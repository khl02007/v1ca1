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
    PANEL_E_EXAMPLES,
    draw_panel_b_visual_epoch_icon,
    load_or_compute_panel_e_example_data,
    plot_panel_e_example,
)
from v1ca1.paper_figures.figure_2_old import (
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    FIGURE_2_PANEL_A_EXAMPLES,
    MIN_PUBLICATION_FONTSIZE_PT,
    PANEL_A_EXAMPLE_Y_MAX_OVERRIDES,
    PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
    build_output_path,
    filter_panel_b_overlap_by_even_odd_stability,
    load_panel_a_example_data,
    load_panel_b_tuning_overlap_table,
    load_panel_e_decoding_error_table,
    parse_dataset_id,
    plot_panel_b_dpp_overlap_scatter,
)
from v1ca1.paper_figures._dark_light import (
    DEFAULT_OUTPUT_DIR,
    PANEL_A_DARK_EPOCH_BACKGROUND,
    PANEL_A_EPOCH_LABELS,
    PANEL_TRAJECTORY_COLORS,
    _plot_panel_e_cross_axis,
    _plot_panel_e_place_axis,
    plot_panel_a_raster_axis,
    plot_epoch_path_rate_axis,
    validate_panel_a_trajectories,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import (
    draw_w_track_schematic,
    get_w_track_geometry,
)


DEFAULT_OUTPUT_NAME = "figure_summary"
FIGURE_FORMATS = ("svg", "png", "pdf")
DEFAULT_OUTPUT_FORMAT = "svg"
LETTER_WIDTH_WITH_HALF_INCH_MARGINS_MM = 7.5 * 25.4
DEFAULT_FIGURE_HEIGHT_MM = 86.0
SUMMARY_WIDTH_RATIOS = (0.36, 0.32, 0.32)
SUMMARY_WSPACE = 0.09
SUMMARY_HSPACE = 0.12
SUMMARY_RIGHT_COLUMN_HEIGHT_RATIOS = (1.0, 1.0)
SUMMARY_RIGHT_COLUMN_HSPACE = 0.22
SUMMARY_EXAMPLE_COUNT = 2
SUMMARY_FIGURE_2_PANEL_A_EXAMPLE_NUMBERS = (1, 3)
SUMMARY_TASK_LEFT_ARM_COLOR = "#E78AC3"
SUMMARY_TASK_RIGHT_ARM_COLOR = "#66C2A5"
SUMMARY_TASK_TRAJECTORY_BLOCK_BOUNDS = (0.084, 0.159, 0.331, 0.662)
SUMMARY_TASK_VISUAL_BLOCK_BOUNDS = (0.54, 0.03, 0.44, 0.92)
SUMMARY_CONDITION_TRACK_BOUNDS = (
    (0.02, 0.24, 0.28, 0.30),
    (0.36, 0.24, 0.28, 0.30),
    (0.70, 0.24, 0.28, 0.30),
)
SUMMARY_VISUAL_STIMULUS_LABEL = (0.50, 0.82)
SUMMARY_TIMELINE_ARROW_Y = 0.21
SUMMARY_RUN_SLEEP_TIMELINE_ARROW_Y = 0.305
SUMMARY_RUN_SLEEP_TIMELINE_TIME_LABEL_OFFSET = 0.045
SUMMARY_TIMELINE_STIMULUS_MARKER_CENTERS = (0.16, 0.50, 0.84)
SUMMARY_VISUAL_STIMULUS_BOUNDS = (0.02, 0.54, 0.92, 0.28)
SUMMARY_VISUAL_STIMULUS_ORDER = ("grating", "dots", "black")
SUMMARY_VISUAL_SCREEN_X = 0.07
SUMMARY_VISUAL_SCREEN_Y = 0.28
SUMMARY_VISUAL_SCREEN_W = 0.224
SUMMARY_VISUAL_SCREEN_H = 0.40
SUMMARY_VISUAL_SCREEN_GAP = 0.075
SUMMARY_VISUAL_CONNECTOR_DASH = (0, (3.0, 3.0))
SUMMARY_VISUAL_CONNECTOR_LINEWIDTH = 0.85
SUMMARY_VISUAL_CONNECTOR_ENDPOINT_FRACTION = 0.06
SUMMARY_PANEL_E_CROSS_AXIS_BOUNDS = (0.06, 0.18, 0.40, 0.62)
SUMMARY_PANEL_E_PLACE_AXIS_BOUNDS = (0.57, 0.18, 0.39, 0.62)
SUMMARY_TOP_HEADER_Y_PAD = 0.012
SUMMARY_TOP_HEADER_LABEL_X_PAD = 0.02
SUMMARY_MIN_PUBLICATION_FONTSIZE_EXEMPT_TEXT = frozenset({"A", "B"})


def _raise_text_to_minimum_fontsize(
    fig: Any,
    min_fontsize: float,
    *,
    additional_exempt_text: Sequence[str] = (),
) -> None:
    """Raise final figure text to a minimum size, preserving W-track A/B labels."""
    exempt_text = SUMMARY_MIN_PUBLICATION_FONTSIZE_EXEMPT_TEXT.union(
        additional_exempt_text
    )

    def _iter_axes_tree(ax: Any) -> Sequence[Any]:
        axes = [ax]
        for child_ax in getattr(ax, "child_axes", ()):
            axes.extend(_iter_axes_tree(child_ax))
        return axes

    def _maybe_raise(text: Any) -> None:
        if text is None:
            return
        if text.get_text().strip() in exempt_text:
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


def _add_aligned_top_panel_headers(
    fig: Any,
    panel_specs: Sequence[tuple[Any, str, str]],
) -> None:
    """Add figure-level headers for top-row panels at a shared y position."""
    top_y = (
        max(ax.get_position().y1 for ax, _label, _title in panel_specs)
        + SUMMARY_TOP_HEADER_Y_PAD
    )
    for ax, panel_label, title in panel_specs:
        bounds = ax.get_position()
        label_x = bounds.x0 - SUMMARY_TOP_HEADER_LABEL_X_PAD * bounds.width
        title_x = 0.5 * (bounds.x0 + bounds.x1)
        fig.text(
            label_x,
            top_y,
            panel_label,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
        fig.text(
            title_x,
            top_y,
            title,
            ha="center",
            va="bottom",
            fontsize=8.0,
        )


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
    arm_side_outlines: bool = False,
) -> None:
    """Draw one summary W-track condition icon."""
    arm_colors = {}
    if left_arm_color is not None:
        arm_colors["left_arm"] = left_arm_color
    if right_arm_color is not None:
        arm_colors["right_arm"] = right_arm_color
    color_kwargs = {}
    if arm_colors:
        color_kwargs = (
            {"arm_side_outline_colors": arm_colors}
            if arm_side_outlines
            else {
                "region_fill_colors": arm_colors,
                "region_fill_alpha": 0.92,
            }
        )
    draw_panel_b_visual_epoch_icon(
        ax,
        fill_track=fill_track,
        **color_kwargs,
    )


def _draw_summary_visual_stimulus_icons(ax: Any) -> None:
    """Draw enlarged visual-stimulus cartoons for the summary task panel."""
    from matplotlib.patches import Ellipse, Rectangle

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    screen_y = SUMMARY_VISUAL_SCREEN_Y
    screen_h = SUMMARY_VISUAL_SCREEN_H
    screen_w = SUMMARY_VISUAL_SCREEN_W
    screen_gap = SUMMARY_VISUAL_SCREEN_GAP
    screen_x = SUMMARY_VISUAL_SCREEN_X
    for index, screen_type in enumerate(SUMMARY_VISUAL_STIMULUS_ORDER):
        x0 = screen_x + index * (screen_w + screen_gap)
        facecolor = "black" if screen_type == "black" else "white"
        if screen_type == "dots":
            facecolor = "0.70"
        edgecolor = {
            "grating": SUMMARY_TASK_LEFT_ARM_COLOR,
            "dots": SUMMARY_TASK_RIGHT_ARM_COLOR,
        }.get(screen_type, "black")
        linewidth = 1.7 if screen_type in {"grating", "dots"} else 0.75
        ax.add_patch(
            Rectangle(
                (x0, screen_y),
                screen_w,
                screen_h,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                transform=ax.transAxes,
            )
        )
        if screen_type == "grating":
            stripe_w = screen_w / 7.0
            for stripe_index in range(0, 7, 2):
                ax.add_patch(
                    Rectangle(
                        (x0 + stripe_index * stripe_w, screen_y),
                        stripe_w,
                        screen_h,
                        facecolor="black",
                        edgecolor="none",
                        transform=ax.transAxes,
                    )
                )
        elif screen_type == "dots":
            for x_frac, y_frac, radius, color in (
                (0.22, 0.72, 0.16, "white"),
                (0.50, 0.54, 0.11, "black"),
                (0.76, 0.72, 0.13, "white"),
                (0.24, 0.28, 0.13, "black"),
                (0.58, 0.24, 0.15, "white"),
                (0.82, 0.34, 0.16, "black"),
            ):
                ax.add_patch(
                    Ellipse(
                        (x0 + x_frac * screen_w, screen_y + y_frac * screen_h),
                        width=2.0 * radius * screen_w,
                        height=2.0 * radius * screen_h,
                        facecolor=color,
                        edgecolor="none",
                        transform=ax.transAxes,
                    )
                )
    ellipsis_x = screen_x + 3.0 * screen_w + 2.0 * screen_gap + 0.055
    ax.text(
        ellipsis_x,
        screen_y + 0.48 * screen_h,
        "...",
        ha="left",
        va="center",
        fontsize=6.0,
        transform=ax.transAxes,
    )


def _visual_stimulus_screen_point(
    stimulus_index: int,
    *,
    x_fraction: float,
    y_fraction: float,
) -> tuple[float, float]:
    """Return a visual-stimulus point in the parent schematic axes."""
    visual_x, visual_y, visual_w, visual_h = SUMMARY_VISUAL_STIMULUS_BOUNDS
    screen_x = SUMMARY_VISUAL_SCREEN_X + stimulus_index * (
        SUMMARY_VISUAL_SCREEN_W + SUMMARY_VISUAL_SCREEN_GAP
    )
    x = visual_x + (screen_x + x_fraction * SUMMARY_VISUAL_SCREEN_W) * visual_w
    y = visual_y + (
        SUMMARY_VISUAL_SCREEN_Y + y_fraction * SUMMARY_VISUAL_SCREEN_H
    ) * visual_h
    return x, y


def _condition_track_data_point(
    condition_index: int,
    *,
    x: float,
    y: float,
    dims: dict[str, float],
) -> tuple[float, float]:
    """Return a W-track data point in the parent schematic axes."""
    bounds = SUMMARY_CONDITION_TRACK_BOUNDS[condition_index]
    source_xlim = (-0.95, dims["x5"] + 0.95)
    source_ylim = (-0.25, dims["y2"] + 0.25)
    x_fraction = (x - source_xlim[0]) / (source_xlim[1] - source_xlim[0])
    y_fraction = (y - source_ylim[0]) / (source_ylim[1] - source_ylim[0])
    return (
        bounds[0] + x_fraction * bounds[2],
        bounds[1] + y_fraction * bounds[3],
    )


def _draw_summary_visual_stimulus_connectors(
    ax: Any,
    condition_axes: Sequence[Any],
) -> None:
    """Draw dashed mappings from visual-stimulus icons to matching W-track arms."""
    from matplotlib.patches import ConnectionPatch

    _outline, _points, dims = get_w_track_geometry()
    arm_inset = 0.08
    left_arm_targets = (dims["x0"] + arm_inset, dims["x1"] - arm_inset)
    right_arm_targets = (dims["x4"] + arm_inset, dims["x5"] - arm_inset)
    connector_specs = (
        ("grating", 0, left_arm_targets, SUMMARY_TASK_LEFT_ARM_COLOR),
        ("dots", 0, right_arm_targets, SUMMARY_TASK_RIGHT_ARM_COLOR),
    )
    target_y = dims["y2"] - 0.22
    for stimulus_name, condition_index, target_xs, color in connector_specs:
        stimulus_index = SUMMARY_VISUAL_STIMULUS_ORDER.index(stimulus_name)
        start_points = (
            _visual_stimulus_screen_point(
                stimulus_index,
                x_fraction=0.08,
                y_fraction=0.0,
            ),
            _visual_stimulus_screen_point(
                stimulus_index,
                x_fraction=0.92,
                y_fraction=0.0,
            ),
        )
        target_points = tuple(
            (target_x, target_y) for target_x in target_xs
        )
        condition_ax = condition_axes[condition_index]
        for start, target in zip(start_points, target_points, strict=True):
            target_parent = _condition_track_data_point(
                condition_index,
                x=target[0],
                y=target[1],
                dims=dims,
            )
            connector = ConnectionPatch(
                xyA=start,
                coordsA=ax.transAxes,
                xyB=target,
                coordsB=condition_ax.transData,
                axesA=ax,
                axesB=condition_ax,
                arrowstyle="-",
                color=color,
                linewidth=SUMMARY_VISUAL_CONNECTOR_LINEWIDTH,
                linestyle=SUMMARY_VISUAL_CONNECTOR_DASH,
                mutation_scale=1.0,
                shrinkA=0.0,
                shrinkB=0.0,
                clip_on=False,
                zorder=30,
            )
            ax.figure.add_artist(connector)

            # Tiny solid endpoint segments avoid a dash-gap at the exact contact points.
            start_cap = (
                start[0]
                + SUMMARY_VISUAL_CONNECTOR_ENDPOINT_FRACTION
                * (target_parent[0] - start[0]),
                start[1]
                + SUMMARY_VISUAL_CONNECTOR_ENDPOINT_FRACTION
                * (target_parent[1] - start[1]),
            )
            ax.plot(
                [start[0], start_cap[0]],
                [start[1], start_cap[1]],
                color=color,
                linewidth=SUMMARY_VISUAL_CONNECTOR_LINEWIDTH,
                solid_capstyle="butt",
                transform=ax.transAxes,
                clip_on=False,
                zorder=31,
            )
            condition_ax.plot(
                [target[0], target[0]],
                [target[1] + 0.16, target[1]],
                color=color,
                linewidth=SUMMARY_VISUAL_CONNECTOR_LINEWIDTH,
                solid_capstyle="butt",
                clip_on=False,
                zorder=31,
            )


def _draw_summary_timeline_stimulus_markers(ax: Any) -> None:
    """Draw compact run labels on the experimental timeline arrow."""
    from matplotlib.patches import Rectangle

    marker_w = 0.18
    marker_h = 0.075
    y0 = SUMMARY_TIMELINE_ARROW_Y - marker_h / 2.0
    for center_x, run_label in zip(
        SUMMARY_TIMELINE_STIMULUS_MARKER_CENTERS,
        ("Run1", "Run2", "Run3"),
        strict=True,
    ):
        x0 = center_x - marker_w / 2.0
        ax.add_patch(
            Rectangle(
                (x0, y0),
                marker_w,
                marker_h,
                facecolor="white",
                edgecolor="black",
                linewidth=0.75,
                transform=ax.transAxes,
                zorder=4,
            )
        )
        ax.text(
            center_x,
            SUMMARY_TIMELINE_ARROW_Y,
            run_label,
            ha="center",
            va="center",
            fontsize=5.2,
            transform=ax.transAxes,
            zorder=5,
        )


def _draw_summary_run_sleep_timeline_markers(
    ax: Any,
    *,
    arrow_y: float = SUMMARY_RUN_SLEEP_TIMELINE_ARROW_Y,
) -> None:
    """Draw the condensed run/sleep sequence as a plain timeline arrow."""
    time_label_y = arrow_y - SUMMARY_RUN_SLEEP_TIMELINE_TIME_LABEL_OFFSET
    ax.annotate(
        "",
        xy=(0.98, arrow_y),
        xytext=(0.02, arrow_y),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "0.25",
            "linewidth": 0.7,
            "mutation_scale": 8.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
        zorder=3,
    )
    ax.text(
        0.50,
        time_label_y,
        "Time",
        ha="center",
        va="top",
        fontsize=5.2,
        transform=ax.transAxes,
        zorder=3,
    )


def _draw_summary_trajectory_cycle(ax: Any) -> None:
    """Draw the four W-track trajectories and transition arrows."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for trajectory_type, bounds in CYCLE_TRAJECTORY_LAYOUT:
        inset = ax.inset_axes(bounds)
        draw_w_track_schematic(
            inset,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=1.2,
            trajectory_linewidth=1.4,
            arrow_mutation_scale=13.0,
            fill_track=False,
        )
        if trajectory_type == "center_to_left":
            _outline, points, dims = get_w_track_geometry()
            label_y = dims["y2"] + 0.08
            for arm_name, label in (("left", "L"), ("center", "C"), ("right", "R")):
                inset.text(
                    points[arm_name][0],
                    label_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=5.2,
                    color="black",
                    clip_on=False,
                )
    for start, end, rad in CYCLE_ARROW_SPECS:
        ax.annotate(
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


def _draw_summary_visual_stimulus_block(
    ax: Any,
    *,
    timeline_style: str = "arrow",
    run_sleep_timeline_arrow_y: float = SUMMARY_RUN_SLEEP_TIMELINE_ARROW_Y,
    arm_side_outlines: bool = False,
) -> None:
    """Draw stimulus-location icons, timeline arrow, and stimulus cartoons."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    condition_specs = (
        {
            "left_arm_color": SUMMARY_TASK_LEFT_ARM_COLOR,
            "right_arm_color": SUMMARY_TASK_RIGHT_ARM_COLOR,
        },
        {
            "left_arm_color": SUMMARY_TASK_RIGHT_ARM_COLOR,
            "right_arm_color": SUMMARY_TASK_LEFT_ARM_COLOR,
        },
        {"fill_track": True},
    )
    if timeline_style != "run_sleep_boxes":
        ax.text(
            *SUMMARY_VISUAL_STIMULUS_LABEL,
            "Visual stimuli",
            ha="center",
            va="bottom",
            fontsize=6.0,
            transform=ax.transAxes,
        )
    condition_axes = []
    for bounds, condition_spec in zip(
        SUMMARY_CONDITION_TRACK_BOUNDS,
        condition_specs,
        strict=True,
    ):
        condition_ax = ax.inset_axes(bounds)
        condition_axes.append(condition_ax)
        _draw_summary_condition_track(
            condition_ax,
            arm_side_outlines=arm_side_outlines,
            **condition_spec,
        )
    if timeline_style == "arrow":
        ax.annotate(
            "",
            xy=(0.98, SUMMARY_TIMELINE_ARROW_Y),
            xytext=(0.02, SUMMARY_TIMELINE_ARROW_Y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "0.25",
                "linewidth": 0.7,
                "mutation_scale": 8.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )
        _draw_summary_timeline_stimulus_markers(ax)
    elif timeline_style == "run_sleep_boxes":
        _draw_summary_run_sleep_timeline_markers(
            ax,
            arrow_y=run_sleep_timeline_arrow_y,
        )
    else:
        raise ValueError("timeline_style must be 'arrow' or 'run_sleep_boxes'.")
    visual_ax = ax.inset_axes(SUMMARY_VISUAL_STIMULUS_BOUNDS)
    _draw_summary_visual_stimulus_icons(visual_ax)
    _draw_summary_visual_stimulus_connectors(ax, condition_axes)


def draw_summary_task_panel(ax: Any, *, timeline_style: str = "arrow") -> None:
    """Draw the compact summary task schematic for Panel A."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    trajectory_ax = ax.inset_axes(SUMMARY_TASK_TRAJECTORY_BLOCK_BOUNDS)
    _draw_summary_trajectory_cycle(trajectory_ax)

    visual_ax = ax.inset_axes(SUMMARY_TASK_VISUAL_BLOCK_BOUNDS)
    _draw_summary_visual_stimulus_block(visual_ax, timeline_style=timeline_style)


def plot_summary_directional_progression_panel(
    ax: Any,
    example: dict[str, Any],
) -> None:
    """Plot the first Figure 1C example as a compact DPP-coding schematic."""
    plot_panel_e_example(
        ax,
        example,
        title=None,
        show_ylabel=True,
        show_rate_legends=False,
    )
    rate_axes = [child_ax for child_ax in ax.child_axes if child_ax.get_xlabel()]
    for rate_index, rate_ax in enumerate(rate_axes):
        rate_ax.set_xticklabels(["0", ""] if rate_index == 0 else ["", "1"])
    for child_ax in ax.child_axes:
        if child_ax.get_xlabel():
            child_ax.set_xlabel("")
        if child_ax.get_ylabel() in {"Trials", "FR (Hz)"}:
            child_ax.yaxis.set_label_coords(-0.16, 0.5)
    ax.text(
        0.505,
        -0.055,
        "Norm. path progression",
        ha="center",
        va="top",
        fontsize=6.0,
        transform=ax.transAxes,
        clip_on=False,
    )


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


def _summary_raster_section_centers(
    example: dict[str, Any],
    trajectories: Sequence[str],
) -> dict[str, float]:
    """Return normalized y centers for trajectory sections in a compact raster."""
    raster_positions = example["epoch_rates"]["dark"]["raster_positions"]
    row_index = 1
    section_centers: dict[str, float] = {}
    for trajectory_type in trajectories:
        n_trials = len(raster_positions[trajectory_type])
        section_centers[trajectory_type] = row_index + max(n_trials - 1, 0) / 2.0
        row_index += n_trials + 1
    y_limit = float(max(1, row_index))
    return {
        trajectory_type: section_center / y_limit
        for trajectory_type, section_center in section_centers.items()
    }


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
    rate_y = 0.20 if show_xlabel else 0.12
    rate_height = 0.36
    raster_y = 0.62
    raster_height = 0.30
    icon_height = 0.075
    section_centers = _summary_raster_section_centers(example, trajectories)
    for trajectory_type in reversed(trajectories):
        icon_y = (
            raster_y
            + raster_height * section_centers[trajectory_type]
            - icon_height / 2.0
        )
        icon_ax = ax.inset_axes([0.025, icon_y, 0.075, icon_height])
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )

    dark_raster_ax = ax.inset_axes([0.14, raster_y, 0.37, raster_height])
    light_raster_ax = ax.inset_axes([0.60, raster_y, 0.37, raster_height])
    dark_raster_ax.set_facecolor(PANEL_A_DARK_EPOCH_BACKGROUND)
    for epoch_key, raster_ax in (
        ("dark", dark_raster_ax),
        ("light", light_raster_ax),
    ):
        plot_panel_a_raster_axis(
            raster_ax,
            example,
            epoch_key,
            trajectories=trajectories,
            show_ylabel=epoch_key == "dark",
            show_title=True,
        )
        raster_ax.set_title(PANEL_A_EPOCH_LABELS[epoch_key], fontsize=6.0, pad=1.0)
        raster_ax.tick_params(labelsize=6.0, length=1.0, pad=0.6)
        if epoch_key == "light":
            raster_ax.set_ylabel("")
        else:
            raster_ax.yaxis.set_label_coords(-0.28, 0.5)

    dark_ax = ax.inset_axes([0.14, rate_y, 0.37, rate_height])
    light_ax = ax.inset_axes([0.60, rate_y, 0.37, rate_height])
    dark_ax.set_facecolor(PANEL_A_DARK_EPOCH_BACKGROUND)
    for epoch_key, rate_ax in (("dark", dark_ax), ("light", light_ax)):
        plot_epoch_path_rate_axis(
            rate_ax,
            example,
            epoch_key,
            trajectories=trajectories,
            y_max=y_max,
            show_ylabel=epoch_key == "dark",
            show_title=False,
            show_correlation=False,
        )
        rate_ax.tick_params(labelsize=6.0, length=1.3, pad=0.8)
        _set_rate_axis_xlabel(rate_ax, show_xlabel=show_xlabel)
        if epoch_key == "light":
            rate_ax.set_ylabel("")
        else:
            rate_ax.yaxis.set_label_coords(-0.28, 0.5)
    if show_xlabel:
        _add_combined_rate_axis_xlabel(ax)


def plot_summary_examples(
    ax: Any,
    examples: Sequence[dict[str, Any]],
    *,
    source_example_numbers: Sequence[int] | None = None,
) -> None:
    """Plot two compact dark/light DPP example cells."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        return

    row_bottom = 0.02
    row_top = 0.96
    row_gap = 0.03
    row_height = (row_top - row_bottom - row_gap) / 2.0
    y_positions = (row_bottom + row_height + row_gap, row_bottom)
    for example_index, (example, y0) in enumerate(
        zip(examples[:SUMMARY_EXAMPLE_COUNT], y_positions, strict=False),
        start=1,
    ):
        row_ax = ax.inset_axes([0.0, y0, 1.0, row_height])
        source_example_number = (
            source_example_numbers[example_index - 1]
            if source_example_numbers is not None
            and example_index <= len(source_example_numbers)
            else example_index
        )
        y_max = _example_rate_y_max(
            example,
            PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(source_example_number),
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
            child_ax.set_xlabel("Dark DPP Index", fontsize=6.0, labelpad=1.0)
            child_ax.set_ylabel("Light DPP Index", fontsize=6.0, labelpad=1.0)
            child_ax.set_xticks([0.0, 0.5, 1.0])
            child_ax.set_yticks([0.0, 0.5, 1.0])
            child_ax.set_xticklabels(["0", "0.5", "1"])
            child_ax.set_yticklabels(["0", "0.5", "1"])
            child_ax.tick_params(labelsize=6.0, length=1.4, pad=1.0)
            for text in child_ax.texts:
                text.set_fontsize(6.0)
        elif child_ax.get_xlabel() == "Frac.":
            child_ax.set_xlabel("Frac.", fontsize=6.0, labelpad=0.8)
            child_ax.set_xticks([0.1])
            child_ax.set_xticklabels(["0.1"])
            child_ax.tick_params(axis="x", labelsize=6.0, length=1.2, pad=0.8)
        if child_ax.get_ylabel() == "Light DPP\noverlap":
            child_ax.set_ylabel("Light DPP Index", fontsize=6.0, labelpad=1.0)
        elif child_ax.get_ylabel() == "Frac.":
            child_ax.set_ylabel("Frac.", fontsize=6.0, labelpad=0.8)
            child_ax.set_yticks([0.1])
            child_ax.set_yticklabels(["0.1"])
            child_ax.tick_params(axis="y", labelsize=6.0, length=1.2, pad=0.8)


def plot_summary_panel_d(
    ax: Any,
    overlap_table: Any,
) -> None:
    """Plot the summary Panel D dark-vs-light DPP scatter."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    plot_panel_b_dpp_overlap_scatter(ax, overlap_table, title=None)
    _format_summary_dppi_scatter(ax)
    for child_ax in ax.child_axes:
        for text in child_ax.texts:
            if text.get_text().strip().startswith("median"):
                text.set_text("")


def plot_summary_panel_e(
    ax: Any,
    decoding_error_table: Any,
) -> None:
    """Plot the summary Panel E decoding comparison."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    cross_ax = ax.inset_axes(SUMMARY_PANEL_E_CROSS_AXIS_BOUNDS)
    place_ax = ax.inset_axes(SUMMARY_PANEL_E_PLACE_AXIS_BOUNDS)
    _plot_panel_e_cross_axis(cross_ax, decoding_error_table)
    _plot_panel_e_place_axis(place_ax, decoding_error_table, ylabel=None)
    for child_ax, title in ((cross_ax, "Cross-path"), (place_ax, "Place")):
        child_ax.set_title(title, fontsize=6.0, pad=1.0)
        child_ax.tick_params(labelsize=6.0, length=1.4, pad=1.0)
    for child_ax in ax.child_axes:
        for text in child_ax.texts:
            if text.get_text().strip().startswith(("Light med.", "Dark med.")):
                text.set_text("")


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
    selected_examples = [
        FIGURE_2_PANEL_A_EXAMPLES[example_number - 1]
        for example_number in SUMMARY_FIGURE_2_PANEL_A_EXAMPLE_NUMBERS
    ]
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
    direction_animal, direction_date, direction_epoch, direction_region, direction_unit = (
        PANEL_E_EXAMPLES[0]
    )
    direction_example = load_or_compute_panel_e_example_data(
        data_root=data_root,
        animal_name=direction_animal,
        date=direction_date,
        epoch=direction_epoch,
        region=direction_region,
        unit_id=direction_unit,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_e_cache_dir=panel_example_cache_dir,
        refresh_panel_e_cache=refresh_panel_example_cache,
    )
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
    decoding_error_table = load_panel_e_decoding_error_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
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
        nrows=2,
        ncols=3,
        width_ratios=SUMMARY_WIDTH_RATIOS,
        height_ratios=(1.0, 1.0),
        wspace=SUMMARY_WSPACE,
        hspace=SUMMARY_HSPACE,
    )
    task_ax = fig.add_subplot(grid[0, 0])
    dppi_ax = fig.add_subplot(grid[1, 0])
    examples_ax = fig.add_subplot(grid[:, 1])
    right_grid = grid[:, 2].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=SUMMARY_RIGHT_COLUMN_HEIGHT_RATIOS,
        hspace=SUMMARY_RIGHT_COLUMN_HSPACE,
    )
    panel_d_axis = fig.add_subplot(right_grid[0, 0])
    panel_e_axis = fig.add_subplot(right_grid[1, 0])

    del asset_dir
    draw_summary_task_panel(task_ax)

    plot_summary_directional_progression_panel(dppi_ax, direction_example)
    dppi_ax.set_title(
        "V1 encodes directional path progression\n(DPP) in darkness",
        fontsize=8.0,
        pad=1.5,
    )
    label_axis(dppi_ax, "B", x=-0.02, y=1.02)

    plot_summary_examples(
        examples_ax,
        panel_examples,
        source_example_numbers=SUMMARY_FIGURE_2_PANEL_A_EXAMPLE_NUMBERS,
    )

    plot_summary_panel_d(panel_d_axis, overlap_table)

    plot_summary_panel_e(panel_e_axis, decoding_error_table)
    panel_e_axis.set_title(
        "Dark vs light decoding",
        fontsize=8.0,
        pad=1.5,
    )
    label_axis(panel_e_axis, "E", x=-0.02, y=1.02)

    _raise_text_to_minimum_fontsize(fig, MIN_PUBLICATION_FONTSIZE_PT)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _add_aligned_top_panel_headers(
        fig,
        (
            (task_ax, "A", "W-track task with visual landmarks"),
            (examples_ax, "C", "Light modulates DPP tuning"),
            (panel_d_axis, "D", "Dark vs light DPP"),
        ),
    )
    fig.canvas.draw()
    save_figure(fig, output_path, dpi=dpi, pad_inches=0.03)
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
