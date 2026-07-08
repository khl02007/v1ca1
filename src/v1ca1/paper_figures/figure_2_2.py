"""Generate Figure 2.2."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.plot_wtrack_schematic import draw_large_ovals, get_w_track_geometry
from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures import figure_2 as _figure_2
from v1ca1.paper_figures.datasets import DatasetId, get_processed_datasets
from v1ca1.paper_figures.figure_1 import (
    DECODING_SIGNIFICANCE_BRACKET_HEIGHT,
    DECODING_SIGNIFICANCE_BRACKET_LINEWIDTH,
    DECODING_SIGNIFICANCE_LABEL_FONTSIZE,
    DECODING_SIGNIFICANCE_LABEL_Y_OFFSET,
)
from v1ca1.paper_figures.old_fig3 import (
    PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM,
    PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT,
    PANEL_G_INDEPENDENT_BASIS_ICON_TOP,
    PANEL_G_INDEPENDENT_BASIS_ICON_WIDTH,
    _draw_panel_g_basis_icon,
    _draw_panel_h_track,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "figure_2_2"
DEFAULT_FIGURE_WIDTH_MM = _figure_2.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = (
    _figure_2.PANEL_A_SINGLE_ROW_HEIGHT_MM
    + _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM
    + _figure_2.PANEL_D_ROW_HEIGHT_MM
)
PANEL_C2_SIGNIFICANCE_BRACKET_X = (1.0, 2.0)
PANEL_C2_SIGNIFICANCE_BRACKET_Y_FRACTION = 0.82
PANEL_C2_SIGNIFICANCE_LABEL = "*"
PANEL_C2_ERROR_AXIS_LABEL = "|Norm. error|"
PANEL_A2_SINGLE_ROW_SCHEMATIC_AXIS_LEFT = -0.055
PANEL_D2_SCHEMATIC_AXIS_BOUNDS = _figure_2.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS
PANEL_D2_RESULT_AXIS_BOUNDS = _figure_2.PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS
PANEL_D2_DARK_TRACK_CENTER_X = 0.18
PANEL_D2_SEGMENT_TRACK_CENTER_X = 0.405
PANEL_D2_LIGHT_TRACK_CENTER_X = 0.615
PANEL_D2_BASIS_ICON_CENTER_X = 0.5 * (
    PANEL_D2_DARK_TRACK_CENTER_X + PANEL_D2_LIGHT_TRACK_CENTER_X
)
PANEL_D2_PREDICT_TRACK_CENTER_X = 0.860
PANEL_D2_SHARED_PLUS_X = 0.295
PANEL_D2_SHARED_ARROW_X = (0.500, 0.545)
PANEL_D2_EQUALS_X = 0.5 * (PANEL_D2_SHARED_ARROW_X[0] + PANEL_D2_SHARED_ARROW_X[1])
PANEL_D2_CUE_SWAP_ARROW_MARGIN = 0.006
PANEL_D2_INDEPENDENT_ROW_Y_OFFSET = 0.0
PANEL_D2_SHARED_ROW_Y_OFFSET = -0.070
PANEL_D2_SEGMENT_LABEL_GAP = 0.095
PANEL_D2_CUE_SWAP_LABEL_Y_OFFSET = 0.080
PANEL_D2_RIGHT_ARM_OUTLINE_COLOR = "#0072B2"
PANEL_D2_PLACE_FIELD_COLORS = ("#221150", "#B73779", "#FCFDBF")
PANEL_D2_DARK_FIELD_PLACE_FIELD_ALPHA = 0.5
PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA = 1.0
PANEL_D2_SEGMENT_OVAL_REGIONS = ("left_arm", "right_arm")
PANEL_D2_SEGMENT_OVAL_FILL_COLOR = "#8A8A8A"
PANEL_D2_SEGMENT_OVAL_EDGE_COLOR = "black"
PANEL_D2_SEGMENT_OVAL_LINEWIDTH = 0.75
PANEL_D2_SEGMENT_OVAL_ALPHAS = (0.46, 0.16)
PANEL_D2_SEGMENT_OUTLINE_COLORS = {
    **_figure_2.PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS,
    "right_arm": PANEL_D2_RIGHT_ARM_OUTLINE_COLOR,
}
PANEL_D2_SEGMENT_OUTLINE_LINEWIDTHS = {
    **_figure_2.PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS,
    "right_arm": _figure_2.PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS["left_arm"],
}
PANEL_D2_BASIS_BOTTOM_LINEWIDTH = 2.6
PANEL_D2_BASIS_LABEL_Y_OFFSET = (
    0.5 - 0.5 * (
        PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM + PANEL_G_INDEPENDENT_BASIS_ICON_TOP
    )
) * (
    PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT
    * _figure_2.PANEL_B_INDEPENDENT_BASIS_ICON_SCALE
)
PANEL_D2_EXAMPLE_SLOT_BOUNDS = (
    (0.125, 0.735, 0.300, 0.195),
    (0.125, 0.405, 0.300, 0.195),
    (0.125, 0.075, 0.300, 0.195),
)
PANEL_D2_EXAMPLE_ICON_BOUNDS = (-0.45, 0.23, 0.27, 0.43)


def __getattr__(name: str) -> Any:
    """Delegate unchanged Figure 2 helpers and constants to the base module."""
    return getattr(_figure_2, name)


def _bounds_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> list[float]:
    """Return inset bounds from center coordinates."""
    return [center_x - width / 2.0, center_y - height / 2.0, width, height]


def _add_panel_c2_light_dark_bracket(ax: Any) -> None:
    """Draw the Figure 2.2 Panel C light-dark significance bracket."""
    x_start, x_stop = PANEL_C2_SIGNIFICANCE_BRACKET_X
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    y = y_min + PANEL_C2_SIGNIFICANCE_BRACKET_Y_FRACTION * y_span
    y_top = y + DECODING_SIGNIFICANCE_BRACKET_HEIGHT
    if y_top > y_max:
        y_top = y_max
        y = y_top - DECODING_SIGNIFICANCE_BRACKET_HEIGHT
    ax.plot(
        [x_start, x_start, x_stop, x_stop],
        [y, y_top, y_top, y],
        color="black",
        linewidth=DECODING_SIGNIFICANCE_BRACKET_LINEWIDTH,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        (x_start + x_stop) / 2.0,
        y_top + DECODING_SIGNIFICANCE_LABEL_Y_OFFSET,
        PANEL_C2_SIGNIFICANCE_LABEL,
        ha="center",
        va="bottom",
        fontsize=DECODING_SIGNIFICANCE_LABEL_FONTSIZE,
        color="black",
        clip_on=False,
        zorder=7,
    )


def add_panel_c2_light_dark_brackets(panel_c_axis: Any) -> None:
    """Add light-dark significance brackets to the two Panel C summary axes."""
    for child_axis in panel_c_axis.child_axes[:2]:
        _add_panel_c2_light_dark_bracket(child_axis)


def format_panel_c2_decoding_axes(panel_c_axis: Any) -> None:
    """Apply Figure 2.2-specific cleanup to Panel C decoding axes."""
    child_axes = panel_c_axis.child_axes[:2]
    for child_axis in child_axes:
        for text in list(child_axis.texts):
            if text.get_text().startswith(("Light med.", "Dark med.")):
                text.remove()
    if child_axes:
        child_axes[0].yaxis.label.set_text(PANEL_C2_ERROR_AXIS_LABEL)


def plot_panel_a2_examples_single_row(
    ax: Any,
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot Figure 2.2 Panel A with W-track icons farther from the rasters."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        return

    column_bounds = _figure_2._equal_width_row_bounds(
        len(examples),
        _figure_2.PANEL_A_SINGLE_ROW_COLUMN_GAP,
    )
    for example_index, (example, (left, column_width)) in enumerate(
        zip(examples, column_bounds, strict=True),
        start=1,
    ):
        example_ax = ax.inset_axes([left, 0.0, column_width, 1.0])
        plot_kwargs: dict[str, Any] = {
            "title": None,
            "dark_epoch_axis_left": _figure_2.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT,
            "light_epoch_axis_left": _figure_2.PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT,
            "epoch_axis_width": _figure_2.PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH,
            "schematic_axis_left": PANEL_A2_SINGLE_ROW_SCHEMATIC_AXIS_LEFT,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        }
        y_max_override = _figure_2.PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(example_index)
        if y_max_override is not None:
            plot_kwargs["y_max"] = y_max_override
        _figure_2.plot_panel_a_example(example_ax, example, **plot_kwargs)
        rate_axes = [
            child_ax for child_ax in example_ax.child_axes if child_ax.get_xlabel()
        ]
        for rate_ax in rate_axes:
            rate_ax.set_xlabel("")
            rate_ax.tick_params(
                axis="x",
                labelsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
                pad=0.4,
            )
        example_ax.text(
            0.5,
            0.985,
            f"Example cell {example_index}",
            ha="center",
            va="top",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
            transform=example_ax.transAxes,
        )
        example_ax.text(
            0.5,
            _figure_2.PANEL_A_SINGLE_ROW_XLABEL_Y,
            "Norm. path progression",
            ha="center",
            va="top",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
            transform=example_ax.transAxes,
            clip_on=False,
        )


def _draw_panel_d2_track(
    ax: Any,
    *,
    center_x: float,
    center_y: float,
    track_kind: str,
    track_size: tuple[float, float],
    **kwargs: Any,
) -> Any:
    """Draw one Figure 2.2 Panel D W-track icon."""
    track_ax = ax.inset_axes(
        _bounds_from_center(center_x, center_y, track_size[0], track_size[1])
    )
    track_ax.patch.set_visible(False)
    _draw_panel_h_track(track_ax, track_kind=track_kind, **kwargs)
    return track_ax


def _set_panel_d2_place_field_alpha(ax: Any, alpha: float) -> None:
    """Set a uniform alpha on place-field ellipses in one Panel D2 icon."""
    from matplotlib.patches import Ellipse

    for patch in ax.patches:
        if type(patch) is Ellipse:
            patch.set_alpha(float(alpha))


def _draw_panel_d2_segment_ovals(ax: Any) -> None:
    """Draw custom gain ovals for the Figure 2.2 dark-scaffold segment icon."""
    _outline, _points, dims = get_w_track_geometry()
    draw_large_ovals(
        ax,
        dims,
        oval_regions=list(PANEL_D2_SEGMENT_OVAL_REGIONS),
        oval_styles=[
            {
                "edge_color": PANEL_D2_SEGMENT_OVAL_EDGE_COLOR,
                "fill_color": PANEL_D2_SEGMENT_OVAL_FILL_COLOR,
                "fill_alpha": alpha,
                "linewidth": PANEL_D2_SEGMENT_OVAL_LINEWIDTH,
            }
            for alpha in PANEL_D2_SEGMENT_OVAL_ALPHAS
        ],
    )


def _draw_panel_d2_basis_icon(
    ax: Any,
    *,
    center_x: float,
    center_y: float,
    scale: float,
) -> None:
    """Draw the independent-basis icon in the Figure 2.2 Panel D schematic."""
    width = PANEL_G_INDEPENDENT_BASIS_ICON_WIDTH * float(scale)
    height = PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT * float(scale)
    visual_center_y = 0.5 * (
        PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM + PANEL_G_INDEPENDENT_BASIS_ICON_TOP
    )
    basis_ax = ax.inset_axes(
        [
            center_x - width / 2.0,
            center_y - height * visual_center_y,
            width,
            height,
        ]
    )
    basis_ax.patch.set_visible(False)
    _draw_panel_g_basis_icon(basis_ax)
    vertical_span = (
        PANEL_G_INDEPENDENT_BASIS_ICON_TOP - PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM
    )
    horizontal_left = 0.5 - vertical_span / 2.0
    horizontal_right = 0.5 + vertical_span / 2.0
    basis_ax.plot(
        [horizontal_left, horizontal_right],
        [PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM] * 2,
        color="black",
        linewidth=PANEL_D2_BASIS_BOTTOM_LINEWIDTH,
        solid_capstyle="butt",
        zorder=5,
    )


def _draw_panel_d2_horizontal_arrow(
    ax: Any,
    *,
    start_x: float,
    end_x: float,
    y: float,
) -> None:
    """Draw a short horizontal arrow in Panel D schematic coordinates."""
    ax.annotate(
        "",
        xy=(end_x, y),
        xytext=(start_x, y),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 0.8,
            "mutation_scale": 7.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def draw_panel_d2_architecture_schematic(
    ax: Any,
    *,
    show_dark_track_labels: bool = True,
    track_size: tuple[float, float] = _figure_2.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE,
) -> None:
    """Draw Panel D architecture with adjacent BA prediction icons."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.patch.set_visible(False)

    schematic_shift = _figure_2.PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
    independent_y = (
        _figure_2.PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y + schematic_shift
        + PANEL_D2_INDEPENDENT_ROW_Y_OFFSET
    )
    shared_y = (
        _figure_2.PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
        + schematic_shift
        + PANEL_D2_SHARED_ROW_Y_OFFSET
    )
    field_label_y = _figure_2.PANEL_B_FIELD_LABEL_Y + schematic_shift
    basis_label_y = (
        _figure_2.PANEL_G_INDEPENDENT_BASIS_LABEL_Y
        + schematic_shift
        + PANEL_D2_BASIS_LABEL_Y_OFFSET
    )
    segment_label_y = shared_y + track_size[1] / 2.0 + PANEL_D2_SEGMENT_LABEL_GAP
    cue_swap_arrow_start_x = (
        PANEL_D2_LIGHT_TRACK_CENTER_X
        + track_size[0] / 2.0
        + PANEL_D2_CUE_SWAP_ARROW_MARGIN
    )
    cue_swap_arrow_end_x = (
        PANEL_D2_PREDICT_TRACK_CENTER_X
        - track_size[0] / 2.0
        - PANEL_D2_CUE_SWAP_ARROW_MARGIN
    )
    text_kwargs = {"transform": ax.transAxes}

    ax.text(
        PANEL_D2_DARK_TRACK_CENTER_X,
        field_label_y,
        "Dark field",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_LIGHT_TRACK_CENTER_X,
        field_label_y,
        "Light field",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_PREDICT_TRACK_CENTER_X,
        field_label_y + PANEL_D2_CUE_SWAP_LABEL_Y_OFFSET,
        "Cue-swap\nprediction",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        _figure_2.PANEL_D_MODEL_LABEL_X,
        independent_y,
        "Independent\nmodel",
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_MODEL_LABEL_FONTSIZE,
        fontweight="bold",
        **text_kwargs,
    )
    ax.text(
        _figure_2.PANEL_D_MODEL_LABEL_X,
        shared_y,
        "Dark scaffold\nmodel",
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_MODEL_LABEL_FONTSIZE,
        fontweight="bold",
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_BASIS_ICON_CENTER_X,
        basis_label_y,
        _figure_2.PANEL_B_INDEPENDENT_BASIS_LABEL,
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_COMPONENT_LABEL_FONTSIZE,
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_SEGMENT_TRACK_CENTER_X,
        segment_label_y,
        _figure_2.PANEL_B_SEGMENT_MODULATION_LABEL,
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_COMPONENT_LABEL_FONTSIZE,
        **text_kwargs,
    )

    independent_dark_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_DARK_TRACK_CENTER_X,
        center_y=independent_y,
        track_size=track_size,
        track_kind="dark",
        show_labels=show_dark_track_labels,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _set_panel_d2_place_field_alpha(
        independent_dark_ax,
        PANEL_D2_DARK_FIELD_PLACE_FIELD_ALPHA,
    )
    _draw_panel_d2_basis_icon(
        ax,
        center_x=PANEL_D2_BASIS_ICON_CENTER_X,
        center_y=independent_y,
        scale=_figure_2.PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
    )
    _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_LIGHT_TRACK_CENTER_X,
        center_y=independent_y,
        track_size=track_size,
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        highlighted_segments=(3,),
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS[
            "stim1"
        ],
        region_fill_alpha=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_PREDICT_TRACK_CENTER_X,
        center_y=independent_y,
        track_size=track_size,
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        highlighted_segments=(3,),
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS[
            "stim2"
        ],
        region_fill_alpha=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="right_arm",
    )
    _draw_panel_d2_horizontal_arrow(
        ax,
        start_x=cue_swap_arrow_start_x,
        end_x=cue_swap_arrow_end_x,
        y=independent_y,
    )

    shared_dark_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_DARK_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="dark",
        show_labels=show_dark_track_labels,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="left_arm",
    )
    _set_panel_d2_place_field_alpha(
        shared_dark_ax,
        PANEL_D2_DARK_FIELD_PLACE_FIELD_ALPHA,
    )
    ax.text(
        PANEL_D2_SHARED_PLUS_X,
        shared_y,
        "+",
        ha="center",
        va="center",
        fontsize=8.0,
        **text_kwargs,
    )
    segment_oval_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_SEGMENT_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="segment_modulation",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        segment_outline_colors={},
        segment_outline_linewidths={},
    )
    _draw_panel_d2_segment_ovals(segment_oval_ax)
    ax.text(
        PANEL_D2_EQUALS_X,
        shared_y,
        "=",
        ha="center",
        va="center",
        fontsize=8.0,
        **text_kwargs,
    )
    shared_light_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_LIGHT_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS[
            "stim1"
        ],
        region_fill_alpha=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _set_panel_d2_place_field_alpha(
        shared_light_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA,
    )
    shared_predict_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_PREDICT_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS[
            "stim2"
        ],
        region_fill_alpha=_figure_2.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="right_arm",
    )
    _set_panel_d2_place_field_alpha(
        shared_predict_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA,
    )
    _draw_panel_d2_horizontal_arrow(
        ax,
        start_x=cue_swap_arrow_start_x,
        end_x=cue_swap_arrow_end_x,
        y=shared_y,
    )


def _plot_panel_d2_swap_results(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot the three swap examples and mean-delta histogram."""
    examples = list(swap_examples.values()) if isinstance(swap_examples, dict) else list(
        swap_examples or []
    )
    example_delta_label_positions = _figure_2.PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS
    example_delta_label_vertical_alignments = (
        _figure_2.PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS
    )
    for example_index, bounds in enumerate(PANEL_D2_EXAMPLE_SLOT_BOUNDS):
        example_ax = ax.inset_axes(bounds)
        example = examples[example_index] if example_index < len(examples) else None
        _figure_2._plot_panel_h_switched_segment_example(
            example_ax,
            example,
            model_name=model_name,
            model_colors=model_colors,
            model_labels=model_labels,
            example_label=f"Example {example_index + 1}",
            show_xlabel=example_index == 2,
            show_ylabel=True,
            show_legend=False,
            show_xticklabels=example_index == 2,
            icon_bounds=PANEL_D2_EXAMPLE_ICON_BOUNDS,
            legend_loc="center left",
            legend_bbox_to_anchor=None,
            delta_label_position=(
                example_delta_label_positions[example_index]
                if example_index < len(example_delta_label_positions)
                else None
            ),
            delta_label_va=(
                example_delta_label_vertical_alignments[example_index]
                if example_index < len(example_delta_label_vertical_alignments)
                else None
            ),
        )
        example_ax.tick_params(labelsize=4.3)
        for text in example_ax.texts:
            if text.get_text().startswith("ΔLL="):
                text.set_fontsize(4.1)
        _figure_2._set_nested_legend_fontsize(example_ax, 3.9)
        _figure_2._replace_nested_text(
            example_ax,
            "Norm. path progression",
            "Norm.\npath progression",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
        )

    histogram_ax = ax.inset_axes(_figure_2.PANEL_E_MEAN_DELTA_AXIS_BOUNDS)
    _figure_2.plot_panel_d_mean_swap_delta_axis(
        histogram_ax,
        swap_delta_table,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def plot_panel_d2_architecture_with_swap_results(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot Figure 2.2 Panel D with the BA icons and swap results."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes(PANEL_D2_SCHEMATIC_AXIS_BOUNDS)
    result_ax = ax.inset_axes(PANEL_D2_RESULT_AXIS_BOUNDS)
    result_ax.set_xlim(0.0, 1.0)
    result_ax.set_ylim(0.0, 1.0)
    result_ax.axis("off")

    draw_panel_d2_architecture_schematic(schematic_ax)
    _plot_panel_d2_swap_results(
        result_ax,
        swap_delta_table,
        swap_examples,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def make_figure_2_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = _figure_2.DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = _figure_2.DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = _figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = _figure_2.DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    dark_tuning_correlation_threshold: float = (
        _figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
    ),
    high_dark_tuning_correlation_threshold: float = (
        _figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
    ),
) -> Path:
    """Build and save Figure 2.2."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else _figure_2.DEFAULT_REGIONS[0]
    panel_glm_payload = _figure_2.load_panel_glm_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        swap_delta_min_tuning_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        swap_model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
        swap_example_count=len(_figure_2.PANEL_C_SWAP_EXAMPLES),
        swap_requested_examples=_figure_2.PANEL_C_SWAP_EXAMPLES,
        dark_light_requested_examples=_figure_2.PANEL_C_DARK_LIGHT_EXAMPLES,
    )
    panel_a_examples = [
        _figure_2.load_panel_a_example_data(
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
        for animal_name, date, region, unit_id, trajectories in (
            _figure_2.FIGURE_2_PANEL_A_EXAMPLES
        )
    ]
    panel_b_overlap_table = _figure_2.load_panel_b_tuning_overlap_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_b_overlap_table = _figure_2.filter_panel_b_overlap_by_even_odd_stability(
        panel_b_overlap_table,
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    )
    panel_e_decoding_error_table = _figure_2.load_panel_e_decoding_error_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        **_figure_2.CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS
    )
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            _figure_2.PANEL_A_SINGLE_ROW_HEIGHT_MM,
            _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM,
            _figure_2.PANEL_D_ROW_HEIGHT_MM,
        ],
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    quant_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=_figure_2.PANEL_BC_ROW_WIDTH_RATIOS,
        wspace=_figure_2.PANEL_BC_ROW_WSPACE,
    )
    panel_b_axis = fig.add_subplot(quant_grid[0, 0])
    panel_c_axis = fig.add_subplot(quant_grid[0, 1])
    panel_d_axis = fig.add_subplot(outer_grid[2, 0])

    plot_panel_a2_examples_single_row(panel_a_axis, panel_a_examples)
    _figure_2.plot_panel_b_dpp_overlap_with_schematic(
        panel_b_axis,
        panel_b_overlap_table,
        example=panel_a_examples[0],
        low_threshold=dark_tuning_correlation_threshold,
        high_threshold=high_dark_tuning_correlation_threshold,
    )
    _figure_2._replace_nested_text(
        panel_b_axis,
        "DPP index",
        "DPP index (DPPI)",
        fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    _figure_2.plot_panel_c_cross_and_place_decoding(
        panel_c_axis,
        panel_e_decoding_error_table,
    )
    format_panel_c2_decoding_axes(panel_c_axis)
    add_panel_c2_light_dark_brackets(panel_c_axis)
    plot_panel_d2_architecture_with_swap_results(
        panel_d_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
        model_colors=_figure_2.PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=_figure_2.PANEL_C_SWAP_MODEL_LABELS_2_3,
    )

    label_axis(panel_a_axis, "A", x=-0.02, y=_figure_2.PANEL_A_LABEL_Y)
    panel_a_label = panel_a_axis.texts[-1]
    panel_a_axis.set_title(
        "Example DPP cells in dark and light",
        fontsize=8,
        pad=_figure_2.PANEL_A_TITLE_PAD,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=_figure_2.PANEL_B_LABEL_Y, va="baseline")
    panel_b_label = panel_b_axis.texts[-1]
    panel_b_title = panel_b_axis.set_title(
        "Dark and light DPP coding",
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    label_axis(panel_c_axis, "C", x=-0.035, y=_figure_2.PANEL_B_LABEL_Y, va="baseline")
    panel_c_label = panel_c_axis.texts[-1]
    label_axis(panel_d_axis, "D", x=-0.02, y=_figure_2.PANEL_BC_LABEL_Y)
    panel_d_label = panel_d_axis.texts[-1]
    panel_c_title = panel_c_axis.set_title(
        "Dark and light decoding comparison",
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    panel_d_title = panel_d_axis.set_title(
        "Two models that relate dark and light activity",
        fontsize=8,
        pad=_figure_2.PANEL_BC_TITLE_PAD,
    )

    _figure_2._raise_text_to_minimum_fontsize(
        fig,
        _figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _figure_2._set_axis_horizontal_bounds(
        panel_a_axis,
        left=_figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[0],
        width=_figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[1],
    )
    panel_a_axis_height = panel_a_axis.get_position().height
    _figure_2._set_axis_height_preserving_top(panel_b_axis, panel_a_axis_height)
    _figure_2._set_axis_height_preserving_top(panel_c_axis, panel_a_axis_height)
    _figure_2._scale_axis_width_from_left(
        panel_b_axis,
        _figure_2.PANEL_B_HORIZONTAL_WIDTH_SCALE,
    )
    fig.canvas.draw()
    _figure_2._align_text_to_reference_display_x(panel_b_label, panel_a_label)
    _figure_2._align_text_to_reference_display_x(panel_d_label, panel_a_label)
    _figure_2._align_texts_to_reference_display_y((panel_d_title, panel_d_label))
    _figure_2._align_texts_to_reference_display_y(
        (panel_b_title, panel_b_label, panel_c_title, panel_c_label)
    )

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 2.2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 2.2 generation."""
    parser = argparse.ArgumentParser(description="Generate Figure 2.2.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_figure_2.DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure output. Default: {_figure_2.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
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
        default=_figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Dark tuning-correlation threshold for Panel B low/high grouping. "
            f"Default: {_figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--high-dark-tuning-correlation-threshold",
        type=float,
        default=_figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Upper dark tuning-correlation threshold for Panel B high group. "
            f"Default: {_figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=_figure_2.FIGURE_FORMATS,
        default=_figure_2.DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {_figure_2.DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_figure_2.parse_dataset_id,
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
            f"Default: {', '.join(_figure_2.DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=_figure_2.DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {_figure_2.DEFAULT_POSITION_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=_figure_2.DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples to ignore. "
            f"Default: {_figure_2.DEFAULT_POSITION_OFFSET}"
        ),
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=_figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {_figure_2.DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=_figure_2.DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in bins. Default: {_figure_2.DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 2.2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else _figure_2.DEFAULT_REGIONS
    output_path = _figure_2.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_2_2(
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
        dark_tuning_correlation_threshold=args.dark_tuning_correlation_threshold,
        high_dark_tuning_correlation_threshold=(
            args.high_dark_tuning_correlation_threshold
        ),
    )


if __name__ == "__main__":
    main()
