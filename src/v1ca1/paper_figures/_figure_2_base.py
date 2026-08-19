"""Private helper backend for the canonical Figure 2 implementation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures import figure_2_common as _figure_2_common
from v1ca1.paper_figures.figure_1 import (
    DELTA_LOG_LIKELIHOOD_AXIS_LABEL,
    PANEL_B_VISUAL_ICON_COLORS,
    PANEL_B_VISUAL_ICON_REGION_FILL_ALPHA,
    PANEL_E_RIGHT_ANNOTATION_X,
    PANEL_E_RIGHT_SUMMARY_POSITION,
    _format_cell_animal_count,
    _format_delta_advantage_summary,
    get_stability_table_path,
    select_units_by_saved_movement_firing_rate,
)
from v1ca1.paper_figures.figure_2_common import *  # noqa: F403
from v1ca1.paper_figures.figure_2_common import (
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    FIGURE_2_CONSTRAINED_LAYOUT_PADS,
    FIGURE_2_PANEL_A_EXAMPLES,
    FIGURE_FORMATS,
    PANEL_AB_WIDTH_RATIOS,
    PANEL_AB_WSPACE,
    PANEL_A_EXAMPLE_ROW_HEIGHT_MM,
    PANEL_A_LABEL_Y,
    PANEL_A_TITLE_PAD,
    PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y,
    PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y,
    PANEL_BC_LABEL_Y,
    PANEL_BC_ROW_HEIGHT_MM,
    PANEL_BC_TITLE_PAD,
    PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
    PANEL_B_EXAMPLE_FIELD_HEIGHT,
    PANEL_B_EXAMPLE_FIELD_Y,
    PANEL_B_EXAMPLE_MODEL_COLORS,
    PANEL_B_EXAMPLE_MODEL_LABELS,
    PANEL_B_FIELD_LABEL_Y,
    PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
    PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
    PANEL_B_INDEPENDENT_BASIS_LABEL,
    PANEL_B_LABEL_Y,
    PANEL_B_MODEL_LABEL_FONTSIZE,
    PANEL_B_MODEL_LABEL_X,
    PANEL_B_SEGMENT_MODULATION_LABEL,
    PANEL_B_SEGMENT_MODULATION_LABEL_Y,
    PANEL_B_TUNING_CORRELATION_TRAJECTORIES,
    PANEL_B_TITLE_PAD,
    PANEL_B_COMPONENT_LABEL_FONTSIZE,
    PANEL_C_DELTA_AXIS_BOUNDS,
    PANEL_C_DARK_LIGHT_EXAMPLES,
    PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS,
    PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS,
    PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y,
    PANEL_C_INDEPENDENT_TRACK_CENTER_Y,
    PANEL_C_PREDICTION_LABEL_FONTSIZE,
    PANEL_C_SCHEMATIC_AXIS_BOUNDS,
    PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y,
    PANEL_C_SHARED_DARK_TRACK_CENTER_Y,
    PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y,
    PANEL_C_SHARED_PREDICTION_LABEL_Y,
    PANEL_C_SWAP_EXAMPLES,
    PANEL_C_SWAP_MODEL_COLORS,
    PANEL_C_SWAP_MODEL_LABELS,
    PANEL_C_SWAP_MODEL_NAME,
    PANEL_A_EXAMPLE_Y_MAX_OVERRIDES,
    build_output_path,
    load_panel_a_example_data,
    load_panel_b_tuning_overlap_table,
    load_panel_glm_data,
    parse_dataset_id,
    plot_panel_a_example,
    plot_panel_b_dpp_overlap_grouped,
    plot_panel_b_dpp_overlap_scatter,
    _align_text_to_reference_display_x,
    _align_texts_to_reference_display_y,
    _shift_axis_horizontally,
)
from v1ca1.paper_figures._dark_light import (
    DEFAULT_OUTPUT_DIR,
    PANEL_G_DARK_TRACK_CENTER_X,
    PANEL_G_INDEPENDENT_BASIS_LABEL_Y,
    PANEL_G_LIGHT_TRACK_CENTER_X,
    PANEL_G_SEGMENT_MODULATION_TRACK_CENTER_X,
    PANEL_G_SHARED_OUTPUT_ARROW_X,
    PANEL_TRAJECTORY_COLORS,
    PANEL_H_DELTA_TRAJECTORIES,
    PANEL_H_DELTA_X_LIMITS,
    _draw_panel_h_track,
    _filter_panel_h_heldout_delta,
    _fraction_histogram_weights,
    get_dark_epoch,
    get_light_epoch,
    _plot_panel_e_cross_axis,
    _panel_model_color,
    _panel_model_label,
    _plot_panel_e_place_axis,
    _plot_panel_g_architecture_schematic,
    _plot_panel_g_example_columns,
    _plot_panel_h_switched_segment_example,
    load_panel_e_decoding_error_table,
)
from v1ca1.paper_figures.style import (
    EMPHASIS_HISTOGRAM_KWARGS,
    NEUTRAL_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_NAME = "figure_2"
PANEL_A_SINGLE_ROW_HEIGHT_MM = PANEL_A_EXAMPLE_ROW_HEIGHT_MM * 0.76
PANEL_BC_QUANT_ROW_HEIGHT_MM = PANEL_A_SINGLE_ROW_HEIGHT_MM
PANEL_D_ROW_HEIGHT_MM = PANEL_A_SINGLE_ROW_HEIGHT_MM
PANEL_E_ROW_HEIGHT_MM = PANEL_A_SINGLE_ROW_HEIGHT_MM
PANEL_BC_ROW_WIDTH_RATIOS = (2.0, 1.0)
PANEL_BC_ROW_WSPACE = 0.03
PANEL_E_HORIZONTAL_SHIFT = 0.0
DEFAULT_FIGURE_HEIGHT_MM = (
    PANEL_A_SINGLE_ROW_HEIGHT_MM
    + PANEL_BC_QUANT_ROW_HEIGHT_MM
    + PANEL_D_ROW_HEIGHT_MM
    + PANEL_E_ROW_HEIGHT_MM
)
CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS = {
    **FIGURE_2_CONSTRAINED_LAYOUT_PADS,
    "hspace": 0.08,
}
PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS = (0.005, 0.04, 0.49, 0.92)
PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS = (0.50, 0.06, 0.50, 0.90)
PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE = (0.194, 0.241)
PANEL_C_SIDE_BY_SIDE_EXAMPLE_ICON_BOUNDS = (0.018, 0.345, 0.050, 0.205)
PANEL_C_SIDE_BY_SIDE_EXAMPLE_XLABEL_Y = 0.06
PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_WIDTH = 0.50
PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_GAP = 0.02
PANEL_C_SIDE_BY_SIDE_EXAMPLE_PLOT_LEFT_OFFSET = 0.13
PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_Y = 0.19
PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_HEIGHT = 0.64
PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_WIDTH = 0.32
PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_GAP = 0.045
PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_GAP = 0.12
PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_HEIGHT = (
    1.0 - PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_GAP
) / 2.0
PANEL_D_COMPACT_SLOT_BOUNDS = (
    (0.055, 0.700, 0.300, 0.215),
    (0.055, 0.410, 0.300, 0.215),
    (0.055, 0.120, 0.300, 0.215),
    (0.650, 0.125, 0.330, 0.700),
)
PANEL_D_COMPACT_EXAMPLE_ICON_BOUNDS = (-0.38, 0.35, 0.18, 0.28)
PANEL_E_EXAMPLE_SLOT_BOUNDS = (
    (0.040, 0.700, 0.420, 0.235),
    (0.040, 0.390, 0.420, 0.235),
    (0.040, 0.080, 0.420, 0.235),
)
PANEL_E_MEAN_DELTA_AXIS_BOUNDS = (0.605, 0.165, 0.380, 0.685)
PANEL_E_EXAMPLE_ICON_BOUNDS = (-0.30, 0.32, 0.135, 0.265)
PANEL_D_SCHEMATIC_TRACK_SIZE = PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE
PANEL_E_SCHEMATIC_AXIS_BOUNDS = PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS
PANEL_E_DELTA_AXIS_BOUNDS = (0.50, 0.08, 0.50, 0.86)
PANEL_A_HORIZONTAL_AXIS_BOUNDS = (0.040, 0.950)
PANEL_B_HORIZONTAL_WIDTH_SCALE = 1.00
PANEL_A_SINGLE_ROW_COLUMN_GAP = 0.035
PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT = 0.05
PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT = 0.58
PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH = 0.37
PANEL_A_SINGLE_ROW_SCHEMATIC_AXIS_LEFT = -0.025
PANEL_A_SINGLE_ROW_XLABEL_Y = 0.075
PANEL_B_DPPI_SCHEMATIC_AXIS_BOUNDS = (-0.015, 0.12, 0.315, 0.76)
PANEL_B_DPPI_TITLE_POSITION = (0.50, 0.99)
PANEL_B_DPPI_OVERLAP_DEFINITION = "same-turn overlap ="
PANEL_B_DPPI_OVERLAP_DEFINITION_POSITION = (0.50, 0.80)
PANEL_B_DPPI_FORMULA_Y = 0.695
PANEL_B_DPPI_FORMULA_NEUTRAL_COLOR = "#404040"
PANEL_B_DPPI_MIN_OUTLINE_COLOR = "#238B45"
PANEL_B_DPPI_MAX_OUTLINE_COLOR = "#6A51A3"
PANEL_B_DPPI_CURVE_AXIS_BOUNDS = (0.08, 0.29, 0.84, 0.28)
PANEL_B_DPPI_RATE_COLORS = ("#000000", "#000000")
PANEL_B_DPPI_GRAY_OVERLAP_COLOR = "#BDBDBD"
PANEL_B_DPPI_GRAY_OVERLAP_ALPHA = 0.55
PANEL_B_DPPI_PATH_COLORED_SECOND_RATE_SCALE = 0.55
PANEL_B_DPPI_RATE_LINEWIDTH = 1.9
PANEL_B_DPPI_OUTLINE_LINEWIDTH = 1.35
PANEL_B_DPPI_MAX_FILL_ALPHA = 0.16
PANEL_B_DPPI_MIN_FILL_ALPHA = 0.24
PANEL_B_DPPI_RATE_LABEL_POSITIONS = ((0.34, 1.20), (0.70, 1.09))
PANEL_B_DPPI_TURN_ICON_BOUNDS = (
    (0.020, 0.575, 0.230, 0.350),
    (0.750, 0.540, 0.230, 0.350),
)
PANEL_B_DPPI_TURN_ICON_TRACK_LINEWIDTH = 0.35
PANEL_B_DPPI_TURN_ICON_TRAJECTORY_LINEWIDTH = 0.55
PANEL_B_DPPI_TURN_ICON_ARROW_SCALE = 4.4
PANEL_B_DPPI_EQUATION = "DPPI = max(left overlap,\nright overlap)"
PANEL_B_DPPI_EQUATION_POSITION = (0.50, 0.08)
PANEL_B_GROUPED_AXIS_BOUNDS = (0.325, 0.32, 0.285, 0.54)
PANEL_B_SCATTER_AXIS_BOUNDS = (0.635, 0.12, 0.360, 0.76)
PANEL_B_DPPI_SCHEMATIC_WITHOUT_GROUP_AXIS_BOUNDS = (0.000, 0.12, 0.380, 0.76)
PANEL_B_SCATTER_WITHOUT_GROUP_AXIS_BOUNDS = (0.425, 0.125, 0.570, 0.80)
PANEL_B_SCATTER_MAIN_TICKS = (0.0, 0.5, 1.0)
PANEL_B_SCATTER_MAIN_TICK_LABELS = ("0", "0.5", "1")
PANEL_C_CROSS_DECODING_AXIS_BOUNDS = (0.06, 0.16, 0.40, 0.66)
PANEL_C_PLACE_DECODING_AXIS_BOUNDS = (0.57, 0.16, 0.39, 0.66)
PANEL_D_MODEL_LABEL_X = PANEL_B_MODEL_LABEL_X
PANEL_D_SCHEMATIC_LABEL_FONTSIZE = 3.8
PANEL_D_TRAIN_TRACK_CENTER_X = PANEL_G_DARK_TRACK_CENTER_X
PANEL_D_PREDICT_TRACK_CENTER_X = PANEL_G_LIGHT_TRACK_CENTER_X
PANEL_D_SHARED_DARK_TRACK_CENTER_X = PANEL_D_TRAIN_TRACK_CENTER_X
PANEL_D_SHARED_PLUS_X = 0.39
PANEL_D_SHARED_SEGMENT_TRACK_CENTER_X = PANEL_G_SEGMENT_MODULATION_TRACK_CENTER_X
PANEL_D_SHARED_ARROW_X = PANEL_G_SHARED_OUTPUT_ARROW_X
PANEL_D_SHARED_ARROW_Y_OFFSET = 0.0
PANEL_D_SHARED_LIGHT_TRACK_CENTER_X = PANEL_D_PREDICT_TRACK_CENTER_X
PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT = -0.105
PANEL_D_INDEPENDENT_TRACK_CENTER_Y = 0.765
PANEL_D_SHARED_TRACK_CENTER_Y = 0.354
PANEL_D_SHARED_PREDICTION_LABEL_Y = 0.124
PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA = PANEL_B_VISUAL_ICON_REGION_FILL_ALPHA
PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS = {
    "left_arm": 1.15,
}
PANEL_D_DARK_PLACE_FIELD_COLORS = ("#D6ECFF", "#1E88E5", "#005CB9")
PANEL_D_LIGHT_PLACE_FIELD_COLORS = ("#FEE08B", "#FDAE61", "#D73027")
PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE = 1.35
PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS = {
    "stim1": {
        "left_arm": PANEL_B_VISUAL_ICON_COLORS["A"],
        "right_arm": PANEL_B_VISUAL_ICON_COLORS["B"],
    },
    "stim2": {
        "left_arm": PANEL_B_VISUAL_ICON_COLORS["B"],
        "right_arm": PANEL_B_VISUAL_ICON_COLORS["A"],
    },
}
PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS = {
    "left_arm": "#E69F00",
}
PANEL_E_RIGHT_ARM_SEGMENT_OUTLINE_COLORS = {
    "right_arm": PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS["left_arm"],
}
PANEL_E_RIGHT_ARM_SEGMENT_OUTLINE_LINEWIDTHS = {
    "right_arm": PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS["left_arm"],
}
PANEL_E_VERTICAL_SHIFT = 0.0
PANEL_2_3_INDEPENDENT_MODEL_COLOR = "#E69F00"
PANEL_2_3_DARK_SCAFFOLD_MODEL_COLOR = "#0072B2"
PANEL_DARK_SCAFFOLD_MODEL_LABEL = "Dark scaffold"
PANEL_B_EXAMPLE_MODEL_COLORS_2_3 = {
    **PANEL_B_EXAMPLE_MODEL_COLORS,
    "visual": PANEL_2_3_INDEPENDENT_MODEL_COLOR,
    PANEL_C_SWAP_MODEL_NAME: PANEL_2_3_DARK_SCAFFOLD_MODEL_COLOR,
}
PANEL_B_EXAMPLE_MODEL_LABELS_2_3 = {
    **PANEL_B_EXAMPLE_MODEL_LABELS,
    PANEL_C_SWAP_MODEL_NAME: PANEL_DARK_SCAFFOLD_MODEL_LABEL,
}
PANEL_C_SWAP_MODEL_COLORS_2_3 = {
    **PANEL_C_SWAP_MODEL_COLORS,
    "visual": PANEL_2_3_INDEPENDENT_MODEL_COLOR,
    PANEL_C_SWAP_MODEL_NAME: PANEL_2_3_DARK_SCAFFOLD_MODEL_COLOR,
}
PANEL_C_SWAP_MODEL_LABELS_2_3 = {
    **PANEL_C_SWAP_MODEL_LABELS,
    PANEL_C_SWAP_MODEL_NAME: PANEL_DARK_SCAFFOLD_MODEL_LABEL,
}
MIN_PUBLICATION_FONTSIZE_PT = 6.0


def __getattr__(name: str) -> Any:
    """Delegate legacy Figure 2 helper lookups to the shared implementation."""
    return getattr(_figure_2_common, name)


def _shift_axis_vertically(ax: Any, dy_figure_fraction: float) -> None:
    """Shift one axis vertically after constrained layout selects its size."""
    if dy_figure_fraction == 0.0:
        return
    box = ax.get_position()
    ax.set_axes_locator(None)
    ax.set_position(
        [
            box.x0,
            box.y0 + dy_figure_fraction,
            box.width,
            box.height,
        ]
    )


def _set_axis_horizontal_bounds(
    ax: Any,
    *,
    left: float,
    width: float,
) -> None:
    """Set one axis horizontal figure bounds after constrained layout."""
    box = ax.get_position()
    ax.set_axes_locator(None)
    ax.set_position([left, box.y0, width, box.height])


def _set_axis_height_preserving_top(ax: Any, height: float) -> None:
    """Set one axis height while preserving its top edge."""
    box = ax.get_position()
    top_y = box.y0 + box.height
    ax.set_axes_locator(None)
    ax.set_position([box.x0, top_y - height, box.width, height])


def _scale_axis_width_from_left(ax: Any, scale: float) -> None:
    """Scale one axis horizontally from its left edge."""
    if scale == 1.0:
        return
    box = ax.get_position()
    ax.set_axes_locator(None)
    ax.set_position([box.x0, box.y0, box.width * scale, box.height])


def _iter_nested_axes(ax: Any) -> Any:
    """Yield all nested inset axes below one axes."""
    for child_ax in getattr(ax, "child_axes", ()):
        yield child_ax
        yield from _iter_nested_axes(child_ax)


def _replace_nested_text(
    ax: Any,
    old_text: str,
    new_text: str,
    *,
    fontsize: float | None = None,
) -> None:
    """Replace matching text in one axes tree."""
    for candidate_ax in (ax, *_iter_nested_axes(ax)):
        for text in (
            candidate_ax.xaxis.label,
            candidate_ax.yaxis.label,
            *candidate_ax.texts,
        ):
            if text.get_text() == old_text:
                text.set_text(new_text)
                if fontsize is not None:
                    text.set_fontsize(fontsize)


def _keep_last_nested_text(
    ax: Any,
    text_value: str,
    *,
    replacement: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Keep only the final matching text in one axes tree."""
    matches = []
    for candidate_ax in (ax, *_iter_nested_axes(ax)):
        for text in (
            candidate_ax.xaxis.label,
            candidate_ax.yaxis.label,
            *candidate_ax.texts,
        ):
            if text.get_text() == text_value:
                matches.append(text)
    for text in matches[:-1]:
        text.set_text("")
    if not matches:
        return
    if replacement is not None:
        matches[-1].set_text(replacement)
    if fontsize is not None:
        matches[-1].set_fontsize(fontsize)


def _set_nested_legend_fontsize(ax: Any, fontsize: float) -> None:
    """Set legend text size throughout one axes tree."""
    for candidate_ax in (ax, *_iter_nested_axes(ax)):
        legend = candidate_ax.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            text.set_fontsize(fontsize)


def _raise_text_to_minimum_fontsize(fig: Any, min_fontsize: float) -> None:
    """Raise final figure text to a minimum size."""
    seen_axes: set[int] = set()

    def _maybe_raise(text: Any) -> None:
        if text is None:
            return
        if text.get_fontsize() < min_fontsize:
            text.set_fontsize(min_fontsize)

    for ax in fig.axes:
        for candidate_ax in (ax, *_iter_nested_axes(ax)):
            axis_id = id(candidate_ax)
            if axis_id in seen_axes:
                continue
            seen_axes.add(axis_id)
            _maybe_raise(candidate_ax.title)
            _maybe_raise(candidate_ax.xaxis.label)
            _maybe_raise(candidate_ax.yaxis.label)
            for tick_label in candidate_ax.get_xticklabels():
                _maybe_raise(tick_label)
            for tick_label in candidate_ax.get_yticklabels():
                _maybe_raise(tick_label)
            for text in candidate_ax.texts:
                _maybe_raise(text)
            legend = candidate_ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    _maybe_raise(text)


def _remove_axis_tick_label_lines(ax: Any, prefixes: Sequence[str]) -> None:
    """Remove tick-label lines that begin with any prefix."""
    cleaned_labels = []
    for tick in ax.get_xticklabels():
        lines = [
            line
            for line in tick.get_text().splitlines()
            if not any(line.startswith(prefix) for prefix in prefixes)
        ]
        cleaned_labels.append("\n".join(lines))
    ax.set_xticklabels(cleaned_labels, fontsize=MIN_PUBLICATION_FONTSIZE_PT)


def _equal_width_row_bounds(
    item_count: int,
    gap: float,
) -> tuple[tuple[float, float], ...]:
    """Return equal-width left and width bounds for one horizontal row."""
    if item_count <= 0:
        return ()
    item_width = (1.0 - float(gap) * (item_count - 1)) / item_count
    return tuple(
        (item_index * (item_width + float(gap)), item_width)
        for item_index in range(item_count)
    )


def plot_panel_a_examples_single_row(
    ax: Any,
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot all Figure 2 Panel A examples in one horizontal row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        return

    column_bounds = _equal_width_row_bounds(
        len(examples),
        PANEL_A_SINGLE_ROW_COLUMN_GAP,
    )
    for example_index, (example, (left, column_width)) in enumerate(
        zip(examples, column_bounds, strict=True),
        start=1,
    ):
        example_ax = ax.inset_axes([left, 0.0, column_width, 1.0])
        plot_kwargs: dict[str, Any] = {
            "title": None,
            "dark_epoch_axis_left": PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT,
            "light_epoch_axis_left": PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT,
            "epoch_axis_width": PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH,
            "schematic_axis_left": PANEL_A_SINGLE_ROW_SCHEMATIC_AXIS_LEFT,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        }
        y_max_override = PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(example_index)
        if y_max_override is not None:
            plot_kwargs["y_max"] = y_max_override
        plot_panel_a_example(example_ax, example, **plot_kwargs)
        rate_axes = [
            child_ax for child_ax in example_ax.child_axes if child_ax.get_xlabel()
        ]
        for rate_ax in rate_axes:
            rate_ax.set_xlabel("")
            rate_ax.tick_params(
                axis="x",
                labelsize=MIN_PUBLICATION_FONTSIZE_PT,
                pad=0.4,
            )
        example_ax.text(
            0.5,
            0.985,
            f"Example cell {example_index}",
            ha="center",
            va="top",
            fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            transform=example_ax.transAxes,
        )
        example_ax.text(
            0.5,
            PANEL_A_SINGLE_ROW_XLABEL_Y,
            "Norm. path progression",
            ha="center",
            va="top",
            fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            transform=example_ax.transAxes,
            clip_on=False,
        )


def _draw_overlap_curve_schematic(
    ax: Any,
    *,
    position: np.ndarray,
    first_rate: np.ndarray,
    second_rate: np.ndarray,
    first_color: str,
    second_color: str,
    gray_overlap_only: bool = False,
) -> None:
    """Draw one schematic same-turn overlap computation."""
    x = np.asarray(position, dtype=float)
    curve_a = np.clip(np.asarray(first_rate, dtype=float), 0.0, None)
    curve_b = np.clip(np.asarray(second_rate, dtype=float), 0.0, None)
    valid = np.isfinite(x) & np.isfinite(curve_a) & np.isfinite(curve_b)
    x = x[valid]
    curve_a = curve_a[valid]
    curve_b = curve_b[valid]
    if x.size < 2:
        return
    scale = max(float(np.nanmax(curve_a)), float(np.nanmax(curve_b)), 1.0)
    display_scale = 1.35
    curve_a = curve_a / scale * display_scale
    curve_b = curve_b / scale * display_scale
    overlap = np.minimum(curve_a, curve_b)
    envelope = np.maximum(curve_a, curve_b)

    if gray_overlap_only:
        ax.fill_between(
            x,
            0.0,
            overlap,
            color=PANEL_B_DPPI_GRAY_OVERLAP_COLOR,
            alpha=PANEL_B_DPPI_GRAY_OVERLAP_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
    else:
        ax.fill_between(
            x,
            0.0,
            envelope,
            color=PANEL_B_DPPI_MAX_OUTLINE_COLOR,
            alpha=PANEL_B_DPPI_MAX_FILL_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
        ax.fill_between(
            x,
            0.0,
            overlap,
            color=PANEL_B_DPPI_MIN_OUTLINE_COLOR,
            alpha=PANEL_B_DPPI_MIN_FILL_ALPHA,
            linewidth=0.0,
            zorder=2,
        )
    ax.plot(
        x,
        curve_a,
        color=first_color,
        linewidth=PANEL_B_DPPI_RATE_LINEWIDTH,
        zorder=3,
    )
    ax.plot(
        x,
        curve_b,
        color=second_color,
        linewidth=PANEL_B_DPPI_RATE_LINEWIDTH,
        zorder=4,
    )
    if not gray_overlap_only:
        ax.plot(
            x,
            envelope,
            color=PANEL_B_DPPI_MAX_OUTLINE_COLOR,
            linewidth=PANEL_B_DPPI_OUTLINE_LINEWIDTH,
            zorder=5,
        )
        ax.plot(
            x,
            overlap,
            color=PANEL_B_DPPI_MIN_OUTLINE_COLOR,
            linewidth=PANEL_B_DPPI_OUTLINE_LINEWIDTH,
            zorder=6,
        )
    for label, color, (label_x, label_y) in (
        ("r1", first_color, PANEL_B_DPPI_RATE_LABEL_POSITIONS[0]),
        ("r2", second_color, PANEL_B_DPPI_RATE_LABEL_POSITIONS[1]),
    ):
        ax.text(
            label_x,
            label_y,
            label,
            color=color,
            ha="center",
            va="center",
            fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            fontweight="bold",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.42)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.45)


def _draw_panel_b_dppi_turn_icon(
    ax: Any,
    *,
    bounds: tuple[float, float, float, float],
    trajectory_name: str,
) -> None:
    """Draw one small physical-turn W-track icon for the DPPI schematic."""
    icon_ax = ax.inset_axes(bounds)
    icon_ax.patch.set_visible(False)
    icon_ax.set_zorder(8)
    draw_w_track_schematic(
        icon_ax,
        trajectory_name=trajectory_name,
        arrow_color=PANEL_TRAJECTORY_COLORS[trajectory_name],
        track_linewidth=PANEL_B_DPPI_TURN_ICON_TRACK_LINEWIDTH,
        trajectory_linewidth=PANEL_B_DPPI_TURN_ICON_TRAJECTORY_LINEWIDTH,
        arrow_mutation_scale=PANEL_B_DPPI_TURN_ICON_ARROW_SCALE,
    )


def _add_centered_colored_text_fragments(
    ax: Any,
    *,
    y: float,
    fragments: Sequence[tuple[str, str]],
    fontsize: float,
) -> None:
    """Draw text fragments centered as one line, allowing per-fragment colors."""
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    widths = []
    scratch_texts = []
    for text_value, _color in fragments:
        scratch_text = ax.text(
            0.0,
            y,
            text_value,
            fontsize=fontsize,
            ha="left",
            va="top",
            transform=ax.transAxes,
            alpha=0.0,
        )
        scratch_texts.append(scratch_text)
        widths.append(scratch_text.get_window_extent(renderer).width)
    for scratch_text in scratch_texts:
        scratch_text.remove()

    axes_width = ax.get_window_extent(renderer).width
    total_width_axes = sum(widths) / axes_width
    x_position = 0.5 - total_width_axes / 2.0
    for (text_value, color), width in zip(fragments, widths, strict=True):
        ax.text(
            x_position,
            y,
            text_value,
            fontsize=fontsize,
            color=color,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        x_position += width / axes_width


def plot_panel_b_dppi_schematic(
    ax: Any,
    example: dict[str, Any],
    *,
    style: str = "legacy",
) -> None:
    """Draw a compact schematic defining DPPI as max same-turn overlap."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    trajectories = tuple(str(trajectory) for trajectory in example["trajectories"])
    if len(trajectories) < 2:
        ax.text(0.5, 0.5, "No example\ncurves", ha="center", va="center")
        return
    first_trajectory, second_trajectory = trajectories[:2]
    if style not in {"legacy", "path_colored_gray_overlap"}:
        raise ValueError(f"Unsupported DPPI schematic style {style!r}.")
    use_path_colored_style = style == "path_colored_gray_overlap"
    curve_position = np.linspace(0.0, 1.0, 121)
    first_rate = np.exp(-0.5 * ((curve_position - 0.44) / 0.075) ** 2)
    second_rate_scale = (
        PANEL_B_DPPI_PATH_COLORED_SECOND_RATE_SCALE
        if use_path_colored_style
        else 0.92
    )
    second_rate = second_rate_scale * np.exp(
        -0.5 * ((curve_position - 0.57) / 0.075) ** 2
    )
    rate_colors = (
        (
            PANEL_TRAJECTORY_COLORS[first_trajectory],
            PANEL_TRAJECTORY_COLORS[second_trajectory],
        )
        if use_path_colored_style
        else PANEL_B_DPPI_RATE_COLORS
    )
    ax.text(
        PANEL_B_DPPI_TITLE_POSITION[0],
        PANEL_B_DPPI_TITLE_POSITION[1],
        "DPP index",
        ha="center",
        va="top",
        fontsize=5.8,
        fontweight="bold",
    )
    ax.text(
        PANEL_B_DPPI_OVERLAP_DEFINITION_POSITION[0],
        PANEL_B_DPPI_OVERLAP_DEFINITION_POSITION[1],
        PANEL_B_DPPI_OVERLAP_DEFINITION,
        ha="center",
        va="top",
        fontsize=4.7,
        linespacing=0.88,
    )
    _add_centered_colored_text_fragments(
        ax,
        y=PANEL_B_DPPI_FORMULA_Y,
        fragments=(
            ("∫ ", PANEL_B_DPPI_FORMULA_NEUTRAL_COLOR),
            (
                "min(r1,r2)",
                (
                    PANEL_B_DPPI_FORMULA_NEUTRAL_COLOR
                    if use_path_colored_style
                    else PANEL_B_DPPI_MIN_OUTLINE_COLOR
                ),
            ),
            (" / ∫ ", PANEL_B_DPPI_FORMULA_NEUTRAL_COLOR),
            (
                "max(r1,r2)",
                (
                    PANEL_B_DPPI_FORMULA_NEUTRAL_COLOR
                    if use_path_colored_style
                    else PANEL_B_DPPI_MAX_OUTLINE_COLOR
                ),
            ),
        ),
        fontsize=MIN_PUBLICATION_FONTSIZE_PT,
    )
    curve_ax = ax.inset_axes(PANEL_B_DPPI_CURVE_AXIS_BOUNDS)
    _draw_overlap_curve_schematic(
        curve_ax,
        position=curve_position,
        first_rate=first_rate,
        second_rate=second_rate,
        first_color=rate_colors[0],
        second_color=rate_colors[1],
        gray_overlap_only=use_path_colored_style,
    )
    for trajectory_name, bounds in zip(
        (first_trajectory, second_trajectory),
        PANEL_B_DPPI_TURN_ICON_BOUNDS,
        strict=True,
    ):
        _draw_panel_b_dppi_turn_icon(
            curve_ax,
            bounds=bounds,
            trajectory_name=trajectory_name,
        )

    ax.text(
        PANEL_B_DPPI_EQUATION_POSITION[0],
        PANEL_B_DPPI_EQUATION_POSITION[1],
        PANEL_B_DPPI_EQUATION,
        ha="center",
        va="center",
        fontsize=MIN_PUBLICATION_FONTSIZE_PT,
        linespacing=0.88,
    )


def _load_panel_b_stable_units_for_epoch(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    epoch_type: str,
    region: str,
    min_movement_firing_rate_hz: float | None,
    min_stability_correlation: float,
) -> Any:
    """Return active units with stable tuning in at least one DPP trajectory."""
    import pandas as pd

    path = get_stability_table_path(data_root, animal_name, date)
    if not path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(path)
    required_columns = [
        "unit",
        "region",
        "epoch",
        "trajectory_type",
        "stability_correlation",
    ]
    if min_movement_firing_rate_hz is not None:
        required_columns.append("firing_rate_hz")
    missing_columns = [column for column in required_columns if column not in table]
    if missing_columns:
        raise ValueError(
            f"Tuning stability table {path} is missing columns {missing_columns!r}."
        )

    rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & (
            table["trajectory_type"]
            .astype(str)
            .isin(PANEL_B_TUNING_CORRELATION_TRAJECTORIES)
        )
    ].copy()
    rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
    active_units = select_units_by_saved_movement_firing_rate(
        rows,
        min_movement_firing_rate_hz,
    )
    rows["stability_correlation"] = pd.to_numeric(
        rows["stability_correlation"],
        errors="coerce",
    )
    rows = rows[
        np.isfinite(rows["unit"].to_numpy(dtype=float))
        & rows["unit"].isin(active_units)
        & np.isfinite(rows["stability_correlation"].to_numpy(dtype=float))
        & (
            rows["stability_correlation"].to_numpy(dtype=float)
            >= float(min_stability_correlation)
        )
    ].copy()
    stability_column = f"max_{epoch_type}_tuning_stability_correlation"
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "unit",
                stability_column,
            ]
        )
    rows["unit"] = rows["unit"].astype(int)
    stable_units = (
        rows.groupby("unit", as_index=False, observed=False)[
            "stability_correlation"
        ]
        .max()
        .rename(columns={"stability_correlation": stability_column})
    )
    return stable_units.assign(animal_name=animal_name, date=date)


def filter_panel_b_overlap_by_even_odd_stability(
    table: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_movement_firing_rate_hz: float | None = (
        PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_stability_correlation: float,
) -> Any:
    """Keep Panel B units active and stable in both light and dark epochs."""
    import pandas as pd

    if table is None or not len(table):
        return table

    stable_tables = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        resolved_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        resolved_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        dark_stable = _load_panel_b_stable_units_for_epoch(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=resolved_dark_epoch,
            epoch_type="dark",
            region=region,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            min_stability_correlation=min_stability_correlation,
        )
        light_stable = _load_panel_b_stable_units_for_epoch(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=resolved_light_epoch,
            epoch_type="light",
            region=region,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            min_stability_correlation=min_stability_correlation,
        )
        stable_tables.append(
            dark_stable.merge(
                light_stable,
                on=["animal_name", "date", "unit"],
                how="inner",
            )
        )

    if not stable_tables:
        return table.iloc[0:0].copy()

    stable_units = pd.concat(stable_tables, axis=0, ignore_index=True)
    if stable_units.empty:
        return table.iloc[0:0].copy()

    filtered = table.copy()
    filtered["unit"] = pd.to_numeric(filtered["unit"], errors="coerce")
    filtered = filtered[np.isfinite(filtered["unit"].to_numpy(dtype=float))].copy()
    filtered["unit"] = filtered["unit"].astype(int)
    return filtered.merge(
        stable_units,
        on=["animal_name", "date", "unit"],
        how="inner",
    )


def _format_panel_b_dppi_scatter_axes(ax: Any) -> None:
    """Format Panel B DPPI scatter marginals for the 6 pt text floor."""
    for child_ax in ax.child_axes:
        if child_ax.get_xlabel() == "Dark DPP\noverlap":
            child_ax.set_xlabel(
                "Dark DPPI",
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
                labelpad=1.0,
            )
            child_ax.xaxis.set_label_coords(0.50, -0.18)
            child_ax.set_ylabel(
                "Light DPPI",
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
                labelpad=1.0,
            )
            child_ax.set_xticks(PANEL_B_SCATTER_MAIN_TICKS)
            child_ax.set_yticks(PANEL_B_SCATTER_MAIN_TICKS)
            child_ax.set_xticklabels(
                PANEL_B_SCATTER_MAIN_TICK_LABELS,
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            )
            child_ax.set_yticklabels(
                PANEL_B_SCATTER_MAIN_TICK_LABELS,
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            )
            child_ax.tick_params(labelsize=MIN_PUBLICATION_FONTSIZE_PT, pad=0.8)
            for text in child_ax.texts:
                text.set_fontsize(max(text.get_fontsize(), MIN_PUBLICATION_FONTSIZE_PT))
        elif child_ax.get_xlabel() == "Frac.":
            child_ax.set_xlabel(
                "Frac.",
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
                labelpad=1.0,
            )
            child_ax.xaxis.set_label_coords(0.50, -0.18)
            child_ax.set_xticks((0.0, 0.1))
            child_ax.set_xticklabels(
                ("0", "0.1"),
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            )
            child_ax.tick_params(axis="x", labelbottom=True, bottom=True, pad=0.8)

        if child_ax.get_ylabel() == "Frac.":
            child_ax.set_ylabel(
                "Frac.",
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
                labelpad=0.4,
            )
            child_ax.yaxis.set_label_coords(-0.103, 0.50)
            child_ax.set_yticks((0.0, 0.1))
            child_ax.set_yticklabels(
                ("0", "0.1"),
                fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            )
            child_ax.tick_params(axis="y", labelleft=True, left=True, pad=0.6)


def plot_panel_b_dpp_overlap_with_schematic(
    ax: Any,
    overlap_table: Any,
    *,
    example: dict[str, Any],
    low_threshold: float,
    high_threshold: float,
    show_grouped: bool = True,
    show_scatter_linear_fit: bool = False,
    show_scatter_r2: bool = False,
    scatter_equal_aspect: bool = False,
    schematic_style: str = "legacy",
) -> None:
    """Plot Figure 2 Panel B with DPPI schematic and overlap summaries."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    schematic_bounds = (
        PANEL_B_DPPI_SCHEMATIC_AXIS_BOUNDS
        if show_grouped
        else PANEL_B_DPPI_SCHEMATIC_WITHOUT_GROUP_AXIS_BOUNDS
    )
    schematic_ax = ax.inset_axes(schematic_bounds)
    scatter_bounds = (
        PANEL_B_SCATTER_AXIS_BOUNDS
        if show_grouped
        else PANEL_B_SCATTER_WITHOUT_GROUP_AXIS_BOUNDS
    )
    if show_grouped:
        grouped_ax = ax.inset_axes(PANEL_B_GROUPED_AXIS_BOUNDS)
    scatter_ax = ax.inset_axes(scatter_bounds)
    plot_panel_b_dppi_schematic(
        schematic_ax,
        example,
        style=schematic_style,
    )
    if show_grouped:
        plot_panel_b_dpp_overlap_grouped(
            grouped_ax,
            overlap_table,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        for text in grouped_ax.texts:
            if "> 0.5" in text.get_text():
                text.set_text("")
    plot_panel_b_dpp_overlap_scatter(
        scatter_ax,
        overlap_table,
        title=None,
        show_linear_fit=show_scatter_linear_fit,
        show_r2_annotation=show_scatter_r2,
        equal_aspect=scatter_equal_aspect,
    )
    _format_panel_b_dppi_scatter_axes(scatter_ax)
    if show_grouped:
        grouped_ax.set_title("")
        grouped_ax.set_xlabel(
            "Dark DPPI",
            fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            labelpad=0.6,
        )
        grouped_ax.set_ylabel(
            "Light DPPI",
            fontsize=MIN_PUBLICATION_FONTSIZE_PT,
            labelpad=1.0,
        )
        low_threshold_label = f"{float(low_threshold):g}"
        high_threshold_label = f"{float(high_threshold):g}"
        grouped_labels = (
            f"Low\n<{low_threshold_label}",
            f"Mid\n{low_threshold_label}-{high_threshold_label}",
            f"High\n>={high_threshold_label}",
        )
        grouped_ax.set_xticklabels(grouped_labels, fontsize=MIN_PUBLICATION_FONTSIZE_PT)
        _remove_axis_tick_label_lines(grouped_ax, prefixes=("n=",))
        grouped_ax.xaxis.set_label_coords(0.5, -0.26)
        grouped_ax.tick_params(axis="x", pad=1.0)


def plot_panel_c_cross_and_place_decoding(
    ax: Any,
    decoding_error_table: Any,
) -> None:
    """Plot Figure 2 Panel C as compact cross-path and place decoding."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    cross_ax = ax.inset_axes(PANEL_C_CROSS_DECODING_AXIS_BOUNDS)
    place_ax = ax.inset_axes(PANEL_C_PLACE_DECODING_AXIS_BOUNDS)
    _plot_panel_e_cross_axis(cross_ax, decoding_error_table)
    cross_ax.set_title("Cross-path\ndecoding", fontsize=5.8, pad=1.5)
    _plot_panel_e_place_axis(place_ax, decoding_error_table, ylabel=None)
    place_ax.set_title("Path-specific\nplace decoding", fontsize=5.8, pad=1.0)


def _draw_panel_d_swap_schematic(
    ax: Any,
    *,
    track_size: tuple[float, float] | None = None,
    show_dark_track_labels: bool = False,
    model_name: str,
    model_labels: Mapping[str, str] | None = None,
    prediction_label_fontsize: float = PANEL_C_PREDICTION_LABEL_FONTSIZE,
) -> None:
    """Draw Panel D swap schematic with a panel-C-like dark-scaffold row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    shared_model_label = (
        "Dark scaffold\nmodel"
        if str(model_name) == "task_segment_bump"
        else f"{_panel_model_label(str(model_name), model_labels)}\nmodel"
    )
    def _bounds_from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> list[float]:
        return [center_x - width / 2.0, center_y - height / 2.0, width, height]

    light_bounds = {
        "width": track_size[0] if track_size is not None else 0.38,
        "height": track_size[1] if track_size is not None else 0.23,
    }
    dark_bounds = {
        "width": track_size[0] if track_size is not None else 0.34,
        "height": track_size[1] if track_size is not None else 0.21,
    }
    train_predict_midpoint_x = 0.5 * (
        PANEL_D_TRAIN_TRACK_CENTER_X + PANEL_D_PREDICT_TRACK_CENTER_X
    )

    ax.text(
        PANEL_D_TRAIN_TRACK_CENTER_X,
        0.98,
        "Train: AB",
        ha="center",
        va="top",
        fontsize=5.8,
    )
    ax.text(
        PANEL_D_PREDICT_TRACK_CENTER_X,
        0.98,
        "Predict: BA",
        ha="center",
        va="top",
        fontsize=5.8,
    )
    ax.text(
        PANEL_D_MODEL_LABEL_X,
        PANEL_D_INDEPENDENT_TRACK_CENTER_Y,
        "Independent\nmodel",
        ha="center",
        va="center",
        fontsize=PANEL_B_MODEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.text(
        PANEL_D_MODEL_LABEL_X,
        PANEL_D_SHARED_TRACK_CENTER_Y,
        shared_model_label,
        ha="center",
        va="center",
        fontsize=PANEL_B_MODEL_LABEL_FONTSIZE,
        fontweight="bold",
    )

    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                PANEL_D_TRAIN_TRACK_CENTER_X,
                PANEL_D_INDEPENDENT_TRACK_CENTER_Y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        highlighted_segments=(3,),
        label_fontsize=PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim1"],
        region_fill_alpha=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D_LIGHT_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                PANEL_D_PREDICT_TRACK_CENTER_X,
                PANEL_D_INDEPENDENT_TRACK_CENTER_Y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        highlighted_segments=(3,),
        label_fontsize=PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim2"],
        region_fill_alpha=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D_LIGHT_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="right_arm",
    )
    ax.text(
        train_predict_midpoint_x,
        PANEL_D_INDEPENDENT_TRACK_CENTER_Y,
        "Visual landmarks\ndrive neural activity",
        ha="center",
        va="center",
        fontsize=prediction_label_fontsize,
        linespacing=0.95,
    )

    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                PANEL_D_SHARED_DARK_TRACK_CENTER_X,
                PANEL_D_SHARED_TRACK_CENTER_Y,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
        show_labels=show_dark_track_labels,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        show_place_field_blob=True,
        place_field_colors=PANEL_D_DARK_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="right_arm",
    )
    ax.text(
        PANEL_D_SHARED_PLUS_X,
        PANEL_D_SHARED_TRACK_CENTER_Y,
        "+",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                PANEL_D_SHARED_SEGMENT_TRACK_CENTER_X,
                PANEL_D_SHARED_TRACK_CENTER_Y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="segment_modulation",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        oval_regions=["left_arm"],
        fill_oval_regions=False,
        label_fontsize=PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=PANEL_B_VISUAL_ICON_COLORS,
        segment_outline_colors=PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS,
        segment_outline_linewidths=PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS,
    )
    ax.annotate(
        "",
        xy=(
            PANEL_D_SHARED_ARROW_X[1],
            PANEL_D_SHARED_TRACK_CENTER_Y + PANEL_D_SHARED_ARROW_Y_OFFSET,
        ),
        xytext=(
            PANEL_D_SHARED_ARROW_X[0],
            PANEL_D_SHARED_TRACK_CENTER_Y + PANEL_D_SHARED_ARROW_Y_OFFSET,
        ),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 0.8,
            "mutation_scale": 8.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                PANEL_D_SHARED_LIGHT_TRACK_CENTER_X,
                PANEL_D_SHARED_TRACK_CENTER_Y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        oval_regions=["right_arm"],
        fill_oval_regions=False,
        label_fontsize=PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=PANEL_B_VISUAL_ICON_COLORS,
        region_fill_colors=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim2"],
        region_fill_alpha=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        show_place_field_blob=True,
        place_field_colors=PANEL_D_DARK_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="right_arm",
        segment_outline_colors=PANEL_E_RIGHT_ARM_SEGMENT_OUTLINE_COLORS,
        segment_outline_linewidths=PANEL_E_RIGHT_ARM_SEGMENT_OUTLINE_LINEWIDTHS,
    )
    ax.text(
        train_predict_midpoint_x,
        PANEL_D_SHARED_PREDICTION_LABEL_Y,
        "Dark scaffold + visual gain\ndrive neural activity",
        ha="center",
        va="bottom",
        fontsize=prediction_label_fontsize,
        linespacing=0.95,
    )


def _summarize_swap_delta_across_trajectories(swap_delta_table: Any) -> Any:
    """Return complete held-out mean swapped-segment delta LL rows per unit."""
    import pandas as pd

    table = _filter_panel_h_heldout_delta(swap_delta_table)
    if table is None or "delta_ll_bits_per_spike" not in table or not len(table):
        return pd.DataFrame(columns=["mean_delta"])

    table = table.copy()
    table["delta_ll_bits_per_spike"] = pd.to_numeric(
        table["delta_ll_bits_per_spike"],
        errors="coerce",
    )
    finite_mask = np.isfinite(table["delta_ll_bits_per_spike"].to_numpy(dtype=float))
    table = table[finite_mask].copy()
    if table.empty:
        return pd.DataFrame(columns=["mean_delta"])

    if "trajectory" in table:
        table = table[
            table["trajectory"].astype(str).isin(PANEL_H_DELTA_TRAJECTORIES)
        ].copy()
    key_columns = [
        column
        for column in (
            "animal_name",
            "date",
            "region",
            "dark_epoch",
            "unit",
            "model_name",
        )
        if column in table
    ]
    if not key_columns:
        values = table["delta_ll_bits_per_spike"].to_numpy(dtype=float)
        return pd.DataFrame({"mean_delta": values[np.isfinite(values)]})

    grouped = table.groupby(key_columns, dropna=False)
    if "trajectory" in table:
        summary = grouped.agg(
            mean_delta=("delta_ll_bits_per_spike", "mean"),
            trajectory_count=("trajectory", lambda values: values.astype(str).nunique()),
        )
        summary = summary[
            summary["trajectory_count"] >= len(PANEL_H_DELTA_TRAJECTORIES)
        ]
    else:
        summary = grouped.agg(mean_delta=("delta_ll_bits_per_spike", "mean"))
    summary = summary.reset_index()
    return summary[
        np.isfinite(summary["mean_delta"].to_numpy(dtype=float))
    ].copy()


def _mean_swap_delta_across_trajectories(swap_delta_table: Any) -> np.ndarray:
    """Return one held-out mean swapped-segment delta LL per complete unit."""
    summary = _summarize_swap_delta_across_trajectories(swap_delta_table)
    return np.asarray(summary["mean_delta"], dtype=float)


def plot_panel_d_mean_swap_delta_axis(
    ax: Any,
    swap_delta_table: Any,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot mean held-out delta LL averaged over four trajectories."""
    visual_color = _panel_model_color("visual", model_colors)
    model_color = _panel_model_color(model_name, model_colors)
    model_label = _panel_model_label(model_name, model_labels)
    summary = _summarize_swap_delta_across_trajectories(swap_delta_table)
    values = np.asarray(summary["mean_delta"], dtype=float)
    x_limits = PANEL_H_DELTA_X_LIMITS
    bin_edges = np.round(np.arange(x_limits[0], x_limits[1] + 0.05, 0.1), 10)

    ax.axvspan(
        x_limits[0],
        0.0,
        color=NEUTRAL_COLORS["dark_epoch_background"],
        alpha=0.65,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    if values.size:
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=model_color,
            **EMPHASIS_HISTOGRAM_KWARGS,
            zorder=2,
        )
        ax.text(
            0.03,
            0.97,
            "Indep. better",
            ha="left",
            va="top",
            fontsize=5.5,
            color=visual_color,
            transform=ax.transAxes,
        )
        ax.text(
            PANEL_E_RIGHT_ANNOTATION_X,
            0.97,
            f"{model_label}\nbetter",
            ha="left",
            va="top",
            fontsize=5.5,
            color=model_color,
            transform=ax.transAxes,
        )
        ax.text(
            PANEL_E_RIGHT_SUMMARY_POSITION[0],
            PANEL_E_RIGHT_SUMMARY_POSITION[1],
            _format_delta_advantage_summary(values, include_median=False),
            ha="right",
            va="bottom",
            fontsize=4.8,
            color=model_color,
            transform=ax.transAxes,
        )
        ax.text(
            0.03,
            0.06,
            _format_cell_animal_count(
                summary,
                value_column="mean_delta",
            ),
            ha="left",
            va="bottom",
            fontsize=4.8,
            color="0.25",
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No mean\nvalues", ha="center", va="center", fontsize=5.0)

    ax.set_xlim(*x_limits)
    ax.set_xlabel(DELTA_LOG_LIKELIHOOD_AXIS_LABEL, fontsize=7, labelpad=2)
    ax.set_ylabel("Fraction", fontsize=7, labelpad=2)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.0, length=1.2, pad=0.8)


def plot_panel_d_compact_swap_delta(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot Figure 2 Panel E with three examples and one mean histogram."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes(PANEL_E_SCHEMATIC_AXIS_BOUNDS)
    result_ax = ax.inset_axes(PANEL_E_DELTA_AXIS_BOUNDS)
    result_ax.set_xlim(0.0, 1.0)
    result_ax.set_ylim(0.0, 1.0)
    result_ax.axis("off")

    _draw_panel_d_swap_schematic(
        schematic_ax,
        track_size=PANEL_D_SCHEMATIC_TRACK_SIZE,
        show_dark_track_labels=True,
        model_name=model_name,
        model_labels=model_labels,
        prediction_label_fontsize=PANEL_C_PREDICTION_LABEL_FONTSIZE,
    )

    examples = list(swap_examples.values()) if isinstance(swap_examples, dict) else list(
        swap_examples or []
    )
    example_delta_label_positions = PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS
    example_delta_label_vertical_alignments = (
        PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS
    )
    for example_index, bounds in enumerate(PANEL_E_EXAMPLE_SLOT_BOUNDS):
        example_ax = result_ax.inset_axes(bounds)
        example = examples[example_index] if example_index < len(examples) else None
        _plot_panel_h_switched_segment_example(
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
            icon_bounds=PANEL_E_EXAMPLE_ICON_BOUNDS,
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
        _set_nested_legend_fontsize(example_ax, 3.9)
        _replace_nested_text(
            example_ax,
            "Norm. path progression",
            "Norm.\npath progression",
            fontsize=MIN_PUBLICATION_FONTSIZE_PT,
        )

    histogram_ax = result_ax.inset_axes(PANEL_E_MEAN_DELTA_AXIS_BOUNDS)
    plot_panel_d_mean_swap_delta_axis(
        histogram_ax,
        swap_delta_table,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def plot_panel_c_model_architecture_row(
    ax: Any,
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot Figure 2 Panel D with schematics left and examples right."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes(PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS)
    _plot_panel_g_architecture_schematic(
        schematic_ax,
        independent_track_center_y=(
            PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y
            + PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
        ),
        shared_track_center_y=(
            PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
            + PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
        ),
        track_size=PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE,
        independent_basis_icon_scale=PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
        independent_basis_label=PANEL_B_INDEPENDENT_BASIS_LABEL,
        independent_basis_label_y=(
            PANEL_G_INDEPENDENT_BASIS_LABEL_Y
            + PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
        ),
        show_dark_track_labels=True,
        field_label_y=(
            PANEL_B_FIELD_LABEL_Y + PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
        ),
        model_label_x=PANEL_B_MODEL_LABEL_X,
        model_label_fontsize=PANEL_B_MODEL_LABEL_FONTSIZE,
        shared_model_label="Dark scaffold\nmodel",
        component_label_fontsize=PANEL_B_COMPONENT_LABEL_FONTSIZE,
        segment_modulation_label_y=(
            PANEL_B_SEGMENT_MODULATION_LABEL_Y
            + PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
        ),
        segment_modulation_label=PANEL_B_SEGMENT_MODULATION_LABEL,
        fill_oval_regions=False,
        independent_light_region_fill_colors=(
            PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim1"]
        ),
        independent_light_region_fill_alpha=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        independent_light_label_colors=PANEL_B_VISUAL_ICON_COLORS,
        shared_light_region_fill_colors=(
            PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim1"]
        ),
        shared_light_region_fill_alpha=PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        shared_light_label_colors=PANEL_B_VISUAL_ICON_COLORS,
        dark_basis_edge_color="white",
        dark_basis_fill_color="none",
        dark_basis_fill_alpha=1.0,
        dark_basis_linewidth=0.30,
        show_place_field_blobs=True,
        independent_dark_place_field_colors=PANEL_D_DARK_PLACE_FIELD_COLORS,
        independent_light_place_field_colors=PANEL_D_LIGHT_PLACE_FIELD_COLORS,
        shared_dark_place_field_colors=PANEL_D_DARK_PLACE_FIELD_COLORS,
        shared_light_place_field_colors=PANEL_D_DARK_PLACE_FIELD_COLORS,
        place_field_blob_size_scale=PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        segment_gain_outline_colors=(
            PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS
        ),
        segment_gain_outline_linewidths=PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS,
        segment_gain_label_colors=PANEL_B_VISUAL_ICON_COLORS,
    )

    example_ax = ax.inset_axes(PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS)
    _plot_panel_g_example_columns(
        example_ax,
        examples,
        field_y=PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_Y,
        field_height=PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_HEIGHT,
        icon_bounds=PANEL_C_SIDE_BY_SIDE_EXAMPLE_ICON_BOUNDS,
        xlabel_y=PANEL_C_SIDE_BY_SIDE_EXAMPLE_XLABEL_Y,
        column_width=PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_WIDTH,
        column_gap=PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_GAP,
        plot_left_offset=PANEL_C_SIDE_BY_SIDE_EXAMPLE_PLOT_LEFT_OFFSET,
        field_width=PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_WIDTH,
        field_gap=PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_GAP,
        layout="rows",
        row_height=PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_HEIGHT,
        row_gap=PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_GAP,
        show_ylabels_for_all_examples=True,
        show_epoch_titles=False,
        show_light_yticklabels=False,
        model_colors=PANEL_B_EXAMPLE_MODEL_COLORS_2_3,
        model_labels=PANEL_B_EXAMPLE_MODEL_LABELS_2_3,
    )
    _replace_nested_text(
        example_ax,
        "Norm. path progression",
        "Norm. path progression",
        fontsize=MIN_PUBLICATION_FONTSIZE_PT,
    )
    _replace_nested_text(
        example_ax,
        "Norm. goal progression",
        "Norm. path progression",
        fontsize=MIN_PUBLICATION_FONTSIZE_PT,
    )
    _set_nested_legend_fontsize(example_ax, 3.9)


def make_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    dark_tuning_correlation_threshold: float = (
        PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
    ),
    high_dark_tuning_correlation_threshold: float = (
        PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
    ),
) -> Path:
    """Build and save Figure 2."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_glm_payload = load_panel_glm_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        swap_delta_min_movement_firing_rate_hz=(
            PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        swap_delta_min_tuning_stability_correlation=(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        swap_model_name=PANEL_C_SWAP_MODEL_NAME,
        swap_example_count=len(PANEL_C_SWAP_EXAMPLES),
        swap_requested_examples=PANEL_C_SWAP_EXAMPLES,
        dark_light_requested_examples=PANEL_C_DARK_LIGHT_EXAMPLES,
    )
    panel_a_examples = [
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
        for animal_name, date, region, unit_id, trajectories in (
            FIGURE_2_PANEL_A_EXAMPLES
        )
    ]
    panel_b_overlap_table = load_panel_b_tuning_overlap_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_b_overlap_table = filter_panel_b_overlap_by_even_odd_stability(
        panel_b_overlap_table,
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_movement_firing_rate_hz=(
            PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        min_stability_correlation=(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    )
    panel_e_decoding_error_table = load_panel_e_decoding_error_table(
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
    fig.get_layout_engine().set(**CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS)
    outer_grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            PANEL_A_SINGLE_ROW_HEIGHT_MM,
            PANEL_BC_QUANT_ROW_HEIGHT_MM,
            PANEL_D_ROW_HEIGHT_MM,
            PANEL_E_ROW_HEIGHT_MM,
        ],
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    quant_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_BC_ROW_WIDTH_RATIOS,
        wspace=PANEL_BC_ROW_WSPACE,
    )
    panel_b_axis = fig.add_subplot(quant_grid[0, 0])
    panel_c_axis = fig.add_subplot(quant_grid[0, 1])
    panel_d_axis = fig.add_subplot(outer_grid[2, 0])
    panel_e_axis = fig.add_subplot(outer_grid[3, 0])

    plot_panel_a_examples_single_row(
        panel_a_axis,
        panel_a_examples,
    )
    plot_panel_b_dpp_overlap_with_schematic(
        panel_b_axis,
        panel_b_overlap_table,
        example=panel_a_examples[0],
        low_threshold=dark_tuning_correlation_threshold,
        high_threshold=high_dark_tuning_correlation_threshold,
    )
    plot_panel_c_cross_and_place_decoding(panel_c_axis, panel_e_decoding_error_table)
    plot_panel_c_model_architecture_row(
        panel_d_axis,
        panel_glm_payload["dark_light_examples"],
    )
    plot_panel_d_compact_swap_delta(
        panel_e_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
        model_name=PANEL_C_SWAP_MODEL_NAME,
        model_colors=PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=PANEL_C_SWAP_MODEL_LABELS_2_3,
    )

    label_axis(panel_a_axis, "A", x=-0.02, y=PANEL_A_LABEL_Y)
    panel_a_label = panel_a_axis.texts[-1]
    panel_a_axis.set_title(
        "Example DPP cells in dark and light",
        fontsize=8,
        pad=PANEL_A_TITLE_PAD,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=PANEL_B_LABEL_Y, va="baseline")
    panel_b_label = panel_b_axis.texts[-1]
    panel_b_title = panel_b_axis.set_title(
        "Dark and light DPP coding",
        fontsize=8,
        pad=PANEL_B_TITLE_PAD,
    )
    label_axis(panel_c_axis, "C", x=-0.035, y=PANEL_B_LABEL_Y, va="baseline")
    panel_c_label = panel_c_axis.texts[-1]
    label_axis(panel_d_axis, "D", x=-0.02, y=PANEL_BC_LABEL_Y)
    panel_d_label = panel_d_axis.texts[-1]
    label_axis(panel_e_axis, "E", x=-0.02, y=PANEL_BC_LABEL_Y)
    panel_e_label = panel_e_axis.texts[-1]
    panel_c_title = panel_c_axis.set_title(
        "Dark and light decoding comparison",
        fontsize=8,
        pad=PANEL_B_TITLE_PAD,
    )
    panel_d_title = panel_d_axis.set_title(
        "Two models that relate dark and light activity",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )
    panel_e_title = panel_e_axis.set_title(
        "Predicting activity in held-out light epoch",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )

    _raise_text_to_minimum_fontsize(fig, MIN_PUBLICATION_FONTSIZE_PT)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _set_axis_horizontal_bounds(
        panel_a_axis,
        left=PANEL_A_HORIZONTAL_AXIS_BOUNDS[0],
        width=PANEL_A_HORIZONTAL_AXIS_BOUNDS[1],
    )
    panel_a_axis_height = panel_a_axis.get_position().height
    _set_axis_height_preserving_top(panel_b_axis, panel_a_axis_height)
    _set_axis_height_preserving_top(panel_c_axis, panel_a_axis_height)
    _scale_axis_width_from_left(panel_b_axis, PANEL_B_HORIZONTAL_WIDTH_SCALE)
    _shift_axis_horizontally(panel_e_axis, PANEL_E_HORIZONTAL_SHIFT)
    _shift_axis_vertically(panel_e_axis, PANEL_E_VERTICAL_SHIFT)
    fig.canvas.draw()
    _align_text_to_reference_display_x(panel_b_label, panel_a_label)
    _align_text_to_reference_display_x(panel_d_label, panel_a_label)
    _align_text_to_reference_display_x(panel_e_label, panel_a_label)
    panel_e_title.set_position((0.5, panel_e_title.get_position()[1]))
    _align_texts_to_reference_display_y((panel_e_title, panel_e_label))
    _align_texts_to_reference_display_y((panel_d_title, panel_d_label))
    _align_texts_to_reference_display_y(
        (panel_b_title, panel_b_label, panel_c_title, panel_c_label)
    )

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 2 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 2."
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
            "Dark tuning-correlation threshold for Panel B low/high grouping. "
            f"Default: {PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--high-dark-tuning-correlation-threshold",
        type=float,
        default=PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Upper dark tuning-correlation threshold for Panel B high group. "
            f"Default: {PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD}"
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
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 2 generation."""
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
    make_figure_2(
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
