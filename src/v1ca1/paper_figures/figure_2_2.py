"""Generate an alternate Figure 2 with Panels C and D on separate rows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.transforms import Bbox
import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures.datasets import DatasetId, get_processed_datasets
from v1ca1.paper_figures.figure_2 import (
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
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
    PANEL_B_INDEPENDENT_BASIS_LABEL,
    PANEL_B_LABEL_Y,
    PANEL_B_MODEL_LABEL_FONTSIZE,
    PANEL_B_MODEL_LABEL_X,
    PANEL_B_SEGMENT_MODULATION_LABEL,
    PANEL_B_SEGMENT_MODULATION_LABEL_Y,
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
    build_output_path,
    load_panel_a_example_data,
    load_panel_b_tuning_overlap_table,
    load_panel_glm_data,
    parse_dataset_id,
    plot_panel_a_examples_row,
    plot_panel_b_dpp_overlap_grouped,
    plot_panel_b_dpp_overlap_scatter,
    _align_text_to_reference_display_x,
    _align_texts_to_reference_display_y,
    _shift_axis_horizontally,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    PANEL_H_DELTA_TRAJECTORIES,
    PANEL_H_DELTA_X_LIMITS,
    _draw_panel_h_track,
    _filter_panel_h_heldout_delta,
    _format_panel_h_delta_summary,
    _fraction_histogram_weights,
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
    OUTLINED_HISTOGRAM_KWARGS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "figure_2_2"
PANEL_C_ROW_HEIGHT_MM = PANEL_BC_ROW_HEIGHT_MM * 0.82
PANEL_D_ROW_HEIGHT_MM = PANEL_BC_ROW_HEIGHT_MM * 1.35
PANEL_CD_ROW_GAP_MM = 0.05
PANEL_TOP_ROW_HEIGHT_MM = PANEL_A_EXAMPLE_ROW_HEIGHT_MM * 1.40
PANEL_TOP_WIDTH_RATIOS = (0.57, 0.43)
PANEL_TOP_RIGHT_HEIGHT_RATIOS = (0.54, 0.46)
PANEL_TOP_RIGHT_HSPACE = 0.04
PANEL_DE_WIDTH_RATIOS = (1.0,)
PANEL_DE_WSPACE = 0.08
PANEL_D_HORIZONTAL_SHIFT = 0.0
FIGURE_2_2_BOTTOM_CROP_MM = 16.0
DEFAULT_FIGURE_HEIGHT_MM = (
    PANEL_TOP_ROW_HEIGHT_MM
    + PANEL_C_ROW_HEIGHT_MM
    + PANEL_CD_ROW_GAP_MM
    + PANEL_D_ROW_HEIGHT_MM
)
FIGURE_2_2_CONSTRAINED_LAYOUT_PADS = {
    **FIGURE_2_CONSTRAINED_LAYOUT_PADS,
    "hspace": 0.08,
}
PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS = (0.0, 0.04, 0.49, 0.92)
PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS = (0.51, 0.262, 0.49, 0.644)
PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE = (0.194, 0.241)
PANEL_C_SIDE_BY_SIDE_EXAMPLE_ICON_BOUNDS = (0.0535, 0.321, 0.063, 0.238)
PANEL_C_SIDE_BY_SIDE_EXAMPLE_XLABEL_Y = 0.02
PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_WIDTH = 0.50
PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_GAP = 0.0
PANEL_C_SIDE_BY_SIDE_EXAMPLE_PLOT_LEFT_OFFSET = 0.26
PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_WIDTH = 0.21
PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_GAP = 0.08
PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_GAP = 0.06
PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_HEIGHT = (
    1.0 - PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_GAP
) / 2.0
PANEL_D_COMPACT_SLOT_BOUNDS = (
    (0.260, 0.577, 0.294, 0.198),
    (0.680, 0.577, 0.294, 0.198),
    (0.260, 0.202, 0.294, 0.198),
    (0.680, 0.170, 0.294, 0.245),
)
PANEL_D_COMPACT_EXAMPLE_ICON_BOUNDS = (-0.38, 0.35, 0.18, 0.28)
PANEL_D_SCHEMATIC_TRACK_SIZE = (0.27, 0.157)
PANEL_D_SCHEMATIC_AXIS_BOUNDS = PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS
PANEL_D_DELTA_AXIS_BOUNDS = PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS
PANEL_B_GROUPED_AXIS_BOUNDS = (0.035, 0.07, 0.405, 0.82)
PANEL_B_SCATTER_AXIS_BOUNDS = (0.51, 0.00, 0.48, 0.92)
PANEL_E_CROSS_DECODING_AXIS_BOUNDS = (0.06, 0.03, 0.40, 0.77)
PANEL_E_PLACE_DECODING_AXIS_BOUNDS = (0.57, 0.03, 0.39, 0.77)
PANEL_D_MODEL_LABEL_X = PANEL_B_MODEL_LABEL_X
PANEL_D_SCHEMATIC_LABEL_FONTSIZE = 3.8
PANEL_D_TRAIN_TRACK_CENTER_X = 0.30
PANEL_D_PREDICT_TRACK_CENTER_X = 0.865
PANEL_D_SHARED_DARK_TRACK_CENTER_X = PANEL_D_TRAIN_TRACK_CENTER_X
PANEL_D_SHARED_PLUS_X = 0.445
PANEL_D_SHARED_SEGMENT_TRACK_CENTER_X = 0.585
PANEL_D_SHARED_ARROW_X = (0.725, 0.745)
PANEL_D_SHARED_ARROW_Y_OFFSET = 0.0
PANEL_D_SHARED_LIGHT_TRACK_CENTER_X = PANEL_D_PREDICT_TRACK_CENTER_X
PANEL_D_INDEPENDENT_TRACK_CENTER_Y = PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y
PANEL_D_INDEPENDENT_PREDICTION_LABEL_Y = 0.615
PANEL_D_SHARED_TRACK_CENTER_Y = PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
PANEL_D_SHARED_PREDICTION_LABEL_Y = 0.17
PANEL_D_VERTICAL_SHIFT = 0.065


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


def _tight_bbox_with_bottom_crop(fig: Any, crop_mm: float) -> Bbox:
    """Return a tight figure bbox with unused bottom whitespace removed."""
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    crop_inches = max(float(crop_mm), 0.0) / 25.4
    return Bbox.from_extents(
        tight_bbox.x0,
        tight_bbox.y0 + crop_inches,
        tight_bbox.x1,
        tight_bbox.y1,
    )


def _draw_panel_d_swap_schematic(
    ax: Any,
    *,
    track_size: tuple[float, float] | None = None,
    show_dark_track_labels: bool = False,
    model_name: str,
    model_labels: Mapping[str, str] | None = None,
    prediction_label_fontsize: float = PANEL_C_PREDICTION_LABEL_FONTSIZE,
) -> None:
    """Draw Panel D swap schematic with a panel-C-like shared-scaffold row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    shared_model_label = (
        "Shared-scaffold\nmodel"
        if str(model_name) == "task_segment_bump"
        else f"{_panel_model_label(str(model_name), model_labels)}\nmodel"
    )
    shared_prediction_label = (
        "\"Light activity is like the same arm\n"
        "dark activity with visual modulation\""
        if str(model_name) == "task_segment_bump"
        else "\"Light activity is like the same arm\n"
        "dark activity with segment gain\""
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
    )
    ax.text(
        train_predict_midpoint_x,
        PANEL_D_INDEPENDENT_PREDICTION_LABEL_Y,
        "\"Light activity is like the other arm\nwith the same visual landmark\"",
        ha="center",
        va="top",
        fontsize=prediction_label_fontsize,
        linespacing=0.9,
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
        label_fontsize=PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
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
        label_fontsize=PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
    )
    ax.text(
        train_predict_midpoint_x,
        PANEL_D_SHARED_PREDICTION_LABEL_Y,
        shared_prediction_label,
        ha="center",
        va="bottom",
        fontsize=prediction_label_fontsize,
        linespacing=0.9,
    )


def _mean_swap_delta_across_trajectories(swap_delta_table: Any) -> np.ndarray:
    """Return one held-out mean swapped-segment delta LL per unit."""
    import pandas as pd

    table = _filter_panel_h_heldout_delta(swap_delta_table)
    if table is None or "delta_ll_bits_per_spike" not in table or not len(table):
        return np.asarray([], dtype=float)

    table = table.copy()
    table["delta_ll_bits_per_spike"] = pd.to_numeric(
        table["delta_ll_bits_per_spike"],
        errors="coerce",
    )
    finite_mask = np.isfinite(table["delta_ll_bits_per_spike"].to_numpy(dtype=float))
    table = table[finite_mask].copy()
    if table.empty:
        return np.asarray([], dtype=float)

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
        return values[np.isfinite(values)]

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
    values = summary["mean_delta"].to_numpy(dtype=float)
    return values[np.isfinite(values)]


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
    values = _mean_swap_delta_across_trajectories(swap_delta_table)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.65, zorder=1)
    if values.size:
        bin_edges = np.linspace(PANEL_H_DELTA_X_LIMITS[0], PANEL_H_DELTA_X_LIMITS[1], 29)
        hist_kwargs = OUTLINED_HISTOGRAM_KWARGS.copy()
        hist_kwargs.update({"edgecolor": "none", "linewidth": 0.0})
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=model_color,
            **hist_kwargs,
            zorder=2,
        )
        ax.text(
            0.00,
            1.05,
            "Indep.\nbetter",
            ha="left",
            va="bottom",
            fontsize=3.4,
            color=visual_color,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            1.00,
            1.05,
            "Shared\nbetter",
            ha="right",
            va="bottom",
            fontsize=3.4,
            color=model_color,
            transform=ax.transAxes,
            clip_on=False,
        )
        summary_text = (
            _format_panel_h_delta_summary(values).replace("\n", ", ")
            + f", n={values.size}"
        )
        ax.text(
            0.97,
            0.88,
            summary_text,
            ha="right",
            va="top",
            fontsize=4.1,
            color=model_color,
            transform=ax.transAxes,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.2,
            },
        )
    else:
        ax.text(0.5, 0.5, "No mean\nvalues", ha="center", va="center", fontsize=5.0)

    ax.set_xlim(*PANEL_H_DELTA_X_LIMITS)
    ax.text(
        0.5,
        1.30,
        "Mean across trajectories",
        ha="center",
        va="bottom",
        fontsize=5.0,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.set_xlabel("Δ log likelihood (bits/spike)", fontsize=4.2, labelpad=0.8)
    ax.set_ylabel("Frac.", fontsize=4.3, labelpad=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=3.4, length=1.0, pad=0.6)


def plot_panel_d_compact_swap_delta(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot Figure 2_2 Panel D with three examples and one mean histogram."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes(PANEL_D_SCHEMATIC_AXIS_BOUNDS)
    result_ax = ax.inset_axes(PANEL_D_DELTA_AXIS_BOUNDS)
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
    for example_index, bounds in enumerate(PANEL_D_COMPACT_SLOT_BOUNDS[:3]):
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
            show_ylabel=example_index in (0, 2),
            show_legend=example_index == 1,
            show_xticklabels=example_index == 2,
            icon_bounds=PANEL_D_COMPACT_EXAMPLE_ICON_BOUNDS,
            legend_loc="upper left",
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

    histogram_ax = result_ax.inset_axes(PANEL_D_COMPACT_SLOT_BOUNDS[3])
    plot_panel_d_mean_swap_delta_axis(
        histogram_ax,
        swap_delta_table,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def plot_panel_e_cross_and_place_decoding(
    ax: Any,
    decoding_error_table: Any,
) -> None:
    """Plot Figure 2_2 Panel E as compact cross-route and place decoding."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    cross_ax = ax.inset_axes(PANEL_E_CROSS_DECODING_AXIS_BOUNDS)
    place_ax = ax.inset_axes(PANEL_E_PLACE_DECODING_AXIS_BOUNDS)
    _plot_panel_e_cross_axis(cross_ax, decoding_error_table)
    _plot_panel_e_place_axis(place_ax, decoding_error_table, ylabel=None)
    place_ax.set_title("Path-specific\nplace decoding", fontsize=5.8, pad=1.0)


def plot_panel_c_model_architecture_row(
    ax: Any,
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot Figure 2_2 Panel C with schematics left and examples right."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes(PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS)
    _plot_panel_g_architecture_schematic(
        schematic_ax,
        independent_track_center_y=PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y,
        shared_track_center_y=PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y,
        track_size=PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE,
        independent_basis_icon_scale=PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
        independent_basis_label=PANEL_B_INDEPENDENT_BASIS_LABEL,
        show_dark_track_labels=True,
        field_label_y=PANEL_B_FIELD_LABEL_Y,
        model_label_x=PANEL_B_MODEL_LABEL_X,
        model_label_fontsize=PANEL_B_MODEL_LABEL_FONTSIZE,
        shared_model_label="Shared-scaffold\nmodel",
        component_label_fontsize=PANEL_B_COMPONENT_LABEL_FONTSIZE,
        segment_modulation_label_y=PANEL_B_SEGMENT_MODULATION_LABEL_Y,
        segment_modulation_label=PANEL_B_SEGMENT_MODULATION_LABEL,
    )

    example_ax = ax.inset_axes(PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS)
    _plot_panel_g_example_columns(
        example_ax,
        examples,
        field_y=PANEL_B_EXAMPLE_FIELD_Y,
        field_height=PANEL_B_EXAMPLE_FIELD_HEIGHT,
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
        model_colors=PANEL_B_EXAMPLE_MODEL_COLORS,
        model_labels=PANEL_B_EXAMPLE_MODEL_LABELS,
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
    """Build and save alternate Figure 2 with Panels C and D stacked."""
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
    fig.get_layout_engine().set(**FIGURE_2_2_CONSTRAINED_LAYOUT_PADS)
    outer_grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            PANEL_TOP_ROW_HEIGHT_MM,
            PANEL_C_ROW_HEIGHT_MM,
            PANEL_CD_ROW_GAP_MM,
            PANEL_D_ROW_HEIGHT_MM,
        ],
    )
    top_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_TOP_WIDTH_RATIOS,
        wspace=PANEL_AB_WSPACE,
    )
    panel_a_axis = fig.add_subplot(top_grid[0, 0])
    top_right_grid = top_grid[0, 1].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=PANEL_TOP_RIGHT_HEIGHT_RATIOS,
        hspace=PANEL_TOP_RIGHT_HSPACE,
    )
    panel_b_axis = fig.add_subplot(top_right_grid[0, 0])
    panel_b_axis.set_xlim(0.0, 1.0)
    panel_b_axis.set_ylim(0.0, 1.0)
    panel_b_axis.axis("off")
    panel_b_grouped_axis = panel_b_axis.inset_axes(PANEL_B_GROUPED_AXIS_BOUNDS)
    panel_b_scatter_axis = panel_b_axis.inset_axes(PANEL_B_SCATTER_AXIS_BOUNDS)
    panel_e_axis = fig.add_subplot(top_right_grid[1, 0])
    panel_c_axis = fig.add_subplot(outer_grid[1, 0])
    panel_d_axis = fig.add_subplot(outer_grid[3, 0])

    plot_panel_a_examples_row(
        panel_a_axis,
        panel_a_examples,
        similarity_annotation="dppi",
        show_similarity_annotation=False,
    )
    plot_panel_b_dpp_overlap_grouped(
        panel_b_grouped_axis,
        panel_b_overlap_table,
        low_threshold=dark_tuning_correlation_threshold,
    )
    plot_panel_b_dpp_overlap_scatter(
        panel_b_scatter_axis,
        panel_b_overlap_table,
        title=None,
    )
    plot_panel_c_model_architecture_row(
        panel_c_axis,
        panel_glm_payload["dark_light_examples"],
    )
    plot_panel_d_compact_swap_delta(
        panel_d_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
        model_name=PANEL_C_SWAP_MODEL_NAME,
        model_colors=PANEL_C_SWAP_MODEL_COLORS,
        model_labels=PANEL_C_SWAP_MODEL_LABELS,
    )
    plot_panel_e_cross_and_place_decoding(panel_e_axis, panel_e_decoding_error_table)

    label_axis(panel_a_axis, "A", x=-0.02, y=PANEL_A_LABEL_Y)
    panel_a_label = panel_a_axis.texts[-1]
    panel_a_axis.set_title(
        "Example DPP cells in dark and light",
        fontsize=8,
        pad=PANEL_A_TITLE_PAD,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=PANEL_B_LABEL_Y)
    panel_b_axis.set_title(
        "Dark-to-light DPP overlap",
        fontsize=8,
        pad=PANEL_B_TITLE_PAD,
    )
    label_axis(panel_c_axis, "D", x=-0.02, y=PANEL_BC_LABEL_Y)
    panel_c_label = panel_c_axis.texts[-1]
    label_axis(panel_d_axis, "E", x=-0.02, y=PANEL_BC_LABEL_Y)
    panel_d_label = panel_d_axis.texts[-1]
    label_axis(panel_e_axis, "C", x=-0.035, y=PANEL_B_LABEL_Y)
    panel_c_title = panel_c_axis.set_title(
        "Two models that relate dark and light activity",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )
    panel_d_title = panel_d_axis.set_title(
        "Predicting activity in held-out light epoch",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )
    panel_e_axis.set_title(
        "Decoding",
        fontsize=7,
        pad=PANEL_B_TITLE_PAD,
    )

    fig.canvas.draw()
    _shift_axis_horizontally(panel_d_axis, PANEL_D_HORIZONTAL_SHIFT)
    _shift_axis_vertically(panel_d_axis, PANEL_D_VERTICAL_SHIFT)
    fig.canvas.draw()
    _align_text_to_reference_display_x(panel_c_label, panel_a_label)
    _align_text_to_reference_display_x(panel_d_label, panel_a_label)
    _align_texts_to_reference_display_y((panel_c_title, panel_c_label))
    _align_texts_to_reference_display_y((panel_d_title, panel_d_label))
    _align_texts_to_reference_display_y((panel_b_axis.title, panel_b_axis.texts[-1]))
    fig.set_layout_engine(None)

    save_bbox = _tight_bbox_with_bottom_crop(fig, FIGURE_2_2_BOTTOM_CROP_MM)
    save_figure(fig, output_path, dpi=dpi, bbox_inches=save_bbox)
    plt.close(fig)
    print(f"Saved Figure 2_2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for alternate Figure 2 generation."""
    parser = argparse.ArgumentParser(
        description="Generate alternate Figure 2 with Panels C and D stacked."
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
    """Run alternate Figure 2 generation."""
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
    )


if __name__ == "__main__":
    main()
