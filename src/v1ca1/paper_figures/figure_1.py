from __future__ import annotations

"""Generate Figure 1 panels for pooled dark-epoch place-field heatmaps."""

import argparse
import hashlib
import html
import io
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    get_analysis_path,
    load_trajectory_intervals,
)
from v1ca1.helper.plot_wtrack_schematic import get_w_track_geometry
from v1ca1.helper.wtrack import get_wtrack_total_length
from v1ca1.paper_figures.datasets import (
    DatasetId,
    FigureEpochDatasetId,
    get_dataset_dark_epoch,
    get_processed_datasets,
    make_dataset_id,
    normalize_dataset_id,
)
from v1ca1.paper_figures.style import (
    COMPACT_HISTOGRAM_KWARGS,
    EMPHASIS_HISTOGRAM_KWARGS,
    ENCODING_COMPARISON_COLORS,
    HISTOGRAM_KWARGS,
    NEUTRAL_COLORS,
    PANEL_LABEL_KWARGS,
    RASTER_TICK_KWARGS,
    REGION_COLORS,
    SCHEMATIC_COLORS,
    TRAJECTORY_COLORS,
    VISUAL_CONDITION_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import (
    draw_w_track_arm_side_outlines,
    draw_w_track_schematic,
)
from v1ca1.raster.plot_place_field_heatmap import (
    DEFAULT_SIGMA_BINS,
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    compute_odd_even_place_tuning_curves,
    compute_unit_order,
    prepare_heatmap_session,
)
from v1ca1.raster.plot_1d_place_field_trajectory import (
    compute_trial_spike_positions,
    make_linear_position_interpolator,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from matplotlib.text import Text


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "figure_1"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_ASSET_DIR = Path("paper_figures") / "assets" / "figure_1"
DEFAULT_PROBE_ASSET_NAME = "probe.jpg"
DEFAULT_HISTOLOGY_ASSET_NAME = "histology.svg"
DEFAULT_BEHAVIOR_ASSET_NAME = "behavior.png"
DEFAULT_FIGURE_1A_BEHAVIOR_ASSET_NAME = "behavior_bright.png"
DEFAULT_POSITION_BIN_COUNT = 50
DEFAULT_REGIONS = ("v1",)
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_TOP_ROW_HEIGHT_MM = 40.0
DEFAULT_HEATMAP_HEIGHT_MM = 84.0
DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM = 0.5
DEFAULT_BOTTOM_ROW_HEIGHT_MM = 30.0
DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION = 2.0 / 3.0
DEFAULT_PANEL_E_WIDTH_FRACTION = 1.0 / 3.0
DEFAULT_PANEL_F_WIDTH_FRACTION = DEFAULT_PANEL_E_WIDTH_FRACTION
DEFAULT_PANEL_G_WIDTH_FRACTION = (1.0 - DEFAULT_PANEL_F_WIDTH_FRACTION) / 2.0
DEFAULT_PANEL_H_WIDTH_FRACTION = DEFAULT_PANEL_G_WIDTH_FRACTION
PANEL_H_DECODING_ANIMALS = ("L12", "L14", "L15", "L19")
BOTTOM_ROW_PANEL_WSPACE = 0.05
BOTTOM_ROW_PLOT_BOUNDS = (0.14, 0.18, 0.78, 0.72)
PANEL_G_TITLE = "Cross path decoding"
PANEL_E_RIGHT_ANNOTATION_X = 0.56
PANEL_E_RIGHT_SUMMARY_POSITION = (0.97, 0.06)
PANEL_F_RIGHT_SUMMARY_POSITION = (0.97, 0.06)
PANEL_F_RIGHT_SUMMARY_Y_STEP = 0.12
PANEL_C_EXAMPLE_TITLE = "Example dark DPP coding cells"
PANEL_C_LABEL_X_OFFSET = -0.04
PANEL_D_LABEL_X = -0.12
PANEL_D_LABEL_Y = 1.04
HEATMAP_COLORBAR_PAD = 0.001
HEATMAP_COLORBAR_ASPECT = 14
HEATMAP_COLORBAR_LABEL_FONTSIZE = 4.9
HEATMAP_COLORBAR_LABELPAD = 0
HEATMAP_TUNING_LABEL_OFFSET = -0.004
HEATMAP_ORDER_LABEL_OFFSET = 0.007
HEATMAP_PATH_LABEL_OFFSET = 0.020
PANEL_D_HEATMAP_BLOCK_OUTLINE_COLOR = "red"
PANEL_D_HEATMAP_BLOCK_OUTLINE_LINEWIDTH = 1.2
PANEL_D_HEATMAP_BLOCK_OUTLINE_PAD = 0.004
PANEL_D_HEATMAP_BLOCK_OUTLINE_LINESTYLE = "-"
PANEL_D_HEATMAP_BLOCK_OUTLINE_SPECS = (
    ((0, 2), (0, 2)),
    ((2, 4), (2, 4)),
)
MIN_FIGURE_1_FONTSIZE_PT = 6.0
PANEL_D_TRAJECTORY_TYPES = (
    "right_to_center",
    "center_to_left",
    "left_to_center",
    "center_to_right",
)
PANEL_D_LINEAR_POSITION_ORIENTATION = "task_progression"
PANEL_D_HEATMAP_CMAP = "viridis"
PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION = "unit_max_per_trajectory"
PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION = "unit_max_across_trajectories"
PANEL_D_FIRING_RATE_NORMALIZATION = PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION
PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ = 0.5
PANEL_D_MIN_TUNING_STABILITY_CORRELATION = 0.5
PANEL_D_CACHE_PREFIX = "figure_1_panel_d"
PANEL_D_CACHE_VERSION = 7
PANEL_D_ACROSS_TRAJECTORY_CACHE_VERSION = 5
PANEL_D_CACHE_METADATA_KEY = "__metadata__"
PANEL_D_CACHE_DATASET_TOKEN_LIMIT = 96
PANEL_E_CACHE_PREFIX = "figure_1_panel_e"
PANEL_E_CACHE_VERSION = 1
PANEL_E_CACHE_METADATA_KEY = "__metadata__"
PANEL_DARK_LIGHT_EXAMPLE_TITLE = "Example visual cell in different visual conditions"
PANEL_B_VISUAL_EXAMPLE = ("L14", "20240611", "v1", 229)
PANEL_B_VISUAL_TRAJECTORIES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
PANEL_B_VISUAL_LIGHT_EPOCHS = ("02_r1", "06_r3")
PANEL_B_VISUAL_EPOCH_LABELS = {
    "02_r1": "02_r1",
    "06_r3": "06_r3",
    "dark": "Dark",
}
PANEL_DARK_LIGHT_VISUAL_LABEL_COLORS = {
    "A": VISUAL_CONDITION_COLORS["02_r1"],
    "B": VISUAL_CONDITION_COLORS["06_r3"],
}
PANEL_DARK_LIGHT_RIGHT_ARM_TRAJECTORIES = ("center_to_right", "right_to_center")
PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS = {
    trajectory_type: {
        "02_r1": VISUAL_CONDITION_COLORS["06_r3"],
        "06_r3": VISUAL_CONDITION_COLORS["02_r1"],
    }
    for trajectory_type in PANEL_DARK_LIGHT_RIGHT_ARM_TRAJECTORIES
}
PANEL_B_VISUAL_ICON_LABEL_X_OFFSET = 1.15
PANEL_B_VISUAL_ICON_COLORS = {
    "A": "#66C2A5",
    "B": "#E78AC3",
}
PANEL_B_VISUAL_ICON_REGION_FILL_ALPHA = 0.92
PANEL_B_VISUAL_ICON_ARM_SIDE_OUTLINE_GAP = 0.32
PANEL_B_VISUAL_ICON_ARM_SIDE_OUTLINE_LINEWIDTH = 1.25
PANEL_B_VISUAL_SHARED_XLABEL_Y = 0.035
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
RASTER_ASSET_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
NEURON_SCALE_BAR_COUNT = 100
PANEL_E_EXAMPLES = (
    ("L14", "20240611", "08_r4", "v1", 34),
    ("L15", "20241121", "10_r5", "v1", 473),
)
PANEL_E_RASTER_TRAJECTORY_LAYOUT = (
    ("center_to_left", "center_to_right"),
    ("right_to_center", "left_to_center"),
)
PANEL_E_FR_TRAJECTORY_PAIRS = (
    ("center_to_left", "right_to_center"),
    ("center_to_right", "left_to_center"),
)
PANEL_C_EXAMPLE_BLOCK_HEIGHT = 0.47
PANEL_C_EXAMPLE_BLOCK_Y_POSITIONS = (0.55, 0.02)
PANEL_C_EXAMPLE_SHARED_XLABEL_Y = 0.005
PANEL_C_EXAMPLE_YLABEL_X = -0.190
PANEL_E_TRAJECTORY_LABELS = {
    "center_to_left": "C→L",
    "center_to_right": "C→R",
    "right_to_center": "R→C",
    "left_to_center": "L→C",
}
PANEL_E_TRAJECTORY_COLORS = TRAJECTORY_COLORS
PANEL_E_RASTER_TICK_MARKERSIZE = RASTER_TICK_KWARGS["markersize"]
PANEL_E_RASTER_TICK_MARKEREDGEWIDTH = RASTER_TICK_KWARGS["markeredgewidth"]
PANEL_E_AXIS_LABEL_FONTSIZE = 5.4
PANEL_E_TICK_LABEL_FONTSIZE = 5.0
TASK_PROGRESSION_XLABEL = "Norm. path progression"
TASK_PROGRESSION_SEGMENT_BOUNDARIES = (0.4, 0.6)
TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR = NEUTRAL_COLORS["segment_boundary"]
TASK_PROGRESSION_SEGMENT_BOUNDARY_LINEWIDTH = 0.45
PANEL_DARK_LIGHT_RASTER_COLOR = "black"
PANEL_DARK_LIGHT_RASTER_BACKGROUND_ALPHA = 0.18
PANEL_DARK_LIGHT_FIRST_SEGMENT = (0.0, TASK_PROGRESSION_SEGMENT_BOUNDARIES[0])
PANEL_DARK_LIGHT_LAST_SEGMENT = (TASK_PROGRESSION_SEGMENT_BOUNDARIES[1], 1.0)
PANEL_DARK_LIGHT_TRAJECTORY_EPOCH_BACKGROUNDS = {
    "center_to_left": {
        "02_r1": ((*PANEL_DARK_LIGHT_LAST_SEGMENT, VISUAL_CONDITION_COLORS["02_r1"]),),
        "06_r3": ((*PANEL_DARK_LIGHT_LAST_SEGMENT, VISUAL_CONDITION_COLORS["06_r3"]),),
    },
    "center_to_right": {
        "02_r1": ((*PANEL_DARK_LIGHT_LAST_SEGMENT, VISUAL_CONDITION_COLORS["06_r3"]),),
        "06_r3": ((*PANEL_DARK_LIGHT_LAST_SEGMENT, VISUAL_CONDITION_COLORS["02_r1"]),),
    },
    "left_to_center": {
        "02_r1": ((*PANEL_DARK_LIGHT_FIRST_SEGMENT, VISUAL_CONDITION_COLORS["02_r1"]),),
        "06_r3": ((*PANEL_DARK_LIGHT_FIRST_SEGMENT, VISUAL_CONDITION_COLORS["06_r3"]),),
    },
    "right_to_center": {
        "02_r1": ((*PANEL_DARK_LIGHT_FIRST_SEGMENT, VISUAL_CONDITION_COLORS["06_r3"]),),
        "06_r3": ((*PANEL_DARK_LIGHT_FIRST_SEGMENT, VISUAL_CONDITION_COLORS["02_r1"]),),
    },
}
STABILITY_TRAJECTORY_LAYOUT = PANEL_E_RASTER_TRAJECTORY_LAYOUT
STABILITY_LEGEND_FONTSIZE = 5.4
STABILITY_AXIS_LABEL_FONTSIZE = 5.7
STABILITY_TICK_LABEL_FONTSIZE = 5.5
STABILITY_TABLE_RELATIVE_PATH = (
    Path("task_progression") / "stability" / "odd_even_task_progression_stability.parquet"
)
MOTOR_NESTED_CV_RELATIVE_DIR = Path("task_progression") / "motor" / "nested_lap_cv"
MOTOR_DELTA_METRIC = "dll_motor_tp_vs_motor_bits_per_spike"
MOTOR_DELTA_REGION = "v1"
MOTOR_NESTED_CV_CONFIG_TOKEN = (
    "bin0p05s_spbin2-8cmn3_order4_gpgap15cm_"
    "zscore_outer5_inner3_ridge0p1-1em06n6"
)
MOTOR_MIN_MOVEMENT_FIRING_RATE_HZ = 0.5
MOTOR_MIN_TUNING_STABILITY_CORRELATION = 0.5
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
ENCODING_COMPARISON_REGION = "v1"
ENCODING_COMPARISON_N_FOLDS = 5
ENCODING_COMPARISON_BIN_SIZE_S = 0.05
ENCODING_COMPARISON_PLACE_BIN_SIZE_CM = DEFAULT_PLACE_BIN_SIZE_CM
ENCODING_COMPARISON_MIN_SPIKES = 0
ENCODING_MIN_MOVEMENT_FIRING_RATE_HZ = 0.5
ENCODING_MIN_TUNING_STABILITY_CORRELATION = 0.5
DECODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "decoding_comparison"
DECODING_COMPARISON_REGION = "v1"
STABILITY_REGIONS = ("v1", "ca1")
STABILITY_REGION_COLORS = REGION_COLORS
ENCODING_DPP_COMPARISONS = (
    (
        "dpp_vs_absolute_place",
        "DPP - absolute place",
        "delta_bits_generalized_place_vs_tp",
    ),
    (
        "dpp_vs_absolute_task_progression",
        "DPP - distance-to-reward",
        "delta_bits_gtp_vs_tp",
    ),
)
ENCODING_DPP_COMPARISON_COLORS = ENCODING_COMPARISON_COLORS
DECODING_CROSS_TRAJECTORY_COMPARISONS = (
    (
        "same_turn_cross_arm",
        "Same turn\ncross arm",
        "same_turn_cross_arm",
        (
            ("center_to_left", "right_to_center"),
            ("right_to_center", "center_to_left"),
            ("center_to_right", "left_to_center"),
            ("left_to_center", "center_to_right"),
        ),
    ),
    (
        "opposite_turn_same_arm",
        "Opposite turn\nsame arm",
        "opposite_turn_same_arm",
        (
            ("center_to_left", "left_to_center"),
            ("left_to_center", "center_to_left"),
            ("center_to_right", "right_to_center"),
            ("right_to_center", "center_to_right"),
        ),
    ),
    (
        "same_inbound_outbound_cross_arm",
        "Same in/out\ncross arm",
        "same_inbound_outbound_cross_arm",
        (
            ("center_to_left", "center_to_right"),
            ("center_to_right", "center_to_left"),
            ("left_to_center", "right_to_center"),
            ("right_to_center", "left_to_center"),
        ),
    ),
)
DECODING_EXAMPLE_TRAIN_TRAJECTORY = "center_to_left"
DECODING_EXAMPLE_TEST_TRAJECTORIES = {
    "same_turn_cross_arm": "right_to_center",
    "opposite_turn_same_arm": "left_to_center",
    "same_inbound_outbound_cross_arm": "center_to_right",
}
DECODING_TRAIN_SCHEMATIC_CENTER_X = -0.075
DECODING_SCHEMATIC_Y = -0.55
DECODING_SCHEMATIC_WIDTH = 0.132
DECODING_SCHEMATIC_HEIGHT = 0.198
DECODING_TRAIN_LABEL_Y = -0.32
TOP_ROW_PANEL_LABEL_X_OFFSETS = (-0.04, -0.02)
TOP_ROW_PANEL_TITLE_FONTSIZES = (8.0, 8.0)
BOTTOM_ROW_PANEL_LABEL_X_OFFSETS = (-0.04, -0.08, -0.04)
DECODING_YLABEL_FONTSIZE = 6.0
DECODING_YLABEL_X = -0.16
DECODING_XTICK_LABEL_FONTSIZE = 5.6
DECODING_MEDIAN_LABEL_FONTSIZE = 4.8
DECODING_MEDIAN_LABEL_X_OFFSET = 0.09
DECODING_REGION_POSITION_OFFSET = 0.10
DECODING_REGION_LEGEND_FONTSIZE = 5.0
DECODING_PERMUTATION_COUNT = 100_000
DECODING_PERMUTATION_SEED = 20260703
DECODING_SIGNIFICANCE_CONTRASTS = (
    ("same_turn_cross_arm", "opposite_turn_same_arm", 0.500),
    ("same_turn_cross_arm", "same_inbound_outbound_cross_arm", 0.635),
)
DECODING_SIGNIFICANCE_BRACKET_HEIGHT = 0.016
DECODING_SIGNIFICANCE_BRACKET_LINEWIDTH = 0.6
DECODING_SIGNIFICANCE_LABEL_FONTSIZE = 7.0
DECODING_SIGNIFICANCE_LABEL_Y_OFFSET = 0.004
DELTA_LOG_LIKELIHOOD_AXIS_LABEL = "Δ log likelihood\n(bits/spike)"
STABILITY_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "unit",
    "region",
    "epoch",
    "trajectory_type",
    "firing_rate_hz",
    "stability_correlation",
    "n_odd_trials",
    "n_even_trials",
    "odd_duration_s",
    "even_duration_s",
)
MOTOR_DELTA_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "unit",
    "delta_log_likelihood_bits_per_spike",
    "source_path",
)
ENCODING_DELTA_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "unit",
    "n_spikes",
    "comparison",
    "comparison_label",
    "delta_log_likelihood_bits_per_spike",
    "source_path",
)
DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "comparison",
    "comparison_label",
    "transfer_family",
    "encoding_trajectory",
    "decoding_trajectory",
    "absolute_error",
    "true_path",
    "decoded_path",
)
DECODING_TRIAL_ERROR_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "comparison",
    "comparison_label",
    "transfer_family",
    "encoding_trajectory",
    "decoding_trajectory",
    "trial_index",
    "trial_start",
    "trial_end",
    "trial_median_absolute_error",
    "n_samples",
    "true_path",
    "decoded_path",
)
DECODING_PERMUTATION_RESULT_COLUMNS = (
    "animal_name",
    "comparison_a",
    "comparison_b",
    "median_difference",
    "p_two_sided",
    "p_less",
    "p_greater",
    "n_permutations",
)
CYCLE_TRAJECTORY_LAYOUT = (
    ("left_to_center", (0.59, 0.60, 0.39, 0.39)),
    ("center_to_right", (0.59, 0.04, 0.39, 0.39)),
    ("right_to_center", (0.02, 0.04, 0.39, 0.39)),
    ("center_to_left", (0.02, 0.60, 0.39, 0.39)),
)
CYCLE_ARROW_SPECS = (
    ((0.785, 0.60), (0.785, 0.45), 0.0),
    ((0.59, 0.235), (0.41, 0.235), 0.0),
    ((0.215, 0.45), (0.215, 0.60), 0.0),
    ((0.41, 0.795), (0.59, 0.795), 0.0),
)
CYCLE_ARROW_LINEWIDTH = 1.08
CYCLE_ARROW_MUTATION_SCALE = 12.6
TASK_DESIGN_TOP_LEFT_BOUNDS = (0.02, 0.40, 0.46, 0.56)
TASK_DESIGN_TOP_RIGHT_BOUNDS = (0.52, 0.40, 0.46, 0.56)
TASK_DESIGN_BOTTOM_BOUNDS = (0.02, 0.02, 0.96, 0.28)
TASK_DESIGN_CONDITION_TRACK_BOUNDS = (
    (0.05, 0.25, 0.26, 0.55),
    (0.37, 0.25, 0.26, 0.55),
    (0.69, 0.25, 0.26, 0.55),
)
TASK_DESIGN_VISUAL_TIMELINE_ARROW_Y = 0.19
TASK_DESIGN_DIVIDER_Y = 0.35
TASK_DESIGN_DIVIDER_HEADING = "Visual conditions over time"
TASK_DESIGN_PHASE_LABEL_Y = 0.82
TASK_DESIGN_PHASE_LABELS = (
    "Initial A/B",
    "Cue swap (B/A)",
    "Dark",
)
TASK_DESIGN_PROGRESSION_SEGMENTS = (
    ("AB", ("A", "B")),
    ("gray", ("gray",)),
    ("BA", ("B", "A")),
    ("dark", ("dark",)),
)
TASK_DESIGN_TITLE = "W-track task with visual landmarks"
TASK_DESIGN_PROGRESSION_SLEEP_LABEL = "sleep"
TASK_DESIGN_PROGRESSION_DURATION_LABEL = "Run/sleep epochs (~25 min each)"
TASK_DESIGN_RUN_EPOCH_LINEWIDTH = 1.20
TASK_DESIGN_SLEEP_EPOCH_LINEWIDTH = 0.45
MOVEMENT_AXIS_Y = -0.13
MOVEMENT_AXIS_ARROW_MARGIN = 0.12


def get_trajectory_endpoint_labels(
    trajectory_type: str,
    *,
    axis_orientation: str = "branch_position",
) -> tuple[str, str]:
    """Return normalized-axis endpoint labels for one trajectory type."""
    if axis_orientation == "task_progression":
        if trajectory_type == "center_to_left":
            return "C", "L"
        if trajectory_type == "left_to_center":
            return "L", "C"
        if trajectory_type == "center_to_right":
            return "C", "R"
        if trajectory_type == "right_to_center":
            return "R", "C"
        raise ValueError(f"Unknown trajectory type {trajectory_type!r}.")
    if axis_orientation != "branch_position":
        raise ValueError(f"Unknown axis_orientation {axis_orientation!r}.")
    if "left" in trajectory_type:
        return "C", "L"
    if "right" in trajectory_type:
        return "C", "R"
    raise ValueError(f"Unknown trajectory type {trajectory_type!r}.")


def get_movement_arrow_points(
    trajectory_type: str,
    *,
    axis_orientation: str = "branch_position",
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return axes-fraction start/end points for the movement-direction arrow."""
    if axis_orientation == "task_progression":
        return (
            (MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
            (1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
        )
    if axis_orientation != "branch_position":
        raise ValueError(f"Unknown axis_orientation {axis_orientation!r}.")
    if trajectory_type.startswith("center_to_"):
        return (
            (MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
            (1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
        )
    if trajectory_type.endswith("_to_center"):
        return (
            (1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
            (MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y),
        )
    raise ValueError(f"Unknown trajectory type {trajectory_type!r}.")


def add_normalized_path_heatmap_axis(ax: "Axes") -> None:
    """Label one bottom-row heatmap with normalized path coordinates."""
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["0", "1"])
    ax.tick_params(
        axis="x",
        labelsize=PANEL_E_TICK_LABEL_FONTSIZE,
        length=0.9,
        width=0.35,
        pad=1,
    )


def add_centered_below_axis_text(
    fig: Any,
    axes: Sequence["Axes"],
    text: str,
    *,
    y_offset: float = 0.01,
    fontsize: float = 9.0,
) -> "Text":
    """Add text centered below a group of axes."""
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    return fig.text(
        (x0 + x1) / 2,
        y0 - y_offset,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
    )


def add_centered_axis_text(
    fig: Any,
    axes: Sequence["Axes"],
    text: str,
    *,
    y_offset: float = 0.01,
    rotation: float = 0.0,
    fontsize: float = 9.0,
) -> "Text":
    """Add text centered over or beside a group of axes."""
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    if rotation:
        return fig.text(
            x0 - y_offset,
            (y0 + y1) / 2,
            text,
            ha="center",
            va="center",
            rotation=rotation,
            fontsize=fontsize,
        )
    return fig.text(
        (x0 + x1) / 2,
        y1 + y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def add_aligned_panel_headers(
    fig: Any,
    axes: Sequence["Axes"],
    *,
    labels: Sequence[str],
    titles: Sequence[str],
    label_x_offsets: Sequence[float],
    fontsize: float = 8.0,
    title_fontsizes: Sequence[float] | None = None,
) -> None:
    """Draw panel labels and titles on one shared figure-level baseline."""
    if title_fontsizes is None:
        title_fontsizes = (fontsize,) * len(axes)
    if not (len(axes) == len(labels) == len(titles) == len(label_x_offsets)):
        raise ValueError("axes, labels, titles, and label_x_offsets must have equal length.")
    if len(title_fontsizes) != len(axes):
        raise ValueError("title_fontsizes must have one value per axis.")

    title_y_values = [
        fig.transFigure.inverted()
        .transform(axis.title.get_transform().transform(axis.title.get_position()))[1]
        for axis in axes
    ]
    header_y = max(title_y_values)

    label_kwargs = PANEL_LABEL_KWARGS.copy()
    label_kwargs["fontsize"] = fontsize
    for axis, label, title, label_x_offset, title_fontsize in zip(
        axes,
        labels,
        titles,
        label_x_offsets,
        title_fontsizes,
        strict=True,
    ):
        box = axis.get_position()
        axis.set_title("")
        fig.text(
            box.x0 + label_x_offset * box.width,
            header_y,
            label,
            transform=fig.transFigure,
            **label_kwargs,
        )
        fig.text(
            (box.x0 + box.x1) / 2.0,
            header_y,
            title,
            ha="center",
            va="bottom",
            fontsize=title_fontsize,
            transform=fig.transFigure,
        )


def add_aligned_panel_headers_at_label_positions(
    fig: Any,
    axes: Sequence["Axes"],
    *,
    labels: Sequence[str],
    titles: Sequence[str],
    label_x_positions: Sequence[float],
    fontsize: float = 8.0,
) -> None:
    """Draw panel labels at fixed figure x positions with aligned titles."""
    if not (len(axes) == len(labels) == len(titles) == len(label_x_positions)):
        raise ValueError(
            "axes, labels, titles, and label_x_positions must have equal length."
        )

    title_y_values = [
        fig.transFigure.inverted()
        .transform(axis.title.get_transform().transform(axis.title.get_position()))[1]
        for axis in axes
    ]
    header_y = max(title_y_values)

    label_kwargs = PANEL_LABEL_KWARGS.copy()
    label_kwargs["fontsize"] = fontsize
    for axis, label, title, label_x in zip(
        axes,
        labels,
        titles,
        label_x_positions,
        strict=True,
    ):
        box = axis.get_position()
        axis.set_title("")
        fig.text(
            label_x,
            header_y,
            label,
            transform=fig.transFigure,
            **label_kwargs,
        )
        fig.text(
            (box.x0 + box.x1) / 2.0,
            header_y,
            title,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            transform=fig.transFigure,
        )


def add_panel_header_at_reference_y(
    fig: Any,
    axis: "Axes",
    *,
    label: str,
    title: str,
    label_x_offset: float,
    reference_axis: "Axes",
    reference_y: float,
    fontsize: float = 8.0,
) -> tuple["Text", "Text"]:
    """Draw one panel label/title at a reference axes-relative y level."""
    reference_y_figure = fig.transFigure.inverted().transform(
        reference_axis.transAxes.transform((0.0, reference_y))
    )[1]
    box = axis.get_position()

    label_kwargs = PANEL_LABEL_KWARGS.copy()
    label_kwargs["fontsize"] = fontsize
    label_text = fig.text(
        box.x0 + label_x_offset * box.width,
        reference_y_figure,
        label,
        transform=fig.transFigure,
        **label_kwargs,
    )
    title_text = fig.text(
        (box.x0 + box.x1) / 2.0,
        reference_y_figure,
        title,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        transform=fig.transFigure,
    )
    return label_text, title_text


def raise_figure_text_to_minimum_fontsize(fig: Any, min_fontsize: float) -> None:
    """Raise all figure text below a minimum font size."""

    def _iter_axes_tree(ax: Any) -> Sequence[Any]:
        axes = [ax]
        for child_ax in getattr(ax, "child_axes", ()):
            axes.extend(_iter_axes_tree(child_ax))
        return axes

    def _maybe_raise(text: Any) -> None:
        if text is not None and text.get_fontsize() < min_fontsize:
            text.set_fontsize(min_fontsize)

    seen_axes: set[int] = set()
    for root_ax in fig.axes:
        for ax in _iter_axes_tree(root_ax):
            ax_id = id(ax)
            if ax_id in seen_axes:
                continue
            seen_axes.add(ax_id)
            _maybe_raise(ax.title)
            _maybe_raise(ax.xaxis.label)
            _maybe_raise(ax.yaxis.label)
            _maybe_raise(ax.xaxis.get_offset_text())
            _maybe_raise(ax.yaxis.get_offset_text())
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
    for text in fig.texts:
        _maybe_raise(text)


def get_dark_epoch(animal_name: str, date: str, dark_epoch: str | None = None) -> str:
    """Return the dark run epoch label for one session."""
    del date
    if dark_epoch is not None:
        return str(dark_epoch)
    return get_dataset_dark_epoch(animal_name)


def parse_dataset_id(value: str) -> DatasetId:
    """Parse one `animal:date[:dark_epoch]` data-set identifier."""
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(
            "Data sets must be specified as animal:date or animal:date:dark_epoch, "
            "for example L14:20240611 or L15:20241121:10_r5."
        )
    return make_dataset_id(*parts)


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the figure output path for one requested format."""
    if output_format not in FIGURE_FORMATS:
        raise ValueError(
            f"Unknown output format {output_format!r}. Expected one of {FIGURE_FORMATS!r}."
        )
    return Path(output_dir) / f"{output_name}.{output_format}"


def get_figure_1_asset_path(asset_dir: Path, asset_name: str) -> Path:
    """Return the path to one Figure 1 external asset."""
    return Path(asset_dir) / asset_name


def _parse_svg_aspect_ratio(svg_path: Path) -> float:
    """Return width divided by height for one SVG file."""
    root = ET.parse(svg_path).getroot()
    viewbox = root.attrib.get("viewBox")
    if viewbox is not None:
        values = [float(value) for value in viewbox.replace(",", " ").split()]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return values[2] / values[3]

    width = _parse_svg_length(root.attrib.get("width"))
    height = _parse_svg_length(root.attrib.get("height"))
    if width is None or height is None or height <= 0:
        return 1.0
    return width / height


def _parse_svg_length(value: str | None) -> float | None:
    """Parse one SVG length string and ignore its unit."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    number = []
    for character in stripped:
        if character.isdigit() or character in ".+-eE":
            number.append(character)
        else:
            break
    if not number:
        return None
    return float("".join(number))


def _find_svg_sidecar_raster(svg_path: Path) -> Path | None:
    """Return a same-stem raster export for an SVG when one exists."""
    for extension in RASTER_ASSET_EXTENSIONS:
        sidecar_path = svg_path.with_suffix(extension)
        if sidecar_path.exists():
            return sidecar_path
    return None


def _render_svg_with_cairosvg(svg_path: Path, *, output_width_px: int) -> np.ndarray | None:
    """Rasterize an SVG with cairosvg when that optional dependency is installed."""
    try:
        import cairosvg
        import matplotlib.image as mpimg
    except ImportError:
        return None

    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=int(output_width_px),
    )
    return np.asarray(mpimg.imread(io.BytesIO(png_bytes), format="png"))


def _find_chrome_executable() -> str | None:
    """Return an available Chrome executable for SVG rasterization."""
    for executable in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(executable)
        if path is not None:
            return path
    return None


def _render_svg_with_chrome(svg_path: Path, *, output_width_px: int) -> np.ndarray | None:
    """Rasterize an SVG with headless Chrome when available."""
    import matplotlib.image as mpimg

    chrome_path = _find_chrome_executable()
    if chrome_path is None:
        return None

    aspect_ratio = _parse_svg_aspect_ratio(svg_path)
    output_height_px = max(1, int(round(output_width_px / aspect_ratio)))
    with tempfile.TemporaryDirectory(prefix="v1ca1-figure-asset-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        html_path = temp_dir / "asset.html"
        png_path = temp_dir / "asset.png"
        html_path.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    "<html>",
                    "<head>",
                    "<meta charset='utf-8'>",
                    "<style>",
                    "html, body { margin: 0; width: 100%; height: 100%; background: white; }",
                    "img { width: 100vw; height: 100vh; object-fit: contain; display: block; }",
                    "</style>",
                    "</head>",
                    "<body>",
                    f"<img src='{html.escape(svg_path.resolve().as_uri(), quote=True)}'>",
                    "</body>",
                    "</html>",
                ]
            ),
            encoding="utf-8",
        )
        command = [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-crash-reporter",
            "--disable-breakpad",
            "--disable-features=Crashpad",
            f"--user-data-dir={temp_dir / 'chrome-profile'}",
            f"--screenshot={png_path}",
            f"--window-size={int(output_width_px)},{output_height_px}",
            html_path.as_uri(),
        ]
        env = {
            **os.environ,
            "HOME": str(temp_dir),
            "XDG_CONFIG_HOME": str(temp_dir),
            "XDG_CACHE_HOME": str(temp_dir),
        }
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as error:
            message = (error.stderr or error.stdout or "").strip()
            raise RuntimeError(
                "Could not rasterize SVG asset with headless Chrome. "
                "Install cairosvg or place a same-stem raster export next to the SVG. "
                f"Chrome output: {message}"
            ) from error

        return np.asarray(mpimg.imread(png_path))


def load_panel_asset_image(asset_path: Path, *, svg_output_width_px: int = 1200) -> np.ndarray:
    """Load one raster image asset, rasterizing SVG assets when needed."""
    import matplotlib.image as mpimg

    asset_path = Path(asset_path)
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing Figure 1 asset: {asset_path}")

    suffix = asset_path.suffix.lower()
    if suffix in RASTER_ASSET_EXTENSIONS:
        return np.asarray(mpimg.imread(asset_path))
    if suffix != ".svg":
        raise ValueError(f"Unsupported Figure 1 asset format: {asset_path}")

    sidecar_path = _find_svg_sidecar_raster(asset_path)
    if sidecar_path is not None:
        return np.asarray(mpimg.imread(sidecar_path))

    image = _render_svg_with_cairosvg(asset_path, output_width_px=svg_output_width_px)
    if image is not None:
        return image

    image = _render_svg_with_chrome(asset_path, output_width_px=svg_output_width_px)
    if image is not None:
        return image

    raise RuntimeError(
        "Could not rasterize SVG asset. Install cairosvg or place a same-stem "
        f"raster export next to {asset_path}."
    )


def build_normalized_position_bins(position_bin_count: int) -> np.ndarray:
    """Return normalized trajectory-position bin edges from 0 to 1."""
    if position_bin_count <= 0:
        raise ValueError("--position-bin-count must be positive.")
    return np.linspace(0.0, 1.0, int(position_bin_count) + 1)


def get_stability_table_path(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression stability table path for one session."""
    return Path(data_root) / animal_name / date / STABILITY_TABLE_RELATIVE_PATH


def get_motor_nested_cv_dir(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression motor nested-CV directory."""
    return Path(data_root) / animal_name / date / MOTOR_NESTED_CV_RELATIVE_DIR


def _format_float_token(value: float) -> str:
    """Return a path-safe compact token for one numeric value."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def format_place_bin_size_token(place_bin_size_cm: float) -> str:
    """Return the filename token for one place-bin-size setting."""
    return f"placebin{_format_float_token(place_bin_size_cm)}cm"


def format_time_bin_size_token(bin_size_s: float) -> str:
    """Return the filename token for one temporal-bin-size setting."""
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive and finite.")
    return f"bin{_format_float_token(bin_size_s)}s"


def get_encoding_comparison_dir(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression encoding-comparison directory."""
    return Path(data_root) / animal_name / date / ENCODING_COMPARISON_RELATIVE_DIR


def get_decoding_comparison_dir(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved task-progression decoding-comparison directory."""
    return Path(data_root) / animal_name / date / DECODING_COMPARISON_RELATIVE_DIR


def find_motor_nested_cv_path(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> Path:
    """Return the configured motor nested-CV output path for one session epoch."""
    data_dir = get_motor_nested_cv_dir(data_root, animal_name, date)
    path = data_dir / (
        f"{region}_{epoch}_nested_lapcv_{MOTOR_NESTED_CV_CONFIG_TOKEN}.nc"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing configured task-progression motor nested-CV output. "
            f"Expected {path}. "
            "Run `python -m v1ca1.task_progression.motor "
            f"--animal-name {animal_name} --date {date} --regions {region} "
            f"--epochs {epoch} --motor-feature-mode zscore` first."
        )
    return path


def find_encoding_summary_path(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    n_folds: int = ENCODING_COMPARISON_N_FOLDS,
    bin_size_s: float = ENCODING_COMPARISON_BIN_SIZE_S,
    place_bin_size_cm: float = ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
) -> Path:
    """Return the preferred encoding-comparison summary table path."""
    data_dir = get_encoding_comparison_dir(data_root, animal_name, date)
    time_bin_token = format_time_bin_size_token(bin_size_s)
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    path = data_dir / (
        f"{region}_{epoch}_cv{n_folds}_{time_bin_token}_{place_bin_token}"
        "_encoding_summary.parquet"
    )
    if path.exists():
        return path
    raise FileNotFoundError(
        "Missing task-progression encoding-comparison summary. Expected "
        f"{path}. Run "
        "`python -m v1ca1.task_progression.encoding_comparison "
        f"--animal-name {animal_name} --date {date} --dark-epoch {epoch} "
        f"--regions {region} --bin-size-s {bin_size_s} "
        f"--place-bin-size-cm {place_bin_size_cm}` first."
    )


def get_cross_trajectory_decoding_tsd_paths(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    transfer_family: str,
    encoding_trajectory: str,
    decoding_trajectory: str,
) -> tuple[Path, Path]:
    """Return true and decoded cross-trajectory task-progression `.npz` paths."""
    data_dir = get_decoding_comparison_dir(data_root, animal_name, date)
    suffix = f"{transfer_family}_{encoding_trajectory}_to_{decoding_trajectory}"
    true_path = data_dir / f"{region}_{epoch}_{suffix}_true_tp_cross_traj.npz"
    decoded_path = data_dir / f"{region}_{epoch}_{suffix}_decoded_tp_cross_traj.npz"
    return true_path, decoded_path


def _load_tsd_npz(path: Path) -> Any:
    """Load one pynapple-backed `.npz` time-series artifact."""
    import pynapple as nap

    return nap.load_file(path)


def load_decoding_absolute_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = DECODING_COMPARISON_REGION,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = DECODING_CROSS_TRAJECTORY_COMPARISONS,
) -> Any:
    """Load pooled sample-level cross-trajectory decoding absolute errors."""
    import pandas as pd

    from v1ca1.task_progression.decoding_comparison import align_true_to_decoded

    tables = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        for comparison, label, transfer_family, trajectory_pairs in comparisons:
            for encoding_trajectory, decoding_trajectory in trajectory_pairs:
                true_path, decoded_path = get_cross_trajectory_decoding_tsd_paths(
                    data_root=data_root,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    epoch=epoch,
                    transfer_family=transfer_family,
                    encoding_trajectory=encoding_trajectory,
                    decoding_trajectory=decoding_trajectory,
                )
                missing_paths = [
                    path for path in (true_path, decoded_path) if not path.exists()
                ]
                if missing_paths:
                    raise FileNotFoundError(
                        "Missing task-progression decoding-comparison output. "
                        f"Expected {missing_paths[0]}. Run "
                        "`python -m v1ca1.task_progression.decoding_comparison "
                        f"--animal-name {animal_name} --date {date} "
                        f"--dark-epoch {epoch} --regions {region}` first."
                    )

                true_tsd = _load_tsd_npz(true_path)
                decoded_tsd = _load_tsd_npz(decoded_path)
                true_values, decoded_values = align_true_to_decoded(true_tsd, decoded_tsd)
                absolute_error = np.abs(decoded_values - true_values)
                absolute_error = absolute_error[np.isfinite(absolute_error)]
                if absolute_error.size == 0:
                    continue
                tables.append(
                    pd.DataFrame(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "region": region,
                            "comparison": comparison,
                            "comparison_label": label,
                            "transfer_family": transfer_family,
                            "encoding_trajectory": encoding_trajectory,
                            "decoding_trajectory": decoding_trajectory,
                            "absolute_error": absolute_error,
                            "true_path": str(true_path),
                            "decoded_path": str(decoded_path),
                        }
                    )
                )

    if not tables:
        return pd.DataFrame(columns=DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS)
    return pd.concat(tables, axis=0, ignore_index=True)


def _align_absolute_error_with_times(
    true_tsd: Any,
    decoded_tsd: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decoded timestamps and finite aligned absolute errors."""
    if len(np.asarray(decoded_tsd.t)) == 0 or len(np.asarray(true_tsd.t)) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    support = true_tsd.time_support.intersect(decoded_tsd.time_support)
    true_restricted = true_tsd.restrict(support)
    decoded_restricted = decoded_tsd.restrict(support)
    true_at_decoded = true_restricted.interpolate(
        decoded_restricted,
        ep=support,
        left=np.nan,
        right=np.nan,
    )
    timestamps = np.asarray(decoded_restricted.t, dtype=float)
    true_values = np.asarray(true_at_decoded.d, dtype=float)
    decoded_values = np.asarray(decoded_restricted.d, dtype=float)
    valid = (
        np.isfinite(timestamps)
        & np.isfinite(true_values)
        & np.isfinite(decoded_values)
    )
    return (
        timestamps[valid],
        np.abs(decoded_values[valid] - true_values[valid]),
    )


def _intervalset_to_arrays(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted start and end arrays for one interval set."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Trajectory interval starts and ends have mismatched shapes: "
            f"{starts.shape} vs {ends.shape}."
        )
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts)
    return starts[order], ends[order]


def build_decoding_trial_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = DECODING_COMPARISON_REGION,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = DECODING_CROSS_TRAJECTORY_COMPARISONS,
) -> Any:
    """Build lap-level median decoding errors for Figure 1 inference."""
    import pandas as pd

    records = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        analysis_path = get_analysis_path(animal_name, date, Path(data_root))
        trajectory_intervals, _source = load_trajectory_intervals(
            analysis_path,
            [epoch],
        )
        if epoch not in trajectory_intervals:
            raise ValueError(
                f"Trajectory intervals do not contain epoch {epoch!r} for "
                f"{animal_name} {date}."
            )

        for comparison, label, transfer_family, trajectory_pairs in comparisons:
            for encoding_trajectory, decoding_trajectory in trajectory_pairs:
                if decoding_trajectory not in trajectory_intervals[epoch]:
                    raise ValueError(
                        "Trajectory intervals do not contain decoding trajectory "
                        f"{decoding_trajectory!r} for {animal_name} {date} {epoch}."
                    )
                true_path, decoded_path = get_cross_trajectory_decoding_tsd_paths(
                    data_root=data_root,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    epoch=epoch,
                    transfer_family=transfer_family,
                    encoding_trajectory=encoding_trajectory,
                    decoding_trajectory=decoding_trajectory,
                )
                missing_paths = [
                    path for path in (true_path, decoded_path) if not path.exists()
                ]
                if missing_paths:
                    raise FileNotFoundError(
                        "Missing task-progression decoding-comparison output. "
                        f"Expected {missing_paths[0]}. Run "
                        "`python -m v1ca1.task_progression.decoding_comparison "
                        f"--animal-name {animal_name} --date {date} "
                        f"--dark-epoch {epoch} --regions {region}` first."
                    )

                true_tsd = _load_tsd_npz(true_path)
                decoded_tsd = _load_tsd_npz(decoded_path)
                timestamps, absolute_error = _align_absolute_error_with_times(
                    true_tsd,
                    decoded_tsd,
                )
                starts, ends = _intervalset_to_arrays(
                    trajectory_intervals[epoch][decoding_trajectory]
                )
                for trial_index, (start, end) in enumerate(
                    zip(starts, ends, strict=True)
                ):
                    in_trial = (timestamps >= start) & (timestamps < end)
                    values = absolute_error[in_trial]
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue
                    records.append(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "region": region,
                            "comparison": comparison,
                            "comparison_label": label,
                            "transfer_family": transfer_family,
                            "encoding_trajectory": encoding_trajectory,
                            "decoding_trajectory": decoding_trajectory,
                            "trial_index": int(trial_index),
                            "trial_start": float(start),
                            "trial_end": float(end),
                            "trial_median_absolute_error": float(
                                np.median(values)
                            ),
                            "n_samples": int(values.size),
                            "true_path": str(true_path),
                            "decoded_path": str(decoded_path),
                        }
                    )

    return pd.DataFrame.from_records(
        records,
        columns=DECODING_TRIAL_ERROR_TABLE_COLUMNS,
    )


def stratified_median_permutation_test(
    trial_table: Any,
    comparison_a: str,
    comparison_b: str,
    *,
    n_permutations: int,
    rng: Any,
) -> dict[str, float | int]:
    """Compare lap medians after shuffling labels within decoding paths."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    required_columns = {
        "comparison",
        "decoding_trajectory",
        "trial_median_absolute_error",
    }
    missing_columns = required_columns.difference(trial_table.columns)
    if missing_columns:
        raise ValueError(
            "Trial-error table is missing required columns: "
            f"{sorted(missing_columns)!r}."
        )

    comparison_values = trial_table["comparison"].astype(str)
    value_column = "trial_median_absolute_error"
    values_a = np.asarray(
        trial_table.loc[comparison_values == comparison_a, value_column],
        dtype=float,
    )
    values_b = np.asarray(
        trial_table.loc[comparison_values == comparison_b, value_column],
        dtype=float,
    )
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if values_a.size == 0 or values_b.size == 0:
        raise ValueError(
            "Both decoding comparisons must contain finite trial-level errors: "
            f"{comparison_a!r} has {values_a.size}, "
            f"{comparison_b!r} has {values_b.size}."
        )

    observed = float(np.median(values_a) - np.median(values_b))
    selected = trial_table.loc[
        comparison_values.isin((comparison_a, comparison_b))
    ]
    strata = []
    for decoding_trajectory, trajectory_table in selected.groupby(
        "decoding_trajectory",
        sort=False,
    ):
        trajectory_comparisons = trajectory_table["comparison"].astype(str)
        stratum_a = np.asarray(
            trajectory_table.loc[
                trajectory_comparisons == comparison_a,
                value_column,
            ],
            dtype=float,
        )
        stratum_b = np.asarray(
            trajectory_table.loc[
                trajectory_comparisons == comparison_b,
                value_column,
            ],
            dtype=float,
        )
        stratum_a = stratum_a[np.isfinite(stratum_a)]
        stratum_b = stratum_b[np.isfinite(stratum_b)]
        if stratum_a.size == 0 or stratum_b.size == 0:
            raise ValueError(
                "Both decoding comparisons must contain finite trials in "
                f"stratum {decoding_trajectory!r}: {comparison_a!r} has "
                f"{stratum_a.size}, {comparison_b!r} has {stratum_b.size}."
            )
        strata.append(
            (
                np.concatenate((stratum_a, stratum_b)),
                int(stratum_a.size),
                int(stratum_b.size),
            )
        )
    if not strata:
        raise ValueError("No decoding-trajectory strata were available.")

    null_differences = np.empty(n_permutations, dtype=float)
    for permutation_index in range(n_permutations):
        permuted_a = []
        permuted_b = []
        for values, n_a, n_b in strata:
            order = rng.permutation(values.size)
            permuted_a.append(values[order[:n_a]])
            permuted_b.append(values[order[n_a : n_a + n_b]])
        null_differences[permutation_index] = float(
            np.median(np.concatenate(permuted_a))
            - np.median(np.concatenate(permuted_b))
        )

    p_two_sided = (
        np.count_nonzero(np.abs(null_differences) >= abs(observed)) + 1.0
    ) / (n_permutations + 1.0)
    p_less = (
        np.count_nonzero(null_differences <= observed) + 1.0
    ) / (n_permutations + 1.0)
    p_greater = (
        np.count_nonzero(null_differences >= observed) + 1.0
    ) / (n_permutations + 1.0)
    return {
        "median_difference": observed,
        "p_two_sided": float(p_two_sided),
        "p_less": float(p_less),
        "p_greater": float(p_greater),
        "n_permutations": int(n_permutations),
    }


def compute_decoding_permutation_tests(
    trial_table: Any,
    *,
    contrasts: Sequence[tuple[str, str, float]] = DECODING_SIGNIFICANCE_CONTRASTS,
    n_permutations: int = DECODING_PERMUTATION_COUNT,
    seed: int = DECODING_PERMUTATION_SEED,
) -> Any:
    """Run Figure 1's planned permutation tests separately per animal."""
    import pandas as pd

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if "animal_name" not in trial_table.columns:
        raise ValueError("Trial-error table is missing required column 'animal_name'.")

    comparison_strata = {
        comparison: {
            decoding_trajectory
            for (
                candidate_comparison,
                _label,
                _transfer_family,
                trajectory_pairs,
            ) in DECODING_CROSS_TRAJECTORY_COMPARISONS
            if candidate_comparison == comparison
            for _encoding_trajectory, decoding_trajectory in trajectory_pairs
        }
        for comparison_a, comparison_b, _y in contrasts
        for comparison in (comparison_a, comparison_b)
    }
    rng = np.random.default_rng(seed)
    records = []
    for animal_name, animal_table in trial_table.groupby(
        "animal_name",
        sort=True,
    ):
        for comparison_a, comparison_b, _y in contrasts:
            selected = animal_table.loc[
                animal_table["comparison"].astype(str).isin(
                    (comparison_a, comparison_b)
                )
            ]
            observed_strata = set(
                selected["decoding_trajectory"].astype(str).unique()
            )
            expected_strata = comparison_strata.get(
                comparison_a,
                set(),
            ) | comparison_strata.get(comparison_b, set())
            if observed_strata != expected_strata:
                raise ValueError(
                    "Incomplete decoding-trajectory coverage for "
                    f"{animal_name} contrast {comparison_a!r} vs "
                    f"{comparison_b!r}: expected "
                    f"{sorted(expected_strata)!r}, observed "
                    f"{sorted(observed_strata)!r}."
                )
            result = stratified_median_permutation_test(
                animal_table,
                comparison_a,
                comparison_b,
                n_permutations=n_permutations,
                rng=rng,
            )
            records.append(
                {
                    "animal_name": str(animal_name),
                    "comparison_a": comparison_a,
                    "comparison_b": comparison_b,
                    **result,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=DECODING_PERMUTATION_RESULT_COLUMNS,
    )


def significance_stars(p_value: float) -> str:
    """Return the Figure 1 significance label for one two-sided p-value."""
    if not np.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
        raise ValueError("p_value must be finite and between 0 and 1.")
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def build_decoding_significance_brackets(
    per_animal_results: Any,
    *,
    animal_names: Sequence[str] = PANEL_H_DECODING_ANIMALS,
    contrasts: Sequence[tuple[str, str, float]] = DECODING_SIGNIFICANCE_CONTRASTS,
) -> tuple[tuple[float, float, float, str], ...]:
    """Aggregate per-animal tests into conservative Figure 1 brackets."""
    required_columns = {
        "animal_name",
        "comparison_a",
        "comparison_b",
        "median_difference",
        "p_two_sided",
    }
    missing_columns = required_columns.difference(per_animal_results.columns)
    if missing_columns:
        raise ValueError(
            "Permutation-test table is missing required columns: "
            f"{sorted(missing_columns)!r}."
        )

    expected_animals = tuple(str(animal_name) for animal_name in animal_names)
    if not expected_animals or len(set(expected_animals)) != len(expected_animals):
        raise ValueError("animal_names must contain unique animal identifiers.")
    observed_animals = set(per_animal_results["animal_name"].astype(str))
    if observed_animals != set(expected_animals):
        raise ValueError(
            "Permutation-test animals do not match Figure 1 animals: "
            f"expected {sorted(expected_animals)!r}, "
            f"observed {sorted(observed_animals)!r}."
        )

    comparison_positions = {
        comparison: float(index)
        for index, (comparison, _label, _family, _pairs) in enumerate(
            DECODING_CROSS_TRAJECTORY_COMPARISONS,
            start=1,
        )
    }
    brackets = []
    result_animals = per_animal_results["animal_name"].astype(str)
    result_comparison_a = per_animal_results["comparison_a"].astype(str)
    result_comparison_b = per_animal_results["comparison_b"].astype(str)
    for comparison_a, comparison_b, y in contrasts:
        if (
            comparison_a not in comparison_positions
            or comparison_b not in comparison_positions
        ):
            raise ValueError(
                "Significance contrast references a comparison that is not "
                f"plotted: {comparison_a!r} vs {comparison_b!r}."
            )
        contrast_results = per_animal_results.loc[
            (result_comparison_a == comparison_a)
            & (result_comparison_b == comparison_b)
            & result_animals.isin(expected_animals)
        ]
        contrast_animals = contrast_results["animal_name"].astype(str)
        counts = contrast_animals.value_counts()
        if (
            set(contrast_animals) != set(expected_animals)
            or len(contrast_results) != len(expected_animals)
            or not np.all(counts.to_numpy(dtype=int) == 1)
        ):
            raise ValueError(
                "Expected exactly one permutation result per Figure 1 animal "
                f"for {comparison_a!r} vs {comparison_b!r}."
            )
        p_values = np.asarray(
            contrast_results["p_two_sided"],
            dtype=float,
        )
        median_differences = np.asarray(
            contrast_results["median_difference"],
            dtype=float,
        )
        if not np.all(np.isfinite(p_values)) or np.any(
            (p_values < 0.0) | (p_values > 1.0)
        ):
            raise ValueError(
                "Permutation results contain invalid two-sided p-values for "
                f"{comparison_a!r} vs {comparison_b!r}."
            )
        if not np.all(np.isfinite(median_differences)) or np.any(
            median_differences >= 0.0
        ):
            raise ValueError(
                "Figure 1 expects comparison A to have lower median trial "
                f"errors for every animal in {comparison_a!r} vs "
                f"{comparison_b!r}."
            )

        aggregate_p_value = float(np.max(p_values))
        x_a = comparison_positions[comparison_a]
        x_b = comparison_positions[comparison_b]
        brackets.append(
            (
                min(x_a, x_b),
                max(x_a, x_b),
                float(y),
                significance_stars(aggregate_p_value),
            )
        )
    return tuple(brackets)


def filter_datasets_by_animals(
    datasets: Sequence[DatasetId | FigureEpochDatasetId | tuple[str, str]],
    animal_names: Sequence[str],
) -> list[DatasetId]:
    """Return normalized data-set IDs for selected animals."""
    selected_animals = {str(animal_name) for animal_name in animal_names}
    return [
        normalized_dataset
        for dataset in datasets
        if (normalized_dataset := normalize_dataset_id(dataset))[0] in selected_animals
    ]


def load_dark_epoch_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str] = STABILITY_REGIONS,
) -> Any:
    """Load pooled dark-epoch stability rows for the requested data sets."""
    import pandas as pd

    tables = []
    selected_regions = set(regions)
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        table_path = get_stability_table_path(data_root, animal_name, date)
        if not table_path.exists():
            raise FileNotFoundError(
                "Missing task-progression stability table. Expected "
                f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
                "for this session first."
            )
        table = pd.read_parquet(table_path)
        table = table[
            (table["epoch"].astype(str) == dark_epoch)
            & (table["region"].astype(str).isin(selected_regions))
            & (table["trajectory_type"].astype(str).isin(TRAJECTORY_TYPES))
        ].copy()
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=STABILITY_TABLE_COLUMNS)
    return pd.concat(tables, axis=0, ignore_index=True)


def load_motor_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = MOTOR_DELTA_REGION,
    delta_metric: str = MOTOR_DELTA_METRIC,
    min_movement_firing_rate_hz: float | None = (
        MOTOR_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_tuning_stability_correlation: float | None = (
        MOTOR_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load motor-model deltas for active, stable dark-epoch V1 units."""
    import pandas as pd
    import xarray as xr

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        stable_units = load_units_by_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
        )
        stable_unit_set = (
            {int(unit) for unit in np.asarray(stable_units).reshape(-1)}
            if stable_units is not None
            else None
        )
        nested_cv_path = find_motor_nested_cv_path(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
        )
        fit_dataset = xr.open_dataset(nested_cv_path)
        try:
            values = np.asarray(
                fit_dataset["pooled_delta_bits_per_spike"]
                .sel(delta_metric=delta_metric)
                .values,
                dtype=float,
            ).reshape(-1)
            units = np.asarray(fit_dataset.coords["unit"].values)
        finally:
            fit_dataset.close()

        for unit, value in zip(units, values, strict=True):
            unit_id = int(unit)
            if stable_unit_set is not None and unit_id not in stable_unit_set:
                continue
            rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": epoch,
                    "region": region,
                    "unit": unit_id,
                    "delta_log_likelihood_bits_per_spike": float(value),
                    "source_path": str(nested_cv_path),
                }
            )

    if not rows:
        return pd.DataFrame(columns=MOTOR_DELTA_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=MOTOR_DELTA_TABLE_COLUMNS)


def _to_scalar(value: Any) -> Any:
    """Return a Python scalar for NumPy scalar values."""
    return value.item() if hasattr(value, "item") else value


def load_encoding_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str = ENCODING_COMPARISON_REGION,
    n_folds: int = ENCODING_COMPARISON_N_FOLDS,
    bin_size_s: float = ENCODING_COMPARISON_BIN_SIZE_S,
    place_bin_size_cm: float = ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
    min_spikes: int = ENCODING_COMPARISON_MIN_SPIKES,
    min_movement_firing_rate_hz: float | None = (
        ENCODING_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_tuning_stability_correlation: float | None = (
        ENCODING_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load DPP-model deltas for active, stable dark-epoch V1 units."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        stable_units = load_units_by_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
        )
        stable_unit_set = (
            {int(unit) for unit in np.asarray(stable_units).reshape(-1)}
            if stable_units is not None
            else None
        )
        summary_path = find_encoding_summary_path(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            n_folds=n_folds,
            bin_size_s=bin_size_s,
            place_bin_size_cm=place_bin_size_cm,
        )
        table = pd.read_parquet(summary_path)
        missing_columns = [
            source_column
            for _comparison, _label, source_column in ENCODING_DPP_COMPARISONS
            if source_column not in table.columns
        ]
        if "n_spikes" not in table.columns:
            missing_columns.append("n_spikes")
        if missing_columns:
            raise ValueError(
                f"Encoding summary {summary_path} is missing columns "
                f"{missing_columns!r}."
            )

        if int(min_spikes) > 0:
            table = table[np.asarray(table["n_spikes"], dtype=float) >= int(min_spikes)]
        if stable_unit_set is not None:
            units = np.asarray(table.index)
            stable_mask = np.asarray(
                [int(unit) in stable_unit_set for unit in units],
                dtype=bool,
            )
            table = table.iloc[stable_mask]
        units = np.asarray(table.index)
        n_spikes = np.asarray(table["n_spikes"], dtype=int)
        for comparison, label, source_column in ENCODING_DPP_COMPARISONS:
            values = -np.asarray(table[source_column], dtype=float).reshape(-1)
            rows.extend(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": epoch,
                    "region": region,
                    "unit": _to_scalar(unit),
                    "n_spikes": int(spike_count),
                    "comparison": comparison,
                    "comparison_label": label,
                    "delta_log_likelihood_bits_per_spike": float(value),
                    "source_path": str(summary_path),
                }
                for unit, spike_count, value in zip(units, n_spikes, values, strict=True)
            )

    if not rows:
        return pd.DataFrame(columns=ENCODING_DELTA_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=ENCODING_DELTA_TABLE_COLUMNS)


def build_unit_keys(
    animal_name: str,
    date: str,
    region: str,
    units: np.ndarray,
) -> np.ndarray:
    """Return globally unique unit keys for pooled heatmap alignment."""
    return np.asarray(
        [f"{animal_name}:{date}:{region}:{unit}" for unit in units.tolist()],
        dtype=object,
    )


def has_plottable_values(values: np.ndarray) -> bool:
    """Return whether one matrix contains positive finite values."""
    values = np.asarray(values, dtype=float)
    return bool(values.size and np.isfinite(values).any() and np.nanmax(values) > 0)


def compute_tuning_curve_peak_positions(values: np.ndarray) -> np.ndarray:
    """Return midpoint peak bins for finite, non-flat tuning curves."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D tuning matrix, got shape {values.shape}.")

    peak_positions = np.full(values.shape[0], np.nan, dtype=float)
    for row_index, row_values in enumerate(values):
        finite = np.isfinite(row_values)
        if np.count_nonzero(finite) < 2:
            continue
        finite_values = row_values[finite]
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        if np.isclose(minimum, maximum):
            continue
        peak_bins = np.flatnonzero(
            finite & np.isclose(row_values, maximum, rtol=1e-10, atol=1e-12)
        )
        if peak_bins.size:
            peak_positions[row_index] = float(np.mean(peak_bins))
    return peak_positions


def extract_tuning_curve_arrays(tuning_curve: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return unit labels and a unit-by-position value matrix from one tuning curve."""
    if len(tuning_curve.dims) != 2:
        raise ValueError(
            "Expected a 2D tuning curve with unit and position dimensions. "
            f"Got dims {tuning_curve.dims!r}."
        )
    unit_dim, pos_dim = tuning_curve.dims
    values = np.asarray(
        tuning_curve.transpose(unit_dim, pos_dim).values,
        dtype=float,
    )
    units = np.asarray(tuning_curve.coords[unit_dim].values)
    return units, values


def compute_unit_movement_firing_rates(spikes: Any, movement_interval: Any) -> dict[Any, float]:
    """Return mean firing rates for all units during one movement interval."""
    duration = float(movement_interval.tot_length())
    unit_ids = list(spikes.keys())
    if duration <= 0:
        return {unit_id: 0.0 for unit_id in unit_ids}

    counts = spikes.count(ep=movement_interval).to_numpy()
    rates = np.sum(np.asarray(counts, dtype=float), axis=0).ravel() / duration
    return {
        unit_id: float(rate)
        for unit_id, rate in zip(unit_ids, rates, strict=True)
    }


def select_units_by_movement_firing_rate(
    movement_firing_rates: dict[Any, float],
    min_movement_firing_rate_hz: float | None,
) -> np.ndarray:
    """Return units whose movement firing rate meets the requested threshold."""
    if min_movement_firing_rate_hz is None:
        return np.asarray(list(movement_firing_rates.keys()))
    if min_movement_firing_rate_hz < 0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")
    return np.asarray(
        [
            unit_id
            for unit_id, rate in movement_firing_rates.items()
            if rate >= min_movement_firing_rate_hz
        ]
    )


def select_units_by_tuning_stability(
    stability_table: Any,
    min_tuning_stability_correlation: float | None,
) -> np.ndarray:
    """Return units stable in at least one trajectory at the requested threshold."""
    if min_tuning_stability_correlation is None:
        return np.asarray(stability_table["unit"].drop_duplicates())
    if min_tuning_stability_correlation < -1.0:
        raise ValueError("min_tuning_stability_correlation must be at least -1.")

    correlations = np.asarray(stability_table["stability_correlation"], dtype=float)
    stable_rows = stability_table[
        np.isfinite(correlations)
        & (correlations >= float(min_tuning_stability_correlation))
    ]
    return np.asarray(stable_rows["unit"].drop_duplicates())


def select_units_by_saved_movement_firing_rate(
    stability_table: Any,
    min_movement_firing_rate_hz: float | None,
) -> np.ndarray:
    """Return units meeting an epoch-level movement firing-rate threshold."""
    if min_movement_firing_rate_hz is None:
        return np.asarray(stability_table["unit"].drop_duplicates())
    if min_movement_firing_rate_hz < 0.0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")
    if "firing_rate_hz" not in stability_table.columns:
        raise ValueError(
            "Stability table is missing firing_rate_hz required for unit selection."
        )

    selected_units: list[Any] = []
    for unit, unit_rows in stability_table.groupby("unit", sort=False):
        firing_rates = np.asarray(unit_rows["firing_rate_hz"], dtype=float)
        finite_rates = firing_rates[np.isfinite(firing_rates)]
        if finite_rates.size == 0:
            continue
        if not np.allclose(
            finite_rates,
            finite_rates[0],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                f"Unit {unit!r} has inconsistent movement firing rates in one epoch."
            )
        if float(finite_rates[0]) >= float(min_movement_firing_rate_hz):
            selected_units.append(unit)
    return np.asarray(selected_units)


def load_units_by_tuning_stability(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_movement_firing_rate_hz: float | None = None,
    min_tuning_stability_correlation: float | None,
) -> np.ndarray | None:
    """Return the intersection of saved firing-rate and stability criteria."""
    if (
        min_movement_firing_rate_hz is None
        and min_tuning_stability_correlation is None
    ):
        return None

    import pandas as pd

    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(table_path)
    table = table[
        (table["epoch"].astype(str) == str(epoch))
        & (table["region"].astype(str) == str(region))
        & (table["trajectory_type"].astype(str).isin(PANEL_D_TRAJECTORY_TYPES))
    ]
    firing_rate_units = select_units_by_saved_movement_firing_rate(
        table,
        min_movement_firing_rate_hz,
    )
    stable_units = select_units_by_tuning_stability(
        table,
        min_tuning_stability_correlation,
    )
    return np.intersect1d(firing_rate_units, stable_units)


def filter_tuning_curve_units(tuning_curve: Any | None, included_units: np.ndarray) -> Any | None:
    """Return one tuning curve restricted to the requested units."""
    if tuning_curve is None:
        return None
    if len(tuning_curve.dims) != 2:
        raise ValueError(
            "Expected a 2D tuning curve with unit and position dimensions. "
            f"Got dims {tuning_curve.dims!r}."
        )
    unit_dim = tuning_curve.dims[0]
    units = np.asarray(tuning_curve.coords[unit_dim].values)
    unit_mask = np.isin(units, np.asarray(included_units))
    return tuning_curve.isel({unit_dim: np.flatnonzero(unit_mask)})


def filter_tuning_curves_by_units(
    curves_by_trajectory: dict[str, Any | None],
    included_units: np.ndarray,
) -> dict[str, Any | None]:
    """Return trajectory tuning curves restricted to requested units."""
    return {
        trajectory_type: filter_tuning_curve_units(curve, included_units)
        for trajectory_type, curve in curves_by_trajectory.items()
    }


def normalize_linear_position_by_trajectory(
    animal_name: str,
    linear_position_by_trajectory: dict[str, Any],
) -> dict[str, Any]:
    """Scale linear trajectory coordinates to normalized 0-1 position."""
    import pynapple as nap

    total_length_cm = float(get_wtrack_total_length(animal_name))
    if total_length_cm <= 0:
        raise ValueError(f"W-track total length must be positive for {animal_name!r}.")

    normalized: dict[str, Any] = {}
    for trajectory_type, linear_position in linear_position_by_trajectory.items():
        normalized[trajectory_type] = nap.Tsd(
            t=np.asarray(linear_position.t, dtype=float),
            d=np.asarray(linear_position.d, dtype=float) / total_length_cm,
            time_support=linear_position.time_support,
            time_units="s",
        )
    return normalized


def compute_dark_epoch_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    epoch: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
    min_movement_firing_rate_hz: float | None = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = None,
) -> dict[str, Any]:
    """Compute odd/even normalized-position tuning curves for one dark epoch."""
    session = prepare_heatmap_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )

    selected_epoch = session["run_epochs"][0]
    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        session["position_by_epoch"][selected_epoch],
        session["timestamps_position"][selected_epoch],
        session["trajectory_intervals"][selected_epoch],
        position_offset=position_offset,
        use_trajectory_direction=use_trajectory_direction,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    odd_curves, even_curves = compute_odd_even_place_tuning_curves(
        session["spikes_by_region"][region],
        normalized_position_by_trajectory,
        session["trajectory_intervals"][selected_epoch],
        session["movement_by_run"][selected_epoch],
        bin_edges=build_normalized_position_bins(position_bin_count),
        sigma_bins=sigma_bins,
    )
    movement_firing_rates = compute_unit_movement_firing_rates(
        session["spikes_by_region"][region],
        session["movement_by_run"][selected_epoch],
    )
    included_units = select_units_by_movement_firing_rate(
        movement_firing_rates,
        min_movement_firing_rate_hz,
    )
    stable_units = load_units_by_tuning_stability(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=selected_epoch,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
    )
    if stable_units is not None:
        included_units = np.intersect1d(included_units, stable_units)
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": selected_epoch,
        "odd_curves": filter_tuning_curves_by_units(odd_curves, included_units),
        "even_curves": filter_tuning_curves_by_units(even_curves, included_units),
        "included_units": included_units,
        "movement_firing_rates_hz": movement_firing_rates,
    }


def get_unit_spike_times(spikes: Any, unit_id: int) -> np.ndarray:
    """Return spike times for one unit from a pynapple TsGroup-like object."""
    if unit_id not in list(spikes.keys()):
        raise ValueError(f"Unit {unit_id!r} was not found in the requested spikes.")
    return np.asarray(spikes[unit_id].t, dtype=float)


def extract_unit_rate_curve(
    tuning_curve: Any | None,
    unit_id: int,
    fallback_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized position and firing-rate values for one unit."""
    fallback_position = np.asarray(fallback_position, dtype=float)
    if tuning_curve is None:
        return fallback_position, np.full(fallback_position.shape, np.nan, dtype=float)

    if len(tuning_curve.dims) != 2:
        raise ValueError(
            "Expected a 2D tuning curve with unit and position dimensions. "
            f"Got dims {tuning_curve.dims!r}."
        )
    unit_dim, position_dim = tuning_curve.dims
    units = np.asarray(tuning_curve.coords[unit_dim].values)
    matching_indices = np.flatnonzero(units == unit_id)
    if matching_indices.size == 0:
        return fallback_position, np.full(fallback_position.shape, np.nan, dtype=float)

    values = np.asarray(
        tuning_curve.transpose(unit_dim, position_dim).values,
        dtype=float,
    )
    position = np.asarray(tuning_curve.coords[position_dim].values, dtype=float)
    return position, values[int(matching_indices[0])]


def orient_panel_e_task_progression(linear_position: Any, trajectory_type: str) -> Any:
    """Orient normalized position so 0 is the trajectory start for panel E."""
    import pynapple as nap

    values = np.asarray(linear_position.d, dtype=float)
    if trajectory_type.endswith("_to_center"):
        values = 1.0 - values
    return nap.Tsd(
        t=np.asarray(linear_position.t, dtype=float),
        d=values,
        time_support=linear_position.time_support,
        time_units="s",
    )


def load_panel_e_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load one panel E example unit with normalized-position rasters and rate curves."""
    session = prepare_heatmap_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )
    selected_epoch = session["run_epochs"][0]
    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        session["position_by_epoch"][selected_epoch],
        session["timestamps_position"][selected_epoch],
        session["trajectory_intervals"][selected_epoch],
        position_offset=position_offset,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    task_progression_by_trajectory = {
        trajectory_type: orient_panel_e_task_progression(
            normalized_position,
            trajectory_type,
        )
        for trajectory_type, normalized_position in normalized_position_by_trajectory.items()
    }

    spikes = session["spikes_by_region"][region]
    spike_times_s = get_unit_spike_times(spikes, unit_id)
    bin_edges = build_normalized_position_bins(position_bin_count)
    fallback_position = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    raster_positions: dict[str, list[np.ndarray]] = {}
    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        task_progression = task_progression_by_trajectory[trajectory_type]
        task_progression_interpolator = make_linear_position_interpolator(task_progression)
        trial_positions = compute_trial_spike_positions(
            spike_times_s,
            session["trajectory_intervals"][selected_epoch][trajectory_type],
            task_progression_interpolator,
        )
        raster_positions[trajectory_type] = [
            positions[(positions >= 0.0) & (positions <= 1.0)]
            for positions in trial_positions
        ]

        movement_epochs = session["trajectory_intervals"][selected_epoch][
            trajectory_type
        ].intersect(session["movement_by_run"][selected_epoch])
        try:
            tuning_curve = compute_place_tuning_curve(
                spikes,
                task_progression,
                movement_epochs,
                bin_edges=bin_edges,
                sigma_bins=sigma_bins,
            )
        except (IndexError, UnboundLocalError):
            tuning_curve = None
        firing_rates[trajectory_type] = extract_unit_rate_curve(
            tuning_curve,
            unit_id,
            fallback_position,
        )

    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": selected_epoch,
        "region": region,
        "unit_id": unit_id,
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }


def build_panel_e_cache_metadata(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Return metadata that identifies one Panel E example-cell cache."""
    return {
        "cache_version": PANEL_E_CACHE_VERSION,
        "data_root": str(Path(data_root)),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "region": str(region),
        "unit_id": int(unit_id),
        "trajectory_types": list(TRAJECTORY_TYPES),
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
    }


def build_panel_e_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for one Panel E example-cell payload."""
    dataset_token = "-".join(
        _format_panel_d_cache_token(value)
        for value in (
            metadata["animal_name"],
            metadata["date"],
            metadata["epoch"],
            metadata["region"],
            f"unit{metadata['unit_id']}",
        )
    )
    filename = (
        f"{PANEL_E_CACHE_PREFIX}_{dataset_token}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_d_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_d_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_e_cache_trajectory_token(trajectory_type: str) -> str:
    """Return a compact trajectory token for Panel E cache array names."""
    return _format_panel_d_cache_token(trajectory_type)


def save_panel_e_cache(
    cache_path: Path,
    example_data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write one Panel E example-cell cache as compressed NumPy arrays."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        PANEL_E_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for trajectory_type in TRAJECTORY_TYPES:
        token = _panel_e_cache_trajectory_token(trajectory_type)
        raster_trials = [
            np.asarray(trial_positions, dtype=float)
            for trial_positions in example_data["raster_positions"][trajectory_type]
        ]
        if raster_trials:
            payload[f"raster_{token}_values"] = np.concatenate(raster_trials)
            payload[f"raster_{token}_lengths"] = np.asarray(
                [trial_positions.size for trial_positions in raster_trials],
                dtype=int,
            )
        else:
            payload[f"raster_{token}_values"] = np.asarray([], dtype=float)
            payload[f"raster_{token}_lengths"] = np.asarray([], dtype=int)

        rate_position, rate_values = example_data["firing_rates"][trajectory_type]
        payload[f"rate_{token}_position"] = np.asarray(rate_position, dtype=float)
        payload[f"rate_{token}_values"] = np.asarray(rate_values, dtype=float)

    np.savez_compressed(cache_path, **payload)


def load_panel_e_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return cached Panel E example-cell data when metadata still matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_E_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Panel E cache at {cache_path}.")
                return None

            raster_positions: dict[str, list[np.ndarray]] = {}
            firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for trajectory_type in TRAJECTORY_TYPES:
                token = _panel_e_cache_trajectory_token(trajectory_type)
                values = np.asarray(data[f"raster_{token}_values"], dtype=float)
                lengths = np.asarray(data[f"raster_{token}_lengths"], dtype=int)
                split_points = np.cumsum(lengths)[:-1]
                raster_positions[trajectory_type] = (
                    [
                        np.asarray(trial_positions, dtype=float)
                        for trial_positions in np.split(values, split_points)
                    ]
                    if lengths.size
                    else []
                )
                firing_rates[trajectory_type] = (
                    np.asarray(data[f"rate_{token}_position"], dtype=float),
                    np.asarray(data[f"rate_{token}_values"], dtype=float),
                )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable Panel E cache at {cache_path}: {exc}")
        return None

    return {
        "animal_name": str(expected_metadata["animal_name"]),
        "date": str(expected_metadata["date"]),
        "epoch": str(expected_metadata["epoch"]),
        "region": str(expected_metadata["region"]),
        "unit_id": int(expected_metadata["unit_id"]),
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }


def load_or_compute_panel_e_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_e_cache_dir: Path | None,
    refresh_panel_e_cache: bool,
) -> dict[str, Any]:
    """Load cached Panel E example-cell data or compute and cache it."""
    metadata = build_panel_e_cache_metadata(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        unit_id=unit_id,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    cache_path = (
        build_panel_e_cache_path(panel_e_cache_dir, metadata)
        if panel_e_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_e_cache:
        cached_example = load_panel_e_cache(cache_path, metadata)
        if cached_example is not None:
            print(f"Loaded Panel E cache from {cache_path}.")
            return cached_example

    example_data = load_panel_e_example_data(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        unit_id=unit_id,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    if cache_path is not None:
        save_panel_e_cache(cache_path, example_data, metadata)
        print(f"Saved Panel E cache to {cache_path}.")
    return example_data


def _concatenate_unit_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate pooled unit-key chunks."""
    if not parts:
        return np.asarray([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def _concatenate_value_parts(parts: list[np.ndarray], position_bin_count: int) -> np.ndarray:
    """Concatenate pooled tuning-matrix chunks."""
    if not parts:
        return np.empty((0, position_bin_count), dtype=float)
    return np.vstack(parts)


def align_panel_values_to_unit_order(
    display_values: np.ndarray,
    display_units: np.ndarray,
    reference_units: np.ndarray,
    unit_order: np.ndarray,
) -> np.ndarray:
    """Align one display matrix to reference units and apply the learned unit order."""
    display_values = np.asarray(display_values, dtype=float)
    display_units = np.asarray(display_units)
    reference_units = np.asarray(reference_units)
    unit_order = np.asarray(unit_order, dtype=int)

    if display_values.ndim != 2:
        raise ValueError(f"Expected a 2D tuning matrix, got shape {display_values.shape}.")
    if display_values.shape[0] != display_units.size:
        raise ValueError(
            "Display matrix rows must match the number of display units. "
            f"Got {display_values.shape[0]} rows and {display_units.size} units."
        )
    if unit_order.shape != (reference_units.size,):
        raise ValueError(
            "unit_order must contain one index per reference unit. "
            f"Got shape {unit_order.shape} for {reference_units.size} units."
        )

    aligned = np.full((reference_units.size, display_values.shape[1]), np.nan, dtype=float)
    index_by_unit = {unit: index for index, unit in enumerate(display_units.tolist())}
    for reference_index, unit in enumerate(reference_units.tolist()):
        display_index = index_by_unit.get(unit)
        if display_index is not None:
            aligned[reference_index] = display_values[display_index]
    return aligned[unit_order]


def normalize_panel_values_across_trajectories(
    sorted_values_by_trajectory: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Normalize each unit by its maximum across all displayed trajectories."""
    if not sorted_values_by_trajectory:
        return {}

    combined_values = np.concatenate(
        list(sorted_values_by_trajectory.values()),
        axis=1,
    )
    row_max = np.full(combined_values.shape[0], np.nan, dtype=float)
    finite_rows = np.isfinite(combined_values).any(axis=1)
    if np.any(finite_rows):
        row_max[finite_rows] = np.nanmax(combined_values[finite_rows], axis=1)

    valid_rows = np.isfinite(row_max) & (row_max > 0)
    normalized_by_trajectory: dict[str, np.ndarray] = {}
    for trajectory_type, values in sorted_values_by_trajectory.items():
        normalized = np.full_like(values, np.nan, dtype=float)
        if np.any(valid_rows):
            normalized[valid_rows] = values[valid_rows] / row_max[valid_rows, None]
        normalized_by_trajectory[trajectory_type] = normalized
    return normalized_by_trajectory


def normalize_panel_values_per_trajectory(values: np.ndarray) -> np.ndarray:
    """Normalize each unit by its maximum within one displayed trajectory."""
    values = np.asarray(values, dtype=float)
    row_max = np.full(values.shape[0], np.nan, dtype=float)
    finite_rows = np.isfinite(values).any(axis=1)
    if np.any(finite_rows):
        row_max[finite_rows] = np.nanmax(values[finite_rows], axis=1)

    valid_rows = np.isfinite(row_max) & (row_max > 0)
    normalized = np.full_like(values, np.nan, dtype=float)
    if np.any(valid_rows):
        normalized[valid_rows] = values[valid_rows] / row_max[valid_rows, None]
    return normalized


def _build_pooled_panel_values_order_and_peaks(
    curve_sets: Sequence[dict[str, Any]],
    *,
    position_bin_count: int,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Return pooled heatmaps, row unit keys, and aligned odd-lap peaks."""
    trajectory_types = tuple(trajectory_types)
    if firing_rate_normalization not in {
        PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    }:
        raise ValueError(
            f"Unknown firing_rate_normalization {firing_rate_normalization!r}."
        )
    odd_units_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in trajectory_types
    }
    odd_values_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in trajectory_types
    }
    even_units_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in trajectory_types
    }
    even_values_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in trajectory_types
    }

    for curve_set in curve_sets:
        animal_name = str(curve_set["animal_name"])
        date = str(curve_set["date"])
        region = str(curve_set["region"])
        for trajectory_type in trajectory_types:
            odd_curve = curve_set["odd_curves"].get(trajectory_type)
            if odd_curve is not None:
                units, values = extract_tuning_curve_arrays(odd_curve)
                odd_units_by_trajectory[trajectory_type].append(
                    build_unit_keys(animal_name, date, region, units)
                )
                odd_values_by_trajectory[trajectory_type].append(values)

            even_curve = curve_set["even_curves"].get(trajectory_type)
            if even_curve is not None:
                units, values = extract_tuning_curve_arrays(even_curve)
                even_units_by_trajectory[trajectory_type].append(
                    build_unit_keys(animal_name, date, region, units)
                )
                even_values_by_trajectory[trajectory_type].append(values)

    panels: dict[tuple[str, str], np.ndarray] = {}
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray] = {}
    ordered_peak_positions_by_trajectory: dict[str, np.ndarray] = {}
    for order_trajectory in trajectory_types:
        reference_units = _concatenate_unit_parts(odd_units_by_trajectory[order_trajectory])
        order_values = _concatenate_value_parts(
            odd_values_by_trajectory[order_trajectory],
            position_bin_count,
        )
        if reference_units.size and has_plottable_values(order_values):
            unit_order = compute_unit_order(order_values)
        else:
            unit_order = np.asarray([], dtype=int)
        ordered_unit_keys_by_trajectory[order_trajectory] = (
            reference_units[unit_order] if unit_order.size else reference_units
        )
        peak_positions = compute_tuning_curve_peak_positions(order_values)
        ordered_peak_positions_by_trajectory[order_trajectory] = (
            peak_positions[unit_order] if unit_order.size else peak_positions
        )

        sorted_values_by_plot_trajectory: dict[str, np.ndarray] = {}
        for plot_trajectory in trajectory_types:
            display_units = _concatenate_unit_parts(even_units_by_trajectory[plot_trajectory])
            display_values = _concatenate_value_parts(
                even_values_by_trajectory[plot_trajectory],
                position_bin_count,
            )
            if unit_order.size == 0 or display_units.size == 0:
                sorted_values_by_plot_trajectory[plot_trajectory] = np.full(
                    (reference_units.size, position_bin_count),
                    np.nan,
                    dtype=float,
                )
                continue
            sorted_values_by_plot_trajectory[plot_trajectory] = (
                align_panel_values_to_unit_order(
                    display_values,
                    display_units,
                    reference_units,
                    unit_order,
                )
            )

        if (
            firing_rate_normalization
            == PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION
        ):
            normalized_values_by_plot_trajectory = (
                normalize_panel_values_across_trajectories(
                    sorted_values_by_plot_trajectory,
                )
            )
        else:
            normalized_values_by_plot_trajectory = {
                trajectory_type: normalize_panel_values_per_trajectory(values)
                for trajectory_type, values in sorted_values_by_plot_trajectory.items()
            }
        for plot_trajectory in trajectory_types:
            panels[(order_trajectory, plot_trajectory)] = (
                normalized_values_by_plot_trajectory[plot_trajectory]
            )
    return (
        panels,
        ordered_unit_keys_by_trajectory,
        ordered_peak_positions_by_trajectory,
    )


def build_pooled_panel_values_and_unit_order(
    curve_sets: Sequence[dict[str, Any]],
    *,
    position_bin_count: int,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """Return normalized pooled heatmaps and their row unit keys."""
    panels, ordered_unit_keys, _ordered_peak_positions = (
        _build_pooled_panel_values_order_and_peaks(
            curve_sets,
            position_bin_count=position_bin_count,
            trajectory_types=trajectory_types,
            firing_rate_normalization=firing_rate_normalization,
        )
    )
    return panels, ordered_unit_keys


def build_pooled_panel_values(
    curve_sets: Sequence[dict[str, Any]],
    *,
    position_bin_count: int,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
) -> dict[tuple[str, str], np.ndarray]:
    """Return normalized heatmap panels pooled across data sets."""
    panels, _ordered_unit_keys_by_trajectory = build_pooled_panel_values_and_unit_order(
        curve_sets,
        position_bin_count=position_bin_count,
        trajectory_types=trajectory_types,
        firing_rate_normalization=firing_rate_normalization,
    )
    return panels


def _format_panel_d_cache_token(value: object) -> str:
    """Return a filesystem-safe token for Panel D cache file names."""
    text = str(value).strip()
    cleaned = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            cleaned.append(character)
        elif character == ".":
            cleaned.append("p")
        else:
            cleaned.append("-")
    token = "".join(cleaned).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "none"


def _format_panel_d_cache_number(value: float | int) -> str:
    """Return a compact numeric token for Panel D cache file names."""
    return _format_panel_d_cache_token(f"{float(value):g}")


def _build_panel_d_dataset_cache_token(
    dataset_metadata: Sequence[dict[str, str]],
) -> str:
    """Return a descriptive cache token for the Panel D data-set list."""
    dataset_tokens = [
        _format_panel_d_cache_token(
            f"{dataset['animal_name']}-{dataset['date']}-{dataset['dark_epoch']}"
        )
        for dataset in dataset_metadata
    ]
    token = "_".join(dataset_tokens) or "none"
    if len(token) <= PANEL_D_CACHE_DATASET_TOKEN_LIMIT:
        return token

    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    prefix = "_".join(dataset_tokens[:2])
    return _format_panel_d_cache_token(
        f"{prefix}_{len(dataset_tokens)}datasets_{digest}"
    )


def build_panel_d_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
    min_movement_firing_rate_hz: float | None = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> dict[str, Any]:
    """Return metadata that identifies one Panel D heatmap cache."""
    if min_movement_firing_rate_hz is not None and min_movement_firing_rate_hz < 0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")
    if (
        min_tuning_stability_correlation is not None
        and min_tuning_stability_correlation < -1.0
    ):
        raise ValueError("min_tuning_stability_correlation must be at least -1.")
    if firing_rate_normalization == PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION:
        cache_version = PANEL_D_CACHE_VERSION
    elif firing_rate_normalization == PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION:
        cache_version = PANEL_D_ACROSS_TRAJECTORY_CACHE_VERSION
    else:
        raise ValueError(
            f"Unknown firing_rate_normalization {firing_rate_normalization!r}."
        )
    dataset_metadata = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        dataset_metadata.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": dark_epoch,
            }
        )

    metadata = {
        "cache_version": cache_version,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "D",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "datasets": dataset_metadata,
        "trajectory_types": list(PANEL_D_TRAJECTORY_TYPES),
        "linear_position_orientation": PANEL_D_LINEAR_POSITION_ORIENTATION,
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "pooled_builder": "build_pooled_panel_values",
    }
    if min_movement_firing_rate_hz is not None:
        metadata["min_movement_firing_rate_hz"] = float(min_movement_firing_rate_hz)
    if min_tuning_stability_correlation is not None:
        metadata["min_tuning_stability_correlation"] = float(
            min_tuning_stability_correlation
        )
    if firing_rate_normalization != PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION:
        metadata["firing_rate_normalization"] = firing_rate_normalization
    return metadata


def build_panel_d_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for one Panel D heatmap payload."""
    region_token = _format_panel_d_cache_token(metadata["region"])
    dataset_metadata = metadata["datasets"]
    dark_epochs = [
        _format_panel_d_cache_token(dataset["dark_epoch"])
        for dataset in dataset_metadata
    ]
    unique_dark_epochs = list(dict.fromkeys(dark_epochs))
    dark_epoch_token = (
        unique_dark_epochs[0]
        if len(unique_dark_epochs) == 1
        else "mixed-" + "_".join(unique_dark_epochs)
    )
    dataset_token = _build_panel_d_dataset_cache_token(dataset_metadata)
    orientation_token = _format_panel_d_cache_token(
        metadata["linear_position_orientation"]
    )
    filename = (
        f"{PANEL_D_CACHE_PREFIX}_{region_token}_dark{dark_epoch_token}"
        f"_datasets-{dataset_token}"
        f"_orient{orientation_token}"
    )
    if "firing_rate_normalization" in metadata:
        normalization_token = _format_panel_d_cache_token(
            metadata["firing_rate_normalization"]
        )
        filename += f"_norm{normalization_token}"
    if "min_movement_firing_rate_hz" in metadata:
        filename += (
            "_minmovefr"
            f"{_format_panel_d_cache_number(metadata['min_movement_firing_rate_hz'])}"
        )
    if "min_tuning_stability_correlation" in metadata:
        filename += (
            "_minstab"
            f"{_format_panel_d_cache_number(metadata['min_tuning_stability_correlation'])}"
        )
    filename += (
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_d_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_d_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_d_metadata_trajectory_types(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return the Panel D trajectory order encoded in cache metadata."""
    return tuple(str(trajectory) for trajectory in metadata["trajectory_types"])


def _panel_d_cache_array_name(order_trajectory: str, plot_trajectory: str) -> str:
    """Return the array name for one Panel D heatmap matrix."""
    return f"{order_trajectory}__{plot_trajectory}"


def _panel_d_cache_unit_order_array_name(order_trajectory: str) -> str:
    """Return the array name for one Panel D ordered unit-key vector."""
    return f"unit_order__{order_trajectory}"


def _panel_d_cache_order_peak_array_name(order_trajectory: str) -> str:
    """Return the array name for one Panel D aligned odd-lap peak vector."""
    return f"order_peak_position__{order_trajectory}"


def save_panel_d_cache(
    cache_path: Path,
    panels: dict[tuple[str, str], np.ndarray],
    metadata: dict[str, Any],
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray] | None = None,
    ordered_peak_positions_by_trajectory: dict[str, np.ndarray] | None = None,
) -> None:
    """Write one Panel D heatmap cache as compressed NumPy arrays."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        PANEL_D_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    trajectory_types = _panel_d_metadata_trajectory_types(metadata)
    for order_trajectory in trajectory_types:
        for plot_trajectory in trajectory_types:
            payload[_panel_d_cache_array_name(order_trajectory, plot_trajectory)] = (
                np.asarray(panels[(order_trajectory, plot_trajectory)], dtype=float)
            )
        if ordered_unit_keys_by_trajectory is not None:
            payload[_panel_d_cache_unit_order_array_name(order_trajectory)] = np.asarray(
                ordered_unit_keys_by_trajectory[order_trajectory],
                dtype=str,
            )
        if ordered_peak_positions_by_trajectory is not None:
            payload[_panel_d_cache_order_peak_array_name(order_trajectory)] = np.asarray(
                ordered_peak_positions_by_trajectory[order_trajectory],
                dtype=float,
            )
    np.savez_compressed(cache_path, **payload)


def load_panel_d_cache_full_payload(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
] | None:
    """Return cached heatmaps, row keys, and odd peaks when metadata matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_D_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Panel D cache at {cache_path}.")
                return None

            panels: dict[tuple[str, str], np.ndarray] = {}
            trajectory_types = _panel_d_metadata_trajectory_types(expected_metadata)
            for order_trajectory in trajectory_types:
                for plot_trajectory in trajectory_types:
                    array_name = _panel_d_cache_array_name(
                        order_trajectory,
                        plot_trajectory,
                    )
                    panels[(order_trajectory, plot_trajectory)] = np.asarray(
                        data[array_name],
                        dtype=float,
                    )
            ordered_unit_keys_by_trajectory = {}
            ordered_peak_positions_by_trajectory = {}
            for order_trajectory in trajectory_types:
                array_name = _panel_d_cache_unit_order_array_name(order_trajectory)
                if array_name in data.files:
                    ordered_unit_keys_by_trajectory[order_trajectory] = np.asarray(
                        data[array_name],
                        dtype=str,
                    )
                peak_array_name = _panel_d_cache_order_peak_array_name(
                    order_trajectory
                )
                if peak_array_name in data.files:
                    ordered_peak_positions_by_trajectory[order_trajectory] = (
                        np.asarray(
                            data[peak_array_name],
                            dtype=float,
                        )
                    )
            return (
                panels,
                ordered_unit_keys_by_trajectory,
                ordered_peak_positions_by_trajectory,
            )
    except Exception as exc:
        print(f"Ignoring unreadable Panel D cache at {cache_path}: {exc}")
        return None


def load_panel_d_cache_payload(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]] | None:
    """Return cached Panel D heatmaps and row unit keys when metadata matches."""
    payload = load_panel_d_cache_full_payload(cache_path, expected_metadata)
    if payload is None:
        return None
    panels, ordered_unit_keys, _ordered_peak_positions = payload
    return panels, ordered_unit_keys


def load_panel_d_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray] | None:
    """Return cached Panel D heatmap matrices when metadata still matches."""
    payload = load_panel_d_cache_payload(cache_path, expected_metadata)
    if payload is None:
        return None
    panels, _ordered_unit_keys_by_trajectory = payload
    return panels


def load_panel_d_ordered_unit_keys(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    """Return cached Panel D row unit keys when metadata still matches."""
    payload = load_panel_d_cache_payload(cache_path, expected_metadata)
    if payload is None:
        return None
    _panels, ordered_unit_keys_by_trajectory = payload
    trajectory_types = _panel_d_metadata_trajectory_types(expected_metadata)
    if any(trajectory not in ordered_unit_keys_by_trajectory for trajectory in trajectory_types):
        return None
    return ordered_unit_keys_by_trajectory


def load_panel_d_order_peak_positions(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    """Return cached Panel D odd-lap peak positions when metadata matches."""
    payload = load_panel_d_cache_full_payload(cache_path, expected_metadata)
    if payload is None:
        return None
    _panels, _ordered_unit_keys, ordered_peak_positions = payload
    trajectory_types = _panel_d_metadata_trajectory_types(expected_metadata)
    if any(
        trajectory not in ordered_peak_positions
        for trajectory in trajectory_types
    ):
        return None
    return ordered_peak_positions


def load_or_compute_panel_d_heatmap_full_payload(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
    min_movement_firing_rate_hz: float | None = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
    require_ordered_unit_keys: bool = False,
    require_order_peak_positions: bool = False,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Load or compute Panel D heatmaps, row keys, and odd-lap peak positions."""
    metadata = build_panel_d_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        firing_rate_normalization=firing_rate_normalization,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
    )
    cache_path = (
        build_panel_d_cache_path(panel_d_cache_dir, metadata)
        if panel_d_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_d_cache:
        cached_payload = load_panel_d_cache_full_payload(cache_path, metadata)
        if cached_payload is not None:
            (
                panels,
                ordered_unit_keys_by_trajectory,
                ordered_peak_positions_by_trajectory,
            ) = cached_payload
            trajectory_types = _panel_d_metadata_trajectory_types(metadata)
            has_unit_order = all(
                trajectory in ordered_unit_keys_by_trajectory
                for trajectory in trajectory_types
            )
            has_order_peaks = all(
                trajectory in ordered_peak_positions_by_trajectory
                for trajectory in trajectory_types
            )
            if (
                (has_unit_order or not require_ordered_unit_keys)
                and (
                    has_order_peaks
                    or not require_order_peak_positions
                )
            ):
                print(f"Loaded Panel D cache from {cache_path}.")
                return (
                    panels,
                    ordered_unit_keys_by_trajectory,
                    ordered_peak_positions_by_trajectory,
                )

    print(f"Building pooled dark-epoch heatmap for region {region}.")
    curve_sets = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        print(f"  Loading {animal_name} {date} epoch {epoch}.")
        curve_sets.append(
            compute_dark_epoch_tuning_curves(
                animal_name=animal_name,
                date=date,
                data_root=data_root,
                region=region,
                epoch=epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                use_trajectory_direction=True,
                min_movement_firing_rate_hz=min_movement_firing_rate_hz,
                min_tuning_stability_correlation=min_tuning_stability_correlation,
            )
        )

    (
        panels,
        ordered_unit_keys_by_trajectory,
        ordered_peak_positions_by_trajectory,
    ) = _build_pooled_panel_values_order_and_peaks(
        curve_sets,
        position_bin_count=position_bin_count,
        trajectory_types=PANEL_D_TRAJECTORY_TYPES,
        firing_rate_normalization=firing_rate_normalization,
    )
    if cache_path is not None:
        save_panel_d_cache(
            cache_path,
            panels,
            metadata,
            ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
            ordered_peak_positions_by_trajectory=(
                ordered_peak_positions_by_trajectory
            ),
        )
        print(f"Saved Panel D cache to {cache_path}.")
    return (
        panels,
        ordered_unit_keys_by_trajectory,
        ordered_peak_positions_by_trajectory,
    )


def load_or_compute_panel_d_heatmap_payload(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
    min_movement_firing_rate_hz: float | None = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
    require_ordered_unit_keys: bool = False,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """Load or compute Panel D heatmaps and their row unit keys."""
    panels, ordered_unit_keys, _ordered_peak_positions = (
        load_or_compute_panel_d_heatmap_full_payload(
            data_root=data_root,
            datasets=datasets,
            region=region,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_d_cache_dir=panel_d_cache_dir,
            refresh_panel_d_cache=refresh_panel_d_cache,
            firing_rate_normalization=firing_rate_normalization,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
            require_ordered_unit_keys=require_ordered_unit_keys,
        )
    )
    return panels, ordered_unit_keys


def load_or_compute_panel_d_heatmap_panels(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
    firing_rate_normalization: str = PANEL_D_FIRING_RATE_NORMALIZATION,
    min_movement_firing_rate_hz: float | None = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> dict[tuple[str, str], np.ndarray]:
    """Load cached Panel D panels or compute and cache them."""
    panels, _ordered_unit_keys_by_trajectory = load_or_compute_panel_d_heatmap_payload(
        data_root=data_root,
        datasets=datasets,
        region=region,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=refresh_panel_d_cache,
        firing_rate_normalization=firing_rate_normalization,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
    )
    return panels


def plot_dark_heatmap_regions(
    heatmap_axes: np.ndarray,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None = None,
    refresh_panel_d_cache: bool = False,
) -> "AxesImage | None":
    """Plot pooled dark-epoch heatmaps for all requested regions."""
    color_image = None
    for region_index, region in enumerate(regions):
        panels = load_or_compute_panel_d_heatmap_panels(
            data_root=data_root,
            datasets=datasets,
            region=region,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_d_cache_dir=panel_d_cache_dir,
            refresh_panel_d_cache=refresh_panel_d_cache,
        )
        start_row = region_index * len(PANEL_D_TRAJECTORY_TYPES)
        stop_row = start_row + len(PANEL_D_TRAJECTORY_TYPES)
        image = plot_pooled_heatmap_grid(
            heatmap_axes[start_row:stop_row, :],
            panels,
            trajectory_types=PANEL_D_TRAJECTORY_TYPES,
            axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
            cmap=PANEL_D_HEATMAP_CMAP,
        )
        if color_image is None and image is not None:
            color_image = image
    return color_image


def plot_pooled_heatmap_grid(
    axes: np.ndarray,
    panels: dict[tuple[str, str], np.ndarray],
    *,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    axis_orientation: str = "branch_position",
    cmap: str = "viridis",
) -> "AxesImage | None":
    """Plot one pooled 4x4 odd/even trajectory heatmap grid."""
    trajectory_types = tuple(trajectory_types)
    color_image = None
    for row_index, order_trajectory in enumerate(trajectory_types):
        row_size = max(
            panels[(order_trajectory, plot_trajectory)].shape[0]
            for plot_trajectory in trajectory_types
        )
        y_limit = max(row_size, 1)
        for col_index, plot_trajectory in enumerate(trajectory_types):
            ax: Axes = axes[row_index, col_index]
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_limit, 0)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_index == len(trajectory_types) - 1:
                add_normalized_path_heatmap_axis(ax)

            panel_values = panels[(order_trajectory, plot_trajectory)]
            if not has_plottable_values(panel_values):
                add_task_progression_segment_boundary_lines(ax)
                continue
            image = ax.imshow(
                panel_values,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                extent=[0.0, 1.0, panel_values.shape[0], 0],
                vmin=0.0,
                vmax=1.0,
                cmap=cmap,
            )
            add_task_progression_segment_boundary_lines(ax)
            if color_image is None:
                color_image = image
    return color_image


def add_panel_d_heatmap_block_outlines(axes: np.ndarray) -> tuple[Any, ...]:
    """Draw outlines around the diagonal 2x2 blocks in the Panel D heatmap grid."""
    from matplotlib.patches import Rectangle

    heatmap_axes = np.asarray(axes, dtype=object)
    if heatmap_axes.ndim != 2:
        raise ValueError("axes must be a 2D array.")

    if (
        heatmap_axes.shape[0] < len(PANEL_D_TRAJECTORY_TYPES)
        or heatmap_axes.shape[1] < len(PANEL_D_TRAJECTORY_TYPES)
    ):
        return ()

    fig = heatmap_axes.flat[0].figure
    outlines = []
    region_row_count = len(PANEL_D_TRAJECTORY_TYPES)
    for row_start in range(0, heatmap_axes.shape[0], region_row_count):
        if row_start + region_row_count > heatmap_axes.shape[0]:
            break
        for (relative_row_start, relative_row_stop), (col_start, col_stop) in (
            PANEL_D_HEATMAP_BLOCK_OUTLINE_SPECS
        ):
            block_axes = heatmap_axes[
                row_start + relative_row_start : row_start + relative_row_stop,
                col_start:col_stop,
            ]
            positions = [block_ax.get_position() for block_ax in block_axes.ravel()]
            x0 = min(position.x0 for position in positions)
            y0 = min(position.y0 for position in positions)
            x1 = max(position.x1 for position in positions)
            y1 = max(position.y1 for position in positions)
            x0 -= PANEL_D_HEATMAP_BLOCK_OUTLINE_PAD
            y0 -= PANEL_D_HEATMAP_BLOCK_OUTLINE_PAD
            x1 += PANEL_D_HEATMAP_BLOCK_OUTLINE_PAD
            y1 += PANEL_D_HEATMAP_BLOCK_OUTLINE_PAD
            outline = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=PANEL_D_HEATMAP_BLOCK_OUTLINE_COLOR,
                linewidth=PANEL_D_HEATMAP_BLOCK_OUTLINE_LINEWIDTH,
                linestyle=PANEL_D_HEATMAP_BLOCK_OUTLINE_LINESTYLE,
                transform=fig.transFigure,
                clip_on=False,
                zorder=1000,
            )
            fig.add_artist(outline)
            outlines.append(outline)
    return tuple(outlines)


def add_task_progression_segment_boundary_lines(ax: "Axes") -> None:
    """Draw normalized task-progression segment boundaries."""
    for boundary in TASK_PROGRESSION_SEGMENT_BOUNDARIES:
        ax.axvline(
            boundary,
            color=TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR,
            linewidth=TASK_PROGRESSION_SEGMENT_BOUNDARY_LINEWIDTH,
            zorder=1,
        )


def draw_neuron_scale_bar(
    ax: "Axes",
    *,
    neuron_count: int = NEURON_SCALE_BAR_COUNT,
    x: float = 1.08,
) -> None:
    """Draw a vertical data-scaled neuron count bar beside one heatmap axis."""
    from matplotlib.transforms import blended_transform_factory

    if neuron_count <= 0:
        raise ValueError("neuron_count must be positive.")

    y_limits = [float(value) for value in ax.get_ylim()]
    y_min = min(y_limits)
    y_max = max(y_limits)
    y_span = y_max - y_min
    margin = max(8.0, 0.28 * y_span)
    if y_span >= neuron_count + margin:
        y_bottom = y_max - margin
        y_top = y_bottom - float(neuron_count)
    else:
        y_top = y_min
        y_bottom = min(y_max, y_min + float(neuron_count))

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(
        [x, x],
        [y_bottom, y_top],
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
        transform=transform,
        clip_on=False,
    )
    ax.text(
        x + 0.05,
        (y_bottom + y_top) / 2,
        f"{neuron_count} neurons",
        ha="left",
        va="center",
        rotation=90,
        transform=transform,
        clip_on=False,
    )


def draw_order_schematic(
    ax: "Axes",
    trajectory_type: str,
    *,
    arrow_color: str = "red",
    fill_track: bool = True,
) -> "Axes":
    """Draw one compact order schematic centered in a heatmap-height row."""
    ax.axis("off")
    inset = ax.inset_axes([0.0, 0.29, 1.0, 0.42])
    draw_w_track_schematic(
        inset,
        trajectory_name=trajectory_type,
        arrow_color=arrow_color,
        fill_track=fill_track,
    )
    return inset


def draw_image_asset(ax: "Axes", image: np.ndarray, *, aspect: str = "equal") -> None:
    """Draw one external raster asset in an axis."""
    ax.imshow(image, aspect=aspect)
    ax.axis("off")


def draw_panel_a_assets(
    ax: "Axes",
    *,
    asset_dir: Path,
    probe_asset_name: str = DEFAULT_PROBE_ASSET_NAME,
    histology_asset_name: str = DEFAULT_HISTOLOGY_ASSET_NAME,
    behavior_asset_name: str = DEFAULT_BEHAVIOR_ASSET_NAME,
) -> None:
    """Draw panel A external assets: probe, behavior, and histology."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    probe_path = get_figure_1_asset_path(asset_dir, probe_asset_name)
    histology_path = get_figure_1_asset_path(asset_dir, histology_asset_name)
    behavior_path = get_figure_1_asset_path(asset_dir, behavior_asset_name)
    probe_image = np.rot90(load_panel_asset_image(probe_path))
    histology_image = load_panel_asset_image(histology_path)
    behavior_image = load_panel_asset_image(behavior_path)

    probe_ax = ax.inset_axes([0.12, 0.508, 0.24, 0.624])
    behavior_ax = ax.inset_axes([0.00, -0.18, 0.48, 0.76])
    histology_ax = ax.inset_axes([0.47, -0.18, 0.86, 1.36])
    draw_image_asset(probe_ax, probe_image)
    draw_image_asset(behavior_ax, behavior_image)
    draw_image_asset(histology_ax, histology_image)


def draw_panel_a_anatomy_assets(
    ax: "Axes",
    *,
    asset_dir: Path,
    probe_asset_name: str = DEFAULT_PROBE_ASSET_NAME,
    histology_asset_name: str = DEFAULT_HISTOLOGY_ASSET_NAME,
) -> None:
    """Draw the Figure 1A probe and histology assets without behavior."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    probe_path = get_figure_1_asset_path(asset_dir, probe_asset_name)
    histology_path = get_figure_1_asset_path(asset_dir, histology_asset_name)
    probe_image = np.rot90(load_panel_asset_image(probe_path))
    histology_image = load_panel_asset_image(histology_path)

    probe_ax = ax.inset_axes([0.02, 0.21, 0.23, 0.60])
    histology_ax = ax.inset_axes([0.30, -0.08, 0.72, 1.16])
    draw_image_asset(probe_ax, probe_image)
    draw_image_asset(histology_ax, histology_image)


def draw_behavior_task_design_panel(
    ax: "Axes",
    *,
    asset_dir: Path,
    behavior_asset_name: str = DEFAULT_BEHAVIOR_ASSET_NAME,
    rotate_behavior_180: bool = False,
) -> None:
    """Draw Figure 1A task-design schematics."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    del asset_dir, behavior_asset_name, rotate_behavior_180

    from v1ca1.paper_figures import figure_summary

    trajectory_ax = ax.inset_axes(TASK_DESIGN_TOP_LEFT_BOUNDS)
    arena_ax = ax.inset_axes(TASK_DESIGN_TOP_RIGHT_BOUNDS)
    condition_ax = ax.inset_axes(TASK_DESIGN_BOTTOM_BOUNDS)
    ax.plot(
        [0.02, 0.98],
        [TASK_DESIGN_DIVIDER_Y, TASK_DESIGN_DIVIDER_Y],
        color="0.75",
        linewidth=0.6,
        transform=ax.transAxes,
        clip_on=False,
        zorder=20,
    )
    ax.text(
        0.5,
        TASK_DESIGN_DIVIDER_Y,
        TASK_DESIGN_DIVIDER_HEADING,
        ha="center",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        transform=ax.transAxes,
        zorder=21,
    )
    draw_w_track_cycle_panel(trajectory_ax, include_visual_stimuli=False)
    draw_visual_stimuli_schematic(
        arena_ax,
        show_condition_labels=True,
        show_monitor_legend=True,
    )
    draw_task_design_condition_timeline_block(
        condition_ax,
        figure_summary=figure_summary,
    )


def draw_task_design_condition_timeline_block(
    ax: "Axes",
    *,
    figure_summary: Any,
) -> None:
    """Draw the three Figure 1A visual conditions over a time arrow."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    color_a = figure_summary.SUMMARY_TASK_RIGHT_ARM_COLOR
    color_b = figure_summary.SUMMARY_TASK_LEFT_ARM_COLOR
    condition_specs = (
        {"left_arm_color": color_a, "right_arm_color": color_b},
        {"left_arm_color": color_b, "right_arm_color": color_a},
        {"fill_track": True},
    )
    condition_centers = tuple(
        bounds[0] + bounds[2] / 2.0
        for bounds in TASK_DESIGN_CONDITION_TRACK_BOUNDS
    )
    for bounds, condition_spec in zip(
        TASK_DESIGN_CONDITION_TRACK_BOUNDS,
        condition_specs,
        strict=True,
    ):
        condition_ax = ax.inset_axes(bounds)
        figure_summary._draw_summary_condition_track(
            condition_ax,
            arm_side_outlines=True,
            **condition_spec,
        )
    neutral_text_color = "0.20"
    phase_label_fragments = {
        "Initial A/B": (
            ("Initial ", neutral_text_color),
            ("A", color_a),
            ("/", neutral_text_color),
            ("B", color_b),
        ),
        "Cue swap (B/A)": (
            ("Cue swap (", neutral_text_color),
            ("B", color_b),
            ("/", neutral_text_color),
            ("A", color_a),
            (")", neutral_text_color),
        ),
        "Dark": (("Dark", neutral_text_color),),
    }
    for center_x, phase_label in zip(
        condition_centers,
        TASK_DESIGN_PHASE_LABELS,
        strict=True,
    ):
        _add_colored_task_design_phase_label(
            ax,
            x=center_x,
            y=TASK_DESIGN_PHASE_LABEL_Y,
            fragments=phase_label_fragments[phase_label],
        )
    figure_summary._draw_summary_run_sleep_timeline_markers(
        ax,
        arrow_y=TASK_DESIGN_VISUAL_TIMELINE_ARROW_Y,
    )
    ax.plot(
        condition_centers,
        [TASK_DESIGN_VISUAL_TIMELINE_ARROW_Y] * len(condition_centers),
        linestyle="none",
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgecolor="0.25",
        markeredgewidth=0.65,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )


def _add_colored_task_design_phase_label(
    ax: "Axes",
    *,
    x: float,
    y: float,
    fragments: Sequence[tuple[str, str]],
) -> None:
    """Draw one centered task-design label with per-fragment colors."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    text_box = HPacker(
        children=[
            TextArea(
                text,
                textprops={
                    "fontsize": MIN_FIGURE_1_FONTSIZE_PT,
                    "fontweight": "bold",
                    "color": color,
                },
            )
            for text, color in fragments
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    ax.add_artist(
        AnnotationBbox(
            text_box,
            (x, y),
            xycoords=ax.transAxes,
            box_alignment=(0.5, 0.0),
            frameon=False,
            pad=0,
            zorder=5,
        )
    )


def _scale_points_to_axes(
    points: Sequence[tuple[float, float]],
    *,
    bounds: tuple[float, float, float, float],
    source_xlim: tuple[float, float],
    source_ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    """Scale points from one source coordinate system into axes coordinates."""
    x0, y0, width, height = bounds
    xmin, xmax = source_xlim
    ymin, ymax = source_ylim
    return [
        (
            x0 + (float(x) - xmin) / (xmax - xmin) * width,
            y0 + (float(y) - ymin) / (ymax - ymin) * height,
        )
        for x, y in points
    ]


def _scale_rect_to_axes(
    rect: tuple[float, float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    source_xlim: tuple[float, float],
    source_ylim: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Scale one rectangle from source coordinates into axes coordinates."""
    (x, y, width, height) = rect
    (x0, y0), (x1, y1) = _scale_points_to_axes(
        [(x, y), (x + width, y + height)],
        bounds=bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )
    return x0, y0, x1 - x0, y1 - y0


def _get_axes_display_aspect(ax: "Axes") -> float:
    """Return physical width/height for one axes."""
    fig_width, fig_height = ax.figure.get_size_inches()
    box = ax.get_position()
    height = box.height * fig_height
    if height <= 0.0:
        return 1.0
    return (box.width * fig_width) / height


def draw_visual_stimuli_schematic(
    ax: "Axes",
    *,
    show_condition_labels: bool = False,
    show_monitor_legend: bool = False,
) -> None:
    """Draw a compact W-track, monitor layout, walls, and visual stimuli."""
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.axis("off")

    transform = ax.transAxes
    outline, points, dims = get_w_track_geometry()
    source_xlim = (dims["x0"] - 0.35, dims["x5"] + 0.35)
    source_ylim = (dims["y0"] - 0.55, dims["y2"] + 0.45)
    axes_aspect = _get_axes_display_aspect(ax)
    track_height = 0.50
    track_width = track_height * (
        (source_xlim[1] - source_xlim[0])
        / (source_ylim[1] - source_ylim[0])
        / axes_aspect
    )
    track_bounds = ((1.0 - track_width) / 2.0, 0.43, track_width, track_height)
    scaled_outline = _scale_points_to_axes(
        outline,
        bounds=track_bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )

    monitor_color = SCHEMATIC_COLORS["visual_stimulus"]
    monitor_edge_color = "0.25"
    monitor_bar_w = 0.26
    monitor_y = dims["y1"] + 0.12
    monitor_h = dims["y2"] - dims["y1"] - 0.32
    monitor_source_rects = (
        (dims["x0"] - 0.29, monitor_y, monitor_bar_w, monitor_h),
        (dims["x1"] + 0.05, monitor_y, monitor_bar_w, monitor_h),
        (dims["x4"] - 0.31, monitor_y, monitor_bar_w, monitor_h),
        (dims["x5"] + 0.05, monitor_y, monitor_bar_w, monitor_h),
        (
            dims["x0"],
            dims["y0"] - 0.47,
            dims["x5"] - dims["x0"],
            monitor_bar_w,
        ),
    )
    for source_rect in monitor_source_rects:
        rect = _scale_rect_to_axes(
            source_rect,
            bounds=track_bounds,
            source_xlim=source_xlim,
            source_ylim=source_ylim,
        )
        ax.add_patch(
            Rectangle(
                rect[:2],
                rect[2],
                rect[3],
                facecolor=monitor_color,
                edgecolor=monitor_edge_color,
                linewidth=0.45,
                transform=transform,
                zorder=1,
            )
        )
    wall_top_y = dims["y2"] + 0.20
    wall_bottom_y = dims["y1"] + 0.06
    wall_color = "0.45"
    wall_linewidth = 1.0
    horizontal_wall = _scale_points_to_axes(
        [
            (dims["x0"] - 0.18, wall_top_y),
            (dims["x5"] + 0.18, wall_top_y),
        ],
        bounds=track_bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )
    ax.plot(
        [horizontal_wall[0][0], horizontal_wall[1][0]],
        [horizontal_wall[0][1], horizontal_wall[1][1]],
        color=wall_color,
        linewidth=wall_linewidth,
        transform=transform,
        clip_on=False,
        zorder=4,
    )
    wall_x_positions = (
        (dims["x1"] + dims["x2"]) / 2,
        (dims["x3"] + dims["x4"]) / 2,
    )
    for wall_x in wall_x_positions:
        (x, y0), (_x, y1) = _scale_points_to_axes(
            [(wall_x, wall_bottom_y), (wall_x, wall_top_y)],
            bounds=track_bounds,
            source_xlim=source_xlim,
            source_ylim=source_ylim,
        )
        ax.plot(
            [x, x],
            [y0, y1],
            color=wall_color,
            linewidth=wall_linewidth,
            transform=transform,
            clip_on=False,
            zorder=4,
        )
    ax.add_patch(
        Polygon(
            scaled_outline,
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=1.15,
            transform=transform,
            zorder=2,
        )
    )
    label_points = _scale_points_to_axes(
        [
            (points["left"][0], wall_top_y + 0.10),
            (points["center"][0], wall_top_y + 0.10),
            (points["right"][0], wall_top_y + 0.10),
        ],
        bounds=track_bounds,
        source_xlim=source_xlim,
        source_ylim=source_ylim,
    )
    for (x, y), label in zip(label_points, ("L", "C", "R"), strict=True):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=6.3,
            fontweight="bold",
            color="black",
            transform=transform,
            zorder=4,
        )

    screen_y = 0.025
    screen_h = 0.175
    screen_w = screen_h / axes_aspect * 1.10
    screen_gap = 0.045
    ellipsis_gap = 0.045
    ellipsis_step = 0.028
    stimulus_row_width = (
        3 * screen_w
        + 2 * screen_gap
        + ellipsis_gap
        + 2 * ellipsis_step
    )
    screen_start = (1.0 - stimulus_row_width) / 2.0
    screen_specs = (
        (screen_start, "grating"),
        (screen_start + screen_w + screen_gap, "dots"),
        (screen_start + 2 * (screen_w + screen_gap), "black"),
    )
    stimulus_center = (screen_specs[0][0] + screen_specs[-1][0] + screen_w) / 2

    def add_display_circle(
        x: float,
        y: float,
        radius: float,
        color: str,
        *,
        zorder: int = 3,
    ) -> None:
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2 * radius / axes_aspect,
                height=2 * radius,
                facecolor=color,
                edgecolor="none",
                transform=transform,
                zorder=zorder,
            )
        )

    if show_monitor_legend:
        monitor_legend_y = 0.375
        monitor_legend_x = 0.285
        monitor_legend_width = 0.075
        monitor_legend_height = 0.025
        ax.add_patch(
            Rectangle(
                (
                    monitor_legend_x,
                    monitor_legend_y - monitor_legend_height / 2.0,
                ),
                monitor_legend_width,
                monitor_legend_height,
                facecolor=monitor_color,
                edgecolor=monitor_edge_color,
                linewidth=0.45,
                transform=transform,
                zorder=4,
            )
        )
        ax.text(
            monitor_legend_x + monitor_legend_width + 0.025,
            monitor_legend_y,
            "Monitor locations",
            ha="left",
            va="center",
            fontsize=5.8,
            color="black",
            transform=transform,
            zorder=4,
        )

    if show_condition_labels:
        label_y = screen_y + screen_h + 0.025
        label_specs = (
            (0, "A", PANEL_B_VISUAL_ICON_COLORS["A"]),
            (1, "B", PANEL_B_VISUAL_ICON_COLORS["B"]),
            (2, "Dark", "black"),
        )
        for screen_index, label, color in label_specs:
            ax.text(
                screen_specs[screen_index][0] + screen_w / 2.0,
                label_y,
                label,
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                color=color,
                transform=transform,
                zorder=4,
            )
    else:
        ax.text(
            stimulus_center,
            0.325,
            "Visual stimuli",
            ha="center",
            va="bottom",
            fontsize=5.6,
            transform=transform,
            zorder=4,
        )

    for x0, screen_type in screen_specs:
        if screen_type == "black":
            facecolor = "black"
        elif screen_type == "dots":
            facecolor = "0.65"
        else:
            facecolor = "white"
        edgecolor = "black"
        linewidth = 0.8
        if show_condition_labels and screen_type == "grating":
            edgecolor = PANEL_B_VISUAL_ICON_COLORS["A"]
            linewidth = 1.25
        elif show_condition_labels and screen_type == "dots":
            edgecolor = PANEL_B_VISUAL_ICON_COLORS["B"]
            linewidth = 1.25
        ax.add_patch(
            Rectangle(
                (x0, screen_y),
                screen_w,
                screen_h,
                facecolor=facecolor,
                edgecolor="none",
                linewidth=0.0,
                transform=transform,
                zorder=2,
            )
        )
        if screen_type == "grating":
            stripe_w = screen_w / 8.0
            for stripe_index in range(0, 8, 2):
                ax.add_patch(
                    Rectangle(
                        (x0 + stripe_index * stripe_w, screen_y),
                        stripe_w,
                        screen_h,
                        facecolor="black",
                        edgecolor="none",
                        transform=transform,
                        zorder=3,
                    )
                )
        elif screen_type == "dots":
            dot_specs = (
                (0.187, 0.696, 0.168, "white"),
                (0.432, 0.320, 0.104, "white"),
                (0.665, 0.688, 0.080, "black"),
                (0.852, 0.624, 0.120, "white"),
                (0.148, 0.192, 0.112, "black"),
                (0.652, 0.184, 0.104, "white"),
                (0.813, 0.240, 0.160, "black"),
                (0.587, 0.432, 0.064, "black"),
            )
            for x_frac, y_frac, radius_frac, color in dot_specs:
                add_display_circle(
                    x0 + x_frac * screen_w,
                    screen_y + y_frac * screen_h,
                    radius_frac * screen_h,
                    color,
                )
        ax.add_patch(
            Rectangle(
                (x0, screen_y),
                screen_w,
                screen_h,
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=linewidth,
                transform=transform,
                zorder=4,
            )
        )

    ellipsis_start = screen_specs[-1][0] + screen_w + ellipsis_gap
    for x in (
        ellipsis_start,
        ellipsis_start + ellipsis_step,
        ellipsis_start + 2 * ellipsis_step,
    ):
        add_display_circle(x, screen_y + 0.52 * screen_h, 0.007, "black")


def _task_design_progression_color(label: str) -> str:
    """Return the display color for one task-design progression token."""
    if label in {"AB", "BA"}:
        return "white"
    if label == "A":
        return VISUAL_CONDITION_COLORS["02_r1"]
    if label == "B":
        return VISUAL_CONDITION_COLORS["06_r3"]
    if label == "gray":
        return "0.70"
    if label == "dark":
        return "black"
    if label == "sleep":
        return "white"
    raise ValueError(f"Unknown task-design progression token {label!r}.")


def draw_task_design_progression_bar(ax: "Axes") -> None:
    """Draw the run/sleep epoch progression schematic."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.axis("off")

    transform = ax.transAxes
    left = 0.035
    rectangle_right = 0.875
    arrow_right = 0.955
    y0 = 0.36
    height = 0.24
    gap = 0.008
    progression_epochs = []
    for index, run_epoch in enumerate(TASK_DESIGN_PROGRESSION_SEGMENTS):
        progression_epochs.append(("run", *run_epoch))
        if index < len(TASK_DESIGN_PROGRESSION_SEGMENTS) - 1:
            progression_epochs.append(
                (
                    "sleep",
                    TASK_DESIGN_PROGRESSION_SLEEP_LABEL,
                    (TASK_DESIGN_PROGRESSION_SLEEP_LABEL,),
                )
            )
    epoch_width = (
        rectangle_right - left - gap * (len(progression_epochs) - 1)
    ) / len(progression_epochs)

    def add_epoch_rectangle(
        x0: float,
        width: float,
        token: str,
        *,
        edgecolor: str = "black",
        linewidth: float = TASK_DESIGN_SLEEP_EPOCH_LINEWIDTH,
    ) -> None:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                facecolor=_task_design_progression_color(token),
                edgecolor=edgecolor,
                linewidth=linewidth,
                transform=transform,
            )
        )

    def add_centered_label(
        x: float,
        y: float,
        label: str,
        *,
        fontsize: float,
        color: str = "black",
        va: str = "center",
    ) -> None:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va=va,
            fontsize=fontsize,
            color=color,
            transform=transform,
        )

    def add_colored_epoch_letters(
        x: float,
        y: float,
        letters: Sequence[str],
        *,
        fontsize: float,
    ) -> None:
        alignment = ("right", "left")
        for letter, ha in zip(letters, alignment, strict=True):
            ax.text(
                x,
                y,
                letter,
                ha=ha,
                va="center",
                fontsize=fontsize,
                color=_task_design_progression_color(letter),
                transform=transform,
            )

    ax.text(
        0.5,
        0.82,
        TASK_DESIGN_PROGRESSION_DURATION_LABEL,
        ha="center",
        va="center",
        fontsize=5.8,
        color="black",
        transform=transform,
    )
    ax.add_patch(
        FancyArrowPatch(
            (left, 0.20),
            (arrow_right, 0.20),
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.6,
            color="0.25",
            transform=transform,
            clip_on=False,
        )
    )
    for index, (epoch_type, epoch_label, tokens) in enumerate(progression_epochs):
        x0 = left + index * (epoch_width + gap)
        if epoch_type == "sleep":
            add_epoch_rectangle(
                x0,
                epoch_width,
                TASK_DESIGN_PROGRESSION_SLEEP_LABEL,
            )
            add_centered_label(
                x0 + epoch_width / 2.0,
                y0 + height / 2.0,
                TASK_DESIGN_PROGRESSION_SLEEP_LABEL,
                fontsize=3.8,
                color="0.25",
            )
            continue

        add_epoch_rectangle(
            x0,
            epoch_width,
            epoch_label,
            linewidth=TASK_DESIGN_RUN_EPOCH_LINEWIDTH,
        )
        if all(token in {"A", "B"} for token in tokens):
            add_colored_epoch_letters(
                x0 + epoch_width / 2.0,
                y0 + height / 2.0,
                tokens,
                fontsize=5.4,
            )
        else:
            add_centered_label(
                x0 + epoch_width / 2.0,
                y0 + height / 2.0,
                epoch_label,
                fontsize=4.4,
                color="white" if epoch_label == "dark" else "black",
            )
    ax.text(
        rectangle_right + 0.048,
        y0 + height / 2.0,
        "...",
        ha="center",
        va="center",
        fontsize=8.0,
        color="black",
        transform=transform,
    )


def draw_w_track_cycle_panel(
    ax: "Axes",
    *,
    include_visual_stimuli: bool = True,
) -> None:
    """Draw the four-trajectory W-track task cycle schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
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
    if include_visual_stimuli:
        draw_visual_stimuli_schematic(ax)


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return weights that normalize one histogram to a fraction of units."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _format_delta_advantage_summary(
    values: np.ndarray,
    *,
    label: str | None = None,
    include_median: bool = True,
) -> str:
    """Return compact DPP-side summary text for delta log-likelihood values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    prefix = "" if label is None else f"{label}\n"
    if values.size == 0:
        if include_median:
            return f"{prefix}n/a >0, med. n/a"
        return f"{prefix}n/a >0"
    fraction_positive = float(np.mean(values > 0.0))
    if not include_median:
        return f"{prefix}{fraction_positive:.0%} >0"
    median = float(np.median(values))
    return f"{prefix}{fraction_positive:.0%} >0, med. {median:.2f}"


def _format_cell_animal_count(
    table: Any,
    *,
    value_column: str | None = None,
) -> str:
    """Return a compact unique cell and animal count label for panel annotations."""
    columns = set(getattr(table, "columns", []))
    count_table = table
    if value_column is not None and value_column in columns:
        count_table = table[np.isfinite(np.asarray(table[value_column], dtype=float))]

    if {"animal_name", "date", "unit"}.issubset(columns):
        cell_columns = ["animal_name", "date", "unit"]
        n_cells = int(count_table.loc[:, cell_columns].drop_duplicates().shape[0])
    elif "unit" in columns:
        n_cells = int(count_table["unit"].nunique())
    else:
        n_cells = int(len(count_table))

    n_animals = (
        int(count_table["animal_name"].nunique()) if "animal_name" in columns else 0
    )
    cell_word = "cell" if n_cells == 1 else "cells"
    animal_word = "animal" if n_animals == 1 else "animals"
    return f"n = {n_cells} {cell_word}\n{n_animals} {animal_word}"


def build_zero_including_histogram_bins(
    values: np.ndarray,
    *,
    n_bins: int | None = None,
) -> np.ndarray:
    """Return histogram bin edges with zero included as a bin edge."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-0.1, 0.1, 17)
    if n_bins is not None:
        span = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.1)
        bin_width = (2.0 * span) / max(int(n_bins), 1)
        n_left = int(np.ceil(abs(float(np.nanmin(values))) / bin_width))
        n_right = int(np.ceil(abs(float(np.nanmax(values))) / bin_width))
        bin_edges = bin_width * np.arange(-n_left, n_right + 1, dtype=float)
        if bin_edges.size < 2:
            return np.asarray([-bin_width, 0.0, bin_width], dtype=float)
        return bin_edges
    if np.allclose(values, values[0]):
        half_width = max(0.1, abs(float(values[0])) * 0.1 + 0.1)
        bin_edges = np.linspace(
            float(values[0]) - half_width,
            float(values[0]) + half_width,
            17,
        )
    else:
        bin_edges = np.histogram_bin_edges(values, bins="auto")

    if not np.any(np.isclose(bin_edges, 0.0)):
        bin_edges = np.sort(np.unique(np.concatenate([bin_edges, np.array([0.0])])))
    return np.asarray(bin_edges, dtype=float)


def plot_stability_panel(
    ax: "Axes",
    stability_table: Any,
    *,
    regions: Sequence[str] = STABILITY_REGIONS,
) -> None:
    """Plot pooled odd/even stability histograms with trajectory schematics."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    bins = np.linspace(-1.0, 1.0, 25)
    cell_width = 0.46
    cell_height = 0.43
    x_positions = (0.02, 0.52)
    y_positions = (0.53, 0.04)
    legend_handles = []
    legend_labels = []

    for row, trajectory_row in enumerate(STABILITY_TRAJECTORY_LAYOUT):
        for col, trajectory_type in enumerate(trajectory_row):
            x0 = x_positions[col]
            y0 = y_positions[row]
            schematic_ax = ax.inset_axes(
                [x0 + 0.16, y0 + cell_height * 0.67, 0.14, 0.17]
            )
            draw_w_track_schematic(
                schematic_ax,
                trajectory_name=trajectory_type,
                arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
                track_linewidth=0.5,
                trajectory_linewidth=0.75,
                arrow_mutation_scale=6.5,
                fill_track=True,
            )

            hist_ax = ax.inset_axes([x0, y0, cell_width, cell_height * 0.62])
            trajectory_rows = stability_table[
                stability_table["trajectory_type"].astype(str) == trajectory_type
            ]
            for region in regions:
                values = np.asarray(
                    trajectory_rows.loc[
                        trajectory_rows["region"].astype(str) == region,
                        "stability_correlation",
                    ],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                counts, _edges, patches = hist_ax.hist(
                    values,
                    bins=bins,
                    weights=_fraction_histogram_weights(values),
                    color=STABILITY_REGION_COLORS.get(region),
                    **HISTOGRAM_KWARGS,
                )
                del counts
                if row == 0 and col == 0 and len(patches) > 0:
                    legend_handles.append(patches[0])
                    legend_labels.append(region.upper())

            hist_ax.set_xlim(-1.0, 1.0)
            hist_ax.set_ylim(bottom=0.0)
            hist_ax.spines["top"].set_visible(False)
            hist_ax.spines["right"].set_visible(False)
            hist_ax.tick_params(
                labelsize=STABILITY_TICK_LABEL_FONTSIZE,
                length=2,
                pad=1,
            )
            if row == len(STABILITY_TRAJECTORY_LAYOUT) - 1:
                hist_ax.set_xlabel(
                    "Odd/even corr.",
                    fontsize=STABILITY_AXIS_LABEL_FONTSIZE,
                    labelpad=1,
                )
            else:
                hist_ax.set_xticklabels([])
            if col == 0:
                hist_ax.set_ylabel(
                    "Frac.",
                    fontsize=STABILITY_AXIS_LABEL_FONTSIZE,
                    labelpad=1,
                )
            else:
                hist_ax.set_yticklabels([])

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=STABILITY_LEGEND_FONTSIZE,
            handlelength=1.0,
            borderpad=0.1,
            labelspacing=0.2,
        )


def plot_motor_delta_panel(ax: "Axes", motor_delta_table: Any) -> None:
    """Plot pooled V1 motor+DPP versus motor delta log-likelihood values."""
    x_limits = (-1.0, 1.0)
    bin_edges = np.round(np.arange(x_limits[0], x_limits[1] + 0.05, 0.1), 10)
    values = np.asarray(
        motor_delta_table["delta_log_likelihood_bits_per_spike"],
        dtype=float,
    )
    values = values[np.isfinite(values)]

    ax.axvspan(
        x_limits[0],
        0.0,
        color=NEUTRAL_COLORS["dark_epoch_background"],
        alpha=0.65,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    if values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=REGION_COLORS.get(MOTOR_DELTA_REGION, REGION_COLORS["v1"]),
            **EMPHASIS_HISTOGRAM_KWARGS,
            zorder=2,
        )

    ax.text(
        0.03,
        0.97,
        "Motor only better",
        ha="left",
        va="top",
        fontsize=5.5,
        transform=ax.transAxes,
    )
    ax.text(
        PANEL_E_RIGHT_ANNOTATION_X,
        0.97,
        "Motor+DPP better",
        ha="left",
        va="top",
        fontsize=5.5,
        transform=ax.transAxes,
    )
    ax.text(
        PANEL_E_RIGHT_SUMMARY_POSITION[0],
        PANEL_E_RIGHT_SUMMARY_POSITION[1],
        _format_delta_advantage_summary(values, include_median=False),
        ha="right",
        va="bottom",
        fontsize=4.8,
        color=REGION_COLORS.get(MOTOR_DELTA_REGION, REGION_COLORS["v1"]),
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.06,
        _format_cell_animal_count(
            motor_delta_table,
            value_column="delta_log_likelihood_bits_per_spike",
        ),
        ha="left",
        va="bottom",
        fontsize=4.8,
        color="0.25",
        transform=ax.transAxes,
    )
    ax.set_xlim(*x_limits)
    ax.set_xlabel(DELTA_LOG_LIKELIHOOD_AXIS_LABEL, fontsize=7, labelpad=2)
    ax.set_ylabel("Fraction", fontsize=7, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, length=2, pad=1)


def plot_encoding_delta_panel(ax: "Axes", encoding_delta_table: Any) -> None:
    """Plot pooled V1 DPP-versus-absolute-model delta log-likelihood values."""
    x_limits = (-1.0, 1.0)
    bin_edges = np.round(np.arange(x_limits[0], x_limits[1] + 0.05, 0.1), 10)
    all_values = np.asarray(
        encoding_delta_table["delta_log_likelihood_bits_per_spike"],
        dtype=float,
    )
    all_values = all_values[np.isfinite(all_values)]

    ax.axvspan(
        x_limits[0],
        0.0,
        color=NEUTRAL_COLORS["dark_epoch_background"],
        alpha=0.65,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=1)
    summary_rows = []
    if all_values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        for comparison, label, _source_column in ENCODING_DPP_COMPARISONS:
            values = np.asarray(
                encoding_delta_table.loc[
                    encoding_delta_table["comparison"].astype(str) == comparison,
                    "delta_log_likelihood_bits_per_spike",
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            color = ENCODING_DPP_COMPARISON_COLORS.get(comparison, "0.4")
            ax.hist(
                values,
                bins=bin_edges,
                weights=_fraction_histogram_weights(values),
                color=color,
                label=label,
                **COMPACT_HISTOGRAM_KWARGS,
                zorder=2,
            )
            summary_rows.append(
                (
                    comparison,
                    f"{np.mean(values > 0.0):.0%} >0",
                    color,
                )
            )

    ax.text(
        0.03,
        0.97,
        "Abs place better",
        ha="left",
        va="top",
        fontsize=4.8,
        color=ENCODING_DPP_COMPARISON_COLORS["dpp_vs_absolute_place"],
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.82,
        "Distance-to-reward\nbetter",
        ha="left",
        va="top",
        fontsize=4.8,
        color=ENCODING_DPP_COMPARISON_COLORS["dpp_vs_absolute_task_progression"],
        transform=ax.transAxes,
    )
    ax.text(
        0.67,
        0.97,
        "DPP better",
        ha="left",
        va="top",
        fontsize=4.8,
        color="black",
        transform=ax.transAxes,
    )
    for row_index, (_comparison, summary, color) in enumerate(summary_rows):
        ax.text(
            PANEL_F_RIGHT_SUMMARY_POSITION[0],
            PANEL_F_RIGHT_SUMMARY_POSITION[1]
            + PANEL_F_RIGHT_SUMMARY_Y_STEP * (len(summary_rows) - row_index - 1),
            summary,
            ha="right",
            va="bottom",
            fontsize=4.2,
            color=color,
            transform=ax.transAxes,
        )
    ax.text(
        0.03,
        0.06,
        _format_cell_animal_count(
            encoding_delta_table,
            value_column="delta_log_likelihood_bits_per_spike",
        ),
        ha="left",
        va="bottom",
        fontsize=4.8,
        color="0.25",
        transform=ax.transAxes,
    )
    ax.set_xlim(*x_limits)
    ax.set_xlabel(DELTA_LOG_LIKELIHOOD_AXIS_LABEL, fontsize=7, labelpad=2)
    ax.set_ylabel("Fraction", fontsize=7, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, length=2, pad=1)


def add_decoding_significance_brackets(
    ax: "Axes",
    *,
    brackets: Sequence[tuple[float, float, float, str]] = (),
) -> None:
    """Draw data-derived significance brackets above Panel G summaries."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for x_start, x_stop, y, label in brackets:
        if x_start < x_min or x_stop > x_max or y < y_min or y > y_max:
            continue
        y_top = y + DECODING_SIGNIFICANCE_BRACKET_HEIGHT
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
            label,
            ha="center",
            va="bottom",
            fontsize=DECODING_SIGNIFICANCE_LABEL_FONTSIZE,
            color="black",
            clip_on=False,
            zorder=7,
        )


def plot_decoding_error_panel(
    ax: "Axes",
    decoding_error_table: Any,
    *,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = DECODING_CROSS_TRAJECTORY_COMPARISONS,
    significance_brackets: Sequence[tuple[float, float, float, str]] = (),
    region: str = DECODING_COMPARISON_REGION,
    regions: Sequence[str] | None = None,
    region_position_offset: float = DECODING_REGION_POSITION_OFFSET,
    show_region_legend: bool = False,
    show_median_labels: bool = True,
    xtick_label_fontsize: float = DECODING_XTICK_LABEL_FONTSIZE,
) -> None:
    """Plot sample summaries with separately computed lap-level inference."""
    plot_regions = (str(region),) if regions is None else tuple(map(str, regions))
    if not plot_regions or len(set(plot_regions)) != len(plot_regions):
        raise ValueError("regions must contain unique region names.")
    if not np.isfinite(region_position_offset) or region_position_offset < 0:
        raise ValueError("region_position_offset must be finite and non-negative.")
    if len(plot_regions) > 1 and "region" not in decoding_error_table.columns:
        raise ValueError(
            "A multi-region decoding plot requires a 'region' table column."
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    plot_ax = ax.inset_axes([0.0, 0.0, 1.0, 1.0])
    positions = np.arange(1, len(comparisons) + 1, dtype=float)
    labels = [label for _comparison, label, _family, _pairs in comparisons]
    if len(plot_regions) == 1:
        region_offsets = np.zeros(1, dtype=float)
    else:
        region_offsets = np.linspace(
            -region_position_offset,
            region_position_offset,
            len(plot_regions),
        )

    plotted_any = False
    plotted_region_count = 0
    for plot_region, position_offset in zip(
        plot_regions,
        region_offsets,
        strict=True,
    ):
        plot_table = decoding_error_table.copy()
        if "region" in plot_table.columns:
            plot_table = plot_table.loc[
                plot_table["region"].astype(str) == plot_region
            ]
        medians = []
        q25_values = []
        q75_values = []
        plot_positions = []
        base_plot_positions = []
        for position, (comparison, _label, _family, _pairs) in zip(
            positions,
            comparisons,
            strict=True,
        ):
            values = np.asarray(
                plot_table.loc[
                    plot_table["comparison"].astype(str) == comparison,
                    "absolute_error",
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            medians.append(float(np.median(values)))
            q25_values.append(float(np.quantile(values, 0.25)))
            q75_values.append(float(np.quantile(values, 0.75)))
            plot_positions.append(float(position + position_offset))
            base_plot_positions.append(float(position))

        if not medians:
            continue
        plotted_any = True
        plotted_region_count += 1
        color = REGION_COLORS.get(plot_region, REGION_COLORS["v1"])
        plot_ax.vlines(
            plot_positions,
            q25_values,
            q75_values,
            colors=color,
            linewidth=1.0,
            alpha=0.75,
            zorder=3,
        )
        plot_ax.scatter(
            plot_positions,
            medians,
            c=color,
            s=14,
            edgecolors="black",
            linewidths=0.3,
            label=plot_region.upper() if show_region_legend else None,
            zorder=4,
        )
        if show_median_labels:
            for base_position, plot_position, median in zip(
                base_plot_positions,
                plot_positions,
                medians,
                strict=True,
            ):
                is_last_position = np.isclose(
                    base_position,
                    positions[-1],
                )
                median_label_offset = (
                    -DECODING_MEDIAN_LABEL_X_OFFSET
                    if is_last_position
                    else DECODING_MEDIAN_LABEL_X_OFFSET
                )
                plot_ax.text(
                    plot_position + median_label_offset,
                    median,
                    f"med. {median:.2f}",
                    ha="right" if is_last_position else "left",
                    va="center",
                    fontsize=DECODING_MEDIAN_LABEL_FONTSIZE,
                    color="0.20",
                    zorder=5,
                )

    if not plotted_any:
        plot_ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=plot_ax.transAxes,
        )

    plot_ax.set_xticks(positions)
    plot_ax.set_xticklabels(labels, fontsize=xtick_label_fontsize)
    plot_ax.set_xlim(0.5, len(comparisons) + 0.5)
    plot_ax.set_ylim(0.0, 0.72)
    plot_ax.set_ylabel(
        "|Norm. error|",
        fontsize=DECODING_YLABEL_FONTSIZE,
        labelpad=2,
    )
    plot_ax.yaxis.set_label_coords(DECODING_YLABEL_X, 0.5)
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.tick_params(axis="y", labelsize=7, length=2, pad=1)
    plot_ax.tick_params(axis="x", length=0, pad=1)
    if show_region_legend and plotted_region_count:
        plot_ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=DECODING_REGION_LEGEND_FONTSIZE,
            borderpad=0.1,
            handletextpad=0.3,
            labelspacing=0.2,
        )
    if plotted_any and significance_brackets:
        add_decoding_significance_brackets(
            plot_ax,
            brackets=significance_brackets,
        )

    train_center = DECODING_TRAIN_SCHEMATIC_CENTER_X
    ax.text(
        train_center,
        DECODING_TRAIN_LABEL_Y,
        "Train",
        ha="center",
        va="bottom",
        fontsize=5.2,
        transform=ax.transAxes,
    )
    train_ax = ax.inset_axes(
        [
            train_center - DECODING_SCHEMATIC_WIDTH / 2,
            DECODING_SCHEMATIC_Y,
            DECODING_SCHEMATIC_WIDTH,
            DECODING_SCHEMATIC_HEIGHT,
        ]
    )
    draw_w_track_schematic(
        train_ax,
        trajectory_name=DECODING_EXAMPLE_TRAIN_TRAJECTORY,
        arrow_color=PANEL_E_TRAJECTORY_COLORS[DECODING_EXAMPLE_TRAIN_TRAJECTORY],
        track_linewidth=0.45,
        trajectory_linewidth=0.65,
        arrow_mutation_scale=5.8,
        fill_track=True,
    )

    plot_left = 0.0
    plot_width = 1.0
    icon_width = DECODING_SCHEMATIC_WIDTH
    icon_height = DECODING_SCHEMATIC_HEIGHT
    for position, (comparison, _label, _family, _pairs) in zip(
        positions,
        comparisons,
        strict=True,
    ):
        test_trajectory = DECODING_EXAMPLE_TEST_TRAJECTORIES.get(comparison)
        if test_trajectory is None:
            continue
        x_center = plot_left + plot_width * (position - 0.5) / len(comparisons)
        icon_ax = ax.inset_axes(
            [
                x_center - icon_width / 2,
                DECODING_SCHEMATIC_Y,
                icon_width,
                icon_height,
            ]
        )
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=test_trajectory,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[test_trajectory],
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=True,
        )


def plot_position_aligned_raster_axis(
    ax: "Axes",
    trial_positions: Sequence[np.ndarray],
    trajectory_type: str,
    *,
    show_ylabel: bool = False,
    ylabel_x: float | None = None,
) -> None:
    """Plot spikes by normalized task progression across trajectory trials."""
    for trial_index, positions in enumerate(trial_positions, start=1):
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            continue
        ax.plot(
            positions,
            np.full(positions.shape, trial_index, dtype=float),
            "|",
            color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            **RASTER_TICK_KWARGS,
        )

    add_task_progression_segment_boundary_lines(ax)

    n_trials = len(trial_positions)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1, n_trials) + 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks([])
    if show_ylabel:
        ax.set_ylabel("Trials", fontsize=PANEL_E_AXIS_LABEL_FONTSIZE, labelpad=1)
        if ylabel_x is not None:
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0.9, width=0.35, pad=1)


def draw_panel_e_raster_schematic(ax: "Axes", trajectory_type: str) -> None:
    """Draw one compact dark-epoch trajectory icon for a panel E raster."""
    draw_w_track_schematic(
        ax,
        trajectory_name=trajectory_type,
        arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
        track_linewidth=0.45,
        trajectory_linewidth=0.65,
        arrow_mutation_scale=5.8,
        fill_track=True,
    )


def plot_panel_e_rate_axis(
    ax: "Axes",
    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]],
    trajectory_pair: Sequence[str],
    *,
    y_max: float,
    show_ylabel: bool = False,
    show_legend: bool = True,
    show_xlabel: bool = True,
    ylabel_x: float | None = None,
) -> None:
    """Plot occupancy-normalized firing rates for a pair of trajectories."""
    for trajectory_type in trajectory_pair:
        position, rate = firing_rates[trajectory_type]
        ax.plot(
            position,
            rate,
            color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            linewidth=0.8,
            label=PANEL_E_TRAJECTORY_LABELS[trajectory_type],
        )

    add_task_progression_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    if show_xlabel:
        ax.set_xlabel(
            TASK_PROGRESSION_XLABEL,
            fontsize=PANEL_E_AXIS_LABEL_FONTSIZE,
            labelpad=1,
        )
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=PANEL_E_AXIS_LABEL_FONTSIZE, labelpad=1)
        if ylabel_x is not None:
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
    if show_legend:
        ax.legend(frameon=False, fontsize=3.8, handlelength=0.9, borderpad=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        labelsize=PANEL_E_TICK_LABEL_FONTSIZE,
        length=0.9,
        width=0.35,
        pad=1,
    )


def plot_panel_e_example(
    ax: "Axes",
    example: dict[str, Any],
    *,
    title: str | None = None,
    show_ylabel: bool = True,
    show_rate_legends: bool = True,
    show_rate_xlabels: bool = True,
    show_shared_xlabel: bool = False,
) -> None:
    """Plot one panel E example with trajectory rasters and firing-rate curves."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if title is not None:
        ax.text(
            0.50,
            0.995,
            title,
            ha="center",
            va="top",
            fontsize=5.8,
            transform=ax.transAxes,
        )

    raster_positions = example["raster_positions"]
    firing_rates = example["firing_rates"]
    schematic_bounds = {
        "center_to_left": (0.210, 0.880, 0.09, 0.070),
        "center_to_right": (0.710, 0.880, 0.09, 0.070),
        "right_to_center": (0.210, 0.565, 0.09, 0.070),
        "left_to_center": (0.710, 0.565, 0.09, 0.070),
    }
    raster_bounds = {
        "center_to_left": (0.05, 0.65, 0.41, 0.19),
        "center_to_right": (0.55, 0.65, 0.41, 0.19),
        "right_to_center": (0.05, 0.35, 0.41, 0.19),
        "left_to_center": (0.55, 0.35, 0.41, 0.19),
    }
    for row in PANEL_E_RASTER_TRAJECTORY_LAYOUT:
        for trajectory_type in row:
            schematic_ax = ax.inset_axes(schematic_bounds[trajectory_type])
            draw_panel_e_raster_schematic(schematic_ax, trajectory_type)
            raster_ax = ax.inset_axes(raster_bounds[trajectory_type])
            plot_position_aligned_raster_axis(
                raster_ax,
                raster_positions[trajectory_type],
                trajectory_type,
                show_ylabel=show_ylabel
                and trajectory_type in {"center_to_left", "right_to_center"},
                ylabel_x=PANEL_C_EXAMPLE_YLABEL_X,
            )

    finite_rate_maxima = [
        float(np.nanmax(rate))
        for _position, rate in firing_rates.values()
        if np.isfinite(rate).any()
    ]
    y_max = 1.0 if not finite_rate_maxima else max(1.0, np.ceil(max(finite_rate_maxima)))
    rate_bounds = ((0.05, 0.08, 0.41, 0.25), (0.55, 0.08, 0.41, 0.25))
    for pair_index, trajectory_pair in enumerate(PANEL_E_FR_TRAJECTORY_PAIRS):
        rate_ax = ax.inset_axes(rate_bounds[pair_index])
        plot_panel_e_rate_axis(
            rate_ax,
            firing_rates,
            trajectory_pair,
            y_max=y_max,
            show_ylabel=show_ylabel and pair_index == 0,
            show_legend=show_rate_legends,
            show_xlabel=show_rate_xlabels,
            ylabel_x=PANEL_C_EXAMPLE_YLABEL_X,
        )
    if show_shared_xlabel:
        ax.text(
            0.5,
            PANEL_C_EXAMPLE_SHARED_XLABEL_Y,
            TASK_PROGRESSION_XLABEL,
            ha="center",
            va="top",
            fontsize=6.0,
            transform=ax.transAxes,
        )


def plot_panel_e_examples(
    ax: "Axes",
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot all panel E example units stacked in one panel axis."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center", transform=ax.transAxes)
        return

    block_height = PANEL_C_EXAMPLE_BLOCK_HEIGHT
    if len(examples) == len(PANEL_C_EXAMPLE_BLOCK_Y_POSITIONS):
        y_positions = np.asarray(PANEL_C_EXAMPLE_BLOCK_Y_POSITIONS, dtype=float)
    else:
        y_positions = np.linspace(1.0 - block_height, 0.02, len(examples))
    for example_index, (y0, example) in enumerate(
        zip(y_positions, examples, strict=False),
        start=1,
    ):
        example_ax = ax.inset_axes([0.0, float(y0), 1.0, block_height])
        plot_panel_e_example(
            example_ax,
            example,
            title=f"Example cell {example_index}",
            show_rate_xlabels=False,
            show_shared_xlabel=True,
        )


def find_dataset_dark_epoch(
    datasets: Sequence[DatasetId],
    animal_name: str,
    date: str,
) -> str | None:
    """Return the requested dark epoch for one data set, if it is listed."""
    for dataset in datasets:
        dataset_animal, dataset_date, dataset_dark_epoch = normalize_dataset_id(dataset)
        if dataset_animal == animal_name and dataset_date == date:
            return dataset_dark_epoch
    return None


def build_panel_b_visual_epoch_specs(
    animal_name: str,
    date: str,
    *,
    dark_epoch: str | None,
) -> tuple[tuple[str, str, str], ...]:
    """Return panel-B visual epoch keys, labels, and run epoch IDs."""
    light_specs = tuple(
        (epoch, PANEL_B_VISUAL_EPOCH_LABELS[epoch], epoch)
        for epoch in PANEL_B_VISUAL_LIGHT_EPOCHS
    )
    return (
        *light_specs,
        (
            "dark",
            PANEL_B_VISUAL_EPOCH_LABELS["dark"],
            get_dark_epoch(animal_name, date, dark_epoch),
        ),
    )


def load_panel_b_visual_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    unit_id: int,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> dict[str, Any]:
    """Load the Figure 1B example cell rasters and rate curves across epochs."""
    from v1ca1.paper_figures._dark_light import load_or_compute_panel_example_data

    epoch_specs = build_panel_b_visual_epoch_specs(
        animal_name,
        date,
        dark_epoch=dark_epoch,
    )
    epoch_examples = {
        epoch_key: load_or_compute_panel_example_data(
            data_root=data_root,
            panel_name="A",
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region=region,
            unit_id=unit_id,
            trajectories=PANEL_B_VISUAL_TRAJECTORIES,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
        for epoch_key, _epoch_label, epoch in epoch_specs
    }
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "unit_id": unit_id,
        "epoch_order": tuple(epoch_key for epoch_key, _epoch_label, _epoch in epoch_specs),
        "epoch_labels": {
            epoch_key: epoch_label for epoch_key, epoch_label, _epoch in epoch_specs
        },
        "epoch_examples": epoch_examples,
        "trajectories": PANEL_B_VISUAL_TRAJECTORIES,
    }


def _get_panel_b_visual_y_max(example: dict[str, Any]) -> float:
    """Return a shared firing-rate limit for the Figure 1B example."""
    maxima: list[float] = []
    for epoch_payload in example["epoch_examples"].values():
        for _position, rate in epoch_payload["firing_rates"].values():
            rate = np.asarray(rate, dtype=float)
            if np.isfinite(rate).any():
                maxima.append(float(np.nanmax(rate)))
    if not maxima:
        return 1.0
    return max(1.0, float(np.ceil(max(maxima))))


def draw_panel_b_visual_epoch_icon(
    ax: "Axes",
    *,
    left_label: str | None = None,
    right_label: str | None = None,
    fill_track: bool = False,
    label_colors: Mapping[str, str] | None = None,
    region_fill_colors: Mapping[str, str] | None = None,
    region_fill_alpha: float | None = None,
    arm_side_outline_colors: Mapping[str, str] | None = None,
) -> None:
    """Draw one Figure 1B epoch-condition W-track icon."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon, Rectangle

    outline, _points, dims = get_w_track_geometry()
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="black" if fill_track else "none",
            edgecolor="none",
            linewidth=0.0,
            joinstyle="miter",
            zorder=1,
        )
    )
    region_rectangles = {
        "left_arm": (
            dims["x0"],
            dims["y1"],
            dims["x1"] - dims["x0"],
            dims["y2"] - dims["y1"],
        ),
        "right_arm": (
            dims["x4"],
            dims["y1"],
            dims["x5"] - dims["x4"],
            dims["y2"] - dims["y1"],
        ),
    }
    for region_name, color in (region_fill_colors or {}).items():
        x, y, width, height = region_rectangles[region_name]
        facecolor = (
            color
            if region_fill_alpha is None
            else to_rgba(color, alpha=region_fill_alpha)
        )
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=facecolor,
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
            )
        )
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=0.45,
            joinstyle="miter",
            zorder=3,
        )
    )
    if arm_side_outline_colors is not None:
        draw_w_track_arm_side_outlines(
            ax,
            arm_colors=arm_side_outline_colors,
            gap=PANEL_B_VISUAL_ICON_ARM_SIDE_OUTLINE_GAP,
            linewidth=PANEL_B_VISUAL_ICON_ARM_SIDE_OUTLINE_LINEWIDTH,
            label_prefix="_panel_b_visual_icon_arm_side_outline",
            zorder=4,
        )
    if left_label is not None:
        ax.text(
            dims["x0"] - PANEL_B_VISUAL_ICON_LABEL_X_OFFSET,
            dims["y2"] / 2,
            left_label,
            ha="center",
            va="center",
            fontsize=5.2,
            color=(
                label_colors.get(left_label, "black")
                if label_colors is not None
                else "black"
            ),
    )
    if right_label is not None:
        ax.text(
            dims["x5"] + PANEL_B_VISUAL_ICON_LABEL_X_OFFSET,
            dims["y2"] / 2,
            right_label,
            ha="center",
            va="center",
            fontsize=5.2,
            color=(
                label_colors.get(right_label, "black")
                if label_colors is not None
                else "black"
            ),
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.95, dims["x5"] + 0.95)
    ax.set_ylim(-0.25, dims["y2"] + 0.25)
    ax.axis("off")


def plot_panel_b_visual_rate_axis(
    ax: "Axes",
    example: dict[str, Any],
    trajectory_type: str,
    *,
    y_max: float,
    show_ylabel: bool = False,
    show_legend: bool = False,
    show_xlabel: bool = True,
    trajectory_epoch_color_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    """Plot Figure 1B firing-rate curves for one route across epochs."""
    trajectory_epoch_colors = (
        trajectory_epoch_color_overrides.get(trajectory_type, {})
        if trajectory_epoch_color_overrides is not None
        else {}
    )
    for epoch_key in example["epoch_order"]:
        position, rate = example["epoch_examples"][epoch_key]["firing_rates"][
            trajectory_type
        ]
        ax.plot(
            position,
            rate,
            color=trajectory_epoch_colors.get(
                epoch_key,
                VISUAL_CONDITION_COLORS[epoch_key],
            ),
            linewidth=0.85,
            label=example["epoch_labels"][epoch_key],
        )
    add_task_progression_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    if show_xlabel:
        ax.set_xlabel(TASK_PROGRESSION_XLABEL, fontsize=4.8, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.8, labelpad=1)
    if show_legend:
        ax.legend(frameon=False, fontsize=4.2, handlelength=1.1, borderpad=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def plot_panel_b_visual_combined_raster_axis(
    ax: "Axes",
    example: dict[str, Any],
    trajectory_type: str,
    *,
    trajectory_epoch_backgrounds: Mapping[
        str,
        Mapping[str, Sequence[tuple[float, float, str]]],
    ]
    | None = None,
    epoch_background_alpha: float = PANEL_DARK_LIGHT_RASTER_BACKGROUND_ALPHA,
    raster_color: str = PANEL_DARK_LIGHT_RASTER_COLOR,
) -> None:
    """Plot all Figure 1B epoch rasters in one stacked axis."""
    epoch_order = tuple(example["epoch_order"])
    n_epochs = len(epoch_order)
    for epoch_index, epoch_key in enumerate(epoch_order):
        epoch_base = float(n_epochs - epoch_index - 1)
        if epoch_key == "dark":
            ax.axhspan(
                epoch_base,
                epoch_base + 1.0,
                color=NEUTRAL_COLORS["dark_epoch_background"],
                linewidth=0,
                zorder=0,
            )
        if trajectory_epoch_backgrounds is not None:
            for x0, x1, color in trajectory_epoch_backgrounds.get(
                trajectory_type,
                {},
            ).get(epoch_key, ()):
                ax.axvspan(
                    x0,
                    x1,
                    ymin=epoch_base / n_epochs,
                    ymax=(epoch_base + 1.0) / n_epochs,
                    color=color,
                    alpha=epoch_background_alpha,
                    linewidth=0,
                    zorder=0,
                )

        trial_positions = example["epoch_examples"][epoch_key]["raster_positions"][
            trajectory_type
        ]
        n_trials = len(trial_positions)
        for trial_index, positions in enumerate(trial_positions, start=1):
            positions = np.asarray(positions, dtype=float)
            if positions.size == 0:
                continue
            y_position = epoch_base + (trial_index / max(n_trials + 1, 1))
            ax.plot(
                positions,
                np.full(positions.shape, y_position, dtype=float),
                "|",
                color=raster_color,
                **RASTER_TICK_KWARGS,
                zorder=3,
            )

    for separator in range(1, n_epochs):
        ax.axhline(separator, color="0.82", linewidth=0.35, zorder=1)
    add_task_progression_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(max(1, n_epochs)))
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=1.5, pad=1)


def plot_panel_b_visual_example(
    ax: "Axes",
    example: dict[str, Any],
    *,
    visual_label_colors: Mapping[str, str] | None = None,
    trajectory_epoch_color_overrides: Mapping[str, Mapping[str, str]] | None = None,
    trajectory_epoch_backgrounds: Mapping[
        str,
        Mapping[str, Sequence[tuple[float, float, str]]],
    ]
    | None = None,
    epoch_background_alpha: float = PANEL_DARK_LIGHT_RASTER_BACKGROUND_ALPHA,
    raster_color: str = PANEL_DARK_LIGHT_RASTER_COLOR,
) -> None:
    """Plot the Figure 1B visual example rasters and firing-rate curves."""
    trajectories = tuple(str(trajectory) for trajectory in example["trajectories"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not trajectories:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center", transform=ax.transAxes)
        return

    left_margin = 0.13
    right_margin = 0.012
    column_gap = 0.026
    column_width = (
        1.0
        - left_margin
        - right_margin
        - column_gap * (len(trajectories) - 1)
    ) / len(trajectories)
    raster_bottom = 0.34
    raster_height = 0.43
    y_max = _get_panel_b_visual_y_max(example)

    icon_specs = (
        {
            "left_label": "A",
            "right_label": "B",
            "fill_track": False,
            "arm_side_outline_colors": {
                "left_arm": PANEL_B_VISUAL_ICON_COLORS["A"],
                "right_arm": PANEL_B_VISUAL_ICON_COLORS["B"],
            },
        },
        {
            "left_label": "B",
            "right_label": "A",
            "fill_track": False,
            "arm_side_outline_colors": {
                "left_arm": PANEL_B_VISUAL_ICON_COLORS["B"],
                "right_arm": PANEL_B_VISUAL_ICON_COLORS["A"],
            },
        },
        {"left_label": None, "right_label": None, "fill_track": True},
    )
    for row_index, icon_spec in enumerate(icon_specs):
        epoch_center = raster_bottom + raster_height * (
            len(icon_specs) - row_index - 0.5
        ) / len(icon_specs)
        icon_ax = ax.inset_axes([0.026, epoch_center - 0.061, 0.070, 0.122])
        draw_panel_b_visual_epoch_icon(
            icon_ax,
            **icon_spec,
            label_colors=visual_label_colors,
        )

    for trajectory_index, trajectory_type in enumerate(trajectories):
        left = left_margin + trajectory_index * (column_width + column_gap)
        schematic_ax = ax.inset_axes(
            [left + 0.34 * column_width, 0.80, 0.32 * column_width, 0.12]
        )
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )

        raster_ax = ax.inset_axes([left, raster_bottom, column_width, raster_height])
        plot_panel_b_visual_combined_raster_axis(
            raster_ax,
            example,
            trajectory_type,
            trajectory_epoch_backgrounds=trajectory_epoch_backgrounds,
            epoch_background_alpha=epoch_background_alpha,
            raster_color=raster_color,
        )

        rate_ax = ax.inset_axes([left, 0.147, column_width, 0.16])
        plot_panel_b_visual_rate_axis(
            rate_ax,
            example,
            trajectory_type,
            y_max=y_max,
            show_ylabel=trajectory_index == 0,
            show_legend=False,
            show_xlabel=False,
            trajectory_epoch_color_overrides=trajectory_epoch_color_overrides,
        )
    rate_area_left = left_margin
    rate_area_right = left_margin + len(trajectories) * column_width + (
        len(trajectories) - 1
    ) * column_gap
    ax.text(
        (rate_area_left + rate_area_right) / 2.0,
        PANEL_B_VISUAL_SHARED_XLABEL_Y,
        TASK_PROGRESSION_XLABEL,
        ha="center",
        va="top",
        fontsize=6.0,
        transform=ax.transAxes,
    )


def plot_dark_light_example_panel(
    ax: "Axes",
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path | None,
    refresh_panel_example_cache: bool,
) -> None:
    """Draw the Figure 1B visual-condition example neuron."""
    animal_name, date, region, unit_id = PANEL_B_VISUAL_EXAMPLE
    example = load_panel_b_visual_example_data(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        unit_id=unit_id,
        dark_epoch=find_dataset_dark_epoch(datasets, animal_name, date),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=refresh_panel_example_cache,
    )
    plot_panel_b_visual_example(
        ax,
        example,
        visual_label_colors=PANEL_B_VISUAL_ICON_COLORS,
        trajectory_epoch_color_overrides=PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS,
        trajectory_epoch_backgrounds=PANEL_DARK_LIGHT_TRAJECTORY_EPOCH_BACKGROUNDS,
        epoch_background_alpha=PANEL_DARK_LIGHT_RASTER_BACKGROUND_ALPHA,
        raster_color=PANEL_DARK_LIGHT_RASTER_COLOR,
    )


def make_figure_1(
    *,
    data_root: Path,
    asset_dir: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    encoding_bin_size_s: float,
    encoding_place_bin_size_cm: float,
    dpi: int,
    decoding_n_permutations: int = DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = DECODING_PERMUTATION_SEED,
    panel_d_cache_dir: Path | None = None,
    refresh_panel_d_cache: bool = False,
    panel_e_cache_dir: Path | None = None,
    refresh_panel_e_cache: bool = False,
    panel_dark_light_example_cache_dir: Path | None = None,
    refresh_panel_dark_light_example_cache: bool = False,
) -> Path:
    """Build and save Figure 1."""
    import matplotlib.pyplot as plt

    if decoding_n_permutations <= 0:
        raise ValueError("decoding_n_permutations must be positive.")
    if decoding_permutation_seed < 0:
        raise ValueError("decoding_permutation_seed must be non-negative.")
    panel_d_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_d_cache_dir is None
        else Path(panel_d_cache_dir)
    )
    panel_e_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_e_cache_dir is None
        else Path(panel_e_cache_dir)
    )
    panel_dark_light_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_dark_light_example_cache_dir is None
        else Path(panel_dark_light_example_cache_dir)
    )
    apply_paper_style()
    n_region_rows = len(regions) * len(PANEL_D_TRAJECTORY_TYPES)
    heatmap_height_mm = DEFAULT_HEATMAP_HEIGHT_MM * max(len(regions), 1)
    fig_height_mm = (
        DEFAULT_TOP_ROW_HEIGHT_MM
        + heatmap_height_mm
        + DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM
        + DEFAULT_BOTTOM_ROW_HEIGHT_MM
    )
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            DEFAULT_TOP_ROW_HEIGHT_MM,
            heatmap_height_mm,
            DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM,
            DEFAULT_BOTTOM_ROW_HEIGHT_MM,
        ],
    )

    main_grid = outer_grid[:2].subgridspec(
        nrows=2,
        ncols=2,
        wspace=BOTTOM_ROW_PANEL_WSPACE,
        hspace=0.08,
        height_ratios=[
            DEFAULT_TOP_ROW_HEIGHT_MM,
            heatmap_height_mm,
        ],
        width_ratios=[
            DEFAULT_PANEL_E_WIDTH_FRACTION,
            DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
        ],
    )
    panel_b_axis = fig.add_subplot(main_grid[0, 0])
    draw_behavior_task_design_panel(
        panel_b_axis,
        asset_dir=asset_dir,
        behavior_asset_name=DEFAULT_FIGURE_1A_BEHAVIOR_ASSET_NAME,
        rotate_behavior_180=True,
    )
    panel_b_axis.set_title(TASK_DESIGN_TITLE, fontsize=8, pad=2)

    panel_dark_light_axis = fig.add_subplot(main_grid[0, 1])
    plot_dark_light_example_panel(
        panel_dark_light_axis,
        data_root=data_root,
        datasets=datasets,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_example_cache_dir=panel_dark_light_example_cache_dir,
        refresh_panel_example_cache=refresh_panel_dark_light_example_cache,
    )
    panel_dark_light_axis.set_title(
        PANEL_DARK_LIGHT_EXAMPLE_TITLE,
        fontsize=8,
        pad=2,
    )

    heatmap_grid = main_grid[1, 1].subgridspec(
        nrows=n_region_rows + 1,
        ncols=len(PANEL_D_TRAJECTORY_TYPES) + 1,
        height_ratios=[0.42, *([1.0] * n_region_rows)],
        width_ratios=[0.48, *([1.0] * len(PANEL_D_TRAJECTORY_TYPES))],
    )
    panel_d_axis = fig.add_subplot(main_grid[1, 0])
    panel_e_examples = [
        load_or_compute_panel_e_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region=region,
            unit_id=unit_id,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_e_cache_dir=panel_e_cache_dir,
            refresh_panel_e_cache=refresh_panel_e_cache,
        )
        for animal_name, date, epoch, region, unit_id in PANEL_E_EXAMPLES
    ]
    plot_panel_e_examples(panel_d_axis, panel_e_examples)
    panel_d_axis.set_title(PANEL_C_EXAMPLE_TITLE, fontsize=8, pad=2)

    spacer_axis = fig.add_subplot(outer_grid[2])
    spacer_axis.axis("off")

    final_row_grid = outer_grid[3].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            DEFAULT_PANEL_F_WIDTH_FRACTION,
            DEFAULT_PANEL_G_WIDTH_FRACTION,
            DEFAULT_PANEL_H_WIDTH_FRACTION,
        ],
    )
    panel_f_axis = fig.add_subplot(final_row_grid[0, 0])
    panel_g_axis = fig.add_subplot(final_row_grid[0, 1])
    panel_h_axis = fig.add_subplot(final_row_grid[0, 2])
    for final_row_axis in (panel_f_axis, panel_g_axis, panel_h_axis):
        final_row_axis.axis("off")
    panel_f_plot_axis = panel_f_axis.inset_axes(BOTTOM_ROW_PLOT_BOUNDS)
    panel_g_plot_axis = panel_g_axis.inset_axes(BOTTOM_ROW_PLOT_BOUNDS)
    panel_h_plot_axis = panel_h_axis.inset_axes(BOTTOM_ROW_PLOT_BOUNDS)
    motor_delta_table = load_motor_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=MOTOR_DELTA_REGION,
    )
    plot_motor_delta_panel(panel_f_plot_axis, motor_delta_table)
    panel_f_axis.set_title("Comparison to motor", fontsize=8, pad=2)
    encoding_delta_table = load_encoding_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=ENCODING_COMPARISON_REGION,
        bin_size_s=encoding_bin_size_s,
        place_bin_size_cm=encoding_place_bin_size_cm,
    )
    plot_encoding_delta_panel(panel_g_plot_axis, encoding_delta_table)
    panel_g_axis.set_title("Comparison to alternative codes", fontsize=8, pad=2)
    decoding_datasets = filter_datasets_by_animals(
        datasets,
        PANEL_H_DECODING_ANIMALS,
    )
    decoding_dataset_animals = [
        animal_name for animal_name, _date, _epoch in decoding_datasets
    ]
    if (
        len(decoding_datasets) != len(PANEL_H_DECODING_ANIMALS)
        or set(decoding_dataset_animals) != set(PANEL_H_DECODING_ANIMALS)
    ):
        raise ValueError(
            "Figure 1 decoding inference requires exactly one data set for "
            f"each animal {PANEL_H_DECODING_ANIMALS!r}; received "
            f"{decoding_datasets!r}."
        )
    decoding_error_table = load_decoding_absolute_error_table(
        data_root=data_root,
        datasets=decoding_datasets,
        region=DECODING_COMPARISON_REGION,
    )
    decoding_trial_error_table = build_decoding_trial_error_table(
        data_root=data_root,
        datasets=decoding_datasets,
        region=DECODING_COMPARISON_REGION,
    )
    decoding_permutation_results = compute_decoding_permutation_tests(
        decoding_trial_error_table,
        n_permutations=decoding_n_permutations,
        seed=decoding_permutation_seed,
    )
    decoding_significance_brackets = build_decoding_significance_brackets(
        decoding_permutation_results,
    )
    plot_decoding_error_panel(
        panel_h_plot_axis,
        decoding_error_table,
        significance_brackets=decoding_significance_brackets,
    )
    panel_h_axis.set_title(PANEL_G_TITLE, fontsize=8, pad=2)

    axes = np.asarray(
        [
            [
                fig.add_subplot(heatmap_grid[row, col])
                for col in range(len(PANEL_D_TRAJECTORY_TYPES) + 1)
            ]
            for row in range(n_region_rows + 1)
        ],
        dtype=object,
    )

    corner_axis = axes[0, 0]
    corner_axis.axis("off")
    tuning_schematic_axes = axes[0, 1:]
    order_schematic_axes = axes[1:, 0]
    heatmap_axes = axes[1:, 1:]
    for ax, trajectory_type in zip(
        tuning_schematic_axes,
        PANEL_D_TRAJECTORY_TYPES,
        strict=True,
    ):
        draw_w_track_schematic(
            ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            fill_track=True,
        )
    for row_index, ax in enumerate(order_schematic_axes):
        trajectory_type = PANEL_D_TRAJECTORY_TYPES[
            row_index % len(PANEL_D_TRAJECTORY_TYPES)
        ]
        draw_order_schematic(
            ax,
            trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
        )

    color_image = plot_dark_heatmap_regions(
        heatmap_axes,
        data_root=data_root,
        datasets=datasets,
        regions=regions,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=refresh_panel_d_cache,
    )

    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=heatmap_axes.ravel().tolist(),
            shrink=0.24,
            pad=HEATMAP_COLORBAR_PAD,
            aspect=HEATMAP_COLORBAR_ASPECT,
            ticks=[0.0, 1.0],
        )
        colorbar.ax.set_yticklabels(["0", "1"])
        colorbar.ax.tick_params(length=2)
        colorbar.set_label(
            "Norm. FR",
            rotation=90,
            labelpad=HEATMAP_COLORBAR_LABELPAD,
            fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
        )

    draw_neuron_scale_bar(heatmap_axes[-1, -1])

    fig.canvas.draw()
    add_panel_d_heatmap_block_outlines(heatmap_axes)
    fig.set_constrained_layout(False)
    panel_e_position = panel_f_axis.get_position()
    panel_c_position = panel_d_axis.get_position()
    final_row_left = panel_c_position.x0
    final_row_right = panel_dark_light_axis.get_position().x1
    final_row_panel_width = panel_c_position.width
    final_row_gap = (
        final_row_right - final_row_left - 3.0 * final_row_panel_width
    ) / 2.0
    for panel_index, final_row_axis in enumerate(
        (panel_f_axis, panel_g_axis, panel_h_axis)
    ):
        final_row_axis.set_position(
            [
                final_row_left
                + panel_index * (final_row_panel_width + final_row_gap),
                panel_e_position.y0,
                final_row_panel_width,
                panel_e_position.height,
            ]
        )
    panel_d_axis.set_title("")
    add_aligned_panel_headers(
        fig,
        (panel_b_axis, panel_dark_light_axis),
        labels=("A", "B"),
        titles=(TASK_DESIGN_TITLE, PANEL_DARK_LIGHT_EXAMPLE_TITLE),
        label_x_offsets=TOP_ROW_PANEL_LABEL_X_OFFSETS,
        title_fontsizes=TOP_ROW_PANEL_TITLE_FONTSIZES,
    )
    add_centered_axis_text(
        fig,
        tuning_schematic_axes,
        "Tuning",
        y_offset=HEATMAP_TUNING_LABEL_OFFSET,
    )
    add_centered_axis_text(
        fig,
        order_schematic_axes,
        "Order",
        y_offset=HEATMAP_ORDER_LABEL_OFFSET,
        rotation=90,
    )
    add_centered_below_axis_text(
        fig,
        heatmap_axes[-1, :],
        TASK_PROGRESSION_XLABEL,
        y_offset=HEATMAP_PATH_LABEL_OFFSET,
        fontsize=PANEL_E_AXIS_LABEL_FONTSIZE,
    )
    add_panel_header_at_reference_y(
        fig,
        panel_d_axis,
        label="C",
        title=PANEL_C_EXAMPLE_TITLE,
        label_x_offset=PANEL_C_LABEL_X_OFFSET,
        reference_axis=corner_axis,
        reference_y=PANEL_D_LABEL_Y,
        fontsize=8,
    )
    label_axis(corner_axis, "D", x=PANEL_D_LABEL_X, y=PANEL_D_LABEL_Y)
    panel_c_label_x = (
        panel_d_axis.get_position().x0
        + PANEL_C_LABEL_X_OFFSET * panel_d_axis.get_position().width
    )
    panel_d_label_x = fig.transFigure.inverted().transform(
        corner_axis.transAxes.transform((PANEL_D_LABEL_X, PANEL_D_LABEL_Y))
    )[0]
    panel_g_label_x = (
        panel_h_axis.get_position().x0
        + BOTTOM_ROW_PANEL_LABEL_X_OFFSETS[2] * panel_h_axis.get_position().width
    )
    add_aligned_panel_headers_at_label_positions(
        fig,
        (panel_f_axis, panel_g_axis, panel_h_axis),
        labels=("E", "F", "G"),
        titles=(
            "Comparison to motor",
            "Comparison to alternative codes",
            PANEL_G_TITLE,
        ),
        label_x_positions=(panel_c_label_x, panel_d_label_x, panel_g_label_x),
        fontsize=8,
    )
    raise_figure_text_to_minimum_fontsize(fig, MIN_FIGURE_1_FONTSIZE_PT)
    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 1 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 1 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 pooled dark-epoch place-field heatmaps."
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
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help=f"Directory containing Figure 1 external assets. Default: {DEFAULT_ASSET_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--panel-d-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel D heatmap matrices. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-d-cache",
        action="store_true",
        help="Recompute Panel D and overwrite its cache even when a matching cache exists.",
    )
    parser.add_argument(
        "--panel-e-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel E example-cell data. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-e-cache",
        action="store_true",
        help="Recompute Panel E and overwrite its cache even when a matching cache exists.",
    )
    parser.add_argument(
        "--panel-dark-light-example-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached moved Figure 2A example-cell data. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-dark-light-example-cache",
        action="store_true",
        help=(
            "Recompute the moved Figure 2A example cell and overwrite its "
            "cache even when a matching cache exists."
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
        "--encoding-bin-size-s",
        type=float,
        default=ENCODING_COMPARISON_BIN_SIZE_S,
        help=(
            "Time-bin size used to find encoding-comparison summary files. "
            f"Default: {ENCODING_COMPARISON_BIN_SIZE_S}"
        ),
    )
    parser.add_argument(
        "--encoding-place-bin-size-cm",
        type=float,
        default=ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
        help=(
            "Place-bin size used to find encoding-comparison summary files. "
            f"Default: {ENCODING_COMPARISON_PLACE_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--decoding-n-permutations",
        type=int,
        default=DECODING_PERMUTATION_COUNT,
        help=(
            "Label permutations used for Figure 1 decoding inference. "
            f"Default: {DECODING_PERMUTATION_COUNT}"
        ),
    )
    parser.add_argument(
        "--decoding-permutation-seed",
        type=int,
        default=DECODING_PERMUTATION_SEED,
        help=(
            "Random seed used for Figure 1 decoding inference. "
            f"Default: {DECODING_PERMUTATION_SEED}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 1 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_d_cache_dir = (
        args.panel_d_cache_dir
        if args.panel_d_cache_dir is not None
        else args.output_dir / "cache"
    )
    panel_e_cache_dir = (
        args.panel_e_cache_dir
        if args.panel_e_cache_dir is not None
        else args.output_dir / "cache"
    )
    panel_dark_light_example_cache_dir = (
        args.panel_dark_light_example_cache_dir
        if args.panel_dark_light_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_1(
        data_root=args.data_root,
        asset_dir=args.asset_dir,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        encoding_bin_size_s=args.encoding_bin_size_s,
        encoding_place_bin_size_cm=args.encoding_place_bin_size_cm,
        dpi=args.dpi,
        decoding_n_permutations=args.decoding_n_permutations,
        decoding_permutation_seed=args.decoding_permutation_seed,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=args.refresh_panel_d_cache,
        panel_e_cache_dir=panel_e_cache_dir,
        refresh_panel_e_cache=args.refresh_panel_e_cache,
        panel_dark_light_example_cache_dir=panel_dark_light_example_cache_dir,
        refresh_panel_dark_light_example_cache=(
            args.refresh_panel_dark_light_example_cache
        ),
    )


if __name__ == "__main__":
    main()
