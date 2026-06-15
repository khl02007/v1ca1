from __future__ import annotations

"""Generate Figure 4 panels moved from the dark-light Figure 3 layout."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_processed_datasets,
)
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_2_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_2_WIDTH_MM,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_REGIONS,
    FIGURE_FORMATS,
    PANEL_GH_WIDTH_RATIOS,
    PANEL_H_INDEPENDENT_TRACK_CENTER_Y,
    PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y,
    PANEL_H_SHARED_DARK_TRACK_CENTER_Y,
    PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y,
    build_output_path,
    load_panel_glm_data,
    parse_dataset_id,
    plot_panel_g_model_architecture,
    plot_panel_h_swap_delta,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "figure_4"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_2_WIDTH_MM
PANEL_A_TO_GH_HEIGHT_RATIOS = (0.637, 1.3)
DEFAULT_FIGURE_HEIGHT_MM = (
    FIGURE_2_HEIGHT_MM
    * 1.3
    * PANEL_A_TO_GH_HEIGHT_RATIOS[1]
    / sum(PANEL_A_TO_GH_HEIGHT_RATIOS)
)
FIGURE_4_CONSTRAINED_LAYOUT_PADS = {
    "h_pad": 0.01,
    "w_pad": 0.01,
    "hspace": 0.01,
    "wspace": 0.02,
}
PANEL_BC_LABEL_Y = 1.03
PANEL_BC_TITLE_PAD = 0.5
PANEL_B_SCHEMATIC_HEIGHT_FRACTION = 0.72
PANEL_B_SCHEMATIC_TRACK_SIZE = (0.2025, 0.2547)
PANEL_B_INDEPENDENT_BASIS_ICON_SCALE = 0.70
PANEL_B_INDEPENDENT_BASIS_LABEL = "Independent"
PANEL_B_EXAMPLE_AXIS_BOUNDS = (0.0, 0.01, 1.0, 0.44)
PANEL_B_EXAMPLE_FIELD_Y = 0.13
PANEL_B_EXAMPLE_FIELD_HEIGHT = 0.62
PANEL_B_EXAMPLE_ICON_BOUNDS = (0.04, 0.27, 0.09, 0.34)
PANEL_B_EXAMPLE_XLABEL_Y = 0.02
PANEL_B_EXAMPLE_COLUMN_WIDTH = 0.50
PANEL_B_EXAMPLE_COLUMN_GAP = 0.0
PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET = 0.20
PANEL_B_EXAMPLE_FIELD_WIDTH = 0.28
PANEL_B_EXAMPLE_FIELD_GAP = 0.075
PANEL_B_EXAMPLE_LAYOUT = "rows"
PANEL_B_EXAMPLE_ROW_HEIGHT = 0.46
PANEL_B_EXAMPLE_ROW_GAP = 0.05
PANEL_B_MODEL_LABEL_X = 0.03
PANEL_B_MODEL_LABEL_FONTSIZE = 5.8
PANEL_B_COMPONENT_LABEL_FONTSIZE = 5.8
PANEL_B_SEGMENT_MODULATION_LABEL = "Segment-specific\nmodulation"
PANEL_B_SEGMENT_MODULATION_LABEL_Y = 0.545
PANEL_INDEPENDENT_MODEL_COLOR = "#0072B2"
PANEL_SHARED_SCAFFOLD_MODEL_COLOR = "#CC79A7"
PANEL_B_EXAMPLE_MODEL_COLORS = {
    "visual": PANEL_INDEPENDENT_MODEL_COLOR,
    "task_segment_bump": PANEL_SHARED_SCAFFOLD_MODEL_COLOR,
}
PANEL_B_EXAMPLE_MODEL_LABELS = {
    "visual": "Independent",
    "task_segment_bump": "Shared-scaffold",
}
PANEL_B_ALIGNMENT_SCHEMATIC_AXIS_BOUNDS = (-0.06, 0.39, 0.40, 0.58)
PANEL_C_SCHEMATIC_AXIS_BOUNDS = (-0.08, 0.25, 0.40, 0.72)
PANEL_C_DELTA_AXIS_BOUNDS = (0.39, 0.35, 0.60, 0.59)
PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION = 0.5
PANEL_C_DELTA_GRID_BOUNDS = (
    (0.035, 0.42, 0.445, 0.50),
    (0.535, 0.42, 0.445, 0.50),
    (0.035, -0.22, 0.445, 0.50),
    (0.535, -0.22, 0.445, 0.50),
)
PANEL_C_DELTA_XLABEL_Y = -0.40
PANEL_C_SWAP_EXAMPLES = (
    ("L15", "20241121", "v1", 27, "center_to_right"),
    ("L19", "20250930", "v1", 4, "center_to_left"),
    ("L15", "20241121", "v1", 146, "center_to_right"),
)
PANEL_C_EXAMPLE_AXIS_BOUNDS = (
    (0.095, -0.18, 0.20, 0.19),
    (0.405, -0.18, 0.20, 0.19),
    (0.715, -0.18, 0.20, 0.19),
)
PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS = (
    (0.96, 0.94),
    (0.96, 0.06),
    (0.96, 0.94),
)
PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS = ("top", "bottom", "top")
PANEL_C_EXAMPLE_ICON_BOUNDS = (-0.46, 0.28, 0.26, 0.38)
PANEL_C_PREDICTION_LABEL_FONTSIZE = 5.8
PANEL_C_SWAP_MODEL_NAME = "task_segment_scalar"
PANEL_C_SWAP_MODEL_COLORS = {
    "visual": PANEL_INDEPENDENT_MODEL_COLOR,
    PANEL_C_SWAP_MODEL_NAME: PANEL_SHARED_SCAFFOLD_MODEL_COLOR,
}
PANEL_C_SWAP_MODEL_LABELS = {
    PANEL_C_SWAP_MODEL_NAME: "Shared-scaffold",
}
PANEL_C_INDEPENDENT_TRACK_CENTER_Y = 0.742
PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y = 0.60
PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y = 0.34
PANEL_C_SHARED_DARK_TRACK_CENTER_Y = 0.0
PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y = 0.17
PANEL_C_SHARED_PREDICTION_LABEL_Y = -0.24
PANEL_C_SCHEMATIC_TRACK_SIZE = (0.628, 0.316)
PANEL_C_HORIZONTAL_SHIFT = -0.025


def _panel_b_schematic_center_y_for_panel_c_center_y(panel_c_center_y: float) -> float:
    """Return a Panel B schematic y-center aligned to one Panel C schematic row."""
    panel_b_schematic_bottom = 1.0 - PANEL_B_SCHEMATIC_HEIGHT_FRACTION
    panel_c_parent_center_y = (
        PANEL_B_ALIGNMENT_SCHEMATIC_AXIS_BOUNDS[1]
        + PANEL_B_ALIGNMENT_SCHEMATIC_AXIS_BOUNDS[3] * float(panel_c_center_y)
    )
    return (
        panel_c_parent_center_y - panel_b_schematic_bottom
    ) / PANEL_B_SCHEMATIC_HEIGHT_FRACTION


PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y = (
    _panel_b_schematic_center_y_for_panel_c_center_y(
        PANEL_H_INDEPENDENT_TRACK_CENTER_Y
    )
)
PANEL_B_FIELD_LABEL_Y = 0.9619
PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y = (
    _panel_b_schematic_center_y_for_panel_c_center_y(
        (
            PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y
            + PANEL_H_SHARED_DARK_TRACK_CENTER_Y
            + PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y
        )
        / 3.0
    )
)


def _shift_axis_horizontally(ax: Any, dx_figure_fraction: float) -> None:
    """Shift one axis after constrained layout has selected its size."""
    if dx_figure_fraction == 0.0:
        return
    box = ax.get_position()
    ax.set_axes_locator(None)
    ax.set_position(
        [
            box.x0 + dx_figure_fraction,
            box.y0,
            box.width,
            box.height,
        ]
    )


def _align_texts_to_reference_display_y(texts: Sequence[Any]) -> None:
    """Align text artists to the first text's rendered vertical position."""
    if len(texts) < 2:
        return
    for text in texts:
        axes = getattr(text, "axes", None)
        if axes is not None and text is axes.title:
            axes._autotitlepos = False
    reference_display = texts[0].get_transform().transform(texts[0].get_position())
    for text in texts[1:]:
        x_position, _y_position = text.get_position()
        display_position = text.get_transform().transform(text.get_position())
        adjusted_position = text.get_transform().inverted().transform(
            (display_position[0], reference_display[1])
        )
        text.set_position((x_position, float(adjusted_position[1])))


def make_figure_4(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
) -> Path:
    """Build and save Figure 4."""
    import matplotlib.pyplot as plt

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
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(**FIGURE_4_CONSTRAINED_LAYOUT_PADS)
    glm_grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_GH_WIDTH_RATIOS,
    )
    panel_b_axis = fig.add_subplot(glm_grid[0, 0])
    panel_c_axis = fig.add_subplot(glm_grid[0, 1])

    plot_panel_g_model_architecture(
        panel_b_axis,
        panel_glm_payload["dark_light_examples"],
        independent_track_center_y=PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y,
        shared_track_center_y=PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y,
        schematic_height_fraction=PANEL_B_SCHEMATIC_HEIGHT_FRACTION,
        schematic_track_size=PANEL_B_SCHEMATIC_TRACK_SIZE,
        independent_basis_icon_scale=PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
        independent_basis_label=PANEL_B_INDEPENDENT_BASIS_LABEL,
        show_dark_track_labels=True,
        field_label_y=PANEL_B_FIELD_LABEL_Y,
        model_label_x=PANEL_B_MODEL_LABEL_X,
        model_label_fontsize=PANEL_B_MODEL_LABEL_FONTSIZE,
        component_label_fontsize=PANEL_B_COMPONENT_LABEL_FONTSIZE,
        segment_modulation_label_y=PANEL_B_SEGMENT_MODULATION_LABEL_Y,
        segment_modulation_label=PANEL_B_SEGMENT_MODULATION_LABEL,
        example_axis_bounds=PANEL_B_EXAMPLE_AXIS_BOUNDS,
        example_field_y=PANEL_B_EXAMPLE_FIELD_Y,
        example_field_height=PANEL_B_EXAMPLE_FIELD_HEIGHT,
        example_icon_bounds=PANEL_B_EXAMPLE_ICON_BOUNDS,
        example_xlabel_y=PANEL_B_EXAMPLE_XLABEL_Y,
        example_column_width=PANEL_B_EXAMPLE_COLUMN_WIDTH,
        example_column_gap=PANEL_B_EXAMPLE_COLUMN_GAP,
        example_plot_left_offset=PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET,
        example_field_width=PANEL_B_EXAMPLE_FIELD_WIDTH,
        example_field_gap=PANEL_B_EXAMPLE_FIELD_GAP,
        example_layout=PANEL_B_EXAMPLE_LAYOUT,
        example_row_height=PANEL_B_EXAMPLE_ROW_HEIGHT,
        example_row_gap=PANEL_B_EXAMPLE_ROW_GAP,
        model_colors=PANEL_B_EXAMPLE_MODEL_COLORS,
        model_labels=PANEL_B_EXAMPLE_MODEL_LABELS,
    )
    plot_panel_h_swap_delta(
        panel_c_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
        model_name=PANEL_C_SWAP_MODEL_NAME,
        model_colors=PANEL_C_SWAP_MODEL_COLORS,
        model_labels=PANEL_C_SWAP_MODEL_LABELS,
        schematic_axis_bounds=PANEL_C_SCHEMATIC_AXIS_BOUNDS,
        delta_axis_bounds=PANEL_C_DELTA_AXIS_BOUNDS,
        example_axis_bounds=PANEL_C_EXAMPLE_AXIS_BOUNDS,
        schematic_track_size=PANEL_C_SCHEMATIC_TRACK_SIZE,
        show_dark_track_labels=True,
        show_model_labels=False,
        prediction_label_fontsize=PANEL_C_PREDICTION_LABEL_FONTSIZE,
        independent_track_center_y=PANEL_C_INDEPENDENT_TRACK_CENTER_Y,
        independent_prediction_label_y=PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y,
        segment_modulation_track_center_y=PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y,
        shared_dark_track_center_y=PANEL_C_SHARED_DARK_TRACK_CENTER_Y,
        shared_light_track_center_y=PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y,
        shared_prediction_label_y=PANEL_C_SHARED_PREDICTION_LABEL_Y,
        delta_grid_bounds=PANEL_C_DELTA_GRID_BOUNDS,
        delta_xlabel_y=PANEL_C_DELTA_XLABEL_Y,
        example_delta_label_positions=PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS,
        example_delta_label_vertical_alignments=(
            PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS
        ),
        example_icon_bounds=PANEL_C_EXAMPLE_ICON_BOUNDS,
    )

    label_axis(panel_b_axis, "A", x=-0.035, y=PANEL_BC_LABEL_Y)
    panel_b_label = panel_b_axis.texts[-1]
    label_axis(panel_c_axis, "B", x=-0.035, y=PANEL_BC_LABEL_Y)
    panel_c_label = panel_c_axis.texts[-1]
    panel_b_title = panel_b_axis.set_title(
        "Two models that relate dark and light activity",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )
    panel_c_title = panel_c_axis.set_title(
        "Predicting activity in held-out light epoch",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )

    fig.canvas.draw()
    _shift_axis_horizontally(panel_c_axis, PANEL_C_HORIZONTAL_SHIFT)
    fig.canvas.draw()
    _align_texts_to_reference_display_y((panel_b_label, panel_c_label))
    _align_texts_to_reference_display_y((panel_b_title, panel_c_title))
    fig.set_layout_engine(None)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 4 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 4 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 dark-light example and GLM panels."
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
        "--light-epoch",
        default=None,
        help=(
            "Light run epoch for GLM panels. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help=(
            "Dark run epoch. "
            f"Default: registry value, currently {DEFAULT_DARK_EPOCH} unless overridden."
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
    """Run Figure 4 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_figure_4(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
