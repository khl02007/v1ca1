"""Generate Figure 3 from the model and cue-swap panels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures import figure_2_old as _figure_2
from v1ca1.paper_figures._dark_light import (
    load_panel_h_swap_delta_table,
    load_panel_h_swap_examples,
)
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_DIR = _figure_2.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "figure_3"
DEFAULT_OUTPUT_FORMAT = _figure_2.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = _figure_2.FIGURE_FORMATS
DEFAULT_REGIONS = _figure_2.DEFAULT_REGIONS
DEFAULT_FIGURE_WIDTH_MM = _figure_2.DEFAULT_FIGURE_WIDTH_MM / 2.0
TOP_ROW_HEIGHT_MM = _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM
BOTTOM_ROW_HEIGHT_MM = _figure_2.PANEL_D_ROW_HEIGHT_MM
DEFAULT_FIGURE_HEIGHT_MM = TOP_ROW_HEIGHT_MM + BOTTOM_ROW_HEIGHT_MM
PANEL_TITLES = (
    "Two models that relate dark and light activity",
    "Dark and light stimulus-swap prediction comparison",
)
PANEL_LABEL_FONTSIZE = 8.0
PANEL_A_B_LABEL_X = -0.035
PANEL_B_LABEL_X = PANEL_A_B_LABEL_X
PANEL_C_LABEL_X = _figure_2.PANEL_D2_HISTOGRAM_AXIS_BOUNDS[0] - 0.03
PANEL_BC_SPLIT_LABEL_Y = 0.92
PANEL_BC_TITLE_PAD = _figure_2.PANEL_BC_TITLE_PAD + 2.5


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the Figure 3 output path for a supported format."""
    return _figure_2.build_output_path(output_dir, output_name, output_format)


def load_figure_3_panel_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    dark_epoch: str | None,
) -> dict[str, Any]:
    """Load only the cue-swap data used by Figure 3 Panel B."""
    dataset_ids = tuple(datasets)
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    swap_examples = load_panel_h_swap_examples(
        data_root=data_root,
        datasets=dataset_ids,
        region=quant_region,
        dark_epoch=dark_epoch,
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
        example_count=len(_figure_2.PANEL_C_SWAP_EXAMPLES),
        requested_examples=_figure_2.PANEL_C_SWAP_EXAMPLES,
    )
    swap_delta_table = load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=quant_region,
        dark_epoch=dark_epoch,
        min_movement_firing_rate_hz=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        min_tuning_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
    )
    return {
        "swap_delta": swap_delta_table,
        "swap_examples": swap_examples,
    }


def make_figure_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    dark_epoch: str | None,
    dpi: int,
) -> Path:
    """Build and save Figure 3."""
    import matplotlib.pyplot as plt

    panel_data = load_figure_3_panel_data(
        data_root=data_root,
        datasets=datasets,
        regions=regions,
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
        nrows=2,
        ncols=1,
        height_ratios=(TOP_ROW_HEIGHT_MM, BOTTOM_ROW_HEIGHT_MM),
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    panel_b_axis = fig.add_subplot(outer_grid[1, 0])

    _figure_2.plot_panel_d2_architecture_panel(panel_a_axis)
    _figure_2.plot_panel_d2_swap_results_panel(
        panel_b_axis,
        panel_data["swap_delta"],
        panel_data["swap_examples"],
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
        model_colors=_figure_2.PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=_figure_2.PANEL_C_SWAP_MODEL_LABELS_2_3,
    )

    label_axis(
        panel_a_axis,
        "A",
        x=PANEL_A_B_LABEL_X,
        y=_figure_2.PANEL_B_LABEL_Y,
        va="baseline",
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    panel_a_label = panel_a_axis.texts[-1]
    label_axis(
        panel_b_axis,
        "B",
        x=PANEL_B_LABEL_X,
        y=PANEL_BC_SPLIT_LABEL_Y,
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    panel_b_label = panel_b_axis.texts[-1]
    label_axis(
        panel_b_axis,
        "C",
        x=PANEL_C_LABEL_X,
        y=PANEL_BC_SPLIT_LABEL_Y,
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    panel_c_label = panel_b_axis.texts[-1]
    panel_a_axis.set_title(
        PANEL_TITLES[0],
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    panel_b_axis.set_title(
        PANEL_TITLES[1],
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )

    _figure_2._raise_text_to_minimum_fontsize(
        fig,
        _figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _figure_2._align_text_to_reference_display_x(panel_b_label, panel_a_label)
    _figure_2._align_texts_to_reference_display_y(
        (panel_b_label, panel_c_label)
    )
    _figure_2._align_text_tops_to_reference_display_y(
        fig,
        (panel_b_label, panel_c_label),
    )

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 3 generation."""
    parser = argparse.ArgumentParser(description="Generate Figure 3.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Base directory containing analysis outputs. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
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
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 3 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_figure_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
