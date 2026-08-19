"""Generate Supplementary Figure 3 without its cvPCA panel A."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from v1ca1.paper_figures import supplementary_figure_3 as base
from v1ca1.paper_figures.style import apply_paper_style, figure_size, save_figure


DEFAULT_OUTPUT_DIR = base.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "supplementary_figure_3_3"
DEFAULT_OUTPUT_FORMAT = base.DEFAULT_OUTPUT_FORMAT
DEFAULT_FIGURE_WIDTH_MM = base.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = (
    base.DEFAULT_MOTOR_GRID_HEIGHT_MM
    + base.DEFAULT_BOTTOM_SECTION_SPACER_MM
    + base.DEFAULT_MOTOR_SUMMARY_HEIGHT_MM
)


def make_supplementary_figure_3_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[base.DatasetId],
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 3 panels B and C."""
    import matplotlib.pyplot as plt

    dataset_ids = [base.normalize_dataset_id(dataset) for dataset in datasets]

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=False,
    )
    if not dataset_ids:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        print(f"Saved Supplementary Figure 3_3 to {output_path}")
        return output_path

    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            base.DEFAULT_MOTOR_GRID_HEIGHT_MM,
            base.DEFAULT_BOTTOM_SECTION_SPACER_MM,
            base.DEFAULT_MOTOR_SUMMARY_HEIGHT_MM,
        ],
        hspace=base.PANEL_GRID_HSPACE,
        left=base.PANEL_A_GRID_LEFT,
        right=base.PANEL_A_GRID_RIGHT,
        top=base.PANEL_A_GRID_TOP,
        bottom=base.PANEL_A_GRID_BOTTOM,
    )
    motor_axes, motor_summary_axes = base.add_supplementary_figure_3_bc_panels(
        fig,
        outer_grid[0, 0],
        outer_grid[1, 0],
        outer_grid[2, 0],
        data_root=data_root,
        datasets=dataset_ids,
        panel_b_label="A",
        panel_c_label="B",
    )
    for ax in motor_axes[-1, :]:
        ax.xaxis.label.set_text("Norm. path progression")
    for ax in motor_summary_axes:
        ax.xaxis.label.set_text("Dark-light correlation")

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 3_3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 3_3."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 3 panels B and C."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=base.DEFAULT_DATA_ROOT,
        help=(
            "Base directory containing analysis outputs. "
            f"Default: {base.DEFAULT_DATA_ROOT}"
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
        choices=base.FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=base.parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date[:dark_epoch]. "
            "May be repeated. Default: use v1ca1.paper_figures.datasets."
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
    """Run Supplementary Figure 3_3 generation."""
    args = parse_arguments(argv)
    datasets = (
        args.dataset if args.dataset is not None else base.get_processed_datasets()
    )
    output_path = base.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_3_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
