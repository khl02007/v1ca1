from __future__ import annotations

"""Generate Supplementary Figure 4 per-animal Figure 4C histograms."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_REGIONS,
    FIGURE_FORMATS,
    build_output_path,
    load_panel_h_swap_delta_table,
    parse_dataset_id,
)
from v1ca1.paper_figures.figure_3 import (
    _plot_panel_h_delta_grid as plot_figure_4c_histogram_grid,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "supplementary_figure_4"
LETTER_PAPER_WIDTH_IN = 8.5
LETTER_HORIZONTAL_MARGIN_IN = 1.0
DEFAULT_FIGURE_WIDTH_MM = (
    LETTER_PAPER_WIDTH_IN - 2.0 * LETTER_HORIZONTAL_MARGIN_IN
) * 25.4
DEFAULT_ANIMAL_ROW_HEIGHT_MM = 35.0
ANIMAL_ROW_LABEL_FONTSIZE = 5.2
PER_ANIMAL_GRID_HSPACE = 0.42
PANEL_TITLE_FONTSIZE = 8.0


def group_datasets_by_animal(
    datasets: Sequence[DatasetId],
) -> dict[str, list[DatasetId]]:
    """Return normalized data sets grouped by animal in input order."""
    grouped: dict[str, list[DatasetId]] = {}
    for dataset in datasets:
        normalized = normalize_dataset_id(dataset)
        animal_name = str(normalized[0])
        grouped.setdefault(animal_name, []).append(normalized)
    return grouped


def format_animal_row_label(animal_name: str, datasets: Sequence[DatasetId]) -> str:
    """Return a compact label for one per-animal row."""
    dates = []
    for dataset in datasets:
        _animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        if str(date) not in dates:
            dates.append(str(date))
    if not dates:
        return str(animal_name)
    return f"{animal_name}\n{', '.join(dates)}"


def get_figure_height_mm(n_animal_rows: int) -> float:
    """Return the Supplementary Figure 4 height for the requested row count."""
    return DEFAULT_ANIMAL_ROW_HEIGHT_MM * max(int(n_animal_rows), 1)


def make_supplementary_figure_4(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 4."""
    import matplotlib.pyplot as plt

    datasets = [normalize_dataset_id(dataset) for dataset in datasets]
    animal_groups = group_datasets_by_animal(datasets)

    apply_paper_style()
    fig_height_mm = get_figure_height_mm(len(animal_groups))
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=False,
    )
    if not datasets:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        print(f"Saved Supplementary Figure 4 to {output_path}")
        return output_path

    outer_grid = fig.add_gridspec(
        nrows=len(animal_groups),
        ncols=1,
        hspace=PER_ANIMAL_GRID_HSPACE,
        left=0.125,
        right=0.985,
        top=0.94,
        bottom=0.06,
    )
    for row_index, (animal_name, animal_datasets) in enumerate(animal_groups.items()):
        axis = fig.add_subplot(outer_grid[row_index, 0])
        swap_delta_table = load_panel_h_swap_delta_table(
            data_root=data_root,
            datasets=animal_datasets,
            region=region,
            dark_epoch=dark_epoch,
        )
        plot_figure_4c_histogram_grid(axis, swap_delta_table)
        axis.text(
            -0.075,
            0.50,
            format_animal_row_label(animal_name, animal_datasets),
            ha="right",
            va="center",
            fontsize=ANIMAL_ROW_LABEL_FONTSIZE,
            transform=axis.transAxes,
            color="0.25",
        )

        if row_index == 0:
            axis.set_title(
                "Figure 4C delta LL histograms",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=2,
            )
            label_axis(axis, "A", x=-0.115, y=1.02)

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 4 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 4 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 4 per-animal Figure 4C histograms."
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
        choices=REGIONS,
        default=DEFAULT_REGIONS[0],
        help=f"Region to include. Default: {DEFAULT_REGIONS[0]}.",
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
    """Run Supplementary Figure 4 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_4(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
