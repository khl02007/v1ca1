from __future__ import annotations

"""Generate Supplementary Figure 3 per-animal swap-model histograms."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import _format_cell_animal_count
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_3_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_3_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_REGIONS,
    FIGURE_FORMATS,
    PANEL_C_TRAJECTORY_COLORS,
    PANEL_H_DELTA_TRAJECTORIES,
    PANEL_H_HELDOUT_LIGHT_EPOCH,
    PANEL_H_TRAIN_LIGHT_EPOCH,
    _filter_panel_h_heldout_delta,
    _plot_panel_h_delta_axis,
    build_output_path,
    load_panel_h_swap_delta_table,
    parse_dataset_id,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_NAME = "supplementary_figure_3"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_3_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = FIGURE_3_HEIGHT_MM
ANIMAL_ROWS_AXIS_BOUNDS = (0.11, 0.07, 0.87, 0.88)
ANIMAL_ROW_GAP = 0.035
ANIMAL_ROW_LABEL_FONTSIZE = 5.2
TRAJECTORY_CELL_WIDTH = 0.205
TRAJECTORY_ICON_BOUNDS = (0.315, 0.78, 0.37, 0.17)
TRAJECTORY_HISTOGRAM_BOUNDS = (0.0, 0.15, 1.0, 0.58)
COUNT_TEXT_FONTSIZE = 3.2


def group_datasets_by_animal(datasets: Sequence[DatasetId]) -> dict[str, list[DatasetId]]:
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


def select_animal_swap_delta_table(swap_delta_table: Any, animal_name: str) -> Any:
    """Return swap-delta rows for one animal."""
    if swap_delta_table is None or "animal_name" not in swap_delta_table:
        return swap_delta_table
    return swap_delta_table[
        swap_delta_table["animal_name"].astype(str) == str(animal_name)
    ]


def select_trajectory_swap_delta_table(
    swap_delta_table: Any,
    trajectory_type: str,
) -> Any:
    """Return swap-delta rows for one trajectory."""
    swap_delta_table = _filter_panel_h_heldout_delta(swap_delta_table)
    if swap_delta_table is None or "trajectory" not in swap_delta_table:
        return swap_delta_table
    return swap_delta_table[
        swap_delta_table["trajectory"].astype(str) == str(trajectory_type)
    ]


def add_cell_animal_count_text(ax: Any, swap_delta_table: Any) -> None:
    """Add the Figure-1F-style cell and animal count label."""
    if swap_delta_table is None:
        return
    ax.text(
        0.03,
        0.06,
        _format_cell_animal_count(
            swap_delta_table,
            value_column="delta_ll_bits_per_spike",
        ),
        ha="left",
        va="bottom",
        fontsize=COUNT_TEXT_FONTSIZE,
        color="0.25",
        transform=ax.transAxes,
    )


def plot_panel_h_animal_histogram_row(
    ax: Any,
    swap_delta_table: Any,
    *,
    show_xticklabels: bool,
) -> list[Any]:
    """Plot the four Panel-H trajectory histograms in one horizontal row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    cell_gap = (1.0 - len(PANEL_H_DELTA_TRAJECTORIES) * TRAJECTORY_CELL_WIDTH) / (
        len(PANEL_H_DELTA_TRAJECTORIES) - 1
    )
    histogram_axes = []
    for trajectory_index, trajectory_type in enumerate(PANEL_H_DELTA_TRAJECTORIES):
        cell_left = trajectory_index * (TRAJECTORY_CELL_WIDTH + cell_gap)
        icon_x, icon_y, icon_width, icon_height = TRAJECTORY_ICON_BOUNDS
        icon_ax = ax.inset_axes(
            [
                cell_left + icon_x * TRAJECTORY_CELL_WIDTH,
                icon_y,
                icon_width * TRAJECTORY_CELL_WIDTH,
                icon_height,
            ]
        )
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_C_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.42,
            trajectory_linewidth=0.68,
            arrow_mutation_scale=5.4,
            fill_track=False,
        )

        hist_x, hist_y, hist_width, hist_height = TRAJECTORY_HISTOGRAM_BOUNDS
        hist_ax = ax.inset_axes(
            [
                cell_left + hist_x * TRAJECTORY_CELL_WIDTH,
                hist_y,
                hist_width * TRAJECTORY_CELL_WIDTH,
                hist_height,
            ]
        )
        _plot_panel_h_delta_axis(
            hist_ax,
            swap_delta_table,
            trajectory_type=trajectory_type,
            show_xticklabels=show_xticklabels,
            show_yticklabels=trajectory_index == 0,
        )
        trajectory_table = select_trajectory_swap_delta_table(
            swap_delta_table,
            trajectory_type,
        )
        add_cell_animal_count_text(hist_ax, trajectory_table)
        histogram_axes.append(hist_ax)
    return histogram_axes


def plot_panel_h_animal_histogram_rows(
    ax: Any,
    swap_delta_table: Any,
    *,
    datasets: Sequence[DatasetId],
) -> list[Any]:
    """Plot one horizontal Panel-H histogram row per animal."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    grouped_datasets = group_datasets_by_animal(datasets)
    if not grouped_datasets:
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        return []

    axis_x, axis_y, axis_width, axis_height = ANIMAL_ROWS_AXIS_BOUNDS
    n_animals = len(grouped_datasets)
    slot_height = (axis_height - ANIMAL_ROW_GAP * (n_animals - 1)) / n_animals
    row_axes = []
    for animal_index, (animal_name, animal_datasets) in enumerate(
        grouped_datasets.items()
    ):
        row_bottom = (
            axis_y
            + axis_height
            - (animal_index + 1) * slot_height
            - animal_index * ANIMAL_ROW_GAP
        )
        ax.text(
            0.01,
            row_bottom + 0.5 * slot_height,
            format_animal_row_label(animal_name, animal_datasets),
            ha="left",
            va="center",
            fontsize=ANIMAL_ROW_LABEL_FONTSIZE,
            transform=ax.transAxes,
        )
        row_ax = ax.inset_axes([axis_x, row_bottom, axis_width, slot_height])
        animal_table = select_animal_swap_delta_table(swap_delta_table, animal_name)
        plot_panel_h_animal_histogram_row(
            row_ax,
            animal_table,
            show_xticklabels=animal_index == n_animals - 1,
        )
        row_axes.append(row_ax)

    ax.text(
        axis_x + 0.5 * axis_width,
        0.02,
        "Delta log likelihood (bits/spike)",
        ha="center",
        va="bottom",
        fontsize=6.0,
        transform=ax.transAxes,
    )
    ax.text(
        axis_x - 0.045,
        axis_y + 0.5 * axis_height,
        "Frac.",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        transform=ax.transAxes,
    )
    return row_axes


def make_supplementary_figure_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 3."""
    import matplotlib.pyplot as plt

    swap_delta_table = load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        light_epoch_pairs=((PANEL_H_TRAIN_LIGHT_EPOCH, PANEL_H_HELDOUT_LIGHT_EPOCH),),
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    panel_a_axis = fig.add_subplot(1, 1, 1)
    plot_panel_h_animal_histogram_rows(
        panel_a_axis,
        swap_delta_table,
        datasets=datasets,
    )

    panel_a_axis.set_title(
        "Predicting activity in held-out light epoch by animal",
        fontsize=8,
        pad=2,
    )
    label_axis(panel_a_axis, "A", x=-0.02, y=1.01)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Supplementary Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 3 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 3 per-animal Panel H histograms."
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
        help="Dark run epoch. Default: registry value for each animal.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Supplementary Figure 3 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
