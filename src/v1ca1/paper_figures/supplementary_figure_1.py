from __future__ import annotations

"""Generate Supplementary Figure 1 per-data-set model comparison panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DECODING_COMPARISON_REGION,
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_1_WIDTH_MM,
    ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
    ENCODING_COMPARISON_REGION,
    MOTOR_DELTA_REGION,
    STABILITY_REGIONS,
    draw_panel_a_anatomy_assets,
    load_decoding_absolute_error_table,
    load_dark_epoch_stability_table,
    load_encoding_delta_table,
    load_motor_delta_table,
    parse_dataset_id,
    plot_decoding_error_panel,
    plot_encoding_delta_panel,
    plot_motor_delta_panel,
    plot_stability_panel,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "supplementary_figure_1"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_1_WIDTH_MM
DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM = 40.0
DEFAULT_DATASET_PANEL_HEIGHT_MM = 26.0
DEFAULT_SECTION_SPACER_MM = 5.0
MODEL_COMPARISON_GRID_WSPACE = -0.10
MODEL_COMPARISON_SECOND_COLUMN_SHIFT_PT = -10.0
MODEL_COMPARISON_THIRD_COLUMN_SHIFT_PT = -33.0
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
DATASET_STACK_ROW_GAP = 0.045
DATASET_STACK_LABELED_AXIS_BOUNDS = (0.16, 0.10, 0.82, 0.80)
DATASET_STACK_UNLABELED_AXIS_BOUNDS = (0.04, 0.10, 0.94, 0.80)
DATASET_LABEL_FONTSIZE = 4.8
DATASET_LABEL_X = 0.01
MOTOR_DATASET_STACK_AXIS_BOUNDS = (0.22, 0.10, 0.76, 0.80)
MOTOR_DATASET_LABEL_X = 0.045


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


def shift_axis_horizontally(ax: Any, dx_figure_fraction: float) -> None:
    """Shift an axis horizontally in figure coordinates."""
    if dx_figure_fraction == 0.0:
        return
    box = ax.get_position()
    ax.set_position(
        [
            box.x0 + dx_figure_fraction,
            box.y0,
            box.width,
            box.height,
        ]
    )


def shift_model_comparison_columns(fig: Any, axes: Sequence[Any]) -> None:
    """Tighten spacing between the model-comparison columns."""
    fig_width_pt = float(fig.get_figwidth()) * 72.0
    if fig_width_pt <= 0.0:
        return
    shifts_pt = (
        0.0,
        MODEL_COMPARISON_SECOND_COLUMN_SHIFT_PT,
        MODEL_COMPARISON_THIRD_COLUMN_SHIFT_PT,
    )
    for ax, shift_pt in zip(axes, shifts_pt, strict=True):
        shift_axis_horizontally(ax, shift_pt / fig_width_pt)


def format_dataset_label(dataset: DatasetId) -> str:
    """Return a compact row label for one data set."""
    animal_name, date, epoch = normalize_dataset_id(dataset)
    return f"{animal_name}\n{date}\n{epoch}"


def format_dataset_animal_label(dataset: DatasetId) -> str:
    """Return the animal-only row label for one data set."""
    animal_name, _date, _epoch = normalize_dataset_id(dataset)
    return animal_name


def plot_dataset_stack_panel(
    ax: Any,
    *,
    datasets: Sequence[DatasetId],
    plot_dataset: Any,
    axis_bounds: tuple[float, float, float, float] | None = None,
    show_dataset_labels: bool = True,
    dataset_label_formatter: Any = format_dataset_label,
    dataset_label_x: float = DATASET_LABEL_X,
    dataset_label_ha: str = "left",
) -> list[Any]:
    """Plot one vertically stacked per-data-set panel."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not datasets:
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        return []

    n_datasets = len(datasets)
    row_gap = DATASET_STACK_ROW_GAP
    slot_height = (1.0 - row_gap * (n_datasets - 1)) / n_datasets
    if axis_bounds is None:
        axis_bounds = (
            DATASET_STACK_LABELED_AXIS_BOUNDS
            if show_dataset_labels
            else DATASET_STACK_UNLABELED_AXIS_BOUNDS
        )
    row_axes = []
    for dataset_index, dataset in enumerate(datasets):
        row_bottom = 1.0 - (dataset_index + 1) * slot_height - dataset_index * row_gap
        plot_x, plot_y, plot_width, plot_height = axis_bounds
        row_ax = ax.inset_axes(
            [
                plot_x,
                row_bottom + plot_y * slot_height,
                plot_width,
                plot_height * slot_height,
            ]
        )
        plot_dataset(row_ax, dataset)
        if show_dataset_labels:
            ax.text(
                dataset_label_x,
                row_bottom + 0.5 * slot_height,
                dataset_label_formatter(dataset),
                ha=dataset_label_ha,
                va="center",
                fontsize=DATASET_LABEL_FONTSIZE,
                transform=ax.transAxes,
            )
        row_axes.append(row_ax)
    return row_axes


def keep_only_bottom_x_axis_labels(row_axes: Sequence[Any]) -> None:
    """Hide x-axis text above the bottom row in a stacked panel."""
    for row_ax in list(row_axes)[:-1]:
        for plot_ax in [row_ax, *list(row_ax.child_axes)]:
            plot_ax.set_xlabel("")
            plot_ax.tick_params(axis="x", labelbottom=False)


def plot_motor_delta_dataset_stack_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    show_dataset_labels: bool = True,
) -> list[Any]:
    """Plot Figure 1F-style motor comparisons for each data set."""

    def plot_dataset(row_ax: Any, dataset: DatasetId) -> None:
        motor_delta_table = load_motor_delta_table(
            data_root=data_root,
            datasets=[dataset],
            region=MOTOR_DELTA_REGION,
        )
        plot_motor_delta_panel(row_ax, motor_delta_table)

    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
        axis_bounds=MOTOR_DATASET_STACK_AXIS_BOUNDS,
        show_dataset_labels=show_dataset_labels,
        dataset_label_formatter=format_dataset_animal_label,
        dataset_label_x=MOTOR_DATASET_LABEL_X,
        dataset_label_ha="right",
    )
    keep_only_bottom_x_axis_labels(row_axes)
    return row_axes


def plot_encoding_delta_dataset_stack_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    place_bin_size_cm: float,
    show_dataset_labels: bool = True,
) -> list[Any]:
    """Plot Figure 1G-style encoding comparisons for each data set."""

    def plot_dataset(row_ax: Any, dataset: DatasetId) -> None:
        encoding_delta_table = load_encoding_delta_table(
            data_root=data_root,
            datasets=[dataset],
            region=ENCODING_COMPARISON_REGION,
            place_bin_size_cm=place_bin_size_cm,
        )
        plot_encoding_delta_panel(row_ax, encoding_delta_table)

    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
        show_dataset_labels=show_dataset_labels,
    )
    keep_only_bottom_x_axis_labels(row_axes)
    return row_axes


def remove_decoding_schematic_icons(ax: Any) -> None:
    """Remove Figure 1H trajectory schematics from a decoding-error axis."""
    for child_ax in list(ax.child_axes)[1:]:
        child_ax.remove()
    for text in list(ax.texts):
        if text.get_text() == "Train":
            text.remove()


def plot_decoding_error_dataset_stack_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    show_dataset_labels: bool = True,
) -> list[Any]:
    """Plot Figure 1H-style decoding errors for each data set."""

    def plot_dataset(row_ax: Any, dataset: DatasetId) -> None:
        decoding_error_table = load_decoding_absolute_error_table(
            data_root=data_root,
            datasets=[dataset],
            region=DECODING_COMPARISON_REGION,
        )
        plot_decoding_error_panel(row_ax, decoding_error_table)
        remove_decoding_schematic_icons(row_ax)

    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
        show_dataset_labels=show_dataset_labels,
    )
    keep_only_bottom_x_axis_labels(row_axes)
    return row_axes


def plot_pooled_stability_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
) -> None:
    """Plot the pooled Figure 1C tuning-stability panel."""
    stability_table = load_dark_epoch_stability_table(
        data_root=data_root,
        datasets=datasets,
        regions=STABILITY_REGIONS,
    )
    plot_stability_panel(ax, stability_table)


def make_supplementary_figure_1(
    *,
    data_root: Path,
    asset_dir: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    encoding_place_bin_size_cm: float,
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 1."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    model_row_height_mm = DEFAULT_DATASET_PANEL_HEIGHT_MM * max(len(datasets), 1)
    fig_height_mm = (
        DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM
        + DEFAULT_SECTION_SPACER_MM
        + model_row_height_mm
    )
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
            DEFAULT_SECTION_SPACER_MM,
            model_row_height_mm,
        ],
    )
    moved_grid = outer_grid[0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[0.42, 0.58],
        wspace=0.15,
    )
    moved_anatomy_axis = fig.add_subplot(moved_grid[0, 0])
    draw_panel_a_anatomy_assets(moved_anatomy_axis, asset_dir=asset_dir)
    moved_anatomy_axis.set_title("Probe and histology", fontsize=8, pad=2)
    label_axis(moved_anatomy_axis, "A", x=-0.02, y=1.01)

    moved_stability_axis = fig.add_subplot(moved_grid[0, 1])
    plot_pooled_stability_panel(
        moved_stability_axis,
        data_root=data_root,
        datasets=datasets,
    )
    moved_stability_axis.set_title("Pooled tuning stability", fontsize=8, pad=2)
    label_axis(moved_stability_axis, "B", x=-0.02, y=1.01)

    moved_spacer_axis = fig.add_subplot(outer_grid[1])
    moved_spacer_axis.axis("off")

    model_grid = outer_grid[2].subgridspec(
        nrows=1,
        ncols=3,
        wspace=MODEL_COMPARISON_GRID_WSPACE,
    )
    panel_a_axis = fig.add_subplot(model_grid[0, 0])
    panel_b_axis = fig.add_subplot(model_grid[0, 1])
    panel_c_axis = fig.add_subplot(model_grid[0, 2])
    plot_motor_delta_dataset_stack_panel(
        panel_a_axis,
        data_root=data_root,
        datasets=datasets,
    )
    panel_a_axis.set_title("Comparison to motor", fontsize=8, pad=2)
    plot_encoding_delta_dataset_stack_panel(
        panel_b_axis,
        data_root=data_root,
        datasets=datasets,
        place_bin_size_cm=encoding_place_bin_size_cm,
        show_dataset_labels=False,
    )
    panel_b_axis.set_title("Comparison to alternative codes", fontsize=8, pad=2)
    plot_decoding_error_dataset_stack_panel(
        panel_c_axis,
        data_root=data_root,
        datasets=datasets,
        show_dataset_labels=False,
    )
    panel_c_axis.set_title("Cross trajectory decoding", fontsize=8, pad=2)
    for ax, label in zip(
        (panel_a_axis, panel_b_axis, panel_c_axis),
        ("C", "D", "E"),
        strict=True,
    ):
        label_axis(ax, label, x=-0.02, y=1.01)

    fig.canvas.draw()
    shift_model_comparison_columns(
        fig,
        (panel_a_axis, panel_b_axis, panel_c_axis),
    )
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Supplementary Figure 1 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 1 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 1 per-data-set model comparisons."
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
        help=f"Directory containing moved Figure 1 assets. Default: {DEFAULT_ASSET_DIR}",
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
        "--encoding-place-bin-size-cm",
        type=float,
        default=ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
        help=(
            "Place-bin size used to find encoding-comparison summary files. "
            f"Default: {ENCODING_COMPARISON_PLACE_BIN_SIZE_CM}"
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
    """Run Supplementary Figure 1 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_1(
        data_root=args.data_root,
        asset_dir=args.asset_dir,
        output_path=output_path,
        datasets=datasets,
        encoding_place_bin_size_cm=args.encoding_place_bin_size_cm,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
