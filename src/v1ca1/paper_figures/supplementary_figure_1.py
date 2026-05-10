from __future__ import annotations

"""Generate Supplementary Figure 1 per-data-set model comparison panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, TRAJECTORY_TYPES
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DECODING_COMPARISON_REGION,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_1_WIDTH_MM,
    ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
    ENCODING_COMPARISON_REGION,
    MOTOR_DELTA_REGION,
    PANEL_E_TRAJECTORY_COLORS,
    STABILITY_AXIS_LABEL_FONTSIZE,
    STABILITY_LEGEND_FONTSIZE,
    STABILITY_REGIONS,
    STABILITY_REGION_COLORS,
    STABILITY_TICK_LABEL_FONTSIZE,
    load_decoding_absolute_error_table,
    load_dark_epoch_stability_table,
    load_encoding_delta_table,
    load_motor_delta_table,
    parse_dataset_id,
    plot_decoding_error_panel,
    plot_encoding_delta_panel,
    plot_motor_delta_panel,
)
from v1ca1.paper_figures.style import (
    HISTOGRAM_KWARGS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "supplementary_figure_1"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_1_WIDTH_MM
DEFAULT_STABILITY_ROW_HEIGHT_MM = 20.0
DEFAULT_DATASET_PANEL_HEIGHT_MM = 26.0
DEFAULT_SECTION_SPACER_MM = 5.0
MODEL_COMPARISON_GRID_WSPACE = -0.10
MODEL_COMPARISON_PANEL_C_SHIFT_PT = -10.0
MODEL_COMPARISON_PANEL_D_SHIFT_PT = -33.0
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
DATASET_STACK_ROW_GAP = 0.045
DATASET_STACK_LABELED_AXIS_BOUNDS = (0.16, 0.10, 0.82, 0.80)
DATASET_STACK_UNLABELED_AXIS_BOUNDS = (0.04, 0.10, 0.94, 0.80)
DATASET_LABEL_FONTSIZE = 4.8
STABILITY_ROW_LABEL_FONTSIZE = 5.0
STABILITY_ROW_AXIS_BOUNDS = (0.12, 0.07, 0.86, 0.82)
STABILITY_SUMMARY_FONTSIZE = 3.2


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
    """Tighten spacing between the B-D model-comparison columns."""
    fig_width_pt = float(fig.get_figwidth()) * 72.0
    if fig_width_pt <= 0.0:
        return
    shifts_pt = (
        0.0,
        MODEL_COMPARISON_PANEL_C_SHIFT_PT,
        MODEL_COMPARISON_PANEL_D_SHIFT_PT,
    )
    for ax, shift_pt in zip(axes, shifts_pt, strict=True):
        shift_axis_horizontally(ax, shift_pt / fig_width_pt)


def format_dataset_label(dataset: DatasetId) -> str:
    """Return a compact row label for one data set."""
    animal_name, date, epoch = normalize_dataset_id(dataset)
    return f"{animal_name}\n{date}\n{epoch}"


def make_fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return per-sample weights that sum one non-empty histogram to one."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def format_stability_summary(region: str, values: np.ndarray) -> str:
    """Return compact median and high-stability fraction text."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    label = str(region).upper()
    if values.size == 0:
        return f"{label} med n/a, >0.5 n/a"
    median = float(np.median(values))
    fraction_high = float(np.mean(values > 0.5))
    return f"{label} med {median:.2f}, >0.5 {fraction_high:.0%}"


def plot_stability_dataset_row(
    ax: Any,
    stability_table: Any,
    *,
    regions: Sequence[str] = STABILITY_REGIONS,
    show_xlabel: bool = True,
    show_legend: bool = False,
) -> None:
    """Plot Figure 1C-style tuning stability in one horizontal trajectory row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    bins = np.linspace(-1.0, 1.0, 25)
    legend_handles = []
    legend_labels = []
    cell_width = 0.225
    cell_gap = (1.0 - len(TRAJECTORY_TYPES) * cell_width) / (len(TRAJECTORY_TYPES) - 1)
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        x0 = trajectory_index * (cell_width + cell_gap)
        schematic_ax = ax.inset_axes([x0 + 0.078, 0.65, 0.070, 0.28])
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.5,
            trajectory_linewidth=0.75,
            arrow_mutation_scale=6.5,
            fill_track=True,
        )

        hist_ax = ax.inset_axes([x0, 0.08, cell_width, 0.52])
        trajectory_rows = stability_table[
            stability_table["trajectory_type"].astype(str) == trajectory_type
        ]
        values_by_region = {}
        for region in regions:
            values = np.asarray(
                trajectory_rows.loc[
                    trajectory_rows["region"].astype(str) == region,
                    "stability_correlation",
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            values_by_region[region] = values
            if values.size == 0:
                continue
            _counts, _edges, patches = hist_ax.hist(
                values,
                bins=bins,
                weights=make_fraction_histogram_weights(values),
                color=STABILITY_REGION_COLORS.get(region),
                **HISTOGRAM_KWARGS,
            )
            if trajectory_index == 0 and len(patches) > 0:
                legend_handles.append(patches[0])
                legend_labels.append(region.upper())

        for summary_index, region in enumerate(regions):
            hist_ax.text(
                0.03,
                0.96 - 0.16 * summary_index,
                format_stability_summary(region, values_by_region.get(region, np.asarray([]))),
                ha="left",
                va="top",
                fontsize=STABILITY_SUMMARY_FONTSIZE,
                color=STABILITY_REGION_COLORS.get(region, "black"),
                transform=hist_ax.transAxes,
            )

        hist_ax.set_xlim(-1.0, 1.0)
        hist_ax.set_ylim(bottom=0.0)
        hist_ax.spines["top"].set_visible(False)
        hist_ax.spines["right"].set_visible(False)
        hist_ax.tick_params(
            labelsize=STABILITY_TICK_LABEL_FONTSIZE,
            length=2,
            pad=1,
        )
        if show_xlabel:
            hist_ax.set_xlabel(
                "Odd/even corr.",
                fontsize=STABILITY_AXIS_LABEL_FONTSIZE,
                labelpad=1,
            )
        else:
            hist_ax.set_xticklabels([])
        if trajectory_index == 0:
            hist_ax.set_ylabel(
                "Frac.",
                fontsize=STABILITY_AXIS_LABEL_FONTSIZE,
                labelpad=1,
            )
        else:
            hist_ax.set_yticklabels([])

    if show_legend and legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=STABILITY_LEGEND_FONTSIZE,
            handlelength=1.0,
            borderpad=0.1,
            labelspacing=0.2,
        )


def plot_stability_dataset_rows_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str] = STABILITY_REGIONS,
) -> list[Any]:
    """Plot one horizontal Figure 1C-style stability row per data set."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not datasets:
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        return []

    n_datasets = len(datasets)
    row_gap = DATASET_STACK_ROW_GAP
    slot_height = (1.0 - row_gap * (n_datasets - 1)) / n_datasets
    row_axes = []
    for dataset_index, dataset in enumerate(datasets):
        row_bottom = 1.0 - (dataset_index + 1) * slot_height - dataset_index * row_gap
        plot_x, plot_y, plot_width, plot_height = STABILITY_ROW_AXIS_BOUNDS
        row_ax = ax.inset_axes(
            [
                plot_x,
                row_bottom + plot_y * slot_height,
                plot_width,
                plot_height * slot_height,
            ]
        )
        stability_table = load_dark_epoch_stability_table(
            data_root=data_root,
            datasets=[dataset],
            regions=regions,
        )
        plot_stability_dataset_row(
            row_ax,
            stability_table,
            regions=regions,
            show_xlabel=dataset_index == n_datasets - 1,
            show_legend=dataset_index == 0,
        )
        ax.text(
            0.01,
            row_bottom + 0.5 * slot_height,
            format_dataset_label(dataset),
            ha="left",
            va="center",
            fontsize=STABILITY_ROW_LABEL_FONTSIZE,
            transform=ax.transAxes,
        )
        row_axes.append(row_ax)
    return row_axes


def plot_dataset_stack_panel(
    ax: Any,
    *,
    datasets: Sequence[DatasetId],
    plot_dataset: Any,
    axis_bounds: tuple[float, float, float, float] | None = None,
    show_dataset_labels: bool = True,
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
                0.01,
                row_bottom + 0.5 * slot_height,
                format_dataset_label(dataset),
                ha="left",
                va="center",
                fontsize=DATASET_LABEL_FONTSIZE,
                transform=ax.transAxes,
            )
        row_axes.append(row_ax)
    return row_axes


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

    return plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
        show_dataset_labels=show_dataset_labels,
    )


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

    return plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
        show_dataset_labels=show_dataset_labels,
    )


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

    return plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
        show_dataset_labels=show_dataset_labels,
    )


def make_supplementary_figure_1(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    encoding_place_bin_size_cm: float,
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 1."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    stability_row_height_mm = DEFAULT_STABILITY_ROW_HEIGHT_MM * max(len(datasets), 1)
    model_row_height_mm = DEFAULT_DATASET_PANEL_HEIGHT_MM * max(len(datasets), 1)
    fig_height_mm = (
        stability_row_height_mm
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
            stability_row_height_mm,
            DEFAULT_SECTION_SPACER_MM,
            model_row_height_mm,
        ],
    )
    stability_axis = fig.add_subplot(outer_grid[0])
    plot_stability_dataset_rows_panel(
        stability_axis,
        data_root=data_root,
        datasets=datasets,
    )
    stability_axis.set_title("Tuning stability", fontsize=8, pad=2)
    label_axis(stability_axis, "A", x=-0.02, y=1.01)

    spacer_axis = fig.add_subplot(outer_grid[1])
    spacer_axis.axis("off")

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
        ("B", "C", "D"),
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
        output_path=output_path,
        datasets=datasets,
        encoding_place_bin_size_cm=args.encoding_place_bin_size_cm,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
