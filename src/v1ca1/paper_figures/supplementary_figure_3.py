"""Generate data-only Figure 2B--C summaries separately for each animal."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures import figure_2
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_DIR = figure_2.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "supplementary_figure_3"
DEFAULT_OUTPUT_FORMAT = figure_2.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = figure_2.FIGURE_FORMATS
DEFAULT_REGION = figure_2.DEFAULT_REGIONS[0]
DEFAULT_FIGURE_WIDTH_MM = figure_2.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_ANIMAL_ROW_HEIGHT_MM = 40.0
ANIMAL_ROW_LABEL_FONTSIZE = 6.0
PANEL_TITLE_FONTSIZE = 8.0
PANEL_LABELS = ("A", "B")
PANEL_TITLES = (
    "Tuning shift across dark and light",
    "Path-invariance across dark and light",
)
PANEL_A_DATA_BOUNDS = (0.12, 0.15, 0.82, 0.73)
PANEL_B_DATA_BOUNDS = (0.02, 0.04, 0.96, 0.90)
PANEL_GRID_WSPACE = 0.08
PANEL_GRID_HSPACE = 0.10
PANEL_GRID_LEFT = 0.10
PANEL_GRID_RIGHT = 0.985
PANEL_GRID_TOP = 0.96
PANEL_GRID_BOTTOM = 0.03
PATH_INVARIANCE_MARGINAL_FRACTION_LIMIT = 0.21


def normalize_individual_animal_datasets(
    datasets: Sequence[DatasetId],
) -> list[DatasetId]:
    """Normalize data sets and require one session for each animal row."""
    normalized = [normalize_dataset_id(dataset) for dataset in datasets]
    animal_names = [animal_name for animal_name, _date, _epoch in normalized]
    if len(set(animal_names)) != len(animal_names):
        raise ValueError(
            "Supplementary Figure 3 requires exactly one data set per animal; "
            f"received {normalized!r}."
        )
    return normalized


def format_animal_row_label(dataset: DatasetId) -> str:
    """Return the animal name shown beside one figure row."""
    animal_name, _date, _epoch = normalize_dataset_id(dataset)
    return animal_name


def get_figure_height_mm(n_animal_rows: int) -> float:
    """Return the figure height for the requested number of animal rows."""
    return DEFAULT_ANIMAL_ROW_HEIGHT_MM * max(int(n_animal_rows), 1)


def filter_table_by_animal(table: Any, animal_name: str) -> Any:
    """Return rows belonging to one animal without changing table columns."""
    if "animal_name" not in table.columns:
        raise ValueError("Panel table is missing required column 'animal_name'.")
    return table.loc[table["animal_name"].astype(str) == str(animal_name)].copy()


def load_individual_animal_panel_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    panel_tuning_similarity_cache_dir: Path | None,
    refresh_panel_tuning_similarity_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        figure_2.DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        figure_2.DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> dict[str, Any]:
    """Load the exact quantitative tables used by Figure 2B and 2C."""
    dataset_ids = tuple(normalize_individual_animal_datasets(datasets))
    shift_profile_table = figure_2.load_or_compute_panel_h_shift_profile_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=panel_tuning_similarity_cache_dir,
        refresh_cache=refresh_panel_tuning_similarity_cache,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=(
            min_path_movement_firing_rate_hz
        ),
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )

    legacy_figure_2 = figure_2._figure_2._figure_2
    overlap_table = legacy_figure_2.load_panel_b_tuning_overlap_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    overlap_table = legacy_figure_2.filter_panel_b_overlap_by_even_odd_stability(
        overlap_table,
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_movement_firing_rate_hz=(
            legacy_figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        min_stability_correlation=(
            legacy_figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    )
    return {
        "shift_profile": shift_profile_table,
        "path_invariance": overlap_table,
    }


def plot_path_invariance_data(ax: Any, table: Any) -> None:
    """Plot Figure 2C's scatter and marginals without its schematic."""
    legacy_figure_2 = figure_2._figure_2._figure_2
    legacy_figure_2.plot_panel_b_dpp_overlap_scatter(
        ax,
        table,
        title=None,
        show_linear_fit=True,
        show_r2_annotation=True,
        equal_aspect=True,
    )
    legacy_figure_2._format_panel_b_dppi_scatter_axes(ax)
    for child_ax in ax.child_axes:
        if child_ax.get_ylabel() == "Frac.":
            child_ax.set_ylim(0.0, PATH_INVARIANCE_MARGINAL_FRACTION_LIMIT)
            child_ax.set_yticks((0.0, 0.1, 0.2), labels=("0", "0.1", "0.2"))
        if child_ax.get_xlabel() == "Frac.":
            child_ax.set_xlim(0.0, PATH_INVARIANCE_MARGINAL_FRACTION_LIMIT)
            child_ax.set_xticks((0.0, 0.1, 0.2), labels=("0", "0.1", "0.2"))
    for old_text, new_text in (
        ("Dark DPPI", "Dark PII"),
        ("Light DPPI", "Light PII"),
    ):
        legacy_figure_2._replace_nested_text(
            ax,
            old_text,
            new_text,
            fontsize=legacy_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
        )


def make_supplementary_figure_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    panel_tuning_similarity_cache_dir: Path | None = None,
    refresh_panel_tuning_similarity_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        figure_2.DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        figure_2.DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Path:
    """Build and save data-only Figure 2B--C panels by animal."""
    import matplotlib.pyplot as plt

    dataset_ids = normalize_individual_animal_datasets(datasets)
    panel_tuning_similarity_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_tuning_similarity_cache_dir is None
        else Path(panel_tuning_similarity_cache_dir)
    )
    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            get_figure_height_mm(len(dataset_ids)),
        ),
        constrained_layout=False,
    )
    if not dataset_ids:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        print(f"Saved Supplementary Figure 3 to {output_path}")
        return output_path

    panel_data = load_individual_animal_panel_data(
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        panel_tuning_similarity_cache_dir=(
            panel_tuning_similarity_cache_dir
        ),
        refresh_panel_tuning_similarity_cache=(
            refresh_panel_tuning_similarity_cache
        ),
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=(
            min_path_movement_firing_rate_hz
        ),
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    grid = fig.add_gridspec(
        nrows=len(dataset_ids),
        ncols=2,
        wspace=PANEL_GRID_WSPACE,
        hspace=PANEL_GRID_HSPACE,
        left=PANEL_GRID_LEFT,
        right=PANEL_GRID_RIGHT,
        top=PANEL_GRID_TOP,
        bottom=PANEL_GRID_BOTTOM,
    )
    panel_b_outer_axes = []

    for row_index, dataset in enumerate(dataset_ids):
        animal_name, _date, _epoch = dataset
        panel_a_outer_ax = fig.add_subplot(grid[row_index, 0])
        panel_b_outer_ax = fig.add_subplot(grid[row_index, 1])
        panel_a_outer_ax.axis("off")
        panel_b_outer_ax.axis("off")
        panel_b_outer_axes.append(panel_b_outer_ax)

        panel_a_data_ax = panel_a_outer_ax.inset_axes(PANEL_A_DATA_BOUNDS)
        panel_b_data_ax = panel_b_outer_ax.inset_axes(PANEL_B_DATA_BOUNDS)
        shift_profile_rows = filter_table_by_animal(
            panel_data["shift_profile"],
            animal_name,
        )
        path_invariance_rows = filter_table_by_animal(
            panel_data["path_invariance"],
            animal_name,
        )
        figure_2.plot_population_shift_profile(
            panel_a_data_ax,
            shift_profile_rows,
        )
        plot_path_invariance_data(panel_b_data_ax, path_invariance_rows)

        panel_a_outer_ax.text(
            -0.055,
            0.50,
            format_animal_row_label(dataset),
            ha="right",
            va="center",
            fontsize=ANIMAL_ROW_LABEL_FONTSIZE,
            color="0.25",
            transform=panel_a_outer_ax.transAxes,
        )
        if row_index == 0:
            for outer_ax, panel_label, panel_title in zip(
                (panel_a_outer_ax, panel_b_outer_ax),
                PANEL_LABELS,
                PANEL_TITLES,
                strict=True,
            ):
                outer_ax.set_title(
                    panel_title,
                    fontsize=PANEL_TITLE_FONTSIZE,
                    pad=2,
                )
                label_axis(outer_ax, panel_label, x=-0.025, y=1.02)

    figure_2._figure_2._raise_text_to_minimum_fontsize(
        fig,
        figure_2._figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    fig.canvas.draw()
    for panel_b_outer_ax in panel_b_outer_axes:
        figure_2._figure_2._align_panel_b_top_histogram_label_to_scatter(
            fig,
            panel_b_outer_ax,
        )
    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 3."""
    parser = argparse.ArgumentParser(
        description="Generate data-only Figure 2B--C panels by animal."
    )
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
        type=figure_2._figure_2.parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date[:dark_epoch]. "
            "May be repeated. Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default=DEFAULT_REGION,
        help=f"Region to include. Default: {DEFAULT_REGION}.",
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--panel-tuning-similarity-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for Figure 2 tuning-similarity caches. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-tuning-similarity-cache",
        action="store_true",
        help="Recompute Figure 2 tuning-shift values and overwrite matching caches.",
    )
    parser.add_argument(
        "--min-epoch-movement-firing-rate-hz",
        type=float,
        default=figure_2.DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ,
        help=(
            "Minimum whole-epoch movement firing rate in both conditions. "
            f"Default: {figure_2.DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-path-movement-firing-rate-hz",
        type=float,
        default=(
            figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        help=(
            "Minimum path movement firing rate in both conditions. "
            "Default: 0.5"
        ),
    )
    parser.add_argument(
        "--min-segment-mean-firing-rate-hz",
        type=float,
        default=figure_2.DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ,
        help=(
            "Minimum segment mean firing rate in both conditions. "
            f"Default: {figure_2.DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-stability-correlation",
        type=float,
        default=(
            figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        help="Minimum odd/even stability in both conditions. Default: 0.5",
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
    output_path = figure_2.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_tuning_similarity_cache_dir = (
        args.panel_tuning_similarity_cache_dir
        if args.panel_tuning_similarity_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_supplementary_figure_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
        panel_tuning_similarity_cache_dir=(
            panel_tuning_similarity_cache_dir
        ),
        refresh_panel_tuning_similarity_cache=(
            args.refresh_panel_tuning_similarity_cache
        ),
        min_epoch_movement_firing_rate_hz=(
            args.min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=(
            args.min_path_movement_firing_rate_hz
        ),
        min_segment_mean_firing_rate_hz=(
            args.min_segment_mean_firing_rate_hz
        ),
        min_stability_correlation=args.min_stability_correlation,
    )


if __name__ == "__main__":
    main()
