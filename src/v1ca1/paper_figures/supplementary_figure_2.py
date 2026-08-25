from __future__ import annotations

"""Generate Figure 1 panels E--G separately for each animal."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures import figure_1
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


DEFAULT_OUTPUT_DIR = figure_1.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "supplementary_figure_2"
DEFAULT_OUTPUT_FORMAT = figure_1.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = figure_1.FIGURE_FORMATS
DEFAULT_FIGURE_WIDTH_MM = figure_1.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_ANIMAL_ROW_HEIGHT_MM = 40.0
ANIMAL_ROW_LABEL_FONTSIZE = 6.0
PANEL_TITLE_FONTSIZE = 8.0
PANEL_LABELS = ("A", "B", "C")
PANEL_TITLES = (
    "Comparison to motor",
    "Comparison to alternative codes",
    figure_1.PANEL_G_TITLE,
)
PANEL_PLOT_BOUNDS = (0.14, 0.34, 0.78, 0.58)
PANEL_GRID_WSPACE = 0.05
PANEL_GRID_HSPACE = 0.10
PANEL_GRID_LEFT = 0.13
PANEL_GRID_RIGHT = 0.985
PANEL_GRID_TOP = 0.96
PANEL_GRID_BOTTOM = 0.03


def normalize_individual_animal_datasets(
    datasets: Sequence[DatasetId],
) -> list[DatasetId]:
    """Normalize data sets and require one session for each animal row."""
    normalized = [normalize_dataset_id(dataset) for dataset in datasets]
    animal_names = [animal_name for animal_name, _date, _epoch in normalized]
    if len(set(animal_names)) != len(animal_names):
        raise ValueError(
            "Supplementary Figure 2 requires exactly one data set per animal; "
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
    encoding_bin_size_s: float = figure_1.ENCODING_COMPARISON_BIN_SIZE_S,
    encoding_place_bin_size_cm: float = (
        figure_1.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
    ),
    decoding_n_permutations: int = figure_1.DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = figure_1.DECODING_PERMUTATION_SEED,
) -> dict[str, Any]:
    """Load Figure 1 E--G data and per-animal decoding inference."""
    if decoding_n_permutations <= 0:
        raise ValueError("decoding_n_permutations must be positive.")
    if decoding_permutation_seed < 0:
        raise ValueError("decoding_permutation_seed must be non-negative.")

    dataset_ids = tuple(normalize_individual_animal_datasets(datasets))
    motor_delta_table = figure_1.load_motor_delta_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=figure_1.MOTOR_DELTA_REGION,
    )
    encoding_delta_table = figure_1.load_encoding_delta_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=figure_1.ENCODING_COMPARISON_REGION,
        bin_size_s=encoding_bin_size_s,
        place_bin_size_cm=encoding_place_bin_size_cm,
    )
    decoding_error_table = figure_1.load_decoding_absolute_error_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=figure_1.DECODING_COMPARISON_REGION,
    )
    decoding_trial_error_table = figure_1.build_decoding_trial_error_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=figure_1.DECODING_COMPARISON_REGION,
    )
    decoding_permutation_results = figure_1.compute_decoding_permutation_tests(
        decoding_trial_error_table,
        n_permutations=decoding_n_permutations,
        seed=decoding_permutation_seed,
    )
    return {
        "motor_delta": motor_delta_table,
        "encoding_delta": encoding_delta_table,
        "decoding_error": decoding_error_table,
        "decoding_permutation": decoding_permutation_results,
    }


def make_supplementary_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    dpi: int,
    encoding_bin_size_s: float = figure_1.ENCODING_COMPARISON_BIN_SIZE_S,
    encoding_place_bin_size_cm: float = (
        figure_1.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
    ),
    decoding_n_permutations: int = figure_1.DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = figure_1.DECODING_PERMUTATION_SEED,
) -> Path:
    """Build and save Figure 1 panels E--G with one animal per row."""
    import matplotlib.pyplot as plt

    dataset_ids = normalize_individual_animal_datasets(datasets)
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
        print(f"Saved Supplementary Figure 2 to {output_path}")
        return output_path

    panel_data = load_individual_animal_panel_data(
        data_root=data_root,
        datasets=dataset_ids,
        encoding_bin_size_s=encoding_bin_size_s,
        encoding_place_bin_size_cm=encoding_place_bin_size_cm,
        decoding_n_permutations=decoding_n_permutations,
        decoding_permutation_seed=decoding_permutation_seed,
    )
    grid = fig.add_gridspec(
        nrows=len(dataset_ids),
        ncols=3,
        wspace=PANEL_GRID_WSPACE,
        hspace=PANEL_GRID_HSPACE,
        left=PANEL_GRID_LEFT,
        right=PANEL_GRID_RIGHT,
        top=PANEL_GRID_TOP,
        bottom=PANEL_GRID_BOTTOM,
    )

    for row_index, dataset in enumerate(dataset_ids):
        animal_name, _date, _epoch = dataset
        outer_axes = [fig.add_subplot(grid[row_index, col]) for col in range(3)]
        for outer_ax in outer_axes:
            outer_ax.axis("off")
        plot_axes = [outer_ax.inset_axes(PANEL_PLOT_BOUNDS) for outer_ax in outer_axes]

        motor_rows = filter_table_by_animal(panel_data["motor_delta"], animal_name)
        encoding_rows = filter_table_by_animal(
            panel_data["encoding_delta"],
            animal_name,
        )
        decoding_rows = filter_table_by_animal(
            panel_data["decoding_error"],
            animal_name,
        )
        permutation_rows = filter_table_by_animal(
            panel_data["decoding_permutation"],
            animal_name,
        )
        decoding_brackets = figure_1.build_decoding_significance_brackets(
            permutation_rows,
            animal_names=(animal_name,),
        )

        figure_1.plot_motor_delta_panel(plot_axes[0], motor_rows)
        figure_1.plot_encoding_delta_panel(plot_axes[1], encoding_rows)
        figure_1.plot_decoding_error_panel(
            plot_axes[2],
            decoding_rows,
            significance_brackets=decoding_brackets,
        )
        outer_axes[0].text(
            -0.075,
            0.50,
            format_animal_row_label(dataset),
            ha="right",
            va="center",
            fontsize=ANIMAL_ROW_LABEL_FONTSIZE,
            color="0.25",
            transform=outer_axes[0].transAxes,
        )

        if row_index == 0:
            for outer_ax, panel_label, panel_title in zip(
                outer_axes,
                PANEL_LABELS,
                PANEL_TITLES,
                strict=True,
            ):
                outer_ax.set_title(
                    panel_title,
                    fontsize=PANEL_TITLE_FONTSIZE,
                    pad=2,
                )
                label_axis(outer_ax, panel_label, x=-0.035, y=1.02)

    figure_1.raise_figure_text_to_minimum_fontsize(
        fig,
        figure_1.MIN_FIGURE_1_FONTSIZE_PT,
    )
    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 2."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 panels E--G separately for each animal."
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
        type=figure_1.parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date[:dark_epoch]. "
            "May be repeated. Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--encoding-bin-size-s",
        type=float,
        default=figure_1.ENCODING_COMPARISON_BIN_SIZE_S,
        help=(
            "Time-bin size used to find encoding-comparison summary files. "
            f"Default: {figure_1.ENCODING_COMPARISON_BIN_SIZE_S}"
        ),
    )
    parser.add_argument(
        "--encoding-place-bin-size-cm",
        type=float,
        default=figure_1.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
        help=(
            "Place-bin size used to find encoding-comparison summary files. "
            f"Default: {figure_1.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--decoding-n-permutations",
        type=int,
        default=figure_1.DECODING_PERMUTATION_COUNT,
        help=(
            "Label permutations used for per-animal decoding inference. "
            f"Default: {figure_1.DECODING_PERMUTATION_COUNT}"
        ),
    )
    parser.add_argument(
        "--decoding-permutation-seed",
        type=int,
        default=figure_1.DECODING_PERMUTATION_SEED,
        help=(
            "Random seed used for per-animal decoding inference. "
            f"Default: {figure_1.DECODING_PERMUTATION_SEED}"
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
    """Run Supplementary Figure 2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = figure_1.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        dpi=args.dpi,
        encoding_bin_size_s=args.encoding_bin_size_s,
        encoding_place_bin_size_cm=args.encoding_place_bin_size_cm,
        decoding_n_permutations=args.decoding_n_permutations,
        decoding_permutation_seed=args.decoding_permutation_seed,
    )


if __name__ == "__main__":
    main()
