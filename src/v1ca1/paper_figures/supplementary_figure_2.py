"""Generate Supplementary Figure 2 from two existing summary panels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures import figure_2_old as figure_2
from v1ca1.paper_figures import supplementary_figure_3
from v1ca1.paper_figures._dark_light import load_panel_e_decoding_error_table
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
DEFAULT_OUTPUT_NAME = "supplementary_figure_2"
DEFAULT_OUTPUT_FORMAT = figure_2.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = figure_2.FIGURE_FORMATS
DEFAULT_REGION = figure_2.DEFAULT_REGIONS[0]
DEFAULT_FIGURE_WIDTH_MM = figure_2.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = figure_2.PANEL_D_ROW_HEIGHT_MM
PANEL_WIDTH_RATIOS = figure_2.PANEL_D2E2_ROW_WIDTH_RATIOS
PANEL_WSPACE = figure_2.PANEL_D2E2_ROW_WSPACE
DECODING_PANEL_TITLE = "Dark and light decoding comparison"
PANEL_TITLES = (
    DECODING_PANEL_TITLE,
    supplementary_figure_3.PANEL_A_CV_PCA_TITLE,
)


def load_panel_a_decoding_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    decoding_n_permutations: int = figure_2.DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = figure_2.DECODING_PERMUTATION_SEED,
) -> dict[str, Any]:
    """Load decoding data for Supplementary Figure 2 Panel A."""
    if decoding_n_permutations <= 0:
        raise ValueError("decoding_n_permutations must be positive.")
    if decoding_permutation_seed < 0:
        raise ValueError("decoding_permutation_seed must be non-negative.")

    dataset_ids = tuple(datasets)
    normalized_datasets = [
        normalize_dataset_id(dataset) for dataset in dataset_ids
    ]
    decoding_animal_names = tuple(
        animal_name for animal_name, _date, _epoch in normalized_datasets
    )
    if (
        not decoding_animal_names
        or len(set(decoding_animal_names)) != len(decoding_animal_names)
    ):
        raise ValueError(
            "Supplementary Figure 2 decoding inference requires exactly one "
            "data set per animal; "
            f"received {normalized_datasets!r}."
        )

    decoding_error_table = load_panel_e_decoding_error_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    decoding_trial_error_table = figure_2.build_panel_e_decoding_trial_error_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    permutation_results = figure_2.compute_panel_e_decoding_permutation_tests(
        decoding_trial_error_table,
        n_permutations=decoding_n_permutations,
        seed=decoding_permutation_seed,
    )
    significance_labels = figure_2.build_panel_e_decoding_significance_labels(
        permutation_results,
        animal_names=decoding_animal_names,
    )
    return {
        "decoding_error": decoding_error_table,
        "decoding_significance_labels": significance_labels,
    }


def make_supplementary_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    decoding_n_permutations: int = figure_2.DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = figure_2.DECODING_PERMUTATION_SEED,
) -> Path:
    """Build and save Supplementary Figure 2 panels A and B."""
    import matplotlib.pyplot as plt

    dataset_ids = tuple(datasets)
    panel_a_data = load_panel_a_decoding_data(
        data_root=data_root,
        datasets=dataset_ids,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        decoding_n_permutations=decoding_n_permutations,
        decoding_permutation_seed=decoding_permutation_seed,
    )
    panel_b_table = (
        supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table(
            data_root=data_root,
            datasets=dataset_ids,
        )
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        **figure_2.CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS
    )
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_WIDTH_RATIOS,
        wspace=PANEL_WSPACE,
    )
    panel_a_axis = fig.add_subplot(grid[0, 0])
    panel_b_axis = fig.add_subplot(grid[0, 1])

    figure_2.plot_panel_e2_decoding_panel(
        panel_a_axis,
        panel_a_data["decoding_error"],
        significance_labels=panel_a_data["decoding_significance_labels"],
    )
    panel_a_title = panel_a_axis.set_title(
        PANEL_TITLES[0],
        fontsize=8,
        pad=figure_2.PANEL_BC_TITLE_PAD,
    )
    label_axis(
        panel_a_axis,
        "A",
        x=-0.035,
        y=figure_2.PANEL_BC_LABEL_Y,
    )
    panel_a_label = panel_a_axis.texts[-1]
    figure_2._raise_text_to_minimum_fontsize(
        fig,
        figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )

    supplementary_figure_3.plot_panel_a_cv_pca_participation_ratios(
        panel_b_axis,
        panel_b_table,
    )
    panel_b_title = panel_b_axis.set_title(
        PANEL_TITLES[1],
        fontsize=supplementary_figure_3.PANEL_TITLE_FONTSIZE,
        pad=2,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=1.05)
    panel_b_label = panel_b_axis.texts[-1]

    fig.canvas.draw()
    fig.set_layout_engine(None)
    header_texts = (
        panel_a_title,
        panel_a_label,
        panel_b_title,
        panel_b_label,
    )
    figure_2._align_texts_to_reference_display_y(header_texts)
    figure_2._align_text_tops_to_reference_display_y(fig, header_texts)

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 2."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 2 panels A and B."
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
        type=figure_2.parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date[:dark_epoch]. "
            "May be repeated. Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default=DEFAULT_REGION,
        help=f"Decoding region. Default: {DEFAULT_REGION}",
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--decoding-n-permutations",
        type=int,
        default=figure_2.DECODING_PERMUTATION_COUNT,
        help=(
            "Label permutations used for decoding inference. "
            f"Default: {figure_2.DECODING_PERMUTATION_COUNT}"
        ),
    )
    parser.add_argument(
        "--decoding-permutation-seed",
        type=int,
        default=figure_2.DECODING_PERMUTATION_SEED,
        help=(
            "Random seed used for decoding inference. "
            f"Default: {figure_2.DECODING_PERMUTATION_SEED}"
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
    datasets = (
        args.dataset if args.dataset is not None else get_processed_datasets()
    )
    output_path = figure_2.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
        decoding_n_permutations=args.decoding_n_permutations,
        decoding_permutation_seed=args.decoding_permutation_seed,
    )


if __name__ == "__main__":
    main()
