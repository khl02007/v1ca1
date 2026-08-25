"""Generate pooled and per-animal Supplementary Figure 4 summaries."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures import _figure_2_panels as figure_2
from v1ca1.paper_figures import supplementary_figure_5
from v1ca1.paper_figures._dark_light import (
    PANEL_E_NORM_ERROR_YLIM,
    PANEL_E_PLACE_ERROR_YLIM,
    load_panel_e_decoding_error_table,
)
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
DEFAULT_OUTPUT_NAME = "supplementary_figure_4"
DEFAULT_OUTPUT_FORMAT = figure_2.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = figure_2.FIGURE_FORMATS
DEFAULT_REGION = figure_2.DEFAULT_REGIONS[0]
DEFAULT_FIGURE_WIDTH_MM = figure_2.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_FIGURE_ROW_HEIGHT_MM = figure_2.PANEL_D_ROW_HEIGHT_MM
DEFAULT_ANIMAL_COUNT = 4
PANEL_C_ANIMALS_PER_ROW = 2
DEFAULT_FIGURE_HEIGHT_MM = (
    1.0 + math.ceil(DEFAULT_ANIMAL_COUNT / PANEL_C_ANIMALS_PER_ROW)
) * DEFAULT_FIGURE_ROW_HEIGHT_MM
PANEL_WIDTH_RATIOS = figure_2.PANEL_D2E2_ROW_WIDTH_RATIOS
PANEL_WSPACE = figure_2.PANEL_D2E2_ROW_WSPACE
PANEL_HSPACE = 0.10
PANEL_C_ROW_HSPACE = 0.08
DECODING_PANEL_TITLE = "Dark and light decoding comparison"
INDIVIDUAL_DECODING_PANEL_TITLE = (
    "Individual-animal dark and light decoding comparison"
)
PANEL_TITLES = (
    DECODING_PANEL_TITLE,
    supplementary_figure_5.PANEL_A_CV_PCA_TITLE,
    INDIVIDUAL_DECODING_PANEL_TITLE,
)
PANEL_C_ANIMAL_LABEL_X = -0.005
PANEL_C_ANIMAL_LABEL_FONTSIZE = 6.0
PANEL_C_YLIM_PADDING = 1.08
PANEL_C_FINE_YLIM_STEP = 0.05
PANEL_C_STANDARD_YLIM_STEP = 0.1
PANEL_C_WIDE_YLIM_STEP = 0.5
PANEL_C_PLACE_YLIMS_BY_ANIMAL = {
    "L12": (0.0, 0.2),
    "L14": (0.0, 0.2),
    "L15": (0.0, 0.2),
    "L19": (0.0, 1.5),
}


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
    """Load decoding data for Supplementary Figure 4 Panel A."""
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
            "Supplementary Figure 4 decoding inference requires exactly one "
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
    individual_decoding_tables = []
    for dataset, normalized_dataset in zip(
        dataset_ids,
        normalized_datasets,
        strict=True,
    ):
        animal_name, date, _epoch = normalized_dataset
        animal_table = load_panel_e_decoding_error_table(
            data_root=data_root,
            datasets=(dataset,),
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ).copy()
        animal_table.loc[:, "animal_name"] = str(animal_name)
        animal_table.loc[:, "date"] = str(date)
        individual_decoding_tables.append(animal_table)

    import pandas as pd

    individual_decoding_error_table = pd.concat(
        individual_decoding_tables,
        ignore_index=True,
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
    result_animal_names = permutation_results["animal_name"].astype(str)
    individual_significance_labels = {
        str(animal_name): figure_2.build_panel_e_decoding_significance_labels(
            permutation_results.loc[
                result_animal_names == str(animal_name)
            ].copy(),
            animal_names=(str(animal_name),),
        )
        for animal_name in decoding_animal_names
    }
    return {
        "decoding_error": decoding_error_table,
        "individual_decoding_error": individual_decoding_error_table,
        "decoding_significance_labels": significance_labels,
        "individual_decoding_significance_labels": (
            individual_significance_labels
        ),
    }


def get_panel_c_row_count(animal_count: int) -> int:
    """Return the rows needed to display two Panel C animals per row."""
    return math.ceil(
        max(int(animal_count), 0) / PANEL_C_ANIMALS_PER_ROW
    )


def get_figure_height_mm(animal_count: int) -> float:
    """Return one top-row height plus the required Panel C rows."""
    return DEFAULT_FIGURE_ROW_HEIGHT_MM * (
        1.0 + get_panel_c_row_count(animal_count)
    )


def _center_axis_title_over_axes(title: Any, axes: Sequence[Any]) -> None:
    """Center one axis title over the horizontal extent of several axes."""
    if not axes:
        return
    anchor_box = title.axes.get_position()
    target_center = 0.5 * (
        min(axis.get_position().x0 for axis in axes)
        + max(axis.get_position().x1 for axis in axes)
    )
    title.set_x((target_center - anchor_box.x0) / anchor_box.width)


def filter_decoding_table_by_animal(table: Any, animal_name: str) -> Any:
    """Return the decoding summary rows for one animal."""
    if "animal_name" not in table.columns:
        raise ValueError("Decoding summary is missing column 'animal_name'.")
    return table.loc[
        table["animal_name"].astype(str) == str(animal_name)
    ].copy()


def _round_panel_c_ylim_upper(value: float) -> float:
    """Round a data-derived Panel C upper limit to a readable value."""
    if value <= 0.6:
        step = PANEL_C_FINE_YLIM_STEP
    elif value <= 3.0:
        step = PANEL_C_STANDARD_YLIM_STEP
    else:
        step = PANEL_C_WIDE_YLIM_STEP
    return round(math.ceil((value - 1e-12) / step) * step, 10)


def _get_panel_c_analysis_ylim(
    table: Any,
    analysis: str,
    default_ylim: tuple[float, float],
) -> tuple[float, float]:
    """Return a linear limit that contains one animal's median and IQR."""
    rows = table.loc[table["analysis"].astype(str) == analysis]
    if analysis == "cross_trajectory" and figure_2.PANEL_E_CROSS_COMPARISONS:
        comparison = figure_2.PANEL_E_CROSS_COMPARISONS[0][0]
        rows = rows.loc[rows["comparison"].astype(str) == comparison]
    values = rows[["q25_error", "median_error", "q75_error"]].to_numpy(
        dtype=float,
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return default_ylim
    padded_upper = max(
        float(default_ylim[1]),
        max(0.0, float(np.max(finite_values))) * PANEL_C_YLIM_PADDING,
    )
    return float(default_ylim[0]), _round_panel_c_ylim_upper(padded_upper)


def get_panel_c_decoding_ylims(
    table: Any,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return row-specific cross-path and place-decoding limits."""
    cross_ylim = _get_panel_c_analysis_ylim(
        table,
        "cross_trajectory",
        PANEL_E_NORM_ERROR_YLIM,
    )
    place_ylim = _get_panel_c_analysis_ylim(
        table,
        "place",
        PANEL_E_PLACE_ERROR_YLIM,
    )
    animal_names = table["animal_name"].dropna().astype(str).unique()
    if len(animal_names) == 1:
        place_ylim = PANEL_C_PLACE_YLIMS_BY_ANIMAL.get(
            str(animal_names[0]),
            place_ylim,
        )
    return cross_ylim, place_ylim


def make_supplementary_figure_4(
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
    """Build and save Supplementary Figure 4 panels A--C."""
    import matplotlib.pyplot as plt

    dataset_ids = tuple(datasets)
    normalized_datasets = tuple(
        normalize_dataset_id(dataset) for dataset in dataset_ids
    )
    decoding_animal_names = tuple(
        str(animal_name)
        for animal_name, _date, _epoch in normalized_datasets
    )
    panel_c_row_count = get_panel_c_row_count(len(decoding_animal_names))
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
        supplementary_figure_5.load_panel_a_cv_pca_participation_ratio_table(
            data_root=data_root,
            datasets=dataset_ids,
        )
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            get_figure_height_mm(len(decoding_animal_names)),
        ),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        **figure_2.CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=(1.0, float(panel_c_row_count)),
        hspace=PANEL_HSPACE,
    )
    top_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_WIDTH_RATIOS,
        wspace=PANEL_WSPACE,
    )
    panel_a_axis = fig.add_subplot(top_grid[0, 0])
    panel_b_axis = fig.add_subplot(top_grid[0, 1])
    panel_c_grid = outer_grid[1, 0].subgridspec(
        nrows=panel_c_row_count,
        ncols=PANEL_C_ANIMALS_PER_ROW,
        width_ratios=PANEL_WIDTH_RATIOS,
        wspace=PANEL_WSPACE,
        hspace=PANEL_C_ROW_HSPACE,
    )

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

    supplementary_figure_5.plot_panel_a_cv_pca_participation_ratios(
        panel_b_axis,
        panel_b_table,
    )
    panel_b_title = panel_b_axis.set_title(
        PANEL_TITLES[1],
        fontsize=supplementary_figure_5.PANEL_TITLE_FONTSIZE,
        pad=2,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=1.05)
    panel_b_label = panel_b_axis.texts[-1]

    panel_c_title = None
    panel_c_label = None
    individual_table = panel_a_data["individual_decoding_error"]
    individual_labels = panel_a_data[
        "individual_decoding_significance_labels"
    ]
    panel_c_axes = []
    for animal_index, animal_name in enumerate(decoding_animal_names):
        row_index, column_index = divmod(
            animal_index,
            PANEL_C_ANIMALS_PER_ROW,
        )
        panel_c_axis = fig.add_subplot(
            panel_c_grid[row_index, column_index]
        )
        panel_c_axes.append(panel_c_axis)
        animal_table = filter_decoding_table_by_animal(
            individual_table,
            animal_name,
        )
        cross_ylim, place_ylim = get_panel_c_decoding_ylims(animal_table)
        figure_2.plot_panel_e2_decoding_panel(
            panel_c_axis,
            animal_table,
            significance_labels=individual_labels[animal_name],
            cross_ylim=cross_ylim,
            place_ylim=place_ylim,
        )
        panel_c_axis.text(
            PANEL_C_ANIMAL_LABEL_X,
            0.5,
            animal_name,
            ha="right",
            va="center",
            fontsize=PANEL_C_ANIMAL_LABEL_FONTSIZE,
            color="0.25",
            transform=panel_c_axis.transAxes,
        )
        if animal_index == 0:
            panel_c_title = panel_c_axis.set_title(
                PANEL_TITLES[2],
                fontsize=8,
                pad=figure_2.PANEL_BC_TITLE_PAD,
            )
            label_axis(
                panel_c_axis,
                "C",
                x=-0.017,
                y=figure_2.PANEL_BC_LABEL_Y,
            )
            panel_c_label = panel_c_axis.texts[-1]
    figure_2._raise_text_to_minimum_fontsize(
        fig,
        figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )

    fig.canvas.draw()
    fig.set_layout_engine(None)
    if panel_c_title is None or panel_c_label is None:
        raise ValueError("Supplementary Figure 4C requires at least one animal.")
    _center_axis_title_over_axes(
        panel_c_title,
        panel_c_axes[:PANEL_C_ANIMALS_PER_ROW],
    )
    top_header_texts = (
        panel_a_title,
        panel_a_label,
        panel_b_title,
        panel_b_label,
    )
    figure_2._align_texts_to_reference_display_y(top_header_texts)
    figure_2._align_text_tops_to_reference_display_y(fig, top_header_texts)
    lower_header_texts = (panel_c_title, panel_c_label)
    figure_2._align_texts_to_reference_display_y(lower_header_texts)
    figure_2._align_text_tops_to_reference_display_y(fig, lower_header_texts)

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 4 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 4."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 4 panels A--C."
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
    """Run Supplementary Figure 4 generation."""
    args = parse_arguments(argv)
    datasets = (
        args.dataset if args.dataset is not None else get_processed_datasets()
    )
    output_path = figure_2.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_4(
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
