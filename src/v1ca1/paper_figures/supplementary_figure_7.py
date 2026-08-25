"""Generate Supplementary Figure 7 from existing ripple-analysis panels."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures._ripple_panels import (
    DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    DEFAULT_MINIMUM_RIPPLE_MEAN_ZSCORE,
    DEFAULT_REGIONS,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    DEFAULT_RIPPLE_WINDOW_S,
    FIGURE_FORMATS,
    PANEL_A_EPOCH_ORDER,
    PANEL_C_SOURCE_MIXED_MODEL_PERMUTATIONS,
    PANEL_C_SOURCE_MIXED_MODEL_PERMUTATION_BATCH_SIZE,
    PANEL_C_SOURCE_MIXED_MODEL_RANDOM_SEED,
    PANEL_E_GLM_EPOCH_ORDER,
    add_aligned_panel_headers,
    compute_source_predictor_paired_mixed_model_permutation,
    filter_epoch_payloads,
    load_glm_source_predictor_comparison_tables,
    load_pooled_ripple_heatmap_epoch_tables,
    parse_dataset_id,
    plot_epoch_modulation_histogram_panel,
    plot_epoch_ripple_heatmap_panel,
    plot_glm_source_predictor_comparison_panel,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    save_figure,
)


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "supplementary_figure_7"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_PANEL_A_ROW_HEIGHT_MM = 65.0
DEFAULT_ANIMAL_ROW_HEIGHT_MM = 52.5
DEFAULT_ANIMAL_COUNT = 4
ANIMALS_PER_ROW = 2
DEFAULT_FIGURE_HEIGHT_MM = (
    DEFAULT_PANEL_A_ROW_HEIGHT_MM
    + math.ceil(DEFAULT_ANIMAL_COUNT / ANIMALS_PER_ROW)
    * DEFAULT_ANIMAL_ROW_HEIGHT_MM
)
OUTPUT_BOTTOM_CROP_MM = 6.0
PANEL_A_TOP_WIDTH_RATIOS = (1.0, 1.35, 1.0)
ANIMAL_PANEL_WIDTH_RATIOS = (0.22, 1.35, 1.0)
ANIMAL_GRID_WSPACE = 0.12
ANIMAL_GRID_HSPACE = 0.14
ANIMAL_PANEL_WSPACE = 0.08
PANEL_SECTION_HSPACE = 0.12
POOLED_PANEL_LABELS = ("A", "B", "C")
INDIVIDUAL_PANEL_LABEL = "D"
PANEL_LABEL_X_OFFSETS = (0.0, -0.08, -0.10)
ANIMAL_LABEL_X = 0.5
ANIMAL_LABEL_Y = 0.5
ANIMAL_LABEL_FONTSIZE = 7.0
L14_PANEL_C_AXIS_LIMITS = (-0.2, 0.25)
PANEL_A_HISTOGRAM_BOTTOM = 0.20
PANEL_A_HISTOGRAM_HEIGHT = 0.70
PANEL_A_HEATMAP_VERTICAL_BOUNDS = (
    PANEL_A_HISTOGRAM_BOTTOM,
    PANEL_A_HISTOGRAM_BOTTOM + PANEL_A_HISTOGRAM_HEIGHT,
)
PANEL_TITLES = (
    "Ripple-triggered\nmean firing rates",
    "Ripple modulation index",
    "CA1 spike vector vs.\nmean CA1 activity",
)
RIPPLE_SELECTION_CHOICES = ("allripples", "deduped", "single")


def group_datasets_by_animal(
    datasets: Sequence[DatasetId],
) -> dict[str, list[DatasetId]]:
    """Return normalized data sets grouped by animal in input order."""
    grouped: dict[str, list[DatasetId]] = {}
    for dataset in datasets:
        normalized = normalize_dataset_id(dataset)
        grouped.setdefault(str(normalized[0]), []).append(normalized)
    return grouped


def get_animal_row_count(animal_count: int) -> int:
    """Return the rows needed to display two animals per row."""
    return max(math.ceil(max(int(animal_count), 0) / ANIMALS_PER_ROW), 1)


def get_figure_height_mm(animal_count: int) -> float:
    """Return the Panel A height plus the required animal rows."""
    return (
        DEFAULT_PANEL_A_ROW_HEIGHT_MM
        + get_animal_row_count(animal_count) * DEFAULT_ANIMAL_ROW_HEIGHT_MM
    )


def filter_table_by_animal(table: Any, animal_name: str) -> Any:
    """Return table rows belonging to one animal."""
    if table is None:
        return None
    if "animal_name" not in table.columns:
        if len(table) == 0:
            return table.copy()
        raise ValueError("Panel table is missing required column 'animal_name'.")
    return table.loc[
        table["animal_name"].astype(str) == str(animal_name)
    ].copy()


def filter_epoch_tables_by_animal(
    epoch_tables: Sequence[Mapping[str, Any]],
    animal_name: str,
) -> list[dict[str, Any]]:
    """Filter ripple heatmap and modulation payloads to one animal."""
    filtered_payloads = []
    for payload in epoch_tables:
        filtered = dict(payload)
        for key in ("firing_rate_table", "summary_table"):
            if key in payload:
                filtered[key] = filter_table_by_animal(
                    payload[key],
                    animal_name,
                )
        if "datasets" in payload:
            filtered_datasets = tuple(
                dataset
                for dataset in payload["datasets"]
                if normalize_dataset_id(dataset)[0] == str(animal_name)
            )
            filtered["datasets"] = filtered_datasets
            filtered["n_datasets"] = len(filtered_datasets)
            filtered["epochs"] = tuple(
                normalize_dataset_id(dataset)[2]
                for dataset in filtered_datasets
            )
        filtered_payloads.append(filtered)
    return filtered_payloads


def filter_source_payload_by_animal(
    payload: Mapping[str, Any],
    animal_name: str,
) -> dict[str, Any]:
    """Filter the CA1-source comparison payload to one animal."""
    filtered = dict(payload)
    filtered["comparison_table"] = filter_table_by_animal(
        payload.get("comparison_table"),
        animal_name,
    )
    filtered["missing_artifacts"] = [
        artifact
        for artifact in payload.get("missing_artifacts", ())
        if str(artifact.get("animal_name")) == str(animal_name)
    ]
    return filtered


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the requested Supplementary Figure 7 output path."""
    if output_format not in FIGURE_FORMATS:
        raise ValueError(
            f"Unknown output format {output_format!r}. "
            f"Expected one of {FIGURE_FORMATS!r}."
        )
    return Path(output_dir) / f"{output_name}.{output_format}"


def make_supplementary_figure_7(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    ripple_threshold_zscore: float | None,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ripple_selection: str,
    ridge_strength: float,
    dpi: int,
    source_comparison_n_permutations: int = (
        PANEL_C_SOURCE_MIXED_MODEL_PERMUTATIONS
    ),
    source_comparison_permutation_seed: int = (
        PANEL_C_SOURCE_MIXED_MODEL_RANDOM_SEED
    ),
    source_comparison_permutation_batch_size: int = (
        PANEL_C_SOURCE_MIXED_MODEL_PERMUTATION_BATCH_SIZE
    ),
) -> Path:
    """Build Supplementary Figure 7 with Panels B and C split by animal."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    if source_comparison_n_permutations <= 0:
        raise ValueError("source_comparison_n_permutations must be positive.")
    if source_comparison_permutation_seed < 0:
        raise ValueError(
            "source_comparison_permutation_seed must be non-negative."
        )
    if source_comparison_permutation_batch_size <= 0:
        raise ValueError(
            "source_comparison_permutation_batch_size must be positive."
        )

    animal_groups = group_datasets_by_animal(datasets)
    if not animal_groups:
        raise ValueError("Supplementary Figure 7 requires at least one animal.")
    normalized_datasets = tuple(
        dataset
        for animal_datasets in animal_groups.values()
        for dataset in animal_datasets
    )
    animal_names = tuple(animal_groups)
    animal_row_count = get_animal_row_count(len(animal_names))

    heatmap_epoch_tables = load_pooled_ripple_heatmap_epoch_tables(
        data_root,
        normalized_datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    panel_a_epoch_tables = filter_epoch_payloads(
        heatmap_epoch_tables,
        PANEL_A_EPOCH_ORDER,
    )
    source_comparison_payload = load_glm_source_predictor_comparison_tables(
        data_root,
        normalized_datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        epoch_types=PANEL_E_GLM_EPOCH_ORDER,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )
    pooled_source_statistics = (
        compute_source_predictor_paired_mixed_model_permutation(
            source_comparison_payload.get("comparison_table"),
            n_permutations=source_comparison_n_permutations,
            random_seed=source_comparison_permutation_seed,
            permutation_batch_size=(
                source_comparison_permutation_batch_size
            ),
        )
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            get_figure_height_mm(len(animal_names)),
        ),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=(
            DEFAULT_PANEL_A_ROW_HEIGHT_MM,
            animal_row_count * DEFAULT_ANIMAL_ROW_HEIGHT_MM,
        ),
        hspace=PANEL_SECTION_HSPACE,
    )
    pooled_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=PANEL_A_TOP_WIDTH_RATIOS,
        wspace=ANIMAL_PANEL_WSPACE,
    )
    pooled_axes = [
        fig.add_subplot(pooled_grid[0, column_index])
        for column_index in range(3)
    ]
    panel_a_axis, pooled_panel_b_axis, pooled_panel_c_axis = pooled_axes
    animal_grid = outer_grid[1, 0].subgridspec(
        nrows=animal_row_count,
        ncols=ANIMALS_PER_ROW,
        wspace=ANIMAL_GRID_WSPACE,
        hspace=ANIMAL_GRID_HSPACE,
    )

    plot_epoch_ripple_heatmap_panel(
        panel_a_axis,
        panel_a_epoch_tables,
        regions=regions,
        expand_heatmaps_vertically=True,
        show_modulation_histogram=False,
        heatmap_vertical_bounds=PANEL_A_HEATMAP_VERTICAL_BOUNDS,
    )
    plot_epoch_modulation_histogram_panel(
        pooled_panel_b_axis,
        panel_a_epoch_tables,
        regions=regions,
        bottom=PANEL_A_HISTOGRAM_BOTTOM,
        height=PANEL_A_HISTOGRAM_HEIGHT,
    )
    plot_glm_source_predictor_comparison_panel(
        pooled_panel_c_axis,
        source_comparison_payload,
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
        annotate_pooled_inference=True,
        pooled_statistics=pooled_source_statistics,
    )
    for axis, title in zip(pooled_axes, PANEL_TITLES, strict=True):
        axis.set_title(title, fontsize=7.2, pad=2)

    animal_axes: list[tuple[Any, Any]] = []
    animal_sign_tests: dict[str, dict[str, Any] | None] = {}
    for animal_index, animal_name in enumerate(animal_names):
        row_index, column_index = divmod(animal_index, ANIMALS_PER_ROW)
        animal_panel_grid = animal_grid[row_index, column_index].subgridspec(
            nrows=1,
            ncols=3,
            width_ratios=ANIMAL_PANEL_WIDTH_RATIOS,
            wspace=ANIMAL_PANEL_WSPACE,
        )
        animal_label_axis = fig.add_subplot(animal_panel_grid[0, 0])
        panel_b_axis = fig.add_subplot(animal_panel_grid[0, 1])
        panel_c_axis = fig.add_subplot(animal_panel_grid[0, 2])
        animal_label_axis.set_axis_off()
        animal_epoch_tables = filter_epoch_tables_by_animal(
            panel_a_epoch_tables,
            animal_name,
        )
        animal_source_payload = filter_source_payload_by_animal(
            source_comparison_payload,
            animal_name,
        )
        plot_epoch_modulation_histogram_panel(
            panel_b_axis,
            animal_epoch_tables,
            regions=regions,
            bottom=PANEL_A_HISTOGRAM_BOTTOM,
            height=PANEL_A_HISTOGRAM_HEIGHT,
        )
        animal_sign_tests[animal_name] = (
            plot_glm_source_predictor_comparison_panel(
                panel_c_axis,
                animal_source_payload,
                include_per_animal=False,
                include_pooled=True,
                compact_labels=True,
                show_color_note=False,
                annotate_pooled_sign_test=False,
                axis_limits=(
                    L14_PANEL_C_AXIS_LIMITS
                    if animal_name == "L14"
                    else None
                ),
            )
        )
        animal_label_axis.text(
            ANIMAL_LABEL_X,
            ANIMAL_LABEL_Y,
            animal_name,
            ha="center",
            va="center",
            fontsize=ANIMAL_LABEL_FONTSIZE,
            fontweight="bold",
            color="0.15",
            transform=animal_label_axis.transAxes,
            clip_on=False,
            zorder=10,
        )
        animal_axes.append((panel_b_axis, panel_c_axis))

    fig.canvas.draw()
    fig.set_layout_engine(None)
    add_aligned_panel_headers(
        fig,
        pooled_axes,
        labels=POOLED_PANEL_LABELS,
        titles=PANEL_TITLES,
        label_x_offsets=PANEL_LABEL_X_OFFSETS,
        fontsize=7.2,
    )
    top_animal_axes = [
        axis
        for panel_axes in animal_axes[:ANIMALS_PER_ROW]
        for axis in panel_axes
    ]
    add_aligned_panel_headers(
        fig,
        top_animal_axes,
        labels=(
            INDIVIDUAL_PANEL_LABEL,
            *("" for _axis in top_animal_axes[1:]),
        ),
        titles=tuple("" for _axis in top_animal_axes),
        label_x_offsets=tuple(
            offset
            for _animal_index in range(len(top_animal_axes) // 2)
            for offset in PANEL_LABEL_X_OFFSETS[1:]
        ),
        fontsize=7.2,
    )
    figure_width_in, figure_height_in = fig.get_size_inches()
    bottom_crop_in = OUTPUT_BOTTOM_CROP_MM / 25.4
    output_bbox = Bbox.from_bounds(
        0.0,
        bottom_crop_in,
        figure_width_in,
        figure_height_in - bottom_crop_in,
    )
    save_figure(fig, output_path, dpi=dpi, bbox_inches=output_bbox)
    plt.close(fig)

    for missing in source_comparison_payload["missing_artifacts"]:
        print(
            "Supplementary Figure 7 source-comparison missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']} "
            f"({missing['source_predictor_mode']}): {missing['path']}"
        )
    pooled_p_value = float(pooled_source_statistics["p_value"])
    pooled_p_value_text = (
        f"{pooled_p_value:.3g}" if np.isfinite(pooled_p_value) else "nan"
    )
    pooled_coefficient = float(pooled_source_statistics["coefficient"])
    pooled_coefficient_text = (
        f"{pooled_coefficient:.4g}"
        if np.isfinite(pooled_coefficient)
        else "nan"
    )
    print(
        "Supplementary Figure 7 Panel C paired-delta random-intercept "
        "LMM permutation: "
        f"beta={pooled_coefficient_text}; "
        f"n={pooled_source_statistics['n_finite_pairs']} V1 units across "
        f"{pooled_source_statistics['n_animals']} animals; "
        f"one-sided p={pooled_p_value_text}"
    )
    for group_name, sign_test in animal_sign_tests.items():
        if sign_test is None:
            continue
        p_value = float(sign_test["p_value"])
        p_value_text = f"{p_value:.3g}" if np.isfinite(p_value) else "nan"
        print(
            f"Supplementary Figure 7 {group_name} one-sided paired sign test: "
            f"{sign_test['n_vector_greater']}/"
            f"{sign_test['n_tested']} non-tied V1 units favor the "
            "CA1 vector model; "
            f"{sign_test['n_ties']} ties; p={p_value_text}"
        )
    print(f"Saved Supplementary Figure 7 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Supplementary Figure 7 command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Supplementary Figure 7 ripple heatmap, modulation, "
            "and CA1-source comparison panels."
        )
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
            "Animal/date/epoch data set as animal:date[:epoch]. May be repeated. "
            "Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=DEFAULT_REGIONS,
        help=(
            "Region to include in panels A and B. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument("--light-epoch", default=None)
    parser.add_argument("--dark-epoch", default=None)
    parser.add_argument("--sleep-epoch", default=None)
    parser.add_argument(
        "--minimum-ripple-mean-zscore",
        "--ripple-threshold-zscore",
        dest="ripple_threshold_zscore",
        type=float,
        default=DEFAULT_MINIMUM_RIPPLE_MEAN_ZSCORE,
        help="Optional minimum event mean z-score for cached ripple outputs.",
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_S,
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    )
    parser.add_argument(
        "--ripple-selection",
        choices=RIPPLE_SELECTION_CHOICES,
        default=DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
    )
    parser.add_argument(
        "--panel-c-permutations",
        type=int,
        default=PANEL_C_SOURCE_MIXED_MODEL_PERMUTATIONS,
        help=(
            "Paired label-swap permutations for the Panel C animal "
            "random-intercept LMM. "
            f"Default: {PANEL_C_SOURCE_MIXED_MODEL_PERMUTATIONS}"
        ),
    )
    parser.add_argument(
        "--panel-c-permutation-seed",
        type=int,
        default=PANEL_C_SOURCE_MIXED_MODEL_RANDOM_SEED,
        help=(
            "Random seed for the Panel C mixed-model permutation test. "
            f"Default: {PANEL_C_SOURCE_MIXED_MODEL_RANDOM_SEED}"
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
    """Run Supplementary Figure 7 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_7(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        sleep_epoch=args.sleep_epoch,
        ripple_threshold_zscore=args.ripple_threshold_zscore,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection=args.ripple_selection,
        ridge_strength=args.ridge_strength,
        dpi=args.dpi,
        source_comparison_n_permutations=args.panel_c_permutations,
        source_comparison_permutation_seed=args.panel_c_permutation_seed,
    )


if __name__ == "__main__":
    main()
