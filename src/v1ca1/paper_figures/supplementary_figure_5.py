"""Generate Supplementary Figure 5 from existing ripple-analysis panels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures.datasets import DatasetId, get_processed_datasets
from v1ca1.paper_figures.figure_3_old import (
    DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    DEFAULT_MINIMUM_RIPPLE_MEAN_ZSCORE,
    DEFAULT_REGIONS,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    DEFAULT_RIPPLE_WINDOW_S,
    FIGURE_FORMATS,
    PANEL_A_EPOCH_ORDER,
    PANEL_E_GLM_EPOCH_ORDER,
    add_aligned_panel_headers,
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
DEFAULT_OUTPUT_NAME = "supplementary_figure_5"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_FIGURE_HEIGHT_MM = 70.0
OUTPUT_BOTTOM_CROP_MM = 6.0
PANEL_WIDTH_RATIOS = (1.0, 1.35, 1.0)
PANEL_WSPACE = 0.08
PANEL_LABEL_X_OFFSETS = (0.0, -0.08, -0.10)
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


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the requested Supplementary Figure 5 output path."""
    if output_format not in FIGURE_FORMATS:
        raise ValueError(
            f"Unknown output format {output_format!r}. "
            f"Expected one of {FIGURE_FORMATS!r}."
        )
    return Path(output_dir) / f"{output_name}.{output_format}"


def make_supplementary_figure_5(
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
) -> Path:
    """Build Supplementary Figure 5 from three existing panel plotters."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    heatmap_epoch_tables = load_pooled_ripple_heatmap_epoch_tables(
        data_root,
        datasets,
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
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        epoch_types=PANEL_E_GLM_EPOCH_ORDER,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=PANEL_WIDTH_RATIOS,
        wspace=PANEL_WSPACE,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    plot_epoch_ripple_heatmap_panel(
        axes[0],
        panel_a_epoch_tables,
        regions=regions,
        expand_heatmaps_vertically=True,
        show_modulation_histogram=False,
        heatmap_vertical_bounds=PANEL_A_HEATMAP_VERTICAL_BOUNDS,
    )
    plot_epoch_modulation_histogram_panel(
        axes[1],
        panel_a_epoch_tables,
        regions=regions,
        bottom=PANEL_A_HISTOGRAM_BOTTOM,
        height=PANEL_A_HISTOGRAM_HEIGHT,
    )
    pooled_sign_test = plot_glm_source_predictor_comparison_panel(
        axes[2],
        source_comparison_payload,
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
        annotate_pooled_sign_test=True,
    )
    for axis, title in zip(axes, PANEL_TITLES, strict=True):
        axis.set_title(title, fontsize=7.2, pad=2)

    fig.canvas.draw()
    fig.set_layout_engine(None)
    add_aligned_panel_headers(
        fig,
        axes,
        labels=("A", "B", "C"),
        titles=PANEL_TITLES,
        label_x_offsets=PANEL_LABEL_X_OFFSETS,
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
            "Supplementary Figure 5 source-comparison missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']} "
            f"({missing['source_predictor_mode']}): {missing['path']}"
        )
    if pooled_sign_test is not None:
        p_value = float(pooled_sign_test["p_value"])
        p_value_text = f"{p_value:.3g}" if np.isfinite(p_value) else "nan"
        print(
            "Supplementary Figure 5 pooled one-sided paired sign test: "
            f"{pooled_sign_test['n_vector_greater']}/"
            f"{pooled_sign_test['n_tested']} non-tied V1 units favor the "
            "CA1 vector model; "
            f"{pooled_sign_test['n_ties']} ties; p={p_value_text}"
        )
    print(f"Saved Supplementary Figure 5 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Supplementary Figure 5 command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Supplementary Figure 5 ripple heatmap, modulation, "
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
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Supplementary Figure 5 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_5(
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
    )


if __name__ == "__main__":
    main()
