from __future__ import annotations

"""Generate Figure 4 panels moved from the dark-light Figure 3 layout."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_processed_datasets,
)
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_2_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_2_WIDTH_MM,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    FIGURE_FORMATS,
    PANEL_A_EXAMPLE,
    PANEL_GH_WIDTH_RATIOS,
    build_output_path,
    load_panel_a_example_data,
    load_panel_glm_data,
    parse_dataset_id,
    plot_panel_a_example,
    plot_panel_g_model_architecture,
    plot_panel_h_swap_delta,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "figure_4"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_2_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = FIGURE_2_HEIGHT_MM
PANEL_A_TO_GH_HEIGHT_RATIOS = (1.0, 1.0)


def make_figure_4(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    dpi: int,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> Path:
    """Build and save Figure 4."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_glm_payload = load_panel_glm_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=PANEL_A_TO_GH_HEIGHT_RATIOS,
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    glm_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_GH_WIDTH_RATIOS,
    )
    panel_b_axis = fig.add_subplot(glm_grid[0, 0])
    panel_c_axis = fig.add_subplot(glm_grid[0, 1])

    panel_a_animal, panel_a_date, panel_a_region, panel_a_unit = PANEL_A_EXAMPLE
    panel_a_example = load_panel_a_example_data(
        data_root=data_root,
        animal_name=panel_a_animal,
        date=panel_a_date,
        region=panel_a_region,
        unit_id=panel_a_unit,
        dark_epoch=dark_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=refresh_panel_example_cache,
    )
    plot_panel_a_example(panel_a_axis, panel_a_example)
    plot_panel_g_model_architecture(
        panel_b_axis,
        panel_glm_payload["dark_light_examples"],
    )
    plot_panel_h_swap_delta(
        panel_c_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
    )

    label_axis(panel_a_axis, "A", x=-0.02, y=1.00)
    label_axis(panel_b_axis, "B", x=-0.035, y=1.12)
    label_axis(panel_c_axis, "C", x=-0.06, y=1.02)
    panel_a_axis.set_title(
        "Example visual cell in different visual conditions",
        fontsize=8,
        pad=2,
    )
    panel_b_axis.set_title(
        "Two possible models that relate dark and light activity",
        fontsize=8,
        pad=2,
    )
    panel_c_axis.set_title(
        "Predicting activity in held-out light epoch",
        fontsize=8,
        pad=2,
    )

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 4 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 4 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 dark-light example and GLM panels."
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
        "--panel-example-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel A example-cell rasters and rate curves. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help="Recompute Panel A example-cell data and overwrite matching caches.",
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
        action="append",
        choices=REGIONS,
        help=(
            "Region to include. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument(
        "--light-epoch",
        default=None,
        help=(
            "Light run epoch for GLM panels. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
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
        "--position-bin-count",
        type=int,
        default=DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {DEFAULT_POSITION_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 4 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_4(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        dpi=args.dpi,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
    )


if __name__ == "__main__":
    main()
