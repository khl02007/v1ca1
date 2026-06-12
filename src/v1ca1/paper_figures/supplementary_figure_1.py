from __future__ import annotations

"""Generate Supplementary Figure 1 pooled summary panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_1_WIDTH_MM,
    STABILITY_REGIONS,
    draw_panel_a_anatomy_assets,
    load_dark_epoch_stability_table,
    parse_dataset_id,
    plot_stability_panel,
)
from v1ca1.paper_figures.figure_2 import load_dark_movement_firing_rate_table
from v1ca1.paper_figures.style import (
    EMPHASIS_HISTOGRAM_KWARGS,
    REGION_COLORS,
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
DARK_MOVEMENT_FIRING_RATE_REGION = "v1"
DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ = 0.5
DARK_MOVEMENT_FIRING_RATE_BIN_COUNT = 24
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")


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


def load_pooled_dark_movement_firing_rate_table(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    region: str = DARK_MOVEMENT_FIRING_RATE_REGION,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> Any:
    """Return pooled V1 dark movement firing rates across data sets."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        table = load_dark_movement_firing_rate_table(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            dark_epoch=dark_epoch,
            region=region,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        ).copy()
        table["animal_name"] = animal_name
        table["date"] = date
        table["dark_epoch"] = dark_epoch
        tables.append(table)

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "dark_epoch",
                "unit",
                "dark_firing_rate_hz",
            ]
        )
    return pd.concat(tables, ignore_index=True)


def get_log_histogram_edges(
    values: np.ndarray,
    *,
    reference_hz: float,
    bin_count: int,
) -> np.ndarray:
    """Return log-spaced histogram edges spanning positive values and a reference."""
    if reference_hz <= 0.0:
        raise ValueError("reference_hz must be positive for a log-scale histogram.")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")

    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size:
        lower_reference = min(float(np.nanmin(positive_values)), float(reference_hz))
        upper_reference = max(float(np.nanmax(positive_values)), float(reference_hz))
    else:
        lower_reference = float(reference_hz) / 10.0
        upper_reference = float(reference_hz) * 10.0

    lower_edge = 10.0 ** np.floor(np.log10(lower_reference))
    upper_edge = 10.0 ** np.ceil(np.log10(upper_reference))
    if upper_edge <= lower_edge:
        upper_edge = lower_edge * 10.0

    edges = np.geomspace(lower_edge, upper_edge, int(bin_count) + 1)
    return np.sort(np.unique(np.append(edges, float(reference_hz))))


def plot_pooled_dark_movement_firing_rate_histogram(
    ax: Any,
    dark_movement_firing_rate_table: Any,
    *,
    threshold_hz: float = DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ,
    region: str = DARK_MOVEMENT_FIRING_RATE_REGION,
    bin_count: int = DARK_MOVEMENT_FIRING_RATE_BIN_COUNT,
) -> None:
    """Plot pooled dark movement firing-rate distribution for V1 cells."""
    if (
        dark_movement_firing_rate_table is None
        or "dark_firing_rate_hz" not in dark_movement_firing_rate_table
    ):
        values = np.asarray([], dtype=float)
    else:
        values = np.asarray(
            dark_movement_firing_rate_table["dark_firing_rate_hz"],
            dtype=float,
        )
    finite_values = values[np.isfinite(values)]
    positive_values = finite_values[finite_values > 0.0]
    bin_edges = get_log_histogram_edges(
        positive_values,
        reference_hz=threshold_hz,
        bin_count=bin_count,
    )

    ax.axvline(
        threshold_hz,
        color="0.20",
        linestyle="--",
        linewidth=0.8,
        zorder=3,
    )
    if finite_values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite\nrates",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
    else:
        if positive_values.size:
            weights = np.full(
                positive_values.shape,
                1.0 / finite_values.size,
                dtype=float,
            )
            ax.hist(
                positive_values,
                bins=bin_edges,
                weights=weights,
                color=REGION_COLORS.get(region, "0.4"),
                **EMPHASIS_HISTOGRAM_KWARGS,
                zorder=2,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No positive\nrates",
                ha="center",
                va="center",
                fontsize=6,
                transform=ax.transAxes,
            )

        fraction_above_threshold = float(np.mean(finite_values > float(threshold_hz)))
        ax.text(
            0.04,
            0.96,
            f"{fraction_above_threshold:.0%} > {threshold_hz:g} Hz\nn={finite_values.size}",
            ha="left",
            va="top",
            fontsize=5.6,
            color=REGION_COLORS.get(region, "0.25"),
            transform=ax.transAxes,
        )
    ax.set_xscale("log")
    ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Mean firing rate (Hz)", fontsize=6.2, labelpad=1.0)
    ax.set_ylabel("Frac.", fontsize=6.2, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def plot_pooled_dark_movement_firing_rate_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    cache_dir: Path | None,
    refresh_cache: bool = False,
) -> None:
    """Load and plot pooled V1 dark movement firing rates."""
    table = load_pooled_dark_movement_firing_rate_table(
        data_root,
        datasets,
        region=DARK_MOVEMENT_FIRING_RATE_REGION,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )
    plot_pooled_dark_movement_firing_rate_histogram(
        ax,
        table,
        threshold_hz=DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ,
        region=DARK_MOVEMENT_FIRING_RATE_REGION,
    )


def make_supplementary_figure_1(
    *,
    data_root: Path,
    asset_dir: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    dpi: int,
    dark_movement_fr_cache_dir: Path | None = None,
    refresh_dark_movement_fr_cache: bool = False,
) -> Path:
    """Build and save Supplementary Figure 1."""
    import matplotlib.pyplot as plt

    dark_movement_fr_cache_dir = (
        Path(output_path).parent / "cache"
        if dark_movement_fr_cache_dir is None
        else Path(dark_movement_fr_cache_dir)
    )
    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
        ),
        constrained_layout=True,
    )
    moved_grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[0.30, 0.30, 0.40],
        wspace=0.18,
    )
    moved_anatomy_axis = fig.add_subplot(moved_grid[0, 0])
    draw_panel_a_anatomy_assets(moved_anatomy_axis, asset_dir=asset_dir)
    moved_anatomy_axis.set_title("Probe and histology", fontsize=8, pad=2)
    label_axis(moved_anatomy_axis, "A", x=-0.02, y=1.01)

    moved_firing_rate_axis = fig.add_subplot(moved_grid[0, 1])
    plot_pooled_dark_movement_firing_rate_panel(
        moved_firing_rate_axis,
        data_root=data_root,
        datasets=datasets,
        cache_dir=dark_movement_fr_cache_dir,
        refresh_cache=refresh_dark_movement_fr_cache,
    )
    moved_firing_rate_axis.set_title(
        "V1 firing rate in darkness",
        fontsize=8,
        pad=2,
    )
    label_axis(moved_firing_rate_axis, "B", x=-0.02, y=1.01)

    moved_stability_axis = fig.add_subplot(moved_grid[0, 2])
    plot_pooled_stability_panel(
        moved_stability_axis,
        data_root=data_root,
        datasets=datasets,
    )
    moved_stability_axis.set_title("Tuning stability", fontsize=8, pad=2)
    label_axis(moved_stability_axis, "C", x=-0.02, y=1.01)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Supplementary Figure 1 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 1 generation."""
    parser = argparse.ArgumentParser(description="Generate Supplementary Figure 1.")
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
        "--dark-movement-fr-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached dark movement firing-rate tables. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-dark-movement-fr-cache",
        action="store_true",
        help="Recompute and overwrite cached dark movement firing-rate tables.",
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
    dark_movement_fr_cache_dir = (
        args.dark_movement_fr_cache_dir
        if args.dark_movement_fr_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_supplementary_figure_1(
        data_root=args.data_root,
        asset_dir=args.asset_dir,
        output_path=output_path,
        datasets=datasets,
        dpi=args.dpi,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
    )


if __name__ == "__main__":
    main()
