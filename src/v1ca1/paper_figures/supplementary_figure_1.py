from __future__ import annotations

"""Generate Supplementary Figure 1 per-data-set model comparison panels."""

import argparse
import json
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
    DECODING_COMPARISON_REGION,
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_1_WIDTH_MM,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    ENCODING_COMPARISON_PLACE_BIN_SIZE_CM,
    ENCODING_COMPARISON_REGION,
    HEATMAP_COLORBAR_LABELPAD,
    HEATMAP_COLORBAR_LABEL_FONTSIZE,
    HEATMAP_COLORBAR_PAD,
    MOTOR_DELTA_REGION,
    PANEL_D_CACHE_METADATA_KEY,
    PANEL_D_CACHE_PREFIX,
    PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    PANEL_D_HEATMAP_CMAP,
    PANEL_D_LINEAR_POSITION_ORIENTATION,
    PANEL_D_TRAJECTORY_TYPES,
    PANEL_E_TRAJECTORY_COLORS,
    STABILITY_REGIONS,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    build_panel_d_cache_metadata,
    draw_panel_a_anatomy_assets,
    draw_order_schematic,
    load_decoding_absolute_error_table,
    load_dark_epoch_stability_table,
    load_encoding_delta_table,
    load_motor_delta_table,
    load_or_compute_panel_d_heatmap_panels,
    parse_dataset_id,
    plot_decoding_error_panel,
    plot_encoding_delta_panel,
    plot_motor_delta_panel,
    plot_pooled_heatmap_grid,
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
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "supplementary_figure_1"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_1_WIDTH_MM
DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM = 40.0
DEFAULT_DATASET_PANEL_HEIGHT_MM = 26.0
DEFAULT_SECTION_SPACER_MM = 5.0
PANEL_D_NORMALIZATION_COMPARISON_WSPACE = 0.08
PANEL_D_NORMALIZATION_HEATMAP_WSPACE = 0.0
PANEL_D_NORMALIZATION_HEATMAP_HSPACE = 0.0
PANEL_D_NORMALIZATION_TITLE_FONTSIZE = 7.2
PANEL_D_PER_TRAJECTORY_CACHE_VERSION = 2
PANEL_D_NORMALIZATION_REGION = "v1"
DARK_MOVEMENT_FIRING_RATE_REGION = "v1"
DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ = 0.5
DARK_MOVEMENT_FIRING_RATE_BIN_COUNT = 24
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
        zero_count = int(np.sum(finite_values <= 0.0))
        if zero_count:
            ax.text(
                0.04,
                0.06,
                f"{zero_count} at 0 Hz",
                ha="left",
                va="bottom",
                fontsize=4.8,
                color="0.35",
                transform=ax.transAxes,
            )

    ax.set_xscale("log")
    ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Movement firing rate (Hz)", fontsize=6.2, labelpad=1.0)
    ax.set_ylabel("Frac. V1 cells", fontsize=6.2, labelpad=1.0)
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


def build_panel_d_per_trajectory_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Return metadata for the legacy per-trajectory Figure 1D cache."""
    metadata = build_panel_d_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        min_movement_firing_rate_hz=None,
        min_tuning_stability_correlation=None,
    )
    metadata["cache_version"] = PANEL_D_PER_TRAJECTORY_CACHE_VERSION
    metadata.pop("firing_rate_normalization", None)
    return metadata


def load_panel_d_cache_if_metadata_matches(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray] | None:
    """Return Panel D arrays from one cache only when metadata matches exactly."""
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if PANEL_D_CACHE_METADATA_KEY not in data.files:
                return None
            cached_metadata = json.loads(str(data[PANEL_D_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                return None

            trajectory_types = tuple(
                str(trajectory) for trajectory in expected_metadata["trajectory_types"]
            )
            panels: dict[tuple[str, str], np.ndarray] = {}
            for order_trajectory in trajectory_types:
                for plot_trajectory in trajectory_types:
                    array_name = f"{order_trajectory}__{plot_trajectory}"
                    if array_name not in data.files:
                        return None
                    panels[(order_trajectory, plot_trajectory)] = np.asarray(
                        data[array_name],
                        dtype=float,
                    )
            return panels
    except Exception:
        return None


def load_panel_d_per_trajectory_panels(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path,
) -> dict[tuple[str, str], np.ndarray]:
    """Load the saved legacy Figure 1D per-trajectory normalization cache."""
    metadata = build_panel_d_per_trajectory_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    cache_paths = sorted(
        Path(panel_d_cache_dir).glob(
            f"{PANEL_D_CACHE_PREFIX}_*cachev{PANEL_D_PER_TRAJECTORY_CACHE_VERSION}.npz"
        )
    )
    for cache_path in cache_paths:
        panels = load_panel_d_cache_if_metadata_matches(cache_path, metadata)
        if panels is not None:
            print(f"Loaded per-trajectory Panel D cache from {cache_path}.")
            return panels

    raise FileNotFoundError(
        "No matching legacy per-trajectory Figure 1D cache was found in "
        f"{panel_d_cache_dir}. Regenerate the cachev2 Panel D heatmap before "
        "building this comparison."
    )


def plot_panel_d_normalization_heatmap(
    fig: Any,
    subplot_spec: Any,
    panels: dict[tuple[str, str], np.ndarray],
    *,
    title: str,
    panel_label: str,
) -> Any | None:
    """Plot one compact Figure 1D heatmap grid for a normalization mode."""
    trajectory_types = PANEL_D_TRAJECTORY_TYPES
    grid = subplot_spec.subgridspec(
        nrows=len(trajectory_types) + 2,
        ncols=len(trajectory_types) + 1,
        height_ratios=[0.24, 0.42, *([1.0] * len(trajectory_types))],
        width_ratios=[0.48, *([1.0] * len(trajectory_types))],
        wspace=PANEL_D_NORMALIZATION_HEATMAP_WSPACE,
        hspace=PANEL_D_NORMALIZATION_HEATMAP_HSPACE,
    )
    title_axis = fig.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        fontsize=PANEL_D_NORMALIZATION_TITLE_FONTSIZE,
    )
    label_axis(title_axis, panel_label, x=-0.02, y=0.0)

    axes = np.asarray(
        [
            [
                fig.add_subplot(grid[row + 1, col])
                for col in range(len(trajectory_types) + 1)
            ]
            for row in range(len(trajectory_types) + 1)
        ],
        dtype=object,
    )
    corner_axis = axes[0, 0]
    corner_axis.axis("off")
    for ax, trajectory_type in zip(axes[0, 1:], trajectory_types, strict=True):
        draw_w_track_schematic(
            ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            fill_track=True,
        )
    for row_index, ax in enumerate(axes[1:, 0]):
        trajectory_type = trajectory_types[row_index % len(trajectory_types)]
        draw_order_schematic(
            ax,
            trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
        )

    heatmap_axes = axes[1:, 1:]
    color_image = plot_pooled_heatmap_grid(
        heatmap_axes,
        panels,
        trajectory_types=trajectory_types,
        axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
        cmap=PANEL_D_HEATMAP_CMAP,
    )
    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=heatmap_axes.ravel().tolist(),
            shrink=0.28,
            pad=HEATMAP_COLORBAR_PAD,
            aspect=7,
            ticks=[0.0, 1.0],
        )
        colorbar.ax.set_yticklabels(["0", "1"])
        colorbar.ax.tick_params(length=2)
        colorbar.set_label(
            "Norm. FR",
            rotation=90,
            labelpad=HEATMAP_COLORBAR_LABELPAD,
            fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
        )
    return color_image


def plot_panel_d_normalization_comparison(
    fig: Any,
    subplot_spec: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    panel_d_cache_dir: Path,
) -> None:
    """Plot side-by-side Figure 1D heatmaps for the two normalization modes."""
    per_trajectory_panels = load_panel_d_per_trajectory_panels(
        data_root=data_root,
        datasets=datasets,
        region=PANEL_D_NORMALIZATION_REGION,
        position_bin_count=DEFAULT_POSITION_BIN_COUNT,
        position_offset=DEFAULT_POSITION_OFFSET,
        speed_threshold_cm_s=DEFAULT_SPEED_THRESHOLD_CM_S,
        sigma_bins=DEFAULT_SIGMA_BINS,
        panel_d_cache_dir=panel_d_cache_dir,
    )
    across_trajectory_panels = load_or_compute_panel_d_heatmap_panels(
        data_root=data_root,
        datasets=datasets,
        region=PANEL_D_NORMALIZATION_REGION,
        position_bin_count=DEFAULT_POSITION_BIN_COUNT,
        position_offset=DEFAULT_POSITION_OFFSET,
        speed_threshold_cm_s=DEFAULT_SPEED_THRESHOLD_CM_S,
        sigma_bins=DEFAULT_SIGMA_BINS,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=False,
        firing_rate_normalization=PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        min_movement_firing_rate_hz=None,
        min_tuning_stability_correlation=None,
    )

    comparison_grid = subplot_spec.subgridspec(
        nrows=1,
        ncols=2,
        wspace=PANEL_D_NORMALIZATION_COMPARISON_WSPACE,
    )
    plot_panel_d_normalization_heatmap(
        fig,
        comparison_grid[0, 0],
        per_trajectory_panels,
        title="Per-trajectory normalization",
        panel_label="D",
    )
    plot_panel_d_normalization_heatmap(
        fig,
        comparison_grid[0, 1],
        across_trajectory_panels,
        title="Across-trajectory normalization",
        panel_label="E",
    )


def make_supplementary_figure_1(
    *,
    data_root: Path,
    asset_dir: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    encoding_place_bin_size_cm: float,
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
    moved_firing_rate_axis.set_title("Dark movement firing rates", fontsize=8, pad=2)
    label_axis(moved_firing_rate_axis, "B", x=-0.02, y=1.01)

    moved_stability_axis = fig.add_subplot(moved_grid[0, 2])
    plot_pooled_stability_panel(
        moved_stability_axis,
        data_root=data_root,
        datasets=datasets,
    )
    moved_stability_axis.set_title("Pooled tuning stability", fontsize=8, pad=2)
    label_axis(moved_stability_axis, "C", x=-0.02, y=1.01)

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
        ("D", "E", "F"),
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
        encoding_place_bin_size_cm=args.encoding_place_bin_size_cm,
        dpi=args.dpi,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
    )


if __name__ == "__main__":
    main()
