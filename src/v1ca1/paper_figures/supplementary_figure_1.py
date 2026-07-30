from __future__ import annotations

"""Generate Supplementary Figure 1 pooled summary panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
)
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    BOTTOM_ROW_PANEL_WSPACE,
    BOTTOM_ROW_PLOT_BOUNDS,
    DECODING_CROSS_TRAJECTORY_COMPARISONS,
    DECODING_SIGNIFICANCE_CONTRASTS,
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_1_WIDTH_MM,
    DEFAULT_HEATMAP_HEIGHT_MM,
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
    DEFAULT_PANEL_E_WIDTH_FRACTION,
    DEFAULT_POSITION_BIN_COUNT,
    HEATMAP_COLORBAR_ASPECT,
    HEATMAP_COLORBAR_LABEL_FONTSIZE,
    HEATMAP_COLORBAR_LABELPAD,
    HEATMAP_COLORBAR_PAD,
    HEATMAP_ORDER_LABEL_OFFSET,
    HEATMAP_PATH_LABEL_OFFSET,
    PANEL_D_LABEL_X,
    PANEL_D_LABEL_Y,
    PANEL_D_HEATMAP_BLOCK_OUTLINE_COLOR,
    PANEL_D_HEATMAP_BLOCK_OUTLINE_LINEWIDTH,
    PANEL_D_TRAJECTORY_TYPES,
    PANEL_E_AXIS_LABEL_FONTSIZE,
    PANEL_H_DECODING_ANIMALS,
    STABILITY_REGIONS,
    TASK_PROGRESSION_XLABEL,
    add_centered_axis_text,
    add_centered_below_axis_text,
    add_panel_d_heatmap_block_outlines,
    build_decoding_trial_error_table,
    compute_tuning_curve_peak_positions,
    compute_decoding_permutation_tests,
    draw_panel_a_anatomy_assets,
    draw_neuron_scale_bar,
    load_dark_epoch_stability_table,
    load_decoding_absolute_error_table,
    load_or_compute_panel_d_heatmap_full_payload,
    parse_dataset_id,
    plot_dark_heatmap_regions,
    plot_decoding_error_panel,
    plot_stability_panel,
    significance_stars,
)
from v1ca1.paper_figures.figure_3 import load_dark_movement_firing_rate_table
from v1ca1.paper_figures.heatmaps import setup_heatmap_comparison_panel
from v1ca1.paper_figures.style import (
    EMPHASIS_HISTOGRAM_KWARGS,
    REGION_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.raster.plot_place_field_heatmap import DEFAULT_SIGMA_BINS


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "supplementary_figure_1"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_1_WIDTH_MM
DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM = 40.0
DEFAULT_CA1_HEATMAP_ROW_HEIGHT_MM = DEFAULT_HEATMAP_HEIGHT_MM
DEFAULT_FIGURE_HEIGHT_MM = (
    DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM + DEFAULT_CA1_HEATMAP_ROW_HEIGHT_MM
)
DARK_MOVEMENT_FIRING_RATE_REGION = "v1"
DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ = 0.5
DARK_MOVEMENT_FIRING_RATE_BIN_COUNT = 24
CA1_HEATMAP_REGION = "ca1"
CA1_HEATMAP_TITLE = "CA1 task-progression tuning in darkness"
ORDER_PRESERVATION_REGIONS = ("v1", "ca1")
ORDER_PRESERVATION_REGION_LABELS = {"v1": "V1", "ca1": "CA1"}
ORDER_PRESERVATION_TRAJECTORY_LABELS = {
    "right_to_center": "R→C",
    "center_to_left": "C→L",
    "left_to_center": "L→C",
    "center_to_right": "C→R",
}
ORDER_PRESERVATION_TITLE = "Cross-path order preservation"
ORDER_PRESERVATION_CMAP = "RdBu_r"
ORDER_PRESERVATION_MIN_UNITS = 3
DECODING_COMPARISON_REGIONS = ("v1", "ca1")
TUNING_SIMILARITY_TITLE = "Same-turn tuning similarity"
DECODING_COMPARISON_TITLE = "Same-turn cross-path decoding"
TURN_PAIR_SPECS = (
    (
        "left_turn",
        "Left-turn\npair",
        "center_to_left",
        "right_to_center",
    ),
    (
        "right_turn",
        "Right-turn\npair",
        "center_to_right",
        "left_to_center",
    ),
)
TURN_DISTRIBUTION_REGION_OFFSET = 0.14
TURN_DISTRIBUTION_HALF_WIDTH = 0.11
TUNING_SIMILARITY_Y_LIMITS = (-1.0, 1.0)
DECODING_ERROR_Y_LIMITS = (0.0, 0.45)
SAME_TURN_DECODING_COMPARISONS = tuple(
    comparison_spec
    for comparison_spec in DECODING_CROSS_TRAJECTORY_COMPARISONS
    if comparison_spec[0] == "same_turn_cross_arm"
)
CA1_DECODING_REGION = "ca1"
CA1_DECODING_COMPARISONS = tuple(
    (
        comparison,
        "Opp. turn\nsame arm" if comparison == "opposite_turn_same_arm" else label,
        transfer_family,
        trajectory_pairs,
    )
    for comparison, label, transfer_family, trajectory_pairs in (
        DECODING_CROSS_TRAJECTORY_COMPARISONS
    )
)
CA1_DECODING_PLOT_BOUNDS = (
    BOTTOM_ROW_PLOT_BOUNDS[0],
    0.40,
    BOTTOM_ROW_PLOT_BOUNDS[2],
    0.50,
)
CA1_DECODING_XTICK_LABEL_FONTSIZE = 4.6
LOWER_ROW_RIGHT_HEIGHT_RATIOS = (1.0, 1.0)
LOWER_ROW_RIGHT_HSPACE = 0.28
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


def plot_pooled_ca1_dark_heatmap_panel(
    fig: Any,
    grid_spec: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
) -> dict[str, Any]:
    """Plot pooled dark-epoch CA1 heatmaps using the Figure 1D analysis."""
    panel = setup_heatmap_comparison_panel(
        fig,
        grid_spec,
        trajectory_types=PANEL_D_TRAJECTORY_TYPES,
        fill_track=True,
    )
    heatmap_axes = panel["heatmap_axes"]
    color_image = plot_dark_heatmap_regions(
        heatmap_axes,
        data_root=data_root,
        datasets=datasets,
        regions=(CA1_HEATMAP_REGION,),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=refresh_panel_d_cache,
    )
    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=heatmap_axes.ravel().tolist(),
            shrink=0.24,
            pad=HEATMAP_COLORBAR_PAD,
            aspect=HEATMAP_COLORBAR_ASPECT,
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

    draw_neuron_scale_bar(heatmap_axes[-1, -1])
    return panel


def compute_rowwise_tuning_curve_correlations(
    first_curves: np.ndarray,
    second_curves: np.ndarray,
) -> np.ndarray:
    """Return Pearson correlations between corresponding tuning-curve rows."""
    first_curves = np.asarray(first_curves, dtype=float)
    second_curves = np.asarray(second_curves, dtype=float)
    if first_curves.ndim != 2 or second_curves.ndim != 2:
        raise ValueError("Tuning-curve arrays must both be two-dimensional.")
    if first_curves.shape != second_curves.shape:
        raise ValueError(
            "Tuning-curve arrays must have matching shapes. "
            f"Got {first_curves.shape} and {second_curves.shape}."
        )

    correlations = np.full(first_curves.shape[0], np.nan, dtype=float)
    for row_index, (first_curve, second_curve) in enumerate(
        zip(first_curves, second_curves, strict=True)
    ):
        valid = np.isfinite(first_curve) & np.isfinite(second_curve)
        if np.count_nonzero(valid) < 3:
            continue
        first_values = first_curve[valid]
        second_values = second_curve[valid]
        if (
            np.ptp(first_values) <= 0.0
            or np.ptp(second_values) <= 0.0
        ):
            continue
        correlations[row_index] = float(
            np.corrcoef(first_values, second_values)[0, 1]
        )
    return correlations


def build_turn_tuning_similarity_table(
    panels_by_region: dict[str, dict[tuple[str, str], np.ndarray]],
    ordered_unit_keys_by_region: dict[str, dict[str, np.ndarray]],
    *,
    regions: Sequence[str] = DECODING_COMPARISON_REGIONS,
) -> Any:
    """Return neuron-level same-turn tuning correlations for both regions."""
    import pandas as pd

    records = []
    for region in regions:
        panels = panels_by_region[region]
        ordered_unit_keys = ordered_unit_keys_by_region[region]
        for turn_type, turn_label, first_trajectory, second_trajectory in (
            TURN_PAIR_SPECS
        ):
            first_curves = np.asarray(
                panels[(first_trajectory, first_trajectory)],
                dtype=float,
            )
            second_curves = np.asarray(
                panels[(first_trajectory, second_trajectory)],
                dtype=float,
            )
            unit_keys = np.asarray(
                ordered_unit_keys[first_trajectory],
                dtype=str,
            )
            if first_curves.shape[0] != unit_keys.size:
                raise ValueError(
                    "Tuning-curve rows must match ordered unit keys. "
                    f"Got {first_curves.shape[0]} rows and {unit_keys.size} keys "
                    f"for {region} {first_trajectory}."
                )
            correlations = compute_rowwise_tuning_curve_correlations(
                first_curves,
                second_curves,
            )
            for unit_key, correlation in zip(
                unit_keys,
                correlations,
                strict=True,
            ):
                if not np.isfinite(correlation):
                    continue
                animal_name = str(unit_key).split(":", maxsplit=1)[0]
                records.append(
                    {
                        "region": str(region),
                        "animal_name": animal_name,
                        "turn_type": turn_type,
                        "turn_label": turn_label,
                        "first_trajectory": first_trajectory,
                        "second_trajectory": second_trajectory,
                        "unit_key": str(unit_key),
                        "tuning_correlation": float(correlation),
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "region",
            "animal_name",
            "turn_type",
            "turn_label",
            "first_trajectory",
            "second_trajectory",
            "unit_key",
            "tuning_correlation",
        ],
    )


def load_pooled_turn_tuning_similarity_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
) -> Any:
    """Load Figure 1D curves and derive same-turn neuron correlations."""
    panels_by_region = {}
    ordered_unit_keys_by_region = {}
    for region in DECODING_COMPARISON_REGIONS:
        panels, ordered_unit_keys, _ordered_peak_positions = (
            load_or_compute_panel_d_heatmap_full_payload(
                data_root=data_root,
                datasets=datasets,
                region=region,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                panel_d_cache_dir=panel_d_cache_dir,
                refresh_panel_d_cache=refresh_panel_d_cache,
                require_ordered_unit_keys=True,
            )
        )
        panels_by_region[region] = panels
        ordered_unit_keys_by_region[region] = ordered_unit_keys
    return build_turn_tuning_similarity_table(
        panels_by_region,
        ordered_unit_keys_by_region,
    )


def compute_equal_animal_weights(
    animal_names: Sequence[str],
    strata: Sequence[str] | None = None,
) -> np.ndarray:
    """Return weights equal across animals and optional within-animal strata."""
    animal_names = np.asarray(animal_names, dtype=str)
    if strata is None:
        strata = np.full(animal_names.size, "all", dtype=str)
    else:
        strata = np.asarray(strata, dtype=str)
        if strata.shape != animal_names.shape:
            raise ValueError("Weighting strata must match the animal-name shape.")
    weights = np.zeros(animal_names.size, dtype=float)
    unique_animals = np.unique(animal_names)
    if unique_animals.size == 0:
        return weights
    for animal_name in unique_animals:
        animal_rows = animal_names == animal_name
        animal_strata = np.unique(strata[animal_rows])
        for stratum in animal_strata:
            stratum_rows = animal_rows & (strata == stratum)
            weights[stratum_rows] = 1.0 / (
                unique_animals.size
                * animal_strata.size
                * np.count_nonzero(stratum_rows)
            )
    return weights


def weighted_quantile(
    values: np.ndarray,
    quantiles: Sequence[float],
    weights: np.ndarray,
) -> np.ndarray:
    """Return linearly interpolated weighted empirical quantiles."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("Values and weights must be matching one-dimensional arrays.")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("Quantiles must lie between zero and one.")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.full(quantiles.shape, np.nan, dtype=float)

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    weighted_positions = (
        np.cumsum(weights) - 0.5 * weights
    ) / np.sum(weights)
    return np.interp(
        quantiles,
        weighted_positions,
        values,
        left=values[0],
        right=values[-1],
    )


def _plot_weighted_violin(
    ax: Any,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    center: float,
    color: str,
    y_limits: tuple[float, float],
) -> None:
    """Plot one weighted violin with its weighted median and IQR."""
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return

    if values.size >= 2 and np.ptp(values) > 0.0:
        density_grid = np.linspace(y_limits[0], y_limits[1], 256)
        density = gaussian_kde(values, weights=weights)(density_grid)
        if np.any(np.isfinite(density)) and np.nanmax(density) > 0.0:
            half_width = (
                TURN_DISTRIBUTION_HALF_WIDTH
                * density
                / np.nanmax(density)
            )
            ax.fill_betweenx(
                density_grid,
                center - half_width,
                center + half_width,
                facecolor=color,
                edgecolor=color,
                linewidth=0.45,
                alpha=0.32,
                zorder=1,
            )

    lower_quartile, median, upper_quartile = weighted_quantile(
        values,
        (0.25, 0.50, 0.75),
        weights,
    )
    ax.plot(
        [center, center],
        [lower_quartile, upper_quartile],
        color=color,
        linewidth=1.15,
        solid_capstyle="round",
        zorder=3,
    )
    ax.scatter(
        [center],
        [median],
        s=10,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )


def plot_turn_region_distribution_panel(
    ax: Any,
    table: Any,
    *,
    value_column: str,
    ylabel: str,
    y_limits: tuple[float, float],
    show_legend: bool,
    show_zero_line: bool = False,
    annotation: str | None = None,
    within_animal_weight_columns: Sequence[str] = (),
) -> None:
    """Plot equal-animal-weighted left/right distributions for V1 and CA1."""
    from matplotlib.lines import Line2D

    required_columns = {
        "region",
        "animal_name",
        "turn_type",
        value_column,
    }
    missing_columns = required_columns.difference(table.columns)
    missing_columns.update(
        set(within_animal_weight_columns).difference(table.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Distribution table is missing columns {sorted(missing_columns)!r}."
        )

    region_offsets = {
        region: offset
        for region, offset in zip(
            DECODING_COMPARISON_REGIONS,
            (
                -TURN_DISTRIBUTION_REGION_OFFSET,
                TURN_DISTRIBUTION_REGION_OFFSET,
            ),
            strict=True,
        )
    }
    for turn_index, (turn_type, _label, _first, _second) in enumerate(
        TURN_PAIR_SPECS,
        start=1,
    ):
        for region in DECODING_COMPARISON_REGIONS:
            selected = table.loc[
                (table["turn_type"].astype(str) == turn_type)
                & (table["region"].astype(str) == region)
            ]
            values = np.asarray(selected[value_column], dtype=float)
            finite = np.isfinite(values)
            values = values[finite]
            animal_names = np.asarray(
                selected.loc[finite, "animal_name"],
                dtype=str,
            )
            strata = None
            if within_animal_weight_columns:
                strata = (
                    selected.loc[finite, list(within_animal_weight_columns)]
                    .astype(str)
                    .agg("\x1f".join, axis=1)
                    .to_numpy(dtype=str)
                )
            weights = compute_equal_animal_weights(
                animal_names,
                strata,
            )
            _plot_weighted_violin(
                ax,
                values,
                weights,
                center=float(turn_index) + region_offsets[region],
                color=REGION_COLORS[region],
                y_limits=y_limits,
            )

    if show_zero_line:
        ax.axhline(
            0.0,
            color="0.65",
            linestyle="--",
            linewidth=0.55,
            zorder=0,
        )
    ax.set_xlim(0.55, len(TURN_PAIR_SPECS) + 0.45)
    ax.set_ylim(*y_limits)
    ax.set_xticks(
        np.arange(1, len(TURN_PAIR_SPECS) + 1),
        [turn_label for _turn, turn_label, _first, _second in TURN_PAIR_SPECS],
    )
    ax.set_ylabel(ylabel, fontsize=6.0, labelpad=1.2)
    ax.tick_params(axis="both", labelsize=5.5, length=1.8, pad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if annotation is not None:
        ax.text(
            0.98,
            0.96,
            annotation,
            ha="right",
            va="top",
            fontsize=4.6,
            color="0.35",
            transform=ax.transAxes,
        )
    if show_legend:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=3.4,
                markerfacecolor=REGION_COLORS[region],
                markeredgewidth=0.0,
                label=ORDER_PRESERVATION_REGION_LABELS[region],
            )
            for region in DECODING_COMPARISON_REGIONS
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="lower left",
            fontsize=5.0,
            handletextpad=0.3,
            borderaxespad=0.2,
        )


def plot_pooled_turn_tuning_similarity_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
) -> Any:
    """Load and plot pooled same-turn tuning-curve correlations."""
    table = load_pooled_turn_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=refresh_panel_d_cache,
    )
    plot_turn_region_distribution_panel(
        ax,
        table,
        value_column="tuning_correlation",
        ylabel="Tuning-curve\nPearson r",
        y_limits=TUNING_SIMILARITY_Y_LIMITS,
        show_legend=True,
        show_zero_line=True,
    )
    return table


def load_pooled_same_turn_trial_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
) -> Any:
    """Return pooled lap errors for left- and right-turn cross-path decoding."""
    import pandas as pd

    decoding_datasets = [
        normalize_dataset_id(dataset)
        for dataset in datasets
    ]
    decoding_animals = [
        animal_name
        for animal_name, _date, _epoch in decoding_datasets
    ]
    if not decoding_datasets or len(set(decoding_animals)) != len(
        decoding_animals
    ):
        raise ValueError(
            "Supplementary Figure 1 decoding requires one unique data set "
            f"per animal; received {decoding_datasets!r}."
        )

    tables = []
    for region in DECODING_COMPARISON_REGIONS:
        table = build_decoding_trial_error_table(
            data_root=data_root,
            datasets=decoding_datasets,
            region=region,
            comparisons=SAME_TURN_DECODING_COMPARISONS,
        ).copy()
        table = table.loc[
            table["comparison"].astype(str) == "same_turn_cross_arm"
        ].copy()
        tables.append(table)
    pooled_table = pd.concat(tables, ignore_index=True)

    turn_by_pair = {
        frozenset((first_trajectory, second_trajectory)): (
            turn_type,
            turn_label,
        )
        for turn_type, turn_label, first_trajectory, second_trajectory in (
            TURN_PAIR_SPECS
        )
    }
    turn_types = []
    turn_labels = []
    for encoding_trajectory, decoding_trajectory in zip(
        pooled_table["encoding_trajectory"].astype(str),
        pooled_table["decoding_trajectory"].astype(str),
        strict=True,
    ):
        pair = frozenset((encoding_trajectory, decoding_trajectory))
        if pair not in turn_by_pair:
            raise ValueError(
                "Unexpected same-turn decoding transfer "
                f"{encoding_trajectory!r} to {decoding_trajectory!r}."
            )
        turn_type, turn_label = turn_by_pair[pair]
        turn_types.append(turn_type)
        turn_labels.append(turn_label)
    pooled_table["turn_type"] = turn_types
    pooled_table["turn_label"] = turn_labels

    lap_key_columns = [
        "animal_name",
        "date",
        "epoch",
        "encoding_trajectory",
        "decoding_trajectory",
        "trial_index",
    ]
    lap_keys_by_region = {}
    for region in DECODING_COMPARISON_REGIONS:
        region_table = pooled_table.loc[
            pooled_table["region"].astype(str) == region
        ]
        lap_keys = set(
            region_table[lap_key_columns]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if len(lap_keys) != len(region_table):
            raise ValueError(f"Duplicate same-turn lap keys for region {region!r}.")
        lap_keys_by_region[region] = lap_keys
    reference_region = DECODING_COMPARISON_REGIONS[0]
    reference_lap_keys = lap_keys_by_region[reference_region]
    for region in DECODING_COMPARISON_REGIONS[1:]:
        if lap_keys_by_region[region] != reference_lap_keys:
            raise ValueError(
                "V1 and CA1 same-turn decoding tables do not contain the same "
                f"laps ({reference_region!r} vs {region!r})."
            )

    expected_transfers = {
        (
            region,
            animal_name,
            encoding_trajectory,
            decoding_trajectory,
        )
        for region in DECODING_COMPARISON_REGIONS
        for animal_name in decoding_animals
        for (
            _comparison,
            _label,
            _transfer_family,
            trajectory_pairs,
        ) in SAME_TURN_DECODING_COMPARISONS
        for encoding_trajectory, decoding_trajectory in trajectory_pairs
    }
    observed_transfers = set(
        pooled_table[
            [
                "region",
                "animal_name",
                "encoding_trajectory",
                "decoding_trajectory",
            ]
        ]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed_transfers != expected_transfers:
        missing_transfers = sorted(expected_transfers - observed_transfers)
        extra_transfers = sorted(observed_transfers - expected_transfers)
        raise ValueError(
            "Incomplete same-turn decoding coverage. "
            f"Missing {missing_transfers!r}; extra {extra_transfers!r}."
        )
    return pooled_table


def plot_pooled_same_turn_decoding_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
) -> Any:
    """Load and plot pooled lap-level same-turn decoding errors."""
    table = load_pooled_same_turn_trial_error_table(
        data_root=data_root,
        datasets=datasets,
    )
    plot_turn_region_distribution_panel(
        ax,
        table,
        value_column="trial_median_absolute_error",
        ylabel="Median |normalized\nerror| per lap",
        y_limits=DECODING_ERROR_Y_LIMITS,
        show_legend=False,
        annotation="All eligible units",
        within_animal_weight_columns=(
            "encoding_trajectory",
            "decoding_trajectory",
        ),
    )
    return table


def compute_curve_peak_positions(curves: np.ndarray) -> np.ndarray:
    """Return midpoint peak bins for finite, non-flat tuning curves."""
    return compute_tuning_curve_peak_positions(curves)


def compute_peak_order_preservation_score(
    panel_values: np.ndarray,
    ordered_unit_keys: np.ndarray,
    ordered_peak_positions: np.ndarray,
    animal_name: str,
    *,
    min_units: int = ORDER_PRESERVATION_MIN_UNITS,
) -> tuple[float, int]:
    """Correlate one animal's odd- and even-lap tuning peak positions."""
    from scipy.stats import spearmanr

    panel_values = np.asarray(panel_values, dtype=float)
    ordered_unit_keys = np.asarray(ordered_unit_keys, dtype=str)
    ordered_peak_positions = np.asarray(ordered_peak_positions, dtype=float)
    if panel_values.ndim != 2:
        raise ValueError(
            f"Expected a 2D tuning matrix, got shape {panel_values.shape}."
        )
    if panel_values.shape[0] != ordered_unit_keys.size:
        raise ValueError(
            "Panel rows must match ordered unit keys. "
            f"Got {panel_values.shape[0]} rows and {ordered_unit_keys.size} keys."
        )
    if ordered_peak_positions.shape != (ordered_unit_keys.size,):
        raise ValueError(
            "Odd-lap peak positions must match ordered unit keys. "
            f"Got {ordered_peak_positions.shape} for {ordered_unit_keys.size} keys."
        )
    if min_units < 3:
        raise ValueError("min_units must be at least 3.")

    animal_rows = np.flatnonzero(
        np.char.startswith(ordered_unit_keys, f"{animal_name}:")
    )
    odd_peak_positions = ordered_peak_positions[animal_rows]
    even_peak_positions = compute_curve_peak_positions(panel_values[animal_rows])
    valid = np.isfinite(odd_peak_positions) & np.isfinite(even_peak_positions)
    valid_count = int(np.count_nonzero(valid))
    if (
        valid_count < min_units
        or np.unique(odd_peak_positions[valid]).size < 2
        or np.unique(even_peak_positions[valid]).size < 2
    ):
        return np.nan, valid_count

    score = float(
        spearmanr(
            odd_peak_positions[valid],
            even_peak_positions[valid],
        ).statistic
    )
    return score, valid_count


def build_order_preservation_score_table(
    panels_by_region: dict[str, dict[tuple[str, str], np.ndarray]],
    ordered_unit_keys_by_region: dict[str, dict[str, np.ndarray]],
    ordered_peak_positions_by_region: dict[str, dict[str, np.ndarray]],
    animal_names: Sequence[str],
    *,
    regions: Sequence[str] = ORDER_PRESERVATION_REGIONS,
    trajectory_types: Sequence[str] = PANEL_D_TRAJECTORY_TYPES,
) -> Any:
    """Return animal-level odd/even peak-position preservation scores."""
    import pandas as pd

    records = []
    for region in regions:
        panels = panels_by_region[region]
        ordered_keys = ordered_unit_keys_by_region[region]
        ordered_peak_positions = ordered_peak_positions_by_region[region]
        for animal_name in animal_names:
            for order_trajectory in trajectory_types:
                for plot_trajectory in trajectory_types:
                    score, unit_count = compute_peak_order_preservation_score(
                        panels[(order_trajectory, plot_trajectory)],
                        ordered_keys[order_trajectory],
                        ordered_peak_positions[order_trajectory],
                        animal_name,
                    )
                    records.append(
                        {
                            "region": region,
                            "animal_name": str(animal_name),
                            "order_trajectory": order_trajectory,
                            "plot_trajectory": plot_trajectory,
                            "spearman_rho": score,
                            "n_units": unit_count,
                        }
                    )
    return pd.DataFrame.from_records(records)


def summarize_order_preservation_scores(
    score_table: Any,
    animal_names: Sequence[str],
    *,
    regions: Sequence[str] = ORDER_PRESERVATION_REGIONS,
    trajectory_types: Sequence[str] = PANEL_D_TRAJECTORY_TYPES,
) -> dict[str, np.ndarray]:
    """Return equal-animal mean 4x4 order-preservation matrices."""
    expected_animals = {str(animal_name) for animal_name in animal_names}
    matrices = {
        str(region): np.full(
            (len(trajectory_types), len(trajectory_types)),
            np.nan,
            dtype=float,
        )
        for region in regions
    }
    for region in regions:
        region_values = score_table.loc[
            score_table["region"].astype(str) == str(region)
        ]
        for row_index, order_trajectory in enumerate(trajectory_types):
            for column_index, plot_trajectory in enumerate(trajectory_types):
                cell = region_values.loc[
                    (
                        region_values["order_trajectory"].astype(str)
                        == order_trajectory
                    )
                    & (
                        region_values["plot_trajectory"].astype(str)
                        == plot_trajectory
                    )
                ]
                observed_animals = set(cell["animal_name"].astype(str))
                values = np.asarray(cell["spearman_rho"], dtype=float)
                if (
                    observed_animals == expected_animals
                    and len(cell) == len(expected_animals)
                    and np.all(np.isfinite(values))
                ):
                    matrices[str(region)][row_index, column_index] = float(
                        np.mean(values)
                    )
    return matrices


def load_order_preservation_score_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    refresh_panel_d_cache: bool,
) -> Any:
    """Load Figure 1D cache payloads and derive animal-level order scores."""
    panels_by_region = {}
    ordered_unit_keys_by_region = {}
    ordered_peak_positions_by_region = {}
    for region in ORDER_PRESERVATION_REGIONS:
        panels, ordered_unit_keys, ordered_peak_positions = (
            load_or_compute_panel_d_heatmap_full_payload(
                data_root=data_root,
                datasets=datasets,
                region=region,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                panel_d_cache_dir=panel_d_cache_dir,
                refresh_panel_d_cache=refresh_panel_d_cache,
                require_ordered_unit_keys=True,
                require_order_peak_positions=True,
            )
        )
        panels_by_region[region] = panels
        ordered_unit_keys_by_region[region] = ordered_unit_keys
        ordered_peak_positions_by_region[region] = ordered_peak_positions

    animal_names = [
        normalize_dataset_id(dataset)[0]
        for dataset in datasets
    ]
    return build_order_preservation_score_table(
        panels_by_region,
        ordered_unit_keys_by_region,
        ordered_peak_positions_by_region,
        animal_names,
    )


def plot_order_preservation_panel(
    ax: Any,
    matrices: dict[str, np.ndarray],
) -> None:
    """Plot side-by-side V1 and CA1 odd/even peak-correlation matrices."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    ax.axis("off")
    cmap = plt.get_cmap(ORDER_PRESERVATION_CMAP).copy()
    cmap.set_bad("#D9D9D9")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    matrix_bounds = (
        (0.13, 0.25, 0.34, 0.55),
        (0.53, 0.25, 0.34, 0.55),
    )
    matrix_axes = []
    image = None
    labels = [
        ORDER_PRESERVATION_TRAJECTORY_LABELS[trajectory_type]
        for trajectory_type in PANEL_D_TRAJECTORY_TYPES
    ]
    for region_index, (region, bounds) in enumerate(
        zip(ORDER_PRESERVATION_REGIONS, matrix_bounds, strict=True)
    ):
        matrix_ax = ax.inset_axes(bounds)
        matrix_axes.append(matrix_ax)
        values = np.asarray(matrices[region], dtype=float)
        image = matrix_ax.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        matrix_ax.set_title(
            ORDER_PRESERVATION_REGION_LABELS[region],
            color=REGION_COLORS[region],
            fontsize=6.0,
            pad=1,
        )
        matrix_ax.set_xticks(np.arange(len(labels)))
        matrix_ax.set_xticklabels(
            labels,
            rotation=90,
            fontsize=3.8,
        )
        matrix_ax.set_yticks(np.arange(len(labels)))
        matrix_ax.set_yticklabels(
            labels if region_index == 0 else (),
            fontsize=3.8,
        )
        matrix_ax.tick_params(length=0, pad=0.5)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                label = "—" if not np.isfinite(value) else f"{value:.2f}"
                matrix_ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=3.5,
                    color=(
                        "white"
                        if np.isfinite(value) and abs(value) >= 0.55
                        else "black"
                    ),
                )
        for start in (0, 2):
            matrix_ax.add_patch(
                Rectangle(
                    (start - 0.5, start - 0.5),
                    2.0,
                    2.0,
                    fill=False,
                    edgecolor=PANEL_D_HEATMAP_BLOCK_OUTLINE_COLOR,
                    linewidth=0.6 * PANEL_D_HEATMAP_BLOCK_OUTLINE_LINEWIDTH,
                    clip_on=False,
                )
            )

    ax.text(
        0.50,
        0.02,
        "Even-lap tuning",
        ha="center",
        va="bottom",
        fontsize=4.8,
        transform=ax.transAxes,
    )
    ax.text(
        0.015,
        0.525,
        "Odd-lap order",
        ha="center",
        va="center",
        rotation=90,
        fontsize=4.8,
        transform=ax.transAxes,
    )
    if image is not None:
        colorbar_ax = ax.inset_axes([0.91, 0.31, 0.025, 0.42])
        colorbar = ax.figure.colorbar(
            image,
            cax=colorbar_ax,
            ticks=[-1.0, 0.0, 1.0],
        )
        colorbar.ax.tick_params(labelsize=3.8, length=1.5, pad=1)
        colorbar.set_label("Mean ρ", fontsize=4.5, labelpad=1)
        colorbar.outline.set_linewidth(0.4)


def build_ca1_decoding_significance_brackets(
    per_animal_results: Any,
    *,
    animal_names: Sequence[str] = PANEL_H_DECODING_ANIMALS,
    contrasts: Sequence[tuple[str, str, float]] = DECODING_SIGNIFICANCE_CONTRASTS,
) -> tuple[tuple[float, float, float, str], ...]:
    """Return brackets only for a consistent significant same-turn advantage."""
    expected_animals = {str(animal_name) for animal_name in animal_names}
    comparison_positions = {
        comparison: float(index)
        for index, (comparison, _label, _family, _pairs) in enumerate(
            DECODING_CROSS_TRAJECTORY_COMPARISONS,
            start=1,
        )
    }
    brackets = []
    for comparison_a, comparison_b, y in contrasts:
        selected = per_animal_results.loc[
            (
                per_animal_results["comparison_a"].astype(str)
                == comparison_a
            )
            & (
                per_animal_results["comparison_b"].astype(str)
                == comparison_b
            )
        ]
        observed_animals = set(selected["animal_name"].astype(str))
        if (
            observed_animals != expected_animals
            or len(selected) != len(expected_animals)
        ):
            raise ValueError(
                "Expected exactly one CA1 permutation result per animal for "
                f"{comparison_a!r} vs {comparison_b!r}."
            )
        p_values = np.asarray(selected["p_two_sided"], dtype=float)
        median_differences = np.asarray(
            selected["median_difference"],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(p_values))
            or np.any((p_values < 0.0) | (p_values > 1.0))
            or not np.all(np.isfinite(median_differences))
        ):
            raise ValueError(
                "CA1 permutation results must be finite with valid p-values."
            )
        if not np.all(median_differences < 0.0):
            continue
        aggregate_p_value = float(np.max(p_values))
        label = significance_stars(aggregate_p_value)
        if label == "n.s.":
            continue
        brackets.append(
            (
                comparison_positions[comparison_a],
                comparison_positions[comparison_b],
                float(y),
                label,
            )
        )
    return tuple(brackets)


def plot_cross_path_decoding_region_comparison_panel(
    ax: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    n_permutations: int,
    permutation_seed: int,
) -> Any:
    """Plot V1 and CA1 Figure 1G summaries with separate regional inference."""
    import pandas as pd

    decoding_datasets = [
        normalize_dataset_id(dataset)
        for dataset in datasets
    ]
    decoding_animals = [
        animal_name
        for animal_name, _date, _epoch in decoding_datasets
    ]
    if not decoding_datasets or len(set(decoding_animals)) != len(
        decoding_animals
    ):
        raise ValueError(
            "Supplementary Figure 1 decoding requires one unique data set "
            f"per animal; received {decoding_datasets!r}."
        )

    decoding_error_tables = {
        region: load_decoding_absolute_error_table(
            data_root=data_root,
            datasets=decoding_datasets,
            region=region,
        )
        for region in DECODING_COMPARISON_REGIONS
    }
    decoding_error_table = pd.concat(
        list(decoding_error_tables.values()),
        axis=0,
        ignore_index=True,
    )
    permutation_results = {}
    for region in DECODING_COMPARISON_REGIONS:
        trial_error_table = build_decoding_trial_error_table(
            data_root=data_root,
            datasets=decoding_datasets,
            region=region,
        )
        permutation_results[region] = compute_decoding_permutation_tests(
            trial_error_table,
            n_permutations=n_permutations,
            seed=permutation_seed,
        )

    plot_decoding_error_panel(
        ax,
        decoding_error_table,
        comparisons=CA1_DECODING_COMPARISONS,
        significance_brackets=(),
        regions=DECODING_COMPARISON_REGIONS,
        show_region_legend=True,
        show_median_labels=False,
        xtick_label_fontsize=CA1_DECODING_XTICK_LABEL_FONTSIZE,
    )
    return permutation_results


def add_ca1_heatmap_title(
    fig: Any,
    corner_axis: Any,
    tuning_schematic_axes: Sequence[Any],
) -> Any:
    """Add the CA1 heatmap title on the Panel D header baseline."""
    boxes = [axis.get_position() for axis in tuning_schematic_axes]
    title_x = (min(box.x0 for box in boxes) + max(box.x1 for box in boxes)) / 2
    title_y = fig.transFigure.inverted().transform(
        corner_axis.transAxes.transform((0.0, PANEL_D_LABEL_Y))
    )[1]
    return fig.text(
        title_x,
        title_y,
        CA1_HEATMAP_TITLE,
        ha="center",
        va="bottom",
        fontsize=8,
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
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    panel_d_cache_dir: Path | None = None,
    refresh_panel_d_cache: bool = False,
) -> Path:
    """Build and save Supplementary Figure 1."""
    import matplotlib.pyplot as plt

    dark_movement_fr_cache_dir = (
        Path(output_path).parent / "cache"
        if dark_movement_fr_cache_dir is None
        else Path(dark_movement_fr_cache_dir)
    )
    panel_d_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_d_cache_dir is None
        else Path(panel_d_cache_dir)
    )
    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            DEFAULT_FIGURE_HEIGHT_MM,
        ),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[
            DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
            DEFAULT_CA1_HEATMAP_ROW_HEIGHT_MM,
        ],
        hspace=0.08,
    )
    moved_grid = outer_grid[0, 0].subgridspec(
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

    lower_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[
            DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
            DEFAULT_PANEL_E_WIDTH_FRACTION,
        ],
        wspace=BOTTOM_ROW_PANEL_WSPACE,
    )
    lower_right_grid = lower_grid[0, 1].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=LOWER_ROW_RIGHT_HEIGHT_RATIOS,
        hspace=LOWER_ROW_RIGHT_HSPACE,
    )
    tuning_similarity_axis = fig.add_subplot(lower_right_grid[0, 0])
    plot_pooled_turn_tuning_similarity_panel(
        tuning_similarity_axis,
        data_root=data_root,
        datasets=datasets,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=refresh_panel_d_cache,
    )
    tuning_similarity_axis.set_title(
        TUNING_SIMILARITY_TITLE,
        fontsize=8,
        pad=2,
    )

    ca1_panel = plot_pooled_ca1_dark_heatmap_panel(
        fig,
        lower_grid[0, 0],
        data_root=data_root,
        datasets=datasets,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=False,
    )
    decoding_axis = fig.add_subplot(lower_right_grid[1, 0])
    plot_pooled_same_turn_decoding_panel(
        decoding_axis,
        data_root=data_root,
        datasets=datasets,
    )
    decoding_axis.set_title(
        DECODING_COMPARISON_TITLE,
        fontsize=8,
        pad=2,
    )

    corner_axis = ca1_panel["corner_axis"]
    tuning_schematic_axes = ca1_panel["tuning_schematic_axes"]
    order_schematic_axes = ca1_panel["order_schematic_axes"]
    heatmap_axes = ca1_panel["heatmap_axes"]

    fig.canvas.draw()
    add_panel_d_heatmap_block_outlines(heatmap_axes)
    fig.set_constrained_layout(False)
    label_axis(corner_axis, "D", x=PANEL_D_LABEL_X, y=PANEL_D_LABEL_Y)
    label_axis(tuning_similarity_axis, "E", x=-0.03, y=1.01)
    label_axis(decoding_axis, "F", x=-0.03, y=1.01)
    add_ca1_heatmap_title(fig, corner_axis, tuning_schematic_axes)
    add_centered_axis_text(
        fig,
        order_schematic_axes,
        "Order",
        y_offset=HEATMAP_ORDER_LABEL_OFFSET,
        rotation=90,
        fontsize=PANEL_E_AXIS_LABEL_FONTSIZE,
    )
    add_centered_below_axis_text(
        fig,
        heatmap_axes[-1, :],
        TASK_PROGRESSION_XLABEL,
        y_offset=HEATMAP_PATH_LABEL_OFFSET,
        fontsize=PANEL_E_AXIS_LABEL_FONTSIZE,
    )

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
        "--panel-d-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel D V1/CA1 heatmap matrices. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-d-cache",
        action="store_true",
        help=(
            "Recompute the Panel D V1/CA1 heatmaps and overwrite their caches even "
            "when a matching cache exists."
        ),
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
        help=(
            "Number of leading position samples to ignore. "
            f"Default: {DEFAULT_POSITION_OFFSET}"
        ),
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
    panel_d_cache_dir = (
        args.panel_d_cache_dir
        if args.panel_d_cache_dir is not None
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
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_d_cache_dir=panel_d_cache_dir,
        refresh_panel_d_cache=args.refresh_panel_d_cache,
    )


if __name__ == "__main__":
    main()
