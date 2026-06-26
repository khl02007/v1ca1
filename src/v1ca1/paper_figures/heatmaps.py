from __future__ import annotations

"""Generate side-by-side Figure 1D and Figure 3B heatmap comparisons."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_POSITION_BIN_COUNT,
    HEATMAP_COLORBAR_LABEL_FONTSIZE,
    HEATMAP_COLORBAR_LABELPAD,
    PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    PANEL_D_FIRING_RATE_NORMALIZATION,
    PANEL_D_HEATMAP_CMAP,
    PANEL_D_LINEAR_POSITION_ORIENTATION,
    PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    PANEL_D_TRAJECTORY_TYPES,
    PANEL_E_AXIS_LABEL_FONTSIZE,
    PANEL_E_TRAJECTORY_COLORS,
    TASK_PROGRESSION_XLABEL,
    add_centered_below_axis_text,
    add_centered_axis_text,
    align_panel_values_to_unit_order,
    build_unit_keys,
    draw_order_schematic,
    draw_neuron_scale_bar,
    extract_tuning_curve_arrays,
    load_or_compute_panel_d_heatmap_payload,
    load_or_compute_panel_d_heatmap_panels,
    normalize_panel_values_across_trajectories,
    normalize_panel_values_per_trajectory,
    plot_pooled_heatmap_grid,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    FIGURE_FORMATS,
    PANEL_B_HEATMAP_CMAP,
    PANEL_B_LINEAR_POSITION_ORIENTATION,
    PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_B_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_TRAJECTORY_TYPES,
    build_output_path,
    compute_light_epoch_tuning_curves,
    get_tuning_similarity_path,
    load_or_compute_panel_b_heatmap_panels,
    parse_dataset_id,
)
from v1ca1.paper_figures.style import apply_paper_style, figure_size, save_figure
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_NAME = "heatmaps"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_FIGURE_WIDTH_MM = 220.0
DEFAULT_FIGURE_HEIGHT_MM = 510.0
HEATMAP_GRID_MARGINS = {
    "left": 0.085,
    "right": 0.920,
    "top": 0.955,
    "bottom": 0.065,
    "hspace": 0.30,
    "wspace": 0.16,
}
HEATMAP_COLORBAR_AXIS_BOUNDS = (0.945, 0.34, 0.012, 0.32)
LIGHT_ORDER_ORIGINAL = "light_order"
LIGHT_ORDER_DARK = "dark_order"
DARK_DPP_FILTER_THRESHOLD = 0.5
DARK_DPP_CORRELATION_FILTER = "correlation"
DARK_DPP_OVERLAP_FILTER = "absolute_overlap"
DARK_DPP_COMPARISON_LABELS = ("left_turn", "right_turn")
HEATMAP_ROW_SPECS = (
    (
        "Light order; per unit, per trajectory",
        PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        LIGHT_ORDER_ORIGINAL,
        None,
    ),
    (
        "Light order; per unit, across four subpanels",
        PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        LIGHT_ORDER_ORIGINAL,
        None,
    ),
    (
        "Dark order; per unit, per trajectory",
        PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        LIGHT_ORDER_DARK,
        None,
    ),
    (
        "Dark order; per unit, across four subpanels",
        PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        LIGHT_ORDER_DARK,
        None,
    ),
    (
        "Dark DPP corr. > 0.5; dark order",
        PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        LIGHT_ORDER_DARK,
        DARK_DPP_CORRELATION_FILTER,
    ),
    (
        "Dark DPP overlap > 0.5; dark order",
        PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
        LIGHT_ORDER_DARK,
        DARK_DPP_OVERLAP_FILTER,
    ),
)
HEATMAP_COLUMN_TITLES = (
    "Fig. 1D dark heatmaps",
    "Fig. 3B light heatmaps",
)
HEATMAP_DARK_ORDER_LIGHT_TITLE = "Fig. 3B light heatmaps, dark order"
HEATMAP_TITLE_FONTSIZE = 8.0
HEATMAP_ROW_LABEL_FONTSIZE = 7.2
HEATMAP_AXIS_LABEL_FONTSIZE = PANEL_E_AXIS_LABEL_FONTSIZE


def setup_heatmap_comparison_panel(
    fig: Any,
    grid_spec: Any,
    *,
    trajectory_types: Sequence[str],
    fill_track: bool,
) -> dict[str, Any]:
    """Create one 4-by-4 heatmap grid with trajectory schematics."""
    trajectory_types = tuple(trajectory_types)
    heatmap_grid = grid_spec.subgridspec(
        nrows=len(trajectory_types) + 1,
        ncols=len(trajectory_types) + 1,
        height_ratios=[0.42, *([1.0] * len(trajectory_types))],
        width_ratios=[0.48, *([1.0] * len(trajectory_types))],
    )
    axes = np.asarray(
        [
            [
                fig.add_subplot(heatmap_grid[row, col])
                for col in range(len(trajectory_types) + 1)
            ]
            for row in range(len(trajectory_types) + 1)
        ],
        dtype=object,
    )

    corner_axis = axes[0, 0]
    corner_axis.axis("off")
    tuning_schematic_axes = axes[0, 1:]
    order_schematic_axes = axes[1:, 0]
    heatmap_axes = axes[1:, 1:]

    for ax, trajectory_type in zip(tuning_schematic_axes, trajectory_types, strict=True):
        draw_w_track_schematic(
            ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            fill_track=fill_track,
        )
    for ax, trajectory_type in zip(order_schematic_axes, trajectory_types, strict=True):
        draw_order_schematic(
            ax,
            trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            fill_track=fill_track,
        )
    return {
        "corner_axis": corner_axis,
        "tuning_schematic_axes": tuning_schematic_axes,
        "order_schematic_axes": order_schematic_axes,
        "heatmap_axes": heatmap_axes,
    }


def _concatenate_unit_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate pooled unit-key chunks."""
    if not parts:
        return np.asarray([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def _concatenate_value_parts(parts: list[np.ndarray], position_bin_count: int) -> np.ndarray:
    """Concatenate pooled tuning-matrix chunks."""
    if not parts:
        return np.empty((0, position_bin_count), dtype=float)
    return np.vstack(parts)


def build_light_curve_sets(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> list[dict[str, Any]]:
    """Compute light-epoch tuning curves for dark-order comparisons."""
    curve_sets = []
    for animal_name, date, _dark_epoch in datasets:
        curve_sets.append(
            compute_light_epoch_tuning_curves(
                animal_name=str(animal_name),
                date=str(date),
                data_root=data_root,
                region=region,
                light_epoch=light_epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                use_trajectory_direction=True,
                min_movement_firing_rate_hz=PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ,
                min_tuning_stability_correlation=PANEL_B_MIN_TUNING_STABILITY_CORRELATION,
            )
        )
    return curve_sets


def build_light_panels_in_dark_order(
    curve_sets: Sequence[dict[str, Any]],
    *,
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    position_bin_count: int,
    trajectory_types: Sequence[str],
    firing_rate_normalization: str,
) -> dict[tuple[str, str], np.ndarray]:
    """Return light-epoch heatmaps aligned to the dark-epoch row order."""
    trajectory_types = tuple(trajectory_types)
    even_units_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in trajectory_types
    }
    even_values_by_trajectory: dict[str, list[np.ndarray]] = {
        trajectory_type: [] for trajectory_type in trajectory_types
    }

    for curve_set in curve_sets:
        animal_name = str(curve_set["animal_name"])
        date = str(curve_set["date"])
        region = str(curve_set["region"])
        for trajectory_type in trajectory_types:
            even_curve = curve_set["even_curves"].get(trajectory_type)
            if even_curve is None:
                continue
            units, values = extract_tuning_curve_arrays(even_curve)
            even_units_by_trajectory[trajectory_type].append(
                build_unit_keys(animal_name, date, region, units)
            )
            even_values_by_trajectory[trajectory_type].append(values)

    panels: dict[tuple[str, str], np.ndarray] = {}
    for order_trajectory in trajectory_types:
        reference_units = np.asarray(
            ordered_unit_keys_by_trajectory.get(
                order_trajectory,
                np.asarray([], dtype=object),
            ),
            dtype=object,
        )
        unit_order = np.arange(reference_units.size, dtype=int)
        sorted_values_by_plot_trajectory: dict[str, np.ndarray] = {}
        for plot_trajectory in trajectory_types:
            display_units = _concatenate_unit_parts(even_units_by_trajectory[plot_trajectory])
            display_values = _concatenate_value_parts(
                even_values_by_trajectory[plot_trajectory],
                position_bin_count,
            )
            if reference_units.size == 0 or display_units.size == 0:
                sorted_values_by_plot_trajectory[plot_trajectory] = np.full(
                    (reference_units.size, position_bin_count),
                    np.nan,
                    dtype=float,
                )
                continue
            sorted_values_by_plot_trajectory[plot_trajectory] = (
                align_panel_values_to_unit_order(
                    display_values,
                    display_units,
                    reference_units,
                    unit_order,
                )
            )

        if (
            firing_rate_normalization
            == PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION
        ):
            normalized_values_by_plot_trajectory = (
                normalize_panel_values_across_trajectories(
                    sorted_values_by_plot_trajectory
                )
            )
        else:
            normalized_values_by_plot_trajectory = {
                trajectory_type: normalize_panel_values_per_trajectory(values)
                for trajectory_type, values in sorted_values_by_plot_trajectory.items()
            }
        for plot_trajectory in trajectory_types:
            panels[(order_trajectory, plot_trajectory)] = (
                normalized_values_by_plot_trajectory[plot_trajectory]
            )
    return panels


def load_dark_dpp_filtered_unit_keys(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    similarity_metric: str,
    threshold: float = DARK_DPP_FILTER_THRESHOLD,
) -> set[str]:
    """Return pooled unit keys whose max same-turn dark DPP exceeds threshold."""
    import pandas as pd

    unit_key_parts: list[np.ndarray] = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        path = get_tuning_similarity_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=dark_epoch,
            similarity_metric=similarity_metric,
        )
        if path.exists():
            table = pd.read_parquet(path)
        else:
            from v1ca1.paper_figures.figure_4 import (
                _compute_similarity_from_saved_curves,
            )

            table = _compute_similarity_from_saved_curves(
                data_root,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_epoch,
                region=region,
                tuning_similarity_metric=similarity_metric,
            )
        missing_columns = [
            column
            for column in ("unit", "region", "epoch", "comparison_label", "similarity")
            if column not in table.columns
        ]
        if missing_columns:
            raise ValueError(
                f"DPP similarity table {path} is missing columns {missing_columns!r}."
            )
        rows = table[
            (table["region"].astype(str) == str(region))
            & (table["epoch"].astype(str) == str(dark_epoch))
            & (
                table["comparison_label"]
                .astype(str)
                .isin(DARK_DPP_COMPARISON_LABELS)
            )
        ].copy()
        rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
        rows["similarity"] = pd.to_numeric(rows["similarity"], errors="coerce")
        rows = rows[
            np.isfinite(rows["unit"].to_numpy(dtype=float))
            & np.isfinite(rows["similarity"].to_numpy(dtype=float))
        ].copy()
        if rows.empty:
            continue
        rows["unit"] = rows["unit"].astype(int)
        rows["comparison_label"] = rows["comparison_label"].astype(str)
        selected_rows = (
            rows.sort_values(
                ["unit", "similarity", "comparison_label"],
                ascending=[True, False, True],
            )
            .drop_duplicates("unit", keep="first")
            .loc[lambda frame: frame["similarity"] > float(threshold)]
        )
        if selected_rows.empty:
            continue
        unit_key_parts.append(
            build_unit_keys(
                animal_name,
                date,
                region,
                selected_rows["unit"].to_numpy(dtype=int),
            )
        )
    return set(_concatenate_unit_parts(unit_key_parts).astype(str).tolist())


def filter_heatmap_panels_to_unit_keys(
    panels: dict[tuple[str, str], np.ndarray],
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    selected_unit_keys: set[str],
    *,
    trajectory_types: Sequence[str],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """Keep only selected row-unit keys while preserving dark-order alignment."""
    trajectory_types = tuple(trajectory_types)
    selected_keys = {str(unit_key) for unit_key in selected_unit_keys}
    filtered_panels: dict[tuple[str, str], np.ndarray] = {}
    filtered_unit_keys_by_trajectory: dict[str, np.ndarray] = {}
    for order_trajectory in trajectory_types:
        reference_units = np.asarray(
            ordered_unit_keys_by_trajectory.get(
                order_trajectory,
                np.asarray([], dtype=object),
            ),
            dtype=object,
        )
        keep_mask = np.asarray(
            [str(unit_key) in selected_keys for unit_key in reference_units],
            dtype=bool,
        )
        filtered_unit_keys_by_trajectory[order_trajectory] = reference_units[keep_mask]
        for plot_trajectory in trajectory_types:
            values = np.asarray(
                panels[(order_trajectory, plot_trajectory)],
                dtype=float,
            )
            if values.shape[0] != reference_units.size:
                raise ValueError(
                    "Heatmap panel row count does not match ordered unit keys for "
                    f"{order_trajectory!r}."
                )
            filtered_panels[(order_trajectory, plot_trajectory)] = values[keep_mask, :]
    return filtered_panels, filtered_unit_keys_by_trajectory


def _annotate_heatmap_panel(
    fig: Any,
    panel: dict[str, Any],
    *,
    title: str,
) -> None:
    """Add the shared labels around one heatmap panel."""
    add_centered_axis_text(
        fig,
        panel["tuning_schematic_axes"],
        title,
        y_offset=0.004,
        fontsize=HEATMAP_TITLE_FONTSIZE,
    )
    add_centered_axis_text(
        fig,
        panel["tuning_schematic_axes"],
        "Tuning",
        y_offset=-0.003,
        fontsize=HEATMAP_AXIS_LABEL_FONTSIZE,
    )
    add_centered_axis_text(
        fig,
        panel["order_schematic_axes"],
        "Order",
        y_offset=0.006,
        rotation=90,
        fontsize=HEATMAP_AXIS_LABEL_FONTSIZE,
    )
    add_centered_below_axis_text(
        fig,
        panel["heatmap_axes"][-1, :],
        TASK_PROGRESSION_XLABEL,
        y_offset=0.013,
        fontsize=HEATMAP_AXIS_LABEL_FONTSIZE,
    )


def _add_row_label(fig: Any, axes: np.ndarray, label: str) -> None:
    """Add a left-side label for one normalization row."""
    boxes = [ax.get_position() for ax in axes.ravel()]
    x0 = min(box.x0 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    fig.text(
        x0 - 0.075,
        (y0 + y1) / 2.0,
        label,
        ha="right",
        va="center",
        rotation=90,
        fontsize=HEATMAP_ROW_LABEL_FONTSIZE,
    )


def make_heatmaps_figure(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_d_cache_dir: Path | None,
    panel_b_cache_dir: Path | None,
    refresh_panel_cache: bool,
    dpi: int,
) -> Path:
    """Build and save the heatmap normalization comparison figure."""
    import matplotlib.pyplot as plt

    region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_d_cache_dir = (
        output_path.parent / "cache"
        if panel_d_cache_dir is None
        else Path(panel_d_cache_dir)
    )
    panel_b_cache_dir = (
        output_path.parent / "cache"
        if panel_b_cache_dir is None
        else Path(panel_b_cache_dir)
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=False,
    )
    outer_grid = fig.add_gridspec(
        nrows=len(HEATMAP_ROW_SPECS),
        ncols=len(HEATMAP_COLUMN_TITLES),
        **HEATMAP_GRID_MARGINS,
    )

    all_heatmap_axes = []
    color_image = None
    row_axes_by_index: list[np.ndarray] = []
    panel_records: list[tuple[dict[str, Any], str]] = []
    dark_payloads_by_normalization: dict[
        str,
        tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]],
    ] = {}
    dark_order_light_curve_sets: list[dict[str, Any]] | None = None
    dpp_unit_keys_by_metric: dict[str, set[str]] = {}

    def _get_dark_payload(
        normalization: str,
    ) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
        if normalization not in dark_payloads_by_normalization:
            dark_payloads_by_normalization[normalization] = (
                load_or_compute_panel_d_heatmap_payload(
                    data_root=data_root,
                    datasets=datasets,
                    region=region,
                    position_bin_count=position_bin_count,
                    position_offset=position_offset,
                    speed_threshold_cm_s=speed_threshold_cm_s,
                    sigma_bins=sigma_bins,
                    panel_d_cache_dir=panel_d_cache_dir,
                    refresh_panel_d_cache=refresh_panel_cache,
                    firing_rate_normalization=normalization,
                    require_ordered_unit_keys=True,
                )
            )
        return dark_payloads_by_normalization[normalization]

    def _get_dpp_unit_keys(similarity_metric: str) -> set[str]:
        if similarity_metric not in dpp_unit_keys_by_metric:
            dpp_unit_keys_by_metric[similarity_metric] = (
                load_dark_dpp_filtered_unit_keys(
                    data_root=data_root,
                    datasets=datasets,
                    region=region,
                    similarity_metric=similarity_metric,
                    threshold=DARK_DPP_FILTER_THRESHOLD,
                )
            )
        return dpp_unit_keys_by_metric[similarity_metric]

    for row_index, (
        _row_label,
        normalization,
        light_order,
        dpp_filter_metric,
    ) in enumerate(HEATMAP_ROW_SPECS):
        row_heatmap_axes = []
        dark_panel = setup_heatmap_comparison_panel(
            fig,
            outer_grid[row_index, 0],
            trajectory_types=PANEL_D_TRAJECTORY_TYPES,
            fill_track=True,
        )
        filtered_dark_unit_keys_by_trajectory = None
        if dpp_filter_metric is None:
            dark_panels = load_or_compute_panel_d_heatmap_panels(
                data_root=data_root,
                datasets=datasets,
                region=region,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                panel_d_cache_dir=panel_d_cache_dir,
                refresh_panel_d_cache=refresh_panel_cache,
                firing_rate_normalization=normalization,
            )
        else:
            dark_panels, dark_unit_keys_by_trajectory = _get_dark_payload(normalization)
            dark_panels, filtered_dark_unit_keys_by_trajectory = (
                filter_heatmap_panels_to_unit_keys(
                    dark_panels,
                    dark_unit_keys_by_trajectory,
                    _get_dpp_unit_keys(dpp_filter_metric),
                    trajectory_types=PANEL_D_TRAJECTORY_TYPES,
                )
            )
        color_image = plot_pooled_heatmap_grid(
            dark_panel["heatmap_axes"],
            dark_panels,
            trajectory_types=PANEL_D_TRAJECTORY_TYPES,
            axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
            cmap=PANEL_D_HEATMAP_CMAP,
        )
        panel_records.append((dark_panel, HEATMAP_COLUMN_TITLES[0]))
        draw_neuron_scale_bar(dark_panel["heatmap_axes"][-1, -1])
        row_heatmap_axes.extend(dark_panel["heatmap_axes"].ravel().tolist())
        all_heatmap_axes.extend(dark_panel["heatmap_axes"].ravel().tolist())

        light_panel = setup_heatmap_comparison_panel(
            fig,
            outer_grid[row_index, 1],
            trajectory_types=PANEL_B_TRAJECTORY_TYPES,
            fill_track=False,
        )
        if light_order == LIGHT_ORDER_DARK:
            if filtered_dark_unit_keys_by_trajectory is None:
                _dark_panels, ordered_unit_keys_by_trajectory = _get_dark_payload(
                    PANEL_D_FIRING_RATE_NORMALIZATION
                )
            else:
                ordered_unit_keys_by_trajectory = filtered_dark_unit_keys_by_trajectory
            if dark_order_light_curve_sets is None:
                dark_order_light_curve_sets = build_light_curve_sets(
                    data_root=data_root,
                    datasets=datasets,
                    region=region,
                    light_epoch=light_epoch,
                    position_bin_count=position_bin_count,
                    position_offset=position_offset,
                    speed_threshold_cm_s=speed_threshold_cm_s,
                    sigma_bins=sigma_bins,
                )
            light_panels = build_light_panels_in_dark_order(
                dark_order_light_curve_sets,
                ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
                position_bin_count=position_bin_count,
                trajectory_types=PANEL_B_TRAJECTORY_TYPES,
                firing_rate_normalization=normalization,
            )
            light_title = HEATMAP_DARK_ORDER_LIGHT_TITLE
        else:
            light_panels = load_or_compute_panel_b_heatmap_panels(
                data_root=data_root,
                datasets=datasets,
                region=region,
                light_epoch=light_epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                panel_b_cache_dir=panel_b_cache_dir,
                refresh_panel_b_cache=refresh_panel_cache,
                firing_rate_normalization=normalization,
            )
            light_title = HEATMAP_COLUMN_TITLES[1]
        light_image = plot_pooled_heatmap_grid(
            light_panel["heatmap_axes"],
            light_panels,
            trajectory_types=PANEL_B_TRAJECTORY_TYPES,
            axis_orientation=PANEL_B_LINEAR_POSITION_ORIENTATION,
            cmap=PANEL_B_HEATMAP_CMAP,
        )
        if color_image is None:
            color_image = light_image
        panel_records.append((light_panel, light_title))
        draw_neuron_scale_bar(light_panel["heatmap_axes"][-1, -1])
        row_heatmap_axes.extend(light_panel["heatmap_axes"].ravel().tolist())
        all_heatmap_axes.extend(light_panel["heatmap_axes"].ravel().tolist())
        row_axes_by_index.append(np.asarray(row_heatmap_axes, dtype=object))

    if color_image is not None:
        colorbar_axis = fig.add_axes(HEATMAP_COLORBAR_AXIS_BOUNDS)
        colorbar = fig.colorbar(
            color_image,
            cax=colorbar_axis,
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

    for panel, title in panel_records:
        _annotate_heatmap_panel(fig, panel, title=title)

    for row_axes, (row_label, _normalization, _light_order, _dpp_filter_metric) in zip(
        row_axes_by_index,
        HEATMAP_ROW_SPECS,
        strict=True,
    ):
        _add_row_label(fig, row_axes, row_label)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved heatmaps figure to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for heatmap comparison generation."""
    parser = argparse.ArgumentParser(
        description="Generate side-by-side Figure 1D and Figure 3B heatmap comparisons."
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
        help="Light run epoch for Figure 3B heatmaps. Default: registry value.",
    )
    parser.add_argument(
        "--panel-d-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for Figure 1D heatmap cache files. "
            "Default: output-dir/cache."
        ),
    )
    parser.add_argument(
        "--panel-b-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for Figure 3B heatmap cache files. "
            "Default: output-dir/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-cache",
        action="store_true",
        help="Recompute heatmap panels and overwrite matching cache files.",
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
    """Run heatmap comparison generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_heatmaps_figure(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_d_cache_dir=args.panel_d_cache_dir,
        panel_b_cache_dir=args.panel_b_cache_dir,
        refresh_panel_cache=args.refresh_panel_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
