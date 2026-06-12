from __future__ import annotations

"""Generate Supplementary Figure 3 cvPCA and motor-control panels."""

import argparse
import hashlib
import json
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
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
    PANEL_D_HEATMAP_CMAP,
    align_panel_values_to_unit_order,
    build_normalized_position_bins,
    build_unit_keys,
    compute_unit_movement_firing_rates,
    extract_tuning_curve_arrays,
    get_stability_table_path,
    load_or_compute_panel_d_heatmap_payload,
    normalize_linear_position_by_trajectory,
    normalize_panel_values_per_trajectory,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_3_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_3_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PANEL_AB_HEIGHT_MM,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    FIGURE_FORMATS,
    PANEL_B_LINEAR_POSITION_ORIENTATION,
    PANEL_B_TRAJECTORY_TYPES,
    add_centered_axis_text,
    add_segment_boundary_lines,
    build_output_path,
    get_dark_epoch,
    get_light_epoch,
    parse_dataset_id,
    plot_pooled_heatmap_grid,
)
from v1ca1.motor.compare_epoch_motor_behavior import (
    MOTOR_VARIABLES,
    VARIABLE_LABELS,
)
from v1ca1.raster.plot_place_field_heatmap import (
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    prepare_heatmap_session,
    smooth_tuning_curve_nan_aware,
)
from v1ca1.paper_figures.style import (
    COMPACT_HISTOGRAM_KWARGS,
    EPOCH_TYPE_COLORS,
    NEUTRAL_COLORS,
    TRAJECTORY_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "supplementary_figure_3"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_3_WIDTH_MM
DEFAULT_SECTION_SPACER_MM = 10.0
DEFAULT_BOTTOM_SECTION_SPACER_MM = 22.0
PANEL_A_CV_PCA_SIZE_FRACTION = 0.40
DEFAULT_REORDERED_HEATMAP_HEIGHT_MM = (
    DEFAULT_PANEL_AB_HEIGHT_MM * PANEL_A_CV_PCA_SIZE_FRACTION
)
DEFAULT_MOTOR_GRID_HEIGHT_MM = FIGURE_3_HEIGHT_MM + 55.0
DEFAULT_MOTOR_SUMMARY_HEIGHT_MM = 35.0
DEFAULT_FIGURE_HEIGHT_MM = (
    DEFAULT_REORDERED_HEATMAP_HEIGHT_MM
    + DEFAULT_SECTION_SPACER_MM
    + DEFAULT_MOTOR_GRID_HEIGHT_MM
    + DEFAULT_BOTTOM_SECTION_SPACER_MM
    + DEFAULT_MOTOR_SUMMARY_HEIGHT_MM
)
MOTOR_GRID_HSPACE = 0.28
MOTOR_GRID_WSPACE = 0.18
MOTOR_SUMMARY_GRID_WSPACE = 0.24
MOTOR_PANEL_ANIMAL_NAME = "L14"
MOTOR_PANEL_LIGHT_EPOCH = "02_r1"
MOTOR_PANEL_RELATIVE_PATH = Path("motor") / "epoch_motor_progression_summary.parquet"
MOTOR_PANEL_COLUMNS = (
    "epoch",
    "trajectory_type",
    "variable",
    "progression_bin_index",
    "progression_bin_center",
    "median",
    "q25",
    "q75",
)
MOTOR_PANEL_EPOCH_COLORS = {
    "dark": EPOCH_TYPE_COLORS["dark"],
    "light": EPOCH_TYPE_COLORS["light"],
}
MOTOR_SUMMARY_ANIMAL_COLORS = {
    "L12": "#4C78A8",
    "L14": "#66C2A5",
    "L15": "#FC8D62",
    "L19": "#E78AC3",
}
MOTOR_PANEL_TRAJECTORY_LABELS = {
    "right_to_center": "R -> C",
    "center_to_left": "C -> L",
    "left_to_center": "L -> C",
    "center_to_right": "C -> R",
}
MOTOR_PANEL_VARIABLE_LABELS = {
    "speed_cm_s": "Speed\n(cm/s)",
    "acceleration_cm_s2": "Accel.\n(cm/s2)",
    "head_direction_deg": "Head dir.\n(deg)",
    "head_angular_velocity_deg_s": "Head ang. vel.\n(deg/s)",
    "head_angular_acceleration_deg_s2": "Head ang. accel.\n(deg/s2)",
    "head_angular_speed_deg_s": "Head ang. speed\n(deg/s)",
}
PANEL_TITLE_FONTSIZE = 8.0
STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION = 0.5
DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ = 0.5
DARK_LIGHT_CORRELATION_BINS = np.linspace(-1.0, 1.0, 21)
DARK_LIGHT_CORRELATION_GRID_WSPACE = 0.24
DARK_LIGHT_CORRELATION_TUNING_CURVE_RELATIVE_DIR = (
    Path("task_progression") / "compute_tuning_curves"
)
DARK_LIGHT_CORRELATION_RATE_RELATIVE_DIR = (
    Path("task_progression") / "dark_light_glm" / "selected"
)
DARK_LIGHT_CORRELATION_RATE_MODEL = "visual"
DARK_LIGHT_CORRELATION_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_epoch",
    "trajectory_type",
    "unit",
    "dark_movement_firing_rate_hz",
    "light_movement_firing_rate_hz",
    "correlation",
)
LIGHT_TUNING_STABILITY_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "light_epoch",
    "trajectory_type",
    "unit",
    "stability_correlation",
)
PANEL_A_SCATTER_ALPHA = 0.30
PANEL_A_GRID_LEFT = 0.045
PANEL_A_GRID_RIGHT = 0.965
PANEL_A_GRID_TOP = 0.965
PANEL_A_GRID_BOTTOM = 0.055
PANEL_A_FIGURE_1D_WIDTH_CORRECTION = 0.995515695
PANEL_A_HEATMAP_WIDTH_FRACTION = (
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION * PANEL_A_FIGURE_1D_WIDTH_CORRECTION
)
PANEL_A_HEATMAP_SIDE_SPACER_FRACTION = (
    (PANEL_A_GRID_RIGHT - PANEL_A_GRID_LEFT - PANEL_A_HEATMAP_WIDTH_FRACTION) / 2.0
)
REORDERED_HEATMAP_TITLE = "Fig. 1D cells in light"
REORDERED_HEATMAP_CMAP = PANEL_D_HEATMAP_CMAP
REORDERED_HEATMAP_VMAX = 1.0
REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION = 0.5
PANEL_A_CV_PCA_REGION = "v1"
PANEL_A_CV_PCA_LIGHT_EPOCH = "06_r3"
PANEL_A_CV_PCA_TITLE = "V1 cvPCA dimensionality"
PANEL_A_CV_PCA_RELATIVE_DIR = Path("signal_dim") / "cv_pca"
PANEL_A_CV_PCA_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_epoch",
    "source_condition",
    "target_condition",
    "n_units",
    "source_cv_participation_ratio",
)
PANEL_A_FIGURE_1D_ORDER_MODE = "figure_1d_order"
PANEL_A_ORDER_MODES = (
    PANEL_A_FIGURE_1D_ORDER_MODE,
)
PANEL_A_CACHE_PREFIX = "supplementary_figure_3_panel_a"
PANEL_A_CACHE_VERSION = 2
PANEL_A_CACHE_METADATA_KEY = "__metadata__"
PANEL_A_CACHE_DATASET_TOKEN_LIMIT = 120


def group_datasets_by_animal(datasets: Sequence[DatasetId]) -> dict[str, list[DatasetId]]:
    """Return normalized data sets grouped by animal in input order."""
    grouped: dict[str, list[DatasetId]] = {}
    for dataset in datasets:
        normalized = normalize_dataset_id(dataset)
        animal_name = str(normalized[0])
        grouped.setdefault(animal_name, []).append(normalized)
    return grouped


def format_animal_row_label(animal_name: str, datasets: Sequence[DatasetId]) -> str:
    """Return a compact label for one per-animal row."""
    dates = []
    for dataset in datasets:
        _animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        if str(date) not in dates:
            dates.append(str(date))
    if not dates:
        return str(animal_name)
    return f"{animal_name}\n{', '.join(dates)}"


def hide_x_axis_labels(ax: object) -> None:
    """Hide x-axis label text and tick labels for repeated row panels."""
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelbottom=False)


def set_panel_a_dot_alpha(ax: object) -> None:
    """Make the per-animal Figure 3C scatter points more visible."""
    for collection in ax.collections:
        collection.set_alpha(PANEL_A_SCATTER_ALPHA)


def build_panel_a_cv_pca_summary_path(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    dark_epoch: str,
) -> Path:
    """Return the hardcoded V1 light-vs-registered-dark cvPCA summary path."""
    stem = (
        f"{PANEL_A_CV_PCA_REGION}_{PANEL_A_CV_PCA_LIGHT_EPOCH}_vs_"
        f"{dark_epoch}_cv_pca_summary.parquet"
    )
    return (
        Path(data_root)
        / str(animal_name)
        / str(date)
        / PANEL_A_CV_PCA_RELATIVE_DIR
        / stem
    )


def _require_table_columns(table: Any, path: Path, columns: Sequence[str]) -> None:
    """Validate that a loaded table has the required columns."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Table {path} is missing columns {missing!r}.")


def load_panel_a_cv_pca_participation_ratio_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
) -> Any:
    """Load paired dark/light cvPCA participation ratios for Supplementary Figure 3A."""
    import pandas as pd

    missing_paths = []
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        path = build_panel_a_cv_pca_summary_path(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            dark_epoch=dark_epoch,
        )
        if not path.exists():
            missing_paths.append(
                f"{animal_name} {date} {PANEL_A_CV_PCA_REGION} "
                f"{PANEL_A_CV_PCA_LIGHT_EPOCH} vs {dark_epoch}: {path}"
            )
            continue

        table = pd.read_parquet(path)
        _require_table_columns(table, path, PANEL_A_CV_PCA_COLUMNS)
        for condition in ("dark", "light"):
            matches = table.loc[
                (table["source_condition"] == condition)
                & (table["target_condition"] == condition)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {condition!r} within-condition cvPCA row in "
                    f"{path}, found {len(matches)}."
                )
            match = matches.iloc[0]
            participation_ratio = float(match["source_cv_participation_ratio"])
            if not np.isfinite(participation_ratio):
                raise ValueError(
                    f"Participation ratio for {condition!r} in {path} is not finite."
                )
            rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "region": PANEL_A_CV_PCA_REGION,
                    "dark_epoch": dark_epoch,
                    "light_epoch": PANEL_A_CV_PCA_LIGHT_EPOCH,
                    "condition": condition,
                    "participation_ratio": participation_ratio,
                    "n_units": int(match["n_units"]),
                    "source_path": path,
                }
            )

    if missing_paths:
        message = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing Supplementary Figure 3A cvPCA summary files:\n" + message
        )
    return pd.DataFrame(rows)


def plot_panel_a_cv_pca_participation_ratios(ax: object, table: Any) -> None:
    """Plot paired dark/light cvPCA participation ratios for manuscript sessions."""
    condition_positions = {"dark": 0.0, "light": 1.0}
    dark_color = EPOCH_TYPE_COLORS["dark"]
    light_color = EPOCH_TYPE_COLORS["light"]
    for (_animal_name, _date), session_table in table.groupby(
        ["animal_name", "date"],
        sort=False,
    ):
        values = {}
        for condition in ("dark", "light"):
            condition_values = session_table.loc[
                session_table["condition"] == condition,
                "participation_ratio",
            ].to_numpy(dtype=float)
            if condition_values.size != 1:
                raise ValueError(
                    "Expected one participation ratio per condition for each "
                    f"session, found {condition_values.size} for {condition!r}."
                )
            values[condition] = float(condition_values[0])
        ax.plot(
            [condition_positions["dark"], condition_positions["light"]],
            [values["dark"], values["light"]],
            color="0.58",
            linewidth=0.85,
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            [condition_positions["dark"], condition_positions["light"]],
            [values["dark"], values["light"]],
            color=[dark_color, light_color],
            edgecolor="black",
            linewidth=0.35,
            s=16,
            zorder=2,
        )

    values = table["participation_ratio"].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        value_range = value_max - value_min
        pad = max(0.08 * value_range, 0.08)
        ax.set_ylim(value_min - pad, value_max + pad)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks(
        [condition_positions["dark"], condition_positions["light"]],
        ["Dark", "Light"],
    )
    ax.set_ylabel("Participation ratio")
    ax.set_title(PANEL_A_CV_PCA_TITLE, fontsize=PANEL_TITLE_FONTSIZE, pad=2)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_motor_progression_summary_path(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
) -> Path:
    """Return the motor progression summary path for one data set."""
    return Path(data_root) / str(animal_name) / str(date) / MOTOR_PANEL_RELATIVE_PATH


def load_panel_b_motor_progression_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    light_epoch: str = MOTOR_PANEL_LIGHT_EPOCH,
) -> Any:
    """Load motor progression summaries for the Supplementary Figure 3B grid."""
    import pandas as pd

    missing_paths = []
    missing_entries = []
    tables = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        path = build_motor_progression_summary_path(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
        )
        if not path.exists():
            missing_paths.append(f"{animal_name} {date}: {path}")
            continue

        table = pd.read_parquet(path)
        _require_table_columns(table, path, MOTOR_PANEL_COLUMNS)
        table = table.copy()
        for column in ("epoch", "trajectory_type", "variable"):
            table[column] = table[column].astype(str)

        required_epochs = {"dark": str(dark_epoch), "light": str(light_epoch)}
        for epoch_type, epoch in required_epochs.items():
            epoch_table = table.loc[table["epoch"] == epoch]
            if epoch_table.empty:
                missing_entries.append(
                    f"{animal_name} {date}: missing {epoch_type} epoch {epoch!r}"
                )
                continue
            for variable_name in MOTOR_VARIABLES:
                for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
                    has_rows = np.any(
                        (epoch_table["variable"] == str(variable_name))
                        & (epoch_table["trajectory_type"] == str(trajectory_type))
                    )
                    if not has_rows:
                        missing_entries.append(
                            f"{animal_name} {date}: missing {epoch!r} "
                            f"{variable_name} {trajectory_type}"
                        )

        selected = table.loc[
            table["epoch"].isin(required_epochs.values())
            & table["variable"].isin(MOTOR_VARIABLES)
            & table["trajectory_type"].isin(PANEL_B_TRAJECTORY_TYPES)
        ].copy()
        selected["animal_name"] = animal_name
        selected["date"] = date
        selected["dark_epoch"] = dark_epoch
        selected["light_epoch"] = str(light_epoch)
        selected["dataset_label"] = f"{animal_name} {date}"
        selected["epoch_type"] = np.where(
            selected["epoch"].to_numpy(dtype=str) == str(dark_epoch),
            "dark",
            "light",
        )
        selected["source_path"] = path
        tables.append(selected)

    if missing_paths:
        message = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing Supplementary Figure 3B motor progression files:\n" + message
        )
    if missing_entries:
        message = "\n".join(f"- {entry}" for entry in missing_entries)
        raise ValueError(
            "Motor progression summaries are incomplete for Supplementary Figure 3B:\n"
            + message
        )
    if not tables:
        raise ValueError("No motor progression tables were loaded for Supplementary Figure 3B.")
    return pd.concat(tables, ignore_index=True)


def _set_motor_panel_row_limits(axes: np.ndarray, table: Any) -> None:
    """Apply one y-axis range per motor variable row."""
    for row_index, variable_name in enumerate(MOTOR_VARIABLES):
        values = table.loc[
            table["variable"].astype(str) == str(variable_name),
            ["q25", "median", "q75"],
        ].to_numpy(dtype=float).ravel()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        y_range = y_max - y_min
        y_pad = max(0.06 * y_range, 1.0 if y_range == 0.0 else 0.0)
        for ax in axes[row_index, :]:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_panel_b_motor_progression_grid(
    axes: np.ndarray,
    table: Any,
    *,
    datasets: Sequence[DatasetId],
) -> None:
    """Plot one animal's median motor variables and IQR over normalized position."""
    from matplotlib.lines import Line2D

    axes = np.asarray(axes, dtype=object)
    expected_shape = (len(MOTOR_VARIABLES), len(PANEL_B_TRAJECTORY_TYPES))
    if axes.shape != expected_shape:
        raise ValueError(f"Expected axes shape {expected_shape}, got {axes.shape}.")

    dataset_order = []
    for dataset in datasets:
        normalized_dataset = normalize_dataset_id(dataset)
        if str(normalized_dataset[0]) == MOTOR_PANEL_ANIMAL_NAME:
            dataset_order.append(normalized_dataset)
    if not dataset_order:
        raise ValueError(
            "Supplementary Figure 3B is configured to plot "
            f"{MOTOR_PANEL_ANIMAL_NAME}, but no matching dataset was supplied."
        )

    panel_table = table.loc[
        table["animal_name"].astype(str) == MOTOR_PANEL_ANIMAL_NAME
    ].copy()
    if panel_table.empty:
        raise ValueError(
            "Supplementary Figure 3B motor table has no rows for "
            f"{MOTOR_PANEL_ANIMAL_NAME}."
        )
    epoch_labels = {"dark": "Dark", "light": "Light"}

    for row_index, variable_name in enumerate(MOTOR_VARIABLES):
        variable_table = panel_table.loc[
            panel_table["variable"].astype(str) == str(variable_name)
        ]
        for column_index, trajectory_type in enumerate(PANEL_B_TRAJECTORY_TYPES):
            ax = axes[row_index, column_index]
            trajectory_table = variable_table.loc[
                variable_table["trajectory_type"].astype(str) == str(trajectory_type)
            ]
            for animal_name, date, _dark_epoch in dataset_order:
                dataset_table = trajectory_table.loc[
                    (trajectory_table["animal_name"].astype(str) == str(animal_name))
                    & (trajectory_table["date"].astype(str) == str(date))
                ]
                for epoch_type in ("dark", "light"):
                    epoch_table = dataset_table.loc[
                        dataset_table["epoch_type"].astype(str) == epoch_type
                    ].sort_values("progression_bin_index", kind="stable")
                    if epoch_table.empty:
                        continue
                    color = MOTOR_PANEL_EPOCH_COLORS[epoch_type]
                    x_values = epoch_table["progression_bin_center"].to_numpy(dtype=float)
                    q25_values = epoch_table["q25"].to_numpy(dtype=float)
                    q75_values = epoch_table["q75"].to_numpy(dtype=float)
                    ax.fill_between(
                        x_values,
                        q25_values,
                        q75_values,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                    )
                    ax.plot(
                        x_values,
                        epoch_table["median"].to_numpy(dtype=float),
                        color=color,
                        linewidth=0.9,
                        alpha=0.90,
                    )

            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.18, linewidth=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_index == 0:
                ax.set_title(
                    MOTOR_PANEL_TRAJECTORY_LABELS.get(str(trajectory_type), str(trajectory_type)),
                    fontsize=6.0,
                    pad=1.5,
                )
            if column_index == 0:
                ax.set_ylabel(
                    MOTOR_PANEL_VARIABLE_LABELS.get(
                        str(variable_name),
                        VARIABLE_LABELS.get(str(variable_name), str(variable_name)),
                    ),
                    fontsize=5.2,
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if row_index == len(MOTOR_VARIABLES) - 1:
                ax.set_xlabel("Norm. position", fontsize=5.4)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="both", labelsize=4.8, length=2.0)

    _set_motor_panel_row_limits(axes, panel_table)

    epoch_handles = [
        Line2D([0], [0], color=MOTOR_PANEL_EPOCH_COLORS["dark"], linewidth=1.0),
        Line2D([0], [0], color=MOTOR_PANEL_EPOCH_COLORS["light"], linewidth=1.0),
    ]
    axes[0, 0].legend(
        epoch_handles,
        [epoch_labels["dark"], epoch_labels["light"]],
        frameon=False,
        fontsize=4.8,
        loc="upper left",
        handlelength=1.5,
        borderpad=0.1,
        labelspacing=0.2,
    )


def _motor_summary_animal_color(animal_name: str) -> str:
    """Return a stable color for one motor-summary data set."""
    return MOTOR_SUMMARY_ANIMAL_COLORS.get(str(animal_name), "0.35")


def compute_motor_profile_correlation(dark_values: np.ndarray, light_values: np.ndarray) -> float:
    """Return the Pearson correlation between paired dark and light motor profiles."""
    dark_values = np.asarray(dark_values, dtype=float).reshape(-1)
    light_values = np.asarray(light_values, dtype=float).reshape(-1)
    if dark_values.shape != light_values.shape:
        raise ValueError(
            "Dark and light motor profiles must have matching shapes. "
            f"Got {dark_values.shape} and {light_values.shape}."
        )
    finite_mask = np.isfinite(dark_values) & np.isfinite(light_values)
    if np.count_nonzero(finite_mask) < 2:
        return float("nan")
    dark_finite = dark_values[finite_mask]
    light_finite = light_values[finite_mask]
    if np.nanstd(dark_finite) <= 0.0 or np.nanstd(light_finite) <= 0.0:
        return float("nan")
    return float(np.corrcoef(dark_finite, light_finite)[0, 1])


def build_panel_c_motor_profile_correlation_table(
    table: Any,
    *,
    datasets: Sequence[DatasetId],
) -> Any:
    """Summarize dark-light motor profile correlations by data set and trajectory."""
    import pandas as pd

    required_columns = (
        "animal_name",
        "date",
        "dark_epoch",
        "light_epoch",
        "epoch_type",
        "trajectory_type",
        "variable",
        "progression_bin_index",
        "median",
    )
    _require_table_columns(table, Path("motor_progression_table"), required_columns)

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        dataset_table = table.loc[
            (table["animal_name"].astype(str) == str(animal_name))
            & (table["date"].astype(str) == str(date))
        ]
        for variable_name in MOTOR_VARIABLES:
            variable_table = dataset_table.loc[
                dataset_table["variable"].astype(str) == str(variable_name)
            ]
            for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
                trajectory_table = variable_table.loc[
                    variable_table["trajectory_type"].astype(str) == str(trajectory_type)
                ]
                dark_table = trajectory_table.loc[
                    trajectory_table["epoch_type"].astype(str) == "dark"
                ][["progression_bin_index", "median"]]
                light_table = trajectory_table.loc[
                    trajectory_table["epoch_type"].astype(str) == "light"
                ][["progression_bin_index", "median"]]
                paired = pd.merge(
                    dark_table,
                    light_table,
                    on="progression_bin_index",
                    suffixes=("_dark", "_light"),
                    how="inner",
                ).sort_values("progression_bin_index", kind="stable")
                correlation = compute_motor_profile_correlation(
                    paired["median_dark"].to_numpy(dtype=float),
                    paired["median_light"].to_numpy(dtype=float),
                )
                dark_epoch = (
                    dataset_table["dark_epoch"].astype(str).iloc[0]
                    if not dataset_table.empty
                    else ""
                )
                light_epoch = (
                    dataset_table["light_epoch"].astype(str).iloc[0]
                    if not dataset_table.empty
                    else ""
                )
                rows.append(
                    {
                        "animal_name": str(animal_name),
                        "date": str(date),
                        "dark_epoch": dark_epoch,
                        "light_epoch": light_epoch,
                        "trajectory_type": str(trajectory_type),
                        "variable": str(variable_name),
                        "correlation": correlation,
                        "n_bins": int(len(paired)),
                    }
                )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "animal_name",
            "date",
            "dark_epoch",
            "light_epoch",
            "trajectory_type",
            "variable",
            "correlation",
            "n_bins",
        ],
    )


def plot_panel_c_motor_profile_correlations(
    axes: Sequence[Any],
    table: Any,
    *,
    datasets: Sequence[DatasetId],
) -> None:
    """Plot dark-light motor profile correlations for each data set."""
    from matplotlib.lines import Line2D

    axes = np.asarray(axes, dtype=object).reshape(-1)
    expected_count = len(PANEL_B_TRAJECTORY_TYPES)
    if axes.shape != (expected_count,):
        raise ValueError(f"Expected {expected_count} axes, got {axes.shape}.")

    animal_order = list(
        dict.fromkeys(str(normalize_dataset_id(dataset)[0]) for dataset in datasets)
    )
    y_positions = np.arange(len(MOTOR_VARIABLES), dtype=float)
    if len(animal_order) <= 1:
        jitter_values = np.zeros(len(animal_order), dtype=float)
    else:
        jitter_values = np.linspace(-0.18, 0.18, len(animal_order))
    y_jitter_by_animal = {
        animal_name: jitter
        for animal_name, jitter in zip(animal_order, jitter_values, strict=True)
    }

    for column_index, trajectory_type in enumerate(PANEL_B_TRAJECTORY_TYPES):
        ax = axes[column_index]
        trajectory_table = table.loc[
            table["trajectory_type"].astype(str) == str(trajectory_type)
        ]
        for variable_index, variable_name in enumerate(MOTOR_VARIABLES):
            variable_table = trajectory_table.loc[
                trajectory_table["variable"].astype(str) == str(variable_name)
            ]
            for animal_name in animal_order:
                animal_table = variable_table.loc[
                    variable_table["animal_name"].astype(str) == animal_name
                ]
                if animal_table.empty:
                    continue
                value = float(animal_table["correlation"].iloc[0])
                if not np.isfinite(value):
                    continue
                ax.scatter(
                    value,
                    float(variable_index) + y_jitter_by_animal[animal_name],
                    s=14,
                    color=_motor_summary_animal_color(animal_name),
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.95,
                    zorder=3,
                )

        ax.axvline(0.0, color="0.65", linewidth=0.6, linestyle="--", zorder=1)
        ax.axvline(1.0, color="0.80", linewidth=0.5, zorder=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.6, len(MOTOR_VARIABLES) - 0.4)
        ax.invert_yaxis()
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_title(
            MOTOR_PANEL_TRAJECTORY_LABELS.get(str(trajectory_type), str(trajectory_type)),
            fontsize=6.0,
            pad=1.5,
        )
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.4)
        ax.grid(True, axis="y", alpha=0.10, linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=4.8, length=2.0)
        ax.set_xlabel("Profile corr.", fontsize=5.4)
        if column_index == 0:
            ax.set_yticks(
                y_positions,
                [
                    MOTOR_PANEL_VARIABLE_LABELS.get(
                        str(variable_name),
                        VARIABLE_LABELS.get(str(variable_name), str(variable_name)),
                    ).replace("\n", " ")
                    for variable_name in MOTOR_VARIABLES
                ],
            )
            ax.set_ylabel("Motor variable", fontsize=5.4)
        else:
            ax.set_yticks(y_positions)
            ax.tick_params(axis="y", labelleft=False)

    animal_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=_motor_summary_animal_color(animal_name),
            markeredgecolor="white",
            markeredgewidth=0.25,
            markersize=3.5,
        )
        for animal_name in animal_order
    ]
    axes[-1].legend(
        animal_handles,
        animal_order,
        frameon=False,
        fontsize=4.8,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.3,
    )


def load_epoch_stable_units_by_tuning_stability(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_correlation: float = STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION,
) -> set[int]:
    """Return units with odd/even stability above threshold in any trajectory."""
    if min_correlation < -1.0:
        raise ValueError("min_correlation must be at least -1.")

    import pandas as pd

    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(table_path)
    _require_table_columns(
        table,
        table_path,
        ("unit", "region", "epoch", "trajectory_type", "stability_correlation"),
    )
    correlations = pd.to_numeric(table["stability_correlation"], errors="coerce")
    stable_rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & (table["trajectory_type"].astype(str).isin(PANEL_B_TRAJECTORY_TYPES))
        & np.isfinite(correlations.to_numpy(dtype=float))
        & (correlations.to_numpy(dtype=float) > float(min_correlation))
    ]
    return set(
        pd.to_numeric(stable_rows["unit"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )


def filter_panel_d_similarity_table_by_tuning_stability(
    similarity_table: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_correlation: float = STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION,
) -> Any:
    """Keep similarity rows for units stable in at least one trajectory in both epochs."""
    import pandas as pd

    required_columns = ("animal_name", "date", "unit")
    _require_table_columns(similarity_table, Path("similarity_table"), required_columns)
    stable_units_by_session: dict[tuple[str, str], set[int]] = {}
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        selected_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        light_stable_units = load_epoch_stable_units_by_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_light_epoch,
            min_correlation=min_correlation,
        )
        dark_stable_units = load_epoch_stable_units_by_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_dark_epoch,
            min_correlation=min_correlation,
        )
        stable_units_by_session[(str(animal_name), str(date))] = (
            light_stable_units & dark_stable_units
        )

    table = similarity_table.copy()
    unit_values = pd.to_numeric(table["unit"], errors="coerce")
    keep_mask = []
    for animal_name, date, unit in zip(
        table["animal_name"].astype(str),
        table["date"].astype(str),
        unit_values,
        strict=True,
    ):
        keep_mask.append(
            np.isfinite(unit)
            and int(unit) in stable_units_by_session.get((animal_name, date), set())
        )
    return table.loc[np.asarray(keep_mask, dtype=bool)].copy()


def concatenate_unit_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate unit-key chunks for pooled heatmap ordering."""
    if not parts:
        return np.asarray([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def concatenate_value_parts(
    parts: list[np.ndarray],
    position_bin_count: int,
) -> np.ndarray:
    """Concatenate tuning-curve chunks for one pooled heatmap panel."""
    if not parts:
        return np.empty((0, position_bin_count), dtype=float)
    return np.vstack(parts)


def collect_curve_arrays(
    curve_sets: Sequence[dict[str, Any]],
    *,
    curve_key: str,
    trajectory_type: str,
    position_bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled unit keys and tuning values for one trajectory."""
    unit_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for curve_set in curve_sets:
        curve = curve_set[curve_key].get(trajectory_type)
        if curve is None:
            continue
        units, values = extract_tuning_curve_arrays(curve)
        unit_parts.append(
            build_unit_keys(
                animal_name=str(curve_set["animal_name"]),
                date=str(curve_set["date"]),
                region=str(curve_set["region"]),
                units=units,
            )
        )
        value_parts.append(values)
    return (
        concatenate_unit_parts(unit_parts),
        concatenate_value_parts(value_parts, position_bin_count),
    )


def compute_epoch_all_trial_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    epoch: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
) -> dict[str, Any]:
    """Compute all-trial movement tuning curves and movement firing rates."""
    session = prepare_heatmap_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )
    selected_epoch = session["run_epochs"][0]
    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        session["position_by_epoch"][selected_epoch],
        session["timestamps_position"][selected_epoch],
        session["trajectory_intervals"][selected_epoch],
        position_offset=position_offset,
        use_trajectory_direction=use_trajectory_direction,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    bin_edges = build_normalized_position_bins(position_bin_count)
    curves = {}
    for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
        epochs = session["trajectory_intervals"][selected_epoch][
            trajectory_type
        ].intersect(session["movement_by_run"][selected_epoch])
        curves[trajectory_type] = compute_place_tuning_curve(
            session["spikes_by_region"][region],
            normalized_position_by_trajectory[trajectory_type],
            epochs,
            bin_edges=bin_edges,
            sigma_bins=sigma_bins,
        )
    movement_firing_rates = compute_unit_movement_firing_rates(
        session["spikes_by_region"][region],
        session["movement_by_run"][selected_epoch],
    )
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": selected_epoch,
        "all_curves": curves,
        "movement_firing_rates_hz": movement_firing_rates,
    }


def compute_light_epoch_all_trial_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
) -> dict[str, Any]:
    """Compute all-trial light movement tuning curves."""
    return compute_epoch_all_trial_tuning_curves(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        region=region,
        epoch=get_light_epoch(animal_name, date, light_epoch),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        use_trajectory_direction=use_trajectory_direction,
    )


def compute_dark_epoch_all_trial_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
) -> dict[str, Any]:
    """Compute all-trial dark movement tuning curves."""
    return compute_epoch_all_trial_tuning_curves(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        region=region,
        epoch=get_dark_epoch(animal_name, date, dark_epoch),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        use_trajectory_direction=use_trajectory_direction,
    )


def get_saved_task_progression_tuning_curve_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
) -> Path:
    """Return one saved trajectory tuning-curve artifact path."""
    return (
        Path(data_root)
        / animal_name
        / date
        / DARK_LIGHT_CORRELATION_TUNING_CURVE_RELATIVE_DIR
        / f"{region}_{epoch}_place_{trajectory_type}_tuning_curves.nc"
    )


def get_dark_light_movement_rate_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_name: str = DARK_LIGHT_CORRELATION_RATE_MODEL,
) -> Path:
    """Return the selected dark/light artifact that stores movement rates."""
    return (
        Path(data_root)
        / animal_name
        / date
        / DARK_LIGHT_CORRELATION_RATE_RELATIVE_DIR
        / f"{region}_{light_epoch}_vs_{dark_epoch}_{model_name}_selected.nc"
    )


def load_saved_epoch_tuning_curve_set(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load saved empirical trajectory tuning curves for one epoch."""
    import xarray as xr

    curves = {}
    for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
        path = get_saved_task_progression_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory_type=trajectory_type,
        )
        if not path.exists():
            raise FileNotFoundError(
                "Missing saved task-progression tuning curve. Expected "
                f"{path}. Run `python -m v1ca1.task_progression.compute_tuning_curves` "
                "for this session first."
            )
        dataset = xr.open_dataset(path)
        try:
            if "firing_rate_hz" not in dataset:
                raise ValueError(f"Tuning curve artifact {path} lacks firing_rate_hz.")
            curve = dataset["firing_rate_hz"].load()
            if float(sigma_bins) > 0.0:
                curve = smooth_tuning_curve_nan_aware(
                    curve,
                    sigma_bins=float(sigma_bins),
                )
            curves[trajectory_type] = curve
        finally:
            dataset.close()
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
        "all_curves": curves,
    }


def saved_epoch_tuning_curve_set_exists(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> bool:
    """Return whether all saved trajectory tuning curves exist for one epoch."""
    return all(
        get_saved_task_progression_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory_type=trajectory_type,
        ).exists()
        for trajectory_type in PANEL_B_TRAJECTORY_TYPES
    )


def load_dark_light_movement_firing_rates(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
) -> tuple[dict[int, float], dict[int, float]]:
    """Load saved dark and light movement firing rates keyed by unit."""
    import xarray as xr

    path = get_dark_light_movement_rate_path(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing selected dark/light model artifact with movement firing rates. "
            f"Expected {path}. Run `python -m v1ca1.task_progression.dark_light_glm` "
            "for this session first."
        )
    dataset = xr.open_dataset(path)
    try:
        for variable in (
            "dark_movement_firing_rate_hz",
            "light_movement_firing_rate_hz",
        ):
            if variable not in dataset:
                raise ValueError(f"Dark/light artifact {path} lacks {variable}.")
        units = np.asarray(dataset.coords["unit"].values, dtype=int)
        dark_rates = np.asarray(dataset["dark_movement_firing_rate_hz"].values, dtype=float)
        light_rates = np.asarray(
            dataset["light_movement_firing_rate_hz"].values,
            dtype=float,
        )
    finally:
        dataset.close()
    return (
        {int(unit): float(rate) for unit, rate in zip(units, dark_rates, strict=True)},
        {int(unit): float(rate) for unit, rate in zip(units, light_rates, strict=True)},
    )


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return weights that normalize one histogram to a fraction of cells."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def compute_tuning_curve_correlation(
    dark_values: np.ndarray,
    light_values: np.ndarray,
) -> float:
    """Return Pearson correlation between matched dark and light tuning curves."""
    dark_values = np.asarray(dark_values, dtype=float).reshape(-1)
    light_values = np.asarray(light_values, dtype=float).reshape(-1)
    if dark_values.shape != light_values.shape:
        raise ValueError(
            "Dark and light tuning curves must have matching shapes. "
            f"Got {dark_values.shape} and {light_values.shape}."
        )
    finite_mask = np.isfinite(dark_values) & np.isfinite(light_values)
    if int(np.count_nonzero(finite_mask)) < 2:
        return float("nan")
    dark_finite = dark_values[finite_mask]
    light_finite = light_values[finite_mask]
    if np.nanstd(dark_finite) <= 0.0 or np.nanstd(light_finite) <= 0.0:
        return float("nan")
    return float(np.corrcoef(dark_finite, light_finite)[0, 1])


def _movement_rate_lookup(movement_firing_rates: dict[Any, float]) -> dict[int, float]:
    """Return movement firing rates keyed by integer unit id."""
    return {
        int(unit_id): float(rate)
        for unit_id, rate in movement_firing_rates.items()
    }


def build_dark_light_tuning_correlation_table(
    dark_curve_sets: Sequence[dict[str, Any]],
    light_curve_sets: Sequence[dict[str, Any]],
    *,
    min_movement_firing_rate_hz: float = (
        DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return per-unit dark/light tuning-curve correlations by trajectory."""
    if min_movement_firing_rate_hz < 0.0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")

    import pandas as pd

    rows: list[dict[str, Any]] = []
    for dark_set, light_set in zip(dark_curve_sets, light_curve_sets, strict=True):
        dark_rates = _movement_rate_lookup(dark_set["movement_firing_rates_hz"])
        light_rates = _movement_rate_lookup(light_set["movement_firing_rates_hz"])
        for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
            dark_curve = dark_set["all_curves"].get(trajectory_type)
            light_curve = light_set["all_curves"].get(trajectory_type)
            if dark_curve is None or light_curve is None:
                continue
            dark_units, dark_values = extract_tuning_curve_arrays(dark_curve)
            light_units, light_values = extract_tuning_curve_arrays(light_curve)
            dark_rows = {int(unit): index for index, unit in enumerate(dark_units)}
            light_rows = {int(unit): index for index, unit in enumerate(light_units)}
            for unit_id in sorted(set(dark_rows).intersection(light_rows)):
                dark_rate = dark_rates.get(unit_id, 0.0)
                light_rate = light_rates.get(unit_id, 0.0)
                if (
                    dark_rate < float(min_movement_firing_rate_hz)
                    or light_rate < float(min_movement_firing_rate_hz)
                ):
                    continue
                correlation = compute_tuning_curve_correlation(
                    dark_values[dark_rows[unit_id]],
                    light_values[light_rows[unit_id]],
                )
                if not np.isfinite(correlation):
                    continue
                rows.append(
                    {
                        "animal_name": str(dark_set["animal_name"]),
                        "date": str(dark_set["date"]),
                        "region": str(dark_set["region"]),
                        "dark_epoch": str(dark_set["epoch"]),
                        "light_epoch": str(light_set["epoch"]),
                        "trajectory_type": trajectory_type,
                        "unit": int(unit_id),
                        "dark_movement_firing_rate_hz": float(dark_rate),
                        "light_movement_firing_rate_hz": float(light_rate),
                        "correlation": float(correlation),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=DARK_LIGHT_CORRELATION_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=DARK_LIGHT_CORRELATION_TABLE_COLUMNS)


def load_dark_light_tuning_correlation_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    min_movement_firing_rate_hz: float = (
        DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Compute pooled dark/light tuning-curve correlations for V1 cells."""

    dark_curve_sets = []
    light_curve_sets = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        selected_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        dark_rates, light_rates = load_dark_light_movement_firing_rates(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=selected_light_epoch,
            dark_epoch=selected_dark_epoch,
        )
        if saved_epoch_tuning_curve_set_exists(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_dark_epoch,
        ) and saved_epoch_tuning_curve_set_exists(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_light_epoch,
        ):
            dark_curve_set = load_saved_epoch_tuning_curve_set(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=selected_dark_epoch,
                sigma_bins=sigma_bins,
            )
            light_curve_set = load_saved_epoch_tuning_curve_set(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=selected_light_epoch,
                sigma_bins=sigma_bins,
            )
        else:
            dark_curve_set = compute_dark_epoch_all_trial_tuning_curves(
                animal_name=animal_name,
                date=date,
                data_root=data_root,
                region=region,
                dark_epoch=selected_dark_epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                use_trajectory_direction=True,
            )
            light_curve_set = compute_light_epoch_all_trial_tuning_curves(
                animal_name=animal_name,
                date=date,
                data_root=data_root,
                region=region,
                light_epoch=selected_light_epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                use_trajectory_direction=True,
            )
        dark_curve_set["movement_firing_rates_hz"] = dark_rates
        light_curve_set["movement_firing_rates_hz"] = light_rates
        dark_curve_sets.append(dark_curve_set)
        light_curve_sets.append(light_curve_set)
    return build_dark_light_tuning_correlation_table(
        dark_curve_sets,
        light_curve_sets,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
    )


def load_light_tuning_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
) -> Any:
    """Load pooled light-epoch odd/even tuning-stability rows by trajectory."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        table_path = get_stability_table_path(data_root, animal_name, date)
        if not table_path.exists():
            raise FileNotFoundError(
                "Missing task-progression stability table. Expected "
                f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
                "for this session first."
            )
        table = pd.read_parquet(table_path)
        _require_table_columns(
            table,
            table_path,
            ("unit", "region", "epoch", "trajectory_type", "stability_correlation"),
        )
        filtered = table[
            (table["region"].astype(str) == str(region))
            & (table["epoch"].astype(str) == str(selected_light_epoch))
            & (table["trajectory_type"].astype(str).isin(PANEL_B_TRAJECTORY_TYPES))
        ].copy()
        filtered["stability_correlation"] = pd.to_numeric(
            filtered["stability_correlation"],
            errors="coerce",
        )
        filtered["unit"] = pd.to_numeric(filtered["unit"], errors="coerce")
        filtered = filtered[
            np.isfinite(filtered["stability_correlation"].to_numpy(dtype=float))
            & np.isfinite(filtered["unit"].to_numpy(dtype=float))
        ].copy()
        if filtered.empty:
            continue
        filtered = filtered.assign(
            animal_name=animal_name,
            date=date,
            light_epoch=selected_light_epoch,
        )
        filtered["unit"] = filtered["unit"].astype(int)
        tables.append(
            filtered[
                [
                    "animal_name",
                    "date",
                    "region",
                    "light_epoch",
                    "trajectory_type",
                    "unit",
                    "stability_correlation",
                ]
            ]
        )

    if not tables:
        return pd.DataFrame(columns=LIGHT_TUNING_STABILITY_TABLE_COLUMNS)
    return pd.concat(tables, axis=0, ignore_index=True)


def _format_trajectory_label(trajectory_type: str) -> str:
    """Return compact trajectory labels for histogram titles."""
    labels = {
        "center_to_left": "C-L",
        "center_to_right": "C-R",
        "right_to_center": "R-C",
        "left_to_center": "L-C",
    }
    return labels.get(trajectory_type, trajectory_type)


def plot_dark_light_tuning_correlation_histograms(
    axes: Sequence["Axes"],
    correlation_table: Any,
) -> None:
    """Plot dark/light tuning-curve correlation histograms by trajectory."""
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    if axes_array.size != len(PANEL_B_TRAJECTORY_TYPES):
        raise ValueError(
            "Expected one axis per trajectory type. "
            f"Got {axes_array.size} axes for {len(PANEL_B_TRAJECTORY_TYPES)} trajectories."
        )

    for axis_index, (ax, trajectory_type) in enumerate(
        zip(axes_array, PANEL_B_TRAJECTORY_TYPES, strict=True)
    ):
        rows = correlation_table[
            correlation_table["trajectory_type"].astype(str) == trajectory_type
        ]
        values = np.asarray(rows["correlation"], dtype=float)
        values = values[np.isfinite(values)]
        ax.axvspan(
            -1.0,
            0.0,
            color=NEUTRAL_COLORS["dark_epoch_background"],
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
        if values.size:
            ax.hist(
                values,
                bins=DARK_LIGHT_CORRELATION_BINS,
                weights=_fraction_histogram_weights(values),
                color=TRAJECTORY_COLORS[trajectory_type],
                **COMPACT_HISTOGRAM_KWARGS,
                zorder=2,
            )
            summary = f"n = {values.size}\nmed. {np.median(values):.2f}"
        else:
            summary = "n = 0\nmed. n/a"
        ax.text(
            0.04,
            0.94,
            summary,
            ha="left",
            va="top",
            fontsize=5.2,
            transform=ax.transAxes,
            color="0.25",
        )
        ax.set_title(
            _format_trajectory_label(trajectory_type),
            fontsize=6.2,
            pad=1.5,
            color=TRAJECTORY_COLORS[trajectory_type],
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Dark-light tuning corr.", fontsize=6.2, labelpad=1.5)
        if axis_index == 0:
            ax.set_ylabel("Frac.", fontsize=6.4, labelpad=1.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.8, length=1.8, pad=1)


def plot_dark_light_with_light_stability_histograms(
    axes: Sequence["Axes"],
    correlation_table: Any,
    stability_table: Any,
) -> None:
    """Plot dark/light tuning similarity with light odd/even stability overlays."""
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    if axes_array.size != len(PANEL_B_TRAJECTORY_TYPES):
        raise ValueError(
            "Expected one axis per trajectory type. "
            f"Got {axes_array.size} axes for {len(PANEL_B_TRAJECTORY_TYPES)} trajectories."
        )

    for axis_index, (ax, trajectory_type) in enumerate(
        zip(axes_array, PANEL_B_TRAJECTORY_TYPES, strict=True)
    ):
        correlation_rows = correlation_table[
            correlation_table["trajectory_type"].astype(str) == trajectory_type
        ]
        correlation_values = np.asarray(correlation_rows["correlation"], dtype=float)
        correlation_values = correlation_values[np.isfinite(correlation_values)]
        stability_rows = stability_table[
            stability_table["trajectory_type"].astype(str) == trajectory_type
        ]
        stability_values = np.asarray(
            stability_rows["stability_correlation"],
            dtype=float,
        )
        stability_values = stability_values[np.isfinite(stability_values)]

        ax.axvspan(
            -1.0,
            0.0,
            color=NEUTRAL_COLORS["dark_epoch_background"],
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
        if correlation_values.size:
            ax.hist(
                correlation_values,
                bins=DARK_LIGHT_CORRELATION_BINS,
                weights=_fraction_histogram_weights(correlation_values),
                color=TRAJECTORY_COLORS[trajectory_type],
                label="Dark-light",
                **COMPACT_HISTOGRAM_KWARGS,
                zorder=2,
            )
        if stability_values.size:
            ax.hist(
                stability_values,
                bins=DARK_LIGHT_CORRELATION_BINS,
                weights=_fraction_histogram_weights(stability_values),
                histtype="step",
                color="black",
                linewidth=0.85,
                label="Light odd/even",
                zorder=3,
            )
        if axis_index == 0:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(
                    handles,
                    labels,
                    frameon=False,
                    fontsize=4.8,
                    loc="upper left",
                    handlelength=1.4,
                )
        ax.set_title(
            _format_trajectory_label(trajectory_type),
            fontsize=6.2,
            pad=1.5,
            color=TRAJECTORY_COLORS[trajectory_type],
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Correlation", fontsize=6.2, labelpad=1.5)
        if axis_index == 0:
            ax.set_ylabel("Frac.", fontsize=6.4, labelpad=1.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.8, length=1.8, pad=1)


def select_unit_keys_by_light_tuning_stability(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    min_stability_correlation: float = (
        REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION
    ),
) -> set[str]:
    """Return unit keys stable in at least one light-epoch trajectory."""
    if min_stability_correlation < -1.0:
        raise ValueError("min_stability_correlation must be at least -1.")

    import pandas as pd

    included_unit_keys: set[str] = set()
    for dataset in datasets:
        animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        table_path = get_stability_table_path(data_root, animal_name, date)
        if not table_path.exists():
            raise FileNotFoundError(
                "Missing task-progression stability table. Expected "
                f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
                "for this session first."
            )
        table = pd.read_parquet(table_path)
        table = table[
            (table["epoch"].astype(str) == str(selected_light_epoch))
            & (table["region"].astype(str) == str(region))
            & (table["trajectory_type"].astype(str).isin(PANEL_B_TRAJECTORY_TYPES))
        ]
        correlations = np.asarray(table["stability_correlation"], dtype=float)
        stable_rows = table[
            np.isfinite(correlations)
            & (correlations > float(min_stability_correlation))
        ]
        for unit_id in stable_rows["unit"].drop_duplicates().to_numpy():
            included_unit_keys.add(f"{animal_name}:{date}:{region}:{int(unit_id)}")
    return included_unit_keys


def filter_ordered_unit_keys_by_unit_set(
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    included_unit_keys: set[str],
) -> dict[str, np.ndarray]:
    """Return cached row unit keys restricted to an included unit-key set."""
    return {
        trajectory_type: np.asarray(
            [
                str(unit_key)
                for unit_key in ordered_unit_keys_by_trajectory[trajectory_type]
                if str(unit_key) in included_unit_keys
            ],
            dtype=object,
        )
        for trajectory_type in PANEL_B_TRAJECTORY_TYPES
    }


def build_figure_1d_ordered_light_panel_values(
    *,
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    light_curve_sets: Sequence[dict[str, Any]],
    position_bin_count: int,
    curve_key: str = "all_curves",
) -> dict[tuple[str, str], np.ndarray]:
    """Return light heatmaps aligned to a fixed unit-key row order."""
    panels: dict[tuple[str, str], np.ndarray] = {}
    for order_trajectory in PANEL_B_TRAJECTORY_TYPES:
        reference_units = np.asarray(
            ordered_unit_keys_by_trajectory[order_trajectory],
            dtype=object,
        )
        unit_order = np.arange(reference_units.size, dtype=int)
        sorted_values_by_plot_trajectory: dict[str, np.ndarray] = {}
        for plot_trajectory in PANEL_B_TRAJECTORY_TYPES:
            display_units, display_values = collect_curve_arrays(
                light_curve_sets,
                curve_key=curve_key,
                trajectory_type=plot_trajectory,
                position_bin_count=position_bin_count,
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
        for plot_trajectory, values in sorted_values_by_plot_trajectory.items():
            panels[(order_trajectory, plot_trajectory)] = (
                normalize_panel_values_per_trajectory(values)
            )
    return panels


def _format_panel_a_cache_token(value: object) -> str:
    """Return a filesystem-safe token for Supplementary Figure 3A caches."""
    text = str(value).strip()
    cleaned = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            cleaned.append(character)
        elif character == ".":
            cleaned.append("p")
        else:
            cleaned.append("-")
    token = "".join(cleaned).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "none"


def _format_panel_a_cache_number(value: float | int) -> str:
    """Return a compact numeric token for Supplementary Figure 3A caches."""
    return _format_panel_a_cache_token(f"{float(value):g}")


def _build_panel_a_dataset_cache_token(
    dataset_metadata: Sequence[dict[str, str]],
) -> str:
    """Return a descriptive cache token for the Supplementary Figure 3A data sets."""
    dataset_tokens = [
        _format_panel_a_cache_token(
            f"{dataset['animal_name']}-{dataset['date']}-"
            f"{dataset['dark_epoch']}-{dataset['light_epoch']}"
        )
        for dataset in dataset_metadata
    ]
    token = "_".join(dataset_tokens) or "none"
    if len(token) <= PANEL_A_CACHE_DATASET_TOKEN_LIMIT:
        return token

    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    prefix = "_".join(dataset_tokens[:2])
    return _format_panel_a_cache_token(
        f"{prefix}_{len(dataset_tokens)}datasets_{digest}"
    )


def build_panel_a_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    order_mode: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Return metadata identifying one Supplementary Figure 3A heatmap cache."""
    if order_mode not in PANEL_A_ORDER_MODES:
        raise ValueError(f"Unknown panel A order_mode {order_mode!r}.")
    dataset_metadata = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_metadata.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": dark_epoch if dark_epoch is not None else dataset_dark_epoch,
                "light_epoch": get_light_epoch(animal_name, date, light_epoch),
            }
        )
    return {
        "cache_version": PANEL_A_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "A",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "light_epoch_argument": light_epoch,
        "dark_epoch_argument": dark_epoch,
        "datasets": dataset_metadata,
        "trajectory_types": list(PANEL_B_TRAJECTORY_TYPES),
        "linear_position_orientation": PANEL_B_LINEAR_POSITION_ORIENTATION,
        "order_mode": order_mode,
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "min_light_stability_correlation": float(
            REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION
        ),
        "firing_rate_normalization": "unit_max_per_trajectory",
        "order_trials": "figure_1d_dark_odd",
        "display_trials": "light_all",
        "source_unit_set": "figure_1d_cache_v6",
    }


def build_panel_a_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the cache path for one Supplementary Figure 3A heatmap payload."""
    dataset_token = _build_panel_a_dataset_cache_token(metadata["datasets"])
    region_token = _format_panel_a_cache_token(metadata["region"])
    light_epochs = [
        _format_panel_a_cache_token(dataset["light_epoch"])
        for dataset in metadata["datasets"]
    ]
    unique_light_epochs = list(dict.fromkeys(light_epochs))
    light_epoch_token = (
        unique_light_epochs[0]
        if len(unique_light_epochs) == 1
        else "mixed-" + "_".join(unique_light_epochs)
    )
    order_token = _format_panel_a_cache_token(metadata["order_mode"])
    filename = (
        f"{PANEL_A_CACHE_PREFIX}_{region_token}_light{light_epoch_token}"
        f"_datasets-{dataset_token}"
        f"_order{order_token}"
        f"_minlightstab"
        f"{_format_panel_a_cache_number(metadata['min_light_stability_correlation'])}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_a_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_a_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_a_cache_array_name(order_trajectory: str, plot_trajectory: str) -> str:
    """Return the cache array name for one Supplementary Figure 3A panel."""
    return f"{order_trajectory}__{plot_trajectory}"


def _panel_a_cache_unit_order_array_name(order_trajectory: str) -> str:
    """Return the cache array name for one Supplementary Figure 3A unit order."""
    return f"unit_order__{order_trajectory}"


def save_panel_a_cache(
    cache_path: Path,
    panels: dict[tuple[str, str], np.ndarray],
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write one Supplementary Figure 3A heatmap cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        PANEL_A_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    trajectory_types = tuple(str(trajectory) for trajectory in metadata["trajectory_types"])
    for order_trajectory in trajectory_types:
        payload[_panel_a_cache_unit_order_array_name(order_trajectory)] = np.asarray(
            ordered_unit_keys_by_trajectory[order_trajectory],
            dtype=str,
        )
        for plot_trajectory in trajectory_types:
            payload[_panel_a_cache_array_name(order_trajectory, plot_trajectory)] = (
                np.asarray(panels[(order_trajectory, plot_trajectory)], dtype=float)
            )
    np.savez_compressed(cache_path, **payload)


def load_panel_a_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]] | None:
    """Return cached Supplementary Figure 3A heatmaps when metadata matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_A_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Supplementary Figure 3A cache at {cache_path}.")
                return None

            trajectory_types = tuple(
                str(trajectory) for trajectory in expected_metadata["trajectory_types"]
            )
            panels: dict[tuple[str, str], np.ndarray] = {}
            ordered_unit_keys_by_trajectory: dict[str, np.ndarray] = {}
            for order_trajectory in trajectory_types:
                ordered_unit_keys_by_trajectory[order_trajectory] = np.asarray(
                    data[_panel_a_cache_unit_order_array_name(order_trajectory)],
                    dtype=str,
                )
                for plot_trajectory in trajectory_types:
                    panels[(order_trajectory, plot_trajectory)] = np.asarray(
                        data[_panel_a_cache_array_name(order_trajectory, plot_trajectory)],
                        dtype=float,
                    )
            return panels, ordered_unit_keys_by_trajectory
    except Exception as exc:
        print(f"Ignoring unreadable Supplementary Figure 3A cache at {cache_path}: {exc}")
        return None


def build_panel_a_heatmap_payloads(
    *,
    figure_1d_ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    light_curve_sets: Sequence[dict[str, Any]],
    position_bin_count: int,
) -> dict[str, tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]]:
    """Return the Supplementary Figure 3A Fig. 1D-order heatmap payload."""
    figure_1d_panels = build_figure_1d_ordered_light_panel_values(
        ordered_unit_keys_by_trajectory=figure_1d_ordered_unit_keys_by_trajectory,
        light_curve_sets=light_curve_sets,
        position_bin_count=position_bin_count,
        curve_key="all_curves",
    )
    return {
        PANEL_A_FIGURE_1D_ORDER_MODE: (
            figure_1d_panels,
            figure_1d_ordered_unit_keys_by_trajectory,
        ),
    }


def load_dark_ordered_light_panel_values(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    figure_1d_cache_dir: Path | None = None,
    panel_a_cache_dir: Path | None = None,
    refresh_panel_a_cache: bool = False,
) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """Load Fig. 1D cells in light, cached for the displayed row order."""
    figure_1d_cache_dir = (
        DEFAULT_OUTPUT_DIR / "cache"
        if figure_1d_cache_dir is None
        else Path(figure_1d_cache_dir)
    )
    panel_a_cache_dir = (
        DEFAULT_OUTPUT_DIR / "cache"
        if panel_a_cache_dir is None
        else Path(panel_a_cache_dir)
    )
    figure_1d_datasets = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        figure_1d_datasets.append(
            (
                animal_name,
                date,
                dark_epoch if dark_epoch is not None else dataset_dark_epoch,
            )
        )

    metadata_by_order_mode = {
        order_mode: build_panel_a_cache_metadata(
            data_root=data_root,
            datasets=figure_1d_datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            order_mode=order_mode,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        )
        for order_mode in PANEL_A_ORDER_MODES
    }
    cache_paths_by_order_mode = {
        order_mode: build_panel_a_cache_path(panel_a_cache_dir, metadata)
        for order_mode, metadata in metadata_by_order_mode.items()
    }
    loaded_payloads = {}
    if not refresh_panel_a_cache:
        for order_mode, cache_path in cache_paths_by_order_mode.items():
            cached_payload = load_panel_a_cache(
                cache_path,
                metadata_by_order_mode[order_mode],
            )
            if cached_payload is not None:
                print(f"Loaded Supplementary Figure 3A cache from {cache_path}.")
                loaded_payloads[order_mode] = cached_payload

    missing_order_modes = [
        order_mode
        for order_mode in PANEL_A_ORDER_MODES
        if order_mode not in loaded_payloads
    ]
    if missing_order_modes:
        light_curve_sets = []
        for dataset in datasets:
            animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
            light_curve_sets.append(
                compute_light_epoch_all_trial_tuning_curves(
                    animal_name=animal_name,
                    date=date,
                    data_root=data_root,
                    region=region,
                    light_epoch=light_epoch,
                    position_bin_count=position_bin_count,
                    position_offset=position_offset,
                    speed_threshold_cm_s=speed_threshold_cm_s,
                    sigma_bins=sigma_bins,
                    use_trajectory_direction=True,
                )
            )
        _figure_1d_panels, ordered_unit_keys_by_trajectory = (
            load_or_compute_panel_d_heatmap_payload(
                data_root=data_root,
                datasets=figure_1d_datasets,
                region=region,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                panel_d_cache_dir=figure_1d_cache_dir,
                refresh_panel_d_cache=False,
                require_ordered_unit_keys=True,
            )
        )
        light_stable_unit_keys = select_unit_keys_by_light_tuning_stability(
            data_root=data_root,
            datasets=figure_1d_datasets,
            region=region,
            light_epoch=light_epoch,
        )
        ordered_unit_keys_by_trajectory = filter_ordered_unit_keys_by_unit_set(
            ordered_unit_keys_by_trajectory,
            light_stable_unit_keys,
        )
        computed_payloads = build_panel_a_heatmap_payloads(
            figure_1d_ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
            light_curve_sets=light_curve_sets,
            position_bin_count=position_bin_count,
        )
        for order_mode in missing_order_modes:
            panels, ordered_unit_keys = computed_payloads[order_mode]
            cache_path = cache_paths_by_order_mode[order_mode]
            save_panel_a_cache(
                cache_path,
                panels,
                ordered_unit_keys,
                metadata_by_order_mode[order_mode],
            )
            print(f"Saved Supplementary Figure 3A cache to {cache_path}.")
            loaded_payloads[order_mode] = panels, ordered_unit_keys

    return {
        order_mode: loaded_payloads[order_mode][0]
        for order_mode in PANEL_A_ORDER_MODES
    }


def set_heatmap_display_style(
    heatmap_axes: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Set the colormap and display range for all images in one heatmap grid."""
    for ax in np.asarray(heatmap_axes, dtype=object).ravel():
        for image in ax.images:
            image.set_cmap(cmap)
            image.set_clim(vmin=float(vmin), vmax=float(vmax))


def plot_dark_ordered_light_heatmap_regions(
    heatmap_axes: np.ndarray,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    figure_1d_cache_dir: Path | None = None,
    panel_a_cache_dir: Path | None = None,
    refresh_panel_a_cache: bool = False,
    order_mode: str = PANEL_A_FIGURE_1D_ORDER_MODE,
) -> "AxesImage | None":
    """Plot light-epoch heatmaps with rows sorted by Figure 1D dark order."""
    if order_mode not in PANEL_A_ORDER_MODES:
        raise ValueError(f"Unknown panel A order_mode {order_mode!r}.")
    color_image = None
    for region_index, region in enumerate(regions):
        panels_by_order_mode = load_dark_ordered_light_panel_values(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            figure_1d_cache_dir=figure_1d_cache_dir,
            panel_a_cache_dir=panel_a_cache_dir,
            refresh_panel_a_cache=refresh_panel_a_cache,
        )
        panels = panels_by_order_mode[order_mode]
        start_row = region_index * len(PANEL_B_TRAJECTORY_TYPES)
        stop_row = start_row + len(PANEL_B_TRAJECTORY_TYPES)
        image = plot_pooled_heatmap_grid(
            heatmap_axes[start_row:stop_row, :],
            panels,
            trajectory_types=PANEL_B_TRAJECTORY_TYPES,
            axis_orientation=PANEL_B_LINEAR_POSITION_ORIENTATION,
            cmap=REORDERED_HEATMAP_CMAP,
        )
        set_heatmap_display_style(
            heatmap_axes[start_row:stop_row, :],
            cmap=REORDERED_HEATMAP_CMAP,
            vmin=0.0,
            vmax=REORDERED_HEATMAP_VMAX,
        )
        for heatmap_ax in heatmap_axes[start_row:stop_row, :].ravel():
            add_segment_boundary_lines(heatmap_ax)
        if color_image is None and image is not None:
            color_image = image
    return color_image


def make_supplementary_figure_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    figure_1d_cache_dir: Path | None = None,
    panel_a_cache_dir: Path | None = None,
    refresh_panel_a_cache: bool = False,
) -> Path:
    """Build and save Supplementary Figure 3."""
    import matplotlib.pyplot as plt

    datasets = [normalize_dataset_id(dataset) for dataset in datasets]

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=False,
    )
    if not datasets:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved Supplementary Figure 3 to {output_path}")
        return output_path

    outer_grid = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[
            DEFAULT_REORDERED_HEATMAP_HEIGHT_MM,
            DEFAULT_SECTION_SPACER_MM,
            DEFAULT_MOTOR_GRID_HEIGHT_MM,
            DEFAULT_BOTTOM_SECTION_SPACER_MM,
            DEFAULT_MOTOR_SUMMARY_HEIGHT_MM,
        ],
        hspace=0.04,
        left=PANEL_A_GRID_LEFT,
        right=PANEL_A_GRID_RIGHT,
        top=PANEL_A_GRID_TOP,
        bottom=PANEL_A_GRID_BOTTOM,
    )
    panel_a_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            (1.0 - PANEL_A_CV_PCA_SIZE_FRACTION) / 2.0,
            PANEL_A_CV_PCA_SIZE_FRACTION,
            (1.0 - PANEL_A_CV_PCA_SIZE_FRACTION) / 2.0,
        ],
        wspace=0.0,
    )
    panel_a_axis = fig.add_subplot(panel_a_grid[0, 1])
    panel_a_cv_pca_table = load_panel_a_cv_pca_participation_ratio_table(
        data_root=data_root,
        datasets=datasets,
    )
    plot_panel_a_cv_pca_participation_ratios(panel_a_axis, panel_a_cv_pca_table)

    spacer_axis = fig.add_subplot(outer_grid[1, 0])
    spacer_axis.axis("off")

    motor_grid = outer_grid[2, 0].subgridspec(
        nrows=len(MOTOR_VARIABLES),
        ncols=len(PANEL_B_TRAJECTORY_TYPES),
        hspace=MOTOR_GRID_HSPACE,
        wspace=MOTOR_GRID_WSPACE,
    )
    motor_axes = np.asarray(
        [
            [
                fig.add_subplot(motor_grid[row_index, column_index])
                for column_index in range(len(PANEL_B_TRAJECTORY_TYPES))
            ]
            for row_index in range(len(MOTOR_VARIABLES))
        ],
        dtype=object,
    )
    motor_table = load_panel_b_motor_progression_table(
        data_root=data_root,
        datasets=datasets,
    )
    plot_panel_b_motor_progression_grid(
        motor_axes,
        motor_table,
        datasets=datasets,
    )

    bottom_spacer_axis = fig.add_subplot(outer_grid[3, 0])
    bottom_spacer_axis.axis("off")
    motor_summary_grid = outer_grid[4, 0].subgridspec(
        nrows=1,
        ncols=len(PANEL_B_TRAJECTORY_TYPES),
        wspace=MOTOR_SUMMARY_GRID_WSPACE,
    )
    motor_summary_axes = [
        fig.add_subplot(motor_summary_grid[0, column_index])
        for column_index in range(len(PANEL_B_TRAJECTORY_TYPES))
    ]
    motor_profile_correlation_table = build_panel_c_motor_profile_correlation_table(
        motor_table,
        datasets=datasets,
    )
    plot_panel_c_motor_profile_correlations(
        motor_summary_axes,
        motor_profile_correlation_table,
        datasets=datasets,
    )

    fig.canvas.draw()
    add_centered_axis_text(
        fig,
        motor_summary_axes,
        "Dark-light motor profile correlation",
        y_offset=0.025,
        fontsize=PANEL_TITLE_FONTSIZE,
    )
    label_axis(motor_summary_axes[0], "C", x=-0.30, y=1.05)
    add_centered_axis_text(
        fig,
        motor_axes[0, :],
        "Motor variables over normalized path progression across dark and light",
        y_offset=0.010,
        fontsize=PANEL_TITLE_FONTSIZE,
    )
    label_axis(motor_axes[0, 0], "B", x=-0.28, y=1.05)
    label_axis(
        panel_a_axis,
        "A",
        x=-0.035,
        y=1.05,
    )

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Supplementary Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 3 generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Supplementary Figure 3 cvPCA and per-animal "
            "Figure 3C-E panels."
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
            "Animal/date data set to include as animal:date. May be repeated. "
            "Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default=DEFAULT_REGIONS[0],
        help=f"Region to include. Default: {DEFAULT_REGIONS[0]}.",
    )
    parser.add_argument(
        "--light-epoch",
        default=None,
        help=(
            "Light run epoch for the reordered heatmap and Figure 3D-F panels. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help="Dark run epoch. Default: registry value for each animal.",
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
        "--panel-a-cache-dir",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility option; current Supplementary Figure 3A "
            "reads cvPCA parquet summaries."
        ),
    )
    parser.add_argument(
        "--refresh-panel-a-cache",
        action="store_true",
        help=(
            "Deprecated compatibility option; current Supplementary Figure 3A "
            "does not use cached panel data."
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
    """Run Supplementary Figure 3 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_a_cache_dir=args.panel_a_cache_dir,
        refresh_panel_a_cache=args.refresh_panel_a_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
