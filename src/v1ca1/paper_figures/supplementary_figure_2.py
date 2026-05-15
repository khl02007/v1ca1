from __future__ import annotations

"""Generate Supplementary Figure 2 ripple-GLM scatter panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, get_analysis_path
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_figure_epoch_dataset_id,
)
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_SELECTION,
    DEFAULT_RIPPLE_WINDOW_S,
    FIGURE_FORMATS,
    MODEL_COLOR,
    NONSIGNIFICANT_COLOR,
    PANEL_E_GLM_SOURCE_WINDOW_OFFSET_S,
    PANEL_E_GLM_TARGET_WINDOW_OFFSETS_S,
    get_ripple_glm_model_window_path,
    load_glm_dark_activity_devexp_tables,
    load_glm_epoch_summary_tables,
    parse_dataset_id,
)
from v1ca1.paper_figures.style import (
    EPOCH_TYPE_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "supplementary_figure_2"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_SECTION_HEADER_HEIGHT_MM = 8.0
DEFAULT_DATASET_ROW_HEIGHT_MM = 13.0
DEFAULT_PER_ANIMAL_ROW_HEIGHT_MM = 20.0
DEFAULT_SECTION_GAP_MM = 5.0
RIPPLE_SELECTION_MODE_CHOICES = ("allripples", "single")
DEFAULT_RIPPLE_SELECTION_MODES = (DEFAULT_RIPPLE_SELECTION,)
DEFAULT_EPOCH_TYPES = ("light", "dark", "sleep")
DEFAULT_PER_ANIMAL_EPOCH_TYPES = ("light", "sleep")
EPOCH_TYPE_LABELS = {
    "light": "Light",
    "dark": "Dark",
    "sleep": "Sleep",
}
PER_ANIMAL_EPOCH_LABELS = {
    "light": "Run",
    "sleep": "Sleep",
}
SELECTION_LABELS = {
    "allripples": "All ripples",
    "single": "Single-ripple windows",
}
PLOT_X_LIMITS = (-0.1, 0.5)
PANEL_GRID_LEFT = 0.13
PANEL_GRID_RIGHT = 0.99
PANEL_GRID_BOTTOM = 0.11
PANEL_GRID_TOP = 0.88
DATASET_ROW_GAP = 0.045
EPOCH_COLUMN_GAP = 0.018
SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE = 0.005


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


def get_dataset_analysis_path(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the analysis directory for one animal/date pair."""
    return get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=Path(data_root),
    )


def format_dataset_epoch_label(dataset_row: dict[str, Any]) -> str:
    """Return a compact row label for one data-set epoch."""
    animal_name = str(dataset_row["animal_name"])
    date = str(dataset_row["date"])
    epoch = str(dataset_row.get("epoch", ""))
    epoch_type = dataset_row.get("epoch_type")
    if epoch_type is None:
        return f"{animal_name}\n{date}"
    epoch_label = EPOCH_TYPE_LABELS.get(str(epoch_type), str(epoch_type).title())
    return f"{animal_name}\n{date}\n{epoch_label} {epoch}"


def get_epoch_type_color(epoch_type: str | None) -> str:
    """Return the Figure 2 color for one epoch type."""
    if epoch_type is None:
        return MODEL_COLOR
    return EPOCH_TYPE_COLORS.get(str(epoch_type), MODEL_COLOR)


def iter_dataset_epoch_rows(
    dataset: DatasetId,
    epoch_types: Sequence[str] = DEFAULT_EPOCH_TYPES,
) -> list[dict[str, str]]:
    """Return registered epoch rows to include for one data set."""
    animal_name, date, light_epoch, dark_epoch, sleep_epoch = (
        normalize_figure_epoch_dataset_id(dataset)
    )
    epoch_by_type = {
        "light": light_epoch,
        "dark": dark_epoch,
        "sleep": sleep_epoch,
    }
    rows = []
    for epoch_type in epoch_types:
        epoch = epoch_by_type[str(epoch_type)]
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dataset_dark_epoch": dark_epoch,
                "epoch_type": str(epoch_type),
                "epoch": epoch,
            }
        )
    return rows


def format_target_window_label(
    target_window_offset_s: float,
    *,
    target_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
) -> str:
    """Return a compact target-window label in milliseconds."""
    start_ms = int(round(1000.0 * float(target_window_offset_s)))
    end_ms = int(round(1000.0 * (float(target_window_offset_s) + float(target_window_s))))
    return f"{start_ms} to {end_ms} ms"


def get_available_offset_glm_artifacts(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_selection: str,
    target_window_offsets_s: Sequence[float] = PANEL_E_GLM_TARGET_WINDOW_OFFSETS_S,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> list[dict[str, Any]]:
    """Return available CA1-to-V1 target-window offset GLM artifacts."""
    artifacts = []
    for target_window_offset_s in target_window_offsets_s:
        path = get_ripple_glm_model_window_path(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            source_window_s=ripple_window_s,
            source_window_offset_s=PANEL_E_GLM_SOURCE_WINDOW_OFFSET_S,
            target_window_s=ripple_window_s,
            target_window_offset_s=float(target_window_offset_s),
            ripple_selection=ripple_selection,
            ridge_strength=ridge_strength,
        )
        if not path.exists():
            continue
        artifacts.append(
            {
                "target_window_offset_s": float(target_window_offset_s),
                "target_window_label": format_target_window_label(
                    float(target_window_offset_s),
                    target_window_s=ripple_window_s,
                ),
                "path": path,
            }
        )
    return artifacts


def load_glm_scatter_table(path: Path) -> Any:
    """Load one ripple-GLM artifact as a per-unit summary table."""
    import pandas as pd
    import xarray as xr

    dataset = xr.open_dataset(path)
    try:
        required_variables = ("ripple_devexp_mean", "ripple_devexp_p_value")
        missing_variables = [
            variable for variable in required_variables if variable not in dataset
        ]
        if missing_variables:
            raise ValueError(
                f"Ripple-GLM artifact {path} is missing variables "
                f"{missing_variables!r}."
            )
        unit_ids = np.asarray(dataset.coords.get("unit", np.arange(dataset.sizes["unit"])))
        table = pd.DataFrame(
            {
                "unit_id": unit_ids,
                "ripple_devexp_mean": np.asarray(
                    dataset["ripple_devexp_mean"].values,
                    dtype=float,
                ),
                "ripple_devexp_p_value": np.asarray(
                    dataset["ripple_devexp_p_value"].values,
                    dtype=float,
                ),
            }
        )
        table["n_ripples"] = int(
            dataset.attrs.get(
                "n_ripples_after_selection",
                dataset.attrs.get("n_ripples", 0),
            )
        )
        table["n_shuffles"] = int(dataset.sizes.get("shuffle", 0))
        table["source_path"] = str(path)
    finally:
        dataset.close()
    return table


def load_available_glm_scatter_payload(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    ripple_selection_modes: Sequence[str] = DEFAULT_RIPPLE_SELECTION_MODES,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load available target-offset ripple-GLM artifacts by data-set epoch."""
    payload_rows: dict[str, list[dict[str, Any]]] = {}
    for ripple_selection in ripple_selection_modes:
        selection_rows = []
        for dataset in datasets:
            for epoch_row in iter_dataset_epoch_rows(dataset):
                animal_name = epoch_row["animal_name"]
                date = epoch_row["date"]
                epoch = epoch_row["epoch"]
                artifacts = []
                for artifact in get_available_offset_glm_artifacts(
                    data_root,
                    animal_name=animal_name,
                    date=date,
                    epoch=epoch,
                    ripple_selection=ripple_selection,
                    ripple_window_s=ripple_window_s,
                    ridge_strength=ridge_strength,
                ):
                    path = artifact["path"]
                    artifacts.append(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "epoch_type": epoch_row["epoch_type"],
                            "epoch": epoch,
                            "ripple_selection": ripple_selection,
                            "target_window_offset_s": artifact[
                                "target_window_offset_s"
                            ],
                            "target_window_label": artifact["target_window_label"],
                            "summary_table": load_glm_scatter_table(path),
                            "source_path": str(path),
                        }
                    )
                selection_rows.append(
                    {
                        "dataset": (
                            animal_name,
                            date,
                            epoch_row["dataset_dark_epoch"],
                        ),
                        "animal_name": animal_name,
                        "date": date,
                        "epoch_type": epoch_row["epoch_type"],
                        "epoch": epoch,
                        "ripple_selection": ripple_selection,
                        "artifacts": artifacts,
                    }
                )
        payload_rows[ripple_selection] = selection_rows
    return {
        "rows_by_selection": payload_rows,
        "ripple_selection_modes": tuple(ripple_selection_modes),
        "epoch_types": tuple(DEFAULT_EPOCH_TYPES),
        "ripple_window_s": float(ripple_window_s),
        "ridge_strength": float(ridge_strength),
    }


def get_payload_neglog_p_limit(payload: dict[str, Any]) -> float:
    """Return a shared y-axis maximum for all loaded scatter panels."""
    neglog_values = []
    for selection_rows in payload["rows_by_selection"].values():
        for dataset_row in selection_rows:
            for artifact in dataset_row["artifacts"]:
                table = artifact["summary_table"]
                p_values = np.asarray(table["ripple_devexp_p_value"], dtype=float)
                p_values = p_values[np.isfinite(p_values)]
                if p_values.size:
                    neglog_values.append(-np.log10(np.clip(p_values, 1e-12, 1.0)))
    if not neglog_values:
        return -np.log10(SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE) + 0.35
    return max(
        -np.log10(SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE) + 0.35,
        float(np.nanmax(np.concatenate(neglog_values))) + 0.35,
    )


def plot_glm_scatter_axis(
    ax: "Axes",
    summary_table: Any,
    *,
    title: str,
    color: str,
    x_limits: tuple[float, float],
    y_limit: float,
    show_yticklabels: bool,
    show_xticklabels: bool,
) -> None:
    """Plot one Figure-2C-style deviance/significance scatter."""
    values = np.asarray(summary_table["ripple_devexp_mean"], dtype=float)
    p_values = np.asarray(summary_table["ripple_devexp_p_value"], dtype=float)
    valid = np.isfinite(values) & np.isfinite(p_values)

    ax.axvline(0.0, color="0.45", linewidth=0.45, zorder=1)
    ax.axhline(
        -np.log10(SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE),
        color="0.25",
        linestyle="--",
        linewidth=0.55,
        zorder=1,
    )
    if np.any(valid):
        finite_values = values[valid]
        finite_p_values = p_values[valid]
        neglog_p = -np.log10(np.clip(finite_p_values, 1e-12, 1.0))
        significant = finite_p_values < SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE
        if np.any(~significant):
            ax.scatter(
                finite_values[~significant],
                neglog_p[~significant],
                s=3.2,
                color=NONSIGNIFICANT_COLOR,
                alpha=0.42,
                edgecolors="none",
                zorder=2,
            )
        if np.any(significant):
            ax.scatter(
                finite_values[significant],
                neglog_p[significant],
                s=3.8,
                color=color,
                alpha=0.52,
                edgecolors="none",
                zorder=3,
            )
        ax.text(
            0.96,
            0.05,
            f"n={int(np.sum(valid))}\nsig={np.mean(significant):.2f}",
            ha="right",
            va="bottom",
            fontsize=3.9,
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No finite\nvalues",
            ha="center",
            va="center",
            fontsize=4.8,
            transform=ax.transAxes,
        )

    ax.set_title(title, fontsize=4.8, pad=1.0)
    ax.set_xlim(*x_limits)
    ax.set_ylim(0.0, y_limit)
    if not show_xticklabels:
        ax.set_xticklabels([])
    if not show_yticklabels:
        ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.2, length=1.3, pad=1)


def plot_selection_scatter_grid(
    ax: "Axes",
    selection_rows: Sequence[dict[str, Any]],
    *,
    selection_label: str,
    x_limits: tuple[float, float],
    y_limit: float,
) -> list[Any]:
    """Plot one section of per-data-set, per-offset GLM scatter panels."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    if not selection_rows:
        ax.text(0.5, 0.5, "No data sets", ha="center", va="center", fontsize=6.0)
        return []

    row_axes = []
    n_rows = len(selection_rows)
    grid_height = PANEL_GRID_TOP - PANEL_GRID_BOTTOM
    row_gap = min(
        DATASET_ROW_GAP,
        grid_height * 0.18 / max(n_rows - 1, 1),
    )
    row_height = (grid_height - row_gap * (n_rows - 1)) / n_rows
    grid_width = PANEL_GRID_RIGHT - PANEL_GRID_LEFT
    plotted_epochs = sorted(
        {
            str(artifact["epoch"])
            for dataset_row in selection_rows
            for artifact in dataset_row["artifacts"]
        }
    )
    if len(plotted_epochs) == 1:
        header_label = f"{selection_label} ({plotted_epochs[0]})"
    else:
        header_label = selection_label
    plotted_offsets = sorted(
        {
            float(artifact["target_window_offset_s"])
            for dataset_row in selection_rows
            for artifact in dataset_row["artifacts"]
        }
    )
    offset_labels = {
        offset: format_target_window_label(offset)
        for offset in plotted_offsets
    }

    ax.text(
        PANEL_GRID_LEFT,
        0.98,
        header_label,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        transform=ax.transAxes,
    )
    for row_index, dataset_row in enumerate(selection_rows):
        row_bottom = (
            PANEL_GRID_BOTTOM
            + grid_height
            - (row_index + 1) * row_height
            - row_index * row_gap
        )
        ax.text(
            0.01,
            row_bottom + 0.5 * row_height,
            format_dataset_epoch_label(dataset_row),
            ha="left",
            va="center",
            fontsize=4.5,
            transform=ax.transAxes,
        )
        artifacts = dataset_row["artifacts"]
        if not artifacts:
            empty_ax = ax.inset_axes([PANEL_GRID_LEFT, row_bottom, grid_width, row_height])
            empty_ax.axis("off")
            empty_ax.text(
                0.5,
                0.5,
                "No offset artifacts",
                ha="center",
                va="center",
                fontsize=5.0,
                color="0.35",
                transform=empty_ax.transAxes,
            )
            row_axes.append(empty_ax)
            continue

        artifacts_by_offset = {
            float(artifact["target_window_offset_s"]): artifact
            for artifact in artifacts
        }
        n_columns = max(len(plotted_offsets), 1)
        column_width = (
            grid_width - EPOCH_COLUMN_GAP * (n_columns - 1)
        ) / n_columns
        for column_index, offset in enumerate(plotted_offsets):
            column_left = PANEL_GRID_LEFT + column_index * (
                column_width + EPOCH_COLUMN_GAP
            )
            scatter_ax = ax.inset_axes(
                [column_left, row_bottom, column_width, row_height]
            )
            artifact = artifacts_by_offset.get(offset)
            if artifact is None:
                scatter_ax.axis("off")
                scatter_ax.set_title(
                    offset_labels[offset],
                    fontsize=4.8,
                    pad=1.0,
                )
                scatter_ax.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    fontsize=4.6,
                    color="0.4",
                    transform=scatter_ax.transAxes,
                )
                row_axes.append(scatter_ax)
                continue
            plot_glm_scatter_axis(
                scatter_ax,
                artifact["summary_table"],
                title=offset_labels[offset],
                color=get_epoch_type_color(
                    artifact.get("epoch_type", dataset_row.get("epoch_type"))
                ),
                x_limits=x_limits,
                y_limit=y_limit,
                show_yticklabels=column_index == 0,
                show_xticklabels=row_index == n_rows - 1,
            )
            row_axes.append(scatter_ax)

    ax.text(
        0.5 * (PANEL_GRID_LEFT + PANEL_GRID_RIGHT),
        0.015,
        "Ripple deviance explained",
        ha="center",
        va="bottom",
        fontsize=6.0,
        transform=ax.transAxes,
    )
    ax.text(
        PANEL_GRID_LEFT - 0.065,
        0.5 * (PANEL_GRID_BOTTOM + PANEL_GRID_TOP),
        "-log10 shuffle p",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        transform=ax.transAxes,
    )
    return row_axes


def resolve_primary_ripple_selection(
    ripple_selection_modes: Sequence[str],
) -> str:
    """Return the ripple-selection mode used by the per-animal scatter figure."""
    if not ripple_selection_modes:
        return DEFAULT_RIPPLE_SELECTION
    return str(tuple(ripple_selection_modes)[0])


def filter_glm_epoch_payloads(
    payloads: Sequence[dict[str, Any]],
    epoch_types: Sequence[str],
) -> list[dict[str, Any]]:
    """Return GLM payloads ordered by requested epoch type."""
    payload_by_epoch_type = {
        str(payload["epoch_type"]): payload
        for payload in payloads
        if "epoch_type" in payload
    }
    return [
        payload_by_epoch_type[str(epoch_type)]
        for epoch_type in epoch_types
        if str(epoch_type) in payload_by_epoch_type
    ]


def load_per_animal_glm_scatter_payload(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    epoch_types: Sequence[str] = DEFAULT_PER_ANIMAL_EPOCH_TYPES,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load Figure-2C-style GLM summaries for per-animal run/sleep scatter panels."""
    epoch_tables = load_glm_epoch_summary_tables(
        data_root,
        datasets,
        ripple_window_s=ripple_window_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )
    selected_epoch_tables = filter_glm_epoch_payloads(epoch_tables, epoch_types)
    return {
        "epoch_tables": selected_epoch_tables,
        "epoch_types": tuple(epoch_types),
        "ripple_selection": ripple_selection,
        "ripple_window_s": float(ripple_window_s),
        "ridge_strength": float(ridge_strength),
    }


def iter_per_animal_glm_keys(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return animal/date keys in first-seen order across GLM epoch tables."""
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for epoch_payload in payload.get("epoch_tables", []):
        table = epoch_payload.get("summary_table")
        if table is None or len(table) == 0:
            continue
        if not {"animal_name", "date"}.issubset(table.columns):
            continue
        key_rows = table[["animal_name", "date"]].drop_duplicates()
        for animal_name, date in key_rows.itertuples(index=False, name=None):
            key = (str(animal_name), str(date))
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def subset_glm_table_for_animal_date(table: Any, animal_name: str, date: str) -> Any:
    """Return one animal/date subset from a pooled GLM summary table."""
    if {"animal_name", "date"}.issubset(table.columns):
        return table[
            (table["animal_name"].astype(str) == str(animal_name))
            & (table["date"].astype(str) == str(date))
        ]
    return table


def get_per_animal_neglog_p_limit(payload: dict[str, Any]) -> float:
    """Return a shared y-axis maximum for per-animal GLM scatter panels."""
    neglog_values = []
    for epoch_payload in payload.get("epoch_tables", []):
        table = epoch_payload.get("summary_table")
        if table is None or len(table) == 0:
            continue
        p_values = np.asarray(table["ripple_devexp_p_value"], dtype=float)
        p_values = p_values[np.isfinite(p_values)]
        if p_values.size:
            neglog_values.append(-np.log10(np.clip(p_values, 1e-12, 1.0)))
    if not neglog_values:
        return -np.log10(SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE) + 0.35
    return max(
        -np.log10(SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE) + 0.35,
        float(np.nanmax(np.concatenate(neglog_values))) + 0.35,
    )


def iter_dark_activity_dataset_keys(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return animal/date keys for dark-rate versus deviance plots."""
    table = payload.get("devexp_table")
    if table is None or len(table) == 0:
        return []
    if not {"animal_name", "date"}.issubset(table.columns):
        return []
    key_rows = table[["animal_name", "date"]].drop_duplicates()
    return [
        (str(animal_name), str(date))
        for animal_name, date in key_rows.itertuples(index=False, name=None)
    ]


def get_dark_activity_scatter_limits(
    payload: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return shared x/y limits for deviance versus dark movement rate plots."""
    table = payload.get("devexp_table")
    if table is None or len(table) == 0:
        return (-0.05, 0.40), (0.1, 100.0)

    x_values = np.asarray(table["ripple_devexp_mean"], dtype=float)
    y_values = np.asarray(table["dark_firing_rate_hz"], dtype=float)
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values) & (y_values > 0.0)]
    if finite_x.size:
        x_min = min(-0.05, float(np.nanmin(finite_x)) - 0.02)
        x_max = max(0.40, float(np.nanmax(finite_x)) + 0.02)
    else:
        x_min, x_max = -0.05, 0.40
    if finite_y.size:
        y_min = max(0.03, float(np.nanmin(finite_y)) / 1.4)
        y_max = max(y_min * 1.5, float(np.nanmax(finite_y)) * 1.4)
    else:
        y_min, y_max = 0.1, 100.0
    return (x_min, x_max), (y_min, y_max)


def plot_per_animal_glm_scatter_grid(
    ax: "Axes",
    payload: dict[str, Any],
    *,
    x_limits: tuple[float, float] = (-0.05, 0.40),
    y_limit: float | None = None,
) -> list[Any]:
    """Plot Figure 2C's run/sleep GLM scatter on separate rows for each animal."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    epoch_tables = list(payload.get("epoch_tables", []))
    animal_date_keys = iter_per_animal_glm_keys(payload)
    if not epoch_tables or not animal_date_keys:
        ax.text(
            0.5,
            0.5,
            "No GLM data",
            ha="center",
            va="center",
            fontsize=6.0,
            transform=ax.transAxes,
        )
        return []

    y_limit = get_per_animal_neglog_p_limit(payload) if y_limit is None else y_limit
    n_rows = len(animal_date_keys)
    n_columns = len(epoch_tables)
    grid_height = PANEL_GRID_TOP - PANEL_GRID_BOTTOM
    row_gap = min(DATASET_ROW_GAP, grid_height * 0.18 / max(n_rows - 1, 1))
    row_height = (grid_height - row_gap * (n_rows - 1)) / n_rows
    grid_width = PANEL_GRID_RIGHT - PANEL_GRID_LEFT
    column_width = (grid_width - EPOCH_COLUMN_GAP * (n_columns - 1)) / n_columns

    ax.text(
        PANEL_GRID_LEFT,
        0.98,
        f"Figure 2C scatter by animal ({payload['ripple_selection']})",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        transform=ax.transAxes,
    )

    child_axes = []
    for row_index, (animal_name, date) in enumerate(animal_date_keys):
        row_bottom = (
            PANEL_GRID_BOTTOM
            + grid_height
            - (row_index + 1) * row_height
            - row_index * row_gap
        )
        ax.text(
            0.01,
            row_bottom + 0.5 * row_height,
            f"{animal_name}\n{date}",
            ha="left",
            va="center",
            fontsize=5.0,
            transform=ax.transAxes,
        )
        for column_index, epoch_payload in enumerate(epoch_tables):
            column_left = PANEL_GRID_LEFT + column_index * (
                column_width + EPOCH_COLUMN_GAP
            )
            scatter_ax = ax.inset_axes(
                [column_left, row_bottom, column_width, row_height]
            )
            epoch_type = str(epoch_payload["epoch_type"])
            table = subset_glm_table_for_animal_date(
                epoch_payload["summary_table"],
                animal_name,
                date,
            )
            title = (
                PER_ANIMAL_EPOCH_LABELS.get(epoch_type, EPOCH_TYPE_LABELS.get(epoch_type, epoch_type))
                if row_index == 0
                else ""
            )
            plot_glm_scatter_axis(
                scatter_ax,
                table,
                title=title,
                color=get_epoch_type_color(epoch_type),
                x_limits=x_limits,
                y_limit=float(y_limit),
                show_yticklabels=column_index == 0,
                show_xticklabels=row_index == n_rows - 1,
            )
            child_axes.append(scatter_ax)

    ax.text(
        0.5 * (PANEL_GRID_LEFT + PANEL_GRID_RIGHT),
        0.015,
        "Ripple deviance explained",
        ha="center",
        va="bottom",
        fontsize=6.0,
        transform=ax.transAxes,
    )
    ax.text(
        PANEL_GRID_LEFT - 0.065,
        0.5 * (PANEL_GRID_BOTTOM + PANEL_GRID_TOP),
        "-log10 shuffle p",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        transform=ax.transAxes,
    )
    return child_axes


def plot_dark_firing_rate_devexp_grid(
    ax: "Axes",
    payload: dict[str, Any],
    *,
    epoch_types: Sequence[str] = DEFAULT_PER_ANIMAL_EPOCH_TYPES,
) -> list[Any]:
    """Plot dark-epoch movement firing rate versus ripple deviance explained."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    table = payload.get("devexp_table")
    animal_date_keys = iter_dark_activity_dataset_keys(payload)
    if table is None or len(table) == 0 or not animal_date_keys:
        ax.text(
            0.5,
            0.5,
            "No dark movement firing-rate data",
            ha="center",
            va="center",
            fontsize=6.0,
            transform=ax.transAxes,
        )
        return []

    x_limits, y_limits = get_dark_activity_scatter_limits(payload)
    n_rows = len(animal_date_keys)
    n_columns = len(epoch_types)
    grid_height = PANEL_GRID_TOP - PANEL_GRID_BOTTOM
    row_gap = min(DATASET_ROW_GAP, grid_height * 0.18 / max(n_rows - 1, 1))
    row_height = (grid_height - row_gap * (n_rows - 1)) / n_rows
    grid_width = PANEL_GRID_RIGHT - PANEL_GRID_LEFT
    column_width = (grid_width - EPOCH_COLUMN_GAP * (n_columns - 1)) / n_columns

    ax.text(
        PANEL_GRID_LEFT,
        0.98,
        "Dark movement firing rate versus deviance explained",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        transform=ax.transAxes,
    )

    child_axes = []
    for row_index, (animal_name, date) in enumerate(animal_date_keys):
        row_bottom = (
            PANEL_GRID_BOTTOM
            + grid_height
            - (row_index + 1) * row_height
            - row_index * row_gap
        )
        ax.text(
            0.01,
            row_bottom + 0.5 * row_height,
            f"{animal_name}\n{date}",
            ha="left",
            va="center",
            fontsize=5.0,
            transform=ax.transAxes,
        )
        for column_index, epoch_type in enumerate(epoch_types):
            column_left = PANEL_GRID_LEFT + column_index * (
                column_width + EPOCH_COLUMN_GAP
            )
            scatter_ax = ax.inset_axes(
                [column_left, row_bottom, column_width, row_height]
            )
            rows = table[
                (table["animal_name"].astype(str) == animal_name)
                & (table["date"].astype(str) == date)
                & (table["epoch_type"].astype(str) == str(epoch_type))
            ]
            x_values = np.asarray(rows["ripple_devexp_mean"], dtype=float)
            y_values = np.asarray(rows["dark_firing_rate_hz"], dtype=float)
            p_values = np.asarray(rows["ripple_devexp_p_value"], dtype=float)
            valid = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & (y_values > 0.0)
                & np.isfinite(p_values)
            )
            scatter_ax.axvline(0.0, color="0.45", linewidth=0.45, zorder=1)
            if np.any(valid):
                significant = p_values[valid] < SUPPLEMENTARY_FIGURE_2_SIGNIFICANCE_P_VALUE
                finite_x = x_values[valid]
                finite_y = y_values[valid]
                if np.any(~significant):
                    scatter_ax.scatter(
                        finite_x[~significant],
                        finite_y[~significant],
                        s=3.2,
                        color=NONSIGNIFICANT_COLOR,
                        alpha=0.42,
                        edgecolors="none",
                        zorder=2,
                    )
                if np.any(significant):
                    scatter_ax.scatter(
                        finite_x[significant],
                        finite_y[significant],
                        s=3.8,
                        color=get_epoch_type_color(str(epoch_type)),
                        alpha=0.52,
                        edgecolors="none",
                        zorder=3,
                    )
                scatter_ax.text(
                    0.96,
                    0.05,
                    f"n={int(np.sum(valid))}\nsig={np.mean(significant):.2f}",
                    ha="right",
                    va="bottom",
                    fontsize=3.9,
                    transform=scatter_ax.transAxes,
                )
            else:
                scatter_ax.text(
                    0.5,
                    0.5,
                    "No finite\nvalues",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    transform=scatter_ax.transAxes,
                )

            if row_index == 0:
                scatter_ax.set_title(
                    PER_ANIMAL_EPOCH_LABELS.get(
                        str(epoch_type),
                        EPOCH_TYPE_LABELS.get(str(epoch_type), str(epoch_type)),
                    ),
                    fontsize=4.8,
                    pad=1.0,
                )
            scatter_ax.set_xlim(*x_limits)
            scatter_ax.set_yscale("log")
            scatter_ax.set_ylim(*y_limits)
            if not (row_index == n_rows - 1):
                scatter_ax.set_xticklabels([])
            if column_index != 0:
                scatter_ax.set_yticklabels([])
            scatter_ax.spines["top"].set_visible(False)
            scatter_ax.spines["right"].set_visible(False)
            scatter_ax.tick_params(labelsize=4.2, length=1.3, pad=1)
            child_axes.append(scatter_ax)

    ax.text(
        0.5 * (PANEL_GRID_LEFT + PANEL_GRID_RIGHT),
        0.015,
        "Ripple deviance explained",
        ha="center",
        va="bottom",
        fontsize=6.0,
        transform=ax.transAxes,
    )
    ax.text(
        PANEL_GRID_LEFT - 0.065,
        0.5 * (PANEL_GRID_BOTTOM + PANEL_GRID_TOP),
        "Dark movement firing rate (Hz)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        transform=ax.transAxes,
    )
    return child_axes


def make_supplementary_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    ripple_selection_modes: Sequence[str],
    ripple_window_s: float,
    ridge_strength: float,
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 2."""
    import matplotlib.pyplot as plt

    ripple_selection = resolve_primary_ripple_selection(ripple_selection_modes)
    payload = load_per_animal_glm_scatter_payload(
        data_root,
        datasets,
        ripple_selection=ripple_selection,
        ripple_window_s=ripple_window_s,
        ridge_strength=ridge_strength,
    )
    dark_activity_payload = load_glm_dark_activity_devexp_tables(
        data_root,
        datasets,
        ripple_selection=ripple_selection,
        ripple_window_s=ripple_window_s,
        ridge_strength=ridge_strength,
        epoch_types=DEFAULT_PER_ANIMAL_EPOCH_TYPES,
    )
    y_limit = get_per_animal_neglog_p_limit(payload)

    apply_paper_style()
    n_animal_rows = max(
        len(iter_per_animal_glm_keys(payload)),
        len(iter_dark_activity_dataset_keys(dark_activity_payload)),
        len(datasets),
        1,
    )
    section_height_mm = (
        DEFAULT_SECTION_HEADER_HEIGHT_MM
        + DEFAULT_PER_ANIMAL_ROW_HEIGHT_MM * n_animal_rows
    )
    figure_height_mm = 2.0 * section_height_mm + DEFAULT_SECTION_GAP_MM
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, figure_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            section_height_mm,
            DEFAULT_SECTION_GAP_MM,
            section_height_mm,
        ],
    )
    section_ax = fig.add_subplot(outer_grid[0])
    plot_per_animal_glm_scatter_grid(
        section_ax,
        payload,
        y_limit=y_limit,
    )
    label_axis(section_ax, "A", x=-0.01, y=1.01)
    spacer_ax = fig.add_subplot(outer_grid[1])
    spacer_ax.axis("off")
    dark_activity_ax = fig.add_subplot(outer_grid[2])
    plot_dark_firing_rate_devexp_grid(
        dark_activity_ax,
        dark_activity_payload,
    )
    label_axis(dark_activity_ax, "B", x=-0.01, y=1.01)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    for missing in dark_activity_payload["missing_artifacts"]:
        print(
            "Supplementary Figure 2 missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']}: {missing.get('path', '')}"
        )
    print(f"Saved Supplementary Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 2 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 2 ripple-GLM scatter panels."
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
        "--ripple-selection",
        nargs="+",
        choices=RIPPLE_SELECTION_MODE_CHOICES,
        default=list(DEFAULT_RIPPLE_SELECTION_MODES),
        help=(
            "Ripple-selection mode to plot. If multiple values are passed, "
            "only the first is used for the per-animal scatter grid. "
            f"Default: {list(DEFAULT_RIPPLE_SELECTION_MODES)!r}"
        ),
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_S,
        help=f"Ripple-GLM window length in seconds. Default: {DEFAULT_RIPPLE_WINDOW_S}",
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Ridge strength to plot. Default: {DEFAULT_RIDGE_STRENGTH:g}",
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
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        ripple_selection_modes=tuple(args.ripple_selection),
        ripple_window_s=args.ripple_window_s,
        ridge_strength=args.ridge_strength,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
