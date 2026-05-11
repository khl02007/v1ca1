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
    make_figure_2_epoch_ids,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_S,
    FIGURE_FORMATS,
    MODEL_COLOR,
    NONSIGNIFICANT_COLOR,
    PANEL_E_GLM_SOURCE_WINDOW_OFFSET_S,
    PANEL_E_GLM_TARGET_WINDOW_OFFSETS_S,
    SIGNIFICANCE_P_VALUE,
    get_ripple_glm_model_window_path,
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
DEFAULT_FIGURE_WIDTH_MM = 180.0
DEFAULT_SECTION_HEADER_HEIGHT_MM = 8.0
DEFAULT_DATASET_ROW_HEIGHT_MM = 19.0
DEFAULT_SECTION_GAP_MM = 5.0
DEFAULT_RIPPLE_SELECTION_MODES = ("allripples", "single")
SELECTION_LABELS = {
    "allripples": "All ripples",
    "single": "Single-ripple windows",
}
SELECTION_COLORS = {
    "allripples": EPOCH_TYPE_COLORS["sleep"],
    "single": EPOCH_TYPE_COLORS["light"],
}
PLOT_X_LIMITS = (-0.1, 0.5)
PANEL_GRID_LEFT = 0.13
PANEL_GRID_RIGHT = 0.99
PANEL_GRID_BOTTOM = 0.11
PANEL_GRID_TOP = 0.88
DATASET_ROW_GAP = 0.045
EPOCH_COLUMN_GAP = 0.018


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


def format_dataset_label(dataset: DatasetId) -> str:
    """Return a compact row label for one data set."""
    animal_name, date, _epoch = normalize_dataset_id(dataset)
    return f"{animal_name}\n{date}"


def get_dataset_light_epoch(dataset: DatasetId) -> str:
    """Return the registered light epoch for one data set."""
    animal_name, date, dark_epoch = normalize_dataset_id(dataset)
    epoch_ids = make_figure_2_epoch_ids(
        animal_name,
        date,
        dark_epoch=dark_epoch,
    )
    return normalize_dataset_id(epoch_ids["light"])[2]


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
    """Load available 02_r1 target-offset ripple-GLM artifacts by data set."""
    payload_rows: dict[str, list[dict[str, Any]]] = {}
    for ripple_selection in ripple_selection_modes:
        selection_rows = []
        for dataset in datasets:
            animal_name, date, dark_epoch = normalize_dataset_id(dataset)
            epoch = get_dataset_light_epoch(dataset)
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
                        "epoch": epoch,
                        "ripple_selection": ripple_selection,
                        "target_window_offset_s": artifact["target_window_offset_s"],
                        "target_window_label": artifact["target_window_label"],
                        "summary_table": load_glm_scatter_table(path),
                        "source_path": str(path),
                    }
                )
            selection_rows.append(
                {
                    "dataset": (animal_name, date, dark_epoch),
                    "animal_name": animal_name,
                    "date": date,
                    "ripple_selection": ripple_selection,
                    "artifacts": artifacts,
                }
            )
        payload_rows[ripple_selection] = selection_rows
    return {
        "rows_by_selection": payload_rows,
        "ripple_selection_modes": tuple(ripple_selection_modes),
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
        return 2.0
    return max(2.0, float(np.nanmax(np.concatenate(neglog_values))) + 0.35)


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
        -np.log10(SIGNIFICANCE_P_VALUE),
        color="0.25",
        linestyle="--",
        linewidth=0.55,
        zorder=1,
    )
    if np.any(valid):
        finite_values = values[valid]
        finite_p_values = p_values[valid]
        neglog_p = -np.log10(np.clip(finite_p_values, 1e-12, 1.0))
        significant = finite_p_values < SIGNIFICANCE_P_VALUE
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
    color: str,
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
    row_height = (grid_height - DATASET_ROW_GAP * (n_rows - 1)) / n_rows
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
            - row_index * DATASET_ROW_GAP
        )
        ax.text(
            0.01,
            row_bottom + 0.5 * row_height,
            format_dataset_label(dataset_row["dataset"]),
            ha="left",
            va="center",
            fontsize=5.0,
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
                color=color,
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

    payload = load_available_glm_scatter_payload(
        data_root,
        datasets,
        ripple_selection_modes=ripple_selection_modes,
        ripple_window_s=ripple_window_s,
        ridge_strength=ridge_strength,
    )
    y_limit = get_payload_neglog_p_limit(payload)

    apply_paper_style()
    n_dataset_rows = max(len(datasets), 1)
    section_height_mm = (
        DEFAULT_SECTION_HEADER_HEIGHT_MM
        + DEFAULT_DATASET_ROW_HEIGHT_MM * n_dataset_rows
    )
    figure_height_mm = (
        section_height_mm * len(ripple_selection_modes)
        + DEFAULT_SECTION_GAP_MM * max(len(ripple_selection_modes) - 1, 0)
    )
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, figure_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=len(ripple_selection_modes) * 2 - 1,
        ncols=1,
        height_ratios=[
            section_height_mm if index % 2 == 0 else DEFAULT_SECTION_GAP_MM
            for index in range(len(ripple_selection_modes) * 2 - 1)
        ],
    )
    panel_labels = ("A", "B", "C", "D")
    panel_index = 0
    for selection_index, ripple_selection in enumerate(ripple_selection_modes):
        axis_index = selection_index * 2
        section_ax = fig.add_subplot(outer_grid[axis_index])
        plot_selection_scatter_grid(
            section_ax,
            payload["rows_by_selection"].get(ripple_selection, []),
            selection_label=SELECTION_LABELS.get(
                ripple_selection,
                str(ripple_selection),
            ),
            color=SELECTION_COLORS.get(ripple_selection, MODEL_COLOR),
            x_limits=PLOT_X_LIMITS,
            y_limit=y_limit,
        )
        if panel_index < len(panel_labels):
            label_axis(section_ax, panel_labels[panel_index], x=-0.01, y=1.01)
        panel_index += 1
        if selection_index < len(ripple_selection_modes) - 1:
            spacer_ax = fig.add_subplot(outer_grid[axis_index + 1])
            spacer_ax.axis("off")

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
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
        choices=DEFAULT_RIPPLE_SELECTION_MODES,
        default=list(DEFAULT_RIPPLE_SELECTION_MODES),
        help=(
            "Ripple-selection modes to plot, in order. "
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
