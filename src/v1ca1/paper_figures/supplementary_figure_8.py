from __future__ import annotations

"""Generate Figure 4D--F summaries separately for each animal."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures import _ripple_panels as _figure_3
from v1ca1.paper_figures import figure_4
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_DIR = figure_4.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "supplementary_figure_8"
DEFAULT_OUTPUT_FORMAT = figure_4.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = figure_4.FIGURE_FORMATS
DEFAULT_FIGURE_WIDTH_MM = figure_4.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_ANIMAL_ROW_HEIGHT_MM = 35.0
ANIMAL_ROW_LABEL_FONTSIZE = 6.0
PANEL_LABELS = ("A", "B", "C")
PANEL_WIDTH_RATIOS = (1.0, 1.8, 1.15)
PANEL_GRID_WSPACE = 0.16
PANEL_GRID_HSPACE = 0.12
PANEL_GRID_LEFT = 0.09
PANEL_GRID_RIGHT = 0.985
PANEL_GRID_TOP = 0.945
PANEL_GRID_BOTTOM = 0.03
PANEL_D_SCATTER_AXIS_BOUNDS = (0.20, 0.48, 0.76, 0.43)
PANEL_D_BOX_AXIS_BOUNDS = (0.20, 0.10, 0.76, 0.25)
PANEL_E_SINGLE_EPOCH_COLUMN_BOUNDS = (
    (0.05, 0.39),
    (0.57, 0.38),
    (0.0, 0.0),
)
PANEL_E_SINGLE_EPOCH_AXIS_VERTICAL_BOUNDS = (0.15, 0.73)
PANEL_F_AXIS_BOUNDS = (0.15, 0.15, 0.81, 0.73)


def group_datasets_by_animal(
    datasets: Sequence[DatasetId],
) -> dict[str, list[DatasetId]]:
    """Return normalized data sets grouped by animal in input order."""
    grouped: dict[str, list[DatasetId]] = {}
    for dataset in datasets:
        normalized = normalize_dataset_id(dataset)
        grouped.setdefault(normalized[0], []).append(normalized)
    return grouped


def get_figure_height_mm(n_animal_rows: int) -> float:
    """Return the figure height for the requested number of animal rows."""
    return DEFAULT_ANIMAL_ROW_HEIGHT_MM * max(int(n_animal_rows), 1)


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
    """Filter Figure 4D epoch payloads to one animal."""
    filtered_payloads = []
    for payload in epoch_tables:
        filtered_datasets = tuple(
            dataset
            for dataset in payload.get("datasets", ())
            if normalize_dataset_id(dataset)[0] == str(animal_name)
        )
        filtered_payloads.append(
            {
                **payload,
                "datasets": filtered_datasets,
                "n_datasets": len(filtered_datasets),
                "summary_table": filter_table_by_animal(
                    payload["summary_table"],
                    animal_name,
                ),
            }
        )
    return filtered_payloads


def filter_behavior_payload_by_animal(
    payload: Mapping[str, Any],
    animal_name: str,
) -> dict[str, Any]:
    """Filter Figure 4E/F data and reference populations to one animal."""
    filtered = dict(payload)
    for key in (
        "devexp_table",
        "dark_activity_reference_table",
        "dark_active_dppi_reference_table",
    ):
        filtered[key] = filter_table_by_animal(payload.get(key), animal_name)
    filtered["missing_artifacts"] = [
        artifact
        for artifact in payload.get("missing_artifacts", ())
        if str(artifact.get("animal_name")) == str(animal_name)
    ]
    return filtered


def load_supplementary_figure_8_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ripple_selection: str,
    ridge_strength: float,
    dark_movement_fr_cache_dir: Path | None,
    refresh_dark_movement_fr_cache: bool,
    tuning_similarity_metric: str = (
        _figure_3.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
    ),
) -> dict[str, Any]:
    """Load the quantitative tables used by Figure 4D--F."""
    dataset_ids = tuple(normalize_dataset_id(dataset) for dataset in datasets)
    panel_d_epoch_tables = _figure_3.load_glm_epoch_summary_tables(
        data_root,
        dataset_ids,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        epoch_types=_figure_3.PANEL_C_EPOCH_ORDER,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )
    panel_ef_behavior_payload = _figure_3.load_glm_dark_activity_devexp_tables(
        data_root,
        dataset_ids,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=refresh_dark_movement_fr_cache,
        tuning_similarity_metric=tuning_similarity_metric,
    )
    return {
        "panel_d_epoch_tables": panel_d_epoch_tables,
        "panel_ef_behavior_payload": panel_ef_behavior_payload,
    }


def _set_child_axis_bounds(
    parent_ax: Any,
    child_ax: Any,
    bounds: tuple[float, float, float, float],
) -> None:
    """Place a child axis at parent-relative bounds."""
    parent_box = parent_ax.get_position()
    left, bottom, width, height = bounds
    child_ax.set_axes_locator(None)
    child_ax.set_position(
        (
            parent_box.x0 + left * parent_box.width,
            parent_box.y0 + bottom * parent_box.height,
            width * parent_box.width,
            height * parent_box.height,
        )
    )


def plot_figure_4_panel_d(
    ax: Any,
    epoch_tables: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any]:
    """Plot only Figure 4D's GLM performance summary column."""
    if len(epoch_tables) != 1:
        raise ValueError(
            "Supplementary Figure 8 expects one Figure 4D epoch payload."
        )
    _figure_3.plot_glm_analysis_panel(
        ax,
        epoch_tables,
        ripple_trace=None,
        prediction_examples=(),
    )
    child_axes = tuple(ax.child_axes)
    if len(child_axes) != 3:
        raise RuntimeError(
            "Expected Figure 4D plotting to create one schematic and two "
            "summary axes."
        )
    child_axes[0].remove()
    scatter_ax, box_ax = child_axes[1:]
    _set_child_axis_bounds(ax, scatter_ax, PANEL_D_SCATTER_AXIS_BOUNDS)
    _set_child_axis_bounds(ax, box_ax, PANEL_D_BOX_AXIS_BOUNDS)
    box_labels = [label.get_text() for label in box_ax.get_yticklabels()]
    if "n.s." in box_labels:
        box_ax.set_yticks(
            box_ax.get_yticks(),
            labels=["" if label == "n.s." else label for label in box_labels],
            fontsize=6,
        )
    return scatter_ax, box_ax


def plot_figure_4_panel_e(
    ax: Any,
    payload: Mapping[str, Any],
    *,
    n_permutations: int = _figure_3.PANEL_E_ACTIVITY_NULL_PERMUTATIONS,
    random_seed: int = _figure_3.PANEL_E_ACTIVITY_NULL_RANDOM_SEED,
    devexp_batch_size: int = _figure_3.PANEL_E_DEVEXP_PERMUTATION_BATCH_SIZE,
) -> tuple[Any, Any]:
    """Plot Figure 4E's dark-activity summaries for one animal."""
    statistics_by_epoch = _figure_3.compute_glm_dark_activity_statistics(
        payload,
        epoch_types=_figure_3.PANEL_D_EPOCH_ORDER,
        n_permutations=n_permutations,
        random_seed=random_seed,
        devexp_batch_size=devexp_batch_size,
    )
    _figure_3.plot_glm_behavior_association_panel(
        ax,
        payload,
        show_note=False,
        show_significance_marker=False,
        include_similarity=False,
        single_epoch_column_bounds=PANEL_E_SINGLE_EPOCH_COLUMN_BOUNDS,
        single_epoch_axis_vertical_bounds=(
            PANEL_E_SINGLE_EPOCH_AXIS_VERTICAL_BOUNDS
        ),
        single_line_axis_labels=True,
        activity_statistics_by_epoch=statistics_by_epoch,
    )
    if len(ax.child_axes) != 2:
        raise RuntimeError("Expected Figure 4E to create two summary axes.")
    return tuple(ax.child_axes)


def plot_figure_4_panel_f(
    ax: Any,
    payload: Mapping[str, Any],
    *,
    n_permutations: int = _figure_3.PANEL_F_DPPI_NULL_PERMUTATIONS,
    random_seed: int = _figure_3.PANEL_F_DPPI_NULL_RANDOM_SEED,
) -> Any:
    """Plot Figure 4F's dark path-invariance distribution for one animal."""
    _figure_3.plot_dark_active_dppi_distribution_panel(
        ax,
        payload,
        n_permutations=n_permutations,
        random_seed=random_seed,
        axis_bounds=PANEL_F_AXIS_BOUNDS,
        show_significance_marker=False,
    )
    if len(ax.child_axes) != 1:
        raise RuntimeError("Expected Figure 4F to create one histogram axis.")
    histogram_ax = ax.child_axes[0]
    if histogram_ax.axison:
        histogram_ax.set_xlabel(figure_4.PANEL_F_XLABEL, fontsize=6, labelpad=1.0)
    return histogram_ax


def make_supplementary_figure_8(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ripple_selection: str,
    ridge_strength: float,
    dark_movement_fr_cache_dir: Path | None,
    refresh_dark_movement_fr_cache: bool,
    dpi: int,
    dppi_n_permutations: int = _figure_3.PANEL_F_DPPI_NULL_PERMUTATIONS,
    dppi_random_seed: int = _figure_3.PANEL_F_DPPI_NULL_RANDOM_SEED,
    activity_n_permutations: int = (
        _figure_3.PANEL_E_ACTIVITY_NULL_PERMUTATIONS
    ),
    activity_random_seed: int = _figure_3.PANEL_E_ACTIVITY_NULL_RANDOM_SEED,
    activity_devexp_batch_size: int = (
        _figure_3.PANEL_E_DEVEXP_PERMUTATION_BATCH_SIZE
    ),
    tuning_similarity_metric: str = (
        _figure_3.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
    ),
) -> Path:
    """Build and save Figure 4D--F panels with one row per animal."""
    import matplotlib.pyplot as plt

    animal_groups = group_datasets_by_animal(datasets)
    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            get_figure_height_mm(len(animal_groups)),
        ),
        constrained_layout=False,
    )
    if not animal_groups:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        print(f"Saved Supplementary Figure 8 to {output_path}")
        return output_path

    normalized_datasets = [
        dataset
        for animal_datasets in animal_groups.values()
        for dataset in animal_datasets
    ]
    panel_data = load_supplementary_figure_8_data(
        data_root=data_root,
        datasets=normalized_datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=refresh_dark_movement_fr_cache,
        tuning_similarity_metric=tuning_similarity_metric,
    )
    grid = fig.add_gridspec(
        nrows=len(animal_groups),
        ncols=3,
        width_ratios=PANEL_WIDTH_RATIOS,
        wspace=PANEL_GRID_WSPACE,
        hspace=PANEL_GRID_HSPACE,
        left=PANEL_GRID_LEFT,
        right=PANEL_GRID_RIGHT,
        top=PANEL_GRID_TOP,
        bottom=PANEL_GRID_BOTTOM,
    )

    for row_index, animal_name in enumerate(animal_groups):
        row_axes = [
            fig.add_subplot(grid[row_index, column_index])
            for column_index in range(3)
        ]
        panel_d_epoch_tables = filter_epoch_tables_by_animal(
            panel_data["panel_d_epoch_tables"],
            animal_name,
        )
        behavior_payload = filter_behavior_payload_by_animal(
            panel_data["panel_ef_behavior_payload"],
            animal_name,
        )
        plot_figure_4_panel_d(row_axes[0], panel_d_epoch_tables)
        plot_figure_4_panel_e(
            row_axes[1],
            behavior_payload,
            n_permutations=activity_n_permutations,
            random_seed=activity_random_seed,
            devexp_batch_size=activity_devexp_batch_size,
        )
        plot_figure_4_panel_f(
            row_axes[2],
            behavior_payload,
            n_permutations=dppi_n_permutations,
            random_seed=dppi_random_seed,
        )
        row_axes[0].text(
            -0.12,
            0.50,
            animal_name,
            ha="right",
            va="center",
            fontsize=ANIMAL_ROW_LABEL_FONTSIZE,
            color="0.25",
            transform=row_axes[0].transAxes,
        )
        if row_index == 0:
            for axis, panel_label in zip(
                row_axes,
                PANEL_LABELS,
                strict=True,
            ):
                label_axis(axis, panel_label, x=-0.03, y=1.02)

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    for missing in panel_data["panel_ef_behavior_payload"]["missing_artifacts"]:
        print(
            "Supplementary Figure 8 missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']}: {missing['path']}"
        )
    print(f"Saved Supplementary Figure 8 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 8."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 4D--F summaries separately by animal."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Base directory containing analysis outputs. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
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
        type=_figure_3.parse_dataset_id,
        help=(
            "Animal/date/dark-epoch data set as animal:date[:epoch]. May be "
            "repeated. Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument("--sleep-epoch", default=None, help="Sleep epoch.")
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=_figure_3.DEFAULT_RIPPLE_WINDOW_S,
        help=(
            "Ripple-GLM window length in seconds. "
            f"Default: {_figure_3.DEFAULT_RIPPLE_WINDOW_S}"
        ),
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=_figure_3.DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        help=(
            "Ripple-GLM window offset in seconds. "
            f"Default: {_figure_3.DEFAULT_RIPPLE_WINDOW_OFFSET_S}"
        ),
    )
    parser.add_argument(
        "--ripple-selection",
        choices=("allripples", "deduped", "single"),
        default=_figure_3.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
        help=(
            "Ripple-GLM selection suffix. "
            f"Default: {_figure_3.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION}"
        ),
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=_figure_3.DEFAULT_RIDGE_STRENGTH,
        help=(
            "Ripple-GLM ridge strength. "
            f"Default: {_figure_3.DEFAULT_RIDGE_STRENGTH:g}"
        ),
    )
    parser.add_argument(
        "--dark-movement-fr-cache-dir",
        type=Path,
        default=_figure_3.DEFAULT_FIGURE_CACHE_DIR,
        help=(
            "Directory for cached dark movement firing-rate tables. "
            f"Default: {_figure_3.DEFAULT_FIGURE_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--refresh-dark-movement-fr-cache",
        action="store_true",
        help="Recompute and overwrite cached dark movement firing-rate tables.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Supplementary Figure 8 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = figure_4.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_8(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        sleep_epoch=args.sleep_epoch,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection=args.ripple_selection,
        ridge_strength=args.ridge_strength,
        dark_movement_fr_cache_dir=args.dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
