from __future__ import annotations

"""Generate Figure 3 panels for light-epoch place fields."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    get_analysis_path,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_dataset_dark_epoch,
    get_dataset_light_epoch,
    get_processed_datasets,
    make_dataset_id,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_HEATMAP_HEIGHT_MM,
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_PANEL_E_WIDTH_FRACTION,
    add_centered_axis_text,
    build_normalized_position_bins,
    build_pooled_panel_values,
    compute_dark_epoch_tuning_curves,
    draw_neuron_scale_bar,
    draw_order_schematic,
    extract_unit_rate_curve,
    get_unit_spike_times,
    load_panel_e_example_data,
    normalize_linear_position_by_trajectory,
    orient_panel_e_task_progression,
    plot_pooled_heatmap_grid,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.helper.plot_wtrack_schematic import get_w_track_geometry
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic
from v1ca1.raster.plot_place_field_heatmap import (
    DEFAULT_SIGMA_BINS,
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    prepare_heatmap_session,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "figure_3"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_REGIONS = ("v1",)
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_PANEL_A_HEIGHT_MM = 44.8
DEFAULT_PANEL_BC_HEIGHT_MM = DEFAULT_HEATMAP_HEIGHT_MM
DEFAULT_PANEL_DEF_HEIGHT_MM = 30.0
DEFAULT_FIGURE_HEIGHT_MM = (
    DEFAULT_PANEL_A_HEIGHT_MM
    + DEFAULT_PANEL_BC_HEIGHT_MM
    + DEFAULT_PANEL_DEF_HEIGHT_MM
)
DEFAULT_PANEL_B_WIDTH_FRACTION = DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION
DEFAULT_PANEL_C_WIDTH_FRACTION = DEFAULT_PANEL_E_WIDTH_FRACTION
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
TUNING_ANALYSIS_RELATIVE_DIR = Path("task_progression") / "tuning_analysis"
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
DECODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "decoding_comparison"
PANEL_A_EXAMPLE = ("L14", "20240611", "v1", 229)
PANEL_A_TRAJECTORIES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
PANEL_A_LIGHT_EPOCHS = ("02_r1", "06_r3")
PANEL_A_EPOCH_LABELS = {
    "02_r1": "02_r1",
    "06_r3": "06_r3",
    "dark": "Dark",
}
PANEL_A_EPOCH_COLORS = {
    "02_r1": "#4C72B0",
    "06_r3": "#55A868",
    "dark": "#222222",
}
SEGMENT_BOUNDARIES = (0.4, 0.6)
SEGMENT_BOUNDARY_COLOR = "0.65"
SEGMENT_BOUNDARY_LINEWIDTH = 0.45
PANEL_C_EXAMPLES = (
    ("L14", "20240611", "v1", 34, ("center_to_left", "right_to_center")),
    ("L15", "20241121", "v1", 473, ("center_to_right", "left_to_center")),
)
PANEL_TRAJECTORY_LABELS = {
    "center_to_left": "C to L",
    "right_to_center": "R to C",
    "center_to_right": "C to R",
    "left_to_center": "L to C",
}
PANEL_C_TRAJECTORY_COLORS = {
    "center_to_left": "#4C72B0",
    "center_to_right": "#C44E52",
    "right_to_center": "#55A868",
    "left_to_center": "#DD8452",
}
PANEL_C_EPOCH_LABELS = {
    "dark": "Dark",
    "light": "Light",
}
PANEL_QUANT_EPOCH_ORDER = ("light", "dark")
PANEL_QUANT_EPOCH_LABELS = {
    "light": "Light",
    "dark": "Dark",
}
PANEL_QUANT_EPOCH_COLORS = {
    "light": PANEL_A_EPOCH_COLORS["02_r1"],
    "dark": PANEL_A_EPOCH_COLORS["dark"],
}
PANEL_D_COMPARISON_LABELS = ("left_turn", "right_turn")
PANEL_D_COMPARISON_COLORS = {
    "left_turn": PANEL_C_TRAJECTORY_COLORS["center_to_left"],
    "right_turn": PANEL_C_TRAJECTORY_COLORS["center_to_right"],
}
PANEL_D_COMPARISON_DISPLAY_LABELS = {
    "left_turn": "Left turn",
    "right_turn": "Right turn",
}
PANEL_E_ENCODING_N_FOLDS = 5
PANEL_E_PLACE_BIN_SIZE_CM = DEFAULT_PLACE_BIN_SIZE_CM
PANEL_E_DELTA_COLUMN = "delta_bits_place_vs_tp"
PANEL_F_DECODING_MODELS = ("task_progression", "place")
PANEL_F_DECODING_METRIC = "median_abs_error"


def get_dark_epoch(animal_name: str, date: str, dark_epoch: str | None = None) -> str:
    """Return the dark run epoch label for one session."""
    del date
    if dark_epoch is not None:
        return str(dark_epoch)
    return get_dataset_dark_epoch(animal_name)


def get_light_epoch(animal_name: str, date: str, light_epoch: str | None = None) -> str:
    """Return the light run epoch label for one session."""
    del date
    if light_epoch is not None:
        return str(light_epoch)
    return get_dataset_light_epoch(animal_name)


def parse_dataset_id(value: str) -> DatasetId:
    """Parse one `animal:date[:dark_epoch]` data-set identifier."""
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(
            "Data sets must be specified as animal:date or animal:date:dark_epoch, "
            "for example L14:20240611 or L15:20241121:10_r5."
        )
    return make_dataset_id(*parts)


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


def format_place_bin_size_token(place_bin_size_cm: float) -> str:
    """Return the filename token used by task-progression encoding summaries."""
    value_text = f"{float(place_bin_size_cm):g}".replace("-", "m").replace(".", "p")
    return f"placebin{value_text}cm"


def get_tuning_similarity_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    similarity_metric: str = "correlation",
) -> Path:
    """Return one tuning-analysis within-epoch similarity artifact path."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / TUNING_ANALYSIS_RELATIVE_DIR
        / f"{region}_{epoch}_{similarity_metric}_within_epoch_similarity.parquet"
    )


def get_encoding_summary_candidate_paths(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    n_folds: int = PANEL_E_ENCODING_N_FOLDS,
    place_bin_size_cm: float = PANEL_E_PLACE_BIN_SIZE_CM,
) -> tuple[Path, ...]:
    """Return current and legacy encoding-summary artifact paths."""
    data_dir = (
        get_dataset_analysis_path(data_root, animal_name, date)
        / ENCODING_COMPARISON_RELATIVE_DIR
    )
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    return (
        data_dir / f"{region}_{epoch}_cv{n_folds}_{place_bin_token}_encoding_summary.parquet",
        data_dir / f"{region}_{epoch}_cv{n_folds}_encoding_summary.parquet",
    )


def get_decoding_summary_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> Path:
    """Return one decoding-comparison summary artifact path."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / DECODING_COMPARISON_RELATIVE_DIR
        / f"{region}_{epoch}_decoding_summary.parquet"
    )


def read_parquet_table(path: Path) -> Any:
    """Load one parquet table with a focused missing-file message."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Parquet table not found: {path}")
    import pandas as pd

    return pd.read_parquet(path)


def _resolve_existing_path(paths: Sequence[Path]) -> Path | None:
    """Return the first existing path from a candidate list."""
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def make_light_epoch_dataset_ids(
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
) -> list[DatasetId]:
    """Return data-set IDs with registered light epochs for each animal/date."""
    light_datasets: list[DatasetId] = []
    for dataset in datasets:
        animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        light_datasets.append(
            make_dataset_id(
                animal_name,
                date,
                get_light_epoch(animal_name, date, light_epoch),
            )
        )
    return light_datasets


def compute_light_epoch_tuning_curves(
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
) -> dict[str, Any]:
    """Compute odd/even normalized-position tuning curves for one light epoch."""
    return compute_dark_epoch_tuning_curves(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        region=region,
        epoch=get_light_epoch(animal_name, date, light_epoch),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )


def validate_trajectories(trajectories: Sequence[str], *, panel_name: str) -> tuple[str, ...]:
    """Return validated trajectory names for one figure panel."""
    validated = tuple(str(trajectory) for trajectory in trajectories)
    if not validated:
        raise ValueError(f"Panel {panel_name} examples must include at least one trajectory.")
    unknown = [trajectory for trajectory in validated if trajectory not in TRAJECTORY_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown panel {panel_name} trajectory type(s): {unknown!r}. "
            f"Expected one of {TRAJECTORY_TYPES!r}."
        )
    return validated


def validate_panel_c_trajectories(trajectories: Sequence[str]) -> tuple[str, ...]:
    """Return validated panel-C trajectory names."""
    return validate_trajectories(trajectories, panel_name="C")


def build_panel_a_epoch_specs(
    animal_name: str,
    date: str,
    *,
    dark_epoch: str | None,
) -> tuple[tuple[str, str, str], ...]:
    """Return panel-A epoch keys, labels, and run epoch IDs."""
    light_specs = tuple((epoch, PANEL_A_EPOCH_LABELS[epoch], epoch) for epoch in PANEL_A_LIGHT_EPOCHS)
    return (
        *light_specs,
        ("dark", PANEL_A_EPOCH_LABELS["dark"], get_dark_epoch(animal_name, date, dark_epoch)),
    )


def load_panel_a_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    unit_id: int,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load the panel-A example cell rasters and rate curves across epochs."""
    epoch_specs = build_panel_a_epoch_specs(
        animal_name,
        date,
        dark_epoch=dark_epoch,
    )
    epoch_examples = {
        epoch_key: load_panel_e_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region=region,
            unit_id=unit_id,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        )
        for epoch_key, _epoch_label, epoch in epoch_specs
    }
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "unit_id": unit_id,
        "epoch_order": tuple(epoch_key for epoch_key, _epoch_label, _epoch in epoch_specs),
        "epoch_labels": {
            epoch_key: epoch_label for epoch_key, epoch_label, _epoch in epoch_specs
        },
        "epoch_examples": epoch_examples,
        "trajectories": PANEL_A_TRAJECTORIES,
    }


def _get_panel_a_y_max(example: dict[str, Any]) -> float:
    """Return a shared firing-rate limit for the panel-A example."""
    maxima: list[float] = []
    for epoch_payload in example["epoch_examples"].values():
        for _position, rate in epoch_payload["firing_rates"].values():
            rate = np.asarray(rate, dtype=float)
            if np.isfinite(rate).any():
                maxima.append(float(np.nanmax(rate)))
    if not maxima:
        return 1.0
    return max(1.0, float(np.ceil(max(maxima))))


def add_segment_boundary_lines(ax: "Axes") -> None:
    """Draw normalized task-progression segment boundaries."""
    for boundary in SEGMENT_BOUNDARIES:
        ax.axvline(
            boundary,
            color=SEGMENT_BOUNDARY_COLOR,
            linewidth=SEGMENT_BOUNDARY_LINEWIDTH,
            zorder=1,
        )


def plot_panel_a_rate_axis(
    ax: "Axes",
    example: dict[str, Any],
    trajectory_type: str,
    *,
    y_max: float,
    show_ylabel: bool = False,
    show_legend: bool = False,
) -> None:
    """Plot panel-A firing-rate curves for one trajectory across epochs."""
    for epoch_key in example["epoch_order"]:
        position, rate = example["epoch_examples"][epoch_key]["firing_rates"][
            trajectory_type
        ]
        ax.plot(
            position,
            rate,
            color=PANEL_A_EPOCH_COLORS[epoch_key],
            linewidth=0.85,
            label=example["epoch_labels"][epoch_key],
        )
    add_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    ax.set_xlabel("Norm. task progression", fontsize=4.8, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.8, labelpad=1)
    if show_legend:
        ax.legend(frameon=False, fontsize=4.2, handlelength=1.1, borderpad=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def plot_panel_a_raster_axis(
    ax: "Axes",
    trial_positions: Sequence[np.ndarray],
    *,
    color: str,
) -> None:
    """Plot one panel-A position-aligned spike raster with segment boundaries."""
    for trial_index, positions in enumerate(trial_positions, start=1):
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            continue
        ax.plot(
            positions,
            np.full(positions.shape, trial_index, dtype=float),
            "|",
            color=color,
            markersize=1.2,
            markeredgewidth=0.45,
        )

    add_segment_boundary_lines(ax)

    n_trials = len(trial_positions)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1, n_trials) + 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=1.5, pad=1)


def draw_panel_a_epoch_icon(
    ax: "Axes",
    *,
    left_label: str | None = None,
    right_label: str | None = None,
    fill_track: bool = False,
) -> None:
    """Draw one panel-A epoch-condition W-track icon."""
    from matplotlib.patches import Polygon

    outline, _points, dims = get_w_track_geometry()
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="black" if fill_track else "none",
            edgecolor="black",
            linewidth=0.45,
            joinstyle="miter",
        )
    )
    if left_label is not None:
        ax.text(
            dims["x0"] - 0.58,
            dims["y2"] / 2,
            left_label,
            ha="center",
            va="center",
            fontsize=5.2,
        )
    if right_label is not None:
        ax.text(
            dims["x5"] + 0.58,
            dims["y2"] / 2,
            right_label,
            ha="center",
            va="center",
            fontsize=5.2,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.95, dims["x5"] + 0.95)
    ax.set_ylim(-0.25, dims["y2"] + 0.25)
    ax.axis("off")


def plot_panel_a_example(ax: "Axes", example: dict[str, Any]) -> None:
    """Plot the panel-A example rasters and firing-rate curves."""
    trajectories = validate_trajectories(example["trajectories"], panel_name="A")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        f"{example['animal_name']} {example['region'].upper()} cell {example['unit_id']}",
        ha="left",
        va="top",
        fontsize=5.8,
        transform=ax.transAxes,
    )

    left_margin = 0.13
    right_margin = 0.012
    column_gap = 0.026
    column_width = (
        1.0
        - left_margin
        - right_margin
        - column_gap * (len(trajectories) - 1)
    ) / len(trajectories)
    raster_height = 0.105
    raster_y = (0.66, 0.51, 0.36)
    y_max = _get_panel_a_y_max(example)

    icon_specs = (
        {"left_label": "A", "right_label": "B", "fill_track": False},
        {"left_label": "B", "right_label": "A", "fill_track": False},
        {"left_label": None, "right_label": None, "fill_track": True},
    )
    for row_index, icon_spec in enumerate(icon_specs):
        icon_ax = ax.inset_axes([0.026, raster_y[row_index] - 0.008, 0.070, 0.122])
        draw_panel_a_epoch_icon(icon_ax, **icon_spec)

    for trajectory_index, trajectory_type in enumerate(trajectories):
        left = left_margin + trajectory_index * (column_width + column_gap)
        schematic_ax = ax.inset_axes(
            [left + 0.34 * column_width, 0.80, 0.32 * column_width, 0.12]
        )
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color="red",
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )

        for epoch_index, epoch_key in enumerate(example["epoch_order"]):
            raster_ax = ax.inset_axes(
                [left, raster_y[epoch_index], column_width, raster_height]
            )
            plot_panel_a_raster_axis(
                raster_ax,
                example["epoch_examples"][epoch_key]["raster_positions"][trajectory_type],
                color=PANEL_A_EPOCH_COLORS[epoch_key],
            )
            if trajectory_index == 0:
                raster_ax.set_ylabel("")

        rate_ax = ax.inset_axes([left, 0.07, column_width, 0.20])
        plot_panel_a_rate_axis(
            rate_ax,
            example,
            trajectory_type,
            y_max=y_max,
            show_ylabel=trajectory_index == 0,
            show_legend=trajectory_index == len(trajectories) - 1,
        )


def load_epoch_unit_rate_curves(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    unit_id: int,
    trajectories: Sequence[str],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load one unit's full-epoch tuning curves for selected trajectories."""
    trajectories = validate_panel_c_trajectories(trajectories)
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
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    task_progression_by_trajectory = {
        trajectory_type: orient_panel_e_task_progression(
            normalized_position_by_trajectory[trajectory_type],
            trajectory_type,
        )
        for trajectory_type in trajectories
    }

    spikes = session["spikes_by_region"][region]
    get_unit_spike_times(spikes, unit_id)
    bin_edges = build_normalized_position_bins(position_bin_count)
    fallback_position = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory_type in trajectories:
        movement_epochs = session["trajectory_intervals"][selected_epoch][
            trajectory_type
        ].intersect(session["movement_by_run"][selected_epoch])
        tuning_curve = compute_place_tuning_curve(
            spikes,
            task_progression_by_trajectory[trajectory_type],
            movement_epochs,
            bin_edges=bin_edges,
            sigma_bins=sigma_bins,
        )
        firing_rates[trajectory_type] = extract_unit_rate_curve(
            tuning_curve,
            unit_id,
            fallback_position,
        )

    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": selected_epoch,
        "region": region,
        "unit_id": unit_id,
        "firing_rates": firing_rates,
    }


def load_panel_c_example_data(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    unit_id: int,
    trajectories: Sequence[str],
    dark_epoch: str | None,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load one dark-vs-light example unit for panel C."""
    trajectories = validate_panel_c_trajectories(trajectories)
    dark_epoch_id = get_dark_epoch(animal_name, date, dark_epoch)
    light_epoch_id = get_light_epoch(animal_name, date, light_epoch)
    epoch_rates = {
        "dark": load_epoch_unit_rate_curves(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            epoch=dark_epoch_id,
            region=region,
            unit_id=unit_id,
            trajectories=trajectories,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        ),
        "light": load_epoch_unit_rate_curves(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            epoch=light_epoch_id,
            region=region,
            unit_id=unit_id,
            trajectories=trajectories,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        ),
    }
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "unit_id": unit_id,
        "trajectories": trajectories,
        "epoch_rates": epoch_rates,
    }


def _get_panel_c_y_max(example: dict[str, Any]) -> float:
    """Return a shared y-limit for one dark-light tuning example."""
    maxima: list[float] = []
    for epoch_payload in example["epoch_rates"].values():
        for _position, rate in epoch_payload["firing_rates"].values():
            rate = np.asarray(rate, dtype=float)
            if np.isfinite(rate).any():
                maxima.append(float(np.nanmax(rate)))
    if not maxima:
        return 1.0
    return max(1.0, float(np.ceil(max(maxima))))


def plot_epoch_path_rate_axis(
    ax: "Axes",
    example: dict[str, Any],
    epoch_key: str,
    *,
    y_max: float,
    trajectories: Sequence[str] | None = None,
    show_ylabel: bool = False,
    show_legend: bool = False,
) -> None:
    """Plot selected path-type tuning curves for one epoch."""
    trajectories = (
        validate_panel_c_trajectories(example["trajectories"])
        if trajectories is None
        else validate_panel_c_trajectories(trajectories)
    )
    for trajectory_type in trajectories:
        position, rate = example["epoch_rates"][epoch_key]["firing_rates"][trajectory_type]
        ax.plot(
            position,
            rate,
            color=PANEL_C_TRAJECTORY_COLORS[trajectory_type],
            linestyle="-",
            linewidth=0.9,
            label=PANEL_TRAJECTORY_LABELS[trajectory_type],
        )
    add_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    ax.set_xlabel("Norm. task progression", fontsize=4.8, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.8, labelpad=1)
    ax.set_title(PANEL_C_EPOCH_LABELS[epoch_key], fontsize=5.3, pad=1)
    if show_legend:
        ax.legend(frameon=False, fontsize=4.2, handlelength=1.1, borderpad=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def plot_panel_c_example(
    ax: "Axes",
    example: dict[str, Any],
    *,
    title: str | None = None,
) -> None:
    """Plot one panel-C example cell with dark and light rate curves."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if title is not None:
        ax.text(
            0.02,
            0.995,
            title,
            ha="left",
            va="top",
            fontsize=5.8,
            transform=ax.transAxes,
        )

    trajectories = validate_panel_c_trajectories(example["trajectories"])
    y_max = _get_panel_c_y_max(example)
    n_schematics = len(trajectories)
    schematic_gap = 0.04 if n_schematics > 1 else 0.0
    schematic_width = (0.42 - schematic_gap * (n_schematics - 1)) / n_schematics
    for trajectory_index, trajectory_type in enumerate(trajectories):
        left = 0.29 + trajectory_index * (schematic_width + schematic_gap)
        schematic_ax = ax.inset_axes([left, 0.72, schematic_width, 0.18])
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color="red",
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )
    dark_ax = ax.inset_axes([0.04, 0.12, 0.43, 0.50])
    light_ax = ax.inset_axes([0.53, 0.12, 0.43, 0.50])
    plot_epoch_path_rate_axis(
        dark_ax,
        example,
        "dark",
        trajectories=trajectories,
        y_max=y_max,
        show_ylabel=True,
    )
    plot_epoch_path_rate_axis(
        light_ax,
        example,
        "light",
        trajectories=trajectories,
        y_max=y_max,
        show_legend=True,
    )


def plot_panel_c_examples(
    ax: "Axes",
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot all panel-C examples stacked in one axis."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center", transform=ax.transAxes)
        return

    block_height = 0.47
    y_positions = np.linspace(1.0 - block_height, 0.02, len(examples))
    for example_index, (y0, example) in enumerate(
        zip(y_positions, examples, strict=False),
        start=1,
    ):
        example_ax = ax.inset_axes([0.0, float(y0), 1.0, block_height])
        plot_panel_c_example(
            example_ax,
            example,
            title=f"Example cell {example_index}",
        )


def setup_light_heatmap_panel(
    fig: Any,
    grid_spec: Any,
    *,
    regions: Sequence[str],
) -> dict[str, Any]:
    """Create the panel-A light-epoch heatmap axes."""
    n_region_rows = len(regions) * len(TRAJECTORY_TYPES)
    heatmap_grid = grid_spec.subgridspec(
        nrows=n_region_rows + 1,
        ncols=len(TRAJECTORY_TYPES) + 1,
        height_ratios=[0.42, *([1.0] * n_region_rows)],
        width_ratios=[0.48, *([1.0] * len(TRAJECTORY_TYPES))],
    )
    axes = np.asarray(
        [
            [fig.add_subplot(heatmap_grid[row, col]) for col in range(len(TRAJECTORY_TYPES) + 1)]
            for row in range(n_region_rows + 1)
        ],
        dtype=object,
    )
    corner_axis = axes[0, 0]
    corner_axis.axis("off")
    tuning_schematic_axes = axes[0, 1:]
    order_schematic_axes = axes[1:, 0]
    heatmap_axes = axes[1:, 1:]
    for ax, trajectory_type in zip(tuning_schematic_axes, TRAJECTORY_TYPES, strict=True):
        draw_w_track_schematic(
            ax,
            trajectory_name=trajectory_type,
            arrow_color="red",
            fill_track=False,
        )
    for row_index, ax in enumerate(order_schematic_axes):
        draw_order_schematic(
            ax,
            TRAJECTORY_TYPES[row_index % len(TRAJECTORY_TYPES)],
            arrow_color="red",
            fill_track=False,
        )
    return {
        "corner_axis": corner_axis,
        "tuning_schematic_axes": tuning_schematic_axes,
        "order_schematic_axes": order_schematic_axes,
        "heatmap_axes": heatmap_axes,
    }


def plot_light_heatmap_regions(
    heatmap_axes: np.ndarray,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> "AxesImage | None":
    """Plot pooled light-epoch heatmaps for all requested regions."""
    color_image = None
    for region_index, region in enumerate(regions):
        print(f"Building pooled light-epoch heatmap for region {region}.")
        curve_sets = []
        for dataset in datasets:
            animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
            epoch = get_light_epoch(animal_name, date, light_epoch)
            print(f"  Loading {animal_name} {date} epoch {epoch}.")
            curve_sets.append(
                compute_light_epoch_tuning_curves(
                    animal_name=animal_name,
                    date=date,
                    data_root=data_root,
                    region=region,
                    light_epoch=light_epoch,
                    position_bin_count=position_bin_count,
                    position_offset=position_offset,
                    speed_threshold_cm_s=speed_threshold_cm_s,
                    sigma_bins=sigma_bins,
                )
            )

        panels = build_pooled_panel_values(
            curve_sets,
            position_bin_count=position_bin_count,
        )
        start_row = region_index * len(TRAJECTORY_TYPES)
        stop_row = start_row + len(TRAJECTORY_TYPES)
        image = plot_pooled_heatmap_grid(
            heatmap_axes[start_row:stop_row, :],
            panels,
        )
        for heatmap_ax in heatmap_axes[start_row:stop_row, :].ravel():
            add_segment_boundary_lines(heatmap_ax)
        if color_image is None and image is not None:
            color_image = image
    return color_image


def build_panel_quant_epoch_specs(
    animal_name: str,
    date: str,
    *,
    light_epoch: str | None,
    dark_epoch: str | None,
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return light and dark epoch labels for quantitative artifact panels."""
    return (
        ("light", get_light_epoch(animal_name, date, light_epoch)),
        ("dark", get_dark_epoch(animal_name, date, dark_epoch)),
    )


def _missing_panel_quant_artifacts(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    encoding_n_folds: int,
    place_bin_size_cm: float,
) -> list[dict[str, str]]:
    """Return missing D/E/F artifact records before any quantitative plotting."""
    missing: list[dict[str, str]] = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        for epoch_type, epoch in build_panel_quant_epoch_specs(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ):
            tuning_path = get_tuning_similarity_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
            )
            if not tuning_path.exists():
                missing.append(
                    {
                        "artifact": "tuning_analysis",
                        "animal_name": animal_name,
                        "date": date,
                        "epoch_type": epoch_type,
                        "epoch": epoch,
                        "path": str(tuning_path),
                    }
                )

            encoding_paths = get_encoding_summary_candidate_paths(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                n_folds=encoding_n_folds,
                place_bin_size_cm=place_bin_size_cm,
            )
            if _resolve_existing_path(encoding_paths) is None:
                missing.append(
                    {
                        "artifact": "encoding_comparison",
                        "animal_name": animal_name,
                        "date": date,
                        "epoch_type": epoch_type,
                        "epoch": epoch,
                        "path": str(encoding_paths[0]),
                    }
                )

            decoding_path = get_decoding_summary_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
            )
            if not decoding_path.exists():
                missing.append(
                    {
                        "artifact": "decoding_comparison",
                        "animal_name": animal_name,
                        "date": date,
                        "epoch_type": epoch_type,
                        "epoch": epoch,
                        "path": str(decoding_path),
                    }
                )
    return missing


def _raise_for_missing_panel_quant_artifacts(missing: Sequence[dict[str, str]]) -> None:
    """Raise a concise error listing missing D/E/F artifacts."""
    if not missing:
        return
    lines = [
        "Missing required Figure 3 D/E/F artifact(s). Run the listed analysis "
        "workflow(s) first:"
    ]
    lines.extend(
        (
            f"- {record['artifact']} for {record['animal_name']} {record['date']} "
            f"{record['epoch']} ({record['epoch_type']}): {record['path']}"
        )
        for record in missing
    )
    raise FileNotFoundError("\n".join(lines))


def _require_columns(table: Any, path: Path, columns: Sequence[str]) -> None:
    """Validate that one loaded artifact table has required columns."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Artifact table {path} is missing columns {missing!r}.")


def load_panel_d_similarity_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
) -> Any:
    """Load same-turn tuning-curve correlations for light and dark epochs."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        for epoch_type, epoch in build_panel_quant_epoch_specs(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ):
            path = get_tuning_similarity_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
            )
            table = read_parquet_table(path)
            _require_columns(
                table,
                path,
                ("unit", "region", "epoch", "comparison_label", "similarity"),
            )
            filtered = table[
                (table["region"].astype(str) == region)
                & (table["epoch"].astype(str) == epoch)
                & (table["comparison_label"].astype(str).isin(PANEL_D_COMPARISON_LABELS))
            ].copy()
            filtered["similarity"] = pd.to_numeric(
                filtered["similarity"],
                errors="coerce",
            )
            filtered = filtered[
                np.isfinite(filtered["similarity"].to_numpy(dtype=float))
            ].copy()
            if filtered.empty:
                continue
            filtered = filtered.assign(
                animal_name=animal_name,
                date=date,
                epoch_type=epoch_type,
                source_path=str(path),
            )
            tables.append(
                filtered[
                    [
                        "animal_name",
                        "date",
                        "epoch_type",
                        "epoch",
                        "unit",
                        "comparison_label",
                        "similarity",
                        "source_path",
                    ]
                ]
            )

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "epoch_type",
                "epoch",
                "unit",
                "comparison_label",
                "similarity",
                "source_path",
            ]
        )
    return pd.concat(tables, axis=0, ignore_index=True)


def load_panel_e_encoding_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    n_folds: int = PANEL_E_ENCODING_N_FOLDS,
    place_bin_size_cm: float = PANEL_E_PLACE_BIN_SIZE_CM,
) -> Any:
    """Load directional path progression minus place delta log likelihoods."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        for epoch_type, epoch in build_panel_quant_epoch_specs(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ):
            path = _resolve_existing_path(
                get_encoding_summary_candidate_paths(
                    data_root,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    epoch=epoch,
                    n_folds=n_folds,
                    place_bin_size_cm=place_bin_size_cm,
                )
            )
            if path is None:
                continue
            table = read_parquet_table(path)
            _require_columns(table, path, (PANEL_E_DELTA_COLUMN, "n_spikes"))
            unit_ids = pd.to_numeric(table.index.to_numpy(), errors="coerce")
            values = -pd.to_numeric(table[PANEL_E_DELTA_COLUMN], errors="coerce").to_numpy(
                dtype=float
            )
            rows = pd.DataFrame(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch_type": epoch_type,
                    "epoch": epoch,
                    "unit": unit_ids,
                    "n_spikes": pd.to_numeric(table["n_spikes"], errors="coerce").to_numpy(),
                    "delta_bits_tp_vs_place": values,
                    "source_path": str(path),
                }
            )
            rows = rows[
                np.isfinite(rows["unit"].to_numpy(dtype=float))
                & np.isfinite(rows["delta_bits_tp_vs_place"].to_numpy(dtype=float))
            ].copy()
            if rows.empty:
                continue
            rows["unit"] = rows["unit"].astype(int)
            tables.append(rows)

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "epoch_type",
                "epoch",
                "unit",
                "n_spikes",
                "delta_bits_tp_vs_place",
                "source_path",
            ]
        )
    return pd.concat(tables, axis=0, ignore_index=True)


def load_panel_f_decoding_summary_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
) -> Any:
    """Load per-session TP and place decoding-error summaries."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        for epoch_type, epoch in build_panel_quant_epoch_specs(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ):
            path = get_decoding_summary_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
            )
            table = read_parquet_table(path)
            _require_columns(
                table,
                path,
                ("model", "n_units", PANEL_F_DECODING_METRIC),
            )
            filtered = table[
                table["model"].astype(str).isin(PANEL_F_DECODING_MODELS)
            ].copy()
            filtered[PANEL_F_DECODING_METRIC] = pd.to_numeric(
                filtered[PANEL_F_DECODING_METRIC],
                errors="coerce",
            )
            filtered = filtered[
                np.isfinite(filtered[PANEL_F_DECODING_METRIC].to_numpy(dtype=float))
            ].copy()
            if filtered.empty:
                continue
            filtered = filtered.assign(
                animal_name=animal_name,
                date=date,
                epoch_type=epoch_type,
                epoch=epoch,
                source_path=str(path),
            )
            tables.append(
                filtered[
                    [
                        "animal_name",
                        "date",
                        "epoch_type",
                        "epoch",
                        "model",
                        "n_units",
                        PANEL_F_DECODING_METRIC,
                        "source_path",
                    ]
                ]
            )

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "epoch_type",
                "epoch",
                "model",
                "n_units",
                PANEL_F_DECODING_METRIC,
                "source_path",
            ]
        )
    return pd.concat(tables, axis=0, ignore_index=True)


def load_panel_quantification_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    encoding_n_folds: int = PANEL_E_ENCODING_N_FOLDS,
    place_bin_size_cm: float = PANEL_E_PLACE_BIN_SIZE_CM,
) -> dict[str, Any]:
    """Load the saved-artifact payload for panels D, E, and F."""
    missing = _missing_panel_quant_artifacts(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        encoding_n_folds=encoding_n_folds,
        place_bin_size_cm=place_bin_size_cm,
    )
    _raise_for_missing_panel_quant_artifacts(missing)
    return {
        "similarity": load_panel_d_similarity_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ),
        "encoding_delta": load_panel_e_encoding_delta_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            n_folds=encoding_n_folds,
            place_bin_size_cm=place_bin_size_cm,
        ),
        "decoding_summary": load_panel_f_decoding_summary_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ),
    }


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return weights that make a histogram sum to one."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return np.full(values.shape, 1.0 / values.size, dtype=float)


def _finite_column_values(table: Any, column: str) -> np.ndarray:
    """Return finite numeric values from one table column."""
    if table is None or column not in table:
        return np.asarray([], dtype=float)
    values = np.asarray(table[column], dtype=float)
    return values[np.isfinite(values)]


def _symmetric_limits(values: np.ndarray, *, minimum: float) -> tuple[float, float]:
    """Return symmetric finite limits for one distribution."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -minimum, minimum
    limit = max(minimum, float(np.nanmax(np.abs(finite))))
    limit = float(np.ceil(limit * 10.0) / 10.0)
    return -limit, limit


def build_panel_d_similarity_pairs(similarity_table: Any) -> Any:
    """Return paired light and dark per-unit max same-turn correlations."""
    import pandas as pd

    required_columns = (
        "animal_name",
        "date",
        "unit",
        "comparison_label",
        "epoch_type",
        "similarity",
    )
    missing_columns = [
        column for column in required_columns if column not in similarity_table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Panel D similarity table is missing columns {missing_columns!r}."
        )

    table = similarity_table.copy()
    table = table[
        table["epoch_type"].astype(str).isin(PANEL_QUANT_EPOCH_ORDER)
        & table["comparison_label"].astype(str).isin(PANEL_D_COMPARISON_LABELS)
    ].copy()
    table["similarity"] = pd.to_numeric(table["similarity"], errors="coerce")
    table["unit"] = pd.to_numeric(table["unit"], errors="coerce")
    table = table[
        np.isfinite(table["similarity"].to_numpy(dtype=float))
        & np.isfinite(table["unit"].to_numpy(dtype=float))
    ].copy()
    if table.empty:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "unit",
                "similarity_light",
                "similarity_dark",
            ]
        )
    table["unit"] = table["unit"].astype(int)
    key_columns = ["animal_name", "date", "unit"]
    table = (
        table.groupby([*key_columns, "epoch_type"], as_index=False, observed=False)[
            "similarity"
        ]
        .max()
    )
    light = table[table["epoch_type"].astype(str) == "light"][
        key_columns + ["similarity"]
    ].rename(columns={"similarity": "similarity_light"})
    dark = table[table["epoch_type"].astype(str) == "dark"][
        key_columns + ["similarity"]
    ].rename(columns={"similarity": "similarity_dark"})
    pairs = light.merge(dark, on=key_columns, how="inner")
    pairs = pairs[
        np.isfinite(pairs["similarity_light"].to_numpy(dtype=float))
        & np.isfinite(pairs["similarity_dark"].to_numpy(dtype=float))
    ].copy()
    return pairs


def plot_panel_d_similarity(ax: "Axes", similarity_table: Any) -> None:
    """Plot paired light-vs-dark max same-turn tuning-curve correlations."""
    paired = build_panel_d_similarity_pairs(similarity_table)
    ax.plot(
        [-1.0, 1.0],
        [-1.0, 1.0],
        color="0.35",
        linestyle="--",
        linewidth=0.65,
        zorder=1,
    )
    if len(paired) > 0:
        ax.scatter(
            paired["similarity_light"].to_numpy(dtype=float),
            paired["similarity_dark"].to_numpy(dtype=float),
            s=6,
            color=PANEL_QUANT_EPOCH_COLORS["light"],
            alpha=0.18,
            edgecolors="none",
            zorder=2,
        )
        x_values = paired["similarity_light"].to_numpy(dtype=float)
        y_values = paired["similarity_dark"].to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        summary = f"n={int(np.sum(valid))}"
        if np.sum(valid) > 1:
            correlation = float(np.corrcoef(x_values[valid], y_values[valid])[0, 1])
            if np.isfinite(correlation):
                summary = f"{summary}\nr={correlation:.2f}"
        ax.text(
            0.04,
            0.96,
            summary,
            ha="left",
            va="top",
            fontsize=5.0,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No paired\nsimilarity", ha="center", va="center")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Light same-turn corr.", fontsize=6.2, labelpad=1.5)
    ax.set_ylabel("Dark same-turn corr.", fontsize=6.2, labelpad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def plot_panel_e_encoding_delta_histogram(ax: "Axes", delta_table: Any) -> None:
    """Plot TP minus place encoding delta log-likelihood histograms."""
    values_by_epoch = {
        epoch_type: _finite_column_values(
            delta_table[delta_table["epoch_type"].astype(str) == epoch_type],
            "delta_bits_tp_vs_place",
        )
        for epoch_type in PANEL_QUANT_EPOCH_ORDER
    }
    all_values = np.concatenate(
        [values for values in values_by_epoch.values() if values.size]
    ) if any(values.size for values in values_by_epoch.values()) else np.asarray([], dtype=float)
    x_limits = _symmetric_limits(all_values, minimum=0.35)
    bin_edges = np.linspace(x_limits[0], x_limits[1], 27)

    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    plotted_any = False
    summary_lines = []
    for epoch_type in PANEL_QUANT_EPOCH_ORDER:
        values = values_by_epoch[epoch_type]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            alpha=0.48 if epoch_type == "light" else 0.34,
            edgecolor="white",
            linewidth=0.25,
            label=PANEL_QUANT_EPOCH_LABELS[epoch_type],
            zorder=2,
        )
        median_value = float(np.nanmedian(values))
        ax.axvline(
            median_value,
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            linewidth=0.85,
            zorder=3,
        )
        summary_lines.append(
            f"{PANEL_QUANT_EPOCH_LABELS[epoch_type]}: "
            f"n={values.size}, med={median_value:.2f}"
        )
        plotted_any = True

    if plotted_any:
        ax.legend(frameon=False, fontsize=5.2, handlelength=1.0, borderpad=0.1)
        ax.text(
            0.98,
            0.95,
            "\n".join(summary_lines),
            ha="right",
            va="top",
            fontsize=4.8,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No encoding\nvalues", ha="center", va="center")

    ax.set_xlim(*x_limits)
    ax.set_xlabel("TP - place LL\n(bits/spike)", fontsize=5.8, labelpad=1.5)
    ax.set_ylabel("Frac. units", fontsize=6.2, labelpad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def _plot_panel_f_model_axis(
    ax: "Axes",
    decoding_table: Any,
    *,
    model: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot one decoding-error model across light and dark epochs."""
    model_table = decoding_table[decoding_table["model"].astype(str) == model].copy()
    positions = np.arange(1, len(PANEL_QUANT_EPOCH_ORDER) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [PANEL_QUANT_EPOCH_LABELS[epoch_type] for epoch_type in PANEL_QUANT_EPOCH_ORDER]
    )
    ax.set_title(title, fontsize=5.8, pad=1.5)

    if model_table.empty:
        ax.text(0.5, 0.5, "No decoding\nsummary", ha="center", va="center")
        ax.set_xlim(0.5, len(PANEL_QUANT_EPOCH_ORDER) + 0.5)
        return

    animal_names = list(dict.fromkeys(model_table["animal_name"].astype(str)))
    for animal_name in animal_names:
        animal_table = model_table[model_table["animal_name"].astype(str) == animal_name]
        animal_x = []
        animal_y = []
        for position, epoch_type in zip(positions, PANEL_QUANT_EPOCH_ORDER, strict=True):
            values = _finite_column_values(
                animal_table[animal_table["epoch_type"].astype(str) == epoch_type],
                PANEL_F_DECODING_METRIC,
            )
            if values.size == 0:
                continue
            animal_x.append(float(position))
            animal_y.append(float(np.nanmedian(values)))
        if len(animal_x) >= 2:
            ax.plot(animal_x, animal_y, color="0.72", linewidth=0.55, zorder=1)
        if animal_x:
            ax.scatter(
                animal_x,
                animal_y,
                s=9,
                color="white",
                edgecolor="0.25",
                linewidth=0.45,
                zorder=3,
            )

    all_values = []
    for position, epoch_type in zip(positions, PANEL_QUANT_EPOCH_ORDER, strict=True):
        values = _finite_column_values(
            model_table[model_table["epoch_type"].astype(str) == epoch_type],
            PANEL_F_DECODING_METRIC,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        all_values.append(values)
        median_value = float(np.nanmedian(values))
        q25, q75 = np.quantile(values, [0.25, 0.75]).astype(float)
        color = PANEL_QUANT_EPOCH_COLORS[epoch_type]
        ax.vlines(position, q25, q75, color=color, linewidth=1.0, zorder=4)
        ax.scatter(
            [position],
            [median_value],
            s=16,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=5,
        )

    finite_all = np.concatenate(all_values) if all_values else np.asarray([], dtype=float)
    y_max = float(np.nanmax(finite_all)) if finite_all.size else 1.0
    ax.set_ylim(0.0, max(y_max * 1.18, 0.05))
    ax.set_xlim(0.5, len(PANEL_QUANT_EPOCH_ORDER) + 0.5)
    ax.set_ylabel(ylabel, fontsize=5.4, labelpad=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.0, length=1.5, pad=1)


def plot_panel_f_decoding_error(ax: "Axes", decoding_table: Any) -> None:
    """Plot TP and place decoding errors for light and dark epochs."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    tp_ax = ax.inset_axes([0.08, 0.20, 0.38, 0.66])
    place_ax = ax.inset_axes([0.58, 0.20, 0.38, 0.66])
    _plot_panel_f_model_axis(
        tp_ax,
        decoding_table,
        model="task_progression",
        title="TP",
        ylabel="Median abs.\nerror (norm.)",
    )
    _plot_panel_f_model_axis(
        place_ax,
        decoding_table,
        model="place",
        title="Place",
        ylabel="Median abs.\nerror (cm)",
    )


def make_figure_3(
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
) -> Path:
    """Build and save Figure 3."""
    import matplotlib.pyplot as plt

    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_quant_payload = load_panel_quantification_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

    apply_paper_style()
    fig_height_mm = DEFAULT_PANEL_A_HEIGHT_MM + (
        DEFAULT_PANEL_BC_HEIGHT_MM * max(len(regions), 1)
    ) + DEFAULT_PANEL_DEF_HEIGHT_MM
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            DEFAULT_PANEL_A_HEIGHT_MM,
            DEFAULT_PANEL_BC_HEIGHT_MM * max(len(regions), 1),
            DEFAULT_PANEL_DEF_HEIGHT_MM,
        ],
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    middle_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[
            DEFAULT_PANEL_B_WIDTH_FRACTION,
            DEFAULT_PANEL_C_WIDTH_FRACTION,
        ],
    )
    panel_b = setup_light_heatmap_panel(
        fig,
        middle_grid[0, 0],
        regions=regions,
    )
    panel_c_axis = fig.add_subplot(middle_grid[0, 1])
    bottom_grid = outer_grid[2, 0].subgridspec(nrows=1, ncols=3)
    panel_d_axis = fig.add_subplot(bottom_grid[0, 0])
    panel_e_axis = fig.add_subplot(bottom_grid[0, 1])
    panel_f_axis = fig.add_subplot(bottom_grid[0, 2])

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
    )
    plot_panel_a_example(panel_a_axis, panel_a_example)

    color_image = plot_light_heatmap_regions(
        panel_b["heatmap_axes"],
        data_root=data_root,
        datasets=datasets,
        regions=regions,
        light_epoch=light_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=panel_b["heatmap_axes"].ravel().tolist(),
            shrink=0.24,
            pad=0.01,
            aspect=7,
            ticks=[0.0, 1.0],
        )
        colorbar.ax.set_yticklabels(["0", "1"])
        colorbar.ax.tick_params(length=2)
        colorbar.set_label("Norm. FR", rotation=90, labelpad=4)
    draw_neuron_scale_bar(panel_b["heatmap_axes"][-1, -1])

    examples = [
        load_panel_c_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            unit_id=unit_id,
            trajectories=trajectories,
            dark_epoch=dark_epoch,
            light_epoch=light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        )
        for animal_name, date, region, unit_id, trajectories in PANEL_C_EXAMPLES
    ]
    plot_panel_c_examples(panel_c_axis, examples)
    plot_panel_d_similarity(panel_d_axis, panel_quant_payload["similarity"])
    plot_panel_e_encoding_delta_histogram(
        panel_e_axis,
        panel_quant_payload["encoding_delta"],
    )
    plot_panel_f_decoding_error(
        panel_f_axis,
        panel_quant_payload["decoding_summary"],
    )

    fig.canvas.draw()
    add_centered_axis_text(fig, panel_b["tuning_schematic_axes"], "Tuning", y_offset=0.005)
    add_centered_axis_text(
        fig,
        panel_b["order_schematic_axes"],
        "Order",
        y_offset=0.018,
        rotation=90,
    )
    label_axis(panel_a_axis, "A", x=-0.02, y=1.00)
    label_axis(panel_b["corner_axis"], "B", x=-0.12, y=1.04)
    label_axis(panel_c_axis, "C", x=-0.04, y=1.02)
    label_axis(panel_d_axis, "D", x=-0.18, y=1.12)
    label_axis(panel_e_axis, "E", x=-0.06, y=1.02)
    label_axis(panel_f_axis, "F", x=-0.06, y=1.02)
    panel_a_axis.set_title("Example cell rasters across run epochs", fontsize=8, pad=2)
    panel_b["corner_axis"].set_title("Light epoch", fontsize=8, pad=2)
    panel_c_axis.set_title("Dark-light tuning examples", fontsize=8, pad=2)
    panel_d_axis.set_title("Same-turn tuning similarity", fontsize=8, pad=5)
    panel_e_axis.set_title("Encoding comparison", fontsize=8, pad=2)
    panel_f_axis.set_title("Decoding error", fontsize=8, pad=2)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 3 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 light-epoch heatmaps and dark-light tuning examples."
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
        help=(
            "Light run epoch for panel A and panel B. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help=(
            "Dark run epoch for panel B. "
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
    """Run Figure 3 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_figure_3(
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
    )


if __name__ == "__main__":
    main()
