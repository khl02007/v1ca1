from __future__ import annotations

"""Shared helpers for dark-light task-progression figures."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    TRAJECTORY_TYPES,
    get_analysis_path,
)
from v1ca1.helper.plot_wtrack_schematic import (
    get_w_track_geometry,
    trajectory_points,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_dataset_dark_epoch,
    get_dataset_light_epoch,
    make_dataset_id,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_POSITION_BIN_COUNT,
    DECODING_CROSS_TRAJECTORY_COMPARISONS,
    HEATMAP_PATH_LABEL_OFFSET,
    PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_E_AXIS_LABEL_FONTSIZE,
    TASK_PROGRESSION_XLABEL,
    add_centered_axis_text,
    add_centered_below_axis_text,
    build_normalized_position_bins,
    build_pooled_panel_values,
    compute_dark_epoch_tuning_curves,
    draw_order_schematic,
    extract_unit_rate_curve,
    get_stability_table_path,
    get_unit_spike_times,
    normalize_linear_position_by_trajectory,
    orient_panel_e_task_progression,
    plot_pooled_heatmap_grid,
    select_units_by_saved_movement_firing_rate,
)
from v1ca1.paper_figures.style import (
    EPOCH_TYPE_COLORS,
    EPOCH_HISTOGRAM_ALPHA,
    MODEL_CLASS_COLORS,
    NEUTRAL_COLORS,
    OUTLINED_HISTOGRAM_KWARGS,
    PANEL_LABEL_KWARGS,
    RASTER_TICK_KWARGS,
    SCHEMATIC_COLORS,
    TRAJECTORY_COLORS,
    label_axis,
)
from v1ca1.helper.wtrack import get_wtrack_total_length
from v1ca1.paper_figures.w_track_schematic import (
    draw_w_track_basis_schematic,
    draw_w_track_schematic,
)
from v1ca1.raster.plot_place_field_heatmap import (
    DEFAULT_SIGMA_BINS,
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    prepare_heatmap_session,
)
from v1ca1.raster.plot_1d_place_field_trajectory import (
    compute_trial_spike_positions,
    make_linear_position_interpolator,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from matplotlib.text import Text


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
LEGACY_CACHE_FIGURE_NAME = "old_fig3"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_REGIONS = ("v1",)
PANEL_DEF_AXIS_BOTTOM = 0.10
PANEL_DEF_AXIS_HEIGHT = 0.60
PANEL_GH_WIDTH_RATIOS = (0.4, 0.6)
PANEL_CD_GROUP_TITLE_Y = 1.05
PANEL_CD_GROUP_TITLE_FONTSIZE = 7.2
PANEL_C_SCATTER_AXIS_BOUNDS = (
    0.07,
    PANEL_DEF_AXIS_BOTTOM,
    0.44,
    PANEL_DEF_AXIS_HEIGHT,
)
PANEL_C_CROSS_ROUTE_AXIS_BOUNDS = (
    0.62,
    PANEL_DEF_AXIS_BOTTOM,
    0.32,
    PANEL_DEF_AXIS_HEIGHT,
)
PANEL_D_HISTOGRAM_AXIS_BOUNDS = (
    0.06,
    PANEL_DEF_AXIS_BOTTOM,
    0.54,
    PANEL_DEF_AXIS_HEIGHT,
)
PANEL_D_PLACE_DECODING_AXIS_BOUNDS = (
    0.70,
    PANEL_DEF_AXIS_BOTTOM,
    0.26,
    PANEL_DEF_AXIS_HEIGHT,
)
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
PANEL_B_TRAJECTORY_TYPES = (
    "right_to_center",
    "center_to_left",
    "left_to_center",
    "center_to_right",
)
PANEL_B_LINEAR_POSITION_ORIENTATION = "task_progression"
PANEL_B_HEATMAP_CMAP = "viridis"
PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ
PANEL_B_MIN_TUNING_STABILITY_CORRELATION = PANEL_D_MIN_TUNING_STABILITY_CORRELATION
PANEL_B_FIRING_RATE_NORMALIZATION = PANEL_D_PER_TRAJECTORY_FIRING_RATE_NORMALIZATION
PANEL_B_CACHE_VERSION = 3
PANEL_B_CACHE_PREFIX = "figure_3_panel_b"
PANEL_B_CACHE_METADATA_KEY = "__metadata__"
PANEL_B_CACHE_DATASET_TOKEN_LIMIT = 120
PANEL_EXAMPLE_CACHE_VERSION = 1
PANEL_EXAMPLE_CACHE_PREFIX = "figure_3_panel_example"
PANEL_EXAMPLE_CACHE_METADATA_KEY = "__metadata__"
# Preserve the historical cache token so relabeling the example block as Panel A
# does not force expensive local example-cell recomputation.
PANEL_A_EXAMPLE_CACHE_PANEL_NAME = "C"
TUNING_ANALYSIS_RELATIVE_DIR = Path("task_progression") / "tuning_analysis"
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
DECODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "decoding_comparison"
DARK_LIGHT_GLM_RELATIVE_DIR = Path("task_progression") / "dark_light_glm"
SWAP_GLM_COMPARISON_RELATIVE_DIR = Path("task_progression") / "swap_glm_comparison"
COMPUTE_TUNING_CURVES_RELATIVE_DIR = Path("task_progression") / "compute_tuning_curves"
SEGMENT_BOUNDARIES = (0.4, 0.6)
SEGMENT_BOUNDARY_COLOR = NEUTRAL_COLORS["segment_boundary"]
SEGMENT_BOUNDARY_LINEWIDTH = 0.45
PANEL_A_EXAMPLES = (
    ("L14", "20240611", "v1", 34, ("center_to_left", "right_to_center")),
    ("L15", "20241121", "v1", 473, ("center_to_right", "left_to_center")),
)
PANEL_TRAJECTORY_LABELS = {
    "center_to_left": "C to L",
    "right_to_center": "R to C",
    "center_to_right": "C to R",
    "left_to_center": "L to C",
}
PANEL_TRAJECTORY_COLORS = TRAJECTORY_COLORS
PANEL_A_EPOCH_LABELS = {
    "dark": "Dark",
    "light": "Light",
}
PANEL_A_DARK_EPOCH_BACKGROUND = NEUTRAL_COLORS["dark_epoch_background"]
PANEL_A_EXAMPLE_BLOCK_HEIGHT = 0.66
PANEL_A_EXAMPLE_TOP = 1.16
PANEL_A_EXAMPLE_BOTTOM = -0.044
PANEL_A_FIRST_EXAMPLE_Y_SHIFT = 0.09
PANEL_A_EXAMPLE_RASTER_Y = 0.48
PANEL_A_EXAMPLE_RASTER_HEIGHT = 0.34
PANEL_A_EXAMPLE_RATE_Y = 0.16
PANEL_A_EXAMPLE_RATE_HEIGHT = 0.31
PANEL_AB_HEADER_Y_OFFSET = 0.012
PANEL_QUANT_EPOCH_ORDER = ("light", "dark")
PANEL_QUANT_EPOCH_LABELS = {
    "light": "Light",
    "dark": "Dark",
}
PANEL_QUANT_EPOCH_COLORS = {
    "light": EPOCH_TYPE_COLORS["light"],
    "dark": EPOCH_TYPE_COLORS["dark"],
}
PANEL_C_SIMILARITY_COMPARISON_LABELS = ("left_turn", "right_turn")
PANEL_C_SCATTER_SIZE = 3.0
PANEL_C_SCATTER_ALPHA = 0.2
PANEL_D_ENCODING_N_FOLDS = 5
PANEL_D_PLACE_BIN_SIZE_CM = DEFAULT_PLACE_BIN_SIZE_CM
PANEL_D_ENCODING_DELTA_COLUMN = "delta_bits_place_vs_tp"
PANEL_D_MIN_TUNING_STABILITY_CORRELATION = 0.5
PANEL_D_ENCODING_X_LIMITS = (-0.75, 0.75)
PANEL_E_DECODING_MODELS = ("task_progression", "place")
PANEL_E_DECODING_METRIC = "median_abs_error"
PANEL_E_PLACE_MODEL_NAME = "place"
PANEL_E_POOLED_LABEL = "pooled"
PANEL_E_CROSS_COMPARISONS = DECODING_CROSS_TRAJECTORY_COMPARISONS[:1]
PANEL_E_NORM_ERROR_YLIM = (0.0, 0.5)
PANEL_E_PLACE_ERROR_YLIM = (0.0, 0.2)
PANEL_E_YLABEL_FONTSIZE = 5.8
PANEL_E_CROSS_AXIS_BOUNDS = (0.04, PANEL_DEF_AXIS_BOTTOM, 0.42, PANEL_DEF_AXIS_HEIGHT)
PANEL_E_PLACE_AXIS_BOUNDS = (0.56, PANEL_DEF_AXIS_BOTTOM, 0.42, PANEL_DEF_AXIS_HEIGHT)
PANEL_E_SUMMARY_TEXT_X = 0.97
PANEL_E_PLACE_SUMMARY_TEXT_X = 0.04
PANEL_QUANT_SUMMARY_TEXT_FONTSIZE = 4.2
PANEL_E_ERROR_SUMMARY_COLUMNS = (
    "animal_name",
    "date",
    "epoch_type",
    "epoch",
    "analysis",
    "comparison",
    "comparison_label",
    "q25_error",
    "median_error",
    "q75_error",
    "n_samples",
)
GLM_MODEL_LABELS = {
    "visual": "Independent",
    "task_segment_bump": "Shared scaffold",
    "task_segment_scalar": "Segment scalar",
}
GLM_MODEL_COLORS = MODEL_CLASS_COLORS
GLM_BASIS_DARK_COLOR = SCHEMATIC_COLORS["dark_basis"]
GLM_BASIS_LIGHT_COLOR = SCHEMATIC_COLORS["light_basis"]
GLM_TRAJECTORY_ARROW_COLOR = SCHEMATIC_COLORS["trajectory_arrow"]
GLM_EMPIRICAL_COLOR = NEUTRAL_COLORS["empirical"]
PANEL_G_COMPARISON_MODEL_NAME = "task_segment_scalar"
PANEL_G_MODELS = ("visual", PANEL_G_COMPARISON_MODEL_NAME)
PANEL_G_MODEL_LABELS = {
    "visual": "Independent",
    PANEL_G_COMPARISON_MODEL_NAME: GLM_MODEL_LABELS[PANEL_G_COMPARISON_MODEL_NAME],
}
PANEL_G_MODEL_COLORS = MODEL_CLASS_COLORS
PANEL_G_SCHEMATIC_WIDTH_FRACTION = 1.00
PANEL_G_EXAMPLE_WIDTH_FRACTION = 1.00
PANEL_G_SCHEMATIC_HEIGHT_FRACTION = 0.66
PANEL_G_EXAMPLE_HEIGHT_FRACTION = 0.30
PANEL_G_EXAMPLES = (
    ("L15", "20241121", "v1", 426, "center_to_right"),
    ("L14", "20240611", "v1", 99, "center_to_left"),
)
PANEL_G_BASIS_DARK_COLOR = SCHEMATIC_COLORS["dark_basis"]
PANEL_G_BASIS_LIGHT_COLOR = SCHEMATIC_COLORS["light_basis"]
PANEL_G_ARROW_COLOR = SCHEMATIC_COLORS["trajectory_arrow"]
PANEL_G_SHARED_SCAFFOLD_BASIS_LINEWIDTH = 0.10
PANEL_G_SHARED_SCAFFOLD_OVAL_LINEWIDTH = 0.45
PANEL_G_EMPIRICAL_COLOR = NEUTRAL_COLORS["empirical"]
PANEL_G_EXAMPLE_MODEL_COLORS = MODEL_CLASS_COLORS
PANEL_G_EXAMPLE_COUNT = 2
PANEL_G_INDEPENDENT_TRACK_CENTER_Y = 0.715
PANEL_G_SHARED_TRACK_CENTER_Y = 0.205
PANEL_G_DARK_TRACK_CENTER_X = 0.25
PANEL_G_LIGHT_TRACK_CENTER_X = 0.86
PANEL_G_SEGMENT_MODULATION_TRACK_CENTER_X = 0.53
PANEL_G_SHARED_OUTPUT_ARROW_X = (0.69, 0.74)
PANEL_G_SCHEMATIC_INSET_ZORDER = -1.0
PANEL_G_SCHEMATIC_TEXT_ZORDER = 5.0
PANEL_G_FIELD_LABEL_Y = 0.98
PANEL_G_INDEPENDENT_BASIS_LABEL_Y = 0.87
PANEL_G_COMPONENT_LABEL_FONTSIZE = 4.3
PANEL_G_SEGMENT_MODULATION_LABEL_GAP = 0.045
PANEL_G_SEGMENT_GAIN_OUTLINE_OUTSET = 0.16
PANEL_G_PLACE_FIELD_PATH_ARROW_TRAJECTORY = "center_to_left"
PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_COLOR = "white"
PANEL_G_PLACE_FIELD_PATH_ARROW_COLOR = "black"
PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_ALPHA = 0.88
PANEL_G_PLACE_FIELD_PATH_ARROW_ALPHA = 0.78
PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_LINEWIDTH = 0.72
PANEL_G_PLACE_FIELD_PATH_ARROW_LINEWIDTH = 0.40
PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_LENGTH = 0.42
PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_WIDTH = 0.38
PANEL_G_PLACE_FIELD_PATH_ARROW_ZORDER = 3.30
PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_OUTLINE_ZORDER = 3.34
PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_ZORDER = 3.35
PANEL_G_INDEPENDENT_BASIS_ICON_WIDTH = 0.16
PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT = 0.24
PANEL_G_INDEPENDENT_BASIS_ICON_LEFT_X = 0.43
PANEL_G_INDEPENDENT_BASIS_ICON_RIGHT_X = 0.57
PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM = 0.18
PANEL_G_INDEPENDENT_BASIS_ICON_TOP = 0.58
PANEL_H_SWAP_LIGHT_EPOCH_PAIRS = (("02_r1", "06_r3"), ("06_r3", "02_r1"))
PANEL_H_HELDOUT_LIGHT_EPOCH = "06_r3"
PANEL_H_TRAIN_LIGHT_EPOCH = "02_r1"
PANEL_H_SWAP_DELTA_VARIABLE = (
    "test_light_swapped_segment_swapped_delta_model_minus_visual_raw_ll_bits_per_spike"
)
PANEL_H_DEFAULT_MODEL_NAME = "task_segment_bump"
PANEL_H_DELTA_X_LIMITS = (-1.0, 1.0)
PANEL_H_DELTA_TRAJECTORIES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
PANEL_H_EXAMPLES = (
    ("L15", "20241121", "v1", 27, "center_to_right"),
    ("L14", "20240611", "v1", 368, "right_to_center"),
)
PANEL_H_SCHEMATIC_TRACK_LINEWIDTH = 0.55
PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH = 0.85
PANEL_H_SCHEMATIC_BASIS_LINEWIDTH = 0.145
PANEL_H_SCHEMATIC_DARK_BASIS_LINEWIDTH = 0.25
PANEL_H_SCHEMATIC_OVAL_LINEWIDTH = 0.45
PANEL_H_SCHEMATIC_ARROW_SCALE = 6.5
PANEL_H_SCHEMATIC_BASIS_RADIUS = 0.30
PANEL_H_SCHEMATIC_BASIS_SPACING = 0.34
PANEL_H_SCHEMATIC_AXIS_BOUNDS = (-0.025, 0.38, 0.34, 0.58)
PANEL_H_DELTA_AXIS_BOUNDS = (0.35, 0.42, 0.62, 0.52)
PANEL_H_INDEPENDENT_TRACK_CENTER_Y = 0.765
PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y = 0.380
PANEL_H_SHARED_DARK_TRACK_CENTER_Y = 0.190
PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y = 0.250
PANEL_H_EXAMPLE_AXIS_BOUNDS = (
    (0.18, 0.06, 0.24, 0.25),
    (0.58, 0.06, 0.24, 0.25),
)


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
    n_folds: int = PANEL_D_ENCODING_N_FOLDS,
    place_bin_size_cm: float = PANEL_D_PLACE_BIN_SIZE_CM,
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


def get_within_epoch_decoding_tsd_paths(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Return true and decoded within-epoch decoding `.npz` artifact paths."""
    data_dir = (
        get_dataset_analysis_path(data_root, animal_name, date)
        / DECODING_COMPARISON_RELATIVE_DIR
    )
    return (
        data_dir / f"{region}_{epoch}_true_{model_name}.npz",
        data_dir / f"{region}_{epoch}_decoded_{model_name}.npz",
    )


def get_cross_trajectory_decoding_tsd_paths(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    transfer_family: str,
    encoding_trajectory: str,
    decoding_trajectory: str,
) -> tuple[Path, Path]:
    """Return true and decoded cross-trajectory TP decoding `.npz` artifact paths."""
    data_dir = (
        get_dataset_analysis_path(data_root, animal_name, date)
        / DECODING_COMPARISON_RELATIVE_DIR
    )
    suffix = f"{transfer_family}_{encoding_trajectory}_to_{decoding_trajectory}"
    return (
        data_dir / f"{region}_{epoch}_{suffix}_true_tp_cross_traj.npz",
        data_dir / f"{region}_{epoch}_{suffix}_decoded_tp_cross_traj.npz",
    )


def get_dark_light_glm_selected_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_name: str,
) -> Path:
    """Return one selected dark/light GLM artifact path."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / DARK_LIGHT_GLM_RELATIVE_DIR
        / "selected"
        / f"{region}_{light_epoch}_vs_{dark_epoch}_{model_name}_selected.nc"
    )


def get_compute_tuning_curve_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory: str,
) -> Path:
    """Return one empirical trajectory place-tuning artifact path."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / COMPUTE_TUNING_CURVES_RELATIVE_DIR
        / f"{region}_{epoch}_place_{trajectory}_tuning_curves.nc"
    )


def get_swap_glm_selected_comparison_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    swap_light_offset: bool = False,
) -> Path:
    """Return one selected-source swap-GLM comparison artifact path."""
    suffix = "_swap_light_offset" if swap_light_offset else ""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / SWAP_GLM_COMPARISON_RELATIVE_DIR
        / (
            f"{region}_{dark_epoch}_traindark_"
            f"{light_train_epoch}_trainlight_"
            f"{light_test_epoch}_testlight_"
            f"dark_light_selected_swap{suffix}.nc"
        )
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


def _format_panel_b_cache_token(value: object) -> str:
    """Return a filesystem-safe token for Panel B cache file names."""
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


def _format_panel_b_cache_number(value: float | int) -> str:
    """Return a compact numeric token for Panel B cache file names."""
    return _format_panel_b_cache_token(f"{float(value):g}")


def _build_panel_b_dataset_cache_token(
    dataset_metadata: Sequence[dict[str, str]],
) -> str:
    """Return a descriptive cache token for the Panel B data-set list."""
    dataset_tokens = [
        _format_panel_b_cache_token(
            f"{dataset['animal_name']}-{dataset['date']}-{dataset['light_epoch']}"
        )
        for dataset in dataset_metadata
    ]
    token = "_".join(dataset_tokens) or "none"
    if len(token) <= PANEL_B_CACHE_DATASET_TOKEN_LIMIT:
        return token

    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    prefix = "_".join(dataset_tokens[:2])
    return _format_panel_b_cache_token(
        f"{prefix}_{len(dataset_tokens)}datasets_{digest}"
    )


def build_panel_b_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    min_movement_firing_rate_hz: float | None = PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = (
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    ),
    firing_rate_normalization: str = PANEL_B_FIRING_RATE_NORMALIZATION,
) -> dict[str, Any]:
    """Return metadata that identifies one Panel B heatmap cache."""
    if min_movement_firing_rate_hz is not None and min_movement_firing_rate_hz < 0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")
    if (
        min_tuning_stability_correlation is not None
        and min_tuning_stability_correlation < -1.0
    ):
        raise ValueError("min_tuning_stability_correlation must be at least -1.")
    dataset_metadata = []
    for dataset in datasets:
        animal_name, date, dark_epoch = normalize_dataset_id(dataset)
        dataset_metadata.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": dark_epoch,
                "light_epoch": get_light_epoch(animal_name, date, light_epoch),
            }
        )

    metadata = {
        "cache_version": PANEL_B_CACHE_VERSION,
        "figure": LEGACY_CACHE_FIGURE_NAME,
        "panel": "B",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "light_epoch_argument": light_epoch,
        "datasets": dataset_metadata,
        "trajectory_types": list(PANEL_B_TRAJECTORY_TYPES),
        "linear_position_orientation": PANEL_B_LINEAR_POSITION_ORIENTATION,
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "pooled_builder": "build_pooled_panel_values",
    }
    if min_movement_firing_rate_hz is not None:
        metadata["min_movement_firing_rate_hz"] = float(min_movement_firing_rate_hz)
    if min_tuning_stability_correlation is not None:
        metadata["min_tuning_stability_correlation"] = float(
            min_tuning_stability_correlation
        )
    if firing_rate_normalization != PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION:
        metadata["firing_rate_normalization"] = str(firing_rate_normalization)
    return metadata


def build_panel_b_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for one Panel B heatmap payload."""
    region_token = _format_panel_b_cache_token(metadata["region"])
    dataset_metadata = metadata["datasets"]
    light_epochs = [
        _format_panel_b_cache_token(dataset["light_epoch"])
        for dataset in dataset_metadata
    ]
    unique_light_epochs = list(dict.fromkeys(light_epochs))
    light_epoch_token = (
        unique_light_epochs[0]
        if len(unique_light_epochs) == 1
        else "mixed-" + "_".join(unique_light_epochs)
    )
    dataset_token = _build_panel_b_dataset_cache_token(dataset_metadata)
    filename = (
        f"{PANEL_B_CACHE_PREFIX}_{region_token}_light{light_epoch_token}"
        f"_datasets-{dataset_token}"
        f"_orient{_format_panel_b_cache_token(metadata['linear_position_orientation'])}"
    )
    if "min_movement_firing_rate_hz" in metadata:
        filename += (
            "_minmovefr"
            f"{_format_panel_b_cache_number(metadata['min_movement_firing_rate_hz'])}"
        )
    if "min_tuning_stability_correlation" in metadata:
        filename += (
            "_minstab"
            f"{_format_panel_b_cache_number(metadata['min_tuning_stability_correlation'])}"
        )
    if "firing_rate_normalization" in metadata:
        filename += (
            "_norm"
            f"{_format_panel_b_cache_token(metadata['firing_rate_normalization'])}"
        )
    filename += (
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_b_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_b_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_b_cache_array_name(order_trajectory: str, plot_trajectory: str) -> str:
    """Return the array name for one Panel B heatmap matrix."""
    return f"{order_trajectory}__{plot_trajectory}"


def save_panel_b_cache(
    cache_path: Path,
    panels: dict[tuple[str, str], np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write one Panel B heatmap cache as compressed NumPy arrays."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        PANEL_B_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for order_trajectory in PANEL_B_TRAJECTORY_TYPES:
        for plot_trajectory in PANEL_B_TRAJECTORY_TYPES:
            payload[_panel_b_cache_array_name(order_trajectory, plot_trajectory)] = np.asarray(
                panels[(order_trajectory, plot_trajectory)],
                dtype=float,
            )
    np.savez_compressed(cache_path, **payload)


def load_panel_b_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray] | None:
    """Return cached Panel B heatmap matrices when metadata still matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_B_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Panel B cache at {cache_path}.")
                return None

            panels: dict[tuple[str, str], np.ndarray] = {}
            for order_trajectory in PANEL_B_TRAJECTORY_TYPES:
                for plot_trajectory in PANEL_B_TRAJECTORY_TYPES:
                    array_name = _panel_b_cache_array_name(
                        order_trajectory,
                        plot_trajectory,
                    )
                    panels[(order_trajectory, plot_trajectory)] = np.asarray(
                        data[array_name],
                        dtype=float,
                    )
            return panels
    except Exception as exc:
        print(f"Ignoring unreadable Panel B cache at {cache_path}: {exc}")
        return None


def load_or_compute_panel_b_heatmap_panels(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_b_cache_dir: Path | None,
    refresh_panel_b_cache: bool,
    min_movement_firing_rate_hz: float | None = PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float | None = (
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    ),
    firing_rate_normalization: str = PANEL_B_FIRING_RATE_NORMALIZATION,
) -> dict[tuple[str, str], np.ndarray]:
    """Load cached Panel B panels or compute and cache them."""
    metadata = build_panel_b_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
        firing_rate_normalization=firing_rate_normalization,
    )
    cache_path = (
        build_panel_b_cache_path(panel_b_cache_dir, metadata)
        if panel_b_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_b_cache:
        cached_panels = load_panel_b_cache(cache_path, metadata)
        if cached_panels is not None:
            print(f"Loaded Panel B cache from {cache_path}.")
            return cached_panels

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
                use_trajectory_direction=True,
                min_movement_firing_rate_hz=min_movement_firing_rate_hz,
                min_tuning_stability_correlation=min_tuning_stability_correlation,
            )
        )

    panels = build_pooled_panel_values(
        curve_sets,
        position_bin_count=position_bin_count,
        trajectory_types=PANEL_B_TRAJECTORY_TYPES,
        firing_rate_normalization=firing_rate_normalization,
    )
    if cache_path is not None:
        save_panel_b_cache(cache_path, panels, metadata)
        print(f"Saved Panel B cache to {cache_path}.")
    return panels


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
    use_trajectory_direction: bool = False,
    min_movement_firing_rate_hz: float | None = None,
    min_tuning_stability_correlation: float | None = None,
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
        use_trajectory_direction=use_trajectory_direction,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
    )


def build_panel_example_cache_metadata(
    *,
    data_root: Path,
    panel_name: str,
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
    """Return metadata that identifies one Panel A/C example-cell cache."""
    return {
        "cache_version": PANEL_EXAMPLE_CACHE_VERSION,
        "figure": LEGACY_CACHE_FIGURE_NAME,
        "panel": str(panel_name).upper(),
        "payload": "raster_positions_and_firing_rates",
        "data_root": str(Path(data_root)),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "region": str(region),
        "unit_id": int(unit_id),
        "trajectory_types": list(trajectories),
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
    }


def build_panel_example_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for one Panel A/C example-cell payload."""
    panel_token = _format_panel_b_cache_token(metadata["panel"]).lower()
    dataset_token = "-".join(
        _format_panel_b_cache_token(value)
        for value in (
            metadata["animal_name"],
            metadata["date"],
            metadata["epoch"],
            metadata["region"],
            f"unit{metadata['unit_id']}",
        )
    )
    trajectory_token = "-".join(
        _format_panel_b_cache_token(trajectory)
        for trajectory in metadata["trajectory_types"]
    )
    filename = (
        f"{PANEL_EXAMPLE_CACHE_PREFIX}_{panel_token}_{dataset_token}"
        f"_traj-{trajectory_token}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_b_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_b_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_example_cache_trajectory_token(trajectory_type: str) -> str:
    """Return a compact trajectory token for Panel A/C cache array names."""
    return _format_panel_b_cache_token(trajectory_type)


def save_panel_example_cache(
    cache_path: Path,
    example_data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write one Panel A/C example-cell cache as compressed NumPy arrays."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        PANEL_EXAMPLE_CACHE_METADATA_KEY: np.asarray(
            json.dumps(metadata, sort_keys=True)
        )
    }
    for trajectory_type in metadata["trajectory_types"]:
        token = _panel_example_cache_trajectory_token(str(trajectory_type))
        raster_trials = [
            np.asarray(trial_positions, dtype=float)
            for trial_positions in example_data["raster_positions"][trajectory_type]
        ]
        if raster_trials:
            payload[f"raster_{token}_values"] = np.concatenate(raster_trials)
            payload[f"raster_{token}_lengths"] = np.asarray(
                [trial_positions.size for trial_positions in raster_trials],
                dtype=int,
            )
        else:
            payload[f"raster_{token}_values"] = np.asarray([], dtype=float)
            payload[f"raster_{token}_lengths"] = np.asarray([], dtype=int)

        rate_position, rate_values = example_data["firing_rates"][trajectory_type]
        payload[f"rate_{token}_position"] = np.asarray(rate_position, dtype=float)
        payload[f"rate_{token}_values"] = np.asarray(rate_values, dtype=float)

    np.savez_compressed(cache_path, **payload)


def load_panel_example_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return cached Panel A/C example-cell data when metadata still matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(
                str(data[PANEL_EXAMPLE_CACHE_METADATA_KEY].item())
            )
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Panel example cache at {cache_path}.")
                return None

            raster_positions: dict[str, list[np.ndarray]] = {}
            firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for trajectory_type in expected_metadata["trajectory_types"]:
                trajectory_type = str(trajectory_type)
                token = _panel_example_cache_trajectory_token(trajectory_type)
                values = np.asarray(data[f"raster_{token}_values"], dtype=float)
                lengths = np.asarray(data[f"raster_{token}_lengths"], dtype=int)
                split_points = np.cumsum(lengths)[:-1]
                raster_positions[trajectory_type] = (
                    [
                        np.asarray(trial_positions, dtype=float)
                        for trial_positions in np.split(values, split_points)
                    ]
                    if lengths.size
                    else []
                )
                firing_rates[trajectory_type] = (
                    np.asarray(data[f"rate_{token}_position"], dtype=float),
                    np.asarray(data[f"rate_{token}_values"], dtype=float),
                )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable Panel example cache at {cache_path}: {exc}")
        return None

    return {
        "animal_name": str(expected_metadata["animal_name"]),
        "date": str(expected_metadata["date"]),
        "epoch": str(expected_metadata["epoch"]),
        "region": str(expected_metadata["region"]),
        "unit_id": int(expected_metadata["unit_id"]),
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }


def load_or_compute_panel_example_data(
    *,
    data_root: Path,
    panel_name: str,
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
    panel_example_cache_dir: Path | None,
    refresh_panel_example_cache: bool,
) -> dict[str, Any]:
    """Load cached Panel A/C example-cell data or compute and cache it."""
    trajectories = validate_trajectories(trajectories, panel_name=panel_name)
    metadata = build_panel_example_cache_metadata(
        data_root=data_root,
        panel_name=panel_name,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        unit_id=unit_id,
        trajectories=trajectories,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    cache_path = (
        build_panel_example_cache_path(panel_example_cache_dir, metadata)
        if panel_example_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_example_cache:
        cached_example = load_panel_example_cache(cache_path, metadata)
        if cached_example is not None:
            print(f"Loaded Panel {metadata['panel']} example cache from {cache_path}.")
            return cached_example

    example_data = load_epoch_unit_rate_curves(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        unit_id=unit_id,
        trajectories=trajectories,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    if cache_path is not None:
        save_panel_example_cache(cache_path, example_data, metadata)
        print(f"Saved Panel {metadata['panel']} example cache to {cache_path}.")
    return example_data


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


def validate_panel_a_trajectories(trajectories: Sequence[str]) -> tuple[str, ...]:
    """Return validated panel-A trajectory names."""
    return validate_trajectories(trajectories, panel_name="A")

def add_segment_boundary_lines(ax: "Axes") -> None:
    """Draw normalized task-progression segment boundaries."""
    for boundary in SEGMENT_BOUNDARIES:
        ax.axvline(
            boundary,
            color=SEGMENT_BOUNDARY_COLOR,
            linewidth=SEGMENT_BOUNDARY_LINEWIDTH,
            zorder=1,
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
    """Load one unit's full-epoch rasters and tuning curves."""
    trajectories = validate_panel_a_trajectories(trajectories)
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
    spike_times_s = get_unit_spike_times(spikes, unit_id)
    bin_edges = build_normalized_position_bins(position_bin_count)
    fallback_position = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    raster_positions: dict[str, list[np.ndarray]] = {}
    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory_type in trajectories:
        task_progression = task_progression_by_trajectory[trajectory_type]
        task_progression_interpolator = make_linear_position_interpolator(task_progression)
        trial_positions = compute_trial_spike_positions(
            spike_times_s,
            session["trajectory_intervals"][selected_epoch][trajectory_type],
            task_progression_interpolator,
        )
        raster_positions[trajectory_type] = [
            positions[(positions >= 0.0) & (positions <= 1.0)]
            for positions in trial_positions
        ]

        movement_epochs = session["trajectory_intervals"][selected_epoch][
            trajectory_type
        ].intersect(session["movement_by_run"][selected_epoch])
        tuning_curve = compute_place_tuning_curve(
            spikes,
            task_progression,
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
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }


def load_panel_a_example_data(
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
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> dict[str, Any]:
    """Load one dark-vs-light example unit for Panel A."""
    trajectories = validate_panel_a_trajectories(trajectories)
    dark_epoch_id = get_dark_epoch(animal_name, date, dark_epoch)
    light_epoch_id = get_light_epoch(animal_name, date, light_epoch)
    epoch_rates = {
        "dark": load_or_compute_panel_example_data(
            data_root=data_root,
            panel_name=PANEL_A_EXAMPLE_CACHE_PANEL_NAME,
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
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        ),
        "light": load_or_compute_panel_example_data(
            data_root=data_root,
            panel_name=PANEL_A_EXAMPLE_CACHE_PANEL_NAME,
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
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
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


def _get_panel_a_y_max(example: dict[str, Any]) -> float:
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


def _compute_panel_a_rate_correlation(
    example: dict[str, Any],
    epoch_key: str,
    trajectories: Sequence[str],
) -> float:
    """Return Pearson correlation between the two panel-A FR curves."""
    if len(trajectories) != 2:
        return float("nan")
    rates = []
    for trajectory_type in trajectories:
        _position, rate = example["epoch_rates"][epoch_key]["firing_rates"][
            trajectory_type
        ]
        rates.append(np.asarray(rate, dtype=float))
    if rates[0].shape != rates[1].shape:
        return float("nan")
    valid = np.isfinite(rates[0]) & np.isfinite(rates[1])
    if np.sum(valid) < 2:
        return float("nan")
    first = rates[0][valid]
    second = rates[1][valid]
    if np.nanstd(first) <= 0.0 or np.nanstd(second) <= 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _compute_panel_a_rate_overlap_index(
    example: dict[str, Any],
    epoch_key: str,
    trajectories: Sequence[str],
) -> float:
    """Return the sum(min) / sum(max) index between two rate curves."""
    if len(trajectories) != 2:
        return float("nan")
    rates = []
    for trajectory_type in trajectories:
        _position, rate = example["epoch_rates"][epoch_key]["firing_rates"][
            trajectory_type
        ]
        rates.append(np.asarray(rate, dtype=float))
    if rates[0].shape != rates[1].shape:
        return float("nan")
    valid = np.isfinite(rates[0]) & np.isfinite(rates[1])
    if np.sum(valid) < 2:
        return float("nan")
    first = np.clip(rates[0][valid], 0.0, None)
    second = np.clip(rates[1][valid], 0.0, None)
    denominator = float(np.sum(np.maximum(first, second)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(np.minimum(first, second)) / denominator)


def _format_panel_a_similarity_annotation(
    example: dict[str, Any],
    epoch_key: str,
    trajectories: Sequence[str],
    annotation: str,
) -> str:
    """Return the requested compact similarity annotation for one epoch."""
    if annotation == "correlation":
        value = _compute_panel_a_rate_correlation(example, epoch_key, trajectories)
        return f"r={value:.2f}" if np.isfinite(value) else "r=n/a"
    if annotation == "dppi":
        value = _compute_panel_a_rate_overlap_index(example, epoch_key, trajectories)
        return f"DPPI={value:.2f}" if np.isfinite(value) else "DPPI=n/a"
    raise ValueError("annotation must be 'correlation' or 'dppi'.")


def plot_epoch_path_rate_axis(
    ax: "Axes",
    example: dict[str, Any],
    epoch_key: str,
    *,
    y_max: float,
    trajectories: Sequence[str] | None = None,
    show_ylabel: bool = False,
    show_legend: bool = False,
    show_title: bool = True,
    show_correlation: bool = True,
    similarity_annotation: str = "correlation",
    correlation_text_position: tuple[float, float] = (0.96, 0.92),
    correlation_text_ha: str = "right",
) -> None:
    """Plot selected path-type tuning curves for one epoch."""
    trajectories = (
        validate_panel_a_trajectories(example["trajectories"])
        if trajectories is None
        else validate_panel_a_trajectories(trajectories)
    )
    for trajectory_type in trajectories:
        position, rate = example["epoch_rates"][epoch_key]["firing_rates"][trajectory_type]
        ax.plot(
            position,
            rate,
            color=PANEL_TRAJECTORY_COLORS[trajectory_type],
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
    ax.set_xlabel(TASK_PROGRESSION_XLABEL, fontsize=4.8, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.8, labelpad=1)
    if show_title:
        ax.set_title(PANEL_A_EPOCH_LABELS[epoch_key], fontsize=5.3, pad=1)
    if show_legend:
        ax.legend(frameon=False, fontsize=4.2, handlelength=1.1, borderpad=0.1)
    if show_correlation:
        label = _format_panel_a_similarity_annotation(
            example,
            epoch_key,
            trajectories,
            similarity_annotation,
        )
        ax.text(
            correlation_text_position[0],
            correlation_text_position[1],
            label,
            ha=correlation_text_ha,
            va="top",
            fontsize=4.6,
            transform=ax.transAxes,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def plot_panel_a_raster_axis(
    ax: "Axes",
    example: dict[str, Any],
    epoch_key: str,
    *,
    trajectories: Sequence[str] | None = None,
    show_ylabel: bool = False,
    show_title: bool = False,
) -> None:
    """Plot selected trajectory spike rasters for one Panel A epoch."""
    trajectories = (
        validate_panel_a_trajectories(example["trajectories"])
        if trajectories is None
        else validate_panel_a_trajectories(trajectories)
    )
    raster_positions = example["epoch_rates"][epoch_key]["raster_positions"]
    row_index = 1
    for trajectory_type in trajectories:
        color = PANEL_TRAJECTORY_COLORS[trajectory_type]
        for positions in raster_positions[trajectory_type]:
            positions = np.asarray(positions, dtype=float)
            if positions.size:
                ax.plot(
                    positions,
                    np.full(positions.shape, row_index, dtype=float),
                    "|",
                    color=color,
                    **RASTER_TICK_KWARGS,
                )
            row_index += 1
        row_index += 1

    add_segment_boundary_lines(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(1, row_index))
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks([])
    if show_ylabel:
        ax.set_ylabel("Trials", fontsize=4.8, labelpad=1)
        ax.yaxis.set_label_coords(-0.32, 0.5)
    if show_title:
        ax.set_title(PANEL_A_EPOCH_LABELS[epoch_key], fontsize=5.3, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=1.0, width=0.35, pad=1)


def _panel_a_raster_section_centers(
    example: dict[str, Any],
    trajectories: Sequence[str],
    *,
    epoch_key: str = "dark",
) -> dict[str, float]:
    """Return normalized raster-axis centers for each trajectory section."""
    epoch_rates = example["epoch_rates"]
    if epoch_key not in epoch_rates:
        epoch_key = next(iter(epoch_rates))
    raster_positions = epoch_rates[epoch_key]["raster_positions"]

    section_centers: dict[str, float] = {}
    row_index = 1
    for trajectory_type in trajectories:
        n_trials = len(raster_positions[trajectory_type])
        section_centers[trajectory_type] = row_index + max(n_trials - 1, 0) / 2.0
        row_index += n_trials
        row_index += 1

    y_limit = float(max(1, row_index))
    return {
        trajectory_type: section_center / y_limit
        for trajectory_type, section_center in section_centers.items()
    }


def plot_panel_a_example(
    ax: "Axes",
    example: dict[str, Any],
    *,
    title: str | None = None,
    y_shift: float = 0.0,
    y_max: float | None = None,
    dark_epoch_axis_left: float = 0.10,
    light_epoch_axis_left: float = 0.56,
    epoch_axis_width: float = 0.40,
    schematic_axis_left: float = 0.012,
    schematic_axis_width: float = 0.070,
    schematic_axis_height: float = 0.075,
    schematic_track_linewidth: float = 0.45,
    schematic_trajectory_linewidth: float = 0.65,
    show_correlation: bool = False,
    similarity_annotation: str = "correlation",
    correlation_text_position: tuple[float, float] = (0.96, 0.92),
    correlation_text_ha: str = "right",
) -> None:
    """Plot one Panel A example cell with dark and light rate curves."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if title is not None:
        ax.text(
            0.50,
            0.885 + y_shift,
            title,
            ha="center",
            va="top",
            fontsize=5.8,
            transform=ax.transAxes,
        )

    trajectories = validate_panel_a_trajectories(example["trajectories"])
    y_max = _get_panel_a_y_max(example) if y_max is None else float(y_max)
    raster_y = PANEL_A_EXAMPLE_RASTER_Y + y_shift
    raster_height = PANEL_A_EXAMPLE_RASTER_HEIGHT
    section_centers = _panel_a_raster_section_centers(example, trajectories)
    for trajectory_type in reversed(trajectories):
        schematic_y = (
            raster_y
            + raster_height * section_centers[trajectory_type]
            - schematic_axis_height / 2.0
        )
        schematic_ax = ax.inset_axes(
            [
                schematic_axis_left,
                schematic_y,
                schematic_axis_width,
                schematic_axis_height,
            ]
        )
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=schematic_track_linewidth,
            trajectory_linewidth=schematic_trajectory_linewidth,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )
    dark_raster_ax = ax.inset_axes(
        [dark_epoch_axis_left, raster_y, epoch_axis_width, raster_height]
    )
    light_raster_ax = ax.inset_axes(
        [light_epoch_axis_left, raster_y, epoch_axis_width, raster_height]
    )
    dark_raster_ax.set_facecolor(PANEL_A_DARK_EPOCH_BACKGROUND)
    plot_panel_a_raster_axis(
        dark_raster_ax,
        example,
        "dark",
        trajectories=trajectories,
        show_ylabel=True,
        show_title=True,
    )
    plot_panel_a_raster_axis(
        light_raster_ax,
        example,
        "light",
        trajectories=trajectories,
        show_title=True,
    )

    dark_ax = ax.inset_axes(
        [
            dark_epoch_axis_left,
            PANEL_A_EXAMPLE_RATE_Y + y_shift,
            epoch_axis_width,
            PANEL_A_EXAMPLE_RATE_HEIGHT,
        ]
    )
    light_ax = ax.inset_axes(
        [
            light_epoch_axis_left,
            PANEL_A_EXAMPLE_RATE_Y + y_shift,
            epoch_axis_width,
            PANEL_A_EXAMPLE_RATE_HEIGHT,
        ]
    )
    dark_ax.set_facecolor(PANEL_A_DARK_EPOCH_BACKGROUND)
    plot_epoch_path_rate_axis(
        dark_ax,
        example,
        "dark",
        trajectories=trajectories,
        y_max=y_max,
        show_ylabel=True,
        show_title=False,
        show_correlation=show_correlation,
        similarity_annotation=similarity_annotation,
        correlation_text_position=correlation_text_position,
        correlation_text_ha=correlation_text_ha,
    )
    plot_epoch_path_rate_axis(
        light_ax,
        example,
        "light",
        trajectories=trajectories,
        y_max=y_max,
        show_title=False,
        show_correlation=show_correlation,
        similarity_annotation=similarity_annotation,
        correlation_text_position=correlation_text_position,
        correlation_text_ha=correlation_text_ha,
    )


def plot_panel_a_examples(
    ax: "Axes",
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot all Panel A examples stacked in one axis."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center", transform=ax.transAxes)
        return

    block_height = PANEL_A_EXAMPLE_BLOCK_HEIGHT
    y_positions = np.linspace(
        PANEL_A_EXAMPLE_TOP - block_height,
        PANEL_A_EXAMPLE_BOTTOM,
        len(examples),
    )
    for example_index, (y0, example) in enumerate(
        zip(y_positions, examples, strict=False),
        start=1,
    ):
        example_ax = ax.inset_axes([0.0, float(y0), 1.0, block_height])
        plot_panel_a_example(
            example_ax,
            example,
            title=f"Example cell {example_index}",
            y_shift=PANEL_A_FIRST_EXAMPLE_Y_SHIFT if example_index == 1 else 0.0,
        )


def setup_light_heatmap_panel(
    fig: Any,
    grid_spec: Any,
    *,
    regions: Sequence[str],
) -> dict[str, Any]:
    """Create the Panel B light-epoch heatmap axes."""
    n_region_rows = len(regions) * len(PANEL_B_TRAJECTORY_TYPES)
    heatmap_grid = grid_spec.subgridspec(
        nrows=n_region_rows + 1,
        ncols=len(PANEL_B_TRAJECTORY_TYPES) + 1,
        height_ratios=[0.42, *([1.0] * n_region_rows)],
        width_ratios=[0.48, *([1.0] * len(PANEL_B_TRAJECTORY_TYPES))],
    )
    axes = np.asarray(
        [
            [
                fig.add_subplot(heatmap_grid[row, col])
                for col in range(len(PANEL_B_TRAJECTORY_TYPES) + 1)
            ]
            for row in range(n_region_rows + 1)
        ],
        dtype=object,
    )
    corner_axis = axes[0, 0]
    corner_axis.axis("off")
    tuning_schematic_axes = axes[0, 1:]
    order_schematic_axes = axes[1:, 0]
    heatmap_axes = axes[1:, 1:]
    for ax, trajectory_type in zip(
        tuning_schematic_axes,
        PANEL_B_TRAJECTORY_TYPES,
        strict=True,
    ):
        draw_w_track_schematic(
            ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_TRAJECTORY_COLORS[trajectory_type],
            fill_track=False,
        )
    for row_index, ax in enumerate(order_schematic_axes):
        trajectory_type = PANEL_B_TRAJECTORY_TYPES[
            row_index % len(PANEL_B_TRAJECTORY_TYPES)
        ]
        draw_order_schematic(
            ax,
            trajectory_type,
            arrow_color=PANEL_TRAJECTORY_COLORS[trajectory_type],
            fill_track=False,
        )
    return {
        "corner_axis": corner_axis,
        "tuning_schematic_axes": tuning_schematic_axes,
        "order_schematic_axes": order_schematic_axes,
        "heatmap_axes": heatmap_axes,
    }


def shift_axes_horizontally(axes: Sequence["Axes"], dx: float) -> None:
    """Shift axes by a fixed figure-coordinate offset without resizing them."""
    if dx == 0:
        return
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0 + dx, box.y0, box.width, box.height])


def _axis_to_figure_coordinates(
    fig: Any,
    ax: "Axes",
    x: float,
    y: float,
) -> tuple[float, float]:
    """Convert one axes-relative coordinate to figure coordinates."""
    figure_x, figure_y = fig.transFigure.inverted().transform(
        ax.transAxes.transform((x, y))
    )
    return float(figure_x), float(figure_y)


def _axis_group_center_x(axes: Sequence["Axes"]) -> float:
    """Return the center x-coordinate of an axes group in figure coordinates."""
    boxes = [ax.get_position() for ax in axes]
    return float((min(box.x0 for box in boxes) + max(box.x1 for box in boxes)) / 2.0)


def _axis_group_top_y(axes: Sequence["Axes"]) -> float:
    """Return the top y-coordinate of an axes group in figure coordinates."""
    boxes = [ax.get_position() for ax in axes]
    return float(max(box.y1 for box in boxes))


def _add_panel_label_at_figure_y(
    fig: Any,
    ax: "Axes",
    label: str,
    *,
    x: float,
    y: float,
) -> "Text":
    """Add a panel label using an axes-relative x and figure-level y."""
    figure_x, _figure_y = _axis_to_figure_coordinates(fig, ax, x, 0.0)
    text_kwargs = PANEL_LABEL_KWARGS.copy()
    text_kwargs["va"] = "center"
    return fig.text(figure_x, y, label, **text_kwargs)


def _add_centered_axis_group_text_at_y(
    fig: Any,
    axes: Sequence["Axes"],
    text: str,
    *,
    y: float,
    fontsize: float,
) -> "Text":
    """Add text centered over an axes group at a fixed figure-level y."""
    return fig.text(
        _axis_group_center_x(axes),
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def add_panel_b_path_progression_label(fig: Any, heatmap_axes: np.ndarray) -> "Text":
    """Add the shared Panel B normalized path-progression x-axis label."""
    return add_centered_below_axis_text(
        fig,
        heatmap_axes[-1, :],
        TASK_PROGRESSION_XLABEL,
        y_offset=HEATMAP_PATH_LABEL_OFFSET,
        fontsize=PANEL_E_AXIS_LABEL_FONTSIZE,
    )


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
    panel_b_cache_dir: Path | None = None,
    refresh_panel_b_cache: bool = False,
) -> "AxesImage | None":
    """Plot pooled light-epoch heatmaps for all requested regions."""
    color_image = None
    for region_index, region in enumerate(regions):
        panels = load_or_compute_panel_b_heatmap_panels(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_b_cache_dir=panel_b_cache_dir,
            refresh_panel_b_cache=refresh_panel_b_cache,
        )
        start_row = region_index * len(PANEL_B_TRAJECTORY_TYPES)
        stop_row = start_row + len(PANEL_B_TRAJECTORY_TYPES)
        image = plot_pooled_heatmap_grid(
            heatmap_axes[start_row:stop_row, :],
            panels,
            trajectory_types=PANEL_B_TRAJECTORY_TYPES,
            axis_orientation=PANEL_B_LINEAR_POSITION_ORIENTATION,
            cmap=PANEL_B_HEATMAP_CMAP,
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
    """Return missing C/D/E artifact records before quantitative plotting."""
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
    """Raise a concise error listing missing dark/light panel artifacts."""
    if not missing:
        return
    lines = [
        "Missing required dark/light panel artifact(s). Run the listed analysis "
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


def load_panel_d_stable_units(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_tuning_stability_correlation: float | None = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> np.ndarray | None:
    """Return units with odd/even tuning stability above threshold in any trajectory."""
    if min_tuning_stability_correlation is None:
        return None
    if min_tuning_stability_correlation < -1.0:
        raise ValueError("min_tuning_stability_correlation must be at least -1.")

    import pandas as pd

    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = read_parquet_table(table_path)
    _require_columns(
        table,
        table_path,
        ("unit", "region", "epoch", "trajectory_type", "stability_correlation"),
    )
    correlations = np.asarray(table["stability_correlation"], dtype=float)
    stable_rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & (table["trajectory_type"].astype(str).isin(TRAJECTORY_TYPES))
        & np.isfinite(correlations)
        & (correlations > float(min_tuning_stability_correlation))
    ]
    stable_units = pd.to_numeric(stable_rows["unit"], errors="coerce")
    stable_units = stable_units[np.isfinite(stable_units.to_numpy(dtype=float))]
    return stable_units.astype(int).drop_duplicates().to_numpy(dtype=int)


def load_panel_c_similarity_table(
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
                & (table["comparison_label"].astype(str).isin(PANEL_C_SIMILARITY_COMPARISON_LABELS))
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


def load_panel_d_encoding_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    n_folds: int = PANEL_D_ENCODING_N_FOLDS,
    place_bin_size_cm: float = PANEL_D_PLACE_BIN_SIZE_CM,
    min_tuning_stability_correlation: float | None = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
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
            _require_columns(table, path, (PANEL_D_ENCODING_DELTA_COLUMN, "n_spikes"))
            stable_units = load_panel_d_stable_units(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                min_tuning_stability_correlation=min_tuning_stability_correlation,
            )
            if stable_units is not None:
                stable_unit_set = {int(unit) for unit in np.asarray(stable_units)}
                candidate_units = pd.to_numeric(
                    table.index.to_numpy(),
                    errors="coerce",
                )
                keep_mask = np.asarray(
                    [
                        np.isfinite(unit_id) and int(unit_id) in stable_unit_set
                        for unit_id in candidate_units
                    ],
                    dtype=bool,
                )
                table = table.iloc[keep_mask]
            unit_ids = pd.to_numeric(table.index.to_numpy(), errors="coerce")
            values = -pd.to_numeric(table[PANEL_D_ENCODING_DELTA_COLUMN], errors="coerce").to_numpy(
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


def _load_decoding_tsd(path: Path) -> Any:
    """Load one pynapple-backed decoding `.npz` artifact."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing decoding-comparison time-series artifact: {path}")
    import pynapple as nap

    return nap.load_file(path)


def _load_absolute_normalized_decoding_errors(
    true_path: Path,
    decoded_path: Path,
    *,
    normalization: float,
) -> np.ndarray:
    """Return finite absolute decoding errors normalized by a coordinate length."""
    from v1ca1.task_progression.decoding_comparison import align_true_to_decoded

    true_tsd = _load_decoding_tsd(true_path)
    decoded_tsd = _load_decoding_tsd(decoded_path)
    true_values, decoded_values = align_true_to_decoded(true_tsd, decoded_tsd)
    if normalization <= 0.0:
        raise ValueError(f"normalization must be positive, got {normalization!r}.")
    errors = np.abs(decoded_values - true_values) / float(normalization)
    return errors[np.isfinite(errors)]


def _summarize_panel_e_errors(
    values: np.ndarray,
    *,
    animal_name: str,
    date: str,
    epoch_type: str,
    epoch: str,
    analysis: str,
    comparison: str,
    comparison_label: str,
) -> dict[str, Any] | None:
    """Return one Panel E median/IQR error row."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75]).astype(float)
    return {
        "animal_name": animal_name,
        "date": date,
        "epoch_type": epoch_type,
        "epoch": epoch,
        "analysis": analysis,
        "comparison": comparison,
        "comparison_label": comparison_label,
        "q25_error": float(q25),
        "median_error": float(median),
        "q75_error": float(q75),
        "n_samples": int(values.size),
    }


def load_panel_e_decoding_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = PANEL_E_CROSS_COMPARISONS,
) -> Any:
    """Load pooled normalized decoding-error medians and IQRs for Panel E."""
    import pandas as pd

    pooled_place_values: dict[str, list[np.ndarray]] = {}
    pooled_cross_values: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        place_normalization = get_wtrack_total_length(animal_name)
        for epoch_type, epoch in build_panel_quant_epoch_specs(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ):
            true_place_path, decoded_place_path = get_within_epoch_decoding_tsd_paths(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                model_name=PANEL_E_PLACE_MODEL_NAME,
            )
            place_values = _load_absolute_normalized_decoding_errors(
                true_place_path,
                decoded_place_path,
                normalization=place_normalization,
            )
            if place_values.size > 0:
                pooled_place_values.setdefault(epoch_type, []).append(place_values)

            for comparison, label, transfer_family, trajectory_pairs in comparisons:
                comparison_values = []
                for encoding_trajectory, decoding_trajectory in trajectory_pairs:
                    true_cross_path, decoded_cross_path = (
                        get_cross_trajectory_decoding_tsd_paths(
                            data_root,
                            animal_name=animal_name,
                            date=date,
                            region=region,
                            epoch=epoch,
                            transfer_family=transfer_family,
                            encoding_trajectory=encoding_trajectory,
                            decoding_trajectory=decoding_trajectory,
                        )
                    )
                    comparison_values.append(
                        _load_absolute_normalized_decoding_errors(
                            true_cross_path,
                            decoded_cross_path,
                            normalization=1.0,
                        )
                    )
                finite_values = [
                    values for values in comparison_values if values.size > 0
                ]
                if not finite_values:
                    continue
                pooled_cross_values.setdefault(
                    (epoch_type, comparison, label),
                    [],
                ).append(np.concatenate(finite_values))

    rows: list[dict[str, Any]] = []
    for epoch_type, values in pooled_place_values.items():
        row = _summarize_panel_e_errors(
            np.concatenate(values),
            animal_name=PANEL_E_POOLED_LABEL,
            date=PANEL_E_POOLED_LABEL,
            epoch_type=epoch_type,
            epoch=PANEL_E_POOLED_LABEL,
            analysis="place",
            comparison="place",
            comparison_label="Place",
        )
        if row is not None:
            rows.append(row)

    for (
        epoch_type,
        comparison,
        comparison_label,
    ), values in pooled_cross_values.items():
        row = _summarize_panel_e_errors(
            np.concatenate(values),
            animal_name=PANEL_E_POOLED_LABEL,
            date=PANEL_E_POOLED_LABEL,
            epoch_type=epoch_type,
            epoch=PANEL_E_POOLED_LABEL,
            analysis="cross_trajectory",
            comparison=comparison,
            comparison_label=comparison_label,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=PANEL_E_ERROR_SUMMARY_COLUMNS)
    return pd.DataFrame(rows)


def load_panel_quantification_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    encoding_n_folds: int = PANEL_D_ENCODING_N_FOLDS,
    place_bin_size_cm: float = PANEL_D_PLACE_BIN_SIZE_CM,
) -> dict[str, Any]:
    """Load the saved-artifact payload for panels C, D, and E."""
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
        "similarity": load_panel_c_similarity_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ),
        "encoding_delta": load_panel_d_encoding_delta_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            n_folds=encoding_n_folds,
            place_bin_size_cm=place_bin_size_cm,
        ),
        "decoding_error": load_panel_e_decoding_error_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ),
    }


def _panel_model_label(
    model_name: str,
    model_labels: Mapping[str, str] | None = None,
) -> str:
    """Return the displayed label for one GLM comparison model."""
    if model_labels is not None and str(model_name) in model_labels:
        return str(model_labels[str(model_name)])
    return GLM_MODEL_LABELS.get(str(model_name), str(model_name))


def _panel_model_color(
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
) -> str:
    """Return the plotted color for one GLM comparison model."""
    if model_colors is not None and str(model_name) in model_colors:
        return str(model_colors[str(model_name)])
    return GLM_MODEL_COLORS.get(str(model_name), "0.2")


def _ensure_panel_h_model(dataset_obj: Any, model_name: str, source_path: Path) -> None:
    """Raise a focused error if one Panel H comparison model is unavailable."""
    available = [str(value) for value in dataset_obj.coords["model"].values]
    if str(model_name) in available:
        return
    raise KeyError(
        f"{source_path} is missing model {model_name!r}. "
        f"Available models: {', '.join(available)}"
    )


def _load_panel_g_selected_dataset(path: Path) -> Any:
    """Load one selected dark/light GLM NetCDF dataset."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dark/light GLM selected output not found: {path}")
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def _normalize_panel_g_empirical_position(tuning_curve: Any) -> np.ndarray:
    """Return normalized task progression positions for one empirical curve."""
    position = np.asarray(tuning_curve.coords["linpos"].values, dtype=float)
    try:
        bin_edges = json.loads(str(tuning_curve.attrs["bin_edges"]))
        max_edge = float(np.asarray(bin_edges, dtype=float).reshape(-1)[-1])
        if np.isfinite(max_edge) and max_edge > 0.0:
            return position / max_edge
    except Exception:
        pass

    finite = position[np.isfinite(position)]
    if finite.size < 2:
        return np.zeros_like(position, dtype=float)
    span = float(finite[-1] - finite[0])
    if span <= 0.0:
        return np.zeros_like(position, dtype=float)
    return (position - float(finite[0])) / span


def _load_panel_g_empirical_curve(
    path: Path,
    *,
    unit_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one empirical trajectory field for Panel G."""
    if not Path(path).exists():
        return None
    import xarray as xr

    with xr.open_dataarray(path) as tuning_curve:
        if int(unit_id) not in set(np.asarray(tuning_curve.coords["unit"].values, dtype=int)):
            return None
        unit_curve = tuning_curve.sel(unit=int(unit_id)).load()
    return (
        _normalize_panel_g_empirical_position(unit_curve),
        np.asarray(unit_curve.values, dtype=float),
    )


def _add_panel_g_empirical_curves(
    candidate: dict[str, Any],
    *,
    data_root: Path,
) -> dict[str, Any] | None:
    """Return a candidate enriched with empirical dark/light fields, if available."""
    empirical: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for epoch_key, epoch in (
        ("dark", candidate["dark_epoch"]),
        ("light", candidate["light_epoch"]),
    ):
        path = get_compute_tuning_curve_path(
            data_root,
            animal_name=str(candidate["animal_name"]),
            date=str(candidate["date"]),
            region=str(candidate["region"]),
            epoch=str(epoch),
            trajectory=str(candidate["trajectory"]),
        )
        curve = _load_panel_g_empirical_curve(path, unit_id=int(candidate["unit_id"]))
        if curve is None:
            return None
        empirical[epoch_key] = curve

    enriched = dict(candidate)
    enriched["empirical"] = empirical
    return enriched


def _panel_g_candidate_examples_from_pair(
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    visual_dataset: Any,
    comparison_dataset: Any,
    visual_path: Path,
    comparison_path: Path,
    comparison_model_name: str = PANEL_G_COMPARISON_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Return scored example candidates from one visual/comparison selected pair."""
    comparison_model_name = str(comparison_model_name)
    trajectories = [str(value) for value in visual_dataset.coords["trajectory"].values]
    units = np.asarray(visual_dataset.coords["unit"].values)
    tp_grid = np.asarray(visual_dataset.coords["tp_grid"].values, dtype=float)
    segment_edges = np.asarray(visual_dataset.coords["segment_edge"].values, dtype=float)
    visual_score = np.asarray(
        visual_dataset["ll_bits_per_spike_cv_light"].values,
        dtype=float,
    )
    comparison_score = np.asarray(
        comparison_dataset["ll_bits_per_spike_cv_light"].values,
        dtype=float,
    )
    combined_score = np.minimum(visual_score, comparison_score)

    candidates: list[dict[str, Any]] = []
    for trajectory_index, trajectory_type in enumerate(trajectories):
        for unit_index, unit_id in enumerate(units):
            score = float(combined_score[trajectory_index, unit_index])
            if not np.isfinite(score):
                continue
            candidates.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "region": region,
                    "light_epoch": light_epoch,
                    "dark_epoch": dark_epoch,
                    "trajectory": trajectory_type,
                    "unit_id": int(unit_id),
                    "score": score,
                    "tp_grid": tp_grid,
                    "segment_edges": segment_edges,
                    "models": {
                        "visual": {
                            "dark_hz": np.asarray(
                                visual_dataset["dark_hz_grid"].isel(
                                    trajectory=trajectory_index,
                                    unit=unit_index,
                                ).values,
                                dtype=float,
                            ),
                            "light_hz": np.asarray(
                                visual_dataset["light_hz_grid"].isel(
                                    trajectory=trajectory_index,
                                    unit=unit_index,
                                ).values,
                                dtype=float,
                            ),
                            "score": float(visual_score[trajectory_index, unit_index]),
                            "source_path": str(visual_path),
                        },
                        comparison_model_name: {
                            "dark_hz": np.asarray(
                                comparison_dataset["dark_hz_grid"].isel(
                                    trajectory=trajectory_index,
                                    unit=unit_index,
                                ).values,
                                dtype=float,
                            ),
                            "light_hz": np.asarray(
                                comparison_dataset["light_hz_grid"].isel(
                                    trajectory=trajectory_index,
                                    unit=unit_index,
                                ).values,
                                dtype=float,
                            ),
                            "score": float(
                                comparison_score[trajectory_index, unit_index]
                            ),
                            "source_path": str(comparison_path),
                        },
                    },
                }
            )
    return candidates


def _select_requested_panel_g_examples(
    candidates: Sequence[dict[str, Any]],
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    example_count: int,
    requested_examples: Sequence[tuple[str, str, str, int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return configured Panel G examples when they belong to the requested inputs."""
    dataset_keys = {
        normalize_dataset_id(dataset)[:2]
        for dataset in datasets
    }
    example_specs = (
        PANEL_G_EXAMPLES if requested_examples is None else requested_examples
    )
    requested = [
        (animal_name, date, requested_region, int(unit_id), trajectory)
        for animal_name, date, requested_region, unit_id, trajectory in example_specs
        if (animal_name, date) in dataset_keys and requested_region == region
    ][: max(int(example_count), 0)]
    if not requested:
        return []

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for animal_name, date, requested_region, unit_id, trajectory in requested:
        matching = [
            candidate
            for candidate in candidates
            if str(candidate["animal_name"]) == animal_name
            and str(candidate["date"]) == date
            and str(candidate["region"]) == requested_region
            and int(candidate["unit_id"]) == unit_id
            and str(candidate["trajectory"]) == trajectory
        ]
        matching.sort(key=lambda candidate: candidate["score"], reverse=True)
        for candidate in matching:
            enriched_candidate = _add_panel_g_empirical_curves(
                candidate,
                data_root=data_root,
            )
            if enriched_candidate is not None:
                selected.append(enriched_candidate)
                break
        else:
            missing.append(
                f"{animal_name} {requested_region.upper()} cell {unit_id} "
                f"{PANEL_TRAJECTORY_LABELS.get(trajectory, trajectory)}"
            )

    if missing:
        raise ValueError(
            "Configured Panel G dark_light_glm example(s) were not available: "
            + ", ".join(missing)
        )
    return selected


def load_panel_g_dark_light_glm_examples(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    example_count: int = PANEL_G_EXAMPLE_COUNT,
    requested_examples: Sequence[tuple[str, str, str, int, str]] | None = None,
    comparison_model_name: str = PANEL_G_COMPARISON_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Load high-scoring visual and comparison GLM example fits."""
    candidates: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    comparison_model_name = str(comparison_model_name)
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        visual_path = get_dark_light_glm_selected_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=dataset_light_epoch,
            dark_epoch=dataset_dark_epoch,
            model_name="visual",
        )
        comparison_path = get_dark_light_glm_selected_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=dataset_light_epoch,
            dark_epoch=dataset_dark_epoch,
            model_name=comparison_model_name,
        )
        if not visual_path.exists() or not comparison_path.exists():
            missing_paths.extend(
                path for path in (visual_path, comparison_path) if not path.exists()
            )
            continue

        visual_dataset = _load_panel_g_selected_dataset(visual_path)
        comparison_dataset = _load_panel_g_selected_dataset(comparison_path)
        candidates.extend(
            _panel_g_candidate_examples_from_pair(
                animal_name=animal_name,
                date=date,
                region=region,
                light_epoch=dataset_light_epoch,
                dark_epoch=dataset_dark_epoch,
                visual_dataset=visual_dataset,
                comparison_dataset=comparison_dataset,
                visual_path=visual_path,
                comparison_path=comparison_path,
                comparison_model_name=comparison_model_name,
            )
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    requested_examples = _select_requested_panel_g_examples(
        candidates,
        data_root=data_root,
        datasets=datasets,
        region=region,
        example_count=example_count,
        requested_examples=requested_examples,
    )
    if requested_examples:
        return requested_examples

    selected: list[dict[str, Any]] = []
    seen_units: set[tuple[str, str, int]] = set()
    for candidate in candidates:
        enriched_candidate = _add_panel_g_empirical_curves(
            candidate,
            data_root=data_root,
        )
        if enriched_candidate is None:
            continue
        unit_key = (
            str(enriched_candidate["animal_name"]),
            str(enriched_candidate["date"]),
            int(enriched_candidate["unit_id"]),
        )
        if unit_key in seen_units:
            continue
        selected.append(enriched_candidate)
        seen_units.add(unit_key)
        if len(selected) >= int(example_count):
            break

    if selected:
        return selected

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No Panel G dark_light_glm selected artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    raise ValueError("No finite Panel G dark_light_glm examples were found.")


def load_panel_h_swap_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_epoch_pairs: Sequence[tuple[str, str]] = PANEL_H_SWAP_LIGHT_EPOCH_PAIRS,
    min_movement_firing_rate_hz: float | None = None,
    min_tuning_stability_correlation: float | None = None,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
) -> Any:
    """Load model-minus-independent swapped-segment LL values."""
    import pandas as pd
    import xarray as xr

    tables = []
    missing_paths: list[Path] = []
    model_name = str(model_name)
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        stable_units = load_dark_epoch_units_exceeding_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=dataset_dark_epoch,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
        )
        stable_unit_set = (
            {int(unit) for unit in np.asarray(stable_units).reshape(-1)}
            if stable_units is not None
            else None
        )
        for light_train_epoch, light_test_epoch in light_epoch_pairs:
            path = get_swap_glm_selected_comparison_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                dark_epoch=dataset_dark_epoch,
                light_train_epoch=light_train_epoch,
                light_test_epoch=light_test_epoch,
            )
            if not path.exists():
                missing_paths.append(path)
                continue
            with xr.open_dataset(path) as dataset_obj:
                if PANEL_H_SWAP_DELTA_VARIABLE not in dataset_obj:
                    raise KeyError(
                        f"{path} is missing {PANEL_H_SWAP_DELTA_VARIABLE!r}."
                    )
                _ensure_panel_h_model(dataset_obj, model_name, path)
                delta = np.asarray(
                    dataset_obj[PANEL_H_SWAP_DELTA_VARIABLE]
                    .sel(model=model_name)
                    .values,
                    dtype=float,
                )
                trajectories = [str(value) for value in dataset_obj.coords["trajectory"].values]
                units = np.asarray(dataset_obj.coords["unit"].values)
                if stable_unit_set is not None:
                    unit_mask = np.asarray(
                        [int(unit) in stable_unit_set for unit in units],
                        dtype=bool,
                    )
                    delta = delta[:, unit_mask]
                    units = units[unit_mask]

            trajectory_grid, unit_grid = np.meshgrid(
                np.asarray(trajectories, dtype=object),
                units,
                indexing="ij",
            )
            table = pd.DataFrame(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "region": region,
                    "dark_epoch": dataset_dark_epoch,
                    "light_train_epoch": light_train_epoch,
                    "light_test_epoch": light_test_epoch,
                    "model_name": model_name,
                    "trajectory": trajectory_grid.ravel(),
                    "unit": unit_grid.ravel(),
                    "delta_ll_bits_per_spike": delta.ravel(),
                    "source_path": str(path),
                }
            )
            table = table[
                np.isfinite(table["delta_ll_bits_per_spike"].to_numpy(dtype=float))
            ].copy()
            if not table.empty:
                tables.append(table)

    if tables:
        return pd.concat(tables, axis=0, ignore_index=True)

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No Panel H swap_glm_comparison artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    return pd.DataFrame(
        columns=[
            "animal_name",
            "date",
            "region",
            "dark_epoch",
            "light_train_epoch",
            "light_test_epoch",
            "model_name",
            "trajectory",
            "unit",
            "delta_ll_bits_per_spike",
            "source_path",
        ]
    )


def load_dark_epoch_units_exceeding_tuning_stability(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_movement_firing_rate_hz: float | None = None,
    min_tuning_stability_correlation: float | None,
) -> np.ndarray | None:
    """Return dark-epoch units meeting movement-rate and stability thresholds."""
    if (
        min_movement_firing_rate_hz is None
        and min_tuning_stability_correlation is None
    ):
        return None
    if (
        min_tuning_stability_correlation is not None
        and min_tuning_stability_correlation < -1.0
    ):
        raise ValueError("min_tuning_stability_correlation must be at least -1.")

    import pandas as pd

    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(table_path)
    required_columns = [
        "unit",
        "region",
        "epoch",
        "trajectory_type",
        "stability_correlation",
    ]
    if min_movement_firing_rate_hz is not None:
        required_columns.append("firing_rate_hz")
    _require_columns(table, table_path, required_columns)
    table = table[
        (table["epoch"].astype(str) == str(epoch))
        & (table["region"].astype(str) == str(region))
        & (table["trajectory_type"].astype(str).isin(TRAJECTORY_TYPES))
    ].copy()
    active_units = select_units_by_saved_movement_firing_rate(
        table,
        min_movement_firing_rate_hz,
    )
    correlations = np.asarray(table["stability_correlation"], dtype=float)
    keep_mask = table["unit"].isin(active_units) & np.isfinite(correlations)
    if min_tuning_stability_correlation is not None:
        keep_mask &= correlations >= float(min_tuning_stability_correlation)
    stable_rows = table[keep_mask]
    return np.asarray(stable_rows["unit"].drop_duplicates())


def _panel_h_swap_examples_from_dataset(
    dataset_obj: Any,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    source_path: Path,
    example_count: int,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Return the strongest model-advantage switched-segment examples."""
    model_name = str(model_name)
    _ensure_panel_h_model(dataset_obj, model_name, source_path)
    delta = np.asarray(
        dataset_obj[PANEL_H_SWAP_DELTA_VARIABLE].sel(model=model_name).values,
        dtype=float,
    )
    if not np.isfinite(delta).any():
        return []

    tp_grid = np.asarray(dataset_obj.coords["tp_grid"].values, dtype=float)
    observed_position = np.asarray(dataset_obj.coords["tp_observed_bin"].values, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(delta.ravel()))
    ordered_indices = finite_indices[np.argsort(delta.ravel()[finite_indices])[::-1]]
    examples: list[dict[str, Any]] = []
    for flat_index in ordered_indices[: max(int(example_count), 0)]:
        trajectory_index, unit_index = np.unravel_index(flat_index, delta.shape)
        examples.append(
            _panel_h_swap_example_from_indices(
                dataset_obj,
                animal_name=animal_name,
                date=date,
                region=region,
                dark_epoch=dark_epoch,
                light_train_epoch=light_train_epoch,
                light_test_epoch=light_test_epoch,
                source_path=source_path,
                trajectory_index=int(trajectory_index),
                unit_index=int(unit_index),
                tp_grid=tp_grid,
                observed_position=observed_position,
                delta=delta,
                model_name=model_name,
            )
        )
    return examples


def _panel_h_swap_example_from_indices(
    dataset_obj: Any,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    source_path: Path,
    trajectory_index: int,
    unit_index: int,
    tp_grid: np.ndarray | None = None,
    observed_position: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Return one switched-segment example at explicit trajectory/unit indices."""
    model_name = str(model_name)
    _ensure_panel_h_model(dataset_obj, model_name, source_path)
    if tp_grid is None:
        tp_grid = np.asarray(dataset_obj.coords["tp_grid"].values, dtype=float)
    if observed_position is None:
        observed_position = np.asarray(
            dataset_obj.coords["tp_observed_bin"].values,
            dtype=float,
        )
    if delta is None:
        delta = np.asarray(
            dataset_obj[PANEL_H_SWAP_DELTA_VARIABLE]
            .sel(model=model_name)
            .values,
            dtype=float,
        )

    trajectory = str(dataset_obj.coords["trajectory"].values[trajectory_index])
    unit_id = int(np.asarray(dataset_obj.coords["unit"].values)[unit_index])
    segment_start = float(
        dataset_obj["swap_segment_start"].isel(trajectory=trajectory_index).values
    )
    segment_end = float(
        dataset_obj["swap_segment_end"].isel(trajectory=trajectory_index).values
    )
    observed_rate = np.asarray(
        dataset_obj["test_light_observed_rate_hz"].isel(
            trajectory=trajectory_index,
            unit=unit_index,
        ).values,
        dtype=float,
    )
    models = {}
    for plotted_model_name in ("visual", model_name):
        models[plotted_model_name] = np.asarray(
            dataset_obj["test_light_swapped_hz_grid"]
            .sel(model=plotted_model_name)
            .isel(trajectory=trajectory_index, unit=unit_index)
            .values,
            dtype=float,
        )

    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "dark_epoch": dark_epoch,
        "light_train_epoch": light_train_epoch,
        "light_test_epoch": light_test_epoch,
        "model_name": model_name,
        "trajectory": trajectory,
        "unit_id": unit_id,
        "delta_ll_bits_per_spike": float(delta[trajectory_index, unit_index]),
        "segment_start": segment_start,
        "segment_end": segment_end,
        "tp_grid": tp_grid,
        "observed_position": observed_position,
        "observed_rate_hz": observed_rate,
        "models": models,
        "swap_source_trajectory": str(
            dataset_obj["swap_source_trajectory"]
            .isel(trajectory=trajectory_index)
            .values
        ),
        "source_path": str(source_path),
    }


def load_panel_h_swap_example(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = PANEL_H_TRAIN_LIGHT_EPOCH,
    light_test_epoch: str = PANEL_H_HELDOUT_LIGHT_EPOCH,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
) -> dict[str, Any] | None:
    """Load one switched-segment example for Panel H."""
    examples = load_panel_h_swap_examples(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
        model_name=model_name,
        example_count=1,
    )
    return examples[0] if examples else None


def load_panel_h_swap_examples(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = PANEL_H_TRAIN_LIGHT_EPOCH,
    light_test_epoch: str = PANEL_H_HELDOUT_LIGHT_EPOCH,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    example_count: int = 2,
    requested_examples: Sequence[tuple[str, str, str, int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Load top switched-segment examples for Panel H."""
    import xarray as xr

    model_name = str(model_name)
    example_specs = PANEL_H_EXAMPLES if requested_examples is None else requested_examples
    dataset_keys = {normalize_dataset_id(dataset)[:2] for dataset in datasets}
    requested = [
        (animal_name, date, requested_region, int(unit_id), trajectory)
        for animal_name, date, requested_region, unit_id, trajectory in example_specs
        if (animal_name, date) in dataset_keys and requested_region == region
    ][: max(int(example_count), 0)]
    requested_examples: list[dict[str, Any]] = []
    missing_requested: list[str] = []
    candidates: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        path = get_swap_glm_selected_comparison_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        if not path.exists():
            continue
        with xr.open_dataset(path) as dataset_obj:
            for (
                requested_animal,
                requested_date,
                requested_region,
                requested_unit,
                requested_trajectory,
            ) in requested:
                if (
                    requested_animal != animal_name
                    or requested_date != date
                    or requested_region != region
                ):
                    continue
                trajectories = [
                    str(value) for value in dataset_obj.coords["trajectory"].values
                ]
                units = np.asarray(dataset_obj.coords["unit"].values, dtype=int)
                if requested_trajectory not in trajectories or requested_unit not in set(units):
                    missing_requested.append(
                        f"{requested_animal} {requested_region.upper()} "
                        f"{requested_unit} "
                        f"{PANEL_TRAJECTORY_LABELS.get(requested_trajectory, requested_trajectory)}"
                    )
                    continue
                trajectory_index = trajectories.index(requested_trajectory)
                unit_index = int(np.flatnonzero(units == requested_unit)[0])
                requested_example = _panel_h_swap_example_from_indices(
                    dataset_obj,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    dark_epoch=dataset_dark_epoch,
                    light_train_epoch=light_train_epoch,
                    light_test_epoch=light_test_epoch,
                    source_path=path,
                    trajectory_index=trajectory_index,
                    unit_index=unit_index,
                    model_name=model_name,
                )
                if np.isfinite(float(requested_example["delta_ll_bits_per_spike"])):
                    requested_examples.append(requested_example)
                else:
                    missing_requested.append(
                        f"{requested_animal} {requested_region.upper()} "
                        f"{requested_unit} "
                        f"{PANEL_TRAJECTORY_LABELS.get(requested_trajectory, requested_trajectory)}"
                    )
            candidates.extend(
                _panel_h_swap_examples_from_dataset(
                    dataset_obj,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    dark_epoch=dataset_dark_epoch,
                    light_train_epoch=light_train_epoch,
                    light_test_epoch=light_test_epoch,
                    source_path=path,
                    model_name=model_name,
                    example_count=example_count,
                )
            )
    if missing_requested:
        raise ValueError(
            "Configured Panel H swap example(s) were not available: "
            + ", ".join(missing_requested)
        )
    if requested_examples:
        requested_order = {
            (
                animal_name,
                date,
                requested_region,
                int(unit_id),
                trajectory,
            ): requested_index
            for requested_index, (
                animal_name,
                date,
                requested_region,
                unit_id,
                trajectory,
            ) in enumerate(requested)
        }
        requested_examples.sort(
            key=lambda example: requested_order[
                (
                    str(example["animal_name"]),
                    str(example["date"]),
                    str(example["region"]),
                    int(example["unit_id"]),
                    str(example["trajectory"]),
                )
            ]
        )
        return requested_examples[: max(int(example_count), 0)]

    candidates.sort(
        key=lambda example: float(example["delta_ll_bits_per_spike"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for example in candidates:
        example_key = (
            str(example["animal_name"]),
            str(example["date"]),
            str(example["trajectory"]),
            int(example["unit_id"]),
        )
        if example_key in seen:
            continue
        selected.append(example)
        seen.add(example_key)
        if len(selected) >= int(example_count):
            break
    return selected


def load_panel_glm_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    swap_delta_min_movement_firing_rate_hz: float | None = None,
    swap_delta_min_tuning_stability_correlation: float | None = None,
    swap_model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    swap_example_count: int = 2,
    swap_requested_examples: Sequence[tuple[str, str, str, int, str]] | None = None,
    dark_light_requested_examples: Sequence[
        tuple[str, str, str, int, str]
    ] | None = None,
) -> dict[str, Any]:
    """Load saved GLM artifacts for the Figure 2 GLM panels."""
    swap_examples = load_panel_h_swap_examples(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        model_name=swap_model_name,
        example_count=swap_example_count,
        requested_examples=swap_requested_examples,
    )
    return {
        "dark_light_examples": load_panel_g_dark_light_glm_examples(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            example_count=(
                len(dark_light_requested_examples)
                if dark_light_requested_examples is not None
                else PANEL_G_EXAMPLE_COUNT
            ),
            requested_examples=dark_light_requested_examples,
        ),
        "swap_delta": load_panel_h_swap_delta_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            dark_epoch=dark_epoch,
            min_movement_firing_rate_hz=(
                swap_delta_min_movement_firing_rate_hz
            ),
            min_tuning_stability_correlation=(
                swap_delta_min_tuning_stability_correlation
            ),
            model_name=swap_model_name,
        ),
        "swap_examples": swap_examples,
        "swap_example": swap_examples[0] if swap_examples else None,
    }


def _coerce_panel_h_swap_examples(
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return a list of Panel H examples from old or new call signatures."""
    if swap_examples is None:
        return []
    if isinstance(swap_examples, dict):
        return [swap_examples]
    return list(swap_examples)


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


def build_panel_c_similarity_pairs(similarity_table: Any) -> Any:
    """Return paired light/dark correlations for each unit's best dark same-turn pair."""
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
            f"Panel C similarity table is missing columns {missing_columns!r}."
        )

    table = similarity_table.copy()
    table = table[
        table["epoch_type"].astype(str).isin(PANEL_QUANT_EPOCH_ORDER)
        & table["comparison_label"].astype(str).isin(PANEL_C_SIMILARITY_COMPARISON_LABELS)
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
                "comparison_label",
                "similarity_light",
                "similarity_dark",
            ]
        )
    table["unit"] = table["unit"].astype(int)
    table["epoch_type"] = table["epoch_type"].astype(str)
    table["comparison_label"] = table["comparison_label"].astype(str)
    key_columns = ["animal_name", "date", "unit"]
    pair_columns = [*key_columns, "comparison_label"]
    table = (
        table.groupby([*pair_columns, "epoch_type"], as_index=False, observed=False)[
            "similarity"
        ]
        .max()
    )
    light = table[table["epoch_type"].astype(str) == "light"][
        pair_columns + ["similarity"]
    ].rename(columns={"similarity": "similarity_light"})
    dark = table[table["epoch_type"].astype(str) == "dark"][
        pair_columns + ["similarity"]
    ].rename(columns={"similarity": "similarity_dark"})
    pairs = dark.merge(light, on=pair_columns, how="inner")
    pairs = pairs[
        np.isfinite(pairs["similarity_light"].to_numpy(dtype=float))
        & np.isfinite(pairs["similarity_dark"].to_numpy(dtype=float))
    ].copy()
    if pairs.empty:
        return pairs

    comparison_order = {
        comparison_label: index
        for index, comparison_label in enumerate(PANEL_C_SIMILARITY_COMPARISON_LABELS)
    }
    pairs["_comparison_order"] = (
        pairs["comparison_label"].map(comparison_order).fillna(len(comparison_order))
    )
    pairs = (
        pairs.sort_values(
            [*key_columns, "similarity_dark", "_comparison_order"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(key_columns, keep="first")
        .drop(columns="_comparison_order")
        .reset_index(drop=True)
    )
    return pairs


def plot_panel_c_similarity(ax: "Axes", similarity_table: Any) -> None:
    """Plot light-vs-dark tuning similarity for the best dark same-turn pair."""
    paired = build_panel_c_similarity_pairs(similarity_table)
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
            s=PANEL_C_SCATTER_SIZE,
            color=PANEL_QUANT_EPOCH_COLORS["light"],
            alpha=PANEL_C_SCATTER_ALPHA,
            edgecolors="none",
            zorder=2,
        )
        x_values = paired["similarity_light"].to_numpy(dtype=float)
        y_values = paired["similarity_dark"].to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        ax.text(
            0.96,
            0.04,
            f"n={int(np.sum(valid))}",
            ha="right",
            va="bottom",
            fontsize=5.0,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No paired\nsimilarity", ha="center", va="center")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box", anchor="S")
    ax.set_xlabel("Light tuning corr.", fontsize=6.2, labelpad=1.5)
    ax.set_ylabel("Dark tuning corr.", fontsize=6.2, labelpad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def _panel_d_epoch_cell_counts(delta_table: Any) -> tuple[dict[str, int], int]:
    """Return separate light/dark cell counts and a shared animal count for Panel D."""
    columns = set(getattr(delta_table, "columns", []))
    if "delta_bits_tp_vs_place" in columns:
        count_table = delta_table[
            np.isfinite(np.asarray(delta_table["delta_bits_tp_vs_place"], dtype=float))
        ]
    else:
        count_table = delta_table

    if {"animal_name", "date", "unit"}.issubset(columns):
        cell_columns = ["animal_name", "date", "unit"]
        count_cells = lambda table: int(table.loc[:, cell_columns].drop_duplicates().shape[0])
    elif "unit" in columns:
        count_cells = lambda table: int(table["unit"].nunique())
    else:
        count_cells = lambda table: int(len(table))

    cell_counts = {}
    for epoch_type in PANEL_QUANT_EPOCH_ORDER:
        epoch_table = count_table[
            count_table["epoch_type"].astype(str) == epoch_type
        ] if "epoch_type" in columns else count_table
        cell_counts[epoch_type] = count_cells(epoch_table)

    n_animals = (
        int(count_table["animal_name"].nunique()) if "animal_name" in columns else 0
    )
    return cell_counts, n_animals


def _add_panel_d_count_text(ax: "Axes", delta_table: Any) -> None:
    """Draw color-coded Panel D cell counts with a shared animal count."""
    cell_counts, n_animals = _panel_d_epoch_cell_counts(delta_table)
    y_by_epoch = {"light": 0.40, "dark": 0.24}
    for epoch_type in PANEL_QUANT_EPOCH_ORDER:
        n_cells = cell_counts.get(epoch_type, 0)
        cell_word = "cell" if n_cells == 1 else "cells"
        ax.text(
            0.03,
            y_by_epoch.get(epoch_type, 0.145),
            f"{PANEL_QUANT_EPOCH_LABELS[epoch_type]}: n = {n_cells} {cell_word}",
            ha="left",
            va="bottom",
            fontsize=PANEL_QUANT_SUMMARY_TEXT_FONTSIZE,
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            transform=ax.transAxes,
        )

    animal_word = "animal" if n_animals == 1 else "animals"
    ax.text(
        0.03,
        0.08,
        f"{n_animals} {animal_word}",
        ha="left",
        va="bottom",
        fontsize=PANEL_QUANT_SUMMARY_TEXT_FONTSIZE,
        color="0.25",
        transform=ax.transAxes,
    )


def plot_panel_d_encoding_delta_histogram(ax: "Axes", delta_table: Any) -> None:
    """Plot light- and dark-epoch TP minus place encoding delta log-likelihoods."""
    bin_edges = np.linspace(PANEL_D_ENCODING_X_LIMITS[0], PANEL_D_ENCODING_X_LIMITS[1], 27)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.6, zorder=1)
    ax.text(
        0.03,
        0.97,
        "Route-specific\nplace better",
        ha="left",
        va="top",
        fontsize=4.8,
        transform=ax.transAxes,
    )
    ax.text(
        0.67,
        0.97,
        "DPP better",
        ha="left",
        va="top",
        fontsize=4.8,
        transform=ax.transAxes,
    )
    summary_y_by_epoch = {"light": 0.76, "dark": 0.50}
    plotted_any = False
    for epoch_index, epoch_type in enumerate(PANEL_QUANT_EPOCH_ORDER):
        values = _finite_column_values(
            delta_table[delta_table["epoch_type"].astype(str) == epoch_type],
            "delta_bits_tp_vs_place",
        )
        if values.size == 0:
            continue
        plotted_any = True
        hist_kwargs = OUTLINED_HISTOGRAM_KWARGS.copy()
        hist_kwargs["alpha"] = EPOCH_HISTOGRAM_ALPHA.get(
            epoch_type,
            OUTLINED_HISTOGRAM_KWARGS["alpha"],
        )
        hist_kwargs["edgecolor"] = "none"
        hist_kwargs["linewidth"] = 0.0
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            label=PANEL_QUANT_EPOCH_LABELS[epoch_type],
            **hist_kwargs,
            zorder=2 + epoch_index,
        )
        median_value = float(np.nanmedian(values))
        fraction_positive = float(np.mean(values > 0.0))
        summary_text = (
            f"{PANEL_QUANT_EPOCH_LABELS[epoch_type]}: "
            f"{fraction_positive:.0%} >0\nmed. {median_value:.2f}"
        )
        ax.text(
            0.67,
            summary_y_by_epoch.get(epoch_type, 0.76 - 0.26 * epoch_index),
            summary_text,
            ha="left",
            va="top",
            fontsize=PANEL_QUANT_SUMMARY_TEXT_FONTSIZE,
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            transform=ax.transAxes,
        )
    if not plotted_any:
        ax.text(0.5, 0.5, "No encoding\nvalues", ha="center", va="center")
    _add_panel_d_count_text(ax, delta_table)

    ax.set_xlim(*PANEL_D_ENCODING_X_LIMITS)
    ax.set_xlabel(
        "Δ log likelihood (bits/spike)",
        fontsize=5.8,
        labelpad=1.5,
    )
    ax.set_ylabel("Frac.", fontsize=6.2, labelpad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def _set_panel_e_error_ylim(ax: "Axes", table: Any) -> None:
    """Set a normalized-error y-limit that preserves the Figure 1 convention."""
    q75_values = _finite_column_values(table, "q75_error") if table is not None else np.asarray([])
    upper = PANEL_E_NORM_ERROR_YLIM[1]
    if q75_values.size:
        upper = max(upper, float(np.nanmax(q75_values)) * 1.08)
    ax.set_ylim(PANEL_E_NORM_ERROR_YLIM[0], upper)


def _plot_panel_e_interval_point(
    ax: "Axes",
    *,
    x: float,
    q25: float,
    median: float,
    q75: float,
    color: str,
    marker: str = "o",
    size: float = 13,
    linewidth: float = 1.0,
    alpha: float = 0.75,
    label: str | None = None,
) -> None:
    """Plot one median/IQR point, marking values clipped by the current y-limit."""
    y_min, y_max = ax.get_ylim()
    clipped_q25 = float(np.clip(q25, y_min, y_max))
    clipped_q75 = float(np.clip(q75, y_min, y_max))
    clipped_median = float(np.clip(median, y_min, y_max))
    ax.vlines(
        x,
        clipped_q25,
        clipped_q75,
        colors=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=3,
    )
    marker_to_draw = "^" if median > y_max else marker
    ax.scatter(
        [x],
        [clipped_median],
        c=color,
        marker=marker_to_draw,
        s=size,
        edgecolors="black",
        linewidths=0.3,
        label=label,
        zorder=4,
        clip_on=False,
    )
    if q75 > y_max and median <= y_max:
        ax.scatter(
            [x],
            [y_max],
            c=color,
            marker="^",
            s=max(size * 0.65, 7),
            edgecolors="black",
            linewidths=0.25,
            zorder=5,
            clip_on=False,
        )


def _style_panel_e_error_axis(ax: "Axes", ylabel: str | None = "Abs. norm. error") -> None:
    """Apply compact normalized-error axis styling for Panel E."""
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PANEL_E_YLABEL_FONTSIZE, labelpad=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=5.0, length=1.5, pad=1)
    ax.tick_params(axis="x", labelsize=4.7, length=0, pad=1)


def _add_panel_e_error_summary_text(
    ax: "Axes",
    table: Any,
    *,
    x: float = PANEL_E_SUMMARY_TEXT_X,
    horizontal_alignment: str = "right",
) -> None:
    """Add compact median summaries following the decoding-summary style."""
    if table.empty:
        return
    for row_index, epoch_type in enumerate(PANEL_QUANT_EPOCH_ORDER):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ax.text(
            x,
            0.95 - 0.13 * row_index,
            f"{PANEL_QUANT_EPOCH_LABELS[epoch_type]} med. "
            f"{float(row['median_error']):.2f}",
            ha=horizontal_alignment,
            va="top",
            fontsize=PANEL_QUANT_SUMMARY_TEXT_FONTSIZE,
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            transform=ax.transAxes,
        )


def _plot_panel_e_place_axis(
    ax: "Axes",
    decoding_error_table: Any,
    *,
    ylabel: str | None = "Abs. norm. error",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot pooled within-epoch place-decoding median/IQR errors by epoch."""
    table = decoding_error_table[
        decoding_error_table["analysis"].astype(str) == "place"
    ].copy()
    positions = np.arange(1, len(PANEL_QUANT_EPOCH_ORDER) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [PANEL_QUANT_EPOCH_LABELS[epoch_type] for epoch_type in PANEL_QUANT_EPOCH_ORDER]
    )
    ax.set_xlim(0.5, len(PANEL_QUANT_EPOCH_ORDER) + 0.5)
    ax.set_ylim(*(PANEL_E_PLACE_ERROR_YLIM if ylim is None else ylim))
    ax.set_title("Route-specific\nplace decoding", fontsize=5.8, pad=1.5)
    if table.empty:
        ax.text(0.5, 0.5, "No place\ndecoding", ha="center", va="center")
        _style_panel_e_error_axis(ax, ylabel=ylabel)
        return

    for position, epoch_type in zip(positions, PANEL_QUANT_EPOCH_ORDER, strict=True):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if rows.empty:
            continue
        row = rows.iloc[0]
        _plot_panel_e_interval_point(
            ax,
            x=float(position),
            q25=float(row["q25_error"]),
            median=float(row["median_error"]),
            q75=float(row["q75_error"]),
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            marker="o",
        )

    _add_panel_e_error_summary_text(
        ax,
        table,
        x=PANEL_E_PLACE_SUMMARY_TEXT_X,
        horizontal_alignment="left",
    )
    _style_panel_e_error_axis(ax, ylabel=ylabel)


def _plot_panel_e_cross_axis(
    ax: "Axes",
    decoding_error_table: Any,
    *,
    ylabel: str | None = "Abs. norm. error",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot pooled cross-trajectory TP decoding median/IQR errors by epoch."""
    table = decoding_error_table[
        decoding_error_table["analysis"].astype(str) == "cross_trajectory"
    ].copy()
    comparisons = list(PANEL_E_CROSS_COMPARISONS)
    positions = np.arange(1, len(PANEL_QUANT_EPOCH_ORDER) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [PANEL_QUANT_EPOCH_LABELS[epoch_type] for epoch_type in PANEL_QUANT_EPOCH_ORDER]
    )
    ax.set_xlim(0.5, len(PANEL_QUANT_EPOCH_ORDER) + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title("Cross-route\ndecoding", fontsize=5.8, pad=1.5)
    if table.empty:
        ax.text(0.5, 0.5, "No cross-route\ndecoding", ha="center", va="center")
        if ylim is None:
            _set_panel_e_error_ylim(ax, table)
        _style_panel_e_error_axis(ax, ylabel=ylabel)
        return

    comparison = comparisons[0][0] if comparisons else None
    for position, epoch_type in zip(positions, PANEL_QUANT_EPOCH_ORDER, strict=True):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if comparison is not None:
            rows = rows[rows["comparison"].astype(str) == comparison]
        if rows.empty:
            continue
        row = rows.iloc[0]
        _plot_panel_e_interval_point(
            ax,
            x=float(position),
            q25=float(row["q25_error"]),
            median=float(row["median_error"]),
            q75=float(row["q75_error"]),
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            marker="o",
            size=11,
            linewidth=0.85,
            alpha=0.70,
        )

    _add_panel_e_error_summary_text(ax, table)
    if ylim is None:
        _set_panel_e_error_ylim(ax, table)
    _style_panel_e_error_axis(ax, ylabel=ylabel)


def _add_panel_cd_group_title(ax: "Axes", title: str) -> None:
    """Add the conceptual title for one grouped bottom-row panel."""
    ax.text(
        0.5,
        PANEL_CD_GROUP_TITLE_Y,
        title,
        ha="center",
        va="top",
        fontsize=PANEL_CD_GROUP_TITLE_FONTSIZE,
        transform=ax.transAxes,
        clip_on=False,
    )


def _add_panel_cd_label(ax: "Axes", label: str) -> None:
    """Add a bottom-row panel label aligned with its group title."""
    label_axis(ax, label, x=0.00, y=PANEL_CD_GROUP_TITLE_Y, va="top")


def plot_panel_c_vision_tuning_panel(
    ax: "Axes",
    similarity_table: Any,
    decoding_error_table: Any,
) -> None:
    """Plot vision-related DPP tuning changes and cross-route decoding."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    _add_panel_cd_group_title(ax, "Vision changes DPP tuning")
    similarity_ax = ax.inset_axes(PANEL_C_SCATTER_AXIS_BOUNDS)
    cross_route_ax = ax.inset_axes(PANEL_C_CROSS_ROUTE_AXIS_BOUNDS)
    plot_panel_c_similarity(similarity_ax, similarity_table)
    _plot_panel_e_cross_axis(cross_route_ax, decoding_error_table)
    similarity_ax.set_title(
        "Same-turn route\ntuning similarity",
        fontsize=5.8,
        pad=1.5,
    )


def plot_panel_d_route_place_panel(
    ax: "Axes",
    delta_table: Any,
    decoding_error_table: Any,
) -> None:
    """Plot the shift from DPP coding toward route-specific place coding."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    _add_panel_cd_group_title(ax, "Shift toward route-specific place coding")
    encoding_ax = ax.inset_axes(PANEL_D_HISTOGRAM_AXIS_BOUNDS)
    place_ax = ax.inset_axes(PANEL_D_PLACE_DECODING_AXIS_BOUNDS)
    plot_panel_d_encoding_delta_histogram(encoding_ax, delta_table)
    _plot_panel_e_place_axis(place_ax, decoding_error_table)
    encoding_ax.set_title("Encoding comparison", fontsize=5.8, pad=1.5)


def plot_panel_e_decoding_error(ax: "Axes", decoding_error_table: Any) -> None:
    """Plot normalized place and cross-trajectory decoding errors."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    cross_ax = ax.inset_axes(PANEL_E_CROSS_AXIS_BOUNDS)
    place_ax = ax.inset_axes(PANEL_E_PLACE_AXIS_BOUNDS)
    _plot_panel_e_cross_axis(cross_ax, decoding_error_table)
    _plot_panel_e_place_axis(place_ax, decoding_error_table, ylabel=None)


def _panel_g_basis_styles(
    *,
    edge_color: str,
    fill_color: str,
    fill_alpha: float,
    linewidth: float,
) -> list[dict[str, Any]]:
    """Return consistent three-segment basis styles for Panel G schematics."""
    return [
        {
            "edge_color": edge_color,
            "fill_color": fill_color,
            "fill_alpha": fill_alpha,
            "linewidth": linewidth,
            "radius": 0.30,
            "spacing": 0.34,
        }
        for _segment_index in range(3)
    ]


def _panel_basis_styles_with_highlighted_segments(
    highlighted_segments: Sequence[int],
) -> list[dict[str, Any]]:
    """Return basis styles with selected 1-based segments filled orange."""
    highlighted = {int(segment_index) for segment_index in highlighted_segments}
    styles = []
    for segment_index in range(1, 4):
        if segment_index in highlighted:
            styles.append(
                {
                    "edge_color": "black",
                    "fill_color": PANEL_G_BASIS_LIGHT_COLOR,
                    "fill_alpha": 0.76,
                    "linewidth": 0.145,
                    "radius": 0.30,
                    "spacing": 0.34,
                }
            )
        else:
            styles.append(
                {
                    "edge_color": "black",
                    "fill_color": "none",
                    "fill_alpha": 1.0,
                    "linewidth": 0.145,
                    "radius": 0.30,
                    "spacing": 0.34,
                }
            )
    return styles


def _panel_g_oval_styles(count: int, *, fill: bool = True) -> list[dict[str, Any]]:
    """Return orange segment-modulation styles for Panel G track overlays."""
    linewidth = PANEL_G_SHARED_SCAFFOLD_OVAL_LINEWIDTH if fill else 0.90
    return [
        {
            "edge_color": PANEL_G_BASIS_LIGHT_COLOR,
            "fill_color": PANEL_G_BASIS_LIGHT_COLOR if fill else "none",
            "fill_alpha": 0.38 if fill else 1.0,
            "linewidth": linewidth,
        }
        for _index in range(count)
    ]


def _draw_panel_g_basis_icon(ax: "Axes") -> None:
    """Draw a compact independent-basis icon."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    line_kwargs = {"color": "black", "linewidth": 1.05, "solid_capstyle": "butt"}
    vertical_span = (
        PANEL_G_INDEPENDENT_BASIS_ICON_TOP
        - PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM
    )
    horizontal_left = 0.5 - vertical_span / 2.0
    horizontal_right = 0.5 + vertical_span / 2.0
    ax.plot(
        [horizontal_left, horizontal_right],
        [PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM] * 2,
        **line_kwargs,
    )
    ax.plot(
        [PANEL_G_INDEPENDENT_BASIS_ICON_LEFT_X] * 2,
        [
            PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM,
            PANEL_G_INDEPENDENT_BASIS_ICON_TOP,
        ],
        **line_kwargs,
    )
    ax.plot(
        [PANEL_G_INDEPENDENT_BASIS_ICON_RIGHT_X] * 2,
        [
            PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM,
            PANEL_G_INDEPENDENT_BASIS_ICON_TOP,
        ],
        **line_kwargs,
    )


def _draw_panel_g_place_field_blob(
    ax: "Axes",
    *,
    colors: Sequence[str] | None = None,
    size_scale: float = 1.0,
    arm: str = "left",
) -> None:
    """Draw a compact path-aligned place-field heat strip on a W-track schematic."""
    from matplotlib.patches import Ellipse

    selected_colors = tuple(colors or ("#FEE08B", "#FDAE61", "#D73027"))
    selected_size_scale = float(size_scale)
    _outline, _points, dims = get_w_track_geometry()
    arm_centers = {
        "left": (dims["x0"] + dims["x1"]) / 2.0,
        "left_arm": (dims["x0"] + dims["x1"]) / 2.0,
        "center": (dims["x2"] + dims["x3"]) / 2.0,
        "center_arm": (dims["x2"] + dims["x3"]) / 2.0,
        "right": (dims["x4"] + dims["x5"]) / 2.0,
        "right_arm": (dims["x4"] + dims["x5"]) / 2.0,
    }
    center_x = arm_centers[arm]
    field_center_y = dims["y1"] + 1.45
    field_sigma = 0.58
    y_values = np.linspace(dims["y1"] + 0.35, dims["y2"] - 0.28, 8)
    for y in y_values:
        relative_rate = float(np.exp(-0.5 * ((y - field_center_y) / field_sigma) ** 2))
        if relative_rate < 0.06:
            continue
        color = selected_colors[2] if relative_rate > 0.72 else selected_colors[1]
        if relative_rate < 0.32:
            color = selected_colors[0]
        ax.add_patch(
            Ellipse(
                (center_x, y),
                dims["corridor_w"] * 0.96 * selected_size_scale,
                0.72 * selected_size_scale,
                facecolor=color,
                edgecolor="none",
                alpha=0.24 + 0.74 * relative_rate,
                zorder=4.0 + relative_rate,
            )
        )


def _panel_g_arrow_head_geometry(
    tail: tuple[float, float],
    tip: tuple[float, float],
    *,
    head_length: float = PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_LENGTH,
    head_width: float = PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_WIDTH,
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Return triangle vertices and shaft endpoint for a directed arrow head."""
    tail_xy = np.asarray(tail, dtype=float)
    tip_xy = np.asarray(tip, dtype=float)
    direction = tip_xy - tail_xy
    segment_length = float(np.hypot(direction[0], direction[1]))
    if segment_length == 0.0:
        raise ValueError("Cannot draw an arrow head on a zero-length segment.")

    unit = direction / segment_length
    base_center = tip_xy - unit * min(head_length, segment_length)
    perpendicular = np.asarray((-unit[1], unit[0]), dtype=float)
    base_left = base_center + perpendicular * head_width / 2.0
    base_right = base_center - perpendicular * head_width / 2.0
    vertices = np.vstack((tip_xy, base_left, base_right))
    return (
        [(float(x), float(y)) for x, y in vertices],
        (float(base_center[0]), float(base_center[1])),
    )


def _draw_panel_g_place_field_path_arrow(
    ax: "Axes",
    *,
    trajectory_name: str = PANEL_G_PLACE_FIELD_PATH_ARROW_TRAJECTORY,
) -> None:
    """Draw a low-contrast haloed path arrow over a place-field schematic."""
    from matplotlib import patheffects
    from matplotlib.colors import to_rgba
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Polygon

    _outline, points, _dims = get_w_track_geometry()
    path = trajectory_points(trajectory_name, points)
    arrow_head_vertices, shaft_end = _panel_g_arrow_head_geometry(path[-2], path[-1])
    shaft_path = [*path[:-1], shaft_end]
    xs, ys = zip(*shaft_path, strict=True)
    arrow_effects = [
        patheffects.Stroke(
            linewidth=PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_LINEWIDTH,
            foreground=to_rgba(
                PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_COLOR,
                PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_ALPHA,
            ),
        ),
        patheffects.Normal(),
    ]
    (line,) = ax.plot(
        xs,
        ys,
        color=to_rgba(
            PANEL_G_PLACE_FIELD_PATH_ARROW_COLOR,
            PANEL_G_PLACE_FIELD_PATH_ARROW_ALPHA,
        ),
        linewidth=PANEL_G_PLACE_FIELD_PATH_ARROW_LINEWIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=PANEL_G_PLACE_FIELD_PATH_ARROW_ZORDER,
        label="_place_field_path_arrow",
    )
    line.set_path_effects(arrow_effects)
    ax.add_patch(
        PathPatch(
            Path(
                [
                    arrow_head_vertices[1],
                    arrow_head_vertices[0],
                    arrow_head_vertices[2],
                ],
                [Path.MOVETO, Path.LINETO, Path.LINETO],
            ),
            facecolor="none",
            edgecolor=to_rgba(
                PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_COLOR,
                PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_ALPHA,
            ),
            linewidth=PANEL_G_PLACE_FIELD_PATH_ARROW_HALO_LINEWIDTH,
            capstyle="round",
            joinstyle="round",
            zorder=PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_OUTLINE_ZORDER,
            label="_place_field_path_arrow_head_outline",
        )
    )
    ax.add_patch(
        Polygon(
            arrow_head_vertices,
            closed=True,
            facecolor=to_rgba(
                PANEL_G_PLACE_FIELD_PATH_ARROW_COLOR,
                PANEL_G_PLACE_FIELD_PATH_ARROW_ALPHA,
            ),
            edgecolor="none",
            linewidth=0.0,
            zorder=PANEL_G_PLACE_FIELD_PATH_ARROW_HEAD_ZORDER,
            label="_place_field_path_arrow_head",
        )
    )


def _draw_panel_g_segment_gain_outlines(
    ax: "Axes",
    *,
    outline_colors: Mapping[str, Any],
    outline_linewidths: Mapping[str, float] | None = None,
) -> None:
    """Draw colored segment outlines outside the W-track corridor boundary."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    _outline, _points, dims = get_w_track_geometry()
    region_rectangles = {
        "left_arm": (
            dims["x0"],
            dims["y1"],
            dims["x1"] - dims["x0"],
            dims["y2"] - dims["y1"],
        ),
        "center_arm": (
            dims["x2"],
            dims["y1"],
            dims["x3"] - dims["x2"],
            dims["y2"] - dims["y1"],
        ),
        "right_arm": (
            dims["x4"],
            dims["y1"],
            dims["x5"] - dims["x4"],
            dims["y2"] - dims["y1"],
        ),
        "left_center_connector": (
            dims["x0"],
            dims["y0"],
            dims["x3"] - dims["x0"],
            dims["y1"] - dims["y0"],
        ),
        "center_right_connector": (
            dims["x2"],
            dims["y0"],
            dims["x5"] - dims["x2"],
            dims["y1"] - dims["y0"],
        ),
    }
    outset = PANEL_G_SEGMENT_GAIN_OUTLINE_OUTSET
    for region_name, color in outline_colors.items():
        x, y, width, height = region_rectangles[region_name]
        left = x - outset
        right = x + width + outset
        bottom = y - outset
        top = y + height + outset
        ax.add_patch(
            PathPatch(
                Path(
                    [(left, bottom), (left, top), (right, top), (right, bottom)],
                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO],
                ),
                facecolor="none",
                edgecolor=color,
                linewidth=(
                    outline_linewidths.get(region_name, 1.4)
                    if outline_linewidths is not None
                    else 1.4
                ),
                joinstyle="miter",
                capstyle="butt",
                zorder=6.5,
            )
        )


def _draw_panel_g_track(
    ax: "Axes",
    *,
    track_kind: str,
    show_labels: bool = False,
    trajectory_name: str = "center_to_left",
    stimulus_layout: str = "stim1",
    highlighted_segments: Sequence[int] | None = None,
    oval_regions: Sequence[str] | None = None,
    fill_oval_regions: bool = True,
    label_fontsize: float = 4.8,
    label_colors: Mapping[str, Any] | None = None,
    region_fill_colors: Mapping[str, Any] | None = None,
    region_fill_alpha: float | None = None,
    segment_outline_colors: Mapping[str, Any] | None = None,
    segment_outline_linewidths: Mapping[str, float] | None = None,
    dark_basis_edge_color: str = "black",
    dark_basis_fill_color: str = PANEL_G_BASIS_DARK_COLOR,
    dark_basis_fill_alpha: float = 0.7,
    dark_basis_linewidth: float = 0.25,
    show_place_field_blob: bool = False,
    place_field_colors: Sequence[str] | None = None,
    place_field_blob_size_scale: float = 1.0,
    place_field_arm: str = "left",
    show_place_field_path_arrow: bool = False,
) -> None:
    """Draw one W-track field component for the Panel G model schematic."""
    trajectory_color = PANEL_G_ARROW_COLOR
    if track_kind == "dark":
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            fill_track_black=True,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_color="white",
            label_fontsize=label_fontsize,
            show_trajectory=not show_place_field_blob,
            show_basis=not show_place_field_blob,
            basis_segment_styles=_panel_g_basis_styles(
                edge_color=dark_basis_edge_color,
                fill_color=dark_basis_fill_color,
                fill_alpha=dark_basis_fill_alpha,
                linewidth=dark_basis_linewidth,
            ),
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.85,
            arrow_mutation_scale=6.5,
        )
        if show_place_field_blob:
            _draw_panel_g_place_field_blob(
                ax,
                colors=place_field_colors,
                size_scale=place_field_blob_size_scale,
                arm=place_field_arm,
            )
        if show_place_field_path_arrow:
            _draw_panel_g_place_field_path_arrow(ax)
        _remove_w_track_center_label(ax)
        return

    if track_kind == "independent_light":
        basis_segment_styles = (
            _panel_basis_styles_with_highlighted_segments(highlighted_segments)
            if highlighted_segments is not None
            else _panel_g_basis_styles(
                edge_color="black",
                fill_color=PANEL_G_BASIS_LIGHT_COLOR,
                fill_alpha=0.76,
                linewidth=0.145,
            )
        )
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_colors=label_colors,
            label_fontsize=label_fontsize,
            show_trajectory=not show_place_field_blob,
            show_basis=not show_place_field_blob,
            basis_segment_styles=basis_segment_styles,
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.85,
            arrow_mutation_scale=6.5,
            region_fill_colors=region_fill_colors,
            region_fill_alpha=region_fill_alpha,
        )
        if show_place_field_blob:
            _draw_panel_g_place_field_blob(
                ax,
                colors=place_field_colors,
                size_scale=place_field_blob_size_scale,
                arm=place_field_arm,
            )
        if show_place_field_path_arrow:
            _draw_panel_g_place_field_path_arrow(ax)
        _remove_w_track_center_label(ax)
        return

    if track_kind == "segment_modulation":
        selected_oval_regions = list(
            oval_regions or ["left_arm", "center_arm", "left_center_connector"]
        )
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_colors=label_colors,
            label_fontsize=label_fontsize,
            show_arrow=segment_outline_colors is None,
            show_trajectory=segment_outline_colors is None,
            show_large_ovals=segment_outline_colors is None,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_g_oval_styles(
                len(selected_oval_regions),
                fill=fill_oval_regions,
            ),
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.78,
            arrow_mutation_scale=6.0,
        )
        if segment_outline_colors is not None:
            _draw_panel_g_segment_gain_outlines(
                ax,
                outline_colors=segment_outline_colors,
                outline_linewidths=segment_outline_linewidths,
            )
        _remove_w_track_center_label(ax)
        return

    if track_kind == "shared_light":
        selected_oval_regions = list(
            oval_regions or ["left_arm", "center_arm", "left_center_connector"]
        )
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_colors=label_colors,
            label_fontsize=label_fontsize,
            show_trajectory=not show_place_field_blob,
            show_basis=not show_place_field_blob,
            basis_segment_styles=_panel_g_basis_styles(
                edge_color="black",
                fill_color=PANEL_G_BASIS_DARK_COLOR,
                fill_alpha=0.7,
                linewidth=0.25,
            ),
            show_large_ovals=not show_place_field_blob,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_g_oval_styles(
                len(selected_oval_regions),
                fill=fill_oval_regions,
            ),
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.85,
            arrow_mutation_scale=6.5,
            region_fill_colors=region_fill_colors,
            region_fill_alpha=region_fill_alpha,
        )
        if show_place_field_blob:
            _draw_panel_g_place_field_blob(
                ax,
                colors=place_field_colors,
                size_scale=place_field_blob_size_scale,
                arm=place_field_arm,
            )
        if show_place_field_path_arrow:
            _draw_panel_g_place_field_path_arrow(ax)
        if segment_outline_colors is not None:
            _draw_panel_g_segment_gain_outlines(
                ax,
                outline_colors=segment_outline_colors,
                outline_linewidths=segment_outline_linewidths,
            )
        _remove_w_track_center_label(ax)
        return

    raise ValueError(f"Unknown Panel G track_kind {track_kind!r}.")


def _plot_panel_g_architecture_schematic(
    ax: "Axes",
    *,
    independent_track_center_y: float = PANEL_G_INDEPENDENT_TRACK_CENTER_Y,
    shared_track_center_y: float = PANEL_G_SHARED_TRACK_CENTER_Y,
    track_size: tuple[float, float] | None = None,
    independent_basis_icon_scale: float = 1.0,
    independent_basis_label: str = "Independent\nbasis functions",
    independent_basis_label_y: float = PANEL_G_INDEPENDENT_BASIS_LABEL_Y,
    show_dark_track_labels: bool = False,
    field_label_y: float = PANEL_G_FIELD_LABEL_Y,
    model_label_x: float = 0.08,
    model_label_fontsize: float | None = None,
    shared_model_label: str = "Shared-scaffold\nmodel",
    component_label_fontsize: float = PANEL_G_COMPONENT_LABEL_FONTSIZE,
    segment_modulation_label_y: float | None = None,
    segment_modulation_label: str = "Segment-specific modulation",
    segment_modulation_label_gap: float = PANEL_G_SEGMENT_MODULATION_LABEL_GAP,
    fill_oval_regions: bool = True,
    independent_light_region_fill_colors: Mapping[str, Any] | None = None,
    independent_light_region_fill_alpha: float | None = None,
    independent_light_label_colors: Mapping[str, Any] | None = None,
    shared_light_region_fill_colors: Mapping[str, Any] | None = None,
    shared_light_region_fill_alpha: float | None = None,
    shared_light_label_colors: Mapping[str, Any] | None = None,
    dark_basis_edge_color: str = "black",
    dark_basis_fill_color: str = PANEL_G_BASIS_DARK_COLOR,
    dark_basis_fill_alpha: float = 0.7,
    dark_basis_linewidth: float = 0.25,
    show_place_field_blobs: bool = False,
    independent_dark_place_field_colors: Sequence[str] | None = None,
    independent_light_place_field_colors: Sequence[str] | None = None,
    shared_dark_place_field_colors: Sequence[str] | None = None,
    shared_light_place_field_colors: Sequence[str] | None = None,
    place_field_blob_size_scale: float = 1.0,
    show_place_field_path_arrow: bool = False,
    segment_gain_outline_colors: Mapping[str, Any] | None = None,
    segment_gain_outline_linewidths: Mapping[str, float] | None = None,
    segment_gain_label_colors: Mapping[str, Any] | None = None,
) -> None:
    """Draw the compact dark/light GLM architecture schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.patch.set_visible(False)

    def _bounds_from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> list[float]:
        return [center_x - width / 2.0, center_y - height / 2.0, width, height]

    def _schematic_inset(bounds: list[float]) -> "Axes":
        inset_ax = ax.inset_axes(bounds)
        inset_ax.set_zorder(PANEL_G_SCHEMATIC_INSET_ZORDER)
        inset_ax.patch.set_visible(False)
        return inset_ax

    dark_center_x = PANEL_G_DARK_TRACK_CENTER_X
    light_center_x = PANEL_G_LIGHT_TRACK_CENTER_X
    independent_basis_center_x = 0.5 * (dark_center_x + light_center_x)
    default_dark_bounds = {"width": 0.16, "height": 0.31}
    default_light_bounds = {"width": 0.18, "height": 0.34}
    dark_bounds = (
        {"width": track_size[0], "height": track_size[1]}
        if track_size is not None
        else default_dark_bounds
    )
    light_bounds = (
        {"width": track_size[0], "height": track_size[1]}
        if track_size is not None
        else default_light_bounds
    )
    basis_icon_visual_center_y = 0.5 * (
        PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM
        + PANEL_G_INDEPENDENT_BASIS_ICON_TOP
    )
    basis_icon_width = PANEL_G_INDEPENDENT_BASIS_ICON_WIDTH * float(
        independent_basis_icon_scale
    )
    basis_icon_height = PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT * float(
        independent_basis_icon_scale
    )
    basis_icon_y = (
        independent_track_center_y
        - basis_icon_height * basis_icon_visual_center_y
    )
    selected_segment_modulation_label_y = segment_modulation_label_y or min(
        basis_icon_y - 0.035,
        shared_track_center_y
        + light_bounds["height"] / 2.0
        + segment_modulation_label_gap,
    )
    independent_model_fontsize = 4.1 if model_label_fontsize is None else model_label_fontsize
    shared_model_fontsize = 3.8 if model_label_fontsize is None else model_label_fontsize

    text_kwargs = {"zorder": PANEL_G_SCHEMATIC_TEXT_ZORDER}
    ax.text(
        dark_center_x,
        field_label_y,
        "Dark field",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        light_center_x,
        field_label_y,
        "Light field",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        model_label_x,
        independent_track_center_y,
        "Independent\nmodel",
        ha="center",
        va="center",
        fontsize=independent_model_fontsize,
        fontweight="bold",
        **text_kwargs,
    )
    ax.text(
        model_label_x,
        shared_track_center_y,
        shared_model_label,
        ha="center",
        va="center",
        fontsize=shared_model_fontsize,
        fontweight="bold",
        **text_kwargs,
    )
    ax.text(
        independent_basis_center_x,
        independent_basis_label_y,
        independent_basis_label,
        ha="center",
        va="center",
        fontsize=component_label_fontsize,
        **text_kwargs,
    )
    ax.text(
        PANEL_G_SEGMENT_MODULATION_TRACK_CENTER_X,
        selected_segment_modulation_label_y,
        segment_modulation_label,
        ha="center",
        va="center",
        fontsize=component_label_fontsize,
        **text_kwargs,
    )

    _draw_panel_g_track(
        _schematic_inset(
            _bounds_from_center(
                dark_center_x,
                independent_track_center_y,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
        show_labels=show_dark_track_labels,
        dark_basis_edge_color=dark_basis_edge_color,
        dark_basis_fill_color=dark_basis_fill_color,
        dark_basis_fill_alpha=dark_basis_fill_alpha,
        dark_basis_linewidth=dark_basis_linewidth,
        show_place_field_blob=show_place_field_blobs,
        place_field_colors=independent_dark_place_field_colors,
        place_field_blob_size_scale=place_field_blob_size_scale,
        show_place_field_path_arrow=show_place_field_path_arrow,
    )
    basis_ax = ax.inset_axes(
        [
            independent_basis_center_x - basis_icon_width / 2.0,
            basis_icon_y,
            basis_icon_width,
            basis_icon_height,
        ]
    )
    basis_ax.set_zorder(PANEL_G_SCHEMATIC_INSET_ZORDER)
    basis_ax.patch.set_visible(False)
    _draw_panel_g_basis_icon(basis_ax)
    _draw_panel_g_track(
        _schematic_inset(
            _bounds_from_center(
                light_center_x,
                independent_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="independent_light",
        show_labels=True,
        label_colors=independent_light_label_colors,
        region_fill_colors=independent_light_region_fill_colors,
        region_fill_alpha=independent_light_region_fill_alpha,
        show_place_field_blob=show_place_field_blobs,
        place_field_colors=independent_light_place_field_colors,
        place_field_blob_size_scale=place_field_blob_size_scale,
        show_place_field_path_arrow=show_place_field_path_arrow,
    )

    _draw_panel_g_track(
        _schematic_inset(
            _bounds_from_center(
                dark_center_x,
                shared_track_center_y,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
        show_labels=show_dark_track_labels,
        dark_basis_edge_color=dark_basis_edge_color,
        dark_basis_fill_color=dark_basis_fill_color,
        dark_basis_fill_alpha=dark_basis_fill_alpha,
        dark_basis_linewidth=dark_basis_linewidth,
        show_place_field_blob=show_place_field_blobs,
        place_field_colors=shared_dark_place_field_colors,
        place_field_blob_size_scale=place_field_blob_size_scale,
        show_place_field_path_arrow=show_place_field_path_arrow,
    )
    ax.text(
        0.39,
        shared_track_center_y,
        "+",
        ha="center",
        va="center",
        fontsize=8.0,
        **text_kwargs,
    )
    _draw_panel_g_track(
        _schematic_inset(
            _bounds_from_center(
                PANEL_G_SEGMENT_MODULATION_TRACK_CENTER_X,
                shared_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="segment_modulation",
        show_labels=True,
        label_colors=segment_gain_label_colors,
        fill_oval_regions=fill_oval_regions,
        segment_outline_colors=segment_gain_outline_colors,
        segment_outline_linewidths=segment_gain_outline_linewidths,
    )
    ax.annotate(
        "",
        xy=(PANEL_G_SHARED_OUTPUT_ARROW_X[1], shared_track_center_y + 0.005),
        xytext=(PANEL_G_SHARED_OUTPUT_ARROW_X[0], shared_track_center_y + 0.005),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 0.8,
            "mutation_scale": 8.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=PANEL_G_SCHEMATIC_TEXT_ZORDER,
    )
    _draw_panel_g_track(
        _schematic_inset(
            _bounds_from_center(
                light_center_x,
                shared_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="shared_light",
        show_labels=True,
        label_colors=shared_light_label_colors,
        fill_oval_regions=fill_oval_regions,
        region_fill_colors=shared_light_region_fill_colors,
        region_fill_alpha=shared_light_region_fill_alpha,
        show_place_field_blob=show_place_field_blobs,
        place_field_colors=shared_light_place_field_colors,
        place_field_blob_size_scale=place_field_blob_size_scale,
        show_place_field_path_arrow=show_place_field_path_arrow,
        segment_outline_colors=segment_gain_outline_colors,
        segment_outline_linewidths=segment_gain_outline_linewidths,
    )


def _panel_g_examples_y_max(examples: Sequence[dict[str, Any]]) -> float:
    """Return a shared y-limit for Panel G example field plots."""
    values: list[np.ndarray] = []
    for example in examples:
        for epoch_key in ("dark", "light"):
            if "empirical" in example:
                values.append(np.asarray(example["empirical"][epoch_key][1], dtype=float))
        for model_payload in example.get("models", {}).values():
            values.extend(
                [
                    np.asarray(model_payload["dark_hz"], dtype=float),
                    np.asarray(model_payload["light_hz"], dtype=float),
                ]
            )
    finite_values = [value[np.isfinite(value)] for value in values]
    finite_values = [value for value in finite_values if value.size]
    if not finite_values:
        return 1.0
    return max(1.0, float(np.ceil(np.nanmax(np.concatenate(finite_values)))))


def _plot_panel_g_example_field_axis(
    ax: "Axes",
    example: dict[str, Any],
    *,
    epoch_key: str,
    y_max: float,
    show_ylabel: bool = False,
    show_title: bool = False,
    show_legend: bool = False,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot empirical and fitted fields for one Panel G example epoch."""
    empirical_position, empirical_rate = example["empirical"][epoch_key]
    ax.plot(
        empirical_position,
        empirical_rate,
        color=PANEL_G_EMPIRICAL_COLOR,
        linewidth=0.9,
        label="Empirical",
        zorder=4,
    )
    field_key = f"{epoch_key}_hz"
    for model_name in PANEL_G_MODELS:
        ax.plot(
            example["tp_grid"],
            example["models"][model_name][field_key],
            color=_panel_model_color(model_name, model_colors),
            linewidth=0.75,
            label=_panel_model_label(model_name, model_labels),
            zorder=3,
        )
    for boundary in np.asarray(example["segment_edges"], dtype=float)[1:-1]:
        ax.axvline(boundary, color=SEGMENT_BOUNDARY_COLOR, linewidth=0.35, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    if show_title:
        ax.set_title(PANEL_QUANT_EPOCH_LABELS[epoch_key], fontsize=4.8, pad=0.8)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.4, labelpad=0.8)
    if show_legend:
        legend_kwargs: dict[str, Any] = {
            "frameon": False,
            "fontsize": 3.4,
            "handlelength": 0.9,
            "loc": legend_loc,
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(
            **legend_kwargs,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.1, length=1.1, pad=0.6)


def _plot_panel_g_example_columns(
    ax: "Axes",
    examples: Sequence[dict[str, Any]],
    *,
    field_y: float = 0.05,
    field_height: float = 0.58,
    icon_bounds: tuple[float, float, float, float] = (-0.045, 0.23, 0.085, 0.26),
    xlabel_y: float = -0.145,
    column_width: float = 0.46,
    column_gap: float = 0.04,
    plot_left_offset: float = 0.12,
    field_width: float = 0.14,
    field_gap: float = 0.035,
    layout: str = "columns",
    row_height: float = 0.46,
    row_gap: float = 0.05,
    show_ylabels_for_all_examples: bool = False,
    show_epoch_titles: bool = True,
    show_light_yticklabels: bool = True,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot example cells below or beside the Panel G schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No GLM\nexamples", ha="center", va="center", fontsize=5.0)
        return

    if layout not in {"columns", "rows"}:
        raise ValueError("Panel G example layout must be 'columns' or 'rows'.")

    def _plot_example_block(
        block_ax: "Axes",
        example: dict[str, Any],
        example_index: int,
        *,
        column_left: float,
        show_legend: bool,
        show_ylabel: bool = True,
        show_epoch_titles: bool = True,
        show_light_yticklabels: bool = True,
    ) -> None:
        y_max = _panel_g_examples_y_max([example])
        plot_left = column_left + plot_left_offset
        plot_center = plot_left + field_width + field_gap / 2.0
        icon_ax = block_ax.inset_axes(
            [
                column_left + icon_bounds[0],
                icon_bounds[1],
                icon_bounds[2],
                icon_bounds[3],
            ]
        )
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=example["trajectory"],
            arrow_color=PANEL_TRAJECTORY_COLORS[example["trajectory"]],
            track_linewidth=0.42,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.4,
            fill_track=False,
        )
        block_ax.text(
            plot_center,
            0.985,
            f"Example {example_index}",
            ha="center",
            va="top",
            fontsize=5.6,
            transform=block_ax.transAxes,
        )
        dark_ax = block_ax.inset_axes([plot_left, field_y, field_width, field_height])
        light_ax = block_ax.inset_axes(
            [
                plot_left + field_width + field_gap,
                field_y,
                field_width,
                field_height,
            ]
        )
        dark_ax.set_facecolor(PANEL_A_DARK_EPOCH_BACKGROUND)
        _plot_panel_g_example_field_axis(
            dark_ax,
            example,
            epoch_key="dark",
            y_max=y_max,
            show_ylabel=show_ylabel,
            show_title=show_epoch_titles,
            model_colors=model_colors,
            model_labels=model_labels,
        )
        _plot_panel_g_example_field_axis(
            light_ax,
            example,
            epoch_key="light",
            y_max=y_max,
            show_title=show_epoch_titles,
            show_legend=show_legend,
            legend_loc="center left",
            legend_bbox_to_anchor=(1.02, 0.5),
            model_colors=model_colors,
            model_labels=model_labels,
        )
        if not show_light_yticklabels:
            light_ax.tick_params(axis="y", labelleft=False)
        block_ax.text(
            plot_center,
            xlabel_y,
            TASK_PROGRESSION_XLABEL,
            ha="center",
            va="top",
            fontsize=3.7,
            transform=block_ax.transAxes,
            clip_on=False,
        )

    if layout == "columns":
        for example_index, example in enumerate(examples[:2], start=1):
            column_left = (example_index - 1) * (column_width + column_gap)
            _plot_example_block(
                ax,
                example,
                example_index,
                column_left=column_left,
                show_legend=example_index == 2,
                show_ylabel=True,
                show_epoch_titles=show_epoch_titles,
                show_light_yticklabels=show_light_yticklabels,
            )
        return

    display_examples = list(examples[:4])
    if len(display_examples) <= 2:
        for example_index, example in enumerate(display_examples, start=1):
            row_bottom = (
                1.0 - example_index * row_height - (example_index - 1) * row_gap
            )
            row_ax = ax.inset_axes([0.0, row_bottom, 1.0, row_height])
            row_ax.set_xlim(0.0, 1.0)
            row_ax.set_ylim(0.0, 1.0)
            row_ax.axis("off")
            _plot_example_block(
                row_ax,
                example,
                example_index,
                column_left=0.0,
                show_legend=example_index == 2,
                show_ylabel=True,
                show_epoch_titles=show_epoch_titles,
                show_light_yticklabels=show_light_yticklabels,
            )
        return

    column_count = 2
    row_count = (len(display_examples) + column_count - 1) // column_count
    cell_width = (1.0 - column_gap * (column_count - 1)) / column_count
    cell_height = (1.0 - row_gap * (row_count - 1)) / row_count
    for example_index, example in enumerate(display_examples, start=1):
        row_index = (example_index - 1) // column_count
        column_index = (example_index - 1) % column_count
        left = column_index * (cell_width + column_gap)
        bottom = 1.0 - (row_index + 1) * cell_height - row_index * row_gap
        cell_ax = ax.inset_axes([left, bottom, cell_width, cell_height])
        cell_ax.set_xlim(0.0, 1.0)
        cell_ax.set_ylim(0.0, 1.0)
        cell_ax.axis("off")
        _plot_example_block(
            cell_ax,
            example,
            example_index,
            column_left=0.0,
            show_legend=example_index == 2,
            show_ylabel=show_ylabels_for_all_examples or column_index == 0,
            show_epoch_titles=show_epoch_titles,
            show_light_yticklabels=show_light_yticklabels,
        )


def plot_panel_g_model_architecture(
    ax: "Axes",
    examples: Sequence[dict[str, Any]] | None = None,
    *,
    independent_track_center_y: float = PANEL_G_INDEPENDENT_TRACK_CENTER_Y,
    shared_track_center_y: float = PANEL_G_SHARED_TRACK_CENTER_Y,
    schematic_height_fraction: float = PANEL_G_SCHEMATIC_HEIGHT_FRACTION,
    schematic_track_size: tuple[float, float] | None = None,
    independent_basis_icon_scale: float = 1.0,
    independent_basis_label: str = "Independent\nbasis functions",
    show_dark_track_labels: bool = False,
    field_label_y: float = PANEL_G_FIELD_LABEL_Y,
    model_label_x: float = 0.08,
    model_label_fontsize: float | None = None,
    shared_model_label: str = "Shared-scaffold\nmodel",
    component_label_fontsize: float = PANEL_G_COMPONENT_LABEL_FONTSIZE,
    segment_modulation_label_y: float | None = None,
    segment_modulation_label: str = "Segment-specific modulation",
    segment_modulation_label_gap: float = PANEL_G_SEGMENT_MODULATION_LABEL_GAP,
    example_axis_bounds: tuple[float, float, float, float] = (
        0.0,
        0.02,
        1.0,
        PANEL_G_EXAMPLE_HEIGHT_FRACTION,
    ),
    example_field_y: float = 0.05,
    example_field_height: float = 0.58,
    example_icon_bounds: tuple[float, float, float, float] = (
        -0.045,
        0.23,
        0.085,
        0.26,
    ),
    example_xlabel_y: float = -0.145,
    example_column_width: float = 0.46,
    example_column_gap: float = 0.04,
    example_plot_left_offset: float = 0.12,
    example_field_width: float = 0.14,
    example_field_gap: float = 0.035,
    example_layout: str = "columns",
    example_row_height: float = 0.46,
    example_row_gap: float = 0.05,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot Panel G example GLM fits and the model schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    schematic_ax = ax.inset_axes(
        [
            0.0,
            1.0 - schematic_height_fraction,
            1.0,
            schematic_height_fraction,
        ]
    )
    _plot_panel_g_architecture_schematic(
        schematic_ax,
        independent_track_center_y=independent_track_center_y,
        shared_track_center_y=shared_track_center_y,
        track_size=schematic_track_size,
        independent_basis_icon_scale=independent_basis_icon_scale,
        independent_basis_label=independent_basis_label,
        show_dark_track_labels=show_dark_track_labels,
        field_label_y=field_label_y,
        model_label_x=model_label_x,
        model_label_fontsize=model_label_fontsize,
        shared_model_label=shared_model_label,
        component_label_fontsize=component_label_fontsize,
        segment_modulation_label_y=segment_modulation_label_y,
        segment_modulation_label=segment_modulation_label,
        segment_modulation_label_gap=segment_modulation_label_gap,
    )
    example_ax = ax.inset_axes(example_axis_bounds)
    _plot_panel_g_example_columns(
        example_ax,
        [] if examples is None else examples,
        field_y=example_field_y,
        field_height=example_field_height,
        icon_bounds=example_icon_bounds,
        xlabel_y=example_xlabel_y,
        column_width=example_column_width,
        column_gap=example_column_gap,
        plot_left_offset=example_plot_left_offset,
        field_width=example_field_width,
        field_gap=example_field_gap,
        layout=example_layout,
        row_height=example_row_height,
        row_gap=example_row_gap,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def _remove_w_track_center_label(ax: "Axes") -> None:
    """Remove the center-arm C label from compact model schematics."""
    for text in list(ax.texts):
        if text.get_text() == "C":
            text.remove()


def _panel_h_basis_styles(
    *,
    edge_color: str,
    fill_color: str,
    fill_alpha: float,
    linewidth: float,
) -> list[dict[str, Any]]:
    """Return thin three-segment basis styles for the scaled Panel H schematic."""
    return [
        {
            "edge_color": edge_color,
            "fill_color": fill_color,
            "fill_alpha": fill_alpha,
            "linewidth": linewidth,
            "radius": PANEL_H_SCHEMATIC_BASIS_RADIUS,
            "spacing": PANEL_H_SCHEMATIC_BASIS_SPACING,
        }
        for _segment_index in range(3)
    ]


def _panel_h_basis_styles_with_highlighted_segments(
    highlighted_segments: Sequence[int],
) -> list[dict[str, Any]]:
    """Return thin Panel H basis styles with selected 1-based segments filled."""
    highlighted = {int(segment_index) for segment_index in highlighted_segments}
    styles = []
    for segment_index in range(1, 4):
        if segment_index in highlighted:
            styles.append(
                {
                    "edge_color": "black",
                    "fill_color": GLM_BASIS_LIGHT_COLOR,
                    "fill_alpha": 0.76,
                    "linewidth": PANEL_H_SCHEMATIC_BASIS_LINEWIDTH,
                    "radius": PANEL_H_SCHEMATIC_BASIS_RADIUS,
                    "spacing": PANEL_H_SCHEMATIC_BASIS_SPACING,
                }
            )
        else:
            styles.append(
                {
                    "edge_color": "black",
                    "fill_color": "none",
                    "fill_alpha": 1.0,
                    "linewidth": PANEL_H_SCHEMATIC_BASIS_LINEWIDTH,
                    "radius": PANEL_H_SCHEMATIC_BASIS_RADIUS,
                    "spacing": PANEL_H_SCHEMATIC_BASIS_SPACING,
                }
            )
    return styles


def _panel_h_shared_basis_styles_with_filled_segments(
    filled_segments: Sequence[int],
) -> list[dict[str, Any]]:
    """Return shared-scaffold basis styles with selected 1-based segments filled."""
    filled = {int(segment_index) for segment_index in filled_segments}
    styles = []
    for segment_index in range(1, 4):
        styles.append(
            {
                "edge_color": "black",
                "fill_color": (
                    GLM_BASIS_DARK_COLOR if segment_index in filled else "none"
                ),
                "fill_alpha": 0.7 if segment_index in filled else 1.0,
                "linewidth": PANEL_H_SCHEMATIC_DARK_BASIS_LINEWIDTH,
                "radius": PANEL_H_SCHEMATIC_BASIS_RADIUS,
                "spacing": PANEL_H_SCHEMATIC_BASIS_SPACING,
            }
        )
    return styles


def _panel_h_oval_styles(count: int, *, fill: bool = True) -> list[dict[str, Any]]:
    """Return thin orange modulation ovals for the scaled Panel H schematic."""
    linewidth = PANEL_H_SCHEMATIC_OVAL_LINEWIDTH if fill else 0.90
    return [
        {
            "edge_color": GLM_BASIS_LIGHT_COLOR,
            "fill_color": GLM_BASIS_LIGHT_COLOR if fill else "none",
            "fill_alpha": 0.38 if fill else 1.0,
            "linewidth": linewidth,
        }
        for _index in range(count)
    ]


def _draw_panel_h_track(
    ax: "Axes",
    *,
    track_kind: str,
    show_labels: bool = False,
    trajectory_name: str = "center_to_left",
    stimulus_layout: str = "stim1",
    highlighted_segments: Sequence[int] | None = None,
    oval_regions: Sequence[str] | None = None,
    fill_oval_regions: bool = True,
    label_fontsize: float = 3.1,
    label_colors: Mapping[str, Any] | None = None,
    region_fill_colors: Mapping[str, Any] | None = None,
    region_fill_alpha: float | None = None,
    show_place_field_blob: bool = False,
    place_field_colors: Sequence[str] | None = None,
    place_field_blob_size_scale: float = 1.0,
    place_field_arm: str = "left",
    segment_outline_colors: Mapping[str, Any] | None = None,
    segment_outline_linewidths: Mapping[str, float] | None = None,
) -> None:
    """Draw one thin W-track component for the scaled Panel H swap schematic."""
    trajectory_color = GLM_TRAJECTORY_ARROW_COLOR
    if track_kind == "dark":
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            fill_track_black=True,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_color="white",
            label_fontsize=label_fontsize,
            show_trajectory=not show_place_field_blob,
            show_basis=not show_place_field_blob,
            basis_segment_styles=_panel_h_basis_styles(
                edge_color="black",
                fill_color=GLM_BASIS_DARK_COLOR,
                fill_alpha=0.7,
                linewidth=PANEL_H_SCHEMATIC_DARK_BASIS_LINEWIDTH,
            ),
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
        )
        if show_place_field_blob:
            _draw_panel_g_place_field_blob(
                ax,
                colors=place_field_colors,
                size_scale=place_field_blob_size_scale,
                arm=place_field_arm,
            )
        _remove_w_track_center_label(ax)
        return

    if track_kind == "independent_light":
        basis_segment_styles = (
            _panel_h_basis_styles_with_highlighted_segments(highlighted_segments)
            if highlighted_segments is not None
            else _panel_h_basis_styles(
                edge_color="black",
                fill_color=GLM_BASIS_LIGHT_COLOR,
                fill_alpha=0.76,
                linewidth=PANEL_H_SCHEMATIC_BASIS_LINEWIDTH,
            )
        )
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_colors=label_colors,
            label_fontsize=label_fontsize,
            show_trajectory=not show_place_field_blob,
            show_basis=not show_place_field_blob,
            basis_segment_styles=basis_segment_styles,
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
            region_fill_colors=region_fill_colors,
            region_fill_alpha=region_fill_alpha,
        )
        if show_place_field_blob:
            _draw_panel_g_place_field_blob(
                ax,
                colors=place_field_colors,
                size_scale=place_field_blob_size_scale,
                arm=place_field_arm,
            )
        _remove_w_track_center_label(ax)
        return

    if track_kind == "segment_modulation":
        selected_oval_regions = list(oval_regions or ["left_arm"])
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_colors=label_colors,
            label_fontsize=label_fontsize,
            show_arrow=segment_outline_colors is None,
            show_trajectory=segment_outline_colors is None,
            show_large_ovals=segment_outline_colors is None,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_h_oval_styles(
                len(selected_oval_regions),
                fill=fill_oval_regions,
            ),
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
        )
        if segment_outline_colors is not None:
            _draw_panel_g_segment_gain_outlines(
                ax,
                outline_colors=segment_outline_colors,
                outline_linewidths=segment_outline_linewidths,
            )
        _remove_w_track_center_label(ax)
        return

    if track_kind == "shared_light":
        selected_oval_regions = list(oval_regions or ["left_arm"])
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_colors=label_colors,
            label_fontsize=label_fontsize,
            show_trajectory=not show_place_field_blob,
            show_basis=not show_place_field_blob,
            basis_segment_styles=_panel_h_shared_basis_styles_with_filled_segments(
                (3,)
            ),
            show_large_ovals=not show_place_field_blob,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_h_oval_styles(
                len(selected_oval_regions),
                fill=fill_oval_regions,
            ),
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
            region_fill_colors=region_fill_colors,
            region_fill_alpha=region_fill_alpha,
        )
        if show_place_field_blob:
            _draw_panel_g_place_field_blob(
                ax,
                colors=place_field_colors,
                size_scale=place_field_blob_size_scale,
                arm=place_field_arm,
            )
        if segment_outline_colors is not None:
            _draw_panel_g_segment_gain_outlines(
                ax,
                outline_colors=segment_outline_colors,
                outline_linewidths=segment_outline_linewidths,
            )
        _remove_w_track_center_label(ax)
        return

    raise ValueError(f"Unknown Panel H track_kind {track_kind!r}.")


def _draw_panel_h_swap_schematic(
    ax: "Axes",
    *,
    track_size: tuple[float, float] | None = None,
    show_dark_track_labels: bool = False,
    show_model_labels: bool = True,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    model_labels: Mapping[str, str] | None = None,
    prediction_label_fontsize: float = 3.0,
    independent_track_center_y: float = PANEL_H_INDEPENDENT_TRACK_CENTER_Y,
    independent_prediction_label_y: float = 0.61,
    segment_modulation_track_center_y: float = PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y,
    shared_dark_track_center_y: float = PANEL_H_SHARED_DARK_TRACK_CENTER_Y,
    shared_light_track_center_y: float = PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y,
    shared_prediction_label_y: float = 0.02,
) -> None:
    """Draw a scaled full-layout train/predict swap schematic for Panel H."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    shared_model_label = (
        "Shared-scaffold\nmodel"
        if str(model_name) == "task_segment_bump"
        else f"{_panel_model_label(str(model_name), model_labels)}\nmodel"
    )
    shared_prediction_label = (
        "\"Light activity is like the same arm\n"
        "dark activity with visual modulation\""
        if str(model_name) == "task_segment_bump"
        else "\"Light activity is like the same arm\n"
        "dark activity with segment gain\""
    )

    def _bounds_from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> list[float]:
        return [center_x - width / 2.0, center_y - height / 2.0, width, height]

    train_center_x = 0.36
    predict_center_x = 0.78
    train_predict_midpoint_x = 0.5 * (train_center_x + predict_center_x)
    default_light_bounds = {"width": 0.38, "height": 0.23}
    default_dark_bounds = {"width": 0.34, "height": 0.21}
    light_bounds = (
        {"width": track_size[0], "height": track_size[1]}
        if track_size is not None
        else default_light_bounds
    )
    dark_bounds = (
        {"width": track_size[0], "height": track_size[1]}
        if track_size is not None
        else default_dark_bounds
    )

    ax.text(train_center_x, 0.98, "Train: AB", ha="center", va="top", fontsize=5.8)
    ax.text(predict_center_x, 0.98, "Predict: BA", ha="center", va="top", fontsize=5.8)
    if show_model_labels:
        ax.text(
            0.045,
            0.72,
            "Independent\nmodel",
            ha="center",
            va="center",
            fontsize=4.1,
            fontweight="bold",
        )
        ax.text(
            0.045,
            0.235,
            shared_model_label,
            ha="center",
            va="center",
            fontsize=3.8,
            fontweight="bold",
        )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                train_center_x,
                independent_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        highlighted_segments=(3,),
        label_fontsize=4.8,
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                predict_center_x,
                independent_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        highlighted_segments=(3,),
        label_fontsize=4.8,
    )
    ax.text(
        train_predict_midpoint_x,
        independent_prediction_label_y,
        "\"Light activity is like the other arm\nwith the same visual landmark\"",
        ha="center",
        va="top",
        fontsize=prediction_label_fontsize,
        linespacing=0.9,
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                train_center_x,
                segment_modulation_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="segment_modulation",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        oval_regions=["left_arm"],
        label_fontsize=4.8,
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                train_center_x,
                shared_dark_track_center_y,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
        show_labels=show_dark_track_labels,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                predict_center_x,
                shared_light_track_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        oval_regions=["right_arm"],
        label_fontsize=4.8,
    )
    ax.text(
        train_predict_midpoint_x,
        shared_prediction_label_y,
        shared_prediction_label,
        ha="center",
        va="bottom",
        fontsize=prediction_label_fontsize,
        linespacing=0.9,
    )


def _filter_panel_h_heldout_delta(swap_delta_table: Any) -> Any:
    """Return Panel H delta rows for the held-out 06_r3 light epoch."""
    table = swap_delta_table
    if table is None or "delta_ll_bits_per_spike" not in table:
        return table
    if "light_test_epoch" in table:
        table = table[table["light_test_epoch"].astype(str) == PANEL_H_HELDOUT_LIGHT_EPOCH]
    if "light_train_epoch" in table:
        table = table[table["light_train_epoch"].astype(str) == PANEL_H_TRAIN_LIGHT_EPOCH]
    return table


def _format_panel_h_delta_summary(values: np.ndarray) -> str:
    """Return two-line fraction-positive and median text for Panel H."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n/a >0\nmed. n/a"
    fraction_positive = float(np.mean(values > 0.0))
    median = float(np.median(values))
    return f"{fraction_positive:.0%} >0\nmed. {median:.2f}"


def _plot_panel_h_delta_axis(
    ax: "Axes",
    swap_delta_table: Any,
    *,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    trajectory_type: str | None = None,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
) -> None:
    """Plot held-out 06_r3 model-minus-independent delta LL values."""
    model_name = str(model_name)
    model_label = _panel_model_label(model_name, model_labels)
    visual_color = _panel_model_color("visual", model_colors)
    model_color = _panel_model_color(model_name, model_colors)
    heldout_table = _filter_panel_h_heldout_delta(swap_delta_table)
    if (
        trajectory_type is not None
        and heldout_table is not None
        and "trajectory" in heldout_table
    ):
        heldout_table = heldout_table[
            heldout_table["trajectory"].astype(str) == str(trajectory_type)
        ]
    values = _finite_column_values(heldout_table, "delta_ll_bits_per_spike")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.65, zorder=1)
    if values.size:
        x_limits = PANEL_H_DELTA_X_LIMITS
        bin_edges = np.linspace(x_limits[0], x_limits[1], 29)
        hist_kwargs = OUTLINED_HISTOGRAM_KWARGS.copy()
        hist_kwargs.update({"edgecolor": "none", "linewidth": 0.0})
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=model_color,
            **hist_kwargs,
            zorder=2,
        )
        ax.text(
            0.03,
            0.94,
            "Independent\nbetter",
            ha="left",
            va="top",
            fontsize=4.0,
            color=visual_color,
            transform=ax.transAxes,
        )
        ax.text(
            0.97,
            0.94,
            f"{model_label}\nbetter",
            ha="right",
            va="top",
            fontsize=4.0,
            color=model_color,
            transform=ax.transAxes,
        )
        ax.text(
            0.97,
            0.56,
            _format_panel_h_delta_summary(values),
            ha="right",
            va="top",
            fontsize=4.8,
            color=model_color,
            transform=ax.transAxes,
        )
        ax.set_xlim(*x_limits)
    else:
        ax.text(0.5, 0.5, "No swap\nvalues", ha="center", va="center")
        ax.set_xlim(*PANEL_H_DELTA_X_LIMITS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_xticklabels:
        ax.tick_params(labelbottom=False)
    if not show_yticklabels:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=3.4, length=1.0, pad=0.6)


def _plot_panel_h_delta_grid(
    ax: "Axes",
    swap_delta_table: Any,
    *,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    grid_bounds: Sequence[tuple[float, float, float, float]] | None = None,
    xlabel_y: float = -0.055,
) -> None:
    """Plot Panel H delta LL histograms split by trajectory."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    selected_grid_bounds = (
        grid_bounds
        or (
            (0.08, 0.55, 0.40, 0.31),
            (0.56, 0.55, 0.40, 0.31),
            (0.08, 0.12, 0.40, 0.31),
            (0.56, 0.12, 0.40, 0.31),
        )
    )
    row_centers = [bounds[1] + bounds[3] / 2 for bounds in selected_grid_bounds]
    ylabel_y = (min(row_centers) + max(row_centers)) / 2
    for trajectory_index, (trajectory_type, bounds) in enumerate(
        zip(PANEL_H_DELTA_TRAJECTORIES, selected_grid_bounds, strict=True)
    ):
        icon_ax = ax.inset_axes(
            [
                bounds[0] + bounds[2] * 0.31,
                bounds[1] + bounds[3] + 0.008,
                bounds[2] * 0.38,
                0.105,
            ]
        )
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.42,
            trajectory_linewidth=0.68,
            arrow_mutation_scale=5.4,
            fill_track=False,
        )
        delta_ax = ax.inset_axes(bounds)
        _plot_panel_h_delta_axis(
            delta_ax,
            swap_delta_table,
            model_name=model_name,
            model_colors=model_colors,
            model_labels=model_labels,
            trajectory_type=trajectory_type,
            show_xticklabels=trajectory_index >= 2,
            show_yticklabels=trajectory_index in (0, 2),
        )

    ax.text(
        0.53,
        xlabel_y,
        "Δ log likelihood (bits/spike)",
        ha="center",
        va="bottom",
        fontsize=4.3,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        -0.055,
        ylabel_y,
        "Frac.",
        ha="left",
        va="center",
        rotation=90,
        fontsize=4.3,
        transform=ax.transAxes,
        clip_on=False,
    )


def _plot_panel_h_switched_segment_example(
    ax: "Axes",
    swap_example: dict[str, Any] | None,
    *,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    example_label: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = True,
    show_xticklabels: bool = True,
    icon_bounds: tuple[float, float, float, float] | None = (1.05, 0.04, 0.28, 0.34),
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: tuple[float, float] | None = (1.02, 1.02),
    delta_label_position: tuple[float, float] | None = None,
    delta_label_va: str | None = None,
) -> None:
    """Plot one empirical and model-predicted switched segment."""
    if swap_example is None:
        ax.text(0.5, 0.5, "No switch\nexample", ha="center", va="center", fontsize=5.0)
        ax.axis("off")
        return

    model_name = str(swap_example.get("model_name", model_name))
    start = float(swap_example["segment_start"])
    end = float(swap_example["segment_end"])
    observed_position = np.asarray(swap_example["observed_position"], dtype=float)
    observed_rate = np.asarray(swap_example["observed_rate_hz"], dtype=float)
    observed_mask = (
        np.isfinite(observed_position)
        & np.isfinite(observed_rate)
        & (observed_position >= start)
        & (observed_position <= end)
    )
    tp_grid = np.asarray(swap_example["tp_grid"], dtype=float)
    grid_mask = (tp_grid >= start) & (tp_grid <= end)
    values = [observed_rate[observed_mask]]
    for plotted_model_name in ("visual", model_name):
        values.append(
            np.asarray(swap_example["models"][plotted_model_name], dtype=float)[grid_mask]
        )
    finite_values = [value[np.isfinite(value)] for value in values if value.size]
    y_max = 1.0 if not finite_values else max(1.0, float(np.ceil(np.nanmax(np.concatenate(finite_values)))))

    ax.plot(
        observed_position[observed_mask],
        observed_rate[observed_mask],
        color=GLM_EMPIRICAL_COLOR,
        linewidth=0.9,
        label="Empirical",
        zorder=4,
    )
    for plotted_model_name in ("visual", model_name):
        ax.plot(
            tp_grid[grid_mask],
            np.asarray(swap_example["models"][plotted_model_name], dtype=float)[grid_mask],
            color=_panel_model_color(plotted_model_name, model_colors),
            linewidth=0.8,
            label=_panel_model_label(plotted_model_name, model_labels),
            zorder=3,
        )
    ax.axvspan(start, end, color=GLM_BASIS_LIGHT_COLOR, alpha=0.10, linewidth=0)
    ax.set_xlim(start, end)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([start, end])
    ax.set_xticklabels([f"{start:.2f}", f"{end:.2f}"])
    if not show_xticklabels:
        ax.tick_params(labelbottom=False)
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    delta_ll = swap_example.get("delta_ll_bits_per_spike")
    delta_label = ""
    if delta_ll is not None and np.isfinite(float(delta_ll)):
        delta_label = f"ΔLL={float(delta_ll):.2f}"
        delta_text_position = delta_label_position or (
            (0.96, 0.94) if example_label == "Example 1" else (0.96, 0.06)
        )
        delta_text_vertical_alignment = delta_label_va or (
            "top" if example_label == "Example 1" else "bottom"
        )
        ax.text(
            *delta_text_position,
            delta_label,
            ha="right",
            va=delta_text_vertical_alignment,
            fontsize=3.8,
            transform=ax.transAxes,
        )
    ax.set_title(
        "Example" if example_label is None else example_label,
        fontsize=5.0,
        pad=0.8,
    )
    if icon_bounds is not None:
        icon_ax = ax.inset_axes(icon_bounds)
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=swap_example["trajectory"],
            arrow_color=PANEL_TRAJECTORY_COLORS[swap_example["trajectory"]],
            track_linewidth=0.34,
            trajectory_linewidth=0.55,
            arrow_mutation_scale=4.8,
            fill_track=False,
        )
    if show_xlabel:
        ax.set_xlabel("Switched segment", fontsize=4.5, labelpad=0.8)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.5, labelpad=0.8)
    if show_legend:
        legend_kwargs: dict[str, Any] = {
            "frameon": False,
            "fontsize": 3.4,
            "handlelength": 0.9,
            "loc": legend_loc,
            "borderaxespad": 0.0,
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(**legend_kwargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.1, length=1.2, pad=0.7)


def plot_panel_h_swap_delta(
    ax: "Axes",
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    *,
    model_name: str = PANEL_H_DEFAULT_MODEL_NAME,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    schematic_axis_bounds: tuple[float, float, float, float] = PANEL_H_SCHEMATIC_AXIS_BOUNDS,
    delta_axis_bounds: tuple[float, float, float, float] = PANEL_H_DELTA_AXIS_BOUNDS,
    example_axis_bounds: Sequence[tuple[float, float, float, float]] = (
        PANEL_H_EXAMPLE_AXIS_BOUNDS
    ),
    schematic_track_size: tuple[float, float] | None = None,
    show_dark_track_labels: bool = False,
    show_model_labels: bool = True,
    prediction_label_fontsize: float = 3.0,
    independent_track_center_y: float = PANEL_H_INDEPENDENT_TRACK_CENTER_Y,
    independent_prediction_label_y: float = 0.61,
    segment_modulation_track_center_y: float = PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y,
    shared_dark_track_center_y: float = PANEL_H_SHARED_DARK_TRACK_CENTER_Y,
    shared_light_track_center_y: float = PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y,
    shared_prediction_label_y: float = 0.02,
    delta_grid_bounds: Sequence[tuple[float, float, float, float]] | None = None,
    delta_xlabel_y: float = -0.055,
    example_delta_label_positions: Sequence[tuple[float, float] | None] | None = None,
    example_delta_label_vertical_alignments: Sequence[str | None] | None = None,
    example_icon_bounds: tuple[float, float, float, float] | None = (
        -0.36,
        0.04,
        0.28,
        0.34,
    ),
) -> None:
    """Plot the Panel H swap schematic, delta LL, and switched-segment examples."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    schematic_ax = ax.inset_axes(schematic_axis_bounds)
    delta_ax = ax.inset_axes(delta_axis_bounds)
    example_axes = [ax.inset_axes(bounds) for bounds in example_axis_bounds]
    _draw_panel_h_swap_schematic(
        schematic_ax,
        track_size=schematic_track_size,
        show_dark_track_labels=show_dark_track_labels,
        show_model_labels=show_model_labels,
        model_name=model_name,
        model_labels=model_labels,
        prediction_label_fontsize=prediction_label_fontsize,
        independent_track_center_y=independent_track_center_y,
        independent_prediction_label_y=independent_prediction_label_y,
        segment_modulation_track_center_y=segment_modulation_track_center_y,
        shared_dark_track_center_y=shared_dark_track_center_y,
        shared_light_track_center_y=shared_light_track_center_y,
        shared_prediction_label_y=shared_prediction_label_y,
    )
    _plot_panel_h_delta_grid(
        delta_ax,
        swap_delta_table,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
        grid_bounds=delta_grid_bounds,
        xlabel_y=delta_xlabel_y,
    )
    delta_label_positions = example_delta_label_positions or (
        (0.96, 0.94),
        (0.96, 0.94),
    )
    delta_label_vertical_alignments = example_delta_label_vertical_alignments or (
        "top",
        "top",
    )
    examples = _coerce_panel_h_swap_examples(swap_examples)[: len(example_axes)]
    legend_example_index = max(min(len(examples), len(example_axes)) - 1, 0)
    for example_index, example_ax in enumerate(example_axes):
        example = examples[example_index] if example_index < len(examples) else None
        delta_label_position = (
            delta_label_positions[example_index]
            if example_index < len(delta_label_positions)
            else None
        )
        delta_label_vertical_alignment = (
            delta_label_vertical_alignments[example_index]
            if example_index < len(delta_label_vertical_alignments)
            else None
        )
        _plot_panel_h_switched_segment_example(
            example_ax,
            example,
            model_name=model_name,
            model_colors=model_colors,
            model_labels=model_labels,
            example_label=f"Example {example_index + 1}",
            show_xlabel=True,
            show_ylabel=True,
            show_legend=example_index == legend_example_index,
            show_xticklabels=True,
            icon_bounds=example_icon_bounds,
            legend_loc="center left",
            legend_bbox_to_anchor=(1.02, 0.5),
            delta_label_position=delta_label_position,
            delta_label_va=delta_label_vertical_alignment,
        )
