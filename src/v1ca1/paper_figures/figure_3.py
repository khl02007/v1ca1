from __future__ import annotations

"""Generate Figure 3 panels for light-epoch place fields."""

import argparse
import hashlib
import json
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
    DECODING_CROSS_TRAJECTORY_COMPARISONS,
    HEATMAP_COLORBAR_LABEL_FONTSIZE,
    HEATMAP_COLORBAR_LABELPAD,
    add_centered_axis_text,
    build_normalized_position_bins,
    build_pooled_panel_values,
    compute_dark_epoch_tuning_curves,
    draw_neuron_scale_bar,
    draw_order_schematic,
    extract_unit_rate_curve,
    get_unit_spike_times,
    normalize_linear_position_by_trajectory,
    orient_panel_e_task_progression,
    plot_pooled_heatmap_grid,
)
from v1ca1.paper_figures.style import (
    EPOCH_TYPE_COLORS,
    EPOCH_HISTOGRAM_ALPHA,
    MODEL_CLASS_COLORS,
    NEUTRAL_COLORS,
    OUTLINED_HISTOGRAM_KWARGS,
    RASTER_TICK_KWARGS,
    SCHEMATIC_COLORS,
    TRAJECTORY_COLORS,
    VISUAL_CONDITION_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.helper.plot_wtrack_schematic import get_w_track_geometry
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


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "figure_3"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_REGIONS = ("v1",)
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_PANEL_A_HEIGHT_MM = 42.56
DEFAULT_PANEL_BC_HEIGHT_MM = DEFAULT_HEATMAP_HEIGHT_MM
DEFAULT_PANEL_DEF_HEIGHT_MM = 30.0
DEFAULT_PANEL_GH_HEIGHT_MM = 42.0
DEFAULT_FIGURE_HEIGHT_MM = (
    DEFAULT_PANEL_A_HEIGHT_MM
    + DEFAULT_PANEL_BC_HEIGHT_MM
    + DEFAULT_PANEL_DEF_HEIGHT_MM
    + DEFAULT_PANEL_GH_HEIGHT_MM
)
DEFAULT_PANEL_B_WIDTH_FRACTION = DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION
DEFAULT_PANEL_C_WIDTH_FRACTION = DEFAULT_PANEL_E_WIDTH_FRACTION
PANEL_DEF_WIDTH_RATIOS = (0.86, 1.30, 1.58)
PANEL_DEF_WSPACE = 0.22
PANEL_DEF_AXIS_BOTTOM = 0.13
PANEL_DEF_AXIS_HEIGHT = 0.70
PANEL_D_AXIS_BOUNDS = (0.30, PANEL_DEF_AXIS_BOTTOM, 0.82, PANEL_DEF_AXIS_HEIGHT)
PANEL_E_AXIS_BOUNDS = (0.06, PANEL_DEF_AXIS_BOTTOM, 0.92, PANEL_DEF_AXIS_HEIGHT)
PANEL_GH_WIDTH_RATIOS = (0.4, 0.6)
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
PANEL_C_COLORBAR_PAD = 0.002
PANEL_C_NEURON_SCALE_BAR_X = 1.02
PANEL_C_HORIZONTAL_SHIFT = 0.012
PANEL_B_CACHE_VERSION = 1
PANEL_B_CACHE_PREFIX = "figure_3_panel_b"
PANEL_B_CACHE_METADATA_KEY = "__metadata__"
PANEL_B_CACHE_DATASET_TOKEN_LIMIT = 120
PANEL_EXAMPLE_CACHE_VERSION = 1
PANEL_EXAMPLE_CACHE_PREFIX = "figure_3_panel_example"
PANEL_EXAMPLE_CACHE_METADATA_KEY = "__metadata__"
TUNING_ANALYSIS_RELATIVE_DIR = Path("task_progression") / "tuning_analysis"
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
DECODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "decoding_comparison"
DARK_LIGHT_GLM_RELATIVE_DIR = Path("task_progression") / "dark_light_glm"
SWAP_GLM_COMPARISON_RELATIVE_DIR = Path("task_progression") / "swap_glm_comparison"
COMPUTE_TUNING_CURVES_RELATIVE_DIR = Path("task_progression") / "compute_tuning_curves"
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
PANEL_A_EPOCH_COLORS = VISUAL_CONDITION_COLORS
SEGMENT_BOUNDARIES = (0.4, 0.6)
SEGMENT_BOUNDARY_COLOR = NEUTRAL_COLORS["segment_boundary"]
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
PANEL_C_TRAJECTORY_COLORS = TRAJECTORY_COLORS
PANEL_C_EPOCH_LABELS = {
    "dark": "Dark",
    "light": "Light",
}
PANEL_C_DARK_EPOCH_BACKGROUND = NEUTRAL_COLORS["dark_epoch_background"]
PANEL_B_EXAMPLE_BLOCK_HEIGHT = 0.66
PANEL_B_EXAMPLE_TOP = 1.16
PANEL_B_EXAMPLE_BOTTOM = -0.044
PANEL_B_FIRST_EXAMPLE_Y_SHIFT = 0.09
PANEL_B_TITLE_Y = 1.30
PANEL_B_EXAMPLE_RASTER_Y = 0.48
PANEL_B_EXAMPLE_RASTER_HEIGHT = 0.34
PANEL_B_EXAMPLE_RATE_Y = 0.16
PANEL_B_EXAMPLE_RATE_HEIGHT = 0.31
PANEL_QUANT_EPOCH_ORDER = ("light", "dark")
PANEL_QUANT_EPOCH_LABELS = {
    "light": "Light",
    "dark": "Dark",
}
PANEL_QUANT_EPOCH_COLORS = {
    "light": EPOCH_TYPE_COLORS["light"],
    "dark": EPOCH_TYPE_COLORS["dark"],
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
PANEL_D_SCATTER_SIZE = 4.2
PANEL_D_SCATTER_ALPHA = 0.126
PANEL_E_ENCODING_N_FOLDS = 5
PANEL_E_PLACE_BIN_SIZE_CM = DEFAULT_PLACE_BIN_SIZE_CM
PANEL_E_DELTA_COLUMN = "delta_bits_place_vs_tp"
PANEL_E_X_LIMITS = (-0.75, 0.75)
PANEL_F_DECODING_MODELS = ("task_progression", "place")
PANEL_F_DECODING_METRIC = "median_abs_error"
PANEL_F_PLACE_MODEL_NAME = "place"
PANEL_F_POOLED_LABEL = "pooled"
PANEL_F_CROSS_COMPARISONS = DECODING_CROSS_TRAJECTORY_COMPARISONS[:1]
PANEL_F_NORM_ERROR_YLIM = (0.0, 0.5)
PANEL_F_PLACE_ERROR_YLIM = (0.0, 0.12)
PANEL_F_YLABEL_FONTSIZE = 5.8
PANEL_F_CROSS_AXIS_BOUNDS = (0.04, PANEL_DEF_AXIS_BOTTOM, 0.42, PANEL_DEF_AXIS_HEIGHT)
PANEL_F_PLACE_AXIS_BOUNDS = (0.56, PANEL_DEF_AXIS_BOTTOM, 0.42, PANEL_DEF_AXIS_HEIGHT)
PANEL_F_SUMMARY_TEXT_FONTSIZE = 4.2
PANEL_F_ERROR_SUMMARY_COLUMNS = (
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
PANEL_G_MODELS = ("visual", "task_segment_bump")
PANEL_G_MODEL_LABELS = {
    "visual": "Independent",
    "task_segment_bump": "Shared scaffold",
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
PANEL_H_SWAP_LIGHT_EPOCH_PAIRS = (("02_r1", "06_r3"), ("06_r3", "02_r1"))
PANEL_H_HELDOUT_LIGHT_EPOCH = "06_r3"
PANEL_H_TRAIN_LIGHT_EPOCH = "02_r1"
PANEL_H_SWAP_DELTA_VARIABLE = (
    "test_light_swapped_segment_swapped_delta_model_minus_visual_raw_ll_bits_per_spike"
)
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
PANEL_H_SCHEMATIC_AXIS_BOUNDS = (-0.025, 0.01, 0.34, 0.96)
PANEL_H_DELTA_AXIS_BOUNDS = (0.32, 0.04, 0.45, 0.88)


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
) -> dict[str, Any]:
    """Return metadata that identifies one Panel B heatmap cache."""
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

    return {
        "cache_version": PANEL_B_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "B",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "light_epoch_argument": light_epoch,
        "datasets": dataset_metadata,
        "trajectory_types": list(TRAJECTORY_TYPES),
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "pooled_builder": "build_pooled_panel_values",
    }


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
    for order_trajectory in TRAJECTORY_TYPES:
        for plot_trajectory in TRAJECTORY_TYPES:
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
            for order_trajectory in TRAJECTORY_TYPES:
                for plot_trajectory in TRAJECTORY_TYPES:
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
            )
        )

    panels = build_pooled_panel_values(
        curve_sets,
        position_bin_count=position_bin_count,
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
        "figure": DEFAULT_OUTPUT_NAME,
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
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> dict[str, Any]:
    """Load the panel-A example cell rasters and rate curves across epochs."""
    epoch_specs = build_panel_a_epoch_specs(
        animal_name,
        date,
        dark_epoch=dark_epoch,
    )
    epoch_examples = {
        epoch_key: load_or_compute_panel_example_data(
            data_root=data_root,
            panel_name="A",
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region=region,
            unit_id=unit_id,
            trajectories=PANEL_A_TRAJECTORIES,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
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
    ax.set_xlabel("Norm. path progression", fontsize=4.8, labelpad=1)
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
            **RASTER_TICK_KWARGS,
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


def plot_panel_a_combined_raster_axis(
    ax: "Axes",
    example: dict[str, Any],
    trajectory_type: str,
) -> None:
    """Plot all panel-A epoch rasters in one stacked axis."""
    epoch_order = tuple(example["epoch_order"])
    n_epochs = len(epoch_order)
    for epoch_index, epoch_key in enumerate(epoch_order):
        epoch_base = float(n_epochs - epoch_index - 1)
        if epoch_key == "dark":
            ax.axhspan(
                epoch_base,
                epoch_base + 1.0,
                color=PANEL_C_DARK_EPOCH_BACKGROUND,
                linewidth=0,
                zorder=0,
            )
        trial_positions = example["epoch_examples"][epoch_key]["raster_positions"][
            trajectory_type
        ]
        n_trials = len(trial_positions)
        color = PANEL_A_EPOCH_COLORS[epoch_key]
        for trial_index, positions in enumerate(trial_positions, start=1):
            positions = np.asarray(positions, dtype=float)
            if positions.size == 0:
                continue
            y_position = epoch_base + (trial_index / max(n_trials + 1, 1))
            ax.plot(
                positions,
                np.full(positions.shape, y_position, dtype=float),
                "|",
                color=color,
                **RASTER_TICK_KWARGS,
                zorder=3,
            )

    for separator in range(1, n_epochs):
        ax.axhline(separator, color="0.82", linewidth=0.35, zorder=1)
    add_segment_boundary_lines(ax)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(max(1, n_epochs)))
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

    left_margin = 0.13
    right_margin = 0.012
    column_gap = 0.026
    column_width = (
        1.0
        - left_margin
        - right_margin
        - column_gap * (len(trajectories) - 1)
    ) / len(trajectories)
    raster_bottom = 0.34
    raster_height = 0.43
    y_max = _get_panel_a_y_max(example)

    icon_specs = (
        {"left_label": "A", "right_label": "B", "fill_track": False},
        {"left_label": "B", "right_label": "A", "fill_track": False},
        {"left_label": None, "right_label": None, "fill_track": True},
    )
    for row_index, icon_spec in enumerate(icon_specs):
        epoch_center = raster_bottom + raster_height * (
            len(icon_specs) - row_index - 0.5
        ) / len(icon_specs)
        icon_ax = ax.inset_axes([0.026, epoch_center - 0.061, 0.070, 0.122])
        draw_panel_a_epoch_icon(icon_ax, **icon_spec)

    for trajectory_index, trajectory_type in enumerate(trajectories):
        left = left_margin + trajectory_index * (column_width + column_gap)
        schematic_ax = ax.inset_axes(
            [left + 0.34 * column_width, 0.80, 0.32 * column_width, 0.12]
        )
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_C_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )

        raster_ax = ax.inset_axes([left, raster_bottom, column_width, raster_height])
        plot_panel_a_combined_raster_axis(raster_ax, example, trajectory_type)
        if trajectory_index == 0:
            raster_ax.set_ylabel("")

        rate_ax = ax.inset_axes([left, 0.147, column_width, 0.16])
        plot_panel_a_rate_axis(
            rate_ax,
            example,
            trajectory_type,
            y_max=y_max,
            show_ylabel=trajectory_index == 0,
            show_legend=False,
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
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> dict[str, Any]:
    """Load one dark-vs-light example unit for panel C."""
    trajectories = validate_panel_c_trajectories(trajectories)
    dark_epoch_id = get_dark_epoch(animal_name, date, dark_epoch)
    light_epoch_id = get_light_epoch(animal_name, date, light_epoch)
    epoch_rates = {
        "dark": load_or_compute_panel_example_data(
            data_root=data_root,
            panel_name="C",
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
            panel_name="C",
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


def _compute_panel_c_rate_correlation(
    example: dict[str, Any],
    epoch_key: str,
    trajectories: Sequence[str],
) -> float:
    """Return Pearson correlation between the two panel-C FR curves."""
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
    ax.set_xlabel("Norm. path progression", fontsize=4.8, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", fontsize=4.8, labelpad=1)
    if show_title:
        ax.set_title(PANEL_C_EPOCH_LABELS[epoch_key], fontsize=5.3, pad=1)
    if show_legend:
        ax.legend(frameon=False, fontsize=4.2, handlelength=1.1, borderpad=0.1)
    correlation = _compute_panel_c_rate_correlation(example, epoch_key, trajectories)
    label = f"r={correlation:.2f}" if np.isfinite(correlation) else "r=n/a"
    ax.text(
        0.96,
        0.92,
        label,
        ha="right",
        va="top",
        fontsize=4.6,
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)


def plot_panel_c_raster_axis(
    ax: "Axes",
    example: dict[str, Any],
    epoch_key: str,
    *,
    trajectories: Sequence[str] | None = None,
    show_ylabel: bool = False,
    show_title: bool = False,
) -> None:
    """Plot selected trajectory spike rasters for one panel-C epoch."""
    trajectories = (
        validate_panel_c_trajectories(example["trajectories"])
        if trajectories is None
        else validate_panel_c_trajectories(trajectories)
    )
    raster_positions = example["epoch_rates"][epoch_key]["raster_positions"]
    row_index = 1
    for trajectory_type in trajectories:
        color = PANEL_C_TRAJECTORY_COLORS[trajectory_type]
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
    if show_title:
        ax.set_title(PANEL_C_EPOCH_LABELS[epoch_key], fontsize=5.3, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=1.0, width=0.35, pad=1)


def plot_panel_c_example(
    ax: "Axes",
    example: dict[str, Any],
    *,
    title: str | None = None,
    y_shift: float = 0.0,
) -> None:
    """Plot one panel-C example cell with dark and light rate curves."""
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

    trajectories = validate_panel_c_trajectories(example["trajectories"])
    y_max = _get_panel_c_y_max(example)
    raster_y = PANEL_B_EXAMPLE_RASTER_Y + y_shift
    raster_height = PANEL_B_EXAMPLE_RASTER_HEIGHT
    schematic_height = 0.075
    schematic_gap = 0.020 if len(trajectories) > 1 else 0.0
    schematic_total_height = len(trajectories) * schematic_height + (
        len(trajectories) - 1
    ) * schematic_gap
    schematic_top = raster_y + (raster_height + schematic_total_height) / 2
    for trajectory_index, trajectory_type in enumerate(trajectories):
        schematic_y = schematic_top - schematic_height - trajectory_index * (
            schematic_height + schematic_gap
        )
        schematic_ax = ax.inset_axes([0.012, schematic_y, 0.070, schematic_height])
        draw_w_track_schematic(
            schematic_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_C_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.45,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )
    dark_raster_ax = ax.inset_axes([0.10, raster_y, 0.40, raster_height])
    light_raster_ax = ax.inset_axes([0.56, raster_y, 0.40, raster_height])
    dark_raster_ax.set_facecolor(PANEL_C_DARK_EPOCH_BACKGROUND)
    plot_panel_c_raster_axis(
        dark_raster_ax,
        example,
        "dark",
        trajectories=trajectories,
        show_ylabel=True,
        show_title=True,
    )
    plot_panel_c_raster_axis(
        light_raster_ax,
        example,
        "light",
        trajectories=trajectories,
        show_title=True,
    )

    dark_ax = ax.inset_axes(
        [0.10, PANEL_B_EXAMPLE_RATE_Y + y_shift, 0.40, PANEL_B_EXAMPLE_RATE_HEIGHT]
    )
    light_ax = ax.inset_axes(
        [0.56, PANEL_B_EXAMPLE_RATE_Y + y_shift, 0.40, PANEL_B_EXAMPLE_RATE_HEIGHT]
    )
    dark_ax.set_facecolor(PANEL_C_DARK_EPOCH_BACKGROUND)
    plot_epoch_path_rate_axis(
        dark_ax,
        example,
        "dark",
        trajectories=trajectories,
        y_max=y_max,
        show_ylabel=True,
        show_title=False,
    )
    plot_epoch_path_rate_axis(
        light_ax,
        example,
        "light",
        trajectories=trajectories,
        y_max=y_max,
        show_title=False,
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

    block_height = PANEL_B_EXAMPLE_BLOCK_HEIGHT
    y_positions = np.linspace(
        PANEL_B_EXAMPLE_TOP - block_height,
        PANEL_B_EXAMPLE_BOTTOM,
        len(examples),
    )
    for example_index, (y0, example) in enumerate(
        zip(y_positions, examples, strict=False),
        start=1,
    ):
        example_ax = ax.inset_axes([0.0, float(y0), 1.0, block_height])
        plot_panel_c_example(
            example_ax,
            example,
            title=f"Example cell {example_index}",
            y_shift=PANEL_B_FIRST_EXAMPLE_Y_SHIFT if example_index == 1 else 0.0,
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
            arrow_color=PANEL_C_TRAJECTORY_COLORS[trajectory_type],
            fill_track=False,
        )
    for row_index, ax in enumerate(order_schematic_axes):
        draw_order_schematic(
            ax,
            TRAJECTORY_TYPES[row_index % len(TRAJECTORY_TYPES)],
            arrow_color=PANEL_C_TRAJECTORY_COLORS[
                TRAJECTORY_TYPES[row_index % len(TRAJECTORY_TYPES)]
            ],
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


def _summarize_panel_f_errors(
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
    """Return one Panel F median/IQR error row."""
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


def load_panel_f_decoding_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = PANEL_F_CROSS_COMPARISONS,
) -> Any:
    """Load pooled normalized decoding-error medians and IQRs for Panel F."""
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
                model_name=PANEL_F_PLACE_MODEL_NAME,
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
        row = _summarize_panel_f_errors(
            np.concatenate(values),
            animal_name=PANEL_F_POOLED_LABEL,
            date=PANEL_F_POOLED_LABEL,
            epoch_type=epoch_type,
            epoch=PANEL_F_POOLED_LABEL,
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
        row = _summarize_panel_f_errors(
            np.concatenate(values),
            animal_name=PANEL_F_POOLED_LABEL,
            date=PANEL_F_POOLED_LABEL,
            epoch_type=epoch_type,
            epoch=PANEL_F_POOLED_LABEL,
            analysis="cross_trajectory",
            comparison=comparison,
            comparison_label=comparison_label,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=PANEL_F_ERROR_SUMMARY_COLUMNS)
    return pd.DataFrame(rows)


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
        "decoding_error": load_panel_f_decoding_error_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ),
    }


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
    bump_dataset: Any,
    visual_path: Path,
    bump_path: Path,
) -> list[dict[str, Any]]:
    """Return scored example candidates from one visual/bump selected pair."""
    trajectories = [str(value) for value in visual_dataset.coords["trajectory"].values]
    units = np.asarray(visual_dataset.coords["unit"].values)
    tp_grid = np.asarray(visual_dataset.coords["tp_grid"].values, dtype=float)
    segment_edges = np.asarray(visual_dataset.coords["segment_edge"].values, dtype=float)
    visual_score = np.asarray(
        visual_dataset["ll_bits_per_spike_cv_light"].values,
        dtype=float,
    )
    bump_score = np.asarray(
        bump_dataset["ll_bits_per_spike_cv_light"].values,
        dtype=float,
    )
    combined_score = np.minimum(visual_score, bump_score)

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
                        "task_segment_bump": {
                            "dark_hz": np.asarray(
                                bump_dataset["dark_hz_grid"].isel(
                                    trajectory=trajectory_index,
                                    unit=unit_index,
                                ).values,
                                dtype=float,
                            ),
                            "light_hz": np.asarray(
                                bump_dataset["light_hz_grid"].isel(
                                    trajectory=trajectory_index,
                                    unit=unit_index,
                                ).values,
                                dtype=float,
                            ),
                            "score": float(bump_score[trajectory_index, unit_index]),
                            "source_path": str(bump_path),
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
) -> list[dict[str, Any]]:
    """Return configured Panel G examples when they belong to the requested inputs."""
    dataset_keys = {
        normalize_dataset_id(dataset)[:2]
        for dataset in datasets
    }
    requested = [
        (animal_name, date, requested_region, int(unit_id), trajectory)
        for animal_name, date, requested_region, unit_id, trajectory in PANEL_G_EXAMPLES
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
) -> list[dict[str, Any]]:
    """Load high-scoring visual and segment-bump GLM example fits."""
    candidates: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
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
        bump_path = get_dark_light_glm_selected_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=dataset_light_epoch,
            dark_epoch=dataset_dark_epoch,
            model_name="task_segment_bump",
        )
        if not visual_path.exists() or not bump_path.exists():
            missing_paths.extend(path for path in (visual_path, bump_path) if not path.exists())
            continue

        visual_dataset = _load_panel_g_selected_dataset(visual_path)
        bump_dataset = _load_panel_g_selected_dataset(bump_path)
        candidates.extend(
            _panel_g_candidate_examples_from_pair(
                animal_name=animal_name,
                date=date,
                region=region,
                light_epoch=dataset_light_epoch,
                dark_epoch=dataset_dark_epoch,
                visual_dataset=visual_dataset,
                bump_dataset=bump_dataset,
                visual_path=visual_path,
                bump_path=bump_path,
            )
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    requested_examples = _select_requested_panel_g_examples(
        candidates,
        data_root=data_root,
        datasets=datasets,
        region=region,
        example_count=example_count,
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
) -> Any:
    """Load segment-bump minus independent swapped-segment LL values."""
    import pandas as pd
    import xarray as xr

    tables = []
    missing_paths: list[Path] = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
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
                delta = np.asarray(
                    dataset_obj[PANEL_H_SWAP_DELTA_VARIABLE]
                    .sel(model="task_segment_bump")
                    .values,
                    dtype=float,
                )
                trajectories = [str(value) for value in dataset_obj.coords["trajectory"].values]
                units = np.asarray(dataset_obj.coords["unit"].values)

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
            "trajectory",
            "unit",
            "delta_ll_bits_per_spike",
            "source_path",
        ]
    )


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
) -> list[dict[str, Any]]:
    """Return the strongest bump-advantage switched-segment examples from one dataset."""
    delta = np.asarray(
        dataset_obj[PANEL_H_SWAP_DELTA_VARIABLE].sel(model="task_segment_bump").values,
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
) -> dict[str, Any]:
    """Return one switched-segment example at explicit trajectory/unit indices."""
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
            .sel(model="task_segment_bump")
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
    for model_name in ("visual", "task_segment_bump"):
        models[model_name] = np.asarray(
            dataset_obj["test_light_swapped_hz_grid"]
            .sel(model=model_name)
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


def _panel_h_swap_example_from_dataset(
    dataset_obj: Any,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    source_path: Path,
) -> dict[str, Any] | None:
    """Return the strongest bump-advantage switched-segment example from one dataset."""
    examples = _panel_h_swap_examples_from_dataset(
        dataset_obj,
        animal_name=animal_name,
        date=date,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
        source_path=source_path,
        example_count=1,
    )
    return examples[0] if examples else None


def load_panel_h_swap_example(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = PANEL_H_TRAIN_LIGHT_EPOCH,
    light_test_epoch: str = PANEL_H_HELDOUT_LIGHT_EPOCH,
) -> dict[str, Any] | None:
    """Load one switched-segment example for Panel H."""
    examples = load_panel_h_swap_examples(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
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
    example_count: int = 2,
) -> list[dict[str, Any]]:
    """Load top switched-segment examples for Panel H."""
    import xarray as xr

    dataset_keys = {normalize_dataset_id(dataset)[:2] for dataset in datasets}
    requested = [
        (animal_name, date, requested_region, int(unit_id), trajectory)
        for animal_name, date, requested_region, unit_id, trajectory in PANEL_H_EXAMPLES
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
) -> dict[str, Any]:
    """Load saved GLM artifacts for panels G and H."""
    swap_examples = load_panel_h_swap_examples(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
    )
    return {
        "dark_light_examples": load_panel_g_dark_light_glm_examples(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        ),
        "swap_delta": load_panel_h_swap_delta_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            dark_epoch=dark_epoch,
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
            s=PANEL_D_SCATTER_SIZE,
            color=PANEL_QUANT_EPOCH_COLORS["light"],
            alpha=PANEL_D_SCATTER_ALPHA,
            edgecolors="none",
            zorder=2,
        )
        x_values = paired["similarity_light"].to_numpy(dtype=float)
        y_values = paired["similarity_dark"].to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        ax.text(
            0.04,
            0.96,
            f"n={int(np.sum(valid))}",
            ha="left",
            va="top",
            fontsize=5.0,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No paired\nsimilarity", ha="center", va="center")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box", anchor="S")
    ax.set_xlabel("Light same-turn\ntuning corr.", fontsize=6.2, labelpad=1.5)
    ax.set_ylabel("Dark same-turn\ntuning corr.", fontsize=6.2, labelpad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def plot_panel_e_encoding_delta_histogram(ax: "Axes", delta_table: Any) -> None:
    """Plot light-epoch TP minus place encoding delta log-likelihoods."""
    epoch_type = "light"
    values = _finite_column_values(
        delta_table[delta_table["epoch_type"].astype(str) == epoch_type],
        "delta_bits_tp_vs_place",
    )
    bin_edges = np.linspace(PANEL_E_X_LIMITS[0], PANEL_E_X_LIMITS[1], 27)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.6, zorder=1)
    if values.size:
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
            zorder=2,
        )
        median_value = float(np.nanmedian(values))
        summary_text = (
            f"{PANEL_QUANT_EPOCH_LABELS[epoch_type]}: "
            f"n={values.size}, med={median_value:.2f}"
        )
        ax.text(
            0.03,
            0.97,
            "Trajectory-specific\nplace better",
            ha="left",
            va="top",
            fontsize=4.8,
            transform=ax.transAxes,
        )
        ax.text(
            0.97,
            0.97,
            "DPP better",
            ha="right",
            va="top",
            fontsize=4.8,
            transform=ax.transAxes,
        )
        ax.text(
            0.98,
            0.76,
            summary_text,
            ha="right",
            va="top",
            fontsize=4.8,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No encoding\nvalues", ha="center", va="center")

    ax.set_xlim(*PANEL_E_X_LIMITS)
    ax.set_xlabel(
        "Delta log likelihood (bits/spike)",
        fontsize=5.8,
        labelpad=1.5,
    )
    ax.set_ylabel("Frac. units", fontsize=6.2, labelpad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.6, length=1.8, pad=1)


def _set_panel_f_error_ylim(ax: "Axes", table: Any) -> None:
    """Set a normalized-error y-limit that preserves the Figure 1 convention."""
    q75_values = _finite_column_values(table, "q75_error") if table is not None else np.asarray([])
    upper = PANEL_F_NORM_ERROR_YLIM[1]
    if q75_values.size:
        upper = max(upper, float(np.nanmax(q75_values)) * 1.08)
    ax.set_ylim(PANEL_F_NORM_ERROR_YLIM[0], upper)


def _plot_panel_f_interval_point(
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


def _style_panel_f_error_axis(ax: "Axes", ylabel: str | None = "Abs. norm. error") -> None:
    """Apply compact normalized-error axis styling for Panel F."""
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PANEL_F_YLABEL_FONTSIZE, labelpad=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=5.0, length=1.5, pad=1)
    ax.tick_params(axis="x", labelsize=4.7, length=0, pad=1)


def _add_panel_f_error_summary_text(ax: "Axes", table: Any) -> None:
    """Add compact median summaries following the Figure 1 Panel F style."""
    if table.empty:
        return
    for row_index, epoch_type in enumerate(PANEL_QUANT_EPOCH_ORDER):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ax.text(
            0.97,
            0.95 - 0.13 * row_index,
            f"{PANEL_QUANT_EPOCH_LABELS[epoch_type]} med. "
            f"{float(row['median_error']):.2f}",
            ha="right",
            va="top",
            fontsize=PANEL_F_SUMMARY_TEXT_FONTSIZE,
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            transform=ax.transAxes,
        )


def _plot_panel_f_place_axis(
    ax: "Axes",
    decoding_error_table: Any,
    *,
    ylabel: str | None = "Abs. norm. error",
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
    ax.set_ylim(*PANEL_F_PLACE_ERROR_YLIM)
    ax.set_title("Trajectory-specific\nplace decoding", fontsize=5.8, pad=1.5)
    if table.empty:
        ax.text(0.5, 0.5, "No place\ndecoding", ha="center", va="center")
        _style_panel_f_error_axis(ax, ylabel=ylabel)
        return

    for position, epoch_type in zip(positions, PANEL_QUANT_EPOCH_ORDER, strict=True):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if rows.empty:
            continue
        row = rows.iloc[0]
        _plot_panel_f_interval_point(
            ax,
            x=float(position),
            q25=float(row["q25_error"]),
            median=float(row["median_error"]),
            q75=float(row["q75_error"]),
            color=PANEL_QUANT_EPOCH_COLORS[epoch_type],
            marker="o",
        )

    _add_panel_f_error_summary_text(ax, table)
    _style_panel_f_error_axis(ax, ylabel=ylabel)


def _plot_panel_f_cross_axis(
    ax: "Axes",
    decoding_error_table: Any,
    *,
    ylabel: str | None = "Abs. norm. error",
) -> None:
    """Plot pooled cross-trajectory TP decoding median/IQR errors by epoch."""
    table = decoding_error_table[
        decoding_error_table["analysis"].astype(str) == "cross_trajectory"
    ].copy()
    comparisons = list(PANEL_F_CROSS_COMPARISONS)
    positions = np.arange(1, len(PANEL_QUANT_EPOCH_ORDER) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [PANEL_QUANT_EPOCH_LABELS[epoch_type] for epoch_type in PANEL_QUANT_EPOCH_ORDER]
    )
    ax.set_xlim(0.5, len(PANEL_QUANT_EPOCH_ORDER) + 0.5)
    ax.set_title("Cross trajectory\ndecoding", fontsize=5.8, pad=1.5)
    if table.empty:
        ax.text(0.5, 0.5, "No cross-trajectory\ndecoding", ha="center", va="center")
        _set_panel_f_error_ylim(ax, table)
        _style_panel_f_error_axis(ax, ylabel=ylabel)
        return

    comparison = comparisons[0][0] if comparisons else None
    for position, epoch_type in zip(positions, PANEL_QUANT_EPOCH_ORDER, strict=True):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if comparison is not None:
            rows = rows[rows["comparison"].astype(str) == comparison]
        if rows.empty:
            continue
        row = rows.iloc[0]
        _plot_panel_f_interval_point(
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

    _add_panel_f_error_summary_text(ax, table)
    _set_panel_f_error_ylim(ax, table)
    _style_panel_f_error_axis(ax, ylabel=ylabel)


def plot_panel_f_decoding_error(ax: "Axes", decoding_error_table: Any) -> None:
    """Plot normalized place and cross-trajectory decoding errors."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    cross_ax = ax.inset_axes(PANEL_F_CROSS_AXIS_BOUNDS)
    place_ax = ax.inset_axes(PANEL_F_PLACE_AXIS_BOUNDS)
    _plot_panel_f_cross_axis(cross_ax, decoding_error_table)
    _plot_panel_f_place_axis(place_ax, decoding_error_table, ylabel=None)


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


def _panel_g_oval_styles(count: int) -> list[dict[str, Any]]:
    """Return orange segment-modulation styles for Panel G track overlays."""
    return [
        {
            "edge_color": PANEL_G_BASIS_LIGHT_COLOR,
            "fill_color": PANEL_G_BASIS_LIGHT_COLOR,
            "fill_alpha": 0.38,
            "linewidth": PANEL_G_SHARED_SCAFFOLD_OVAL_LINEWIDTH,
        }
        for _index in range(count)
    ]


def _draw_panel_g_basis_icon(ax: "Axes") -> None:
    """Draw a compact independent-basis icon."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    line_kwargs = {"color": "black", "linewidth": 1.05, "solid_capstyle": "butt"}
    ax.plot([0.36, 0.64], [0.18, 0.18], **line_kwargs)
    ax.plot([0.45, 0.45], [0.18, 0.86], **line_kwargs)
    ax.plot([0.55, 0.55], [0.18, 0.86], **line_kwargs)


def _draw_panel_g_track(
    ax: "Axes",
    *,
    track_kind: str,
    show_labels: bool = False,
    trajectory_name: str = "center_to_left",
    stimulus_layout: str = "stim1",
    highlighted_segments: Sequence[int] | None = None,
    oval_regions: Sequence[str] | None = None,
    label_fontsize: float = 4.8,
) -> None:
    """Draw one W-track field component for the Panel G model schematic."""
    trajectory_color = PANEL_G_ARROW_COLOR
    if track_kind == "dark":
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            fill_track_black=True,
            show_basis=True,
            basis_segment_styles=_panel_g_basis_styles(
                edge_color=PANEL_G_BASIS_DARK_COLOR,
                fill_color="none",
                fill_alpha=1.0,
                linewidth=0.25,
            ),
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.85,
            arrow_mutation_scale=6.5,
        )
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
            label_fontsize=label_fontsize,
            show_basis=True,
            basis_segment_styles=basis_segment_styles,
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.85,
            arrow_mutation_scale=6.5,
        )
        return

    if track_kind == "segment_modulation":
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_fontsize=label_fontsize,
            show_large_ovals=True,
            oval_regions=list(oval_regions or ["left_arm"]),
            oval_styles=_panel_g_oval_styles(len(list(oval_regions or ["left_arm"]))),
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.78,
            arrow_mutation_scale=6.0,
        )
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
            label_fontsize=label_fontsize,
            show_basis=True,
            basis_segment_styles=_panel_g_basis_styles(
                edge_color=PANEL_G_BASIS_DARK_COLOR,
                fill_color="none",
                fill_alpha=1.0,
                linewidth=PANEL_G_SHARED_SCAFFOLD_BASIS_LINEWIDTH,
            ),
            show_large_ovals=True,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_g_oval_styles(len(selected_oval_regions)),
            arrow_color=trajectory_color,
            track_linewidth=0.55,
            trajectory_linewidth=0.85,
            arrow_mutation_scale=6.5,
        )
        return

    raise ValueError(f"Unknown Panel G track_kind {track_kind!r}.")


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
                    "fill_color": PANEL_G_BASIS_LIGHT_COLOR,
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


def _panel_h_oval_styles(count: int) -> list[dict[str, Any]]:
    """Return thin orange modulation ovals for the scaled Panel H schematic."""
    return [
        {
            "edge_color": PANEL_G_BASIS_LIGHT_COLOR,
            "fill_color": PANEL_G_BASIS_LIGHT_COLOR,
            "fill_alpha": 0.38,
            "linewidth": PANEL_H_SCHEMATIC_OVAL_LINEWIDTH,
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
    label_fontsize: float = 3.1,
) -> None:
    """Draw one thin W-track component for the scaled Panel H swap schematic."""
    trajectory_color = PANEL_G_ARROW_COLOR
    if track_kind == "dark":
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            fill_track_black=True,
            show_basis=True,
            basis_segment_styles=_panel_h_basis_styles(
                edge_color=PANEL_G_BASIS_DARK_COLOR,
                fill_color="none",
                fill_alpha=1.0,
                linewidth=PANEL_H_SCHEMATIC_DARK_BASIS_LINEWIDTH,
            ),
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
        )
        return

    if track_kind == "independent_light":
        basis_segment_styles = (
            _panel_h_basis_styles_with_highlighted_segments(highlighted_segments)
            if highlighted_segments is not None
            else _panel_h_basis_styles(
                edge_color="black",
                fill_color=PANEL_G_BASIS_LIGHT_COLOR,
                fill_alpha=0.76,
                linewidth=PANEL_H_SCHEMATIC_BASIS_LINEWIDTH,
            )
        )
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_fontsize=label_fontsize,
            show_basis=True,
            basis_segment_styles=basis_segment_styles,
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
        )
        return

    if track_kind == "segment_modulation":
        selected_oval_regions = list(oval_regions or ["left_arm"])
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_fontsize=label_fontsize,
            show_large_ovals=True,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_h_oval_styles(len(selected_oval_regions)),
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
        )
        return

    if track_kind == "shared_light":
        selected_oval_regions = list(oval_regions or ["left_arm"])
        draw_w_track_basis_schematic(
            ax,
            trajectory_name=trajectory_name,
            show_labels=show_labels,
            stimulus_layout=stimulus_layout,
            label_fontsize=label_fontsize,
            show_basis=True,
            basis_segment_styles=_panel_h_basis_styles(
                edge_color=PANEL_G_BASIS_DARK_COLOR,
                fill_color="none",
                fill_alpha=1.0,
                linewidth=PANEL_H_SCHEMATIC_DARK_BASIS_LINEWIDTH,
            ),
            show_large_ovals=True,
            oval_regions=selected_oval_regions,
            oval_styles=_panel_h_oval_styles(len(selected_oval_regions)),
            arrow_color=trajectory_color,
            track_linewidth=PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
            trajectory_linewidth=PANEL_H_SCHEMATIC_TRAJECTORY_LINEWIDTH,
            arrow_mutation_scale=PANEL_H_SCHEMATIC_ARROW_SCALE,
        )
        return

    raise ValueError(f"Unknown Panel H track_kind {track_kind!r}.")


def _plot_panel_g_architecture_schematic(ax: "Axes") -> None:
    """Draw the compact dark/light GLM architecture schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    def _bounds_from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> list[float]:
        return [center_x - width / 2.0, center_y - height / 2.0, width, height]

    dark_center_x = 0.25
    light_center_x = 0.78
    independent_basis_center_x = 0.5 * (dark_center_x + light_center_x)
    top_center_y = 0.715
    bottom_center_y = 0.255
    dark_bounds = {
        "width": 0.16,
        "height": 0.31,
    }
    light_bounds = {
        "width": 0.18,
        "height": 0.34,
    }

    ax.text(dark_center_x, 0.98, "Dark field", ha="center", va="top", fontsize=5.8)
    ax.text(light_center_x, 0.98, "Light field", ha="center", va="top", fontsize=5.8)
    ax.text(
        0.08,
        0.72,
        "Independent\nmodel",
        ha="center",
        va="center",
        fontsize=4.1,
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.28,
        "Shared-scaffold\nmodel",
        ha="center",
        va="center",
        fontsize=3.8,
        fontweight="bold",
    )
    ax.text(
        independent_basis_center_x,
        0.78,
        "Independent\nbasis functions",
        ha="center",
        va="center",
        fontsize=3.7,
    )
    ax.text(
        0.55,
        0.47,
        "Segment-specific modulation",
        ha="center",
        va="center",
        fontsize=3.7,
    )

    _draw_panel_g_track(
        ax.inset_axes(
            _bounds_from_center(
                dark_center_x,
                top_center_y,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
    )
    basis_width = 0.08
    basis_ax = ax.inset_axes(
        [
            independent_basis_center_x - basis_width / 2.0,
            0.57,
            basis_width,
            0.15,
        ]
    )
    _draw_panel_g_basis_icon(basis_ax)
    _draw_panel_g_track(
        ax.inset_axes(
            _bounds_from_center(
                light_center_x,
                top_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="independent_light",
        show_labels=True,
    )

    _draw_panel_g_track(
        ax.inset_axes(
            _bounds_from_center(
                dark_center_x,
                bottom_center_y,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
    )
    ax.text(0.39, 0.25, "+", ha="center", va="center", fontsize=8.0)
    _draw_panel_g_track(
        ax.inset_axes([0.44, 0.08, 0.18, 0.34]),
        track_kind="segment_modulation",
        show_labels=True,
    )
    ax.annotate(
        "",
        xy=(0.67, 0.26),
        xytext=(0.62, 0.26),
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
    )
    _draw_panel_g_track(
        ax.inset_axes(
            _bounds_from_center(
                light_center_x,
                bottom_center_y,
                light_bounds["width"],
                light_bounds["height"],
            )
        ),
        track_kind="shared_light",
        show_labels=True,
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
            color=PANEL_G_EXAMPLE_MODEL_COLORS[model_name],
            linewidth=0.75,
            label=PANEL_G_MODEL_LABELS[model_name],
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
) -> None:
    """Plot two example cells below the Panel G schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No GLM\nexamples", ha="center", va="center", fontsize=5.0)
        return

    column_width = 0.46
    column_gap = 0.04
    for example_index, example in enumerate(examples[:2], start=1):
        column_left = (example_index - 1) * (column_width + column_gap)
        y_max = _panel_g_examples_y_max([example])
        icon_ax = ax.inset_axes([column_left - 0.045, 0.23, 0.085, 0.26])
        draw_w_track_schematic(
            icon_ax,
            trajectory_name=example["trajectory"],
            arrow_color=PANEL_C_TRAJECTORY_COLORS[example["trajectory"]],
            track_linewidth=0.42,
            trajectory_linewidth=0.65,
            arrow_mutation_scale=5.4,
            fill_track=False,
        )
        ax.text(
            column_left + 0.29,
            0.93,
            f"Example {example_index}",
            ha="center",
            va="top",
            fontsize=5.6,
            transform=ax.transAxes,
        )
        plot_left = column_left + 0.12
        field_width = 0.14
        dark_ax = ax.inset_axes([plot_left, 0.05, field_width, 0.58])
        light_ax = ax.inset_axes(
            [plot_left + field_width + 0.035, 0.05, field_width, 0.58]
        )
        dark_ax.set_facecolor(PANEL_C_DARK_EPOCH_BACKGROUND)
        _plot_panel_g_example_field_axis(
            dark_ax,
            example,
            epoch_key="dark",
            y_max=y_max,
            show_ylabel=True,
            show_title=True,
        )
        _plot_panel_g_example_field_axis(
            light_ax,
            example,
            epoch_key="light",
            y_max=y_max,
            show_title=True,
            show_legend=example_index == 2,
            legend_loc="center left",
            legend_bbox_to_anchor=(1.02, 0.5),
        )
        dark_ax.set_xlabel("Norm. path progression", fontsize=3.7, labelpad=0.6)
        light_ax.set_xlabel("Norm. path progression", fontsize=3.7, labelpad=0.6)


def plot_panel_g_model_architecture(
    ax: "Axes",
    examples: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Plot Panel G example GLM fits and the model schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    schematic_ax = ax.inset_axes(
        [
            0.0,
            1.0 - PANEL_G_SCHEMATIC_HEIGHT_FRACTION,
            1.0,
            PANEL_G_SCHEMATIC_HEIGHT_FRACTION,
        ]
    )
    _plot_panel_g_architecture_schematic(schematic_ax)
    example_ax = ax.inset_axes(
        [
            0.0,
            0.02,
            1.0,
            PANEL_G_EXAMPLE_HEIGHT_FRACTION,
        ]
    )
    _plot_panel_g_example_columns(example_ax, [] if examples is None else examples)


def _plot_panel_g_curve_schematic(ax: "Axes", *, bump: bool) -> None:
    """Plot a compact dark/light model curve schematic."""
    x = np.linspace(0.0, 1.0, 100)
    dark_curve = 0.25 + 0.55 * np.exp(-0.5 * ((x - 0.32) / 0.12) ** 2)
    if bump:
        gain = 0.45 * np.exp(-0.5 * ((x - 0.73) / 0.10) ** 2)
        light_curve = dark_curve * (1.0 + gain)
    else:
        light_curve = 0.18 + 0.62 * np.exp(-0.5 * ((x - 0.72) / 0.13) ** 2)
    ax.plot(x, dark_curve, color=PANEL_QUANT_EPOCH_COLORS["dark"], linewidth=0.75)
    ax.plot(x, light_curve, color=PANEL_QUANT_EPOCH_COLORS["light"], linewidth=0.75)
    if bump:
        for boundary in SEGMENT_BOUNDARIES:
            ax.axvline(boundary, color=SEGMENT_BOUNDARY_COLOR, linewidth=0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_panel_g_model_schematic(ax: "Axes") -> None:
    """Draw compact schematics for the independent and segment-bump models."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    blocks = (
        ("visual", 0.03, "dark field + light field"),
        ("task_segment_bump", 0.53, "dark field x segment gain"),
    )
    for model_name, x0, subtitle in blocks:
        ax.text(
            x0,
            0.94,
            PANEL_G_MODEL_LABELS[model_name],
            ha="left",
            va="top",
            fontsize=5.6,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            x0,
            0.78,
            subtitle,
            ha="left",
            va="top",
            fontsize=4.5,
            color="0.25",
            transform=ax.transAxes,
        )
        track_ax = ax.inset_axes([x0, 0.20, 0.16, 0.42])
        draw_w_track_schematic(
            track_ax,
            trajectory_name="center_to_left",
            arrow_color=PANEL_G_MODEL_COLORS[model_name],
            track_linewidth=0.45,
            trajectory_linewidth=0.7,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )
        curve_ax = ax.inset_axes([x0 + 0.20, 0.20, 0.22, 0.44])
        _plot_panel_g_curve_schematic(
            curve_ax,
            bump=model_name == "task_segment_bump",
        )


def _plot_panel_g_prediction_axis(
    ax: "Axes",
    example: dict[str, Any],
    model_name: str,
    *,
    y_max: float,
    show_ylabel: bool = False,
) -> None:
    """Plot one example dark/light GLM prediction for one model."""
    model_payload = example["models"][model_name]
    tp_grid = np.asarray(example["tp_grid"], dtype=float)
    ax.plot(
        tp_grid,
        model_payload["dark_hz"],
        color=PANEL_QUANT_EPOCH_COLORS["dark"],
        linewidth=0.75,
        label="Dark",
    )
    ax.plot(
        tp_grid,
        model_payload["light_hz"],
        color=PANEL_QUANT_EPOCH_COLORS["light"],
        linewidth=0.75,
        label="Light",
    )
    for boundary in np.asarray(example["segment_edges"], dtype=float)[1:-1]:
        ax.axvline(boundary, color=SEGMENT_BOUNDARY_COLOR, linewidth=0.45, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    ax.set_title(PANEL_G_MODEL_LABELS[model_name], fontsize=5.2, pad=1.0)
    if show_ylabel:
        ax.set_ylabel("Hz", fontsize=4.8, labelpad=1)
    ax.tick_params(labelsize=4.5, length=1.2, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_g_example_y_max(example: dict[str, Any]) -> float:
    """Return a compact y-limit for one GLM example."""
    values = []
    for model_payload in example["models"].values():
        values.extend(
            [
                np.asarray(model_payload["dark_hz"], dtype=float),
                np.asarray(model_payload["light_hz"], dtype=float),
            ]
        )
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    if finite.size == 0:
        return 1.0
    return max(1.0, float(np.ceil(np.nanmax(finite))))


def plot_panel_g_dark_light_glm(
    ax: "Axes",
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot dark-light GLM schematics and example selected fits."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    schematic_ax = ax.inset_axes([0.02, 0.66, 0.96, 0.30])
    plot_panel_g_model_schematic(schematic_ax)

    if not examples:
        ax.text(0.5, 0.30, "No GLM examples", ha="center", va="center")
        return

    row_height = 0.24
    row_y_positions = [0.36, 0.08]
    for example_index, (example, y0) in enumerate(
        zip(examples, row_y_positions, strict=False),
        start=1,
    ):
        label = (
            f"Ex. {example_index}: {example['animal_name']} "
            f"{example['region'].upper()} cell {example['unit_id']}\n"
            f"{PANEL_TRAJECTORY_LABELS[example['trajectory']]}"
        )
        ax.text(0.02, y0 + row_height * 0.72, label, ha="left", va="top", fontsize=4.7)
        track_ax = ax.inset_axes([0.14, y0 + 0.07, 0.08, 0.12])
        draw_w_track_schematic(
            track_ax,
            trajectory_name=example["trajectory"],
            arrow_color=PANEL_C_TRAJECTORY_COLORS[example["trajectory"]],
            track_linewidth=0.45,
            trajectory_linewidth=0.7,
            arrow_mutation_scale=5.8,
            fill_track=False,
        )
        y_max = _panel_g_example_y_max(example)
        visual_ax = ax.inset_axes([0.27, y0, 0.28, row_height])
        bump_ax = ax.inset_axes([0.66, y0, 0.28, row_height])
        _plot_panel_g_prediction_axis(
            visual_ax,
            example,
            "visual",
            y_max=y_max,
            show_ylabel=True,
        )
        _plot_panel_g_prediction_axis(
            bump_ax,
            example,
            "task_segment_bump",
            y_max=y_max,
        )
        if example_index == 1:
            visual_ax.legend(
                frameon=False,
                fontsize=4.2,
                handlelength=1.0,
                borderpad=0.1,
                loc="upper right",
            )
        if example_index == len(examples):
            visual_ax.set_xlabel("Norm. task progression", fontsize=4.6, labelpad=1)
            bump_ax.set_xlabel("Norm. task progression", fontsize=4.6, labelpad=1)


def _draw_panel_h_swap_schematic(ax: "Axes") -> None:
    """Draw a scaled full-layout train/predict swap schematic for Panel H."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    def _bounds_from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> list[float]:
        return [center_x - width / 2.0, center_y - height / 2.0, width, height]

    train_center_x = 0.36
    predict_center_x = 0.78
    light_bounds = {"width": 0.38, "height": 0.23}
    dark_bounds = {"width": 0.34, "height": 0.21}

    ax.text(train_center_x, 0.98, "Train: AB", ha="center", va="top", fontsize=5.8)
    ax.text(predict_center_x, 0.98, "Predict: BA", ha="center", va="top", fontsize=5.8)
    ax.text(
        0.08,
        0.72,
        "Independent\nmodel",
        ha="center",
        va="center",
        fontsize=4.1,
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.28,
        "Shared-scaffold\nmodel",
        ha="center",
        va="center",
        fontsize=3.8,
        fontweight="bold",
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                train_center_x,
                0.725,
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
                0.725,
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
        predict_center_x,
        0.55,
        "\"Light activity is like the other arm\nwith the same visual landmark\"",
        ha="center",
        va="top",
        fontsize=3.0,
        linespacing=0.9,
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                train_center_x,
                0.475,
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
                0.195,
                dark_bounds["width"],
                dark_bounds["height"],
            )
        ),
        track_kind="dark",
        trajectory_name="center_to_right",
    )
    _draw_panel_h_track(
        ax.inset_axes(
            _bounds_from_center(
                predict_center_x,
                0.335,
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
        predict_center_x,
        0.05,
        "\"Light activity is like the same arm\ndark activity with visual modulation\"",
        ha="center",
        va="bottom",
        fontsize=3.0,
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
    """Return compact fraction-positive and median text for Panel H."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n/a >0, med. n/a"
    fraction_positive = float(np.mean(values > 0.0))
    median = float(np.median(values))
    return f"{fraction_positive:.0%} >0, med. {median:.2f}"


def _plot_panel_h_delta_axis(
    ax: "Axes",
    swap_delta_table: Any,
    *,
    trajectory_type: str | None = None,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
) -> None:
    """Plot held-out 06_r3 segment-bump minus independent delta LL values."""
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
        hist_kwargs["edgecolor"] = "none"
        hist_kwargs["linewidth"] = 0.0
        ax.hist(
            values,
            bins=bin_edges,
            weights=_fraction_histogram_weights(values),
            color=PANEL_G_MODEL_COLORS["task_segment_bump"],
            **hist_kwargs,
            zorder=2,
        )
        ax.text(
            0.62,
            0.92,
            _format_panel_h_delta_summary(values),
            ha="left",
            va="top",
            fontsize=3.4,
            color=PANEL_G_MODEL_COLORS["task_segment_bump"],
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


def _plot_panel_h_delta_grid(ax: "Axes", swap_delta_table: Any) -> None:
    """Plot Panel H delta LL histograms split by trajectory."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    grid_bounds = (
        (0.08, 0.55, 0.40, 0.31),
        (0.56, 0.55, 0.40, 0.31),
        (0.08, 0.12, 0.40, 0.31),
        (0.56, 0.12, 0.40, 0.31),
    )
    for trajectory_index, (trajectory_type, bounds) in enumerate(
        zip(PANEL_H_DELTA_TRAJECTORIES, grid_bounds, strict=True)
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
            arrow_color=PANEL_C_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.42,
            trajectory_linewidth=0.68,
            arrow_mutation_scale=5.4,
            fill_track=False,
        )
        delta_ax = ax.inset_axes(bounds)
        _plot_panel_h_delta_axis(
            delta_ax,
            swap_delta_table,
            trajectory_type=trajectory_type,
            show_xticklabels=trajectory_index >= 2,
            show_yticklabels=trajectory_index in (0, 2),
        )

    ax.text(
        0.53,
        0.02,
        "Delta log likelihood (bits/spike)",
        ha="center",
        va="bottom",
        fontsize=4.3,
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.52,
        "Frac. traj-units",
        ha="left",
        va="center",
        rotation=90,
        fontsize=4.3,
        transform=ax.transAxes,
    )


def _plot_panel_h_switched_segment_example(
    ax: "Axes",
    swap_example: dict[str, Any] | None,
    *,
    example_label: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = True,
    show_xticklabels: bool = True,
) -> None:
    """Plot one empirical and model-predicted switched segment."""
    if swap_example is None:
        ax.text(0.5, 0.5, "No switch\nexample", ha="center", va="center", fontsize=5.0)
        ax.axis("off")
        return

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
    for model_name in ("visual", "task_segment_bump"):
        values.append(np.asarray(swap_example["models"][model_name], dtype=float)[grid_mask])
    finite_values = [value[np.isfinite(value)] for value in values if value.size]
    y_max = 1.0 if not finite_values else max(1.0, float(np.ceil(np.nanmax(np.concatenate(finite_values)))))

    ax.plot(
        observed_position[observed_mask],
        observed_rate[observed_mask],
        color=PANEL_G_EMPIRICAL_COLOR,
        linewidth=0.9,
        label="Emp.",
        zorder=4,
    )
    for model_name in ("visual", "task_segment_bump"):
        ax.plot(
            tp_grid[grid_mask],
            np.asarray(swap_example["models"][model_name], dtype=float)[grid_mask],
            color=PANEL_G_EXAMPLE_MODEL_COLORS[model_name],
            linewidth=0.8,
            label=PANEL_G_MODEL_LABELS[model_name],
            zorder=3,
        )
    ax.axvspan(start, end, color=PANEL_G_BASIS_LIGHT_COLOR, alpha=0.10, linewidth=0)
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
        ax.text(
            0.04,
            0.06,
            delta_label,
            ha="left",
            va="bottom",
            fontsize=3.8,
            transform=ax.transAxes,
        )
    ax.set_title(
        "Example" if example_label is None else example_label,
        fontsize=5.0,
        pad=0.8,
    )
    icon_ax = ax.inset_axes([-0.35, 0.32, 0.28, 0.34])
    draw_w_track_schematic(
        icon_ax,
        trajectory_name=swap_example["trajectory"],
        arrow_color=PANEL_C_TRAJECTORY_COLORS[swap_example["trajectory"]],
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
        ax.legend(frameon=False, fontsize=3.4, handlelength=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.1, length=1.2, pad=0.7)


def plot_panel_h_swap_delta(
    ax: "Axes",
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
) -> None:
    """Plot the Panel H swap schematic, delta LL, and switched-segment examples."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    schematic_ax = ax.inset_axes(PANEL_H_SCHEMATIC_AXIS_BOUNDS)
    delta_ax = ax.inset_axes(PANEL_H_DELTA_AXIS_BOUNDS)
    example_axes = [
        ax.inset_axes([0.82, 0.56, 0.17, 0.34]),
        ax.inset_axes([0.82, 0.11, 0.17, 0.34]),
    ]
    _draw_panel_h_swap_schematic(schematic_ax)
    _plot_panel_h_delta_grid(delta_ax, swap_delta_table)
    examples = _coerce_panel_h_swap_examples(swap_examples)[:2]
    for example_index, example_ax in enumerate(example_axes):
        example = examples[example_index] if example_index < len(examples) else None
        _plot_panel_h_switched_segment_example(
            example_ax,
            example,
            example_label=f"Example {example_index + 1}",
            show_xlabel=example_index == 1,
            show_ylabel=example_index == 0,
            show_legend=example_index == 0,
            show_xticklabels=example_index == 1,
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
    panel_b_cache_dir: Path | None = None,
    refresh_panel_b_cache: bool = False,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> Path:
    """Build and save Figure 3."""
    import matplotlib.pyplot as plt

    panel_b_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_b_cache_dir is None
        else Path(panel_b_cache_dir)
    )
    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_quant_payload = load_panel_quantification_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_glm_payload = load_panel_glm_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

    apply_paper_style()
    fig_height_mm = DEFAULT_PANEL_A_HEIGHT_MM + (
        DEFAULT_PANEL_BC_HEIGHT_MM * max(len(regions), 1)
    ) + DEFAULT_PANEL_DEF_HEIGHT_MM + DEFAULT_PANEL_GH_HEIGHT_MM
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            DEFAULT_PANEL_A_HEIGHT_MM,
            DEFAULT_PANEL_BC_HEIGHT_MM * max(len(regions), 1),
            DEFAULT_PANEL_DEF_HEIGHT_MM,
            DEFAULT_PANEL_GH_HEIGHT_MM,
        ],
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    middle_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[
            DEFAULT_PANEL_C_WIDTH_FRACTION,
            DEFAULT_PANEL_B_WIDTH_FRACTION,
        ],
    )
    panel_c_axis = fig.add_subplot(middle_grid[0, 0])
    panel_b = setup_light_heatmap_panel(
        fig,
        middle_grid[0, 1],
        regions=regions,
    )
    bottom_grid = outer_grid[2, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=PANEL_DEF_WIDTH_RATIOS,
        wspace=PANEL_DEF_WSPACE,
    )
    panel_d_container_axis = fig.add_subplot(bottom_grid[0, 0])
    panel_e_container_axis = fig.add_subplot(bottom_grid[0, 1])
    panel_f_axis = fig.add_subplot(bottom_grid[0, 2])
    panel_d_container_axis.axis("off")
    panel_e_container_axis.axis("off")
    panel_d_axis = panel_d_container_axis.inset_axes(PANEL_D_AXIS_BOUNDS)
    panel_e_axis = panel_e_container_axis.inset_axes(PANEL_E_AXIS_BOUNDS)
    glm_grid = outer_grid[3, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_GH_WIDTH_RATIOS,
    )
    panel_g_axis = fig.add_subplot(glm_grid[0, 0])
    panel_h_axis = fig.add_subplot(glm_grid[0, 1])

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
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=refresh_panel_example_cache,
    )
    plot_panel_a_example(panel_a_axis, panel_a_example)

    colorbar = None
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
        panel_b_cache_dir=panel_b_cache_dir,
        refresh_panel_b_cache=refresh_panel_b_cache,
    )
    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=panel_b["heatmap_axes"].ravel().tolist(),
            shrink=0.24,
            pad=PANEL_C_COLORBAR_PAD,
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
    draw_neuron_scale_bar(
        panel_b["heatmap_axes"][-1, -1],
        x=PANEL_C_NEURON_SCALE_BAR_X,
    )

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
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
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
        panel_quant_payload["decoding_error"],
    )
    plot_panel_g_model_architecture(
        panel_g_axis,
        panel_glm_payload["dark_light_examples"],
    )
    plot_panel_h_swap_delta(
        panel_h_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
    )

    fig.canvas.draw()
    panel_c_axes = [
        panel_b["corner_axis"],
        *panel_b["tuning_schematic_axes"].ravel().tolist(),
        *panel_b["order_schematic_axes"].ravel().tolist(),
        *panel_b["heatmap_axes"].ravel().tolist(),
    ]
    if colorbar is not None:
        panel_c_axes.append(colorbar.ax)
    shift_axes_horizontally(panel_c_axes, PANEL_C_HORIZONTAL_SHIFT)
    add_centered_axis_text(
        fig,
        panel_b["tuning_schematic_axes"],
        "Tuning",
        y_offset=-0.026,
        fontsize=8.0,
    )
    add_centered_axis_text(
        fig,
        panel_b["order_schematic_axes"],
        "Order",
        y_offset=-0.006,
        rotation=90,
    )
    label_axis(panel_a_axis, "A", x=-0.02, y=1.00)
    label_axis(panel_b["corner_axis"], "C", x=-0.12, y=0.52)
    label_axis(panel_c_axis, "B", x=-0.07, y=1.267)
    label_axis(panel_d_container_axis, "D", x=0.00, y=0.98)
    label_axis(panel_e_container_axis, "E", x=0.00, y=0.98)
    label_axis(panel_f_axis, "F", x=0.00, y=0.98)
    label_axis(panel_g_axis, "G", x=-0.035, y=1.02)
    label_axis(panel_h_axis, "H", x=-0.06, y=1.02)
    panel_c_axis.text(
        0.5,
        PANEL_B_TITLE_Y,
        "Example DPP cells in\ndifferent visual conditions",
        ha="center",
        va="top",
        fontsize=7.2,
        linespacing=1.0,
        transform=panel_c_axis.transAxes,
        clip_on=False,
    )
    panel_a_axis.set_title(
        "Example visual cell in different visual conditions",
        fontsize=8,
        pad=2,
    )
    panel_d_axis.set_title("Same-turn tuning similarity", fontsize=8, pad=5)
    panel_e_axis.set_title("Encoding comparison", fontsize=8, pad=2)
    panel_g_axis.set_title(
        "Two possible models that relate dark and light activity",
        fontsize=8,
        pad=2,
    )
    panel_h_axis.set_title(
        "Predicting activity in held-out light epoch",
        fontsize=8,
        pad=2,
    )

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
        "--panel-b-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel B heatmap matrices. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-b-cache",
        action="store_true",
        help="Recompute Panel B and overwrite its cache even when a matching cache exists.",
    )
    parser.add_argument(
        "--panel-example-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel A/C example-cell rasters and rate curves. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help=(
            "Recompute Panel A/C example-cell data and overwrite matching caches."
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
    panel_b_cache_dir = (
        args.panel_b_cache_dir
        if args.panel_b_cache_dir is not None
        else args.output_dir / "cache"
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
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
        panel_b_cache_dir=panel_b_cache_dir,
        refresh_panel_b_cache=args.refresh_panel_b_cache,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
    )


if __name__ == "__main__":
    main()
