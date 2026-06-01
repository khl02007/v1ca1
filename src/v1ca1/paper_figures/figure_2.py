from __future__ import annotations

"""Generate Figure 2 panels for CA1 ripple modulation of V1 activity."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_spikes_by_region,
)
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    make_dataset_id,
    make_figure_2_epoch_ids,
    normalize_dataset_id,
)
from v1ca1.paper_figures.style import (
    ANIMAL_COLORS,
    COMPACT_HISTOGRAM_KWARGS,
    EPOCH_TYPE_COLORS,
    HISTOGRAM_KWARGS,
    MODEL_CLASS_COLORS,
    NEUTRAL_COLORS,
    REGION_COLORS,
    SCHEMATIC_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.xcorr.screen_xcorr import (
    PAIR_STATUS_VALID,
    STATE_CHOICES as XCORR_STATE_CHOICES,
    format_xcorr_settings_suffix,
    get_state_output_parts,
    order_ca1_units_by_best_partner,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_FIGURE_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
DEFAULT_OUTPUT_NAME = "figure_2"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_EXAMPLE_DATASET = ("L14", "20240611", "08_r4")
DEFAULT_XCORR_DATASET = ("L15", "20241121", "02_r1")
DEFAULT_PANEL_B_SCHEMATIC_DATASET = ("L15", "20241121", "10_r5")
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_FIGURE_HEIGHT_MM = 72.0
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
DEFAULT_REGIONS = REGIONS
RIPPLE_EVENT_RELATIVE_PATH = Path("ripple") / "ripple_times.parquet"
RIPPLE_LFP_RELATIVE_DIR = Path("ripple") / "ripple_channels_lfp"
RIPPLE_MODULATION_RELATIVE_DIR = Path("ripple") / "ripple_modulation"
RIPPLE_GLM_RELATIVE_DIR = Path("ripple_glm")
RIPPLE_DECODING_COMPARISON_RELATIVE_DIR = Path("ripple_decoding_comparison")
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
TUNING_ANALYSIS_RELATIVE_DIR = Path("task_progression") / "tuning_analysis"
MOTOR_NESTED_CV_RELATIVE_DIR = Path("task_progression") / "motor" / "nested_lap_cv"
DEFAULT_RIPPLE_THRESHOLD_ZSCORE = 2.0
DEFAULT_BIN_SIZE_S = 20e-3
DEFAULT_TIME_BEFORE_S = 0.5
DEFAULT_TIME_AFTER_S = 0.5
DEFAULT_RESPONSE_WINDOW = (0.0, 0.1)
DEFAULT_BASELINE_WINDOW = (-0.5, -0.3)
DEFAULT_HEATMAP_NORMALIZE = "max"
DEFAULT_REGION_LABEL = "all_regions"
DEFAULT_RIPPLE_WINDOW_S = 0.2
DEFAULT_RIPPLE_WINDOW_OFFSET_S = 0.0
DEFAULT_RIPPLE_SELECTION = "allripples"
DEFAULT_FIGURE_2_GLM_RIPPLE_SELECTION = "single"
DEFAULT_RIDGE_STRENGTH = 1e-1
SOURCE_PREDICTOR_MODE_UNIT_VECTOR = "unit_vector"
SOURCE_PREDICTOR_MODE_MEAN_ACTIVITY = "mean_activity"
SOURCE_PREDICTOR_MODE_CHOICES = (
    SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
    SOURCE_PREDICTOR_MODE_MEAN_ACTIVITY,
)
SOURCE_PREDICTOR_MODE_FILENAME_TOKENS = {
    SOURCE_PREDICTOR_MODE_UNIT_VECTOR: "",
    SOURCE_PREDICTOR_MODE_MEAN_ACTIVITY: "mean_ca1",
}
DEFAULT_LFP_TIME_BEFORE_S = 0.080
DEFAULT_LFP_TIME_AFTER_S = 0.160
DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S = 0.080
DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S = 0.220
DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION = 5
DEFAULT_PANEL_B_SCHEMATIC_TARGET_DURATION_S = 0.150
PANEL_B_SCHEMATIC_CACHE_VERSION = 4
DEFAULT_XCORR_STATE = "ripple"
DEFAULT_XCORR_BIN_SIZE_S = 0.005
DEFAULT_XCORR_MAX_LAG_S = 0.5
DEFAULT_XCORR_TOP_CA1_UNITS = 4
DEFAULT_XCORR_DISPLAY_VMAX = 5.0
DEFAULT_XCORR_LAG_WINDOW_S = (-0.3, 0.3)
DEFAULT_PANEL_D_REGION = "v1"
DEFAULT_PANEL_D_ENCODING_N_FOLDS = 5
DEFAULT_PANEL_D_PLACE_BIN_SIZE_CM = 4.0
DEFAULT_PANEL_D_ENCODING_SOURCE_COLUMN = "delta_bits_generalized_place_vs_tp"
DEFAULT_PANEL_D_MOTOR_DELTA_METRIC = "dll_motor_tp_vs_motor_bits_per_spike"
PANEL_D_MOTOR_PREFERRED_FILENAME_TOKEN = "_zscore_"
DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC = "correlation"
DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL = "pooled_same_turn"
NEURON_SCALE_BAR_COUNT = 100
PANEL_E_CATEGORICAL_METRICS = (
    ("place", "turn_group"),
    ("place", "arm_identity"),
)
PANEL_E_EPOCH_ORDER = ("light", "dark")
PANEL_E_METRIC_LABELS = {
    ("place", "turn_group"): "Turn group",
    ("place", "arm_identity"): "Arm",
}
PANEL_E_CHANCE_LEVELS = {
    "turn_group": 0.5,
    "arm_identity": 1.0 / 3.0,
}
PANEL_E_GLM_TARGET_WINDOW_OFFSETS_S = (-0.4, -0.2, 0.0, 0.2)
PANEL_E_GLM_TARGET_WINDOW_S = DEFAULT_RIPPLE_WINDOW_S
PANEL_E_GLM_SOURCE_WINDOW_OFFSET_S = 0.0
PANEL_E_GLM_EPOCH_ORDER = ("dark",)
HEATMAP_EPOCH_ORDER = ("light", "dark", "sleep")
PANEL_A_EPOCH_ORDER = ("dark",)
PANEL_C_EPOCH_ORDER = ("dark",)
PANEL_D_EPOCH_ORDER = ("dark",)
HEATMAP_EPOCH_LABELS = {
    "light": "Light run",
    "dark": "Dark run",
    "sleep": "Sleep",
}
XCORR_RELATIVE_DIR = Path("xcorr") / "screen_pairs"
XCORR_SUMMARY_FILENAME = "xcorr_summary.parquet"
XCORR_DATASET_FILENAME = "xcorr.nc"
MODEL_COLOR = MODEL_CLASS_COLORS["visual"]
GLM_EPOCH_COLORS = EPOCH_TYPE_COLORS
NONSIGNIFICANT_COLOR = NEUTRAL_COLORS["nonsignificant"]
SIGNIFICANCE_P_VALUE = 0.05
PANEL_C_SIGNIFICANCE_P_VALUE = 0.05
PANEL_BC_SIGNIFICANT_UNIT_COLOR = REGION_COLORS["v1"]
PANEL_C_SOURCE_COMPARISON_COLOR = PANEL_BC_SIGNIFICANT_UNIT_COLOR
PANEL_D_SIGNIFICANCE_P_VALUE = PANEL_C_SIGNIFICANCE_P_VALUE
PANEL_D_MIN_DEVIANCE_EXPLAINED = 0.0
PANEL_D_POINT_COLOR = REGION_COLORS["v1"]
PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ = 0.5
PANEL_D_DARK_ACTIVITY_COLORS = {
    "inactive": EPOCH_TYPE_COLORS["light"],
    "active": SCHEMATIC_COLORS["dark_basis"],
}
PANEL_CD_DEVIANCE_EXPLAINED_LIMITS = (-0.1, 0.3)
PANEL_B_DEVIANCE_EXPLAINED_LIMITS = (-0.1, 0.3)
PANEL_C_SOURCE_COMPARISON_LIMITS = (-0.1, 0.3)
DARK_MOVEMENT_FR_CACHE_VERSION = 1
DARK_MOVEMENT_FR_CACHE_COLUMNS = ("unit", "dark_firing_rate_hz")


def parse_dataset_id(value: str) -> DatasetId:
    """Parse one `animal:date[:epoch]` data-set identifier."""
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(
            "Data sets must be specified as animal:date or animal:date:epoch, "
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


def get_ripple_event_path(data_root: Path, animal_name: str, date: str) -> Path:
    """Return the saved ripple event parquet path for one session."""
    return get_dataset_analysis_path(data_root, animal_name, date) / RIPPLE_EVENT_RELATIVE_PATH


def get_ripple_lfp_path(
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
) -> Path:
    """Return the ripple-band LFP NetCDF path for one epoch."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / RIPPLE_LFP_RELATIVE_DIR
        / f"{epoch}_ripple_channels_lfp.nc"
    )


def format_output_value(value: float | str) -> str:
    """Return a filename-safe value using the ripple-modulation convention."""
    if isinstance(value, str):
        return value
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def build_ripple_modulation_output_stem(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region_label: str,
    ripple_threshold_zscore: float,
    bin_size_s: float,
    time_before_s: float,
    time_after_s: float,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
    heatmap_normalize: str,
) -> str:
    """Return the shared ripple-modulation filename stem for one epoch."""
    return (
        f"{animal_name}_{date}_{epoch}_{region_label}"
        f"_thr_{format_output_value(ripple_threshold_zscore)}"
        f"_bin_{format_output_value(bin_size_s)}"
        f"_tb_{format_output_value(time_before_s)}"
        f"_ta_{format_output_value(time_after_s)}"
        f"_resp_{format_output_value(response_window[0])}_{format_output_value(response_window[1])}"
        f"_base_{format_output_value(baseline_window[0])}_{format_output_value(baseline_window[1])}"
        f"_norm_{heatmap_normalize}"
    )


def get_ripple_modulation_paths(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region_label: str = DEFAULT_REGION_LABEL,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
) -> dict[str, Path]:
    """Return ripple-modulation parquet paths without creating directories."""
    stem = build_ripple_modulation_output_stem(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region_label=region_label,
        ripple_threshold_zscore=ripple_threshold_zscore,
        bin_size_s=bin_size_s,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        response_window=response_window,
        baseline_window=baseline_window,
        heatmap_normalize=heatmap_normalize,
    )
    data_dir = get_dataset_analysis_path(data_root, animal_name, date) / RIPPLE_MODULATION_RELATIVE_DIR
    return {
        "peri_ripple_firing_rate": data_dir / f"{stem}_peri_ripple_firing_rate.parquet",
        "summary": data_dir / f"{stem}_summary.parquet",
    }


def _format_window_suffix_value(value: float) -> str:
    """Return one filesystem-friendly encoded float value."""
    abs_text = f"{abs(float(value)):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"m{abs_text}" if float(value) < 0 else abs_text


def format_ripple_window_suffix(
    ripple_window_s: float,
    *,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> str:
    """Return the ripple-GLM filename suffix for one window setup."""
    window_suffix = f"rw_{_format_window_suffix_value(ripple_window_s)}s"
    if np.isclose(
        float(ripple_window_offset_s),
        DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        rtol=1e-12,
        atol=1e-12,
    ):
        return window_suffix
    return f"{window_suffix}_off_{_format_window_suffix_value(ripple_window_offset_s)}s"


def format_glm_model_window_suffix(
    *,
    source_window_s: float,
    source_window_offset_s: float,
    target_window_s: float,
    target_window_offset_s: float,
) -> str:
    """Return the ripple-GLM filename suffix for source and target windows."""
    source_suffix = format_ripple_window_suffix(
        source_window_s,
        ripple_window_offset_s=source_window_offset_s,
    )
    target_suffix = format_ripple_window_suffix(
        target_window_s,
        ripple_window_offset_s=target_window_offset_s,
    )
    if source_suffix == target_suffix:
        return target_suffix
    return f"src_{source_suffix}_tgt_{target_suffix}"


def format_ridge_strength_suffix(ridge_strength: float) -> str:
    """Return the ripple-GLM filename suffix for one ridge strength."""
    ridge_text = f"{float(ridge_strength):.0e}"
    mantissa, exponent = ridge_text.split("e")
    exponent = exponent.lstrip("+")
    if exponent.startswith("-0"):
        exponent = f"-{exponent[2:]}"
    elif exponent.startswith("0"):
        exponent = exponent[1:]
    return f"ridge_{mantissa}e{exponent}"


def validate_source_predictor_mode(source_predictor_mode: str) -> str:
    """Return a validated ripple-GLM source predictor mode."""
    if source_predictor_mode not in SOURCE_PREDICTOR_MODE_CHOICES:
        raise ValueError(
            "source_predictor_mode must be one of "
            f"{SOURCE_PREDICTOR_MODE_CHOICES!r}, got {source_predictor_mode!r}."
        )
    return str(source_predictor_mode)


def format_source_predictor_filename_component(source_predictor_mode: str) -> str:
    """Return the optional ripple-GLM filename token for one source mode."""
    mode = validate_source_predictor_mode(source_predictor_mode)
    token = SOURCE_PREDICTOR_MODE_FILENAME_TOKENS[mode]
    return f"{token}_" if token else ""


def get_ripple_glm_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    source_predictor_mode: str = SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
) -> Path:
    """Return the NetCDF ripple-GLM result path for one epoch."""
    window_suffix = format_ripple_window_suffix(
        ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    ridge_suffix = format_ridge_strength_suffix(ridge_strength)
    source_component = format_source_predictor_filename_component(source_predictor_mode)
    filename = (
        f"{epoch}_{window_suffix}_{ripple_selection}_{source_component}"
        f"{ridge_suffix}_samplewise_ripple_glm.nc"
    )
    return get_dataset_analysis_path(data_root, animal_name, date) / RIPPLE_GLM_RELATIVE_DIR / filename


def get_ripple_glm_model_window_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    source_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    source_window_offset_s: float = PANEL_E_GLM_SOURCE_WINDOW_OFFSET_S,
    target_window_s: float = PANEL_E_GLM_TARGET_WINDOW_S,
    target_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    source_predictor_mode: str = SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
) -> Path:
    """Return one NetCDF ripple-GLM path for asymmetric model windows."""
    window_suffix = format_glm_model_window_suffix(
        source_window_s=source_window_s,
        source_window_offset_s=source_window_offset_s,
        target_window_s=target_window_s,
        target_window_offset_s=target_window_offset_s,
    )
    ridge_suffix = format_ridge_strength_suffix(ridge_strength)
    source_component = format_source_predictor_filename_component(source_predictor_mode)
    filename = (
        f"{epoch}_{window_suffix}_{ripple_selection}_{source_component}"
        f"{ridge_suffix}_samplewise_ripple_glm.nc"
    )
    return get_dataset_analysis_path(data_root, animal_name, date) / RIPPLE_GLM_RELATIVE_DIR / filename


def format_place_bin_size_token(place_bin_size_cm: float) -> str:
    """Return the filename token used by task-progression encoding summaries."""
    value_text = f"{float(place_bin_size_cm):g}".replace("-", "m").replace(".", "p")
    return f"placebin{value_text}cm"


def get_encoding_comparison_summary_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    n_folds: int = DEFAULT_PANEL_D_ENCODING_N_FOLDS,
    place_bin_size_cm: float = DEFAULT_PANEL_D_PLACE_BIN_SIZE_CM,
) -> Path:
    """Return the preferred task-progression encoding-comparison summary path."""
    data_dir = (
        get_dataset_analysis_path(data_root, animal_name, date)
        / ENCODING_COMPARISON_RELATIVE_DIR
    )
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    preferred_path = data_dir / f"{region}_{epoch}_cv{n_folds}_{place_bin_token}_encoding_summary.parquet"
    if preferred_path.exists():
        return preferred_path
    return data_dir / f"{region}_{epoch}_cv{n_folds}_encoding_summary.parquet"


def find_motor_nested_cv_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> Path:
    """Return the preferred motor nested-CV output path for one session epoch."""
    data_dir = (
        get_dataset_analysis_path(data_root, animal_name, date)
        / MOTOR_NESTED_CV_RELATIVE_DIR
    )
    candidates = sorted(data_dir.glob(f"{region}_{epoch}_nested_lapcv_*.nc"))
    if not candidates:
        raise FileNotFoundError(
            "Missing task-progression motor nested-CV output. Expected a file "
            f"matching {data_dir / f'{region}_{epoch}_nested_lapcv_*.nc'}."
        )
    preferred = [
        path
        for path in candidates
        if PANEL_D_MOTOR_PREFERRED_FILENAME_TOKEN in path.name
    ]
    candidates = preferred if preferred else candidates
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def get_tuning_similarity_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    similarity_metric: str = DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC,
) -> Path:
    """Return the task-progression tuning-analysis within-epoch similarity path."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / TUNING_ANALYSIS_RELATIVE_DIR
        / f"{region}_{epoch}_{similarity_metric}_within_epoch_similarity.parquet"
    )


def get_ripple_decoding_comparison_summary_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
) -> Path:
    """Return the ripple CA1-V1 Bayesian decoding comparison summary path."""
    filename = f"{representation}_train-{train_epoch}_decode-{decode_epoch}_epoch_summary.parquet"
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / RIPPLE_DECODING_COMPARISON_RELATIVE_DIR
        / filename
    )


def read_parquet_table(path: Path) -> Any:
    """Load one parquet table with a focused missing-file message."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Parquet table not found: {path}")
    import pandas as pd

    return pd.read_parquet(path)


def _append_path_parts(base_path: Path, parts: Sequence[str]) -> Path:
    """Append ordered path parts to one base path."""
    output_path = Path(base_path)
    for part in parts:
        output_path = output_path / str(part)
    return output_path


def get_screen_xcorr_candidate_dirs(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    state: str = DEFAULT_XCORR_STATE,
    max_lag_s: float = DEFAULT_XCORR_MAX_LAG_S,
    bin_size_s: float = DEFAULT_XCORR_BIN_SIZE_S,
    ripple_window_s: float | None = None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> list[Path]:
    """Return current and legacy screen-xcorr cache directories for one epoch."""
    analysis_path = get_dataset_analysis_path(data_root, animal_name, date)
    state_parts = get_state_output_parts(
        state,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    settings_suffix = format_xcorr_settings_suffix(
        max_lag_s=max_lag_s,
        bin_size_s=bin_size_s,
    )
    base_path = analysis_path / XCORR_RELATIVE_DIR
    return [
        _append_path_parts(base_path, [*state_parts, settings_suffix, epoch]),
        _append_path_parts(base_path, [*state_parts, epoch]),
    ]


def get_screen_xcorr_paths(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    state: str = DEFAULT_XCORR_STATE,
    max_lag_s: float = DEFAULT_XCORR_MAX_LAG_S,
    bin_size_s: float = DEFAULT_XCORR_BIN_SIZE_S,
    ripple_window_s: float | None = None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> dict[str, Path]:
    """Return the first existing screen-xcorr summary and tensor paths."""
    candidates = get_screen_xcorr_candidate_dirs(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        state=state,
        max_lag_s=max_lag_s,
        bin_size_s=bin_size_s,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    for candidate_dir in candidates:
        paths = {
            "summary": candidate_dir / XCORR_SUMMARY_FILENAME,
            "dataset": candidate_dir / XCORR_DATASET_FILENAME,
        }
        if paths["summary"].exists() and paths["dataset"].exists():
            return paths

    checked = "\n".join(str(candidate_dir) for candidate_dir in candidates)
    raise FileNotFoundError(
        "Could not find matching screen-xcorr summary and NetCDF files. "
        f"Checked:\n{checked}"
    )


def load_ripple_event_table(
    data_root: Path,
    animal_name: str,
    date: str,
) -> Any:
    """Load the flattened ripple event table for one session."""
    table = read_parquet_table(get_ripple_event_path(data_root, animal_name, date))
    rename_columns = {}
    if "start" in table.columns and "start_time" not in table.columns:
        rename_columns["start"] = "start_time"
    if "end" in table.columns and "end_time" not in table.columns:
        rename_columns["end"] = "end_time"
    if rename_columns:
        table = table.rename(columns=rename_columns)

    required_columns = ("epoch", "start_time", "end_time")
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "Ripple event table is missing required columns: "
            f"{missing_columns!r}"
        )
    table = table.copy()
    table["epoch"] = table["epoch"].astype(str)
    table["start_time"] = np.asarray(table["start_time"], dtype=float)
    table["end_time"] = np.asarray(table["end_time"], dtype=float)
    return table


def filter_ripples_by_epoch_and_threshold(
    ripple_table: Any,
    *,
    epoch: str,
    ripple_threshold_zscore: float | None,
) -> Any:
    """Return one epoch's ripples, optionally thresholded by mean z-score."""
    table = ripple_table.loc[ripple_table["epoch"].astype(str) == str(epoch)].copy()
    if ripple_threshold_zscore is None:
        return table.reset_index(drop=True)
    if "mean_zscore" not in table.columns:
        raise ValueError(
            "Ripple thresholding requires a 'mean_zscore' column in "
            "ripple_times.parquet."
        )
    mean_zscore = np.asarray(table["mean_zscore"], dtype=float)
    return table.loc[mean_zscore > float(ripple_threshold_zscore)].reset_index(drop=True)


def load_ripple_count_table(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    ripple_threshold_zscore: float | None = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
) -> Any:
    """Load per-data-set ripple counts from saved event tables."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        event_table = load_ripple_event_table(data_root, animal_name, date)
        epoch_table = filter_ripples_by_epoch_and_threshold(
            event_table,
            epoch=epoch,
            ripple_threshold_zscore=ripple_threshold_zscore,
        )
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "epoch": epoch,
                "n_ripples": int(len(epoch_table)),
            }
        )
    return pd.DataFrame(rows)


def load_epoch_modulation_summary_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region_label: str = DEFAULT_REGION_LABEL,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
) -> Any:
    """Load one epoch's cached ripple-modulation summary table."""
    paths = get_ripple_modulation_paths(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region_label=region_label,
        ripple_threshold_zscore=ripple_threshold_zscore,
        bin_size_s=bin_size_s,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        response_window=response_window,
        baseline_window=baseline_window,
        heatmap_normalize=heatmap_normalize,
    )
    table = read_parquet_table(paths["summary"]).copy()
    required_columns = (
        "animal_name",
        "date",
        "epoch",
        "region",
        "unit_id",
        "ripple_modulation_index",
        "response_zscore",
    )
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "Ripple-modulation summary table is missing required columns: "
            f"{missing_columns!r}"
        )
    table["source_path"] = str(paths["summary"])
    return table


def load_modulation_summary_table(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    region_label: str = DEFAULT_REGION_LABEL,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
) -> Any:
    """Load pooled ripple-modulation summaries for configured data sets."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset)
        table = load_epoch_modulation_summary_table(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region_label=region_label,
            ripple_threshold_zscore=ripple_threshold_zscore,
            bin_size_s=bin_size_s,
            time_before_s=time_before_s,
            time_after_s=time_after_s,
            response_window=response_window,
            baseline_window=baseline_window,
            heatmap_normalize=heatmap_normalize,
        )
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True, sort=False)


def load_peri_ripple_firing_rate_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region_label: str = DEFAULT_REGION_LABEL,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
) -> Any:
    """Load one epoch's cached peri-ripple firing-rate table."""
    paths = get_ripple_modulation_paths(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region_label=region_label,
        ripple_threshold_zscore=ripple_threshold_zscore,
        bin_size_s=bin_size_s,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        response_window=response_window,
        baseline_window=baseline_window,
        heatmap_normalize=heatmap_normalize,
    )
    table = read_parquet_table(paths["peri_ripple_firing_rate"])
    required_columns = (
        "animal_name",
        "date",
        "epoch",
        "region",
        "unit_id",
        "time_s",
        "mean_rate_hz",
    )
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "Peri-ripple firing-rate table is missing required columns: "
            f"{missing_columns!r}"
        )
    return table


def load_ripple_heatmap_epoch_tables(
    data_root: Path,
    epoch_ids: Mapping[str, DatasetId],
    *,
    region_label: str = DEFAULT_REGION_LABEL,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
) -> list[dict[str, Any]]:
    """Load registered light, dark, and sleep peri-ripple heatmap tables."""
    epoch_tables = []
    for epoch_type in HEATMAP_EPOCH_ORDER:
        animal_name, date, epoch = normalize_dataset_id(epoch_ids[epoch_type])
        table = load_peri_ripple_firing_rate_table(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region_label=region_label,
            ripple_threshold_zscore=ripple_threshold_zscore,
            bin_size_s=bin_size_s,
            time_before_s=time_before_s,
            time_after_s=time_after_s,
            response_window=response_window,
            baseline_window=baseline_window,
            heatmap_normalize=heatmap_normalize,
        )
        epoch_tables.append(
            {
                "epoch_type": epoch_type,
                "label": HEATMAP_EPOCH_LABELS[epoch_type],
                "animal_name": animal_name,
                "date": date,
                "epoch": epoch,
                "firing_rate_table": table,
            }
        )
    return epoch_tables


def _format_pooled_epoch_label(epochs: Sequence[str]) -> str:
    """Return a compact epoch label for pooled panel-A tables."""
    unique_epochs = sorted({str(epoch) for epoch in epochs})
    if len(unique_epochs) == 1:
        return unique_epochs[0]
    return "registered"


def load_pooled_ripple_heatmap_epoch_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
    region_label: str = DEFAULT_REGION_LABEL,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window: tuple[float, float] = DEFAULT_RESPONSE_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
) -> list[dict[str, Any]]:
    """Load pooled light, dark, and sleep ripple-modulation tables."""
    import pandas as pd

    grouped_tables: dict[str, dict[str, list[Any]]] = {
        epoch_type: {
            "firing_rate_tables": [],
            "summary_tables": [],
            "epochs": [],
            "datasets": [],
        }
        for epoch_type in HEATMAP_EPOCH_ORDER
    }
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        for epoch_type in HEATMAP_EPOCH_ORDER:
            epoch_animal, epoch_date, epoch = normalize_dataset_id(epoch_ids[epoch_type])
            firing_rate_table = load_peri_ripple_firing_rate_table(
                data_root,
                animal_name=epoch_animal,
                date=epoch_date,
                epoch=epoch,
                region_label=region_label,
                ripple_threshold_zscore=ripple_threshold_zscore,
                bin_size_s=bin_size_s,
                time_before_s=time_before_s,
                time_after_s=time_after_s,
                response_window=response_window,
                baseline_window=baseline_window,
                heatmap_normalize=heatmap_normalize,
            )
            summary_table = load_epoch_modulation_summary_table(
                data_root,
                animal_name=epoch_animal,
                date=epoch_date,
                epoch=epoch,
                region_label=region_label,
                ripple_threshold_zscore=ripple_threshold_zscore,
                bin_size_s=bin_size_s,
                time_before_s=time_before_s,
                time_after_s=time_after_s,
                response_window=response_window,
                baseline_window=baseline_window,
                heatmap_normalize=heatmap_normalize,
            )
            grouped = grouped_tables[epoch_type]
            grouped["firing_rate_tables"].append(firing_rate_table)
            grouped["summary_tables"].append(summary_table)
            grouped["epochs"].append(epoch)
            grouped["datasets"].append((epoch_animal, epoch_date, epoch))

    epoch_tables = []
    for epoch_type in HEATMAP_EPOCH_ORDER:
        grouped = grouped_tables[epoch_type]
        firing_rate_tables = grouped["firing_rate_tables"]
        summary_tables = grouped["summary_tables"]
        epoch_tables.append(
            {
                "epoch_type": epoch_type,
                "label": HEATMAP_EPOCH_LABELS[epoch_type],
                "epoch": _format_pooled_epoch_label(grouped["epochs"]),
                "epochs": tuple(grouped["epochs"]),
                "datasets": tuple(grouped["datasets"]),
                "n_datasets": len(grouped["datasets"]),
                "firing_rate_table": pd.concat(
                    firing_rate_tables,
                    ignore_index=True,
                    sort=False,
                )
                if firing_rate_tables
                else pd.DataFrame(),
                "summary_table": pd.concat(
                    summary_tables,
                    ignore_index=True,
                    sort=False,
                )
                if summary_tables
                else pd.DataFrame(),
            }
        )
    return epoch_tables


def normalize_heatmap_rows(values: np.ndarray) -> np.ndarray:
    """Peak-normalize each heatmap row for display."""
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap matrix, got shape {value_array.shape}.")

    row_scale = np.full(value_array.shape[0], np.nan, dtype=float)
    finite_rows = np.isfinite(value_array).any(axis=1)
    if np.any(finite_rows):
        row_scale[finite_rows] = np.nanmax(value_array[finite_rows], axis=1)
    valid_rows = np.isfinite(row_scale) & (row_scale > 0)
    normalized = np.full_like(value_array, np.nan, dtype=float)
    if np.any(valid_rows):
        normalized[valid_rows] = value_array[valid_rows] / row_scale[valid_rows, None]
    return normalized


def build_peri_ripple_heatmap_payload(
    firing_rate_table: Any,
    *,
    region: str,
) -> dict[str, Any]:
    """Return unit/time matrix payload for one region's peri-ripple heatmap."""
    region_rows = firing_rate_table.loc[firing_rate_table["region"].astype(str) == region].copy()
    if region_rows.empty:
        return {
            "region": region,
            "unit_ids": np.asarray([], dtype=object),
            "time_s": np.asarray([], dtype=float),
            "mean_rate_hz": np.empty((0, 0), dtype=float),
        }

    identity_columns = [
        column
        for column in ("animal_name", "date", "epoch", "unit_id")
        if column in region_rows.columns
    ]
    if "unit_id" not in identity_columns:
        raise ValueError("Peri-ripple firing-rate table is missing required column: 'unit_id'")

    sorted_rows = region_rows.sort_values(
        by=[*identity_columns, "time_s"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_key = identity_columns[0] if len(identity_columns) == 1 else identity_columns
    unit_ids = []
    rate_rows = []
    time_s: np.ndarray | None = None
    for unit_id, group in sorted_rows.groupby(group_key, sort=False):
        unit_time_s = group["time_s"].to_numpy(dtype=float)
        if time_s is None:
            time_s = unit_time_s
        elif unit_time_s.shape != time_s.shape or not np.allclose(unit_time_s, time_s):
            raise ValueError(
                f"Peri-ripple firing-rate table has inconsistent time bins for {region}."
            )
        unit_ids.append(unit_id)
        rate_rows.append(group["mean_rate_hz"].to_numpy(dtype=float))

    matrix = np.vstack(rate_rows) if rate_rows else np.empty((0, 0), dtype=float)
    return {
        "region": region,
        "unit_ids": np.asarray(unit_ids, dtype=object),
        "time_s": np.asarray(time_s if time_s is not None else [], dtype=float),
        "mean_rate_hz": matrix,
    }


def _filter_existing_unit_ids(unit_ids: np.ndarray, available_unit_ids: np.ndarray) -> np.ndarray:
    """Return requested unit IDs that exist in one ordered coordinate array."""
    available = set(np.asarray(available_unit_ids).tolist())
    return np.asarray([unit_id for unit_id in np.asarray(unit_ids).tolist() if unit_id in available])


def load_top_ca1_xcorr_panel_data(
    data_root: Path,
    *,
    animal_name: str = DEFAULT_XCORR_DATASET[0],
    date: str = DEFAULT_XCORR_DATASET[1],
    epoch: str = DEFAULT_XCORR_DATASET[2],
    state: str = DEFAULT_XCORR_STATE,
    top_n_ca1_units: int = DEFAULT_XCORR_TOP_CA1_UNITS,
    max_lag_s: float = DEFAULT_XCORR_MAX_LAG_S,
    bin_size_s: float = DEFAULT_XCORR_BIN_SIZE_S,
    display_vmax: float = DEFAULT_XCORR_DISPLAY_VMAX,
    ripple_window_s: float | None = None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> dict[str, Any]:
    """Load top-CA1 screen-xcorr heatmap data with one shared V1 order."""
    if int(top_n_ca1_units) <= 0:
        raise ValueError("top_n_ca1_units must be positive.")

    paths = get_screen_xcorr_paths(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        state=state,
        max_lag_s=max_lag_s,
        bin_size_s=bin_size_s,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    summary_table = read_parquet_table(paths["summary"])
    required_columns = (
        "ca1_unit_id",
        "v1_unit_id",
        "peak_lag_s",
        "peak_norm_xcorr",
        "status",
    )
    missing_columns = [column for column in required_columns if column not in summary_table.columns]
    if missing_columns:
        raise ValueError(
            "Screen-xcorr summary table is missing required columns: "
            f"{missing_columns!r}"
        )

    valid_summary = summary_table.loc[summary_table["status"] == PAIR_STATUS_VALID].copy()
    if valid_summary.empty:
        raise ValueError(f"No valid screen-xcorr pairs found in {paths['summary']}.")

    ca1_unit_order = order_ca1_units_by_best_partner(valid_summary)
    if ca1_unit_order.size == 0:
        raise ValueError(f"No CA1 units could be ranked from {paths['summary']}.")
    ca1_unit_order = ca1_unit_order[: int(top_n_ca1_units)]
    top_ca1_unit_id = ca1_unit_order[0]
    top_ca1_rows = valid_summary.loc[valid_summary["ca1_unit_id"] == top_ca1_unit_id].copy()
    top_ca1_rows = top_ca1_rows.sort_values(
        by=["peak_norm_xcorr", "peak_lag_s"],
        ascending=[False, True],
        kind="stable",
    )
    if top_ca1_rows.empty:
        raise ValueError(f"No valid V1 partners found for CA1 unit {top_ca1_unit_id!r}.")
    v1_unit_order = top_ca1_rows["v1_unit_id"].to_numpy()

    import xarray as xr

    dataset = xr.load_dataset(paths["dataset"])
    try:
        available_ca1_units = np.asarray(dataset["ca1_unit"].values)
        available_v1_units = np.asarray(dataset["v1_unit"].values)
        ca1_unit_order = _filter_existing_unit_ids(ca1_unit_order, available_ca1_units)
        v1_unit_order = _filter_existing_unit_ids(v1_unit_order, available_v1_units)
        if ca1_unit_order.size == 0 or v1_unit_order.size == 0:
            raise ValueError(
                "Screen-xcorr summary units do not overlap the NetCDF unit coordinates."
            )
        xcorr_values = np.asarray(
            dataset["xcorr"].sel(ca1_unit=ca1_unit_order, v1_unit=v1_unit_order).values,
            dtype=float,
        )
        lag_s = np.asarray(dataset["lag_s"].values, dtype=float)
        attrs = dict(dataset.attrs)
    finally:
        dataset.close()

    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "state": state,
        "summary_path": paths["summary"],
        "dataset_path": paths["dataset"],
        "summary_table": valid_summary,
        "ca1_unit_ids": ca1_unit_order,
        "v1_unit_ids": v1_unit_order,
        "v1_order_reference_ca1_unit": top_ca1_unit_id,
        "lag_s": lag_s,
        "xcorr": xcorr_values,
        "display_vmax": float(display_vmax),
        "attrs": attrs,
    }


def load_example_ripple_lfp_trace(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_threshold_zscore: float | None = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    time_before_s: float = DEFAULT_LFP_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_LFP_TIME_AFTER_S,
) -> dict[str, Any]:
    """Load a ripple-band LFP snippet around the largest ripple in one epoch."""
    ripple_table = load_ripple_event_table(data_root, animal_name, date)
    epoch_table = filter_ripples_by_epoch_and_threshold(
        ripple_table,
        epoch=epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    if epoch_table.empty:
        raise ValueError(
            f"No ripples found for {animal_name} {date} {epoch} "
            f"at threshold {ripple_threshold_zscore}."
        )
    if "mean_zscore" in epoch_table.columns:
        row = epoch_table.iloc[int(np.nanargmax(epoch_table["mean_zscore"].to_numpy(dtype=float)))]
        mean_zscore = float(row["mean_zscore"])
    else:
        row = epoch_table.iloc[0]
        mean_zscore = float("nan")

    ripple_start_s = float(row["start_time"])
    ripple_end_s = float(row["end_time"])
    return load_ripple_lfp_snippet(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_start_s=ripple_start_s,
        ripple_end_s=ripple_end_s,
        mean_zscore=mean_zscore,
        n_ripples=int(len(epoch_table)),
        time_before_s=time_before_s,
        time_after_s=time_after_s,
    )


def load_ripple_lfp_snippet(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_start_s: float,
    ripple_end_s: float,
    mean_zscore: float = float("nan"),
    n_ripples: int = 0,
    time_before_s: float = DEFAULT_LFP_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_LFP_TIME_AFTER_S,
) -> dict[str, Any]:
    """Load one ripple-band LFP snippet around a specified ripple start."""
    import xarray as xr

    lfp_path = get_ripple_lfp_path(data_root, animal_name, date, epoch)
    if not lfp_path.exists():
        raise FileNotFoundError(f"Ripple-band LFP NetCDF not found: {lfp_path}")

    dataset = xr.load_dataset(lfp_path)
    try:
        time_s = np.asarray(dataset["time"].values, dtype=float)
        filtered_lfp = np.asarray(dataset["filtered_lfp"].values, dtype=float)
        channel_ids = np.asarray(dataset["channel"].values)
    finally:
        dataset.close()

    if filtered_lfp.ndim != 2:
        raise ValueError(f"Expected 2D filtered_lfp, got shape {filtered_lfp.shape}.")
    mask = (time_s >= ripple_start_s - time_before_s) & (time_s <= ripple_start_s + time_after_s)
    if not np.any(mask):
        raise ValueError(
            "Ripple-band LFP cache has no samples in the requested snippet window."
        )

    snippet = filtered_lfp[mask, 0]
    relative_time_s = time_s[mask] - ripple_start_s
    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "time_s": relative_time_s,
        "filtered_lfp": snippet,
        "ripple_start_s": ripple_start_s,
        "ripple_end_s": ripple_end_s,
        "ripple_duration_s": ripple_end_s - ripple_start_s,
        "mean_zscore": mean_zscore,
        "channel": channel_ids[0] if channel_ids.size else 0,
        "n_ripples": int(n_ripples),
    }


def build_panel_b_schematic_cache_metadata(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_threshold_zscore: float | None = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    time_before_s: float = DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
    n_units_per_region: int = DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
    target_ripple_duration_s: float = DEFAULT_PANEL_B_SCHEMATIC_TARGET_DURATION_S,
) -> dict[str, Any]:
    """Return metadata identifying the cached panel B real-spike schematic example."""
    return {
        "cache_version": PANEL_B_SCHEMATIC_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "B",
        "artifact": "ripple_glm_schematic_spikes",
        "data_root": str(Path(data_root)),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "ripple_threshold_zscore": None
        if ripple_threshold_zscore is None
        else float(ripple_threshold_zscore),
        "time_before_s": float(time_before_s),
        "time_after_s": float(time_after_s),
        "n_units_per_region": int(n_units_per_region),
        "target_ripple_duration_s": float(target_ripple_duration_s),
    }


def build_panel_b_schematic_cache_path(
    cache_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Return the descriptive cache path for the panel B schematic spike rasters."""
    animal_name = _format_figure_cache_token(metadata["animal_name"])
    date = _format_figure_cache_token(metadata["date"])
    epoch = _format_figure_cache_token(metadata["epoch"])
    if metadata["ripple_threshold_zscore"] is None:
        threshold = "none"
    else:
        threshold = _format_figure_cache_number(float(metadata["ripple_threshold_zscore"]))
    time_before = _format_figure_cache_number(float(metadata["time_before_s"]))
    time_after = _format_figure_cache_number(float(metadata["time_after_s"]))
    n_units = int(metadata["n_units_per_region"])
    target_duration = _format_figure_cache_number(float(metadata["target_ripple_duration_s"]))
    cache_version = int(metadata["cache_version"])
    filename = (
        f"figure_2_panel_b_schematic_{animal_name}_{date}_{epoch}"
        f"_thr{threshold}_tb{time_before}_ta{time_after}_n{n_units}"
        f"_dur{target_duration}"
        f"_cachev{cache_version}.npz"
    )
    return Path(cache_dir) / filename


def _extract_spike_times_by_unit(spikes: Any) -> dict[Any, np.ndarray]:
    """Return sorted spike-time arrays keyed by unit id for a TsGroup-like object."""
    spike_times_by_unit: dict[Any, np.ndarray] = {}
    for unit_id in spikes.keys():
        times = np.asarray(spikes[unit_id].t, dtype=float)
        spike_times_by_unit[unit_id] = np.sort(times[np.isfinite(times)])
    return spike_times_by_unit


def _count_unit_spikes_in_window(
    spike_times_by_unit: Mapping[Any, np.ndarray],
    *,
    start_s: float,
    end_s: float,
) -> dict[Any, int]:
    """Count spikes for each unit in one half-open time window."""
    counts: dict[Any, int] = {}
    for unit_id, spike_times_s in spike_times_by_unit.items():
        left = int(np.searchsorted(spike_times_s, start_s, side="left"))
        right = int(np.searchsorted(spike_times_s, end_s, side="left"))
        counts[unit_id] = max(0, right - left)
    return counts


def _select_top_units_by_count(
    spike_times_by_unit: Mapping[Any, np.ndarray],
    *,
    start_s: float,
    end_s: float,
    n_units: int,
) -> tuple[list[Any], dict[Any, int]]:
    """Select the most active units within one display window."""
    counts = _count_unit_spikes_in_window(
        spike_times_by_unit,
        start_s=start_s,
        end_s=end_s,
    )
    ranked_units = sorted(
        counts,
        key=lambda unit_id: (-counts[unit_id], str(unit_id)),
    )
    return ranked_units[: int(n_units)], counts


def _select_strongly_modulated_ca1_units(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    n_units: int,
    ripple_threshold_zscore: float | None = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
) -> list[Any]:
    """Return CA1 units with the strongest finite ripple modulation."""
    if ripple_threshold_zscore is None:
        raise ValueError("CA1 ripple-modulation ranking requires a finite ripple threshold.")
    table = load_epoch_modulation_summary_table(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_threshold_zscore=float(ripple_threshold_zscore),
    )
    ca1_table = table.loc[table["region"].astype(str) == "ca1"].copy()
    if ca1_table.empty:
        raise ValueError(f"No CA1 ripple-modulation rows found for {animal_name} {date} {epoch}.")

    response_zscore = np.asarray(ca1_table["response_zscore"], dtype=float)
    modulation_index = np.asarray(ca1_table["ripple_modulation_index"], dtype=float)
    finite_zscore = np.isfinite(response_zscore)
    finite_modulation_index = np.isfinite(modulation_index)
    ca1_table["_modulation_rank_1"] = np.where(
        finite_zscore,
        np.abs(response_zscore),
        -np.inf,
    )
    ca1_table["_modulation_rank_2"] = np.where(
        finite_modulation_index,
        np.abs(modulation_index),
        -np.inf,
    )
    ranked = ca1_table.sort_values(
        by=["_modulation_rank_1", "_modulation_rank_2", "unit_id"],
        ascending=[False, False, True],
    )
    ranked = ranked.loc[
        np.isfinite(np.asarray(ranked["_modulation_rank_1"], dtype=float))
        | np.isfinite(np.asarray(ranked["_modulation_rank_2"], dtype=float))
    ]
    return list(ranked["unit_id"].iloc[: int(n_units)])


def _relative_spike_times_for_units(
    spike_times_by_unit: Mapping[Any, np.ndarray],
    unit_ids: Sequence[Any],
    *,
    ripple_start_s: float,
    time_before_s: float,
    time_after_s: float,
) -> tuple[np.ndarray, ...]:
    """Return per-unit spike times relative to ripple onset."""
    window_start_s = ripple_start_s - float(time_before_s)
    window_end_s = ripple_start_s + float(time_after_s)
    relative_times = []
    for unit_id in unit_ids:
        spike_times_s = spike_times_by_unit[unit_id]
        left = int(np.searchsorted(spike_times_s, window_start_s, side="left"))
        right = int(np.searchsorted(spike_times_s, window_end_s, side="left"))
        relative_times.append(np.asarray(spike_times_s[left:right] - ripple_start_s, dtype=float))
    return tuple(relative_times)


def _flatten_spike_rasters(
    spike_rasters: Sequence[Sequence[float] | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ragged spike rasters for compact npz storage."""
    counts = np.asarray([len(row) for row in spike_rasters], dtype=np.int64)
    if counts.size == 0 or int(np.sum(counts)) == 0:
        return np.asarray([], dtype=float), counts
    return np.concatenate([np.asarray(row, dtype=float) for row in spike_rasters]), counts


def _unflatten_spike_rasters(
    flat_spikes: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Restore ragged spike rasters from compact npz storage."""
    flat_spikes = np.asarray(flat_spikes, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    rasters = []
    offset = 0
    for count in counts:
        next_offset = offset + int(count)
        rasters.append(np.asarray(flat_spikes[offset:next_offset], dtype=float))
        offset = next_offset
    return tuple(rasters)


def save_panel_b_schematic_cache(
    cache_path: Path,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Save the real-spike panel B schematic example to a compact npz cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ca1_flat_spikes, ca1_counts = _flatten_spike_rasters(payload["ca1_spike_times_s"])
    v1_flat_spikes, v1_counts = _flatten_spike_rasters(payload["v1_spike_times_s"])
    np.savez_compressed(
        cache_path,
        metadata_json=json.dumps(dict(metadata), sort_keys=True),
        animal_name=str(payload["animal_name"]),
        date=str(payload["date"]),
        epoch=str(payload["epoch"]),
        time_s=np.asarray(payload["time_s"], dtype=float),
        filtered_lfp=np.asarray(payload["filtered_lfp"], dtype=float),
        ripple_start_s=float(payload["ripple_start_s"]),
        ripple_end_s=float(payload["ripple_end_s"]),
        ripple_duration_s=float(payload["ripple_duration_s"]),
        mean_zscore=float(payload["mean_zscore"]),
        channel=payload["channel"],
        n_ripples=int(payload["n_ripples"]),
        time_before_s=float(payload["time_before_s"]),
        time_after_s=float(payload["time_after_s"]),
        n_units_per_region=int(payload["n_units_per_region"]),
        ca1_unit_ids=np.asarray(payload["ca1_unit_ids"]),
        v1_unit_ids=np.asarray(payload["v1_unit_ids"]),
        ca1_spike_times_flat=ca1_flat_spikes,
        ca1_spike_counts=ca1_counts,
        v1_spike_times_flat=v1_flat_spikes,
        v1_spike_counts=v1_counts,
        selection_score=np.asarray(payload["selection_score"], dtype=float),
    )


def load_panel_b_schematic_cache(
    cache_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load a cached panel B schematic example if metadata still match."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_metadata = json.loads(str(cached["metadata_json"]))
            if cached_metadata != dict(metadata):
                return None
            return {
                "animal_name": str(cached["animal_name"]),
                "date": str(cached["date"]),
                "epoch": str(cached["epoch"]),
                "time_s": np.asarray(cached["time_s"], dtype=float),
                "filtered_lfp": np.asarray(cached["filtered_lfp"], dtype=float),
                "ripple_start_s": float(cached["ripple_start_s"]),
                "ripple_end_s": float(cached["ripple_end_s"]),
                "ripple_duration_s": float(cached["ripple_duration_s"]),
                "mean_zscore": float(cached["mean_zscore"]),
                "channel": cached["channel"].item(),
                "n_ripples": int(cached["n_ripples"]),
                "time_before_s": float(cached["time_before_s"]),
                "time_after_s": float(cached["time_after_s"]),
                "n_units_per_region": int(cached["n_units_per_region"]),
                "ca1_unit_ids": np.asarray(cached["ca1_unit_ids"]),
                "v1_unit_ids": np.asarray(cached["v1_unit_ids"]),
                "ca1_spike_times_s": _unflatten_spike_rasters(
                    cached["ca1_spike_times_flat"],
                    cached["ca1_spike_counts"],
                ),
                "v1_spike_times_s": _unflatten_spike_rasters(
                    cached["v1_spike_times_flat"],
                    cached["v1_spike_counts"],
                ),
                "selection_score": np.asarray(cached["selection_score"], dtype=float),
            }
    except (OSError, KeyError, json.JSONDecodeError, ValueError):
        return None


def build_panel_b_schematic_example(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_threshold_zscore: float | None = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    time_before_s: float = DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
    n_units_per_region: int = DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
    target_ripple_duration_s: float = DEFAULT_PANEL_B_SCHEMATIC_TARGET_DURATION_S,
) -> dict[str, Any]:
    """Build a real LFP and spike-raster example for the panel B schematic."""
    ripple_table = load_ripple_event_table(data_root, animal_name, date)
    epoch_table = filter_ripples_by_epoch_and_threshold(
        ripple_table,
        epoch=epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    if epoch_table.empty:
        raise ValueError(
            f"No ripples found for {animal_name} {date} {epoch} "
            f"at threshold {ripple_threshold_zscore}."
        )

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=Path(data_root),
    )
    timestamps_ephys_all, _ = load_ephys_timestamps_all(analysis_path)
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all,
        regions=("ca1", "v1"),
    )
    spike_times_by_region = {
        region: _extract_spike_times_by_unit(spikes_by_region[region])
        for region in ("ca1", "v1")
    }
    ranked_modulated_ca1_unit_ids = [
        unit_id
        for unit_id in _select_strongly_modulated_ca1_units(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            n_units=len(spike_times_by_region["ca1"]),
            ripple_threshold_zscore=ripple_threshold_zscore,
        )
        if unit_id in spike_times_by_region["ca1"]
    ]
    if len(ranked_modulated_ca1_unit_ids) < int(n_units_per_region):
        raise ValueError(
            "Could not match enough strongly modulated CA1 units to spike trains "
            f"for {animal_name} {date} {epoch}: found {len(ranked_modulated_ca1_unit_ids)}."
        )

    best_row_index: int | None = None
    best_score: tuple[float, ...] | None = None
    best_unit_ids: dict[str, list[Any]] = {}
    for row_index, row in epoch_table.reset_index(drop=True).iterrows():
        ripple_start_s = float(row["start_time"])
        ripple_duration_s = float(row["end_time"]) - ripple_start_s
        window_start_s = ripple_start_s - float(time_before_s)
        window_end_s = ripple_start_s + float(time_after_s)
        unit_ids_by_region: dict[str, list[Any]] = {}
        counts_by_region: dict[str, dict[Any, int]] = {}
        ca1_counts_all = _count_unit_spikes_in_window(
            {
                unit_id: spike_times_by_region["ca1"][unit_id]
                for unit_id in ranked_modulated_ca1_unit_ids
            },
            start_s=window_start_s,
            end_s=window_end_s,
        )
        active_modulated_ca1_unit_ids = [
            unit_id for unit_id in ranked_modulated_ca1_unit_ids if ca1_counts_all[unit_id] > 0
        ]
        ca1_unit_ids = active_modulated_ca1_unit_ids[: int(n_units_per_region)]
        if len(ca1_unit_ids) < int(n_units_per_region):
            ca1_unit_ids.extend(
                unit_id
                for unit_id in ranked_modulated_ca1_unit_ids
                if unit_id not in ca1_unit_ids
            )
            ca1_unit_ids = ca1_unit_ids[: int(n_units_per_region)]
        unit_ids_by_region["ca1"] = ca1_unit_ids
        counts_by_region["ca1"] = {
            unit_id: ca1_counts_all[unit_id] for unit_id in unit_ids_by_region["ca1"]
        }
        v1_unit_ids, v1_counts = _select_top_units_by_count(
            spike_times_by_region["v1"],
            start_s=window_start_s,
            end_s=window_end_s,
            n_units=int(n_units_per_region),
        )
        unit_ids_by_region["v1"] = v1_unit_ids
        counts_by_region["v1"] = v1_counts
        active_counts = {
            region: sum(counts_by_region[region][unit_id] > 0 for unit_id in unit_ids_by_region[region])
            for region in ("ca1", "v1")
        }
        ca1_count_total = sum(
            counts_by_region["ca1"][unit_id] for unit_id in unit_ids_by_region["ca1"]
        )
        v1_count_total = sum(
            counts_by_region["v1"][unit_id] for unit_id in unit_ids_by_region["v1"]
        )
        if "mean_zscore" in epoch_table.columns:
            mean_zscore = float(row["mean_zscore"])
        else:
            mean_zscore = float("nan")
        minimum_active_units = int(n_units_per_region)
        score = (
            float(
                active_counts["ca1"] >= minimum_active_units
                and active_counts["v1"] >= minimum_active_units
            ),
            -float(abs(ripple_duration_s - float(target_ripple_duration_s))),
            float(len(active_modulated_ca1_unit_ids)),
            float(active_counts["ca1"]),
            float(ca1_count_total),
            float(active_counts["v1"]),
            float(v1_count_total),
            mean_zscore if np.isfinite(mean_zscore) else -np.inf,
            -float(row_index),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_row_index = int(row_index)
            best_unit_ids = unit_ids_by_region

    if best_row_index is None or best_score is None:
        raise ValueError(f"Could not select a schematic ripple for {animal_name} {date} {epoch}.")

    selected_row = epoch_table.reset_index(drop=True).iloc[best_row_index]
    ripple_start_s = float(selected_row["start_time"])
    ripple_end_s = float(selected_row["end_time"])
    mean_zscore = (
        float(selected_row["mean_zscore"])
        if "mean_zscore" in epoch_table.columns
        else float("nan")
    )
    payload = load_ripple_lfp_snippet(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_start_s=ripple_start_s,
        ripple_end_s=ripple_end_s,
        mean_zscore=mean_zscore,
        n_ripples=int(len(epoch_table)),
        time_before_s=time_before_s,
        time_after_s=time_after_s,
    )
    ca1_unit_ids = best_unit_ids["ca1"]
    v1_unit_ids = best_unit_ids["v1"]
    payload.update(
        {
            "time_before_s": float(time_before_s),
            "time_after_s": float(time_after_s),
            "n_units_per_region": int(n_units_per_region),
            "ca1_unit_ids": np.asarray(ca1_unit_ids),
            "v1_unit_ids": np.asarray(v1_unit_ids),
            "ca1_spike_times_s": _relative_spike_times_for_units(
                spike_times_by_region["ca1"],
                ca1_unit_ids,
                ripple_start_s=ripple_start_s,
                time_before_s=time_before_s,
                time_after_s=time_after_s,
            ),
            "v1_spike_times_s": _relative_spike_times_for_units(
                spike_times_by_region["v1"],
                v1_unit_ids,
                ripple_start_s=ripple_start_s,
                time_before_s=time_before_s,
                time_after_s=time_after_s,
            ),
            "selection_score": np.asarray(best_score, dtype=float),
        }
    )
    return payload


def load_or_build_panel_b_schematic_example(
    data_root: Path,
    *,
    cache_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_threshold_zscore: float | None = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    time_before_s: float = DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
    n_units_per_region: int = DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
    target_ripple_duration_s: float = DEFAULT_PANEL_B_SCHEMATIC_TARGET_DURATION_S,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Load or build the cached real-spike panel B schematic example."""
    metadata = build_panel_b_schematic_cache_metadata(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        n_units_per_region=n_units_per_region,
        target_ripple_duration_s=target_ripple_duration_s,
    )
    cache_path = build_panel_b_schematic_cache_path(cache_dir, metadata)
    if not refresh_cache:
        cached = load_panel_b_schematic_cache(cache_path, metadata)
        if cached is not None:
            return cached
    payload = build_panel_b_schematic_example(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        n_units_per_region=n_units_per_region,
        target_ripple_duration_s=target_ripple_duration_s,
    )
    save_panel_b_schematic_cache(cache_path, payload, metadata)
    return payload


def load_ripple_glm_summary_table(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    source_predictor_mode: str = SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
) -> Any:
    """Load pooled per-unit summary values from ripple-GLM NetCDF outputs."""
    import pandas as pd
    import xarray as xr

    rows: list[dict[str, Any]] = []
    for dataset_id in datasets:
        animal_name, date, epoch = normalize_dataset_id(dataset_id)
        path = get_ripple_glm_path(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            ripple_window_s=ripple_window_s,
            ripple_window_offset_s=ripple_window_offset_s,
            ripple_selection=ripple_selection,
            ridge_strength=ridge_strength,
            source_predictor_mode=source_predictor_mode,
        )
        if not path.exists():
            raise FileNotFoundError(f"Ripple-GLM NetCDF not found: {path}")

        dataset = xr.load_dataset(path)
        try:
            unit_ids = np.asarray(dataset.coords["unit"].values)
            devexp = np.asarray(dataset["ripple_devexp_mean"].values, dtype=float)
            devexp_p = np.asarray(dataset["ripple_devexp_p_value"].values, dtype=float)
            bits_per_spike = np.asarray(
                dataset["ripple_bits_per_spike_mean"].values,
                dtype=float,
            )
            n_ripples = int(dataset.attrs.get("n_ripples_after_selection", dataset.attrs.get("n_ripples", 0)))
            for unit_id, unit_devexp, unit_p, unit_bits_per_spike in zip(
                unit_ids,
                devexp,
                devexp_p,
                bits_per_spike,
                strict=True,
            ):
                rows.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "epoch": epoch,
                        "unit_id": unit_id,
                        "ripple_devexp_mean": float(unit_devexp),
                        "ripple_devexp_p_value": float(unit_p),
                        "ripple_bits_per_spike_mean": float(unit_bits_per_spike),
                        "n_ripples": n_ripples,
                        "source_predictor_mode": source_predictor_mode,
                        "source_path": str(path),
                    }
                )
        finally:
            dataset.close()

    return pd.DataFrame(rows)


def load_glm_epoch_summary_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
    epoch_types: Sequence[str] = HEATMAP_EPOCH_ORDER,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> list[dict[str, Any]]:
    """Load pooled ripple-GLM summaries for selected figure epochs."""
    selected_epoch_types = tuple(str(epoch_type) for epoch_type in epoch_types)
    unknown_epoch_types = sorted(set(selected_epoch_types).difference(HEATMAP_EPOCH_ORDER))
    if unknown_epoch_types:
        raise ValueError(f"Unknown ripple-GLM epoch types: {unknown_epoch_types!r}")
    epoch_datasets: dict[str, list[DatasetId]] = {epoch_type: [] for epoch_type in selected_epoch_types}
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        for epoch_type in selected_epoch_types:
            epoch_datasets[epoch_type].append(normalize_dataset_id(epoch_ids[epoch_type]))

    epoch_tables = []
    for epoch_type in selected_epoch_types:
        selected_datasets = epoch_datasets[epoch_type]
        summary_table = load_ripple_glm_summary_table(
            data_root,
            selected_datasets,
            ripple_window_s=ripple_window_s,
            ripple_window_offset_s=ripple_window_offset_s,
            ripple_selection=ripple_selection,
            ridge_strength=ridge_strength,
        )
        epoch_tables.append(
            {
                "epoch_type": epoch_type,
                "label": HEATMAP_EPOCH_LABELS[epoch_type],
                "epoch": _format_pooled_epoch_label(
                    [epoch for _animal_name, _date, epoch in selected_datasets]
                ),
                "datasets": tuple(selected_datasets),
                "n_datasets": len(selected_datasets),
                "summary_table": summary_table,
            }
        )
    return epoch_tables


def load_glm_source_predictor_comparison_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
    epoch_types: Sequence[str] = PANEL_E_GLM_EPOCH_ORDER,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load paired vector and mean-activity ripple-GLM summaries."""
    import pandas as pd

    rows: list[Any] = []
    missing_artifacts: list[dict[str, str]] = []
    selected_epoch_types = tuple(str(epoch_type) for epoch_type in epoch_types)
    unknown_epoch_types = sorted(set(selected_epoch_types).difference(HEATMAP_EPOCH_ORDER))
    if unknown_epoch_types:
        raise ValueError(f"Unknown ripple-GLM epoch types: {unknown_epoch_types!r}")

    for dataset_id in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset_id)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name=animal_name,
            date=date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        for epoch_type in selected_epoch_types:
            _epoch_animal, _epoch_date, epoch = normalize_dataset_id(epoch_ids[epoch_type])
            paths = {
                mode: get_ripple_glm_path(
                    data_root,
                    animal_name=animal_name,
                    date=date,
                    epoch=epoch,
                    ripple_window_s=ripple_window_s,
                    ripple_window_offset_s=ripple_window_offset_s,
                    ripple_selection=ripple_selection,
                    ridge_strength=ridge_strength,
                    source_predictor_mode=mode,
                )
                for mode in SOURCE_PREDICTOR_MODE_CHOICES
            }
            missing_modes = [
                mode for mode, path in paths.items() if not Path(path).exists()
            ]
            if missing_modes:
                for mode in missing_modes:
                    missing_artifacts.append(
                        {
                            "artifact": "ripple_glm_source_predictor",
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "source_predictor_mode": mode,
                            "path": str(paths[mode]),
                        }
                    )
                continue

            vector_table = load_ripple_glm_summary_table(
                data_root,
                [(animal_name, date, epoch)],
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
                source_predictor_mode=SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
            ).rename(
                columns={
                    "ripple_devexp_mean": "vector_devexp_mean",
                    "ripple_devexp_p_value": "vector_devexp_p_value",
                    "ripple_bits_per_spike_mean": "vector_bits_per_spike_mean",
                    "source_path": "vector_source_path",
                }
            )
            mean_table = load_ripple_glm_summary_table(
                data_root,
                [(animal_name, date, epoch)],
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
                source_predictor_mode=SOURCE_PREDICTOR_MODE_MEAN_ACTIVITY,
            ).rename(
                columns={
                    "ripple_devexp_mean": "mean_activity_devexp_mean",
                    "ripple_devexp_p_value": "mean_activity_devexp_p_value",
                    "ripple_bits_per_spike_mean": "mean_activity_bits_per_spike_mean",
                    "source_path": "mean_activity_source_path",
                }
            )
            joined = vector_table.merge(
                mean_table[
                    [
                        "animal_name",
                        "date",
                        "epoch",
                        "unit_id",
                        "mean_activity_devexp_mean",
                        "mean_activity_devexp_p_value",
                        "mean_activity_bits_per_spike_mean",
                        "mean_activity_source_path",
                    ]
                ],
                on=["animal_name", "date", "epoch", "unit_id"],
                how="inner",
            )
            joined = joined.assign(
                epoch_type=epoch_type,
                label=HEATMAP_EPOCH_LABELS[epoch_type],
                devexp_delta_vector_minus_mean=(
                    joined["vector_devexp_mean"]
                    - joined["mean_activity_devexp_mean"]
                ),
            )
            rows.append(joined)

    comparison_table = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    return {
        "comparison_table": comparison_table,
        "missing_artifacts": missing_artifacts,
        "ripple_selection": ripple_selection,
    }


def load_glm_behavior_association_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
    region: str = DEFAULT_PANEL_D_REGION,
    tuning_similarity_metric: str = DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC,
    tuning_comparison_label: str = DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load dark same-turn tuning similarity joined to light/dark/sleep GLM significance."""
    import pandas as pd

    similarity_rows: list[Any] = []
    missing_artifacts: list[dict[str, str]] = []
    session_unit_columns = ["animal_name", "date", "unit"]

    for dataset_id in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset_id)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name=animal_name,
            date=date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        dark_tuning_epoch = normalize_dataset_id(epoch_ids["dark"])[2]

        tuning_path = get_tuning_similarity_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=dark_tuning_epoch,
            similarity_metric=tuning_similarity_metric,
        )
        if not tuning_path.exists():
            missing_artifacts.append(
                {
                    "artifact": "tuning_analysis",
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": dark_tuning_epoch,
                    "path": str(tuning_path),
                }
            )
            continue

        tuning_table = pd.read_parquet(tuning_path)
        missing_columns = [
            column
            for column in (
                "unit",
                "region",
                "epoch",
                "comparison_label",
                "similarity",
                "firing_rate_hz",
            )
            if column not in tuning_table.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Tuning similarity table {tuning_path} is missing columns "
                f"{missing_columns!r}."
            )
        tuning_rows = tuning_table[
            (tuning_table["region"].astype(str) == region)
            & (tuning_table["epoch"].astype(str) == dark_tuning_epoch)
            & (tuning_table["comparison_label"].astype(str) == tuning_comparison_label)
        ].copy()
        tuning_rows["unit"] = pd.to_numeric(tuning_rows["unit"], errors="coerce")
        tuning_rows = tuning_rows[
            np.isfinite(tuning_rows["unit"].to_numpy(dtype=float))
        ].copy()
        tuning_rows["unit"] = tuning_rows["unit"].astype(int)
        tuning_rows = tuning_rows.assign(
            animal_name=animal_name,
            date=date,
            tuning_epoch=dark_tuning_epoch,
            tuning_source_path=str(tuning_path),
        )[
            session_unit_columns
            + [
                "tuning_epoch",
                "similarity",
                "firing_rate_hz",
                "tuning_source_path",
            ]
        ]
        tuning_rows = tuning_rows.rename(
            columns={"similarity": "same_turn_tuning_similarity"}
        )

        for epoch_type in HEATMAP_EPOCH_ORDER:
            _glm_animal, _glm_date, glm_epoch = normalize_dataset_id(epoch_ids[epoch_type])
            glm_path = get_ripple_glm_path(
                data_root,
                animal_name=animal_name,
                date=date,
                epoch=glm_epoch,
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
            )
            if not glm_path.exists():
                missing_artifacts.append(
                    {
                        "artifact": "ripple_glm",
                        "animal_name": animal_name,
                        "date": date,
                        "epoch": glm_epoch,
                        "path": str(glm_path),
                    }
                )
                continue

            glm_table = load_ripple_glm_summary_table(
                data_root,
                [(animal_name, date, glm_epoch)],
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
            ).rename(columns={"unit_id": "unit", "source_path": "ripple_glm_source_path"})
            glm_table["unit"] = pd.to_numeric(glm_table["unit"], errors="coerce")
            glm_table = glm_table[
                np.isfinite(glm_table["unit"].to_numpy(dtype=float))
            ].copy()
            glm_table["unit"] = glm_table["unit"].astype(int)
            glm_table = glm_table.rename(columns={"epoch": "glm_epoch"})
            glm_table = glm_table[
                session_unit_columns
                + [
                    "glm_epoch",
                    "ripple_devexp_mean",
                    "ripple_devexp_p_value",
                    "ripple_glm_source_path",
                ]
            ].assign(
                epoch_type=epoch_type,
                label=HEATMAP_EPOCH_LABELS[epoch_type],
            )
            similarity_rows.append(
                glm_table.merge(tuning_rows, on=session_unit_columns, how="inner")
            )

    similarity_table = (
        pd.concat(similarity_rows, axis=0, ignore_index=True)
        if similarity_rows
        else pd.DataFrame()
    )
    return {
        "similarity_table": similarity_table,
        "missing_artifacts": missing_artifacts,
        "region": region,
        "tuning_comparison_label": tuning_comparison_label,
        "tuning_similarity_metric": tuning_similarity_metric,
    }


def _format_figure_cache_token(value: Any) -> str:
    """Return a filesystem-safe token for one Figure 2 cache value."""
    token = "".join(
        character if character.isalnum() else "_"
        for character in str(value).strip()
    ).strip("_")
    return token or "none"


def _format_figure_cache_number(value: float) -> str:
    """Return a compact cache token for one numeric setting."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_dark_movement_firing_rate_cache_metadata(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
) -> dict[str, Any]:
    """Return metadata that identifies one dark movement firing-rate cache."""
    return {
        "cache_version": DARK_MOVEMENT_FR_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "D",
        "artifact": "dark_movement_firing_rate",
        "data_root": str(Path(data_root)),
        "animal_name": str(animal_name),
        "date": str(date),
        "dark_epoch": str(dark_epoch),
        "region": str(region),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "columns": list(DARK_MOVEMENT_FR_CACHE_COLUMNS),
    }


def build_dark_movement_firing_rate_cache_path(
    cache_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Return the descriptive cache path for one dark movement firing-rate table."""
    region = _format_figure_cache_token(metadata["region"])
    animal_name = _format_figure_cache_token(metadata["animal_name"])
    date = _format_figure_cache_token(metadata["date"])
    dark_epoch = _format_figure_cache_token(metadata["dark_epoch"])
    speed = _format_figure_cache_number(float(metadata["speed_threshold_cm_s"]))
    cache_version = int(metadata["cache_version"])
    filename = (
        f"figure_2_dark_movement_firing_rate_{region}_{animal_name}_{date}_{dark_epoch}"
        f"_speed{speed}_cachev{cache_version}.parquet"
    )
    return Path(cache_dir) / filename


def _dark_movement_firing_rate_metadata_path(cache_path: Path) -> Path:
    """Return the JSON sidecar path for one dark movement firing-rate cache."""
    return cache_path.with_suffix(".json")


def save_dark_movement_firing_rate_cache(
    cache_path: Path,
    table: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Write one dark movement firing-rate cache table and metadata sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_table = table.loc[:, list(DARK_MOVEMENT_FR_CACHE_COLUMNS)].copy()
    cache_table.to_parquet(cache_path, index=False)
    _dark_movement_firing_rate_metadata_path(cache_path).write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_dark_movement_firing_rate_cache(
    cache_path: Path,
    expected_metadata: Mapping[str, Any],
) -> Any | None:
    """Return cached dark movement firing rates when metadata still matches."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _dark_movement_firing_rate_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None

    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != dict(expected_metadata):
            print(f"Ignoring stale dark movement firing-rate cache at {cache_path}.")
            return None

        table = pd.read_parquet(cache_path)
        missing_columns = [
            column
            for column in DARK_MOVEMENT_FR_CACHE_COLUMNS
            if column not in table.columns
        ]
        if missing_columns:
            print(
                "Ignoring invalid dark movement firing-rate cache at "
                f"{cache_path}: missing columns {missing_columns!r}."
            )
            return None
        return table.loc[:, list(DARK_MOVEMENT_FR_CACHE_COLUMNS)].copy()
    except Exception as exc:
        print(f"Ignoring unreadable dark movement firing-rate cache at {cache_path}: {exc}")
        return None


def load_dark_movement_firing_rate_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str = DEFAULT_PANEL_D_REGION,
    cache_dir: Path | None = DEFAULT_FIGURE_CACHE_DIR,
    refresh_cache: bool = False,
) -> Any:
    """Return per-unit dark movement firing rates for one session and region."""
    import pandas as pd

    from v1ca1.helper.session import compute_movement_firing_rates
    from v1ca1.task_progression._session import prepare_task_progression_session

    metadata = build_dark_movement_firing_rate_cache_metadata(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        dark_epoch=dark_epoch,
        region=region,
    )
    cache_path = (
        build_dark_movement_firing_rate_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached_table = load_dark_movement_firing_rate_cache(cache_path, metadata)
        if cached_table is not None:
            print(f"Loaded dark movement firing-rate cache from {cache_path}.")
            return cached_table

    session = prepare_task_progression_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        selected_run_epochs=[dark_epoch],
        load_body_position=False,
        include_generalized_place=False,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    spikes = session["spikes_by_region"][region]
    unit_ids = np.asarray(list(spikes.keys()), dtype=int)
    firing_rates_hz = np.asarray(movement_firing_rates[region][dark_epoch], dtype=float)
    if unit_ids.shape[0] != firing_rates_hz.shape[0]:
        raise ValueError(
            "Dark movement firing-rate table is not aligned with spike unit IDs: "
            f"{unit_ids.shape[0]} unit IDs and {firing_rates_hz.shape[0]} rates."
        )
    table = pd.DataFrame(
        {
            "unit": unit_ids,
            "dark_firing_rate_hz": firing_rates_hz,
        }
    )
    if cache_path is not None:
        try:
            save_dark_movement_firing_rate_cache(cache_path, table, metadata)
            print(f"Saved dark movement firing-rate cache to {cache_path}.")
        except Exception as exc:
            print(f"Could not save dark movement firing-rate cache to {cache_path}: {exc}")
    return table


def build_glm_dark_activity_devexp_table(
    glm_table: Any,
    dark_activity_table: Any,
    tuning_similarity_table: Any | None = None,
    *,
    animal_name: str,
    date: str,
    glm_epoch: str,
    epoch_type: str,
    dark_epoch: str,
    dark_activity_threshold_hz: float = PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ,
) -> Any:
    """Join ripple-GLM summaries to dark movement firing rates and tuning similarity."""
    import pandas as pd

    glm_rows = glm_table.rename(
        columns={"unit_id": "unit", "source_path": "ripple_glm_source_path"}
    ).copy()
    glm_rows["unit"] = pd.to_numeric(glm_rows["unit"], errors="coerce")
    glm_rows = glm_rows[np.isfinite(glm_rows["unit"].to_numpy(dtype=float))].copy()
    glm_rows["unit"] = glm_rows["unit"].astype(int)
    glm_rows = glm_rows.rename(columns={"epoch": "glm_epoch"})
    glm_rows = glm_rows[
        [
            "unit",
            "glm_epoch",
            "ripple_devexp_mean",
            "ripple_devexp_p_value",
            "ripple_glm_source_path",
        ]
    ]

    dark_rows = dark_activity_table.copy()
    dark_rows["unit"] = pd.to_numeric(dark_rows["unit"], errors="coerce")
    dark_rows = dark_rows[np.isfinite(dark_rows["unit"].to_numpy(dtype=float))].copy()
    dark_rows["unit"] = dark_rows["unit"].astype(int)
    dark_rows = dark_rows[["unit", "dark_firing_rate_hz"]]

    joined = glm_rows.merge(dark_rows, on="unit", how="left")
    if tuning_similarity_table is not None and len(tuning_similarity_table):
        tuning_rows = tuning_similarity_table.copy()
        tuning_rows["unit"] = pd.to_numeric(tuning_rows["unit"], errors="coerce")
        tuning_rows = tuning_rows[
            np.isfinite(tuning_rows["unit"].to_numpy(dtype=float))
        ].copy()
        tuning_rows["unit"] = tuning_rows["unit"].astype(int)
        joined = joined.merge(
            tuning_rows[
                [
                    "unit",
                    "same_turn_tuning_similarity",
                    "tuning_source_path",
                ]
            ],
            on="unit",
            how="left",
        )
    else:
        joined = joined.assign(
            same_turn_tuning_similarity=np.nan,
            tuning_source_path="",
        )
    dark_rate_values = np.asarray(joined["dark_firing_rate_hz"], dtype=float)
    dark_active = np.isfinite(dark_rate_values) & (
        dark_rate_values >= float(dark_activity_threshold_hz)
    )
    joined = joined.assign(
        animal_name=animal_name,
        date=date,
        epoch_type=epoch_type,
        label=HEATMAP_EPOCH_LABELS[epoch_type],
        dark_epoch=dark_epoch,
        dark_active=dark_active,
        dark_activity_group=np.where(dark_active, "Dark active", "Dark inactive"),
    )
    return joined[
        [
            "animal_name",
            "date",
            "unit",
            "epoch_type",
            "label",
            "glm_epoch",
            "dark_epoch",
            "dark_firing_rate_hz",
            "dark_active",
            "dark_activity_group",
            "same_turn_tuning_similarity",
            "tuning_source_path",
            "ripple_devexp_mean",
            "ripple_devexp_p_value",
            "ripple_glm_source_path",
        ]
    ]


def load_dark_same_turn_tuning_similarity_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str = DEFAULT_PANEL_D_REGION,
    tuning_similarity_metric: str = DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC,
    tuning_comparison_label: str = DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL,
) -> Any:
    """Load dark same-turn tuning similarity for one session and region."""
    import pandas as pd

    tuning_path = get_tuning_similarity_path(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=dark_epoch,
        similarity_metric=tuning_similarity_metric,
    )
    if not tuning_path.exists():
        raise FileNotFoundError(str(tuning_path))

    tuning_table = pd.read_parquet(tuning_path)
    missing_columns = [
        column
        for column in ("unit", "region", "epoch", "comparison_label", "similarity")
        if column not in tuning_table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Tuning similarity table {tuning_path} is missing columns "
            f"{missing_columns!r}."
        )
    tuning_rows = tuning_table[
        (tuning_table["region"].astype(str) == region)
        & (tuning_table["epoch"].astype(str) == dark_epoch)
        & (tuning_table["comparison_label"].astype(str) == tuning_comparison_label)
    ].copy()
    tuning_rows["unit"] = pd.to_numeric(tuning_rows["unit"], errors="coerce")
    tuning_rows = tuning_rows[np.isfinite(tuning_rows["unit"].to_numpy(dtype=float))].copy()
    tuning_rows["unit"] = tuning_rows["unit"].astype(int)
    tuning_rows = tuning_rows.assign(tuning_source_path=str(tuning_path))
    return tuning_rows.rename(
        columns={"similarity": "same_turn_tuning_similarity"}
    )[
        [
            "unit",
            "same_turn_tuning_similarity",
            "tuning_source_path",
        ]
    ]


def load_glm_dark_activity_devexp_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
    region: str = DEFAULT_PANEL_D_REGION,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    dark_activity_threshold_hz: float = PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ,
    tuning_similarity_metric: str = DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC,
    tuning_comparison_label: str = DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL,
    epoch_types: Sequence[str] = PANEL_D_EPOCH_ORDER,
    dark_movement_fr_cache_dir: Path | None = DEFAULT_FIGURE_CACHE_DIR,
    refresh_dark_movement_fr_cache: bool = False,
) -> dict[str, Any]:
    """Load ripple-GLM deviance explained split by dark activity threshold."""
    import pandas as pd

    rows: list[Any] = []
    missing_artifacts: list[dict[str, str]] = []
    selected_epoch_types = tuple(epoch_types)

    for dataset_id in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset_id)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name=animal_name,
            date=date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        _dark_animal, _dark_date, dark_run_epoch = normalize_dataset_id(epoch_ids["dark"])
        try:
            dark_activity_table = load_dark_movement_firing_rate_table(
                data_root,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_run_epoch,
                region=region,
                cache_dir=dark_movement_fr_cache_dir,
                refresh_cache=refresh_dark_movement_fr_cache,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            missing_artifacts.append(
                {
                    "artifact": "dark_activity",
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": dark_run_epoch,
                    "path": str(get_dataset_analysis_path(data_root, animal_name, date)),
                    "reason": str(exc),
                }
            )
            continue

        try:
            tuning_similarity_table = load_dark_same_turn_tuning_similarity_table(
                data_root,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_run_epoch,
                region=region,
                tuning_similarity_metric=tuning_similarity_metric,
                tuning_comparison_label=tuning_comparison_label,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            missing_artifacts.append(
                {
                    "artifact": "tuning_analysis",
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": dark_run_epoch,
                    "path": str(
                        get_tuning_similarity_path(
                            data_root,
                            animal_name=animal_name,
                            date=date,
                            region=region,
                            epoch=dark_run_epoch,
                            similarity_metric=tuning_similarity_metric,
                        )
                    ),
                    "reason": str(exc),
                }
            )
            tuning_similarity_table = None

        for epoch_type in selected_epoch_types:
            if epoch_type not in epoch_ids:
                raise ValueError(f"Unknown Figure 2 epoch type: {epoch_type!r}")
            _glm_animal, _glm_date, glm_epoch = normalize_dataset_id(epoch_ids[epoch_type])
            glm_path = get_ripple_glm_path(
                data_root,
                animal_name=animal_name,
                date=date,
                epoch=glm_epoch,
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
            )
            if not glm_path.exists():
                missing_artifacts.append(
                    {
                        "artifact": "ripple_glm",
                        "animal_name": animal_name,
                        "date": date,
                        "epoch": glm_epoch,
                        "path": str(glm_path),
                    }
                )
                continue

            glm_table = load_ripple_glm_summary_table(
                data_root,
                [(animal_name, date, glm_epoch)],
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
            )
            rows.append(
                build_glm_dark_activity_devexp_table(
                    glm_table,
                    dark_activity_table,
                    tuning_similarity_table,
                    animal_name=animal_name,
                    date=date,
                    glm_epoch=glm_epoch,
                    epoch_type=epoch_type,
                    dark_epoch=dark_run_epoch,
                    dark_activity_threshold_hz=dark_activity_threshold_hz,
                )
            )

    devexp_table = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    return {
        "devexp_table": devexp_table,
        "missing_artifacts": missing_artifacts,
        "region": region,
        "dark_activity_threshold_hz": float(dark_activity_threshold_hz),
        "tuning_comparison_label": tuning_comparison_label,
        "tuning_similarity_metric": tuning_similarity_metric,
    }


def load_ripple_decoding_comparison_panel_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    categorical_metrics: Sequence[tuple[str, str]] = PANEL_E_CATEGORICAL_METRICS,
) -> dict[str, Any]:
    """Load CA1-V1 Bayesian ripple decoding categorical agreement summaries."""
    import pandas as pd

    summary_rows: list[dict[str, Any]] = []
    missing_artifacts: list[dict[str, str]] = []
    metrics = tuple((str(representation), str(label_scheme)) for representation, label_scheme in categorical_metrics)
    base_required_columns = {
        "representation",
        "train_epoch",
        "decode_epoch",
        "n_ripples",
        "n_ripple_bins",
        "n_effective_shuffles",
    }

    for dataset_id in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset_id)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name=animal_name,
            date=date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
        )
        for epoch_type in PANEL_E_EPOCH_ORDER:
            _epoch_animal, _epoch_date, epoch = normalize_dataset_id(epoch_ids[epoch_type])
            for representation, label_scheme in metrics:
                summary_path = get_ripple_decoding_comparison_summary_path(
                    data_root,
                    animal_name=animal_name,
                    date=date,
                    representation=representation,
                    train_epoch=epoch,
                    decode_epoch=epoch,
                )
                if not summary_path.exists():
                    missing_artifacts.append(
                        {
                            "artifact": "ripple_decoding_comparison",
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "representation": str(representation),
                            "label_scheme": str(label_scheme),
                            "path": str(summary_path),
                        }
                    )
                    continue

                table = pd.read_parquet(summary_path)
                metric_columns = {
                    f"{label_scheme}_scheme_applicable",
                    f"{label_scheme}_scheme_reason",
                    f"{label_scheme}_n_valid_ripples",
                    f"{label_scheme}_match_rate",
                    f"{label_scheme}_match_rate_shuffle_mean",
                    f"{label_scheme}_match_rate_shuffle_sd",
                    f"{label_scheme}_match_rate_p_value",
                }
                required_columns = base_required_columns | metric_columns
                missing_columns = sorted(required_columns.difference(table.columns))
                if missing_columns:
                    raise ValueError(
                        f"Ripple decoding comparison summary {summary_path} is missing "
                        f"columns {missing_columns!r}."
                    )
                table = table[
                    (table["representation"].astype(str) == representation)
                    & (table["train_epoch"].astype(str) == epoch)
                    & (table["decode_epoch"].astype(str) == epoch)
                ].copy()
                if table.empty:
                    raise ValueError(
                        "Ripple decoding comparison summary did not contain the requested "
                        f"row: {summary_path}"
                    )
                row = table.iloc[0]
                if not bool(row[f"{label_scheme}_scheme_applicable"]):
                    missing_artifacts.append(
                        {
                            "artifact": "ripple_decoding_comparison_scheme",
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "representation": str(representation),
                            "label_scheme": str(label_scheme),
                            "reason": str(row[f"{label_scheme}_scheme_reason"]),
                            "path": str(summary_path),
                        }
                    )
                    continue

                chance_level = PANEL_E_CHANCE_LEVELS.get(str(label_scheme), np.nan)
                summary_rows.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "representation": representation,
                        "train_epoch": epoch,
                        "decode_epoch": epoch,
                        "epoch_type": epoch_type,
                        "epoch_label": HEATMAP_EPOCH_LABELS[epoch_type],
                        "label_scheme": label_scheme,
                        "metric_label": PANEL_E_METRIC_LABELS.get(
                            (representation, label_scheme),
                            str(label_scheme).replace("_", " ").title(),
                        ),
                        "n_ripples": int(row["n_ripples"]),
                        "n_ripple_bins": int(row["n_ripple_bins"]),
                        "n_effective_shuffles": int(row["n_effective_shuffles"]),
                        "categorical_n_valid_ripples": int(
                            row[f"{label_scheme}_n_valid_ripples"]
                        ),
                        "categorical_match_rate": float(row[f"{label_scheme}_match_rate"]),
                        "categorical_match_rate_shuffle_mean": float(
                            row[f"{label_scheme}_match_rate_shuffle_mean"]
                        ),
                        "categorical_match_rate_shuffle_sd": float(
                            row[f"{label_scheme}_match_rate_shuffle_sd"]
                        ),
                        "categorical_match_rate_p_value": float(
                            row[f"{label_scheme}_match_rate_p_value"]
                        ),
                        "chance_level": float(chance_level),
                        "source_path": str(summary_path),
                    }
                )

    summary_table = (
        pd.DataFrame(summary_rows)
        if summary_rows
        else pd.DataFrame()
    )
    return {
        "summary_table": summary_table,
        "missing_artifacts": missing_artifacts,
        "categorical_metrics": metrics,
    }


def _format_panel_e_target_window_label(
    target_window_offset_s: float,
    target_window_s: float,
) -> str:
    """Return a compact millisecond label for one V1 target window."""
    start_ms = int(round(float(target_window_offset_s) * 1000.0))
    end_ms = int(round((float(target_window_offset_s) + float(target_window_s)) * 1000.0))
    return f"{start_ms} to {end_ms}"


def load_glm_offset_panel_tables(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
    epoch_types: Sequence[str] = HEATMAP_EPOCH_ORDER,
    target_window_offsets_s: Sequence[float] = PANEL_E_GLM_TARGET_WINDOW_OFFSETS_S,
    source_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    source_window_offset_s: float = PANEL_E_GLM_SOURCE_WINDOW_OFFSET_S,
    target_window_s: float = PANEL_E_GLM_TARGET_WINDOW_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load complete CA1-to-V1 ripple-GLM target-offset comparisons."""
    import pandas as pd
    import xarray as xr

    offsets = tuple(float(offset) for offset in target_window_offsets_s)
    selected_epoch_types = tuple(str(epoch_type) for epoch_type in epoch_types)
    unknown_epoch_types = sorted(set(selected_epoch_types).difference(HEATMAP_EPOCH_ORDER))
    if unknown_epoch_types:
        raise ValueError(f"Unknown GLM offset epoch types: {unknown_epoch_types!r}")
    unit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    missing_artifacts: list[dict[str, Any]] = []
    skipped_comparisons: list[dict[str, Any]] = []

    for dataset_id in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset_id)
        epoch_ids = make_figure_2_epoch_ids(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dataset_dark_epoch if dark_epoch is None else dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        for epoch_type in selected_epoch_types:
            _epoch_animal, _epoch_date, epoch = normalize_dataset_id(epoch_ids[epoch_type])
            paths_by_offset: dict[float, Path] = {}
            missing_offsets: list[float] = []
            for target_offset_s in offsets:
                path = get_ripple_glm_model_window_path(
                    data_root,
                    animal_name=animal_name,
                    date=date,
                    epoch=epoch,
                    source_window_s=source_window_s,
                    source_window_offset_s=source_window_offset_s,
                    target_window_s=target_window_s,
                    target_window_offset_s=target_offset_s,
                    ripple_selection=ripple_selection,
                    ridge_strength=ridge_strength,
                )
                paths_by_offset[target_offset_s] = path
                if not path.exists():
                    missing_offsets.append(target_offset_s)
                    missing_artifacts.append(
                        {
                            "artifact": "ripple_glm_offset",
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "epoch_type": epoch_type,
                            "target_window_offset_s": target_offset_s,
                            "path": str(path),
                        }
                    )
            if missing_offsets:
                skipped_comparisons.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "epoch": epoch,
                        "epoch_type": epoch_type,
                        "missing_target_window_offsets_s": tuple(missing_offsets),
                    }
                )
                continue

            for target_offset_s in offsets:
                path = paths_by_offset[target_offset_s]
                dataset = xr.load_dataset(path)
                try:
                    required_variables = (
                        "ripple_devexp_mean",
                        "ripple_devexp_p_value",
                    )
                    missing_variables = [
                        variable
                        for variable in required_variables
                        if variable not in dataset.data_vars
                    ]
                    if missing_variables:
                        raise ValueError(
                            f"Ripple-GLM offset output is missing variables "
                            f"{missing_variables!r}: {path}"
                        )
                    unit_ids = np.asarray(dataset.coords["unit"].values)
                    devexp = np.asarray(dataset["ripple_devexp_mean"].values, dtype=float)
                    p_values = np.asarray(dataset["ripple_devexp_p_value"].values, dtype=float)
                    finite_mask = np.isfinite(devexp) & np.isfinite(p_values)
                    significant_mask = finite_mask & (p_values < SIGNIFICANCE_P_VALUE) & (devexp > 0.0)
                    finite_devexp = devexp[finite_mask]
                    significant_devexp = devexp[significant_mask]
                    n_units = int(np.sum(finite_mask))
                    n_significant = int(np.sum(significant_mask))
                    fraction_significant = (
                        n_significant / n_units
                        if n_units > 0
                        else np.nan
                    )
                    summary_rows.append(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "epoch": epoch,
                            "epoch_type": epoch_type,
                            "epoch_label": HEATMAP_EPOCH_LABELS[epoch_type],
                            "target_window_offset_s": target_offset_s,
                            "target_window_label": _format_panel_e_target_window_label(
                                target_offset_s,
                                target_window_s,
                            ),
                            "n_units": n_units,
                            "n_significant_positive": n_significant,
                            "fraction_significant_positive": fraction_significant,
                            "median_devexp_all": (
                                float(np.nanmedian(finite_devexp))
                                if finite_devexp.size
                                else np.nan
                            ),
                            "median_devexp_significant": (
                                float(np.nanmedian(significant_devexp))
                                if significant_devexp.size
                                else np.nan
                            ),
                            "n_ripples": int(
                                dataset.attrs.get(
                                    "n_ripples_after_selection",
                                    dataset.attrs.get("n_ripples", 0),
                                )
                            ),
                            "source_path": str(path),
                        }
                    )
                    for unit_id, unit_devexp, unit_p, is_finite, is_significant in zip(
                        unit_ids,
                        devexp,
                        p_values,
                        finite_mask,
                        significant_mask,
                        strict=True,
                    ):
                        if not bool(is_finite):
                            continue
                        unit_rows.append(
                            {
                                "animal_name": animal_name,
                                "date": date,
                                "epoch": epoch,
                                "epoch_type": epoch_type,
                                "epoch_label": HEATMAP_EPOCH_LABELS[epoch_type],
                                "unit_id": unit_id,
                                "target_window_offset_s": target_offset_s,
                                "target_window_label": _format_panel_e_target_window_label(
                                    target_offset_s,
                                    target_window_s,
                                ),
                                "ripple_devexp_mean": float(unit_devexp),
                                "ripple_devexp_p_value": float(unit_p),
                                "significant_positive": bool(is_significant),
                                "source_path": str(path),
                            }
                        )
                finally:
                    dataset.close()

    return {
        "summary_table": pd.DataFrame(summary_rows),
        "unit_table": pd.DataFrame(unit_rows),
        "missing_artifacts": missing_artifacts,
        "skipped_comparisons": skipped_comparisons,
        "target_window_offsets_s": offsets,
        "source_window_s": float(source_window_s),
        "source_window_offset_s": float(source_window_offset_s),
        "target_window_s": float(target_window_s),
    }


def load_example_glm_prediction(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load observed and predicted counts for the best example V1 GLM unit."""
    import xarray as xr

    path = get_ripple_glm_path(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )
    if not path.exists():
        raise FileNotFoundError(f"Ripple-GLM NetCDF not found: {path}")

    dataset = xr.load_dataset(path)
    try:
        missing_prediction_variables = [
            name
            for name in ("ripple_observed_count_oof", "ripple_predicted_count_oof")
            if name not in dataset.data_vars
        ]
        if missing_prediction_variables:
            raise ValueError(
                "Ripple-GLM output lacks held-out prediction variables "
                f"{missing_prediction_variables!r}: {path}"
            )
        devexp = np.asarray(dataset["ripple_devexp_mean"].values, dtype=float)
        finite_indices = np.flatnonzero(np.isfinite(devexp))
        if finite_indices.size == 0:
            raise ValueError(f"Ripple-GLM output has no finite deviance values: {path}")
        unit_index = int(finite_indices[np.argmax(devexp[finite_indices])])
        unit_id = np.asarray(dataset.coords["unit"].values)[unit_index]
        observed = np.asarray(dataset["ripple_observed_count_oof"].values[:, unit_index], dtype=float)
        predicted = np.asarray(dataset["ripple_predicted_count_oof"].values[:, unit_index], dtype=float)
        p_value = float(dataset["ripple_devexp_p_value"].values[unit_index])
        metric_value = float(devexp[unit_index])
    finally:
        dataset.close()

    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "unit_id": unit_id,
        "observed": observed,
        "predicted": predicted,
        "ripple_devexp_mean": metric_value,
        "ripple_devexp_p_value": p_value,
        "source_path": str(path),
    }


def load_first_available_glm_prediction(
    data_root: Path,
    *,
    preferred_dataset: DatasetId,
    candidate_datasets: Sequence[DatasetId],
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> dict[str, Any]:
    """Load a held-out prediction example from the first compatible GLM file."""
    ordered_datasets = []
    seen: set[DatasetId] = set()
    for dataset in (preferred_dataset, *candidate_datasets):
        normalized_dataset = normalize_dataset_id(dataset)
        if normalized_dataset in seen:
            continue
        ordered_datasets.append(normalized_dataset)
        seen.add(normalized_dataset)

    errors = []
    for animal_name, date, epoch in ordered_datasets:
        try:
            return load_example_glm_prediction(
                data_root,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                ripple_window_s=ripple_window_s,
                ripple_window_offset_s=ripple_window_offset_s,
                ripple_selection=ripple_selection,
                ridge_strength=ridge_strength,
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            errors.append(f"{animal_name} {date} {epoch}: {exc}")

    raise FileNotFoundError(
        "Could not find a compatible ripple-GLM prediction example. "
        + " | ".join(errors)
    )


def draw_ripple_glm_schematic(
    ax: "Axes",
    ripple_trace: Mapping[str, Any] | None = None,
) -> None:
    """Draw a compact schematic of the CA1-to-V1 ripple count GLM."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    transform = ax.transAxes

    ca1_color = REGION_COLORS["ca1"]
    v1_color = REGION_COLORS["v1"]
    time_min_s = -0.08
    time_max_s = 0.22
    row_label_x = 0.095
    trace_x0 = 0.13
    trace_x1 = 0.94
    count_window_x0 = trace_x0 + (0.0 - time_min_s) / (time_max_s - time_min_s) * (
        trace_x1 - trace_x0
    )
    count_window_x1 = trace_x0 + (0.20 - time_min_s) / (time_max_s - time_min_s) * (
        trace_x1 - trace_x0
    )

    ax.add_patch(
        Rectangle(
            (count_window_x0, 0.43),
            count_window_x1 - count_window_x0,
            0.46,
            facecolor=SCHEMATIC_COLORS["ripple_window_fill"],
            edgecolor="none",
            alpha=0.65,
            transform=transform,
            zorder=0,
        )
    )
    ax.plot(
        [count_window_x0, count_window_x0],
        [0.42, 0.91],
        color=SCHEMATIC_COLORS["ripple_onset"],
        linewidth=0.75,
        transform=transform,
        zorder=3,
    )
    ax.plot(
        [count_window_x1, count_window_x1],
        [0.43, 0.88],
        color="0.50",
        linewidth=0.55,
        linestyle=":",
        transform=transform,
        zorder=3,
    )
    ax.text(
        (count_window_x0 + count_window_x1) / 2.0,
        0.905,
        "0-200 ms",
        ha="center",
        va="top",
        fontsize=5.3,
        transform=transform,
    )
    ax.text(count_window_x0, 0.930, "onset", ha="center", va="bottom", fontsize=4.8, transform=transform)

    if ripple_trace is not None:
        trace_time_s = np.asarray(ripple_trace.get("time_s", []), dtype=float)
        trace_lfp = np.asarray(ripple_trace.get("filtered_lfp", []), dtype=float)
        trace_mask = (
            np.isfinite(trace_time_s)
            & np.isfinite(trace_lfp)
            & (trace_time_s >= time_min_s)
            & (trace_time_s <= time_max_s)
        )
    else:
        trace_time_s = np.asarray([], dtype=float)
        trace_lfp = np.asarray([], dtype=float)
        trace_mask = np.asarray([], dtype=bool)
    if np.any(trace_mask):
        plot_time_s = trace_time_s[trace_mask]
        plot_lfp = trace_lfp[trace_mask] - np.nanmedian(trace_lfp[trace_mask])
        lfp_scale = np.nanpercentile(np.abs(plot_lfp), 98)
        if not np.isfinite(lfp_scale) or lfp_scale <= 0:
            lfp_scale = np.nanmax(np.abs(plot_lfp))
        plot_lfp = plot_lfp / lfp_scale if np.isfinite(lfp_scale) and lfp_scale > 0 else plot_lfp
    else:
        plot_time_s = np.linspace(time_min_s, time_max_s, 120)
        envelope = np.exp(-((plot_time_s - 0.055) / 0.065) ** 2)
        plot_lfp = 0.35 * np.sin(2.0 * np.pi * 9.0 * plot_time_s)
        plot_lfp += envelope * np.sin(2.0 * np.pi * 170.0 * plot_time_s)
    ripple_x = trace_x0 + (plot_time_s - time_min_s) / (time_max_s - time_min_s) * (
        trace_x1 - trace_x0
    )
    ripple_y = 0.78 + 0.065 * np.clip(plot_lfp, -1.6, 1.6)
    ax.plot([trace_x0, trace_x1], [0.78, 0.78], color="0.75", linewidth=0.35, transform=transform)
    ax.plot(
        ripple_x,
        ripple_y,
        color=SCHEMATIC_COLORS["ripple_trace"],
        linewidth=0.75,
        transform=transform,
    )
    ax.text(
        row_label_x,
        0.80,
        "ripple\nLFP",
        ha="right",
        va="center",
        fontsize=5.0,
        transform=transform,
    )

    def _time_to_x(time_s: float) -> float:
        return trace_x0 + (float(time_s) - time_min_s) / (time_max_s - time_min_s) * (
            trace_x1 - trace_x0
        )

    def _draw_raster_rows(
        y_top: float,
        color: str,
        spike_times_s: Sequence[Sequence[float]],
        *,
        row_step: float = 0.020,
    ) -> None:
        for row_index, row_spikes in enumerate(spike_times_s):
            y0 = y_top - row_index * row_step
            ax.plot(
                [trace_x0, trace_x1],
                [y0, y0],
                color="0.87",
                linewidth=0.25,
                transform=transform,
                zorder=0,
            )
            for spike_time_s in row_spikes:
                if not np.isfinite(spike_time_s) or spike_time_s < time_min_s or spike_time_s > time_max_s:
                    continue
                spike_x = _time_to_x(spike_time_s)
                ax.plot(
                    [spike_x, spike_x],
                    [y0 - 0.0075, y0 + 0.0075],
                    color=color,
                    linewidth=0.65,
                    solid_capstyle="butt",
                    transform=transform,
                )

    fallback_ca1_spikes = (
        (0.012, 0.038, 0.112, 0.156),
        (0.028, 0.074, 0.136),
        (0.004, 0.094, 0.176),
        (0.018, 0.058, 0.148),
        (0.042, 0.122),
    )
    fallback_v1_spikes = (
        (0.030, 0.126),
        (0.066, 0.088, 0.168),
        (0.046, 0.150),
        (0.024, 0.104, 0.194),
        (0.074, 0.138),
    )

    def _get_spike_raster(key: str, fallback: Sequence[Sequence[float]]) -> tuple[np.ndarray, ...]:
        if ripple_trace is None or key not in ripple_trace:
            return tuple(np.asarray(row, dtype=float) for row in fallback)
        rows = tuple(np.asarray(row, dtype=float) for row in ripple_trace[key])
        return rows if rows else tuple(np.asarray(row, dtype=float) for row in fallback)

    ca1_spikes = _get_spike_raster("ca1_spike_times_s", fallback_ca1_spikes)
    v1_spikes = _get_spike_raster("v1_spike_times_s", fallback_v1_spikes)
    row_step = 0.020
    ca1_top_y = 0.665
    v1_top_y = 0.535
    ca1_center_y = ca1_top_y - row_step * max(len(ca1_spikes) - 1, 0) / 2.0
    v1_center_y = v1_top_y - row_step * max(len(v1_spikes) - 1, 0) / 2.0
    ax.text(row_label_x, ca1_center_y, "CA1", ha="right", va="center", fontsize=5.4, color=ca1_color, transform=transform)
    ax.text(row_label_x, v1_center_y, "V1", ha="right", va="center", fontsize=5.4, color=v1_color, transform=transform)
    _draw_raster_rows(ca1_top_y, ca1_color, ca1_spikes, row_step=row_step)
    _draw_raster_rows(v1_top_y, v1_color, v1_spikes, row_step=row_step)

    circle_y = np.linspace(0.355, 0.120, 8)
    ca1_column_x = 0.17
    v1_column_x = 0.83
    glm_x0 = 0.405
    glm_y0 = float(np.mean(circle_y)) - 0.105 / 2.0
    glm_width = 0.19
    glm_height = 0.105
    glm_center_y = glm_y0 + glm_height / 2.0

    ax.text(
        ca1_column_x,
        0.385,
        "CA1 counts",
        ha="center",
        va="bottom",
        fontsize=4.9,
        color=ca1_color,
        transform=transform,
    )
    ax.text(
        v1_column_x,
        0.385,
        "V1 counts",
        ha="center",
        va="bottom",
        fontsize=4.9,
        color=v1_color,
        transform=transform,
    )
    for y_value in circle_y:
        ax.plot(
            [ca1_column_x + 0.035, glm_x0],
            [y_value, glm_center_y],
            color="0.35",
            linewidth=0.34,
            alpha=0.72,
            transform=transform,
            zorder=1,
        )
        ax.plot(
            [glm_x0 + glm_width, v1_column_x - 0.035],
            [glm_center_y, y_value],
            color="0.35",
            linewidth=0.34,
            alpha=0.72,
            transform=transform,
            zorder=1,
        )
    for start, end in (
        ((ca1_column_x + 0.062, glm_center_y), (glm_x0 - 0.008, glm_center_y)),
        ((glm_x0 + glm_width + 0.008, glm_center_y), (v1_column_x - 0.062, glm_center_y)),
    ):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.55,
                color="0.20",
                transform=transform,
                zorder=2,
            )
        )
    ax.scatter(
        np.full_like(circle_y, ca1_column_x, dtype=float),
        circle_y,
        s=17,
        marker="o",
        facecolor=SCHEMATIC_COLORS["ca1_count_fill"],
        edgecolor=ca1_color,
        linewidth=0.65,
        transform=transform,
        zorder=3,
    )
    ax.scatter(
        np.full_like(circle_y, v1_column_x, dtype=float),
        circle_y,
        s=17,
        marker="o",
        facecolor=SCHEMATIC_COLORS["v1_count_fill"],
        edgecolor=v1_color,
        linewidth=0.65,
        transform=transform,
        zorder=3,
    )
    ax.add_patch(
        Rectangle(
            (glm_x0, glm_y0),
            glm_width,
            glm_height,
            facecolor=SCHEMATIC_COLORS["glm_fill"],
            edgecolor="0.25",
            linewidth=0.65,
            transform=transform,
            zorder=2,
        )
    )
    ax.text(
        glm_x0 + glm_width / 2.0,
        glm_center_y,
        "GLM",
        ha="center",
        va="center",
        fontsize=5.0,
        color="black",
        transform=transform,
        zorder=4,
    )
    ax.text(
        0.50,
        0.045,
        "Held-out prediction",
        ha="center",
        va="center",
        fontsize=5.4,
        transform=transform,
    )


def plot_ripple_lfp_panel(ax: "Axes", trace: dict[str, Any]) -> None:
    """Plot one ripple-band LFP snippet around a ripple start."""
    time_s = np.asarray(trace["time_s"], dtype=float)
    lfp = np.asarray(trace["filtered_lfp"], dtype=float)
    ax.plot(time_s, lfp, color="black", linewidth=0.6)
    ax.axvspan(
        0.0,
        float(trace["ripple_duration_s"]),
        color=SCHEMATIC_COLORS["ripple_span"],
        alpha=0.28,
        linewidth=0,
    )
    ax.axvline(0.0, color=SCHEMATIC_COLORS["ripple_onset"], linewidth=0.7)
    ax.set_xlabel("Time from ripple start (s)")
    ax.set_ylabel("Ripple-band LFP")
    ax.set_title(
        f"{trace['animal_name']} {trace['date']} {trace['epoch']} ch {trace['channel']}",
        fontsize=7,
        pad=2,
    )
    if np.isfinite(trace["mean_zscore"]):
        label = f"z={float(trace['mean_zscore']):.1f}\nn={trace['n_ripples']}"
    else:
        label = f"n={trace['n_ripples']}"
    ax.text(
        0.98,
        0.95,
        label,
        ha="right",
        va="top",
        fontsize=6,
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6, length=2, pad=1)


def plot_peri_ripple_heatmap_panel(
    ax: "Axes",
    firing_rate_table: Any,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
) -> None:
    """Plot example peri-ripple firing-rate heatmaps for each region."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    n_regions = len(regions)
    if n_regions == 0:
        ax.text(0.5, 0.5, "No regions", ha="center", va="center", transform=ax.transAxes)
        return

    image = None
    for index, region in enumerate(regions):
        y_top = 0.92 - index * (0.82 / n_regions)
        height = 0.70 / n_regions
        heatmap_ax = ax.inset_axes([0.14, y_top - height, 0.76, height])
        payload = build_peri_ripple_heatmap_payload(firing_rate_table, region=region)
        matrix = np.asarray(payload["mean_rate_hz"], dtype=float)
        time_s = np.asarray(payload["time_s"], dtype=float)
        if matrix.size == 0 or time_s.size == 0:
            heatmap_ax.text(
                0.5,
                0.5,
                f"No {region.upper()} units",
                ha="center",
                va="center",
                transform=heatmap_ax.transAxes,
            )
        else:
            normalized = normalize_heatmap_rows(matrix)
            row_peak = np.full(normalized.shape[0], -np.inf, dtype=float)
            finite_rows = np.isfinite(matrix).any(axis=1)
            if np.any(finite_rows):
                row_peak[finite_rows] = np.nanmax(matrix[finite_rows], axis=1)
            order = np.argsort(-row_peak, kind="stable")
            image = heatmap_ax.imshow(
                normalized[order],
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                extent=[time_s[0], time_s[-1], normalized.shape[0], 0],
            )
        heatmap_ax.axvline(0.0, color="white", linewidth=0.6, alpha=0.9)
        heatmap_ax.set_ylabel(region.upper(), fontsize=6, labelpad=2)
        heatmap_ax.set_yticks([])
        if index == n_regions - 1:
            heatmap_ax.set_xlabel("Time from ripple start (s)", fontsize=6, labelpad=1)
        else:
            heatmap_ax.set_xticklabels([])
        heatmap_ax.tick_params(labelsize=5, length=2, pad=1)

    if image is not None:
        colorbar_ax = ax.inset_axes([0.93, 0.18, 0.03, 0.68])
        colorbar = ax.figure.colorbar(image, cax=colorbar_ax, ticks=[0.0, 1.0])
        colorbar.ax.tick_params(labelsize=5, length=2, pad=1)
        colorbar.set_label("Norm. FR", fontsize=5, labelpad=2)


def draw_neuron_scale_bar(
    ax: "Axes",
    *,
    neuron_count: int = NEURON_SCALE_BAR_COUNT,
    x: float = 1.02,
) -> None:
    """Draw a vertical data-scaled neuron count bar beside one heatmap axis."""
    from matplotlib.transforms import blended_transform_factory

    if neuron_count <= 0:
        raise ValueError("neuron_count must be positive.")

    y_limits = [float(value) for value in ax.get_ylim()]
    y_min = min(y_limits)
    y_max = max(y_limits)
    y_span = y_max - y_min
    margin = max(8.0, 0.28 * y_span)
    if y_span >= neuron_count + margin:
        y_bottom = y_max - margin
        y_top = y_bottom - float(neuron_count)
    else:
        y_top = y_min
        y_bottom = min(y_max, y_min + float(neuron_count))

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(
        [x, x],
        [y_bottom, y_top],
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
        transform=transform,
        clip_on=False,
    )
    ax.text(
        x + 0.035,
        (y_bottom + y_top) / 2,
        f"{neuron_count} neurons",
        ha="left",
        va="center",
        rotation=90,
        fontsize=5,
        transform=transform,
        clip_on=False,
    )


def plot_epoch_ripple_heatmap_panel(
    ax: "Axes",
    epoch_tables: Sequence[dict[str, Any]],
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
) -> None:
    """Plot ripple-triggered firing-rate heatmaps across light, dark, and sleep epochs."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    n_epochs = len(epoch_tables)
    n_regions = len(regions)
    if n_epochs == 0 or n_regions == 0:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center", transform=ax.transAxes)
        return

    left = 0.10
    right = 0.93
    heatmap_bottom = 0.36
    heatmap_top = 0.96
    hist_bottom = 0.08
    hist_height = 0.18
    column_gap = 0.035
    row_gap = 0.045
    cell_width = (right - left - column_gap * (n_epochs - 1)) / n_epochs
    available_heatmap_height = heatmap_top - heatmap_bottom - row_gap * (n_regions - 1)
    prepared_epoch_payloads = []
    max_region_unit_total = 0
    for epoch_payload in epoch_tables:
        firing_rate_table = epoch_payload["firing_rate_table"]
        region_payloads = []
        region_unit_total = 0
        for region in regions:
            payload = build_peri_ripple_heatmap_payload(firing_rate_table, region=region)
            unit_count = int(np.asarray(payload["mean_rate_hz"], dtype=float).shape[0])
            region_payloads.append(
                {
                    "region": region,
                    "payload": payload,
                    "unit_count": unit_count,
                }
            )
            region_unit_total += unit_count
        max_region_unit_total = max(max_region_unit_total, region_unit_total)
        prepared_epoch_payloads.append(
            {
                "epoch_payload": epoch_payload,
                "region_payloads": region_payloads,
            }
        )
    unit_height = (
        available_heatmap_height / max_region_unit_total
        if max_region_unit_total > 0
        else available_heatmap_height / n_regions
    )
    empty_region_height = min(0.04, available_heatmap_height / n_regions)
    image = None
    last_heatmap_ax = None

    for col_index, prepared_epoch_payload in enumerate(prepared_epoch_payloads):
        epoch_payload = prepared_epoch_payload["epoch_payload"]
        x0 = left + col_index * (cell_width + column_gap)
        panel_a_title = {
            "light": "",
            "sleep": "Sleep",
        }.get(str(epoch_payload["epoch_type"]), str(epoch_payload["label"]))
        if panel_a_title:
            ax.text(
                x0 + cell_width / 2,
                0.96,
                panel_a_title,
                ha="center",
                va="top",
                fontsize=6,
                transform=ax.transAxes,
            )
        y_top = heatmap_top
        for row_index, region_payload in enumerate(prepared_epoch_payload["region_payloads"]):
            region = region_payload["region"]
            payload = region_payload["payload"]
            unit_count = region_payload["unit_count"]
            cell_height = (
                unit_count * unit_height
                if max_region_unit_total > 0
                else unit_height
            )
            if max_region_unit_total > 0 and unit_count == 0:
                cell_height = empty_region_height
            y0 = y_top - cell_height
            heatmap_ax = ax.inset_axes([x0, y0, cell_width, cell_height])
            last_heatmap_ax = heatmap_ax
            matrix = np.asarray(payload["mean_rate_hz"], dtype=float)
            time_s = np.asarray(payload["time_s"], dtype=float)
            if matrix.size == 0 or time_s.size == 0:
                heatmap_ax.text(
                    0.5,
                    0.5,
                    "No units",
                    ha="center",
                    va="center",
                    fontsize=5,
                    transform=heatmap_ax.transAxes,
                )
            else:
                normalized = normalize_heatmap_rows(matrix)
                row_peak = np.full(normalized.shape[0], -np.inf, dtype=float)
                finite_rows = np.isfinite(matrix).any(axis=1)
                if np.any(finite_rows):
                    row_peak[finite_rows] = np.nanmax(matrix[finite_rows], axis=1)
                order = np.argsort(-row_peak, kind="stable")
                image = heatmap_ax.imshow(
                    normalized[order],
                    origin="upper",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=0.0,
                    vmax=1.0,
                    cmap="viridis",
                    extent=[time_s[0], time_s[-1], normalized.shape[0], 0],
                )
            heatmap_ax.axvline(0.0, color="white", linewidth=0.55, alpha=0.9)
            heatmap_ax.set_yticks([])
            if col_index == 0:
                heatmap_ax.set_ylabel(region.upper(), fontsize=6, labelpad=2)
            if row_index == n_regions - 1:
                heatmap_ax.set_xlabel("Time (s)", fontsize=5, labelpad=1)
            else:
                heatmap_ax.set_xticklabels([])
            heatmap_ax.tick_params(labelsize=5, length=1.5, pad=1)
            y_top = y0 - row_gap

        summary_table = epoch_payload.get("summary_table")
        hist_ax = ax.inset_axes([x0, hist_bottom, cell_width, hist_height])
        _plot_modulation_histogram_inset(
            hist_ax,
            summary_table,
            regions=regions,
            show_ylabel=col_index == 0,
            show_legend=col_index == n_epochs - 1,
        )

    if image is not None:
        if last_heatmap_ax is not None:
            draw_neuron_scale_bar(last_heatmap_ax)
        colorbar_height = 0.30
        colorbar_bottom = heatmap_bottom + 0.5 * (
            heatmap_top - heatmap_bottom - colorbar_height
        )
        colorbar_ax = ax.inset_axes([0.975, colorbar_bottom, 0.018, colorbar_height])
        colorbar = ax.figure.colorbar(image, cax=colorbar_ax, ticks=[0.0, 1.0])
        colorbar.ax.tick_params(labelsize=5, length=2, pad=1)
        colorbar.set_label("Norm. FR", fontsize=5, labelpad=2)


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return weights that normalize one histogram to a fraction of units."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _format_region_summary(values: np.ndarray) -> str:
    """Return short median and positive-fraction text for one region."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n=0"
    return f"n={values.size}, med={np.median(values):.2f}, frac>0={np.mean(values > 0):.2f}"


def _get_modulation_index_values(summary_table: Any, region: str) -> np.ndarray:
    """Return finite ripple-modulation index values for one region."""
    if summary_table is None:
        return np.asarray([], dtype=float)
    columns = getattr(summary_table, "columns", ())
    if "region" not in columns or "ripple_modulation_index" not in columns:
        return np.asarray([], dtype=float)
    values = np.asarray(
        summary_table.loc[
            summary_table["region"].astype(str) == region,
            "ripple_modulation_index",
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _plot_modulation_histogram_inset(
    ax: "Axes",
    summary_table: Any,
    *,
    regions: Sequence[str],
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    """Plot a compact region-colored modulation-index histogram."""
    bins = np.linspace(-1.0, 1.0, 21)
    has_values = False
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.55, zorder=1)
    for region in regions:
        values = _get_modulation_index_values(summary_table, region)
        if not values.size:
            continue
        has_values = True
        ax.hist(
            values,
            bins=bins,
            weights=_fraction_histogram_weights(values),
            color=REGION_COLORS.get(region, "0.5"),
            label=region.upper(),
            **COMPACT_HISTOGRAM_KWARGS,
            zorder=2,
        )
    if not has_values:
        ax.text(0.5, 0.5, "No index", ha="center", va="center", fontsize=5, transform=ax.transAxes)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Mod. index", fontsize=5, labelpad=1)
    if show_ylabel:
        ax.set_ylabel("Frac.", fontsize=5, labelpad=1)
    else:
        ax.set_yticklabels([])
    if show_legend and has_values:
        ax.legend(frameon=False, fontsize=5, handlelength=0.8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5, length=1.5, pad=1)


def plot_top_ca1_xcorr_panel(ax: "Axes", payload: dict[str, Any]) -> None:
    """Plot top CA1 units' CA1-V1 xcorr heatmaps with a shared V1 order."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    xcorr_values = np.asarray(payload["xcorr"], dtype=float)
    lag_s = np.asarray(payload["lag_s"], dtype=float)
    ca1_unit_ids = np.asarray(payload["ca1_unit_ids"])
    if xcorr_values.ndim != 3:
        raise ValueError(f"Expected xcorr array with 3 dimensions, got {xcorr_values.shape}.")
    if xcorr_values.size == 0 or lag_s.size == 0:
        ax.text(0.5, 0.5, "No xcorr data", ha="center", va="center", transform=ax.transAxes)
        return

    n_ca1, n_v1, _n_lag = xcorr_values.shape
    display_vmax = float(payload.get("display_vmax", DEFAULT_XCORR_DISPLAY_VMAX))
    lag_min_s, lag_max_s = DEFAULT_XCORR_LAG_WINDOW_S
    lag_mask = (lag_s >= lag_min_s) & (lag_s <= lag_max_s)
    if not np.any(lag_mask):
        raise ValueError(
            f"Screen-xcorr lags do not overlap requested window {DEFAULT_XCORR_LAG_WINDOW_S}."
        )
    lag_plot_s = lag_s[lag_mask]
    xcorr_plot = xcorr_values[:, :, lag_mask]

    left = 0.10
    right = 0.93
    bottom = 0.10
    top = 0.89
    column_gap = 0.022
    cell_width = (right - left - column_gap * (n_ca1 - 1)) / n_ca1
    image = None
    for ca1_index, ca1_unit_id in enumerate(ca1_unit_ids):
        x0 = left + ca1_index * (cell_width + column_gap)
        heatmap_ax = ax.inset_axes([x0, bottom, cell_width, top - bottom])
        image = heatmap_ax.imshow(
            np.clip(xcorr_plot[ca1_index], 0.0, display_vmax),
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=[lag_plot_s[0], lag_plot_s[-1], n_v1, 0],
            vmin=0.0,
            vmax=display_vmax,
        )
        heatmap_ax.axvline(0.0, color="white", linewidth=0.3, alpha=0.9)
        heatmap_ax.set_xlim(lag_min_s, lag_max_s)
        heatmap_ax.set_title(f"CA1 {ca1_unit_id}", fontsize=5.4, pad=1)
        heatmap_ax.set_yticks([])
        heatmap_ax.tick_params(axis="x", labelsize=4.6, length=1.4, pad=1)
        heatmap_ax.tick_params(axis="y", length=0)

    ax.text(
        0.035,
        bottom + 0.5 * (top - bottom),
        "V1 units\n(shared order)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=5,
        transform=ax.transAxes,
    )

    ax.text(
        0.5 * (left + right),
        0.035,
        "Lag (s)",
        ha="center",
        va="bottom",
        fontsize=5,
        transform=ax.transAxes,
    )
    if image is not None:
        colorbar_height = 0.23
        colorbar_bottom = bottom + 0.5 * (top - bottom - colorbar_height)
        colorbar_ax = ax.inset_axes([0.955, colorbar_bottom, 0.026, colorbar_height])
        colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
        colorbar.ax.tick_params(labelsize=5, length=2, pad=1)
        colorbar.set_label("Norm. xcorr", fontsize=5, labelpad=2)


def plot_modulation_index_panel(
    ax: "Axes",
    summary_table: Any,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
) -> None:
    """Plot pooled ripple-modulation index distributions by region."""
    bins = np.linspace(-1.0, 1.0, 31)
    summary_lines = []
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=0.7, zorder=1)
    for region in regions:
        values = np.asarray(
            summary_table.loc[
                summary_table["region"].astype(str) == region,
                "ripple_modulation_index",
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(
                values,
                bins=bins,
                weights=_fraction_histogram_weights(values),
                color=REGION_COLORS.get(region, "0.5"),
                label=region.upper(),
                **HISTOGRAM_KWARGS,
                zorder=2,
            )
        summary_lines.append(f"{region.upper()}: {_format_region_summary(values)}")

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Ripple modulation index")
    ax.set_ylabel("Frac. units")
    ax.legend(frameon=False, fontsize=6, handlelength=1.0)
    ax.text(
        0.03,
        0.96,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=5.2,
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6, length=2, pad=1)


def plot_ripple_count_panel(ax: "Axes", count_table: Any) -> None:
    """Plot the number of thresholded ripples in each data-set epoch."""
    if count_table.empty:
        ax.text(0.5, 0.5, "No ripple counts", ha="center", va="center", transform=ax.transAxes)
        return
    positions = np.arange(len(count_table), dtype=float)
    labels = [
        f"{animal}\n{epoch}"
        for animal, epoch in zip(
            count_table["animal_name"].astype(str),
            count_table["epoch"].astype(str),
            strict=True,
        )
    ]
    ax.bar(
        positions,
        count_table["n_ripples"].to_numpy(dtype=float),
        color=SCHEMATIC_COLORS["ripple_trace"],
        alpha=0.82,
        width=0.7,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Ripples")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=1)


def plot_glm_summary_panel(ax: "Axes", glm_table: Any) -> None:
    """Plot pooled ripple-GLM deviance explained versus shuffle significance."""
    values = np.asarray(glm_table["ripple_devexp_mean"], dtype=float)
    p_values = np.asarray(glm_table["ripple_devexp_p_value"], dtype=float)
    valid = np.isfinite(values) & np.isfinite(p_values)
    ax.axhline(-np.log10(0.05), color="0.25", linestyle="--", linewidth=0.7, zorder=1)
    if np.any(valid):
        ax.scatter(
            values[valid],
            -np.log10(np.clip(p_values[valid], 1e-12, 1.0)),
            s=9,
            color=MODEL_COLOR,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        ax.text(
            0.97,
            0.05,
            f"n={int(np.sum(valid))}\nfrac p<0.05={np.mean(p_values[valid] < 0.05):.2f}",
            ha="right",
            va="bottom",
            fontsize=5.5,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No finite GLM values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Ripple deviance explained")
    ax.set_ylabel("-log10 shuffle p")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6, length=2, pad=1)


def plot_glm_analysis_panel(
    ax: "Axes",
    epoch_tables: Sequence[dict[str, Any]],
    ripple_trace: Mapping[str, Any] | None = None,
) -> None:
    """Plot the ripple-GLM schematic and epoch-specific performance summaries."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not epoch_tables:
        ax.text(0.5, 0.5, "No GLM data", ha="center", va="center", transform=ax.transAxes)
        return

    schematic_ax = ax.inset_axes([0.00, 0.04, 0.39, 0.91])
    draw_ripple_glm_schematic(schematic_ax, ripple_trace=ripple_trace)

    all_neglog_p: list[np.ndarray] = []
    for epoch_payload in epoch_tables:
        table = epoch_payload["summary_table"]
        values = np.asarray(table["ripple_devexp_mean"], dtype=float)
        p_values = np.asarray(table["ripple_devexp_p_value"], dtype=float)
        valid = np.isfinite(values) & np.isfinite(p_values)
        all_neglog_p.append(-np.log10(np.clip(p_values[valid], 1e-12, 1.0)))

    finite_neglog_p = np.concatenate([values for values in all_neglog_p if values.size]) if any(
        values.size for values in all_neglog_p
    ) else np.asarray([], dtype=float)
    x_min, x_max = PANEL_B_DEVIANCE_EXPLAINED_LIMITS
    y_max = max(2.0, float(np.nanmax(finite_neglog_p)) + 0.4) if finite_neglog_p.size else 2.0

    plot_left = 0.52
    plot_right = 0.98
    scatter_bottom = 0.38
    scatter_top = 0.94
    box_bottom = 0.06
    box_top = 0.29
    plot_gap = 0.025
    plot_width = (plot_right - plot_left - plot_gap * (len(epoch_tables) - 1)) / len(epoch_tables)
    for index, epoch_payload in enumerate(epoch_tables):
        table = epoch_payload["summary_table"]
        values = np.asarray(table["ripple_devexp_mean"], dtype=float)
        p_values = np.asarray(table["ripple_devexp_p_value"], dtype=float)
        valid = np.isfinite(values) & np.isfinite(p_values)
        plot_ax = ax.inset_axes(
            [
                plot_left + index * (plot_width + plot_gap),
                scatter_bottom,
                plot_width,
                scatter_top - scatter_bottom,
            ]
        )
        epoch_color = PANEL_BC_SIGNIFICANT_UNIT_COLOR
        plot_ax.axvline(0.0, color="0.45", linewidth=0.45, zorder=1)
        plot_ax.axhline(
            -np.log10(PANEL_C_SIGNIFICANCE_P_VALUE),
            color="0.25",
            linestyle="--",
            linewidth=0.55,
            zorder=1,
        )
        if np.any(valid):
            finite_values = values[valid]
            finite_p_values = p_values[valid]
            neglog_p = -np.log10(np.clip(finite_p_values, 1e-12, 1.0))
            significant = finite_p_values < PANEL_C_SIGNIFICANCE_P_VALUE
            if np.any(~significant):
                plot_ax.scatter(
                    finite_values[~significant],
                    neglog_p[~significant],
                    s=5,
                    color=NONSIGNIFICANT_COLOR,
                    alpha=0.45,
                    edgecolors="none",
                    zorder=2,
                )
            if np.any(significant):
                plot_ax.scatter(
                    finite_values[significant],
                    neglog_p[significant],
                    s=6,
                    color=epoch_color,
                    alpha=0.55,
                    edgecolors="none",
                    zorder=3,
                )
            plot_ax.text(
                0.96,
                0.05,
                f"n={int(np.sum(valid))}\nsig={np.mean(significant):.2f}",
                ha="right",
                va="bottom",
                fontsize=4.8,
                transform=plot_ax.transAxes,
            )
        else:
            plot_ax.text(
                0.5,
                0.5,
                "No finite\nvalues",
                ha="center",
                va="center",
                fontsize=5,
                transform=plot_ax.transAxes,
            )
        panel_c_title = {
            "light": "",
            "sleep": "Sleep",
        }.get(str(epoch_payload["epoch_type"]), str(epoch_payload["label"]))
        if panel_c_title:
            plot_ax.set_title(panel_c_title, fontsize=5.6, pad=1.5)
        plot_ax.set_xlim(x_min, x_max)
        plot_ax.set_ylim(0.0, y_max)
        plot_ax.tick_params(labelbottom=False)
        if index == 0:
            plot_ax.set_ylabel(
                r"-log10 $\mathit{p}$ from shuffle",
                fontsize=5,
                labelpad=1.0,
            )
        else:
            plot_ax.set_yticklabels([])
        plot_ax.spines["top"].set_visible(False)
        plot_ax.spines["right"].set_visible(False)
        plot_ax.tick_params(labelsize=4.8, length=1.5, pad=1)

        box_ax = ax.inset_axes(
            [
                plot_left + index * (plot_width + plot_gap),
                box_bottom,
                plot_width,
                box_top - box_bottom,
            ]
        )
        if np.any(valid):
            finite_values = values[valid]
            finite_p_values = p_values[valid]
            nonsig_values = finite_values[
                finite_p_values >= PANEL_C_SIGNIFICANCE_P_VALUE
            ]
            sig_values = finite_values[finite_p_values < PANEL_C_SIGNIFICANCE_P_VALUE]
            box_data = []
            box_positions = []
            box_colors = []
            if nonsig_values.size:
                box_data.append(nonsig_values)
                box_positions.append(1)
                box_colors.append(NONSIGNIFICANT_COLOR)
            if sig_values.size:
                box_data.append(sig_values)
                box_positions.append(2)
                box_colors.append(epoch_color)
            if box_data:
                box_artists = box_ax.boxplot(
                    box_data,
                    orientation="horizontal",
                    positions=box_positions,
                    widths=0.48,
                    patch_artist=True,
                    whis=(0, 100),
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 0.55},
                    whiskerprops={"color": "0.25", "linewidth": 0.45},
                    capprops={"color": "0.25", "linewidth": 0.45},
                )
                for patch, color in zip(box_artists["boxes"], box_colors, strict=False):
                    patch.set_facecolor(color)
                    patch.set_edgecolor("0.25")
                    patch.set_alpha(0.72)
                    patch.set_linewidth(0.45)
        else:
            box_ax.text(
                0.5,
                0.5,
                "No values",
                ha="center",
                va="center",
                fontsize=4.8,
                transform=box_ax.transAxes,
            )
        box_ax.axvline(0.0, color="0.45", linewidth=0.45, zorder=1)
        box_ax.set_xlim(x_min, x_max)
        box_ax.set_ylim(0.45, 2.55)
        box_ax.set_yticks([1, 2])
        if index == 0:
            from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

            box_ax.set_yticklabels(["n.s.", ""], fontsize=4.8)
            text_props = {"fontsize": 4.8}
            p_label_box = HPacker(
                children=[
                    TextArea("p", textprops={**text_props, "fontstyle": "italic"}),
                    TextArea(f"<{PANEL_C_SIGNIFICANCE_P_VALUE:g}", textprops=text_props),
                ],
                align="center",
                pad=0,
                sep=0,
            )
            box_ax.add_artist(
                AnnotationBbox(
                    p_label_box,
                    (0.0, 2.0),
                    xycoords=box_ax.get_yaxis_transform(),
                    xybox=(-1.5, 0.0),
                    boxcoords="offset points",
                    box_alignment=(1.0, 0.5),
                    frameon=False,
                    pad=0,
                )
            )
        else:
            box_ax.set_yticklabels([])
        if index == len(epoch_tables) - 1:
            box_ax.set_xlabel("Deviance explained", fontsize=5, labelpad=1)
        box_ax.spines["top"].set_visible(False)
        box_ax.spines["right"].set_visible(False)
        box_ax.tick_params(axis="x", labelsize=4.8, length=1.5, pad=1)
        box_ax.tick_params(axis="y", length=0, pad=1)


def compute_significance_distribution_comparison(
    table: Any,
    *,
    metric_column: str,
    p_column: str = "ripple_devexp_p_value",
    n_permutations: int = 10_000,
    random_seed: int = 53,
) -> dict[str, float | int]:
    """Compare significant and nonsignificant metric distributions by stratum permutation."""
    if table is None or len(table) == 0:
        return {
            "n_significant": 0,
            "n_nonsignificant": 0,
            "median_significant": float("nan"),
            "median_nonsignificant": float("nan"),
            "median_difference": float("nan"),
            "p_value": float("nan"),
        }

    metric_values = np.asarray(table[metric_column], dtype=float)
    p_values = np.asarray(table[p_column], dtype=float)
    valid = np.isfinite(metric_values) & np.isfinite(p_values)
    if not np.any(valid):
        return {
            "n_significant": 0,
            "n_nonsignificant": 0,
            "median_significant": float("nan"),
            "median_nonsignificant": float("nan"),
            "median_difference": float("nan"),
            "p_value": float("nan"),
        }

    metric_values = metric_values[valid]
    significant = p_values[valid] < SIGNIFICANCE_P_VALUE
    n_significant = int(np.sum(significant))
    n_nonsignificant = int(np.sum(~significant))
    if n_significant == 0 or n_nonsignificant == 0:
        return {
            "n_significant": n_significant,
            "n_nonsignificant": n_nonsignificant,
            "median_significant": float("nan"),
            "median_nonsignificant": float("nan"),
            "median_difference": float("nan"),
            "p_value": float("nan"),
        }

    median_significant = float(np.nanmedian(metric_values[significant]))
    median_nonsignificant = float(np.nanmedian(metric_values[~significant]))
    observed_difference = median_significant - median_nonsignificant
    stratum_columns = ["animal_name", "date", "epoch"]
    if all(column in table for column in stratum_columns):
        raw_strata = np.asarray(
            [
                "|".join(str(table[column].iloc[index]) for column in stratum_columns)
                for index, keep in enumerate(valid)
                if keep
            ],
            dtype=object,
        )
    else:
        raw_strata = np.full(metric_values.shape, "all", dtype=object)

    rng = np.random.default_rng(random_seed)
    unique_strata = np.unique(raw_strata)
    exceed_count = 0
    for _ in range(int(n_permutations)):
        permuted_significant = significant.copy()
        for stratum in unique_strata:
            stratum_indices = np.flatnonzero(raw_strata == stratum)
            n_stratum_significant = int(np.sum(significant[stratum_indices]))
            if n_stratum_significant in (0, stratum_indices.size):
                continue
            shuffled_indices = rng.permutation(stratum_indices)
            permuted_significant[stratum_indices] = False
            permuted_significant[shuffled_indices[:n_stratum_significant]] = True
        permuted_difference = float(
            np.nanmedian(metric_values[permuted_significant])
            - np.nanmedian(metric_values[~permuted_significant])
        )
        if abs(permuted_difference) >= abs(observed_difference):
            exceed_count += 1

    return {
        "n_significant": n_significant,
        "n_nonsignificant": n_nonsignificant,
        "median_significant": median_significant,
        "median_nonsignificant": median_nonsignificant,
        "median_difference": observed_difference,
        "p_value": (exceed_count + 1.0) / (float(n_permutations) + 1.0),
    }


def plot_metric_significance_distributions(
    ax: "Axes",
    table: Any,
    *,
    metric_column: str,
    x_label: str,
    title: str,
    x_limits: tuple[float, float],
    bin_edges: np.ndarray,
) -> None:
    """Plot dark tuning similarity for units significant in each GLM epoch group."""
    del bin_edges
    p_column = "ripple_devexp_p_value"
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No joined\nunits",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        ax.set_xlim(*x_limits)
    else:
        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.55, zorder=1)
        plot_data = []
        plot_positions = []
        plot_colors = []
        summary_lines = []
        for position, epoch_type in enumerate(HEATMAP_EPOCH_ORDER, start=1):
            epoch_rows = table[table["epoch_type"].astype(str) == epoch_type]
            metric_values = np.asarray(epoch_rows[metric_column], dtype=float)
            p_values = np.asarray(epoch_rows[p_column], dtype=float)
            valid = (
                np.isfinite(metric_values)
                & np.isfinite(p_values)
                & (p_values < SIGNIFICANCE_P_VALUE)
            )
            values = metric_values[valid]
            if not values.size:
                continue
            plot_data.append(values)
            plot_positions.append(position)
            plot_colors.append(GLM_EPOCH_COLORS.get(epoch_type, PANEL_D_POINT_COLOR))
            summary_lines.append(
                f"{HEATMAP_EPOCH_LABELS[epoch_type].split()[0]} n={values.size}, med={np.nanmedian(values):.2f}"
            )
        if plot_data:
            violin_artists = ax.violinplot(
                plot_data,
                positions=plot_positions,
                orientation="horizontal",
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, color in zip(violin_artists["bodies"], plot_colors, strict=False):
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.38)
                body.set_zorder(2)
            box_artists = ax.boxplot(
                plot_data,
                positions=plot_positions,
                orientation="horizontal",
                widths=0.25,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                medianprops={"color": "black", "linewidth": 0.7},
                whiskerprops={"color": "0.25", "linewidth": 0.55},
                capprops={"color": "0.25", "linewidth": 0.55},
            )
            for patch, color in zip(box_artists["boxes"], plot_colors, strict=False):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.72)
                patch.set_linewidth(0.55)
            rng = np.random.default_rng(7)
            for values, position, color in zip(plot_data, plot_positions, plot_colors, strict=True):
                jitter = rng.uniform(-0.10, 0.10, size=values.size)
                ax.scatter(
                    values,
                    np.full(values.shape, float(position)) + jitter,
                    s=3,
                    color=color,
                    alpha=0.22,
                    edgecolors="none",
                    zorder=3,
                )
            if summary_lines:
                ax.text(
                    0.97,
                    0.96,
                    "\n".join(summary_lines),
                    ha="right",
                    va="top",
                    fontsize=5.2,
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No significant\npairs",
                ha="center",
                va="center",
                fontsize=6,
                transform=ax.transAxes,
            )
        ax.set_xlim(*x_limits)

    ax.set_title(title, fontsize=6.2, pad=1.5)
    ax.set_xlabel(x_label, fontsize=5.4, labelpad=1)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Light", "Dark", "Sleep"], fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=5, length=1.5, pad=1)
    ax.tick_params(axis="y", length=0, pad=1)


def plot_ripple_decoding_comparison_panel(
    ax: "Axes",
    payload: Mapping[str, Any],
) -> None:
    """Plot CA1-V1 Bayesian categorical ripple decoding agreement."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    table = payload.get("summary_table")
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No decoding\ncomparison data",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        return

    categorical_metrics = tuple(
        payload.get("categorical_metrics", PANEL_E_CATEGORICAL_METRICS)
    )
    axis_height = 0.34
    axis_gap = 0.09
    top = 0.84
    rng = np.random.default_rng(19)
    x_by_epoch = {"light": 1.0, "dark": 2.0}
    for metric_index, (representation, label_scheme) in enumerate(categorical_metrics):
        bottom = top - axis_height - metric_index * (axis_height + axis_gap)
        metric_ax = ax.inset_axes([0.18, bottom, 0.76, axis_height])
        metric_rows = table[
            (table["representation"].astype(str) == str(representation))
            & (table["label_scheme"].astype(str) == str(label_scheme))
        ].copy()
        if not metric_rows.empty:
            metric_rows["match_rate_over_shuffle"] = (
                metric_rows["categorical_match_rate"].astype(float)
                - metric_rows["categorical_match_rate_shuffle_mean"].astype(float)
            )

        metric_ax.axhline(
            0.0,
            color="0.35",
            linestyle="--",
            linewidth=0.65,
            zorder=0,
        )
        metric_ax.text(
            2.43,
            0.002,
            "shuffle",
            ha="right",
            va="bottom",
            fontsize=4.6,
            color="0.35",
        )

        for (_animal_name, _date), session_rows in metric_rows.groupby(["animal_name", "date"]):
            delta_by_epoch: dict[str, float] = {}
            for epoch_type in PANEL_E_EPOCH_ORDER:
                epoch_rows = session_rows[session_rows["epoch_type"].astype(str) == epoch_type]
                if epoch_rows.empty:
                    continue
                value = float(epoch_rows["match_rate_over_shuffle"].iloc[0])
                if np.isfinite(value):
                    delta_by_epoch[epoch_type] = value
            if all(epoch_type in delta_by_epoch for epoch_type in PANEL_E_EPOCH_ORDER):
                metric_ax.plot(
                    [x_by_epoch[epoch_type] for epoch_type in PANEL_E_EPOCH_ORDER],
                    [delta_by_epoch[epoch_type] for epoch_type in PANEL_E_EPOCH_ORDER],
                    color="0.80",
                    linewidth=0.55,
                    zorder=1,
                )

        for epoch_type in PANEL_E_EPOCH_ORDER:
            epoch_rows = metric_rows[metric_rows["epoch_type"].astype(str) == epoch_type]
            delta = np.asarray(epoch_rows["match_rate_over_shuffle"], dtype=float)
            valid_delta = np.isfinite(delta)
            if not np.any(valid_delta):
                continue

            x_position = x_by_epoch[epoch_type]
            jitter = rng.uniform(-0.055, 0.055, size=int(np.sum(valid_delta)))
            color = GLM_EPOCH_COLORS.get(epoch_type, MODEL_COLOR)
            metric_ax.scatter(
                np.full(int(np.sum(valid_delta)), x_position) + jitter,
                delta[valid_delta],
                s=11,
                color=color,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.25,
                zorder=4,
            )
        delta_values = (
            np.asarray(metric_rows.get("match_rate_over_shuffle", []), dtype=float)
            if not metric_rows.empty
            else np.array([], dtype=float)
        )
        finite_delta = np.abs(delta_values[np.isfinite(delta_values)])
        y_extent = 0.02
        if finite_delta.size:
            y_extent = max(y_extent, float(np.nanmax(finite_delta)) * 1.35)
        y_extent = float(np.ceil(y_extent / 0.01) * 0.01)
        metric_ax.set_xlim(0.55, 2.45)
        metric_ax.set_ylim(-y_extent, y_extent)
        metric_ax.set_title(
            PANEL_E_METRIC_LABELS.get(
                (str(representation), str(label_scheme)),
                str(label_scheme).replace("_", " ").title(),
            ),
            fontsize=6.0,
            pad=1.2,
        )
        metric_ax.set_xticks([1.0, 2.0])
        if metric_index == len(categorical_metrics) - 1:
            metric_ax.set_xticklabels(["Light", "Dark"], fontsize=5)
            metric_ax.set_xlabel("Decode epoch", fontsize=5.2, labelpad=0.5)
        else:
            metric_ax.set_xticklabels([])
        metric_ax.set_ylabel("Above\nshuffle", fontsize=5.2, labelpad=1.2)
        metric_ax.spines["top"].set_visible(False)
        metric_ax.spines["right"].set_visible(False)
        metric_ax.tick_params(axis="x", length=0, pad=1)
        metric_ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1)


def plot_glm_offset_panel(
    ax: "Axes",
    payload: Mapping[str, Any],
) -> None:
    """Plot CA1-to-V1 ripple GLM target-window offset summaries."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    table = payload.get("summary_table")
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No complete\noffset GLM data",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        return

    target_offsets = tuple(float(offset) for offset in payload["target_window_offsets_s"])
    x_by_offset = {offset: float(index + 1) for index, offset in enumerate(target_offsets)}
    target_window_s = float(payload.get("target_window_s", PANEL_E_GLM_TARGET_WINDOW_S))
    x_tick_labels = [
        _format_panel_e_target_window_label(offset, target_window_s).replace(" to ", "\nto ")
        for offset in target_offsets
    ]
    rng = np.random.default_rng(17)
    fraction_ax = ax.inset_axes([0.19, 0.57, 0.77, 0.32])
    devexp_ax = ax.inset_axes([0.19, 0.15, 0.77, 0.32])
    axes = (fraction_ax, devexp_ax)
    metric_columns = (
        "fraction_significant_positive",
        "median_devexp_significant",
    )
    y_labels = (
        "Sig. V1\nfraction",
        "Median devexp\n(sig. V1)",
    )

    included_labels = sorted(
        {
            f"{animal} {epoch_type}"
            for animal, epoch_type in table[["animal_name", "epoch_type"]].itertuples(
                index=False,
                name=None,
            )
        }
    )
    included_epoch_types = sorted({str(value) for value in table["epoch_type"]})
    count_label = "animals" if len(included_epoch_types) == 1 else "animal-epochs"
    ax.text(
        0.96,
        0.96,
        f"n={len(included_labels)} {count_label}",
        ha="right",
        va="top",
        fontsize=4.8,
        color="0.35",
        transform=ax.transAxes,
    )

    for metric_ax, metric_column, y_label in zip(axes, metric_columns, y_labels, strict=True):
        metric_ax.axhline(
            0.0,
            color="0.55",
            linestyle="--",
            linewidth=0.55,
            zorder=0,
        )
        for (_animal_name, _date, epoch_type), group in table.groupby(
            ["animal_name", "date", "epoch_type"]
        ):
            values_by_offset: dict[float, float] = {}
            for target_offset in target_offsets:
                offset_rows = group[
                    np.isclose(
                        np.asarray(group["target_window_offset_s"], dtype=float),
                        target_offset,
                    )
                ]
                if offset_rows.empty:
                    continue
                value = float(offset_rows[metric_column].iloc[0])
                if np.isfinite(value):
                    values_by_offset[target_offset] = value
            if len(values_by_offset) == len(target_offsets):
                color = GLM_EPOCH_COLORS.get(str(epoch_type), MODEL_COLOR)
                metric_ax.plot(
                    [x_by_offset[offset] for offset in target_offsets],
                    [values_by_offset[offset] for offset in target_offsets],
                    color=color,
                    alpha=0.22,
                    linewidth=0.55,
                    zorder=1,
                )

        for epoch_type in HEATMAP_EPOCH_ORDER:
            epoch_rows = table[table["epoch_type"].astype(str) == epoch_type]
            if epoch_rows.empty:
                continue
            color = GLM_EPOCH_COLORS.get(epoch_type, MODEL_COLOR)
            mean_values = []
            for target_offset in target_offsets:
                offset_rows = epoch_rows[
                    np.isclose(
                        np.asarray(epoch_rows["target_window_offset_s"], dtype=float),
                        target_offset,
                    )
                ]
                values = np.asarray(offset_rows[metric_column], dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    mean_values.append(float(np.nanmean(values)))
                else:
                    mean_values.append(np.nan)
                if values.size:
                    jitter = rng.uniform(-0.045, 0.045, size=values.size)
                    metric_ax.scatter(
                        np.full(values.shape, x_by_offset[target_offset]) + jitter,
                        values,
                        s=11,
                        color=color,
                        alpha=0.78,
                        edgecolors="white",
                        linewidths=0.25,
                        zorder=4,
                    )
            finite_mean = np.isfinite(mean_values)
            if np.any(finite_mean):
                x_values = np.asarray(
                    [x_by_offset[offset] for offset in target_offsets],
                    dtype=float,
                )
                metric_ax.plot(
                    x_values[finite_mean],
                    np.asarray(mean_values, dtype=float)[finite_mean],
                    color=color,
                    linewidth=1.1,
                    marker="o",
                    markersize=2.4,
                    zorder=5,
                    label=HEATMAP_EPOCH_LABELS.get(epoch_type, epoch_type),
                )

        metric_ax.set_xlim(0.65, len(target_offsets) + 0.35)
        metric_ax.set_xticks([x_by_offset[offset] for offset in target_offsets])
        metric_ax.set_ylabel(y_label, fontsize=5.2, labelpad=1.0)
        metric_ax.spines["top"].set_visible(False)
        metric_ax.spines["right"].set_visible(False)
        metric_ax.tick_params(axis="x", length=0, pad=1)
        metric_ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1)

    fraction_values = np.asarray(table["fraction_significant_positive"], dtype=float)
    finite_fraction_values = fraction_values[np.isfinite(fraction_values)]
    fraction_top = 1.0
    if finite_fraction_values.size and np.nanmax(finite_fraction_values) < 0.82:
        fraction_top = 0.9
    fraction_ax.set_ylim(0.0, fraction_top)
    fraction_ax.set_xticklabels([])
    fraction_ax.set_title(
        "CA1 0-200 ms -> V1 target window",
        fontsize=6.0,
        pad=1.2,
    )
    devexp_values = np.asarray(table["median_devexp_significant"], dtype=float)
    finite_devexp_values = devexp_values[np.isfinite(devexp_values)]
    if finite_devexp_values.size:
        y_top = float(np.nanmax(finite_devexp_values)) * 1.18
        devexp_ax.set_ylim(0.0, max(0.02, y_top))
    devexp_ax.set_xticklabels(x_tick_labels, fontsize=4.8)
    devexp_ax.set_xlabel("V1 target window (ms)", fontsize=5.2, labelpad=0.5)
    fraction_ax.legend(
        frameon=False,
        fontsize=4.6,
        handlelength=1.0,
        loc="upper left",
        borderpad=0.1,
        labelspacing=0.2,
    )


def _plot_deviance_metric_quartiles(
    ax: "Axes",
    table: Any,
    *,
    y_column: str,
    y_label: str,
    title: str,
    y_limits: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
    y_scale: str = "linear",
    y_axis_side: str = "left",
    reference_y: float | None = None,
    epoch_type: str = "light",
    min_devexp: float = PANEL_D_MIN_DEVIANCE_EXPLAINED,
    show_x_ticklabels: bool = True,
) -> None:
    """Plot one metric across significant positive ripple-GLM deviance quartiles."""
    if reference_y is not None:
        ax.axhline(
            float(reference_y),
            color="0.45",
            linestyle="--",
            linewidth=0.55,
            zorder=0,
        )
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No joined\nunits",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
    else:
        epoch_rows = table[table["epoch_type"].astype(str) == str(epoch_type)]
        x_values = np.asarray(epoch_rows["ripple_devexp_mean"], dtype=float)
        p_values = np.asarray(epoch_rows["ripple_devexp_p_value"], dtype=float)
        y_values = np.asarray(epoch_rows[y_column], dtype=float)
        valid = (
            np.isfinite(x_values)
            & np.isfinite(p_values)
            & np.isfinite(y_values)
            & (x_values > float(min_devexp))
            & (p_values < PANEL_D_SIGNIFICANCE_P_VALUE)
        )
        if y_scale == "log":
            valid &= y_values > 0.0
        color = GLM_EPOCH_COLORS.get(epoch_type, PANEL_D_POINT_COLOR)
        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            sorted_indices = valid_indices[np.argsort(x_values[valid_indices])]
            quartile_indices = np.array_split(sorted_indices, 4)
            point_x_values = []
            point_y_values = []
            median_x_values = []
            median_y_values = []
            for quartile_index, group_indices in enumerate(quartile_indices, start=1):
                if group_indices.size == 0:
                    continue
                group_y_values = y_values[group_indices]
                if group_indices.size == 1:
                    offsets = np.array([0.0])
                else:
                    offsets = np.linspace(-0.16, 0.16, group_indices.size)
                point_x_values.append(quartile_index + offsets)
                point_y_values.append(group_y_values)
                quartile_low, quartile_median, quartile_high = np.nanpercentile(
                    group_y_values,
                    [25.0, 50.0, 75.0],
                )
                median_x_values.append(float(quartile_index))
                median_y_values.append(float(quartile_median))
                ax.plot(
                    [quartile_index, quartile_index],
                    [quartile_low, quartile_high],
                    color="black",
                    linewidth=0.65,
                    solid_capstyle="round",
                    zorder=4,
                )
            ax.scatter(
                np.concatenate(point_x_values),
                np.concatenate(point_y_values),
                s=4,
                color=color,
                alpha=0.22,
                edgecolors="none",
                zorder=2,
            )
            ax.plot(
                median_x_values,
                median_y_values,
                color="black",
                marker="o",
                markersize=2.4,
                markerfacecolor="white",
                markeredgewidth=0.6,
                linewidth=0.7,
                zorder=5,
            )

    ax.set_xlim(0.55, 4.45)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0])
    if show_x_ticklabels:
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=4.4)
    else:
        ax.set_xticklabels([])
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if y_scale == "log":
        ax.set_yscale("log")
        ax.set_yticks([1.0, 10.0, 100.0])
        ax.set_yticklabels(["1", "10", "100"], fontsize=4.4)
    if y_scale == "linear":
        if y_ticks is None:
            y_ticks = [0.0, 0.5, 1.0]
        ax.set_yticks(y_ticks)
    if y_axis_side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
    else:
        ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=5.4, pad=0.8)
    ax.set_ylabel(y_label, fontsize=5.0, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.4, length=1.5, pad=1)
    ax.tick_params(axis="y", labelsize=4.4, length=1.5, pad=1)


def _plot_deviance_metric_quartile_overlays(
    ax: "Axes",
    table: Any,
    *,
    metrics: Sequence[tuple[str, str, str]],
    y_label: str,
    title: str,
    y_limits: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
    reference_y: float | None = None,
    epoch_type: str = "light",
    min_devexp: float = PANEL_D_MIN_DEVIANCE_EXPLAINED,
) -> None:
    """Plot multiple metrics across shared ripple-GLM deviance quartiles."""
    if reference_y is not None:
        ax.axhline(
            float(reference_y),
            color="0.45",
            linestyle="--",
            linewidth=0.55,
            zorder=0,
        )
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No joined\nunits",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
    else:
        epoch_rows = table[table["epoch_type"].astype(str) == str(epoch_type)]
        x_values = np.asarray(epoch_rows["ripple_devexp_mean"], dtype=float)
        p_values = np.asarray(epoch_rows["ripple_devexp_p_value"], dtype=float)
        y_arrays = [
            np.asarray(epoch_rows[column], dtype=float)
            for column, _label, _color in metrics
        ]
        valid = (
            np.isfinite(x_values)
            & np.isfinite(p_values)
            & (x_values > float(min_devexp))
            & (p_values < PANEL_D_SIGNIFICANCE_P_VALUE)
        )
        for values in y_arrays:
            valid &= np.isfinite(values)

        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            sorted_indices = valid_indices[np.argsort(x_values[valid_indices])]
            quartile_indices = np.array_split(sorted_indices, 4)
            metric_offsets = (
                np.linspace(-0.07, 0.07, len(metrics))
                if len(metrics) > 1
                else np.array([0.0])
            )
            for metric_index, (column, label, color) in enumerate(metrics):
                y_values = np.asarray(epoch_rows[column], dtype=float)
                point_x_values = []
                point_y_values = []
                median_x_values = []
                median_y_values = []
                x_offset = float(metric_offsets[metric_index])
                for quartile_index, group_indices in enumerate(quartile_indices, start=1):
                    if group_indices.size == 0:
                        continue
                    group_y_values = y_values[group_indices]
                    if group_indices.size == 1:
                        offsets = np.array([0.0])
                    else:
                        offsets = np.linspace(-0.09, 0.09, group_indices.size)
                    point_x_values.append(quartile_index + x_offset + offsets)
                    point_y_values.append(group_y_values)
                    quartile_low, quartile_median, quartile_high = np.nanpercentile(
                        group_y_values,
                        [25.0, 50.0, 75.0],
                    )
                    median_x = float(quartile_index + x_offset)
                    median_x_values.append(median_x)
                    median_y_values.append(float(quartile_median))
                    ax.plot(
                        [median_x, median_x],
                        [quartile_low, quartile_high],
                        color=color,
                        linewidth=0.65,
                        solid_capstyle="round",
                        zorder=4,
                    )
                ax.scatter(
                    np.concatenate(point_x_values),
                    np.concatenate(point_y_values),
                    s=3.2,
                    color=color,
                    alpha=0.16,
                    edgecolors="none",
                    zorder=2,
                )
                ax.plot(
                    median_x_values,
                    median_y_values,
                    color=color,
                    marker="o",
                    markersize=2.3,
                    markerfacecolor="white",
                    markeredgewidth=0.6,
                    linewidth=0.7,
                    label=label,
                    zorder=5,
                )
            ax.legend(
                frameon=False,
                fontsize=3.8,
                handlelength=1.0,
                loc="upper left",
                borderpad=0.1,
                labelspacing=0.2,
            )

    ax.set_xlim(0.55, 4.45)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=4.4)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.set_title(title, fontsize=5.4, pad=0.8)
    ax.set_ylabel(y_label, fontsize=5.0, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.4, length=1.5, pad=1)
    ax.tick_params(axis="y", labelsize=4.4, length=1.5, pad=1)


def _plot_glm_significance_metric_distribution(
    ax: "Axes",
    table: Any,
    *,
    metric_column: str,
    y_label: str,
    title: str,
    epoch_type: str = "light",
    p_value_threshold: float = PANEL_D_SIGNIFICANCE_P_VALUE,
    min_devexp: float = PANEL_D_MIN_DEVIANCE_EXPLAINED,
    y_limits: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
    y_scale: str = "linear",
    show_x_ticklabels: bool = True,
    show_y_ticklabels: bool = True,
    show_counts: bool = False,
) -> None:
    """Plot one metric for nonsignificant and significant ripple-GLM cells."""
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No joined\nunits",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        values_by_group: list[np.ndarray] = []
    else:
        epoch_rows = table[table["epoch_type"].astype(str) == str(epoch_type)]
        metric_values = np.asarray(epoch_rows[metric_column], dtype=float)
        devexp_values = np.asarray(epoch_rows["ripple_devexp_mean"], dtype=float)
        p_values = np.asarray(epoch_rows["ripple_devexp_p_value"], dtype=float)
        finite = (
            np.isfinite(metric_values)
            & np.isfinite(devexp_values)
            & np.isfinite(p_values)
        )
        if y_scale == "log":
            finite &= metric_values > 0.0
        significant = (
            finite
            & (devexp_values > float(min_devexp))
            & (p_values < float(p_value_threshold))
        )
        values_by_group = [
            metric_values[finite & ~significant],
            metric_values[significant],
        ]
        if any(values.size for values in values_by_group):
            plot_data = [values for values in values_by_group if values.size]
            plot_positions = [
                position
                for position, values in enumerate(values_by_group, start=1)
                if values.size
            ]
            box = ax.boxplot(
                plot_data,
                positions=plot_positions,
                widths=0.38,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                medianprops={"color": "black", "linewidth": 0.7},
                whiskerprops={"color": "0.25", "linewidth": 0.55},
                capprops={"color": "0.25", "linewidth": 0.55},
            )
            colors = [NONSIGNIFICANT_COLOR, GLM_EPOCH_COLORS.get(epoch_type, MODEL_COLOR)]
            for patch, position in zip(box["boxes"], plot_positions, strict=False):
                patch.set_facecolor(colors[position - 1])
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.65)
                patch.set_linewidth(0.55)
            rng = np.random.default_rng(31)
            for position, values in enumerate(values_by_group, start=1):
                if not values.size:
                    continue
                jitter = rng.uniform(-0.14, 0.14, size=values.size)
                ax.scatter(
                    np.full(values.shape, float(position)) + jitter,
                    values,
                    s=3.2,
                    color=colors[position - 1],
                    alpha=0.28,
                    edgecolors="none",
                    zorder=3,
                )
            ns_count = int(values_by_group[0].size)
            sig_count = int(values_by_group[1].size)
            if show_counts:
                ax.text(
                    0.98,
                    0.96,
                    f"NS n={ns_count}\nSig n={sig_count}",
                    ha="right",
                    va="top",
                    fontsize=4.3,
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No finite\nvalues",
                ha="center",
                va="center",
                fontsize=6,
                transform=ax.transAxes,
            )

    finite_values = (
        np.concatenate([values for values in values_by_group if values.size])
        if values_by_group and any(values.size for values in values_by_group)
        else np.asarray([], dtype=float)
    )
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif finite_values.size:
        low, high = np.nanpercentile(finite_values, [2.0, 98.0])
        pad = max(0.01, 0.12 * float(high - low))
        ax.set_ylim(float(low - pad), float(high + pad))
    if y_scale == "log":
        ax.set_yscale("log")
        if y_ticks is None:
            y_ticks = [1.0, 10.0, 100.0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:g}" for tick in y_ticks], fontsize=4.4)
    elif y_ticks is not None:
        ax.set_yticks(y_ticks)
    if not show_y_ticklabels:
        ax.set_yticklabels([])
    ax.set_xlim(0.45, 2.55)
    ax.set_xticks([1.0, 2.0])
    if show_x_ticklabels:
        ax.set_xticklabels(["NS", "Sig"], fontsize=4.4)
    else:
        ax.set_xticklabels([])
    ax.set_title(title, fontsize=5.4, pad=0.8)
    ax.set_ylabel(y_label, fontsize=5.0, labelpad=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.4, length=1.5, pad=1)
    ax.tick_params(axis="y", labelsize=4.4, length=1.5, pad=1)


def _plot_dark_activity_devexp_boxplot(
    ax: "Axes",
    table: Any,
    *,
    epoch_type: str,
    dark_activity_threshold_hz: float,
    p_value_threshold: float,
    title: str,
    y_label: str = "",
    x_limits: tuple[float, float] | None = None,
    show_y_ticklabels: bool = True,
) -> None:
    """Plot significant ripple-GLM deviance explained by dark activity group."""
    ax.axvline(0.0, color="0.55", linewidth=0.55, linestyle="--", zorder=0)
    values_by_group: list[np.ndarray] = []
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No GLM\nunits",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
    else:
        epoch_rows = table[table["epoch_type"].astype(str) == str(epoch_type)]
        devexp_values = np.asarray(epoch_rows["ripple_devexp_mean"], dtype=float)
        p_values = np.asarray(epoch_rows["ripple_devexp_p_value"], dtype=float)
        dark_rates_hz = np.asarray(epoch_rows["dark_firing_rate_hz"], dtype=float)
        significant = (
            np.isfinite(devexp_values)
            & np.isfinite(p_values)
            & np.isfinite(dark_rates_hz)
            & (p_values < float(p_value_threshold))
        )
        inactive = significant & (dark_rates_hz < float(dark_activity_threshold_hz))
        active = significant & (dark_rates_hz >= float(dark_activity_threshold_hz))
        values_by_group = [
            devexp_values[inactive],
            devexp_values[active],
        ]
        if any(values.size for values in values_by_group):
            colors = [
                PANEL_D_DARK_ACTIVITY_COLORS["inactive"],
                PANEL_D_DARK_ACTIVITY_COLORS["active"],
            ]
            finite_values = np.concatenate(
                [values for values in values_by_group if values.size]
            )
            if x_limits is None:
                low, high = np.nanpercentile(finite_values, [1.0, 99.0])
                pad = max(0.02, 0.08 * float(high - low))
                x_limits = (float(low - pad), float(high + pad))
            plot_data = [values for values in values_by_group if values.size]
            plot_positions = [
                position
                for position, values in enumerate(values_by_group, start=1)
                if values.size
            ]
            box_artists = ax.boxplot(
                plot_data,
                orientation="horizontal",
                positions=plot_positions,
                widths=0.48,
                patch_artist=True,
                whis=(0, 100),
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.7, "zorder": 4},
                whiskerprops={"color": "0.25", "linewidth": 0.55, "zorder": 4},
                capprops={"color": "0.25", "linewidth": 0.55, "zorder": 4},
            )
            for patch, position in zip(box_artists["boxes"], plot_positions, strict=False):
                patch.set_facecolor(colors[position - 1])
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.50)
                patch.set_linewidth(0.55)
                patch.set_zorder(2)
            for position, values in zip(plot_positions, plot_data, strict=False):
                rng = np.random.default_rng(21_000 + position)
                y_values = position + rng.uniform(-0.16, 0.16, size=values.size)
                ax.scatter(
                    values,
                    y_values,
                    s=3.8,
                    color=colors[position - 1],
                    alpha=0.38,
                    edgecolors="none",
                    zorder=3,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No significant\nvalues",
                ha="center",
                va="center",
                fontsize=6,
                transform=ax.transAxes,
            )

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_ylim(0.4, 2.6)
    ax.set_yticks([1.0, 2.0])
    if not show_y_ticklabels:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(
            ["Dark-inactive", "Dark active"],
            fontsize=4.8,
        )
    ax.set_title(title, fontsize=5.8, pad=1.2)
    ax.set_xlabel("Dev. explained", fontsize=5.5, labelpad=1.0)
    ax.set_ylabel(y_label, fontsize=5.5, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.8, length=1.5, pad=1)
    ax.tick_params(axis="y", labelsize=5.0, length=1.5, pad=1)


def _plot_dark_active_same_turn_similarity_histogram(
    ax: "Axes",
    table: Any,
    *,
    epoch_type: str,
    dark_activity_threshold_hz: float,
    p_value_threshold: float,
    title: str,
    x_limits: tuple[float, float] = (-0.1, 1.0),
) -> None:
    """Plot dark same-turn tuning similarity for dark-active GLM-significant units."""
    ax.axvline(0.0, color="0.55", linewidth=0.55, linestyle="--", zorder=0)
    if table is None or len(table) == 0 or "same_turn_tuning_similarity" not in table:
        ax.text(
            0.5,
            0.5,
            "No tuning\nvalues",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
    else:
        epoch_rows = table[table["epoch_type"].astype(str) == str(epoch_type)]
        similarity_values = np.asarray(epoch_rows["same_turn_tuning_similarity"], dtype=float)
        p_values = np.asarray(epoch_rows["ripple_devexp_p_value"], dtype=float)
        dark_rates_hz = np.asarray(epoch_rows["dark_firing_rate_hz"], dtype=float)
        active_significant = (
            np.isfinite(similarity_values)
            & np.isfinite(p_values)
            & np.isfinite(dark_rates_hz)
            & (p_values < float(p_value_threshold))
            & (dark_rates_hz >= float(dark_activity_threshold_hz))
        )
        values = similarity_values[active_significant]
        if values.size:
            color = PANEL_D_DARK_ACTIVITY_COLORS["active"]
            bin_size = 0.1
            left_edge = np.floor(float(x_limits[0]) / bin_size) * bin_size
            right_edge = np.ceil(float(x_limits[1]) / bin_size) * bin_size
            bins = np.round(
                np.arange(left_edge, right_edge + 0.5 * bin_size, bin_size),
                10,
            )
            if not np.any(np.isclose(bins, 0.0)):
                bins = np.sort(np.unique(np.append(bins, 0.0)))
            weights = np.full(values.size, 1.0 / values.size, dtype=float)
            ax.hist(
                values,
                bins=bins,
                weights=weights,
                color=color,
                alpha=0.50,
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
            )
            median_value = float(np.nanmedian(values))
            ax.axvline(
                median_value,
                color=color,
                linewidth=0.9,
                zorder=4,
            )
            ax.text(
                0.96,
                0.94,
                f"median={median_value:.2f}",
                ha="right",
                va="top",
                fontsize=4.6,
                transform=ax.transAxes,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No significant\nvalues",
                ha="center",
                va="center",
                fontsize=6,
                transform=ax.transAxes,
            )

    ax.set_xlim(*x_limits)
    ax.set_title(title, fontsize=5.8, pad=1.2)
    ax.set_xlabel("Dark DPP corr.", fontsize=5.5, labelpad=1.0)
    ax.set_ylabel("Frac. units", fontsize=5.5, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.8, length=1.5, pad=1)
    ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1)


def _plot_dark_activity_significant_composition(
    ax: "Axes",
    table: Any,
    *,
    epoch_type: str,
    dark_activity_threshold_hz: float,
    p_value_threshold: float,
    title: str,
) -> None:
    """Plot dark activity composition among ripple-GLM significant cells."""
    import pandas as pd

    colors = [
        PANEL_D_DARK_ACTIVITY_COLORS["inactive"],
        PANEL_D_DARK_ACTIVITY_COLORS["active"],
    ]
    group_labels = ["Dark-inactive", "Dark active"]
    fractions = [np.nan, np.nan]
    counts = [0, 0]
    total_significant_count = 0
    if table is None or len(table) == 0:
        ax.text(0.5, 0.5, "No GLM\nunits", ha="center", va="center", fontsize=6, transform=ax.transAxes)
    else:
        epoch_rows = table[table["epoch_type"].astype(str) == str(epoch_type)].copy()
        p_values = np.asarray(epoch_rows["ripple_devexp_p_value"], dtype=float)
        dark_rates_hz = np.asarray(epoch_rows["dark_firing_rate_hz"], dtype=float)
        significant = (
            np.isfinite(p_values)
            & np.isfinite(dark_rates_hz)
            & (p_values < float(p_value_threshold))
        )
        group_masks = [
            significant & (dark_rates_hz < float(dark_activity_threshold_hz)),
            significant & (dark_rates_hz >= float(dark_activity_threshold_hz)),
        ]
        total_significant_count = int(np.sum(significant))
        for group_index, mask in enumerate(group_masks):
            counts[group_index] = int(np.sum(mask))
            if total_significant_count:
                fractions[group_index] = float(counts[group_index] / total_significant_count)

        positions = np.arange(1, 3, dtype=float)
        widths = np.nan_to_num(np.asarray(fractions, dtype=float), nan=0.0)
        ax.barh(
            positions,
            widths,
            height=0.58,
            color=colors,
            edgecolor=colors,
            linewidth=0.65,
            alpha=0.36,
            zorder=2,
        )
        ax.scatter(
            widths,
            positions,
            s=12,
            color=colors,
            edgecolors="0.2",
            linewidths=0.35,
            zorder=4,
        )
        if {"animal_name", "date"}.issubset(epoch_rows.columns):
            dataset_fractions_by_key: dict[tuple[str, str], list[float]] = {}
            for (_animal_name, _date), dataset_rows in epoch_rows.groupby(
                ["animal_name", "date"], sort=True
            ):
                dataset_p_values = pd.to_numeric(
                    dataset_rows["ripple_devexp_p_value"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                dataset_dark_rates_hz = pd.to_numeric(
                    dataset_rows["dark_firing_rate_hz"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                dataset_significant = (
                    np.isfinite(dataset_p_values)
                    & np.isfinite(dataset_dark_rates_hz)
                    & (dataset_p_values < float(p_value_threshold))
                )
                dataset_total = int(np.sum(dataset_significant))
                if not dataset_total:
                    continue
                key = (str(_animal_name), str(_date))
                dataset_inactive = int(
                    np.sum(
                        dataset_significant
                        & (dataset_dark_rates_hz < float(dark_activity_threshold_hz))
                    )
                )
                dataset_active = int(
                    np.sum(
                        dataset_significant
                        & (dataset_dark_rates_hz >= float(dark_activity_threshold_hz))
                    )
                )
                dataset_fractions_by_key[key] = [
                    float(dataset_inactive / dataset_total),
                    float(dataset_active / dataset_total),
                ]
            if dataset_fractions_by_key:
                dataset_keys = sorted(dataset_fractions_by_key)
                if len(dataset_keys) == 1:
                    offsets = np.asarray([0.0], dtype=float)
                else:
                    offsets = np.linspace(-0.08, 0.08, len(dataset_keys), dtype=float)
                offset_by_key = dict(zip(dataset_keys, offsets, strict=True))
                for key in dataset_keys:
                    dataset_fractions = np.asarray(dataset_fractions_by_key[key], dtype=float)
                    finite_pair = np.isfinite(dataset_fractions)
                    y_positions = positions + float(offset_by_key[key])
                    if np.all(finite_pair):
                        ax.plot(
                            dataset_fractions,
                            y_positions,
                            color="0.45",
                            linewidth=0.45,
                            alpha=0.55,
                            zorder=3,
                        )
                for group_index in range(2):
                    group_keys = [
                        key
                        for key in dataset_keys
                        if np.isfinite(dataset_fractions_by_key[key][group_index])
                    ]
                    if not group_keys:
                        continue
                    ax.scatter(
                        [
                            dataset_fractions_by_key[key][group_index]
                            for key in group_keys
                        ],
                        [
                            positions[group_index] + float(offset_by_key[key])
                            for key in group_keys
                        ],
                        s=5.5,
                        color=colors[group_index],
                        alpha=0.7,
                        edgecolors="none",
                        zorder=5,
                    )
        for position, fraction, count in zip(positions, fractions, counts, strict=True):
            label = f"n={count}"
            label_x = 0.04
            label_ha = "left"
            if np.isfinite(fraction):
                label = f"{fraction:.2f}\n{label}"
                if fraction > 0.82:
                    label_x = max(0.05, fraction - 0.055)
                    label_ha = "right"
                else:
                    label_x = min(0.98, fraction + 0.05)
            ax.text(
                label_x,
                position,
                label,
                ha=label_ha,
                va="center",
                fontsize=4.6,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.4, 2.6)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels(group_labels, fontsize=4.8)
    ax.set_title(title, fontsize=5.8, pad=1.2)
    ax.set_xlabel(f"p<{p_value_threshold:g} frac.", fontsize=5.5, labelpad=1.0)
    ax.set_ylabel("", fontsize=5.5, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.8, length=1.5, pad=1)
    ax.tick_params(axis="y", labelsize=5.0, length=1.5, pad=1)


def plot_glm_behavior_association_panel(
    ax: "Axes",
    payload: Mapping[str, Any],
    *,
    show_note: bool = True,
) -> None:
    """Plot ripple-GLM deviance explained by dark activity group."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    table = payload.get("devexp_table")
    dark_activity_threshold_hz = float(
        payload.get("dark_activity_threshold_hz", PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ)
    )
    epoch_rows = tuple(PANEL_D_EPOCH_ORDER)
    x_limits = PANEL_CD_DEVIANCE_EXPLAINED_LIMITS

    bottom = 0.17
    height = 0.72
    if len(epoch_rows) == 1:
        axis_layouts = [
            (
                epoch_rows[0],
                0.02,
                0.20,
                0.27,
                0.27,
                0.68,
                0.28,
            )
        ]
    else:
        panel_left = 0.08
        panel_right = 0.96
        pair_gap = 0.07
        inset_gap = 0.025
        fraction_width = 0.13
        devexp_width = 0.14
        pair_width = (
            (panel_right - panel_left - pair_gap * (len(epoch_rows) - 1)) / len(epoch_rows)
            if epoch_rows
            else 0.0
        )
        similarity_width = pair_width - fraction_width - devexp_width - 2.0 * inset_gap
        axis_layouts = [
            (
                epoch_type,
                panel_left + column_index * (pair_width + pair_gap),
                fraction_width,
                panel_left + column_index * (pair_width + pair_gap) + fraction_width + inset_gap,
                devexp_width,
                panel_left
                + column_index * (pair_width + pair_gap)
                + fraction_width
                + devexp_width
                + 2.0 * inset_gap,
                similarity_width,
            )
            for column_index, epoch_type in enumerate(epoch_rows)
        ]

    for (
        epoch_type,
        fraction_left,
        fraction_width,
        devexp_left,
        devexp_width,
        similarity_left,
        similarity_width,
    ) in axis_layouts:
        fraction_ax = ax.inset_axes([fraction_left, bottom, fraction_width, height])
        _plot_dark_activity_significant_composition(
            fraction_ax,
            table,
            epoch_type=epoch_type,
            dark_activity_threshold_hz=dark_activity_threshold_hz,
            p_value_threshold=PANEL_C_SIGNIFICANCE_P_VALUE,
            title="",
        )
        devexp_ax = ax.inset_axes(
            [devexp_left, bottom, devexp_width, height]
        )
        _plot_dark_activity_devexp_boxplot(
            devexp_ax,
            table,
            epoch_type=epoch_type,
            dark_activity_threshold_hz=dark_activity_threshold_hz,
            p_value_threshold=PANEL_D_SIGNIFICANCE_P_VALUE,
            title="",
            y_label="",
            x_limits=x_limits,
            show_y_ticklabels=False,
        )
        similarity_ax = ax.inset_axes(
            [similarity_left, bottom, similarity_width, height]
        )
        _plot_dark_active_same_turn_similarity_histogram(
            similarity_ax,
            table,
            epoch_type=epoch_type,
            dark_activity_threshold_hz=dark_activity_threshold_hz,
            p_value_threshold=PANEL_D_SIGNIFICANCE_P_VALUE,
            title="",
        )
    if show_note:
        ax.text(
            0.50,
            0.035,
            "Plots show p<0.05 units; dark-active split uses 0.5 Hz",
            ha="center",
            va="bottom",
            fontsize=5.0,
            transform=ax.transAxes,
        )


def _compute_source_comparison_axis_limits(table: Any) -> tuple[float, float]:
    """Return shared deviance-explained limits for source-mode comparisons."""
    return PANEL_C_SOURCE_COMPARISON_LIMITS


def _plot_source_predictor_comparison_axis(
    ax: "Axes",
    table: Any,
    *,
    title: str,
    axis_limits: tuple[float, float],
    pooled: bool = False,
    p_value_threshold: float = SIGNIFICANCE_P_VALUE,
    summary_location: str = "upper_left",
) -> None:
    """Plot vector-model deviance explained against mean-activity control."""
    lower, upper = axis_limits
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="0.55",
        linestyle="--",
        linewidth=0.55,
        zorder=1,
    )
    if table is None or len(table) == 0:
        ax.text(
            0.5,
            0.5,
            "No paired\nGLM data",
            ha="center",
            va="center",
            fontsize=5.5,
            transform=ax.transAxes,
        )
    else:
        x_values = np.asarray(table["mean_activity_devexp_mean"], dtype=float)
        y_values = np.asarray(table["vector_devexp_mean"], dtype=float)
        p_values = (
            np.asarray(table["vector_devexp_p_value"], dtype=float)
            if "vector_devexp_p_value" in table
            else np.full(len(table), np.nan, dtype=float)
        )
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        significant = (
            valid
            & np.isfinite(p_values)
            & (p_values < float(p_value_threshold))
        )
        if np.any(significant):
            if pooled and {"animal_name", "date"}.issubset(table.columns):
                for (_animal_name, _date), dataset_rows in table.loc[significant].groupby(
                    ["animal_name", "date"],
                    sort=True,
                ):
                    ax.scatter(
                        dataset_rows["mean_activity_devexp_mean"],
                        dataset_rows["vector_devexp_mean"],
                        s=4.2,
                        color=PANEL_C_SOURCE_COMPARISON_COLOR,
                        alpha=0.42,
                        edgecolors="none",
                        rasterized=True,
                        zorder=3,
                    )
            else:
                ax.scatter(
                    x_values[significant],
                    y_values[significant],
                    s=4.2,
                    color=PANEL_C_SOURCE_COMPARISON_COLOR,
                    alpha=0.42,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                )
            deltas = y_values[significant] - x_values[significant]
            vector_greater_fraction = float(np.mean(deltas > 0.0))
            if summary_location == "lower_right":
                text_x = 0.97
                text_y = 0.05
                text_ha = "right"
                text_va = "bottom"
            elif summary_location == "upper_right":
                text_x = 0.97
                text_y = 0.95
                text_ha = "right"
                text_va = "top"
            else:
                text_x = 0.05
                text_y = 0.95
                text_ha = "left"
                text_va = "top"
            ax.text(
                text_x,
                text_y,
                f"n={int(np.sum(significant))}\n"
                f"frac vector>mean={vector_greater_fraction:.2f}",
                ha=text_ha,
                va=text_va,
                fontsize=4.6,
                transform=ax.transAxes,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No p<0.05\nunits",
                ha="center",
                va="center",
                fontsize=5.5,
                transform=ax.transAxes,
            )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=5.8, pad=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=4.7, length=1.5, pad=1)


def plot_glm_source_predictor_comparison_panel(
    ax: "Axes",
    payload: Mapping[str, Any],
    *,
    include_per_animal: bool = True,
    include_pooled: bool = True,
    compact_labels: bool = False,
    show_color_note: bool = True,
) -> None:
    """Plot full CA1 vector GLM performance against mean CA1 activity."""
    import pandas as pd

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    table = payload.get("comparison_table")
    if table is None:
        table = pd.DataFrame()
    axis_limits = _compute_source_comparison_axis_limits(table)
    if len(table):
        dataset_keys = sorted(
            {
                (str(row.animal_name), str(row.date))
                for row in table[["animal_name", "date"]].drop_duplicates().itertuples()
            }
        )
    else:
        dataset_keys = []
    groups: list[tuple[str, Any, bool]] = []
    if include_per_animal:
        for animal_name, date in dataset_keys:
            dataset_rows = table[
                (table["animal_name"].astype(str) == animal_name)
                & (table["date"].astype(str) == date)
            ]
            groups.append((animal_name, dataset_rows, False))
    if include_pooled and len(table):
        groups.append(("Pooled", table, True))

    if not groups:
        ax.text(
            0.5,
            0.5,
            "No paired vector/mean GLM data",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        return

    left = 0.10 if compact_labels else 0.075
    right = 0.985
    bottom = 0.10 if compact_labels else 0.20
    height = 0.82 if compact_labels else 0.70
    gap = 0.030
    width = (right - left - gap * (len(groups) - 1)) / len(groups)
    for index, (title, rows, pooled) in enumerate(groups):
        child_ax = ax.inset_axes([left + index * (width + gap), bottom, width, height])
        child_title = "" if compact_labels else title
        _plot_source_predictor_comparison_axis(
            child_ax,
            rows,
            title=child_title,
            axis_limits=axis_limits,
            pooled=pooled,
            p_value_threshold=SIGNIFICANCE_P_VALUE,
            summary_location="upper_right" if compact_labels else "upper_left",
        )
        if compact_labels:
            child_ax.set_xlabel("Mean CA1 devexp", fontsize=5.3, labelpad=1.0)
            if index == 0:
                child_ax.set_ylabel("CA1 vector devexp", fontsize=5.3, labelpad=1.0)
        if index > 0:
            child_ax.set_yticklabels([])

    if not compact_labels:
        ax.text(
            0.52,
            0.035,
            "Mean CA1 activity deviance explained",
            ha="center",
            va="bottom",
            fontsize=6.0,
            transform=ax.transAxes,
        )
    if show_color_note:
        ax.text(
            0.98,
            0.035,
            f"Showing vector p<{SIGNIFICANCE_P_VALUE:g} units",
            ha="right",
            va="bottom",
            fontsize=5.2,
            transform=ax.transAxes,
        )
    if not compact_labels:
        ax.text(
            0.018,
            0.56,
            "CA1 spike vector deviance explained",
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.0,
            transform=ax.transAxes,
        )


def plot_glm_prediction_context_panel(
    ax: "Axes",
    behavior_payload: Mapping[str, Any],
    source_comparison_payload: Mapping[str, Any],
) -> None:
    """Stack the pooled source-control plot above the dark-activity summary."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    source_ax = ax.inset_axes([0.0, 0.55, 1.0, 0.42])
    plot_glm_source_predictor_comparison_panel(
        source_ax,
        source_comparison_payload,
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
    )
    source_ax.set_title("Pooled vector vs mean CA1 activity", fontsize=5.8, pad=1.2)

    behavior_ax = ax.inset_axes([0.0, 0.01, 1.0, 0.49])
    plot_glm_behavior_association_panel(
        behavior_ax,
        behavior_payload,
        show_note=False,
    )


def filter_epoch_payloads(
    payloads: Sequence[Mapping[str, Any]],
    epoch_order: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Return payloads ordered by a selected set of epoch types."""
    payload_by_epoch_type = {
        str(payload["epoch_type"]): payload
        for payload in payloads
        if "epoch_type" in payload
    }
    return [
        payload_by_epoch_type[str(epoch_type)]
        for epoch_type in epoch_order
        if str(epoch_type) in payload_by_epoch_type
    ]


def plot_observed_predicted_panel(ax: "Axes", example: dict[str, Any]) -> None:
    """Plot held-out observed versus predicted ripple counts for one V1 unit."""
    observed = np.asarray(example["observed"], dtype=float)
    predicted = np.asarray(example["predicted"], dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if np.any(valid):
        ax.scatter(
            observed[valid],
            predicted[valid],
            s=11,
            color=MODEL_COLOR,
            alpha=0.58,
            edgecolors="none",
        )
        max_value = float(max(np.nanmax(observed[valid]), np.nanmax(predicted[valid]), 1.0))
        ax.plot([0.0, max_value], [0.0, max_value], color="black", linestyle="--", linewidth=0.7)
        ax.set_xlim(0.0, max_value)
        ax.set_ylim(0.0, max_value)
    else:
        ax.text(0.5, 0.5, "No finite samples", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Observed count")
    ax.set_ylabel("Predicted count")
    ax.set_title(
        (
            f"{example['animal_name']} {example['date']} {example['epoch']} "
            f"unit {example['unit_id']}"
        ),
        fontsize=7,
        pad=2,
    )
    ax.text(
        0.04,
        0.96,
        f"devexp={float(example['ripple_devexp_mean']):.2f}\np={float(example['ripple_devexp_p_value']):.3f}",
        ha="left",
        va="top",
        fontsize=5.5,
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6, length=2, pad=1)


def make_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    example_dataset: DatasetId,
    xcorr_dataset: DatasetId,
    xcorr_state: str,
    xcorr_top_ca1_units: int,
    xcorr_bin_size_s: float,
    xcorr_max_lag_s: float,
    xcorr_display_vmax: float,
    light_epoch: str | None,
    dark_epoch: str | None,
    sleep_epoch: str | None,
    regions: Sequence[str],
    ripple_threshold_zscore: float,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ripple_selection: str,
    ridge_strength: float,
    dark_movement_fr_cache_dir: Path | None,
    refresh_dark_movement_fr_cache: bool,
    refresh_panel_b_schematic_cache: bool,
    dpi: int,
) -> Path:
    """Build and save Figure 2."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    heatmap_epoch_tables = load_pooled_ripple_heatmap_epoch_tables(
        data_root,
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    glm_epoch_tables = load_glm_epoch_summary_tables(
        data_root,
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        epoch_types=PANEL_C_EPOCH_ORDER,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )
    schematic_animal, schematic_date, schematic_epoch = normalize_dataset_id(
        DEFAULT_PANEL_B_SCHEMATIC_DATASET
    )
    ripple_schematic_trace: dict[str, Any] | None = None
    try:
        ripple_schematic_trace = load_or_build_panel_b_schematic_example(
            data_root,
            cache_dir=DEFAULT_FIGURE_CACHE_DIR,
            animal_name=schematic_animal,
            date=schematic_date,
            epoch=schematic_epoch,
            ripple_threshold_zscore=ripple_threshold_zscore,
            time_before_s=DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
            time_after_s=DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
            n_units_per_region=DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
            target_ripple_duration_s=DEFAULT_PANEL_B_SCHEMATIC_TARGET_DURATION_S,
            refresh_cache=refresh_panel_b_schematic_cache,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(
            "Panel B using fallback schematic spikes because the real-spike cache "
            f"could not be built for {schematic_animal} {schematic_date} "
            f"{schematic_epoch}: {exc}"
        )
        example_animal, example_date, example_epoch = normalize_dataset_id(example_dataset)
        try:
            ripple_schematic_trace = load_example_ripple_lfp_trace(
                data_root,
                animal_name=example_animal,
                date=example_date,
                epoch=example_epoch,
                ripple_threshold_zscore=ripple_threshold_zscore,
                time_before_s=DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
                time_after_s=DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
            )
        except (FileNotFoundError, KeyError, ValueError) as fallback_exc:
            print(
                "Panel B using fully synthetic schematic because saved ripple-band LFP "
                f"was unavailable for {example_animal} {example_date} "
                f"{example_epoch}: {fallback_exc}"
            )
    panel_d_payload = load_glm_dark_activity_devexp_tables(
        data_root,
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        dark_movement_fr_cache_dir=dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=refresh_dark_movement_fr_cache,
    )
    panel_e_payload = load_glm_source_predictor_comparison_tables(
        data_root,
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
    )
    panel_a_epoch_tables = filter_epoch_payloads(heatmap_epoch_tables, PANEL_A_EPOCH_ORDER)
    panel_c_epoch_tables = filter_epoch_payloads(glm_epoch_tables, PANEL_C_EPOCH_ORDER)
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=8,
        height_ratios=[0.46, 0.54],
    )
    axes = [
        fig.add_subplot(outer_grid[:, :2]),
        fig.add_subplot(outer_grid[:, 2:5]),
        fig.add_subplot(outer_grid[0, 5:]),
        fig.add_subplot(outer_grid[1, 5:]),
    ]

    plot_epoch_ripple_heatmap_panel(axes[0], panel_a_epoch_tables, regions=regions)
    axes[0].set_title("Ripple-triggered\nmean firing rates", fontsize=7.2, pad=2)
    plot_glm_analysis_panel(axes[1], panel_c_epoch_tables, ripple_trace=ripple_schematic_trace)
    axes[1].set_title(
        "Predicting V1 activity during ripples\nwith CA1 activity",
        fontsize=7.2,
        pad=2,
    )
    plot_glm_source_predictor_comparison_panel(
        axes[2],
        panel_e_payload,
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
    )
    axes[2].set_title(
        "CA1 spike vector vs. mean CA1 activity",
        fontsize=7.2,
        pad=2,
    )
    plot_glm_behavior_association_panel(
        axes[3],
        panel_d_payload,
        show_note=False,
    )
    axes[3].set_title(
        "Relationship to dark-active DPP cells",
        fontsize=7.2,
        pad=2,
    )

    for ax, label in zip(axes, ("A", "B", "C", "D"), strict=True):
        label_axis(ax, label, x=-0.10, y=1.04)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    for missing in panel_d_payload["missing_artifacts"]:
        print(
            "Panel C dark-activity missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']}: {missing['path']}"
        )
    for missing in panel_e_payload["missing_artifacts"]:
        print(
            "Panel C source-comparison missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']} "
            f"({missing['source_predictor_mode']}): {missing['path']}"
        )
    print(f"Saved Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 2 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 2 CA1 ripple modulation and CA1-to-V1 GLM panels."
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
            "Animal/date/epoch data set to include as animal:date:epoch. "
            "May be repeated. Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--example-dataset",
        type=parse_dataset_id,
        default=DEFAULT_EXAMPLE_DATASET,
        help=(
            "Data set used for the example ripple-band LFP trace in panel C. "
            "Format: animal:date:epoch."
        ),
    )
    parser.add_argument(
        "--xcorr-dataset",
        type=parse_dataset_id,
        default=DEFAULT_XCORR_DATASET,
        help=(
            "Deprecated screen-xcorr data set argument. "
            "Format: animal:date:epoch. Default: L15:20241121:02_r1."
        ),
    )
    parser.add_argument(
        "--xcorr-state",
        choices=XCORR_STATE_CHOICES,
        default=DEFAULT_XCORR_STATE,
        help=f"Deprecated screen-xcorr state argument. Default: {DEFAULT_XCORR_STATE}.",
    )
    parser.add_argument(
        "--xcorr-top-ca1-units",
        type=int,
        default=DEFAULT_XCORR_TOP_CA1_UNITS,
        help=(
            "Deprecated number of top-ranked CA1 units for screen-xcorr. "
            f"Default: {DEFAULT_XCORR_TOP_CA1_UNITS}."
        ),
    )
    parser.add_argument(
        "--xcorr-bin-size-s",
        type=float,
        default=DEFAULT_XCORR_BIN_SIZE_S,
        help=f"Screen-xcorr bin size in seconds. Default: {DEFAULT_XCORR_BIN_SIZE_S:g}.",
    )
    parser.add_argument(
        "--xcorr-max-lag-s",
        type=float,
        default=DEFAULT_XCORR_MAX_LAG_S,
        help=f"Screen-xcorr maximum lag in seconds. Default: {DEFAULT_XCORR_MAX_LAG_S:g}.",
    )
    parser.add_argument(
        "--xcorr-display-vmax",
        type=float,
        default=DEFAULT_XCORR_DISPLAY_VMAX,
        help=f"Deprecated normalized-xcorr color maximum. Default: {DEFAULT_XCORR_DISPLAY_VMAX:g}.",
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help=(
            "Region to include in peri-ripple and modulation panels. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument(
        "--light-epoch",
        default=None,
        help=(
            "Light run epoch for panel A. "
            "Default: use v1ca1.paper_figures.datasets registry."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help=(
            "Dark run epoch for panel A. "
            "Default: use each data set's registered dark epoch."
        ),
    )
    parser.add_argument(
        "--sleep-epoch",
        default=None,
        help=(
            "Sleep epoch for panel A. "
            "Default: use v1ca1.paper_figures.datasets registry."
        ),
    )
    parser.add_argument(
        "--dark-movement-fr-cache-dir",
        type=Path,
        default=DEFAULT_FIGURE_CACHE_DIR,
        help=(
            "Directory for cached dark movement firing-rate tables used by Panel C. "
            f"Default: {DEFAULT_FIGURE_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--refresh-dark-movement-fr-cache",
        action="store_true",
        help="Recompute and overwrite cached dark movement firing-rate tables.",
    )
    parser.add_argument(
        "--refresh-panel-b-schematic-cache",
        action="store_true",
        help="Recompute and overwrite the cached real-spike panel B schematic example.",
    )
    parser.add_argument(
        "--ripple-threshold-zscore",
        type=float,
        default=DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
        help=(
            "Ripple mean-zscore threshold matching cached ripple-modulation outputs. "
            f"Default: {DEFAULT_RIPPLE_THRESHOLD_ZSCORE:g}"
        ),
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_S,
        help=f"Ripple-GLM window length in seconds. Default: {DEFAULT_RIPPLE_WINDOW_S}",
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        help=(
            "Ripple-GLM window offset in seconds. "
            f"Default: {DEFAULT_RIPPLE_WINDOW_OFFSET_S}"
        ),
    )
    parser.add_argument(
        "--ripple-selection",
        choices=("allripples", "deduped", "single"),
        default=DEFAULT_FIGURE_2_GLM_RIPPLE_SELECTION,
        help=(
            "Ripple-GLM selection suffix for Figure 2 Panels C and D. "
            f"Default: {DEFAULT_FIGURE_2_GLM_RIPPLE_SELECTION}"
        ),
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Ripple-GLM ridge strength. Default: {DEFAULT_RIDGE_STRENGTH:g}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        example_dataset=args.example_dataset,
        xcorr_dataset=args.xcorr_dataset,
        xcorr_state=args.xcorr_state,
        xcorr_top_ca1_units=args.xcorr_top_ca1_units,
        xcorr_bin_size_s=args.xcorr_bin_size_s,
        xcorr_max_lag_s=args.xcorr_max_lag_s,
        xcorr_display_vmax=args.xcorr_display_vmax,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        sleep_epoch=args.sleep_epoch,
        regions=regions,
        ripple_threshold_zscore=args.ripple_threshold_zscore,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection=args.ripple_selection,
        ridge_strength=args.ridge_strength,
        dark_movement_fr_cache_dir=args.dark_movement_fr_cache_dir,
        refresh_dark_movement_fr_cache=args.refresh_dark_movement_fr_cache,
        refresh_panel_b_schematic_cache=args.refresh_panel_b_schematic_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
