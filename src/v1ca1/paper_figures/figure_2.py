from __future__ import annotations

"""Generate Figure 2 panels for CA1 ripple modulation of V1 activity."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS, get_analysis_path
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    make_dataset_id,
    make_figure_2_epoch_ids,
    normalize_dataset_id,
)
from v1ca1.paper_figures.style import (
    REGION_COLORS,
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
DEFAULT_OUTPUT_NAME = "figure_2"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_EXAMPLE_DATASET = ("L14", "20240611", "08_r4")
DEFAULT_XCORR_DATASET = ("L15", "20241121", "02_r1")
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_FIGURE_HEIGHT_MM = 126.0
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
DEFAULT_REGIONS = REGIONS
RIPPLE_EVENT_RELATIVE_PATH = Path("ripple") / "ripple_times.parquet"
RIPPLE_LFP_RELATIVE_DIR = Path("ripple") / "ripple_channels_lfp"
RIPPLE_MODULATION_RELATIVE_DIR = Path("ripple") / "ripple_modulation"
RIPPLE_GLM_RELATIVE_DIR = Path("ripple_glm")
ENCODING_COMPARISON_RELATIVE_DIR = Path("task_progression") / "encoding_comparison"
TUNING_ANALYSIS_RELATIVE_DIR = Path("task_progression") / "tuning_analysis"
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
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_LFP_TIME_BEFORE_S = 0.080
DEFAULT_LFP_TIME_AFTER_S = 0.160
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
DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC = "correlation"
DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL = "pooled_same_turn"
HEATMAP_EPOCH_ORDER = ("light", "dark", "sleep")
HEATMAP_EPOCH_LABELS = {
    "light": "Light run",
    "dark": "Dark run",
    "sleep": "Sleep",
}
XCORR_RELATIVE_DIR = Path("xcorr") / "screen_pairs"
XCORR_SUMMARY_FILENAME = "xcorr_summary.parquet"
XCORR_DATASET_FILENAME = "xcorr.nc"
MODEL_COLOR = "#55A868"
GLM_EPOCH_COLORS = {
    "light": "#4C72B0",
    "dark": "#55A868",
    "sleep": "#C44E52",
}
NONSIGNIFICANT_COLOR = "0.70"
SIGNIFICANCE_P_VALUE = 0.05
PANEL_D_POINT_COLOR = REGION_COLORS["v1"]


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
) -> Path:
    """Return the NetCDF ripple-GLM result path for one epoch."""
    window_suffix = format_ripple_window_suffix(
        ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    ridge_suffix = format_ridge_strength_suffix(ridge_strength)
    filename = f"{epoch}_{window_suffix}_{ripple_selection}_{ridge_suffix}_samplewise_ripple_glm.nc"
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
    import xarray as xr

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
        "n_ripples": int(len(epoch_table)),
    }


def load_ripple_glm_summary_table(
    data_root: Path,
    datasets: Sequence[DatasetId],
    *,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
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
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> list[dict[str, Any]]:
    """Load pooled ripple-GLM summaries for light, dark, and sleep epochs."""
    epoch_datasets: dict[str, list[DatasetId]] = {epoch_type: [] for epoch_type in HEATMAP_EPOCH_ORDER}
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
            epoch_datasets[epoch_type].append(normalize_dataset_id(epoch_ids[epoch_type]))

    epoch_tables = []
    for epoch_type in HEATMAP_EPOCH_ORDER:
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


def draw_ripple_glm_schematic(ax: "Axes") -> None:
    """Draw a compact CA1-to-V1 ripple GLM schematic."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    transform = ax.transAxes
    ca1_rect = Rectangle(
        (0.07, 0.32),
        0.25,
        0.36,
        facecolor="#f6efe6",
        edgecolor="black",
        linewidth=0.7,
        transform=transform,
    )
    v1_rect = Rectangle(
        (0.68, 0.32),
        0.25,
        0.36,
        facecolor="#e7eef8",
        edgecolor="black",
        linewidth=0.7,
        transform=transform,
    )
    window_rect = Rectangle(
        (0.40, 0.43),
        0.20,
        0.14,
        facecolor="#e8f3e8",
        edgecolor="0.25",
        linewidth=0.6,
        transform=transform,
    )
    ax.add_patch(ca1_rect)
    ax.add_patch(v1_rect)
    ax.add_patch(window_rect)
    ax.text(0.195, 0.50, "CA1\ncounts", ha="center", va="center", transform=transform)
    ax.text(0.805, 0.50, "V1\ncounts", ha="center", va="center", transform=transform)
    ax.text(0.50, 0.50, "0.2 s\nripple", ha="center", va="center", fontsize=6, transform=transform)
    ax.add_patch(
        FancyArrowPatch(
            (0.32, 0.50),
            (0.40, 0.50),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=0.8,
            color="black",
            transform=transform,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.60, 0.50),
            (0.68, 0.50),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=0.8,
            color="black",
            transform=transform,
        )
    )
    ax.text(
        0.50,
        0.18,
        "Ridge Poisson GLM, held-out ripples",
        ha="center",
        va="center",
        fontsize=6,
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
        color="#d9a441",
        alpha=0.28,
        linewidth=0,
    )
    ax.axvline(0.0, color="#9a6a00", linewidth=0.7)
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
    heatmap_top = 0.84
    hist_bottom = 0.08
    hist_height = 0.18
    column_gap = 0.035
    row_gap = 0.045
    cell_width = (right - left - column_gap * (n_epochs - 1)) / n_epochs
    cell_height = (heatmap_top - heatmap_bottom - row_gap * (n_regions - 1)) / n_regions
    image = None

    for col_index, epoch_payload in enumerate(epoch_tables):
        x0 = left + col_index * (cell_width + column_gap)
        ax.text(
            x0 + cell_width / 2,
            0.96,
            f"{epoch_payload['label']}\n{epoch_payload['epoch']}",
            ha="center",
            va="top",
            fontsize=6,
            transform=ax.transAxes,
        )
        firing_rate_table = epoch_payload["firing_rate_table"]
        for row_index, region in enumerate(regions):
            y0 = heatmap_top - (row_index + 1) * cell_height - row_index * row_gap
            heatmap_ax = ax.inset_axes([x0, y0, cell_width, cell_height])
            payload = build_peri_ripple_heatmap_payload(firing_rate_table, region=region)
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
        colorbar_ax = ax.inset_axes([0.955, heatmap_bottom, 0.018, heatmap_top - heatmap_bottom])
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
            alpha=0.48,
            edgecolor="none",
            label=region.upper(),
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
    bottom = 0.16
    top = 0.78
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
        heatmap_ax.set_xlabel("Lag (s)", fontsize=5, labelpad=1)
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
        0.50,
        0.97,
        (
            f"{payload['animal_name']} {payload['date']} {payload['epoch']}; "
            f"V1 order from CA1 {payload['v1_order_reference_ca1_unit']}"
        ),
        ha="center",
        va="top",
        fontsize=5.2,
        transform=ax.transAxes,
    )
    if image is not None:
        colorbar_ax = ax.inset_axes([0.955, bottom, 0.026, top - bottom])
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
                alpha=0.52,
                edgecolor="none",
                label=region.upper(),
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
        color="#9c755f",
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
) -> None:
    """Plot the ripple-GLM schematic and epoch-specific performance summaries."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not epoch_tables:
        ax.text(0.5, 0.5, "No GLM data", ha="center", va="center", transform=ax.transAxes)
        return

    schematic_ax = ax.inset_axes([0.02, 0.16, 0.25, 0.70])
    draw_ripple_glm_schematic(schematic_ax)

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
    x_min = -0.1
    x_max = 0.5
    y_max = max(2.0, float(np.nanmax(finite_neglog_p)) + 0.4) if finite_neglog_p.size else 2.0

    plot_left = 0.34
    plot_right = 0.98
    scatter_bottom = 0.39
    scatter_top = 0.82
    box_bottom = 0.13
    box_top = 0.31
    plot_gap = 0.035
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
        epoch_color = GLM_EPOCH_COLORS.get(epoch_payload["epoch_type"], MODEL_COLOR)
        plot_ax.axvline(0.0, color="0.45", linewidth=0.45, zorder=1)
        plot_ax.axhline(
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
        plot_ax.set_title(
            f"{epoch_payload['label']}\n{epoch_payload['epoch']}",
            fontsize=5.6,
            pad=1.5,
        )
        plot_ax.set_xlim(x_min, x_max)
        plot_ax.set_ylim(0.0, y_max)
        plot_ax.tick_params(labelbottom=False)
        if index == 0:
            plot_ax.set_ylabel("-log10 p", fontsize=5, labelpad=1)
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
            nonsig_values = finite_values[finite_p_values >= SIGNIFICANCE_P_VALUE]
            sig_values = finite_values[finite_p_values < SIGNIFICANCE_P_VALUE]
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
            box_ax.set_yticklabels(["n.s.", "p<0.05"], fontsize=4.8)
        else:
            box_ax.set_yticklabels([])
        if index == 1:
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


def plot_glm_behavior_association_panel(
    ax: "Axes",
    payload: Mapping[str, Any],
) -> None:
    """Plot dark same-turn tuning similarity by ripple-GLM significance."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    similarity_ax = ax.inset_axes([0.12, 0.18, 0.78, 0.68])
    plot_metric_significance_distributions(
        similarity_ax,
        payload.get("similarity_table"),
        metric_column="same_turn_tuning_similarity",
        x_label="Dark same-turn\ntuning similarity",
        title="Dark same-turn tuning",
        x_limits=(-0.1, 1.0),
        bin_edges=np.linspace(-0.1, 1.0, 23),
    )


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
    dpi: int,
) -> Path:
    """Build and save Figure 2."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    xcorr_animal, xcorr_date, xcorr_epoch = normalize_dataset_id(xcorr_dataset)
    heatmap_epoch_tables = load_pooled_ripple_heatmap_epoch_tables(
        data_root,
        datasets,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        sleep_epoch=sleep_epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    xcorr_payload = load_top_ca1_xcorr_panel_data(
        data_root,
        animal_name=xcorr_animal,
        date=xcorr_date,
        epoch=xcorr_epoch,
        state=xcorr_state,
        top_n_ca1_units=xcorr_top_ca1_units,
        bin_size_s=xcorr_bin_size_s,
        max_lag_s=xcorr_max_lag_s,
        display_vmax=xcorr_display_vmax,
    )
    glm_epoch_tables = load_glm_epoch_summary_tables(
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
    panel_d_payload = load_glm_behavior_association_tables(
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

    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.12, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
    )
    axes = [
        fig.add_subplot(outer_grid[0, :2]),
        fig.add_subplot(outer_grid[0, 2]),
        fig.add_subplot(outer_grid[1, :2]),
        fig.add_subplot(outer_grid[1, 2]),
    ]

    plot_epoch_ripple_heatmap_panel(axes[0], heatmap_epoch_tables, regions=regions)
    axes[0].set_title("Ripple-triggered firing rates across animals", fontsize=8, pad=2)
    plot_top_ca1_xcorr_panel(axes[1], xcorr_payload)
    axes[1].set_title("Top CA1-V1 ripple xcorr", fontsize=8, pad=2)
    plot_glm_analysis_panel(axes[2], glm_epoch_tables)
    axes[2].set_title("CA1-to-V1 ripple GLM", fontsize=8, pad=2)
    plot_glm_behavior_association_panel(axes[3], panel_d_payload)
    axes[3].set_title("Dark tuning in ripple-significant V1 units", fontsize=8, pad=2)

    for ax, label in zip(axes, ("A", "B", "C", "D"), strict=True):
        label_axis(ax, label, x=-0.10, y=1.04)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    for missing in panel_d_payload["missing_artifacts"]:
        print(
            "Panel D missing "
            f"{missing['artifact']} for {missing['animal_name']} "
            f"{missing['date']} {missing['epoch']}: {missing['path']}"
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
            "Deprecated; retained for compatibility with earlier Figure 2 "
            "versions. Format: animal:date:epoch."
        ),
    )
    parser.add_argument(
        "--xcorr-dataset",
        type=parse_dataset_id,
        default=DEFAULT_XCORR_DATASET,
        help=(
            "Screen-xcorr data set used for panel B. "
            "Format: animal:date:epoch. Default: L15:20241121:02_r1."
        ),
    )
    parser.add_argument(
        "--xcorr-state",
        choices=XCORR_STATE_CHOICES,
        default=DEFAULT_XCORR_STATE,
        help=f"Screen-xcorr state for panel B. Default: {DEFAULT_XCORR_STATE}.",
    )
    parser.add_argument(
        "--xcorr-top-ca1-units",
        type=int,
        default=DEFAULT_XCORR_TOP_CA1_UNITS,
        help=(
            "Number of top-ranked CA1 units to show in panel B. "
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
        help=f"Panel B normalized-xcorr color maximum. Default: {DEFAULT_XCORR_DISPLAY_VMAX:g}.",
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
        default=DEFAULT_RIPPLE_SELECTION,
        help=f"Ripple-GLM selection suffix. Default: {DEFAULT_RIPPLE_SELECTION}",
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
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
