from __future__ import annotations

"""Plot ripple-triggered firing-rate heatmaps for one session."""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
    load_spikes_by_region,
)


DEFAULT_RIPPLE_THRESHOLD_ZSCORE = 4.0
DEFAULT_BIN_SIZE_S = 20e-3
DEFAULT_TIME_BEFORE_S = 0.5
DEFAULT_TIME_AFTER_S = 0.5
DEFAULT_RESPONSE_WINDOW_START_S = 0.0
DEFAULT_RESPONSE_WINDOW_END_S = 0.1
DEFAULT_BASELINE_WINDOW_START_S = -0.5
DEFAULT_BASELINE_WINDOW_END_S = -0.3
HEATMAP_NORMALIZE_CHOICES = ("max", "zscore")
DEFAULT_HEATMAP_NORMALIZE = "max"
OUTPUT_DIRNAME = "ripple_modulation"
RIPPLE_EVENT_FILENAME = "ripple_times.parquet"
SUMMARY_COLUMNS = [
    "animal_name",
    "date",
    "epoch",
    "region",
    "unit_id",
    "n_ripples",
    "bin_size_s",
    "time_before_s",
    "time_after_s",
    "baseline_mean_hz",
    "baseline_std_hz",
    "response_mean_hz",
    "ripple_modulation_index",
    "response_zscore",
    "invalid_reason",
]
PERI_RIPPLE_FIRING_RATE_COLUMNS = [
    "animal_name",
    "date",
    "epoch",
    "region",
    "unit_id",
    "n_ripples",
    "bin_size_s",
    "time_before_s",
    "time_after_s",
    "time_s",
    "mean_rate_hz",
]


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI argument ranges."""
    if args.ripple_threshold_zscore <= 0:
        raise ValueError("--ripple-threshold-zscore must be positive.")
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.time_before_s <= 0:
        raise ValueError("--time-before-s must be positive.")
    if args.time_after_s <= 0:
        raise ValueError("--time-after-s must be positive.")
    if args.response_window_start_s >= args.response_window_end_s:
        raise ValueError("--response-window-start-s must be smaller than --response-window-end-s.")
    if args.baseline_window_start_s >= args.baseline_window_end_s:
        raise ValueError("--baseline-window-start-s must be smaller than --baseline-window-end-s.")


def load_detect_ripple_event_table(analysis_path: Path) -> pd.DataFrame:
    """Load the canonical ripple-event parquet written by `detect_ripples.py`."""
    ripple_path = analysis_path / "ripple" / RIPPLE_EVENT_FILENAME
    if not ripple_path.exists():
        raise FileNotFoundError(
            "Could not find saved ripple events at "
            f"{ripple_path}. Run `python -m v1ca1.ripple.detect_ripples "
            "--animal-name ... --date ...` first."
        )

    table = pd.read_parquet(ripple_path)
    rename_columns = {}
    if "start" in table.columns and "start_time" not in table.columns:
        rename_columns["start"] = "start_time"
    if "end" in table.columns and "end_time" not in table.columns:
        rename_columns["end"] = "end_time"
    if rename_columns:
        table = table.rename(columns=rename_columns)

    required_columns = ["epoch", "start_time", "end_time"]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "Saved ripple-event parquet is missing required columns: "
            f"{missing_columns!r}"
        )

    table = table.copy()
    table["epoch"] = table["epoch"].astype(str)
    table["start_time"] = np.asarray(table["start_time"], dtype=float)
    table["end_time"] = np.asarray(table["end_time"], dtype=float)
    return table


def load_detect_ripple_tables(analysis_path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-epoch ripple tables from the saved detect-ripples parquet."""
    flat_table = load_detect_ripple_event_table(analysis_path)
    ripple_tables: dict[str, pd.DataFrame] = {}
    for epoch, group in flat_table.groupby("epoch", sort=False):
        ripple_tables[str(epoch)] = group.drop(columns="epoch").reset_index(drop=True)
    return ripple_tables, "detect_ripples_parquet"


def select_epochs(
    available_epochs: list[str],
    ripple_tables: dict[str, pd.DataFrame],
    requested_epochs: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Select epochs, skipping any that are missing from `ripple_times.parquet`."""
    available_epoch_set = set(available_epochs)
    ripple_epoch_set = set(ripple_tables)

    if requested_epochs is not None:
        missing_epochs = [epoch for epoch in requested_epochs if epoch not in available_epoch_set]
        if missing_epochs:
            raise ValueError(
                f"Requested epochs were not found in the saved ephys epochs {available_epochs!r}: "
                f"{missing_epochs!r}"
            )
        selected_epochs = [epoch for epoch in requested_epochs if epoch in ripple_epoch_set]
        skipped_epochs = [epoch for epoch in requested_epochs if epoch not in ripple_epoch_set]
        return selected_epochs, skipped_epochs

    selected_epochs = [epoch for epoch in available_epochs if epoch in ripple_epoch_set]
    skipped_epochs = [epoch for epoch in available_epochs if epoch not in ripple_epoch_set]
    if not selected_epochs:
        raise ValueError(
            "No saved ephys epochs have matching ripple outputs from detect_ripples.py. "
            f"Ephys epochs: {available_epochs!r}; ripple epochs: {sorted(ripple_epoch_set)!r}"
        )
    return selected_epochs, skipped_epochs


def validate_region_sortings(analysis_path: Path, regions: tuple[str, ...]) -> None:
    """Ensure requested sorting outputs exist before loading them."""
    missing_paths = [
        analysis_path / f"sorting_{region}"
        for region in regions
        if not (analysis_path / f"sorting_{region}").exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing sorting output for the requested region(s): "
            f"{[str(path) for path in missing_paths]!r}"
        )


def filter_ripple_table_by_threshold(
    ripple_table: pd.DataFrame,
    *,
    epoch: str,
    ripple_threshold_zscore: float,
) -> pd.DataFrame:
    """Return ripples above threshold for one epoch."""
    if "mean_zscore" not in ripple_table.columns:
        raise ValueError(
            "Ripple thresholding requires a 'mean_zscore' column in the detect_ripples.py output. "
            f"Epoch {epoch!r} columns: {list(ripple_table.columns)!r}"
        )
    mean_zscores = np.asarray(ripple_table["mean_zscore"], dtype=float)
    return ripple_table.loc[mean_zscores > ripple_threshold_zscore].reset_index(drop=True)


def build_ripple_start_times(
    ripple_table: pd.DataFrame,
    *,
    epoch: str,
    ripple_threshold_zscore: float,
    epoch_timestamps: np.ndarray,
) -> tuple[Any | None, int]:
    """Build a pynapple timestamp series of ripple starts for one epoch."""
    import pynapple as nap

    filtered_table = filter_ripple_table_by_threshold(
        ripple_table,
        epoch=epoch,
        ripple_threshold_zscore=ripple_threshold_zscore,
    )
    if filtered_table.empty:
        return None, 0

    ripple_start_times = nap.Ts(
        t=filtered_table["start_time"].to_numpy(dtype=float),
        time_units="s",
        time_support=nap.IntervalSet(
            start=float(epoch_timestamps[0]),
            end=float(epoch_timestamps[-1]),
            time_units="s",
        ),
    )
    return ripple_start_times, int(len(filtered_table))


def extract_mean_rate_trace(perievent: Any, *, bin_size_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one mean firing-rate trace from a perievent object."""
    counts = perievent.count(bin_size_s)
    if not hasattr(counts, "index"):
        raise ValueError("Perievent count output must expose bin times via an 'index' attribute.")

    time_values = np.asarray(counts.index, dtype=float)
    count_values = counts.to_numpy() if hasattr(counts, "to_numpy") else np.asarray(counts)
    count_values = np.asarray(count_values, dtype=float)
    if count_values.ndim == 1:
        mean_counts = count_values
    elif count_values.ndim == 2:
        mean_counts = np.mean(count_values, axis=1)
    else:
        raise ValueError(
            "Perievent count output must be one- or two-dimensional, "
            f"got shape {count_values.shape}."
        )

    if time_values.shape[0] != mean_counts.shape[0]:
        raise ValueError(
            "Perievent bin times and mean counts must have matching lengths. "
            f"Got {time_values.shape[0]} and {mean_counts.shape[0]}."
        )
    return time_values, mean_counts / bin_size_s


def compute_modulation_stats(
    time_values: np.ndarray,
    mean_rate_hz: np.ndarray,
    *,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
) -> dict[str, Any]:
    """Compute baseline, response, modulation-index, and z-score metrics for one unit trace."""
    response_mask = (time_values >= response_window[0]) & (time_values < response_window[1])
    baseline_mask = (time_values >= baseline_window[0]) & (time_values < baseline_window[1])
    if not np.any(response_mask):
        raise ValueError(
            "The response window does not contain any PETH bins. "
            f"Window: {response_window!r}; time range: {(float(time_values[0]), float(time_values[-1]))!r}"
        )
    if not np.any(baseline_mask):
        raise ValueError(
            "The baseline window does not contain any PETH bins. "
            f"Window: {baseline_window!r}; time range: {(float(time_values[0]), float(time_values[-1]))!r}"
        )

    baseline_mean_hz = float(np.mean(mean_rate_hz[baseline_mask]))
    baseline_std_hz = float(np.std(mean_rate_hz[baseline_mask]))
    response_mean_hz = float(np.mean(mean_rate_hz[response_mask]))
    rate_sum_hz = response_mean_hz + baseline_mean_hz
    if np.isclose(rate_sum_hz, 0.0):
        ripple_modulation_index = float("nan")
    else:
        ripple_modulation_index = float(
            (response_mean_hz - baseline_mean_hz) / rate_sum_hz
        )

    invalid_reason: str | None = None
    if np.isclose(baseline_std_hz, 0.0):
        response_zscore = float("nan")
        invalid_reason = "zero_baseline_std"
    else:
        response_zscore = float((response_mean_hz - baseline_mean_hz) / baseline_std_hz)

    return {
        "baseline_mean_hz": baseline_mean_hz,
        "baseline_std_hz": baseline_std_hz,
        "response_mean_hz": response_mean_hz,
        "ripple_modulation_index": ripple_modulation_index,
        "response_zscore": response_zscore,
        "invalid_reason": invalid_reason,
    }


def build_region_epoch_modulation_result(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    region_spikes: Any,
    ripple_start_times: Any,
    n_ripples: int,
    bin_size_s: float,
    time_before_s: float,
    time_after_s: float,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
) -> dict[str, Any]:
    """Return summary rows and the full mean-rate matrix for one region and epoch."""
    import pynapple as nap

    rows: list[dict[str, Any]] = []
    unit_ids: list[Any] = []
    mean_rate_rows: list[np.ndarray] = []
    shared_time_values: np.ndarray | None = None

    for unit_id in region_spikes.keys():
        perievent = nap.compute_perievent(
            timestamps=region_spikes[unit_id],
            tref=ripple_start_times,
            minmax=(-time_before_s, time_after_s),
            time_unit="s",
        )
        time_values, mean_rate_hz = extract_mean_rate_trace(perievent, bin_size_s=bin_size_s)
        if shared_time_values is None:
            shared_time_values = time_values
        elif shared_time_values.shape != time_values.shape or not np.allclose(
            shared_time_values,
            time_values,
        ):
            raise ValueError("All units in one region/epoch must share the same PETH bin times.")

        stats = compute_modulation_stats(
            time_values,
            mean_rate_hz,
            response_window=response_window,
            baseline_window=baseline_window,
        )
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "epoch": epoch,
                "region": region,
                "unit_id": unit_id,
                "n_ripples": n_ripples,
                "bin_size_s": float(bin_size_s),
                "time_before_s": float(time_before_s),
                "time_after_s": float(time_after_s),
                **stats,
            }
        )
        unit_ids.append(unit_id)
        mean_rate_rows.append(np.asarray(mean_rate_hz, dtype=float))

    if shared_time_values is None:
        shared_time_values = np.array([], dtype=float)
    mean_rate_matrix = (
        np.vstack(mean_rate_rows)
        if mean_rate_rows
        else np.empty((0, shared_time_values.size), dtype=float)
    )
    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "region": region,
        "n_ripples": int(n_ripples),
        "bin_size_s": float(bin_size_s),
        "time_before_s": float(time_before_s),
        "time_after_s": float(time_after_s),
        "rows": rows,
        "unit_ids": np.asarray(unit_ids, dtype=object),
        "time_values": np.asarray(shared_time_values, dtype=float),
        "mean_rate_hz": mean_rate_matrix,
    }


def format_output_value(value: float | str) -> str:
    """Return a filename-safe string for one parameter value."""
    if isinstance(value, str):
        return value
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def build_epoch_output_stem(
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
    """Return the shared filename stem for one epoch output."""
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


def get_epoch_output_paths(
    analysis_path: Path,
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
) -> dict[str, Path]:
    """Return output paths for one epoch."""
    stem = build_epoch_output_stem(
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
    data_dir = analysis_path / "ripple" / OUTPUT_DIRNAME
    figure_dir = analysis_path / "figs" / OUTPUT_DIRNAME
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "peri_ripple_firing_rate": data_dir / f"{stem}_peri_ripple_firing_rate.parquet",
        "summary": data_dir / f"{stem}_summary.parquet",
        "figure": figure_dir / f"{stem}_heatmap.png",
    }


def save_summary_table(summary_table: pd.DataFrame, path: Path) -> Path:
    """Write the ripple modulation summary table as parquet."""
    ordered_columns = [column for column in SUMMARY_COLUMNS if column in summary_table.columns]
    ordered_columns.extend(column for column in summary_table.columns if column not in ordered_columns)
    output_table = summary_table.loc[:, ordered_columns].copy()
    output_table.to_parquet(path, index=False)
    return path


def build_peri_ripple_firing_rate_table(
    heatmap_payloads: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Flatten cached per-unit peri-ripple firing-rate traces into one table."""
    rows: list[dict[str, Any]] = []
    for region, payload in heatmap_payloads.items():
        unit_ids = np.asarray(payload["unit_ids"], dtype=object)
        time_values = np.asarray(payload["time_values"], dtype=float)
        mean_rate_matrix = np.asarray(payload["mean_rate_hz"], dtype=float)
        if mean_rate_matrix.shape != (unit_ids.size, time_values.size):
            raise ValueError(
                "Heatmap payload matrix shape does not match unit/time coordinates for "
                f"{region} {payload['epoch']}: {mean_rate_matrix.shape}, {unit_ids.size}, {time_values.size}."
            )

        for row_index, unit_id in enumerate(unit_ids):
            for time_index, time_value in enumerate(time_values):
                rows.append(
                    {
                        "animal_name": payload["animal_name"],
                        "date": payload["date"],
                        "epoch": payload["epoch"],
                        "region": region,
                        "unit_id": unit_id,
                        "n_ripples": payload["n_ripples"],
                        "bin_size_s": payload["bin_size_s"],
                        "time_before_s": payload["time_before_s"],
                        "time_after_s": payload["time_after_s"],
                        "time_s": float(time_value),
                        "mean_rate_hz": float(mean_rate_matrix[row_index, time_index]),
                    }
                )

    return pd.DataFrame(rows, columns=PERI_RIPPLE_FIRING_RATE_COLUMNS)


def save_peri_ripple_firing_rate_table(table: pd.DataFrame, path: Path) -> Path:
    """Write the cached peri-ripple firing-rate table as parquet."""
    ordered_columns = [
        column for column in PERI_RIPPLE_FIRING_RATE_COLUMNS if column in table.columns
    ]
    ordered_columns.extend(column for column in table.columns if column not in ordered_columns)
    output_table = table.loc[:, ordered_columns].copy()
    output_table.to_parquet(path, index=False)
    return path


def load_peri_ripple_firing_rate_table(path: Path) -> pd.DataFrame:
    """Load one cached peri-ripple firing-rate parquet table."""
    if not path.exists():
        raise FileNotFoundError(f"Peri-ripple firing-rate parquet not found: {path}")
    table = pd.read_parquet(path)
    missing_columns = [
        column for column in PERI_RIPPLE_FIRING_RATE_COLUMNS if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            "Saved peri-ripple firing-rate parquet is missing required columns: "
            f"{missing_columns!r}"
        )
    return table


def build_heatmap_payloads_from_table(
    firing_rate_table: pd.DataFrame,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Reconstruct one epoch's heatmap payloads from a cached firing-rate table."""
    if firing_rate_table.empty:
        raise ValueError("Saved peri-ripple firing-rate parquet is empty.")

    epoch_values = firing_rate_table["epoch"].astype(str).unique().tolist()
    if len(epoch_values) != 1:
        raise ValueError(
            "Expected one cached peri-ripple firing-rate parquet per epoch, "
            f"found epochs {epoch_values!r}."
        )
    epoch = str(epoch_values[0])
    payloads: dict[str, dict[str, Any]] = {}

    group_columns = [
        "animal_name",
        "date",
        "epoch",
        "region",
        "unit_id",
        "n_ripples",
        "bin_size_s",
        "time_before_s",
        "time_after_s",
    ]
    for region, group in firing_rate_table.groupby("region", sort=False):
        sorted_group = group.sort_values(by=["unit_id", "time_s"], kind="mergesort").reset_index(drop=True)
        unit_groups = list(sorted_group.groupby(group_columns, sort=False))
        first_key = unit_groups[0][0]
        time_values = unit_groups[0][1]["time_s"].to_numpy(dtype=float)
        unit_ids: list[Any] = []
        mean_rate_rows: list[np.ndarray] = []
        for key, unit_group in unit_groups:
            unit_time_values = unit_group["time_s"].to_numpy(dtype=float)
            if unit_time_values.shape != time_values.shape or not np.allclose(
                unit_time_values,
                time_values,
            ):
                raise ValueError(
                    "Saved peri-ripple firing-rate parquet has inconsistent time bins "
                    f"for {region} {epoch}."
                )
            unit_ids.append(key[4])
            mean_rate_rows.append(unit_group["mean_rate_hz"].to_numpy(dtype=float))

        payloads[str(region)] = {
            "animal_name": str(first_key[0]),
            "date": str(first_key[1]),
            "epoch": str(first_key[2]),
            "region": str(first_key[3]),
            "n_ripples": int(first_key[5]),
            "bin_size_s": float(first_key[6]),
            "time_before_s": float(first_key[7]),
            "time_after_s": float(first_key[8]),
            "unit_ids": np.asarray(unit_ids, dtype=object),
            "time_values": np.asarray(time_values, dtype=float),
            "mean_rate_hz": np.vstack(mean_rate_rows),
        }
    return epoch, payloads


def build_summary_table_from_firing_rate_table(
    firing_rate_table: pd.DataFrame,
    *,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
) -> pd.DataFrame:
    """Build the per-unit summary table from a cached firing-rate table."""
    rows: list[dict[str, Any]] = []
    group_columns = [
        "animal_name",
        "date",
        "epoch",
        "region",
        "unit_id",
        "n_ripples",
        "bin_size_s",
        "time_before_s",
        "time_after_s",
    ]
    for key, group in firing_rate_table.groupby(group_columns, sort=False):
        ordered_group = group.sort_values(by="time_s", kind="mergesort").reset_index(drop=True)
        stats = compute_modulation_stats(
            ordered_group["time_s"].to_numpy(dtype=float),
            ordered_group["mean_rate_hz"].to_numpy(dtype=float),
            response_window=response_window,
            baseline_window=baseline_window,
        )
        rows.append(
            {
                "animal_name": key[0],
                "date": key[1],
                "epoch": key[2],
                "region": key[3],
                "unit_id": key[4],
                "n_ripples": int(key[5]),
                "bin_size_s": float(key[6]),
                "time_before_s": float(key[7]),
                "time_after_s": float(key[8]),
                **stats,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def normalize_heatmap_rows(
    values: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Normalize each unit row for heatmap display."""
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap matrix, got shape {value_array.shape}.")

    if mode == "max":
        row_scale = np.full(value_array.shape[0], np.nan, dtype=float)
        finite_rows = np.isfinite(value_array).any(axis=1)
        if np.any(finite_rows):
            row_scale[finite_rows] = np.nanmax(value_array[finite_rows], axis=1)
        valid_rows = np.isfinite(row_scale) & (row_scale > 0)
        normalized = np.full_like(value_array, np.nan, dtype=float)
        if np.any(valid_rows):
            normalized[valid_rows] = value_array[valid_rows] / row_scale[valid_rows, None]
        return normalized

    if mode == "zscore":
        row_mean = np.nanmean(value_array, axis=1)
        row_std = np.nanstd(value_array, axis=1)
        valid_rows = np.isfinite(row_mean) & np.isfinite(row_std) & (row_std > 0)
        normalized = np.full_like(value_array, np.nan, dtype=float)
        if np.any(valid_rows):
            normalized[valid_rows] = (
                value_array[valid_rows] - row_mean[valid_rows, None]
            ) / row_std[valid_rows, None]
        return normalized

    raise ValueError(
        f"Unsupported heatmap normalization {mode!r}. "
        f"Expected one of {HEATMAP_NORMALIZE_CHOICES!r}."
    )


def compute_heatmap_unit_order(values: np.ndarray) -> np.ndarray:
    """Order units by peak mean firing rate from highest to lowest."""
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap matrix, got shape {value_array.shape}.")
    if value_array.shape[0] == 0:
        return np.array([], dtype=int)

    row_peak = np.full(value_array.shape[0], -np.inf, dtype=float)
    finite_rows = np.isfinite(value_array).any(axis=1)
    if np.any(finite_rows):
        row_peak[finite_rows] = np.nanmax(value_array[finite_rows], axis=1)
    return np.argsort(-row_peak, kind="stable")


def save_epoch_heatmap_figure(
    epoch_payloads: dict[str, dict[str, Any]],
    *,
    animal_name: str,
    date: str,
    epoch: str,
    selected_regions: tuple[str, ...],
    heatmap_normalize: str,
    out_path: Path,
    show: bool = False,
) -> Path:
    """Save one epoch-specific grid of ripple-triggered firing-rate heatmaps."""
    nrows = len(selected_regions)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8, max(3 * nrows, 3)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    zscore_abs_max = 1.0
    if heatmap_normalize == "zscore":
        valid_values: list[np.ndarray] = []
        for payload in epoch_payloads.values():
            normalized_matrix = normalize_heatmap_rows(payload["mean_rate_hz"], mode=heatmap_normalize)
            finite_values = normalized_matrix[np.isfinite(normalized_matrix)]
            if finite_values.size:
                valid_values.append(finite_values)
        if valid_values:
            zscore_abs_max = max(1.0, float(np.max(np.abs(np.concatenate(valid_values)))))

    for row_idx, region in enumerate(selected_regions):
        ax = axes[row_idx, 0]
        payload = epoch_payloads.get(region)
        if payload is None or payload["n_ripples"] == 0:
            ax.text(
                0.5,
                0.5,
                "No ripples above threshold",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            n_units = 0
            ripple_count = 0
        elif payload["mean_rate_hz"].shape[0] == 0:
            ax.text(
                0.5,
                0.5,
                "No units",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            n_units = 0
            ripple_count = int(payload["n_ripples"])
        else:
            raw_matrix = np.asarray(payload["mean_rate_hz"], dtype=float)
            normalized_matrix = normalize_heatmap_rows(raw_matrix, mode=heatmap_normalize)
            unit_order = compute_heatmap_unit_order(raw_matrix)
            ordered_matrix = normalized_matrix[unit_order]
            time_values = np.asarray(payload["time_values"], dtype=float)
            if heatmap_normalize == "max":
                image = ax.imshow(
                    ordered_matrix,
                    origin="upper",
                    aspect="auto",
                    interpolation="nearest",
                    extent=[time_values[0], time_values[-1], ordered_matrix.shape[0], 0],
                    vmin=0.0,
                    vmax=1.0,
                    cmap="viridis",
                )
            else:
                image = ax.imshow(
                    ordered_matrix,
                    origin="upper",
                    aspect="auto",
                    interpolation="nearest",
                    extent=[time_values[0], time_values[-1], ordered_matrix.shape[0], 0],
                    vmin=-zscore_abs_max,
                    vmax=zscore_abs_max,
                    cmap="RdBu_r",
                )
            n_units = int(payload["mean_rate_hz"].shape[0])
            ripple_count = int(payload["n_ripples"])

        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{region} {epoch}\n{ripple_count} ripples, {n_units} units")
        ax.set_ylabel("units ordered by peak firing")
        if row_idx == nrows - 1:
            ax.set_xlabel("Time from ripple start (s)")
        ax.set_yticks([])

    fig.suptitle(f"{animal_name} {date} {epoch} ripple-triggered firing-rate heatmaps")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
        if heatmap_normalize == "max":
            colorbar.set_label("max-normalized mean firing rate")
        else:
            colorbar.set_label("z-scored mean firing rate")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_ripple_modulation_for_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    region: str | None = None,
    epochs: list[str] | None = None,
    ripple_threshold_zscore: float = DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    time_before_s: float = DEFAULT_TIME_BEFORE_S,
    time_after_s: float = DEFAULT_TIME_AFTER_S,
    response_window_start_s: float = DEFAULT_RESPONSE_WINDOW_START_S,
    response_window_end_s: float = DEFAULT_RESPONSE_WINDOW_END_S,
    baseline_window_start_s: float = DEFAULT_BASELINE_WINDOW_START_S,
    baseline_window_end_s: float = DEFAULT_BASELINE_WINDOW_END_S,
    heatmap_normalize: str = DEFAULT_HEATMAP_NORMALIZE,
    overwrite: bool = False,
    show: bool = False,
) -> dict[str, Any]:
    """Compute ripple modulation summaries and save one cache/summary/figure set per epoch."""
    print(f"Starting ripple modulation for {animal_name} {date}")
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    selected_regions = (region,) if region is not None else REGIONS
    validate_region_sortings(analysis_path, selected_regions)
    print(f"Using analysis path: {analysis_path}")
    print(f"Selected regions: {list(selected_regions)!r}")

    response_window = (response_window_start_s, response_window_end_s)
    baseline_window = (baseline_window_start_s, baseline_window_end_s)

    print("Loading detect_ripples.py output...")
    ripple_tables, ripple_source = load_detect_ripple_tables(analysis_path)
    print("Loading saved ephys timestamps by epoch...")
    available_epochs, timestamps_ephys_by_epoch, timestamps_source = load_ephys_timestamps_by_epoch(
        analysis_path
    )
    selected_epochs, skipped_epochs_without_ripple_output = select_epochs(
        available_epochs,
        ripple_tables,
        requested_epochs=epochs,
    )
    if skipped_epochs_without_ripple_output:
        print(
            "Skipping epochs not present in ripple_times.parquet: "
            f"{skipped_epochs_without_ripple_output!r}"
        )
    print(f"Selected epochs: {selected_epochs!r}")

    timestamps_ephys_all: np.ndarray | None = None
    timestamps_all_source = "not_loaded_used_saved_peri_ripple_firing_rate_for_all_epochs"
    spikes_by_region: dict[str, Any] | None = None

    epoch_results: dict[str, dict[str, Any]] = {}
    summary_tables: list[pd.DataFrame] = []
    saved_peri_paths: list[Path] = []
    saved_summary_paths: list[Path] = []
    saved_figure_paths: list[Path] = []
    skipped_epochs_below_threshold: list[str] = []

    for epoch in selected_epochs:
        print(f"Preparing outputs for epoch {epoch}")
        region_label = region if region is not None else "all_regions"
        output_paths = get_epoch_output_paths(
            analysis_path,
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
        peri_path = output_paths["peri_ripple_firing_rate"]

        if peri_path.exists() and not overwrite:
            print(f"Using saved peri-ripple firing-rate table for epoch {epoch}: {peri_path}")
            firing_rate_table = load_peri_ripple_firing_rate_table(peri_path)
            cached_epoch, epoch_payloads = build_heatmap_payloads_from_table(firing_rate_table)
            if cached_epoch != epoch:
                raise ValueError(
                    f"Cached peri-ripple firing-rate table {peri_path} is for epoch {cached_epoch!r}, "
                    f"expected {epoch!r}."
                )
        else:
            if overwrite and peri_path.exists():
                print(f"Recomputing saved peri-ripple firing-rate table for epoch {epoch}: {peri_path}")
            else:
                print(f"Computing peri-ripple firing-rate table for epoch {epoch}")

            ripple_start_times, n_ripples = build_ripple_start_times(
                ripple_tables[epoch],
                epoch=epoch,
                ripple_threshold_zscore=ripple_threshold_zscore,
                epoch_timestamps=timestamps_ephys_by_epoch[epoch],
            )
            if n_ripples == 0:
                skipped_epochs_below_threshold.append(epoch)
                print(
                    "Skipping epoch with no ripples above threshold: "
                    f"{animal_name} {date} {epoch}"
                )
                continue

            if timestamps_ephys_all is None:
                print("Loading concatenated ephys timestamps...")
                timestamps_ephys_all, timestamps_all_source = load_ephys_timestamps_all(analysis_path)
            if spikes_by_region is None:
                print("Loading spike trains by region...")
                spikes_by_region = load_spikes_by_region(
                    analysis_path,
                    timestamps_ephys_all,
                    regions=selected_regions,
                )

            epoch_payloads = {}
            for region_name in selected_regions:
                print(f"Computing peri-ripple firing rates for {epoch} {region_name}")
                epoch_payloads[region_name] = build_region_epoch_modulation_result(
                    animal_name=animal_name,
                    date=date,
                    epoch=epoch,
                    region=region_name,
                    region_spikes=spikes_by_region[region_name],
                    ripple_start_times=ripple_start_times,
                    n_ripples=n_ripples,
                    bin_size_s=bin_size_s,
                    time_before_s=time_before_s,
                    time_after_s=time_after_s,
                    response_window=response_window,
                    baseline_window=baseline_window,
                )

            firing_rate_table = build_peri_ripple_firing_rate_table(epoch_payloads)
            save_peri_ripple_firing_rate_table(firing_rate_table, peri_path)
            print(f"Saved peri-ripple firing-rate table for epoch {epoch} to {peri_path}")

        if firing_rate_table.empty:
            raise RuntimeError(f"Peri-ripple firing-rate table is empty for epoch {epoch}: {peri_path}")

        summary_table = build_summary_table_from_firing_rate_table(
            firing_rate_table,
            response_window=response_window,
            baseline_window=baseline_window,
        )
        summary_path = save_summary_table(summary_table, output_paths["summary"])
        print(f"Saved summary table for epoch {epoch} to {summary_path}")
        figure_path = save_epoch_heatmap_figure(
            epoch_payloads,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            selected_regions=selected_regions,
            heatmap_normalize=heatmap_normalize,
            out_path=output_paths["figure"],
            show=show,
        )
        print(f"Saved heatmap figure for epoch {epoch} to {figure_path}")

        epoch_results[epoch] = {
            "peri_ripple_firing_rate_path": peri_path,
            "summary_path": summary_path,
            "figure_path": figure_path,
            "summary_table": summary_table,
            "heatmap_payloads": epoch_payloads,
            "ripple_count": int(next(iter(epoch_payloads.values()))["n_ripples"]),
        }
        summary_tables.append(summary_table)
        saved_peri_paths.append(peri_path)
        saved_summary_paths.append(summary_path)
        saved_figure_paths.append(figure_path)

    if not epoch_results:
        raise RuntimeError(
            "No epochs produced ripple modulation outputs. "
            f"Selected epochs: {selected_epochs!r}; threshold: {ripple_threshold_zscore}"
        )

    summary_table_all = pd.concat(summary_tables, ignore_index=True, sort=False)
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name=Path(__file__).stem,
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "region": region,
            "epochs": epochs,
            "ripple_threshold_zscore": ripple_threshold_zscore,
            "bin_size_s": bin_size_s,
            "time_before_s": time_before_s,
            "time_after_s": time_after_s,
            "response_window_start_s": response_window_start_s,
            "response_window_end_s": response_window_end_s,
            "baseline_window_start_s": baseline_window_start_s,
            "baseline_window_end_s": baseline_window_end_s,
            "heatmap_normalize": heatmap_normalize,
            "overwrite": overwrite,
            "show": show,
        },
        outputs={
            "sources": {
                "timestamps_ephys": timestamps_source,
                "timestamps_ephys_all": timestamps_all_source,
                "ripple_events": ripple_source,
            },
            "selected_regions": list(selected_regions),
            "selected_epochs": selected_epochs,
            "successful_epochs": sorted(epoch_results),
            "skipped_epochs_without_ripple_output": skipped_epochs_without_ripple_output,
            "skipped_epochs_below_threshold": skipped_epochs_below_threshold,
            "saved_peri_ripple_firing_rate_paths": saved_peri_paths,
            "saved_summary_paths": saved_summary_paths,
            "saved_figure_paths": saved_figure_paths,
            "n_summary_rows": int(len(summary_table_all)),
            "n_invalid_zero_baseline_std": int(
                np.sum(summary_table_all["invalid_reason"].fillna("").to_numpy() == "zero_baseline_std")
            ),
        },
    )
    print(f"Saved run metadata to {log_path}")

    return {
        "analysis_path": analysis_path,
        "epoch_results": epoch_results,
        "log_path": log_path,
        "summary_table": summary_table_all,
        "selected_epochs": selected_epochs,
        "selected_regions": selected_regions,
        "skipped_epochs_without_ripple_output": skipped_epochs_without_ripple_output,
        "skipped_epochs_below_threshold": skipped_epochs_below_threshold,
    }


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ripple modulation workflow."""
    parser = argparse.ArgumentParser(
        description="Plot ripple-triggered firing-rate heatmaps for one session"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        help="Only process one region. Default: process all regions.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="Optional subset of epochs to process. Default: all epochs with ripple outputs.",
    )
    parser.add_argument(
        "--ripple-threshold-zscore",
        type=float,
        default=DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
        help=(
            "Minimum ripple mean z-score to keep one ripple event. "
            f"Default: {DEFAULT_RIPPLE_THRESHOLD_ZSCORE}"
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"PETH bin size in seconds. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--time-before-s",
        type=float,
        default=DEFAULT_TIME_BEFORE_S,
        help=f"Seconds before ripple start for the PETH window. Default: {DEFAULT_TIME_BEFORE_S}",
    )
    parser.add_argument(
        "--time-after-s",
        type=float,
        default=DEFAULT_TIME_AFTER_S,
        help=f"Seconds after ripple start for the PETH window. Default: {DEFAULT_TIME_AFTER_S}",
    )
    parser.add_argument(
        "--response-window-start-s",
        type=float,
        default=DEFAULT_RESPONSE_WINDOW_START_S,
        help=(
            "Start of the response window in seconds relative to ripple start. "
            f"Default: {DEFAULT_RESPONSE_WINDOW_START_S}"
        ),
    )
    parser.add_argument(
        "--response-window-end-s",
        type=float,
        default=DEFAULT_RESPONSE_WINDOW_END_S,
        help=(
            "End of the response window in seconds relative to ripple start. "
            f"Default: {DEFAULT_RESPONSE_WINDOW_END_S}"
        ),
    )
    parser.add_argument(
        "--baseline-window-start-s",
        type=float,
        default=DEFAULT_BASELINE_WINDOW_START_S,
        help=(
            "Start of the baseline window in seconds relative to ripple start. "
            f"Default: {DEFAULT_BASELINE_WINDOW_START_S}"
        ),
    )
    parser.add_argument(
        "--baseline-window-end-s",
        type=float,
        default=DEFAULT_BASELINE_WINDOW_END_S,
        help=(
            "End of the baseline window in seconds relative to ripple start. "
            f"Default: {DEFAULT_BASELINE_WINDOW_END_S}"
        ),
    )
    parser.add_argument(
        "--heatmap-normalize",
        choices=HEATMAP_NORMALIZE_CHOICES,
        default=DEFAULT_HEATMAP_NORMALIZE,
        help=(
            "How to normalize each unit row before plotting the heatmap. "
            f"Default: {DEFAULT_HEATMAP_NORMALIZE}"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite saved peri-ripple firing-rate outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the ripple modulation CLI."""
    args = parse_arguments(argv)
    validate_arguments(args)
    try:
        plot_ripple_modulation_for_session(
            animal_name=args.animal_name,
            date=args.date,
            data_root=args.data_root,
            region=args.region,
            epochs=args.epochs,
            ripple_threshold_zscore=args.ripple_threshold_zscore,
            bin_size_s=args.bin_size_s,
            time_before_s=args.time_before_s,
            time_after_s=args.time_after_s,
            response_window_start_s=args.response_window_start_s,
            response_window_end_s=args.response_window_end_s,
            baseline_window_start_s=args.baseline_window_start_s,
            baseline_window_end_s=args.baseline_window_end_s,
            heatmap_normalize=args.heatmap_normalize,
            overwrite=args.overwrite,
            show=args.show,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
