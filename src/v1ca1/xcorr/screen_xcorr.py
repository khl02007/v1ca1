from __future__ import annotations

"""Screen cross-region spike-timing structure for one session/state.

This CLI pools CA1 and V1 spikes within one requested state, computes
target-rate-normalized cross-correlograms with pynapple, and summarizes the
strongest feature across the computed lag range for each CA1-V1 pair. The
primary outputs are:

- one parquet summary table with one row per retained CA1-V1 pair
- one NetCDF-backed `xarray.Dataset` storing the normalized xcorr tensor with
  dimensions `(ca1_unit, v1_unit, lag_s)`
- one session-level overview figure
- one CA1-centered heatmap per retained CA1 unit

The workflow is state-specific. For `ripple`, spikes are restricted to the
detected ripple intervals themselves. For `run`, `immobility`, and `sleep`,
the script reuses the session interval artifacts already produced elsewhere in
the repo. A JSON run log is written under `analysis_path / "v1ca1_log"`.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
    get_analysis_path,
    load_epoch_tags,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
    load_spikes_by_region,
)
from v1ca1.ripple.ripple_glm import (
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    _format_window_suffix_value,
    format_ripple_window_suffix,
    load_ripple_tables,
)


STATE_CHOICES = ("ripple", "run", "immobility", "sleep")
POOLED_EPOCH_SENTINEL = "pooled"
DEFAULT_BIN_SIZE_S = 0.005
DEFAULT_MAX_LAG_S = 0.5
DEFAULT_MIN_STATE_SPIKES = 30
DEFAULT_EXTREMUM_HALF_WIDTH_BINS = 1
DEFAULT_DISPLAY_VMAX = 5.0
DEFAULT_OUTPUT_DIRNAME = "screen_pairs"
SUMMARY_FILENAME = "xcorr_summary.parquet"
DATASET_FILENAME = "xcorr.nc"
OVERVIEW_FIGURE_FILENAME = "overview.png"
CA1_HEATMAP_DIRNAME = "ca1_heatmaps"
PAIR_STATUS_VALID = "valid"
PAIR_STATUS_NO_FINITE_BINS = "no_finite_bins"


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numeric CLI ranges."""
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.max_lag_s <= 0:
        raise ValueError("--max-lag-s must be positive.")
    if args.ripple_window_s is not None and args.ripple_window_s <= 0:
        raise ValueError("--ripple-window-s must be positive.")
    if not np.isfinite(args.ripple_window_offset_s):
        raise ValueError("--ripple-window-offset-s must be finite.")
    if args.state != "ripple":
        if args.ripple_window_s is not None or not np.isclose(
            float(args.ripple_window_offset_s),
            DEFAULT_RIPPLE_WINDOW_OFFSET_S,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "--ripple-window-s and --ripple-window-offset-s are supported only for --state ripple."
            )
    elif args.ripple_window_s is None and not np.isclose(
        float(args.ripple_window_offset_s),
        DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("--ripple-window-offset-s requires --ripple-window-s.")
    if args.min_state_spikes < 0:
        raise ValueError("--min-state-spikes must be non-negative.")
    if args.extremum_half_width_bins < 0:
        raise ValueError("--extremum-half-width-bins must be non-negative.")
    if args.display_vmax <= 0:
        raise ValueError("--display-vmax must be positive.")


def require_xarray():
    """Return `xarray` and fail clearly when NetCDF output is requested."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `xarray` to save xcorr tensors as NetCDF."
        ) from exc
    return xr


def _extract_intervalset_epoch_labels(intervals: Any) -> list[str]:
    """Return epoch labels stored on a pynapple IntervalSet."""
    try:
        epoch_info = intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is not None:
        epoch_array = np.asarray(epoch_info)
        if epoch_array.size:
            return [str(epoch) for epoch in epoch_array.tolist()]

    if hasattr(intervals, "as_dataframe"):
        interval_df = intervals.as_dataframe()
        if "epoch" in interval_df.columns:
            return [str(epoch) for epoch in interval_df["epoch"].tolist()]

    raise ValueError("Could not extract epoch labels from the saved IntervalSet.")


def _interval_rows_to_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return aligned interval rows with canonical start/end column names."""
    interval_table = table.copy()
    if {"start", "end"}.issubset(interval_table.columns):
        interval_table = interval_table.loc[:, ["start", "end"]].copy()
    elif {"start_time", "end_time"}.issubset(interval_table.columns):
        interval_table = interval_table.loc[:, ["start_time", "end_time"]].rename(
            columns={"start_time": "start", "end_time": "end"}
        )
    else:
        raise ValueError(
            "Expected interval rows to contain either start/end or start_time/end_time "
            f"columns. Found {list(interval_table.columns)!r}."
        )

    interval_table["start"] = interval_table["start"].astype(float)
    interval_table["end"] = interval_table["end"].astype(float)
    interval_table = interval_table.loc[interval_table["end"] > interval_table["start"]].reset_index(
        drop=True
    )
    return interval_table


def _get_ripple_anchor_starts(ripple_table: pd.DataFrame) -> np.ndarray:
    """Return ripple start times from one saved ripple table."""
    if "start_time" in ripple_table.columns:
        return np.asarray(ripple_table["start_time"], dtype=float)
    if "start" in ripple_table.columns:
        return np.asarray(ripple_table["start"], dtype=float)
    raise ValueError(
        "Expected ripple table to contain either 'start_time' or 'start' columns. "
        f"Found {list(ripple_table.columns)!r}."
    )


def merge_interval_rows(interval_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping interval rows and return canonical start/end columns."""
    table = _interval_rows_to_table(interval_rows)
    if table.empty:
        return table

    starts = table["start"].to_numpy(dtype=float)
    ends = table["end"].to_numpy(dtype=float)
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    ends = ends[order]

    merged_starts = [float(starts[0])]
    merged_ends = [float(ends[0])]
    for start, end in zip(starts[1:], ends[1:], strict=True):
        if float(start) <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], float(end))
        else:
            merged_starts.append(float(start))
            merged_ends.append(float(end))

    return pd.DataFrame({"start": merged_starts, "end": merged_ends})


def _load_interval_rows_from_npz(path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-epoch intervals from one root-level pynapple IntervalSet output."""
    import pynapple as nap

    if not path.exists():
        raise FileNotFoundError(path)

    intervals = nap.load_file(path)
    epoch_labels = _extract_intervalset_epoch_labels(intervals)
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape or starts.size != len(epoch_labels):
        raise ValueError(
            f"Saved interval output {path} has mismatched starts, ends, and epoch labels."
        )

    rows = pd.DataFrame(
        {
            "epoch": [str(epoch) for epoch in epoch_labels],
            "start": starts,
            "end": ends,
        }
    )
    intervals_by_epoch = {
        epoch: interval_rows.reset_index(drop=True)
        for epoch, interval_rows in rows.groupby("epoch", sort=False)[["start", "end"]]
    }
    return intervals_by_epoch, "pynapple"


def _load_interval_rows_from_root_pickle(path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-epoch intervals from one root-level legacy pickle mapping."""
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as file:
        raw_value = pickle.load(file)
    if not isinstance(raw_value, dict):
        raise ValueError(f"Expected a dict in {path}, found {type(raw_value).__name__}.")
    return {
        str(epoch): _interval_rows_to_table(pd.DataFrame(interval_rows))
        for epoch, interval_rows in raw_value.items()
    }, "pickle"


def _load_interval_rows_from_legacy_dir(path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-epoch intervals from a legacy directory of pickle tables."""
    if not path.exists():
        raise FileNotFoundError(path)

    interval_paths = sorted(path.glob("*.pkl"))
    if not interval_paths:
        raise FileNotFoundError(f"Could not find any interval pickle files under {path}.")
    intervals_by_epoch: dict[str, pd.DataFrame] = {}
    for interval_path in interval_paths:
        with open(interval_path, "rb") as file:
            interval_rows = pickle.load(file)
        intervals_by_epoch[str(interval_path.stem)] = _interval_rows_to_table(pd.DataFrame(interval_rows))
    return intervals_by_epoch, "legacy_dir_pickle"


def _load_interval_rows_from_parquet(path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-epoch intervals from one modern root-level parquet export."""
    if not path.exists():
        raise FileNotFoundError(path)

    table = pd.read_parquet(path)
    required_columns = {"epoch", "start", "end"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"Sleep interval parquet {path} is missing required columns {sorted(missing_columns)!r}."
        )
    table = table.loc[:, ["epoch", "start", "end"]].copy()
    table["epoch"] = table["epoch"].astype(str)
    table["start"] = table["start"].astype(float)
    table["end"] = table["end"].astype(float)
    intervals_by_epoch = {
        epoch: epoch_rows.loc[:, ["start", "end"]].reset_index(drop=True)
        for epoch, epoch_rows in table.groupby("epoch", sort=False)
    }
    return intervals_by_epoch, "parquet"


def load_state_interval_tables(
    analysis_path: Path,
    state: str,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Load per-epoch intervals for the requested state."""
    if state == "ripple":
        ripple_tables, source = load_ripple_tables(analysis_path)
        return {
            str(epoch): _interval_rows_to_table(interval_rows)
            for epoch, interval_rows in ripple_tables.items()
        }, source

    if state == "run":
        parquet_path = analysis_path / "run_times.parquet"
        if parquet_path.exists():
            return _load_interval_rows_from_parquet(parquet_path)
        npz_path = analysis_path / "run_times.npz"
        if npz_path.exists():
            return _load_interval_rows_from_npz(npz_path)
        pickle_path = analysis_path / "run_times.pkl"
        if pickle_path.exists():
            return _load_interval_rows_from_root_pickle(pickle_path)
        legacy_dir = analysis_path / "run_times"
        return _load_interval_rows_from_legacy_dir(legacy_dir)

    if state == "immobility":
        parquet_path = analysis_path / "immobility_times.parquet"
        if parquet_path.exists():
            return _load_interval_rows_from_parquet(parquet_path)
        npz_path = analysis_path / "immobility_times.npz"
        if npz_path.exists():
            return _load_interval_rows_from_npz(npz_path)
        pickle_path = analysis_path / "immobility_times.pkl"
        if pickle_path.exists():
            return _load_interval_rows_from_root_pickle(pickle_path)
        legacy_dir = analysis_path / "immobility_times"
        return _load_interval_rows_from_legacy_dir(legacy_dir)

    if state == "sleep":
        parquet_path = analysis_path / "sleep_times.parquet"
        if parquet_path.exists():
            return _load_interval_rows_from_parquet(parquet_path)
        pickle_path = analysis_path / "sleep_times.pkl"
        if pickle_path.exists():
            return _load_interval_rows_from_root_pickle(pickle_path)
        legacy_dir = analysis_path / "sleep_times"
        return _load_interval_rows_from_legacy_dir(legacy_dir)

    raise ValueError(f"Unsupported state {state!r}. Expected one of {STATE_CHOICES!r}.")


def build_fixed_ripple_window_tables(
    analysis_path: Path,
    *,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    timestamps_by_epoch: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], str, dict[str, dict[str, int]]]:
    """Build merged fixed ripple-start-aligned interval rows for each epoch."""
    ripple_tables, source = load_ripple_tables(analysis_path)

    intervals_by_epoch: dict[str, pd.DataFrame] = {}
    metadata_by_epoch: dict[str, dict[str, int]] = {}
    for epoch, ripple_table in ripple_tables.items():
        epoch_key = str(epoch)
        if epoch_key not in timestamps_by_epoch:
            raise ValueError(
                f"Could not find ephys timestamps for ripple epoch {epoch_key!r}."
            )

        epoch_timestamps = np.asarray(timestamps_by_epoch[epoch_key], dtype=float).ravel()
        if epoch_timestamps.size == 0:
            raise ValueError(f"Epoch {epoch_key!r} has no ephys timestamps.")

        ripple_starts = _get_ripple_anchor_starts(ripple_table)
        window_starts = ripple_starts + float(ripple_window_offset_s)
        window_ends = ripple_starts + float(ripple_window_offset_s) + float(ripple_window_s)

        epoch_start = float(epoch_timestamps[0])
        epoch_end = float(epoch_timestamps[-1])
        clipped_starts = np.maximum(window_starts, epoch_start)
        clipped_ends = np.minimum(window_ends, epoch_end)
        clipped_rows = pd.DataFrame({"start": clipped_starts, "end": clipped_ends})
        clipped_rows = clipped_rows.loc[clipped_rows["end"] > clipped_rows["start"]].reset_index(
            drop=True
        )
        merged_rows = merge_interval_rows(clipped_rows)
        intervals_by_epoch[epoch_key] = merged_rows
        metadata_by_epoch[epoch_key] = {
            "original_ripple_count": int(len(ripple_table)),
            "clipped_window_count": int(len(clipped_rows)),
            "merged_interval_count": int(len(merged_rows)),
        }

    return intervals_by_epoch, source, metadata_by_epoch


def resolve_epoch_groups(
    saved_epoch_tags: list[str],
    intervals_by_epoch: dict[str, pd.DataFrame],
    requested_epochs: list[str] | None,
) -> list[tuple[str, list[str]]]:
    """Return ordered epoch groups for separate or pooled analysis."""
    available_epochs = [epoch for epoch in saved_epoch_tags if epoch in intervals_by_epoch]
    if not available_epochs:
        raise ValueError("No saved intervals were found for the selected state.")
    if requested_epochs is None:
        return [(epoch, [epoch]) for epoch in available_epochs]

    selected_epochs = list(dict.fromkeys(str(epoch) for epoch in requested_epochs))
    selected_epoch_tokens = [epoch.lower() for epoch in selected_epochs]
    if POOLED_EPOCH_SENTINEL in selected_epoch_tokens:
        if len(selected_epochs) != 1:
            raise ValueError(
                f"When using --epochs {POOLED_EPOCH_SENTINEL!r}, do not pass additional epoch labels."
            )
        return [(POOLED_EPOCH_SENTINEL, available_epochs)]

    missing_epochs = [epoch for epoch in selected_epochs if epoch not in intervals_by_epoch]
    if missing_epochs:
        raise ValueError(
            "Requested epochs do not have saved interval output for the selected state: "
            f"{missing_epochs!r}."
        )
    ordered_epochs = [epoch for epoch in saved_epoch_tags if epoch in selected_epochs]
    return [(epoch, [epoch]) for epoch in ordered_epochs]


def build_state_intervalset(
    intervals_by_epoch: dict[str, pd.DataFrame],
    selected_epochs: list[str],
) -> Any:
    """Return one combined IntervalSet across all selected epochs."""
    import pynapple as nap

    start_chunks: list[np.ndarray] = []
    end_chunks: list[np.ndarray] = []
    for epoch in selected_epochs:
        interval_rows = intervals_by_epoch.get(epoch)
        if interval_rows is None or interval_rows.empty:
            continue
        start_chunks.append(interval_rows["start"].to_numpy(dtype=float))
        end_chunks.append(interval_rows["end"].to_numpy(dtype=float))

    if not start_chunks:
        return nap.IntervalSet(
            start=np.array([], dtype=float),
            end=np.array([], dtype=float),
            time_units="s",
        )

    starts = np.concatenate(start_chunks).astype(float, copy=False)
    ends = np.concatenate(end_chunks).astype(float, copy=False)
    order = np.argsort(starts)
    return nap.IntervalSet(
        start=starts[order],
        end=ends[order],
        time_units="s",
    )


def get_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned interval start/end arrays."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError("IntervalSet start/end arrays do not align.")
    return starts, ends


def count_spikes_in_intervals(spike_times: np.ndarray, intervals: Any) -> int:
    """Return the number of spikes that fall inside one IntervalSet."""
    starts, ends = get_interval_bounds(intervals)
    if starts.size == 0:
        return 0

    spike_array = np.asarray(spike_times, dtype=float).ravel()
    if spike_array.size == 0:
        return 0

    spike_count = 0
    for start, end in zip(starts, ends, strict=True):
        left = int(np.searchsorted(spike_array, start, side="left"))
        right = int(np.searchsorted(spike_array, end, side="right"))
        spike_count += right - left
    return int(spike_count)


def build_unit_spike_count_table(
    spikes: Any,
    intervals: Any,
    *,
    region: str,
    min_state_spikes: int,
) -> pd.DataFrame:
    """Return one per-unit spike-count filter table for the selected state."""
    unit_ids = np.asarray(list(spikes.keys()))
    spike_counts = np.asarray(
        [count_spikes_in_intervals(np.asarray(spikes[unit_id].t, dtype=float), intervals) for unit_id in unit_ids],
        dtype=int,
    )
    keep_unit = spike_counts >= int(min_state_spikes)
    return pd.DataFrame(
        {
            "region": region,
            "unit_id": unit_ids,
            "state_spike_count": spike_counts,
            "passes_state_spike_count": keep_unit,
            "keep_unit": keep_unit,
        }
    )


def subset_spikes_by_unit_ids(spikes: Any, unit_ids: np.ndarray) -> Any:
    """Return one TsGroup-like subset containing only the requested units."""
    import pynapple as nap

    return nap.TsGroup(
        {unit_id: spikes[unit_id] for unit_id in unit_ids.tolist()},
        time_units="s",
    )


def compute_xcorr(
    *,
    ca1_spikes: Any,
    v1_spikes: Any,
    intervals: Any,
    bin_size_s: float,
    max_lag_s: float,
) -> pd.DataFrame:
    """Compute CA1-V1 cross-correlograms with pynapple target-rate normalization."""
    import pynapple as nap

    return nap.compute_crosscorrelogram(
        group=(ca1_spikes, v1_spikes),
        binsize=float(bin_size_s),
        windowsize=float(max_lag_s),
        time_units="s",
        norm=True,
        ep=intervals,
    )


def xcorr_frame_to_tensor(
    xcorr: pd.DataFrame,
    ca1_unit_ids: np.ndarray,
    v1_unit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a pynapple xcorr frame into a `(ca1, v1, lag)` tensor."""
    lag_times = xcorr.index.to_numpy(dtype=float)
    tensor = np.full(
        (ca1_unit_ids.size, v1_unit_ids.size, lag_times.size),
        np.nan,
        dtype=float,
    )
    ca1_index = {unit_id: idx for idx, unit_id in enumerate(ca1_unit_ids.tolist())}
    v1_index = {unit_id: idx for idx, unit_id in enumerate(v1_unit_ids.tolist())}
    for column in xcorr.columns:
        if not isinstance(column, tuple) or len(column) != 2:
            raise ValueError(
                "Expected xcorr columns to be `(ca1_unit_id, v1_unit_id)` tuples. "
                f"Found {column!r}."
            )
        ca1_unit_id, v1_unit_id = column
        if ca1_unit_id not in ca1_index or v1_unit_id not in v1_index:
            continue
        tensor[ca1_index[ca1_unit_id], v1_index[v1_unit_id], :] = xcorr[column].to_numpy(
            dtype=float
        )
    return lag_times, tensor


def summarize_pair_curve(
    *,
    xcorr_curve: np.ndarray,
    lag_times: np.ndarray,
    extremum_half_width_bins: int,
) -> dict[str, Any]:
    """Return one pair's strongest feature from normalized xcorr."""
    xcorr_array = np.asarray(xcorr_curve, dtype=float).reshape(-1)
    lag_array = np.asarray(lag_times, dtype=float).reshape(-1)

    finite_indices = np.flatnonzero(np.isfinite(xcorr_array))
    if finite_indices.size == 0:
        return {
            "status": PAIR_STATUS_NO_FINITE_BINS,
            "peak_lag_s": np.nan,
            "peak_norm_xcorr": np.nan,
        }

    finite_values = xcorr_array[finite_indices]
    local_extremum_index = int(np.nanargmax(finite_values))
    peak_index = int(finite_indices[local_extremum_index])
    neighborhood_start = max(0, peak_index - int(extremum_half_width_bins))
    neighborhood_end = min(len(xcorr_array), peak_index + int(extremum_half_width_bins) + 1)
    neighborhood = xcorr_array[neighborhood_start:neighborhood_end]
    peak_norm_xcorr = float(np.nanmean(neighborhood))
    return {
        "status": PAIR_STATUS_VALID,
        "peak_lag_s": float(lag_array[peak_index]),
        "peak_norm_xcorr": peak_norm_xcorr,
    }


def build_pair_summary_table(
    *,
    xcorr: np.ndarray,
    ca1_unit_ids: np.ndarray,
    v1_unit_ids: np.ndarray,
    ca1_spike_counts: np.ndarray,
    v1_spike_counts: np.ndarray,
    lag_times: np.ndarray,
    extremum_half_width_bins: int,
) -> pd.DataFrame:
    """Return one summary row per retained CA1-V1 pair."""
    rows: list[dict[str, Any]] = []
    for ca1_index, ca1_unit_id in enumerate(ca1_unit_ids.tolist()):
        for v1_index, v1_unit_id in enumerate(v1_unit_ids.tolist()):
            summary = summarize_pair_curve(
                xcorr_curve=xcorr[ca1_index, v1_index],
                lag_times=lag_times,
                extremum_half_width_bins=extremum_half_width_bins,
            )
            rows.append(
                {
                    "ca1_unit_id": ca1_unit_id,
                    "v1_unit_id": v1_unit_id,
                    "n_ca1_state_spikes": int(ca1_spike_counts[ca1_index]),
                    "n_v1_state_spikes": int(v1_spike_counts[v1_index]),
                    "peak_lag_s": summary["peak_lag_s"],
                    "peak_norm_xcorr": summary["peak_norm_xcorr"],
                    "status": summary["status"],
                }
            )
    return pd.DataFrame(rows)


def order_v1_partners_for_ca1(pair_summary: pd.DataFrame) -> np.ndarray:
    """Return ordered summary-row indices for one CA1 unit's valid V1 partners."""
    valid_rows = pair_summary.loc[pair_summary["status"] == PAIR_STATUS_VALID].copy()
    if valid_rows.empty:
        return np.array([], dtype=int)
    valid_rows = valid_rows.sort_values(
        by=["peak_lag_s", "peak_norm_xcorr"],
        ascending=[True, False],
        kind="stable",
    )
    return valid_rows.index.to_numpy(dtype=int)


def order_ca1_units_by_best_partner(pair_summary: pd.DataFrame) -> np.ndarray:
    """Return CA1 unit ids sorted by strongest valid partner normalized xcorr."""
    valid_rows = pair_summary.loc[pair_summary["status"] == PAIR_STATUS_VALID].copy()
    if valid_rows.empty:
        return np.array([], dtype=pair_summary["ca1_unit_id"].dtype)
    ordered = (
        valid_rows.groupby("ca1_unit_id", sort=False)["peak_norm_xcorr"]
        .max()
        .sort_values(ascending=False, kind="stable")
    )
    return ordered.index.to_numpy()


def save_xcorr_dataset(
    *,
    output_path: Path,
    xcorr: np.ndarray,
    lag_times: np.ndarray,
    ca1_unit_ids: np.ndarray,
    v1_unit_ids: np.ndarray,
    attrs: dict[str, Any],
) -> Path:
    """Write the full CA1-V1-by-lag tensors as one NetCDF dataset."""
    xr = require_xarray()

    dataset = xr.Dataset(
        data_vars={
            "xcorr": (
                ("ca1_unit", "v1_unit", "lag_s"),
                np.asarray(xcorr, dtype=np.float32),
            ),
        },
        coords={
            "ca1_unit": np.asarray(ca1_unit_ids),
            "v1_unit": np.asarray(v1_unit_ids),
            "lag_s": np.asarray(lag_times, dtype=np.float32),
        },
        attrs={str(key): value for key, value in attrs.items()},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    return output_path


def save_session_overview_figure(
    *,
    pair_summary: pd.DataFrame,
    out_path: Path,
    animal_name: str,
    date: str,
    state: str,
    show: bool = False,
) -> Path | None:
    """Save one session-level histogram/scatter overview for valid pairs."""
    import matplotlib.pyplot as plt

    valid_rows = pair_summary.loc[pair_summary["status"] == PAIR_STATUS_VALID].copy()
    if valid_rows.empty:
        return None

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        constrained_layout=True,
    )

    axes[0].hist(
        valid_rows["peak_norm_xcorr"].to_numpy(dtype=float),
        bins=30,
        color="#b2182b",
        alpha=0.8,
    )
    axes[0].set_title("Peak normalized xcorr")
    axes[0].set_xlabel("peak normalized xcorr")
    axes[0].set_ylabel("pair count")

    axes[1].scatter(
        valid_rows["peak_lag_s"].to_numpy(dtype=float),
        valid_rows["peak_norm_xcorr"].to_numpy(dtype=float),
        c="#2166ac",
        s=12,
        alpha=0.7,
    )
    axes[1].set_title("Lag vs peak normalized xcorr")
    axes[1].set_xlabel("peak lag (s)")
    axes[1].set_ylabel("peak normalized xcorr")

    ca1_count = valid_rows["ca1_unit_id"].nunique()
    v1_count = valid_rows["v1_unit_id"].nunique()
    pair_count = len(valid_rows)
    fig.suptitle(
        f"{animal_name} {date} {state} xcorr overview\n"
        f"{ca1_count} CA1 units, {v1_count} V1 units, {pair_count} valid pairs"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _append_path_parts(base_path: Path, parts: list[str]) -> Path:
    """Append ordered string path parts onto one base path."""
    out_path = base_path
    for part in parts:
        out_path = out_path / part
    return out_path


def format_xcorr_settings_suffix(*, max_lag_s: float, bin_size_s: float) -> str:
    """Return a filesystem-friendly suffix for one xcorr lag/bin setting."""
    return (
        f"ml_{_format_window_suffix_value(max_lag_s)}s_"
        f"bs_{_format_window_suffix_value(bin_size_s)}s"
    )


def get_state_output_parts(
    state: str,
    *,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> list[str]:
    """Return path parts encoding the state and optional ripple window setup."""
    parts = [state]
    if state == "ripple" and ripple_window_s is not None:
        parts.append(
            format_ripple_window_suffix(
                float(ripple_window_s),
                ripple_window_offset_s=float(ripple_window_offset_s),
            )
        )
    return parts


def get_output_dir(
    analysis_path: Path,
    state: str,
    epoch_group_label: str,
    *,
    max_lag_s: float,
    bin_size_s: float,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> Path:
    """Return the structured state- and epoch-group-specific output directory."""
    return _append_path_parts(
        analysis_path / "xcorr" / DEFAULT_OUTPUT_DIRNAME,
        get_state_output_parts(
            state,
            ripple_window_s=ripple_window_s,
            ripple_window_offset_s=ripple_window_offset_s,
        )
        + [format_xcorr_settings_suffix(max_lag_s=max_lag_s, bin_size_s=bin_size_s)]
        + [epoch_group_label],
    )


def get_figure_dir(
    analysis_path: Path,
    state: str,
    epoch_group_label: str,
    *,
    max_lag_s: float,
    bin_size_s: float,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> Path:
    """Return the structured state- and epoch-group-specific figure directory."""
    return _append_path_parts(
        analysis_path / "figs" / "xcorr" / DEFAULT_OUTPUT_DIRNAME,
        get_state_output_parts(
            state,
            ripple_window_s=ripple_window_s,
            ripple_window_offset_s=ripple_window_offset_s,
        )
        + [format_xcorr_settings_suffix(max_lag_s=max_lag_s, bin_size_s=bin_size_s)]
        + [epoch_group_label],
    )


def summarize_counts(pair_summary: pd.DataFrame) -> dict[str, int]:
    """Return retained unit and valid-pair counts for one summary table."""
    valid_rows = pair_summary.loc[pair_summary["status"] == PAIR_STATUS_VALID]
    return {
        "n_ca1_units": int(valid_rows["ca1_unit_id"].nunique()),
        "n_v1_units": int(valid_rows["v1_unit_id"].nunique()),
        "n_valid_pairs": int(len(valid_rows)),
    }


def _format_seconds(value_s: float) -> str:
    """Return one concise human-readable seconds string."""
    return f"{float(value_s):.3f}s"


def _screen_xcorr_for_epoch_group(
    *,
    analysis_path: Path,
    animal_name: str,
    date: str,
    state: str,
    interval_source: str,
    intervals_by_epoch: dict[str, pd.DataFrame],
    interval_metadata_by_epoch: dict[str, dict[str, int]] | None,
    timestamps_source: str,
    spikes_by_region: dict[str, Any],
    epoch_group_label: str,
    selected_epochs: list[str],
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    ripple_window_s: float | None = None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    min_state_spikes: int = DEFAULT_MIN_STATE_SPIKES,
    extremum_half_width_bins: int = DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
    display_vmax: float = DEFAULT_DISPLAY_VMAX,
    show: bool = False,
) -> dict[str, Any]:
    """Run the xcorr screening workflow for one epoch group."""
    print(
        f"Analyzing epoch group {epoch_group_label!r} for state={state} "
        f"using epochs {selected_epochs!r}."
    )
    for epoch in selected_epochs:
        interval_rows = intervals_by_epoch.get(epoch, pd.DataFrame(columns=["start", "end"]))
        print(f"  epoch {epoch}: {len(interval_rows)} intervals")
        if interval_metadata_by_epoch is not None and epoch in interval_metadata_by_epoch:
            epoch_metadata = interval_metadata_by_epoch[epoch]
            print(
                "    fixed ripple windows: "
                f"{epoch_metadata['original_ripple_count']} original ripples, "
                f"{epoch_metadata['clipped_window_count']} clipped windows, "
                f"{epoch_metadata['merged_interval_count']} merged intervals"
            )
    state_intervals = build_state_intervalset(intervals_by_epoch, selected_epochs)
    print(
        "State support built: "
        f"{len(state_intervals)} merged intervals spanning "
        f"{_format_seconds(state_intervals.tot_length())} total."
    )

    ca1_filter = build_unit_spike_count_table(
        spikes_by_region["ca1"],
        state_intervals,
        region="ca1",
        min_state_spikes=min_state_spikes,
    )
    v1_filter = build_unit_spike_count_table(
        spikes_by_region["v1"],
        state_intervals,
        region="v1",
        min_state_spikes=min_state_spikes,
    )
    kept_ca1_units = ca1_filter.loc[ca1_filter["keep_unit"], "unit_id"].to_numpy()
    kept_v1_units = v1_filter.loc[v1_filter["keep_unit"], "unit_id"].to_numpy()
    print(
        "Unit filtering complete: "
        f"kept {kept_ca1_units.size}/{len(ca1_filter)} CA1 units and "
        f"{kept_v1_units.size}/{len(v1_filter)} V1 units with at least "
        f"{min_state_spikes} spikes in the selected state."
    )

    output_dir = get_output_dir(
        analysis_path,
        state,
        epoch_group_label,
        max_lag_s=max_lag_s,
        bin_size_s=bin_size_s,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = get_figure_dir(
        analysis_path,
        state,
        epoch_group_label,
        max_lag_s=max_lag_s,
        bin_size_s=bin_size_s,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    dataset_path = output_dir / DATASET_FILENAME
    overview_path = figure_dir / OVERVIEW_FIGURE_FILENAME
    heatmap_dir = figure_dir / CA1_HEATMAP_DIRNAME
    print(f"Data outputs will be saved under {output_dir}.")
    print(f"Figure outputs will be saved under {figure_dir}.")

    outputs: dict[str, Any] = {
        "output_dir": output_dir,
        "figure_dir": figure_dir,
        "state_interval_source": interval_source,
        "timestamps_ephys_all_source": timestamps_source,
        "selected_epochs": selected_epochs,
        "state_interval_count": int(len(state_intervals)),
        "state_total_duration_s": float(state_intervals.tot_length()),
        "ripple_window_s": None if ripple_window_s is None else float(ripple_window_s),
        "ripple_window_offset_s": float(ripple_window_offset_s),
        "ca1_unit_filter_path": output_dir / "ca1_unit_filter.parquet",
        "v1_unit_filter_path": output_dir / "v1_unit_filter.parquet",
        "xcorr_summary_path": summary_path,
        "xcorr_dataset_path": None,
        "overview_figure_path": None,
        "ca1_heatmap_dir": heatmap_dir,
        "ca1_heatmap_paths": [],
    }
    if interval_metadata_by_epoch is not None:
        outputs["interval_metadata_by_epoch"] = {
            epoch: interval_metadata_by_epoch[epoch]
            for epoch in selected_epochs
            if epoch in interval_metadata_by_epoch
        }

    ca1_filter.to_parquet(outputs["ca1_unit_filter_path"], index=False)
    v1_filter.to_parquet(outputs["v1_unit_filter_path"], index=False)

    if kept_ca1_units.size == 0 or kept_v1_units.size == 0:
        print("No retained units remain after filtering; writing empty summary outputs.")
        empty_summary = pd.DataFrame(
            columns=[
                "ca1_unit_id",
                "v1_unit_id",
                "n_ca1_state_spikes",
                "n_v1_state_spikes",
                "peak_lag_s",
                "peak_norm_xcorr",
                "status",
            ]
        )
        empty_summary.to_parquet(summary_path, index=False)
        outputs.update(
            {
                "message": (
                    "No units passed the state spike-count threshold for the selected state."
                ),
                "counts": {
                    "n_ca1_units": int(kept_ca1_units.size),
                    "n_v1_units": int(kept_v1_units.size),
                    "n_valid_pairs": 0,
                },
            }
        )
        log_path = write_run_log(
            analysis_path=analysis_path,
            script_name="v1ca1.xcorr.screen_xcorr",
            parameters={
                "animal_name": animal_name,
                "date": date,
                "state": state,
                "epoch_group_label": epoch_group_label,
                "selected_epochs": selected_epochs,
                "bin_size_s": bin_size_s,
                "max_lag_s": max_lag_s,
                "ripple_window_s": None if ripple_window_s is None else float(ripple_window_s),
                "ripple_window_offset_s": float(ripple_window_offset_s),
                "min_state_spikes": min_state_spikes,
                "extremum_half_width_bins": extremum_half_width_bins,
                "display_vmax": display_vmax,
            },
            outputs=outputs,
        )
        outputs["log_path"] = log_path
        return outputs

    filtered_ca1_spikes = subset_spikes_by_unit_ids(spikes_by_region["ca1"], kept_ca1_units)
    filtered_v1_spikes = subset_spikes_by_unit_ids(spikes_by_region["v1"], kept_v1_units)
    print(
        "Computing cross-correlograms for "
        f"{kept_ca1_units.size * kept_v1_units.size} CA1-V1 pairs "
        f"with bin_size_s={bin_size_s} and max_lag_s={max_lag_s}."
    )
    xcorr = compute_xcorr(
        ca1_spikes=filtered_ca1_spikes,
        v1_spikes=filtered_v1_spikes,
        intervals=state_intervals,
        bin_size_s=bin_size_s,
        max_lag_s=max_lag_s,
    )
    lag_times, xcorr_values = xcorr_frame_to_tensor(xcorr, kept_ca1_units, kept_v1_units)
    print(f"Computed normalized xcorr tensor with shape {xcorr_values.shape}.")
    print("Summarizing the strongest normalized xcorr feature across the full lag range.")
    print(
        "Heatmaps will use clipped normalized xcorr values from 0 to "
        f"{display_vmax:.3f}. A value of 1 corresponds to baseline firing rate."
    )

    pair_summary = build_pair_summary_table(
        xcorr=xcorr_values,
        ca1_unit_ids=kept_ca1_units,
        v1_unit_ids=kept_v1_units,
        ca1_spike_counts=ca1_filter.loc[ca1_filter["keep_unit"], "state_spike_count"].to_numpy(
            dtype=int
        ),
        v1_spike_counts=v1_filter.loc[v1_filter["keep_unit"], "state_spike_count"].to_numpy(
            dtype=int
        ),
        lag_times=lag_times,
        extremum_half_width_bins=extremum_half_width_bins,
    )
    pair_summary.to_parquet(summary_path, index=False)
    print(f"Saved xcorr summary table to {summary_path}.")

    valid_pair_summary = pair_summary.loc[pair_summary["status"] == PAIR_STATUS_VALID]
    if not valid_pair_summary.empty:
        dataset_attrs = {
            "animal_name": animal_name,
            "date": date,
            "state": state,
            "epoch_group_label": epoch_group_label,
            "bin_size_s": float(bin_size_s),
            "max_lag_s": float(max_lag_s),
            "ripple_window_s": (
                np.nan if ripple_window_s is None else float(ripple_window_s)
            ),
            "ripple_window_offset_s": float(ripple_window_offset_s),
            "min_state_spikes": int(min_state_spikes),
            "extremum_half_width_bins": int(extremum_half_width_bins),
            "display_vmax": float(display_vmax),
            "selected_epochs_json": json.dumps(selected_epochs),
            "state_interval_source": interval_source,
        }
        save_xcorr_dataset(
            output_path=dataset_path,
            xcorr=xcorr_values,
            lag_times=lag_times,
            ca1_unit_ids=kept_ca1_units,
            v1_unit_ids=kept_v1_units,
            attrs=dataset_attrs,
        )
        outputs["xcorr_dataset_path"] = dataset_path
        print(f"Saved xcorr dataset to {dataset_path}.")
        overview_figure_path = save_session_overview_figure(
            pair_summary=pair_summary,
            out_path=overview_path,
            animal_name=animal_name,
            date=date,
            state=state,
            show=show,
        )
        outputs["overview_figure_path"] = overview_figure_path
        if overview_figure_path is not None:
            print(f"Saved overview figure to {overview_figure_path}.")

        v1_unit_order = kept_v1_units
        ca1_unit_order = order_ca1_units_by_best_partner(pair_summary)
        if ca1_unit_order.size:
            v1_index_lookup = {unit_id: idx for idx, unit_id in enumerate(v1_unit_order.tolist())}
            ca1_index_lookup = {unit_id: idx for idx, unit_id in enumerate(kept_ca1_units.tolist())}
            print(f"Saving {ca1_unit_order.size} CA1-centered heatmaps.")
            for rank, ca1_unit_id in enumerate(ca1_unit_order.tolist(), start=1):
                print(
                    f"  heatmap {rank}/{ca1_unit_order.size}: "
                    f"CA1 unit {ca1_unit_id}"
                )
                ca1_rows = pair_summary.loc[pair_summary["ca1_unit_id"] == ca1_unit_id]
                ordered_summary_indices = order_v1_partners_for_ca1(ca1_rows)
                if ordered_summary_indices.size == 0:
                    print("    skipping because no valid V1 partners remain after ranking.")
                    continue
                ordered_rows = pair_summary.loc[ordered_summary_indices]
                heatmap_matrix = np.vstack(
                    [
                        xcorr_values[
                            ca1_index_lookup[ca1_unit_id],
                            v1_index_lookup[v1_unit_id],
                            :,
                        ]
                        for v1_unit_id in ordered_rows["v1_unit_id"].tolist()
                    ]
                )
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(
                    figsize=(8, max(3, 0.2 * heatmap_matrix.shape[0] + 1.5)),
                    constrained_layout=True,
                )
                image = ax.imshow(
                    np.clip(heatmap_matrix, 0.0, display_vmax),
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    extent=[lag_times[0], lag_times[-1], heatmap_matrix.shape[0], 0],
                    vmin=0.0,
                    vmax=display_vmax,
                )
                ax.axvline(0.0, color="black", linewidth=1.0)
                ax.set_xlabel("lag (s)")
                ax.set_ylabel("V1 partners")
                ax.set_yticks([])
                ax.set_title(f"{animal_name} {date} {state} CA1 unit {ca1_unit_id}")
                colorbar = fig.colorbar(image, ax=ax)
                colorbar.set_label("normalized xcorr")
                heatmap_dir.mkdir(parents=True, exist_ok=True)
                heatmap_path = heatmap_dir / f"{rank:03d}_ca1_unit_{ca1_unit_id}.png"
                fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close(fig)
                outputs["ca1_heatmap_paths"].append(heatmap_path)
                print(f"    saved to {heatmap_path}")
    else:
        print("No valid pairs had finite normalized xcorr structure; skipping dataset and figures.")

    outputs["counts"] = summarize_counts(pair_summary)
    print(
        "Final counts: "
        f"{outputs['counts']['n_ca1_units']} CA1 units, "
        f"{outputs['counts']['n_v1_units']} V1 units, "
        f"{outputs['counts']['n_valid_pairs']} valid pairs."
    )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.xcorr.screen_xcorr",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "state": state,
            "epoch_group_label": epoch_group_label,
            "selected_epochs": selected_epochs,
            "bin_size_s": bin_size_s,
            "max_lag_s": max_lag_s,
            "ripple_window_s": None if ripple_window_s is None else float(ripple_window_s),
            "ripple_window_offset_s": float(ripple_window_offset_s),
            "min_state_spikes": min_state_spikes,
            "extremum_half_width_bins": extremum_half_width_bins,
            "display_vmax": display_vmax,
        },
        outputs=outputs,
    )
    outputs["log_path"] = log_path
    return outputs


def screen_xcorr_for_session(
    *,
    animal_name: str,
    date: str,
    state: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    epochs: list[str] | None = None,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    ripple_window_s: float | None = None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    min_state_spikes: int = DEFAULT_MIN_STATE_SPIKES,
    extremum_half_width_bins: int = DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
    display_vmax: float = DEFAULT_DISPLAY_VMAX,
    show: bool = False,
) -> dict[str, Any]:
    """Run xcorr screening for one session using epoch-by-epoch or pooled mode."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    print(f"Processing {animal_name} {date} state={state}.")
    print(
        "Analysis settings: "
        f"bin_size_s={bin_size_s}, "
        f"max_lag_s={max_lag_s}, "
        f"min_state_spikes={min_state_spikes}, "
        f"extremum_half_width_bins={extremum_half_width_bins}, "
        f"display_vmax={display_vmax}."
    )
    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    interval_metadata_by_epoch: dict[str, dict[str, int]] | None = None
    if state == "ripple" and ripple_window_s is not None:
        _ephys_epoch_tags, timestamps_by_epoch, timestamps_by_epoch_source = load_ephys_timestamps_by_epoch(
            analysis_path
        )
        print(
            "Loaded per-epoch ephys timestamps from "
            f"{timestamps_by_epoch_source} to build fixed ripple windows."
        )
        intervals_by_epoch, interval_source, interval_metadata_by_epoch = build_fixed_ripple_window_tables(
            analysis_path,
            ripple_window_s=float(ripple_window_s),
            ripple_window_offset_s=float(ripple_window_offset_s),
            timestamps_by_epoch=timestamps_by_epoch,
        )
        ripple_window_suffix = format_ripple_window_suffix(
            float(ripple_window_s),
            ripple_window_offset_s=float(ripple_window_offset_s),
        )
        print(
            "Using fixed ripple windows "
            f"{ripple_window_suffix}: intervals are built from "
            "[ripple_start + offset, ripple_start + offset + ripple_window_s] "
            "and merged after clipping to epoch bounds."
        )
        interval_source = f"fixed_ripple_windows[{interval_source}]"
    else:
        intervals_by_epoch, interval_source = load_state_interval_tables(analysis_path, state)
        if state == "ripple":
            print("Using detected ripple intervals directly (no fixed ripple window override).")
    epoch_groups = resolve_epoch_groups(epoch_tags, intervals_by_epoch, epochs)
    analysis_mode = (
        "pooled"
        if len(epoch_groups) == 1 and epoch_groups[0][0] == POOLED_EPOCH_SENTINEL
        else "epoch_by_epoch"
    )
    print(
        f"Loaded {state} intervals from {interval_source}. "
        f"Analysis mode: {analysis_mode}. "
        f"Epoch groups: {[label for label, _ in epoch_groups]!r}"
    )

    timestamps_ephys_all, timestamps_source = load_ephys_timestamps_all(analysis_path)
    print(f"Loaded concatenated ephys timestamps from {timestamps_source}.")
    spikes_by_region = load_spikes_by_region(
        analysis_path=analysis_path,
        timestamps_ephys_all=timestamps_ephys_all,
        regions=REGIONS,
    )
    print("Loaded spike trains for CA1 and V1.")

    group_outputs: dict[str, dict[str, Any]] = {}
    for group_index, (epoch_group_label, selected_epochs) in enumerate(epoch_groups, start=1):
        print(
            f"Epoch group {group_index}/{len(epoch_groups)}: "
            f"{epoch_group_label!r}"
        )
        group_outputs[epoch_group_label] = _screen_xcorr_for_epoch_group(
            analysis_path=analysis_path,
            animal_name=animal_name,
            date=date,
            state=state,
            interval_source=interval_source,
            intervals_by_epoch=intervals_by_epoch,
            interval_metadata_by_epoch=interval_metadata_by_epoch,
            timestamps_source=timestamps_source,
            spikes_by_region=spikes_by_region,
            epoch_group_label=epoch_group_label,
            selected_epochs=selected_epochs,
            bin_size_s=bin_size_s,
            max_lag_s=max_lag_s,
            ripple_window_s=ripple_window_s,
            ripple_window_offset_s=ripple_window_offset_s,
            min_state_spikes=min_state_spikes,
            extremum_half_width_bins=extremum_half_width_bins,
            display_vmax=display_vmax,
            show=show,
        )

    return {
        "analysis_mode": analysis_mode,
        "epoch_groups": [label for label, _ in epoch_groups],
        "group_outputs": group_outputs,
    }


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for xcorr screening."""
    parser = argparse.ArgumentParser(
        description="Screen CA1-V1 cross-correlogram structure for one state"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--state",
        required=True,
        choices=STATE_CHOICES,
        help="Behavioral/state interval to restrict spikes to.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help=(
            "Optional epoch selection. By default, analyze each epoch present in the "
            "selected state's saved interval table separately. Pass one or more epoch "
            f"labels to analyze only those epochs separately, or pass {POOLED_EPOCH_SENTINEL!r} "
            "to pool all available epochs into one analysis."
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Cross-correlogram bin size in seconds. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--max-lag-s",
        type=float,
        default=DEFAULT_MAX_LAG_S,
        help=f"Maximum absolute lag in seconds. Default: {DEFAULT_MAX_LAG_S}",
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=None,
        help=(
            "Optional fixed ripple window length in seconds. When set for "
            "--state ripple, both CA1 and V1 spikes are restricted to "
            "[ripple_start + ripple_window_offset_s, ripple_start + "
            "ripple_window_offset_s + ripple_window_s]. Default: use the "
            "detected ripple intervals."
        ),
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        help=(
            "Offset in seconds applied to the optional fixed ripple window "
            "relative to ripple start. "
            f"Default: {DEFAULT_RIPPLE_WINDOW_OFFSET_S}"
        ),
    )
    parser.add_argument(
        "--min-state-spikes",
        type=int,
        default=DEFAULT_MIN_STATE_SPIKES,
        help=(
            "Minimum number of spikes within the selected state required to keep one "
            f"unit. Default: {DEFAULT_MIN_STATE_SPIKES}"
        ),
    )
    parser.add_argument(
        "--extremum-half-width-bins",
        type=int,
        default=DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
        help=(
            "Half-width in lag bins used to average around the strongest "
            f"extremum. Default: {DEFAULT_EXTREMUM_HALF_WIDTH_BINS}"
        ),
    )
    parser.add_argument(
        "--display-vmax",
        "--display-zlim",
        dest="display_vmax",
        type=float,
        default=DEFAULT_DISPLAY_VMAX,
        help=(
            "Upper color limit used for normalized-xcorr heatmap display. "
            f"Default: {DEFAULT_DISPLAY_VMAX}"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the xcorr screening CLI."""
    args = parse_arguments(argv)
    validate_arguments(args)
    outputs = screen_xcorr_for_session(
        animal_name=args.animal_name,
        date=args.date,
        state=args.state,
        data_root=args.data_root,
        epochs=args.epochs,
        bin_size_s=args.bin_size_s,
        max_lag_s=args.max_lag_s,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        min_state_spikes=args.min_state_spikes,
        extremum_half_width_bins=args.extremum_half_width_bins,
        display_vmax=args.display_vmax,
        show=args.show,
    )
    if outputs.get("message"):
        print(outputs["message"])
    print(
        f"Completed {len(outputs['epoch_groups'])} xcorr analyses in "
        f"{outputs['analysis_mode']} mode."
    )
    for epoch_group_label in outputs["epoch_groups"]:
        group_output = outputs["group_outputs"][epoch_group_label]
        print(
            f"  {epoch_group_label}: summary={group_output['xcorr_summary_path']}"
        )


if __name__ == "__main__":
    main()
