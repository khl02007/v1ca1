from __future__ import annotations

"""Fit ripple-triggered population GLMs for one session.

This CLI fits the modern CA1-to-V1 ripple population GLM workflow for one
session, one epoch at a time. Each ripple window is converted into one sample
whose predictors are CA1 unit spike counts and whose targets are V1 unit spike
counts. The model is a ridge-regularized Poisson population GLM trained and
evaluated on ripple windows with contiguous cross-validation folds.

Performance is summarized per V1 unit with pseudo-R^2, mean absolute error
(MAE), deviance explained, and bits/spike. Ripple metrics also include an
empirical shuffle null built by refitting after independently shuffling each
V1 unit's ripple responses, which yields per-unit shuffle means, shuffle
standard deviations, and one-sided p-values.

Successful fits are saved under the session analysis directory as one
NetCDF-backed xarray dataset per epoch and ridge strength in `ripple_glm/`.
Each dataset stores the raw ripple fold arrays, ripple shuffle arrays,
per-unit ripple summary variables, unit IDs, full-fit CA1 coefficient arrays
with coefficient-aligned CA1 unit IDs, and fit metadata. The script also
writes one summary figure per epoch and ridge strength in `figs/ripple_glm/`.
That figure contains one subplot each for pseudo-R^2, MAE, deviance explained,
and bits/spike, with ripple effect size plotted versus `-log10(shuffle p)`.
A JSON run log is written under `v1ca1_log/`.

When this script instantiates `nemos.glm.PopulationGLM`, it selects the solver
backend from the installed `nemos` version. For `nemos<=0.2.5`, it keeps the
legacy `solver_name="LBFGS"` behavior. For `nemos>0.2.5`, it uses
`solver_name="LBFGS[jaxopt]"` because the default solver backend changed in
`0.2.6` and the ripple GLM optimizer hyperparameters are tuned for the jaxopt
backend.
"""

import argparse
import gc
import json
import os
import pickle
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from packaging.version import InvalidVersion, Version
from scipy.special import gammaln
from sklearn.model_selection import KFold

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
    load_spikes_by_region,
)

if TYPE_CHECKING:
    import pynapple as nap
    import xarray as xr


TARGET_REGION = "v1"
SOURCE_REGION = "ca1"

DEFAULT_RIPPLE_WINDOW_S = 0.2
DEFAULT_RIPPLE_WINDOW_OFFSET_S = 0.0
DEFAULT_MIN_SPIKES_PER_RIPPLE = 0.1
DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE = 0.0
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES_RIPPLE = 100
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_RIDGE_STRENGTH_SWEEP = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_MAXITER = 6000
DEFAULT_TOL = 1e-7
DEFAULT_XLA_PREALLOCATE = "false"
DEFAULT_XLA_MEM_FRACTION = "0.70"
RIPPLE_SELECTION_MODE_ALL = "allripples"
RIPPLE_SELECTION_MODE_DEDUPED = "deduped"
RIPPLE_SELECTION_MODE_SINGLE = "single"

HIGHER_IS_BETTER_BY_METRIC = {
    "pseudo_r2": True,
    "mae": False,
    "devexp": True,
    "bits_per_spike": True,
}

EMPIRICAL_P_VALUE_RTOL = 1e-12
EMPIRICAL_P_VALUE_ATOL = 1e-12

RIPPLE_FIGURE_COLOR = "#1f77b4"
NEMOS_JAXOPT_SOLVER_MIN_VERSION = Version("0.2.6")
NEMOS_LEGACY_SOLVER_NAME = "LBFGS"
NEMOS_JAXOPT_SOLVER_NAME = "LBFGS[jaxopt]"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the ripple GLM workflow."""
    parser = argparse.ArgumentParser(
        description="Fit a CA1-to-V1 ripple population GLM for one session"
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
        "--epochs",
        nargs="+",
        help="Optional subset of epoch labels to process. Default: all saved epochs.",
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_S,
        help=f"Fixed ripple window length in seconds. Default: {DEFAULT_RIPPLE_WINDOW_S}",
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        help=(
            "Offset in seconds applied to each fixed ripple window relative to ripple "
            "start time, so the modeled window becomes "
            "[start + offset, start + ripple_window_s + offset]. "
            f"Default: {DEFAULT_RIPPLE_WINDOW_OFFSET_S}"
        ),
    )
    parser.add_argument(
        "--remove-duplicate-ripples",
        action="store_true",
        help=(
            "Drop any ripple whose start falls inside the configured ripple window "
            "of the previous kept ripple, so nearby ripple clusters contribute "
            "only one modeled window."
        ),
    )
    parser.add_argument(
        "--keep-single-ripple-windows",
        action="store_true",
        help=(
            "Keep only fixed ripple windows whose interior contains no other "
            "ripple start times."
        ),
    )
    parser.add_argument(
        "--min-spikes-per-ripple",
        type=float,
        default=DEFAULT_MIN_SPIKES_PER_RIPPLE,
        help=(
            "Minimum average V1 spikes per ripple required to keep one unit. "
            f"Default: {DEFAULT_MIN_SPIKES_PER_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--min-ca1-spikes-per-ripple",
        type=float,
        default=DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE,
        help=(
            "Minimum average CA1 spikes per ripple required to keep one source unit. "
            f"Default: {DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of ripple CV folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--n-shuffles-ripple",
        type=int,
        default=DEFAULT_N_SHUFFLES_RIPPLE,
        help=(
            "Number of shuffle refits per fold for held-out ripple evaluation. "
            f"Default: {DEFAULT_N_SHUFFLES_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--ridge-strength",
        dest="ridge_strengths",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGE_STRENGTH_SWEEP),
        help=(
            "Ridge regularization strength values to run. Pass one value to fit "
            "a single model, or multiple values to sweep. "
            f"Default: {list(DEFAULT_RIDGE_STRENGTH_SWEEP)!r}"
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        dest="shuffle_seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for response shuffles. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    parser.add_argument(
        "--random-seed",
        dest="shuffle_seed",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help=f"Maximum number of LBFGS iterations. Default: {DEFAULT_MAXITER}",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"LBFGS optimizer tolerance. Default: {DEFAULT_TOL}",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Optional value to assign to CUDA_VISIBLE_DEVICES before importing "
            "nemos/JAX, for example '0' or '0,1'."
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI argument ranges."""
    if args.remove_duplicate_ripples and args.keep_single_ripple_windows:
        raise ValueError(
            "--remove-duplicate-ripples and --keep-single-ripple-windows are mutually "
            "exclusive because they define different ripple-selection rules. "
            "Choose only one."
        )
    if not np.isfinite(args.ripple_window_offset_s):
        raise ValueError("--ripple-window-offset-s must be finite.")
    if args.ripple_window_s <= 0:
        raise ValueError("--ripple-window-s must be positive.")
    if args.min_spikes_per_ripple < 0:
        raise ValueError("--min-spikes-per-ripple must be non-negative.")
    if args.min_ca1_spikes_per_ripple < 0:
        raise ValueError("--min-ca1-spikes-per-ripple must be non-negative.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.n_shuffles_ripple < 0:
        raise ValueError("--n-shuffles-ripple must be non-negative.")
    if not args.ridge_strengths:
        raise ValueError("--ridge-strength must contain at least one value.")
    if any(ridge_strength < 0 for ridge_strength in args.ridge_strengths):
        raise ValueError("--ridge-strength must contain only non-negative values.")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive.")
    if args.tol <= 0:
        raise ValueError("--tol must be positive.")


def validate_epochs(
    available_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return selected epochs after validating any requested subset."""
    if requested_epochs is None:
        return available_epochs

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in available_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in the saved session epochs {available_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return requested_epochs


def validate_selected_epochs_across_sources(
    selected_epochs: list[str],
    *,
    source_epochs: dict[str, list[str] | dict[str, Any]],
) -> None:
    """Ensure every selected epoch exists in each required per-epoch source."""
    missing_by_source: dict[str, list[str]] = {}
    for source_name, source in source_epochs.items():
        available_epochs = set(source.keys() if isinstance(source, dict) else source)
        missing_epochs = [epoch for epoch in selected_epochs if epoch not in available_epochs]
        if missing_epochs:
            missing_by_source[source_name] = missing_epochs

    if missing_by_source:
        details = "; ".join(
            f"{source_name}: {missing_epochs!r}"
            for source_name, missing_epochs in missing_by_source.items()
        )
        raise ValueError(
            "Selected epochs are missing required ripple GLM inputs. "
            f"Missing epochs by source: {details}"
        )


def restrict_epochs_to_ripple_event_epochs(
    selected_epochs: list[str],
    *,
    ripple_event_epochs: list[str],
    ripple_event_source: str,
) -> list[str]:
    """Keep only epochs that are present in the loaded ripple event table."""
    ripple_epoch_set = set(ripple_event_epochs)
    kept_epochs = [epoch for epoch in selected_epochs if epoch in ripple_epoch_set]
    skipped_epochs = [epoch for epoch in selected_epochs if epoch not in ripple_epoch_set]

    if ripple_event_source == "parquet":
        source_label = "ripple_times.parquet"
    else:
        source_label = f"the loaded ripple event source ({ripple_event_source})"

    print(f"Restricting ripple GLM to epochs present in {source_label}.")
    print(f"Processing epochs: {kept_epochs!r}")
    if skipped_epochs:
        print(f"Skipping epochs without saved ripple events in {source_label}: {skipped_epochs!r}")

    if not kept_epochs:
        raise ValueError(
            "No selected epochs are present in the loaded ripple event table. "
            f"Selected epochs: {selected_epochs!r}; source: {source_label}."
        )
    return kept_epochs


def normalize_ripple_table(result: Any, epoch: str) -> pd.DataFrame:
    """Normalize one ripple-event payload into a dataframe."""
    if result is None or (isinstance(result, dict) and not result):
        dataframe = pd.DataFrame(columns=["start_time", "end_time"])
    elif isinstance(result, pd.DataFrame):
        dataframe = result.copy()
    elif hasattr(result, "to_dataframe"):
        dataframe = result.to_dataframe()
    else:
        dataframe = pd.DataFrame(result)

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Ripple event payload for epoch {epoch!r} could not be normalized to a DataFrame."
        )

    dataframe = dataframe.reset_index(drop=True)
    if dataframe.empty:
        for column_name in ("start_time", "end_time"):
            if column_name not in dataframe.columns:
                dataframe[column_name] = pd.Series(dtype=float)

    missing_columns = [
        column_name
        for column_name in ("start_time", "end_time")
        if column_name not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Ripple events for epoch {epoch!r} are missing required columns: {missing_columns!r}."
        )

    dataframe["start_time"] = np.asarray(dataframe["start_time"], dtype=float)
    dataframe["end_time"] = np.asarray(dataframe["end_time"], dtype=float)
    return dataframe


def remove_duplicate_ripples(
    ripple_table: pd.DataFrame,
    *,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop ripples whose starts fall inside the previous kept ripple's analysis window."""
    normalized_table = ripple_table.sort_values("start_time").reset_index(drop=True)
    total_ripples = int(len(normalized_table))
    if total_ripples <= 1:
        return normalized_table, np.ones(total_ripples, dtype=bool)

    starts = np.asarray(normalized_table["start_time"], dtype=float)
    window_starts, window_ends = _build_ripple_sample_windows(
        normalized_table,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    keep = np.ones(total_ripples, dtype=bool)

    previous_window_start = float(window_starts[0])
    previous_window_end = float(window_ends[0])
    for ripple_index in range(1, total_ripples):
        current_start = float(starts[ripple_index])
        if previous_window_start <= current_start < previous_window_end:
            keep[ripple_index] = False
            continue

        previous_window_start = float(window_starts[ripple_index])
        previous_window_end = float(window_ends[ripple_index])

    filtered_table = normalized_table.loc[keep].reset_index(drop=True)
    return filtered_table, keep


def keep_single_ripple_windows(
    ripple_table: pd.DataFrame,
    *,
    ripple_window_s: float,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep only fixed windows whose interior contains exactly one ripple start."""
    normalized_table = ripple_table.sort_values("start_time").reset_index(drop=True)
    total_ripples = int(len(normalized_table))
    if total_ripples <= 1:
        return normalized_table, np.ones(total_ripples, dtype=bool)

    starts = np.asarray(normalized_table["start_time"], dtype=float)
    window_starts, window_ends = _build_ripple_sample_windows(
        normalized_table,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    interval_counts = np.searchsorted(starts, window_ends, side="left") - np.searchsorted(
        starts,
        window_starts,
        side="left",
    )
    self_inside = (window_starts <= starts) & (starts < window_ends)
    previous_window_contains_current = np.zeros(total_ripples, dtype=bool)
    previous_window_contains_current[1:] = (
        (window_starts[:-1] <= starts[1:]) & (starts[1:] < window_ends[:-1])
    )
    keep = (interval_counts - self_inside.astype(np.int64) == 0) & (
        ~previous_window_contains_current
    )

    filtered_table = normalized_table.loc[keep].reset_index(drop=True)
    return filtered_table, keep


def resolve_ripple_selection_mode(args: argparse.Namespace) -> str:
    """Return the configured ripple selection mode."""
    if args.remove_duplicate_ripples:
        return RIPPLE_SELECTION_MODE_DEDUPED
    if args.keep_single_ripple_windows:
        return RIPPLE_SELECTION_MODE_SINGLE
    return RIPPLE_SELECTION_MODE_ALL


def load_ripple_tables_from_parquet_output(path: Path) -> dict[str, pd.DataFrame]:
    """Load ripple events from the modern flattened parquet output."""
    if not path.exists():
        raise FileNotFoundError(f"Ripple parquet output not found: {path}")

    flat_table = pd.read_parquet(path)
    if flat_table.empty:
        return {}

    working_table = flat_table.copy()
    rename_columns = {}
    if "start" in working_table.columns and "start_time" not in working_table.columns:
        rename_columns["start"] = "start_time"
    if "end" in working_table.columns and "end_time" not in working_table.columns:
        rename_columns["end"] = "end_time"
    if rename_columns:
        working_table = working_table.rename(columns=rename_columns)

    if "epoch" not in working_table.columns:
        raise ValueError(f"Ripple parquet output is missing the required 'epoch' column: {path}")

    ripple_tables: dict[str, pd.DataFrame] = {}
    for epoch, group in working_table.groupby("epoch", sort=False):
        ripple_tables[str(epoch)] = normalize_ripple_table(
            group.drop(columns="epoch").reset_index(drop=True),
            epoch=str(epoch),
        )
    return ripple_tables


def load_ripple_tables_from_interval_output(path: Path) -> dict[str, pd.DataFrame]:
    """Load ripple events from a saved pynapple `IntervalSet` output."""
    if not path.exists():
        raise FileNotFoundError(f"Ripple interval output not found: {path}")

    try:
        import pynapple as nap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pynapple is required to load ripple interval outputs."
        ) from exc

    ripple_intervals = nap.load_file(path)
    try:
        epoch_info = ripple_intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is None:
        if hasattr(ripple_intervals, "as_dataframe"):
            interval_df = ripple_intervals.as_dataframe()
        elif hasattr(ripple_intervals, "_metadata"):
            interval_df = ripple_intervals._metadata.copy()  # type: ignore[attr-defined]
        else:
            raise ValueError("Could not read metadata from the pynapple IntervalSet.")
        if "epoch" not in interval_df.columns:
            raise ValueError(
                f"Ripple interval output is missing the required 'epoch' metadata: {path}"
            )
        epoch_array = np.asarray(interval_df["epoch"])
    else:
        epoch_array = np.asarray(epoch_info)

    start_array = np.asarray(ripple_intervals.start, dtype=float).ravel()
    end_array = np.asarray(ripple_intervals.end, dtype=float).ravel()
    if start_array.shape != end_array.shape:
        raise ValueError(
            f"Ripple interval output has mismatched start/end shapes: {path}"
        )
    if len(epoch_array) != len(start_array):
        raise ValueError(
            "Ripple interval output has mismatched epoch/start metadata lengths: "
            f"{path}"
        )

    flat_table = pd.DataFrame(
        {
            "epoch": [str(epoch) for epoch in epoch_array.tolist()],
            "start_time": start_array,
            "end_time": end_array,
        }
    )
    ripple_tables: dict[str, pd.DataFrame] = {}
    for epoch, group in flat_table.groupby("epoch", sort=False):
        ripple_tables[str(epoch)] = normalize_ripple_table(
            group.drop(columns="epoch").reset_index(drop=True),
            epoch=str(epoch),
        )
    return ripple_tables


def load_ripple_tables_from_legacy_pickle(path: Path) -> dict[str, pd.DataFrame]:
    """Load ripple events from the legacy detector pickle."""
    if not path.exists():
        raise FileNotFoundError(f"Legacy ripple detector pickle not found: {path}")

    with open(path, "rb") as file:
        loaded = pickle.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Legacy ripple detector pickle is not a dictionary: {path}")

    return {
        str(epoch): normalize_ripple_table(table, epoch=str(epoch))
        for epoch, table in loaded.items()
    }


def load_ripple_tables(analysis_path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """Load ripple events, preferring the supported parquet output."""
    parquet_path = analysis_path / "ripple" / "ripple_times.parquet"
    legacy_path = analysis_path / "ripple" / "Kay_ripple_detector.pkl"

    parquet_error: Exception | None = None
    if parquet_path.exists():
        try:
            return load_ripple_tables_from_parquet_output(parquet_path), "parquet"
        except Exception as exc:
            parquet_error = exc

    if legacy_path.exists():
        return load_ripple_tables_from_legacy_pickle(legacy_path), "pickle"

    if parquet_error is not None:
        raise ValueError(
            f"Failed to load {parquet_path} and no legacy pickle fallback was found."
        ) from parquet_error
    raise FileNotFoundError(
        f"Could not find {parquet_path} or {legacy_path} under {analysis_path}."
    )


def prepare_ripple_glm_session(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Load one session's timestamps, spikes, and ripple events."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, timestamps_ephys_by_epoch, ephys_source = load_ephys_timestamps_by_epoch(
        analysis_path
    )
    timestamps_ephys_all, ephys_all_source = load_ephys_timestamps_all(analysis_path)
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all,
        regions=(TARGET_REGION, SOURCE_REGION),
    )
    loaded_ripple_tables, ripple_source = load_ripple_tables(analysis_path)
    ripple_event_epochs = list(loaded_ripple_tables.keys())
    ripple_tables = {
        epoch: loaded_ripple_tables.get(epoch, normalize_ripple_table(None, epoch))
        for epoch in epoch_tags
    }
    for epoch, table in loaded_ripple_tables.items():
        if epoch not in ripple_tables:
            ripple_tables[epoch] = table
    epoch_intervals = build_epoch_intervals(timestamps_ephys_by_epoch)

    return {
        "analysis_path": analysis_path,
        "epoch_tags": epoch_tags,
        "timestamps_ephys_by_epoch": timestamps_ephys_by_epoch,
        "timestamps_ephys_all": timestamps_ephys_all,
        "epoch_intervals": epoch_intervals,
        "spikes_by_region": spikes_by_region,
        "ripple_event_epochs": ripple_event_epochs,
        "ripple_tables": ripple_tables,
        "sources": {
            "timestamps_ephys": ephys_source,
            "timestamps_ephys_all": ephys_all_source,
            "sorting": "spikeinterface",
            "ripple_events": ripple_source,
        },
    }


def build_epoch_intervals(
    timestamps_by_epoch: dict[str, np.ndarray],
) -> dict[str, "nap.IntervalSet"]:
    """Return one pynapple IntervalSet per epoch from saved ephys timestamps."""
    import pynapple as nap

    epoch_intervals: dict[str, nap.IntervalSet] = {}
    for epoch, timestamps in timestamps_by_epoch.items():
        epoch_timestamps = np.asarray(timestamps, dtype=float)
        if epoch_timestamps.ndim != 1 or epoch_timestamps.size == 0:
            raise ValueError(f"Ephys timestamps for epoch {epoch!r} are empty or malformed.")
        epoch_intervals[epoch] = nap.IntervalSet(
            start=float(epoch_timestamps[0]),
            end=float(epoch_timestamps[-1]),
            time_units="s",
        )
    return epoch_intervals


def poisson_ll_per_neuron(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Return per-neuron Poisson log-likelihood summed over samples."""
    lam = np.clip(np.asarray(lam, dtype=float), 1e-12, None)
    y = np.asarray(y, dtype=float)
    return np.sum(y * np.log(lam) - lam - gammaln(y + 1), axis=0)


def mcfadden_pseudo_r2_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Return McFadden pseudo-R^2 per neuron against a train-fitted constant-rate null."""
    ll_model = poisson_ll_per_neuron(y_test, lam_test)

    lam0 = np.mean(np.asarray(y_train, dtype=float), axis=0, keepdims=True)
    lam0 = np.repeat(lam0, np.asarray(y_test).shape[0], axis=0)
    ll_null = poisson_ll_per_neuron(y_test, lam0)
    ll_null = np.where(np.abs(ll_null) < 1e-12, -1e-12, ll_null)
    return 1.0 - (ll_model / ll_null)


def poisson_ll_saturated_per_neuron(y: np.ndarray) -> np.ndarray:
    """Return the saturated Poisson log-likelihood per neuron."""
    y = np.asarray(y, dtype=float)
    y_safe = np.clip(y, 1e-12, None)
    return np.sum(y * np.log(y_safe) - y - gammaln(y + 1), axis=0)


def mae_per_neuron(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return mean absolute error per neuron."""
    return np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)), axis=0)


def deviance_explained_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """Return Poisson deviance explained per neuron."""
    y_test = np.asarray(y_test, dtype=float)
    lam_test = np.asarray(lam_test, dtype=float)
    y_null_fit = np.asarray(y_null_fit, dtype=float)

    ll_model = poisson_ll_per_neuron(y_test, lam_test)
    lam0 = np.mean(y_null_fit, axis=0, keepdims=True)
    lam0 = np.repeat(lam0, y_test.shape[0], axis=0)
    ll_null = poisson_ll_per_neuron(y_test, lam0)
    ll_sat = poisson_ll_saturated_per_neuron(y_test)

    denom = ll_sat - ll_null
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return (ll_model - ll_null) / denom


def bits_per_spike_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """Return information gain in bits/spike per neuron."""
    y_test = np.asarray(y_test, dtype=float)
    lam_test = np.asarray(lam_test, dtype=float)
    y_null_fit = np.asarray(y_null_fit, dtype=float)

    ll_model = poisson_ll_per_neuron(y_test, lam_test)
    lam0 = np.mean(y_null_fit, axis=0)
    lam0 = np.clip(lam0, 1e-12, None)
    ll_null = np.sum(
        y_test * np.log(lam0[None, :]) - lam0[None, :] - gammaln(y_test + 1),
        axis=0,
    )

    spikes = np.sum(y_test, axis=0)
    denom = spikes * np.log(2.0)
    bits_per_spike = (ll_model - ll_null) / np.clip(denom, 1e-12, None)
    return np.where(spikes > 0, bits_per_spike, np.nan)


def shuffle_time_per_neuron(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle samples independently within each neuron."""
    y = np.asarray(y)
    n_samples, n_neurons = y.shape
    indices = np.vstack([rng.permutation(n_samples) for _ in range(n_neurons)]).T
    return np.take_along_axis(y, indices, axis=0)


def _first_nonfinite_info(values: np.ndarray) -> str | None:
    """Return a short description of the first non-finite value, if any."""
    values = np.asarray(values)
    mask = ~np.isfinite(values)
    if not np.any(mask):
        return None
    index = np.argwhere(mask)[0]
    bad_value = values[tuple(index)]
    return f"shape={values.shape}, first_bad_idx={tuple(index)}, value={bad_value}"


def _preprocess_X_fit_apply(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the legacy feature preprocessing on train and apply it to train/test."""
    keep_x = np.asarray(X_train, dtype=float).std(axis=0) > 1e-6
    X_train_kept = np.asarray(X_train, dtype=float)[:, keep_x]
    X_test_kept = np.asarray(X_test, dtype=float)[:, keep_x]

    mean = X_train_kept.mean(axis=0, keepdims=True)
    std = X_train_kept.std(axis=0, keepdims=True)

    X_train_pp = (X_train_kept - mean) / (std + 1e-8)
    X_test_pp = (X_test_kept - mean) / (std + 1e-8)

    scale = np.sqrt(max(X_train_pp.shape[1], 1))
    X_train_pp /= scale
    X_test_pp /= scale

    X_train_pp = np.clip(X_train_pp, -10.0, 10.0)
    X_test_pp = np.clip(X_test_pp, -10.0, 10.0)
    return X_train_pp, X_test_pp, keep_x, mean.ravel(), std.ravel()


def _coef_feat_by_unit(model: Any, n_features: int) -> np.ndarray:
    """Return coefficients as (n_features, n_units) regardless of internal shape."""
    coef = np.asarray(model.coef_)
    if coef.shape[0] != n_features:
        coef = coef.T
    return coef


def get_epoch_skip_reason(
    *,
    n_ripples: int,
    n_splits: int,
    n_kept_v1_units: int | None = None,
    n_kept_ca1_units: int | None = None,
) -> str | None:
    """Return a skip reason string for common weak-epoch cases."""
    if n_ripples < n_splits:
        return f"Not enough ripples for CV: n_ripples={n_ripples}, n_splits={n_splits}"
    if n_kept_v1_units is not None and n_kept_v1_units <= 0:
        return "No V1 units passed the minimum spikes-per-ripple threshold."
    if n_kept_ca1_units is not None and n_kept_ca1_units <= 0:
        return "No CA1 units passed the minimum spikes-per-ripple threshold."
    return None


def resolve_ridge_strengths(args: argparse.Namespace) -> list[float]:
    """Return the ridge strengths to run for this invocation."""
    return [float(ridge_strength) for ridge_strength in args.ridge_strengths]


def _clear_jax_caches() -> None:
    """Best-effort cleanup of JAX caches between large fits."""
    try:
        import jax
    except ModuleNotFoundError:
        return
    jax.clear_caches()


def _print_progress(step_label: str, message: str) -> None:
    """Print one compact progress message for long-running CLI work."""
    print(f"[{step_label}] {message}")


def configure_jax_environment(cuda_visible_devices: str | None) -> None:
    """Configure JAX memory defaults before importing JAX/NEMOS."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_XLA_PREALLOCATE)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", DEFAULT_XLA_MEM_FRACTION)


def _resolve_nemos_population_glm_solver(nmo: Any) -> tuple[str, str]:
    """Return the installed `nemos` version and matching `PopulationGLM` solver."""
    version_text = getattr(nmo, "__version__", None)
    if version_text is None:
        try:
            version_text = package_version("nemos")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Could not determine installed `nemos` version from "
                "`nemos.__version__` or importlib.metadata.version('nemos')."
            ) from exc

    try:
        nemos_version = Version(str(version_text))
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Could not parse installed `nemos` version {version_text!r}."
        ) from exc

    if nemos_version < NEMOS_JAXOPT_SOLVER_MIN_VERSION:
        solver_name = NEMOS_LEGACY_SOLVER_NAME
    else:
        solver_name = NEMOS_JAXOPT_SOLVER_NAME
    return str(nemos_version), solver_name


def _format_nemos_solver_selection_message(
    nemos_version: str,
    solver_name: str,
) -> str:
    """Return the runtime message describing the selected `nemos` solver."""
    return (
        f"Using nemos {nemos_version} with PopulationGLM solver_name={solver_name!r}. "
        "For nemos>0.2.5, ripple_glm selects 'LBFGS[jaxopt]' because the default "
        "solver backend changed in 0.2.6 and these hyperparameters are tuned for "
        "the jaxopt backend."
    )


def _build_ripple_sample_windows(
    ripple_table: pd.DataFrame,
    *,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ordered `(start, end)` window per ripple row without merging overlaps."""
    normalized_table = ripple_table.sort_values("start_time").reset_index(drop=True)
    ripple_anchor_starts = np.asarray(normalized_table["start_time"], dtype=float)
    ripple_offset_s = float(ripple_window_offset_s)
    ripple_starts = ripple_anchor_starts + ripple_offset_s
    if ripple_window_s is None:
        ripple_anchor_ends = np.asarray(normalized_table["end_time"], dtype=float)
        ripple_ends = ripple_anchor_ends + ripple_offset_s
    else:
        ripple_ends = ripple_anchor_starts + ripple_offset_s + float(ripple_window_s)
    return ripple_starts, ripple_ends


def _count_spikes_in_windows(
    spike_group: Any,
    *,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count spikes per unit for overlapping windows while preserving one row per window."""
    starts = np.asarray(window_starts, dtype=float).ravel()
    ends = np.asarray(window_ends, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            f"Window start/end arrays must have the same shape, got {starts.shape} and {ends.shape}."
        )

    unit_ids = np.asarray(list(spike_group.keys()))
    counts = np.zeros((starts.size, unit_ids.size), dtype=np.float64)
    for unit_index, unit_id in enumerate(unit_ids.tolist()):
        spike_train = spike_group[unit_id]
        spike_times = np.asarray(spike_train.t, dtype=float).ravel()
        end_counts = np.searchsorted(spike_times, ends, side="left")
        start_counts = np.searchsorted(spike_times, starts, side="left")
        counts[:, unit_index] = end_counts - start_counts
    return counts, unit_ids


def _prepare_ripple_glm_epoch_inputs(
    epoch: str,
    *,
    spikes: dict[str, Any],
    ripple_table: pd.DataFrame,
    min_spikes_per_ripple: float,
    min_ca1_spikes_per_ripple: float,
    ripple_window_s: float | None,
    n_splits: int,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> dict[str, Any]:
    """Build one epoch's ridge-independent ripple GLM design data."""
    normalized_ripple_table = ripple_table.sort_values("start_time").reset_index(drop=True)
    ripple_start_times = np.asarray(normalized_ripple_table["start_time"], dtype=float)
    ripple_starts, ripple_ends = _build_ripple_sample_windows(
        normalized_ripple_table,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )

    _print_progress(epoch, "Counting CA1 and V1 spikes in ripple windows.")
    X_r, ca1_unit_ids = _count_spikes_in_windows(
        spikes[SOURCE_REGION],
        window_starts=ripple_starts,
        window_ends=ripple_ends,
    )
    y_r, v1_unit_ids = _count_spikes_in_windows(
        spikes[TARGET_REGION],
        window_starts=ripple_starts,
        window_ends=ripple_ends,
    )

    n_ripples = int(ripple_starts.size)
    skip_reason = get_epoch_skip_reason(n_ripples=n_ripples, n_splits=n_splits)
    if skip_reason is not None:
        raise ValueError(skip_reason)

    for name, values in (("X_r", X_r), ("y_r", y_r)):
        bad_value_info = _first_nonfinite_info(values)
        if bad_value_info is not None:
            raise ValueError(f"Non-finite values found in {name}: {bad_value_info}")

    keep_y = (y_r.sum(axis=0) / max(n_ripples, 1)) >= float(min_spikes_per_ripple)
    y_r = y_r[:, keep_y]
    kept_v1_unit_ids = v1_unit_ids[keep_y]

    keep_x_ripple = (X_r.sum(axis=0) / max(n_ripples, 1)) >= float(min_ca1_spikes_per_ripple)
    X_r = X_r[:, keep_x_ripple]
    kept_ca1_unit_ids = ca1_unit_ids[keep_x_ripple]

    n_cells = int(y_r.shape[1])
    n_ca1_cells = int(X_r.shape[1])
    skip_reason = get_epoch_skip_reason(
        n_ripples=n_ripples,
        n_splits=n_splits,
        n_kept_v1_units=n_cells,
        n_kept_ca1_units=n_ca1_cells,
    )
    if skip_reason is not None:
        raise ValueError(skip_reason)

    cv_splits = [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )
        for train_idx, test_idx in KFold(
            n_splits=n_splits,
            shuffle=False,
            random_state=None,
        ).split(X_r)
    ]
    return {
        "X_r": X_r,
        "y_r": y_r,
        "ca1_unit_ids": kept_ca1_unit_ids,
        "v1_unit_ids": kept_v1_unit_ids,
        "ripple_start_times": ripple_start_times.astype(np.float64),
        "ripple_starts": ripple_starts.astype(np.float64),
        "ripple_ends": ripple_ends.astype(np.float64),
        "n_ripples": n_ripples,
        "n_cells": n_cells,
        "n_ca1_cells": n_ca1_cells,
        "cv_splits": cv_splits,
    }


def _fit_ripple_glm_on_prepared_epoch(
    epoch: str,
    *,
    prepared_epoch: dict[str, Any],
    n_shuffles_ripple: int = DEFAULT_N_SHUFFLES_RIPPLE,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    ripple_window_s: float | None = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    """Fit the ripple GLM on a prepared epoch design matrix."""
    configure_jax_environment(cuda_visible_devices=None)
    import nemos as nmo

    nemos_version, solver_name = _resolve_nemos_population_glm_solver(nmo)
    print(_format_nemos_solver_selection_message(nemos_version, solver_name))

    rng = np.random.default_rng(shuffle_seed)
    X_r = np.asarray(prepared_epoch["X_r"], dtype=np.float64)
    y_r = np.asarray(prepared_epoch["y_r"], dtype=np.float64)
    kept_ca1_unit_ids = np.asarray(prepared_epoch["ca1_unit_ids"])
    kept_v1_unit_ids = np.asarray(prepared_epoch["v1_unit_ids"])
    ripple_start_times = np.asarray(prepared_epoch["ripple_start_times"], dtype=np.float64)
    ripple_starts = np.asarray(prepared_epoch["ripple_starts"], dtype=np.float64)
    ripple_ends = np.asarray(prepared_epoch["ripple_ends"], dtype=np.float64)
    n_ripples = int(prepared_epoch["n_ripples"])
    n_cells = int(prepared_epoch["n_cells"])
    n_ca1_cells = int(prepared_epoch["n_ca1_cells"])
    cv_splits = [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )
        for train_idx, test_idx in prepared_epoch["cv_splits"]
    ]
    n_splits = len(cv_splits)

    _print_progress(
        epoch,
        f"Running {n_splits}-fold CV with {n_shuffles_ripple} shuffle refits per fold.",
    )

    def _alloc_metric_array() -> np.ndarray:
        return np.full((n_splits, n_cells), np.nan, dtype=np.float32)

    pseudo_r2_ripple = _alloc_metric_array()
    mae_ripple = _alloc_metric_array()
    devexp_ripple = _alloc_metric_array()
    bits_per_spike_ripple = _alloc_metric_array()
    ripple_observed_count_oof = np.full((n_ripples, n_cells), np.nan, dtype=np.float32)
    ripple_predicted_count_oof = np.full((n_ripples, n_cells), np.nan, dtype=np.float32)
    ripple_fold_index = np.full(n_ripples, -1, dtype=np.int32)

    pseudo_r2_ripple_shuff = np.full(
        (n_splits, n_shuffles_ripple, n_cells),
        np.nan,
        dtype=np.float32,
    )
    mae_ripple_shuff = np.full(
        (n_splits, n_shuffles_ripple, n_cells),
        np.nan,
        dtype=np.float32,
    )
    devexp_ripple_shuff = np.full(
        (n_splits, n_shuffles_ripple, n_cells),
        np.nan,
        dtype=np.float32,
    )
    bits_per_spike_ripple_shuff = np.full(
        (n_splits, n_shuffles_ripple, n_cells),
        np.nan,
        dtype=np.float32,
    )
    for fold_index, (train_idx, test_idx) in enumerate(cv_splits):
        _print_progress(epoch, f"Fold {fold_index + 1}/{n_splits}: fitting ripple model.")
        X_train_r = X_r[train_idx]
        X_test_r = X_r[test_idx]
        y_train_r = y_r[train_idx]
        y_test_r = y_r[test_idx]

        X_train_pp, X_test_r_pp, _keep_x, _mean, _std = _preprocess_X_fit_apply(
            X_train=X_train_r,
            X_test=X_test_r,
        )
        if X_train_pp.shape[1] == 0:
            raise ValueError(
                f"Epoch {epoch!r}, fold {fold_index}: all CA1 features were near-constant."
            )

        glm = nmo.glm.PopulationGLM(
            solver_name=solver_name,
            regularizer="Ridge",
            regularizer_strength=float(ridge_strength),
            solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
        )
        glm.fit(X_train_pp, y_train_r)
        lam_r = np.asarray(glm.predict(X_test_r_pp), dtype=np.float64)

        pseudo_r2_ripple[fold_index] = mcfadden_pseudo_r2_per_neuron(
            y_test_r,
            lam_r,
            y_train_r,
        ).astype(np.float32)
        mae_ripple[fold_index] = mae_per_neuron(y_test_r, lam_r).astype(np.float32)
        devexp_ripple[fold_index] = deviance_explained_per_neuron(
            y_test=y_test_r,
            lam_test=lam_r,
            y_null_fit=y_train_r,
        ).astype(np.float32)
        bits_per_spike_ripple[fold_index] = bits_per_spike_per_neuron(
            y_test=y_test_r,
            lam_test=lam_r,
            y_null_fit=y_train_r,
        ).astype(np.float32)
        ripple_observed_count_oof[test_idx] = y_test_r.astype(np.float32)
        ripple_predicted_count_oof[test_idx] = lam_r.astype(np.float32)
        ripple_fold_index[test_idx] = int(fold_index)

        if n_shuffles_ripple > 0:
            _print_progress(
                epoch,
                f"Fold {fold_index + 1}/{n_splits}: running shuffle refits.",
            )
        for shuffle_index in range(n_shuffles_ripple):
            y_train_shuffled = shuffle_time_per_neuron(y_train_r, rng)
            glm_shuffle = nmo.glm.PopulationGLM(
                solver_name=solver_name,
                regularizer="Ridge",
                regularizer_strength=float(ridge_strength),
                solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
            )
            glm_shuffle.fit(X_train_pp, y_train_shuffled)
            lam_r_shuffled = np.asarray(glm_shuffle.predict(X_test_r_pp), dtype=np.float64)

            pseudo_r2_ripple_shuff[fold_index, shuffle_index] = mcfadden_pseudo_r2_per_neuron(
                y_test_r,
                lam_r_shuffled,
                y_train_r,
            ).astype(np.float32)
            mae_ripple_shuff[fold_index, shuffle_index] = mae_per_neuron(
                y_test_r,
                lam_r_shuffled,
            ).astype(np.float32)
            devexp_ripple_shuff[fold_index, shuffle_index] = deviance_explained_per_neuron(
                y_test=y_test_r,
                lam_test=lam_r_shuffled,
                y_null_fit=y_train_r,
            ).astype(np.float32)
            bits_per_spike_ripple_shuff[fold_index, shuffle_index] = bits_per_spike_per_neuron(
                y_test=y_test_r,
                lam_test=lam_r_shuffled,
                y_null_fit=y_train_r,
            ).astype(np.float32)

            del glm_shuffle, y_train_shuffled, lam_r_shuffled

        del glm, lam_r
        _clear_jax_caches()
        gc.collect()

    X_r_all_pp, _, keep_x_all, _, _ = _preprocess_X_fit_apply(X_r, X_r)
    if X_r_all_pp.shape[1] == 0:
        raise ValueError(f"Epoch {epoch!r}: all CA1 features were near-constant.")

    _print_progress(epoch, "Fitting final full-data model for coefficient export.")
    coef_ca1_full_all = np.full((X_r_all_pp.shape[1], n_cells), np.nan, dtype=np.float32)
    coef_intercept_full_all = np.full(n_cells, np.nan, dtype=np.float32)
    coef_ca1_unit_ids = kept_ca1_unit_ids[keep_x_all]

    glm_all = nmo.glm.PopulationGLM(
        solver_name=solver_name,
        regularizer="Ridge",
        regularizer_strength=float(ridge_strength),
        solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
    )
    glm_all.fit(X_r_all_pp, y_r)
    coef_ca1_full_all = _coef_feat_by_unit(glm_all, n_features=X_r_all_pp.shape[1]).astype(
        np.float32
    )
    coef_intercept_full_all = np.asarray(glm_all.intercept_, dtype=np.float32).reshape(-1)
    del glm_all
    _clear_jax_caches()
    gc.collect()

    results = {
        "epoch": epoch,
        "shuffle_seed": int(shuffle_seed),
        "min_spikes_per_ripple": float(np.nan),
        "min_ca1_spikes_per_ripple": float(np.nan),
        "n_splits": int(n_splits),
        "n_shuffles_ripple": int(n_shuffles_ripple),
        "ripple_window_s": None if ripple_window_s is None else float(ripple_window_s),
        "ripple_window_offset_s": float(ripple_window_offset_s),
        "n_ripples": n_ripples,
        "n_cells": n_cells,
        "n_ca1_cells": n_ca1_cells,
        "v1_unit_ids": kept_v1_unit_ids,
        "ca1_unit_ids": kept_ca1_unit_ids,
        "coef_ca1_unit_ids": coef_ca1_unit_ids,
        "pseudo_r2_ripple_folds": pseudo_r2_ripple,
        "mae_ripple_folds": mae_ripple,
        "devexp_ripple_folds": devexp_ripple,
        "bits_per_spike_ripple_folds": bits_per_spike_ripple,
        "pseudo_r2_ripple_shuff_folds": pseudo_r2_ripple_shuff,
        "mae_ripple_shuff_folds": mae_ripple_shuff,
        "devexp_ripple_shuff_folds": devexp_ripple_shuff,
        "bits_per_spike_ripple_shuff_folds": bits_per_spike_ripple_shuff,
        "ripple_start_time_s": ripple_start_times,
        "ripple_window_start_s": ripple_starts,
        "ripple_window_end_s": ripple_ends,
        "ripple_fold_index": ripple_fold_index,
        "ripple_observed_count_oof": ripple_observed_count_oof,
        "ripple_predicted_count_oof": ripple_predicted_count_oof,
        "coef_ca1_full_all": coef_ca1_full_all,
        "coef_intercept_full_all": coef_intercept_full_all,
    }
    return results


def fit_ripple_glm_train_on_ripple(
    epoch: str,
    *,
    spikes: dict[str, Any],
    epoch_interval: "nap.IntervalSet",
    ripple_table: pd.DataFrame,
    min_spikes_per_ripple: float = DEFAULT_MIN_SPIKES_PER_RIPPLE,
    min_ca1_spikes_per_ripple: float = DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE,
    n_shuffles_ripple: int = DEFAULT_N_SHUFFLES_RIPPLE,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    ripple_window_s: float | None = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    n_splits: int = DEFAULT_N_SPLITS,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    """Fit the ripple-only GLM workflow for one epoch."""
    prepared_epoch = _prepare_ripple_glm_epoch_inputs(
        epoch,
        spikes=spikes,
        ripple_table=ripple_table,
        min_spikes_per_ripple=min_spikes_per_ripple,
        min_ca1_spikes_per_ripple=min_ca1_spikes_per_ripple,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        n_splits=n_splits,
    )
    results = _fit_ripple_glm_on_prepared_epoch(
        epoch,
        prepared_epoch=prepared_epoch,
        n_shuffles_ripple=n_shuffles_ripple,
        shuffle_seed=shuffle_seed,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ridge_strength=ridge_strength,
        maxiter=maxiter,
        tol=tol,
    )
    results["min_spikes_per_ripple"] = float(min_spikes_per_ripple)
    results["min_ca1_spikes_per_ripple"] = float(min_ca1_spikes_per_ripple)
    return results


def _as_1d_float(values: Any) -> np.ndarray:
    """Return a 1D float array, handling object arrays from `.npz` payloads."""
    array = np.asarray(values)
    if array.dtype == object:
        array = np.asarray(list(array), dtype=float)
    return np.asarray(array, dtype=float).ravel()


def _as_2d_float(values: Any) -> np.ndarray:
    """Return a 2D float array, handling object arrays from `.npz` payloads."""
    array = np.asarray(values)
    if array.dtype == object:
        return np.stack(array.tolist(), axis=0).astype(float)
    return np.asarray(array, dtype=float)


def _as_3d_float(values: Any) -> np.ndarray:
    """Return a 3D float array, handling object arrays from `.npz` payloads."""
    array = np.asarray(values)
    if array.dtype == object:
        return np.stack(array.tolist(), axis=0).astype(float)
    return np.asarray(array, dtype=float)


def nansem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the NaN-aware SEM using population SD."""
    values = np.asarray(values, dtype=float)
    counts = np.sum(np.isfinite(values), axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=0)
    return np.where(counts > 0, sd / np.sqrt(counts), np.nan)


def _format_window_suffix_value(value: float) -> str:
    """Return one filesystem-friendly encoded float value."""
    abs_text = f"{abs(float(value)):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"m{abs_text}" if float(value) < 0 else abs_text


def format_ripple_window_offset_suffix(ripple_window_offset_s: float) -> str:
    """Return a filesystem-friendly suffix for one ripple-window offset."""
    return f"off_{_format_window_suffix_value(ripple_window_offset_s)}s"


def format_ripple_window_suffix(
    ripple_window_s: float,
    *,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> str:
    """Return a filesystem-friendly suffix for one ripple window setup."""
    window_suffix = f"rw_{_format_window_suffix_value(ripple_window_s)}s"
    if np.isclose(
        float(ripple_window_offset_s),
        DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        rtol=1e-12,
        atol=1e-12,
    ):
        return window_suffix
    return f"{window_suffix}_{format_ripple_window_offset_suffix(ripple_window_offset_s)}"


def format_ripple_selection_suffix(ripple_selection_mode: str) -> str:
    """Return a filesystem-friendly suffix describing ripple selection."""
    if ripple_selection_mode == RIPPLE_SELECTION_MODE_ALL:
        return RIPPLE_SELECTION_MODE_ALL
    if ripple_selection_mode == RIPPLE_SELECTION_MODE_DEDUPED:
        return RIPPLE_SELECTION_MODE_DEDUPED
    if ripple_selection_mode == RIPPLE_SELECTION_MODE_SINGLE:
        return RIPPLE_SELECTION_MODE_SINGLE
    raise ValueError(f"Unknown ripple selection mode: {ripple_selection_mode!r}")


def format_ridge_strength_suffix(ridge_strength: float) -> str:
    """Return a filesystem-friendly suffix for one ridge strength."""
    ridge_text = f"{float(ridge_strength):.0e}"
    mantissa, exponent = ridge_text.split("e")
    exponent = exponent.lstrip("+")
    if exponent.startswith("-0"):
        exponent = f"-{exponent[2:]}"
    elif exponent.startswith("0"):
        exponent = exponent[1:]
    return f"ridge_{mantissa}e{exponent}"


def empirical_p_values(
    observed: np.ndarray,
    null_samples: np.ndarray,
    *,
    higher_is_better: bool,
) -> np.ndarray:
    """Return one-sided empirical p-values per unit with inclusive ties."""
    observed = np.asarray(observed, dtype=float)
    null_samples = np.asarray(null_samples, dtype=float)
    if null_samples.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)

    valid = np.isfinite(null_samples) & np.isfinite(observed[None, :])
    equal_with_tolerance = np.isclose(
        null_samples,
        observed[None, :],
        rtol=EMPIRICAL_P_VALUE_RTOL,
        atol=EMPIRICAL_P_VALUE_ATOL,
    )
    if higher_is_better:
        extreme = (null_samples > observed[None, :]) | equal_with_tolerance
    else:
        extreme = (null_samples < observed[None, :]) | equal_with_tolerance

    counts = np.sum(extreme & valid, axis=0)
    totals = np.sum(valid, axis=0)
    return np.where(totals > 0, (counts + 1) / (totals + 1), np.nan)


def summarize_ripple_metric_against_shuffle(
    real_folds: Any,
    shuffle_folds: Any,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Summarize one ripple metric and its shuffle null at the unit level."""
    real_folds_array = _as_2d_float(real_folds)
    shuffle_folds_array = _as_3d_float(shuffle_folds)

    real_mean = np.nanmean(real_folds_array, axis=0)
    real_sem = nansem(real_folds_array, axis=0)

    if shuffle_folds_array.shape[1] == 0:
        unit_null_samples = np.empty((0, real_folds_array.shape[1]), dtype=float)
        shuffle_mean = np.full(real_mean.shape, np.nan, dtype=float)
        shuffle_sd = np.full(real_mean.shape, np.nan, dtype=float)
        unit_p_value = np.full(real_mean.shape, np.nan, dtype=float)
    else:
        unit_null_samples = np.nanmean(shuffle_folds_array, axis=0)
        shuffle_mean = np.nanmean(unit_null_samples, axis=0)
        shuffle_sd = np.nanstd(unit_null_samples, axis=0, ddof=0)
        unit_p_value = empirical_p_values(
            real_mean,
            unit_null_samples,
            higher_is_better=higher_is_better,
        )

    return {
        "real_mean": real_mean,
        "real_sem": real_sem,
        "shuffle_mean": shuffle_mean,
        "shuffle_sd": shuffle_sd,
        "unit_p_value": unit_p_value,
    }


def build_metric_figure_data(
    results: dict[str, Any],
    *,
    metric_name: str,
) -> dict[str, np.ndarray]:
    """Return the per-unit arrays needed for one metric-centric figure."""
    summary = summarize_ripple_metric_against_shuffle(
        results[f"{metric_name}_ripple_folds"],
        results[f"{metric_name}_ripple_shuff_folds"],
        higher_is_better=HIGHER_IS_BETTER_BY_METRIC[metric_name],
    )
    return {
        "ripple_values": np.asarray(summary["real_mean"], dtype=float),
        "ripple_p_value": np.asarray(summary["unit_p_value"], dtype=float),
    }


def build_epoch_fit_dataset(
    results: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    epoch: str,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build one epoch-level ripple GLM fit dataset for NetCDF export."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save ripple GLM fits as NetCDF."
        ) from exc

    metric_names = ("pseudo_r2", "mae", "devexp", "bits_per_spike")
    unit_ids = np.asarray(results["v1_unit_ids"])
    ca1_unit_ids = np.asarray(results["ca1_unit_ids"])
    n_folds = int(_as_2d_float(results["pseudo_r2_ripple_folds"]).shape[0])
    n_shuffles = int(_as_3d_float(results["pseudo_r2_ripple_shuff_folds"]).shape[1])
    n_samples = int(_as_1d_float(results["ripple_window_start_s"]).shape[0])
    coef_ca1_unit_ids = np.asarray(results["coef_ca1_unit_ids"])
    coef_ca1_full_all = _as_2d_float(results["coef_ca1_full_all"])
    coef_intercept_full_all = _as_1d_float(results["coef_intercept_full_all"])
    ripple_start_time_s = _as_1d_float(
        results.get("ripple_start_time_s", results["ripple_window_start_s"])
    )
    ripple_window_start_s = _as_1d_float(results["ripple_window_start_s"])
    ripple_window_end_s = _as_1d_float(results["ripple_window_end_s"])
    ripple_fold_index = np.asarray(results["ripple_fold_index"], dtype=np.int32).ravel()
    ripple_observed_count_oof = _as_2d_float(results["ripple_observed_count_oof"])
    ripple_predicted_count_oof = _as_2d_float(results["ripple_predicted_count_oof"])

    attrs = {
        "schema_version": "6",
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
        "model_direction": f"{SOURCE_REGION}_to_{TARGET_REGION}",
        "n_ripples": int(results["n_ripples"]),
        "ripple_selection_mode": str(fit_parameters["ripple_selection_mode"]),
        "n_ripples_before_selection": int(fit_parameters["n_ripples_before_selection"]),
        "n_ripples_removed_by_selection": int(fit_parameters["n_ripples_removed_by_selection"]),
        "n_ripples_after_selection": int(fit_parameters["n_ripples_after_selection"]),
        "ripple_window_s": float(fit_parameters["ripple_window_s"]),
        "ripple_window_offset_s": float(
            fit_parameters.get("ripple_window_offset_s", DEFAULT_RIPPLE_WINDOW_OFFSET_S)
        ),
        "n_units": int(len(unit_ids)),
        "n_ca1_units": int(len(ca1_unit_ids)),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        "coef_ca1_full_all_space": "preprocessed_predictor",
        "coef_ca1_full_all_preprocess_json": json.dumps(
            {
                "center": True,
                "scale": True,
                "divide_by_sqrt_n_features": True,
                "clip_abs": 10.0,
            },
            sort_keys=True,
        ),
    }

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "ca1_unit_id": (("source_unit",), ca1_unit_ids),
        "coef_ca1_unit_id": (("coef_source_unit",), coef_ca1_unit_ids),
        "ripple_start_time_s": (("sample",), ripple_start_time_s),
        "ripple_window_start_s": (("sample",), ripple_window_start_s),
        "ripple_window_end_s": (("sample",), ripple_window_end_s),
        "ripple_fold_index": (("sample",), ripple_fold_index),
        "ripple_observed_count_oof": (("sample", "unit"), ripple_observed_count_oof),
        "ripple_predicted_count_oof": (("sample", "unit"), ripple_predicted_count_oof),
        "coef_ca1_full_all": (("coef_source_unit", "unit"), coef_ca1_full_all),
        "coef_intercept_full_all": (("unit",), coef_intercept_full_all),
    }
    for metric_name in metric_names:
        ripple_folds = _as_2d_float(results[f"{metric_name}_ripple_folds"])
        ripple_shuffle_folds = _as_3d_float(results[f"{metric_name}_ripple_shuff_folds"])
        ripple_summary = summarize_ripple_metric_against_shuffle(
            ripple_folds,
            ripple_shuffle_folds,
            higher_is_better=HIGHER_IS_BETTER_BY_METRIC[metric_name],
        )

        data_vars[f"{metric_name}_ripple_folds"] = (("fold", "unit"), ripple_folds)
        data_vars[f"{metric_name}_ripple_shuff_folds"] = (
            ("fold", "shuffle", "unit"),
            ripple_shuffle_folds,
        )
        data_vars[f"ripple_{metric_name}_mean"] = (
            ("unit",),
            np.asarray(ripple_summary["real_mean"], dtype=float),
        )
        data_vars[f"ripple_{metric_name}_sem"] = (
            ("unit",),
            np.asarray(ripple_summary["real_sem"], dtype=float),
        )
        data_vars[f"ripple_{metric_name}_shuffle_mean"] = (
            ("unit",),
            np.asarray(ripple_summary["shuffle_mean"], dtype=float),
        )
        data_vars[f"ripple_{metric_name}_shuffle_sd"] = (
            ("unit",),
            np.asarray(ripple_summary["shuffle_sd"], dtype=float),
        )
        data_vars[f"ripple_{metric_name}_p_value"] = (
            ("unit",),
            np.asarray(ripple_summary["unit_p_value"], dtype=float),
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "sample": np.arange(n_samples, dtype=int),
            "fold": np.arange(n_folds, dtype=int),
            "shuffle": np.arange(n_shuffles, dtype=int),
            "unit": unit_ids,
            "source_unit": np.arange(len(ca1_unit_ids), dtype=int),
            "coef_source_unit": np.arange(len(coef_ca1_unit_ids), dtype=int),
        },
        attrs=attrs,
    )


def _get_pyplot():
    """Return pyplot configured for headless script execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_summary_axis(
    *,
    ax: Any,
    ripple_values: Any,
    ripple_p_value: Any,
    metric_label: str,
) -> None:
    """Plot one metric-centric scatter panel as effect size versus significance."""
    ripple_array = _as_1d_float(ripple_values)
    p_value_array = _as_1d_float(ripple_p_value)
    sig_ax = ax

    valid = np.isfinite(ripple_array) & np.isfinite(p_value_array)
    if np.any(valid):
        effect_sizes = ripple_array[valid]
        neglog10_p = -np.log10(np.clip(p_value_array[valid], 1e-12, 1.0))
        sig_ax.scatter(
            effect_sizes,
            neglog10_p,
            s=18,
            alpha=0.7,
            color=RIPPLE_FIGURE_COLOR,
            label="Units",
        )
        sig_ax.axhline(
            -np.log10(0.05),
            linestyle="--",
            linewidth=1,
            color="black",
            label="p = 0.05",
        )
        frac_sig = float(np.mean(p_value_array[valid] < 0.05))
        sig_ax.text(
            0.98,
            0.02,
            f"n={int(np.sum(valid))}\nfrac p<0.05 = {frac_sig:.3f}",
            transform=sig_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )
    else:
        sig_ax.text(0.5, 0.5, "No finite ripple p-values", ha="center", va="center")
        sig_ax.axhline(
            -np.log10(0.05),
            linestyle="--",
            linewidth=1,
            color="black",
            label="p = 0.05",
        )
    sig_ax.set_xlabel(f"Ripple {metric_label} (mean over folds)")
    sig_ax.set_ylabel(r"$-\log_{10}(p)$ (shuffle)")
    sig_ax.set_title("Ripple effect size vs significance")


def plot_epoch_metric_summary(
    *,
    metric_panels: list[dict[str, Any]],
    animal_name: str,
    date: str,
    epoch: str,
    ridge_strength: float,
    out_path: Path,
) -> Path:
    """Plot all metric-centric scatter panels into one combined figure."""
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    for ax, metric_panel in zip(axes.flat, metric_panels, strict=False):
        plot_metric_summary_axis(
            ax=ax,
            ripple_values=metric_panel["ripple_values"],
            ripple_p_value=metric_panel["ripple_p_value"],
            metric_label=metric_panel["metric_label"],
        )

    fig.suptitle(
        f"{animal_name} {date} {epoch} ripple metrics ridge={ridge_strength:.1e}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_top_deviance_predicted_vs_observed(
    *,
    results: dict[str, Any],
    animal_name: str,
    date: str,
    epoch: str,
    ridge_strength: float,
    out_path: Path,
) -> Path:
    """Plot held-out observed versus predicted counts for top deviance-explained units."""
    plt = _get_pyplot()
    devexp = np.asarray(build_metric_figure_data(results, metric_name="devexp")["ripple_values"])
    observed = _as_2d_float(results["ripple_observed_count_oof"])
    predicted = _as_2d_float(results["ripple_predicted_count_oof"])
    unit_ids = np.asarray(results["v1_unit_ids"])

    finite_unit_indices = np.flatnonzero(np.isfinite(devexp))
    top_unit_indices = finite_unit_indices[np.argsort(devexp[finite_unit_indices])[::-1][:10]]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    axes_flat = axes.ravel()

    if top_unit_indices.size == 0:
        for ax in axes_flat:
            ax.axis("off")
        fig.text(0.5, 0.5, "No finite deviance explained values.", ha="center", va="center")
    else:
        for panel_index, ax in enumerate(axes_flat):
            if panel_index >= top_unit_indices.size:
                ax.axis("off")
                continue

            unit_index = int(top_unit_indices[panel_index])
            observed_unit = observed[:, unit_index]
            predicted_unit = predicted[:, unit_index]
            valid = np.isfinite(observed_unit) & np.isfinite(predicted_unit)

            if np.any(valid):
                ax.scatter(
                    observed_unit[valid],
                    predicted_unit[valid],
                    s=18,
                    alpha=0.7,
                    color=RIPPLE_FIGURE_COLOR,
                )
                max_value = float(
                    max(
                        np.nanmax(observed_unit[valid]),
                        np.nanmax(predicted_unit[valid]),
                        1.0,
                    )
                )
                ax.plot([0.0, max_value], [0.0, max_value], linestyle="--", linewidth=1, color="black")
                ax.set_xlim(0.0, max_value)
                ax.set_ylim(0.0, max_value)
            else:
                ax.text(0.5, 0.5, "No finite held-out values", ha="center", va="center")

            ax.set_title(f"V1 unit {int(unit_ids[unit_index])}\ndevexp={float(devexp[unit_index]):.3f}")
            ax.set_xlabel("Observed spike count")
            ax.set_ylabel("Predicted spike count")

    fig.suptitle(
        f"{animal_name} {date} {epoch} top devexp units ridge={ridge_strength:.1e}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_epoch_figures(
    *,
    results: dict[str, Any],
    fig_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_window_s: float,
    ripple_selection_suffix: str,
    ridge_strength: float,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> list[Path]:
    """Save the configured epoch summary figure."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    ripple_window_suffix = format_ripple_window_suffix(
        ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    ridge_strength_suffix = format_ridge_strength_suffix(ridge_strength)
    metric_specs = [
        ("pseudo_r2", "Pseudo R^2"),
        ("mae", "MAE"),
        ("devexp", "Deviance explained"),
        ("bits_per_spike", "Bits/spike"),
    ]

    metric_panels = []
    for metric_name, metric_label in metric_specs:
        metric_data = build_metric_figure_data(results, metric_name=metric_name)
        metric_panels.append(
            {
                "metric_name": metric_name,
                "metric_label": metric_label,
                "ripple_values": metric_data["ripple_values"],
                "ripple_p_value": metric_data["ripple_p_value"],
            }
        )

    out_path = fig_dir / (
        f"{epoch}_{ripple_window_suffix}_{ripple_selection_suffix}_"
        f"{ridge_strength_suffix}_samplewise_metrics_summary.png"
    )
    predicted_vs_observed_out_path = fig_dir / (
        f"{epoch}_{ripple_window_suffix}_{ripple_selection_suffix}_"
        f"{ridge_strength_suffix}_samplewise_top10_devexp_observed_vs_predicted.png"
    )
    return [
        plot_epoch_metric_summary(
            metric_panels=metric_panels,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            ridge_strength=ridge_strength,
            out_path=out_path,
        ),
        plot_top_deviance_predicted_vs_observed(
            results=results,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            ridge_strength=ridge_strength,
            out_path=predicted_vs_observed_out_path,
        ),
    ]


def main() -> None:
    """Run the ripple GLM CLI."""
    args = parse_arguments()
    validate_arguments(args)
    _print_progress("1/5", "Validated CLI arguments.")
    configure_jax_environment(args.cuda_visible_devices)
    if args.cuda_visible_devices is not None:
        print(f"Setting CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r} for ripple GLM.")

    _print_progress(
        "2/5",
        f"Loading session data for {args.animal_name} {args.date}.",
    )
    session = prepare_ripple_glm_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    _print_progress("2/5", "Loaded spikes, ripple tables, and epoch intervals.")
    requested_epochs = validate_epochs(session["epoch_tags"], args.epochs)
    selected_epochs = restrict_epochs_to_ripple_event_epochs(
        requested_epochs,
        ripple_event_epochs=session["ripple_event_epochs"],
        ripple_event_source=session["sources"]["ripple_events"],
    )
    validate_selected_epochs_across_sources(
        selected_epochs,
        source_epochs={
            "ephys_timestamps": session["timestamps_ephys_by_epoch"],
            "ripple_events": session["ripple_tables"],
            "epoch_intervals": session["epoch_intervals"],
        },
    )
    _print_progress(
        "3/5",
        f"Selected {len(selected_epochs)} ripple event epoch(s): {selected_epochs!r}",
    )

    analysis_path = session["analysis_path"]
    data_dir = analysis_path / "ripple_glm"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = analysis_path / "figs" / "ripple_glm"
    ridge_strengths = resolve_ridge_strengths(args)
    ripple_selection_mode = resolve_ripple_selection_mode(args)
    _print_progress(
        "4/5",
        f"Preparing outputs under {analysis_path} with ridge strengths {ridge_strengths!r}.",
    )
    print(f"Sweeping ridge strengths: {ridge_strengths!r}")
    run_parameters = {
        "animal_name": args.animal_name,
        "date": args.date,
        "data_root": str(args.data_root),
        "epochs": list(selected_epochs),
        "ripple_window_s": float(args.ripple_window_s),
        "ripple_window_offset_s": float(args.ripple_window_offset_s),
        "ripple_selection_mode": ripple_selection_mode,
        "remove_duplicate_ripples": bool(args.remove_duplicate_ripples),
        "keep_single_ripple_windows": bool(args.keep_single_ripple_windows),
        "min_spikes_per_ripple": float(args.min_spikes_per_ripple),
        "min_ca1_spikes_per_ripple": float(args.min_ca1_spikes_per_ripple),
        "n_splits": int(args.n_splits),
        "n_shuffles_ripple": int(args.n_shuffles_ripple),
        "ridge_strengths": list(ridge_strengths),
        "shuffle_seed": int(args.shuffle_seed),
        "maxiter": int(args.maxiter),
        "tol": float(args.tol),
        "cuda_visible_devices": args.cuda_visible_devices,
    }

    saved_datasets: list[Path] = []
    saved_figures: list[Path] = []
    skipped_epochs: list[dict[str, Any]] = []
    ripple_window_suffix = format_ripple_window_suffix(
        args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
    )
    ripple_selection_suffix = format_ripple_selection_suffix(ripple_selection_mode)

    if ripple_selection_mode == RIPPLE_SELECTION_MODE_DEDUPED:
        print(
            "Removing duplicate ripples whose starts fall inside the configured "
            "ripple window of the previous kept ripple."
        )
        print(
            "Outputs for duplicate-removed fits will include "
            f"'{ripple_selection_suffix}' in their filenames."
        )
    elif ripple_selection_mode == RIPPLE_SELECTION_MODE_SINGLE:
        print(
            "Keeping only fixed ripple windows whose interior contains a single "
            "ripple start."
        )
        print(
            "Outputs for single-ripple-window fits will include "
            f"'{ripple_selection_suffix}' in their filenames."
        )
    else:
        print(
            "Outputs for default fits will include "
            f"'{ripple_selection_suffix}' in their filenames."
        )

    for epoch in selected_epochs:
        _print_progress("5/5", f"Starting epoch {epoch}.")
        ripple_table = session["ripple_tables"][epoch].copy()
        total_ripples_before_selection = int(len(ripple_table))
        removed_ripples_by_selection = 0
        if ripple_selection_mode == RIPPLE_SELECTION_MODE_DEDUPED:
            ripple_table, keep_mask = remove_duplicate_ripples(
                ripple_table,
                ripple_window_s=args.ripple_window_s,
                ripple_window_offset_s=args.ripple_window_offset_s,
            )
            removed_ripples_by_selection = int(np.size(keep_mask) - np.sum(keep_mask))
            print(
                f"{epoch}: removed {removed_ripples_by_selection}, kept {len(ripple_table)}, "
                f"total {total_ripples_before_selection} because they started inside the "
                "previous kept ripple's analysis window."
            )
        elif ripple_selection_mode == RIPPLE_SELECTION_MODE_SINGLE:
            ripple_table, keep_mask = keep_single_ripple_windows(
                ripple_table,
                ripple_window_s=args.ripple_window_s,
                ripple_window_offset_s=args.ripple_window_offset_s,
            )
            removed_ripples_by_selection = int(np.size(keep_mask) - np.sum(keep_mask))
            print(
                f"{epoch}: removed {removed_ripples_by_selection}, kept {len(ripple_table)}, "
                f"total {total_ripples_before_selection} because another ripple start fell "
                "inside the fixed analysis window."
            )
        try:
            prepared_epoch = _prepare_ripple_glm_epoch_inputs(
                epoch,
                spikes=session["spikes_by_region"],
                ripple_table=ripple_table,
                min_spikes_per_ripple=args.min_spikes_per_ripple,
                min_ca1_spikes_per_ripple=args.min_ca1_spikes_per_ripple,
                ripple_window_s=args.ripple_window_s,
                ripple_window_offset_s=args.ripple_window_offset_s,
                n_splits=args.n_splits,
            )
        except ValueError as exc:
            skipped_epochs.append(
                {
                    "epoch": epoch,
                    "reason": str(exc),
                }
            )
            print(f"Skipping {args.animal_name} {args.date} {epoch}: {exc}")
            continue
        for ridge_strength in ridge_strengths:
            ridge_strength_suffix = format_ridge_strength_suffix(ridge_strength)
            _print_progress(
                epoch,
                "Running fit for "
                f"ridge_strength={ridge_strength:.1e} "
                f"with {len(ripple_table)} modeled ripple(s).",
            )
            try:
                results = _fit_ripple_glm_on_prepared_epoch(
                    epoch,
                    prepared_epoch=prepared_epoch,
                    n_shuffles_ripple=args.n_shuffles_ripple,
                    shuffle_seed=args.shuffle_seed,
                    ripple_window_s=args.ripple_window_s,
                    ripple_window_offset_s=args.ripple_window_offset_s,
                    ridge_strength=ridge_strength,
                    maxiter=args.maxiter,
                    tol=args.tol,
                )
                results["min_spikes_per_ripple"] = float(args.min_spikes_per_ripple)
                results["min_ca1_spikes_per_ripple"] = float(args.min_ca1_spikes_per_ripple)
            except ValueError as exc:
                skipped_epochs.append(
                    {
                        "epoch": epoch,
                        "ridge_strength": float(ridge_strength),
                        "reason": str(exc),
                    }
                )
                print(
                    f"Skipping {args.animal_name} {args.date} {epoch} "
                    f"(ridge_strength={ridge_strength:.1e}): {exc}"
                )
                continue
            except Exception as exc:
                skipped_epochs.append(
                    {
                        "epoch": epoch,
                        "ridge_strength": float(ridge_strength),
                        "reason": "fit failed",
                        "error": str(exc),
                    }
                )
                print(
                    f"Skipping {args.animal_name} {args.date} {epoch} "
                    f"(ridge_strength={ridge_strength:.1e}): fit failed: {exc}"
                )
                continue

            fit_parameters = dict(run_parameters)
            fit_parameters["ridge_strength"] = float(ridge_strength)
            fit_parameters["n_ripples_before_selection"] = int(total_ripples_before_selection)
            fit_parameters["n_ripples_removed_by_selection"] = int(removed_ripples_by_selection)
            fit_parameters["n_ripples_after_selection"] = int(len(ripple_table))
            _print_progress(epoch, "Building xarray dataset for export.")
            fit_dataset = build_epoch_fit_dataset(
                results,
                animal_name=args.animal_name,
                date=args.date,
                epoch=epoch,
                sources=session["sources"],
                fit_parameters=fit_parameters,
            )
            result_path = (
                data_dir
                / (
                    f"{epoch}_{ripple_window_suffix}_{ripple_selection_suffix}_"
                    f"{ridge_strength_suffix}_samplewise_ripple_glm.nc"
                )
            )
            _print_progress(epoch, f"Saving NetCDF to {result_path}.")
            fit_dataset.to_netcdf(result_path)
            saved_datasets.append(result_path)
            _print_progress(epoch, "Saving summary figure.")
            saved_figures.extend(
                save_epoch_figures(
                    results=results,
                    fig_dir=fig_dir,
                    animal_name=args.animal_name,
                    date=args.date,
                    epoch=epoch,
                    ripple_window_s=args.ripple_window_s,
                    ripple_window_offset_s=args.ripple_window_offset_s,
                    ripple_selection_suffix=ripple_selection_suffix,
                    ridge_strength=ridge_strength,
                )
            )
            _print_progress(epoch, f"Finished ridge_strength={ridge_strength:.1e}.")

    if not saved_datasets:
        raise RuntimeError(
            "All selected epochs were skipped. "
            f"Epoch reasons: {skipped_epochs!r}"
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.ripple.ripple_glm",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "epochs": selected_epochs,
            "ripple_window_s": args.ripple_window_s,
            "ripple_window_offset_s": args.ripple_window_offset_s,
            "ripple_selection_mode": ripple_selection_mode,
            "remove_duplicate_ripples": args.remove_duplicate_ripples,
            "keep_single_ripple_windows": args.keep_single_ripple_windows,
            "min_spikes_per_ripple": args.min_spikes_per_ripple,
            "min_ca1_spikes_per_ripple": args.min_ca1_spikes_per_ripple,
            "n_splits": args.n_splits,
            "n_shuffles_ripple": args.n_shuffles_ripple,
            "ridge_strengths": ridge_strengths,
            "shuffle_seed": args.shuffle_seed,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "cuda_visible_devices": args.cuda_visible_devices,
            "model_direction": f"{SOURCE_REGION}_to_{TARGET_REGION}",
        },
        outputs={
            "selected_epochs": selected_epochs,
            "saved_datasets": saved_datasets,
            "saved_figures": saved_figures,
            "skipped_epochs": skipped_epochs,
        },
    )
    _print_progress(
        "done",
        f"Saved {len(saved_datasets)} dataset(s), {len(saved_figures)} figure(s), and run log.",
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
