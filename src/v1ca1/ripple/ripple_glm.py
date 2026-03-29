from __future__ import annotations

"""Fit ripple-triggered population GLMs for one session.

This CLI fits the legacy CA1-to-V1 ripple population GLM workflow for one
session, one epoch at a time. Each ripple window is converted into one sample
whose predictors are CA1 unit spike counts and whose targets are V1 unit spike
counts. The model is a ridge-regularized Poisson population GLM trained on
ripple windows, evaluated on held-out ripple windows with contiguous
cross-validation folds, and then applied to paired pre-ripple windows from the
same epoch.

Performance is summarized per V1 unit with pseudo-R^2, mean absolute error
(MAE), deviance explained, and bits/spike. Ripple metrics also include an
empirical shuffle null built by refitting after independently shuffling each
V1 unit's ripple responses, which yields per-unit shuffle means, shuffle
standard deviations, and one-sided p-values.

Successful fits are saved under the session analysis directory as one
NetCDF-backed xarray dataset per epoch and ridge strength in `ripple_glm/`.
Each dataset stores the raw ripple fold arrays, ripple shuffle arrays,
pre-ripple metrics, per-unit ripple summary variables, unit IDs, and fit
metadata. The script also writes four summary figures per epoch and ridge
strength in `figs/ripple_glm/`, one each for pseudo-R^2, MAE, deviance
explained, and bits/spike. Each figure has a left panel with overlaid ripple
and pre-ripple unit distributions plus compact boxplots, and a right panel
with ripple effect size versus `-log10(shuffle p)`. A JSON run log is written
under `v1ca1_log/`.

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
DEFAULT_PRE_BUFFER_S = 0.02
DEFAULT_PRE_EXCLUDE_GUARD_S = 0.05
DEFAULT_MIN_SPIKES_PER_RIPPLE = 0.1
DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE = 0.0
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES_RIPPLE = 100
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_RIDGE_STRENGTH_SWEEP = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_MAXITER = 6000
DEFAULT_TOL = 1e-7

HIGHER_IS_BETTER_BY_METRIC = {
    "pseudo_r2": True,
    "mae": False,
    "devexp": True,
    "bits_per_spike": True,
}

EMPIRICAL_P_VALUE_RTOL = 1e-12
EMPIRICAL_P_VALUE_ATOL = 1e-12

RIPPLE_FIGURE_COLOR = "#1f77b4"
PRE_FIGURE_COLOR = "#d95f02"
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
        "--pre-window-s",
        type=float,
        help="Optional fixed pre-ripple window length in seconds. Defaults to --ripple-window-s.",
    )
    parser.add_argument(
        "--pre-buffer-s",
        type=float,
        default=DEFAULT_PRE_BUFFER_S,
        help=f"Gap between the pre window and ripple start in seconds. Default: {DEFAULT_PRE_BUFFER_S}",
    )
    parser.add_argument(
        "--exclude-ripples",
        action="store_true",
        help="Exclude ripple intervals (plus the configured guard) from pre windows.",
    )
    parser.add_argument(
        "--pre-exclude-guard-s",
        type=float,
        default=DEFAULT_PRE_EXCLUDE_GUARD_S,
        help=(
            "Guard width in seconds when excluding ripples from pre windows. "
            f"Default: {DEFAULT_PRE_EXCLUDE_GUARD_S}"
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
    if args.ripple_window_s <= 0:
        raise ValueError("--ripple-window-s must be positive.")
    if args.pre_window_s is not None and args.pre_window_s <= 0:
        raise ValueError("--pre-window-s must be positive when provided.")
    if args.pre_buffer_s < 0:
        raise ValueError("--pre-buffer-s must be non-negative.")
    if args.pre_exclude_guard_s < 0:
        raise ValueError("--pre-exclude-guard-s must be non-negative.")
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


def make_preripple_ep(
    ripple_ep: "nap.IntervalSet",
    epoch_ep: "nap.IntervalSet",
    *,
    window_s: float | None = None,
    buffer_s: float = DEFAULT_PRE_BUFFER_S,
    exclude_ripples: bool = False,
    exclude_ripple_guard_s: float = DEFAULT_PRE_EXCLUDE_GUARD_S,
) -> "nap.IntervalSet":
    """Create the paired pre-ripple IntervalSet used by the legacy workflow."""
    import pynapple as nap

    r_start = np.asarray(ripple_ep.start, dtype=float)
    r_end = np.asarray(ripple_ep.end, dtype=float)

    if window_s is None:
        duration = r_end - r_start
    else:
        duration = np.full_like(r_start, float(window_s))

    pre_start = r_start - duration - float(buffer_s)
    pre_end = r_start - float(buffer_s)

    keep = pre_end > pre_start
    pre_start = pre_start[keep]
    pre_end = pre_end[keep]

    if pre_start.size == 0:
        return nap.IntervalSet(
            start=np.array([], dtype=float),
            end=np.array([], dtype=float),
            time_units="s",
        )

    pre_ep = nap.IntervalSet(start=pre_start, end=pre_end, time_units="s").intersect(epoch_ep)

    if exclude_ripples and len(pre_ep) > 0 and len(ripple_ep) > 0:
        ripple_exclusion = nap.IntervalSet(
            start=np.asarray(ripple_ep.start, dtype=float) - float(exclude_ripple_guard_s),
            end=np.asarray(ripple_ep.end, dtype=float) + float(exclude_ripple_guard_s),
            time_units="s",
        ).intersect(epoch_ep)
        pre_ep = pre_ep.set_diff(ripple_exclusion)

    return pre_ep


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


def fit_ripple_glm_train_on_ripple_predict_pre(
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
    pre_window_s: float | None = None,
    pre_buffer_s: float = DEFAULT_PRE_BUFFER_S,
    exclude_ripples: bool = False,
    pre_exclude_guard_s: float = DEFAULT_PRE_EXCLUDE_GUARD_S,
    n_splits: int = DEFAULT_N_SPLITS,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    """Fit the legacy ripple GLM workflow for one epoch."""
    import nemos as nmo
    nemos_version, solver_name = _resolve_nemos_population_glm_solver(nmo)
    print(_format_nemos_solver_selection_message(nemos_version, solver_name))
    import pynapple as nap

    rng = np.random.default_rng(shuffle_seed)
    ripple_starts = np.asarray(ripple_table["start_time"], dtype=float)
    ripple_ends = np.asarray(ripple_table["end_time"], dtype=float)

    if ripple_window_s is None:
        ripple_ep = nap.IntervalSet(start=ripple_starts, end=ripple_ends, time_units="s")
    else:
        ripple_ep = nap.IntervalSet(
            start=ripple_starts,
            end=ripple_starts + float(ripple_window_s),
            time_units="s",
        )

    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=epoch_interval,
        window_s=pre_window_s,
        buffer_s=pre_buffer_s,
        exclude_ripples=exclude_ripples,
        exclude_ripple_guard_s=pre_exclude_guard_s,
    )

    X_r = np.asarray(spikes[SOURCE_REGION].count(ep=ripple_ep), dtype=np.float64)
    y_r = np.asarray(spikes[TARGET_REGION].count(ep=ripple_ep), dtype=np.float64)
    X_p = np.asarray(spikes[SOURCE_REGION].count(ep=pre_ep), dtype=np.float64)
    y_p = np.asarray(spikes[TARGET_REGION].count(ep=pre_ep), dtype=np.float64)

    n_ripples = int(X_r.shape[0])
    n_pre = int(X_p.shape[0])
    skip_reason = get_epoch_skip_reason(n_ripples=n_ripples, n_splits=n_splits)
    if skip_reason is not None:
        raise ValueError(skip_reason)

    for name, values in (("X_r", X_r), ("y_r", y_r), ("X_p", X_p), ("y_p", y_p)):
        bad_value_info = _first_nonfinite_info(values)
        if bad_value_info is not None:
            raise ValueError(f"Non-finite values found in {name}: {bad_value_info}")

    keep_y = (y_r.sum(axis=0) / max(n_ripples, 1)) >= float(min_spikes_per_ripple)
    y_r = y_r[:, keep_y]
    y_p = y_p[:, keep_y]

    v1_unit_ids = np.asarray(list(spikes[TARGET_REGION].keys()))
    kept_v1_unit_ids = v1_unit_ids[keep_y]
    keep_x_ripple = (X_r.sum(axis=0) / max(n_ripples, 1)) >= float(min_ca1_spikes_per_ripple)
    X_r = X_r[:, keep_x_ripple]
    X_p = X_p[:, keep_x_ripple]
    ca1_unit_ids = np.asarray(list(spikes[SOURCE_REGION].keys()))
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

    kfold = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    def _alloc_metric_array() -> np.ndarray:
        return np.full((n_splits, n_cells), np.nan, dtype=np.float32)

    pseudo_r2_ripple = _alloc_metric_array()
    mae_ripple = _alloc_metric_array()
    devexp_ripple = _alloc_metric_array()
    bits_per_spike_ripple = _alloc_metric_array()

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
    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(X_r)):
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
            if (shuffle_index + 1) % 5 == 0:
                _clear_jax_caches()
                gc.collect()

        del glm, lam_r
        _clear_jax_caches()
        gc.collect()

    X_r_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(X_r, X_r)
    if X_r_all_pp.shape[1] == 0:
        raise ValueError(f"Epoch {epoch!r}: all CA1 features were near-constant.")

    pre_pseudo_r2 = np.full(n_cells, np.nan, dtype=np.float32)
    pre_mae = np.full(n_cells, np.nan, dtype=np.float32)
    pre_devexp = np.full(n_cells, np.nan, dtype=np.float32)
    pre_bits_per_spike = np.full(n_cells, np.nan, dtype=np.float32)

    if n_pre > 0:
        X_p_kept = X_p[:, keep_x_all]
        X_p_pp = (X_p_kept - mean_all[None, :]) / (std_all[None, :] + 1e-8)
        X_p_pp /= np.sqrt(max(X_r_all_pp.shape[1], 1))
        X_p_pp = np.clip(X_p_pp, -10.0, 10.0)

        glm_all = nmo.glm.PopulationGLM(
            solver_name=solver_name,
            regularizer="Ridge",
            regularizer_strength=float(ridge_strength),
            solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
        )
        glm_all.fit(X_r_all_pp, y_r)
        lam_pre = np.asarray(glm_all.predict(X_p_pp), dtype=np.float64)

        pre_pseudo_r2 = mcfadden_pseudo_r2_per_neuron(y_p, lam_pre, y_p).astype(np.float32)
        pre_mae = mae_per_neuron(y_p, lam_pre).astype(np.float32)
        pre_devexp = deviance_explained_per_neuron(
            y_test=y_p,
            lam_test=lam_pre,
            y_null_fit=y_p,
        ).astype(np.float32)
        pre_bits_per_spike = bits_per_spike_per_neuron(
            y_test=y_p,
            lam_test=lam_pre,
            y_null_fit=y_p,
        ).astype(np.float32)

        del glm_all, lam_pre
        _clear_jax_caches()
        gc.collect()

    results = {
        "epoch": epoch,
        "shuffle_seed": int(shuffle_seed),
        "min_spikes_per_ripple": float(min_spikes_per_ripple),
        "min_ca1_spikes_per_ripple": float(min_ca1_spikes_per_ripple),
        "n_splits": int(n_splits),
        "n_shuffles_ripple": int(n_shuffles_ripple),
        "ripple_window_s": None if ripple_window_s is None else float(ripple_window_s),
        "pre_window_s": None if pre_window_s is None else float(pre_window_s),
        "pre_buffer_s": float(pre_buffer_s),
        "pre_exclude_guard_s": float(pre_exclude_guard_s),
        "exclude_ripples": bool(exclude_ripples),
        "n_ripples": n_ripples,
        "n_pre": n_pre,
        "n_cells": n_cells,
        "n_ca1_cells": n_ca1_cells,
        "v1_unit_ids": kept_v1_unit_ids,
        "ca1_unit_ids": kept_ca1_unit_ids,
        "pseudo_r2_ripple_folds": pseudo_r2_ripple,
        "mae_ripple_folds": mae_ripple,
        "devexp_ripple_folds": devexp_ripple,
        "bits_per_spike_ripple_folds": bits_per_spike_ripple,
        "pseudo_r2_ripple_shuff_folds": pseudo_r2_ripple_shuff,
        "mae_ripple_shuff_folds": mae_ripple_shuff,
        "devexp_ripple_shuff_folds": devexp_ripple_shuff,
        "bits_per_spike_ripple_shuff_folds": bits_per_spike_ripple_shuff,
        "pseudo_r2_pre": pre_pseudo_r2,
        "mae_pre": pre_mae,
        "devexp_pre": pre_devexp,
        "bits_per_spike_pre": pre_bits_per_spike,
    }
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


def format_ripple_window_suffix(ripple_window_s: float) -> str:
    """Return a filesystem-friendly suffix for one ripple window length."""
    window_text = f"{float(ripple_window_s):.6f}".rstrip("0").rstrip(".")
    return f"rw_{window_text.replace('.', 'p')}s"


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
        "pre_values": _as_1d_float(results[f"{metric_name}_pre"]),
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

    attrs = {
        "schema_version": "1",
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
        "model_direction": f"{SOURCE_REGION}_to_{TARGET_REGION}",
        "n_ripples": int(results["n_ripples"]),
        "n_pre_windows": int(results["n_pre"]),
        "n_units": int(len(unit_ids)),
        "n_ca1_units": int(len(ca1_unit_ids)),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
    }

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "ca1_unit_id": (("source_unit",), ca1_unit_ids),
    }
    for metric_name in metric_names:
        ripple_folds = _as_2d_float(results[f"{metric_name}_ripple_folds"])
        ripple_shuffle_folds = _as_3d_float(results[f"{metric_name}_ripple_shuff_folds"])
        pre_values = _as_1d_float(results[f"{metric_name}_pre"])
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
        data_vars[f"{metric_name}_pre"] = (("unit",), pre_values)
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
            "fold": np.arange(n_folds, dtype=int),
            "shuffle": np.arange(n_shuffles, dtype=int),
            "unit": unit_ids,
            "source_unit": np.arange(len(ca1_unit_ids), dtype=int),
        },
        attrs=attrs,
    )


def _get_pyplot():
    """Return pyplot configured for headless script execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finite_values(values: Any) -> np.ndarray:
    """Return only the finite values from an array-like input."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _metric_histogram_bins(ripple_values: np.ndarray, pre_values: np.ndarray) -> np.ndarray:
    """Return shared histogram bins derived from ripple and pre values."""
    finite_combined = np.concatenate([_finite_values(ripple_values), _finite_values(pre_values)])
    if finite_combined.size == 0:
        return np.linspace(0.0, 1.0, 31)

    x_min = float(np.min(finite_combined))
    x_max = float(np.max(finite_combined))
    if x_min == x_max:
        pad = max(abs(x_min) * 0.05, 0.5)
        x_min -= pad
        x_max += pad
    return np.linspace(x_min, x_max, 31)


def plot_metric_summary(
    *,
    ripple_values: Any,
    pre_values: Any,
    ripple_p_value: Any,
    animal_name: str,
    date: str,
    epoch: str,
    ridge_strength: float,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot one metric-centric summary figure with distribution and significance panels."""
    plt = _get_pyplot()

    ripple_array = _as_1d_float(ripple_values)
    pre_array = _as_1d_float(pre_values)
    p_value_array = _as_1d_float(ripple_p_value)

    ripple_finite = _finite_values(ripple_array)
    pre_finite = _finite_values(pre_array)
    bins = _metric_histogram_bins(ripple_finite, pre_finite)
    x_limits = (float(bins[0]), float(bins[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    hist_ax, sig_ax = axes

    if ripple_finite.size:
        hist_ax.hist(
            ripple_finite,
            bins=bins,
            weights=np.full(ripple_finite.size, 1.0 / ripple_finite.size),
            alpha=0.55,
            color=RIPPLE_FIGURE_COLOR,
            label="Ripple",
            edgecolor="none",
        )
    if pre_finite.size:
        hist_ax.hist(
            pre_finite,
            bins=bins,
            weights=np.full(pre_finite.size, 1.0 / pre_finite.size),
            alpha=0.55,
            color=PRE_FIGURE_COLOR,
            label="Pre",
            edgecolor="none",
        )
    hist_ax.set_xlim(*x_limits)
    hist_ax.set_xlabel(metric_label)
    hist_ax.set_ylabel("Fraction of units")
    hist_ax.set_title("Ripple vs pre distributions")
    if ripple_finite.size or pre_finite.size:
        hist_ax.legend(loc="upper right")

    stats_text = (
        f"Ripple n={ripple_finite.size}, mean={np.nanmean(ripple_array):.4f}\n"
        f"Pre n={pre_finite.size}, mean={np.nanmean(pre_array):.4f}"
    )
    hist_ax.text(
        0.02,
        0.98,
        stats_text,
        transform=hist_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    box_ax = hist_ax.inset_axes([0.0, 0.82, 1.0, 0.18])
    box_data: list[np.ndarray] = []
    positions: list[int] = []
    labels: list[str] = []
    colors: list[str] = []
    if ripple_finite.size:
        box_data.append(ripple_finite)
        positions.append(2)
        labels.append("Ripple")
        colors.append(RIPPLE_FIGURE_COLOR)
    if pre_finite.size:
        box_data.append(pre_finite)
        positions.append(1)
        labels.append("Pre")
        colors.append(PRE_FIGURE_COLOR)
    if box_data:
        boxplot = box_ax.boxplot(
            box_data,
            vert=False,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        for patch, color in zip(boxplot["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in boxplot["medians"]:
            median.set_color("black")
            median.set_linewidth(1.3)
        box_ax.set_yticks(positions, labels)
        box_ax.set_xlim(*x_limits)
    else:
        box_ax.set_yticks([])
    box_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine_name in ("top", "right", "bottom"):
        box_ax.spines[spine_name].set_visible(False)
    box_ax.set_facecolor("none")

    valid = np.isfinite(ripple_array) & np.isfinite(p_value_array)
    if np.any(valid):
        effect_sizes = ripple_array[valid]
        neglog10_p = -np.log10(np.clip(p_value_array[valid], 1e-12, 1.0))
        sig_ax.scatter(effect_sizes, neglog10_p, s=18, alpha=0.7, color=RIPPLE_FIGURE_COLOR)
        sig_ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, color="black")
        frac_sig = float(np.mean(p_value_array[valid] < 0.05))
        sig_ax.text(
            0.98,
            0.05,
            f"frac p<0.05 = {frac_sig:.3f}",
            transform=sig_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )
    else:
        sig_ax.text(0.5, 0.5, "No finite ripple p-values", ha="center", va="center")
        sig_ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, color="black")
    sig_ax.set_xlabel(f"Ripple {metric_label} (mean over folds)")
    sig_ax.set_ylabel(r"$-\log_{10}(p)$ (shuffle)")
    sig_ax.set_title("Ripple effect size vs significance")

    fig.suptitle(
        f"{animal_name} {date} {epoch} {metric_label} ridge={ridge_strength:.1e}",
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
    ridge_strength: float,
) -> list[Path]:
    """Save the configured epoch summary figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    ripple_window_suffix = format_ripple_window_suffix(ripple_window_s)
    ridge_strength_suffix = format_ridge_strength_suffix(ridge_strength)
    metric_specs = [
        ("pseudo_r2", "Pseudo R^2"),
        ("mae", "MAE"),
        ("devexp", "Deviance explained"),
        ("bits_per_spike", "Bits/spike"),
    ]

    figure_paths: list[Path] = []
    for metric_name, metric_label in metric_specs:
        metric_data = build_metric_figure_data(results, metric_name=metric_name)
        figure_paths.append(
            plot_metric_summary(
                ripple_values=metric_data["ripple_values"],
                pre_values=metric_data["pre_values"],
                ripple_p_value=metric_data["ripple_p_value"],
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                ridge_strength=ridge_strength,
                metric_label=metric_label,
                out_path=fig_dir
                / f"{epoch}_{ripple_window_suffix}_{ridge_strength_suffix}_{metric_name}_summary.png",
            )
        )
    return figure_paths


def main() -> None:
    """Run the ripple GLM CLI."""
    args = parse_arguments()
    validate_arguments(args)
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Setting CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r} for ripple GLM.")

    session = prepare_ripple_glm_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
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

    analysis_path = session["analysis_path"]
    data_dir = analysis_path / "ripple_glm"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = analysis_path / "figs" / "ripple_glm"
    pre_window_s = args.pre_window_s if args.pre_window_s is not None else args.ripple_window_s
    ridge_strengths = resolve_ridge_strengths(args)
    print(f"Sweeping ridge strengths: {ridge_strengths!r}")
    run_parameters = {
        "animal_name": args.animal_name,
        "date": args.date,
        "data_root": str(args.data_root),
        "epochs": list(selected_epochs),
        "ripple_window_s": float(args.ripple_window_s),
        "pre_window_s": float(pre_window_s),
        "pre_buffer_s": float(args.pre_buffer_s),
        "exclude_ripples": bool(args.exclude_ripples),
        "pre_exclude_guard_s": float(args.pre_exclude_guard_s),
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
    ripple_window_suffix = format_ripple_window_suffix(args.ripple_window_s)

    for epoch in selected_epochs:
        ripple_table = session["ripple_tables"][epoch]
        for ridge_strength in ridge_strengths:
            ridge_strength_suffix = format_ridge_strength_suffix(ridge_strength)
            print(
                "Fitting ripple GLM for "
                f"{args.animal_name} {args.date} {epoch} with ridge_strength={ridge_strength:.1e}"
            )
            try:
                results = fit_ripple_glm_train_on_ripple_predict_pre(
                    epoch=epoch,
                    spikes=session["spikes_by_region"],
                    epoch_interval=session["epoch_intervals"][epoch],
                    ripple_table=ripple_table,
                    min_spikes_per_ripple=args.min_spikes_per_ripple,
                    min_ca1_spikes_per_ripple=args.min_ca1_spikes_per_ripple,
                    n_shuffles_ripple=args.n_shuffles_ripple,
                    shuffle_seed=args.shuffle_seed,
                    ripple_window_s=args.ripple_window_s,
                    pre_window_s=pre_window_s,
                    pre_buffer_s=args.pre_buffer_s,
                    exclude_ripples=args.exclude_ripples,
                    pre_exclude_guard_s=args.pre_exclude_guard_s,
                    n_splits=args.n_splits,
                    ridge_strength=ridge_strength,
                    maxiter=args.maxiter,
                    tol=args.tol,
                )
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
                / f"{epoch}_{ripple_window_suffix}_{ridge_strength_suffix}_ripple_glm.nc"
            )
            fit_dataset.to_netcdf(result_path)
            saved_datasets.append(result_path)
            saved_figures.extend(
                save_epoch_figures(
                    results=results,
                    fig_dir=fig_dir,
                    animal_name=args.animal_name,
                    date=args.date,
                    epoch=epoch,
                    ripple_window_s=args.ripple_window_s,
                    ridge_strength=ridge_strength,
                )
            )

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
            "pre_window_s": pre_window_s,
            "pre_buffer_s": args.pre_buffer_s,
            "exclude_ripples": args.exclude_ripples,
            "pre_exclude_guard_s": args.pre_exclude_guard_s,
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
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
