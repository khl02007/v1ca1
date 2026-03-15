from __future__ import annotations

"""Fit ripple-triggered population GLMs for one session.

This modernized CLI replaces the legacy ripple population GLM lab script with
validated session loading, explicit command-line arguments, per-epoch parquet
summaries, optional legacy `.npz` outputs, and run logging under the session
analysis directory.
"""

import argparse
import gc
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
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


TARGET_REGION = "v1"
SOURCE_REGION = "ca1"

DEFAULT_RIPPLE_WINDOW_S = 0.2
DEFAULT_PRE_BUFFER_S = 0.02
DEFAULT_PRE_EXCLUDE_GUARD_S = 0.05
DEFAULT_MIN_SPIKES_PER_RIPPLE = 0.1
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES_RIPPLE = 100
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_MAXITER = 6000
DEFAULT_TOL = 1e-7

HIGHER_IS_BETTER_BY_METRIC = {
    "pseudo_r2": True,
    "mae": False,
    "bits_per_spike": True,
}


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
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Ridge regularization strength. Default: {DEFAULT_RIDGE_STRENGTH}",
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
        "--save-legacy-npz",
        action="store_true",
        help="Also save the detailed legacy-style raw `.npz` result files.",
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
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.n_shuffles_ripple < 0:
        raise ValueError("--n-shuffles-ripple must be non-negative.")
    if args.ridge_strength < 0:
        raise ValueError("--ridge-strength must be non-negative.")
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


def _extract_interval_dataframe(intervals: Any) -> pd.DataFrame:
    """Return a dataframe-like view of a pynapple IntervalSet."""
    if hasattr(intervals, "as_dataframe"):
        interval_df = intervals.as_dataframe()
        if isinstance(interval_df, pd.DataFrame):
            return interval_df.copy()
    if hasattr(intervals, "_metadata"):
        metadata = intervals._metadata  # type: ignore[attr-defined]
        if isinstance(metadata, pd.DataFrame):
            return metadata.copy()
    return pd.DataFrame()


def _extract_epoch_metadata(intervals: Any) -> np.ndarray:
    """Extract the saved epoch labels from a pynapple IntervalSet."""
    try:
        epoch_values = intervals.get_info("epoch")
    except Exception:
        epoch_values = None

    if epoch_values is not None:
        epoch_array = np.asarray(epoch_values)
        if epoch_array.size:
            return epoch_array.astype(str)

    interval_df = _extract_interval_dataframe(intervals)
    if "epoch" in interval_df.columns:
        return np.asarray(interval_df["epoch"], dtype=str)

    raise ValueError("The ripple interval output does not contain saved epoch labels.")


def load_ripple_tables_from_interval_output(path: Path) -> dict[str, pd.DataFrame]:
    """Load ripple events from the modern flattened pynapple interval output."""
    import pynapple as nap

    if not path.exists():
        raise FileNotFoundError(f"Ripple interval output not found: {path}")

    interval_set = nap.load_file(path)
    epoch_values = _extract_epoch_metadata(interval_set)
    start_values = np.asarray(interval_set.start, dtype=float).ravel()
    end_values = np.asarray(interval_set.end, dtype=float).ravel()
    if start_values.shape != end_values.shape or start_values.size != epoch_values.size:
        raise ValueError(
            "The ripple interval output has mismatched start/end/epoch lengths: "
            f"{start_values.shape}, {end_values.shape}, {epoch_values.shape}."
        )

    flat_table = pd.DataFrame(
        {
            "epoch": epoch_values.astype(str),
            "start_time": start_values,
            "end_time": end_values,
        }
    )
    if flat_table.empty:
        return {}

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
    """Load ripple events, preferring the modern interval output."""
    interval_path = analysis_path / "ripple" / "ripple_times.npz"
    legacy_path = analysis_path / "ripple" / "Kay_ripple_detector.pkl"

    interval_error: Exception | None = None
    if interval_path.exists():
        try:
            return load_ripple_tables_from_interval_output(interval_path), "pynapple"
        except Exception as exc:
            interval_error = exc

    if legacy_path.exists():
        return load_ripple_tables_from_legacy_pickle(legacy_path), "pickle"

    if interval_error is not None:
        raise ValueError(
            f"Failed to load {interval_path} and no legacy pickle fallback was found."
        ) from interval_error
    raise FileNotFoundError(
        f"Could not find {interval_path} or {legacy_path} under {analysis_path}."
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


def save_results_npz(out_path: Path, results: dict[str, Any]) -> Path:
    """Save one raw legacy-style result payload as a compressed `.npz`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    to_save: dict[str, Any] = {}
    for key, value in results.items():
        if key == "fold_info":
            to_save[key] = np.array(value, dtype=object)
        elif value is None:
            to_save[key] = np.array(None, dtype=object)
        else:
            to_save[key] = value
    np.savez_compressed(out_path, **to_save)
    return out_path


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
) -> str | None:
    """Return a skip reason string for common weak-epoch cases."""
    if n_ripples < n_splits:
        return f"Not enough ripples for CV: n_ripples={n_ripples}, n_splits={n_splits}"
    if n_kept_v1_units is not None and n_kept_v1_units <= 0:
        return "No V1 units passed the minimum spikes-per-ripple threshold."
    return None


def _clear_jax_caches() -> None:
    """Best-effort cleanup of JAX caches between large fits."""
    try:
        import jax
    except ModuleNotFoundError:
        return
    jax.clear_caches()


def fit_ripple_glm_train_on_ripple_predict_pre(
    epoch: str,
    *,
    spikes: dict[str, Any],
    epoch_interval: "nap.IntervalSet",
    ripple_table: pd.DataFrame,
    min_spikes_per_ripple: float = DEFAULT_MIN_SPIKES_PER_RIPPLE,
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
    store_ripple_shuffle_preds: bool = False,
) -> dict[str, Any]:
    """Fit the legacy ripple GLM workflow for one epoch."""
    import nemos as nmo
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
    ca1_unit_ids = np.asarray(list(spikes[SOURCE_REGION].keys()))
    kept_v1_unit_ids = v1_unit_ids[keep_y]
    n_cells = int(y_r.shape[1])
    skip_reason = get_epoch_skip_reason(
        n_ripples=n_ripples,
        n_splits=n_splits,
        n_kept_v1_units=n_cells,
    )
    if skip_reason is not None:
        raise ValueError(skip_reason)

    kfold = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    y_r_test_full = np.full((n_ripples, n_cells), np.nan, dtype=np.float32)
    y_r_hat_full = np.full((n_ripples, n_cells), np.nan, dtype=np.float32)

    y_r_hat_shuff: np.ndarray | None = None
    if store_ripple_shuffle_preds and n_shuffles_ripple > 0:
        y_r_hat_shuff = np.full(
            (n_shuffles_ripple, n_ripples, n_cells),
            np.nan,
            dtype=np.float32,
        )

    def _alloc_metric_array() -> np.ndarray:
        return np.full((n_splits, n_cells), np.nan, dtype=np.float32)

    pseudo_r2_ripple = _alloc_metric_array()
    mae_ripple = _alloc_metric_array()
    ll_ripple = _alloc_metric_array()
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
    ll_ripple_shuff = np.full(
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

    fold_info: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(X_r)):
        X_train_r = X_r[train_idx]
        X_test_r = X_r[test_idx]
        y_train_r = y_r[train_idx]
        y_test_r = y_r[test_idx]

        X_train_pp, X_test_r_pp, keep_x, mean, std = _preprocess_X_fit_apply(
            X_train=X_train_r,
            X_test=X_test_r,
        )
        if X_train_pp.shape[1] == 0:
            raise ValueError(
                f"Epoch {epoch!r}, fold {fold_index}: all CA1 features were near-constant."
            )

        glm = nmo.glm.PopulationGLM(
            solver_name="LBFGS",
            regularizer="Ridge",
            regularizer_strength=float(ridge_strength),
            solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
        )
        glm.fit(X_train_pp, y_train_r)
        lam_r = np.asarray(glm.predict(X_test_r_pp), dtype=np.float64)

        y_r_hat_full[test_idx] = lam_r.astype(np.float32)
        y_r_test_full[test_idx] = y_test_r.astype(np.float32)

        pseudo_r2_ripple[fold_index] = mcfadden_pseudo_r2_per_neuron(
            y_test_r,
            lam_r,
            y_train_r,
        ).astype(np.float32)
        mae_ripple[fold_index] = mae_per_neuron(y_test_r, lam_r).astype(np.float32)
        ll_ripple[fold_index] = poisson_ll_per_neuron(y_test_r, lam_r).astype(np.float32)
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
                solver_name="LBFGS",
                regularizer="Ridge",
                regularizer_strength=float(ridge_strength),
                solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
            )
            glm_shuffle.fit(X_train_pp, y_train_shuffled)
            lam_r_shuffled = np.asarray(glm_shuffle.predict(X_test_r_pp), dtype=np.float64)

            if y_r_hat_shuff is not None:
                y_r_hat_shuff[shuffle_index, test_idx] = lam_r_shuffled.astype(np.float32)

            pseudo_r2_ripple_shuff[fold_index, shuffle_index] = mcfadden_pseudo_r2_per_neuron(
                y_test_r,
                lam_r_shuffled,
                y_train_r,
            ).astype(np.float32)
            mae_ripple_shuff[fold_index, shuffle_index] = mae_per_neuron(
                y_test_r,
                lam_r_shuffled,
            ).astype(np.float32)
            ll_ripple_shuff[fold_index, shuffle_index] = poisson_ll_per_neuron(
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

        fold_info.append(
            {
                "fold": int(fold_index),
                "train_idx": np.asarray(train_idx, dtype=int),
                "test_idx": np.asarray(test_idx, dtype=int),
                "keep_x": np.asarray(keep_x, dtype=bool),
                "x_mean": np.asarray(mean, dtype=float),
                "x_std": np.asarray(std, dtype=float),
                "n_ripples": n_ripples,
                "n_pre": n_pre,
            }
        )

        del glm, lam_r
        _clear_jax_caches()
        gc.collect()

    for name, values in (
        ("y_r_test_full", y_r_test_full),
        ("y_r_hat_full", y_r_hat_full),
    ):
        if np.isnan(values).any():
            raise ValueError(f"{name} has NaNs: some ripple rows were never assigned to a test fold.")

    if y_r_hat_shuff is not None and np.isnan(y_r_hat_shuff).any():
        raise ValueError(
            "y_r_hat_shuff has NaNs: some ripple rows were never assigned to a test fold."
        )

    X_r_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(X_r, X_r)
    if X_r_all_pp.shape[1] == 0:
        raise ValueError(f"Epoch {epoch!r}: all CA1 features were near-constant.")

    pre_pseudo_r2 = np.full(n_cells, np.nan, dtype=np.float32)
    pre_mae = np.full(n_cells, np.nan, dtype=np.float32)
    pre_ll = np.full(n_cells, np.nan, dtype=np.float32)
    pre_devexp = np.full(n_cells, np.nan, dtype=np.float32)
    pre_bits_per_spike = np.full(n_cells, np.nan, dtype=np.float32)
    yhat_pre = np.empty((n_pre, n_cells), dtype=np.float32)

    if n_pre > 0:
        X_p_kept = X_p[:, keep_x_all]
        X_p_pp = (X_p_kept - mean_all[None, :]) / (std_all[None, :] + 1e-8)
        X_p_pp /= np.sqrt(max(X_r_all_pp.shape[1], 1))
        X_p_pp = np.clip(X_p_pp, -10.0, 10.0)

        glm_all = nmo.glm.PopulationGLM(
            solver_name="LBFGS",
            regularizer="Ridge",
            regularizer_strength=float(ridge_strength),
            solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
        )
        glm_all.fit(X_r_all_pp, y_r)
        lam_pre = np.asarray(glm_all.predict(X_p_pp), dtype=np.float64)
        yhat_pre = lam_pre.astype(np.float32)

        pre_pseudo_r2 = mcfadden_pseudo_r2_per_neuron(y_p, lam_pre, y_p).astype(np.float32)
        pre_mae = mae_per_neuron(y_p, lam_pre).astype(np.float32)
        pre_ll = poisson_ll_per_neuron(y_p, lam_pre).astype(np.float32)
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
        "v1_unit_ids": kept_v1_unit_ids,
        "ca1_unit_ids": ca1_unit_ids,
        "fold_info": fold_info,
        "pseudo_r2_ripple_folds": pseudo_r2_ripple,
        "mae_ripple_folds": mae_ripple,
        "ll_ripple_folds": ll_ripple,
        "devexp_ripple_folds": devexp_ripple,
        "bits_per_spike_ripple_folds": bits_per_spike_ripple,
        "pseudo_r2_ripple_shuff_folds": pseudo_r2_ripple_shuff,
        "mae_ripple_shuff_folds": mae_ripple_shuff,
        "ll_ripple_shuff_folds": ll_ripple_shuff,
        "devexp_ripple_shuff_folds": devexp_ripple_shuff,
        "bits_per_spike_ripple_shuff_folds": bits_per_spike_ripple_shuff,
        "y_ripple_test": y_r_test_full,
        "yhat_ripple": y_r_hat_full,
        "yhat_ripple_shuff": y_r_hat_shuff,
        "y_pre_test": y_p.astype(np.float32),
        "yhat_pre": yhat_pre,
        "pseudo_r2_pre": pre_pseudo_r2,
        "mae_pre": pre_mae,
        "ll_pre": pre_ll,
        "devexp_pre": pre_devexp,
        "bits_per_spike_pre": pre_bits_per_spike,
        "keep_x_all": np.asarray(keep_x_all, dtype=bool),
        "x_mean_all": np.asarray(mean_all, dtype=float),
        "x_std_all": np.asarray(std_all, dtype=float),
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


def empirical_p_values(
    observed: np.ndarray,
    null_samples: np.ndarray,
    *,
    higher_is_better: bool,
) -> np.ndarray:
    """Return one-sided empirical p-values per unit."""
    observed = np.asarray(observed, dtype=float)
    null_samples = np.asarray(null_samples, dtype=float)
    if null_samples.size == 0:
        return np.full(observed.shape, np.nan, dtype=float)

    valid = np.isfinite(null_samples) & np.isfinite(observed[None, :])
    if higher_is_better:
        extreme = null_samples >= observed[None, :]
    else:
        extreme = null_samples <= observed[None, :]

    counts = np.sum(extreme & valid, axis=0)
    totals = np.sum(valid, axis=0)
    return np.where(totals > 0, (counts + 1) / (totals + 1), np.nan)


def empirical_population_p_value(
    observed: float,
    null_samples: np.ndarray,
    *,
    higher_is_better: bool,
) -> float:
    """Return a one-sided empirical p-value for one population summary."""
    null_samples = np.asarray(null_samples, dtype=float)
    valid = null_samples[np.isfinite(null_samples)]
    if valid.size == 0 or not np.isfinite(observed):
        return float("nan")

    if higher_is_better:
        count = int(np.sum(valid >= observed))
    else:
        count = int(np.sum(valid <= observed))
    return float((count + 1) / (valid.size + 1))


def summarize_ripple_metric_against_shuffle(
    real_folds: Any,
    shuffle_folds: Any,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Summarize one ripple metric and its shuffle null at unit and population levels."""
    real_folds_array = _as_2d_float(real_folds)
    shuffle_folds_array = _as_3d_float(shuffle_folds)

    real_mean = np.nanmean(real_folds_array, axis=0)
    real_sem = nansem(real_folds_array, axis=0)

    if shuffle_folds_array.shape[1] == 0:
        unit_null_samples = np.empty((0, real_folds_array.shape[1]), dtype=float)
        shuffle_mean = np.full(real_mean.shape, np.nan, dtype=float)
        shuffle_sd = np.full(real_mean.shape, np.nan, dtype=float)
        unit_p_value = np.full(real_mean.shape, np.nan, dtype=float)
        population_null_samples = np.empty(0, dtype=float)
        population_p_value = float("nan")
    else:
        unit_null_samples = np.nanmean(shuffle_folds_array, axis=0)
        shuffle_mean = np.nanmean(unit_null_samples, axis=0)
        shuffle_sd = np.nanstd(unit_null_samples, axis=0, ddof=0)
        unit_p_value = empirical_p_values(
            real_mean,
            unit_null_samples,
            higher_is_better=higher_is_better,
        )
        population_null_samples = np.nanmean(unit_null_samples, axis=1)
        population_p_value = empirical_population_p_value(
            float(np.nanmean(real_mean)),
            population_null_samples,
            higher_is_better=higher_is_better,
        )

    return {
        "real_mean": real_mean,
        "real_sem": real_sem,
        "shuffle_mean": shuffle_mean,
        "shuffle_sd": shuffle_sd,
        "unit_p_value": unit_p_value,
        "population_real_mean": float(np.nanmean(real_mean)),
        "population_null_samples": population_null_samples,
        "population_p_value": population_p_value,
    }


def build_unit_summary_table(
    results: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    epoch: str,
) -> pd.DataFrame:
    """Build the per-unit ripple GLM summary table for one epoch."""
    unit_ids = np.asarray(results["v1_unit_ids"])
    unit_summary = pd.DataFrame(
        {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "v1_unit_id": unit_ids,
            "n_ripples": int(results["n_ripples"]),
            "n_pre_windows": int(results["n_pre"]),
            "n_ca1_units": int(len(np.asarray(results["ca1_unit_ids"]))),
        }
    )

    for metric_name in ("pseudo_r2", "mae", "ll", "devexp", "bits_per_spike"):
        real_folds = _as_2d_float(results[f"{metric_name}_ripple_folds"])
        unit_summary[f"ripple_{metric_name}_mean"] = np.nanmean(real_folds, axis=0)
        unit_summary[f"ripple_{metric_name}_sem"] = nansem(real_folds, axis=0)

    for metric_name in ("pseudo_r2", "mae", "bits_per_spike"):
        summary = summarize_ripple_metric_against_shuffle(
            results[f"{metric_name}_ripple_folds"],
            results[f"{metric_name}_ripple_shuff_folds"],
            higher_is_better=HIGHER_IS_BETTER_BY_METRIC[metric_name],
        )
        unit_summary[f"ripple_{metric_name}_shuffle_mean"] = summary["shuffle_mean"]
        unit_summary[f"ripple_{metric_name}_shuffle_sd"] = summary["shuffle_sd"]
        unit_summary[f"ripple_{metric_name}_p_value"] = summary["unit_p_value"]

    unit_summary["pre_pseudo_r2"] = _as_1d_float(results["pseudo_r2_pre"])
    unit_summary["pre_mae"] = _as_1d_float(results["mae_pre"])
    unit_summary["pre_ll"] = _as_1d_float(results["ll_pre"])
    unit_summary["pre_devexp"] = _as_1d_float(results["devexp_pre"])
    unit_summary["pre_bits_per_spike"] = _as_1d_float(results["bits_per_spike_pre"])
    return unit_summary


def build_population_summary_table(
    results: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    epoch: str,
) -> pd.DataFrame:
    """Build the one-row epoch-level ripple GLM population summary."""
    row: dict[str, Any] = {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "n_ripples": int(results["n_ripples"]),
        "n_pre_windows": int(results["n_pre"]),
        "n_v1_units": int(len(np.asarray(results["v1_unit_ids"]))),
        "n_ca1_units": int(len(np.asarray(results["ca1_unit_ids"]))),
    }

    for metric_name in ("pseudo_r2", "mae", "ll", "devexp", "bits_per_spike"):
        real_folds = _as_2d_float(results[f"{metric_name}_ripple_folds"])
        row[f"ripple_{metric_name}_mean"] = float(np.nanmean(np.nanmean(real_folds, axis=0)))

    row["pre_pseudo_r2_mean"] = float(np.nanmean(_as_1d_float(results["pseudo_r2_pre"])))
    row["pre_mae_mean"] = float(np.nanmean(_as_1d_float(results["mae_pre"])))
    row["pre_ll_mean"] = float(np.nanmean(_as_1d_float(results["ll_pre"])))
    row["pre_devexp_mean"] = float(np.nanmean(_as_1d_float(results["devexp_pre"])))
    row["pre_bits_per_spike_mean"] = float(
        np.nanmean(_as_1d_float(results["bits_per_spike_pre"]))
    )

    for metric_name in ("pseudo_r2", "mae", "bits_per_spike"):
        summary = summarize_ripple_metric_against_shuffle(
            results[f"{metric_name}_ripple_folds"],
            results[f"{metric_name}_ripple_shuff_folds"],
            higher_is_better=HIGHER_IS_BETTER_BY_METRIC[metric_name],
        )
        row[f"ripple_{metric_name}_shuffle_mean"] = float(np.nanmean(summary["shuffle_mean"]))
        row[f"ripple_{metric_name}_shuffle_sd"] = float(np.nanmean(summary["shuffle_sd"]))
        row[f"ripple_{metric_name}_population_p_value"] = summary["population_p_value"]

    return pd.DataFrame([row])


def _get_pyplot():
    """Return pyplot configured for headless script execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ECDF coordinates after dropping NaNs."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return valid, valid
    x_values = np.sort(valid)
    y_values = np.arange(1, x_values.size + 1) / x_values.size
    return x_values, y_values


def plot_ripple_metric_summary(
    *,
    real_folds: Any,
    shuffle_folds: Any,
    animal_name: str,
    date: str,
    epoch: str,
    metric_label: str,
    out_path: Path,
    higher_is_better: bool,
) -> Path:
    """Plot one ripple metric against its shuffle null."""
    plt = _get_pyplot()

    real_folds_array = _as_2d_float(real_folds)
    real_unit_values = np.nanmean(real_folds_array, axis=0)
    summary = summarize_ripple_metric_against_shuffle(
        real_folds,
        shuffle_folds,
        higher_is_better=higher_is_better,
    )
    null_unit_samples = np.asarray(
        np.nanmean(_as_3d_float(shuffle_folds), axis=0)
        if _as_3d_float(shuffle_folds).shape[1] > 0
        else np.empty((0, real_unit_values.size), dtype=float)
    )
    null_flat = null_unit_samples.ravel() if null_unit_samples.size else np.array([], dtype=float)
    pop_null = np.asarray(summary["population_null_samples"], dtype=float)
    p_values = np.asarray(summary["unit_p_value"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    finite_real = real_unit_values[np.isfinite(real_unit_values)]
    if finite_real.size:
        ax.hist(finite_real, bins=30, alpha=0.8)
    ax.set_xlabel(f"Per-unit ripple {metric_label}")
    ax.set_ylabel("Count")
    ax.set_title("Per-unit distribution")
    ax.text(
        0.02,
        0.98,
        (
            f"mean={np.nanmean(real_unit_values):.4f}\n"
            f"median={np.nanmedian(real_unit_values):.4f}\n"
            f"units={real_unit_values.size}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax = axes[1]
    if pop_null.size:
        ax.hist(pop_null, bins=30, alpha=0.75, label="shuffle")
        ax.axvline(summary["population_real_mean"], color="black", linewidth=2, label="observed")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No shuffle data", ha="center", va="center")
    ax.set_xlabel(f"Population mean ripple {metric_label}")
    ax.set_ylabel("Count")
    ax.set_title("Population null")

    ax = axes[2]
    real_x, real_y = _ecdf(real_unit_values)
    if real_x.size:
        ax.plot(real_x, real_y, label="observed", linewidth=2)
    if null_flat.size:
        null_x, null_y = _ecdf(null_flat)
        ax.plot(null_x, null_y, label="shuffle", linewidth=2)
        ax.legend(loc="best")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("ECDF")
    ax.set_title("Observed vs shuffle ECDF")

    ax = axes[3]
    finite_p = p_values[np.isfinite(p_values)]
    if finite_p.size:
        ax.hist(-np.log10(np.clip(finite_p, 1e-12, 1.0)), bins=30, alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No shuffle p-values", ha="center", va="center")
    ax.set_xlabel("-log10(p)")
    ax.set_ylabel("Count")
    ax.set_title("Per-unit shuffle comparison")

    fig.suptitle(f"{animal_name} {date} {epoch} ripple {metric_label}", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_pre_metric_summary(
    *,
    values: Any,
    spike_counts: Any,
    animal_name: str,
    date: str,
    epoch: str,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot one pre-ripple metric summary."""
    plt = _get_pyplot()

    metric_values = _as_1d_float(values)
    spike_counts_array = _as_1d_float(spike_counts)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    finite_values = metric_values[np.isfinite(metric_values)]
    if finite_values.size:
        ax.hist(finite_values, bins=30, alpha=0.8)
    ax.set_xlabel(f"Per-unit pre {metric_label}")
    ax.set_ylabel("Count")
    ax.set_title("Per-unit distribution")
    ax.text(
        0.02,
        0.98,
        (
            f"mean={np.nanmean(metric_values):.4f}\n"
            f"median={np.nanmedian(metric_values):.4f}\n"
            f"units={metric_values.size}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax = axes[1]
    ecdf_x, ecdf_y = _ecdf(metric_values)
    if ecdf_x.size:
        ax.plot(ecdf_x, ecdf_y, linewidth=2)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF")

    ax = axes[2]
    valid = np.isfinite(metric_values) & np.isfinite(spike_counts_array)
    if np.any(valid):
        ax.scatter(spike_counts_array[valid], metric_values[valid], alpha=0.7)
    ax.set_xlabel("Pre spikes per unit")
    ax.set_ylabel(metric_label)
    ax.set_title("Metric vs pre spikes")

    ax = axes[3]
    if np.any(valid):
        ax.scatter(np.arange(valid.sum()), np.sort(metric_values[valid]), alpha=0.7)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
    ax.set_xlabel("Sorted unit rank")
    ax.set_ylabel(metric_label)
    ax.set_title("Sorted values")

    fig.suptitle(f"{animal_name} {date} {epoch} pre {metric_label}", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_epoch_tables(
    *,
    unit_summary: pd.DataFrame,
    population_summary: pd.DataFrame,
    data_dir: Path,
    epoch: str,
) -> dict[str, Path]:
    """Save the per-unit and per-epoch summary tables."""
    data_dir.mkdir(parents=True, exist_ok=True)
    unit_summary_path = data_dir / f"{epoch}_unit_summary.parquet"
    population_summary_path = data_dir / f"{epoch}_population_summary.parquet"
    unit_summary.to_parquet(unit_summary_path, index=False)
    population_summary.to_parquet(population_summary_path, index=False)
    return {
        "unit_summary": unit_summary_path,
        "population_summary": population_summary_path,
    }


def save_epoch_figures(
    *,
    results: dict[str, Any],
    fig_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
) -> list[Path]:
    """Save the six configured epoch summary figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    pre_spike_counts = np.sum(np.asarray(results["y_pre_test"], dtype=float), axis=0)

    figure_paths = [
        plot_ripple_metric_summary(
            real_folds=results["pseudo_r2_ripple_folds"],
            shuffle_folds=results["pseudo_r2_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            metric_label="Pseudo R^2",
            out_path=fig_dir / f"{epoch}_ripple_pseudo_r2_summary.png",
            higher_is_better=True,
        ),
        plot_ripple_metric_summary(
            real_folds=results["mae_ripple_folds"],
            shuffle_folds=results["mae_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            metric_label="MAE",
            out_path=fig_dir / f"{epoch}_ripple_mae_summary.png",
            higher_is_better=False,
        ),
        plot_ripple_metric_summary(
            real_folds=results["bits_per_spike_ripple_folds"],
            shuffle_folds=results["bits_per_spike_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            metric_label="Bits/spike",
            out_path=fig_dir / f"{epoch}_ripple_bits_per_spike_summary.png",
            higher_is_better=True,
        ),
        plot_pre_metric_summary(
            values=results["pseudo_r2_pre"],
            spike_counts=pre_spike_counts,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            metric_label="Pseudo R^2",
            out_path=fig_dir / f"{epoch}_pre_pseudo_r2_summary.png",
        ),
        plot_pre_metric_summary(
            values=results["mae_pre"],
            spike_counts=pre_spike_counts,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            metric_label="MAE",
            out_path=fig_dir / f"{epoch}_pre_mae_summary.png",
        ),
        plot_pre_metric_summary(
            values=results["bits_per_spike_pre"],
            spike_counts=pre_spike_counts,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            metric_label="Bits/spike",
            out_path=fig_dir / f"{epoch}_pre_bits_per_spike_summary.png",
        ),
    ]
    return figure_paths


def main() -> None:
    """Run the ripple GLM CLI."""
    args = parse_arguments()
    validate_arguments(args)

    session = prepare_ripple_glm_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    selected_epochs = validate_epochs(session["epoch_tags"], args.epochs)
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
    fig_dir = analysis_path / "figs" / "ripple_glm"
    pre_window_s = args.pre_window_s if args.pre_window_s is not None else args.ripple_window_s

    saved_unit_tables: list[Path] = []
    saved_population_tables: list[Path] = []
    saved_figures: list[Path] = []
    saved_legacy_npz: list[Path] = []
    skipped_epochs: list[dict[str, Any]] = []

    for epoch in selected_epochs:
        ripple_table = session["ripple_tables"][epoch]
        try:
            results = fit_ripple_glm_train_on_ripple_predict_pre(
                epoch=epoch,
                spikes=session["spikes_by_region"],
                epoch_interval=session["epoch_intervals"][epoch],
                ripple_table=ripple_table,
                min_spikes_per_ripple=args.min_spikes_per_ripple,
                n_shuffles_ripple=args.n_shuffles_ripple,
                shuffle_seed=args.shuffle_seed,
                ripple_window_s=args.ripple_window_s,
                pre_window_s=pre_window_s,
                pre_buffer_s=args.pre_buffer_s,
                exclude_ripples=args.exclude_ripples,
                pre_exclude_guard_s=args.pre_exclude_guard_s,
                n_splits=args.n_splits,
                ridge_strength=args.ridge_strength,
                maxiter=args.maxiter,
                tol=args.tol,
                store_ripple_shuffle_preds=args.save_legacy_npz,
            )
        except ValueError as exc:
            skipped_epochs.append({"epoch": epoch, "reason": str(exc)})
            print(f"Skipping {args.animal_name} {args.date} {epoch}: {exc}")
            continue
        except Exception as exc:
            skipped_epochs.append(
                {
                    "epoch": epoch,
                    "reason": "fit failed",
                    "error": str(exc),
                }
            )
            print(f"Skipping {args.animal_name} {args.date} {epoch}: fit failed: {exc}")
            continue

        unit_summary = build_unit_summary_table(
            results,
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
        )
        population_summary = build_population_summary_table(
            results,
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
        )
        saved_tables = save_epoch_tables(
            unit_summary=unit_summary,
            population_summary=population_summary,
            data_dir=data_dir,
            epoch=epoch,
        )
        saved_unit_tables.append(saved_tables["unit_summary"])
        saved_population_tables.append(saved_tables["population_summary"])
        saved_figures.extend(
            save_epoch_figures(
                results=results,
                fig_dir=fig_dir,
                animal_name=args.animal_name,
                date=args.date,
                epoch=epoch,
            )
        )

        if args.save_legacy_npz:
            legacy_npz_path = save_results_npz(data_dir / f"{epoch}_legacy_results.npz", results)
            saved_legacy_npz.append(legacy_npz_path)

    if not saved_unit_tables:
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
            "n_splits": args.n_splits,
            "n_shuffles_ripple": args.n_shuffles_ripple,
            "ridge_strength": args.ridge_strength,
            "shuffle_seed": args.shuffle_seed,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "save_legacy_npz": args.save_legacy_npz,
            "model_direction": f"{SOURCE_REGION}_to_{TARGET_REGION}",
        },
        outputs={
            "sources": session["sources"],
            "selected_epochs": selected_epochs,
            "saved_unit_tables": saved_unit_tables,
            "saved_population_tables": saved_population_tables,
            "saved_figures": saved_figures,
            "saved_legacy_npz": saved_legacy_npz,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
