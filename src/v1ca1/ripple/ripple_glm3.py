from __future__ import annotations

"""Legacy-style ripple GLM outputs backed by legacy pickle session artifacts.

This script keeps the legacy `ripple_glm_population_pre.py` result contract:
legacy `.npz` payloads and figure paths under the session analysis directory.
Like the legacy script, it loads pickle-based timestamps and ripple events:
`timestamps_ephys.pkl`, `timestamps_ephys_all.pkl`,
`ripple/Kay_ripple_detector.pkl`, and `sorting_v1` / `sorting_ca1`.
"""

import argparse
import gc
import json
import os
import pickle
import sys
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
    load_spikes_by_region,
)
from v1ca1.ripple.ripple_glm import (
    EMPIRICAL_P_VALUE_ATOL,
    EMPIRICAL_P_VALUE_RTOL,
    build_epoch_intervals,
    load_ripple_tables_from_legacy_pickle,
)

if TYPE_CHECKING:
    import pynapple as nap


DEFAULT_RIPPLE_WINDOW_S = 0.2
DEFAULT_PRE_BUFFER_S = 0.02
DEFAULT_PRE_EXCLUDE_GUARD_S = 0.05
DEFAULT_MIN_SPIKES_PER_RIPPLE = 0.1
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES_RIPPLE = 100
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_RANDOM_SEED = 45
DEFAULT_MAXITER = 6000
DEFAULT_TOL = 1e-7
DEFAULT_RIDGE_SWEEP = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1e0,
    1e1,
)
DEFAULT_XLA_PREALLOCATE = "false"
DEFAULT_XLA_MEM_FRACTION = "0.70"

SOURCE_REGION = "ca1"
TARGET_REGION = "v1"
REGIONS = (TARGET_REGION, SOURCE_REGION)

RIPPLE_COLOR = "#1f77b4"
SHUFFLE_COLOR = "#bdbdbd"
PRE_COLOR = "#d95f02"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the legacy-style ripple GLM workflow."""
    parser = argparse.ArgumentParser(
        description="Fit the legacy ripple GLM workflow from legacy pickle artifacts."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base analysis directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="Optional subset of epochs to process. Default: all epochs with ripple events.",
    )
    parser.add_argument(
        "--mode",
        choices=("train_on_ripple", "ridge_sweep"),
        default="train_on_ripple",
        help="Run the active legacy workflow or the ridge-sweep utility. Default: train_on_ripple",
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
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude ripple intervals from pre windows. Default: disabled.",
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
        "--n-shuffles-ripple",
        type=int,
        default=DEFAULT_N_SHUFFLES_RIPPLE,
        help=(
            "Number of response-shuffle refits per fold for ripple evaluation. "
            f"Default: {DEFAULT_N_SHUFFLES_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for shuffle models. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of CV folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Ridge strength for train_on_ripple mode. Default: {DEFAULT_RIDGE_STRENGTH}",
    )
    parser.add_argument(
        "--ridge-strengths",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGE_SWEEP),
        help=(
            "Ridge values for ridge_sweep mode. "
            f"Default: {list(DEFAULT_RIDGE_SWEEP)!r}"
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help=f"Maximum LBFGS iterations. Default: {DEFAULT_MAXITER}",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"LBFGS tolerance. Default: {DEFAULT_TOL}",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Optional value assigned to CUDA_VISIBLE_DEVICES before importing "
            "JAX/NEMOS, for example '0' or ''."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute saved `.npz` payloads instead of reusing existing ones.",
    )
    return parser.parse_args(argv)


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
    if args.n_shuffles_ripple < 0:
        raise ValueError("--n-shuffles-ripple must be non-negative.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.ridge_strength < 0:
        raise ValueError("--ridge-strength must be non-negative.")
    if not args.ridge_strengths:
        raise ValueError("--ridge-strengths must contain at least one value.")
    if any(ridge_strength < 0 for ridge_strength in args.ridge_strengths):
        raise ValueError("--ridge-strengths must contain only non-negative values.")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive.")
    if args.tol <= 0:
        raise ValueError("--tol must be positive.")


def configure_jax_environment(cuda_visible_devices: str | None) -> None:
    """Set CUDA visibility and XLA memory behavior before JAX import."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = DEFAULT_XLA_PREALLOCATE
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = DEFAULT_XLA_MEM_FRACTION


def validate_epochs(
    available_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Validate an optional epoch subset against the available session epochs."""
    if requested_epochs is None:
        return list(available_epochs)

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in available_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in the saved session epochs {available_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return list(requested_epochs)


def validate_selected_epochs_across_sources(
    selected_epochs: list[str],
    *,
    source_epochs: dict[str, list[str] | dict[str, Any]],
) -> None:
    """Ensure each selected epoch is present in each required source."""
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
            "Selected epochs are missing required ripple_glm3 inputs. "
            f"Missing epochs by source: {details}"
        )


def require_existing_path(path: Path, *, description: str) -> Path:
    """Return one required path or raise a clear FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(f"Required {description} was not found: {path}")
    return path


def load_legacy_ripple_tables(analysis_path: Path) -> dict[str, pd.DataFrame]:
    """Load ripple events from the legacy detector pickle."""
    legacy_path = require_existing_path(
        analysis_path / "ripple" / "Kay_ripple_detector.pkl",
        description="legacy ripple detector pickle",
    )
    return load_ripple_tables_from_legacy_pickle(legacy_path)


def load_legacy_epoch_timestamps(
    analysis_path: Path,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, "nap.IntervalSet"]]:
    """Load per-epoch ephys timestamps and epoch intervals from pickle."""
    pickle_path = require_existing_path(
        analysis_path / "timestamps_ephys.pkl",
        description="timestamps_ephys.pkl",
    )
    with open(pickle_path, "rb") as file:
        loaded = pickle.load(file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a dictionary in {pickle_path}, found {type(loaded).__name__}.")

    timestamps_by_epoch = {
        str(epoch): np.asarray(timestamps, dtype=float)
        for epoch, timestamps in loaded.items()
    }
    epoch_tags = list(timestamps_by_epoch.keys())
    return epoch_tags, timestamps_by_epoch, build_epoch_intervals(timestamps_by_epoch)


def load_legacy_ephys_timestamps_all(analysis_path: Path) -> np.ndarray:
    """Load concatenated ephys timestamps from the legacy pickle."""
    pickle_path = require_existing_path(
        analysis_path / "timestamps_ephys_all.pkl",
        description="timestamps_ephys_all.pkl",
    )
    with open(pickle_path, "rb") as file:
        loaded = pickle.load(file)
    return np.asarray(loaded, dtype=float)


def load_legacy_spikes(
    analysis_path: Path,
    timestamps_ephys_all: np.ndarray,
) -> dict[str, "nap.TsGroup"]:
    """Load `sorting_v1` and `sorting_ca1` using the legacy timestamp vector."""
    for region in REGIONS:
        require_existing_path(
            analysis_path / f"sorting_{region}",
            description=f"sorting_{region}",
        )
    return load_spikes_by_region(
        analysis_path,
        np.asarray(timestamps_ephys_all, dtype=float),
        regions=REGIONS,
    )


def prepare_ripple_glm3_session(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Load one session using legacy pickle ripple GLM inputs only."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    require_existing_path(analysis_path, description="analysis path")

    epoch_tags, timestamps_ephys_by_epoch, epoch_intervals = load_legacy_epoch_timestamps(
        analysis_path
    )
    timestamps_ephys_all = load_legacy_ephys_timestamps_all(analysis_path)
    spikes = load_legacy_spikes(analysis_path, timestamps_ephys_all)
    loaded_ripple_tables = load_legacy_ripple_tables(analysis_path)
    ripple_event_epochs = list(loaded_ripple_tables.keys())
    ripple_tables = {
        epoch: loaded_ripple_tables.get(epoch, pd.DataFrame(columns=["start_time", "end_time"]))
        for epoch in epoch_tags
    }
    for epoch, table in loaded_ripple_tables.items():
        if epoch not in ripple_tables:
            ripple_tables[epoch] = table

    return {
        "analysis_path": analysis_path,
        "epoch_tags": epoch_tags,
        "timestamps_ephys_by_epoch": timestamps_ephys_by_epoch,
        "timestamps_ephys_all": timestamps_ephys_all,
        "epoch_intervals": epoch_intervals,
        "spikes": spikes,
        "ripple_event_epochs": ripple_event_epochs,
        "ripple_tables": ripple_tables,
        "sources": {
            "timestamps_ephys": "pickle",
            "timestamps_ephys_all": "pickle",
            "sorting": "spikeinterface",
            "ripple_events": "pickle",
        },
    }


def save_results_npz(out_path: Path, results: dict[str, Any]) -> Path:
    """Save one legacy-style result payload as `.npz`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    to_save = dict(results)
    if "fold_info" in to_save:
        to_save["fold_info"] = np.array(to_save["fold_info"], dtype=object)
    np.savez_compressed(out_path, **to_save)
    return out_path


def _normalize_loaded_npz_value(value: Any) -> Any:
    """Return a convenient Python value for one loaded `.npz` entry."""
    array = np.asarray(value)
    if array.dtype == object and array.shape == ():
        return array.item()
    if array.dtype == object and array.ndim == 1:
        return list(array)
    return value


def load_results_npz(path: Path) -> dict[str, Any]:
    """Load one legacy-style `.npz` result payload."""
    loaded = np.load(path, allow_pickle=True)
    return {
        key: _normalize_loaded_npz_value(loaded[key])
        for key in loaded.files
    }


def build_legacy_result_path(
    out_dir: Path,
    *,
    epoch: str,
    exclude_ripples: bool,
    ripple_window_s: float,
    min_spikes_per_ripple: float,
    ridge_strength: float,
) -> Path:
    """Return the legacy `.npz` output path for one epoch."""
    return (
        out_dir
        / (
            f"{epoch}_train_ripple_predict_ripple_and_pre_exclude_ripples_"
            f"{exclude_ripples}_ripple_window_{ripple_window_s}"
            f"_min_spikes_per_ripple_{min_spikes_per_ripple}"
            f"_ridge_strength_{ridge_strength}.npz"
        )
    )


def build_legacy_figure_paths(
    fig_dir: Path,
    *,
    epoch: str,
    exclude_ripples: bool,
    ripple_window_s: float,
) -> dict[str, Path]:
    """Return the six legacy train-on-ripple figure paths for one epoch."""
    stem = f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window_s}"
    return {
        "ripple_pseudo_r2": fig_dir / f"{stem}_RIPPLE_pseudo_r2_summary.png",
        "ripple_mae": fig_dir / f"{stem}_RIPPLE_mae_summary.png",
        "ripple_delta_bits_per_spike": fig_dir / f"{stem}_RIPPLE_delta_bits_per_spike_summary.png",
        "pre_pseudo_r2": fig_dir / f"{stem}_PRE_pseudo_r2_summary.png",
        "pre_mae": fig_dir / f"{stem}_PRE_mae_summary.png",
        "pre_delta_bits_per_spike": fig_dir / f"{stem}_PRE_delta_bits_per_spike_summary.png",
    }


def _get_pyplot():
    """Return pyplot configured for headless script execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _as_1d_float(values: Any) -> np.ndarray:
    """Return a flattened float array, handling object arrays from `.npz`."""
    array = np.asarray(values)
    if array.dtype == object:
        array = np.asarray(list(array), dtype=float)
    return np.asarray(array, dtype=float).ravel()


def _as_2d_float(values: Any) -> np.ndarray:
    """Return a 2D float array, handling object arrays from `.npz`."""
    array = np.asarray(values)
    if array.dtype == object:
        return np.stack(array.tolist(), axis=0).astype(float)
    return np.asarray(array, dtype=float)


def _as_3d_float(values: Any) -> np.ndarray:
    """Return a 3D float array, handling object arrays from `.npz`."""
    array = np.asarray(values)
    if array.dtype == object:
        return np.stack(array.tolist(), axis=0).astype(float)
    return np.asarray(array, dtype=float)


def _finite_values(values: Any) -> np.ndarray:
    """Return only finite values from one array-like input."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _metric_histogram_bins(*arrays: Any) -> np.ndarray:
    """Return shared histogram bins spanning all finite values."""
    finite = np.concatenate([_finite_values(array) for array in arrays])
    if finite.size == 0:
        return np.linspace(0.0, 1.0, 31)

    x_min = float(np.min(finite))
    x_max = float(np.max(finite))
    if np.isclose(x_min, x_max):
        pad = max(abs(x_min) * 0.05, 0.5)
        x_min -= pad
        x_max += pad
    return np.linspace(x_min, x_max, 31)


def _first_nonfinite_info(values: np.ndarray) -> str | None:
    """Return a short description of the first non-finite value, if any."""
    values = np.asarray(values)
    mask = ~np.isfinite(values)
    if not np.any(mask):
        return None
    index = np.argwhere(mask)[0]
    return f"shape={values.shape}, first_bad_idx={tuple(index)}, value={values[tuple(index)]}"


def _preprocess_X_fit_apply(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the legacy predictor preprocessing on train and apply to train/test."""
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


def make_preripple_ep(
    ripple_ep: "nap.IntervalSet",
    epoch_ep: "nap.IntervalSet",
    *,
    window_s: float | None = None,
    buffer_s: float = DEFAULT_PRE_BUFFER_S,
    exclude_ripples: bool = True,
    exclude_ripple_guard_s: float = DEFAULT_PRE_EXCLUDE_GUARD_S,
) -> "nap.IntervalSet":
    """Create the legacy paired pre-ripple IntervalSet."""
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
        return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

    pre_ep = nap.IntervalSet(start=pre_start, end=pre_end).intersect(epoch_ep)
    if exclude_ripples and len(pre_ep) > 0 and len(ripple_ep) > 0:
        ripple_exclusion = nap.IntervalSet(
            start=np.asarray(ripple_ep.start, dtype=float) - float(exclude_ripple_guard_s),
            end=np.asarray(ripple_ep.end, dtype=float) + float(exclude_ripple_guard_s),
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
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """Return McFadden pseudo-R^2 per neuron against a train-fitted null."""
    ll_model = poisson_ll_per_neuron(y_test, lam_test)
    lam0 = np.mean(np.asarray(y_null_fit, dtype=float), axis=0, keepdims=True)
    lam0 = np.repeat(lam0, np.asarray(y_test).shape[0], axis=0)
    ll_null = poisson_ll_per_neuron(y_test, lam0)
    ll_null = np.where(np.abs(ll_null) < 1e-12, -1e-12, ll_null)
    return 1.0 - (ll_model / ll_null)


def poisson_ll_saturated_per_neuron(y: np.ndarray) -> np.ndarray:
    """Return saturated-model Poisson log-likelihood per neuron."""
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
    """Shuffle sample order independently within each neuron."""
    y = np.asarray(y)
    n_samples, n_neurons = y.shape
    indices = np.vstack([rng.permutation(n_samples) for _ in range(n_neurons)]).T
    return np.take_along_axis(y, indices, axis=0)


def _clear_jax_caches() -> None:
    """Best-effort JAX cache cleanup between fits."""
    try:
        import jax
    except ModuleNotFoundError:
        return
    jax.clear_caches()


def fit_ripple_glm_train_on_ripple_predict_pre(
    epoch: str,
    *,
    spikes: dict[str, Any],
    regions: tuple[str, str] = REGIONS,
    Kay_ripple_detector: dict[str, pd.DataFrame],
    ep: dict[str, "nap.IntervalSet"],
    min_spikes_per_ripple: float = DEFAULT_MIN_SPIKES_PER_RIPPLE,
    n_shuffles_ripple: int = DEFAULT_N_SHUFFLES_RIPPLE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ripple_window: float | None = None,
    pre_window_s: float | None = None,
    pre_buffer_s: float = DEFAULT_PRE_BUFFER_S,
    exclude_ripples: bool = True,
    pre_exclude_guard_s: float = DEFAULT_PRE_EXCLUDE_GUARD_S,
    n_splits: int = DEFAULT_N_SPLITS,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
    store_ripple_shuffle_preds: bool = True,
) -> dict[str, Any]:
    """Fit the legacy train-on-ripple workflow for one epoch."""
    import jax
    import nemos as nmo
    import pynapple as nap

    v1_region, ca1_region = regions
    rng = np.random.default_rng(random_seed)

    ripple_table = Kay_ripple_detector[epoch]
    if ripple_window is None:
        ripple_ep = nap.IntervalSet(
            start=ripple_table["start_time"].to_numpy(dtype=float),
            end=ripple_table["end_time"].to_numpy(dtype=float),
        )
    else:
        ripple_start = ripple_table["start_time"].to_numpy(dtype=float)
        ripple_ep = nap.IntervalSet(
            start=ripple_start,
            end=ripple_start + float(ripple_window),
        )

    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=ep[epoch],
        window_s=pre_window_s,
        buffer_s=pre_buffer_s,
        exclude_ripples=exclude_ripples,
        exclude_ripple_guard_s=pre_exclude_guard_s,
    )

    X_r = np.asarray(spikes[ca1_region].count(ep=ripple_ep), dtype=np.float64)
    y_r = np.asarray(spikes[v1_region].count(ep=ripple_ep), dtype=np.float64)
    X_p = np.asarray(spikes[ca1_region].count(ep=pre_ep), dtype=np.float64)
    y_p = np.asarray(spikes[v1_region].count(ep=pre_ep), dtype=np.float64)

    n_r, _ = X_r.shape
    n_p = X_p.shape[0]
    if n_r < n_splits:
        raise ValueError(f"Not enough ripples: n_ripples={n_r}, n_splits={n_splits}")

    for name, arr in (("X_r", X_r), ("y_r", y_r), ("X_p", X_p), ("y_p", y_p)):
        info = _first_nonfinite_info(arr)
        if info is not None:
            raise ValueError(f"Non-finite in {name}: {info}")

    keep_y = (y_r.sum(axis=0) / max(n_r, 1)) >= float(min_spikes_per_ripple)
    y_r = y_r[:, keep_y]
    y_p = y_p[:, keep_y]

    v1_unit_ids = np.array(list(spikes[v1_region].keys()))
    ca1_unit_ids = np.array(list(spikes[ca1_region].keys()))
    v1_unit_ids_kept = v1_unit_ids[keep_y]
    n_cells = y_r.shape[1]
    if n_cells == 0:
        raise ValueError(
            f"No V1 units passed min_spikes_per_ripple={min_spikes_per_ripple}. "
            f"n_ripples={n_r}"
        )

    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    y_r_test_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)
    y_r_hat_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)
    y_r_hat_shuff: np.ndarray | None = None
    if store_ripple_shuffle_preds and n_shuffles_ripple > 0:
        y_r_hat_shuff = np.full(
            (n_shuffles_ripple, n_r, n_cells),
            np.nan,
            dtype=np.float32,
        )

    pseudo_r2_ripple = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    mae_ripple = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    ll_ripple = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    devexp_ripple = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    bits_per_spike_ripple = np.full((n_splits, n_cells), np.nan, dtype=np.float32)

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
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_r)):
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
                "All X features were near-constant after filtering in this fold; cannot fit GLM."
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
        pseudo_r2_ripple[fold] = mcfadden_pseudo_r2_per_neuron(
            y_test_r,
            lam_r,
            y_train_r,
        ).astype(np.float32)
        mae_ripple[fold] = mae_per_neuron(y_test_r, lam_r).astype(np.float32)
        ll_ripple[fold] = poisson_ll_per_neuron(y_test_r, lam_r).astype(np.float32)
        devexp_ripple[fold] = deviance_explained_per_neuron(
            y_test=y_test_r,
            lam_test=lam_r,
            y_null_fit=y_train_r,
        ).astype(np.float32)
        bits_per_spike_ripple[fold] = bits_per_spike_per_neuron(
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
            lam_r_shuff = np.asarray(glm_shuffle.predict(X_test_r_pp), dtype=np.float64)

            if y_r_hat_shuff is not None:
                y_r_hat_shuff[shuffle_index, test_idx] = lam_r_shuff.astype(np.float32)

            pseudo_r2_ripple_shuff[fold, shuffle_index] = mcfadden_pseudo_r2_per_neuron(
                y_test_r,
                lam_r_shuff,
                y_train_r,
            ).astype(np.float32)
            mae_ripple_shuff[fold, shuffle_index] = mae_per_neuron(
                y_test_r,
                lam_r_shuff,
            ).astype(np.float32)
            ll_ripple_shuff[fold, shuffle_index] = poisson_ll_per_neuron(
                y_test_r,
                lam_r_shuff,
            ).astype(np.float32)
            devexp_ripple_shuff[fold, shuffle_index] = deviance_explained_per_neuron(
                y_test=y_test_r,
                lam_test=lam_r_shuff,
                y_null_fit=y_train_r,
            ).astype(np.float32)
            bits_per_spike_ripple_shuff[fold, shuffle_index] = bits_per_spike_per_neuron(
                y_test=y_test_r,
                lam_test=lam_r_shuff,
                y_null_fit=y_train_r,
            ).astype(np.float32)

            del glm_shuffle, y_train_shuffled, lam_r_shuff
            if (shuffle_index + 1) % 5 == 0:
                jax.clear_caches()
                gc.collect()

        fold_info.append(
            dict(
                fold=int(fold),
                train_idx=train_idx,
                test_idx=test_idx,
                keep_x=keep_x,
                x_mean=mean,
                x_std=std,
                n_ripples=int(n_r),
                n_pre=int(n_p),
            )
        )
        del glm, lam_r
        jax.clear_caches()
        gc.collect()

    if np.isnan(y_r_test_full).any() or np.isnan(y_r_hat_full).any():
        raise ValueError("Some ripple rows were never assigned to a held-out fold.")
    if y_r_hat_shuff is not None and np.isnan(y_r_hat_shuff).any():
        raise ValueError("Some shuffled ripple rows were never assigned to a held-out fold.")

    X_r_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(X_r, X_r)
    if X_r_all_pp.shape[1] == 0:
        raise ValueError("All X features were near-constant after filtering; cannot fit GLM.")

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
    lam_pre_f32 = lam_pre.astype(np.float32)

    pseudo_r2_pre = mcfadden_pseudo_r2_per_neuron(y_p, lam_pre, y_p).astype(np.float32)
    mae_pre = mae_per_neuron(y_p, lam_pre).astype(np.float32)
    ll_pre = poisson_ll_per_neuron(y_p, lam_pre).astype(np.float32)
    devexp_pre = deviance_explained_per_neuron(
        y_test=y_p,
        lam_test=lam_pre,
        y_null_fit=y_p,
    ).astype(np.float32)
    bits_per_spike_pre = bits_per_spike_per_neuron(
        y_test=y_p,
        lam_test=lam_pre,
        y_null_fit=y_p,
    ).astype(np.float32)

    del glm_all
    jax.clear_caches()
    gc.collect()

    return {
        "epoch": epoch,
        "random_seed": int(random_seed),
        "min_spikes_per_ripple": float(min_spikes_per_ripple),
        "n_splits": int(n_splits),
        "n_shuffles_ripple": int(n_shuffles_ripple),
        "ripple_window": None if ripple_window is None else float(ripple_window),
        "pre_window_s": None if pre_window_s is None else float(pre_window_s),
        "pre_buffer_s": float(pre_buffer_s),
        "pre_exclude_guard_s": float(pre_exclude_guard_s),
        "exclude_ripples": bool(exclude_ripples),
        "n_ripples": int(n_r),
        "n_pre": int(n_p),
        "n_cells": int(n_cells),
        "v1_unit_ids": v1_unit_ids_kept,
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
        "yhat_pre": lam_pre_f32,
        "pseudo_r2_pre": pseudo_r2_pre,
        "mae_pre": mae_pre,
        "ll_pre": ll_pre,
        "devexp_pre": devexp_pre,
        "bits_per_spike_pre": bits_per_spike_pre,
        "keep_x_all": keep_x_all,
        "x_mean_all": mean_all,
        "x_std_all": std_all,
    }


def fit_ripple_glm_ridge_sweep_no_shuffles(
    epoch: str,
    *,
    spikes: dict[str, Any],
    regions: tuple[str, str] = REGIONS,
    Kay_ripple_detector: dict[str, pd.DataFrame],
    ep: dict[str, "nap.IntervalSet"],
    ridge_strengths: list[float] | tuple[float, ...] | np.ndarray,
    min_spikes_per_ripple: float = DEFAULT_MIN_SPIKES_PER_RIPPLE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ripple_window: float | None = None,
    pre_window_s: float | None = None,
    pre_buffer_s: float = DEFAULT_PRE_BUFFER_S,
    exclude_ripples: bool = True,
    pre_exclude_guard_s: float = DEFAULT_PRE_EXCLUDE_GUARD_S,
    n_splits: int = DEFAULT_N_SPLITS,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    """Run the legacy ridge sweep by repeatedly fitting the no-shuffle model."""
    ridge_array = np.asarray(list(ridge_strengths), dtype=float)
    if ridge_array.ndim != 1 or ridge_array.size == 0:
        raise ValueError("ridge_strengths must be a non-empty 1D sequence of floats.")

    fit_results = []
    for ridge_strength in ridge_array:
        fit_results.append(
            fit_ripple_glm_train_on_ripple_predict_pre(
                epoch=epoch,
                spikes=spikes,
                regions=regions,
                Kay_ripple_detector=Kay_ripple_detector,
                ep=ep,
                min_spikes_per_ripple=min_spikes_per_ripple,
                n_shuffles_ripple=0,
                random_seed=random_seed,
                ripple_window=ripple_window,
                pre_window_s=pre_window_s,
                pre_buffer_s=pre_buffer_s,
                exclude_ripples=exclude_ripples,
                pre_exclude_guard_s=pre_exclude_guard_s,
                n_splits=n_splits,
                ridge_strength=float(ridge_strength),
                maxiter=maxiter,
                tol=tol,
                store_ripple_shuffle_preds=False,
            )
        )

    first_result = fit_results[0]
    pseudo_r2_ripple = np.stack(
        [np.asarray(result["pseudo_r2_ripple_folds"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    devexp_ripple = np.stack(
        [np.asarray(result["devexp_ripple_folds"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    ll_ripple = np.stack(
        [np.asarray(result["ll_ripple_folds"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    bits_per_spike_ripple = np.stack(
        [np.asarray(result["bits_per_spike_ripple_folds"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    pseudo_r2_pre = np.stack(
        [np.asarray(result["pseudo_r2_pre"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    devexp_pre = np.stack(
        [np.asarray(result["devexp_pre"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    ll_pre = np.stack(
        [np.asarray(result["ll_pre"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)
    bits_per_spike_pre = np.stack(
        [np.asarray(result["bits_per_spike_pre"], dtype=float) for result in fit_results],
        axis=0,
    ).astype(np.float32)

    return {
        "epoch": epoch,
        "random_seed": int(random_seed),
        "ridge_strengths": ridge_array.astype(np.float64),
        "min_spikes_per_ripple": float(min_spikes_per_ripple),
        "n_splits": int(n_splits),
        "ripple_window": None if ripple_window is None else float(ripple_window),
        "pre_window_s": None if pre_window_s is None else float(pre_window_s),
        "pre_buffer_s": float(pre_buffer_s),
        "pre_exclude_guard_s": float(pre_exclude_guard_s),
        "exclude_ripples": bool(exclude_ripples),
        "n_ripples": int(first_result["n_ripples"]),
        "n_pre": int(first_result["n_pre"]),
        "n_cells": int(first_result["n_cells"]),
        "v1_unit_ids": first_result["v1_unit_ids"],
        "ca1_unit_ids": first_result["ca1_unit_ids"],
        "fold_info": first_result["fold_info"],
        "pseudo_r2_ripple_folds": pseudo_r2_ripple,
        "devexp_ripple_folds": devexp_ripple,
        "ll_ripple_folds": ll_ripple,
        "bits_per_spike_ripple_folds": bits_per_spike_ripple,
        "pseudo_r2_pre": pseudo_r2_pre,
        "devexp_pre": devexp_pre,
        "ll_pre": ll_pre,
        "bits_per_spike_pre": bits_per_spike_pre,
        "pseudo_r2_ripple_pop_mean": np.nanmean(np.nanmean(pseudo_r2_ripple, axis=1), axis=1),
        "devexp_ripple_pop_mean": np.nanmean(np.nanmean(devexp_ripple, axis=1), axis=1),
        "ll_ripple_pop_mean": np.nanmean(np.nanmean(ll_ripple, axis=1), axis=1),
        "bits_per_spike_ripple_pop_mean": np.nanmean(
            np.nanmean(bits_per_spike_ripple, axis=1),
            axis=1,
        ),
        "pseudo_r2_pre_pop_mean": np.nanmean(pseudo_r2_pre, axis=1),
        "devexp_pre_pop_mean": np.nanmean(devexp_pre, axis=1),
        "ll_pre_pop_mean": np.nanmean(ll_pre, axis=1),
        "bits_per_spike_pre_pop_mean": np.nanmean(bits_per_spike_pre, axis=1),
    }


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


def _plot_ripple_metric_summary(
    *,
    real_values: np.ndarray,
    null_samples: np.ndarray,
    higher_is_better: bool,
    animal_name: str,
    date: str,
    epoch: str,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot one ripple metric against its shuffle null."""
    plt = _get_pyplot()

    real_values = _as_1d_float(real_values)
    null_samples = np.asarray(null_samples, dtype=float)
    pooled_null = _finite_values(null_samples)
    bins = _metric_histogram_bins(real_values, pooled_null)
    p_values = empirical_p_values(
        observed=real_values,
        null_samples=null_samples,
        higher_is_better=higher_is_better,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    hist_ax, sig_ax = axes
    real_finite = _finite_values(real_values)
    if pooled_null.size:
        hist_ax.hist(
            pooled_null,
            bins=bins,
            weights=np.full(pooled_null.size, 1.0 / pooled_null.size),
            alpha=0.55,
            color=SHUFFLE_COLOR,
            label="Shuffle",
            edgecolor="none",
        )
    if real_finite.size:
        hist_ax.hist(
            real_finite,
            bins=bins,
            weights=np.full(real_finite.size, 1.0 / real_finite.size),
            alpha=0.55,
            color=RIPPLE_COLOR,
            label="Ripple",
            edgecolor="none",
        )
    hist_ax.set_xlabel(metric_label)
    hist_ax.set_ylabel("Fraction of units")
    hist_ax.set_title("Ripple vs shuffle")
    if pooled_null.size or real_finite.size:
        hist_ax.legend(loc="upper right")

    valid = np.isfinite(real_values) & np.isfinite(p_values)
    if np.any(valid):
        sig_ax.scatter(
            real_values[valid],
            -np.log10(np.clip(p_values[valid], 1e-12, 1.0)),
            s=18,
            alpha=0.7,
            color=RIPPLE_COLOR,
        )
    sig_ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, color="black")
    sig_ax.set_xlabel(metric_label)
    sig_ax.set_ylabel(r"$-\log_{10}(p)$")
    sig_ax.set_title("Ripple effect size vs significance")

    fig.suptitle(f"{animal_name} {date} {epoch} ripple {metric_label}", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_pre_metric_summary(
    *,
    metric_values: np.ndarray,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot one pre-ripple metric against pre spikes per unit."""
    plt = _get_pyplot()

    metric_values = _as_1d_float(metric_values)
    y_pre_test = _as_2d_float(y_pre_test)
    if y_pre_test.ndim != 2 or y_pre_test.shape[1] != metric_values.shape[0]:
        raise ValueError(
            f"y_pre_test must be (n_pre, N={metric_values.shape[0]}), got {y_pre_test.shape}"
        )
    spikes_per_unit = np.sum(y_pre_test, axis=0).astype(float)
    bins = _metric_histogram_bins(metric_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    hist_ax, scatter_ax = axes
    metric_finite = _finite_values(metric_values)
    if metric_finite.size:
        hist_ax.hist(
            metric_finite,
            bins=bins,
            weights=np.full(metric_finite.size, 1.0 / metric_finite.size),
            alpha=0.65,
            color=PRE_COLOR,
            edgecolor="none",
        )
    hist_ax.set_xlabel(metric_label)
    hist_ax.set_ylabel("Fraction of units")
    hist_ax.set_title("Pre-ripple distribution")

    valid = np.isfinite(metric_values) & np.isfinite(spikes_per_unit)
    if np.any(valid):
        scatter_ax.scatter(
            spikes_per_unit[valid],
            metric_values[valid],
            s=18,
            alpha=0.7,
            color=PRE_COLOR,
        )
    scatter_ax.set_xlabel("Pre spikes per unit")
    scatter_ax.set_ylabel(metric_label)
    scatter_ax.set_title("Metric vs pre spikes")

    fig.suptitle(f"{animal_name} {date} {epoch} pre {metric_label}", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_ripple_glm_pseudo_r2(
    *,
    pseudo_r2_real_folds: Any,
    pseudo_r2_shuff_folds: Any,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    null_mode: str = "pooled",
) -> Path:
    """Plot held-out ripple pseudo-R^2 against the shuffle null."""
    del null_mode
    real_mean = np.nanmean(_as_2d_float(pseudo_r2_real_folds), axis=0)
    shuffle_mean = np.nanmean(_as_3d_float(pseudo_r2_shuff_folds), axis=0)
    return _plot_ripple_metric_summary(
        real_values=real_mean,
        null_samples=shuffle_mean,
        higher_is_better=True,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        metric_label="Pseudo R^2",
        out_path=out_path,
    )


def plot_ripple_glm_mae(
    *,
    mae_real_folds: Any,
    mae_shuff_folds: Any,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    null_mode: str = "pooled",
) -> Path:
    """Plot held-out ripple MAE against the shuffle null."""
    del null_mode
    real_mean = np.nanmean(_as_2d_float(mae_real_folds), axis=0)
    shuffle_mean = np.nanmean(_as_3d_float(mae_shuff_folds), axis=0)
    return _plot_ripple_metric_summary(
        real_values=real_mean,
        null_samples=shuffle_mean,
        higher_is_better=False,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        metric_label="MAE",
        out_path=out_path,
    )


def plot_ripple_glm_delta_ll(
    *,
    ll_real_folds: Any,
    ll_shuff_folds: Any,
    y_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    normalize: str = "bits_per_spike",
    null_mode: str = "pooled",
) -> Path:
    """Plot ripple held-out log-likelihood in a legacy-compatible summary form."""
    del null_mode
    real_mean = np.nanmean(_as_2d_float(ll_real_folds), axis=0)
    shuffle_mean = np.nanmean(_as_3d_float(ll_shuff_folds), axis=0)

    if normalize == "bits_per_spike":
        spikes = np.sum(_as_2d_float(y_test), axis=0)
        denom = np.clip(spikes * np.log(2.0), 1e-12, None)
        real_mean = np.where(spikes > 0, real_mean / denom, np.nan)
        shuffle_mean = np.where(spikes[None, :] > 0, shuffle_mean / denom[None, :], np.nan)
        metric_label = "Delta bits/spike"
    else:
        metric_label = "Delta log-likelihood"

    return _plot_ripple_metric_summary(
        real_values=real_mean,
        null_samples=shuffle_mean,
        higher_is_better=True,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        metric_label=metric_label,
        out_path=out_path,
    )


def plot_pre_pseudo_r2_summary(
    *,
    pseudo_r2_pre: Any,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
) -> Path:
    """Plot pre-ripple pseudo-R^2 across units."""
    return _plot_pre_metric_summary(
        metric_values=_as_1d_float(pseudo_r2_pre),
        y_pre_test=y_pre_test,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        metric_label="Pseudo R^2",
        out_path=out_path,
    )


def plot_pre_mae_summary(
    *,
    mae_pre: Any,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
) -> Path:
    """Plot pre-ripple MAE across units."""
    return _plot_pre_metric_summary(
        metric_values=_as_1d_float(mae_pre),
        y_pre_test=y_pre_test,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        metric_label="MAE",
        out_path=out_path,
    )


def plot_pre_ll_summary(
    *,
    ll_pre: Any,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    normalize: str = "bits_per_spike",
) -> Path:
    """Plot pre-ripple log-likelihood across units."""
    ll_values = _as_1d_float(ll_pre)
    if normalize == "bits_per_spike":
        y_pre = _as_2d_float(y_pre_test)
        spikes = np.sum(y_pre, axis=0)
        denom = np.clip(spikes * np.log(2.0), 1e-12, None)
        ll_values = np.where(spikes > 0, ll_values / denom, np.nan)
        metric_label = "Delta bits/spike"
    else:
        metric_label = "Log-likelihood"

    return _plot_pre_metric_summary(
        metric_values=ll_values,
        y_pre_test=y_pre_test,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        metric_label=metric_label,
        out_path=out_path,
    )


def plot_ridge_sweep_summary(
    *,
    results: dict[str, Any],
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
) -> Path:
    """Plot one compact ridge-sweep summary across ridge strengths."""
    plt = _get_pyplot()

    ridge_strengths = _as_1d_float(results["ridge_strengths"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    metric_specs = (
        ("pseudo_r2_ripple_pop_mean", "Ripple pseudo R^2"),
        ("devexp_ripple_pop_mean", "Ripple deviance explained"),
        ("ll_ripple_pop_mean", "Ripple log-likelihood"),
        ("bits_per_spike_ripple_pop_mean", "Ripple bits/spike"),
    )
    for ax, (metric_name, metric_label) in zip(axes.flat, metric_specs, strict=False):
        values = _as_1d_float(results[metric_name])
        ax.plot(ridge_strengths, values, marker="o", color=RIPPLE_COLOR)
        ax.set_xscale("log")
        ax.set_xlabel("Ridge strength")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)

    fig.suptitle(f"{animal_name} {date} {epoch} ridge sweep", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_train_on_ripple_figures(
    *,
    results: dict[str, Any],
    fig_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    exclude_ripples: bool,
    ripple_window_s: float,
) -> list[Path]:
    """Save the six legacy train-on-ripple summary figures."""
    figure_paths = build_legacy_figure_paths(
        fig_dir,
        epoch=epoch,
        exclude_ripples=exclude_ripples,
        ripple_window_s=ripple_window_s,
    )
    return [
        plot_ripple_glm_pseudo_r2(
            pseudo_r2_real_folds=results["pseudo_r2_ripple_folds"],
            pseudo_r2_shuff_folds=results["pseudo_r2_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=figure_paths["ripple_pseudo_r2"],
        ),
        plot_ripple_glm_mae(
            mae_real_folds=results["mae_ripple_folds"],
            mae_shuff_folds=results["mae_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=figure_paths["ripple_mae"],
        ),
        plot_ripple_glm_delta_ll(
            ll_real_folds=results["ll_ripple_folds"],
            ll_shuff_folds=results["ll_ripple_shuff_folds"],
            y_test=results["y_ripple_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=figure_paths["ripple_delta_bits_per_spike"],
            normalize="bits_per_spike",
        ),
        plot_pre_pseudo_r2_summary(
            pseudo_r2_pre=results["pseudo_r2_pre"],
            y_pre_test=results["y_pre_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=figure_paths["pre_pseudo_r2"],
        ),
        plot_pre_mae_summary(
            mae_pre=results["mae_pre"],
            y_pre_test=results["y_pre_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=figure_paths["pre_mae"],
        ),
        plot_pre_ll_summary(
            ll_pre=results["ll_pre"],
            y_pre_test=results["y_pre_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=figure_paths["pre_delta_bits_per_spike"],
            normalize="bits_per_spike",
        ),
    ]


def run_train_on_ripple(
    *,
    args: argparse.Namespace,
    session: dict[str, Any],
    selected_epochs: list[str],
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    """Run the active legacy train-on-ripple workflow."""
    out_dir = session["analysis_path"] / "ripple" / "glm_results_train_on_ripple3"
    fig_dir = session["analysis_path"] / "figs" / "ripple" / "train_on_ripple3"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pre_window_s = args.pre_window_s if args.pre_window_s is not None else args.ripple_window_s
    saved_results: list[Path] = []
    saved_figures: list[Path] = []
    skipped_epochs: list[dict[str, Any]] = []

    for epoch in selected_epochs:
        result_path = build_legacy_result_path(
            out_dir,
            epoch=epoch,
            exclude_ripples=args.exclude_ripples,
            ripple_window_s=args.ripple_window_s,
            min_spikes_per_ripple=args.min_spikes_per_ripple,
            ridge_strength=args.ridge_strength,
        )
        if result_path.exists() and not args.overwrite:
            print(f"Loading existing results: {result_path}")
            results = load_results_npz(result_path)
        else:
            print(f"Computing results for: {epoch}")
            try:
                results = fit_ripple_glm_train_on_ripple_predict_pre(
                    epoch=epoch,
                    spikes=session["spikes"],
                    regions=REGIONS,
                    Kay_ripple_detector=session["ripple_tables"],
                    ep=session["epoch_intervals"],
                    min_spikes_per_ripple=args.min_spikes_per_ripple,
                    n_shuffles_ripple=args.n_shuffles_ripple,
                    random_seed=args.random_seed,
                    ripple_window=args.ripple_window_s,
                    pre_window_s=pre_window_s,
                    pre_buffer_s=args.pre_buffer_s,
                    exclude_ripples=args.exclude_ripples,
                    pre_exclude_guard_s=args.pre_exclude_guard_s,
                    n_splits=args.n_splits,
                    ridge_strength=args.ridge_strength,
                    maxiter=args.maxiter,
                    tol=args.tol,
                )
            except Exception as exc:
                skipped_epochs.append({"epoch": epoch, "reason": str(exc)})
                print(f"Skipping {epoch}: {exc}")
                continue
            save_results_npz(result_path, results)
            print(f"Saved: {result_path}")

        saved_results.append(result_path)
        saved_figures.extend(
            save_train_on_ripple_figures(
                results=results,
                fig_dir=fig_dir,
                animal_name=args.animal_name,
                date=args.date,
                epoch=epoch,
                exclude_ripples=args.exclude_ripples,
                ripple_window_s=args.ripple_window_s,
            )
        )

    return saved_results, saved_figures, skipped_epochs


def run_ridge_sweep(
    *,
    args: argparse.Namespace,
    session: dict[str, Any],
    selected_epochs: list[str],
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    """Run the legacy ridge-sweep utility on legacy pickle inputs."""
    out_dir = session["analysis_path"] / "ripple" / "glm_ridge_sweep"
    fig_dir = session["analysis_path"] / "figs" / "ripple" / "glm_ridge_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pre_window_s = args.pre_window_s if args.pre_window_s is not None else args.ripple_window_s
    saved_results: list[Path] = []
    saved_figures: list[Path] = []
    skipped_epochs: list[dict[str, Any]] = []

    for epoch in selected_epochs:
        result_path = (
            out_dir
            / (
                f"{epoch}_ridge_sweep_exclude_{args.exclude_ripples}_"
                f"ripple_window_{args.ripple_window_s}_minspk_{args.min_spikes_per_ripple}.npz"
            )
        )
        if result_path.exists() and not args.overwrite:
            print(f"Loading existing sweep: {result_path}")
            results = load_results_npz(result_path)
        else:
            print(f"Running ridge sweep: {epoch}")
            try:
                results = fit_ripple_glm_ridge_sweep_no_shuffles(
                    epoch=epoch,
                    spikes=session["spikes"],
                    regions=REGIONS,
                    Kay_ripple_detector=session["ripple_tables"],
                    ep=session["epoch_intervals"],
                    ridge_strengths=args.ridge_strengths,
                    min_spikes_per_ripple=args.min_spikes_per_ripple,
                    random_seed=args.random_seed,
                    ripple_window=args.ripple_window_s,
                    pre_window_s=pre_window_s,
                    pre_buffer_s=args.pre_buffer_s,
                    exclude_ripples=args.exclude_ripples,
                    pre_exclude_guard_s=args.pre_exclude_guard_s,
                    n_splits=args.n_splits,
                    maxiter=args.maxiter,
                    tol=args.tol,
                )
            except Exception as exc:
                skipped_epochs.append({"epoch": epoch, "reason": str(exc)})
                print(f"Skipping ridge sweep for {epoch}: {exc}")
                continue
            save_results_npz(result_path, results)
            print(f"Saved: {result_path}")

        figure_path = (
            fig_dir
            / (
                f"{epoch}_ridge_sweep_exclude_{args.exclude_ripples}_"
                f"ripple_window_{args.ripple_window_s}_summary.png"
            )
        )
        saved_results.append(result_path)
        saved_figures.append(
            plot_ridge_sweep_summary(
                results=results,
                animal_name=args.animal_name,
                date=args.date,
                epoch=epoch,
                out_path=figure_path,
            )
        )

    return saved_results, saved_figures, skipped_epochs


def main(argv: list[str] | None = None) -> None:
    """Run the ripple_glm3 CLI."""
    args = parse_arguments(argv)
    validate_arguments(args)
    configure_jax_environment(args.cuda_visible_devices)

    session = prepare_ripple_glm3_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    requested_epochs = validate_epochs(session["epoch_tags"], args.epochs)
    selected_epochs = [
        epoch
        for epoch in requested_epochs
        if epoch in set(session["ripple_event_epochs"])
    ]
    if not selected_epochs:
        raise ValueError(
            "No selected epochs are present in ripple/Kay_ripple_detector.pkl. "
            f"Requested epochs: {requested_epochs!r}"
        )
    validate_selected_epochs_across_sources(
        selected_epochs,
        source_epochs={
            "timestamps_ephys": session["timestamps_ephys_by_epoch"],
            "ripple_events": session["ripple_tables"],
            "epoch_intervals": session["epoch_intervals"],
        },
    )

    if args.mode == "train_on_ripple":
        saved_results, saved_figures, skipped_epochs = run_train_on_ripple(
            args=args,
            session=session,
            selected_epochs=selected_epochs,
        )
    else:
        saved_results, saved_figures, skipped_epochs = run_ridge_sweep(
            args=args,
            session=session,
            selected_epochs=selected_epochs,
        )

    if not saved_results:
        raise RuntimeError(f"All selected epochs were skipped. Reasons: {skipped_epochs!r}")

    log_path = write_run_log(
        analysis_path=session["analysis_path"],
        script_name="v1ca1.ripple.ripple_glm3",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "epochs": selected_epochs,
            "mode": args.mode,
            "ripple_window_s": args.ripple_window_s,
            "pre_window_s": (
                args.pre_window_s if args.pre_window_s is not None else args.ripple_window_s
            ),
            "pre_buffer_s": args.pre_buffer_s,
            "exclude_ripples": args.exclude_ripples,
            "pre_exclude_guard_s": args.pre_exclude_guard_s,
            "min_spikes_per_ripple": args.min_spikes_per_ripple,
            "n_shuffles_ripple": args.n_shuffles_ripple,
            "random_seed": args.random_seed,
            "n_splits": args.n_splits,
            "ridge_strength": args.ridge_strength,
            "ridge_strengths": list(args.ridge_strengths),
            "maxiter": args.maxiter,
            "tol": args.tol,
            "cuda_visible_devices": args.cuda_visible_devices,
            "overwrite": args.overwrite,
            "sources": session["sources"],
        },
        outputs={
            "saved_results": saved_results,
            "saved_figures": saved_figures,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
