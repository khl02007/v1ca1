from __future__ import annotations

"""Fit ripple-time population GLMs on within-ripple spike-count bins.

This CLI loads one session through the shared session helpers, converts each
ripple into fixed-width within-ripple spike-count bins, and fits CA1-to-V1
population GLMs on those ripple bins with contiguous ripple-wise
cross-validation. The primary model uses lagged CA1 ripple bins as predictors;
the baseline model uses only within-ripple phase basis functions. Results are
saved as one compressed `.npz` payload per epoch under
`ripple_glm_binned/`, summary figures are written under
`figs/ripple_glm_binned/`, and a JSON run log is written under
`v1ca1_log/`.
"""

import argparse
import gc
import os
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
    REGIONS,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
    load_spikes_by_region,
)
from v1ca1.ripple.ripple_glm import build_epoch_intervals, load_ripple_tables

if TYPE_CHECKING:
    import pynapple as nap


TARGET_REGION = "v1"
SOURCE_REGION = "ca1"
DEFAULT_BIN_SIZE_S = 0.01
DEFAULT_MAX_LAG_BINS = 0
DEFAULT_CUDA_VISIBLE_DEVICES = "7"
DEFAULT_MIN_TOTAL_SPIKES = 10
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES_RIPPLE = 100
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_MAXITER = 6000
DEFAULT_TOL = 1e-7
DEFAULT_BASELINE_N_BASIS = 5
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_XLA_PREALLOCATE = "false"
DEFAULT_XLA_MEM_FRACTION = "0.70"


def _parse_cuda_visible_devices(argv: list[str] | None = None) -> str:
    """Return the requested CUDA device visibility before JAX import."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        type=str,
        help=(
            "Value assigned to CUDA_VISIBLE_DEVICES before JAX/NEMOS import. "
            "Examples: '0', '1', '0,1', or '' to hide CUDA devices."
        ),
    )
    parsed, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return str(parsed.cuda_visible_devices)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the binned ripple GLM workflow."""
    parser = argparse.ArgumentParser(
        description="Fit ripple-time CA1-to-V1 population GLMs on within-ripple bins"
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
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Bin size in seconds for within-ripple spike counts. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--max-lag-bins",
        type=int,
        default=DEFAULT_MAX_LAG_BINS,
        help=(
            "Maximum number of causal CA1 lag bins to include. "
            f"Default: {DEFAULT_MAX_LAG_BINS}"
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help=(
            "Value assigned to CUDA_VISIBLE_DEVICES before JAX/NEMOS import. "
            f"Default: {DEFAULT_CUDA_VISIBLE_DEVICES!r}"
        ),
    )
    parser.add_argument(
        "--min-total-spikes",
        type=int,
        default=DEFAULT_MIN_TOTAL_SPIKES,
        help=(
            "Minimum total ripple spikes required to keep one V1 unit. "
            f"Default: {DEFAULT_MIN_TOTAL_SPIKES}"
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of ripple cross-validation folds. Default: {DEFAULT_N_SPLITS}",
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
        "--baseline-n-basis",
        type=int,
        default=DEFAULT_BASELINE_N_BASIS,
        help=(
            "Number of within-ripple phase spline basis functions for the baseline model. "
            f"Default: {DEFAULT_BASELINE_N_BASIS}"
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for response shuffles. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to recompute and overwrite existing saved results. "
            "Use --no-overwrite to reuse an existing `.npz` payload when present."
        ),
    )
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI ranges."""
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.max_lag_bins < 0:
        raise ValueError("--max-lag-bins must be non-negative.")
    if args.min_total_spikes < 0:
        raise ValueError("--min-total-spikes must be non-negative.")
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
    if args.baseline_n_basis <= 0:
        raise ValueError("--baseline-n-basis must be positive.")


def validate_selected_epochs(
    available_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return selected epochs after validating any requested subset."""
    if requested_epochs is None:
        return list(available_epochs)

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in available_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in the saved session epochs {available_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return list(requested_epochs)


def configure_jax_environment(cuda_visible_devices: str) -> None:
    """Configure CUDA visibility before importing JAX/NEMOS."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = DEFAULT_XLA_PREALLOCATE
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = DEFAULT_XLA_MEM_FRACTION


def require_glm_modules() -> tuple[Any, Any]:
    """Import and return JAX and NEMOS after CUDA env configuration."""
    try:
        import jax
        import nemos as nmo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `jax` and `nemos`. Install the `.[glm]` extra "
            "or an equivalent environment before running it."
        ) from exc
    return jax, nmo


def empty_ripple_table() -> pd.DataFrame:
    """Return an empty ripple-event table."""
    return pd.DataFrame(
        {
            "start_time": pd.Series(dtype=float),
            "end_time": pd.Series(dtype=float),
        }
    )


def prepare_binned_ripple_glm_session(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Load one session's timestamps, spikes, and ripple tables."""
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
        epoch: loaded_ripple_tables.get(epoch, empty_ripple_table())
        for epoch in epoch_tags
    }
    for epoch, table in loaded_ripple_tables.items():
        if epoch not in ripple_tables:
            ripple_tables[epoch] = table

    return {
        "analysis_path": analysis_path,
        "epoch_tags": epoch_tags,
        "timestamps_ephys_by_epoch": timestamps_ephys_by_epoch,
        "epoch_intervals": build_epoch_intervals(timestamps_ephys_by_epoch),
        "spikes_by_region": spikes_by_region,
        "ripple_tables": ripple_tables,
        "sources": {
            "timestamps_ephys": ephys_source,
            "timestamps_ephys_all": ephys_all_source,
            "sorting": "spikeinterface",
            "ripple_events": ripple_source,
        },
    }


def save_results_npz(out_path: Path, results: dict[str, Any]) -> None:
    """Save one result payload while preserving object-valued fields."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    to_save = dict(results)
    if "fold_info" in to_save:
        to_save["fold_info"] = np.array(to_save["fold_info"], dtype=object)

    for key, value in list(to_save.items()):
        if value is None:
            to_save[key] = np.array(None, dtype=object)

    np.savez_compressed(out_path, **to_save)


def load_results_npz(path: Path) -> dict[str, Any]:
    """Load one saved result payload."""
    data = np.load(path, allow_pickle=True)
    results = dict(data)

    if "fold_info" in results:
        results["fold_info"] = results["fold_info"].tolist()

    for key, value in list(results.items()):
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            maybe_obj = value.item()
            if maybe_obj is None:
                results[key] = None

    return results


def poisson_ll_per_neuron(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Return the Poisson log-likelihood per neuron."""
    lam = np.clip(lam, 1e-12, None)
    return np.sum(y * np.log(lam) - lam - gammaln(y + 1), axis=0)


def mcfadden_pseudo_r2_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Return McFadden pseudo-R^2 per neuron."""
    ll_model = poisson_ll_per_neuron(y_test, lam_test)

    lam0 = np.mean(y_train, axis=0, keepdims=True)
    lam0 = np.repeat(lam0, y_test.shape[0], axis=0)
    ll_null = poisson_ll_per_neuron(y_test, lam0)
    ll_null = np.where(np.abs(ll_null) < 1e-12, -1e-12, ll_null)
    return 1.0 - (ll_model / ll_null)


def poisson_ll_saturated_per_neuron(y: np.ndarray) -> np.ndarray:
    """Return the saturated-model Poisson log-likelihood per neuron."""
    y = np.asarray(y, dtype=float)
    y_safe = np.clip(y, 1e-12, None)
    return np.sum(y * np.log(y_safe) - y - gammaln(y + 1), axis=0)


def deviance_explained_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """Return deviance explained per neuron."""
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
    """Return bits/spike per neuron."""
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

    n_spikes = np.sum(y_test, axis=0)
    denom = np.clip(n_spikes * np.log(2.0), 1e-12, None)
    bps = (ll_model - ll_null) / denom
    return np.where(n_spikes > 0, bps, np.nan)


def mae_per_neuron(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the mean absolute error per neuron."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def shuffle_time_per_neuron(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each response column independently across time."""
    y = np.asarray(y)
    n_samples, n_neurons = y.shape
    idx = np.vstack([rng.permutation(n_samples) for _ in range(n_neurons)]).T
    return np.take_along_axis(y, idx, axis=0)


def _first_nonfinite_info(array: np.ndarray) -> str | None:
    """Return the first non-finite value summary when present."""
    array = np.asarray(array)
    mask = ~np.isfinite(array)
    if not np.any(mask):
        return None
    index = np.argwhere(mask)[0]
    value = array[tuple(index)]
    return f"shape={array.shape}, first_bad_idx={tuple(index)}, value={value}"


def _preprocess_X_fit_apply(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize one predictor matrix pair using the training partition."""
    keep_x = X_train.std(axis=0) > 1e-6
    X_train_k = X_train[:, keep_x]
    X_test_k = X_test[:, keep_x]

    if X_train_k.shape[1] == 0:
        return (
            np.empty((X_train.shape[0], 0), dtype=np.float64),
            np.empty((X_test.shape[0], 0), dtype=np.float64),
            keep_x,
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    mean = X_train_k.mean(axis=0, keepdims=True)
    std = X_train_k.std(axis=0, keepdims=True)

    X_train_pp = (X_train_k - mean) / (std + 1e-8)
    X_test_pp = (X_test_k - mean) / (std + 1e-8)

    X_train_pp /= np.sqrt(max(X_train_pp.shape[1], 1))
    X_test_pp /= np.sqrt(max(X_train_pp.shape[1], 1))

    X_train_pp = np.clip(X_train_pp, -10.0, 10.0)
    X_test_pp = np.clip(X_test_pp, -10.0, 10.0)

    return X_train_pp, X_test_pp, keep_x, mean.ravel(), std.ravel()


def _make_glm(
    *,
    ridge_strength: float,
    maxiter: int,
    tol: float,
) -> Any:
    """Construct one NEMOS population GLM."""
    _jax, nmo = require_glm_modules()
    return nmo.glm.PopulationGLM(
        solver_name="LBFGS",
        regularizer="Ridge",
        regularizer_strength=float(ridge_strength),
        solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
    )


def _count_unit_spikes_in_full_bins(
    tsgroup: "nap.TsGroup",
    unit_ids: np.ndarray,
    start_s: float,
    end_s: float,
    bin_size_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Count spikes in complete fixed-width bins for one interval."""
    duration_s = float(end_s) - float(start_s)
    n_bins = int(np.floor(duration_s / float(bin_size_s) + 1e-12))
    if n_bins <= 0:
        return (
            np.empty((0, len(unit_ids)), dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    effective_end = float(start_s) + n_bins * float(bin_size_s)
    bin_edges = float(start_s) + np.arange(n_bins + 1, dtype=np.float64) * float(
        bin_size_s
    )
    counts = np.zeros((n_bins, len(unit_ids)), dtype=np.float64)

    for column_index, unit_id in enumerate(unit_ids):
        spike_times = np.asarray(tsgroup[unit_id].t, dtype=np.float64)
        left = np.searchsorted(spike_times, float(start_s), side="left")
        right = np.searchsorted(spike_times, effective_end, side="left")
        local_times = spike_times[left:right]
        counts[:, column_index] = np.diff(
            np.searchsorted(local_times, bin_edges, side="left")
        )

    return counts, bin_edges


def build_binned_ripple_dataset(
    *,
    epoch: str,
    spikes_by_region: dict[str, "nap.TsGroup"],
    ripple_table: pd.DataFrame,
    epoch_interval: "nap.IntervalSet",
    regions: tuple[str, str] = (TARGET_REGION, SOURCE_REGION),
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_bins: int = DEFAULT_MAX_LAG_BINS,
) -> dict[str, np.ndarray]:
    """Build one ripple-binned CA1 predictor and V1 response dataset."""
    if bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    if max_lag_bins < 0:
        raise ValueError("max_lag_bins must be >= 0")

    v1_region, ca1_region = regions
    ca1_unit_ids = np.asarray(list(spikes_by_region[ca1_region].keys()))
    v1_unit_ids = np.asarray(list(spikes_by_region[v1_region].keys()))

    epoch_start = float(np.min(np.asarray(epoch_interval.start, dtype=float)))
    epoch_end = float(np.max(np.asarray(epoch_interval.end, dtype=float)))
    ripple_start = np.asarray(ripple_table.get("start_time", []), dtype=float)
    ripple_end = np.asarray(ripple_table.get("end_time", []), dtype=float)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    sample_ripple_index: list[np.ndarray] = []
    sample_ripple_orig_index: list[np.ndarray] = []
    sample_bin_index: list[np.ndarray] = []
    sample_time_from_start: list[np.ndarray] = []
    sample_phase: list[np.ndarray] = []
    sample_ripple_n_bins: list[np.ndarray] = []

    ripple_start_used: list[float] = []
    ripple_end_used: list[float] = []
    ripple_end_full_bins: list[float] = []
    ripple_n_bins_full: list[int] = []
    ripple_n_bins_kept: list[int] = []
    ripple_orig_index: list[int] = []

    kept_ripple_idx = 0
    for orig_idx, (r_start, r_end) in enumerate(zip(ripple_start, ripple_end)):
        start_s = max(float(r_start), epoch_start)
        end_s = min(float(r_end), epoch_end)

        if end_s - start_s < float(bin_size_s):
            continue

        X_full, bin_edges = _count_unit_spikes_in_full_bins(
            spikes_by_region[ca1_region],
            ca1_unit_ids,
            start_s,
            end_s,
            bin_size_s,
        )
        y_full, _ = _count_unit_spikes_in_full_bins(
            spikes_by_region[v1_region],
            v1_unit_ids,
            start_s,
            end_s,
            bin_size_s,
        )

        n_bins = X_full.shape[0]
        if n_bins <= max_lag_bins:
            continue

        lagged_parts = [
            X_full[max_lag_bins - lag : n_bins - lag]
            for lag in range(max_lag_bins + 1)
        ]
        X_keep = np.hstack(lagged_parts)
        y_keep = y_full[max_lag_bins:]

        kept_bin_idx = np.arange(max_lag_bins, n_bins, dtype=np.int64)
        denom = max(n_bins - 1, 1)
        phase = kept_bin_idx.astype(np.float64) / float(denom)

        X_list.append(X_keep)
        y_list.append(y_keep)
        sample_ripple_index.append(
            np.full(y_keep.shape[0], kept_ripple_idx, dtype=np.int64)
        )
        sample_ripple_orig_index.append(
            np.full(y_keep.shape[0], orig_idx, dtype=np.int64)
        )
        sample_bin_index.append(kept_bin_idx)
        sample_time_from_start.append(
            kept_bin_idx.astype(np.float64) * float(bin_size_s)
        )
        sample_phase.append(phase)
        sample_ripple_n_bins.append(np.full(y_keep.shape[0], n_bins, dtype=np.int64))

        ripple_start_used.append(start_s)
        ripple_end_used.append(end_s)
        ripple_end_full_bins.append(float(bin_edges[-1]))
        ripple_n_bins_full.append(n_bins)
        ripple_n_bins_kept.append(y_keep.shape[0])
        ripple_orig_index.append(orig_idx)
        kept_ripple_idx += 1

    if not X_list:
        raise ValueError(
            f"No valid ripples remained for epoch={epoch}, bin_size_s={bin_size_s}, "
            f"max_lag_bins={max_lag_bins}"
        )

    dataset = {
        "X": np.vstack(X_list).astype(np.float64),
        "y": np.vstack(y_list).astype(np.float64),
        "ca1_unit_ids": ca1_unit_ids,
        "v1_unit_ids": v1_unit_ids,
        "sample_ripple_index": np.concatenate(sample_ripple_index),
        "sample_ripple_orig_index": np.concatenate(sample_ripple_orig_index),
        "sample_bin_index_within_ripple": np.concatenate(sample_bin_index),
        "sample_time_from_ripple_start_s": np.concatenate(sample_time_from_start),
        "sample_phase_within_ripple": np.concatenate(sample_phase),
        "sample_ripple_n_bins": np.concatenate(sample_ripple_n_bins),
        "ripple_start_s": np.asarray(ripple_start_used, dtype=np.float64),
        "ripple_end_s": np.asarray(ripple_end_used, dtype=np.float64),
        "ripple_end_full_bins_s": np.asarray(ripple_end_full_bins, dtype=np.float64),
        "ripple_n_bins_full": np.asarray(ripple_n_bins_full, dtype=np.int64),
        "ripple_n_bins_kept": np.asarray(ripple_n_bins_kept, dtype=np.int64),
        "ripple_orig_index": np.asarray(ripple_orig_index, dtype=np.int64),
        "n_ripples_total_detected": np.int64(ripple_start.size),
        "n_ripples_used": np.int64(len(ripple_start_used)),
    }

    for name, array in dataset.items():
        if isinstance(array, np.ndarray) and array.dtype.kind in {"f", "i", "u"}:
            info = _first_nonfinite_info(array) if array.dtype.kind == "f" else None
            if info:
                raise ValueError(f"Non-finite in dataset[{name}]: {info}")

    return dataset


def _build_phase_basis(
    phase_train: np.ndarray,
    phase_test: np.ndarray,
    *,
    n_basis_funcs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build spline basis features for within-ripple phase."""
    _jax, nmo = require_glm_modules()
    basis = nmo.basis.BSplineEval(
        n_basis_funcs=int(n_basis_funcs),
        order=3,
        bounds=(0.0, 1.0),
        label="ripple_phase",
    )
    phase_train_clip = np.clip(np.asarray(phase_train, dtype=float), 0.0, 1.0)
    phase_test_clip = np.clip(np.asarray(phase_test, dtype=float), 0.0, 1.0)
    X_train = np.asarray(basis.compute_features(phase_train_clip), dtype=np.float64)
    X_test = np.asarray(basis.compute_features(phase_test_clip), dtype=np.float64)
    return X_train, X_test


def _constant_rate_predictor(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Return the constant-rate null predictor fit on the training response."""
    lam0 = np.mean(np.asarray(y_train, dtype=np.float64), axis=0, keepdims=True)
    lam0 = np.clip(lam0, 1e-12, None)
    return np.repeat(lam0, int(n_test), axis=0)


def _alloc_metric_arrays(
    n_splits: int,
    n_cells: int,
    n_shuffles: int,
) -> dict[str, np.ndarray]:
    """Allocate the fold-wise metric arrays used by this workflow."""
    out = {
        "pseudo_r2": np.full((n_splits, n_cells), np.nan, dtype=np.float32),
        "mae": np.full((n_splits, n_cells), np.nan, dtype=np.float32),
        "ll": np.full((n_splits, n_cells), np.nan, dtype=np.float32),
        "devexp": np.full((n_splits, n_cells), np.nan, dtype=np.float32),
        "bps": np.full((n_splits, n_cells), np.nan, dtype=np.float32),
    }
    out["pseudo_r2_shuff"] = np.full(
        (n_splits, n_shuffles, n_cells), np.nan, dtype=np.float32
    )
    out["mae_shuff"] = np.full(
        (n_splits, n_shuffles, n_cells), np.nan, dtype=np.float32
    )
    out["ll_shuff"] = np.full(
        (n_splits, n_shuffles, n_cells), np.nan, dtype=np.float32
    )
    out["devexp_shuff"] = np.full(
        (n_splits, n_shuffles, n_cells), np.nan, dtype=np.float32
    )
    out["bps_shuff"] = np.full(
        (n_splits, n_shuffles, n_cells), np.nan, dtype=np.float32
    )
    return out


def fit_ripple_glm_binned(
    epoch: str,
    *,
    animal_name: str,
    date: str,
    spikes_by_region: dict[str, "nap.TsGroup"],
    ripple_table: pd.DataFrame,
    epoch_interval: "nap.IntervalSet",
    regions: tuple[str, str] = (TARGET_REGION, SOURCE_REGION),
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_bins: int = DEFAULT_MAX_LAG_BINS,
    min_total_spikes: int = DEFAULT_MIN_TOTAL_SPIKES,
    n_shuffles_ripple: int = DEFAULT_N_SHUFFLES_RIPPLE,
    n_splits: int = DEFAULT_N_SPLITS,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
    store_shuffle_preds: bool = True,
    baseline_n_basis: int = DEFAULT_BASELINE_N_BASIS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, Any]:
    """Fit one ripple-binned GLM for one epoch."""
    jax, _nmo = require_glm_modules()
    rng = np.random.default_rng(shuffle_seed)

    dataset = build_binned_ripple_dataset(
        epoch=epoch,
        spikes_by_region=spikes_by_region,
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        regions=regions,
        bin_size_s=bin_size_s,
        max_lag_bins=max_lag_bins,
    )

    X = np.asarray(dataset["X"], dtype=np.float64)
    y = np.asarray(dataset["y"], dtype=np.float64)
    phase = np.asarray(dataset["sample_phase_within_ripple"], dtype=np.float64)
    sample_ripple_index = np.asarray(dataset["sample_ripple_index"], dtype=np.int64)

    n_ripples = int(dataset["n_ripples_used"])
    if n_ripples < n_splits:
        raise ValueError(f"Not enough ripples: n_ripples={n_ripples}, n_splits={n_splits}")

    keep_y = y.sum(axis=0) >= int(min_total_spikes)
    y = y[:, keep_y]
    v1_unit_ids = np.asarray(dataset["v1_unit_ids"])
    v1_unit_ids_kept = v1_unit_ids[keep_y]
    ca1_unit_ids = np.asarray(dataset["ca1_unit_ids"])

    n_samples = X.shape[0]
    n_cells = y.shape[1]
    if n_cells == 0:
        raise ValueError(
            f"No V1 units passed min_total_spikes={min_total_spikes} for epoch={epoch}"
        )

    metric_real = _alloc_metric_arrays(n_splits=n_splits, n_cells=n_cells, n_shuffles=0)
    metric_shuff = _alloc_metric_arrays(
        n_splits=n_splits,
        n_cells=n_cells,
        n_shuffles=n_shuffles_ripple,
    )
    metric_baseline = _alloc_metric_arrays(
        n_splits=n_splits,
        n_cells=n_cells,
        n_shuffles=0,
    )

    y_test_full = np.full((n_samples, n_cells), np.nan, dtype=np.float32)
    yhat_full = np.full((n_samples, n_cells), np.nan, dtype=np.float32)
    yhat_baseline = np.full((n_samples, n_cells), np.nan, dtype=np.float32)

    yhat_shuff: np.ndarray | None = None
    if store_shuffle_preds and n_shuffles_ripple > 0:
        yhat_shuff = np.full(
            (n_shuffles_ripple, n_samples, n_cells),
            np.nan,
            dtype=np.float32,
        )

    splitter = KFold(n_splits=n_splits, shuffle=False, random_state=None)
    ripple_ids = np.arange(n_ripples, dtype=np.int64)
    fold_info: list[dict[str, Any]] = []

    for fold, (train_ripple_idx, test_ripple_idx) in enumerate(splitter.split(ripple_ids)):
        train_mask = np.isin(sample_ripple_index, train_ripple_idx)
        test_mask = np.isin(sample_ripple_index, test_ripple_idx)

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        phase_train = phase[train_mask]
        phase_test = phase[test_mask]

        X_train_pp, X_test_pp, keep_x, x_mean, x_std = _preprocess_X_fit_apply(
            X_train=X_train,
            X_test=X_test,
        )
        if X_train_pp.shape[1] == 0:
            raise ValueError(
                f"Fold {fold}: all CA1 features were near-constant after filtering."
            )

        glm = _make_glm(
            ridge_strength=ridge_strength,
            maxiter=maxiter,
            tol=tol,
        )
        glm.fit(X_train_pp, y_train)
        lam = np.asarray(glm.predict(X_test_pp), dtype=np.float64)

        y_test_full[test_mask] = y_test.astype(np.float32)
        yhat_full[test_mask] = lam.astype(np.float32)

        metric_real["pseudo_r2"][fold] = mcfadden_pseudo_r2_per_neuron(
            y_test,
            lam,
            y_train,
        ).astype(np.float32)
        metric_real["mae"][fold] = mae_per_neuron(y_test, lam).astype(np.float32)
        metric_real["ll"][fold] = poisson_ll_per_neuron(y_test, lam).astype(np.float32)
        metric_real["devexp"][fold] = deviance_explained_per_neuron(
            y_test=y_test,
            lam_test=lam,
            y_null_fit=y_train,
        ).astype(np.float32)
        metric_real["bps"][fold] = bits_per_spike_per_neuron(
            y_test=y_test,
            lam_test=lam,
            y_null_fit=y_train,
        ).astype(np.float32)

        for shuffle_index in range(n_shuffles_ripple):
            y_train_sh = shuffle_time_per_neuron(y_train, rng)

            glm_s = _make_glm(
                ridge_strength=ridge_strength,
                maxiter=maxiter,
                tol=tol,
            )
            glm_s.fit(X_train_pp, y_train_sh)
            lam_s = np.asarray(glm_s.predict(X_test_pp), dtype=np.float64)

            if yhat_shuff is not None:
                yhat_shuff[shuffle_index, test_mask] = lam_s.astype(np.float32)

            metric_shuff["pseudo_r2_shuff"][fold, shuffle_index] = (
                mcfadden_pseudo_r2_per_neuron(y_test, lam_s, y_train).astype(np.float32)
            )
            metric_shuff["mae_shuff"][fold, shuffle_index] = mae_per_neuron(
                y_test,
                lam_s,
            ).astype(np.float32)
            metric_shuff["ll_shuff"][fold, shuffle_index] = poisson_ll_per_neuron(
                y_test,
                lam_s,
            ).astype(np.float32)
            metric_shuff["devexp_shuff"][fold, shuffle_index] = (
                deviance_explained_per_neuron(
                    y_test=y_test,
                    lam_test=lam_s,
                    y_null_fit=y_train,
                ).astype(np.float32)
            )
            metric_shuff["bps_shuff"][fold, shuffle_index] = bits_per_spike_per_neuron(
                y_test=y_test,
                lam_test=lam_s,
                y_null_fit=y_train,
            ).astype(np.float32)

            del glm_s, y_train_sh, lam_s
            if (shuffle_index + 1) % 5 == 0:
                jax.clear_caches()
                gc.collect()

        phase_train_basis, phase_test_basis = _build_phase_basis(
            phase_train,
            phase_test,
            n_basis_funcs=baseline_n_basis,
        )
        (
            phase_train_pp,
            phase_test_pp,
            keep_x_phase,
            phase_mean,
            phase_std,
        ) = _preprocess_X_fit_apply(phase_train_basis, phase_test_basis)

        if phase_train_pp.shape[1] == 0:
            lam_baseline = _constant_rate_predictor(y_train, y_test.shape[0])
        else:
            glm_base = _make_glm(
                ridge_strength=ridge_strength,
                maxiter=maxiter,
                tol=tol,
            )
            glm_base.fit(phase_train_pp, y_train)
            lam_baseline = np.asarray(glm_base.predict(phase_test_pp), dtype=np.float64)
            del glm_base

        yhat_baseline[test_mask] = lam_baseline.astype(np.float32)

        metric_baseline["pseudo_r2"][fold] = mcfadden_pseudo_r2_per_neuron(
            y_test,
            lam_baseline,
            y_train,
        ).astype(np.float32)
        metric_baseline["mae"][fold] = mae_per_neuron(
            y_test,
            lam_baseline,
        ).astype(np.float32)
        metric_baseline["ll"][fold] = poisson_ll_per_neuron(
            y_test,
            lam_baseline,
        ).astype(np.float32)
        metric_baseline["devexp"][fold] = deviance_explained_per_neuron(
            y_test=y_test,
            lam_test=lam_baseline,
            y_null_fit=y_train,
        ).astype(np.float32)
        metric_baseline["bps"][fold] = bits_per_spike_per_neuron(
            y_test=y_test,
            lam_test=lam_baseline,
            y_null_fit=y_train,
        ).astype(np.float32)

        fold_info.append(
            {
                "fold": int(fold),
                "train_ripple_index": np.asarray(train_ripple_idx, dtype=np.int64),
                "test_ripple_index": np.asarray(test_ripple_idx, dtype=np.int64),
                "train_sample_index": np.flatnonzero(train_mask),
                "test_sample_index": np.flatnonzero(test_mask),
                "ca1_keep_x": keep_x,
                "ca1_x_mean": x_mean,
                "ca1_x_std": x_std,
                "baseline_keep_x": keep_x_phase,
                "baseline_x_mean": phase_mean,
                "baseline_x_std": phase_std,
            }
        )

        del glm, lam, lam_baseline
        jax.clear_caches()
        gc.collect()

    for name, array in [
        ("y_test_full", y_test_full),
        ("yhat_full", yhat_full),
        ("yhat_baseline", yhat_baseline),
    ]:
        if np.isnan(array).any():
            raise ValueError(f"{name} has NaNs: some samples were never assigned to a fold")

    if yhat_shuff is not None and np.isnan(yhat_shuff).any():
        raise ValueError("yhat_shuff has NaNs: some samples were never assigned to a fold")

    return {
        "epoch": epoch,
        "animal_name": animal_name,
        "date": date,
        "bin_size_s": float(bin_size_s),
        "max_lag_bins": int(max_lag_bins),
        "min_total_spikes": int(min_total_spikes),
        "n_shuffles_ripple": int(n_shuffles_ripple),
        "n_splits": int(n_splits),
        "ridge_strength": float(ridge_strength),
        "maxiter": int(maxiter),
        "tol": float(tol),
        "baseline_n_basis": int(baseline_n_basis),
        "random_seed": int(shuffle_seed),
        "n_samples": int(n_samples),
        "n_cells": int(n_cells),
        "n_ripples_total_detected": int(dataset["n_ripples_total_detected"]),
        "n_ripples_used": int(dataset["n_ripples_used"]),
        "ca1_unit_ids": ca1_unit_ids,
        "v1_unit_ids": dataset["v1_unit_ids"],
        "v1_unit_ids_kept": v1_unit_ids_kept,
        "y_ripple_test": y_test_full,
        "yhat_ripple": yhat_full,
        "yhat_ripple_shuff": yhat_shuff,
        "yhat_baseline": yhat_baseline,
        "pseudo_r2_ripple_folds": metric_real["pseudo_r2"],
        "mae_ripple_folds": metric_real["mae"],
        "ll_ripple_folds": metric_real["ll"],
        "devexp_ripple_folds": metric_real["devexp"],
        "bits_per_spike_ripple_folds": metric_real["bps"],
        "pseudo_r2_ripple_shuff_folds": metric_shuff["pseudo_r2_shuff"],
        "mae_ripple_shuff_folds": metric_shuff["mae_shuff"],
        "ll_ripple_shuff_folds": metric_shuff["ll_shuff"],
        "devexp_ripple_shuff_folds": metric_shuff["devexp_shuff"],
        "bits_per_spike_ripple_shuff_folds": metric_shuff["bps_shuff"],
        "pseudo_r2_baseline_folds": metric_baseline["pseudo_r2"],
        "mae_baseline_folds": metric_baseline["mae"],
        "ll_baseline_folds": metric_baseline["ll"],
        "devexp_baseline_folds": metric_baseline["devexp"],
        "bits_per_spike_baseline_folds": metric_baseline["bps"],
        "sample_ripple_index": dataset["sample_ripple_index"],
        "sample_ripple_orig_index": dataset["sample_ripple_orig_index"],
        "sample_bin_index_within_ripple": dataset["sample_bin_index_within_ripple"],
        "sample_time_from_ripple_start_s": dataset["sample_time_from_ripple_start_s"],
        "sample_phase_within_ripple": dataset["sample_phase_within_ripple"],
        "sample_ripple_n_bins": dataset["sample_ripple_n_bins"],
        "ripple_start_s": dataset["ripple_start_s"],
        "ripple_end_s": dataset["ripple_end_s"],
        "ripple_end_full_bins_s": dataset["ripple_end_full_bins_s"],
        "ripple_n_bins_full": dataset["ripple_n_bins_full"],
        "ripple_n_bins_kept": dataset["ripple_n_bins_kept"],
        "ripple_orig_index": dataset["ripple_orig_index"],
        "fold_info": fold_info,
    }


def _plot_metric_vs_shuffle(
    metric_real_folds: np.ndarray,
    metric_shuff_folds: np.ndarray,
    *,
    metric_name: str,
    higher_is_better: bool,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
) -> None:
    """Plot the real-vs-shuffle summary for one metric."""
    import matplotlib.pyplot as plt

    sig_thresh = -np.log(0.05)
    real = np.nanmean(np.asarray(metric_real_folds, dtype=float), axis=0)
    null = np.nanmean(np.asarray(metric_shuff_folds, dtype=float), axis=0)
    null_mean = np.nanmean(null, axis=0)
    effect = real - null_mean if higher_is_better else null_mean - real
    p_emp = empirical_p_value_against_null(
        real=real,
        null_draws=null,
        higher_is_better=higher_is_better,
    )
    neg_log_p = -np.log(np.clip(p_emp, 1e-12, None))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].hist(real[np.isfinite(real)], bins=40, alpha=0.8, label="real")
    axes[0].hist(null.ravel()[np.isfinite(null.ravel())], bins=40, alpha=0.5, label="shuffle")
    axes[0].set_title(f"{metric_name}: real vs shuffle")
    axes[0].legend(frameon=False)

    valid = np.isfinite(real) & np.isfinite(null_mean)
    axes[1].scatter(null_mean[valid], real[valid], s=10, alpha=0.8)
    if np.any(valid):
        lo = float(np.nanmin(np.r_[null_mean[valid], real[valid]]))
        hi = float(np.nanmax(np.r_[null_mean[valid], real[valid]]))
        axes[1].plot([lo, hi], [lo, hi], color="k", lw=1, ls="--")
    axes[1].set_xlabel("shuffle mean")
    axes[1].set_ylabel("real")
    axes[1].set_title("Per-cell mean")

    axes[2].hist(effect[np.isfinite(effect)], bins=40, alpha=0.85)
    axes[2].set_title("Effect vs shuffle")
    axes[2].set_xlabel(
        f"{metric_name} improvement"
        if higher_is_better
        else f"{metric_name} reduction"
    )

    valid_sig = np.isfinite(real) & np.isfinite(neg_log_p)
    axes[3].scatter(real[valid_sig], neg_log_p[valid_sig], s=10, alpha=0.8)
    axes[3].axhline(sig_thresh, color="k", lw=1, ls="--")
    axes[3].set_xlabel(metric_name)
    axes[3].set_ylabel("-log(p)")
    axes[3].set_title("Metric vs significance")

    fig.suptitle(f"{animal_name} {date} {epoch}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_model_comparison(
    metric_a_folds: np.ndarray,
    metric_b_folds: np.ndarray,
    *,
    label_a: str,
    label_b: str,
    metric_name: str,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    p_values: np.ndarray | None = None,
    significance_label: str = "-log(p)",
) -> None:
    """Plot the CA1-model versus baseline comparison for one metric."""
    import matplotlib.pyplot as plt

    sig_thresh = -np.log(0.05)
    a = np.nanmean(np.asarray(metric_a_folds, dtype=float), axis=0)
    b = np.nanmean(np.asarray(metric_b_folds, dtype=float), axis=0)
    diff = a - b

    if p_values is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].hist(a[np.isfinite(a)], bins=40, alpha=0.75, label=label_a)
    axes[0].hist(b[np.isfinite(b)], bins=40, alpha=0.55, label=label_b)
    axes[0].legend(frameon=False)
    axes[0].set_title(metric_name)

    valid = np.isfinite(a) & np.isfinite(b)
    axes[1].scatter(b[valid], a[valid], s=10, alpha=0.8)
    if np.any(valid):
        lo = float(np.nanmin(np.r_[a[valid], b[valid]]))
        hi = float(np.nanmax(np.r_[a[valid], b[valid]]))
        axes[1].plot([lo, hi], [lo, hi], color="k", lw=1, ls="--")
    axes[1].set_xlabel(label_b)
    axes[1].set_ylabel(label_a)
    axes[1].set_title("Per-cell mean")

    axes[2].hist(diff[np.isfinite(diff)], bins=40, alpha=0.85)
    axes[2].set_title(f"{label_a} - {label_b}")
    axes[2].set_xlabel(metric_name)

    if p_values is not None:
        neg_log_p = -np.log(np.clip(np.asarray(p_values, dtype=float), 1e-12, None))
        valid_sig = np.isfinite(diff) & np.isfinite(neg_log_p)
        axes[3].scatter(diff[valid_sig], neg_log_p[valid_sig], s=10, alpha=0.8)
        axes[3].axhline(sig_thresh, color="k", lw=1, ls="--")
        axes[3].set_xlabel(f"{label_a} - {label_b}")
        axes[3].set_ylabel(significance_label)
        axes[3].set_title("Difference vs significance")

    fig.suptitle(f"{animal_name} {date} {epoch}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def empirical_p_value_against_null(
    real: np.ndarray,
    null_draws: np.ndarray,
    *,
    higher_is_better: bool,
) -> np.ndarray:
    """Return an empirical one-sided p-value against a null draw matrix."""
    real = np.asarray(real, dtype=float)
    null_draws = np.asarray(null_draws, dtype=float)

    if null_draws.ndim != 2:
        raise ValueError("null_draws must have shape (n_draws, n_cells)")

    finite_null = np.isfinite(null_draws)
    n_draws = np.sum(finite_null, axis=0)

    if higher_is_better:
        extreme = np.sum(finite_null & (null_draws >= real[None, :]), axis=0)
    else:
        extreme = np.sum(finite_null & (null_draws <= real[None, :]), axis=0)

    p = (extreme + 1.0) / np.clip(n_draws + 1.0, 1.0, None)
    p = np.where(n_draws > 0, p, np.nan)
    return p


def format_float_token(value: float) -> str:
    """Return one float token safe for filenames."""
    token = np.format_float_positional(float(value), trim="-")
    return token.replace(".", "p")


def build_output_stem(
    *,
    epoch: str,
    bin_size_s: float,
    max_lag_bins: int,
    min_total_spikes: int,
    ridge_strength: float,
) -> str:
    """Return the shared output stem for one epoch/config pair."""
    return (
        f"{epoch}_ripple_binned_bin_size_{format_float_token(bin_size_s)}"
        f"_max_lag_bins_{max_lag_bins}"
        f"_min_total_spikes_{min_total_spikes}"
        f"_ridge_strength_{format_float_token(ridge_strength)}"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the ripple-binned GLM workflow."""
    cuda_visible_devices = _parse_cuda_visible_devices(argv)
    args = parse_arguments(argv)
    validate_arguments(args)
    configure_jax_environment(cuda_visible_devices)

    session = prepare_binned_ripple_glm_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    selected_epochs = validate_selected_epochs(session["epoch_tags"], args.epochs)
    analysis_path = session["analysis_path"]
    out_dir = analysis_path / Path(__file__).stem
    fig_dir = analysis_path / "figs" / Path(__file__).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fit_parameters = {
        "animal_name": args.animal_name,
        "date": args.date,
        "epochs": selected_epochs,
        "bin_size_s": args.bin_size_s,
        "max_lag_bins": args.max_lag_bins,
        "cuda_visible_devices": args.cuda_visible_devices,
        "min_total_spikes": args.min_total_spikes,
        "n_splits": args.n_splits,
        "n_shuffles_ripple": args.n_shuffles_ripple,
        "ridge_strength": args.ridge_strength,
        "maxiter": args.maxiter,
        "tol": args.tol,
        "baseline_n_basis": args.baseline_n_basis,
        "shuffle_seed": args.shuffle_seed,
        "overwrite": bool(args.overwrite),
    }

    saved_result_paths: list[Path] = []
    saved_figure_paths: list[Path] = []
    successful_jobs: list[dict[str, Any]] = []
    skipped_epochs: list[dict[str, Any]] = []

    for epoch in selected_epochs:
        stem = build_output_stem(
            epoch=epoch,
            bin_size_s=args.bin_size_s,
            max_lag_bins=args.max_lag_bins,
            min_total_spikes=args.min_total_spikes,
            ridge_strength=args.ridge_strength,
        )
        out_path = out_dir / f"{stem}.npz"
        had_existing_results = out_path.exists()

        if had_existing_results and not args.overwrite:
            print(f"Loading existing results for {epoch}: {out_path}")
            results = load_results_npz(out_path)
        else:
            ripple_table = session["ripple_tables"].get(epoch, empty_ripple_table())
            if ripple_table.empty:
                skipped_epochs.append(
                    {
                        "epoch": epoch,
                        "reason": "No ripple events were found for this epoch.",
                    }
                )
                print(f"Skipping {epoch}: no ripple events")
                continue

            try:
                results = fit_ripple_glm_binned(
                    epoch=epoch,
                    animal_name=args.animal_name,
                    date=args.date,
                    spikes_by_region=session["spikes_by_region"],
                    ripple_table=ripple_table,
                    epoch_interval=session["epoch_intervals"][epoch],
                    bin_size_s=args.bin_size_s,
                    max_lag_bins=args.max_lag_bins,
                    min_total_spikes=args.min_total_spikes,
                    n_shuffles_ripple=args.n_shuffles_ripple,
                    n_splits=args.n_splits,
                    ridge_strength=args.ridge_strength,
                    maxiter=args.maxiter,
                    tol=args.tol,
                    baseline_n_basis=args.baseline_n_basis,
                    shuffle_seed=args.shuffle_seed,
                    store_shuffle_preds=True,
                )
            except Exception as exc:
                skipped_epochs.append(
                    {
                        "epoch": epoch,
                        "reason": "Failed to fit ripple-binned GLM.",
                        "error": str(exc),
                    }
                )
                print(f"Skipping {epoch}: ripple-binned GLM failed: {exc}")
                continue

            save_results_npz(out_path, results)
            if had_existing_results:
                print(f"Overwrote results for {epoch}: {out_path}")
            else:
                print(f"Saved results for {epoch}: {out_path}")

        figure_paths = [
            fig_dir / f"{stem}_bits_per_spike_vs_shuffle.png",
            fig_dir / f"{stem}_pseudo_r2_vs_shuffle.png",
            fig_dir / f"{stem}_devexp_vs_shuffle.png",
            fig_dir / f"{stem}_mae_vs_shuffle.png",
            fig_dir / f"{stem}_ca1_vs_baseline_bits_per_spike.png",
            fig_dir / f"{stem}_ca1_vs_baseline_devexp.png",
        ]

        _plot_metric_vs_shuffle(
            results["bits_per_spike_ripple_folds"],
            results["bits_per_spike_ripple_shuff_folds"],
            metric_name="bits/spike",
            higher_is_better=True,
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
            out_path=figure_paths[0],
        )
        _plot_metric_vs_shuffle(
            results["pseudo_r2_ripple_folds"],
            results["pseudo_r2_ripple_shuff_folds"],
            metric_name="pseudo_r2",
            higher_is_better=True,
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
            out_path=figure_paths[1],
        )
        _plot_metric_vs_shuffle(
            results["devexp_ripple_folds"],
            results["devexp_ripple_shuff_folds"],
            metric_name="deviance explained",
            higher_is_better=True,
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
            out_path=figure_paths[2],
        )
        _plot_metric_vs_shuffle(
            results["mae_ripple_folds"],
            results["mae_ripple_shuff_folds"],
            metric_name="mae",
            higher_is_better=False,
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
            out_path=figure_paths[3],
        )
        _plot_model_comparison(
            results["bits_per_spike_ripple_folds"],
            results["bits_per_spike_baseline_folds"],
            label_a="CA1 model",
            label_b="time baseline",
            metric_name="bits/spike",
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
            out_path=figure_paths[4],
            p_values=empirical_p_value_against_null(
                real=np.nanmean(results["bits_per_spike_ripple_folds"], axis=0),
                null_draws=np.nanmean(results["bits_per_spike_ripple_shuff_folds"], axis=0),
                higher_is_better=True,
            ),
            significance_label="-log(p)",
        )
        _plot_model_comparison(
            results["devexp_ripple_folds"],
            results["devexp_baseline_folds"],
            label_a="CA1 model",
            label_b="time baseline",
            metric_name="deviance explained",
            animal_name=args.animal_name,
            date=args.date,
            epoch=epoch,
            out_path=figure_paths[5],
            p_values=empirical_p_value_against_null(
                real=np.nanmean(results["devexp_ripple_folds"], axis=0),
                null_draws=np.nanmean(results["devexp_ripple_shuff_folds"], axis=0),
                higher_is_better=True,
            ),
            significance_label="-log(p)",
        )

        saved_result_paths.append(out_path)
        saved_figure_paths.extend(figure_paths)
        successful_jobs.append(
            {
                "epoch": epoch,
                "result_path": out_path,
                "figure_paths": figure_paths,
                "n_ripples_total_detected": int(results["n_ripples_total_detected"]),
                "n_ripples_used": int(results["n_ripples_used"]),
                "n_cells": int(results["n_cells"]),
            }
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name=Path(__file__).stem,
        parameters=fit_parameters,
        outputs={
            "sources": session["sources"],
            "saved_result_paths": saved_result_paths,
            "saved_figure_paths": saved_figure_paths,
            "successful_jobs": successful_jobs,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")

    if not successful_jobs:
        raise RuntimeError(
            "All requested epochs were skipped. "
            f"Epoch reasons: {skipped_epochs!r}"
        )


if __name__ == "__main__":
    main()
