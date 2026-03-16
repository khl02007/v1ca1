from __future__ import annotations

"""Fit ripple-time V1 GLMs driven by CA1 state decoding.

This CLI loads one session through the shared task-progression session helpers,
builds CA1 tuning curves from each decoded run epoch, builds V1 tuning curves
from one selected run epoch, decodes ripple-time place or task progression
from CA1, converts the decoded state into expected V1 firing-rate covariates,
and then fits a diagonal population GLM on ripple bins to predict V1 spiking.

CA1 tuning and decoding always use the same run epoch. V1 tuning can come from
another run epoch, which defaults to `08_r4`. Per-unit summaries include
pseudo-R^2, deviance explained, bits/spike, and empirical shuffle p-values.
Decoded CA1 trajectories are saved as pynapple-backed `.npz` files, per-unit
metric summaries are saved as parquet tables, raw GLM fit outputs plus aligned
ripple-bin model inputs are saved together in one NetCDF-backed xarray dataset
per epoch, and metric figures are written to the session figure directory. A
JSON run log is written under `v1ca1_log/`.
"""

import argparse
import gc
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
    load_ephys_timestamps_by_epoch,
)
from v1ca1.ripple._decoding import (
    REPRESENTATIONS,
    compute_tuning_curves_for_epoch,
    concatenate_tsds,
    empty_ripple_table,
    extract_time_values,
    get_representation_inputs,
    interpolate_nans_1d,
    make_empty_tsd,
    make_intervalset_from_bounds,
    select_decode_epochs,
    deranged_permutation,
)
from v1ca1.ripple.ripple_glm import (
    bits_per_spike_per_neuron,
    build_epoch_intervals,
    deviance_explained_per_neuron,
    empirical_p_values,
    load_ripple_tables,
    mcfadden_pseudo_r2_per_neuron,
    nansem,
)
from v1ca1.task_progression._session import (
    compute_movement_firing_rates,
    get_analysis_path,
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    import pynapple as nap

METRIC_LABELS = {
    "pseudo_r2": "Pseudo R^2",
    "devexp": "Deviance Explained",
    "bits_per_spike": "Bits/Spike",
}

DEFAULT_V1_TUNING_EPOCH = "08_r4"
DEFAULT_BIN_SIZE_S = 0.002
DEFAULT_RIDGE_STRENGTH = 1e-3
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES = 100
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_V1_MIN_MOVEMENT_FR_HZ = 0.5
DEFAULT_V1_MIN_RIPPLE_FR_HZ = 0.1


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the ripple decoding GLM."""
    parser = argparse.ArgumentParser(
        description="Fit a V1 ripple GLM driven by CA1 ripple decoding"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--representation",
        required=True,
        choices=REPRESENTATIONS,
        help="Decoded state representation: place or task progression",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--decode-epochs",
        nargs="+",
        help="Optional run epochs to decode. Default: all run epochs.",
    )
    parser.add_argument(
        "--v1-tuning-epoch",
        default=DEFAULT_V1_TUNING_EPOCH,
        help=(
            "Run epoch used to build V1 tuning curves for the GLM covariates. "
            f"Default: {DEFAULT_V1_TUNING_EPOCH}"
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Time bin size in seconds for ripple decoding and GLM fitting. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Ridge regularization strength. Default: {DEFAULT_RIDGE_STRENGTH}",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of ripple cross-validation folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help=f"Number of shuffle refits per fold. Default: {DEFAULT_N_SHUFFLES}",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for response shuffles. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    parser.add_argument(
        "--v1-min-movement-fr-hz",
        type=float,
        default=DEFAULT_V1_MIN_MOVEMENT_FR_HZ,
        help=(
            "Minimum V1 firing rate during movement in the V1 tuning epoch. "
            f"Default: {DEFAULT_V1_MIN_MOVEMENT_FR_HZ}"
        ),
    )
    parser.add_argument(
        "--v1-min-ripple-fr-hz",
        type=float,
        default=DEFAULT_V1_MIN_RIPPLE_FR_HZ,
        help=(
            "Minimum V1 firing rate during ripple bins in each decoded epoch. "
            f"Default: {DEFAULT_V1_MIN_RIPPLE_FR_HZ}"
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI ranges."""
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.ridge_strength < 0:
        raise ValueError("--ridge-strength must be non-negative.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.n_shuffles < 0:
        raise ValueError("--n-shuffles must be non-negative.")
    if args.v1_min_movement_fr_hz < 0:
        raise ValueError("--v1-min-movement-fr-hz must be non-negative.")
    if args.v1_min_ripple_fr_hz < 0:
        raise ValueError("--v1-min-ripple-fr-hz must be non-negative.")

def validate_v1_tuning_epoch(run_epochs: list[str], v1_tuning_epoch: str) -> str:
    """Return a validated V1 tuning epoch."""
    if v1_tuning_epoch not in run_epochs:
        raise ValueError(
            f"--v1-tuning-epoch must be one of the saved run epochs {run_epochs!r}. "
            f"Got {v1_tuning_epoch!r}."
        )
    return str(v1_tuning_epoch)


def require_xarray():
    """Return `xarray` and fail clearly when NetCDF outputs are requested."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `xarray` to save raw GLM fit outputs and "
            "aligned ripple-bin model inputs as NetCDF."
        ) from exc
    return xr


def _ordered_tuning_curve_array(tuning_curves: Any, unit_ids: np.ndarray) -> np.ndarray:
    """Return tuning curves ordered by the requested unit ids."""
    ordered = tuning_curves.sel(unit=unit_ids.tolist())
    array = np.asarray(ordered.to_numpy(), dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D tuning curves, got shape {array.shape}.")
    return array


def build_expected_rate_matrix(
    *,
    tuning_curves: Any,
    unit_ids: np.ndarray,
    decoded_state: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Interpolate one expected-rate trace per unit from decoded state."""
    state_values = interpolate_nans_1d(decoded_state)
    if not np.any(np.isfinite(state_values)):
        return np.full((decoded_state.size, unit_ids.size), np.nan, dtype=float)

    curve_array = _ordered_tuning_curve_array(tuning_curves, unit_ids)
    state_grid = 0.5 * (
        np.asarray(bin_edges[:-1], dtype=float) + np.asarray(bin_edges[1:], dtype=float)
    )
    clipped_state = np.clip(state_values, float(state_grid[0]), float(state_grid[-1]))

    expected_rate = np.zeros((state_values.size, unit_ids.size), dtype=float)
    for unit_index in range(unit_ids.size):
        curve = interpolate_nans_1d(curve_array[unit_index])
        if not np.any(np.isfinite(curve)):
            continue
        curve = np.clip(np.nan_to_num(curve, nan=0.0), 0.0, None)
        if np.all(curve <= 0.0):
            continue
        expected_rate[:, unit_index] = np.interp(
            clipped_state,
            state_grid,
            curve,
            left=float(curve[0]),
            right=float(curve[-1]),
        )
    return expected_rate


def validate_count_columns(counts: Any, unit_ids: np.ndarray, *, label: str) -> None:
    """Ensure binned spike counts preserve the expected unit order."""
    count_columns = np.asarray(counts.columns)
    if count_columns.shape != unit_ids.shape or not np.array_equal(count_columns, unit_ids):
        raise ValueError(
            f"{label} spike count columns do not match the expected unit order. "
            f"Expected {unit_ids!r}, got {count_columns!r}."
        )


def assemble_ripple_epoch_data(
    *,
    ca1_spikes: Any,
    v1_spikes: Any,
    ca1_tuning_curves: Any,
    ripple_table: pd.DataFrame,
    epoch_interval: Any,
    bin_size_s: float,
    v1_unit_ids: np.ndarray,
) -> dict[str, Any]:
    """Decode CA1 and bin V1 activity ripple by ripple for one epoch."""
    import pynapple as nap

    decoded_chunks: list[Any] = []
    ripple_starts: list[float] = []
    ripple_ends: list[float] = []
    ripple_source_indices: list[int] = []
    response_chunks: list[np.ndarray] = []
    decoded_state_chunks: list[np.ndarray] = []
    bin_time_chunks: list[np.ndarray] = []
    ripple_id_chunks: list[np.ndarray] = []
    skipped_ripples: list[dict[str, Any]] = []
    kept_ripple_count = 0

    for ripple_row_index, ripple_row in ripple_table.reset_index(drop=True).iterrows():
        ripple_ep = make_intervalset_from_bounds(
            np.array([float(ripple_row["start_time"])], dtype=float),
            np.array([float(ripple_row["end_time"])], dtype=float),
        ).intersect(epoch_interval)
        if float(ripple_ep.tot_length()) <= 0.0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "No overlap with decode epoch interval.",
                }
            )
            continue

        counts = v1_spikes.count(float(bin_size_s), ep=ripple_ep)
        validate_count_columns(counts, v1_unit_ids, label="V1")
        response = np.asarray(counts.d, dtype=float)
        if response.ndim == 1:
            response = response[:, np.newaxis]
        count_times = extract_time_values(counts).reshape(-1)
        if count_times.size != response.shape[0]:
            raise ValueError(
                "Ripple-bin timestamps do not match the number of V1 spike-count rows."
            )
        if response.shape[0] == 0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "Ripple duration produced zero decode bins.",
                }
            )
            continue

        decoded, _ = nap.decode_bayes(
            tuning_curves=ca1_tuning_curves,
            data=ca1_spikes,
            epochs=ripple_ep,
            sliding_window_size=None,
            bin_size=float(bin_size_s),
        )
        if len(np.asarray(decoded.t, dtype=float)) == 0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "CA1 decoding returned no time bins.",
                }
            )
            continue

        decoded_state = decoded.interpolate(
            counts,
            ep=ripple_ep,
            left=np.nan,
            right=np.nan,
        ).to_numpy().reshape(-1)
        decoded_state = interpolate_nans_1d(decoded_state)
        if not np.any(np.isfinite(decoded_state)):
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "Decoded state was non-finite for all ripple bins.",
                }
            )
            continue

        decoded_chunks.append(decoded)
        response_chunks.append(response)
        decoded_state_chunks.append(decoded_state)
        bin_time_chunks.append(count_times)
        ripple_id_chunks.append(
            np.full(response.shape[0], kept_ripple_count, dtype=int)
        )
        ripple_starts.append(float(np.asarray(ripple_ep.start, dtype=float)[0]))
        ripple_ends.append(float(np.asarray(ripple_ep.end, dtype=float)[0]))
        ripple_source_indices.append(int(ripple_row_index))
        kept_ripple_count += 1

    ripple_support = make_intervalset_from_bounds(
        np.asarray(ripple_starts, dtype=float),
        np.asarray(ripple_ends, dtype=float),
    )
    decoded_tsd = concatenate_tsds(decoded_chunks, ripple_support)
    if not response_chunks:
        return {
            "decoded_tsd": decoded_tsd,
            "response": np.zeros((0, v1_unit_ids.size), dtype=float),
            "decoded_state": np.array([], dtype=float),
            "bin_times_s": np.array([], dtype=float),
            "ripple_ids": np.array([], dtype=int),
            "n_ripples_kept": 0,
            "n_bins": 0,
            "ripple_start_times_s": np.array([], dtype=float),
            "ripple_end_times_s": np.array([], dtype=float),
            "ripple_source_indices": np.array([], dtype=int),
            "skipped_ripples": skipped_ripples,
        }

    response_matrix = np.concatenate(response_chunks, axis=0)
    decoded_state = np.concatenate(decoded_state_chunks).astype(float, copy=False)
    bin_times_s = np.concatenate(bin_time_chunks).astype(float, copy=False)
    ripple_ids = np.concatenate(ripple_id_chunks).astype(int, copy=False)
    return {
        "decoded_tsd": decoded_tsd,
        "response": response_matrix,
        "decoded_state": decoded_state,
        "bin_times_s": bin_times_s,
        "ripple_ids": ripple_ids,
        "n_ripples_kept": int(kept_ripple_count),
        "n_bins": int(response_matrix.shape[0]),
        "ripple_start_times_s": np.asarray(ripple_starts, dtype=float),
        "ripple_end_times_s": np.asarray(ripple_ends, dtype=float),
        "ripple_source_indices": np.asarray(ripple_source_indices, dtype=int),
        "skipped_ripples": skipped_ripples,
    }


def build_v1_unit_mask_table(
    *,
    unit_ids: np.ndarray,
    movement_firing_rates_hz: np.ndarray,
    ripple_counts: np.ndarray,
    expected_rate_hz: np.ndarray,
    bin_size_s: float,
    min_movement_fr_hz: float,
    min_ripple_fr_hz: float,
) -> pd.DataFrame:
    """Return one auditable V1 unit mask table for one decode epoch."""
    if ripple_counts.ndim != 2:
        raise ValueError(f"Expected ripple_counts to be 2D, got shape {ripple_counts.shape}.")
    if expected_rate_hz.ndim != 2:
        raise ValueError(
            f"Expected expected_rate_hz to be 2D, got shape {expected_rate_hz.shape}."
        )
    if ripple_counts.shape[1] != unit_ids.size:
        raise ValueError("Ripple count columns do not match unit_ids.")
    if expected_rate_hz.shape[1] != unit_ids.size:
        raise ValueError("Expected-rate columns do not match unit_ids.")

    total_duration_s = float(ripple_counts.shape[0]) * float(bin_size_s)
    if total_duration_s > 0.0:
        ripple_firing_rates_hz = np.sum(ripple_counts, axis=0) / total_duration_s
    else:
        ripple_firing_rates_hz = np.zeros(unit_ids.size, dtype=float)

    finite_expected_rate = np.isfinite(expected_rate_hz)
    finite_expected_rate_count = np.sum(finite_expected_rate, axis=0)
    expected_rate_has_positive = np.any(
        finite_expected_rate & (expected_rate_hz > 0.0),
        axis=0,
    )
    expected_rate_sum_hz = np.sum(
        np.where(finite_expected_rate, expected_rate_hz, 0.0),
        axis=0,
    )
    expected_rate_mean_hz = np.divide(
        expected_rate_sum_hz,
        finite_expected_rate_count,
        out=np.full(unit_ids.size, np.nan, dtype=float),
        where=finite_expected_rate_count > 0,
    )
    expected_rate_max_hz = np.max(
        np.where(finite_expected_rate, expected_rate_hz, -np.inf),
        axis=0,
    )
    expected_rate_max_hz[finite_expected_rate_count == 0] = np.nan

    movement_fr_pass = np.asarray(movement_firing_rates_hz, dtype=float) >= float(
        min_movement_fr_hz
    )
    ripple_fr_pass = ripple_firing_rates_hz >= float(min_ripple_fr_hz)
    keep_unit = movement_fr_pass & ripple_fr_pass & expected_rate_has_positive

    return pd.DataFrame(
        {
            "unit_id": np.asarray(unit_ids),
            "movement_firing_rate_hz": np.asarray(movement_firing_rates_hz, dtype=float),
            "ripple_firing_rate_hz": ripple_firing_rates_hz,
            "expected_rate_mean_hz": expected_rate_mean_hz,
            "expected_rate_max_hz": expected_rate_max_hz,
            "passes_movement_firing_rate": movement_fr_pass,
            "passes_ripple_firing_rate": ripple_fr_pass,
            "passes_expected_rate_predictor": expected_rate_has_positive,
            "keep_unit": keep_unit,
        }
    )


def build_ripple_fold_masks(
    ripple_ids: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test bin masks that keep each ripple within one fold."""
    ripple_ids = np.asarray(ripple_ids, dtype=int).reshape(-1)
    unique_ripple_ids = np.unique(ripple_ids)
    if unique_ripple_ids.size < n_splits:
        raise ValueError(
            "Not enough ripples for CV: "
            f"n_ripples={unique_ripple_ids.size}, n_splits={n_splits}"
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    kfold = KFold(n_splits=n_splits, shuffle=False)
    for train_index, test_index in kfold.split(unique_ripple_ids):
        train_ripples = unique_ripple_ids[train_index]
        test_ripples = unique_ripple_ids[test_index]
        train_mask = np.isin(ripple_ids, train_ripples)
        test_mask = np.isin(ripple_ids, test_ripples)
        folds.append((train_mask, test_mask))
    return folds


def shuffle_response_by_ripple_blocks(
    response: np.ndarray,
    ripple_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle whole ripple-response blocks while preserving within-ripple order."""
    response = np.asarray(response, dtype=float)
    ripple_ids = np.asarray(ripple_ids, dtype=int).reshape(-1)
    if response.shape[0] != ripple_ids.size:
        raise ValueError("Response rows must match ripple_ids length.")
    if response.shape[0] == 0:
        return response.copy()

    unique_ripple_ids = np.unique(ripple_ids)
    blocks = [response[ripple_ids == ripple_id] for ripple_id in unique_ripple_ids]
    permutation = deranged_permutation(len(blocks), rng)
    return np.concatenate([blocks[index] for index in permutation], axis=0)


def preprocess_design_matrix(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the design matrix while preserving all feature columns."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    variable = std.ravel() > 1e-6

    X_train_pp = (X_train - mean) / (std + 1e-8)
    X_test_pp = (X_test - mean) / (std + 1e-8)
    if np.any(~variable):
        X_train_pp[:, ~variable] = 0.0
        X_test_pp[:, ~variable] = 0.0

    scale = np.sqrt(max(X_train.shape[1], 1))
    X_train_pp /= scale
    X_test_pp /= scale
    X_train_pp = np.clip(X_train_pp, -10.0, 10.0)
    X_test_pp = np.clip(X_test_pp, -10.0, 10.0)
    return X_train_pp, X_test_pp, variable, mean.ravel(), std.ravel()


def build_diagonal_feature_mask(n_units: int) -> np.ndarray:
    """Return the diagonal feature mask used by the per-unit population GLM."""
    return np.eye(int(n_units), dtype=bool)


def require_population_glm():
    """Return `PopulationGLM` and fail clearly if diagonal masking is unsupported."""
    try:
        import nemos as nmo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `nemos` to fit the ripple decoding GLM, but it is not "
            "installed in the current environment."
        ) from exc

    population_glm = nmo.glm.PopulationGLM
    signature = inspect.signature(population_glm.__init__)
    if "feature_mask" not in signature.parameters:
        raise ModuleNotFoundError(
            "The installed `nemos` does not support `PopulationGLM(feature_mask=...)`, "
            "which this script requires for the diagonal per-unit design."
        )
    return population_glm


def clear_jax_caches() -> None:
    """Best-effort cleanup between large GLM fits."""
    try:
        import jax
    except ModuleNotFoundError:
        return
    jax.clear_caches()


def fit_expected_rate_glm(
    *,
    X: np.ndarray,
    y: np.ndarray,
    ripple_ids: np.ndarray,
    n_splits: int,
    n_shuffles: int,
    shuffle_seed: int,
    ridge_strength: float,
) -> dict[str, np.ndarray]:
    """Fit the diagonal expected-rate ripple GLM with ripple-wise CV."""
    n_bins, n_units = y.shape
    metric_shape = (n_splits, n_units)
    shuffle_shape = (n_splits, n_shuffles, n_units)

    results = {
        "pseudo_r2_folds": np.full(metric_shape, np.nan, dtype=np.float32),
        "devexp_folds": np.full(metric_shape, np.nan, dtype=np.float32),
        "bits_per_spike_folds": np.full(metric_shape, np.nan, dtype=np.float32),
        "pseudo_r2_shuff_folds": np.full(shuffle_shape, np.nan, dtype=np.float32),
        "devexp_shuff_folds": np.full(shuffle_shape, np.nan, dtype=np.float32),
        "bits_per_spike_shuff_folds": np.full(shuffle_shape, np.nan, dtype=np.float32),
    }
    if n_units == 0 or n_bins == 0:
        return results

    population_glm = require_population_glm()
    feature_mask = build_diagonal_feature_mask(n_units)
    rng = np.random.default_rng(shuffle_seed)
    folds = build_ripple_fold_masks(ripple_ids, n_splits=n_splits)

    for fold_index, (train_mask, test_mask) in enumerate(folds):
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        X_train_pp, X_test_pp, _variable, _mean, _std = preprocess_design_matrix(
            X_train=X_train,
            X_test=X_test,
        )
        glm = population_glm(
            regularizer="Ridge",
            regularizer_strength=float(ridge_strength),
            feature_mask=feature_mask,
        )
        glm.fit(X_train_pp, y_train)
        lam_test = np.asarray(glm.predict(X_test_pp), dtype=np.float64)

        results["pseudo_r2_folds"][fold_index] = mcfadden_pseudo_r2_per_neuron(
            y_test,
            lam_test,
            y_train,
        ).astype(np.float32)
        results["devexp_folds"][fold_index] = deviance_explained_per_neuron(
            y_test=y_test,
            lam_test=lam_test,
            y_null_fit=y_train,
        ).astype(np.float32)
        results["bits_per_spike_folds"][fold_index] = bits_per_spike_per_neuron(
            y_test=y_test,
            lam_test=lam_test,
            y_null_fit=y_train,
        ).astype(np.float32)

        train_ripple_ids = ripple_ids[train_mask]
        for shuffle_index in range(n_shuffles):
            shuffled_response = shuffle_response_by_ripple_blocks(
                y_train,
                train_ripple_ids,
                rng,
            )
            glm_shuffle = population_glm(
                regularizer="Ridge",
                regularizer_strength=float(ridge_strength),
                feature_mask=feature_mask,
            )
            glm_shuffle.fit(X_train_pp, shuffled_response)
            lam_shuffle = np.asarray(glm_shuffle.predict(X_test_pp), dtype=np.float64)

            results["pseudo_r2_shuff_folds"][fold_index, shuffle_index] = (
                mcfadden_pseudo_r2_per_neuron(y_test, lam_shuffle, y_train).astype(np.float32)
            )
            results["devexp_shuff_folds"][fold_index, shuffle_index] = (
                deviance_explained_per_neuron(
                    y_test=y_test,
                    lam_test=lam_shuffle,
                    y_null_fit=y_train,
                ).astype(np.float32)
            )
            results["bits_per_spike_shuff_folds"][fold_index, shuffle_index] = (
                bits_per_spike_per_neuron(
                    y_test=y_test,
                    lam_test=lam_shuffle,
                    y_null_fit=y_train,
                ).astype(np.float32)
            )
            del glm_shuffle, shuffled_response, lam_shuffle

        del glm, lam_test
        clear_jax_caches()
        gc.collect()

    return results


def summarize_metric_against_shuffle(
    real_folds: np.ndarray,
    shuffle_folds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize one held-out metric against its shuffle null."""
    real_folds = np.asarray(real_folds, dtype=float)
    shuffle_folds = np.asarray(shuffle_folds, dtype=float)
    real_mean = np.nanmean(real_folds, axis=0)
    real_sem = nansem(real_folds, axis=0)

    if shuffle_folds.shape[1] == 0:
        null_samples = np.empty((0, real_folds.shape[1]), dtype=float)
        shuffle_mean = np.full(real_mean.shape, np.nan, dtype=float)
        shuffle_sd = np.full(real_mean.shape, np.nan, dtype=float)
        p_value = np.full(real_mean.shape, np.nan, dtype=float)
    else:
        null_samples = np.nanmean(shuffle_folds, axis=0)
        shuffle_mean = np.nanmean(null_samples, axis=0)
        shuffle_sd = np.nanstd(null_samples, axis=0, ddof=0)
        p_value = empirical_p_values(
            observed=real_mean,
            null_samples=null_samples,
            higher_is_better=True,
        )

    return {
        "mean": np.asarray(real_mean, dtype=float),
        "sem": np.asarray(real_sem, dtype=float),
        "shuffle_mean": np.asarray(shuffle_mean, dtype=float),
        "shuffle_sd": np.asarray(shuffle_sd, dtype=float),
        "p_value": np.asarray(p_value, dtype=float),
    }


def build_output_stem(
    *,
    representation: str,
    v1_tuning_epoch: str,
    decode_epoch: str,
) -> str:
    """Return the shared output stem for one decode epoch."""
    return (
        f"{representation}_v1train-{v1_tuning_epoch}"
        f"_ca1train-{decode_epoch}_decode-{decode_epoch}"
    )


def build_decoded_output_name(
    *,
    representation: str,
    decode_epoch: str,
) -> str:
    """Return the CA1 decoded-state filename stem for one epoch."""
    return f"{representation}_ca1train-{decode_epoch}_decode-{decode_epoch}"


def build_metric_summary_table(
    *,
    unit_mask_table: pd.DataFrame,
    kept_unit_ids: np.ndarray,
    fit_results: dict[str, np.ndarray],
    representation: str,
    v1_tuning_epoch: str,
    decode_epoch: str,
    n_ripples: int,
    n_ripple_bins: int,
) -> pd.DataFrame:
    """Return one parquet-ready summary table for one decode epoch."""
    table = unit_mask_table.copy()
    table.insert(0, "n_ripple_bins", int(n_ripple_bins))
    table.insert(0, "n_ripples", int(n_ripples))
    table.insert(0, "decode_epoch", str(decode_epoch))
    table.insert(0, "v1_tuning_epoch", str(v1_tuning_epoch))
    table.insert(0, "representation", str(representation))

    if kept_unit_ids.size == 0:
        for metric_name in METRIC_LABELS:
            for suffix in ("", "_sem", "_shuffle_mean", "_shuffle_sd", "_p_value"):
                table[f"{metric_name}{suffix}"] = np.nan
        return table

    kept_lookup = {unit_id: index for index, unit_id in enumerate(kept_unit_ids.tolist())}
    for metric_name in METRIC_LABELS:
        summary = summarize_metric_against_shuffle(
            fit_results[f"{metric_name}_folds"],
            fit_results[f"{metric_name}_shuff_folds"],
        )
        metric_values = np.full(len(table), np.nan, dtype=float)
        metric_sem = np.full(len(table), np.nan, dtype=float)
        metric_shuffle_mean = np.full(len(table), np.nan, dtype=float)
        metric_shuffle_sd = np.full(len(table), np.nan, dtype=float)
        metric_p_value = np.full(len(table), np.nan, dtype=float)

        for row_index, unit_id in enumerate(table["unit_id"].tolist()):
            kept_index = kept_lookup.get(unit_id)
            if kept_index is None:
                continue
            metric_values[row_index] = summary["mean"][kept_index]
            metric_sem[row_index] = summary["sem"][kept_index]
            metric_shuffle_mean[row_index] = summary["shuffle_mean"][kept_index]
            metric_shuffle_sd[row_index] = summary["shuffle_sd"][kept_index]
            metric_p_value[row_index] = summary["p_value"][kept_index]

        table[metric_name] = metric_values
        table[f"{metric_name}_sem"] = metric_sem
        table[f"{metric_name}_shuffle_mean"] = metric_shuffle_mean
        table[f"{metric_name}_shuffle_sd"] = metric_shuffle_sd
        table[f"{metric_name}_p_value"] = metric_p_value
    return table


def build_epoch_dataset_name(
    *,
    representation: str,
    v1_tuning_epoch: str,
    decode_epoch: str,
) -> str:
    """Return the combined epoch dataset filename for one decode epoch."""
    return (
        f"{build_output_stem(representation=representation, v1_tuning_epoch=v1_tuning_epoch, decode_epoch=decode_epoch)}"
        "_glm_dataset.nc"
    )


def build_epoch_dataset(
    *,
    fit_results: dict[str, np.ndarray],
    unit_mask_table: pd.DataFrame,
    ripple_epoch_data: dict[str, Any],
    expected_rate_hz: np.ndarray,
    kept_unit_ids: np.ndarray,
    animal_name: str,
    date: str,
    representation: str,
    v1_tuning_epoch: str,
    decode_epoch: str,
    bin_size_s: float,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> Any:
    """Build one epoch-level xarray dataset with GLM inputs and raw fit outputs."""
    xr = require_xarray()

    unit_ids = unit_mask_table["unit_id"].to_numpy()
    kept_lookup = {unit_id: index for index, unit_id in enumerate(kept_unit_ids.tolist())}
    response = np.asarray(ripple_epoch_data["response"], dtype=np.float32)
    decoded_state = np.asarray(ripple_epoch_data["decoded_state"], dtype=float)
    bin_times_s = np.asarray(ripple_epoch_data["bin_times_s"], dtype=float)
    ripple_ids = np.asarray(ripple_epoch_data["ripple_ids"], dtype=int)
    expected_rate = np.asarray(expected_rate_hz, dtype=np.float32)
    if response.shape != expected_rate.shape:
        raise ValueError("Observed ripple counts and expected rates must have matching shapes.")
    n_bins = int(response.shape[0])
    if decoded_state.shape[0] != n_bins or bin_times_s.shape[0] != n_bins:
        raise ValueError("Decoded state and ripple-bin timestamps must match the saved bins.")
    if ripple_ids.shape[0] != n_bins:
        raise ValueError("Ripple ids must match the saved bins.")
    n_folds = int(np.asarray(fit_results["pseudo_r2_folds"]).shape[0])
    n_shuffles = int(np.asarray(fit_results["pseudo_r2_shuff_folds"]).shape[1])
    attrs = {
        "schema_version": "1",
        "animal_name": animal_name,
        "date": date,
        "representation": representation,
        "v1_tuning_epoch": v1_tuning_epoch,
        "decode_epoch": decode_epoch,
        "model_direction": "decoded_ca1_state_to_v1",
        "bin_size_s": float(bin_size_s),
        "n_ripples": int(ripple_epoch_data["n_ripples_kept"]),
        "n_ripple_bins": int(n_bins),
        "n_units": int(unit_ids.size),
        "n_kept_units": int(kept_unit_ids.size),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        "skipped_ripples_json": json.dumps(
            ripple_epoch_data["skipped_ripples"], sort_keys=True
        ),
    }

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "bin_time_s": (("bin",), bin_times_s),
        "decoded_state": (("bin",), decoded_state),
        "ripple_id": (("bin",), ripple_ids),
        "observed_count": (("bin", "unit"), response),
        "expected_rate_hz": (("bin", "unit"), expected_rate),
        "movement_firing_rate_hz": (
            ("unit",),
            unit_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "ripple_firing_rate_hz": (
            ("unit",),
            unit_mask_table["ripple_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "expected_rate_mean_hz": (
            ("unit",),
            unit_mask_table["expected_rate_mean_hz"].to_numpy(dtype=float),
        ),
        "expected_rate_max_hz": (
            ("unit",),
            unit_mask_table["expected_rate_max_hz"].to_numpy(dtype=float),
        ),
        "passes_movement_firing_rate": (
            ("unit",),
            unit_mask_table["passes_movement_firing_rate"].to_numpy(dtype=bool),
        ),
        "passes_ripple_firing_rate": (
            ("unit",),
            unit_mask_table["passes_ripple_firing_rate"].to_numpy(dtype=bool),
        ),
        "passes_expected_rate_predictor": (
            ("unit",),
            unit_mask_table["passes_expected_rate_predictor"].to_numpy(dtype=bool),
        ),
        "keep_unit": (("unit",), unit_mask_table["keep_unit"].to_numpy(dtype=bool)),
        "ripple_source_index": (
            ("ripple",),
            np.asarray(ripple_epoch_data["ripple_source_indices"], dtype=int),
        ),
        "ripple_start_time_s": (
            ("ripple",),
            np.asarray(ripple_epoch_data["ripple_start_times_s"], dtype=float),
        ),
        "ripple_end_time_s": (
            ("ripple",),
            np.asarray(ripple_epoch_data["ripple_end_times_s"], dtype=float),
        ),
    }
    for metric_name in METRIC_LABELS:
        folds_kept = np.asarray(fit_results[f"{metric_name}_folds"], dtype=np.float32)
        shuffle_folds_kept = np.asarray(
            fit_results[f"{metric_name}_shuff_folds"], dtype=np.float32
        )
        folds = np.full((n_folds, unit_ids.size), np.nan, dtype=np.float32)
        shuffle_folds = np.full(
            (n_folds, n_shuffles, unit_ids.size), np.nan, dtype=np.float32
        )
        for unit_index, unit_id in enumerate(unit_ids.tolist()):
            kept_index = kept_lookup.get(unit_id)
            if kept_index is None:
                continue
            folds[:, unit_index] = folds_kept[:, kept_index]
            shuffle_folds[:, :, unit_index] = shuffle_folds_kept[:, :, kept_index]
        if kept_unit_ids.size == 0:
            summary = {
                "mean": np.full(unit_ids.size, np.nan, dtype=float),
                "sem": np.full(unit_ids.size, np.nan, dtype=float),
                "shuffle_mean": np.full(unit_ids.size, np.nan, dtype=float),
                "shuffle_sd": np.full(unit_ids.size, np.nan, dtype=float),
                "p_value": np.full(unit_ids.size, np.nan, dtype=float),
            }
        else:
            summary_kept = summarize_metric_against_shuffle(
                folds_kept,
                shuffle_folds_kept,
            )
            summary = {
                key: np.full(unit_ids.size, np.nan, dtype=float)
                for key in ("mean", "sem", "shuffle_mean", "shuffle_sd", "p_value")
            }
            for unit_index, unit_id in enumerate(unit_ids.tolist()):
                kept_index = kept_lookup.get(unit_id)
                if kept_index is None:
                    continue
                for key in summary:
                    summary[key][unit_index] = summary_kept[key][kept_index]
        data_vars[f"{metric_name}_folds"] = (("fold", "unit"), folds)
        data_vars[f"{metric_name}_shuff_folds"] = (
            ("fold", "shuffle", "unit"),
            shuffle_folds,
        )
        data_vars[f"{metric_name}_mean"] = (
            ("unit",),
            np.asarray(summary["mean"], dtype=float),
        )
        data_vars[f"{metric_name}_sem"] = (
            ("unit",),
            np.asarray(summary["sem"], dtype=float),
        )
        data_vars[f"{metric_name}_shuffle_mean"] = (
            ("unit",),
            np.asarray(summary["shuffle_mean"], dtype=float),
        )
        data_vars[f"{metric_name}_shuffle_sd"] = (
            ("unit",),
            np.asarray(summary["shuffle_sd"], dtype=float),
        )
        data_vars[f"{metric_name}_p_value"] = (
            ("unit",),
            np.asarray(summary["p_value"], dtype=float),
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "bin": np.arange(n_bins, dtype=int),
            "fold": np.arange(n_folds, dtype=int),
            "shuffle": np.arange(n_shuffles, dtype=int),
            "unit": unit_ids,
            "ripple": np.arange(int(ripple_epoch_data["n_ripples_kept"]), dtype=int),
        },
        attrs=attrs,
    )


def get_pyplot():
    """Return pyplot configured for headless script execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finite_values(values: Any) -> np.ndarray:
    """Return only finite values from an array-like input."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def plot_metric_scatter_with_marginals(
    *,
    metric_values: Any,
    p_values: Any,
    animal_name: str,
    date: str,
    representation: str,
    v1_tuning_epoch: str,
    decode_epoch: str,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot one metric-vs-p-value scatter with marginal histograms."""
    plt = get_pyplot()
    values = np.asarray(metric_values, dtype=float).reshape(-1)
    p_value_array = np.asarray(p_values, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(p_value_array)
    finite_metric = values[valid]
    finite_p = p_value_array[valid]

    figure = plt.figure(figsize=(7.2, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(4.0, 1.2),
        height_ratios=(4.0, 1.2),
    )
    scatter_ax = figure.add_subplot(grid[0, 0])
    right_ax = figure.add_subplot(grid[0, 1], sharey=scatter_ax)
    bottom_ax = figure.add_subplot(grid[1, 0], sharex=scatter_ax)
    empty_ax = figure.add_subplot(grid[1, 1])
    empty_ax.axis("off")

    if finite_metric.size:
        scatter_ax.scatter(finite_metric, finite_p, s=18, alpha=0.75, color="tab:blue")
        metric_bins = np.linspace(
            float(np.min(finite_metric)),
            float(np.max(finite_metric)),
            26,
        )
        if np.allclose(metric_bins[0], metric_bins[-1]):
            pad = max(abs(metric_bins[0]) * 0.05, 0.5)
            metric_bins = np.linspace(metric_bins[0] - pad, metric_bins[-1] + pad, 26)
        p_bins = np.linspace(0.0, 1.0, 21)
        bottom_ax.hist(
            finite_metric,
            bins=metric_bins,
            weights=np.full(finite_metric.size, 1.0 / finite_metric.size),
            color="tab:blue",
            alpha=0.65,
            edgecolor="none",
        )
        right_ax.hist(
            finite_p,
            bins=p_bins,
            weights=np.full(finite_p.size, 1.0 / finite_p.size),
            orientation="horizontal",
            color="tab:blue",
            alpha=0.65,
            edgecolor="none",
        )
    else:
        scatter_ax.text(0.5, 0.5, "No finite unit metrics", ha="center", va="center")

    scatter_ax.axhline(0.05, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
    scatter_ax.set_xlabel(metric_label)
    scatter_ax.set_ylabel("Shuffle p-value")
    scatter_ax.set_title("Unit metric vs shuffle p-value")
    scatter_ax.grid(True, alpha=0.2)

    bottom_ax.set_xlabel(metric_label)
    bottom_ax.set_ylabel("Fraction of units")
    bottom_ax.grid(True, alpha=0.2)

    right_ax.set_xlabel("Fraction of units")
    right_ax.set_ylabel("Shuffle p-value")
    right_ax.grid(True, alpha=0.2)
    plt.setp(right_ax.get_yticklabels(), visible=False)

    figure.suptitle(
        f"{animal_name} {date} {representation} V1 train {v1_tuning_epoch} decode {decode_epoch}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def save_metric_figures(
    *,
    table: pd.DataFrame,
    fig_dir: Path,
    animal_name: str,
    date: str,
    representation: str,
    v1_tuning_epoch: str,
    decode_epoch: str,
) -> list[Path]:
    """Save the three per-metric figures for one decode epoch."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = build_output_stem(
        representation=representation,
        v1_tuning_epoch=v1_tuning_epoch,
        decode_epoch=decode_epoch,
    )
    figure_paths: list[Path] = []
    for metric_name, metric_label in METRIC_LABELS.items():
        figure_paths.append(
            plot_metric_scatter_with_marginals(
                metric_values=table[metric_name].to_numpy(),
                p_values=table[f"{metric_name}_p_value"].to_numpy(),
                animal_name=animal_name,
                date=date,
                representation=representation,
                v1_tuning_epoch=v1_tuning_epoch,
                decode_epoch=decode_epoch,
                metric_label=metric_label,
                out_path=fig_dir / f"{stem}_{metric_name}_summary.png",
            )
        )
    return figure_paths


def main() -> None:
    """Run the ripple decoding GLM CLI."""
    args = parse_arguments()
    validate_arguments(args)

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=REGIONS,
    )
    decode_epochs = select_decode_epochs(session["run_epochs"], args.decode_epochs)
    v1_tuning_epoch = validate_v1_tuning_epoch(
        session["run_epochs"],
        args.v1_tuning_epoch,
    )

    feature_by_epoch, bin_edges, feature_name = get_representation_inputs(
        session,
        animal_name=args.animal_name,
        representation=args.representation,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = analysis_path / "ripple_decoding_glm"
    fig_dir = analysis_path / "figs" / "ripple_decoding_glm"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _epoch_tags, timestamps_ephys_by_epoch, ephys_source = load_ephys_timestamps_by_epoch(
        analysis_path
    )
    epoch_intervals = build_epoch_intervals(timestamps_ephys_by_epoch)
    ripple_tables, ripple_source = load_ripple_tables(analysis_path)

    sources = dict(session["sources"])
    sources["timestamps_ephys"] = ephys_source
    sources["ripple_events"] = ripple_source
    fit_parameters = {
        "bin_size_s": args.bin_size_s,
        "ridge_strength": args.ridge_strength,
        "n_splits": args.n_splits,
        "n_shuffles": args.n_shuffles,
        "shuffle_seed": args.shuffle_seed,
        "v1_min_movement_fr_hz": args.v1_min_movement_fr_hz,
        "v1_min_ripple_fr_hz": args.v1_min_ripple_fr_hz,
        "model_direction": "decoded_ca1_state_to_v1",
    }

    v1_spikes = session["spikes_by_region"]["v1"]
    ca1_spikes = session["spikes_by_region"]["ca1"]
    v1_unit_ids = np.asarray(list(v1_spikes.keys()))
    v1_tuning_curves = compute_tuning_curves_for_epoch(
        spikes=v1_spikes,
        feature=feature_by_epoch[v1_tuning_epoch],
        movement_interval=session["movement_by_run"][v1_tuning_epoch],
        bin_edges=bin_edges,
        feature_name=feature_name,
    )
    v1_movement_rates = np.asarray(
        movement_firing_rates["v1"][v1_tuning_epoch],
        dtype=float,
    )
    if v1_movement_rates.shape[0] != v1_unit_ids.size:
        raise ValueError("V1 movement firing rates do not match the saved V1 unit count.")

    saved_decoded_paths: list[Path] = []
    saved_table_paths: list[Path] = []
    saved_dataset_paths: list[Path] = []
    saved_figure_paths: list[Path] = []
    skipped_epochs: list[dict[str, Any]] = []

    for decode_epoch in decode_epochs:
        ripple_table = ripple_tables.get(decode_epoch, empty_ripple_table())
        if ripple_table.empty:
            skipped_epochs.append(
                {"epoch": decode_epoch, "reason": "No ripple events were found for this epoch."}
            )
            print(f"Skipping {args.animal_name} {args.date} {decode_epoch}: no ripple events")
            continue

        try:
            ca1_tuning_curves = compute_tuning_curves_for_epoch(
                spikes=ca1_spikes,
                feature=feature_by_epoch[decode_epoch],
                movement_interval=session["movement_by_run"][decode_epoch],
                bin_edges=bin_edges,
                feature_name=feature_name,
            )
            ripple_epoch_data = assemble_ripple_epoch_data(
                ca1_spikes=ca1_spikes,
                v1_spikes=v1_spikes,
                ca1_tuning_curves=ca1_tuning_curves,
                ripple_table=ripple_table,
                epoch_interval=epoch_intervals[decode_epoch],
                bin_size_s=args.bin_size_s,
                v1_unit_ids=v1_unit_ids,
            )
        except Exception as exc:
            skipped_epochs.append(
                {
                    "epoch": decode_epoch,
                    "reason": "failed to assemble ripple decode inputs",
                    "error": str(exc),
                }
            )
            print(
                f"Skipping {args.animal_name} {args.date} {decode_epoch}: "
                f"failed to assemble inputs: {exc}"
            )
            continue

        if ripple_epoch_data["n_ripples_kept"] < args.n_splits:
            skipped_epochs.append(
                {
                    "epoch": decode_epoch,
                    "reason": (
                        "Not enough usable ripples for CV: "
                        f"n_ripples={ripple_epoch_data['n_ripples_kept']}, "
                        f"n_splits={args.n_splits}"
                    ),
                    "skipped_ripples": ripple_epoch_data["skipped_ripples"],
                }
            )
            print(
                f"Skipping {args.animal_name} {args.date} {decode_epoch}: "
                "not enough usable ripples for CV"
            )
            continue

        expected_rate_hz = build_expected_rate_matrix(
            tuning_curves=v1_tuning_curves,
            unit_ids=v1_unit_ids,
            decoded_state=ripple_epoch_data["decoded_state"],
            bin_edges=bin_edges,
        )
        unit_mask_table = build_v1_unit_mask_table(
            unit_ids=v1_unit_ids,
            movement_firing_rates_hz=v1_movement_rates,
            ripple_counts=ripple_epoch_data["response"],
            expected_rate_hz=expected_rate_hz,
            bin_size_s=args.bin_size_s,
            min_movement_fr_hz=args.v1_min_movement_fr_hz,
            min_ripple_fr_hz=args.v1_min_ripple_fr_hz,
        )
        keep_unit = unit_mask_table["keep_unit"].to_numpy(dtype=bool)
        kept_unit_ids = v1_unit_ids[keep_unit]

        fit_results = fit_expected_rate_glm(
            X=np.asarray(expected_rate_hz[:, keep_unit], dtype=float),
            y=np.asarray(ripple_epoch_data["response"][:, keep_unit], dtype=float),
            ripple_ids=np.asarray(ripple_epoch_data["ripple_ids"], dtype=int),
            n_splits=args.n_splits,
            n_shuffles=args.n_shuffles,
            shuffle_seed=args.shuffle_seed,
            ridge_strength=args.ridge_strength,
        )
        summary_table = build_metric_summary_table(
            unit_mask_table=unit_mask_table,
            kept_unit_ids=kept_unit_ids,
            fit_results=fit_results,
            representation=args.representation,
            v1_tuning_epoch=v1_tuning_epoch,
            decode_epoch=decode_epoch,
            n_ripples=ripple_epoch_data["n_ripples_kept"],
            n_ripple_bins=ripple_epoch_data["n_bins"],
        )
        epoch_dataset = build_epoch_dataset(
            fit_results=fit_results,
            unit_mask_table=unit_mask_table,
            ripple_epoch_data=ripple_epoch_data,
            expected_rate_hz=expected_rate_hz,
            kept_unit_ids=kept_unit_ids,
            animal_name=args.animal_name,
            date=args.date,
            representation=args.representation,
            v1_tuning_epoch=v1_tuning_epoch,
            decode_epoch=decode_epoch,
            bin_size_s=args.bin_size_s,
            sources=sources,
            fit_parameters=fit_parameters,
        )

        decoded_path = (
            data_dir
            / f"{build_decoded_output_name(representation=args.representation, decode_epoch=decode_epoch)}.npz"
        )
        ripple_epoch_data["decoded_tsd"].save(decoded_path)
        saved_decoded_paths.append(decoded_path)

        table_path = (
            data_dir
            / f"{build_output_stem(representation=args.representation, v1_tuning_epoch=v1_tuning_epoch, decode_epoch=decode_epoch)}_glm_metrics.parquet"
        )
        summary_table.to_parquet(table_path, index=False)
        saved_table_paths.append(table_path)

        dataset_path = data_dir / build_epoch_dataset_name(
            representation=args.representation,
            v1_tuning_epoch=v1_tuning_epoch,
            decode_epoch=decode_epoch,
        )
        epoch_dataset.to_netcdf(dataset_path)
        saved_dataset_paths.append(dataset_path)

        saved_figure_paths.extend(
            save_metric_figures(
                table=summary_table,
                fig_dir=fig_dir,
                animal_name=args.animal_name,
                date=args.date,
                representation=args.representation,
                v1_tuning_epoch=v1_tuning_epoch,
                decode_epoch=decode_epoch,
            )
        )

    if not saved_table_paths:
        raise RuntimeError(
            "All requested decode epochs were skipped. "
            f"Epoch reasons: {skipped_epochs!r}"
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.ripple.ripple_decoding_glm",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "representation": args.representation,
            "data_root": args.data_root,
            "decode_epochs": decode_epochs,
            "v1_tuning_epoch": v1_tuning_epoch,
            "bin_size_s": args.bin_size_s,
            "ridge_strength": args.ridge_strength,
            "n_splits": args.n_splits,
            "n_shuffles": args.n_shuffles,
            "shuffle_seed": args.shuffle_seed,
            "v1_min_movement_fr_hz": args.v1_min_movement_fr_hz,
            "v1_min_ripple_fr_hz": args.v1_min_ripple_fr_hz,
            "model_direction": "decoded_ca1_state_to_v1",
        },
        outputs={
            "sources": sources,
            "saved_decoded_paths": saved_decoded_paths,
            "saved_table_paths": saved_table_paths,
            "saved_dataset_paths": saved_dataset_paths,
            "saved_figure_paths": saved_figure_paths,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
