from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
    "false"  # don't grab almost all VRAM up front
)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"  # or smaller, e.g. 0.50

import gc
import jax
import nemos as nmo
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import spikeinterface.full as si
import kyutils
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.special import gammaln

from typing import Tuple, Literal, Any, Optional, Dict, Sequence

import inspect


animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

regions = ["v1", "ca1"]

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "ripple" / "Kay_ripple_detector.pkl", "rb") as f:
    Kay_ripple_detector = pickle.load(f)

sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")


def get_tsgroup(sorting):
    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
        )
    tsgroup = nap.TsGroup(data, time_units="s")
    return tsgroup


spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])

ep = {}
for epoch in epoch_list:
    ep[epoch] = nap.IntervalSet(
        start=timestamps_ephys[epoch][0],
        end=timestamps_ephys[epoch][-1],
    )


def save_results_npz(out_path: Path, results: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    to_save = dict(results)
    if "fold_info" in to_save:
        to_save["fold_info"] = np.array(to_save["fold_info"], dtype=object)

    np.savez_compressed(out_path, **to_save)


def make_preripple_ep(
    ripple_ep: nap.IntervalSet,
    epoch_ep: nap.IntervalSet,
    *,
    window_s: Optional[float] = None,
    buffer_s: float = 0.02,
    exclude_ripples: bool = True,
    exclude_ripple_guard_s: float = 0.05,
) -> nap.IntervalSet:
    """
    Create a paired pre-ripple IntervalSet aligned 1:1 with `ripple_ep`.

    If `window_s` is None, uses matched-length windows (same duration as each ripple):
        [t_start - dur - buffer, t_start - buffer]

    If `window_s` is provided, uses fixed-length windows:
        [t_start - window - buffer, t_start - buffer]

    Parameters
    ----------
    ripple_ep
        Ripple intervals.
    epoch_ep
        Epoch bounds used to clip windows.
    window_s
        Fixed pre-window length in seconds. If None, uses per-ripple duration.
    buffer_s
        Gap between pre-window end and ripple start (seconds).
    exclude_ripples
        If True, subtract ripple intervals (expanded by guard) from pre windows.
    exclude_ripple_guard_s
        Guard around ripples when excluding (seconds).

    Returns
    -------
    pre_ep
        IntervalSet of pre windows, intended to remain 1:1 with ripples whenever possible.
        Note: clipping/exclusion can drop/fragment intervals; we handle alignment in code below
        by filtering to windows that remain valid and 1:1.
    """
    r_start = np.asarray(ripple_ep.start, dtype=float)
    r_end = np.asarray(ripple_ep.end, dtype=float)

    if window_s is None:
        dur = r_end - r_start
    else:
        dur = np.full_like(r_start, float(window_s))

    pre_start = r_start - dur - buffer_s
    pre_end = r_start - buffer_s

    keep = pre_end > pre_start
    pre_start = pre_start[keep]
    pre_end = pre_end[keep]

    if pre_start.size == 0:
        return nap.IntervalSet(
            start=np.array([], dtype=float), end=np.array([], dtype=float)
        )

    pre_ep = nap.IntervalSet(start=pre_start, end=pre_end).intersect(epoch_ep)

    if exclude_ripples and len(pre_ep) > 0 and len(ripple_ep) > 0:
        rip_excl = nap.IntervalSet(
            start=np.asarray(ripple_ep.start, float) - exclude_ripple_guard_s,
            end=np.asarray(ripple_ep.end, float) + exclude_ripple_guard_s,
        ).intersect(epoch_ep)
        pre_ep = pre_ep.set_diff(rip_excl)

    return pre_ep


# def make_preripple_ep(
#     ripple_ep: nap.IntervalSet,
#     epoch_ep: nap.IntervalSet,
#     *,
#     buffer_s: float = 0.02,
#     match_ripple_duration: bool = True,
#     # Only used when match_ripple_duration=False
#     window_s: Optional[float] = None,
#     exclude_ripples: bool = True,
#     exclude_ripple_guard_s: float = 0.05,
# ) -> nap.IntervalSet:
#     """
#     Create PRE windows relative to each ripple.

#     If match_ripple_duration=True:
#         PRE_i has the same duration as ripple_i and ends buffer_s before ripple start:
#             PRE_i = [r_start - dur - buffer_s, r_start - buffer_s]
#         No exclusion/guard logic is applied (overlap with other ripples is allowed).
#         PRE is still intersected with epoch_ep to avoid out-of-epoch intervals.

#     If match_ripple_duration=False:
#         Uses the "full" behavior:
#             - If window_s is None: dur_i = ripple duration
#             - Else: dur_i = window_s for all ripples
#         Then optionally excludes ripples (expanded by exclude_ripple_guard_s).

#     Parameters
#     ----------
#     ripple_ep
#         Ripple intervals.
#     epoch_ep
#         Epoch bounds used to clip windows.
#     buffer_s
#         Gap between PRE end and ripple start (seconds).
#     match_ripple_duration
#         If True, force PRE duration = ripple duration and skip exclusion logic.
#     window_s
#         Fixed PRE duration (seconds), only used when match_ripple_duration=False.
#     exclude_ripples
#         Only used when match_ripple_duration=False.
#     exclude_ripple_guard_s
#         Only used when match_ripple_duration=False.

#     Returns
#     -------
#     pre_ep
#         IntervalSet of PRE windows.
#     """
#     r_start = np.asarray(ripple_ep.start, dtype=float)
#     r_end = np.asarray(ripple_ep.end, dtype=float)

#     if r_start.size == 0:
#         return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

#     if match_ripple_duration:
#         dur = r_end - r_start
#     else:
#         if window_s is None:
#             dur = r_end - r_start
#         else:
#             dur = np.full_like(r_start, float(window_s))

#     if np.any(dur <= 0):
#         raise ValueError("Non-positive duration encountered in ripple_ep (or window_s).")

#     pre_start = r_start - dur - float(buffer_s)
#     pre_end = r_start - float(buffer_s)

#     keep = pre_end > pre_start
#     pre_start = pre_start[keep]
#     pre_end = pre_end[keep]

#     if pre_start.size == 0:
#         return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

#     # Always clip to epoch bounds so counts don't blow up outside the recording interval
#     pre_ep = nap.IntervalSet(start=pre_start, end=pre_end).intersect(epoch_ep)

#     # Only do exclusion logic in the "full" mode
#     if (not match_ripple_duration) and exclude_ripples and (len(pre_ep) > 0) and (len(ripple_ep) > 0):
#         rip_excl = nap.IntervalSet(
#             start=np.asarray(ripple_ep.start, dtype=float) - float(exclude_ripple_guard_s),
#             end=np.asarray(ripple_ep.end, dtype=float) + float(exclude_ripple_guard_s),
#         ).intersect(epoch_ep)
#         pre_ep = pre_ep.set_diff(rip_excl)

#     return pre_ep


def bits_per_spike_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """
    Information gain in bits/spike per neuron:
        (LL_model - LL_null) / (N_spikes * ln(2))

    Null model is constant-rate (mean of y_null_fit) evaluated on y_test.

    Returns
    -------
    bps : np.ndarray, shape (n_neurons,)
        Bits per spike per neuron. NaN if the neuron has 0 spikes in y_test.
    """
    y_test = np.asarray(y_test, dtype=float)
    lam_test = np.asarray(lam_test, dtype=float)
    y_null_fit = np.asarray(y_null_fit, dtype=float)

    ll_model = poisson_ll_per_neuron(y_test, lam_test)

    lam0 = np.mean(y_null_fit, axis=0)  # (n_neurons,)
    lam0 = np.clip(lam0, 1e-12, None)

    # Compute LL of constant-rate null without repeating lam0 across samples
    ll_null = np.sum(
        y_test * np.log(lam0[None, :]) - lam0[None, :] - gammaln(y_test + 1),
        axis=0,
    )

    spikes = np.sum(y_test, axis=0)  # (n_neurons,)
    denom = spikes * np.log(2.0)

    bps = (ll_model - ll_null) / np.clip(denom, 1e-12, None)
    bps = np.where(spikes > 0, bps, np.nan)  # undefined if no spikes in test
    return bps


def poisson_ll_per_neuron(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """
    Poisson log-likelihood per neuron (column), summed over samples (rows).

    Parameters
    ----------
    y : np.ndarray, shape (n_samples, n_neurons)
        Observed counts.
    lam : np.ndarray, shape (n_samples, n_neurons)
        Predicted rates (must be > 0).

    Returns
    -------
    ll : np.ndarray, shape (n_neurons,)
        Log-likelihood per neuron.
    """
    lam = np.clip(lam, 1e-12, None)
    return np.sum(y * np.log(lam) - lam - gammaln(y + 1), axis=0)


def mcfadden_pseudo_r2_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """
    McFadden pseudo-R^2 per neuron, using a constant-rate null model fit on y_train.

    Notes
    -----
    Uses:
        $$R^2 = 1 - \\frac{\\ell(\\hat\\lambda)}{\\ell(\\lambda_0)}$$

    where lambda_0 is the mean rate from y_train broadcast across y_test.

    Parameters
    ----------
    y_test : np.ndarray, shape (n_test, n_neurons)
    lam_test : np.ndarray, shape (n_test, n_neurons)
    y_train : np.ndarray, shape (n_train, n_neurons)

    Returns
    -------
    r2 : np.ndarray, shape (n_neurons,)
    """
    ll_model = poisson_ll_per_neuron(y_test, lam_test)

    lam0 = np.mean(y_train, axis=0, keepdims=True)
    lam0 = np.repeat(lam0, y_test.shape[0], axis=0)
    ll_null = poisson_ll_per_neuron(y_test, lam0)

    # guard against division by ~0 (can happen for always-0 cells if you didn't filter)
    ll_null = np.where(np.abs(ll_null) < 1e-12, -1e-12, ll_null)
    return 1.0 - (ll_model / ll_null)


def poisson_ll_saturated_per_neuron(y: np.ndarray) -> np.ndarray:
    """
    Saturated Poisson log-likelihood per neuron (lam = y).

    Notes
    -----
    When y == 0, the term y*log(y) is treated as 0 (via clipping).
    """
    y = np.asarray(y, dtype=float)
    y_safe = np.clip(y, 1e-12, None)
    return np.sum(y * np.log(y_safe) - y - gammaln(y + 1), axis=0)


def mae_per_neuron(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean absolute error per neuron.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_neurons)
    y_pred : np.ndarray, shape (n_samples, n_neurons)

    Returns
    -------
    mae : np.ndarray, shape (n_neurons,)
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)


def deviance_explained_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """
    Poisson deviance explained per neuron.

    Parameters
    ----------
    y_test : np.ndarray, shape (n_samples, n_neurons)
        Observed counts on the evaluation set.
    lam_test : np.ndarray, shape (n_samples, n_neurons)
        Predicted rates (lambda) on the evaluation set.
    y_null_fit : np.ndarray, shape (n_fit, n_neurons)
        Data used to fit the null constant-rate model (typically training set).

    Returns
    -------
    de : np.ndarray, shape (n_neurons,)
        Deviance explained per neuron.
    """
    y_test = np.asarray(y_test, dtype=float)
    lam_test = np.asarray(lam_test, dtype=float)
    y_null_fit = np.asarray(y_null_fit, dtype=float)

    ll_model = poisson_ll_per_neuron(y_test, lam_test)

    lam0 = np.mean(y_null_fit, axis=0, keepdims=True)  # (1, n_neurons)
    lam0 = np.repeat(lam0, y_test.shape[0], axis=0)  # (n_samples, n_neurons)
    ll_null = poisson_ll_per_neuron(y_test, lam0)

    ll_sat = poisson_ll_saturated_per_neuron(y_test)

    denom = ll_sat - ll_null
    # Guard: if denom ~ 0 (e.g., constant/zero counts), deviance explained is undefined.
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    de = (ll_model - ll_null) / denom
    return de


def shuffle_time_per_neuron(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y)
    n_samples, n_neurons = y.shape
    idx = np.vstack([rng.permutation(n_samples) for _ in range(n_neurons)]).T
    return np.take_along_axis(y, idx, axis=0)


def _first_nonfinite_info(a: np.ndarray) -> Optional[str]:
    a = np.asarray(a)
    mask = ~np.isfinite(a)
    if not np.any(mask):
        return None
    idx = np.argwhere(mask)[0]
    val = a[tuple(idx)]
    return f"shape={a.shape}, first_bad_idx={tuple(idx)}, value={val}"


def _preprocess_X_fit_apply(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit preprocessing on X_train and apply to X_train/X_test.

    Steps
    -----
    1) Drop near-constant features (std <= 1e-6) based on X_train.
    2) Z-score using X_train mean/std.
    3) Scale by sqrt(n_features) to stabilize optimization.
    4) Clip to [-10, 10].

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test : np.ndarray, shape (n_test, n_features)

    Returns
    -------
    X_train_pp : np.ndarray, shape (n_train, n_features_kept)
    X_test_pp : np.ndarray, shape (n_test, n_features_kept)
    keep_x : np.ndarray, shape (n_features,)
        Boolean mask of kept features.
    mean : np.ndarray, shape (n_features_kept,)
    std : np.ndarray, shape (n_features_kept,)
    """
    keep_x = X_train.std(axis=0) > 1e-6
    X_train_k = X_train[:, keep_x]
    X_test_k = X_test[:, keep_x]

    mean = X_train_k.mean(axis=0, keepdims=True)
    std = X_train_k.std(axis=0, keepdims=True)

    X_train_pp = (X_train_k - mean) / (std + 1e-8)
    X_test_pp = (X_test_k - mean) / (std + 1e-8)

    X_train_pp /= np.sqrt(max(X_train_pp.shape[1], 1))
    X_test_pp /= np.sqrt(max(X_train_pp.shape[1], 1))

    X_train_pp = np.clip(X_train_pp, -10.0, 10.0)
    X_test_pp = np.clip(X_test_pp, -10.0, 10.0)

    return X_train_pp, X_test_pp, keep_x, mean.ravel(), std.ravel()


# def fit_ripple_glm_train_on_ripple_predict_pre(
#     epoch: str,
#     *,
#     spikes: Dict[str, nap.TsGroup],
#     regions: Tuple[str, str] = ("v1", "ca1"),
#     Kay_ripple_detector: Dict[str, Any],
#     ep: Dict[str, nap.IntervalSet],
#     min_spikes_per_ripple: float = 0.05,
#     # ripple CV shuffles (refit each shuffle model per fold)
#     n_shuffles_ripple: int = 100,
#     random_seed: int = 0,
#     ripple_window: Optional[float] = None,
#     pre_window_s: Optional[float] = None,
#     pre_buffer_s: float = 0.02,
#     exclude_ripples=True,
#     pre_exclude_guard_s: float = 0.05,
#     n_splits: int = 5,
#     ridge_strength: float = 0.1,
#     maxiter: int = 6000,
#     tol: float = 1e-7,
#     store_ripple_shuffle_preds: bool = True,
# ) -> dict:
#     """
#     Logic
#     -----
#     1) RIPPLE (proper CV):
#         - Split ripples into folds.
#         - Fit GLM on ripple-train.
#         - Evaluate on ripple-test.
#         - Build ripple null by refitting on shuffled y_train_r.
#         - Optionally store ripple shuffled predictions on held-out ripples.

#     2) PRE (single evaluation, NO shuffle null):
#         - Fit ONE GLM on ALL ripple data.
#         - Predict on ALL pre windows.
#         - Compute metrics once (pseudo-R2, MAE, LL) using PRE baseline for pseudo-R2.

#     Returns
#     -------
#     dict
#         Includes:
#         - ripple_* arrays: per-fold per-cell metrics, plus yhat/y_test filled across folds,
#                           plus shuffle-fold metrics and optional shuffled predictions.
#         - pre_* arrays: per-cell metrics for the single full-ripple-trained model,
#                         plus y_pre_test and yhat_pre.
#     """
#     v1_region, ca1_region = regions
#     rng = np.random.default_rng(random_seed)

#     # --- Build ripple IntervalSet ---
#     if ripple_window is None:
#         ripple_ep = nap.IntervalSet(
#             start=Kay_ripple_detector[epoch]["start_time"],
#             end=Kay_ripple_detector[epoch]["end_time"],
#         )
#     else:
#         ripple_start = np.asarray(Kay_ripple_detector[epoch]["start_time"], dtype=float)
#         ripple_ep = nap.IntervalSet(
#             start=ripple_start,
#             end=ripple_start + float(ripple_window),
#         )

#     # --- Build pre IntervalSet ---
#     pre_ep = make_preripple_ep(
#         ripple_ep=ripple_ep,
#         epoch_ep=ep[epoch],
#         window_s=pre_window_s,
#         buffer_s=pre_buffer_s,
#         exclude_ripples=exclude_ripples,
#         exclude_ripple_guard_s=pre_exclude_guard_s,
#     )

#     # --- Count spikes ---
#     X_r = np.asarray(spikes[ca1_region].count(ep=ripple_ep), dtype=np.float64)
#     y_r = np.asarray(spikes[v1_region].count(ep=ripple_ep), dtype=np.float64)

#     X_p = np.asarray(spikes[ca1_region].count(ep=pre_ep), dtype=np.float64)
#     y_p = np.asarray(spikes[v1_region].count(ep=pre_ep), dtype=np.float64)

#     n_r, _ = X_r.shape
#     n_p = X_p.shape[0]

#     if n_r < n_splits:
#         raise ValueError(f"Not enough ripples: n_ripples={n_r}, n_splits={n_splits}")

#     for name, arr in [("X_r", X_r), ("y_r", y_r), ("X_p", X_p), ("y_p", y_p)]:
#         info = _first_nonfinite_info(arr)
#         if info:
#             raise ValueError(f"Non-finite in {name}: {info}")

#     # --- Filter V1 neurons based on ripple data only ---
#     keep_y = (y_r.sum(axis=0) / max(n_r, 1)) >= float(min_spikes_per_ripple)
#     y_r = y_r[:, keep_y]
#     y_p = y_p[:, keep_y]

#     v1_unit_ids = np.array(list(spikes[v1_region].keys()))
#     ca1_unit_ids = np.array(list(spikes[ca1_region].keys()))
#     v1_unit_ids_kept = v1_unit_ids[keep_y]

#     n_cells = y_r.shape[1]
#     if n_cells == 0:
#         raise ValueError(
#             f"No V1 units passed min_spikes_per_ripple={min_spikes_per_ripple}. "
#             f"n_ripples={n_r}"
#         )

#     # =========================
#     # 1) RIPPLE: proper CV
#     # =========================
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

#     y_r_test_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)
#     y_r_hat_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)

#     y_r_hat_shuff: Optional[np.ndarray] = None
#     if store_ripple_shuffle_preds and n_shuffles_ripple > 0:
#         y_r_hat_shuff = np.full(
#             (n_shuffles_ripple, n_r, n_cells), np.nan, dtype=np.float32
#         )

#     def _alloc_metrics_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         r2 = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
#         mae = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
#         ll = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
#         return r2, mae, ll

#     r2_r_f, mae_r_f, ll_r_f = _alloc_metrics_2d()
#     devexp_r_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)

#     r2_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
#     mae_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
#     ll_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
#     devexp_r_sh = np.full(
#         (n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32
#     )

#     fold_info: list[dict] = []

#     for fold, (train_idx, test_idx) in enumerate(kf.split(X_r)):
#         X_train_r = X_r[train_idx]
#         X_test_r = X_r[test_idx]
#         y_train_r = y_r[train_idx]
#         y_test_r = y_r[test_idx]

#         X_train_pp, X_test_r_pp, keep_x, mean, std = _preprocess_X_fit_apply(
#             X_train=X_train_r,
#             X_test=X_test_r,
#         )

#         # Guard: if all features were dropped, fitting will likely fail
#         if X_train_pp.shape[1] == 0:
#             raise ValueError(
#                 "All X features were near-constant after filtering in this fold; cannot fit GLM."
#             )

#         glm = nmo.glm.PopulationGLM(
#             solver_name="LBFGS",
#             regularizer="Ridge",
#             regularizer_strength=float(ridge_strength),
#             solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
#         )
#         glm.fit(X_train_pp, y_train_r)

#         lam_r = np.asarray(glm.predict(X_test_r_pp), dtype=np.float64)

#         y_r_hat_full[test_idx] = lam_r.astype(np.float32)
#         y_r_test_full[test_idx] = y_test_r.astype(np.float32)

#         r2_r_f[fold] = mcfadden_pseudo_r2_per_neuron(y_test_r, lam_r, y_train_r).astype(
#             np.float32
#         )
#         mae_r_f[fold] = mae_per_neuron(y_test_r, lam_r).astype(np.float32)
#         ll_r_f[fold] = poisson_ll_per_neuron(y_test_r, lam_r).astype(np.float32)
#         devexp_r_f[fold] = deviance_explained_per_neuron(
#             y_test=y_test_r,
#             lam_test=lam_r,
#             y_null_fit=y_train_r,  # null = constant rate fit on train
#         ).astype(np.float32)

#         for s in range(n_shuffles_ripple):
#             y_train_sh = shuffle_time_per_neuron(y_train_r, rng)

#             glm_s = nmo.glm.PopulationGLM(
#                 solver_name="LBFGS",
#                 regularizer="Ridge",
#                 regularizer_strength=float(ridge_strength),
#                 solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
#             )
#             glm_s.fit(X_train_pp, y_train_sh)

#             lam_r_s = np.asarray(glm_s.predict(X_test_r_pp), dtype=np.float64)

#             if y_r_hat_shuff is not None:
#                 y_r_hat_shuff[s, test_idx] = lam_r_s.astype(np.float32)

#             r2_r_sh[fold, s] = mcfadden_pseudo_r2_per_neuron(
#                 y_test_r, lam_r_s, y_train_r
#             ).astype(np.float32)
#             mae_r_sh[fold, s] = mae_per_neuron(y_test_r, lam_r_s).astype(np.float32)
#             ll_r_sh[fold, s] = poisson_ll_per_neuron(y_test_r, lam_r_s).astype(
#                 np.float32
#             )
#             devexp_r_sh[fold, s] = deviance_explained_per_neuron(
#                 y_test=y_test_r,
#                 lam_test=lam_r_s,
#                 y_null_fit=y_train_r,  # keep the same null fit for comparability
#             ).astype(np.float32)

#             del glm_s, y_train_sh, lam_r_s
#             if (s + 1) % 5 == 0:
#                 jax.clear_caches()
#                 gc.collect()

#         fold_info.append(
#             dict(
#                 fold=int(fold),
#                 train_idx=train_idx,
#                 test_idx=test_idx,
#                 keep_x=keep_x,
#                 x_mean=mean,
#                 x_std=std,
#                 n_ripples=int(n_r),
#                 n_pre=int(n_p),
#             )
#         )

#         del glm, lam_r
#         jax.clear_caches()
#         gc.collect()

#     # ripple sanity checks
#     for name, arr in [("y_r_test_full", y_r_test_full), ("y_r_hat_full", y_r_hat_full)]:
#         if np.isnan(arr).any():
#             raise ValueError(
#                 f"{name} has NaNs: some rows were never assigned to a test fold"
#             )

#     if y_r_hat_shuff is not None and np.isnan(y_r_hat_shuff).any():
#         raise ValueError(
#             "y_r_hat_shuff has NaNs: some rows were never assigned to a test fold"
#         )

#     # =========================
#     # 2) PRE: train on ALL ripples, test on ALL pre (NO shuffles)
#     # =========================
#     X_r_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(
#         X_train=X_r,
#         X_test=X_r,  # unused
#     )

#     if X_r_all_pp.shape[1] == 0:
#         raise ValueError(
#             "All X features were near-constant after filtering; cannot fit GLM."
#         )

#     X_p_k = X_p[:, keep_x_all]
#     X_p_pp = (X_p_k - mean_all[None, :]) / (std_all[None, :] + 1e-8)
#     X_p_pp /= np.sqrt(max(X_r_all_pp.shape[1], 1))
#     X_p_pp = np.clip(X_p_pp, -10.0, 10.0)

#     glm_all = nmo.glm.PopulationGLM(
#         solver_name="LBFGS",
#         regularizer="Ridge",
#         regularizer_strength=float(ridge_strength),
#         solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
#     )
#     glm_all.fit(X_r_all_pp, y_r)

#     lam_pre = np.asarray(glm_all.predict(X_p_pp), dtype=np.float64)  # (n_p, n_cells)
#     lam_pre_f32 = lam_pre.astype(np.float32)

#     # PRE metrics (single model); pseudo-R2 null baseline uses PRE mean (y_train=y_p)
#     r2_pre_cell = mcfadden_pseudo_r2_per_neuron(y_p, lam_pre, y_p).astype(np.float32)
#     mae_pre_cell = mae_per_neuron(y_p, lam_pre).astype(np.float32)
#     ll_pre_cell = poisson_ll_per_neuron(y_p, lam_pre).astype(np.float32)
#     devexp_pre_cell = deviance_explained_per_neuron(
#         y_test=y_p,
#         lam_test=lam_pre,
#         y_null_fit=y_p,  # null baseline = PRE mean, consistent with your PRE pseudo-R2
#     ).astype(np.float32)

#     del glm_all
#     jax.clear_caches()
#     gc.collect()

#     results = dict(
#         epoch=epoch,
#         random_seed=int(random_seed),
#         min_spikes_per_ripple=float(min_spikes_per_ripple),
#         n_splits=int(n_splits),
#         n_shuffles_ripple=int(n_shuffles_ripple),
#         ripple_window=None if ripple_window is None else float(ripple_window),
#         pre_window_s=None if pre_window_s is None else float(pre_window_s),
#         pre_buffer_s=float(pre_buffer_s),
#         pre_exclude_guard_s=float(pre_exclude_guard_s),
#         exclude_ripples=exclude_ripples,
#         n_ripples=int(n_r),
#         n_pre=int(n_p),
#         n_cells=int(n_cells),
#         v1_unit_ids=v1_unit_ids_kept,
#         ca1_unit_ids=ca1_unit_ids,
#         fold_info=fold_info,
#         # --- RIPPLE (proper CV) ---
#         pseudo_r2_ripple_folds=r2_r_f,
#         mae_ripple_folds=mae_r_f,
#         ll_ripple_folds=ll_r_f,
#         devexp_ripple_folds=devexp_r_f,
#         pseudo_r2_ripple_shuff_folds=r2_r_sh,
#         mae_ripple_shuff_folds=mae_r_sh,
#         ll_ripple_shuff_folds=ll_r_sh,
#         devexp_ripple_shuff_folds=devexp_r_sh,
#         y_ripple_test=y_r_test_full,
#         yhat_ripple=y_r_hat_full,
#         yhat_ripple_shuff=y_r_hat_shuff,  # (S_r, n_r, n_cells) or None
#         # --- PRE (single fit on all ripple, single test on all pre) ---
#         y_pre_test=y_p.astype(np.float32),
#         yhat_pre=lam_pre_f32,
#         pseudo_r2_pre=r2_pre_cell,
#         mae_pre=mae_pre_cell,
#         ll_pre=ll_pre_cell,
#         devexp_pre=devexp_pre_cell,
#         # preprocessing used for ALL-ripple fit
#         keep_x_all=keep_x_all,
#         x_mean_all=mean_all,
#         x_std_all=std_all,
#     )
#     return results


def fit_ripple_glm_train_on_ripple_predict_pre(
    epoch: str,
    *,
    spikes: Dict[str, nap.TsGroup],
    regions: Tuple[str, str] = ("v1", "ca1"),
    Kay_ripple_detector: Dict[str, Any],
    ep: Dict[str, nap.IntervalSet],
    min_spikes_per_ripple: float = 0.05,
    # ripple CV shuffles (refit each shuffle model per fold)
    n_shuffles_ripple: int = 100,
    random_seed: int = 0,
    ripple_window: Optional[float] = None,
    pre_window_s: Optional[float] = None,
    pre_buffer_s: float = 0.02,
    exclude_ripples=True,
    pre_exclude_guard_s: float = 0.05,
    n_splits: int = 5,
    ridge_strength: float = 0.1,
    maxiter: int = 6000,
    tol: float = 1e-7,
    store_ripple_shuffle_preds: bool = True,
) -> dict:
    v1_region, ca1_region = regions
    rng = np.random.default_rng(random_seed)

    # --- Build ripple IntervalSet ---
    if ripple_window is None:
        ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"],
            end=Kay_ripple_detector[epoch]["end_time"],
        )
    else:
        ripple_start = np.asarray(Kay_ripple_detector[epoch]["start_time"], dtype=float)
        ripple_ep = nap.IntervalSet(
            start=ripple_start,
            end=ripple_start + float(ripple_window),
        )

    # --- Build pre IntervalSet ---
    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=ep[epoch],
        window_s=pre_window_s,
        buffer_s=pre_buffer_s,
        exclude_ripples=exclude_ripples,
        exclude_ripple_guard_s=pre_exclude_guard_s,
    )

    # --- Count spikes ---
    X_r = np.asarray(spikes[ca1_region].count(ep=ripple_ep), dtype=np.float64)
    y_r = np.asarray(spikes[v1_region].count(ep=ripple_ep), dtype=np.float64)

    X_p = np.asarray(spikes[ca1_region].count(ep=pre_ep), dtype=np.float64)
    y_p = np.asarray(spikes[v1_region].count(ep=pre_ep), dtype=np.float64)

    n_r, _ = X_r.shape
    n_p = X_p.shape[0]

    if n_r < n_splits:
        raise ValueError(f"Not enough ripples: n_ripples={n_r}, n_splits={n_splits}")

    for name, arr in [("X_r", X_r), ("y_r", y_r), ("X_p", X_p), ("y_p", y_p)]:
        info = _first_nonfinite_info(arr)
        if info:
            raise ValueError(f"Non-finite in {name}: {info}")

    # --- Filter V1 neurons based on ripple data only ---
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

    # =========================
    # 1) RIPPLE: proper CV
    # =========================
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    y_r_test_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)
    y_r_hat_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)

    y_r_hat_shuff: Optional[np.ndarray] = None
    if store_ripple_shuffle_preds and n_shuffles_ripple > 0:
        y_r_hat_shuff = np.full(
            (n_shuffles_ripple, n_r, n_cells), np.nan, dtype=np.float32
        )

    def _alloc_metrics_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r2 = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
        mae = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
        ll = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
        return r2, mae, ll

    r2_r_f, mae_r_f, ll_r_f = _alloc_metrics_2d()
    devexp_r_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)

    # NEW: bits/spike arrays (real)
    bps_r_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)

    r2_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
    mae_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
    ll_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
    devexp_r_sh = np.full(
        (n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32
    )

    # NEW: bits/spike arrays (shuffles)
    bps_r_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)

    fold_info: list[dict] = []

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

        r2_r_f[fold] = mcfadden_pseudo_r2_per_neuron(y_test_r, lam_r, y_train_r).astype(
            np.float32
        )
        mae_r_f[fold] = mae_per_neuron(y_test_r, lam_r).astype(np.float32)
        ll_r_f[fold] = poisson_ll_per_neuron(y_test_r, lam_r).astype(np.float32)
        devexp_r_f[fold] = deviance_explained_per_neuron(
            y_test=y_test_r,
            lam_test=lam_r,
            y_null_fit=y_train_r,
        ).astype(np.float32)

        # NEW: bits/spike per fold per cell (vs constant-rate null fit on y_train_r)
        bps_r_f[fold] = bits_per_spike_per_neuron(
            y_test=y_test_r, lam_test=lam_r, y_null_fit=y_train_r
        ).astype(np.float32)

        for s in range(n_shuffles_ripple):
            y_train_sh = shuffle_time_per_neuron(y_train_r, rng)

            glm_s = nmo.glm.PopulationGLM(
                solver_name="LBFGS",
                regularizer="Ridge",
                regularizer_strength=float(ridge_strength),
                solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
            )
            glm_s.fit(X_train_pp, y_train_sh)

            lam_r_s = np.asarray(glm_s.predict(X_test_r_pp), dtype=np.float64)

            if y_r_hat_shuff is not None:
                y_r_hat_shuff[s, test_idx] = lam_r_s.astype(np.float32)

            r2_r_sh[fold, s] = mcfadden_pseudo_r2_per_neuron(
                y_test_r, lam_r_s, y_train_r
            ).astype(np.float32)
            mae_r_sh[fold, s] = mae_per_neuron(y_test_r, lam_r_s).astype(np.float32)
            ll_r_sh[fold, s] = poisson_ll_per_neuron(y_test_r, lam_r_s).astype(
                np.float32
            )
            devexp_r_sh[fold, s] = deviance_explained_per_neuron(
                y_test=y_test_r,
                lam_test=lam_r_s,
                y_null_fit=y_train_r,
            ).astype(np.float32)

            # NEW: bits/spike for shuffle model (vs same train-fold null)
            bps_r_sh[fold, s] = bits_per_spike_per_neuron(
                y_test=y_test_r, lam_test=lam_r_s, y_null_fit=y_train_r
            ).astype(np.float32)

            del glm_s, y_train_sh, lam_r_s
            if (s + 1) % 5 == 0:
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

    # sanity checks for full test/pred arrays
    for name, arr in [("y_r_test_full", y_r_test_full), ("y_r_hat_full", y_r_hat_full)]:
        if np.isnan(arr).any():
            raise ValueError(
                f"{name} has NaNs: some rows were never assigned to a test fold"
            )

    if y_r_hat_shuff is not None and np.isnan(y_r_hat_shuff).any():
        raise ValueError(
            "y_r_hat_shuff has NaNs: some rows were never assigned to a test fold"
        )

    # =========================
    # 2) PRE: train on ALL ripples, test on ALL pre (NO shuffles)
    # =========================
    X_r_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(
        X_train=X_r,
        X_test=X_r,  # unused
    )

    if X_r_all_pp.shape[1] == 0:
        raise ValueError(
            "All X features were near-constant after filtering; cannot fit GLM."
        )

    X_p_k = X_p[:, keep_x_all]
    X_p_pp = (X_p_k - mean_all[None, :]) / (std_all[None, :] + 1e-8)
    X_p_pp /= np.sqrt(max(X_r_all_pp.shape[1], 1))
    X_p_pp = np.clip(X_p_pp, -10.0, 10.0)

    glm_all = nmo.glm.PopulationGLM(
        solver_name="LBFGS",
        regularizer="Ridge",
        regularizer_strength=float(ridge_strength),
        solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
    )
    glm_all.fit(X_r_all_pp, y_r)

    lam_pre = np.asarray(glm_all.predict(X_p_pp), dtype=np.float64)  # (n_p, n_cells)
    lam_pre_f32 = lam_pre.astype(np.float32)

    # PRE metrics (single model); null baseline uses PRE mean (y_null_fit=y_p)
    r2_pre_cell = mcfadden_pseudo_r2_per_neuron(y_p, lam_pre, y_p).astype(np.float32)
    mae_pre_cell = mae_per_neuron(y_p, lam_pre).astype(np.float32)
    ll_pre_cell = poisson_ll_per_neuron(y_p, lam_pre).astype(np.float32)
    devexp_pre_cell = deviance_explained_per_neuron(
        y_test=y_p,
        lam_test=lam_pre,
        y_null_fit=y_p,
    ).astype(np.float32)

    # NEW: bits/spike on PRE (vs PRE constant-rate null)
    bps_pre_cell = bits_per_spike_per_neuron(
        y_test=y_p, lam_test=lam_pre, y_null_fit=y_p
    ).astype(np.float32)

    del glm_all
    jax.clear_caches()
    gc.collect()

    results = dict(
        epoch=epoch,
        random_seed=int(random_seed),
        min_spikes_per_ripple=float(min_spikes_per_ripple),
        n_splits=int(n_splits),
        n_shuffles_ripple=int(n_shuffles_ripple),
        ripple_window=None if ripple_window is None else float(ripple_window),
        pre_window_s=None if pre_window_s is None else float(pre_window_s),
        pre_buffer_s=float(pre_buffer_s),
        pre_exclude_guard_s=float(pre_exclude_guard_s),
        exclude_ripples=exclude_ripples,
        n_ripples=int(n_r),
        n_pre=int(n_p),
        n_cells=int(n_cells),
        v1_unit_ids=v1_unit_ids_kept,
        ca1_unit_ids=ca1_unit_ids,
        fold_info=fold_info,
        # --- RIPPLE (proper CV) ---
        pseudo_r2_ripple_folds=r2_r_f,
        mae_ripple_folds=mae_r_f,
        ll_ripple_folds=ll_r_f,
        devexp_ripple_folds=devexp_r_f,
        # NEW: bits/spike (real)
        bits_per_spike_ripple_folds=bps_r_f,
        pseudo_r2_ripple_shuff_folds=r2_r_sh,
        mae_ripple_shuff_folds=mae_r_sh,
        ll_ripple_shuff_folds=ll_r_sh,
        devexp_ripple_shuff_folds=devexp_r_sh,
        # NEW: bits/spike (shuffles)
        bits_per_spike_ripple_shuff_folds=bps_r_sh,
        y_ripple_test=y_r_test_full,
        yhat_ripple=y_r_hat_full,
        yhat_ripple_shuff=y_r_hat_shuff,  # (S_r, n_r, n_cells) or None
        # --- PRE (single fit on all ripple, single test on all pre) ---
        y_pre_test=y_p.astype(np.float32),
        yhat_pre=lam_pre_f32,
        pseudo_r2_pre=r2_pre_cell,
        mae_pre=mae_pre_cell,
        ll_pre=ll_pre_cell,
        devexp_pre=devexp_pre_cell,
        # NEW: bits/spike (PRE)
        bits_per_spike_pre=bps_pre_cell,
        # preprocessing used for ALL-ripple fit
        keep_x_all=keep_x_all,
        x_mean_all=mean_all,
        x_std_all=std_all,
    )
    return results


def fit_ripple_glm_ridge_sweep_no_shuffles(
    epoch: str,
    *,
    spikes: Dict[str, nap.TsGroup],
    regions: Tuple[str, str] = ("v1", "ca1"),
    Kay_ripple_detector: Dict[str, Any],
    ep: Dict[str, nap.IntervalSet],
    ridge_strengths: Sequence[float],
    min_spikes_per_ripple: float = 0.05,
    random_seed: int = 0,
    ripple_window: Optional[float] = None,
    pre_window_s: Optional[float] = None,
    pre_buffer_s: float = 0.02,
    exclude_ripples: bool = True,
    pre_exclude_guard_s: float = 0.05,
    n_splits: int = 5,
    maxiter: int = 1500,
    tol: float = 1e-6,
    warm_start: bool = True,
    sort_ridges_for_warm_start: bool = True,
    dtype_X: np.dtype = np.float32,
) -> dict:
    """
    Fast ridge sweep (no shuffles) for the ripple->V1 PopulationGLM.

    Computes, for each ridge value:
      - RIPPLE CV metrics per fold per cell
      - PRE metrics (train on all ripples, test on all pre windows) per cell

    Metrics:
      - pseudo R2 (McFadden, constant-rate null fit on train)
      - deviance explained (Poisson)
      - held-out Poisson log-likelihood (sum over bins)
      - bits/spike (info gain vs null)

    Notes on efficiency:
      - counts are computed once
      - folds are fixed once
      - preprocessing is fit+applied once per fold and reused for all ridges
      - null / sat terms are precomputed per fold so metrics are cheap once LL_model is computed
      - optional warm-start across ridge values (with safe fallback if not supported)
    """
    v1_region, ca1_region = regions

    ridge_strengths_in = np.asarray(list(ridge_strengths), dtype=float)
    if ridge_strengths_in.ndim != 1 or ridge_strengths_in.size == 0:
        raise ValueError("ridge_strengths must be a non-empty 1D sequence of floats.")

    # Optionally run ridges in descending order for better warm-start behavior,
    # but return results in the user-provided order.
    if warm_start and sort_ridges_for_warm_start:
        order = np.argsort(ridge_strengths_in)[::-1]
    else:
        order = np.arange(ridge_strengths_in.size)

    ridge_run = ridge_strengths_in[order]
    inv_order = np.argsort(order)

    # --- Build ripple IntervalSet ---
    if ripple_window is None:
        ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"],
            end=Kay_ripple_detector[epoch]["end_time"],
        )
    else:
        ripple_start = np.asarray(Kay_ripple_detector[epoch]["start_time"], dtype=float)
        ripple_ep = nap.IntervalSet(
            start=ripple_start,
            end=ripple_start + float(ripple_window),
        )

    # --- Build pre IntervalSet ---
    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=ep[epoch],
        window_s=pre_window_s,
        buffer_s=pre_buffer_s,
        exclude_ripples=exclude_ripples,
        exclude_ripple_guard_s=pre_exclude_guard_s,
    )

    # --- Count spikes ONCE ---
    X_r = np.asarray(spikes[ca1_region].count(ep=ripple_ep), dtype=np.float64)
    y_r = np.asarray(spikes[v1_region].count(ep=ripple_ep), dtype=np.float64)

    X_p = np.asarray(spikes[ca1_region].count(ep=pre_ep), dtype=np.float64)
    y_p = np.asarray(spikes[v1_region].count(ep=pre_ep), dtype=np.float64)

    n_r, _ = X_r.shape
    n_p = X_p.shape[0]

    if n_r < n_splits:
        raise ValueError(f"Not enough ripples: n_ripples={n_r}, n_splits={n_splits}")

    for name, arr in [("X_r", X_r), ("y_r", y_r), ("X_p", X_p), ("y_p", y_p)]:
        info = _first_nonfinite_info(arr)
        if info:
            raise ValueError(f"Non-finite in {name}: {info}")

    # --- Filter V1 neurons based on ripple data only ---
    keep_y = (y_r.sum(axis=0) / max(n_r, 1)) >= float(min_spikes_per_ripple)
    y_r = y_r[:, keep_y]
    y_p = y_p[:, keep_y]

    v1_unit_ids = np.array(list(spikes[v1_region].keys()))
    ca1_unit_ids = np.array(list(spikes[ca1_region].keys()))
    v1_unit_ids_kept = v1_unit_ids[keep_y]

    n_cells = y_r.shape[1]
    if n_cells == 0:
        raise ValueError(
            f"No V1 units passed min_spikes_per_ripple={min_spikes_per_ripple}. n_ripples={n_r}"
        )

    # =========================
    # CV split + per-fold caches
    # =========================
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    # Determine if nemos supports warm-start via init_params
    _fit_accepts_init = False
    try:
        _fit_sig = inspect.signature(nmo.glm.PopulationGLM.fit)
        _fit_accepts_init = "init_params" in _fit_sig.parameters
    except Exception:
        _fit_accepts_init = False

    def _extract_init_params(glm_obj) -> Optional[Any]:
        """
        Best-effort extraction of parameters for warm-start.
        Works across a range of possible attribute conventions.
        """
        # 1) direct params_ (most robust if it exists)
        if hasattr(glm_obj, "params_"):
            return getattr(glm_obj, "params_")

        # 2) common scikit-ish names
        coef = None
        intercept = None

        for name in ("coef_", "coef", "weights_", "W_", "W", "beta_", "beta"):
            if hasattr(glm_obj, name):
                coef = getattr(glm_obj, name)
                break

        for name in ("intercept_", "intercept", "bias_", "b_", "b", "bias"):
            if hasattr(glm_obj, name):
                intercept = getattr(glm_obj, name)
                break

        if coef is None or intercept is None:
            return None

        return (coef, intercept)

    def _glm_fit(glm_obj, Xtr, ytr, init_params):
        if warm_start and _fit_accepts_init and (init_params is not None):
            try:
                glm_obj.fit(Xtr, ytr, init_params=init_params)
                return True
            except TypeError:
                # signature mismatch or init_params rejected
                glm_obj.fit(Xtr, ytr)
                return False
        else:
            glm_obj.fit(Xtr, ytr)
            return False

    # Fold caches
    fold_info: list[dict] = []
    fold_cache: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_r)):
        X_train_r = X_r[train_idx]
        X_test_r = X_r[test_idx]
        y_train_r = y_r[train_idx]
        y_test_r = y_r[test_idx]

        X_train_pp, X_test_pp, keep_x, mean, std = _preprocess_X_fit_apply(
            X_train=X_train_r,
            X_test=X_test_r,
        )

        if X_train_pp.shape[1] == 0:
            raise ValueError(
                f"Fold {fold}: all X features were near-constant after filtering; cannot fit GLM."
            )

        # Cast X for speed (typical JAX default)
        X_train_pp = np.asarray(X_train_pp, dtype=dtype_X)
        X_test_pp = np.asarray(X_test_pp, dtype=dtype_X)

        # Precompute fold constants for fast metrics
        y_test = np.asarray(y_test_r, dtype=np.float64)
        y_train = np.asarray(y_train_r, dtype=np.float64)

        n_test = y_test.shape[0]

        # per-cell sums
        y_test_sum = np.sum(y_test, axis=0)  # (n_cells,)
        fact_sum = np.sum(gammaln(y_test + 1.0), axis=0)  # (n_cells,)

        lam0 = np.mean(y_train, axis=0)
        lam0 = np.clip(lam0, 1e-12, None)

        # LL_null per cell (constant-rate null fit on y_train, evaluated on y_test)
        ll_null = y_test_sum * np.log(lam0) - n_test * lam0 - fact_sum
        ll_null = np.where(np.abs(ll_null) < 1e-12, -1e-12, ll_null)

        # LL_sat per cell (saturated model on y_test)
        y_safe = np.clip(y_test, 1e-12, None)
        ll_sat = np.sum(y_test * np.log(y_safe) - y_test, axis=0) - fact_sum

        denom_dev = ll_sat - ll_null
        denom_dev = np.where(np.abs(denom_dev) < 1e-12, np.nan, denom_dev)

        denom_bps = y_test_sum * np.log(2.0)
        denom_bps = np.where(y_test_sum > 0, denom_bps, np.nan)

        fold_cache.append(
            dict(
                X_train_pp=X_train_pp,
                X_test_pp=X_test_pp,
                y_train=y_train,  # needed for fitting
                y_test=y_test,  # needed for LL_model term y*log(lam)-lam
                fact_sum=fact_sum,  # constant term in LL
                ll_null=ll_null,
                ll_sat=ll_sat,
                denom_dev=denom_dev,
                denom_bps=denom_bps,
                n_test=n_test,
            )
        )

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

    # =========================
    # PRE preprocessing cache (train on all ripples; test on all pre)
    # =========================
    X_r_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(
        X_train=X_r,
        X_test=X_r,  # unused
    )
    if X_r_all_pp.shape[1] == 0:
        raise ValueError(
            "All X features were near-constant after filtering; cannot fit GLM."
        )

    X_r_all_pp = np.asarray(X_r_all_pp, dtype=dtype_X)

    X_p_k = X_p[:, keep_x_all]
    X_p_pp = (X_p_k - mean_all[None, :]) / (std_all[None, :] + 1e-8)
    X_p_pp /= np.sqrt(max(X_r_all_pp.shape[1], 1))
    X_p_pp = np.clip(X_p_pp, -10.0, 10.0)
    X_p_pp = np.asarray(X_p_pp, dtype=dtype_X)

    # PRE constants for metrics (null fit on y_p itself, as in your earlier code)
    y_pre = np.asarray(y_p, dtype=np.float64)
    n_pre = y_pre.shape[0]
    y_pre_sum = np.sum(y_pre, axis=0)
    fact_sum_pre = np.sum(gammaln(y_pre + 1.0), axis=0)

    lam0_pre = np.mean(y_pre, axis=0)
    lam0_pre = np.clip(lam0_pre, 1e-12, None)

    ll_null_pre = y_pre_sum * np.log(lam0_pre) - n_pre * lam0_pre - fact_sum_pre
    ll_null_pre = np.where(np.abs(ll_null_pre) < 1e-12, -1e-12, ll_null_pre)

    y_pre_safe = np.clip(y_pre, 1e-12, None)
    ll_sat_pre = np.sum(y_pre * np.log(y_pre_safe) - y_pre, axis=0) - fact_sum_pre

    denom_dev_pre = ll_sat_pre - ll_null_pre
    denom_dev_pre = np.where(np.abs(denom_dev_pre) < 1e-12, np.nan, denom_dev_pre)

    denom_bps_pre = y_pre_sum * np.log(2.0)
    denom_bps_pre = np.where(y_pre_sum > 0, denom_bps_pre, np.nan)

    # =========================
    # Allocate sweep outputs (in run order)
    # =========================
    R = ridge_run.size
    F = n_splits
    N = n_cells

    # RIPPLE CV: (R, F, N)
    pseudo_r2_ripple = np.full((R, F, N), np.nan, dtype=np.float32)
    devexp_ripple = np.full((R, F, N), np.nan, dtype=np.float32)
    ll_ripple = np.full((R, F, N), np.nan, dtype=np.float32)
    bps_ripple = np.full((R, F, N), np.nan, dtype=np.float32)

    # PRE: (R, N)
    pseudo_r2_pre = np.full((R, N), np.nan, dtype=np.float32)
    devexp_pre = np.full((R, N), np.nan, dtype=np.float32)
    ll_pre = np.full((R, N), np.nan, dtype=np.float32)
    bps_pre = np.full((R, N), np.nan, dtype=np.float32)

    # Warm-start state
    init_params_folds: list[Optional[Any]] = [None] * F
    init_params_all: Optional[Any] = None

    # =========================
    # Ridge sweep
    # =========================
    for r_idx, ridge in enumerate(ridge_run):
        ridge = float(ridge)

        # ---- RIPPLE CV fits ----
        for fold in range(F):
            fc = fold_cache[fold]

            glm = nmo.glm.PopulationGLM(
                solver_name="LBFGS",
                regularizer="Ridge",
                regularizer_strength=ridge,
                solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
            )

            _glm_fit(glm, fc["X_train_pp"], fc["y_train"], init_params_folds[fold])

            # extract params for next ridge (best effort)
            if warm_start and _fit_accepts_init:
                init_params_folds[fold] = _extract_init_params(glm)

            lam = np.asarray(glm.predict(fc["X_test_pp"]), dtype=np.float64)
            lam = np.clip(lam, 1e-12, None)

            # LL_model per cell (avoid gammaln each time via cached fact_sum)
            y_test = fc["y_test"]
            ll_model = np.sum(y_test * np.log(lam) - lam, axis=0) - fc["fact_sum"]

            ll_null = fc["ll_null"]
            denom_dev = fc["denom_dev"]
            denom_bps = fc["denom_bps"]

            r2 = 1.0 - (ll_model / ll_null)
            de = (ll_model - ll_null) / denom_dev
            bps = (ll_model - ll_null) / denom_bps

            pseudo_r2_ripple[r_idx, fold] = r2.astype(np.float32)
            devexp_ripple[r_idx, fold] = de.astype(np.float32)
            ll_ripple[r_idx, fold] = ll_model.astype(np.float32)
            bps_ripple[r_idx, fold] = bps.astype(np.float32)

            del glm, lam, ll_model

        # ---- PRE: fit on ALL ripples; test on ALL pre ----
        glm_all = nmo.glm.PopulationGLM(
            solver_name="LBFGS",
            regularizer="Ridge",
            regularizer_strength=ridge,
            solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
        )

        _glm_fit(glm_all, X_r_all_pp, y_r, init_params_all)

        if warm_start and _fit_accepts_init:
            init_params_all = _extract_init_params(glm_all)

        lam_pre = np.asarray(glm_all.predict(X_p_pp), dtype=np.float64)
        lam_pre = np.clip(lam_pre, 1e-12, None)

        ll_model_pre = np.sum(y_pre * np.log(lam_pre) - lam_pre, axis=0) - fact_sum_pre

        r2_pre = 1.0 - (ll_model_pre / ll_null_pre)
        de_pre = (ll_model_pre - ll_null_pre) / denom_dev_pre
        bps_pre_cell = (ll_model_pre - ll_null_pre) / denom_bps_pre

        pseudo_r2_pre[r_idx] = r2_pre.astype(np.float32)
        devexp_pre[r_idx] = de_pre.astype(np.float32)
        ll_pre[r_idx] = ll_model_pre.astype(np.float32)
        bps_pre[r_idx] = bps_pre_cell.astype(np.float32)

        del glm_all, lam_pre, ll_model_pre

        # Light GC per ridge (no jax.clear_caches here; let JAX reuse compiled code)
        gc.collect()

    # =========================
    # Reorder back to input ridge_strengths order
    # =========================
    pseudo_r2_ripple = pseudo_r2_ripple[inv_order]
    devexp_ripple = devexp_ripple[inv_order]
    ll_ripple = ll_ripple[inv_order]
    bps_ripple = bps_ripple[inv_order]

    pseudo_r2_pre = pseudo_r2_pre[inv_order]
    devexp_pre = devexp_pre[inv_order]
    ll_pre = ll_pre[inv_order]
    bps_pre = bps_pre[inv_order]

    # Convenience summaries
    pseudo_r2_ripple_mean_over_folds = np.nanmean(pseudo_r2_ripple, axis=1)  # (R, N)
    devexp_ripple_mean_over_folds = np.nanmean(devexp_ripple, axis=1)
    ll_ripple_mean_over_folds = np.nanmean(ll_ripple, axis=1)
    bps_ripple_mean_over_folds = np.nanmean(bps_ripple, axis=1)

    pseudo_r2_ripple_pop_mean = np.nanmean(
        pseudo_r2_ripple_mean_over_folds, axis=1
    )  # (R,)
    devexp_ripple_pop_mean = np.nanmean(devexp_ripple_mean_over_folds, axis=1)
    ll_ripple_pop_mean = np.nanmean(ll_ripple_mean_over_folds, axis=1)
    bps_ripple_pop_mean = np.nanmean(bps_ripple_mean_over_folds, axis=1)

    results = dict(
        epoch=epoch,
        random_seed=int(random_seed),
        ridge_strengths=ridge_strengths_in.astype(np.float64),
        ridge_sweep_warm_start=bool(warm_start and _fit_accepts_init),
        ridge_sweep_sort_desc_for_warm_start=bool(
            warm_start and sort_ridges_for_warm_start
        ),
        min_spikes_per_ripple=float(min_spikes_per_ripple),
        n_splits=int(n_splits),
        ripple_window=None if ripple_window is None else float(ripple_window),
        pre_window_s=None if pre_window_s is None else float(pre_window_s),
        pre_buffer_s=float(pre_buffer_s),
        pre_exclude_guard_s=float(pre_exclude_guard_s),
        exclude_ripples=bool(exclude_ripples),
        n_ripples=int(n_r),
        n_pre=int(n_p),
        n_cells=int(n_cells),
        v1_unit_ids=v1_unit_ids_kept,
        ca1_unit_ids=ca1_unit_ids,
        fold_info=fold_info,
        keep_x_all=keep_x_all,
        x_mean_all=mean_all,
        x_std_all=std_all,
        # ----- RIPPLE CV (R, F, N) -----
        pseudo_r2_ripple_folds=pseudo_r2_ripple,
        devexp_ripple_folds=devexp_ripple,
        ll_ripple_folds=ll_ripple,
        bits_per_spike_ripple_folds=bps_ripple,
        # convenience summaries
        pseudo_r2_ripple_mean_over_folds=pseudo_r2_ripple_mean_over_folds,
        devexp_ripple_mean_over_folds=devexp_ripple_mean_over_folds,
        ll_ripple_mean_over_folds=ll_ripple_mean_over_folds,
        bits_per_spike_ripple_mean_over_folds=bps_ripple_mean_over_folds,
        pseudo_r2_ripple_pop_mean=pseudo_r2_ripple_pop_mean,
        devexp_ripple_pop_mean=devexp_ripple_pop_mean,
        ll_ripple_pop_mean=ll_ripple_pop_mean,
        bits_per_spike_ripple_pop_mean=bps_ripple_pop_mean,
        # ----- PRE (R, N) -----
        pseudo_r2_pre=pseudo_r2_pre,
        devexp_pre=devexp_pre,
        ll_pre=ll_pre,
        bits_per_spike_pre=bps_pre,
    )
    return results


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = (np.arange(x.size) + 1) / max(x.size, 1)
    return x, y


def _as_2d_float(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        return np.stack(arr.tolist(), axis=0).astype(float)
    return np.asarray(arr, dtype=float)


def _as_3d_float(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        return np.stack(arr.tolist(), axis=0).astype(float)
    return np.asarray(arr, dtype=float)


def plot_ripple_glm_pseudo_r2(
    *,
    pseudo_r2_real_folds: Any,
    pseudo_r2_shuff_folds: Any,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.7,
    null_mode: Literal["fold_mean", "pooled"] = "fold_mean",
    # "fold_mean" is recommended when the same test set is reused each fold (your PRE case).
    # "pooled" reproduces your prior behavior.
) -> None:
    """
    Plot pseudo-R2-focused panels and save as a single figure.

    Parameters
    ----------
    pseudo_r2_real_folds : array-like, shape (F, N)
        Real pseudo-R2 per fold per cell.
    pseudo_r2_shuff_folds : array-like, shape (F, S, N)
        Shuffle pseudo-R2 per fold per shuffle per cell.
    null_mode : {"fold_mean", "pooled"}
        How to build the null distribution used for p-values/ECDF/panels.
        - "fold_mean": first average across folds -> per-shuffle statistic (S, N).
          This treats folds as repeated trainings, not independent tests. Recommended for PRE.
        - "pooled": pool across folds and shuffles -> (F*S, N). Matches your old behavior.

    Notes
    -----
    For your Option-1 PRE evaluation, folds are not independent test sets. Use "fold_mean".
    For RIPPLE CV, either can be used, but "fold_mean" is still a conservative choice.
    """
    real_f = _as_2d_float(pseudo_r2_real_folds)  # (F, N)
    shuf_f = _as_3d_float(pseudo_r2_shuff_folds)  # (F, S, N)

    if real_f.ndim != 2:
        raise ValueError(f"pseudo_r2_real_folds must be 2D (F,N), got {real_f.shape}")
    if shuf_f.ndim != 3:
        raise ValueError(
            f"pseudo_r2_shuff_folds must be 3D (F,S,N), got {shuf_f.shape}"
        )
    if real_f.shape[0] != shuf_f.shape[0] or real_f.shape[1] != shuf_f.shape[2]:
        raise ValueError(
            f"Shape mismatch: real {real_f.shape} vs shuff {shuf_f.shape} (expected (F,N) and (F,S,N))"
        )

    F, N = real_f.shape
    S = shuf_f.shape[1]

    # Effect size per cell (mean across folds)
    real_cell = np.mean(real_f, axis=0)  # (N,)

    # Build per-cell null samples depending on mode
    if null_mode == "fold_mean":
        # Average across folds first, yielding one sample per shuffle:
        # null_cell_samples: (S, N)
        null_cell_samples = np.mean(shuf_f, axis=0)
        # Population mean null distribution: (S,)
        null_pop = np.mean(null_cell_samples, axis=1)
    elif null_mode == "pooled":
        # Pool folds and shuffles:
        # null_cell_samples: (F*S, N)
        null_cell_samples = shuf_f.reshape(F * S, N)
        null_pop = np.mean(null_cell_samples, axis=1)  # (F*S,)
    else:
        raise ValueError("null_mode must be 'fold_mean' or 'pooled'")

    # Per-cell z-score vs null
    null_mu = np.mean(null_cell_samples, axis=0)
    null_sd = np.std(null_cell_samples, axis=0) + 1e-12
    z_per_cell = (real_cell - null_mu) / null_sd

    # One-sided p-values: P(null >= observed)
    n_null = null_cell_samples.shape[0]
    p_per_cell = (np.sum(null_cell_samples >= real_cell[None, :], axis=0) + 1) / (
        n_null + 1
    )
    neglog10_p = -np.log10(p_per_cell)

    # ECDF: compare observed per-cell distribution to pooled null values
    null_flat = null_cell_samples.ravel()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # A: per-cell pseudo-R2 distribution
    ax = axes[0]
    ax.hist(
        real_cell,
        bins=np.linspace(-0.4, 0.4, 81),
        weights=np.ones_like(real_cell) / max(len(real_cell), 1),
        alpha=alpha,
    )
    ax.set_xlabel("Pseudo $R^2$ McFadden (mean over folds)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell pseudo-$R^2$")

    stats_text = (
        f"Mean:   {np.mean(real_cell):.4f}\n"
        f"Median: {np.median(real_cell):.4f}\n"
        f"25%:    {np.percentile(real_cell, 25):.4f}\n"
        f"75%:    {np.percentile(real_cell, 75):.4f}\n"
        f"Folds:  {F} | Null: {null_mode}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    # B: null distribution of population mean
    ax = axes[1]
    ax.hist(null_pop, alpha=0.75, bins=40)
    ax.axvline(np.mean(real_cell), linewidth=2)
    ax.set_xlabel("Population mean pseudo $R^2$")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: population mean")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(real_cell):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # D: ECDF
    ax = axes[2]
    xr, yr = _ecdf(real_cell)
    xs, ys = _ecdf(null_flat)
    ax.plot(xr, yr, linewidth=2, label="Real (cells)")
    ax.plot(xs, ys, linewidth=2, label="Null (pooled)", alpha=0.8)
    ax.set_xlabel("Pseudo $R^2$")
    ax.set_ylabel("ECDF")
    ax.set_title("D. ECDF: real vs null")
    ax.legend(frameon=False)

    # E: z-score histogram
    ax = axes[3]
    ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score vs null (per cell)")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized effect size")

    # F: effect vs significance
    ax = axes[4]
    ax.scatter(real_cell, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.set_xlabel("Pseudo $R^2$ (effect size)")
    ax.set_ylabel(r"$-\log_{10}(p)$ (shuffle)")
    ax.set_title("F. Effect size vs significance")
    frac_sig = float(np.mean(p_per_cell < 0.05))
    ax.text(
        0.98,
        0.05,
        f"frac p<0.05 = {frac_sig:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # Notes
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Notes:\n"
        "- Effect size is per-cell mean over folds.\n"
        "- p-values are one-sided: P(null ≥ observed).\n"
        "- null_mode='fold_mean' is recommended when the same test set is reused each fold (PRE).\n"
        "- ECDF uses pooled null values for visualization.\n",
        ha="left",
        va="top",
        fontsize=10,
        transform=axes[5].transAxes,
    )

    fig.suptitle(
        f"{animal_name} {date} {epoch} — Pseudo-$R^2$ summary",
        fontsize=14,
        y=1.02,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ripple_glm_mae(
    *,
    mae_real_folds: Any,
    mae_shuff_folds: Any,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.7,
    null_mode: Literal["fold_mean", "pooled"] = "fold_mean",
    # "fold_mean" recommended when the same test set is reused each fold (your PRE case).
    # "pooled" reproduces fold+shuffle pooling.
    bins: int = 60,
    clip_percentile: float = 99.5,
) -> None:
    """
    Plot MAE-focused panels and save as a single figure.

    Parameters
    ----------
    mae_real_folds : array-like, shape (F, N)
        Real MAE per fold per cell.
    mae_shuff_folds : array-like, shape (F, S, N)
        Shuffle MAE per fold per shuffle per cell.
    null_mode : {"fold_mean", "pooled"}
        How to build the null distribution used for p-values/ECDF/panels.
        - "fold_mean": average across folds first -> per-shuffle statistic (S, N).
          Recommended when folds reuse the same test set (PRE).
        - "pooled": pool folds and shuffles -> (F*S, N).
    bins : int
        Number of histogram bins.
    clip_percentile : float
        Upper percentile used to set x-limits for histograms/ECDF for readability.

    Notes
    -----
    For MAE, smaller is better. P-values are one-sided: P(null <= observed),
    i.e. probability that the null achieves MAE at least as small as observed.
    """
    real_f = _as_2d_float(mae_real_folds)  # (F, N)
    shuf_f = _as_3d_float(mae_shuff_folds)  # (F, S, N)

    if real_f.ndim != 2:
        raise ValueError(f"mae_real_folds must be 2D (F,N), got {real_f.shape}")
    if shuf_f.ndim != 3:
        raise ValueError(f"mae_shuff_folds must be 3D (F,S,N), got {shuf_f.shape}")
    if real_f.shape[0] != shuf_f.shape[0] or real_f.shape[1] != shuf_f.shape[2]:
        raise ValueError(
            f"Shape mismatch: real {real_f.shape} vs shuff {shuf_f.shape} (expected (F,N) and (F,S,N))"
        )

    F, N = real_f.shape
    S = shuf_f.shape[1]

    # Effect size per cell: mean MAE over folds
    real_cell = np.mean(real_f, axis=0)  # (N,)

    # Build per-cell null samples
    if null_mode == "fold_mean":
        # (S, N)
        null_cell_samples = np.mean(shuf_f, axis=0)
        # population mean null distribution (S,)
        null_pop = np.mean(null_cell_samples, axis=1)
    elif null_mode == "pooled":
        # (F*S, N)
        null_cell_samples = shuf_f.reshape(F * S, N)
        # (F*S,)
        null_pop = np.mean(null_cell_samples, axis=1)
    else:
        raise ValueError("null_mode must be 'fold_mean' or 'pooled'")

    # Per-cell z-score: how many SDs *below* null mean (positive is better)
    null_mu = np.mean(null_cell_samples, axis=0)
    null_sd = np.std(null_cell_samples, axis=0) + 1e-12
    z_improve = (null_mu - real_cell) / null_sd

    # One-sided p-values for improvement: P(null <= observed) (smaller MAE is better)
    n_null = null_cell_samples.shape[0]
    p_per_cell = (np.sum(null_cell_samples <= real_cell[None, :], axis=0) + 1) / (
        n_null + 1
    )
    neglog10_p = -np.log10(p_per_cell)

    # ECDF uses pooled null values for visualization
    null_flat = null_cell_samples.ravel()

    # Robust x-limits for readability
    finite_all = np.concatenate(
        [real_cell[np.isfinite(real_cell)], null_flat[np.isfinite(null_flat)]]
    )
    if finite_all.size == 0:
        raise ValueError("No finite MAE values to plot.")
    x_max = np.percentile(finite_all, clip_percentile)
    x_max = max(float(x_max), 1e-12)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # A: per-cell MAE distribution
    ax = axes[0]
    ax.hist(
        np.clip(real_cell, 0.0, x_max),
        bins=bins,
        weights=np.ones_like(real_cell) / max(len(real_cell), 1),
        alpha=alpha,
    )
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("MAE per cell (mean over folds)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell MAE")

    stats_text = (
        f"Mean:   {np.mean(real_cell):.4g}\n"
        f"Median: {np.median(real_cell):.4g}\n"
        f"25%:    {np.percentile(real_cell, 25):.4g}\n"
        f"75%:    {np.percentile(real_cell, 75):.4g}\n"
        f"Folds:  {F} | Null: {null_mode}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    # B: null distribution of population mean MAE
    ax = axes[1]
    ax.hist(np.clip(null_pop, 0.0, x_max), alpha=0.75, bins=40)
    ax.axvline(np.mean(real_cell), linewidth=2)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Population mean MAE")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: population mean MAE")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(real_cell):.4g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # D: ECDF
    ax = axes[2]
    xr, yr = _ecdf(real_cell)
    xs, ys = _ecdf(null_flat)
    ax.plot(xr, yr, linewidth=2, label="Real (cells)")
    ax.plot(xs, ys, linewidth=2, label="Null (pooled)", alpha=0.8)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("MAE")
    ax.set_ylabel("ECDF")
    ax.set_title("D. ECDF: real vs null")
    ax.legend(frameon=False)

    # E: z-score histogram (improvement)
    ax = axes[3]
    ax.hist(z_improve[np.isfinite(z_improve)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score improvement (null mean - real) / null std")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized improvement")

    # F: effect vs significance
    ax = axes[4]
    ax.scatter(real_cell, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("MAE (effect size; smaller is better)")
    ax.set_ylabel("-log10(p) where p = P(null ≤ real)")
    ax.set_title("F. Effect size vs significance")
    frac_sig = float(np.mean(p_per_cell < 0.05))
    ax.text(
        0.98,
        0.05,
        f"frac p<0.05 = {frac_sig:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # Notes
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Notes:\n"
        "- Smaller MAE is better.\n"
        "- p-values are one-sided: P(null MAE ≤ real MAE) (null does as well or better).\n"
        "- z-score is defined so positive means improvement (real < null mean).\n"
        "- null_mode='fold_mean' is recommended when folds reuse the same test set (PRE).\n",
        ha="left",
        va="top",
        fontsize=10,
        transform=axes[5].transAxes,
    )

    fig.suptitle(
        f"{animal_name} {date} {epoch} — MAE summary",
        fontsize=14,
        y=1.02,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ripple_glm_delta_ll(
    *,
    ll_real_folds: Any,
    ll_shuff_folds: Any,
    y_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.7,
    normalize: Literal["none", "bits_per_spike"] = "bits_per_spike",
    null_mode: Literal["fold_mean", "pooled"] = "fold_mean",
    bins: int = 60,
    clip_percentile: float = 99.5,
) -> None:
    """
    Plot log-likelihood improvement (real vs shuffle).

    Parameters
    ----------
    ll_real_folds : array-like, shape (F, N)
        Held-out Poisson LL per fold per cell for the real model.
    ll_shuff_folds : array-like, shape (F, S, N)
        Held-out Poisson LL per fold per cell for each shuffle model.
    y_test : np.ndarray, shape (n_samples, N)
        Observed counts on the evaluation set being plotted (RIPPLE or PRE).
        Used only for bits/spike normalization.
    normalize : {"none", "bits_per_spike"}
        If "bits_per_spike", divides ΔLL by total spikes and ln(2).
    null_mode : {"fold_mean", "pooled"}
        - "fold_mean": average across folds first -> S null draws (recommended when the
          same test set is reused each fold, i.e. your PRE case).
        - "pooled": pool folds and shuffles -> F*S null draws (ok for RIPPLE CV).
    """
    ll_real_f = _as_2d_float(ll_real_folds)  # (F, N)
    ll_shuf_f = _as_3d_float(ll_shuff_folds)  # (F, S, N)

    if ll_real_f.ndim != 2:
        raise ValueError(f"ll_real_folds must be 2D (F,N), got {ll_real_f.shape}")
    if ll_shuf_f.ndim != 3:
        raise ValueError(f"ll_shuff_folds must be 3D (F,S,N), got {ll_shuf_f.shape}")
    if (
        ll_real_f.shape[0] != ll_shuf_f.shape[0]
        or ll_real_f.shape[1] != ll_shuf_f.shape[2]
    ):
        raise ValueError(
            f"Shape mismatch: real {ll_real_f.shape} vs shuff {ll_shuf_f.shape} (expected (F,N) and (F,S,N))"
        )

    F, N = ll_real_f.shape
    S = ll_shuf_f.shape[1]

    # ΔLL per fold, shuffle, cell: (F, S, N)
    # (positive is better)
    delta_ll_fsn = ll_real_f[:, None, :] - ll_shuf_f

    # observed effect per cell: mean across folds of ΔLL vs mean-null-per-fold
    # (this matches your earlier definition but is now explicit)
    delta_ll_obs_cell = np.mean(np.mean(delta_ll_fsn, axis=1), axis=0)  # (N,)

    # build per-cell null samples
    if null_mode == "fold_mean":
        # average across folds -> (S, N)
        delta_ll_null_cell = np.mean(delta_ll_fsn, axis=0)
        # population mean null distribution: (S,)
        delta_ll_null_pop = np.mean(delta_ll_null_cell, axis=1)
    elif null_mode == "pooled":
        # pool folds and shuffles -> (F*S, N)
        delta_ll_null_cell = delta_ll_fsn.reshape(F * S, N)
        delta_ll_null_pop = np.mean(delta_ll_null_cell, axis=1)
    else:
        raise ValueError("null_mode must be 'fold_mean' or 'pooled'")

    # normalization
    if normalize == "bits_per_spike":
        spikes_per_cell = np.sum(y_test, axis=0).astype(np.float64)  # (N,)
        denom = np.clip(spikes_per_cell, 1.0, None) * np.log(2.0)
        delta_ll_obs_cell = delta_ll_obs_cell / denom
        delta_ll_null_cell = delta_ll_null_cell / denom[None, :]
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'bits_per_spike'")

    # z-score per cell (positive = better)
    null_mu = np.mean(delta_ll_null_cell, axis=0)
    null_sd = np.std(delta_ll_null_cell, axis=0) + 1e-12
    z_per_cell = (delta_ll_obs_cell - null_mu) / null_sd

    # one-sided p-values: P(null >= observed)
    n_null = delta_ll_null_cell.shape[0]
    p_per_cell = (
        np.sum(delta_ll_null_cell >= delta_ll_obs_cell[None, :], axis=0) + 1
    ) / (n_null + 1)
    neglog10_p = -np.log10(p_per_cell)

    # pooled for ECDF
    null_flat = delta_ll_null_cell.ravel()

    # robust x-limits
    finite_all = np.concatenate(
        [
            delta_ll_obs_cell[np.isfinite(delta_ll_obs_cell)],
            null_flat[np.isfinite(null_flat)],
        ]
    )
    if finite_all.size == 0:
        raise ValueError("No finite ΔLL values to plot.")
    x_max = float(np.percentile(finite_all, clip_percentile))
    x_min = float(np.percentile(finite_all, 100.0 - clip_percentile))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = float(np.min(finite_all)), float(np.max(finite_all))

    if normalize == "bits_per_spike":
        x_label = "Δ bits/spike (real - shuffle)"
    else:
        x_label = "ΔLL (real - shuffle)"

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # A
    ax = axes[0]
    ax.hist(
        np.clip(delta_ll_obs_cell, x_min, x_max),
        bins=bins,
        weights=np.ones_like(delta_ll_obs_cell) / max(len(delta_ll_obs_cell), 1),
        alpha=alpha,
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"{x_label} per cell (effect; mean over folds)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell improvement")

    stats_text = (
        f"Mean:   {np.mean(delta_ll_obs_cell):.4g}\n"
        f"Median: {np.median(delta_ll_obs_cell):.4g}\n"
        f"25%:    {np.percentile(delta_ll_obs_cell, 25):.4g}\n"
        f"75%:    {np.percentile(delta_ll_obs_cell, 75):.4g}\n"
        f"Folds:  {F} | Null: {null_mode}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    # B
    ax = axes[1]
    ax.hist(np.clip(delta_ll_null_pop, x_min, x_max), alpha=0.75, bins=40)
    ax.axvline(np.mean(delta_ll_obs_cell), linewidth=2)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"Population mean {x_label}")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: population mean improvement")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(delta_ll_obs_cell):.4g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # D
    ax = axes[2]
    xr, yr = _ecdf(delta_ll_obs_cell)
    xs, ys = _ecdf(null_flat)
    ax.plot(xr, yr, linewidth=2, label="Real (cells)")
    ax.plot(xs, ys, linewidth=2, label="Null (pooled)", alpha=0.8)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel("ECDF")
    ax.set_title("D. ECDF: real vs null")
    ax.legend(frameon=False)

    # E
    ax = axes[3]
    ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score vs null (per cell)")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized effect size")

    # F
    ax = axes[4]
    ax.scatter(delta_ll_obs_cell, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle=":", linewidth=1)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"{x_label} (effect size)")
    ax.set_ylabel(r"$-\log_{10}(p)$ (one-sided)")
    ax.set_title("F. Effect size vs significance")
    frac_sig = float(np.mean(p_per_cell < 0.05))
    ax.text(
        0.98,
        0.05,
        f"frac p<0.05 = {frac_sig:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # Notes
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Notes:\n"
        "- Δ = LL(real) - LL(shuffle) on the evaluation set.\n"
        "- p-values are one-sided: P(null ≥ observed).\n"
        "- null_mode='fold_mean' recommended when the same test set is reused each fold (PRE).\n",
        ha="left",
        va="top",
        fontsize=10,
        transform=axes[5].transAxes,
    )

    fig.suptitle(
        f"{animal_name} {date} {epoch} — {x_label} summary",
        fontsize=14,
        y=1.02,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _as_1d_float(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu) / sd


def plot_pre_pseudo_r2_summary(
    *,
    pseudo_r2_pre: Any,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.75,
    r2_xlim: tuple[float, float] = (-0.4, 0.4),
) -> None:
    """
    PRE-only pseudo-R2 summary without shuffles.

    Parameters
    ----------
    pseudo_r2_pre : array-like, shape (N,)
        Per-cell PRE pseudo-R2 values.
    y_pre_test : np.ndarray, shape (n_pre, N)
        PRE observed counts (used for spike totals panel).
    """
    r2 = _as_1d_float(pseudo_r2_pre)
    if y_pre_test.ndim != 2 or y_pre_test.shape[1] != r2.shape[0]:
        raise ValueError(
            f"y_pre_test must be (n_pre, N={r2.shape[0]}), got {y_pre_test.shape}"
        )

    spikes_per_cell = np.sum(y_pre_test, axis=0).astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    # A: histogram
    ax = axes[0]
    bins = np.linspace(r2_xlim[0], r2_xlim[1], 81)
    ax.hist(
        np.clip(r2, r2_xlim[0], r2_xlim[1]),
        bins=bins,
        weights=np.ones_like(r2) / max(len(r2), 1),
        alpha=alpha,
    )
    ax.set_xlim(*r2_xlim)
    ax.set_xlabel("PRE pseudo $R^2$ (McFadden)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell pseudo-$R^2$")

    stats_text = (
        f"N cells: {len(r2)}\n"
        f"Mean:   {np.nanmean(r2):.4f}\n"
        f"Median: {np.nanmedian(r2):.4f}\n"
        f"25%:    {np.nanpercentile(r2, 25):.4f}\n"
        f"75%:    {np.nanpercentile(r2, 75):.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    # B: ECDF
    ax = axes[1]
    xr, yr = _ecdf(r2)
    ax.plot(xr, yr, linewidth=2)
    ax.set_xlim(*r2_xlim)
    ax.set_xlabel("PRE pseudo $R^2$")
    ax.set_ylabel("ECDF")
    ax.set_title("B. ECDF")

    # C: pseudo-R2 vs spikes (diagnostic)
    ax = axes[2]
    ax.scatter(spikes_per_cell, r2, s=14, alpha=0.6)
    ax.set_xlabel("Total PRE spikes per cell")
    ax.set_ylabel("PRE pseudo $R^2$")
    ax.set_title("C. Effect vs spike count")
    ax.set_ylim(*r2_xlim)

    # D: across-cell z-score (descriptive)
    ax = axes[3]
    z = _safe_zscore(r2)
    ax.hist(z[np.isfinite(z)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score across cells")
    ax.set_ylabel("Count")
    ax.set_title("D. Standardized (across-cell)")

    fig.suptitle(
        f"{animal_name} {date} {epoch} — PRE pseudo-$R^2$ summary", fontsize=13, y=1.02
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pre_mae_summary(
    *,
    mae_pre: Any,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.75,
    clip_percentile: float = 99.5,
) -> None:
    """
    PRE-only MAE summary without shuffles.

    Parameters
    ----------
    mae_pre : array-like, shape (N,)
        Per-cell MAE on PRE windows.
    y_pre_test : np.ndarray, shape (n_pre, N)
        PRE observed counts (used for spike totals panel).
    """
    mae = _as_1d_float(mae_pre)
    if y_pre_test.ndim != 2 or y_pre_test.shape[1] != mae.shape[0]:
        raise ValueError(
            f"y_pre_test must be (n_pre, N={mae.shape[0]}), got {y_pre_test.shape}"
        )

    spikes_per_cell = np.sum(y_pre_test, axis=0).astype(float)

    finite = mae[np.isfinite(mae)]
    if finite.size == 0:
        raise ValueError("No finite MAE values to plot.")
    x_max = float(np.nanpercentile(finite, clip_percentile))
    x_max = max(x_max, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    # A: histogram
    ax = axes[0]
    ax.hist(
        np.clip(mae, 0.0, x_max),
        bins=60,
        weights=np.ones_like(mae) / max(len(mae), 1),
        alpha=alpha,
    )
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("PRE MAE per cell")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell MAE")

    stats_text = (
        f"N cells: {len(mae)}\n"
        f"Mean:   {np.nanmean(mae):.4g}\n"
        f"Median: {np.nanmedian(mae):.4g}\n"
        f"25%:    {np.nanpercentile(mae, 25):.4g}\n"
        f"75%:    {np.nanpercentile(mae, 75):.4g}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    # B: ECDF
    ax = axes[1]
    x, y = _ecdf(mae)
    ax.plot(x, y, linewidth=2)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("PRE MAE")
    ax.set_ylabel("ECDF")
    ax.set_title("B. ECDF")

    # C: MAE vs spikes
    ax = axes[2]
    ax.scatter(spikes_per_cell, mae, s=14, alpha=0.6)
    ax.set_xlabel("Total PRE spikes per cell")
    ax.set_ylabel("PRE MAE")
    ax.set_title("C. Error vs spike count")
    ax.set_ylim(0.0, x_max)

    # D: across-cell z-score (descriptive)
    ax = axes[3]
    z = _safe_zscore(mae)
    ax.hist(z[np.isfinite(z)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Z-score across cells")
    ax.set_ylabel("Count")
    ax.set_title("D. Standardized (across-cell)")

    fig.suptitle(f"{animal_name} {date} {epoch} — PRE MAE summary", fontsize=13, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pre_ll_summary(
    *,
    ll_pre: Any,
    y_pre_test: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    normalize: Literal["none", "bits_per_spike"] = "bits_per_spike",
    alpha: float = 0.75,
    clip_percentile: float = 99.5,
) -> None:
    """
    PRE-only Poisson log-likelihood summary without shuffles.

    Parameters
    ----------
    ll_pre : array-like, shape (N,)
        Per-cell Poisson LL on PRE windows (sum over time bins).
    y_pre_test : np.ndarray, shape (n_pre, N)
        PRE observed counts; used for bits/spike normalization and spike totals panel.
    normalize : {"none", "bits_per_spike"}
        If "bits_per_spike", divides LL by total spikes and ln(2).
    """
    ll = _as_1d_float(ll_pre)
    if y_pre_test.ndim != 2 or y_pre_test.shape[1] != ll.shape[0]:
        raise ValueError(
            f"y_pre_test must be (n_pre, N={ll.shape[0]}), got {y_pre_test.shape}"
        )

    spikes_per_cell = np.sum(y_pre_test, axis=0).astype(float)

    if normalize == "bits_per_spike":
        denom = np.clip(spikes_per_cell, 1.0, None) * np.log(2.0)
        ll_plot = ll / denom
        x_label = "Poisson LL (bits/spike)"
    elif normalize == "none":
        ll_plot = ll
        x_label = "Poisson LL (sum over bins)"
    else:
        raise ValueError("normalize must be 'none' or 'bits_per_spike'")

    finite = ll_plot[np.isfinite(ll_plot)]
    if finite.size == 0:
        raise ValueError("No finite LL values to plot.")
    x_hi = float(np.nanpercentile(finite, clip_percentile))
    x_lo = float(np.nanpercentile(finite, 100.0 - clip_percentile))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo == x_hi:
        x_lo, x_hi = float(np.nanmin(finite)), float(np.nanmax(finite))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    # A: histogram
    ax = axes[0]
    ax.hist(
        np.clip(ll_plot, x_lo, x_hi),
        bins=60,
        weights=np.ones_like(ll_plot) / max(len(ll_plot), 1),
        alpha=alpha,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell LL")

    stats_text = (
        f"N cells: {len(ll_plot)}\n"
        f"Mean:   {np.nanmean(ll_plot):.4g}\n"
        f"Median: {np.nanmedian(ll_plot):.4g}\n"
        f"25%:    {np.nanpercentile(ll_plot, 25):.4g}\n"
        f"75%:    {np.nanpercentile(ll_plot, 75):.4g}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    # B: ECDF
    ax = axes[1]
    x, y = _ecdf(ll_plot)
    ax.plot(x, y, linewidth=2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel("ECDF")
    ax.set_title("B. ECDF")

    # C: LL vs spikes
    ax = axes[2]
    ax.scatter(spikes_per_cell, ll_plot, s=14, alpha=0.6)
    ax.set_xlabel("Total PRE spikes per cell")
    ax.set_ylabel(x_label)
    ax.set_title("C. LL vs spike count")
    ax.set_ylim(x_lo, x_hi)

    # D: across-cell z-score (descriptive)
    ax = axes[3]
    z = _safe_zscore(ll_plot)
    ax.hist(z[np.isfinite(z)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Z-score across cells")
    ax.set_ylabel("Count")
    ax.set_title("D. Standardized (across-cell)")

    fig.suptitle(f"{animal_name} {date} {epoch} — PRE LL summary", fontsize=13, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _popmean_sem_over_folds(metric_rfn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    metric_rfn: (R, F, N)  ridge x folds x cells
    Returns:
      mu:  (R,) mean over folds of (mean over cells)
      sem: (R,) SEM across folds of (mean over cells)
    """
    m = np.asarray(metric_rfn, dtype=float)
    if m.ndim != 3:
        raise ValueError(f"Expected metric shape (R,F,N), got {m.shape}")

    # population mean per fold: (R, F)
    pop_rf = np.nanmean(m, axis=2)

    mu = np.nanmean(pop_rf, axis=1)

    n_finite = np.sum(np.isfinite(pop_rf), axis=1).astype(float)
    std = np.nanstd(pop_rf, axis=1, ddof=1)
    sem = std / np.sqrt(np.maximum(n_finite, 1.0))
    sem = np.where(n_finite > 1.0, sem, 0.0)

    return mu, sem


def _select_best_and_1se(
    ridge_sorted: np.ndarray,
    mu_sorted: np.ndarray,
    sem_sorted: np.ndarray,
) -> Dict[str, Any]:
    """
    Select:
      - best: argmax(mu)
      - 1-SE: largest ridge whose mu >= mu_best - sem_best
    """
    ridge_sorted = np.asarray(ridge_sorted, dtype=float)
    mu_sorted = np.asarray(mu_sorted, dtype=float)
    sem_sorted = np.asarray(sem_sorted, dtype=float)

    if ridge_sorted.ndim != 1 or mu_sorted.ndim != 1 or sem_sorted.ndim != 1:
        raise ValueError("ridge_sorted, mu_sorted, sem_sorted must be 1D arrays.")
    if not (ridge_sorted.size == mu_sorted.size == sem_sorted.size):
        raise ValueError("ridge_sorted, mu_sorted, sem_sorted must have same length.")

    finite = np.isfinite(mu_sorted)
    if not np.any(finite):
        raise ValueError("No finite metric values to choose best ridge from.")

    best_idx = int(np.nanargmax(mu_sorted))
    best_ridge = float(ridge_sorted[best_idx])

    thresh = mu_sorted[best_idx] - sem_sorted[best_idx]
    candidates = np.where(mu_sorted >= thresh)[0]
    if candidates.size == 0:
        one_se_idx = best_idx
    else:
        # 1-SE rule: choose largest ridge (most regularized) within 1 SEM of best
        one_se_idx = int(candidates[-1])

    one_se_ridge = float(ridge_sorted[one_se_idx])

    return dict(
        best_idx=best_idx,
        best_ridge=best_ridge,
        one_se_idx=one_se_idx,
        one_se_ridge=one_se_ridge,
        threshold=float(thresh),
    )


def plot_ridge_sweep_summary(
    results: Dict[str, Any],
    *,
    criterion: Literal[
        "bits_per_spike", "devexp", "pseudo_r2", "ll"
    ] = "bits_per_spike",
    include_ll_panel: bool = False,
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_one_se: bool = True,
    figsize: Tuple[float, float] = (12, 9),
) -> Dict[str, Any]:
    """
    Plot ridge sweep curves (RIPPLE CV) and return selected ridge.

    Selection is done on RIPPLE CV using `criterion`, using:
      - best ridge: maximizes mean (pop mean across cells, then mean across folds)
      - 1-SE ridge: largest ridge within 1 SEM of best (SEM across folds)

    Returns dict with:
      - best_ridge, one_se_ridge, best_idx, one_se_idx
      - ridge_sorted, selection_metric_mu, selection_metric_sem
    """
    # --- ridge values ---
    ridge = np.asarray(results["ridge_strengths"], dtype=float)
    if ridge.ndim != 1:
        raise ValueError(f"results['ridge_strengths'] must be 1D, got {ridge.shape}")

    sort_idx = np.argsort(ridge)
    ridge_s = ridge[sort_idx]

    # --- metric key map (RIPPLE CV arrays must be (R,F,N)) ---
    key_map = dict(
        bits_per_spike="bits_per_spike_ripple_folds",
        devexp="devexp_ripple_folds",
        pseudo_r2="pseudo_r2_ripple_folds",
        ll="ll_ripple_folds",
    )

    def get_mu_sem(metric_name: str) -> Tuple[np.ndarray, np.ndarray]:
        key = key_map[metric_name]
        if key not in results:
            raise KeyError(
                f"Missing key '{key}' in results (needed for {metric_name})."
            )
        arr = np.asarray(results[key])
        if arr.ndim != 3:
            raise ValueError(
                f"Expected results['{key}'] shape (R,F,N), got {arr.shape}"
            )
        arr = arr[sort_idx]  # sort by ridge
        return _popmean_sem_over_folds(arr)

    # compute curves
    mu_bps, sem_bps = get_mu_sem("bits_per_spike")
    mu_dev, sem_dev = get_mu_sem("devexp")
    mu_r2, sem_r2 = get_mu_sem("pseudo_r2")
    if include_ll_panel:
        mu_ll, sem_ll = get_mu_sem("ll")

    # choose ridge by criterion
    mu_sel, sem_sel = get_mu_sem(criterion)
    sel = _select_best_and_1se(ridge_s, mu_sel, sem_sel)

    best_ridge = sel["best_ridge"]
    one_se_ridge = sel["one_se_ridge"]

    # --- plotting ---
    n_panels = 4 if include_ll_panel else 3
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    def _panel(ax, mu, sem, ylabel: str):
        ax.errorbar(ridge_s, mu, yerr=sem, marker="o", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Ridge strength")
        ax.set_ylabel(ylabel)
        ax.axvline(best_ridge, linewidth=1.5)
        if show_one_se and (one_se_ridge != best_ridge):
            ax.axvline(one_se_ridge, linestyle="--", linewidth=1.25)
        ax.grid(True, which="both", axis="x", alpha=0.3)

    _panel(axes[0], mu_bps, sem_bps, "RIPPLE CV pop mean bits/spike")
    _panel(axes[1], mu_dev, sem_dev, "RIPPLE CV pop mean deviance explained")
    _panel(axes[2], mu_r2, sem_r2, "RIPPLE CV pop mean pseudo $R^2$")

    if include_ll_panel:
        _panel(axes[3], mu_ll, sem_ll, "RIPPLE CV pop mean Poisson LL (sum)")
    else:
        axes[3].axis("off")
        axes[3].text(
            0.0,
            1.0,
            "Selection:\n"
            f"- criterion: {criterion}\n"
            f"- best ridge: {best_ridge:g}\n"
            f"- 1-SE ridge: {one_se_ridge:g}\n"
            "(1-SE = largest ridge within 1 SEM of best across folds)",
            ha="left",
            va="top",
            transform=axes[3].transAxes,
            fontsize=11,
        )

    if title is None:
        title = f"{results.get('epoch', '')} — Ridge sweep (RIPPLE CV)"
    fig.suptitle(title, fontsize=14, y=1.02)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # return selection info + the actual curve used
    return dict(
        best_ridge=best_ridge,
        one_se_ridge=one_se_ridge,
        best_idx=sel["best_idx"],
        one_se_idx=sel["one_se_idx"],
        ridge_sorted=ridge_s,
        selection_metric=criterion,
        selection_metric_mu=mu_sel,
        selection_metric_sem=sem_sel,
        threshold=sel["threshold"],
    )


def load_results_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    results = dict(data)

    # restore fold_info if present
    if "fold_info" in results:
        results["fold_info"] = results["fold_info"].tolist()

    return results


def fit_ripple_glm_total_ca1_train_on_ripple_predict_pre(
    epoch: str,
    *,
    spikes: Dict[str, nap.TsGroup],
    regions: Tuple[str, str] = ("v1", "ca1"),
    Kay_ripple_detector: Dict[str, Any],
    ep: Dict[str, nap.IntervalSet],
    # Key choice:
    ripple_window: float = 0.2,  # fixed window in seconds (recommended)
    pre_window_s: Optional[float] = None,  # if None, uses ripple_window
    pre_buffer_s: float = 0.02,
    exclude_ripples: bool = True,
    pre_exclude_guard_s: float = 0.05,
    min_spikes_per_ripple: float = 0.05,
    n_splits: int = 5,
    ridge_strength: float = 0.1,
    maxiter: int = 6000,
    tol: float = 1e-7,
    # CV choice:
    cv_mode: str = "blocked",  # {"random", "blocked", "timeseries"}
    random_seed: int = 0,
    # Optional shuffle null (event-label shuffle that preserves V1 population structure)
    n_shuffles_ripple: int = 0,
    store_ripple_shuffle_preds: bool = False,
) -> Dict[str, Any]:
    """
    Predict V1 spike counts from TOTAL CA1 spike counts (single predictor).

    cv_mode:
      - "random":    KFold(shuffle=True)
      - "blocked":   KFold(shuffle=False) after sorting ripples by time (contiguous test blocks)
      - "timeseries": TimeSeriesSplit (train strictly before test)

    Shuffles (optional):
      - event-label shuffle within the training set (permute ripple labels), same perm for all V1 neurons.
    """
    v1_region, ca1_region = regions
    rng = np.random.default_rng(random_seed)

    # -----------------------
    # Build ripple IntervalSet
    # -----------------------
    r_start = np.asarray(Kay_ripple_detector[epoch]["start_time"], dtype=float)

    if ripple_window is None:
        # variable durations (not recommended unless you add log-duration offset)
        r_end = np.asarray(Kay_ripple_detector[epoch]["end_time"], dtype=float)
    else:
        r_end = r_start + float(ripple_window)

    # Sort by time so "blocked" or "timeseries" CV really corresponds to time blocks
    sort_idx = np.argsort(r_start)
    r_start = r_start[sort_idx]
    r_end = r_end[sort_idx]

    ripple_ep = nap.IntervalSet(start=r_start, end=r_end).intersect(ep[epoch])

    # -----------------------
    # Build pre IntervalSet
    # -----------------------
    if pre_window_s is None:
        pre_window_s = float(ripple_window) if ripple_window is not None else None

    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=ep[epoch],
        window_s=pre_window_s,
        buffer_s=pre_buffer_s,
        exclude_ripples=exclude_ripples,
        exclude_ripple_guard_s=pre_exclude_guard_s,
    )

    # -----------------------
    # Count spikes
    # -----------------------
    # Full CA1 counts per ripple: (n_ripples, n_ca1_units)
    Xr_full = np.asarray(spikes[ca1_region].count(ep=ripple_ep), dtype=np.float64)
    yr = np.asarray(spikes[v1_region].count(ep=ripple_ep), dtype=np.float64)

    Xp_full = np.asarray(spikes[ca1_region].count(ep=pre_ep), dtype=np.float64)
    yp = np.asarray(spikes[v1_region].count(ep=pre_ep), dtype=np.float64)

    n_r = Xr_full.shape[0]
    n_p = Xp_full.shape[0]

    if n_r < n_splits:
        raise ValueError(f"Not enough ripples: n_ripples={n_r}, n_splits={n_splits}")

    for name, arr in [
        ("Xr_full", Xr_full),
        ("yr", yr),
        ("Xp_full", Xp_full),
        ("yp", yp),
    ]:
        info = _first_nonfinite_info(arr)
        if info:
            raise ValueError(f"Non-finite in {name}: {info}")

    # TOTAL CA1 spike count per window: (n_samples, 1)
    Xr_tot = Xr_full.sum(axis=1, keepdims=True)
    Xp_tot = Xp_full.sum(axis=1, keepdims=True)

    # Filter V1 neurons based on ripple data only (same logic as your original)
    keep_y = (yr.sum(axis=0) / max(n_r, 1)) >= float(min_spikes_per_ripple)
    yr = yr[:, keep_y]
    yp = yp[:, keep_y]

    v1_unit_ids = np.array(list(spikes[v1_region].keys()))
    ca1_unit_ids = np.array(list(spikes[ca1_region].keys()))
    v1_unit_ids_kept = v1_unit_ids[keep_y]

    n_cells = yr.shape[1]
    if n_cells == 0:
        raise ValueError(
            f"No V1 units passed min_spikes_per_ripple={min_spikes_per_ripple}. n_ripples={n_r}"
        )

    # -----------------------
    # Choose CV splitter
    # -----------------------
    cv_mode = str(cv_mode).lower()
    if cv_mode == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    elif cv_mode == "blocked":
        # contiguous test blocks IF rows are time-sorted (we sorted above)
        splitter = KFold(n_splits=n_splits, shuffle=False)
    elif cv_mode == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError("cv_mode must be one of {'random','blocked','timeseries'}")

    # -----------------------
    # Allocate outputs
    # -----------------------
    y_r_test_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)
    y_r_hat_full = np.full((n_r, n_cells), np.nan, dtype=np.float32)

    pseudo_r2_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    devexp_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    ll_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)
    bps_f = np.full((n_splits, n_cells), np.nan, dtype=np.float32)

    # Optional shuffles
    yhat_shuff = None
    if store_ripple_shuffle_preds and n_shuffles_ripple > 0:
        yhat_shuff = np.full(
            (n_shuffles_ripple, n_r, n_cells), np.nan, dtype=np.float32
        )

    pseudo_r2_sh = np.full(
        (n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32
    )
    devexp_sh = np.full(
        (n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32
    )
    ll_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)
    bps_sh = np.full((n_splits, n_shuffles_ripple, n_cells), np.nan, dtype=np.float32)

    fold_info = []

    # Helper: event-label shuffle (same permutation for all V1 neurons)
    def _shuffle_rows(y_train: np.ndarray) -> np.ndarray:
        perm = rng.permutation(y_train.shape[0])
        return y_train[perm]

    # -----------------------
    # CV loop
    # -----------------------
    for fold, (train_idx, test_idx) in enumerate(splitter.split(Xr_tot)):
        X_train = Xr_tot[train_idx]
        X_test = Xr_tot[test_idx]
        y_train = yr[train_idx]
        y_test = yr[test_idx]

        # Preprocess scalar predictor (reusing your helper)
        X_train_pp, X_test_pp, keep_x, mean, std = _preprocess_X_fit_apply(
            X_train, X_test
        )
        if X_train_pp.shape[1] == 0:
            raise ValueError(f"Fold {fold}: predictor became constant after filtering.")

        glm = nmo.glm.PopulationGLM(
            solver_name="LBFGS",
            regularizer="Ridge",
            regularizer_strength=float(ridge_strength),
            solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
        )
        glm.fit(X_train_pp, y_train)

        lam = np.asarray(glm.predict(X_test_pp), dtype=np.float64)

        y_r_hat_full[test_idx] = lam.astype(np.float32)
        y_r_test_full[test_idx] = y_test.astype(np.float32)

        pseudo_r2_f[fold] = mcfadden_pseudo_r2_per_neuron(y_test, lam, y_train).astype(
            np.float32
        )
        devexp_f[fold] = deviance_explained_per_neuron(
            y_test=y_test, lam_test=lam, y_null_fit=y_train
        ).astype(np.float32)
        ll_f[fold] = poisson_ll_per_neuron(y_test, lam).astype(np.float32)
        bps_f[fold] = bits_per_spike_per_neuron(
            y_test=y_test, lam_test=lam, y_null_fit=y_train
        ).astype(np.float32)

        # Optional shuffle nulls
        for s in range(n_shuffles_ripple):
            y_train_sh = _shuffle_rows(y_train)

            glm_s = nmo.glm.PopulationGLM(
                solver_name="LBFGS",
                regularizer="Ridge",
                regularizer_strength=float(ridge_strength),
                solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
            )
            glm_s.fit(X_train_pp, y_train_sh)
            lam_s = np.asarray(glm_s.predict(X_test_pp), dtype=np.float64)

            if yhat_shuff is not None:
                yhat_shuff[s, test_idx] = lam_s.astype(np.float32)

            pseudo_r2_sh[fold, s] = mcfadden_pseudo_r2_per_neuron(
                y_test, lam_s, y_train
            ).astype(np.float32)
            devexp_sh[fold, s] = deviance_explained_per_neuron(
                y_test=y_test, lam_test=lam_s, y_null_fit=y_train
            ).astype(np.float32)
            ll_sh[fold, s] = poisson_ll_per_neuron(y_test, lam_s).astype(np.float32)
            bps_sh[fold, s] = bits_per_spike_per_neuron(
                y_test=y_test, lam_test=lam_s, y_null_fit=y_train
            ).astype(np.float32)

            del glm_s, lam_s, y_train_sh

        fold_info.append(
            dict(
                fold=int(fold),
                train_idx=train_idx,
                test_idx=test_idx,
                x_mean=float(mean[0]) if mean.size else np.nan,
                x_std=float(std[0]) if std.size else np.nan,
                keep_x=keep_x,
                n_ripples=int(n_r),
                n_pre=int(n_p),
            )
        )

        del glm, lam
        jax.clear_caches()
        gc.collect()

    # sanity check
    if np.isnan(y_r_hat_full).any() or np.isnan(y_r_test_full).any():
        raise ValueError(
            "Some rows were never assigned to a test fold (check CV splitter)."
        )

    # -----------------------
    # PRE: train on ALL ripples, predict ALL pre
    # -----------------------
    X_all_pp, _, keep_x_all, mean_all, std_all = _preprocess_X_fit_apply(Xr_tot, Xr_tot)
    if X_all_pp.shape[1] == 0:
        raise ValueError(
            "Total CA1 predictor became constant after filtering on all ripples."
        )

    # Apply same preprocessing to PRE
    Xp_k = Xp_tot[:, keep_x_all]
    Xp_pp = (Xp_k - mean_all[None, :]) / (std_all[None, :] + 1e-8)
    Xp_pp /= np.sqrt(max(X_all_pp.shape[1], 1))
    Xp_pp = np.clip(Xp_pp, -10.0, 10.0)

    glm_all = nmo.glm.PopulationGLM(
        solver_name="LBFGS",
        regularizer="Ridge",
        regularizer_strength=float(ridge_strength),
        solver_kwargs=dict(maxiter=int(maxiter), tol=float(tol), stepsize=0),
    )
    glm_all.fit(X_all_pp, yr)

    lam_pre = np.asarray(glm_all.predict(Xp_pp), dtype=np.float64)
    lam_pre_f32 = lam_pre.astype(np.float32)

    pseudo_r2_pre = mcfadden_pseudo_r2_per_neuron(yp, lam_pre, yp).astype(np.float32)
    devexp_pre = deviance_explained_per_neuron(
        y_test=yp, lam_test=lam_pre, y_null_fit=yp
    ).astype(np.float32)
    ll_pre = poisson_ll_per_neuron(yp, lam_pre).astype(np.float32)
    bps_pre = bits_per_spike_per_neuron(
        y_test=yp, lam_test=lam_pre, y_null_fit=yp
    ).astype(np.float32)

    del glm_all
    jax.clear_caches()
    gc.collect()

    return dict(
        epoch=epoch,
        cv_mode=cv_mode,
        ripple_window=float(ripple_window) if ripple_window is not None else None,
        pre_window_s=float(pre_window_s) if pre_window_s is not None else None,
        pre_buffer_s=float(pre_buffer_s),
        exclude_ripples=bool(exclude_ripples),
        pre_exclude_guard_s=float(pre_exclude_guard_s),
        min_spikes_per_ripple=float(min_spikes_per_ripple),
        n_splits=int(n_splits),
        n_shuffles_ripple=int(n_shuffles_ripple),
        ridge_strength=float(ridge_strength),
        random_seed=int(random_seed),
        n_ripples=int(n_r),
        n_pre=int(n_p),
        n_cells=int(n_cells),
        v1_unit_ids=v1_unit_ids_kept,
        ca1_unit_ids=ca1_unit_ids,
        fold_info=fold_info,
        # Ripple CV outputs
        y_ripple_test=y_r_test_full,
        yhat_ripple=y_r_hat_full,
        pseudo_r2_ripple_folds=pseudo_r2_f,
        devexp_ripple_folds=devexp_f,
        ll_ripple_folds=ll_f,
        bits_per_spike_ripple_folds=bps_f,
        # Optional shuffle outputs
        yhat_ripple_shuff=yhat_shuff,
        pseudo_r2_ripple_shuff_folds=pseudo_r2_sh,
        devexp_ripple_shuff_folds=devexp_sh,
        ll_ripple_shuff_folds=ll_sh,
        bits_per_spike_ripple_shuff_folds=bps_sh,
        # PRE outputs
        y_pre_test=yp.astype(np.float32),
        yhat_pre=lam_pre_f32,
        pseudo_r2_pre=pseudo_r2_pre,
        devexp_pre=devexp_pre,
        ll_pre=ll_pre,
        bits_per_spike_pre=bps_pre,
    )


# compute cv prediction for during and pre ripple
def main():
    out_dir = analysis_path / "ripple" / "glm_results_train_on_ripple"
    fig_dir = analysis_path / "figs" / "ripple" / "train_on_ripple"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    exclude_ripples = False
    ripple_window = 0.2
    min_spikes_per_ripple = 0.1
    ridge_strength = 1e-1

    for epoch in epoch_list:
        out_path = (
            out_dir
            / f"{epoch}_train_ripple_predict_ripple_and_pre_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_min_spikes_per_ripple_{min_spikes_per_ripple}_ridge_strength_{ridge_strength}.npz"
        )
        if out_path.exists():
            print("Loading existing results:", out_path)
            results = load_results_npz(out_path)
        else:
            print("Computing results for:", epoch)
            results = fit_ripple_glm_train_on_ripple_predict_pre(
                epoch=epoch,
                spikes=spikes,
                regions=("v1", "ca1"),
                Kay_ripple_detector=Kay_ripple_detector,
                ep=ep,
                min_spikes_per_ripple=min_spikes_per_ripple,
                n_shuffles_ripple=100,
                random_seed=45,
                ripple_window=ripple_window,
                exclude_ripples=exclude_ripples,
                pre_window_s=ripple_window,
                pre_buffer_s=0.02,
                pre_exclude_guard_s=0.05,
                n_splits=5,
                ridge_strength=ridge_strength,
            )

            save_results_npz(out_path, results)
            print("Saved:", out_path)

        # --- Plot RIPPLE performance ---
        # RIPPLE
        plot_ripple_glm_pseudo_r2(
            pseudo_r2_real_folds=results["pseudo_r2_ripple_folds"],
            pseudo_r2_shuff_folds=results["pseudo_r2_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=fig_dir
            / f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_RIPPLE_pseudo_r2_summary.png",
            null_mode="pooled",
        )

        plot_ripple_glm_mae(
            mae_real_folds=results["mae_ripple_folds"],
            mae_shuff_folds=results["mae_ripple_shuff_folds"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=fig_dir
            / f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_RIPPLE_mae_summary.png",
            null_mode="pooled",
        )
        plot_ripple_glm_delta_ll(
            ll_real_folds=results["ll_ripple_folds"],
            ll_shuff_folds=results["ll_ripple_shuff_folds"],
            y_test=results["y_ripple_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=fig_dir
            / f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_RIPPLE_delta_bits_per_spike_summary.png",
            normalize="bits_per_spike",
            null_mode="pooled",
        )
        # --- Plot PRE performance (same plotting code; different arrays) ---
        # PRE

        plot_pre_pseudo_r2_summary(
            pseudo_r2_pre=results["pseudo_r2_pre"],
            y_pre_test=results["y_pre_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=fig_dir
            / f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_PRE_pseudo_r2_summary.png",
        )

        plot_pre_mae_summary(
            mae_pre=results["mae_pre"],
            y_pre_test=results["y_pre_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=fig_dir
            / f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_PRE_mae_summary.png",
        )

        plot_pre_ll_summary(
            ll_pre=results["ll_pre"],
            y_pre_test=results["y_pre_test"],
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            out_path=fig_dir
            / f"{epoch}_exclude_ripples_{exclude_ripples}_ripple_window_{ripple_window}_PRE_delta_bits_per_spike_summary.png",
            normalize="bits_per_spike",
        )


# ridge strength hyperparameter sweep
# def main():
#     out_dir = analysis_path / "ripple" / "glm_ridge_sweep"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     fig_dir = analysis_path / "figs" / "ripple" / "glm_ridge_sweep"
#     fig_dir.mkdir(parents=True, exist_ok=True)

#     exclude_ripples = False
#     ripple_window = 0.2
#     min_spikes_per_ripple = 0.1

#     # Example sweep grid
#     ridge_strengths = np.logspace(-6, 1, 15)

#     for epoch in epoch_list:
#         out_path = (
#             out_dir
#             / f"{epoch}_ridge_sweep_exclude_{exclude_ripples}_ripple_window_{ripple_window}_minspk_{min_spikes_per_ripple}.npz"
#         )

#         if out_path.exists():
#             print("Loading existing sweep:", out_path)
#             results = load_results_npz(out_path)
#         else:
#             print("Running ridge sweep:", epoch)
#             results = fit_ripple_glm_ridge_sweep_no_shuffles(
#                 epoch=epoch,
#                 spikes=spikes,
#                 regions=("v1", "ca1"),
#                 Kay_ripple_detector=Kay_ripple_detector,
#                 ep=ep,
#                 ridge_strengths=ridge_strengths,
#                 min_spikes_per_ripple=min_spikes_per_ripple,
#                 random_seed=47,
#                 ripple_window=ripple_window,
#                 exclude_ripples=exclude_ripples,
#                 pre_window_s=ripple_window,
#                 pre_buffer_s=0.02,
#                 pre_exclude_guard_s=0.05,
#                 n_splits=5,
#                 maxiter=1500,  # lower is usually fine with warm-start
#                 tol=1e-6,
#                 warm_start=True,
#                 sort_ridges_for_warm_start=True,
#             )
#             save_results_npz(out_path, results)
#             print("Saved:", out_path)

#         # Example: pick ridge by population mean bits/spike on RIPPLE CV
#         bps_pop = np.asarray(results["bits_per_spike_ripple_pop_mean"], float)  # (R,)
#         best_idx = int(np.nanargmax(bps_pop))
#         best_ridge = float(np.asarray(results["ridge_strengths"], float)[best_idx])
#         print(
#             f"{epoch}: best ridge (by ripple CV pop mean bits/spike) = {best_ridge:g}"
#         )

#         sel = plot_ridge_sweep_summary(
#             results,
#             criterion="bits_per_spike",  # or "devexp" / "pseudo_r2" / "ll"
#             include_ll_panel=False,
#             out_path=fig_dir / f"{epoch}_ridge_sweep_summary.png",
#         )

#         print("Best ridge:", sel["best_ridge"])
#         print("1-SE ridge:", sel["one_se_ridge"])


# null model with total CA1 spikes
# def main():
#     out_dir = analysis_path / "ripple" / "glm_total_null"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # fig_dir = analysis_path / "figs" / "ripple" / "glm_total_null"
#     # fig_dir.mkdir(parents=True, exist_ok=True)

#     exclude_ripples = False
#     ripple_window = 0.2
#     min_spikes_per_ripple = 0.1

#     for epoch in epoch_list:
#         out_path = (
#             out_dir
#             / f"{epoch}_total_null_exclude_{exclude_ripples}_ripple_window_{ripple_window}_minspk_{min_spikes_per_ripple}.npz"
#         )
#         res_tot = fit_ripple_glm_total_ca1_train_on_ripple_predict_pre(
#             epoch=epoch,
#             spikes=spikes,
#             regions=("v1", "ca1"),
#             Kay_ripple_detector=Kay_ripple_detector,
#             ep=ep,
#             ripple_window=0.2,
#             pre_window_s=0.2,
#             cv_mode="blocked",  # see below
#             ridge_strength=0.03,
#             n_splits=5,
#             n_shuffles_ripple=100,
#             store_ripple_shuffle_preds=False,
#             random_seed=47,
#         )
#         save_results_npz(out_path, res_tot)


if __name__ == "__main__":
    main()
