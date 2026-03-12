from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import scipy
import position_tools as pt
import spikeinterface.full as si
import kyutils
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.special import gammaln

from dataclasses import dataclass
from typing import Iterator, Tuple, Literal, Any

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

time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)
speed_threshold = 4  # cm/s
position_offset = 10

trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]


with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "body_position.pkl", "rb") as f:
    body_position_dict = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)

with open(analysis_path / "ripple" / "Kay_ripple_detector.pkl", "rb") as f:
    Kay_ripple_detector = pickle.load(f)

sleep_times = {}
for epoch in epoch_list:
    with open(analysis_path / "sleep_times" / f"{epoch}.pkl", "rb") as f:
        sleep_times[epoch] = pickle.load(f)

immobility_times = {}
for epoch in epoch_list:
    with open(analysis_path / "immobility_times" / f"{epoch}.pkl", "rb") as f:
        immobility_times[epoch] = pickle.load(f)

run_times = {}
for epoch in epoch_list:
    with open(analysis_path / "run_times" / f"{epoch}.pkl", "rb") as f:
        run_times[epoch] = pickle.load(f)

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


def poisson_ll_per_neuron(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Poisson log-likelihood per neuron (sum over time)."""
    lam = np.clip(lam, 1e-12, None)
    return np.sum(y * np.log(lam) - lam - gammaln(y + 1), axis=0)


def mcfadden_pseudo_r2_per_neuron(
    y_test: np.ndarray,
    lam_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """McFadden pseudo-R2 per neuron using train-fitted null rate."""
    ll_model = poisson_ll_per_neuron(y_test, lam_test)

    lam0 = np.mean(y_train, axis=0, keepdims=True)  # (1, N)
    lam0 = np.repeat(lam0, y_test.shape[0], axis=0)  # (T_test, N)
    ll_null = poisson_ll_per_neuron(y_test, lam0)

    return 1.0 - (ll_model / ll_null)


def mae_per_neuron(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean absolute error per neuron (mean over time).

    Parameters
    ----------
    y_true : np.ndarray, shape (T, N)
    y_pred : np.ndarray, shape (T, N)

    Returns
    -------
    mae : np.ndarray, shape (N,)
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)


def shuffle_time_per_neuron(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle time independently within each neuron (column)."""
    y_shuff = y.copy()
    for n in range(y_shuff.shape[1]):
        rng.shuffle(y_shuff[:, n])
    return y_shuff


spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])


def save_results_npz(out_path: Path, results: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        # metadata
        epoch=results["epoch"],
        time_bin_size=results["time_bin_size"],
        min_spikes=results["min_spikes"],
        n_shuffles=results["n_shuffles"],
        random_seed=results["random_seed"],
        v1_unit_ids=results["v1_unit_ids"],
        ca1_unit_ids=results["ca1_unit_ids"],
        # metrics (store as object arrays to be safe)
        pseudo_r2_real_folds=np.array(results["pseudo_r2_real_folds"], dtype=object),
        mae_real_folds=np.array(results["mae_real_folds"], dtype=object),
        pseudo_r2_shuff_folds=np.array(results["pseudo_r2_shuff_folds"], dtype=object),
        mae_shuff_folds=np.array(results["mae_shuff_folds"], dtype=object),
        fold_info=np.array(results["fold_info"], dtype=object),
        yhat_real_folds=np.array(results["yhat_real_folds"], dtype=object),
        y_test_folds=np.array(results["y_test_folds"], dtype=object),
    )


BalanceMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class BalancedPerNeuronKFold:
    """
    K-fold splitter that approximately balances per-neuron spike counts across folds.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Shuffle tie-breaks and within equal-size groups.
    random_state : int
        Seed for deterministic behavior.
    balance_mode : {"absolute", "relative"}
        "relative" normalizes each neuron's total to reduce dominance by high-rate neurons.
    min_total_per_neuron : int
        Neurons with total spikes < this threshold are ignored in the balancing objective.
        They are still included in y (i.e. in the GLM), this only affects fold assignment.
    eps : float
        Numerical stability constant.
    """

    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 0
    balance_mode: BalanceMode = "relative"
    min_total_per_neuron: int = 0
    eps: float = 1e-12

    def split(self, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError("y must be 2D, shape (n_samples, n_neurons)")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if y.shape[0] < self.n_splits:
            raise ValueError("n_samples must be >= n_splits")
        if np.any(~np.isfinite(y)):
            raise ValueError("y must be finite")
        if np.any(y < 0):
            raise ValueError("y must be nonnegative")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_neurons = y.shape

        totals = y.sum(axis=0)  # (N,)
        use = totals >= float(self.min_total_per_neuron)
        if not np.any(use):
            # Fall back: no neurons eligible for balancing -> random-ish split by bin weight
            use = np.ones(n_neurons, dtype=bool)

        y_use = y[:, use]

        if self.balance_mode == "relative":
            w = 1.0 / (y_use.sum(axis=0) + self.eps)
            y_eff = y_use * w[None, :]
        elif self.balance_mode == "absolute":
            y_eff = y_use
        else:
            raise ValueError("balance_mode must be 'absolute' or 'relative'")

        target = y_eff.sum(axis=0) / float(self.n_splits)  # (N_use,)

        bin_weight = y_eff.sum(axis=1)  # (T,)
        order = np.argsort(-bin_weight, kind="mergesort")

        if self.shuffle:
            bw = bin_weight[order]
            run_starts = np.flatnonzero(np.r_[True, bw[1:] != bw[:-1]])
            run_ends = np.r_[run_starts[1:], bw.size]
            order_shuffled = order.copy()
            for s, e in zip(run_starts, run_ends):
                if e - s > 1:
                    block = order_shuffled[s:e].copy()
                    rng.shuffle(block)
                    order_shuffled[s:e] = block
            order = order_shuffled

        fold_sums = np.zeros((self.n_splits, target.size), dtype=np.float64)
        fold_sizes = np.zeros(self.n_splits, dtype=np.int64)
        fold_ids = np.full(n_samples, -1, dtype=np.int64)

        diff = fold_sums - target[None, :]
        base = np.einsum("kn,kn->k", diff, diff)

        for idx in order:
            v = y_eff[idx].astype(np.float64, copy=False)

            dot = diff @ v  # (K,)
            score = base + 2.0 * dot

            min_score = score.min()
            candidates = np.flatnonzero(score == min_score)
            if candidates.size > 1:
                min_size = fold_sizes[candidates].min()
                candidates = candidates[fold_sizes[candidates] == min_size]
            k = (
                int(rng.choice(candidates))
                if (candidates.size > 1 and self.shuffle)
                else int(candidates[0])
            )

            fold_ids[idx] = k

            v_norm2 = float(v @ v)
            base[k] = float(base[k] + 2.0 * dot[k] + v_norm2)
            fold_sums[k] += v
            diff[k] += v
            fold_sizes[k] += 1

        all_idx = np.arange(n_samples)
        for k in range(self.n_splits):
            test_idx = all_idx[fold_ids == k]
            train_idx = all_idx[fold_ids != k]
            yield train_idx, test_idx

    def balance_report(self, y: np.ndarray) -> dict:
        """
        Quick diagnostics: fold sizes + per-neuron fraction-of-total per fold.
        """
        y = np.asarray(y)
        totals = y.sum(axis=0) + self.eps
        fold_sizes = []
        frac_cv = []

        fold_fracs = []
        for _, test_idx in self.split(y):
            fold_sizes.append(int(test_idx.size))
            fold_fracs.append(y[test_idx].sum(axis=0) / totals)

        fold_fracs = np.stack(fold_fracs, axis=0)  # (K, N)
        # coefficient of variation over folds, per neuron
        cv = np.std(fold_fracs, axis=0) / (np.mean(fold_fracs, axis=0) + self.eps)
        frac_cv = cv[np.isfinite(cv)]

        return {
            "fold_sizes": fold_sizes,
            "median_frac_cv": float(np.median(frac_cv)),
            "p90_frac_cv": float(np.percentile(frac_cv, 90)),
        }


def fit_ripple_glm(epoch, min_spikes=10, n_shuffles=50, random_seed=0, mode="during"):

    if mode == "during":
        ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"],
            end=Kay_ripple_detector[epoch]["end_time"],
        )
    elif mode == "pre":
        ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"] - 0.2,
            end=Kay_ripple_detector[epoch]["start_time"],
        )

    spike_counts = {}
    for region in regions:
        spike_counts[region] = spikes[region].count(ep=ripple_ep)

    v1_unit_ids = np.array(list(spikes["v1"].keys()))
    ca1_unit_ids = np.array(list(spikes["ca1"].keys()))

    # -----------------------
    # Data
    # -----------------------
    X = np.asarray(spike_counts["ca1"], dtype=np.float64)
    y = np.asarray(spike_counts["v1"], dtype=np.float64)

    # global y filter (keeps neuron set fixed)
    keep_y = y.sum(axis=0) >= min_spikes
    y = y[:, y.sum(axis=0) >= min_spikes]
    v1_unit_ids_kept = v1_unit_ids[keep_y]

    # kf = KFold(n_splits=5, shuffle=True, random_state=1)
    # rng = np.random.default_rng(random_seed)

    rng = np.random.default_rng(random_seed)

    kf = BalancedPerNeuronKFold(
        n_splits=5,
        shuffle=True,
        random_state=1,
        balance_mode="relative",
        min_total_per_neuron=min_spikes,  # ignore super-silent neurons in the balancing objective
    )

    rep = kf.balance_report(y)
    print("BalancedPerNeuronKFold fold_sizes:", rep["fold_sizes"])
    print(
        "Balance quality (fraction-of-total CV): median=",
        rep["median_frac_cv"],
        "p90=",
        rep["p90_frac_cv"],
    )

    # store per-neuron metrics (per fold)
    pseudo_r2_real_folds = []  # list of arrays (N_fold,)
    mae_real_folds = []  # list of arrays (N_fold,)

    pseudo_r2_shuff_folds = []  # list of arrays (n_shuffles, N_fold)
    mae_shuff_folds = []  # list of arrays (n_shuffles, N_fold)

    # NEW: store predictions/observations for the real model (per fold)
    yhat_real_folds = []  # list of arrays (T_test, N)
    y_test_folds = []  # list of arrays (T_test, N)

    # Also store fold bookkeeping (very useful later)
    fold_info = []  # list of dicts per fold

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---- X preprocessing fit on train ----
        keep_x = X_train.std(axis=0) > 1e-6
        X_train = X_train[:, keep_x]
        X_test = X_test[:, keep_x]

        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)

        X_train = (X_train - mean) / (std + 1e-8)
        X_test = (X_test - mean) / (std + 1e-8)

        X_train /= np.sqrt(X_train.shape[1])
        X_test /= np.sqrt(X_test.shape[1])

        X_train = np.clip(X_train, -10.0, 10.0)
        X_test = np.clip(X_test, -10.0, 10.0)

        # -------------------------
        # REAL fit
        # -------------------------
        glm = nmo.glm.PopulationGLM(
            solver_name="LBFGS",
            regularizer="Ridge",
            regularizer_strength=0.1,
            solver_kwargs=dict(maxiter=6000, tol=1e-7, stepsize=0),
        )
        glm.fit(X_train, y_train)

        lam_test = glm.predict(X_test)  # predicted mean counts, (T_test, N)

        # NEW: save held-out predictions + held-out observed counts
        yhat_real_folds.append(np.asarray(lam_test, dtype=np.float64))
        y_test_folds.append(np.asarray(y_test, dtype=np.float64))

        r2_real = mcfadden_pseudo_r2_per_neuron(y_test, lam_test, y_train)
        mae_real = mae_per_neuron(y_test, lam_test)

        pseudo_r2_real_folds.append(r2_real)
        mae_real_folds.append(mae_real)

        # -------------------------
        # SHUFFLED null: shuffle y_train BEFORE fit (repeat)
        # -------------------------
        r2_shuff = np.zeros((n_shuffles, y_train.shape[1]), dtype=np.float64)
        mae_shuff = np.zeros((n_shuffles, y_train.shape[1]), dtype=np.float64)

        for s in range(n_shuffles):
            y_train_shuff = shuffle_time_per_neuron(y_train, rng)

            glm_s = nmo.glm.PopulationGLM(
                solver_name="LBFGS",
                regularizer="Ridge",
                regularizer_strength=0.1,
                solver_kwargs=dict(maxiter=6000, tol=1e-7, stepsize=0),
            )
            glm_s.fit(X_train, y_train_shuff)

            lam_test_s = glm_s.predict(X_test)

            r2_shuff[s] = mcfadden_pseudo_r2_per_neuron(y_test, lam_test_s, y_train)
            mae_shuff[s] = mae_per_neuron(y_test, lam_test_s)

            # ---- cleanup ----
            del glm_s, lam_test_s, y_train_shuff
            if (s + 1) % 5 == 0:
                jax.clear_caches()
                gc.collect()

        pseudo_r2_shuff_folds.append(r2_shuff)
        mae_shuff_folds.append(mae_shuff)

        fold_info.append(
            dict(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                keep_x=keep_x,
                x_mean=mean.ravel(),
                x_std=std.ravel(),
                # balance diagnostics
                test_total_v1_spikes=float(np.sum(y[test_idx])),
                train_total_v1_spikes=float(np.sum(y[train_idx])),
                test_n_bins=int(test_idx.size),
                train_n_bins=int(train_idx.size),
            )
        )

        # -------------------------
        # Fold summaries + p-values (mean across neurons)
        # -------------------------
        obs_r2_mean = float(np.nanmean(r2_real))
        shuff_r2_means = np.nanmean(r2_shuff, axis=1)
        p_r2 = float(np.mean(shuff_r2_means >= obs_r2_mean))

        obs_mae_mean = float(np.nanmean(mae_real))
        shuff_mae_means = np.nanmean(mae_shuff, axis=1)
        # for MAE, "better" is smaller
        p_mae = float(np.mean(shuff_mae_means <= obs_mae_mean))

        print(
            f"Fold {fold}: "
            f"mean pseudo-R2={obs_r2_mean:.4f} (p={p_r2:.4g}) | "
            f"mean MAE={obs_mae_mean:.4f} (p={p_mae:.4g}) | "
            f"N={r2_real.size}"
        )

    # -------------------------
    # Aggregate across folds (optional)
    # -------------------------
    r2_real_all = np.concatenate(pseudo_r2_real_folds)
    mae_real_all = np.concatenate(mae_real_folds)

    r2_shuff_all_means = np.concatenate(
        [np.nanmean(a, axis=1) for a in pseudo_r2_shuff_folds]
    )
    mae_shuff_all_means = np.concatenate(
        [np.nanmean(a, axis=1) for a in mae_shuff_folds]
    )

    print("Overall mean pseudo-R2 (pooled neurons):", float(np.nanmean(r2_real_all)))
    print("Overall mean MAE (pooled neurons):", float(np.nanmean(mae_real_all)))

    print(
        "Overall shuffle p (pseudo-R2):",
        float(np.mean(r2_shuff_all_means >= np.nanmean(r2_real_all))),
    )
    print(
        "Overall shuffle p (MAE; smaller is better):",
        float(np.mean(mae_shuff_all_means <= np.nanmean(mae_real_all))),
    )

    results = dict(
        epoch=epoch,
        time_bin_size=time_bin_size,
        min_spikes=min_spikes,
        n_shuffles=n_shuffles,
        random_seed=random_seed,
        v1_unit_ids=v1_unit_ids_kept,
        ca1_unit_ids=ca1_unit_ids,
        pseudo_r2_real_folds=pseudo_r2_real_folds,
        mae_real_folds=mae_real_folds,
        pseudo_r2_shuff_folds=pseudo_r2_shuff_folds,
        mae_shuff_folds=mae_shuff_folds,
        fold_info=fold_info,
        yhat_real_folds=yhat_real_folds,
        y_test_folds=y_test_folds,
    )

    return results


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = (np.arange(x.size) + 1) / x.size
    return x, y


def _as_2d_float(a: Any) -> np.ndarray:
    """
    Convert folds list/array into a numeric 2D array (n_folds, n_cells).
    Handles object arrays from np.load(..., allow_pickle=True).
    """
    arr = np.asarray(a)
    if arr.dtype == object:
        return np.stack(arr.tolist(), axis=0).astype(float)
    return np.asarray(arr, dtype=float)


def _as_3d_float(a: Any) -> np.ndarray:
    """
    Convert shuffle folds list/array into numeric 3D array
    (n_folds, n_shuffles, n_cells). Handles object arrays.
    """
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
) -> None:
    """
    Plot pseudo-R2-focused panels and save as a single figure.

    Panels
    ------
    A: Per-cell pseudo-R2 histogram (mean over CV folds) + stats box
    B: Shuffle null distribution of mean pseudo-R2 (mean over cells), with observed mean line
    D: ECDF real vs shuffle (per-cell, fold-averaged shuffle samples pooled)
    E: Z-score histogram (per-cell, vs per-cell shuffle distribution)
    F: Effect size vs significance scatter (pseudo-R2 vs -log10 p)
    """
    pseudo_r2_real_folds = _as_2d_float(pseudo_r2_real_folds)  # (F, N)
    pseudo_r2_shuff_folds = _as_3d_float(pseudo_r2_shuff_folds)  # (F, S, N)

    # effect size per cell (mean across folds)
    real_r2 = np.mean(pseudo_r2_real_folds, axis=0)  # (N,)

    # per-cell shuffle samples pooled across folds: (F*S, N)
    shuff_r2_per_cell = pseudo_r2_shuff_folds.reshape(
        -1, pseudo_r2_shuff_folds.shape[-1]
    )

    # per-shuffle population mean: (F*S,)
    shuff_r2_mean = np.mean(pseudo_r2_shuff_folds, axis=2).ravel()

    # per-cell z-scores vs shuffle
    shuff_mu = np.mean(shuff_r2_per_cell, axis=0)
    shuff_sd = np.std(shuff_r2_per_cell, axis=0) + 1e-12
    z_per_cell = (real_r2 - shuff_mu) / shuff_sd

    # per-cell p-values vs shuffle (one-sided: shuffle >= observed)
    n_null = shuff_r2_per_cell.shape[0]
    p_per_cell = (np.sum(shuff_r2_per_cell >= real_r2[None, :], axis=0) + 1) / (
        n_null + 1
    )
    neglog10_p = -np.log10(p_per_cell)

    # ECDF shuffle uses pooled per-cell null values
    shuffled_r2_flat = shuff_r2_per_cell.ravel()

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
        real_r2,
        bins=np.linspace(-0.4, 0.4, 81),
        weights=np.ones_like(real_r2) / len(real_r2),
        alpha=alpha,
    )
    ax.set_xlabel("Pseudo $R^2$ McFadden (mean over CV folds)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell pseudo-$R^2$")

    stats_text = (
        f"Mean:   {np.mean(real_r2):.4f}\n"
        f"Median: {np.median(real_r2):.4f}\n"
        f"25%:    {np.percentile(real_r2, 25):.4f}\n"
        f"75%:    {np.percentile(real_r2, 75):.4f}"
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
    ax.hist(shuff_r2_mean, color="gray", alpha=0.75, bins=40)
    ax.axvline(np.mean(real_r2), color="red", linewidth=2)
    ax.set_xlabel("Pseudo $R^2$ (mean over cells & CV folds)")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: mean pseudo-$R^2$")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(real_r2):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # D (ECDF)
    ax = axes[2]
    xr, yr = _ecdf(real_r2)
    xs, ys = _ecdf(shuffled_r2_flat)
    ax.plot(xr, yr, linewidth=2, label="Real")
    ax.plot(xs, ys, linewidth=2, label="Shuffle", color="gray")
    ax.set_xlabel("Pseudo $R^2$")
    ax.set_ylabel("ECDF")
    ax.set_title("D. ECDF: real vs shuffle")
    ax.legend(frameon=False)

    # E (z-score)
    ax = axes[3]
    ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(2.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score vs shuffle (per cell)")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized effect size")

    # F (effect vs significance)
    ax = axes[4]
    ax.scatter(real_r2, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
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

    # last panel: leave empty or use as a notes panel
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Notes:\n"
        "- p-values are one-sided: P(shuffle ≥ observed).\n"
        "- Z-score uses per-cell shuffle mean/std.\n"
        "- Shuffle ECDF pools per-cell null values.",
        ha="left",
        va="top",
        fontsize=10,
        transform=axes[5].transAxes,
    )

    fig.suptitle(
        f"{animal_name} {date} {epoch} — Pseudo-$R^2$ summary", fontsize=14, y=1.02
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ripple_glm_prediction_gain(
    *,
    mae_real_folds: Any,
    mae_shuff_folds: Any,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.7,
) -> None:
    """
    Plot prediction-gain-focused panels and save as a single figure.

    Definitions
    -----------
    MAE per cell is averaged over time, then averaged over CV folds.
    Prediction gain per cell:
        gain = MAE_shuffle / MAE_real
    (so >1 means the real model improves over shuffle)

    Panels
    ------
    A: Histogram of prediction gain (per cell)
    B: Shuffle null distribution of mean gain (mean over cells), with observed mean line
    E: Z-score histogram (per cell, vs per-cell shuffle gain distribution)
    F: Effect size vs significance scatter (gain vs -log10 p)
    """
    mae_real_folds = _as_2d_float(mae_real_folds)  # (F, N)
    mae_shuff_folds = _as_3d_float(mae_shuff_folds)  # (F, S, N)

    # per-cell real MAE (mean over folds)
    mae_real = np.mean(mae_real_folds, axis=0)  # (N,)

    # per-cell shuffled MAE samples pooled across folds: (F*S, N)
    mae_shuff_per_cell = mae_shuff_folds.reshape(-1, mae_shuff_folds.shape[-1])

    # per-cell shuffled MAE mean (for gain computation)
    mae_shuff_mean = np.mean(mae_shuff_per_cell, axis=0)  # (N,)

    # prediction gain per cell (using mean shuffled MAE / real MAE)
    gain_real = mae_shuff_mean / (mae_real + 1e-12)

    # per-cell gain null distribution: gain_shuff_sample = MAE_shuff_sample / MAE_real
    gain_shuff_per_cell = mae_shuff_per_cell / (mae_real[None, :] + 1e-12)  # (F*S, N)

    # population mean gain null distribution (one value per null sample)
    gain_shuff_mean = np.mean(gain_shuff_per_cell, axis=1)  # (F*S,)

    # per-cell z-scores vs gain null
    shuff_mu = np.mean(gain_shuff_per_cell, axis=0)
    shuff_sd = np.std(gain_shuff_per_cell, axis=0) + 1e-12
    z_per_cell = (gain_real - shuff_mu) / shuff_sd

    # per-cell p-values: P(null gain >= observed gain) (one-sided)
    n_null = gain_shuff_per_cell.shape[0]
    p_per_cell = (np.sum(gain_shuff_per_cell >= gain_real[None, :], axis=0) + 1) / (
        n_null + 1
    )
    neglog10_p = -np.log10(p_per_cell)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 8),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # A: gain distribution
    ax = axes[0]
    ax.hist(
        gain_real,
        bins=np.linspace(0.5, 2.0, 31),
        weights=np.ones_like(gain_real) / len(gain_real),
        alpha=alpha,
    )
    ax.axvline(1.0, color="red", linestyle="--")
    ax.set_xlabel("Prediction gain per cell (MAE shuffle / MAE real)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Prediction gain distribution")
    ax.text(
        0.98,
        0.95,
        f"Mean = {np.mean(gain_real):.4f}\nMedian = {np.median(gain_real):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # B: null distribution of mean gain (over cells)
    ax = axes[1]
    ax.hist(gain_shuff_mean, color="gray", alpha=0.75, bins=40)
    ax.axvline(np.mean(gain_real), color="red", linewidth=2)
    ax.set_xlabel("Mean prediction gain (mean over cells)")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: mean gain")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(gain_real):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # E: z-score distribution
    ax = axes[2]
    ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(2.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score vs shuffle (per cell)")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized effect size (gain)")

    # F: effect vs significance
    ax = axes[3]
    ax.scatter(gain_real, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Prediction gain (effect size)")
    ax.set_ylabel(r"$-\log_{10}(p)$ (shuffle)")
    ax.set_title("F. Effect size vs significance (gain)")
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

    fig.suptitle(
        f"{animal_name} {date} {epoch} — Prediction gain summary", fontsize=14, y=1.02
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = analysis_path / "ripple" / "glm_results"
    for mode in ["during", "pre"]:
        for epoch in epoch_list:
            results = fit_ripple_glm(
                epoch=epoch, min_spikes=10, n_shuffles=50, random_seed=1, mode=mode
            )
            out_path = out_dir / f"{epoch}_{mode}_ripple_glm.npz"
            save_results_npz(out_path, results)
            print("Saved:", out_path)

            plot_ripple_glm_pseudo_r2(
                pseudo_r2_real_folds=results["pseudo_r2_real_folds"],
                pseudo_r2_shuff_folds=results["pseudo_r2_shuff_folds"],
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                out_path=analysis_path
                / "figs"
                / "ripple"
                / f"{epoch}_{mode}_pseudo_r2_summary.png",
            )

            plot_ripple_glm_prediction_gain(
                mae_real_folds=results["mae_real_folds"],
                mae_shuff_folds=results["mae_shuff_folds"],
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                out_path=analysis_path
                / "figs"
                / "ripple"
                / f"{epoch}_{mode}_prediction_gain_summary.png",
            )


if __name__ == "__main__":
    main()
