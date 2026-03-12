from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from typing import Iterator, Tuple, Literal, Any, Optional


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


def _first_nonfinite_info(a: np.ndarray) -> Optional[str]:
    a = np.asarray(a)
    mask = ~np.isfinite(a)
    if not np.any(mask):
        return None
    idx = np.argwhere(mask)[0]
    val = a[tuple(idx)]
    return f"shape={a.shape}, first_bad_idx={tuple(idx)}, value={val}"


@dataclass(frozen=True)
class SpikeBalancedKFold1D:
    """
    K-fold splitter that approximately balances the *total spike count* of a
    single neuron across folds.

    This assigns each time bin to a fold via greedy bin packing on y (counts),
    while ensuring each fold has at least one sample.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle tie-breaks (recommended).
    random_state : int
        RNG seed.
    """

    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 0

    def split(self, y_1d: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_idx, test_idx) splits.

        Parameters
        ----------
        y_1d : np.ndarray, shape (n_samples,)
            Nonnegative spike counts for one neuron.

        Yields
        ------
        train_idx, test_idx : tuple[np.ndarray, np.ndarray]
            Integer indices into y_1d.
        """
        y = np.asarray(y_1d)
        if y.ndim != 1:
            raise ValueError("y_1d must be 1D, shape (n_samples,)")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        n_samples = y.shape[0]
        if n_samples < self.n_splits:
            raise ValueError("n_samples must be >= n_splits")
        if np.any(~np.isfinite(y)) or np.any(y < 0):
            raise ValueError("y_1d must be finite and nonnegative")

        rng = np.random.default_rng(self.random_state)

        # Sort bins by descending spike count (heaviest first)
        order = np.argsort(-y, kind="mergesort")

        # Optional shuffle within equal-count runs (stable + random tie breaks)
        if self.shuffle:
            yy = y[order]
            run_starts = np.flatnonzero(np.r_[True, yy[1:] != yy[:-1]])
            run_ends = np.r_[run_starts[1:], yy.size]
            order2 = order.copy()
            for s, e in zip(run_starts, run_ends):
                if e - s > 1:
                    block = order2[s:e].copy()
                    rng.shuffle(block)
                    order2[s:e] = block
            order = order2

        fold_ids = np.full(n_samples, -1, dtype=np.int64)
        fold_spikes = np.zeros(self.n_splits, dtype=np.float64)
        fold_sizes = np.zeros(self.n_splits, dtype=np.int64)

        # Guarantee non-empty folds: assign first K items one per fold
        for k, idx in enumerate(order[: self.n_splits]):
            fold_ids[idx] = k
            fold_spikes[k] += float(y[idx])
            fold_sizes[k] += 1

        # Greedy pack remaining bins by current smallest fold spike total
        for idx in order[self.n_splits :]:
            min_sp = fold_spikes.min()
            candidates = np.flatnonzero(fold_spikes == min_sp)

            # tie-break: smaller fold size
            if candidates.size > 1:
                min_sz = fold_sizes[candidates].min()
                candidates = candidates[fold_sizes[candidates] == min_sz]

            k = (
                int(rng.choice(candidates))
                if (candidates.size > 1 and self.shuffle)
                else int(candidates[0])
            )

            fold_ids[idx] = k
            fold_spikes[k] += float(y[idx])
            fold_sizes[k] += 1

        if np.any(fold_sizes == 0):
            raise ValueError(f"Empty fold produced: fold_sizes={fold_sizes.tolist()}")

        all_idx = np.arange(n_samples)
        for k in range(self.n_splits):
            test_idx = all_idx[fold_ids == k]
            train_idx = all_idx[fold_ids != k]
            yield train_idx, test_idx


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

    info = _first_nonfinite_info(X)
    if info:
        raise ValueError(f"Non-finite in X right after count(): {info}")

    info = _first_nonfinite_info(y)
    if info:
        raise ValueError(f"Non-finite in y right after count(): {info}")

    # global y filter (keeps neuron set fixed)
    keep_y = y.sum(axis=0) >= min_spikes
    y = y[:, keep_y]
    v1_unit_ids_kept = v1_unit_ids[keep_y]
    n_folds = 5
    splitter = SpikeBalancedKFold1D(n_splits=n_folds, shuffle=True, random_state=1)
    rng = np.random.default_rng(random_seed)

    n_neurons = y.shape[1]

    # store per-neuron metrics (per fold)
    pseudo_r2_real_folds = np.full((n_folds, n_neurons), np.nan, dtype=np.float64)
    mae_real_folds = np.full((n_folds, n_neurons), np.nan, dtype=np.float64)

    pseudo_r2_shuff_folds = np.full(
        (n_folds, n_shuffles, n_neurons), np.nan, dtype=np.float64
    )
    mae_shuff_folds = np.full(
        (n_folds, n_shuffles, n_neurons), np.nan, dtype=np.float64
    )

    # Store predictions/observations for the real model (per neuron, per fold)
    yhat_real_per_neuron: list[list[np.ndarray]] = [[] for _ in range(n_neurons)]
    y_test_per_neuron: list[list[np.ndarray]] = [[] for _ in range(n_neurons)]

    # Fold bookkeeping (per neuron, per fold)
    fold_info: list[dict] = []

    for n in range(n_neurons):
        unit_id = v1_unit_ids_kept[n]
        y_n = y[:, n]  # (T,)

        # Build neuron-specific balanced splits
        splits = list(splitter.split(y_n))

        for fold, (train_idx, test_idx) in enumerate(splits):
            if test_idx.size == 0 or train_idx.size == 0:
                raise ValueError(
                    f"Empty split: neuron={n}, fold={fold}, "
                    f"train={train_idx.size}, test={test_idx.size}, n_samples={y.shape[0]}"
                )

            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx, n]  # (T_train, )
            y_test = y[test_idx, n]  # (T_test, )

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

            info = _first_nonfinite_info(X_train)
            if info:
                raise ValueError(
                    f"Non-finite in X_train after preprocessing: neuron={n}, fold={fold}: {info}"
                )
            info = _first_nonfinite_info(X_test)
            if info:
                raise ValueError(
                    f"Non-finite in X_test after preprocessing: neuron={n}, fold={fold}: {info}"
                )

            # -------------------------
            # REAL fit (single neuron)
            # -------------------------
            glm = nmo.glm.GLM(
                solver_name="LBFGS",
                regularizer="Ridge",
                regularizer_strength=0.1,
                solver_kwargs=dict(maxiter=6000, tol=1e-7, stepsize=0),
            )
            glm.fit(X_train, y_train)

            lam_test = glm.predict(X_test)  # (T_test, 1)

            info = _first_nonfinite_info(lam_test)
            if info:
                raise ValueError(
                    f"Non-finite in lam_test (REAL): neuron={n}, fold={fold}: {info}"
                )

            yhat_real_per_neuron[n].append(lam_test)
            y_test_per_neuron[n].append(np.asarray(y_test, dtype=np.float64))

            r2_real = mcfadden_pseudo_r2_per_neuron(
                y_test[:, None], lam_test[:, None], y_train[:, None]
            )[0]
            mae_real = mae_per_neuron(y_test[:, None], lam_test[:, None])[0]

            pseudo_r2_real_folds[fold, n] = r2_real
            mae_real_folds[fold, n] = mae_real

            # -------------------------
            # SHUFFLED null (per neuron)
            # -------------------------
            for s in range(n_shuffles):
                y_train_shuff = y_train.copy()
                rng.shuffle(y_train_shuff)

                glm_s = nmo.glm.GLM(
                    solver_name="LBFGS",
                    regularizer="Ridge",
                    regularizer_strength=0.1,
                    solver_kwargs=dict(maxiter=6000, tol=1e-7, stepsize=0),
                )
                glm_s.fit(X_train, y_train_shuff)

                lam_test_s = glm_s.predict(X_test)
                info = _first_nonfinite_info(lam_test_s)
                if info:
                    raise ValueError(
                        f"Non-finite in lam_test_s: neuron={n}, fold={fold}, shuffle={s}: {info}"
                    )

                pseudo_r2_shuff_folds[fold, s, n] = mcfadden_pseudo_r2_per_neuron(
                    y_test[:, None], lam_test_s[:, None], y_train[:, None]
                )[0]
                mae_shuff_folds[fold, s, n] = mae_per_neuron(
                    y_test[:, None], lam_test_s[:, None]
                )[0]

                del glm_s, lam_test_s, y_train_shuff
                if (s + 1) % 5 == 0:
                    jax.clear_caches()
                    gc.collect()

            fold_info.append(
                dict(
                    neuron_index=int(n),
                    v1_unit_id=unit_id,
                    fold=int(fold),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    keep_x=keep_x,
                    x_mean=mean.ravel(),
                    x_std=std.ravel(),
                    test_total_spikes=float(np.sum(y_test)),
                    train_total_spikes=float(np.sum(y_train)),
                    test_n_bins=int(test_idx.size),
                    train_n_bins=int(train_idx.size),
                )
            )

        # Optional: quick per-neuron summary
        obs_r2 = float(np.nanmean(pseudo_r2_real_folds[:, n]))
        null_r2 = np.nanmean(pseudo_r2_shuff_folds[:, :, n], axis=1)  # (folds,)
        # foldwise null means -> pooled p-value
        p_r2 = float(np.mean(null_r2 >= obs_r2))
        print(
            f"Neuron {n}/{n_neurons} (unit {unit_id}): "
            f"mean pseudo-R2={obs_r2:.4f} (p~{p_r2:.4g})"
        )

    # -------------------------
    # Aggregate across folds/neuron (optional)
    # -------------------------
    r2_real_all = pseudo_r2_real_folds[np.isfinite(pseudo_r2_real_folds)]
    mae_real_all = mae_real_folds[np.isfinite(mae_real_folds)]

    shuff_r2_all_means = np.nanmean(pseudo_r2_shuff_folds, axis=2).ravel()
    shuff_mae_all_means = np.nanmean(mae_shuff_folds, axis=2).ravel()

    print(
        "Overall mean pseudo-R2 (pooled cells/folds):", float(np.nanmean(r2_real_all))
    )
    print("Overall mean MAE (pooled cells/folds):", float(np.nanmean(mae_real_all)))

    print(
        "Overall shuffle p (pseudo-R2):",
        float(np.mean(shuff_r2_all_means >= np.nanmean(r2_real_all))),
    )
    print(
        "Overall shuffle p (MAE; smaller is better):",
        float(np.mean(shuff_mae_all_means <= np.nanmean(mae_real_all))),
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
        yhat_real_folds=yhat_real_per_neuron,  # NOTE: per-neuron list-of-lists
        y_test_folds=y_test_per_neuron,  # NOTE: per-neuron list-of-lists
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
            out_path = out_dir / f"{epoch}_{mode}_single_neuron_ripple_glm.npz"
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
