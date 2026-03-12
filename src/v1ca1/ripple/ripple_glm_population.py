from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
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
from typing import Iterator, Tuple, Literal, Any, Optional, Dict


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


spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])

ep = {}
for epoch in epoch_list:
    ep[epoch] = nap.IntervalSet(
        start=timestamps_ephys[epoch][0],
        end=timestamps_ephys[epoch][-1],
    )


def make_preripple_baseline_ep(
    ripple_ep: nap.IntervalSet,
    epoch_ep: nap.IntervalSet,
    *,
    buffer_s: float = 0.10,
    exclude_ripples: bool = True,
    exclude_ripple_guard_s: float = 0.10,
    duration=None,
) -> nap.IntervalSet:
    """Create a paired pre-ripple baseline IntervalSet.

    For each ripple [t_start, t_end], create a baseline window of equal length
    immediately before ripple start, separated by a buffer:

        [t_start - dur - buffer_s, t_start - buffer_s]

    Parameters
    ----------
    ripple_ep
        Ripple intervals for a single epoch.
    epoch_ep
        The containing epoch interval(s) to clip baseline windows to.
    buffer_s
        Gap between baseline end and ripple start (seconds).
    exclude_ripples
        If True, subtract ripple intervals (optionally expanded by guard).
    exclude_ripple_guard_s
        Guard added around ripples when excluding them (seconds).

    Returns
    -------
    baseline_ep
        IntervalSet of baseline windows.
    """
    r_start = np.asarray(ripple_ep.start, dtype=float)
    r_end = np.asarray(ripple_ep.end, dtype=float)
    if duration is None:
        dur = r_end - r_start
    else:
        dur = duration

    b_start = r_start - dur - buffer_s
    b_end = r_start - buffer_s

    # drop invalid / negative-length windows
    keep = b_end > b_start
    b_start = b_start[keep]
    b_end = b_end[keep]

    if b_start.size == 0:
        return nap.IntervalSet(
            start=np.array([], dtype=float), end=np.array([], dtype=float)
        )

    baseline_ep = nap.IntervalSet(start=b_start, end=b_end)

    # clip to epoch bounds
    baseline_ep = baseline_ep.intersect(epoch_ep)

    if exclude_ripples and len(baseline_ep) > 0 and len(ripple_ep) > 0:
        # exclude ripples (+ optional guard) from baseline
        if exclude_ripple_guard_s > 0:
            rip_excl = nap.IntervalSet(
                start=np.asarray(ripple_ep.start, float) - exclude_ripple_guard_s,
                end=np.asarray(ripple_ep.end, float) + exclude_ripple_guard_s,
            ).intersect(epoch_ep)
        else:
            rip_excl = ripple_ep

        baseline_ep = baseline_ep.set_diff(rip_excl)

    return baseline_ep


def firing_rate_over_intervals(
    spikes: nap.TsGroup,
    intervals: nap.IntervalSet,
) -> np.ndarray:
    """Compute per-unit firing rate over an IntervalSet.

    Returns
    -------
    fr : np.ndarray, shape (n_units,)
        Total spikes in `intervals` divided by total interval duration.
    """
    tot = float(intervals.tot_length())
    if tot <= 0:
        return np.zeros(len(spikes), dtype=float)

    counts = spikes.count(ep=intervals).to_numpy()  # (n_intervals, n_units)
    return counts.sum(axis=0) / tot


# ---------------------------------------------------------------------
# Build baseline firing rates (paired pre-ripple) for all regions/epochs
# ---------------------------------------------------------------------

# baseline_ep: Dict[str, nap.IntervalSet] = {}
# fr_preripple_baseline: Dict[str, Dict[str, np.ndarray]] = {
#     region: {} for region in regions
# }

# for epoch in epoch_list:
#     baseline_ep[epoch] = make_preripple_baseline_ep(
#         ripple_ep=ripple_ep[epoch],
#         epoch_ep=ep[epoch],
#         buffer_s=0.45,  # gap before ripple
#         exclude_ripples=True,  # recommended
#         exclude_ripple_guard_s=0.10,  # also exclude ripple-adjacent time
#     )

# for region in regions:
#     for epoch in epoch_list:
#         fr_preripple_baseline[region][epoch] = firing_rate_over_intervals(
#             spikes=spikes[region],
#             intervals=baseline_ep[epoch],
#         )


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


def save_results_npz(out_path: Path, results: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        # metadata
        epoch=results["epoch"],
        min_spikes_per_ripple=results["min_spikes_per_ripple"],
        n_shuffles=results["n_shuffles"],
        random_seed=results["random_seed"],
        v1_unit_ids=results["v1_unit_ids"],
        ca1_unit_ids=results["ca1_unit_ids"],
        # metrics (store as object arrays to be safe)
        pseudo_r2_real_folds=results["pseudo_r2_real_folds"],
        mae_real_folds=results["mae_real_folds"],
        pseudo_r2_shuff_folds=results["pseudo_r2_shuff_folds"],
        mae_shuff_folds=results["mae_shuff_folds"],
        ll_real_folds=results["ll_real_folds"],  # (F, N)
        ll_shuff_folds=results["ll_shuff_folds"],  # (F, S, N)
        fold_info=np.array(results["fold_info"], dtype=object),
        y_test_folds=results["y_test_folds"],
        yhat_real_folds=results["yhat_real_folds"],
        yhat_shuff_folds=results["yhat_shuff_folds"],
    )


def _first_nonfinite_info(a: np.ndarray) -> Optional[str]:
    a = np.asarray(a)
    mask = ~np.isfinite(a)
    if not np.any(mask):
        return None
    idx = np.argwhere(mask)[0]
    val = a[tuple(idx)]
    return f"shape={a.shape}, first_bad_idx={tuple(idx)}, value={val}"


# BalanceMode = Literal["absolute", "relative"]


# @dataclass(frozen=True)
# class BalancedPerNeuronKFold:
#     """
#     K-fold splitter that approximately balances per-neuron spike counts across folds.

#     This assigns each time bin to a fold using a greedy objective that minimizes
#     the squared error between each fold's per-neuron totals and the target totals.
#     It additionally guarantees that no fold is empty (when n_samples >= n_splits).

#     Parameters
#     ----------
#     n_splits : int
#         Number of folds (K).
#     shuffle : bool
#         Whether to shuffle tie-breaks.
#     random_state : int
#         Seed for deterministic behavior.
#     balance_mode : {"absolute", "relative"}
#         "absolute": balance raw counts (high-rate neurons dominate).
#         "relative": normalize each neuron's total to 1 before balancing.
#     min_total_per_neuron : int
#         Neurons with total spikes < this threshold are ignored in the balancing objective.
#         They are still included in y (this only affects fold assignment).
#     eps : float
#         Numerical stability constant.

#     Notes
#     -----
#     - This expects y to be nonnegative and finite, shape (n_samples, n_neurons).
#     - It is not a standard CV splitter; it's specialized for count matrices where you
#       want per-neuron totals similar across folds.
#     """

#     n_splits: int = 5
#     shuffle: bool = True
#     random_state: int = 0
#     balance_mode: BalanceMode = "relative"
#     min_total_per_neuron: int = 0
#     eps: float = 1e-12

#     def split(self, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
#         """
#         Generate (train_idx, test_idx) splits.

#         Parameters
#         ----------
#         y : np.ndarray, shape (n_samples, n_neurons)
#             Nonnegative count matrix (e.g., spike counts).

#         Yields
#         ------
#         train_idx, test_idx : tuple[np.ndarray, np.ndarray]
#             Integer indices into rows of y.
#         """
#         y = np.asarray(y)
#         if y.ndim != 2:
#             raise ValueError("y must be 2D, shape (n_samples, n_neurons)")
#         if self.n_splits < 2:
#             raise ValueError("n_splits must be >= 2")
#         n_samples, n_neurons = y.shape
#         if n_samples < self.n_splits:
#             raise ValueError("n_samples must be >= n_splits")
#         if np.any(~np.isfinite(y)):
#             raise ValueError("y must be finite")
#         if np.any(y < 0):
#             raise ValueError("y must be nonnegative")

#         rng = np.random.default_rng(self.random_state)

#         totals = y.sum(axis=0)  # (N,)
#         use = totals >= float(self.min_total_per_neuron)
#         if not np.any(use):
#             use = np.ones(n_neurons, dtype=bool)

#         y_use = y[:, use]  # (T, N_use)

#         if self.balance_mode == "relative":
#             # Each neuron's total contributes ~1 overall
#             w = 1.0 / (y_use.sum(axis=0) + self.eps)  # (N_use,)
#             y_eff = y_use * w[None, :]
#         elif self.balance_mode == "absolute":
#             y_eff = y_use.astype(np.float64, copy=False)
#         else:
#             raise ValueError("balance_mode must be 'absolute' or 'relative'")

#         # Target per fold in the effective space
#         target = y_eff.sum(axis=0) / float(self.n_splits)  # (N_use,)

#         # Assign heavier bins first for better greedy behavior
#         bin_weight = y_eff.sum(axis=1)  # (T,)
#         order = np.argsort(-bin_weight, kind="mergesort")

#         # Optional shuffle within equal-weight runs (stable + random tie breaks)
#         if self.shuffle:
#             bw = bin_weight[order]
#             run_starts = np.flatnonzero(np.r_[True, bw[1:] != bw[:-1]])
#             run_ends = np.r_[run_starts[1:], bw.size]
#             order_shuffled = order.copy()
#             for s, e in zip(run_starts, run_ends):
#                 if e - s > 1:
#                     block = order_shuffled[s:e].copy()
#                     rng.shuffle(block)
#                     order_shuffled[s:e] = block
#             order = order_shuffled

#         # diff[k] tracks (current_fold_sum[k] - target)
#         diff = np.repeat(-target[None, :], self.n_splits, axis=0)
#         fold_sizes = np.zeros(self.n_splits, dtype=np.int64)
#         fold_ids = np.full(n_samples, -1, dtype=np.int64)

#         for idx in order:
#             v = y_eff[idx].astype(np.float64, copy=False)  # (N_use,)

#             # Guarantee non-empty folds: force-fill empties first.
#             empty = np.flatnonzero(fold_sizes == 0)
#             if empty.size > 0:
#                 candidates = empty
#             else:
#                 # Objective if we assign to k: ||diff[k] + v||^2
#                 # Compute for all k:
#                 # score[k] = sum_n (diff[k,n] + v[n])^2
#                 score = np.einsum("kn,kn->k", diff + v[None, :], diff + v[None, :])
#                 min_score = score.min()
#                 candidates = np.flatnonzero(score == min_score)

#                 # Secondary tie-break: smaller fold size
#                 if candidates.size > 1:
#                     min_size = fold_sizes[candidates].min()
#                     candidates = candidates[fold_sizes[candidates] == min_size]

#             if candidates.size > 1 and self.shuffle:
#                 k = int(rng.choice(candidates))
#             else:
#                 k = int(candidates[0])

#             fold_ids[idx] = k
#             diff[k] += v
#             fold_sizes[k] += 1

#         counts = np.bincount(fold_ids, minlength=self.n_splits)
#         if np.any(counts == 0):
#             raise ValueError(
#                 f"Empty fold(s) produced: counts={counts.tolist()}. "
#                 "This should not happen; please report with your data stats."
#             )

#         all_idx = np.arange(n_samples)
#         for k in range(self.n_splits):
#             test_idx = all_idx[fold_ids == k]
#             train_idx = all_idx[fold_ids != k]
#             yield train_idx, test_idx

#     def per_neuron_fold_stats(self, y: np.ndarray) -> dict:
#         """
#         Compute per-neuron balance diagnostics across folds.

#         Returns per-neuron fraction-of-total in each fold, plus summary stats.

#         Parameters
#         ----------
#         y : np.ndarray, shape (n_samples, n_neurons)

#         Returns
#         -------
#         stats : dict
#             Keys include:
#             - fold_sizes : list[int]
#             - fold_fracs : np.ndarray, shape (K, N) fraction of each neuron's total in each fold
#             - frac_cv : np.ndarray, shape (N,) coefficient of variation across folds for each neuron
#             - median_frac_cv, p90_frac_cv, p99_frac_cv : float
#             - frac_close_to_uniform : float
#                 Fraction of neurons whose fold fractions are all within +/- 20% of uniform (1/K).
#         """
#         y = np.asarray(y)
#         if y.ndim != 2:
#             raise ValueError("y must be 2D")
#         if np.any(~np.isfinite(y)) or np.any(y < 0):
#             raise ValueError("y must be finite and nonnegative")

#         totals = y.sum(axis=0) + self.eps  # (N,)
#         fold_sizes: list[int] = []
#         fold_fracs: list[np.ndarray] = []

#         for _, test_idx in self.split(y):
#             fold_sizes.append(int(test_idx.size))
#             fold_fracs.append(y[test_idx].sum(axis=0) / totals)

#         fold_fracs_arr = np.stack(fold_fracs, axis=0)  # (K, N)
#         mean_frac = np.mean(fold_fracs_arr, axis=0) + self.eps
#         frac_cv = np.std(fold_fracs_arr, axis=0) / mean_frac  # (N,)

#         uniform = 1.0 / float(self.n_splits)
#         within_20pct = np.all(
#             (fold_fracs_arr >= 0.8 * uniform) & (fold_fracs_arr <= 1.2 * uniform),
#             axis=0,
#         )
#         frac_close_to_uniform = float(np.mean(within_20pct))

#         return {
#             "fold_sizes": fold_sizes,
#             "fold_fracs": fold_fracs_arr,
#             "frac_cv": frac_cv,
#             "median_frac_cv": float(np.median(frac_cv[np.isfinite(frac_cv)])),
#             "p90_frac_cv": float(np.percentile(frac_cv[np.isfinite(frac_cv)], 90)),
#             "p99_frac_cv": float(np.percentile(frac_cv[np.isfinite(frac_cv)], 99)),
#             "frac_close_to_uniform": frac_close_to_uniform,
#         }

#     def print_balance_report(
#         self,
#         y: np.ndarray,
#         *,
#         cv_warn: float = 0.5,
#         max_bad_to_print: int = 15,
#     ) -> None:
#         """
#         Print a human-readable balance report.

#         Parameters
#         ----------
#         y : np.ndarray, shape (n_samples, n_neurons)
#             Count matrix used for splitting (typically your V1 y).
#         cv_warn : float
#             Warn threshold for per-neuron fraction CV across folds.
#         max_bad_to_print : int
#             Maximum number of worst neurons (by CV) to print indices for.
#         """
#         stats = self.per_neuron_fold_stats(y)

#         frac_cv = stats["frac_cv"]
#         finite = np.isfinite(frac_cv)
#         if not np.any(finite):
#             print("Balance report: no finite CV values (unexpected).")
#             return

#         order = np.argsort(frac_cv[finite])[::-1]  # worst first within finite subset
#         finite_idx = np.flatnonzero(finite)
#         worst_neuron_ids = finite_idx[order][:max_bad_to_print]

#         print("BalancedPerNeuronKFold fold_sizes:", stats["fold_sizes"])
#         print(
#             "Per-neuron fold fraction CV: "
#             f"median={stats['median_frac_cv']:.4f}, "
#             f"p90={stats['p90_frac_cv']:.4f}, "
#             f"p99={stats['p99_frac_cv']:.4f}"
#         )
#         print(
#             "Uniformity check: "
#             f"frac_neurons_within_±20%_of_uniform={stats['frac_close_to_uniform']:.3f}"
#         )

#         n_bad = int(np.sum(frac_cv[finite] > cv_warn))
#         if n_bad > 0:
#             print(
#                 f"Warning: {n_bad}/{int(np.sum(finite))} neurons have frac_cv > {cv_warn}. "
#                 f"Worst neuron indices (up to {max_bad_to_print}): {worst_neuron_ids.tolist()}"
#             )


# --- ADD THIS helper near your other helpers (e.g., above fit_ripple_glm) ---


def _append_log_duration_feature(
    X: np.ndarray,
    ripple_ep: nap.IntervalSet,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Append log(ripple duration) as the last feature column.

    Notes
    -----
    - Uses duration in seconds.
    - Centers/scales using TRAIN only to avoid leakage.
    - Returns X_train_aug, X_test_aug.

    Parameters
    ----------
    X : np.ndarray, shape (n_ripples, n_features)
        Existing design matrix (e.g., CA1 counts).
    ripple_ep : nap.IntervalSet
        Ripple intervals corresponding to rows of X.
    train_idx : np.ndarray, shape (n_train,)
    test_idx : np.ndarray, shape (n_test,)
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    X_train_aug : np.ndarray, shape (n_train, n_features + 1)
    X_test_aug : np.ndarray, shape (n_test, n_features + 1)
    """
    starts = np.asarray(ripple_ep.start, dtype=np.float64)
    ends = np.asarray(ripple_ep.end, dtype=np.float64)
    dur = np.clip(ends - starts, eps, None)  # (n_ripples,)
    log_dur = np.log(dur)  # (n_ripples,)

    # split
    log_dur_train = log_dur[train_idx]
    log_dur_test = log_dur[test_idx]

    # # standardize using TRAIN only
    # mu = float(np.mean(log_dur_train))
    # sd = float(np.std(log_dur_train) + eps)
    # log_dur_train = (log_dur_train - mu) / sd
    # log_dur_test = (log_dur_test - mu) / sd

    # append as final column
    X_train_aug = np.concatenate([X[train_idx], log_dur_train[:, None]], axis=1)
    X_test_aug = np.concatenate([X[test_idx], log_dur_test[:, None]], axis=1)

    return X_train_aug, X_test_aug


def fit_ripple_glm(
    epoch,
    min_spikes_per_ripple=0.1,
    n_shuffles=50,
    random_seed=0,
    mode="during",
    ripple_window=None,
):
    if ripple_window is None:
        ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"],
            end=Kay_ripple_detector[epoch]["end_time"],
        )
        pre_ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"] - 0.2,
            end=Kay_ripple_detector[epoch]["start_time"],
        )
        baseline_ep = make_preripple_baseline_ep(
            ripple_ep=ripple_ep,
            epoch_ep=ep[epoch],
            buffer_s=0.45,  # gap before ripple
            exclude_ripples=True,  # recommended
            exclude_ripple_guard_s=0.10,  # also exclude ripple-adjacent time
        )
    else:
        ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"],
            end=Kay_ripple_detector[epoch]["start_time"] + ripple_window,
        )
        pre_ripple_ep = nap.IntervalSet(
            start=Kay_ripple_detector[epoch]["start_time"] - ripple_window,
            end=Kay_ripple_detector[epoch]["start_time"],
        )
        baseline_ep = make_preripple_baseline_ep(
            ripple_ep=ripple_ep,
            epoch_ep=ep[epoch],
            buffer_s=0.45,  # gap before ripple
            exclude_ripples=True,  # recommended
            exclude_ripple_guard_s=0.10,  # also exclude ripple-adjacent time
            duration=ripple_window,
        )

    if mode == "during":
        interval = ripple_ep
    elif mode == "baseline":
        interval = baseline_ep
    elif mode == "pre":
        interval = pre_ripple_ep

    spike_counts = {}
    pre_spike_counts = {}
    baseline_spike_counts = {}
    for region in regions:
        spike_counts[region] = spikes[region].count(ep=ripple_ep)
        pre_spike_counts[region] = pre_spike_counts[region].count(ep=pre_ripple_ep)
        baseline_spike_counts[region] = baseline_spike_counts[region].count(
            ep=baseline_ep
        )

    v1_unit_ids = np.array(list(spikes["v1"].keys()))
    ca1_unit_ids = np.array(list(spikes["ca1"].keys()))

    # -----------------------
    # Data
    # -----------------------
    X = np.asarray(spike_counts["ca1"], dtype=np.float64)
    y = np.asarray(spike_counts["v1"], dtype=np.float64)

    X_pre = np.asarray(pre_spike_counts["ca1"], dtype=np.float64)
    y_pre = np.asarray(pre_spike_counts["v1"], dtype=np.float64)

    X_baseline = np.asarray(baseline_spike_counts["ca1"], dtype=np.float64)
    y_baseline = np.asarray(baseline_spike_counts["v1"], dtype=np.float64)

    info = _first_nonfinite_info(X)
    if info:
        raise ValueError(f"Non-finite in X right after count(): {info}")

    info = _first_nonfinite_info(y)
    if info:
        raise ValueError(f"Non-finite in y right after count(): {info}")

    # global y filter (keeps neuron set fixed)
    n_ripples = y.shape[0]
    keep_y = (y.sum(axis=0) / max(n_ripples, 1)) >= min_spikes_per_ripple
    y = y[:, keep_y]
    v1_unit_ids_kept = v1_unit_ids[keep_y]
    n_cells = y.shape[1]

    if y.shape[1] == 0:
        raise ValueError(
            f"No V1 units passed min_spikes_per_ripple={min_spikes_per_ripple}. "
            f"Try lowering it. n_ripples={n_ripples}"
        )

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    n_folds = kf.get_n_splits(X)
    rng = np.random.default_rng(random_seed)

    # kf = BalancedPerNeuronKFold(
    #     n_splits=5,
    #     shuffle=True,
    #     random_state=1,
    #     balance_mode="relative",
    #     min_total_per_neuron=min_spikes,
    # )

    # kf.print_balance_report(y, cv_warn=0.5)

    # -----------------------
    # Preallocate outputs
    # -----------------------
    # predictions/observations (per ripple)
    y_test_full = np.full((n_ripples, n_cells), np.nan, dtype=np.float32)
    yhat_real_full = np.full((n_ripples, n_cells), np.nan, dtype=np.float32)
    yhat_shuff_full = np.full(
        (n_shuffles, n_ripples, n_cells), np.nan, dtype=np.float32
    )

    # metrics (per fold)
    pseudo_r2_real_full = np.full((n_folds, n_cells), np.nan, dtype=np.float32)
    mae_real_full = np.full((n_folds, n_cells), np.nan, dtype=np.float32)
    ll_real_full = np.full((n_folds, n_cells), np.nan, dtype=np.float32)  # (F, N)

    pseudo_r2_shuff_full = np.full(
        (n_folds, n_shuffles, n_cells), np.nan, dtype=np.float32
    )
    mae_shuff_full = np.full((n_folds, n_shuffles, n_cells), np.nan, dtype=np.float32)
    ll_shuff_full = np.full(
        (n_folds, n_shuffles, n_cells), np.nan, dtype=np.float32
    )  # (F, S, N)

    # Also store fold bookkeeping (very useful later)
    fold_info = []  # list of dicts per fold

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        if test_idx.size == 0 or train_idx.size == 0:
            raise ValueError(
                f"Empty split: fold={fold}, train={train_idx.size}, test={test_idx.size}, "
                f"n_samples={y.shape[0]}"
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # X_train, X_test = _append_log_duration_feature(
        #     X=X,
        #     ripple_ep=ripple_ep,
        #     train_idx=train_idx,
        #     test_idx=test_idx,
        # )

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
            raise ValueError(f"Non-finite in X_train after preprocessing: {info}")

        info = _first_nonfinite_info(X_test)
        if info:
            raise ValueError(f"Non-finite in X_test after preprocessing: {info}")

        info = _first_nonfinite_info(y_train)
        if info:
            raise ValueError(f"Non-finite in y_train: {info}")

        info = _first_nonfinite_info(y_test)
        if info:
            raise ValueError(f"Non-finite in y_test: {info}")

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

        info = _first_nonfinite_info(lam_test)
        if info:
            raise ValueError(f"Non-finite in lam_test (REAL) fold={fold}: {info}")

        # write held-out predictions/observations
        yhat_real_full[test_idx, :] = np.asarray(lam_test, dtype=np.float32)
        y_test_full[test_idx, :] = np.asarray(y_test, dtype=np.float32)

        # metrics for this fold (per neuron)
        r2_real = mcfadden_pseudo_r2_per_neuron(y_test, lam_test, y_train).astype(
            np.float32
        )
        mae_real = mae_per_neuron(y_test, lam_test).astype(np.float32)
        ll_real = poisson_ll_per_neuron(y_test, lam_test).astype(np.float32)  # (N,)

        pseudo_r2_real_full[fold, :] = r2_real
        mae_real_full[fold, :] = mae_real
        ll_real_full[fold, :] = ll_real

        # -------------------------
        # SHUFFLED null: shuffle y_train BEFORE fit (repeat)
        # -------------------------
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
            info = _first_nonfinite_info(lam_test_s)
            if info:
                raise ValueError(
                    f"Non-finite in lam_test_s fold={fold} shuffle={s}: {info}"
                )

            # yhat_shuff_fold[s] = np.asarray(lam_test_s, dtype=np.float32)
            yhat_shuff_full[s, test_idx, :] = np.asarray(lam_test_s, dtype=np.float32)
            # metrics
            pseudo_r2_shuff_full[fold, s, :] = mcfadden_pseudo_r2_per_neuron(
                y_test, lam_test_s, y_train
            ).astype(np.float32)
            mae_shuff_full[fold, s, :] = mae_per_neuron(y_test, lam_test_s).astype(
                np.float32
            )
            ll_shuff_full[fold, s, :] = poisson_ll_per_neuron(
                y_test, lam_test_s
            ).astype(np.float32)

            # ---- cleanup ----
            del glm_s, lam_test_s, y_train_shuff
            if (s + 1) % 5 == 0:
                jax.clear_caches()
                gc.collect()

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
        obs_r2_mean = float(np.nanmean(pseudo_r2_real_full[fold]))
        shuff_r2_means = np.nanmean(pseudo_r2_shuff_full[fold], axis=1)
        p_r2 = float(np.mean(shuff_r2_means >= obs_r2_mean))

        obs_mae_mean = float(np.nanmean(mae_real_full[fold]))
        shuff_mae_means = np.nanmean(mae_shuff_full[fold], axis=1)
        p_mae = float(np.mean(shuff_mae_means <= obs_mae_mean))

        print(
            f"Fold {fold}: "
            f"mean pseudo-R2={obs_r2_mean:.4f} (p={p_r2:.4g}) | "
            f"mean MAE={obs_mae_mean:.4f} (p={p_mae:.4g}) | "
            f"N={n_cells}"
        )

    # sanity checks
    if np.isnan(y_test_full).any():
        raise ValueError(
            "y_test_full still has NaNs: some ripples were never assigned to a test fold"
        )
    if np.isnan(yhat_real_full).any():
        raise ValueError(
            "yhat_real_full still has NaNs: some ripples were never predicted in the real model"
        )
    if np.isnan(yhat_shuff_full).any():
        raise ValueError(
            "yhat_shuff_full still has NaNs: some shuffle predictions were never written"
        )
    if np.isnan(pseudo_r2_real_full).any() or np.isnan(mae_real_full).any():
        raise ValueError(
            "Real metric arrays contain NaNs: some fold/cell entries were never written"
        )
    if np.isnan(pseudo_r2_shuff_full).any() or np.isnan(mae_shuff_full).any():
        raise ValueError(
            "Shuffle metric arrays contain NaNs: some fold/shuffle/cell entries were never written"
        )

    results = dict(
        epoch=epoch,
        min_spikes_per_ripple=min_spikes_per_ripple,
        n_shuffles=n_shuffles,
        random_seed=random_seed,
        v1_unit_ids=v1_unit_ids_kept,
        ca1_unit_ids=ca1_unit_ids,
        # metrics (now preallocated arrays)
        pseudo_r2_real_folds=pseudo_r2_real_full,  # (F, N)
        mae_real_folds=mae_real_full,  # (F, N)
        pseudo_r2_shuff_folds=pseudo_r2_shuff_full,  # (F, S, N)
        mae_shuff_folds=mae_shuff_full,  # (F, S, N)
        ll_real_folds=ll_real_full,
        ll_shuff_folds=ll_shuff_full,
        fold_info=fold_info,
        # predictions/observations (preallocated arrays)
        yhat_real_folds=yhat_real_full,  # (R, N)
        y_test_folds=y_test_full,  # (R, N)
        yhat_shuff_folds=yhat_shuff_full,  # (S, R, N)
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


# def plot_ripple_glm_prediction_gain(
#     *,
#     mae_real_folds: Any,
#     mae_shuff_folds: Any,
#     animal_name: str,
#     date: str,
#     epoch: str,
#     out_path: Path,
#     alpha: float = 0.7,
# ) -> None:
#     """
#     Plot prediction-gain-focused panels and save as a single figure.

#     Definitions
#     -----------
#     MAE per cell is averaged over time, then averaged over CV folds.
#     Prediction gain per cell:
#         gain = MAE_shuffle / MAE_real
#     (so >1 means the real model improves over shuffle)

#     Panels
#     ------
#     A: Histogram of prediction gain (per cell)
#     B: Shuffle null distribution of mean gain (mean over cells), with observed mean line
#     E: Z-score histogram (per cell, vs per-cell shuffle gain distribution)
#     F: Effect size vs significance scatter (gain vs -log10 p)
#     """
#     mae_real_folds = _as_2d_float(mae_real_folds)  # (F, N)
#     mae_shuff_folds = _as_3d_float(mae_shuff_folds)  # (F, S, N)

#     # per-cell real MAE (mean over folds)
#     mae_real = np.mean(mae_real_folds, axis=0)  # (N,)

#     # per-cell shuffled MAE samples pooled across folds: (F*S, N)
#     mae_shuff_per_cell = mae_shuff_folds.reshape(-1, mae_shuff_folds.shape[-1])

#     # per-cell shuffled MAE mean (for gain computation)
#     mae_shuff_mean = np.mean(mae_shuff_per_cell, axis=0)  # (N,)

#     # prediction gain per cell (using mean shuffled MAE / real MAE)
#     gain_real = mae_shuff_mean / (mae_real + 1e-12)

#     # per-cell gain null distribution: gain_shuff_sample = MAE_shuff_sample / MAE_real
#     gain_shuff_per_cell = mae_shuff_per_cell / (mae_real[None, :] + 1e-12)  # (F*S, N)

#     # population mean gain null distribution (one value per null sample)
#     gain_shuff_mean = np.mean(gain_shuff_per_cell, axis=1)  # (F*S,)

#     # per-cell z-scores vs gain null
#     shuff_mu = np.mean(gain_shuff_per_cell, axis=0)
#     shuff_sd = np.std(gain_shuff_per_cell, axis=0) + 1e-12
#     z_per_cell = (gain_real - shuff_mu) / shuff_sd

#     # per-cell p-values: P(null gain >= observed gain) (one-sided)
#     n_null = gain_shuff_per_cell.shape[0]
#     p_per_cell = (np.sum(gain_shuff_per_cell >= gain_real[None, :], axis=0) + 1) / (
#         n_null + 1
#     )
#     neglog10_p = -np.log10(p_per_cell)

#     fig, axes = plt.subplots(
#         nrows=2,
#         ncols=2,
#         figsize=(12, 8),
#         constrained_layout=True,
#     )
#     axes = axes.ravel()

#     # A: gain distribution
#     ax = axes[0]
#     ax.hist(
#         gain_real,
#         bins=np.linspace(0.5, 2.0, 31),
#         weights=np.ones_like(gain_real) / len(gain_real),
#         alpha=alpha,
#     )
#     ax.axvline(1.0, color="red", linestyle="--")
#     ax.set_xlabel("Prediction gain per cell (MAE shuffle / MAE real)")
#     ax.set_ylabel("Fraction")
#     ax.set_title("A. Prediction gain distribution")
#     ax.text(
#         0.98,
#         0.95,
#         f"Mean = {np.mean(gain_real):.4f}\nMedian = {np.median(gain_real):.4f}",
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=10,
#         bbox=dict(
#             boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8
#         ),
#     )

#     # B: null distribution of mean gain (over cells)
#     ax = axes[1]
#     ax.hist(gain_shuff_mean, color="gray", alpha=0.75, bins=40)
#     ax.axvline(np.mean(gain_real), color="red", linewidth=2)
#     ax.set_xlabel("Mean prediction gain (mean over cells)")
#     ax.set_ylabel("Count")
#     ax.set_title("B. Shuffle null: mean gain")
#     ax.text(
#         0.98,
#         0.95,
#         f"Observed mean = {np.mean(gain_real):.4f}",
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=10,
#     )

#     # E: z-score distribution
#     ax = axes[2]
#     ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
#     ax.axvline(0.0, color="black", linewidth=1)
#     ax.axvline(2.0, color="red", linestyle="--", linewidth=1)
#     ax.set_xlabel("Z-score vs shuffle (per cell)")
#     ax.set_ylabel("Count")
#     ax.set_title("E. Standardized effect size (gain)")

#     # F: effect vs significance
#     ax = axes[3]
#     ax.scatter(gain_real, neglog10_p, s=12, alpha=0.6)
#     ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
#     ax.axvline(1.0, color="red", linestyle=":", linewidth=1)
#     ax.set_xlabel("Prediction gain (effect size)")
#     ax.set_ylabel(r"$-\log_{10}(p)$ (shuffle)")
#     ax.set_title("F. Effect size vs significance (gain)")
#     frac_sig = float(np.mean(p_per_cell < 0.05))
#     ax.text(
#         0.98,
#         0.05,
#         f"frac p<0.05 = {frac_sig:.3f}",
#         transform=ax.transAxes,
#         ha="right",
#         va="bottom",
#         fontsize=10,
#         bbox=dict(
#             boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8
#         ),
#     )

#     fig.suptitle(
#         f"{animal_name} {date} {epoch} — Prediction gain summary", fontsize=14, y=1.02
#     )

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)


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
    Per-cell real MAE:
        mae_real_cell = mean over CV folds of MAE (per cell)

    Per-cell gain (effect size):
        gain_cell = mean_null_mae_cell / mae_real_cell
    where mean_null_mae_cell is the mean of shuffled MAE samples pooled across folds.

    Per-cell gain null distribution:
        gain_null_sample = mae_shuff_sample / mae_real_cell
    where mae_shuff_sample is a single shuffled MAE sample (pooled across folds).

    Panels
    ------
    A: Per-cell gain histogram (mean over CV folds; null-mean / real)
    B: Shuffle null distribution of population mean gain (mean over cells), with observed mean line
    D: ECDF real vs shuffle (per-cell null samples pooled)
    E: Z-score histogram (per cell, vs per-cell null)
    F: Effect size vs significance scatter (gain vs -log10 p)
    """
    mae_real_folds = _as_2d_float(mae_real_folds)  # (F, N)
    mae_shuff_folds = _as_3d_float(mae_shuff_folds)  # (F, S, N)

    # per-cell real MAE (mean across folds)
    mae_real = np.mean(mae_real_folds, axis=0)  # (N,)
    mae_real = np.clip(mae_real, 1e-12, None)

    # per-cell shuffled MAE samples pooled across folds: (F*S, N)
    mae_shuff_per_cell = mae_shuff_folds.reshape(-1, mae_shuff_folds.shape[-1])

    # observed effect size per cell (use mean shuffled MAE / real MAE)
    mae_shuff_mean = np.mean(mae_shuff_per_cell, axis=0)  # (N,)
    gain_real = mae_shuff_mean / mae_real  # (N,)

    # per-cell gain null distribution (F*S, N)
    gain_shuff_per_cell = mae_shuff_per_cell / mae_real[None, :]

    # population mean gain null distribution: (F*S,)
    gain_shuff_mean = np.mean(gain_shuff_per_cell, axis=1)

    # per-cell z-scores vs shuffle gain null
    shuff_mu = np.mean(gain_shuff_per_cell, axis=0)
    shuff_sd = np.std(gain_shuff_per_cell, axis=0) + 1e-12
    z_per_cell = (gain_real - shuff_mu) / shuff_sd

    # per-cell p-values vs shuffle (one-sided: null >= observed)
    n_null = gain_shuff_per_cell.shape[0]
    p_per_cell = (np.sum(gain_shuff_per_cell >= gain_real[None, :], axis=0) + 1) / (
        n_null + 1
    )
    neglog10_p = -np.log10(p_per_cell)

    # ECDF shuffle uses pooled per-cell null values
    gain_shuffled_flat = gain_shuff_per_cell.ravel()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # A: per-cell gain histogram
    ax = axes[0]
    ax.hist(
        gain_real,
        bins=60,
        weights=np.ones_like(gain_real) / len(gain_real),
        alpha=alpha,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction gain per cell (MAE shuffle / MAE real)")
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell prediction gain")

    stats_text = (
        f"Mean:   {np.mean(gain_real):.4f}\n"
        f"Median: {np.median(gain_real):.4f}\n"
        f"25%:    {np.percentile(gain_real, 25):.4f}\n"
        f"75%:    {np.percentile(gain_real, 75):.4f}"
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

    # B: shuffle null distribution of population mean gain
    ax = axes[1]
    ax.hist(gain_shuff_mean, alpha=0.75, bins=40)
    ax.axvline(np.mean(gain_real), linewidth=2)
    ax.set_xlabel("Mean prediction gain (mean over cells & CV folds)")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: mean prediction gain")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(gain_real):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # D: ECDF real vs shuffle
    ax = axes[2]
    xr, yr = _ecdf(gain_real)
    xs, ys = _ecdf(gain_shuffled_flat)
    ax.plot(xr, yr, linewidth=2, label="Real")
    ax.plot(xs, ys, linewidth=2, label="Shuffle")
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction gain")
    ax.set_ylabel("ECDF")
    ax.set_title("D. ECDF: real vs shuffle")
    ax.legend(frameon=False)

    # E: z-score histogram
    ax = axes[3]
    ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score vs shuffle (per cell)")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized effect size (gain)")

    # F: effect vs significance scatter
    ax = axes[4]
    ax.scatter(gain_real, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.axvline(1.0, linestyle=":", linewidth=1)
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

    # Notes panel
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Notes:\n"
        "- gain = MAE_shuffle / MAE_real (so >1 means improvement).\n"
        "- p-values are one-sided: P(null gain ≥ observed gain).\n"
        "- Z-score uses per-cell null mean/std.\n"
        "- ECDF shuffle pools per-cell null values.",
        ha="left",
        va="top",
        fontsize=10,
        transform=axes[5].transAxes,
    )

    fig.suptitle(
        f"{animal_name} {date} {epoch} — Prediction gain summary", fontsize=14, y=1.02
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ripple_glm_delta_ll(
    *,
    ll_real_folds: Any,
    ll_shuff_folds: Any,
    y_test_folds: np.ndarray,
    animal_name: str,
    date: str,
    epoch: str,
    out_path: Path,
    alpha: float = 0.7,
    normalize: Literal["none", "bits_per_spike"] = "bits_per_spike",
) -> None:
    """
    Plot log-likelihood improvement (real vs shuffle) like the pseudo-R2 summary.

    Parameters
    ----------
    ll_real_folds : array-like, shape (F, N)
        Held-out Poisson LL per fold per cell for the real model.
    ll_shuff_folds : array-like, shape (F, S, N)
        Held-out Poisson LL per fold per cell for each shuffle model.
    y_test_folds : np.ndarray, shape (R, N)
        Held-out observed counts for each ripple and cell (filled across folds).
        Used only for bits/spike normalization.
    normalize : {"none", "bits_per_spike"}
        If "bits_per_spike", divides ΔLL by total spikes and ln(2).
    """
    ll_real_folds = _as_2d_float(ll_real_folds)  # (F, N)
    ll_shuff_folds = _as_3d_float(ll_shuff_folds)  # (F, S, N)

    # ΔLL per fold and cell: (F, N)
    ll_shuff_mean_folds = np.mean(ll_shuff_folds, axis=1)  # (F, N)
    delta_ll_folds = ll_real_folds - ll_shuff_mean_folds  # (F, N)

    # effect size per cell (mean across folds): (N,)
    delta_ll_real = np.mean(delta_ll_folds, axis=0)

    # null distribution per cell: use per-shuffle ΔLL pooled across folds: (F*S, N)
    ll_shuff_per_cell = ll_shuff_folds.reshape(-1, ll_shuff_folds.shape[-1])  # (F*S, N)
    ll_real_rep = np.repeat(ll_real_folds, ll_shuff_folds.shape[1], axis=0)  # (F*S, N)
    delta_ll_shuff_per_cell = ll_real_rep - ll_shuff_per_cell  # (F*S, N)

    # bits/spike normalization (optional)
    if normalize == "bits_per_spike":
        spikes_per_cell = np.sum(y_test_folds, axis=0).astype(np.float64)  # (N,)
        denom = np.clip(spikes_per_cell, 1.0, None) * np.log(2.0)  # avoid /0
        delta_ll_real = delta_ll_real / denom
        delta_ll_shuff_per_cell = delta_ll_shuff_per_cell / denom[None, :]
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'bits_per_spike'")

    # population mean null distribution: (F*S,)
    delta_ll_shuff_mean = np.mean(delta_ll_shuff_per_cell, axis=1)

    # per-cell z-scores vs null
    shuff_mu = np.mean(delta_ll_shuff_per_cell, axis=0)
    shuff_sd = np.std(delta_ll_shuff_per_cell, axis=0) + 1e-12
    z_per_cell = (delta_ll_real - shuff_mu) / shuff_sd

    # per-cell p-values: P(null >= observed) (one-sided)
    n_null = delta_ll_shuff_per_cell.shape[0]
    p_per_cell = (
        np.sum(delta_ll_shuff_per_cell >= delta_ll_real[None, :], axis=0) + 1
    ) / (n_null + 1)
    neglog10_p = -np.log10(p_per_cell)

    # ECDF shuffle uses pooled per-cell null values
    shuffled_flat = delta_ll_shuff_per_cell.ravel()

    x_label = "ΔLL (real - shuffle)"
    if normalize == "bits_per_spike":
        x_label = "Δ bits/spike (real - shuffle)"

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # A: per-cell distribution
    ax = axes[0]
    ax.hist(
        delta_ll_real,
        bins=60,
        weights=np.ones_like(delta_ll_real) / len(delta_ll_real),
        alpha=alpha,
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction")
    ax.set_title("A. Per-cell improvement")

    stats_text = (
        f"Mean:   {np.mean(delta_ll_real):.4f}\n"
        f"Median: {np.median(delta_ll_real):.4f}\n"
        f"25%:    {np.percentile(delta_ll_real, 25):.4f}\n"
        f"75%:    {np.percentile(delta_ll_real, 75):.4f}"
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
    ax.hist(delta_ll_shuff_mean, alpha=0.75, bins=40)
    ax.axvline(np.mean(delta_ll_real), linewidth=2)
    ax.set_xlabel(f"Mean {x_label} (mean over cells)")
    ax.set_ylabel("Count")
    ax.set_title("B. Shuffle null: mean improvement")
    ax.text(
        0.98,
        0.95,
        f"Observed mean = {np.mean(delta_ll_real):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # D: ECDF
    ax = axes[2]
    xr, yr = _ecdf(delta_ll_real)
    xs, ys = _ecdf(shuffled_flat)
    ax.plot(xr, yr, linewidth=2, label="Real")
    ax.plot(xs, ys, linewidth=2, label="Shuffle")
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("ECDF")
    ax.set_title("D. ECDF: real vs shuffle")
    ax.legend(frameon=False)

    # E: z-score
    ax = axes[3]
    ax.hist(z_per_cell[np.isfinite(z_per_cell)], bins=50, alpha=0.85)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Z-score vs shuffle (per cell)")
    ax.set_ylabel("Count")
    ax.set_title("E. Standardized effect size")

    # F: effect vs significance
    ax = axes[4]
    ax.scatter(delta_ll_real, neglog10_p, s=12, alpha=0.6)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle=":", linewidth=1)
    ax.set_xlabel(x_label)
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

    # notes
    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Notes:\n"
        "- Δ = LL(real) - LL(shuffle) on held-out data.\n"
        "- p-values are one-sided: P(null ≥ observed).\n"
        "- ECDF shuffle pools per-cell null values.\n",
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


def main():
    out_dir = analysis_path / "ripple" / "glm_results"
    # ripple_window = 0.2
    for epoch in epoch_list:
        for mode in ["during", "baseline"]:
            results = fit_ripple_glm(
                epoch=epoch,
                min_spikes_per_ripple=0.05,
                n_shuffles=100,
                random_seed=47,
                mode=mode,
                ripple_window=None,
            )
            out_path = out_dir / f"{epoch}_{mode}_population_ripple_glm.npz"
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
            plot_ripple_glm_delta_ll(
                ll_real_folds=results["ll_real_folds"],
                ll_shuff_folds=results["ll_shuff_folds"],
                y_test_folds=results["y_test_folds"],
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                out_path=analysis_path
                / "figs"
                / "ripple"
                / f"{epoch}_{mode}_delta_bits_per_spike.png",
                normalize="bits_per_spike",
            )


if __name__ == "__main__":
    main()
