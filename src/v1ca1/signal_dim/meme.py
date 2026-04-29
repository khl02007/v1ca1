from __future__ import annotations

"""Estimate MEME-based signal dimensionality from spatial tuning curves.

This module adapts the MEME framework of Pospisil and Pillow (2025) from
repeated neural responses to visual stimuli to repeated estimates of spatial
tuning over linearized position. In the paper, each condition is a stimulus and
each repeat is another presentation of that stimulus. Here, each condition is a
retained spatial bin within a trajectory class, and each repeat is a tuning
curve estimated from a disjoint group of laps.

The goal of these changes is to construct the closest spatial analog of the
paper's repeat-by-condition matrix. Single laps are often too sparse and noisy
to serve as stable repeats, so laps are pooled into disjoint groups before
tuning curves are computed. Using multiple groups also provides multiple
cross-group pairings, which helps stabilize the moment estimates. Trajectories
are linearized and occupancy-filtered separately, and only then are valid bins
concatenated into the condition axis so that all repeats are compared on the
same set of well-sampled spatial conditions.

Mean removal follows the paper's disjoint-differencing idea rather than
subtracting the empirical mean across bins. Because spatial bins have a
meaningful order and neighboring bins are correlated, this implementation uses
random disjoint bin pairings instead of a fixed odd/even split. The aim is to
remove the global mean without turning preprocessing into a local spatial
difference operator.

This is therefore a spatial analog of MEME, not a literal reproduction of the
paper's visual-stimulus design. Uncertainty is estimated by rerunning the full
pipeline with different random seeds, so reported intervals reflect sensitivity
to random lap grouping, disjoint condition pairings, and MEME Monte Carlo
choices.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pynapple as nap
import track_linearization as tl

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.wtrack import get_wtrack_branch_side
from v1ca1.signal_dim._shared import (
    DEFAULT_BIN_SIZE_CM,
    DEFAULT_DATA_ROOT,
    DEFAULT_MIN_OCCUPANCY_S,
    DEFAULT_N_GROUPS,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REGIONS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    TRAJECTORY_TYPES,
    get_light_and_dark_epochs,
    get_signal_dim_figure_dir,
    get_signal_dim_output_dir,
    load_signal_dim_session,
    occupancy_mask_1d,
    sample_lap_indices,
    select_run_epochs,
    split_lap_indices_into_groups,
)
from v1ca1.signal_dim.cv_pca import (
    DEFAULT_MIN_CONDITION_SD_HZ,
    DEFAULT_REGION_FR_THRESHOLDS,
    DEFAULT_UNIT_FILTER_MODE,
    classify_and_select_units,
)


DEFAULT_DARK_EPOCH = "08_r4"
DEFAULT_N_RANDOM_REPEATS = 200
DEFAULT_RANDOM_REPEAT_CI = 95.0
DEFAULT_FULL_N_PAIRINGS = 1000
DEFAULT_FULL_N_BIN_PERMS = 5
DEFAULT_RANDOM_REPEAT_N_PAIRINGS = 200
DEFAULT_RANDOM_REPEAT_N_BIN_PERMS = 5
DEFAULT_GROUPMEAN_N_DRAWS = 5000
DEFAULT_MEME_OUTPUT_DIRNAME = "meme"

# Shape notation used throughout this module:
#   R = number of disjoint repeat groups of laps
#   C = number of retained spatial conditions after concatenating trajectory bins
#   N = number of neurons
#   F[r, c, n] = tuning-curve value for neuron n in condition c for repeat group r
# This mirrors the paper's repeat-by-stimulus formulation, but here conditions are
# spatial bins and repeats are grouped laps rather than repeated image presentations.


ArrayF = npt.NDArray[np.floating]
load_meme_session = load_signal_dim_session
_sample_lap_indices = sample_lap_indices
_split_lap_indices_into_groups = split_lap_indices_into_groups
_occupancy_mask_1d = occupancy_mask_1d


@dataclass(frozen=True)
class PRSummary:
    pr_center: float
    ci_low: float
    ci_high: float
    n_eff: int
    center: Literal["mean", "median"] = "mean"
    ci: float = 95.0


def summarize_mc_pr(
    mc_pr: ArrayF,
    *,
    ci: float = 95.0,
    center: Literal["mean", "median"] = "mean",
) -> PRSummary:
    """
    Summarize a 1D Monte Carlo distribution of PR values.
    Returns center + percentile CI.
    """
    mc_pr = np.asarray(mc_pr, dtype=np.float64)
    mc_pr = mc_pr[np.isfinite(mc_pr)]
    n = int(mc_pr.size)

    if n == 0:
        return PRSummary(
            pr_center=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_eff=0,
            center=center,
            ci=ci,
        )

    if center == "mean":
        cval = float(np.mean(mc_pr))
    elif center == "median":
        cval = float(np.median(mc_pr))
    else:
        raise ValueError("center must be 'mean' or 'median'.")

    alpha = (100.0 - float(ci)) / 2.0
    lo, hi = np.percentile(mc_pr, [alpha, 100.0 - alpha])

    return PRSummary(
        pr_center=cval,
        ci_low=float(lo),
        ci_high=float(hi),
        n_eff=n,
        center=center,
        ci=float(ci),
    )


def prepare_F(
    session: dict[str, Any],
    epoch: str,
    region: str,
    *,
    dark_epoch: str = DEFAULT_DARK_EPOCH,
    unit_mask: Optional[np.ndarray] = None,
    bin_size: float = DEFAULT_BIN_SIZE_CM,
    n_groups: int = DEFAULT_N_GROUPS,
    group_seed: int = 0,
    min_occupancy_s: float = DEFAULT_MIN_OCCUPANCY_S,
    lap_fraction: float = 1.0,
) -> ArrayF:
    """Build the MEME input tensor ``F`` for one region and epoch.

    ``F`` has shape ``(R, C, N)``:
    - ``R`` is the number of disjoint repeat groups of laps
    - ``C`` is the number of retained spatial conditions after concatenating bins
      from all trajectory types
    - ``N`` is the number of neurons

    This is the main place where the paper's repeated-stimulus setting is
    adapted to navigation. In the paper, one observes repeated responses to the
    same external stimuli. Here, we instead estimate repeated spatial tuning
    curves from repeated laps. Each repeat is not a single lap, but a disjoint
    group of laps. Each group average acts as a more reliable repeat estimate
    than any individual lap would, because single-lap occupancy and spike-count
    fluctuations are often too noisy for stable high-dimensional covariance
    estimation. Each condition is not an image identity, but a linearized
    spatial bin within a trajectory class.

    Implementation details that matter scientifically:
    - tuning curves are computed only during movement
    - trajectory types are linearized separately before concatenation
    - bins with insufficient occupancy are removed using the intersection across
      repeat groups, so all repeats share the same retained condition axis
    - repeat groups are formed from disjoint laps so the resulting repeats are
      less noisy but still approximately independent
    - optional neuron filtering is applied through an explicit mask, keeping the
      cell inclusion rule fixed across compared epochs
    - if ``lap_fraction < 1``, each lap-resampling draw first subsamples laps before
      constructing repeat groups

    The returned tensor therefore contains grouped estimates of noise-reduced
    spatial tuning, not raw spike counts or single-lap responses.
    """
    rng = np.random.default_rng(group_seed)
    position_offset = int(session["position_offset"])
    trajectory_times = session["trajectory_times"]
    position_dict = session["position_dict"]
    timestamps_position_dict = session["timestamps_position_dict"]
    spikes = session["spikes_by_region"]
    movement_epoch = session["movement_by_epoch"][epoch]
    track_total_length = float(session["track_total_length"])
    track_graphs_by_side = session["track_graphs_by_side"]
    edge_orders_by_side = session["edge_orders_by_side"]
    linear_edge_spacing = session["linear_edge_spacing"]

    # Convert saved trajectory start/stop times into per-trajectory lap intervals.
    traj_starts: Dict[str, np.ndarray] = {}
    traj_ends: Dict[str, np.ndarray] = {}
    n_trials: Dict[str, int] = {}
    for traj in TRAJECTORY_TYPES:
        starts = np.asarray(trajectory_times[epoch][traj][:, 0], dtype=np.float64)
        ends = np.asarray(trajectory_times[epoch][traj][:, -1], dtype=np.float64)
        traj_starts[traj] = starts
        traj_ends[traj] = ends
        n_trials[traj] = len(starts)

    for traj, n_available in n_trials.items():
        if n_available < n_groups:
            raise ValueError(
                f"n_groups={n_groups} exceeds available laps for {traj}: "
                f"n_available={n_available}."
            )

    # Linearize each trajectory class separately so left/right branches retain the
    # intended spatial ordering before their bins are later concatenated.
    linear_position: Dict[str, nap.Tsd] = {}
    for traj in TRAJECTORY_TYPES:
        branch_side = get_wtrack_branch_side(traj)
        track_graph = track_graphs_by_side[branch_side]

        position_df = tl.get_linearized_position(
            position=position_dict[epoch][position_offset:],
            track_graph=track_graph,
            edge_order=edge_orders_by_side[branch_side],
            edge_spacing=linear_edge_spacing,
        )

        traj_ep = nap.IntervalSet(start=traj_starts[traj], end=traj_ends[traj])

        linear_position[traj] = nap.Tsd(
            t=timestamps_position_dict[epoch][position_offset:],
            d=position_df["linear_position"],
            time_support=traj_ep,
        )

    # Spatial conditions are linearized bins spanning one branch traversal.
    bin_edges = np.arange(
        0,
        track_total_length + bin_size,
        bin_size,
        dtype=np.float64,
    )
    n_bins = len(bin_edges) - 1
    if n_bins < 2:
        raise ValueError("binning produced <2 bins; decrease bin_size?")

    # For each trajectory, build repeat groups from disjoint laps and estimate
    # one tuning curve per group. Averaging within a group makes each repeat
    # less noisy than a single lap, which is important because the estimator is
    # trying to recover signal geometry rather than single-traversal variability.
    tc_store: Dict[str, List[np.ndarray]] = {traj: [] for traj in TRAJECTORY_TYPES}
    occ_masks: Dict[str, List[np.ndarray]] = {traj: [] for traj in TRAJECTORY_TYPES}

    for traj in TRAJECTORY_TYPES:
        selected_laps = _sample_lap_indices(
            n_trials[traj],
            lap_fraction=lap_fraction,
            n_groups=n_groups,
            rng=rng,
        )
        lap_groups = _split_lap_indices_into_groups(selected_laps, n_groups, rng=rng)

        for g_inds in lap_groups:
            g_inds = np.sort(g_inds)
            g_starts = traj_starts[traj][g_inds]
            g_ends = traj_ends[traj][g_inds]
            g_ep = nap.IntervalSet(start=g_starts, end=g_ends)

            # Restrict to movement so occupancy and firing rates reflect active
            # traversal of the track rather than immobility periods.
            use_ep = g_ep.intersect(movement_epoch)

            # Tuning curve for one repeat group, shape (neurons, bins).
            tc = nap.compute_tuning_curves(
                data=spikes[region],
                features=linear_position[traj],
                bins=[bin_edges],
                epochs=use_ep,
                feature_names=["linpos"],
            )
            tc_np = np.asarray(tc.to_numpy(), dtype=np.float64)  # (N, n_bins)

            # Keep track of which bins were sufficiently sampled in this group.
            mask = _occupancy_mask_1d(
                linear_position[traj],
                use_ep,
                bin_edges,
                min_occupancy_s=min_occupancy_s,
            )

            if mask.shape[0] != n_bins:
                raise RuntimeError("Occupancy mask length mismatch with bins.")

            tc_store[traj].append(tc_np)
            occ_masks[traj].append(mask)

        # sanity
        if len(tc_store[traj]) != n_groups:
            raise RuntimeError("Unexpected number of groups; check grouping logic.")

    # Keep only bins that are reliable in every repeat group so the condition
    # axis is aligned across repeats before running MEME.
    keep_masks: Dict[str, np.ndarray] = {}
    for traj in TRAJECTORY_TYPES:
        keep = np.logical_and.reduce(occ_masks[traj])
        if keep.sum() < 2:
            raise ValueError(
                f"Too few valid bins after occupancy filtering for {traj}: "
                f"{keep.sum()} kept. Increase laps per group, increase bin_size, "
                f"or lower min_occupancy_s."
            )
        keep_masks[traj] = keep

    # Concatenate trajectory-specific tuning curves into one shared condition
    # axis. This is the spatial analog of stacking all stimulus conditions.
    F_list: List[np.ndarray] = []
    for g in range(n_groups):
        pieces = []
        for traj in TRAJECTORY_TYPES:
            tc_np = tc_store[traj][g]  # (N, n_bins)
            tc_np = np.nan_to_num(tc_np, nan=0.0)  # should be rare after filtering
            kept = keep_masks[traj]
            # transpose to (bins, neurons)
            pieces.append(tc_np[:, kept].T)
        Xg = np.concatenate(pieces, axis=0)  # (C, N)
        F_list.append(Xg)

    F = np.stack(F_list, axis=0)  # (R, C, N)

    # Apply a caller-provided inclusion rule so light/dark comparisons are not
    # driven by epoch-specific cell selection.
    if unit_mask is not None:
        keep_neurons = np.asarray(unit_mask, dtype=bool).reshape(-1)
        if keep_neurons.size != F.shape[-1]:
            raise ValueError(
                "unit_mask must have one value per unit. "
                f"Got {keep_neurons.size} mask values for {F.shape[-1]} units."
            )
        F = F[:, :, keep_neurons]

    # final safety
    if np.any(~np.isfinite(F)):
        print(f"[prepare_F] non-finite values found; replacing with 0. Shape={F.shape}")
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    return F


@dataclass(frozen=True)
class MemeMCResult:
    moments: ArrayF  # (k_moms,)
    pr: float
    mc_moments: ArrayF  # (n_mc, k_moms)
    mc_pr: ArrayF  # (n_mc,)


@dataclass(frozen=True)
class RandomRepeatPRResult:
    pr_samples: ArrayF
    summary: PRSummary
    n_random_repeats: int
    n_successful: int


def _random_disjoint_pairs(
    C: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random disjoint pairing of condition indices.
    Returns a, b with length floor(C/2).
    """
    perm = rng.permutation(C)
    m = (C // 2) * 2
    perm = perm[:m]
    a = perm[0::2]
    b = perm[1::2]
    return a, b


def _disjoint_difference(F: ArrayF, rng: np.random.Generator) -> ArrayF:
    """Apply disjoint differencing across conditions.

    This is the spatial-bin analog of the paper's mean-removal strategy for
    unbiased eigenmoment estimation. Instead of subtracting the empirical mean
    across conditions, we randomly pair conditions and replace them with
    pairwise differences divided by ``sqrt(2)``. This removes the shared mean
    contribution without relying on a noisy sample mean estimate, which is more
    faithful to the eigenmoment derivation used in the paper.

    The same random pairing is used across all repeats so repeat-to-repeat
    comparisons stay aligned.
    """
    R, C, N = F.shape
    a, b = _random_disjoint_pairs(C, rng)
    return (F[:, a, :] - F[:, b, :]) / np.sqrt(2.0)


def meme_eigenmoments_and_pr_mc(
    F: ArrayF,
    *,
    k_moms: int = 6,
    remove_mean: bool = True,
    n_pairings: int = 20,
    n_bin_perms: int = 20,
    max_repeat_pairs: Optional[int] = None,
    random_seed: int = 0,
    use_n_choose_p: bool = True,
) -> MemeMCResult:
    """Estimate MEME eigenmoments and participation ratio from spatial tuning curves.

    This function is the core estimator adapted from the paper. It applies the
    Kong-Valiant-style eigenmoment calculation to ``F``, where ``F`` is built
    from grouped spatial tuning curves instead of repeated responses to visual
    stimuli.

    The workflow is:
    1. optionally apply disjoint differencing across conditions to remove the
       mean in a way that preserves unbiased moment estimation more closely than
       naive sample-mean subtraction
    2. randomly permute the condition order to average over order-specific
       variance in the estimator
    3. compute eigenmoments from all repeat pairs, then average across Monte
       Carlo draws
    4. convert the first two estimated moments to participation ratio

    The borrowed assumption from the paper is that repeat-to-repeat noise is
    independent given the underlying signal. In this script, that assumption is
    approximated by using disjoint groups of laps as repeats of the same
    spatially organized tuning function. Grouping multiple laps into each
    repeat is deliberate: it trades temporal granularity for more reliable
    repeat estimates of the underlying spatial tuning curve.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 3:
        raise ValueError(f"F must be (R, C, N). Got {F.shape}.")

    R, C, N = F.shape
    if R < 2:
        raise ValueError("Need at least 2 repeats (R>=2).")
    if C < 2:
        raise ValueError("Need at least 2 conditions (C>=2).")
    if k_moms < 2:
        raise ValueError("Use k_moms>=2 to compute PR.")
    if n_pairings < 1 or n_bin_perms < 1:
        raise ValueError("n_pairings and n_bin_perms must be >= 1.")

    rng = np.random.default_rng(random_seed)

    # repeat-pairs
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    if max_repeat_pairs is not None and max_repeat_pairs < len(pairs):
        keep = rng.choice(len(pairs), size=max_repeat_pairs, replace=False)
        pairs = [pairs[i] for i in keep]

    n_mc = n_pairings * n_bin_perms
    mc_moments = np.empty((n_mc, k_moms), dtype=np.float64)
    mc_pr = np.empty((n_mc,), dtype=np.float64)

    idx = 0
    for _ in range(n_pairings):
        X = _disjoint_difference(F, rng) if remove_mean else F
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # after differencing, C shrinks
        C2 = X.shape[1]
        if k_moms > C2:
            raise ValueError(
                f"k_moms={k_moms} > #conditions after differencing C'={C2}. "
                "Reduce k_moms or increase #bins."
            )

        for _ in range(n_bin_perms):
            perm = rng.permutation(C2)
            Xp = X[:, perm, :]

            pair_moms = np.empty((len(pairs), k_moms), dtype=np.float64)
            for t, (i, j) in enumerate(pairs):
                pair_moms[t] = _meme_moments_from_pair(
                    Xp[i], Xp[j], k_moms=k_moms, use_n_choose_p=use_n_choose_p
                )

            moms = pair_moms.mean(axis=0)
            mc_moments[idx] = moms

            m1, m2 = float(moms[0]), float(moms[1])
            mc_pr[idx] = float("nan") if m2 <= 0 else float(N) * (m1 * m1) / m2
            idx += 1

    moments = mc_moments.mean(axis=0)
    m1, m2 = float(moments[0]), float(moments[1])
    pr = float("nan") if m2 <= 0 else float(N) * (m1 * m1) / m2

    return MemeMCResult(moments=moments, pr=pr, mc_moments=mc_moments, mc_pr=mc_pr)


def random_seed_repeat_meme_pr(
    session: dict[str, Any],
    epoch: str,
    region: str,
    *,
    dark_epoch: str = DEFAULT_DARK_EPOCH,
    n_random_repeats: int = DEFAULT_N_RANDOM_REPEATS,
    unit_mask: Optional[np.ndarray] = None,
    bin_size: float = DEFAULT_BIN_SIZE_CM,
    n_groups: int = DEFAULT_N_GROUPS,
    min_occupancy_s: float = DEFAULT_MIN_OCCUPANCY_S,
    k_moms: int = 6,
    remove_mean: bool = True,
    n_pairings: int = DEFAULT_RANDOM_REPEAT_N_PAIRINGS,
    n_bin_perms: int = DEFAULT_RANDOM_REPEAT_N_BIN_PERMS,
    max_repeat_pairs: Optional[int] = None,
    random_seed: int = 0,
    ci: float = DEFAULT_RANDOM_REPEAT_CI,
    center: Literal["mean", "median"] = "mean",
) -> RandomRepeatPRResult:
    """Estimate spatial MEME PR stability by full-data random-seed repeats.

    Each repeat uses all available laps, changes the random seed, rebuilds the
    grouped tuning-curve tensor ``F``, and reruns MEME. The resulting interval
    reflects sensitivity to random lap grouping, disjoint condition pairing,
    condition permutation, and MEME Monte Carlo choices, matching the repeat
    strategy used by the cvPCA workflow.
    """
    if n_random_repeats < 1:
        raise ValueError("n_random_repeats must be positive.")

    pr_samples = np.full((n_random_repeats,), np.nan, dtype=np.float64)

    for repeat_index in range(n_random_repeats):
        repeat_seed = int(random_seed) + repeat_index

        try:
            F_repeat = prepare_F(
                session,
                epoch=epoch,
                region=region,
                dark_epoch=dark_epoch,
                unit_mask=unit_mask,
                bin_size=bin_size,
                n_groups=n_groups,
                group_seed=repeat_seed,
                min_occupancy_s=min_occupancy_s,
            )
            res_repeat = meme_eigenmoments_and_pr_mc(
                F_repeat,
                k_moms=k_moms,
                remove_mean=remove_mean,
                n_pairings=n_pairings,
                n_bin_perms=n_bin_perms,
                max_repeat_pairs=max_repeat_pairs,
                random_seed=repeat_seed,
            )
            pr_samples[repeat_index] = res_repeat.pr
        except ValueError as exc:
            print(
                f"[random_seed_repeat_meme_pr] skipping repeat {repeat_index} "
                f"for {region} {epoch}: {exc}"
            )

    summary = summarize_mc_pr(pr_samples, ci=ci, center=center)
    return RandomRepeatPRResult(
        pr_samples=pr_samples,
        summary=summary,
        n_random_repeats=n_random_repeats,
        n_successful=summary.n_eff,
    )


def _meme_moments_from_pair(
    X1: ArrayF,
    X2: ArrayF,
    k_moms: int,
    *,
    use_n_choose_p: bool = True,
) -> ArrayF:
    """Eigenmoment estimator for one repeat pair (Kong–Valiant style).

    Parameters
    ----------
    X1, X2
        Shape (C, N) = (n_conditions, n_neurons).
    k_moms
        Number of eigenmoments (p=1..k_moms).
    use_n_choose_p
        If True (recommended), normalize moment p by choose(C, p) (Eq. 5 / Algorithm 1
        style). If False, uses C**p (older/looser normalization).

    Returns
    -------
    moms
        Shape (k_moms,), where moms[p-1] estimates
        $$ m_p = \\frac{1}{N} \\sum_i \\lambda_i^p $$
        for the (signal) covariance eigenvalues \\(\\lambda_i\\).
    """
    if X1.ndim != 2 or X2.ndim != 2 or X1.shape != X2.shape:
        raise ValueError("X1 and X2 must be 2D and have the same shape (C, N).")

    C, N = X1.shape
    if C < 2:
        raise ValueError("Need at least 2 conditions (C>=2).")
    if k_moms < 1:
        raise ValueError("k_moms must be >= 1.")

    # Cross-repeat Gram matrix
    A = X1 @ X2.T  # (C, C)

    # Strictly upper triangle (diagonal + lower set to 0)
    F = np.triu(A, k=1)

    moms = np.empty((k_moms,), dtype=np.float64)

    # Kong–Valiant Algorithm 1 structure:
    # F_i starts at I, then multiplies by F each iteration.
    F_i = np.eye(C, dtype=np.float64)

    d_dims = float(N)
    n = int(C)

    for i in range(k_moms):
        p = i + 1

        if use_n_choose_p:
            denom = float(math.comb(n, p))
        else:
            denom = float(n**p)

        # If denom is zero (shouldn't happen for valid p<=n), set moment to 0
        if denom <= 0.0:
            moms[i] = 0.0
        else:
            moms[i] = float(np.trace(F_i @ A)) / (d_dims * denom)

        # Advance power: F_i <- F_i F  (so next loop uses F^(p-1))
        F_i = F_i @ F

    return moms


@dataclass(frozen=True)
class NaiveSpectrumResult:
    """
    Repeat-averaged covariance eigenspectrum + participation ratio computed
    directly from a covariance matrix estimated from F.

    Attributes
    ----------
    eigenvalues
        Sorted eigenvalues (descending) of the neuron-by-neuron covariance.
        Shape: (n_neurons,)
    pr
        Participation ratio:
            PR = (sum_i lambda_i)^2 / sum_i lambda_i^2
    cov
        The covariance matrix used for the eigendecomposition.
        Shape: (n_neurons, n_neurons)
    """

    eigenvalues: ArrayF
    pr: float
    cov: ArrayF


def naive_cov_eigs_and_pr(
    F: ArrayF,
    *,
    mode: Literal["concat", "mean_repeat"] = "concat",
    center: bool = True,
    remove_mean_via_disjoint: bool = False,
    eps: float = 1e-12,
) -> NaiveSpectrumResult:
    """
    Compute a covariance-based alternative dimensionality estimate from F and
    return its eigenspectrum and participation ratio (PR).

    Parameters
    ----------
    F
        Array with shape (R, C, N):
            R = repeats/groups
            C = conditions (e.g., position bins)
            N = neurons
    mode
        How to aggregate repeats before forming the covariance:
        - "concat": treat each (repeat, condition) as a sample. Uses X with shape (R*C, N).
                    This typically yields a higher-rank sample covariance.
        - "mean_repeat": average over repeats first to get X shape (C, N), then treat each
                         condition as a sample (like "mean tuning curve" covariance).
    center
        If True, subtract the per-neuron mean across samples before computing covariance.
        Recommended.
    remove_mean_via_disjoint
        If True, apply the same disjoint differencing trick used in MEME along the
        condition axis within each repeat:
            (even - odd)/sqrt(2)
        This makes the centering closer in spirit to the MEME preprocessing.
        If you use this, keep center=True (it typically still helps).
    eps
        Small constant for numerical stability in PR computation.

    Returns
    -------
    result
        NaiveSpectrumResult with eigenvalues, PR, and the covariance matrix.

    Notes
    -----
    - This is a covariance/PCA-based alternative dimensionality estimate. When
      used with ``mode="mean_repeat"``, it reflects the repeat-averaged tuning
      map rather than the repeat-aware MEME estimator.
    - PR computed from eigenvalues is:
        $$
        \\mathrm{PR} = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}.
        $$
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 3:
        raise ValueError(f"F must be (R, C, N). Got {F.shape}.")
    R, C, N = F.shape

    X = np.nan_to_num(F, nan=0.0)

    if remove_mean_via_disjoint:
        C_use = (C // 2) * 2
        if C_use < 2:
            raise ValueError("Need at least 2 conditions for disjoint differencing.")
        X = X[:, :C_use, :]
        even = X[:, 0::2, :]
        odd = X[:, 1::2, :]
        X = (even - odd) / np.sqrt(2.0)
        R, C, N = X.shape

    if mode == "concat":
        Xs = X.reshape(R * C, N)  # (samples, neurons)
    elif mode == "mean_repeat":
        Xs = X.mean(axis=0)  # (conditions, neurons)
    else:
        raise ValueError("mode must be one of {'concat','mean_repeat'}.")

    if Xs.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")

    if center:
        Xs = Xs - Xs.mean(axis=0, keepdims=True)

    # Neuron-by-neuron covariance
    cov = (Xs.T @ Xs) / float(Xs.shape[0])

    # Symmetrize to reduce numerical issues
    cov = 0.5 * (cov + cov.T)

    # Eigenvalues (ascending from eigvalsh), then reverse to descending
    evals = np.linalg.eigvalsh(cov)[::-1]

    # Guard against tiny negative eigenvalues due to numeric issues
    evals = np.clip(evals, 0.0, np.inf)

    s1 = float(evals.sum())
    s2 = float(np.sum(evals * evals))
    pr = (s1 * s1) / max(s2, eps)

    return NaiveSpectrumResult(eigenvalues=evals, pr=pr, cov=cov)


@dataclass(frozen=True)
class PowerLawFit:
    kind: Literal["power_law", "broken_power_law"]
    n_neurons: int
    alpha: Optional[float] = None
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    k0: Optional[int] = None
    c: float = np.nan
    objective: float = np.nan


def _moments_from_lambda(lam: ArrayF, k_moms: int) -> ArrayF:
    """
    Compute averaged eigenmoments:
        m_p = (1/N) * sum_i lam_i^p,  p=1..k_moms
    """
    lam = np.asarray(lam, dtype=np.float64)
    N = lam.size
    p = np.arange(1, k_moms + 1, dtype=np.int64)[:, None]  # (k_moms, 1)
    return (lam[None, :] ** p).mean(axis=1)


def _power_law_lambda(n_neurons: int, alpha: float, c: float) -> ArrayF:
    i = np.arange(1, n_neurons + 1, dtype=np.float64)
    return c * (i ** (-alpha))


def _broken_power_law_lambda(
    n_neurons: int, alpha1: float, alpha2: float, k0: int, c: float
) -> ArrayF:
    """
    Continuous broken power law:
      lam_i = c * i^{-alpha1}                         for i <= k0
      lam_i = c * k0^{alpha2-alpha1} * i^{-alpha2}    for i >  k0
    """
    if not (1 <= k0 <= n_neurons):
        raise ValueError("k0 must be in [1, n_neurons].")
    i = np.arange(1, n_neurons + 1, dtype=np.float64)
    lam = np.empty_like(i)
    left = i <= k0
    lam[left] = c * (i[left] ** (-alpha1))
    lam[~left] = c * (k0 ** (alpha2 - alpha1)) * (i[~left] ** (-alpha2))
    return lam


def fit_power_law_from_meme_moments(
    moments: ArrayF,
    *,
    n_neurons: int,
    alpha_bounds: Tuple[float, float] = (1e-3, 5.0),
    grid_size: int = 400,
    weights: Optional[ArrayF] = None,
    use_log_moments: bool = True,
) -> PowerLawFit:
    """
    Fit a single power-law eigenspectrum lam_i = c * i^{-alpha} to MEME moments.

    We eliminate c exactly using the first moment m1:
        m1 = mean_i lam_i = c * mean_i i^{-alpha}
        => c(alpha) = m1 / mean_i i^{-alpha}

    Then we choose alpha that best matches moments 2..K.

    Parameters
    ----------
    moments
        MEME averaged eigenmoments, shape (K,): m_p = (1/N) sum lam_i^p
    n_neurons
        N, number of eigenvalues
    alpha_bounds
        Range of alpha to scan
    grid_size
        Number of alpha grid points
    weights
        Optional weights for moment orders p=2..K (shape K-1).
        If None, uses 1/p^2 for p=2..K.
    use_log_moments
        If True, fits in log-space for stability.

    Returns
    -------
    fit
        PowerLawFit with alpha, c, and objective value.
    """
    moments = np.asarray(moments, dtype=np.float64)
    if moments.ndim != 1 or moments.size < 2:
        raise ValueError("moments must be shape (K,) with K>=2.")
    if n_neurons < 2:
        raise ValueError("n_neurons must be >= 2.")

    K = int(moments.size)
    m1_hat = float(moments[0])

    i = np.arange(1, n_neurons + 1, dtype=np.float64)

    # Fit moments p=2..K (we use p>=2 so c is fixed by m1)
    target = moments[1:]  # (K-1,)

    p_orders = np.arange(2, K + 1, dtype=np.float64)
    if weights is None:
        w = 1.0 / (p_orders**2)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != target.shape:
            raise ValueError(f"weights must have shape {(K-1,)}.")

    alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], grid_size, dtype=np.float64)

    best_obj = np.inf
    best_alpha = None
    best_c = None

    for alpha in alphas:
        base = i ** (-alpha)
        mean_base = float(base.mean())
        if mean_base <= 0:
            continue

        c = m1_hat / mean_base
        lam = c * base

        model = _moments_from_lambda(lam, K)[1:]  # p=2..K

        if use_log_moments and np.all(model > 0) and np.all(target > 0):
            resid = np.log(model) - np.log(target)
        else:
            resid = (model - target) / np.maximum(np.abs(target), 1e-12)

        obj = float(np.sum(w * (resid**2)))
        if obj < best_obj:
            best_obj = obj
            best_alpha = float(alpha)
            best_c = float(c)

    if best_alpha is None or best_c is None:
        raise RuntimeError("Power-law fit failed; no valid alpha in scan.")

    return PowerLawFit(
        kind="power_law",
        n_neurons=n_neurons,
        alpha=best_alpha,
        c=best_c,
        objective=best_obj,
    )


from scipy.optimize import minimize


def fit_broken_power_law_optimized(
    moments: ArrayF,
    *,
    n_neurons: int,
    alpha_bounds: Tuple[float, float] = (0.0, 5.0),
    k0_candidates: Optional[npt.NDArray[np.integer]] = None,
    weights: Optional[ArrayF] = None,
    use_log_moments: bool = True,
) -> PowerLawFit:
    """
    Fits broken power law using the strategy from Pospisil & Pillow (2025):
    1. Grid search for the break point k0.
    2. For each k0, optimize (alpha1, alpha2) using L-BFGS-B (faster & precise).
    """
    moments = np.asarray(moments, dtype=np.float64)
    K = moments.size
    m1_ref = moments[0]
    target_moments = moments[1:]  # p=2..K

    # Paper-consistent heuristic: 1/p^2 weights if not provided
    p_orders = np.arange(2, K + 1, dtype=np.float64)
    if weights is None:
        w = 1.0 / (p_orders**2)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Candidates for k0 (Break point)
    if k0_candidates is None:
        log_k = np.unique(np.round(np.logspace(0, np.log10(n_neurons), 40)).astype(int))
        small_k = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60], dtype=int)
        k0_candidates = np.unique(
            np.clip(np.concatenate([small_k, log_k]), 1, n_neurons - 1)
        )

    # Pre-allocate array for indices to speed up calculations
    # We do this calculation inside the objective to allow float optimization of alphas
    idx_full = np.arange(1, n_neurons + 1, dtype=np.float64)

    best_global_obj = np.inf
    best_global_params = None  # (alpha1, alpha2, k0, c)

    # Iterate over discrete break points k0 (as per paper)
    for k0 in k0_candidates:

        # Pre-slice indices for this k0
        i_left = idx_full[:k0]
        i_right = idx_full[k0:]

        # Define objective function for this specific k0
        def objective(alphas):
            a1, a2 = alphas

            # 1. Reconstruct Lambda shape
            # Vectorized power calculation is fast enough for N~10k inside optimizer
            term_left = i_left ** (-a1)

            # Continuity term: C * k0^-a1 = C * k0^(a2-a1) * k0^-a2
            # Right side base scaling:
            scale_right = k0 ** (a2 - a1)
            term_right = scale_right * (i_right ** (-a2))

            # 2. Solve for c using first moment constraint (m1)
            # m1 = (c/N) * sum(terms)  => c = (m1 * N) / sum(terms)
            sum_terms = np.sum(term_left) + np.sum(term_right)
            c = (m1_ref * n_neurons) / sum_terms

            # 3. Compute higher moments of the model
            # m_p = c^p * (1/N) * sum(terms^p)
            # Optimization: calculate moments directly from terms
            # Note: terms are proportional to lambda/c

            model_moms = []
            for p in range(2, K + 1):
                # sum(lambda^p) = c^p * sum(base_terms^p)
                # This sum can be numerically large/small, be careful
                sum_p = np.sum(term_left**p) + np.sum(term_right**p)
                m_p = (c**p / n_neurons) * sum_p
                model_moms.append(m_p)

            model_moms = np.array(model_moms)

            # 4. Compute Loss
            if use_log_moments:
                # Add tiny epsilon to avoid log(0)
                resid = np.log(model_moms + 1e-100) - np.log(target_moments + 1e-100)
            else:
                resid = model_moms - target_moments

            return np.sum(w * (resid**2))

        # Initial guess for alphas (start with typical values)
        x0 = [1.0, 1.0]

        # Run optimization for this k0
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=[alpha_bounds, alpha_bounds],
            tol=1e-6,
        )

        if res.fun < best_global_obj:
            best_global_obj = res.fun
            # Recompute c for the best alphas
            a1_opt, a2_opt = res.x

            # Recalculate c (logic copied from objective)
            term_left = i_left ** (-a1_opt)
            scale_right = k0 ** (a2_opt - a1_opt)
            term_right = scale_right * (i_right ** (-a2_opt))
            c_opt = (m1_ref * n_neurons) / (np.sum(term_left) + np.sum(term_right))

            best_global_params = (a1_opt, a2_opt, k0, c_opt)

    if best_global_params is None:
        raise RuntimeError("Broken power-law fit failed.")

    return PowerLawFit(
        kind="broken_power_law",
        n_neurons=n_neurons,
        alpha1=best_global_params[0],
        alpha2=best_global_params[1],
        k0=best_global_params[2],
        c=best_global_params[3],
        objective=best_global_obj,
    )


def eigenspectrum_from_fit(fit: PowerLawFit) -> ArrayF:
    """
    Generate lambda_i (length N) from a fitted parametric model.
    """
    if fit.kind == "power_law":
        if fit.alpha is None:
            raise ValueError("fit.alpha is required for power_law.")
        return _power_law_lambda(fit.n_neurons, float(fit.alpha), float(fit.c))

    if fit.kind == "broken_power_law":
        if fit.alpha1 is None or fit.alpha2 is None or fit.k0 is None:
            raise ValueError(
                "fit.alpha1, fit.alpha2, fit.k0 are required for broken_power_law."
            )
        return _broken_power_law_lambda(
            fit.n_neurons,
            float(fit.alpha1),
            float(fit.alpha2),
            int(fit.k0),
            float(fit.c),
        )

    raise ValueError(f"Unknown fit.kind: {fit.kind}")


def plot_eigenspectrum_comparison(
    region: str,
    all_moments: dict,
    naive_eigenvalues: dict,
    n_neurons: int,
    save_path: Path,
    metric: Literal["eigenvalue", "variance_explained"] = "variance_explained",
    max_rank: int = 100,
) -> Path:
    """Plot MEME and repeat-averaged PCA eigenspectra for visual comparison.

    These figures are intended as a geometry check: the MEME panel shows the
    fitted signal eigenspectrum inferred from eigenmoments, while the
    repeat-averaged PCA panel shows a covariance-based spectrum from the same
    grouped spatial tuning data.
    """

    # Setup y-label
    if metric == "variance_explained":
        ylabel = "Fraction of Variance Explained"
    else:
        ylabel = "Eigenvalue Amplitude"

    # Ensure max_rank doesn't exceed n_neurons
    limit = min(max_rank, n_neurons)

    fig, ax = plt.subplots(figsize=(10, 6), ncols=2, sharey=True)

    # --- 1. MEME PLOTTING ---
    for epoch, moments in all_moments.items():
        # Reconstruct FULL spectrum first
        pl_fit = fit_power_law_from_meme_moments(moments, n_neurons=n_neurons)
        lam_pl = eigenspectrum_from_fit(pl_fit)

        # Normalize using the TOTAL variance (sum of all N neurons)
        # If we normalized by sum(limit), the percentages would be wrong.
        if metric == "variance_explained":
            total_variance = np.sum(lam_pl)
            y_values = lam_pl / total_variance
        else:
            y_values = lam_pl

        # Slice ONLY the data we want to plot (top 100)
        # This forces matplotlib to auto-scale Y to these values only.
        ax[0].loglog(
            np.arange(1, limit + 1),
            y_values[:limit],
            "-",
            linewidth=2,
            label=f"{epoch} (α={pl_fit.alpha:.2f})",
        )

    # --- 2. REPEAT-AVERAGED PCA PLOTTING ---
    for epoch, eigenvalues in naive_eigenvalues.items():
        # Use full spectrum for total variance calculation
        full_evals = eigenvalues[:n_neurons]

        if metric == "variance_explained":
            total_variance = np.sum(full_evals)
            y_values = full_evals / total_variance
        else:
            y_values = full_evals

        # Slice for plotting
        ax[1].loglog(
            np.arange(1, limit + 1),
            y_values[:limit],
            "s",
            markersize=4,
            alpha=0.6,
            label=f"{epoch}",
        )

    # Formatting
    for a in ax.flat:
        a.legend(fontsize="small")
        a.grid(True, which="both", ls="-", alpha=0.2)
        a.set_xlabel("PC Rank (log)")
        # Enforce the x-limit strictly
        a.set_xlim(1, limit)

    ax[0].set_title(f"MEME Estimation\n(First {limit} modes)")
    ax[1].set_title(f"Repeat-averaged PCA\n(First {limit} modes)")
    ax[0].set_ylabel(ylabel)

    fig.suptitle(f"Region: {region} | Metric: {metric}")
    plt.tight_layout()

    filename = f"{region}_eigenspectrum_{metric}_top{limit}.png"
    output_path = save_path / filename
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_broken_power_law_comparison(
    region: str,
    all_moments: dict,
    naive_eigenvalues: dict,
    n_neurons: int,
    save_path: Path,
    metric: Literal["eigenvalue", "variance_explained"] = "variance_explained",
    max_rank: int = 1000,
) -> Path:
    """Plot broken-power-law fits for the inferred eigenspectra.

    These plots are qualitative fit diagnostics. They let the reader inspect
    whether the spatial tuning eigenspectrum is better captured by a broken
    power law than by a single-slope fit, mirroring the paper's emphasis on
    broken-power-law structure.
    """

    # Setup y-label
    if metric == "variance_explained":
        ylabel = "Fraction of Variance Explained"
    else:
        ylabel = "Eigenvalue Amplitude"

    # Ensure max_rank doesn't exceed n_neurons
    limit = min(max_rank, n_neurons)

    fig, ax = plt.subplots(figsize=(12, 6), ncols=2, sharey=True)

    # --- 1. MEME PLOTTING (Broken Power Law) ---
    for epoch, moments in all_moments.items():
        # --- CORRECTION HERE: Use the optimized fitter ---
        # This finds the exact alpha values rather than grid points
        pl_fit = fit_broken_power_law_optimized(moments, n_neurons=n_neurons)

        # Reconstruct spectrum from fit parameters
        lam_pl = eigenspectrum_from_fit(pl_fit)

        # Normalize using the TOTAL variance
        if metric == "variance_explained":
            total_variance = np.sum(lam_pl)
            y_values = lam_pl / total_variance
        else:
            y_values = lam_pl

        # Slice for plotting
        # Note: k0 is an integer, so we format it without decimals
        label_str = (
            f"{epoch}\n"
            f"(α1={pl_fit.alpha1:.2f}, α2={pl_fit.alpha2:.2f}, k0={pl_fit.k0})"
        )

        (line,) = ax[0].loglog(
            np.arange(1, limit + 1), y_values[:limit], "-", linewidth=2, label=label_str
        )

        # Visualize the break point k0
        if pl_fit.k0 < limit:
            ax[0].axvline(pl_fit.k0, color=line.get_color(), linestyle=":", alpha=0.5)

    # --- 2. REPEAT-AVERAGED PCA PLOTTING ---
    for epoch, eigenvalues in naive_eigenvalues.items():
        full_evals = eigenvalues[:n_neurons]

        if metric == "variance_explained":
            total_variance = np.sum(full_evals)
            y_values = full_evals / total_variance
        else:
            y_values = full_evals

        ax[1].loglog(
            np.arange(1, limit + 1),
            y_values[:limit],
            "s",
            markersize=4,
            alpha=0.6,
            label=f"{epoch}",
        )

    # Formatting
    for a in ax.flat:
        a.legend(fontsize="x-small", loc="lower left")
        a.grid(True, which="both", ls="-", alpha=0.2)
        a.set_xlabel("PC Rank (log)")
        a.set_xlim(1, limit)

    ax[0].set_title(f"MEME Estimation\n(Broken Power Law Fit)")
    ax[1].set_title(f"Repeat-averaged PCA\n(Covariance Eigenspectrum)")
    ax[0].set_ylabel(ylabel)

    fig.suptitle(f"Region: {region} | Metric: {metric} | Broken Power Law")
    plt.tight_layout()

    filename = f"{region}_broken_power_law_{metric}_top{limit}.png"
    output_path = save_path / filename
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_pr(region, meme_pr, naive_pr, save_path) -> Path:
    """Plot epoch-wise participation ratio summaries for one region."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = list(meme_pr.keys())
    # Sort epochs if needed, assuming they are strings they might need sorting logic

    ax.plot(epochs, list(meme_pr.values()), "-o", label="MEME")
    ax.plot(epochs, list(naive_pr.values()), "-s", label="Repeat-averaged PCA")

    ax.set_ylabel("Participation Ratio")
    ax.set_title(f"Region: {region}")
    ax.legend()
    output_path = save_path / f"{region}_pr.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_pr_light_dark_epochwise(
    region: str,
    epoch_to_mc_pr: Dict[str, ArrayF],
    epoch_to_condition: Dict[str, str],
    save_path: Path,
    *,
    ci: float = 95.0,
    center: Literal["mean", "median"] = "mean",
    jitter: float = 0.08,
    annotate: bool = True,
    random_seed: int = 0,
    title_suffix: str = "",
):
    """Plot per-epoch PR estimates with random-seed repeat intervals by condition.

    This summarizes the adapted MEME output at the epoch level rather than the
    eigenspectrum level, separating the selected light and dark epochs.
    """
    rng = np.random.default_rng(random_seed)

    # Keep only epochs that appear in both dicts
    epochs = [e for e in epoch_to_mc_pr.keys() if e in epoch_to_condition]
    if len(epochs) == 0:
        raise ValueError(
            "No epochs overlap between epoch_to_mc_pr and epoch_to_condition."
        )

    # Fixed condition order
    cond_order = ["dark", "light"]
    cond_x = {c: i for i, c in enumerate(cond_order)}

    fig, ax = plt.subplots(figsize=(7, 5))

    for epoch in sorted(epochs):
        cond = str(epoch_to_condition[epoch]).lower()
        if cond not in cond_x:
            # skip unknown condition labels
            continue

        summ = summarize_mc_pr(epoch_to_mc_pr[epoch], ci=ci, center=center)
        if not np.isfinite(summ.pr_center):
            continue

        x0 = float(cond_x[cond])
        x = x0 + float(rng.uniform(-jitter, jitter))

        y = summ.pr_center
        yerr_low = y - summ.ci_low
        yerr_high = summ.ci_high - y
        yerr = np.array([[yerr_low], [yerr_high]])

        marker = "o" if cond == "dark" else "s"
        ax.errorbar(x, y, yerr=yerr, fmt=marker, capsize=3, alpha=0.9)

        if annotate:
            ax.text(x + 0.02, y, epoch, fontsize=8, rotation=35, va="center")

    ax.set_xticks([cond_x["dark"], cond_x["light"]])
    ax.set_xticklabels(["Dark", "Light"])
    ax.set_ylabel("Participation Ratio (PR)")
    ax.set_title(f"{region}: PR by epoch with {ci:.0f}% interval{title_suffix}")
    ax.grid(True, alpha=0.25)

    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"{region}_pr_light_vs_dark_epochwise_random_repeats_ci{int(ci)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _mc_mean_across_epochs(
    epoch_to_mc_pr: Dict[str, ArrayF],
    epochs: Iterable[str],
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> ArrayF:
    epochs = list(epochs)
    if len(epochs) == 0:
        return np.full((n_draws,), np.nan, dtype=np.float64)

    # Pre-clean each epoch's mc_pr
    cleaned = []
    for e in epochs:
        x = np.asarray(epoch_to_mc_pr[e], dtype=np.float64)
        x = x[np.isfinite(x)]
        cleaned.append(x)

    draws = np.empty((n_draws,), dtype=np.float64)
    for k in range(n_draws):
        vals = []
        for x in cleaned:
            if x.size == 0:
                vals.append(np.nan)
            else:
                vals.append(float(rng.choice(x)))
        draws[k] = float(np.nanmean(vals))
    return draws


def plot_pr_light_dark_groupmeans(
    region: str,
    epoch_to_mc_pr: Dict[str, ArrayF],
    epoch_to_condition: Dict[str, str],
    save_path: Path,
    *,
    ci: float = 95.0,
    center: Literal["mean", "median"] = "mean",
    n_draws: int = 5000,
    random_seed: int = 0,
    title_suffix: str = "",
):
    """Plot condition-level PR means and return the draw distributions.

    The returned dark, light, and delta draws are also written to summary
    parquet tables so downstream analysis can use the same light-vs-dark
    summaries that fed the figure.
    """
    rng = np.random.default_rng(random_seed)

    # Partition epochs by condition
    dark_condition_epochs = [
        e
        for e, c in epoch_to_condition.items()
        if str(c).lower() == "dark" and e in epoch_to_mc_pr
    ]
    light_condition_epochs = [
        e
        for e, c in epoch_to_condition.items()
        if str(c).lower() == "light" and e in epoch_to_mc_pr
    ]

    if len(dark_condition_epochs) == 0 or len(light_condition_epochs) == 0:
        raise ValueError(
            "Need >=1 epoch in each condition. Found "
            f"dark={len(dark_condition_epochs)}, "
            f"light={len(light_condition_epochs)}."
        )

    dark_draws = _mc_mean_across_epochs(
        epoch_to_mc_pr,
        dark_condition_epochs,
        n_draws=n_draws,
        rng=rng,
    )
    light_draws = _mc_mean_across_epochs(
        epoch_to_mc_pr,
        light_condition_epochs,
        n_draws=n_draws,
        rng=rng,
    )
    delta_draws = light_draws - dark_draws

    dark_sum = summarize_mc_pr(dark_draws, ci=ci, center=center)
    light_sum = summarize_mc_pr(light_draws, ci=ci, center=center)
    delta_sum = summarize_mc_pr(delta_draws, ci=ci, center=center)

    fig, ax = plt.subplots(figsize=(6.5, 5))

    xs = np.array([0.0, 1.0])
    ys = np.array([dark_sum.pr_center, light_sum.pr_center])
    yerr = np.array(
        [
            [
                dark_sum.pr_center - dark_sum.ci_low,
                light_sum.pr_center - light_sum.ci_low,
            ],
            [
                dark_sum.ci_high - dark_sum.pr_center,
                light_sum.ci_high - light_sum.pr_center,
            ],
        ],
        dtype=np.float64,
    )

    ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Dark (mean across epochs)", "Light (mean across epochs)"])
    ax.set_ylabel("Participation Ratio (PR)")
    ax.set_title(f"{region}: mean PR Light vs Dark ({ci:.0f}% interval){title_suffix}")
    ax.grid(True, alpha=0.25)

    # annotate delta
    ax.text(
        0.02,
        0.02,
        f"Δ = Light - Dark: {delta_sum.pr_center:.3g} "
        f"[{delta_sum.ci_low:.3g}, {delta_sum.ci_high:.3g}]",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
    )

    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"{region}_pr_light_vs_dark_groupmean_random_repeats_ci{int(ci)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)

    return out, dark_draws, light_draws, delta_draws


def _region_threshold(region: str, args: argparse.Namespace) -> float:
    """Return the configured movement firing-rate threshold for one region."""
    if region == "v1":
        return float(args.v1_min_fr_hz)
    if region == "ca1":
        return float(args.ca1_min_fr_hz)
    raise ValueError(f"Unsupported region: {region!r}")


def _condition_sd(f_tensor: ArrayF) -> ArrayF:
    """Return per-unit condition SD from the repeat-averaged tuning tensor."""
    tensor = np.asarray(f_tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected F with shape (repeat, condition, unit), got {tensor.shape}."
        )
    return np.std(np.mean(tensor, axis=0), axis=0)


def _minimum_metric_across_epochs(
    values_by_epoch: dict[str, ArrayF],
    epochs: list[str],
) -> ArrayF:
    """Return per-unit minima across selected epochs."""
    if not epochs:
        raise ValueError("At least one epoch is required to aggregate unit metrics.")
    values = [
        np.asarray(values_by_epoch[epoch], dtype=float).reshape(-1)
        for epoch in epochs
    ]
    first_size = values[0].size
    if any(value.size != first_size for value in values):
        raise ValueError("Unit metric sizes must match across epochs.")
    return np.min(np.stack(values, axis=0), axis=0)


def build_meme_unit_mask(
    *,
    session: dict[str, Any],
    f_by_epoch: dict[str, ArrayF],
    region: str,
    light_epochs: list[str],
    dark_epoch: str,
    min_firing_rate_hz: float,
    min_condition_sd_hz: float,
    unit_filter_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cvPCA-style unit mask and class labels for MEME epochs.

    When multiple light epochs are analyzed together, light metrics are the
    per-unit minima across selected light epochs. This keeps one stable unit
    mask for all MEME epoch-level PR estimates.
    """
    dark_firing_rate_hz = np.asarray(
        session["movement_firing_rates_by_region"][region][dark_epoch],
        dtype=float,
    )
    light_firing_rate_hz = _minimum_metric_across_epochs(
        session["movement_firing_rates_by_region"][region],
        light_epochs,
    )
    condition_sd_by_epoch = {
        epoch: _condition_sd(f_by_epoch[epoch]) for epoch in [dark_epoch, *light_epochs]
    }
    dark_condition_sd_hz = condition_sd_by_epoch[dark_epoch]
    light_condition_sd_hz = _minimum_metric_across_epochs(
        condition_sd_by_epoch,
        light_epochs,
    )
    unit_ids = np.arange(dark_firing_rate_hz.size, dtype=int)
    selection = classify_and_select_units(
        unit_ids=unit_ids,
        dark_firing_rate_hz=dark_firing_rate_hz,
        light_firing_rate_hz=light_firing_rate_hz,
        dark_condition_sd_hz=dark_condition_sd_hz,
        light_condition_sd_hz=light_condition_sd_hz,
        min_firing_rate_hz=min_firing_rate_hz,
        min_condition_sd_hz=min_condition_sd_hz,
        unit_filter_mode=unit_filter_mode,
    )
    if not np.any(selection.keep_mask):
        raise ValueError(
            "No units remain after applying firing-rate and condition-SD filters."
        )
    return selection.keep_mask, selection.unit_classes


def _stable_region_neuron_count(region_counts: dict[str, int], region: str) -> int:
    """Return one validated neuron count for a region across selected epochs."""
    unique_counts = set(region_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            f"Expected a stable neuron count across epochs for region {region!r}, "
            f"got {sorted(unique_counts)!r}."
        )
    return next(iter(unique_counts))


def save_epoch_summary_tables(
    out_dir: Path,
    *,
    regions: list[str],
    analysis_epochs: list[str],
    epoch_to_condition: dict[str, str],
    n_neurons_by_region: dict[str, dict[str, int]],
    meme_pr: dict[str, dict[str, float]],
    naive_pr: dict[str, dict[str, float]],
    meme_repeat_summary: dict[str, dict[str, PRSummary]],
    min_firing_rate_by_region: dict[str, float],
    n_units_by_class_by_region: dict[str, dict[str, int]],
    settings: dict[str, Any],
) -> list[Path]:
    """Write one parquet summary row per region and epoch.

    These parquet tables are compact analysis summaries. They record scalar
    outputs of the MEME workflow for each analyzed region/epoch pair, not the full
    vector-valued MEME objects such as all eigenmoments or full eigenspectra.
    """
    saved_paths: list[Path] = []
    for region in regions:
        min_firing_rate_hz = min_firing_rate_by_region[region]
        unit_class_counts = n_units_by_class_by_region[region]
        for epoch in analysis_epochs:
            summary = meme_repeat_summary[region][epoch]
            table = pd.DataFrame(
                [
                    {
                        "region": region,
                        "epoch": epoch,
                        "condition": epoch_to_condition.get(epoch),
                        "n_neurons": n_neurons_by_region[region][epoch],
                        "meme_pr": meme_pr[region][epoch],
                        "repeat_averaged_pca_pr": naive_pr[region][epoch],
                        "random_repeat_pr_center": summary.pr_center,
                        "random_repeat_pr_ci_low": summary.ci_low,
                        "random_repeat_pr_ci_high": summary.ci_high,
                        "random_repeat_pr_n_eff": summary.n_eff,
                        "unit_filter_mode": settings["unit_filter_mode"],
                        "min_firing_rate_hz": min_firing_rate_hz,
                        "min_condition_sd_hz": settings["min_condition_sd_hz"],
                        "n_all_shared_active": unit_class_counts["shared_active"],
                        "n_all_dark_only": unit_class_counts["dark_only"],
                        "n_all_light_only": unit_class_counts["light_only"],
                        "n_all_inactive_or_low_mod": unit_class_counts[
                            "inactive_or_low_mod"
                        ],
                        "bin_size_cm": settings["bin_size_cm"],
                        "n_groups": settings["n_groups"],
                        "n_random_repeats": settings["n_random_repeats"],
                        "random_repeat_ci": settings["random_repeat_ci"],
                        "n_pairings_full": settings["full_n_pairings"],
                        "n_bin_perms_full": settings["full_n_bin_perms"],
                        "n_pairings_random_repeat": settings[
                            "random_repeat_n_pairings"
                        ],
                        "n_bin_perms_random_repeat": settings[
                            "random_repeat_n_bin_perms"
                        ],
                    }
                ]
            )
            path = out_dir / f"{region}_{epoch}_meme_summary.parquet"
            table.to_parquet(path, index=False)
            saved_paths.append(path)
    return saved_paths


def save_light_dark_comparison_tables(
    out_dir: Path,
    *,
    regions: list[str],
    light_epochs: list[str],
    dark_epoch: str,
    comparison_draws_by_region: dict[str, dict[str, ArrayF]],
    ci: float,
    center: Literal["mean", "median"],
    n_draws: int,
) -> list[Path]:
    """Write one light-vs-dark comparison parquet per region.

    Each table stores scalar summaries of the grouped random-repeat draw
    distributions used for the light/dark PR figures. The goal is to preserve
    the comparison results in a tabular, downstream-friendly format.
    """
    saved_paths: list[Path] = []
    light_epoch_value = light_epochs[0] if len(light_epochs) == 1 else None
    light_epochs_label = ",".join(light_epochs)
    for region in regions:
        dark_draws = comparison_draws_by_region[region]["dark_draws"]
        light_draws = comparison_draws_by_region[region]["light_draws"]
        delta_draws = comparison_draws_by_region[region]["delta_draws"]
        dark_summary = summarize_mc_pr(dark_draws, ci=ci, center=center)
        light_summary = summarize_mc_pr(light_draws, ci=ci, center=center)
        delta_summary = summarize_mc_pr(delta_draws, ci=ci, center=center)

        table = pd.DataFrame(
            [
                {
                    "region": region,
                    "summary_kind": "dark",
                    "dark_epoch": dark_epoch,
                    "light_epoch": light_epoch_value,
                    "light_epochs": light_epochs_label,
                    "n_light_epochs": len(light_epochs),
                    "pr_center": dark_summary.pr_center,
                    "pr_ci_low": dark_summary.ci_low,
                    "pr_ci_high": dark_summary.ci_high,
                    "n_eff": dark_summary.n_eff,
                    "n_draws": n_draws,
                },
                {
                    "region": region,
                    "summary_kind": "light",
                    "dark_epoch": dark_epoch,
                    "light_epoch": light_epoch_value,
                    "light_epochs": light_epochs_label,
                    "n_light_epochs": len(light_epochs),
                    "pr_center": light_summary.pr_center,
                    "pr_ci_low": light_summary.ci_low,
                    "pr_ci_high": light_summary.ci_high,
                    "n_eff": light_summary.n_eff,
                    "n_draws": n_draws,
                },
                {
                    "region": region,
                    "summary_kind": "light_minus_dark",
                    "dark_epoch": dark_epoch,
                    "light_epoch": light_epoch_value,
                    "light_epochs": light_epochs_label,
                    "n_light_epochs": len(light_epochs),
                    "pr_center": delta_summary.pr_center,
                    "pr_ci_low": delta_summary.ci_low,
                    "pr_ci_high": delta_summary.ci_high,
                    "n_eff": delta_summary.n_eff,
                    "n_draws": n_draws,
                },
            ]
        )
        path = out_dir / f"{region}_{dark_epoch}_light_dark_comparison.parquet"
        table.to_parquet(path, index=False)
        saved_paths.append(path)
    return saved_paths

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MEME signal-dimension workflow."""
    parser = argparse.ArgumentParser(description="Estimate MEME signal dimensionality")
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--run-epochs",
        nargs="*",
        help="Optional run epoch labels to analyze. Defaults to the first four run epochs.",
    )
    parser.add_argument(
        "--dark-epoch",
        required=True,
        help="Epoch label to treat as dark.",
    )
    parser.add_argument(
        "--light-epoch",
        help=(
            "Optional run epoch label to treat as light. If omitted, all selected "
            "run epochs other than --dark-epoch are treated as light."
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help=f"Regions to analyze. Default: {' '.join(DEFAULT_REGIONS)}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=f"Speed threshold in cm/s used to define movement. Default: {DEFAULT_SPEED_THRESHOLD_CM_S}",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--bin-size-cm",
        type=float,
        default=DEFAULT_BIN_SIZE_CM,
        help=f"Spatial bin size in cm for tuning curves. Default: {DEFAULT_BIN_SIZE_CM}",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=DEFAULT_N_GROUPS,
        help=f"Number of disjoint lap groups. Default: {DEFAULT_N_GROUPS}",
    )
    parser.add_argument(
        "--min-occupancy-s",
        type=float,
        default=DEFAULT_MIN_OCCUPANCY_S,
        help=f"Minimum occupancy in seconds per kept bin. Default: {DEFAULT_MIN_OCCUPANCY_S}",
    )
    parser.add_argument(
        "--unit-filter-mode",
        choices=("shared-active", "dark-active", "union-active"),
        default=DEFAULT_UNIT_FILTER_MODE,
        help=f"Neuron inclusion rule. Default: {DEFAULT_UNIT_FILTER_MODE}",
    )
    parser.add_argument(
        "--v1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help="Minimum movement firing rate for V1 units. Default: 0.5",
    )
    parser.add_argument(
        "--ca1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help="Minimum movement firing rate for CA1 units. Default: 0.0",
    )
    parser.add_argument(
        "--min-condition-sd-hz",
        type=float,
        default=DEFAULT_MIN_CONDITION_SD_HZ,
        help=(
            "Minimum repeat-averaged condition SD for unit inclusion. "
            f"Default: {DEFAULT_MIN_CONDITION_SD_HZ}"
        ),
    )
    parser.add_argument(
        "--n-random-repeats",
        type=int,
        default=DEFAULT_N_RANDOM_REPEATS,
        help=f"Number of full-data random-seed repeats. Default: {DEFAULT_N_RANDOM_REPEATS}",
    )
    parser.add_argument(
        "--random-repeat-ci",
        type=float,
        default=DEFAULT_RANDOM_REPEAT_CI,
        help=f"Interval width in percent for random-seed repeats. Default: {DEFAULT_RANDOM_REPEAT_CI}",
    )
    parser.add_argument(
        "--full-n-pairings",
        type=int,
        default=DEFAULT_FULL_N_PAIRINGS,
        help=f"Number of Monte Carlo pairing draws for full MEME estimation. Default: {DEFAULT_FULL_N_PAIRINGS}",
    )
    parser.add_argument(
        "--full-n-bin-perms",
        type=int,
        default=DEFAULT_FULL_N_BIN_PERMS,
        help=f"Number of bin permutations for full MEME estimation. Default: {DEFAULT_FULL_N_BIN_PERMS}",
    )
    parser.add_argument(
        "--repeat-n-pairings",
        type=int,
        default=DEFAULT_RANDOM_REPEAT_N_PAIRINGS,
        help=(
            "Number of Monte Carlo pairing draws inside random-seed repeat "
            f"estimation. Default: {DEFAULT_RANDOM_REPEAT_N_PAIRINGS}"
        ),
    )
    parser.add_argument(
        "--repeat-n-bin-perms",
        type=int,
        default=DEFAULT_RANDOM_REPEAT_N_BIN_PERMS,
        help=(
            "Number of bin permutations inside random-seed repeat estimation. "
            f"Default: {DEFAULT_RANDOM_REPEAT_N_BIN_PERMS}"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Base random seed for MEME estimation. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for saved parquet tables. Default: "
            f"analysis_path / 'signal_dim' / '{DEFAULT_MEME_OUTPUT_DIRNAME}'"
        ),
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        help=(
            "Directory for saved figures. Default: "
            f"analysis_path / 'figs' / 'signal_dim' / '{DEFAULT_MEME_OUTPUT_DIRNAME}'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the MEME signal-dimension workflow for one session.

    High-level workflow:
    1. load one session and derive movement- and trajectory-aware state
    2. choose run epochs plus one light/dark comparison set
    3. build grouped spatial tuning tensors ``F`` for each region and epoch
    4. estimate MEME and repeat-averaged PCA participation ratios
    5. estimate random-seed repeat stability
    6. save figures, parquet summaries, and a run log
    """
    args = parse_arguments()
    if args.n_random_repeats < 1:
        raise ValueError("--n-random-repeats must be positive.")
    if args.random_repeat_ci <= 0.0 or args.random_repeat_ci >= 100.0:
        raise ValueError("--random-repeat-ci must be between 0 and 100.")
    if args.repeat_n_pairings < 1 or args.repeat_n_bin_perms < 1:
        raise ValueError("--repeat-n-pairings and --repeat-n-bin-perms must be positive.")
    if args.min_condition_sd_hz < 0.0:
        raise ValueError("--min-condition-sd-hz must be nonnegative.")

    session = load_meme_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=list(args.regions),
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    run_epochs = select_run_epochs(
        session["run_epochs"],
        args.run_epochs,
    )
    light_epochs, dark_epoch = get_light_and_dark_epochs(
        run_epochs,
        args.light_epoch,
        args.dark_epoch,
    )
    analysis_epochs = [dark_epoch] + [epoch for epoch in light_epochs if epoch != dark_epoch]
    epoch_to_condition = {epoch: "light" for epoch in light_epochs}
    epoch_to_condition[dark_epoch] = "dark"

    analysis_path = session["analysis_path"]
    out_dir = args.output_dir or get_signal_dim_output_dir(
        analysis_path,
        DEFAULT_MEME_OUTPUT_DIRNAME,
    )
    fig_dir = args.fig_dir or get_signal_dim_figure_dir(
        analysis_path,
        DEFAULT_MEME_OUTPUT_DIRNAME,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "animal_name": args.animal_name,
        "date": args.date,
        "data_root": args.data_root,
        "regions": list(args.regions),
        "run_epochs": run_epochs,
        "analysis_epochs": analysis_epochs,
        "position_source": session["position_source"],
        "skipped_run_epochs": session["skipped_run_epochs"],
        "light_epoch": args.light_epoch,
        "light_epochs": light_epochs,
        "dark_epoch": dark_epoch,
        "speed_threshold_cm_s": args.speed_threshold_cm_s,
        "position_offset": args.position_offset,
        "bin_size_cm": args.bin_size_cm,
        "n_groups": args.n_groups,
        "min_occupancy_s": args.min_occupancy_s,
        "unit_filter_mode": args.unit_filter_mode,
        "v1_min_fr_hz": args.v1_min_fr_hz,
        "ca1_min_fr_hz": args.ca1_min_fr_hz,
        "min_condition_sd_hz": args.min_condition_sd_hz,
        "uncertainty_method": "random_seed_repeats",
        "n_random_repeats": args.n_random_repeats,
        "random_repeat_ci": args.random_repeat_ci,
        "full_n_pairings": args.full_n_pairings,
        "full_n_bin_perms": args.full_n_bin_perms,
        "random_repeat_n_pairings": args.repeat_n_pairings,
        "random_repeat_n_bin_perms": args.repeat_n_bin_perms,
        "random_seed": args.random_seed,
        "output_dir": out_dir,
        "fig_dir": fig_dir,
    }

    meme_moments = {region: {} for region in args.regions}
    naive_eigenvalues = {region: {} for region in args.regions}
    meme_pr = {region: {} for region in args.regions}
    naive_pr = {region: {} for region in args.regions}
    meme_repeat_pr = {region: {} for region in args.regions}
    meme_repeat_summary = {region: {} for region in args.regions}
    n_neurons_by_region = {region: {} for region in args.regions}
    n_units_by_class_by_region: dict[str, dict[str, int]] = {}
    comparison_draws_by_region: dict[str, dict[str, ArrayF]] = {}
    min_firing_rate_by_region = {
        region: _region_threshold(region, args) for region in args.regions
    }

    # Figure families:
    # - eigenspectrum comparison: MEME and repeat-averaged PCA geometry
    # - broken power law: qualitative inspection of spectrum shape
    # - PR figures: scalar dimensionality summaries across epochs and conditions
    saved_figures: list[Path] = []
    for region in args.regions:
        min_firing_rate_hz = min_firing_rate_by_region[region]
        print(f"\n=== Processing Region: {region} ===")

        f_all_by_epoch: dict[str, ArrayF] = {}
        for epoch in analysis_epochs:
            print(f"  -- Epoch: {epoch}")
            f_all_by_epoch[epoch] = prepare_F(
                session,
                epoch=epoch,
                region=region,
                dark_epoch=dark_epoch,
                bin_size=args.bin_size_cm,
                n_groups=args.n_groups,
                group_seed=args.random_seed,
                min_occupancy_s=args.min_occupancy_s,
            )
        unit_mask, unit_classes = build_meme_unit_mask(
            session=session,
            f_by_epoch=f_all_by_epoch,
            region=region,
            light_epochs=light_epochs,
            dark_epoch=dark_epoch,
            min_firing_rate_hz=min_firing_rate_hz,
            min_condition_sd_hz=args.min_condition_sd_hz,
            unit_filter_mode=args.unit_filter_mode,
        )
        n_units_by_class_by_region[region] = {
            unit_class: int(np.sum(unit_classes == unit_class))
            for unit_class in (
                "shared_active",
                "dark_only",
                "light_only",
                "inactive_or_low_mod",
            )
        }

        for epoch in analysis_epochs:
            F = f_all_by_epoch[epoch][:, :, unit_mask]
            n_neurons_by_region[region][epoch] = F.shape[-1]

            res = meme_eigenmoments_and_pr_mc(
                F,
                k_moms=6,
                remove_mean=True,
                n_pairings=args.full_n_pairings,
                n_bin_perms=args.full_n_bin_perms,
                max_repeat_pairs=None,
                random_seed=args.random_seed,
            )
            meme_moments[region][epoch] = res.moments
            meme_pr[region][epoch] = res.pr

            repeat = random_seed_repeat_meme_pr(
                session,
                epoch=epoch,
                region=region,
                dark_epoch=dark_epoch,
                n_random_repeats=args.n_random_repeats,
                unit_mask=unit_mask,
                bin_size=args.bin_size_cm,
                n_groups=args.n_groups,
                min_occupancy_s=args.min_occupancy_s,
                k_moms=6,
                remove_mean=True,
                n_pairings=args.repeat_n_pairings,
                n_bin_perms=args.repeat_n_bin_perms,
                max_repeat_pairs=None,
                random_seed=args.random_seed,
                ci=args.random_repeat_ci,
                center="mean",
            )
            meme_repeat_pr[region][epoch] = repeat.pr_samples
            meme_repeat_summary[region][epoch] = repeat.summary

            naive_pca = naive_cov_eigs_and_pr(F, mode="mean_repeat", center=True)
            naive_eigenvalues[region][epoch] = naive_pca.eigenvalues
            naive_pr[region][epoch] = naive_pca.pr

        n_neurons = _stable_region_neuron_count(n_neurons_by_region[region], region)
        print(f"  Plotting for {region} (N={n_neurons})...")

        saved_figures.append(
            plot_eigenspectrum_comparison(
                region=region,
                all_moments=meme_moments[region],
                naive_eigenvalues=naive_eigenvalues[region],
                n_neurons=n_neurons,
                save_path=fig_dir,
                metric="variance_explained",
            )
        )
        saved_figures.append(
            plot_eigenspectrum_comparison(
                region=region,
                all_moments=meme_moments[region],
                naive_eigenvalues=naive_eigenvalues[region],
                n_neurons=n_neurons,
                save_path=fig_dir,
                metric="eigenvalue",
            )
        )
        saved_figures.append(
            plot_broken_power_law_comparison(
                region=region,
                all_moments=meme_moments[region],
                naive_eigenvalues=naive_eigenvalues[region],
                n_neurons=n_neurons,
                save_path=fig_dir,
                metric="variance_explained",
                max_rank=1000,
            )
        )
        saved_figures.append(
            plot_broken_power_law_comparison(
                region=region,
                all_moments=meme_moments[region],
                naive_eigenvalues=naive_eigenvalues[region],
                n_neurons=n_neurons,
                save_path=fig_dir,
                metric="eigenvalue",
                max_rank=1000,
            )
        )
        saved_figures.append(
            plot_pr(
                region=region,
                meme_pr=meme_pr[region],
                naive_pr=naive_pr[region],
                save_path=fig_dir,
            )
        )
        saved_figures.append(
            plot_pr_light_dark_epochwise(
                region=region,
                epoch_to_mc_pr=meme_repeat_pr[region],
                epoch_to_condition=epoch_to_condition,
                save_path=fig_dir,
                ci=args.random_repeat_ci,
                center="mean",
                annotate=True,
                random_seed=0,
                title_suffix=" (random-seed repeats)",
            )
        )
        groupmean_path, dark_draws, light_draws, delta_draws = plot_pr_light_dark_groupmeans(
            region=region,
            epoch_to_mc_pr=meme_repeat_pr[region],
            epoch_to_condition=epoch_to_condition,
            save_path=fig_dir,
            ci=args.random_repeat_ci,
            center="mean",
            n_draws=DEFAULT_GROUPMEAN_N_DRAWS,
            random_seed=0,
            title_suffix=" (random-seed repeats)",
        )
        saved_figures.append(groupmean_path)
        comparison_draws_by_region[region] = {
            "dark_draws": dark_draws,
            "light_draws": light_draws,
            "delta_draws": delta_draws,
        }

    saved_epoch_tables = save_epoch_summary_tables(
        out_dir,
        regions=list(args.regions),
        analysis_epochs=analysis_epochs,
        epoch_to_condition=epoch_to_condition,
        n_neurons_by_region=n_neurons_by_region,
        meme_pr=meme_pr,
        naive_pr=naive_pr,
        meme_repeat_summary=meme_repeat_summary,
        min_firing_rate_by_region=min_firing_rate_by_region,
        n_units_by_class_by_region=n_units_by_class_by_region,
        settings=settings,
    )
    saved_comparison_tables = save_light_dark_comparison_tables(
        out_dir,
        regions=list(args.regions),
        light_epochs=light_epochs,
        dark_epoch=dark_epoch,
        comparison_draws_by_region=comparison_draws_by_region,
        ci=args.random_repeat_ci,
        center="mean",
        n_draws=DEFAULT_GROUPMEAN_N_DRAWS,
    )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.signal_dim.meme",
        parameters=settings,
        outputs={
            "run_epochs": run_epochs,
            "analysis_epochs": analysis_epochs,
            "light_epoch": args.light_epoch,
            "light_epochs": light_epochs,
            "dark_epoch": dark_epoch,
            "saved_epoch_tables": saved_epoch_tables,
            "saved_comparison_tables": saved_comparison_tables,
            "saved_figures": saved_figures,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
