from __future__ import annotations
import os

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, List, Iterable

import numpy as np
import numpy.typing as npt
import pynapple as nap
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import position_tools as pt
import spikeinterface.full as si
import track_linearization as tl
import math

animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

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

epoch_list = list(timestamps_ephys.keys())
run_epoch_list = epoch_list[1::2]
run_epoch_list = run_epoch_list[:4]
sleep_epoch_list = epoch_list[::2]


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


def _sampling_rate(t_position: np.ndarray) -> float:
    return (len(t_position) - 1) / (t_position[-1] - t_position[0])


spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])


ep = {}
for epoch in epoch_list[:-2]:
    ep[epoch] = nap.IntervalSet(
        start=timestamps_ephys[epoch][0],
        end=timestamps_ephys[epoch][-1],
    )

speed = {}
movement = {}
for epoch in epoch_list[:-2]:
    speed[epoch] = nap.Tsd(
        t=timestamps_position_dict[epoch][position_offset:],
        d=pt.get_speed(
            position=position_dict[epoch][position_offset:],
            time=timestamps_position_dict[epoch][position_offset:],
            sampling_frequency=(
                len(timestamps_position_dict[epoch][position_offset:]) - 1
            )
            / (
                timestamps_position_dict[epoch][position_offset:][-1]
                - timestamps_position_dict[epoch][position_offset:][0]
            ),
            sigma=0.1,
        ),
    )
    movement[epoch] = speed[epoch].threshold(speed_threshold, method="above")

total_fr = {}
for region in regions:
    total_fr[region] = {}
    for epoch in epoch_list[:-2]:
        total_fr[region][epoch] = (
            spikes[region].count(ep=ep[epoch]).to_numpy() / ep[epoch].tot_length()
        ).ravel()

fr_during_movement = {}
for region in regions:
    fr_during_movement[region] = {}
    for epoch in epoch_list[:-2]:
        fr_during_movement[region][epoch] = (
            np.sum(
                spikes[region].count(ep=movement[epoch].time_support).to_numpy()
                / movement[epoch].time_support.tot_length(),
                axis=0,
            )
        ).ravel()


ArrayF = npt.NDArray[np.floating]


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


def _split_indices_into_groups(
    n: int, n_groups: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """
    Randomly assign indices 0..n-1 into n_groups disjoint groups (as equal as possible).
    Returns list of arrays of indices (length n_groups).
    """
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2.")
    if n < n_groups:
        raise ValueError(
            f"Need at least n_groups laps. Got n={n}, n_groups={n_groups}."
        )

    perm = rng.permutation(n)
    groups = np.array_split(perm, n_groups)
    # Sanity: no empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError(
            "Grouping produced <2 non-empty groups; increase laps or reduce n_groups."
        )
    return groups


def _sample_lap_indices(
    n_available: int,
    *,
    lap_fraction: float,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a subset of laps without replacement while keeping enough laps
    to form `n_groups` disjoint groups.
    """
    if not (0.0 < lap_fraction <= 1.0):
        raise ValueError("lap_fraction must be in (0, 1].")
    if n_available < n_groups:
        raise ValueError(
            f"Need at least n_groups laps. Got n_available={n_available}, "
            f"n_groups={n_groups}."
        )

    n_select = int(np.ceil(float(lap_fraction) * n_available))
    n_select = min(n_available, max(n_groups, n_select))
    return np.sort(rng.choice(n_available, size=n_select, replace=False))


def _split_lap_indices_into_groups(
    lap_indices: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Randomly split the provided lap indices into n_groups disjoint groups
    (as equal as possible).
    """
    lap_indices = np.asarray(lap_indices, dtype=np.int64)
    if lap_indices.ndim != 1:
        raise ValueError("lap_indices must be a 1D array.")
    if lap_indices.size < n_groups:
        raise ValueError(
            f"Need at least n_groups sampled laps. Got n={lap_indices.size}, "
            f"n_groups={n_groups}."
        )

    groups = np.array_split(rng.permutation(lap_indices), n_groups)
    groups = [g.astype(np.int64, copy=False) for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError(
            "Grouping produced <2 non-empty groups; increase laps or reduce n_groups."
        )
    return groups


def _occupancy_mask_1d(
    linpos_tsd: nap.Tsd,
    epochs: nap.IntervalSet,
    bin_edges: np.ndarray,
    *,
    min_occupancy_s: float,
) -> np.ndarray:
    """
    Compute a boolean mask over 1D bins indicating bins with >= min_occupancy_s
    of occupancy within `epochs`.

    We compute occupancy from the restricted position samples.
    """
    if min_occupancy_s <= 0:
        raise ValueError("min_occupancy_s must be > 0.")

    pos = linpos_tsd.restrict(epochs)
    # Pynapple Tsd typically has .t (times) and .d (data)
    t = np.asarray(pos.t)
    x = np.asarray(pos.d)

    if x.size < 2:
        # No samples => no valid bins
        return np.zeros(len(bin_edges) - 1, dtype=bool)

    # dt from median sampling interval
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        # fallback: average dt
        dt = float((t[-1] - t[0]) / max(len(t) - 1, 1))

    counts, _ = np.histogram(x, bins=bin_edges)
    occ_s = counts.astype(np.float64) * dt
    return occ_s >= float(min_occupancy_s)


def prepare_F(
    epoch: str,
    region: str,
    *,
    fr_dark_threshold: Optional[float] = None,
    bin_size: float = 4.0,
    n_groups: int = 4,
    group_seed: int = 0,
    min_occupancy_s: float = 0.05,
    lap_fraction: float = 1.0,
) -> ArrayF:
    """
    Build F with shape (R, C, N) where:
      - R = n_groups repeat-groups made from disjoint subsets of laps
      - C = concatenated spatial bins across trajectory types (after dropping low-occupancy bins)
      - N = neurons

    Key differences from your old prepare_F:
      - repeats are *groups of laps*, not single laps
      - no interpolation / ffill / bfill
      - bins with insufficient occupancy are dropped consistently across repeats
      - if lap_fraction < 1, each trajectory first subsamples laps without replacement
    """
    rng = np.random.default_rng(group_seed)

    fr_dark = fr_during_movement[region]["08_r4"]  # (N,)

    # ----- Build trajectory IntervalSets from trajectory_times -----
    traj_starts: Dict[str, np.ndarray] = {}
    traj_ends: Dict[str, np.ndarray] = {}
    n_trials: Dict[str, int] = {}
    for traj in trajectory_types:
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

    # ----- Track geometry (same as your code; kept here) -----
    dx = 9.5
    dy = 9.0
    diagonal_segment_length = np.sqrt(dx**2 + dy**2)

    long_segment_length = 81 - 17 - 2
    short_segment_length = 13.5

    total_length_per_trajectory = (
        long_segment_length * 2 + short_segment_length + 2 * diagonal_segment_length
    )

    node_positions_right = np.array(
        [
            (55.5, 81),
            (55.5, 81 - long_segment_length),
            (55.5 - dx, 81 - long_segment_length - dy),
            (55.5 - dx - short_segment_length, 81 - long_segment_length - dy),
            (55.5 - 2 * dx - short_segment_length, 81 - long_segment_length),
            (55.5 - 2 * dx - short_segment_length, 81),
        ]
    )

    node_positions_left = np.array(
        [
            (55.5, 81),
            (55.5, 81 - long_segment_length),
            (55.5 + dx, 81 - long_segment_length - dy),
            (55.5 + dx + short_segment_length, 81 - long_segment_length - dy),
            (55.5 + 2 * dx + short_segment_length, 81 - long_segment_length),
            (55.5 + 2 * dx + short_segment_length, 81),
        ]
    )

    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    linear_edge_order = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    linear_edge_spacing = 0

    track_graph_left = tl.make_track_graph(node_positions_left, edges)
    track_graph_right = tl.make_track_graph(node_positions_right, edges)

    # ----- Movement epoch -----
    speed_arr = pt.get_speed(
        position=position_dict[epoch][position_offset:],
        time=timestamps_position_dict[epoch][position_offset:],
        sampling_frequency=_sampling_rate(
            timestamps_position_dict[epoch][position_offset:]
        ),
        sigma=0.1,
    )
    speed_tsd = nap.Tsd(
        t=timestamps_position_dict[epoch][position_offset:], d=speed_arr
    )
    movement = speed_tsd.threshold(speed_threshold, method="above")

    # ----- Linearized position per trajectory -----
    linear_position: Dict[str, nap.Tsd] = {}
    for traj in trajectory_types:
        if "right" in traj:
            tg = track_graph_right
        else:
            tg = track_graph_left

        position_df = tl.get_linearized_position(
            position=position_dict[epoch][position_offset:],
            track_graph=tg,
            edge_order=linear_edge_order,
            edge_spacing=linear_edge_spacing,
        )

        traj_ep = nap.IntervalSet(start=traj_starts[traj], end=traj_ends[traj])

        linear_position[traj] = nap.Tsd(
            t=timestamps_position_dict[epoch][position_offset:],
            d=position_df["linear_position"],
            time_support=traj_ep,
        )

    # ----- Bin edges -----
    bin_edges = np.arange(
        0, total_length_per_trajectory + bin_size, bin_size, dtype=np.float64
    )
    n_bins = len(bin_edges) - 1
    if n_bins < 2:
        raise ValueError("binning produced <2 bins; decrease bin_size?")

    # ----- Build grouped epochs and compute tuning curves -----
    # Store per-trajectory per-group tuning curves and occupancy masks
    tc_store: Dict[str, List[np.ndarray]] = {traj: [] for traj in trajectory_types}
    occ_masks: Dict[str, List[np.ndarray]] = {traj: [] for traj in trajectory_types}

    for traj in trajectory_types:
        selected_laps = _sample_lap_indices(
            n_trials[traj],
            lap_fraction=lap_fraction,
            n_groups=n_groups,
            rng=rng,
        )
        lap_groups = _split_lap_indices_into_groups(selected_laps, n_groups, rng=rng)

        for g_inds in lap_groups:
            g_starts = traj_starts[traj][g_inds]
            g_ends = traj_ends[traj][g_inds]
            g_ep = nap.IntervalSet(start=g_starts, end=g_ends)

            # only keep moving time
            use_ep = g_ep.intersect(movement.time_support)

            # tuning curve: (units, bins)
            tc = nap.compute_tuning_curves(
                data=spikes[region],
                features=linear_position[traj],
                bins=[bin_edges],
                epochs=use_ep,
                feature_names=["linpos"],
            )
            tc_np = np.asarray(tc.to_numpy(), dtype=np.float64)  # (N, n_bins)

            # occupancy mask for this group
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

    # ----- Choose bins to keep: intersection across groups (per trajectory) -----
    keep_masks: Dict[str, np.ndarray] = {}
    for traj in trajectory_types:
        keep = np.logical_and.reduce(occ_masks[traj])
        if keep.sum() < 2:
            raise ValueError(
                f"Too few valid bins after occupancy filtering for {traj}: "
                f"{keep.sum()} kept. Increase laps per group, increase bin_size, "
                f"or lower min_occupancy_s."
            )
        keep_masks[traj] = keep

    # ----- Stack into F: (R=n_groups, C=sum kept bins across trajs, N) -----
    F_list: List[np.ndarray] = []
    for g in range(n_groups):
        pieces = []
        for traj in trajectory_types:
            tc_np = tc_store[traj][g]  # (N, n_bins)
            tc_np = np.nan_to_num(tc_np, nan=0.0)  # should be rare after filtering
            kept = keep_masks[traj]
            # transpose to (bins, neurons)
            pieces.append(tc_np[:, kept].T)
        Xg = np.concatenate(pieces, axis=0)  # (C, N)
        F_list.append(Xg)

    F = np.stack(F_list, axis=0)  # (R, C, N)

    # ----- Optional neuron filtering -----
    if fr_dark_threshold is not None:
        keep_neurons = np.asarray(fr_dark > fr_dark_threshold)
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
class BootstrapPRResult:
    pr_samples: ArrayF
    summary: PRSummary
    n_bootstraps: int
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
    """
    Apply (even - odd)/sqrt(2) but with random disjoint pairing.
    Uses the *same* pairing for all repeats in F (important).
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
    """
    Monte Carlo MEME moments for spatial tuning curves:

    For each of n_pairings:
      - form disjoint-difference mean removal (random pairing)  -> reduces C to floor(C/2)
      For each of n_bin_perms:
        - randomly permute condition (bin) order
        - compute Kong–Valiant moments using all (or subsampled) repeat pairs
    Average moments across all MC draws, then PR = N * m1^2 / m2.

    This implements:
      - Eq. 5 eigenmoment estimator (Kong–Valiant style)
      - disjoint-difference mean removal (instead of subtracting sample mean)
      - variance reduction by averaging over many disjoint differences
    See paper Materials & Methods. :contentReference[oaicite:1]{index=1}
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


def bootstrap_meme_pr(
    epoch: str,
    region: str,
    *,
    n_bootstraps: int = 200,
    lap_fraction: float = 0.7,
    fr_dark_threshold: Optional[float] = None,
    bin_size: float = 4.0,
    n_groups: int = 4,
    min_occupancy_s: float = 0.01,
    k_moms: int = 6,
    remove_mean: bool = True,
    n_pairings: int = 200,
    n_bin_perms: int = 5,
    max_repeat_pairs: Optional[int] = None,
    random_seed: int = 0,
    ci: float = 95.0,
    center: Literal["mean", "median"] = "mean",
) -> BootstrapPRResult:
    """
    Estimate PR uncertainty by resampling laps, then rebuilding grouped repeats.

    For each bootstrap draw:
      1. sample lap_fraction of laps per trajectory without replacement
      2. split sampled laps into n_groups disjoint groups
      3. build F and compute PR with MEME
    """
    rng = np.random.default_rng(random_seed)
    pr_samples = np.full((n_bootstraps,), np.nan, dtype=np.float64)

    for b in range(n_bootstraps):
        f_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        meme_seed = int(rng.integers(0, np.iinfo(np.int32).max))

        try:
            F_b = prepare_F(
                epoch=epoch,
                region=region,
                fr_dark_threshold=fr_dark_threshold,
                bin_size=bin_size,
                n_groups=n_groups,
                group_seed=f_seed,
                min_occupancy_s=min_occupancy_s,
                lap_fraction=lap_fraction,
            )
            res_b = meme_eigenmoments_and_pr_mc(
                F_b,
                k_moms=k_moms,
                remove_mean=remove_mean,
                n_pairings=n_pairings,
                n_bin_perms=n_bin_perms,
                max_repeat_pairs=max_repeat_pairs,
                random_seed=meme_seed,
            )
            pr_samples[b] = res_b.pr
        except ValueError as exc:
            print(
                f"[bootstrap_meme_pr] skipping bootstrap {b} for {region} {epoch}: {exc}"
            )

    summary = summarize_mc_pr(pr_samples, ci=ci, center=center)
    return BootstrapPRResult(
        pr_samples=pr_samples,
        summary=summary,
        n_bootstraps=n_bootstraps,
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


def _demean_via_disjoint_differences_per_repeat(
    F: ArrayF,
    *,
    pairing: Literal["adjacent", "random"] = "adjacent",
    random_seed: int = 0,
    block_size: Optional[int] = None,
    return_pairs: bool = False,
) -> ArrayF | Tuple[ArrayF, np.ndarray, np.ndarray]:
    """
    Apply disjoint differences along C:
        (F[:, a, :] - F[:, b, :]) / sqrt(2)

    Returns optionally the index arrays (a, b) used.
    """
    if F.ndim != 3:
        raise ValueError(f"F must have shape (R, C, N). Got {F.shape}.")
    R, C, N = F.shape
    if C < 2:
        raise ValueError("Need at least 2 conditions.")

    rng = np.random.default_rng(random_seed)

    def _pairs(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pairing == "adjacent":
            m = (len(idxs) // 2) * 2
            idxs = idxs[:m]
            return idxs[0::2], idxs[1::2]
        if pairing == "random":
            perm = rng.permutation(idxs)
            m = (len(perm) // 2) * 2
            perm = perm[:m]
            return perm[0::2], perm[1::2]
        raise ValueError(f"Unknown pairing={pairing!r}.")

    if block_size is None:
        a, b = _pairs(np.arange(C))
        D = (F[:, a, :] - F[:, b, :]) / np.sqrt(2.0)
        return (D, a, b) if return_pairs else D

    if block_size <= 0 or C % block_size != 0:
        raise ValueError("block_size must be positive and divide C.")

    diffs = []
    a_all = []
    b_all = []
    for start in range(0, C, block_size):
        idxs = np.arange(start, start + block_size)
        a, b = _pairs(idxs)
        diffs.append((F[:, a, :] - F[:, b, :]) / np.sqrt(2.0))
        a_all.append(a)
        b_all.append(b)

    D = np.concatenate(diffs, axis=1)
    if return_pairs:
        return D, np.concatenate(a_all), np.concatenate(b_all)
    return D


def _demean_via_disjoint_differences_per_repeat2(
    F: ArrayF,
    *,
    pairing: Literal["adjacent", "random"] = "adjacent",
    random_seed: int = 0,
    return_pairs: bool = False,
) -> ArrayF | Tuple[ArrayF, np.ndarray, np.ndarray]:
    """
    Apply disjoint differences along C (trials):
        (F[:, a, :] - F[:, b, :]) / sqrt(2)

    Returns optionally the index arrays (a, b) used.
    """
    if F.ndim != 3:
        raise ValueError(f"F must have shape (R, C, N). Got {F.shape}.")
    R, C, N = F.shape  # R: repeats (or trials), C: conditions (or spatial bins)
    if C < 2:
        raise ValueError("Need at least 2 conditions (C >= 2).")

    rng = np.random.default_rng(random_seed)

    # Create pairs of single trials
    def _pairs(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pairing == "adjacent":
            m = (len(idxs) // 2) * 2  # Ensure even number of trials
            idxs = idxs[:m]  # Adjust for odd length
            return idxs[0::2], idxs[1::2]  # Pair adjacent trials
        if pairing == "random":
            perm = rng.permutation(idxs)
            m = (len(perm) // 2) * 2
            perm = perm[:m]
            return perm[0::2], perm[1::2]  # Random trial pairing
        raise ValueError(f"Unknown pairing={pairing!r}.")

    # For each pair of trials, calculate disjoint differences
    a, b = _pairs(np.arange(C))  # Pair trials (conditions) from C
    D = (F[:, a, :] - F[:, b, :]) / np.sqrt(2.0)

    return (D, a, b) if return_pairs else D


@dataclass(frozen=True)
class EigenmomentResult:
    moments: ArrayF  # (k_moms,)
    pr: float  # participation ratio
    pair_moments: ArrayF  # (n_pairs, k_moms)


def eigenmoments_and_pr_single_trial_pairwise(
    F: ArrayF,
    *,
    k_moms: int = 6,
    remove_mean: bool = True,
    max_pairs: Optional[int] = None,
    pairing="random",  # Pairing can be adjacent or random
    random_seed: int = 0,
) -> EigenmomentResult:
    """
    Computes eigenmoments and participation ratio using single trial pairwise moments.
    """
    if F.ndim != 3:
        raise ValueError(f"Expected F to be 3D (R, C, N). Got {F.shape}.")

    if F.shape[1] < 2:
        raise ValueError(f"Expected at least 2 conditions. Got C={F.shape[1]}.")

    # Get number of repeats (R), conditions (C), and neurons (N)
    R, C, N = F.shape
    if R < 2:
        raise ValueError("Need at least 2 repeats/groups.")
    if k_moms < 2:
        raise ValueError("Use k_moms>=2 to compute participation ratio.")

    # X = (
    #     _demean_via_disjoint_differences_per_repeat(
    #         F, pairing=pairing, random_seed=random_seed
    #     )
    #     if remove_mean
    #     else F
    # )

    X = (
        _demean_via_disjoint_differences_per_repeat2(
            F, pairing=pairing, random_seed=random_seed
        )
        if remove_mean
        else F
    )
    # X = np.nan_to_num(X, nan=0.0).astype(np.float64, copy=False)

    # Continue with pairwise calculations as before
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    if max_pairs is not None and max_pairs < len(pairs):
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in keep]

    pair_moms = np.empty((len(pairs), k_moms), dtype=np.float64)
    for t, (i, j) in enumerate(pairs):
        pair_moms[t] = _meme_moments_from_pair(X[i], X[j], k_moms=k_moms)

    moms = pair_moms.mean(axis=0)
    m1, m2 = float(moms[0]), float(moms[1])

    # Participation ratio
    pr = float("nan") if m2 <= 0 else float(N) * (m1 * m1) / m2

    return EigenmomentResult(moments=moms, pr=pr, pair_moments=pair_moms)


def eigenmoments_and_pr(
    F: ArrayF,
    *,
    k_moms: int = 6,
    remove_mean: bool = True,
    max_pairs: Optional[int] = None,
    pairing="random",
    random_seed: int = 0,
) -> EigenmomentResult:
    """
    Compute eigenmoments and participation ratio from (R, C, N) repeats.

    Parameters
    ----------
    F
        Shape: (R, C, N) = (groups, position_bins, neurons).
        Entries: firing rates (recommended) or counts.
    k_moms
        Number of eigenmoments to estimate (need >=2 for PR).
    remove_mean
        If True, uses disjoint differencing along the position-bin axis.
        Leave True unless you know your tuning curves already have zero-mean across bins.
    max_pairs
        If set, randomly subsample repeat pairs for speed.
    random_seed
        RNG seed for pair subsampling.

    Returns
    -------
    result
        moments: (k_moms,) with m_p = (1/N) * sum lambda_i^p
        pr: m1^2 / m2
        pair_moments: per-pair estimates for uncertainty/bootstrapping
    """
    if F.ndim != 3:
        raise ValueError(f"Expected F to be 3D (R, C, N). Got {F.shape}.")

    R, C, N = F.shape
    if R < 2:
        raise ValueError("Need at least 2 repeats/groups.")
    if k_moms < 2:
        raise ValueError("Use k_moms>=2 to compute participation ratio.")

    X = (
        _demean_via_disjoint_differences_per_repeat(
            F, pairing=pairing, random_seed=random_seed
        )
        if remove_mean
        else F
    )
    X = np.nan_to_num(X, nan=0.0).astype(np.float64, copy=False)

    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    if max_pairs is not None and max_pairs < len(pairs):
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in keep]

    pair_moms = np.empty((len(pairs), k_moms), dtype=np.float64)
    for t, (i, j) in enumerate(pairs):
        pair_moms[t] = _meme_moments_from_pair(X[i], X[j], k_moms=k_moms)

    moms = pair_moms.mean(axis=0)
    m1, m2 = float(moms[0]), float(moms[1])

    # Participation ratio
    # pr = float("nan") if m2 <= 0 else (m1 * m1) / m2
    pr = float("nan") if m2 <= 0 else float(N) * (m1 * m1) / m2

    return EigenmomentResult(moments=moms, pr=pr, pair_moments=pair_moms)


# ----------------------------
# Optional: reconstruct a discrete eigenspectrum by moment-matching
# ----------------------------
def estimate_eigenspectrum_discrete(
    moments: ArrayF,
    *,
    n_grid: int = 200,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    ridge: float = 1e-6,
) -> Tuple[ArrayF, ArrayF]:
    """
    Rough eigenspectrum estimate by fitting a nonnegative discrete distribution
    over eigenvalues that matches the first K moments.

    We solve a ridge-stabilized least-squares:
        V w ≈ m,  w>=0, sum(w)=1

    where V[p, j] = lam_j^(p+1),  m[p] = moment_{p+1}.

    Returns
    -------
    lam_grid
        Eigenvalue grid (log-spaced), shape (n_grid,)
    weights
        Nonnegative weights, shape (n_grid,), sum to ~1
    """
    K = moments.shape[0]
    if K < 2:
        raise ValueError("Need at least 2 moments for a stable spectrum fit.")

    m1, m2 = float(moments[0]), float(moments[1])
    if lam_max is None:
        lam_max = max(10.0 * m1, 1e-8)
    if lam_min is None:
        # crude scale using m2 and m1
        lam_min = max(min(m1 * 1e-3, m2 / (m1 + 1e-12) * 1e-3), 1e-12)

    lam_grid = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_grid)).astype(
        np.float64
    )

    # Vandermonde-like matrix for moments (K x n_grid)
    p = np.arange(1, K + 1, dtype=np.float64)[:, None]
    V = lam_grid[None, :] ** p  # moment p is E[lambda^p]

    # Ridge-stabilized solve for unconstrained weights
    # Then project to simplex (nonnegative, sum=1) via clipping + renorm.
    A = V.T @ V + ridge * np.eye(n_grid)
    b = V.T @ moments
    w = np.linalg.solve(A, b)

    w = np.clip(w, 0.0, np.inf)
    s = float(w.sum())
    if s > 0:
        w /= s

    return lam_grid, w


@dataclass(frozen=True)
class NaiveSpectrumResult:
    """
    Naive (sample) eigenspectrum + participation ratio computed directly
    from a covariance matrix estimated from F.

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
    Compute a naive neuron-by-neuron covariance from F and return its eigenspectrum
    and participation ratio (PR).

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
    - This is a *sample* covariance method; it is generally noise-biased compared to MEME.
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


def fit_broken_power_law_from_meme_moments(
    moments: ArrayF,
    *,
    n_neurons: int,
    alpha_bounds: Tuple[float, float] = (1e-3, 5.0),
    grid_size_alpha: int = 120,
    k0_candidates: Optional[npt.NDArray[np.integer]] = None,
    weights: Optional[ArrayF] = None,
    use_log_moments: bool = True,
) -> PowerLawFit:
    """
    Fit a continuous broken power-law eigenspectrum to MEME moments.

    Uses a grid search over (k0, alpha1, alpha2) and eliminates c using m1:
        c = m1 / mean_i base_i

    Parameters
    ----------
    moments
        MEME averaged eigenmoments (K,)
    n_neurons
        N
    alpha_bounds
        Range for alpha1 and alpha2
    grid_size_alpha
        Grid points for each alpha dimension
    k0_candidates
        Candidate breakpoints; if None uses a reasonable default set
    weights
        Optional weights for p=2..K (shape K-1); if None uses 1/p^2
    use_log_moments
        Fit in log-space

    Returns
    -------
    fit
        PowerLawFit with alpha1, alpha2, k0, c
    """
    moments = np.asarray(moments, dtype=np.float64)
    if moments.ndim != 1 or moments.size < 3:
        raise ValueError("moments must be shape (K,) with K>=3 for broken power-law.")
    if n_neurons < 3:
        raise ValueError("n_neurons must be >= 3.")

    K = int(moments.size)
    m1_hat = float(moments[0])
    target = moments[1:]  # p=2..K

    p_orders = np.arange(2, K + 1, dtype=np.float64)
    if weights is None:
        w = 1.0 / (p_orders**2)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != target.shape:
            raise ValueError(f"weights must have shape {(K-1,)}.")

    if k0_candidates is None:
        # log-spaced + a few small values; avoids huge search
        log_k = np.unique(np.round(np.logspace(0, np.log10(n_neurons), 40)).astype(int))
        small_k = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60], dtype=int)
        k0_candidates = np.unique(
            np.clip(np.concatenate([small_k, log_k]), 1, n_neurons)
        )
    else:
        k0_candidates = np.unique(np.asarray(k0_candidates, dtype=int))
        k0_candidates = k0_candidates[
            (k0_candidates >= 1) & (k0_candidates <= n_neurons)
        ]
        if k0_candidates.size == 0:
            raise ValueError("k0_candidates must include values in [1, n_neurons].")

    i = np.arange(1, n_neurons + 1, dtype=np.float64)
    alphas = np.linspace(
        alpha_bounds[0], alpha_bounds[1], grid_size_alpha, dtype=np.float64
    )

    best_obj = np.inf
    best = None  # (alpha1, alpha2, k0, c)

    for k0 in k0_candidates:
        left = i <= k0
        i_left = i[left]
        i_right = i[~left]

        left_pow = i_left[None, :] ** (-alphas[:, None])  # (A, n_left)
        right_pow = i_right[None, :] ** (-alphas[:, None])  # (A, n_right)

        for a1_idx, alpha1 in enumerate(alphas):
            base_left = left_pow[a1_idx]

            for a2_idx, alpha2 in enumerate(alphas):
                cont = float(k0 ** (alpha2 - alpha1))
                base_right = cont * right_pow[a2_idx]

                base = np.empty(n_neurons, dtype=np.float64)
                base[left] = base_left
                base[~left] = base_right

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
                    best = (float(alpha1), float(alpha2), int(k0), float(c))

    if best is None:
        raise RuntimeError("Broken power-law fit failed.")

    alpha1, alpha2, k0, c = best
    return PowerLawFit(
        kind="broken_power_law",
        n_neurons=n_neurons,
        alpha1=alpha1,
        alpha2=alpha2,
        k0=k0,
        c=c,
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
    max_rank: int = 100,  # <--- NEW PARAMETER
):
    """
    Plot the eigenspectrum focused on the top ranks.

    Parameters
    ----------
    max_rank : int
        Number of eigenvalues to plot (e.g., 100).
        The y-axis will auto-scale to fit these specific values.
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

    # --- 2. NAIVE PLOTTING ---
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
    ax[1].set_title(f"Naive PCA\n(First {limit} modes)")
    ax[0].set_ylabel(ylabel)

    fig.suptitle(f"Region: {region} | Metric: {metric}")
    plt.tight_layout()

    filename = f"{region}_eigenspectrum_{metric}_top{limit}.png"
    plt.savefig(save_path / filename, dpi=300)
    plt.close(fig)

    return None


def plot_broken_power_law_comparison(
    region: str,
    all_moments: dict,
    naive_eigenvalues: dict,
    n_neurons: int,
    save_path: Path,
    metric: Literal["eigenvalue", "variance_explained"] = "variance_explained",
    max_rank: int = 1000,
):
    """
    Plot the eigenspectrum with BROKEN Power Law fits focused on the top ranks.
    Uses the optimized fitting function.
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

    # --- 2. NAIVE PLOTTING ---
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
    ax[1].set_title(f"Naive PCA\n(Sample Eigenspectrum)")
    ax[0].set_ylabel(ylabel)

    fig.suptitle(f"Region: {region} | Metric: {metric} | Broken Power Law")
    plt.tight_layout()

    filename = f"{region}_broken_power_law_{metric}_top{limit}.png"
    plt.savefig(save_path / filename, dpi=300)
    plt.close(fig)

    return None


def plot_pr(region, meme_pr, naive_pr, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = list(meme_pr.keys())
    # Sort epochs if needed, assuming they are strings they might need sorting logic

    ax.plot(epochs, list(meme_pr.values()), "-o", label="MEME")
    ax.plot(epochs, list(naive_pr.values()), "-s", label="Naive")

    ax.set_ylabel("Participation Ratio")
    ax.set_title(f"Region: {region}")
    ax.legend()
    plt.savefig(save_path / f"{region}_pr.png", dpi=300)
    plt.close(fig)
    return None


import matplotlib.pyplot as plt
from pathlib import Path


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
    """
    Plot per-epoch PR estimates with CI error bars, grouped by condition (light/dark).

    epoch_to_mc_pr: epoch -> 1D array of MC PR draws
    epoch_to_condition: epoch -> "light" or "dark"
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
    ax.set_title(f"{region}: PR by epoch with {ci:.0f}% CI{title_suffix}")
    ax.grid(True, alpha=0.25)

    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"{region}_pr_light_vs_dark_epochwise_ci{int(ci)}.png"
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
    """
    Plot condition-level mean PR (averaged across epochs) with CI derived from mc_pr.
    Also returns the MC distributions for dark, light, and (light - dark).
    """
    rng = np.random.default_rng(random_seed)

    # Partition epochs by condition
    dark_epochs = [
        e
        for e, c in epoch_to_condition.items()
        if str(c).lower() == "dark" and e in epoch_to_mc_pr
    ]
    light_epochs = [
        e
        for e, c in epoch_to_condition.items()
        if str(c).lower() == "light" and e in epoch_to_mc_pr
    ]

    if len(dark_epochs) == 0 or len(light_epochs) == 0:
        raise ValueError(
            f"Need >=1 epoch in each condition. Found dark={len(dark_epochs)}, light={len(light_epochs)}."
        )

    dark_draws = _mc_mean_across_epochs(
        epoch_to_mc_pr, dark_epochs, n_draws=n_draws, rng=rng
    )
    light_draws = _mc_mean_across_epochs(
        epoch_to_mc_pr, light_epochs, n_draws=n_draws, rng=rng
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
    ax.set_title(f"{region}: mean PR Light vs Dark ({ci:.0f}% CI){title_suffix}")
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
    out = save_path / f"{region}_pr_light_vs_dark_groupmean_ci{int(ci)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)

    return out, dark_draws, light_draws, delta_draws


def main():
    out_dir = analysis_path / "meme3"
    fig_dir = analysis_path / "figs" / "meme3"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_lap_fraction = 0.7
    n_bootstraps = 200
    bootstrap_ci = 95.0
    full_n_pairings = 1000
    full_n_bin_perms = 5
    bootstrap_n_pairings = 200
    bootstrap_n_bin_perms = 5

    # Initialize storage
    meme_moments = {region: {} for region in regions}
    naive_eigenvalues = {region: {} for region in regions}
    meme_pr = {region: {} for region in regions}
    naive_pr = {region: {} for region in regions}
    meme_boot_pr = {region: {} for region in regions}
    meme_boot_summary = {region: {} for region in regions}

    epoch_to_condition = {
        run_epoch_list[0]: "light",
        run_epoch_list[1]: "light",
        run_epoch_list[2]: "light",
        run_epoch_list[3]: "dark",
    }

    # Track N per region to pass to plotting later
    n_neurons_per_region = {}

    for region in regions:
        if region == "v1":
            # fr_dark_threshold = 0.5
            fr_dark_threshold = None
        else:
            fr_dark_threshold = None

        print(f"\n=== Processing Region: {region} ===")

        # 1. Process all epochs for this region
        for epoch in run_epoch_list:
            print(f"  -- Epoch: {epoch}")

            F = prepare_F(
                epoch=epoch,
                region=region,
                fr_dark_threshold=fr_dark_threshold,
                bin_size=4,
                n_groups=4,
                min_occupancy_s=0.01,
            )

            # Store N for this region (it should be constant across epochs for one sorting)
            # We overwrite it each time, which is fine, or check consistency if desired.
            n_neurons_per_region[region] = F.shape[-1]

            # MEME Estimation
            # res = eigenmoments_and_pr(
            #     F, k_moms=6, remove_mean=True, pairing="random", random_seed=1
            # )

            res = meme_eigenmoments_and_pr_mc(
                F,
                k_moms=6,
                remove_mean=True,
                n_pairings=full_n_pairings,
                n_bin_perms=full_n_bin_perms,
                max_repeat_pairs=None,  # or an int if you want speed
                random_seed=47,
            )
            meme_moments[region][epoch] = res.moments
            meme_pr[region][epoch] = res.pr

            boot = bootstrap_meme_pr(
                epoch=epoch,
                region=region,
                n_bootstraps=n_bootstraps,
                lap_fraction=bootstrap_lap_fraction,
                fr_dark_threshold=fr_dark_threshold,
                bin_size=4,
                n_groups=4,
                min_occupancy_s=0.01,
                k_moms=6,
                remove_mean=True,
                n_pairings=bootstrap_n_pairings,
                n_bin_perms=bootstrap_n_bin_perms,
                max_repeat_pairs=None,
                random_seed=1047,
                ci=bootstrap_ci,
                center="mean",
            )
            meme_boot_pr[region][epoch] = boot.pr_samples
            meme_boot_summary[region][epoch] = boot.summary

            # Naive Estimation
            naive_pca = naive_cov_eigs_and_pr(F, mode="mean_repeat", center=True)
            naive_eigenvalues[region][epoch] = naive_pca.eigenvalues
            naive_pr[region][epoch] = naive_pca.pr

        # 2. Plotting for this region immediately (or you can do it after all regions)
        # Now we retrieve N specifically for this region
        N = n_neurons_per_region[region]

        print(f"  Plotting for {region} (N={N})...")

        # Plot Eigenspectrum (Variance Explained)
        plot_eigenspectrum_comparison(
            region=region,
            all_moments=meme_moments[region],
            naive_eigenvalues=naive_eigenvalues[region],
            n_neurons=N,
            save_path=fig_dir,
            metric="variance_explained",
        )
        plot_eigenspectrum_comparison(
            region=region,
            all_moments=meme_moments[region],
            naive_eigenvalues=naive_eigenvalues[region],
            n_neurons=N,
            save_path=fig_dir,
            metric="eigenvalue",
        )
        plot_broken_power_law_comparison(
            region=region,
            all_moments=meme_moments[region],
            naive_eigenvalues=naive_eigenvalues[region],
            n_neurons=N,
            save_path=fig_dir,
            metric="variance_explained",
            max_rank=1000,
        )
        plot_broken_power_law_comparison(
            region=region,
            all_moments=meme_moments[region],
            naive_eigenvalues=naive_eigenvalues[region],
            n_neurons=N,
            save_path=fig_dir,
            metric="eigenvalue",
            max_rank=1000,
        )
        # Plot Participation Ratio
        plot_pr(
            region=region,
            meme_pr=meme_pr[region],
            naive_pr=naive_pr[region],
            save_path=fig_dir,
        )

        # PR plots using mc_pr distributions
        plot_pr_light_dark_epochwise(
            region=region,
            epoch_to_mc_pr=meme_boot_pr[region],
            epoch_to_condition=epoch_to_condition,
            save_path=fig_dir,
            ci=bootstrap_ci,
            center="mean",
            annotate=True,
            random_seed=0,
            title_suffix=" (bootstrap over laps)",
        )

        plot_pr_light_dark_groupmeans(
            region=region,
            epoch_to_mc_pr=meme_boot_pr[region],
            epoch_to_condition=epoch_to_condition,
            save_path=fig_dir,
            ci=bootstrap_ci,
            center="mean",
            n_draws=5000,
            random_seed=0,
            title_suffix=" (bootstrap over laps)",
        )

    # Save final results
    with open(out_dir / "meme_results.pkl", "wb") as f:
        pickle.dump(
            {
                "pr": meme_pr,
                "moments": meme_moments,
                "bootstrap_pr": meme_boot_pr,
                "bootstrap_summary": meme_boot_summary,
                "settings": {
                    "bootstrap_lap_fraction": bootstrap_lap_fraction,
                    "n_bootstraps": n_bootstraps,
                    "bootstrap_ci": bootstrap_ci,
                    "full_n_pairings": full_n_pairings,
                    "full_n_bin_perms": full_n_bin_perms,
                    "bootstrap_n_pairings": bootstrap_n_pairings,
                    "bootstrap_n_bin_perms": bootstrap_n_bin_perms,
                },
            },
            f,
        )

    with open(out_dir / "naive_results.pkl", "wb") as f:
        pickle.dump({"pca_pr": naive_pr, "eigenvalues": naive_eigenvalues}, f)


if __name__ == "__main__":
    main()
