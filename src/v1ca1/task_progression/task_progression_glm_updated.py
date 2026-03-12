from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from typing import Dict, Hashable, List, Sequence, Tuple, NamedTuple, Any

import numpy as np
import pynapple as nap
import jax.numpy as jnp
from nemos.basis import BSplineEval, RaisedCosineLinearEval, CyclicBSplineEval
from nemos.glm import PopulationGLM
import jax

from dataclasses import dataclass

from scipy.optimize import minimize

import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import scipy
import position_tools as pt
import spikeinterface.full as si
import kyutils
import pandas as pd
import track_linearization as tl
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln


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
    "left_to_center",
    "center_to_right",
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


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")


def _sampling_rate(t_position: np.ndarray) -> float:
    return (len(t_position) - 1) / (t_position[-1] - t_position[0])


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


all_ep = {}
all_ep_first = {}
all_ep_second = {}
trajectory_ep = {}
for epoch in run_epoch_list:
    all_ep[epoch] = nap.IntervalSet(
        start=timestamps_position_dict[epoch][position_offset],
        end=timestamps_position_dict[epoch][-1],
    )

    all_ep_first[epoch] = nap.IntervalSet(
        start=timestamps_position_dict[epoch][position_offset],
        end=timestamps_position_dict[epoch][position_offset]
        + (
            timestamps_position_dict[epoch][-1]
            - timestamps_position_dict[epoch][position_offset]
        )
        / 2,
    )
    all_ep_second[epoch] = nap.IntervalSet(
        start=timestamps_position_dict[epoch][position_offset]
        + (
            timestamps_position_dict[epoch][-1]
            - timestamps_position_dict[epoch][position_offset]
        )
        / 2,
        end=timestamps_position_dict[epoch][-1],
    )
    trajectory_ep[epoch] = {}
    for trajectory_type in trajectory_types:
        trajectory_ep[epoch][trajectory_type] = nap.IntervalSet(
            start=trajectory_times[epoch][trajectory_type][:, 0],
            end=trajectory_times[epoch][trajectory_type][:, -1],
        )

# prepare linearization and movement
dx = 9.5
dy = 9
diagonal_segment_length = np.sqrt(dx**2 + dy**2)

long_segment_length = 81 - 17 - 2
short_segment_length = 13.5

total_length_per_trajectory = (
    long_segment_length * 2 + short_segment_length + 2 * diagonal_segment_length
)
gap = 0

tp_segment_border1 = (
    long_segment_length + diagonal_segment_length / 2
) / total_length_per_trajectory
tp_segment_border2 = (
    long_segment_length
    + diagonal_segment_length
    + short_segment_length
    + diagonal_segment_length / 2
) / total_length_per_trajectory

node_positions_right = np.array(
    [
        (55.5, 81),  # center well
        (55.5, 81 - long_segment_length),
        (55.5 - dx, 81 - long_segment_length - dy),
        (55.5 - dx - short_segment_length, 81 - long_segment_length - dy),
        (55.5 - 2 * dx - short_segment_length, 81 - long_segment_length),
        (55.5 - 2 * dx - short_segment_length, 81),
    ]
)

node_positions_left = np.array(
    [
        (55.5, 81),  # center well
        (55.5, 81 - long_segment_length),
        (55.5 + dx, 81 - long_segment_length - dy),
        (55.5 + dx + short_segment_length, 81 - long_segment_length - dy),
        (55.5 + 2 * dx + short_segment_length, 81 - long_segment_length),
        (55.5 + 2 * dx + short_segment_length, 81),
    ]
)

edges_from_center = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
edges_to_center = np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)])

linear_edge_order_from_center = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
linear_edge_order_to_center = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]

linear_edge_spacing = 0

track_graph_from_center = {}
track_graph_to_center = {}
for trajectory_type in trajectory_types:
    if trajectory_type in ["center_to_right", "right_to_center"]:
        track_graph_from_center[trajectory_type] = tl.make_track_graph(
            node_positions_right, edges_from_center
        )
        track_graph_to_center[trajectory_type] = tl.make_track_graph(
            node_positions_right, edges_to_center
        )
    else:
        track_graph_from_center[trajectory_type] = tl.make_track_graph(
            node_positions_left, edges_from_center
        )
        track_graph_to_center[trajectory_type] = tl.make_track_graph(
            node_positions_left, edges_to_center
        )


speed = {}
movement = {}
for epoch in run_epoch_list:
    sp = pt.get_speed(
        position=position_dict[epoch][position_offset:],
        time=timestamps_position_dict[epoch][position_offset:],
        sampling_frequency=_sampling_rate(
            timestamps_position_dict[epoch][position_offset:]
        ),
        sigma=0.1,
    )
    speed[epoch] = nap.Tsd(t=timestamps_position_dict[epoch][position_offset:], d=sp)
    movement[epoch] = speed[epoch].threshold(speed_threshold, method="above")


total_fr = {}
for region in regions:
    total_fr[region] = {}
    for epoch in run_epoch_list:
        total_fr[region][epoch] = (
            spikes[region].count(ep=all_ep[epoch]).to_numpy()
            / all_ep[epoch].tot_length()
        ).ravel()

fr_during_movement = {}
for region in regions:
    fr_during_movement[region] = {}
    for epoch in run_epoch_list:
        fr_during_movement[region][epoch] = (
            np.sum(
                spikes[region].count(ep=movement[epoch].time_support).to_numpy()
                / movement[epoch].time_support.tot_length(),
                axis=0,
            )
        ).ravel()


def get_task_progression_by_trajectory(task_progression_bin_size):
    # calculate task progression for each trajectory
    task_progression_by_trajectory = {}
    for epoch in run_epoch_list:
        task_progression_by_trajectory[epoch] = {}
        for i, trajectory_type in enumerate(trajectory_types):
            if trajectory_type in ["center_to_left", "center_to_right"]:
                tg = track_graph_from_center[trajectory_type]
                eo = linear_edge_order_from_center
            else:
                tg = track_graph_to_center[trajectory_type]
                eo = linear_edge_order_to_center

            position_df = tl.get_linearized_position(
                position=position_dict[epoch][position_offset:],
                track_graph=tg,
                edge_order=eo,
                edge_spacing=0,
            )
            task_progression_by_trajectory[epoch][trajectory_type] = nap.Tsd(
                t=timestamps_position_dict[epoch][position_offset:],
                d=position_df["linear_position"] / total_length_per_trajectory,
                time_support=trajectory_ep[epoch][trajectory_type],
            )
    # calculate task progression tuning curve and mutual info for each trajectory
    # task_progression_by_trajectory_bins = np.linspace(0, 1, 41)
    task_progression_by_trajectory_bins = np.arange(
        0,
        1 + task_progression_bin_size,
        task_progression_bin_size,
    )
    return task_progression_by_trajectory, task_progression_by_trajectory_bins


def _contiguous_folds(light: np.ndarray, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)

    light = np.asarray(light, dtype=int).reshape(-1)
    n_time = light.size
    all_idx = np.arange(n_time, dtype=np.int64)

    idx_l = np.where(light == 1)[0].astype(np.int64)
    idx_d = np.where(light == 0)[0].astype(np.int64)

    chunks_l = np.array_split(idx_l, n_folds)  # list of int arrays
    chunks_d = np.array_split(idx_d, n_folds)  # list of int arrays

    perm = rng.permutation(n_folds)

    folds = []
    for k in range(n_folds):
        i = int(perm[k])

        test = np.concatenate([chunks_l[i], chunks_d[i]]).astype(np.int64, copy=False)
        test.sort()

        is_test = np.zeros(n_time, dtype=bool)
        is_test[test] = True

        train = all_idx[~is_test]
        folds.append((train, test))

    return folds


def fit_shared_place_light_mod_nemos_per_traj(
    spikes,
    trajectory_ep_by_epoch: Dict[Hashable, Dict[str, any]],
    tp_by_epoch: Dict[Hashable, Dict[str, any]],
    speed_by_epoch: Dict[Hashable, any] | None = None,  # <-- OPTIONAL
    *,
    traj_name: str,
    light_epochs: Sequence[Hashable],
    dark_epochs: Sequence[Hashable],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: Dict[Hashable, any] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Multiplicative model (canonical Poisson exp link) using PopulationGLM.

    Base: [B(p), L, (speed optional)]
    Full: [B(p), L, B(p)*L, (speed optional)]

    Returns a standardized output dict (same keys as other functions).
    """
    all_epochs = list(light_epochs) + list(dark_epochs)
    if len(all_epochs) == 0:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    has_speed = speed_by_epoch is not None

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    unit_ids: np.ndarray | None = None

    for ep in all_epochs:
        ep_traj = trajectory_ep_by_epoch[ep][traj_name]
        if restrict_ep_by_epoch is not None:
            ep_traj = ep_traj.intersect(restrict_ep_by_epoch[ep].time_support)

        sp = spikes if unit_mask is None else spikes[unit_mask]
        counts = sp.count(bin_size_s, ep=ep_traj)

        cols = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = cols
        else:
            if cols.shape != unit_ids.shape or not np.all(cols == unit_ids):
                raise ValueError(
                    "Spike count columns (unit order) differ across epochs."
                )

        y = np.asarray(counts.d, dtype=float)  # (T, U)

        p = tp_by_epoch[ep][traj_name].interpolate(counts).to_numpy().reshape(-1)
        L = (
            np.ones_like(p, dtype=float)
            if ep in light_epochs
            else np.zeros_like(p, dtype=float)
        )

        if has_speed:
            v = speed_by_epoch[ep].interpolate(counts).to_numpy().reshape(-1)
            good = np.isfinite(p) & np.isfinite(v)
            vs.append(v[good])
        else:
            good = np.isfinite(p)

        ys.append(y[good])
        ps.append(p[good])
        ls.append(L[good])

    if unit_ids is None:
        raise ValueError("No data after parsing epochs/trajectory intervals.")

    y_all = np.concatenate(ys, axis=0)  # (T, U)
    p_all = np.concatenate(ps, axis=0).reshape(-1)  # (T,)
    l_all = np.concatenate(ls, axis=0).reshape(-1)  # (T,)

    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need BOTH light and dark bins across the provided epochs.")

    # Optional speed
    if has_speed:
        v_all = np.concatenate(vs, axis=0).reshape(-1)
        speed_mean = float(np.mean(v_all))
        speed_std = float(np.std(v_all) + 1e-12)
        v_all = (v_all - speed_mean) / speed_std
    else:
        v_all = None
        speed_mean = np.nan
        speed_std = np.nan

    # B-spline features
    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    B = np.asarray(basis.compute_features(p_all), dtype=float)  # (T, K)
    K = B.shape[1]
    U = y_all.shape[1]

    # Design matrices
    x_base = np.concatenate([B, l_all[:, None]], axis=1)  # (T, K+1)
    x_full = np.concatenate(
        [B, l_all[:, None], B * l_all[:, None]], axis=1
    )  # (T, 2K+1)
    if has_speed:
        x_base = np.concatenate([x_base, v_all[:, None]], axis=1)  # (T, K+2)
        x_full = np.concatenate([x_full, v_all[:, None]], axis=1)  # (T, 2K+2)

    # CV folds (your helper)
    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    agg_sum_per_neuron = lambda arr: jnp.sum(arr, axis=0)

    ll_base_sum = np.zeros(U, dtype=float)
    ll_full_sum = np.zeros(U, dtype=float)
    spk_sum = np.zeros(U, dtype=float)

    for tr, te in folds:
        m0 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m1 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m0.fit(x_base[tr], y_all[tr])
        m1.fit(x_full[tr], y_all[tr])

        ll0 = np.asarray(
            m0.score(
                x_base[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        ll1 = np.asarray(
            m1.score(
                x_full[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )

        ll_base_sum += ll0
        ll_full_sum += ll1
        spk_sum += np.asarray(y_all[te].sum(axis=0), dtype=float)

    dLL_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spk = np.where(spk_sum > 0, ll_base_sum / spk_sum, np.nan)
        ll_full_per_spk = np.where(spk_sum > 0, ll_full_sum / spk_sum, np.nan)
        dll_per_spk = np.where(spk_sum > 0, dLL_sum / spk_sum, np.nan)

    # Fit base + full on ALL data for coefficients + fields
    m0_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m1_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m0_all.fit(x_base, y_all)
    m1_all.fit(x_full, y_all)

    coef0 = _coef_feat_by_unit(m0_all, n_features=x_base.shape[1])  # (P0,U)
    coef1 = _coef_feat_by_unit(m1_all, n_features=x_full.shape[1])  # (P1,U)

    intercept0 = np.asarray(m0_all.intercept_).reshape(-1)
    intercept1 = np.asarray(m1_all.intercept_).reshape(-1)

    # --- Base coefficients (B, L, [v]) ---
    coef_place_base = coef0[0:K, :]
    coef_light_base = coef0[K, :]  # L always at index K in base
    coef_speed_base = coef0[-1, :] if has_speed else np.full((U,), np.nan)

    # --- Full coefficients (B, L, B*L, [v]) ---
    coef_place_full = coef1[0:K, :]
    coef_light_full = coef1[K, :]
    coef_place_x_light_full = coef1[(K + 1) : (K + 1 + K), :]
    coef_speed_full = coef1[-1, :] if has_speed else np.full((U,), np.nan)

    # Terms not in this model family
    nan_u = np.full((U,), np.nan)
    nan_KU = np.full((K, U), np.nan)

    # Fields (Hz) at mean speed (v_z = 0)
    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    Bgrid = np.asarray(basis.compute_features(grid), dtype=float)  # (G,K)

    dark_lin = intercept1[None, :] + (Bgrid @ coef_place_full)
    light_lin = dark_lin + coef_light_full[None, :] + (Bgrid @ coef_place_x_light_full)

    dark_hz_grid = np.exp(dark_lin) / bin_size_s
    light_hz_grid = np.exp(light_lin) / bin_size_s

    return {
        # --- identity / meta ---
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        # --- CV metrics ---
        "spike_sum_cv": spk_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": dLL_sum,
        "ll_base_per_spike_cv": ll_base_per_spk,
        "ll_full_per_spike_cv": ll_full_per_spk,
        "dll_per_spike_cv": dll_per_spk,
        "ll_base_bits_per_spike_cv": ll_base_per_spk * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spk * inv_log2,
        "dll_bits_per_spike_cv": dll_per_spk * inv_log2,
        # --- coefficients (base fit on all data) ---
        "coef_intercept_base_all": intercept0,
        "coef_place_base_all": coef_place_base,
        "coef_light_base_all": coef_light_base,
        "coef_place_x_light_base_all": nan_KU,  # not in base
        "coef_speed_base_all": coef_speed_base,
        # --- coefficients (full fit on all data) ---
        "coef_intercept_full_all": intercept1,
        "coef_place_full_all": coef_place_full,
        "coef_light_full_all": coef_light_full,
        "coef_place_x_light_full_all": coef_place_x_light_full,
        "coef_speed_full_all": coef_speed_full,
        # --- additive-component coefficients (not in this family) ---
        "coef_add_intercept_full_all": nan_u,
        "coef_add_place_full_all": nan_KU,
        # --- fields (Hz) ---
        "grid_tp": grid,
        "dark_hz_grid": dark_hz_grid,
        "light_hz_grid": light_hz_grid,
    }


# ---------------------------------------------------------------------
# NEW: alternative multiplicative gain models with reduced / decoupled
#      degrees of freedom in the light-modulation term.
# ---------------------------------------------------------------------


def _validate_segment_edges(
    segment_edges: Sequence[float],
    *,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Validate and sanitize segment edges for TP/position.

    Parameters
    ----------
    segment_edges
        Monotonically increasing sequence of edges (len >= 2) within pos_bounds.
        For 3 segments, pass 4 edges: (e0, e1, e2, e3).
    pos_bounds
        Bounds for the position variable.

    Returns
    -------
    edges
        1D float array of edges.
    """
    edges = np.asarray(segment_edges, dtype=float).reshape(-1)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            f"segment_edges must be a 1D sequence with len>=2. Got shape={edges.shape}."
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError("segment_edges must contain only finite values.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            "segment_edges must be strictly increasing (each segment must have positive width)."
        )

    lo, hi = float(pos_bounds[0]), float(pos_bounds[1])
    # Allow tiny numerical tolerance, but keep edges within bounds
    tol = 1e-9
    if edges[0] < lo - tol or edges[-1] > hi + tol:
        raise ValueError(
            f"segment_edges must lie within pos_bounds={pos_bounds}. "
            f"Got [{edges[0]}, {edges[-1]}]."
        )
    edges[0] = max(edges[0], lo)
    edges[-1] = min(edges[-1], hi)
    return edges


def _segment_center_raised_cosine_basis(
    x: np.ndarray,
    segment_edges: Sequence[float],
    *,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Segment-wise raised-cosine 'bump' basis.

    For each segment [a, b], we create one smooth bump function that is:
      - 1 at the segment center
      - 0 at the segment boundaries
      - 0 outside the segment

    This gives a *low-DOF*, smooth approximation to a location-specific modulation
    without letting the model put an arbitrary knob at every location.

    Returns
    -------
    B_gain : array, shape (T, n_segments)
    """
    edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    x = np.asarray(x, dtype=float).reshape(-1)
    n_seg = int(edges.size - 1)
    B = np.zeros((x.size, n_seg), dtype=float)

    for k in range(n_seg):
        a = float(edges[k])
        b = float(edges[k + 1])
        c = 0.5 * (a + b)
        half = 0.5 * (b - a)
        if half <= 0:
            continue

        z = (x - c) / half  # in [-1, 1] inside the segment
        m = np.abs(z) <= 1.0
        # raised cosine: 1 at z=0, 0 at z=±1, smooth at boundaries
        B[m, k] = 0.5 * (1.0 + np.cos(np.pi * z[m]))

    return B


def fit_shared_place_light_mod_nemos_per_traj_segment_gain(
    spikes,
    trajectory_ep_by_epoch: Dict[Hashable, Dict[str, any]],
    tp_by_epoch: Dict[Hashable, Dict[str, any]],
    speed_by_epoch: Dict[Hashable, any] = None,
    *,
    traj_name: str,
    light_epochs: Sequence[Hashable],
    dark_epochs: Sequence[Hashable],
    # --- NEW: gain basis control ---
    segment_edges: Sequence[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    # --- dark-field basis (kept flexible) ---
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray = None,
    restrict_ep_by_epoch: Dict[Hashable, any] = None,
) -> Dict[str, np.ndarray]:
    """Multiplicative model with a *segment-anchored* low-DOF gain term.

    This is a drop-in alternative to `fit_shared_place_light_mod_nemos_per_traj`
    meant to address the concern that the location-specific gain term is too permissive.

    Parameterization (Poisson exp link):

        dark:  log λ = b0 + B_dark(p)·β
        light: log λ = b0 + B_dark(p)·β  +  bL  +  B_gain(p)·γ

    where:
      - B_dark is a cyclic B-spline basis with `n_splines` (same as your current model)
      - B_gain is a *low-DOF* basis with 1 raised-cosine bump per segment, defined by `segment_edges`

    Design matrices:
        Base: [B_dark(p), L, (speed optional)]
        Full: [B_dark(p), L, B_gain(p)*L, (speed optional)]

    Notes
    -----
    - Keeping B_dark flexible avoids underfitting the baseline (dark) field, while making the
      *difference* between light and dark much more constrained.
    - `segment_edges` should be specified in the same coordinate system as tp (typically [0,1]).
    """
    all_epochs = list(light_epochs) + list(dark_epochs)
    if len(all_epochs) == 0:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    has_speed = speed_by_epoch is not None

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    unit_ids = None

    for ep in all_epochs:
        ep_traj = trajectory_ep_by_epoch[ep][traj_name]
        if restrict_ep_by_epoch is not None:
            ep_traj = ep_traj.intersect(restrict_ep_by_epoch[ep].time_support)

        sp = spikes if unit_mask is None else spikes[unit_mask]
        counts = sp.count(bin_size_s, ep=ep_traj)

        cols = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = cols
        else:
            if cols.shape != unit_ids.shape or not np.all(cols == unit_ids):
                raise ValueError(
                    "Spike count columns (unit order) differ across epochs."
                )

        y = np.asarray(counts.d, dtype=float)  # (T, U)

        p = tp_by_epoch[ep][traj_name].interpolate(counts).to_numpy().reshape(-1)
        L = (
            np.ones_like(p, dtype=float)
            if ep in light_epochs
            else np.zeros_like(p, dtype=float)
        )

        if has_speed:
            v = speed_by_epoch[ep].interpolate(counts).to_numpy().reshape(-1)
            good = np.isfinite(p) & np.isfinite(v)
            vs.append(v[good])
        else:
            good = np.isfinite(p)

        ys.append(y[good])
        ps.append(p[good])
        ls.append(L[good])

    if unit_ids is None:
        raise ValueError("No data after parsing epochs/trajectory intervals.")

    y_all = np.concatenate(ys, axis=0)  # (T, U)
    p_all = np.concatenate(ps, axis=0).reshape(-1)  # (T,)
    l_all = np.concatenate(ls, axis=0).reshape(-1)  # (T,)

    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need BOTH light and dark bins across the provided epochs.")

    # Optional speed
    if has_speed:
        v_all = np.concatenate(vs, axis=0).reshape(-1)
        speed_mean = float(np.mean(v_all))
        speed_std = float(np.std(v_all) + 1e-12)
        v_all = (v_all - speed_mean) / speed_std
    else:
        v_all = None
        speed_mean = np.nan
        speed_std = np.nan

    # ---- Dark-field basis (as before) ----
    basis_dark = BSplineEval(
        n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds
    )
    B_dark = np.asarray(basis_dark.compute_features(p_all), dtype=float)  # (T, Kd)
    Kd = B_dark.shape[1]
    U = y_all.shape[1]

    # ---- Gain basis (segment bumps) ----
    edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    B_gain = _segment_center_raised_cosine_basis(
        p_all, edges, pos_bounds=pos_bounds
    )  # (T, Kg)
    Kg = B_gain.shape[1]

    # Design matrices
    x_base = np.concatenate([B_dark, l_all[:, None]], axis=1)  # (T, Kd+1)
    x_full = np.concatenate(
        [B_dark, l_all[:, None], B_gain * l_all[:, None]], axis=1
    )  # (T, Kd+1+Kg)
    if has_speed:
        x_base = np.concatenate([x_base, v_all[:, None]], axis=1)  # (T, Kd+2)
        x_full = np.concatenate([x_full, v_all[:, None]], axis=1)  # (T, Kd+2+Kg)

    # CV folds
    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    agg_sum_per_neuron = lambda arr: jnp.sum(arr, axis=0)

    ll_base_sum = np.zeros(U, dtype=float)
    ll_full_sum = np.zeros(U, dtype=float)
    spk_sum = np.zeros(U, dtype=float)

    for tr, te in folds:
        m0 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m1 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m0.fit(x_base[tr], y_all[tr])
        m1.fit(x_full[tr], y_all[tr])

        ll0 = np.asarray(
            m0.score(
                x_base[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        ll1 = np.asarray(
            m1.score(
                x_full[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )

        ll_base_sum += ll0
        ll_full_sum += ll1
        spk_sum += np.asarray(y_all[te].sum(axis=0), dtype=float)

    dLL_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spk = np.where(spk_sum > 0, ll_base_sum / spk_sum, np.nan)
        ll_full_per_spk = np.where(spk_sum > 0, ll_full_sum / spk_sum, np.nan)
        dll_per_spk = np.where(spk_sum > 0, dLL_sum / spk_sum, np.nan)

    # Fit base + full on ALL data for coefficients + fields
    m0_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m1_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m0_all.fit(x_base, y_all)
    m1_all.fit(x_full, y_all)

    coef0 = _coef_feat_by_unit(m0_all, n_features=x_base.shape[1])
    coef1 = _coef_feat_by_unit(m1_all, n_features=x_full.shape[1])

    intercept0 = np.asarray(m0_all.intercept_).reshape(-1)
    intercept1 = np.asarray(m1_all.intercept_).reshape(-1)

    # ---- Base coefficients: [B_dark, L, (v)] ----
    coef_place_base = coef0[0:Kd, :]
    coef_light_base = coef0[Kd, :]
    coef_speed_base = coef0[-1, :] if has_speed else np.full((U,), np.nan)

    # ---- Full coefficients: [B_dark, L, B_gain*L, (v)] ----
    coef_place_full = coef1[0:Kd, :]
    coef_light_full = coef1[Kd, :]
    coef_gain_full = coef1[(Kd + 1) : (Kd + 1 + Kg), :]
    coef_speed_full = coef1[-1, :] if has_speed else np.full((U,), np.nan)

    # Terms not in this model family
    nan_u = np.full((U,), np.nan)
    nan_KU_dark = np.full((Kd, U), np.nan)
    nan_KU_gain = np.full((Kg, U), np.nan)

    # Fields (Hz) at mean speed (v_z = 0)
    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    Bgrid_dark = np.asarray(basis_dark.compute_features(grid), dtype=float)  # (G,Kd)
    Bgrid_gain = _segment_center_raised_cosine_basis(
        grid, edges, pos_bounds=pos_bounds
    )  # (G,Kg)

    dark_lin = intercept1[None, :] + (Bgrid_dark @ coef_place_full)
    light_lin = dark_lin + coef_light_full[None, :] + (Bgrid_gain @ coef_gain_full)

    dark_hz_grid = np.exp(dark_lin) / bin_size_s
    light_hz_grid = np.exp(light_lin) / bin_size_s

    return {
        # identity / meta
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),  # dark basis count
        "spline_order": int(spline_order),
        "n_splines_gain": int(Kg),  # gain basis count
        "gain_basis": "segment_raised_cosine",
        "segment_edges": np.asarray(edges, dtype=float),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        # CV metrics
        "spike_sum_cv": spk_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": dLL_sum,
        "ll_base_per_spike_cv": ll_base_per_spk,
        "ll_full_per_spike_cv": ll_full_per_spk,
        "dll_per_spike_cv": dll_per_spk,
        "ll_base_bits_per_spike_cv": ll_base_per_spk * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spk * inv_log2,
        "dll_bits_per_spike_cv": dll_per_spk * inv_log2,
        # coefficients (base fit on all data)
        "coef_intercept_base_all": intercept0,
        "coef_place_base_all": coef_place_base,
        "coef_light_base_all": coef_light_base,
        "coef_place_x_light_base_all": nan_KU_gain,  # base has no gain block
        "coef_speed_base_all": coef_speed_base,
        # coefficients (full fit on all data)
        "coef_intercept_full_all": intercept1,
        "coef_place_full_all": coef_place_full,  # dark field coefficients (Kd,U)
        "coef_light_full_all": coef_light_full,  # global light offset (U,)
        "coef_place_x_light_full_all": coef_gain_full,  # gain coefficients (Kg,U)
        "coef_speed_full_all": coef_speed_full,
        # additive-component coefficients (not in this family)
        "coef_add_intercept_full_all": nan_u,
        "coef_add_place_full_all": nan_KU_dark,
        # fields (Hz)
        "grid_tp": grid,
        "dark_hz_grid": dark_hz_grid,
        "light_hz_grid": light_hz_grid,
    }


def fit_shared_place_light_mod_nemos_per_traj_gain_splines(
    spikes,
    trajectory_ep_by_epoch: Dict[Hashable, Dict[str, any]],
    tp_by_epoch: Dict[Hashable, Dict[str, any]],
    speed_by_epoch: Dict[Hashable, any] = None,
    *,
    traj_name: str,
    light_epochs: Sequence[Hashable],
    dark_epochs: Sequence[Hashable],
    bin_size_s: float = 0.02,
    # --- dark-field basis ---
    n_splines: int = 25,
    spline_order: int = 4,
    # --- NEW: gain basis can differ from dark basis ---
    n_splines_gain: int = None,
    spline_order_gain: int = None,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray = None,
    restrict_ep_by_epoch: Dict[Hashable, any] = None,
) -> Dict[str, np.ndarray]:
    """Multiplicative model where the gain basis size can differ from the dark basis.

    This is identical to `fit_shared_place_light_mod_nemos_per_traj` except that the
    light-specific term uses its own spline basis B_gain(p) with `n_splines_gain`
    basis functions (default: same as dark).

        dark:  log λ = b0 + B_dark(p)·β
        light: log λ = b0 + B_dark(p)·β + bL + B_gain(p)·γ

    Design matrices:
        Base: [B_dark(p), L, (speed optional)]
        Full: [B_dark(p), L, B_gain(p)*L, (speed optional)]
    """
    all_epochs = list(light_epochs) + list(dark_epochs)
    if len(all_epochs) == 0:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    if n_splines_gain is None:
        n_splines_gain = int(n_splines)
    if spline_order_gain is None:
        spline_order_gain = int(spline_order)

    has_speed = speed_by_epoch is not None

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    unit_ids = None

    for ep in all_epochs:
        ep_traj = trajectory_ep_by_epoch[ep][traj_name]
        if restrict_ep_by_epoch is not None:
            ep_traj = ep_traj.intersect(restrict_ep_by_epoch[ep].time_support)

        sp = spikes if unit_mask is None else spikes[unit_mask]
        counts = sp.count(bin_size_s, ep=ep_traj)

        cols = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = cols
        else:
            if cols.shape != unit_ids.shape or not np.all(cols == unit_ids):
                raise ValueError(
                    "Spike count columns (unit order) differ across epochs."
                )

        y = np.asarray(counts.d, dtype=float)  # (T, U)

        p = tp_by_epoch[ep][traj_name].interpolate(counts).to_numpy().reshape(-1)
        L = (
            np.ones_like(p, dtype=float)
            if ep in light_epochs
            else np.zeros_like(p, dtype=float)
        )

        if has_speed:
            v = speed_by_epoch[ep].interpolate(counts).to_numpy().reshape(-1)
            good = np.isfinite(p) & np.isfinite(v)
            vs.append(v[good])
        else:
            good = np.isfinite(p)

        ys.append(y[good])
        ps.append(p[good])
        ls.append(L[good])

    if unit_ids is None:
        raise ValueError("No data after parsing epochs/trajectory intervals.")

    y_all = np.concatenate(ys, axis=0)  # (T, U)
    p_all = np.concatenate(ps, axis=0).reshape(-1)  # (T,)
    l_all = np.concatenate(ls, axis=0).reshape(-1)  # (T,)

    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need BOTH light and dark bins across the provided epochs.")

    # Optional speed
    if has_speed:
        v_all = np.concatenate(vs, axis=0).reshape(-1)
        speed_mean = float(np.mean(v_all))
        speed_std = float(np.std(v_all) + 1e-12)
        v_all = (v_all - speed_mean) / speed_std
    else:
        v_all = None
        speed_mean = np.nan
        speed_std = np.nan

    # ---- bases ----
    basis_dark = BSplineEval(
        n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds
    )
    B_dark = np.asarray(basis_dark.compute_features(p_all), dtype=float)  # (T, Kd)
    Kd = B_dark.shape[1]

    basis_gain = BSplineEval(
        n_basis_funcs=int(n_splines_gain),
        order=int(spline_order_gain),
        bounds=pos_bounds,
    )
    B_gain = np.asarray(basis_gain.compute_features(p_all), dtype=float)  # (T, Kg)
    Kg = B_gain.shape[1]

    U = y_all.shape[1]

    # Design matrices
    x_base = np.concatenate([B_dark, l_all[:, None]], axis=1)  # (T, Kd+1)
    x_full = np.concatenate(
        [B_dark, l_all[:, None], B_gain * l_all[:, None]], axis=1
    )  # (T, Kd+1+Kg)
    if has_speed:
        x_base = np.concatenate([x_base, v_all[:, None]], axis=1)
        x_full = np.concatenate([x_full, v_all[:, None]], axis=1)

    # CV folds
    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    agg_sum_per_neuron = lambda arr: jnp.sum(arr, axis=0)

    ll_base_sum = np.zeros(U, dtype=float)
    ll_full_sum = np.zeros(U, dtype=float)
    spk_sum = np.zeros(U, dtype=float)

    for tr, te in folds:
        m0 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m1 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m0.fit(x_base[tr], y_all[tr])
        m1.fit(x_full[tr], y_all[tr])

        ll0 = np.asarray(
            m0.score(
                x_base[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        ll1 = np.asarray(
            m1.score(
                x_full[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )

        ll_base_sum += ll0
        ll_full_sum += ll1
        spk_sum += np.asarray(y_all[te].sum(axis=0), dtype=float)

    dLL_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spk = np.where(spk_sum > 0, ll_base_sum / spk_sum, np.nan)
        ll_full_per_spk = np.where(spk_sum > 0, ll_full_sum / spk_sum, np.nan)
        dll_per_spk = np.where(spk_sum > 0, dLL_sum / spk_sum, np.nan)

    # Fit on ALL data for coefficients + fields
    m0_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m1_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m0_all.fit(x_base, y_all)
    m1_all.fit(x_full, y_all)

    coef0 = _coef_feat_by_unit(m0_all, n_features=x_base.shape[1])
    coef1 = _coef_feat_by_unit(m1_all, n_features=x_full.shape[1])

    intercept0 = np.asarray(m0_all.intercept_).reshape(-1)
    intercept1 = np.asarray(m1_all.intercept_).reshape(-1)

    # Base coefficients
    coef_place_base = coef0[0:Kd, :]
    coef_light_base = coef0[Kd, :]
    coef_speed_base = coef0[-1, :] if has_speed else np.full((U,), np.nan)

    # Full coefficients
    coef_place_full = coef1[0:Kd, :]
    coef_light_full = coef1[Kd, :]
    coef_gain_full = coef1[(Kd + 1) : (Kd + 1 + Kg), :]
    coef_speed_full = coef1[-1, :] if has_speed else np.full((U,), np.nan)

    # Terms not in this model family
    nan_u = np.full((U,), np.nan)
    nan_KU_dark = np.full((Kd, U), np.nan)
    nan_KU_gain = np.full((Kg, U), np.nan)

    # Fields (Hz) at mean speed (v_z = 0)
    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    Bgrid_dark = np.asarray(basis_dark.compute_features(grid), dtype=float)
    Bgrid_gain = np.asarray(basis_gain.compute_features(grid), dtype=float)

    dark_lin = intercept1[None, :] + (Bgrid_dark @ coef_place_full)
    light_lin = dark_lin + coef_light_full[None, :] + (Bgrid_gain @ coef_gain_full)

    dark_hz_grid = np.exp(dark_lin) / bin_size_s
    light_hz_grid = np.exp(light_lin) / bin_size_s

    return {
        # identity / meta
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),  # dark basis count
        "spline_order": int(spline_order),
        "n_splines_gain": int(n_splines_gain),
        "spline_order_gain": int(spline_order_gain),
        "gain_basis": "bspline",
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        # CV metrics
        "spike_sum_cv": spk_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": dLL_sum,
        "ll_base_per_spike_cv": ll_base_per_spk,
        "ll_full_per_spike_cv": ll_full_per_spk,
        "dll_per_spike_cv": dll_per_spk,
        "ll_base_bits_per_spike_cv": ll_base_per_spk * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spk * inv_log2,
        "dll_bits_per_spike_cv": dll_per_spk * inv_log2,
        # coefficients (base fit on all data)
        "coef_intercept_base_all": intercept0,
        "coef_place_base_all": coef_place_base,
        "coef_light_base_all": coef_light_base,
        "coef_place_x_light_base_all": nan_KU_gain,
        "coef_speed_base_all": coef_speed_base,
        # coefficients (full fit on all data)
        "coef_intercept_full_all": intercept1,
        "coef_place_full_all": coef_place_full,
        "coef_light_full_all": coef_light_full,
        "coef_place_x_light_full_all": coef_gain_full,
        "coef_speed_full_all": coef_speed_full,
        # additive-component coefficients (not in this family)
        "coef_add_intercept_full_all": nan_u,
        "coef_add_place_full_all": nan_KU_dark,
        # fields (Hz)
        "grid_tp": grid,
        "dark_hz_grid": dark_hz_grid,
        "light_hz_grid": light_hz_grid,
    }


def _coef_feat_by_unit(model, n_features: int) -> np.ndarray:
    coef = np.asarray(model.coef_)
    n_units = np.asarray(model.intercept_).size
    if coef.shape == (n_units, n_features):
        return coef.T
    if coef.shape == (n_features, n_units):
        return coef
    raise ValueError(f"Unexpected coef_ shape {coef.shape}, expected (U,F) or (F,U)")


def fit_shared_place_light_add_nemos_per_traj(
    spikes,
    trajectory_ep_by_epoch: Dict[Hashable, Dict[str, any]],
    tp_by_epoch: Dict[Hashable, Dict[str, any]],
    speed_by_epoch: Dict[Hashable, any] | None = None,  # <-- OPTIONAL
    *,
    traj_name: str,
    light_epochs: Sequence[Hashable],
    dark_epochs: Sequence[Hashable],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: Dict[Hashable, any] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Softplus-link variant.

    Base: [B(p), L, (speed optional)]
    Full: [B(p), L, B(p)*L, (speed optional)]

    Returns the same standardized dict keys as the exp-link function.
    """
    all_epochs = list(light_epochs) + list(dark_epochs)
    if len(all_epochs) == 0:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    has_speed = speed_by_epoch is not None

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    unit_ids: np.ndarray | None = None

    for ep in all_epochs:
        ep_traj = trajectory_ep_by_epoch[ep][traj_name]
        if restrict_ep_by_epoch is not None:
            ep_traj = ep_traj.intersect(restrict_ep_by_epoch[ep].time_support)

        sp = spikes if unit_mask is None else spikes[unit_mask]
        counts = sp.count(bin_size_s, ep=ep_traj)

        cols = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = cols
        else:
            if cols.shape != unit_ids.shape or not np.all(cols == unit_ids):
                raise ValueError(
                    "Spike count columns (unit order) differ across epochs."
                )

        y = np.asarray(counts.d, dtype=float)  # (T, U)

        p = tp_by_epoch[ep][traj_name].interpolate(counts).to_numpy().reshape(-1)
        L = (
            np.ones_like(p, dtype=float)
            if ep in light_epochs
            else np.zeros_like(p, dtype=float)
        )

        if has_speed:
            v = speed_by_epoch[ep].interpolate(counts).to_numpy().reshape(-1)
            good = np.isfinite(p) & np.isfinite(v)
            vs.append(v[good])
        else:
            good = np.isfinite(p)

        ys.append(y[good])
        ps.append(p[good])
        ls.append(L[good])

    if unit_ids is None:
        raise ValueError("No data after parsing epochs/trajectory intervals.")

    y_all = np.concatenate(ys, axis=0)  # (T, U)
    p_all = np.concatenate(ps, axis=0).reshape(-1)  # (T,)
    l_all = np.concatenate(ls, axis=0).reshape(-1)  # (T,)

    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need BOTH light and dark bins across the provided epochs.")

    # Optional speed
    if has_speed:
        v_all = np.concatenate(vs, axis=0).reshape(-1)
        speed_mean = float(np.mean(v_all))
        speed_std = float(np.std(v_all) + 1e-12)
        v_all = (v_all - speed_mean) / speed_std
    else:
        v_all = None
        speed_mean = np.nan
        speed_std = np.nan

    # B-spline features
    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    B = np.asarray(basis.compute_features(p_all), dtype=float)  # (T, K)
    K = B.shape[1]
    U = y_all.shape[1]

    # Design matrices
    x_base = np.concatenate([B, l_all[:, None]], axis=1)
    x_full = np.concatenate([B, l_all[:, None], B * l_all[:, None]], axis=1)
    if has_speed:
        x_base = np.concatenate([x_base, v_all[:, None]], axis=1)
        x_full = np.concatenate([x_full, v_all[:, None]], axis=1)

    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    agg_sum_per_neuron = lambda arr: jnp.sum(arr, axis=0)

    ll_base_sum = np.zeros(U, dtype=float)
    ll_full_sum = np.zeros(U, dtype=float)
    spk_sum = np.zeros(U, dtype=float)

    for tr, te in folds:
        m0 = PopulationGLM(
            "Poisson",
            inverse_link_function=jax.nn.softplus,
            regularizer="Ridge",
            regularizer_strength=ridge,
        )
        m1 = PopulationGLM(
            "Poisson",
            inverse_link_function=jax.nn.softplus,
            regularizer="Ridge",
            regularizer_strength=ridge,
        )
        m0.fit(x_base[tr], y_all[tr])
        m1.fit(x_full[tr], y_all[tr])

        ll0 = np.asarray(
            m0.score(
                x_base[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        ll1 = np.asarray(
            m1.score(
                x_full[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )

        ll_base_sum += ll0
        ll_full_sum += ll1
        spk_sum += np.asarray(y_all[te].sum(axis=0), dtype=float)

    dLL_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spk = np.where(spk_sum > 0, ll_base_sum / spk_sum, np.nan)
        ll_full_per_spk = np.where(spk_sum > 0, ll_full_sum / spk_sum, np.nan)
        dll_per_spk = np.where(spk_sum > 0, dLL_sum / spk_sum, np.nan)

    # Fit base + full on ALL data
    m0_all = PopulationGLM(
        "Poisson",
        inverse_link_function=jax.nn.softplus,
        regularizer="Ridge",
        regularizer_strength=ridge,
    )
    m1_all = PopulationGLM(
        "Poisson",
        inverse_link_function=jax.nn.softplus,
        regularizer="Ridge",
        regularizer_strength=ridge,
    )
    m0_all.fit(x_base, y_all)
    m1_all.fit(x_full, y_all)

    coef0 = _coef_feat_by_unit(m0_all, n_features=x_base.shape[1])
    coef1 = _coef_feat_by_unit(m1_all, n_features=x_full.shape[1])

    intercept0 = np.asarray(m0_all.intercept_).reshape(-1)
    intercept1 = np.asarray(m1_all.intercept_).reshape(-1)

    # Base coefficients
    coef_place_base = coef0[0:K, :]
    coef_light_base = coef0[K, :]
    coef_speed_base = coef0[-1, :] if has_speed else np.full((U,), np.nan)

    # Full coefficients
    coef_place_full = coef1[0:K, :]
    coef_light_full = coef1[K, :]
    coef_place_x_light_full = coef1[(K + 1) : (K + 1 + K), :]
    coef_speed_full = coef1[-1, :] if has_speed else np.full((U,), np.nan)

    nan_u = np.full((U,), np.nan)
    nan_KU = np.full((K, U), np.nan)

    # Fields (Hz) at mean speed (v_z = 0): expected spikes/bin = softplus(eta)
    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    Bgrid = np.asarray(basis.compute_features(grid), dtype=float)

    dark_lin = intercept1[None, :] + (Bgrid @ coef_place_full)
    light_lin = dark_lin + coef_light_full[None, :] + (Bgrid @ coef_place_x_light_full)

    dark_hz_grid = np.asarray(jax.nn.softplus(dark_lin)) / bin_size_s
    light_hz_grid = np.asarray(jax.nn.softplus(light_lin)) / bin_size_s

    return {
        # identity / meta
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        # CV metrics
        "spike_sum_cv": spk_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": dLL_sum,
        "ll_base_per_spike_cv": ll_base_per_spk,
        "ll_full_per_spike_cv": ll_full_per_spk,
        "dll_per_spike_cv": dll_per_spk,
        "ll_base_bits_per_spike_cv": ll_base_per_spk * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spk * inv_log2,
        "dll_bits_per_spike_cv": dll_per_spk * inv_log2,
        # coefficients (base fit on all data)
        "coef_intercept_base_all": intercept0,
        "coef_place_base_all": coef_place_base,
        "coef_light_base_all": coef_light_base,
        "coef_place_x_light_base_all": nan_KU,  # not in base
        "coef_speed_base_all": coef_speed_base,
        # coefficients (full fit on all data)
        "coef_intercept_full_all": intercept1,
        "coef_place_full_all": coef_place_full,
        "coef_light_full_all": coef_light_full,
        "coef_place_x_light_full_all": coef_place_x_light_full,
        "coef_speed_full_all": coef_speed_full,
        # additive-component coefficients (not in this family)
        "coef_add_intercept_full_all": nan_u,
        "coef_add_place_full_all": nan_KU,
        # fields (Hz)
        "grid_tp": grid,
        "dark_hz_grid": dark_hz_grid,
        "light_hz_grid": light_hz_grid,
    }


@jax.tree_util.register_pytree_node_class
@dataclass
class _AddRateParamsBatched:
    theta0: jnp.ndarray  # (U,)
    beta: jnp.ndarray  # (K,U)
    beta_speed: jnp.ndarray  # (U,)
    phi0: jnp.ndarray  # (U,)
    delta: jnp.ndarray  # (K,U)

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, ...], Any]:
        return (self.theta0, self.beta, self.beta_speed, self.phi0, self.delta), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Tuple[jnp.ndarray, ...]):
        theta0, beta, beta_speed, phi0, delta = children
        return cls(
            theta0=theta0, beta=beta, beta_speed=beta_speed, phi0=phi0, delta=delta
        )


@jax.tree_util.register_pytree_node_class
@dataclass
class _AddRateParamsBatchedNoSpeed:
    theta0: jnp.ndarray  # (U,)
    beta: jnp.ndarray  # (K,U)
    phi0: jnp.ndarray  # (U,)
    delta: jnp.ndarray  # (K,U)

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, ...], Any]:
        return (self.theta0, self.beta, self.phi0, self.delta), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Tuple[jnp.ndarray, ...]):
        theta0, beta, phi0, delta = children
        return cls(theta0=theta0, beta=beta, phi0=phi0, delta=delta)


def _init_additive_from_base(
    *,
    y: np.ndarray,  # (T,U)
    B: np.ndarray,  # (T,K)
    v: np.ndarray | None,  # (T,) or None
    L: np.ndarray,  # (T,) 0/1
    ridge: float,
):
    """
    Warm-start:
      - fit base exp Poisson with nemos:
          if v is not None: exp(theta0 + B@beta + beta_speed*v)
          else:             exp(theta0 + B@beta)
      - init additive component to explain mean(light)-mean(dark)
    """
    T, K = B.shape
    U = y.shape[1]

    if v is not None:
        x_base = np.concatenate([B, v[:, None]], axis=1)  # (T, K+1)
        m = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m.fit(x_base, y)
        coef = _coef_feat_by_unit(m, n_features=K + 1)  # (K+1,U)
        theta0 = np.asarray(m.intercept_).reshape(-1).astype(np.float32)
        beta = coef[:K].astype(np.float32)
        beta_speed = coef[K].astype(np.float32)
    else:
        x_base = B  # (T,K)
        m = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m.fit(x_base, y)
        coef = _coef_feat_by_unit(m, n_features=K)  # (K,U)
        theta0 = np.asarray(m.intercept_).reshape(-1).astype(np.float32)
        beta = coef.astype(np.float32)

    # crude init for additive term
    eps = 1e-6
    Lb = L.astype(bool)
    mu_light = y[Lb].mean(axis=0) if np.any(Lb) else y.mean(axis=0)
    mu_dark = y[~Lb].mean(axis=0) if np.any(~Lb) else y.mean(axis=0)
    add_mu = np.maximum(mu_light - mu_dark, eps)

    phi0 = np.log(add_mu + eps).astype(np.float32)
    delta = np.zeros((K, U), dtype=np.float32)

    if v is not None:
        return _AddRateParamsBatched(
            theta0=jnp.asarray(theta0),
            beta=jnp.asarray(beta),
            beta_speed=jnp.asarray(beta_speed),
            phi0=jnp.asarray(phi0),
            delta=jnp.asarray(delta),
        )
    else:
        return _AddRateParamsBatchedNoSpeed(
            theta0=jnp.asarray(theta0),
            beta=jnp.asarray(beta),
            phi0=jnp.asarray(phi0),
            delta=jnp.asarray(delta),
        )


def _try_import_jaxopt():
    try:
        import jaxopt  # type: ignore

        return jaxopt
    except Exception:
        return None


def _fit_additive_rate_batched_jax(
    *,
    y: np.ndarray,  # (T,U)
    B: np.ndarray,  # (T,K)
    v: np.ndarray | None,  # (T,) or None
    L: np.ndarray,  # (T,)
    ridge: float,
    maxiter: int = 200,
    lr: float = 5e-2,
) -> _AddRateParamsBatched | _AddRateParamsBatchedNoSpeed:
    """
    True additive-in-rate Poisson:
      mu = exp(theta0 + B@beta [+ beta_speed*v]) + L*exp(phi0 + B@delta)

    Penalize beta, delta, and (if present) beta_speed. No penalty on theta0/phi0.
    """
    y_j = jnp.asarray(y, dtype=jnp.float32)
    B_j = jnp.asarray(B, dtype=jnp.float32)
    L_j = jnp.asarray(L, dtype=jnp.float32).reshape(-1)

    idx_l = jnp.where(L_j > 0.5)[0]
    idx_d = jnp.where(L_j <= 0.5)[0]

    if v is not None:
        v_j = jnp.asarray(v, dtype=jnp.float32).reshape(-1)

        def nll(params: _AddRateParamsBatched) -> jnp.ndarray:
            eta = (
                params.theta0[None, :]
                + (B_j @ params.beta)
                + v_j[:, None] * params.beta_speed[None, :]
            )
            z = params.phi0[None, :] + (B_j @ params.delta)

            # dark bins: y*eta - exp(eta)
            eta_d = eta[idx_d, :]
            y_d = y_j[idx_d, :]
            ll_dark = jnp.sum(y_d * eta_d - jnp.exp(eta_d))

            # light bins: y*log(exp(eta)+exp(z)) - (exp(eta)+exp(z))
            eta_l = eta[idx_l, :]
            z_l = z[idx_l, :]
            y_l = y_j[idx_l, :]
            log_rate_l = jnp.logaddexp(eta_l, z_l)
            rate_l = jnp.exp(eta_l) + jnp.exp(z_l)
            ll_light = jnp.sum(y_l * log_rate_l - rate_l)

            pen = (
                0.5
                * ridge
                * (
                    jnp.sum(params.beta**2)
                    + jnp.sum(params.delta**2)
                    + jnp.sum(params.beta_speed**2)
                )
            )
            T = y_j.shape[0]
            ll = ll_dark + ll_light
            data_nll = -(ll / T)  # mean over time (summed over units)
            return data_nll + pen

        init = _init_additive_from_base(y=y, B=B, v=v, L=L, ridge=ridge)
        nll_jit = jax.jit(nll)

        jaxopt = _try_import_jaxopt()
        if jaxopt is not None:
            solver = jaxopt.LBFGS(fun=nll_jit, maxiter=int(maxiter))
            params_hat, _ = solver.run(init)
            return params_hat

        # Adam fallback
        grad_jit = jax.jit(jax.grad(nll))
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def zeros_like(p):
            return _AddRateParamsBatched(
                theta0=jnp.zeros_like(p.theta0),
                beta=jnp.zeros_like(p.beta),
                beta_speed=jnp.zeros_like(p.beta_speed),
                phi0=jnp.zeros_like(p.phi0),
                delta=jnp.zeros_like(p.delta),
            )

        params = init
        m = zeros_like(init)
        vv = zeros_like(init)

        for t in range(int(maxiter)):
            g = grad_jit(params)
            m = _AddRateParamsBatched(
                theta0=beta1 * m.theta0 + (1 - beta1) * g.theta0,
                beta=beta1 * m.beta + (1 - beta1) * g.beta,
                beta_speed=beta1 * m.beta_speed + (1 - beta1) * g.beta_speed,
                phi0=beta1 * m.phi0 + (1 - beta1) * g.phi0,
                delta=beta1 * m.delta + (1 - beta1) * g.delta,
            )
            vv = _AddRateParamsBatched(
                theta0=beta2 * vv.theta0 + (1 - beta2) * (g.theta0**2),
                beta=beta2 * vv.beta + (1 - beta2) * (g.beta**2),
                beta_speed=beta2 * vv.beta_speed + (1 - beta2) * (g.beta_speed**2),
                phi0=beta2 * vv.phi0 + (1 - beta2) * (g.phi0**2),
                delta=beta2 * vv.delta + (1 - beta2) * (g.delta**2),
            )

            t1 = t + 1
            mhat = _AddRateParamsBatched(
                theta0=m.theta0 / (1 - beta1**t1),
                beta=m.beta / (1 - beta1**t1),
                beta_speed=m.beta_speed / (1 - beta1**t1),
                phi0=m.phi0 / (1 - beta1**t1),
                delta=m.delta / (1 - beta1**t1),
            )
            vhat = _AddRateParamsBatched(
                theta0=vv.theta0 / (1 - beta2**t1),
                beta=vv.beta / (1 - beta2**t1),
                beta_speed=vv.beta_speed / (1 - beta2**t1),
                phi0=vv.phi0 / (1 - beta2**t1),
                delta=vv.delta / (1 - beta2**t1),
            )

            params = _AddRateParamsBatched(
                theta0=params.theta0 - lr * mhat.theta0 / (jnp.sqrt(vhat.theta0) + eps),
                beta=params.beta - lr * mhat.beta / (jnp.sqrt(vhat.beta) + eps),
                beta_speed=params.beta_speed
                - lr * mhat.beta_speed / (jnp.sqrt(vhat.beta_speed) + eps),
                phi0=params.phi0 - lr * mhat.phi0 / (jnp.sqrt(vhat.phi0) + eps),
                delta=params.delta - lr * mhat.delta / (jnp.sqrt(vhat.delta) + eps),
            )

        return params

    # ---------- NO SPEED BRANCH ----------
    def nll_ns(params: _AddRateParamsBatchedNoSpeed) -> jnp.ndarray:
        eta = params.theta0[None, :] + (B_j @ params.beta)
        z = params.phi0[None, :] + (B_j @ params.delta)

        eta_d = eta[idx_d, :]
        y_d = y_j[idx_d, :]
        ll_dark = jnp.sum(y_d * eta_d - jnp.exp(eta_d))

        eta_l = eta[idx_l, :]
        z_l = z[idx_l, :]
        y_l = y_j[idx_l, :]
        log_rate_l = jnp.logaddexp(eta_l, z_l)
        rate_l = jnp.exp(eta_l) + jnp.exp(z_l)
        ll_light = jnp.sum(y_l * log_rate_l - rate_l)

        pen = 0.5 * ridge * (jnp.sum(params.beta**2) + jnp.sum(params.delta**2))
        T = y_j.shape[0]
        ll = ll_dark + ll_light
        data_nll = -(ll / T)
        return data_nll + pen

    init = _init_additive_from_base(y=y, B=B, v=None, L=L, ridge=ridge)
    nll_jit = jax.jit(nll_ns)

    jaxopt = _try_import_jaxopt()
    if jaxopt is not None:
        solver = jaxopt.LBFGS(fun=nll_jit, maxiter=int(maxiter))
        params_hat, _ = solver.run(init)
        return params_hat

    # Adam fallback (no-speed)
    grad_jit = jax.jit(jax.grad(nll_ns))
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def zeros_like_ns(p):
        return _AddRateParamsBatchedNoSpeed(
            theta0=jnp.zeros_like(p.theta0),
            beta=jnp.zeros_like(p.beta),
            phi0=jnp.zeros_like(p.phi0),
            delta=jnp.zeros_like(p.delta),
        )

    params = init
    m = zeros_like_ns(init)
    vv = zeros_like_ns(init)

    for t in range(int(maxiter)):
        g = grad_jit(params)
        m = _AddRateParamsBatchedNoSpeed(
            theta0=beta1 * m.theta0 + (1 - beta1) * g.theta0,
            beta=beta1 * m.beta + (1 - beta1) * g.beta,
            phi0=beta1 * m.phi0 + (1 - beta1) * g.phi0,
            delta=beta1 * m.delta + (1 - beta1) * g.delta,
        )
        vv = _AddRateParamsBatchedNoSpeed(
            theta0=beta2 * vv.theta0 + (1 - beta2) * (g.theta0**2),
            beta=beta2 * vv.beta + (1 - beta2) * (g.beta**2),
            phi0=beta2 * vv.phi0 + (1 - beta2) * (g.phi0**2),
            delta=beta2 * vv.delta + (1 - beta2) * (g.delta**2),
        )

        t1 = t + 1
        mhat = _AddRateParamsBatchedNoSpeed(
            theta0=m.theta0 / (1 - beta1**t1),
            beta=m.beta / (1 - beta1**t1),
            phi0=m.phi0 / (1 - beta1**t1),
            delta=m.delta / (1 - beta1**t1),
        )
        vhat = _AddRateParamsBatchedNoSpeed(
            theta0=vv.theta0 / (1 - beta2**t1),
            beta=vv.beta / (1 - beta2**t1),
            phi0=vv.phi0 / (1 - beta2**t1),
            delta=vv.delta / (1 - beta2**t1),
        )

        params = _AddRateParamsBatchedNoSpeed(
            theta0=params.theta0 - lr * mhat.theta0 / (jnp.sqrt(vhat.theta0) + eps),
            beta=params.beta - lr * mhat.beta / (jnp.sqrt(vhat.beta) + eps),
            phi0=params.phi0 - lr * mhat.phi0 / (jnp.sqrt(vhat.phi0) + eps),
            delta=params.delta - lr * mhat.delta / (jnp.sqrt(vhat.delta) + eps),
        )

    return params


def fit_true_additive_rate_poisson_fast_per_traj(
    spikes,
    trajectory_ep_by_epoch: Dict[Hashable, Dict[str, any]],
    tp_by_epoch: Dict[Hashable, Dict[str, any]],
    speed_by_epoch: Dict[Hashable, any] | None = None,  # <-- OPTIONAL
    *,
    traj_name: str,
    light_epochs: Sequence[Hashable],
    dark_epochs: Sequence[Hashable],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: Dict[Hashable, any] | None = None,
    maxiter_full: int = 200,
    lr_full: float = 5e-2,
) -> Dict[str, np.ndarray]:
    """
    True additive-in-rate Poisson model with standardized outputs.

    Base (for CV comparison): exp(theta0 + B@beta [+ beta_speed*v])
    Full: base + L*exp(phi0 + B@delta)

    Returns same standardized dict keys as the other two functions.
    """
    all_epochs = list(light_epochs) + list(dark_epochs)
    if len(all_epochs) == 0:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    has_speed = speed_by_epoch is not None

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    unit_ids: np.ndarray | None = None

    sp = spikes if unit_mask is None else spikes[unit_mask]

    for ep in all_epochs:
        ep_traj = trajectory_ep_by_epoch[ep][traj_name]
        if restrict_ep_by_epoch is not None:
            ep_traj = ep_traj.intersect(restrict_ep_by_epoch[ep].time_support)

        counts = sp.count(bin_size_s, ep=ep_traj)

        cols = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = cols
        else:
            if cols.shape != unit_ids.shape or not np.all(cols == unit_ids):
                raise ValueError(
                    "Spike count columns (unit order) differ across epochs."
                )

        y = np.asarray(counts.d, dtype=np.float32)  # (T,U)

        p = (
            tp_by_epoch[ep][traj_name]
            .interpolate(counts)
            .to_numpy()
            .reshape(-1)
            .astype(np.float32)
        )
        L = (np.ones_like(p) if ep in light_epochs else np.zeros_like(p)).astype(
            np.float32
        )

        if has_speed:
            v = (
                speed_by_epoch[ep]
                .interpolate(counts)
                .to_numpy()
                .reshape(-1)
                .astype(np.float32)
            )
            good = np.isfinite(p) & np.isfinite(v)
            vs.append(v[good])
        else:
            good = np.isfinite(p)

        ys.append(y[good])
        ps.append(p[good])
        ls.append(L[good])

    if unit_ids is None:
        raise ValueError("No data after parsing epochs/trajectory intervals.")

    y_all = np.concatenate(ys, axis=0)  # (T,U)
    p_all = np.concatenate(ps, axis=0).reshape(-1)  # (T,)
    l_all = np.concatenate(ls, axis=0).reshape(-1)  # (T,)

    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need BOTH light and dark bins across the provided epochs.")

    if has_speed:
        v_all = np.concatenate(vs, axis=0).reshape(-1).astype(np.float32)
        speed_mean = float(np.mean(v_all))
        speed_std = float(np.std(v_all) + 1e-6)
        v_all = (v_all - speed_mean) / speed_std
    else:
        v_all = None
        speed_mean = np.nan
        speed_std = np.nan

    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    B_all = np.asarray(basis.compute_features(p_all), dtype=np.float32)  # (T,K)
    K = B_all.shape[1]
    U = y_all.shape[1]

    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)

    ll_base_sum = np.zeros(U, dtype=np.float64)
    ll_full_sum = np.zeros(U, dtype=np.float64)
    spk_sum = np.zeros(U, dtype=np.float64)

    # ----- CV loop -----
    for tr, te in folds:
        B_tr, L_tr, y_tr = B_all[tr], l_all[tr], y_all[tr]
        B_te, L_te, y_te = B_all[te], l_all[te], y_all[te]

        if has_speed:
            v_tr = v_all[tr]
            v_te = v_all[te]
            x_base_tr = np.concatenate([B_tr, v_tr[:, None]], axis=1)  # (Ttr, K+1)
        else:
            v_tr = None
            v_te = None
            x_base_tr = B_tr  # (Ttr, K)

        # base fit (exp-link Poisson) using nemos
        m_base = PopulationGLM(
            "Poisson", regularizer="Ridge", regularizer_strength=ridge
        )
        m_base.fit(x_base_tr, y_tr)

        if has_speed:
            coef_b = _coef_feat_by_unit(m_base, n_features=K + 1)  # (K+1,U)
            theta0_b = np.asarray(m_base.intercept_).reshape(-1).astype(np.float32)
            beta_b = coef_b[:K].astype(np.float32)
            beta_v_b = coef_b[K].astype(np.float32)
            eta_b_te = (
                theta0_b[None, :] + (B_te @ beta_b) + v_te[:, None] * beta_v_b[None, :]
            )
        else:
            coef_b = _coef_feat_by_unit(m_base, n_features=K)  # (K,U)
            theta0_b = np.asarray(m_base.intercept_).reshape(-1).astype(np.float32)
            beta_b = coef_b.astype(np.float32)
            eta_b_te = theta0_b[None, :] + (B_te @ beta_b)

        ll_b = np.sum(y_te * eta_b_te - np.exp(eta_b_te), axis=0)  # (U,)
        ll_norm = np.sum(scipy.special.gammaln(y_te + 1.0), axis=0)
        ll_b = ll_b - ll_norm

        # full fit (true additive)
        pars = _fit_additive_rate_batched_jax(
            y=np.asarray(y_tr, dtype=np.float32),
            B=np.asarray(B_tr, dtype=np.float32),
            v=None if not has_speed else np.asarray(v_tr, dtype=np.float32),
            L=np.asarray(L_tr, dtype=np.float32),
            ridge=ridge,
            maxiter=maxiter_full,
            lr=lr_full,
        )

        if has_speed:
            pars = pars  # _AddRateParamsBatched
            eta_dark_te = (
                np.asarray(pars.theta0)[None, :]
                + (B_te @ np.asarray(pars.beta))
                + v_te[:, None] * np.asarray(pars.beta_speed)[None, :]
            )
            z_add_te = np.asarray(pars.phi0)[None, :] + (B_te @ np.asarray(pars.delta))
        else:
            pars = pars  # _AddRateParamsBatchedNoSpeed
            eta_dark_te = np.asarray(pars.theta0)[None, :] + (
                B_te @ np.asarray(pars.beta)
            )
            z_add_te = np.asarray(pars.phi0)[None, :] + (B_te @ np.asarray(pars.delta))

        is_light = L_te > 0.5
        ll_f = np.zeros(U, dtype=np.float64)

        # dark bins
        if np.any(~is_light):
            eta_d = eta_dark_te[~is_light, :]
            y_d = y_te[~is_light, :]
            ll_f += np.sum(y_d * eta_d - np.exp(eta_d), axis=0)
            # ll_f = ll_f - ll_norm

        # light bins
        if np.any(is_light):
            eta_l = eta_dark_te[is_light, :]
            z_l = z_add_te[is_light, :]
            y_l = y_te[is_light, :]

            log_rate_l = np.logaddexp(eta_l, z_l)
            rate_l = np.exp(eta_l) + np.exp(z_l)
            ll_f += np.sum(y_l * log_rate_l - rate_l, axis=0)
            # ll_f = ll_f - ll_norm

        # subtract factorial term once (applies to all bins)
        ll_f -= ll_norm
        ll_base_sum += ll_b
        ll_full_sum += ll_f
        spk_sum += np.sum(y_te, axis=0)

    dLL_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spk = np.where(spk_sum > 0, ll_base_sum / spk_sum, np.nan)
        ll_full_per_spk = np.where(spk_sum > 0, ll_full_sum / spk_sum, np.nan)
        dll_per_spk = np.where(spk_sum > 0, dLL_sum / spk_sum, np.nan)

    # ----- base fit on ALL data (for base coefficients like other functions) -----
    if has_speed:
        x_base_all = np.concatenate([B_all, v_all[:, None]], axis=1)
        m_base_all = PopulationGLM(
            "Poisson", regularizer="Ridge", regularizer_strength=ridge
        )
        m_base_all.fit(x_base_all, y_all)
        coef0 = _coef_feat_by_unit(m_base_all, n_features=K + 1)
        intercept0 = np.asarray(m_base_all.intercept_).reshape(-1)
        coef_place_base = coef0[:K, :]
        coef_speed_base = coef0[K, :]
    else:
        x_base_all = B_all
        m_base_all = PopulationGLM(
            "Poisson", regularizer="Ridge", regularizer_strength=ridge
        )
        m_base_all.fit(x_base_all, y_all)
        coef0 = _coef_feat_by_unit(m_base_all, n_features=K)
        intercept0 = np.asarray(m_base_all.intercept_).reshape(-1)
        coef_place_base = coef0
        coef_speed_base = np.full((U,), np.nan)

    # ----- full fit on ALL data -----
    pars_all = _fit_additive_rate_batched_jax(
        y=np.asarray(y_all, dtype=np.float32),
        B=np.asarray(B_all, dtype=np.float32),
        v=None if not has_speed else np.asarray(v_all, dtype=np.float32),
        L=np.asarray(l_all, dtype=np.float32),
        ridge=ridge,
        maxiter=maxiter_full,
        lr=lr_full,
    )

    if has_speed:
        theta0_all = np.asarray(pars_all.theta0)
        beta_all = np.asarray(pars_all.beta)
        beta_speed_all = np.asarray(pars_all.beta_speed)
        phi0_all = np.asarray(pars_all.phi0)
        delta_all = np.asarray(pars_all.delta)
    else:
        theta0_all = np.asarray(pars_all.theta0)
        beta_all = np.asarray(pars_all.beta)
        beta_speed_all = np.full((U,), np.nan)
        phi0_all = np.asarray(pars_all.phi0)
        delta_all = np.asarray(pars_all.delta)

    # Fields at mean speed (v_z = 0)
    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200).astype(np.float32)
    Bgrid = np.asarray(basis.compute_features(grid), dtype=np.float32)

    dark_bin = np.exp(theta0_all[None, :] + (Bgrid @ beta_all))  # spikes/bin
    add_bin = np.exp(phi0_all[None, :] + (Bgrid @ delta_all))  # spikes/bin
    light_bin = dark_bin + add_bin

    dark_hz_grid = dark_bin / bin_size_s
    light_hz_grid = light_bin / bin_size_s

    # Fill non-applicable terms with NaNs for standardized output
    nan_u = np.full((U,), np.nan)
    nan_KU = np.full((K, U), np.nan)

    return {
        # identity / meta
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        # CV metrics
        "spike_sum_cv": spk_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": dLL_sum,
        "ll_base_per_spike_cv": ll_base_per_spk,
        "ll_full_per_spike_cv": ll_full_per_spk,
        "dll_per_spike_cv": dll_per_spk,
        "ll_base_bits_per_spike_cv": ll_base_per_spk * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spk * inv_log2,
        "dll_bits_per_spike_cv": dll_per_spk * inv_log2,
        # coefficients (base fit on all data)
        "coef_intercept_base_all": intercept0,
        "coef_place_base_all": coef_place_base,
        "coef_light_base_all": nan_u,  # base has no L here
        "coef_place_x_light_base_all": nan_KU,  # not in base
        "coef_speed_base_all": coef_speed_base,
        # coefficients (full fit on all data) -- baseline component
        "coef_intercept_full_all": theta0_all,
        "coef_place_full_all": beta_all,
        "coef_light_full_all": nan_u,  # no global L term
        "coef_place_x_light_full_all": nan_KU,  # no B*L term
        "coef_speed_full_all": beta_speed_all,
        # additive component (light-gated)
        "coef_add_intercept_full_all": phi0_all,
        "coef_add_place_full_all": delta_all,
        # fields (Hz)
        "grid_tp": np.asarray(grid),
        "dark_hz_grid": dark_hz_grid,
        "light_hz_grid": light_hz_grid,
    }


def fit_separate_dark_light_fields_nemos_per_traj(
    spikes,
    trajectory_ep_by_epoch: Dict[Hashable, Dict[str, any]],
    tp_by_epoch: Dict[Hashable, Dict[str, any]],
    speed_by_epoch: Dict[Hashable, any] | None = None,  # OPTIONAL
    *,
    traj_name: str,
    light_epochs: Sequence[Hashable],
    dark_epochs: Sequence[Hashable],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: Tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: Dict[Hashable, any] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Separate dark and light place fields (no shared place + modulation parameterization).

    Base (same as other exp-link models; for nested comparison):
        X_base = [B(p), L, (speed optional)]
        eta = intercept + B@beta_shared + beta_L*L [+ beta_v*v]

    Full (separate fields):
        X_full = [B(p)*(1-L), B(p)*L, L, (speed optional)]
        eta = intercept + (B*(1-L))@beta_dark + (B*L)@beta_light + beta_L*L [+ beta_v*v]

    Outputs:
      - Same standardized keys as your other model functions
      - dark_hz_grid and light_hz_grid computed from the FULL model at mean speed (v_z=0)
      - CV LL in bits/spike from nemos' log-likelihood (includes -gammaln(y+1) term)
    """
    all_epochs = list(light_epochs) + list(dark_epochs)
    if len(all_epochs) == 0:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    has_speed = speed_by_epoch is not None

    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    unit_ids: np.ndarray | None = None

    for ep in all_epochs:
        ep_traj = trajectory_ep_by_epoch[ep][traj_name]
        if restrict_ep_by_epoch is not None:
            ep_traj = ep_traj.intersect(restrict_ep_by_epoch[ep].time_support)

        sp = spikes if unit_mask is None else spikes[unit_mask]
        counts = sp.count(bin_size_s, ep=ep_traj)

        cols = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = cols
        else:
            if cols.shape != unit_ids.shape or not np.all(cols == unit_ids):
                raise ValueError(
                    "Spike count columns (unit order) differ across epochs."
                )

        y = np.asarray(counts.d, dtype=float)  # (T, U)

        p = tp_by_epoch[ep][traj_name].interpolate(counts).to_numpy().reshape(-1)
        L = (
            np.ones_like(p, dtype=float)
            if ep in light_epochs
            else np.zeros_like(p, dtype=float)
        )

        if has_speed:
            v = speed_by_epoch[ep].interpolate(counts).to_numpy().reshape(-1)
            good = np.isfinite(p) & np.isfinite(v)
            vs.append(v[good])
        else:
            good = np.isfinite(p)

        ys.append(y[good])
        ps.append(p[good])
        ls.append(L[good])

    if unit_ids is None:
        raise ValueError("No data after parsing epochs/trajectory intervals.")

    y_all = np.concatenate(ys, axis=0)  # (T, U)
    p_all = np.concatenate(ps, axis=0).reshape(-1)  # (T,)
    l_all = np.concatenate(ls, axis=0).reshape(-1)  # (T,)

    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need BOTH light and dark bins across the provided epochs.")

    # Optional speed z-score
    if has_speed:
        v_all = np.concatenate(vs, axis=0).reshape(-1)
        speed_mean = float(np.mean(v_all))
        speed_std = float(np.std(v_all) + 1e-12)
        v_all = (v_all - speed_mean) / speed_std
    else:
        v_all = None
        speed_mean = np.nan
        speed_std = np.nan

    # B-spline features
    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    B = np.asarray(basis.compute_features(p_all), dtype=float)  # (T, K)
    K = B.shape[1]
    U = y_all.shape[1]

    # -----------------------
    # Design matrices
    # -----------------------
    # Base: [B, L, (v)]
    x_base = np.concatenate([B, l_all[:, None]], axis=1)  # (T, K+1)
    if has_speed:
        x_base = np.concatenate([x_base, v_all[:, None]], axis=1)  # (T, K+2)

    # Full: [B*(1-L), B*L, L, (v)]
    Bd = B * (1.0 - l_all[:, None])  # dark-only basis
    Bl = B * (l_all[:, None])  # light-only basis
    x_full = np.concatenate([Bd, Bl, l_all[:, None]], axis=1)  # (T, 2K+1)
    if has_speed:
        x_full = np.concatenate([x_full, v_all[:, None]], axis=1)  # (T, 2K+2)

    # CV folds
    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    agg_sum_per_neuron = lambda arr: jnp.sum(arr, axis=0)

    ll_base_sum = np.zeros(U, dtype=float)
    ll_full_sum = np.zeros(U, dtype=float)
    spk_sum = np.zeros(U, dtype=float)

    for tr, te in folds:
        m0 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m1 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m0.fit(x_base[tr], y_all[tr])
        m1.fit(x_full[tr], y_all[tr])

        ll0 = np.asarray(
            m0.score(
                x_base[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        ll1 = np.asarray(
            m1.score(
                x_full[te],
                y_all[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )

        ll_base_sum += ll0
        ll_full_sum += ll1
        spk_sum += np.asarray(y_all[te].sum(axis=0), dtype=float)

    dLL_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spk = np.where(spk_sum > 0, ll_base_sum / spk_sum, np.nan)
        ll_full_per_spk = np.where(spk_sum > 0, ll_full_sum / spk_sum, np.nan)
        dll_per_spk = np.where(spk_sum > 0, dLL_sum / spk_sum, np.nan)

    # -----------------------
    # Fit on ALL data for coefficients + fields
    # -----------------------
    m0_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m1_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m0_all.fit(x_base, y_all)
    m1_all.fit(x_full, y_all)

    coef0 = _coef_feat_by_unit(m0_all, n_features=x_base.shape[1])  # (P0,U)
    coef1 = _coef_feat_by_unit(m1_all, n_features=x_full.shape[1])  # (P1,U)

    intercept0 = np.asarray(m0_all.intercept_).reshape(-1)
    intercept1 = np.asarray(m1_all.intercept_).reshape(-1)

    # ---- Base coefficients: [B, L, (v)] ----
    coef_place_base = coef0[0:K, :]
    coef_light_base = coef0[K, :]
    coef_speed_base = coef0[-1, :] if has_speed else np.full((U,), np.nan)

    # ---- Full coefficients: [Bd, Bl, L, (v)] ----
    coef_place_dark = coef1[0:K, :]  # β_dark
    coef_place_light = coef1[K : 2 * K, :]  # β_light
    coef_light_full = coef1[2 * K, :]  # β_L (global light offset)
    coef_speed_full = coef1[-1, :] if has_speed else np.full((U,), np.nan)

    # For standardized output keys, also provide "light change" in spline space:
    # (this makes it easy to compare to the gain-modulated model outputs)
    coef_place_change = coef_place_light - coef_place_dark  # β_light - β_dark

    # Terms not in this family
    nan_u = np.full((U,), np.nan)
    nan_KU = np.full((K, U), np.nan)

    # -----------------------
    # Fields (Hz) from FULL model at mean speed (v_z=0)
    # -----------------------
    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    Bgrid = np.asarray(basis.compute_features(grid), dtype=float)  # (G,K)

    dark_lin = intercept1[None, :] + (Bgrid @ coef_place_dark)
    light_lin = (
        intercept1[None, :] + coef_light_full[None, :] + (Bgrid @ coef_place_light)
    )

    dark_hz_grid = np.exp(dark_lin) / bin_size_s
    light_hz_grid = np.exp(light_lin) / bin_size_s

    return {
        # identity / meta
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        # CV metrics
        "spike_sum_cv": spk_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": dLL_sum,
        "ll_base_per_spike_cv": ll_base_per_spk,
        "ll_full_per_spike_cv": ll_full_per_spk,
        "dll_per_spike_cv": dll_per_spk,
        "ll_base_bits_per_spike_cv": ll_base_per_spk * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spk * inv_log2,
        "dll_bits_per_spike_cv": dll_per_spk * inv_log2,
        # coefficients (base fit on all data)
        "coef_intercept_base_all": intercept0,
        "coef_place_base_all": coef_place_base,
        "coef_light_base_all": coef_light_base,
        "coef_place_x_light_base_all": nan_KU,  # not in base
        "coef_speed_base_all": coef_speed_base,
        # coefficients (full fit on all data)
        # We store dark place in coef_place_full_all, and (light-dark) in coef_place_x_light_full_all
        "coef_intercept_full_all": intercept1,
        "coef_place_full_all": coef_place_dark,
        "coef_light_full_all": coef_light_full,
        "coef_place_x_light_full_all": coef_place_change,
        "coef_speed_full_all": coef_speed_full,
        # additive-component coefficients (not in this family)
        "coef_add_intercept_full_all": nan_u,
        "coef_add_place_full_all": nan_KU,
        # fields (Hz)
        "grid_tp": grid,
        "dark_hz_grid": dark_hz_grid,
        "light_hz_grid": light_hz_grid,
    }


def main():
    print("Starting main()", flush=True)
    save_dir = analysis_path / "tp_glm2"
    save_dir.mkdir(parents=True, exist_ok=True)

    place_bin_size = 4  # cm
    task_progression_bin_size = place_bin_size / total_length_per_trajectory

    task_progression_by_trajectory, task_progression_by_trajectory_bins = (
        get_task_progression_by_trajectory(
            task_progression_bin_size=task_progression_bin_size
        )
    )

    region = "v1"

    dark_epochs = [run_epoch_list[3]]

    unit_mask = fr_during_movement[region][run_epoch_list[3]] > 0.5
    bin_size_s = 0.02
    seed = 47
    n_folds = 5
    n_splines = 25
    n_splines_gain = 6
    speed_by_epoch = speed
    # speed_by_epoch = None

    for epoch in run_epoch_list[:-1]:
        # for epoch in [run_epoch_list[2]]:
        light_epochs = [epoch]
        mult = {}
        # add = {}
        add_jax = {}
        sep = {}
        mult_per_segment = {}
        mult_fewer_gain_splines = {}
        for traj in trajectory_types:
            print(f"traj={traj}", flush=True)
            mult[traj] = {}
            # add[traj] = {}
            add_jax[traj] = {}
            sep[traj] = {}
            mult_per_segment[traj] = {}
            mult_fewer_gain_splines[traj] = {}
            for ridge in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                print(f"  ridge={ridge}", flush=True)
                mult[traj][ridge] = fit_shared_place_light_mod_nemos_per_traj(
                    spikes=spikes[region],
                    trajectory_ep_by_epoch=trajectory_ep,
                    tp_by_epoch=task_progression_by_trajectory,
                    speed_by_epoch=speed_by_epoch,
                    traj_name=traj,
                    light_epochs=light_epochs,
                    dark_epochs=dark_epochs,
                    bin_size_s=bin_size_s,
                    unit_mask=unit_mask,
                    restrict_ep_by_epoch=movement,
                    seed=seed,
                    ridge=ridge,
                    n_folds=n_folds,
                    n_splines=n_splines,
                )
                mult_per_segment[traj][ridge] = (
                    fit_shared_place_light_mod_nemos_per_traj_segment_gain(
                        spikes=spikes[region],
                        trajectory_ep_by_epoch=trajectory_ep,
                        tp_by_epoch=task_progression_by_trajectory,
                        speed_by_epoch=speed_by_epoch,  # can also pass None
                        traj_name=traj,
                        light_epochs=light_epochs,
                        dark_epochs=dark_epochs,
                        bin_size_s=bin_size_s,
                        unit_mask=unit_mask,
                        restrict_ep_by_epoch=movement,
                        seed=seed,
                        ridge=ridge,
                        n_folds=n_folds,
                        n_splines=n_splines,
                        segment_edges=(
                            0.0,
                            tp_segment_border1,
                            tp_segment_border2,
                            1.0,
                        ),  # <-- YOU set these
                    )
                )
                mult_fewer_gain_splines[traj][ridge] = (
                    fit_shared_place_light_mod_nemos_per_traj_gain_splines(
                        spikes=spikes[region],
                        trajectory_ep_by_epoch=trajectory_ep,
                        tp_by_epoch=task_progression_by_trajectory,
                        speed_by_epoch=speed_by_epoch,  # can also pass None
                        traj_name=traj,
                        light_epochs=light_epochs,
                        dark_epochs=dark_epochs,
                        bin_size_s=bin_size_s,
                        unit_mask=unit_mask,
                        restrict_ep_by_epoch=movement,
                        seed=seed,
                        ridge=ridge,
                        n_folds=n_folds,
                        n_splines=n_splines,
                        n_splines_gain=n_splines_gain,  # gain field (new!)
                    )
                )
                add_jax[traj][ridge] = fit_true_additive_rate_poisson_fast_per_traj(
                    spikes=spikes[region],
                    trajectory_ep_by_epoch=trajectory_ep,
                    tp_by_epoch=task_progression_by_trajectory,
                    speed_by_epoch=speed_by_epoch,
                    traj_name=traj,
                    light_epochs=light_epochs,
                    dark_epochs=dark_epochs,
                    bin_size_s=bin_size_s,
                    unit_mask=unit_mask,
                    restrict_ep_by_epoch=movement,
                    seed=seed,
                    ridge=ridge,
                    n_folds=n_folds,
                )
                sep[traj][ridge] = fit_separate_dark_light_fields_nemos_per_traj(
                    spikes=spikes[region],
                    trajectory_ep_by_epoch=trajectory_ep,
                    tp_by_epoch=task_progression_by_trajectory,
                    speed_by_epoch=speed_by_epoch,
                    traj_name=traj,
                    light_epochs=light_epochs,
                    dark_epochs=dark_epochs,
                    bin_size_s=bin_size_s,
                    unit_mask=unit_mask,
                    restrict_ep_by_epoch=movement,
                    seed=seed,
                    ridge=ridge,
                    n_folds=n_folds,
                )

        print("Saving pickles...", flush=True)
        with open(save_dir / f"{epoch}_mult_speed.pkl", "wb") as f:
            pickle.dump(mult, f)
        with open(save_dir / f"{epoch}_mult_per_segment_speed.pkl", "wb") as f:
            pickle.dump(mult_per_segment, f)
        with open(
            save_dir / f"{epoch}_mult_fewer_gain_splines_{n_splines_gain}_speed.pkl",
            "wb",
        ) as f:
            pickle.dump(mult_fewer_gain_splines, f)
        with open(save_dir / f"{epoch}_add_jax_speed.pkl", "wb") as f:
            pickle.dump(add_jax, f)
        with open(save_dir / f"{epoch}_sep_speed.pkl", "wb") as f:
            pickle.dump(sep, f)

        print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise
