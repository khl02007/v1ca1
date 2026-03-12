"""motor_vs_tp_glm_clean.py

Test whether neurons encode *task progression* beyond *motor variables* on a W track.

This file is intended to be a clean, minimal, and easy-to-read replacement for
larger analysis scripts. It uses **nemos**' `PopulationGLM` so that log-likelihood
scoring matches nemos' internal implementation.

Overview
--------
We compare two Poisson GLMs (exp link) using cross-validated log-likelihood
(reported as bits/spike):

Model M0 ("motor")
    log μ(t,u) = b0_u
                + X_motor(t) @ β_motor_u
                + X_traj(t)  @ β_traj_u

Model M1 ("motor+tp")
    log μ(t,u) = b0_u
                + X_motor(t) @ β_motor_u
                + X_traj(t)  @ β_traj_u
                + Σ_j  I_j(t) * B_tp(tp_shifted(t)) @ β_tp[j]_u

Where
  • μ(t,u) is the expected spike *count* in bin t for unit u (spikes/bin).
  • X_motor(t) are spline features of motor covariates shared across trajectories.
  • X_traj(t) are trajectory one-hot indicators (3 columns; one trajectory is
    the reference) so *both* models can have different mean rates per trajectory.
  • tp_shifted(t) is task progression (linearized fraction along the current
    trajectory) shifted so a single spline basis over [0, 2] can cover all four
    W-track trajectories:

        left-turn trajectories  (center_to_left, left_to_center):  tp ∈ [0, 1]
        right-turn trajectories (center_to_right, right_to_center): tp ∈ [1, 2]

    i.e. we compute tp_raw ∈ [0,1] for each trajectory, then add +1 for the two
    right-turn trajectories.
  • I_j(t) is a 0/1 indicator for the *current* trajectory type (4 types). This
    “gates” task progression so each trajectory gets its own TP tuning curve,
    while motor coefficients remain shared across all trajectories.

Data restriction
----------------
We keep time bins only if they are:
  (a) inside ANY provided trajectory interval, and
  (b) inside `movement_ep`.

Cross-validation
----------------
We use **stratified contiguous folds by trajectory**: within each trajectory we
split time bins into contiguous chunks and assign chunks to folds. This helps
ensure each fold contains samples from all trajectories while avoiding random
time-shuffling leakage.

Log-likelihood normalization
----------------------------
Scoring is done via `PopulationGLM.score(..., score_type="log-likelihood")`,
which computes the full Poisson log-likelihood including the -log(y!) term.
That makes the LL comparable across models.

Outputs
-------
The main function `fit_motor_vs_task_progression_wtrack_epoch` returns:
  • CV bits/spike for both models (overall and per trajectory)
  • coefficients (fit on all data) split into blocks (motor, trajectory, TP)
  • TP “rate curves” in Hz implied by the motor+tp model at mean motor covariates

Dependencies
------------
This script assumes you already have:
  • spike trains as a `pynapple.TsGroup`
  • position and body position as `pynapple.TsdFrame` with columns ['x','y']
  • trajectory intervals as a dict of 4 `pynapple.IntervalSet`
  • a movement IntervalSet (e.g., speed > threshold)

The linearization uses `track_linearization` (your local module) + the W-track
node geometry below.
"""

from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pynapple as nap

import jax.numpy as jnp

from nemos.basis import BSplineEval
from nemos.glm import PopulationGLM

import position_tools as pt
import track_linearization as tl

import pickle
from pathlib import Path
import spikeinterface.full as si
import kyutils


# If these are in another module, import them accordingly:
# from motor_vs_tp_glm_clean import fit_motor_vs_task_progression_wtrack_epoch, get_tsgroup

TRAJECTORY_TYPES = [
    "center_to_left",
    "left_to_center",
    "center_to_right",
    "right_to_center",
]

TP_GROUPS: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("LC_CR", ("left_to_center", "center_to_right")),
    ("RC_CL", ("right_to_center", "center_to_left")),
)


def _sampling_rate(t_position: np.ndarray) -> float:
    t_position = np.asarray(t_position)
    return (len(t_position) - 1) / (t_position[-1] - t_position[0])


def _as_xy(arr) -> np.ndarray:
    """
    Robustly coerce position-like arrays to shape (T, 2).
    Works for numpy arrays and pandas DataFrames (via np.asarray).
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array for position, got shape {a.shape}")
    # common case: (T,2) or (T,>=2)
    if a.shape[1] >= 2:
        return a[:, :2]
    # sometimes stored as (2,T)
    if a.shape[0] >= 2:
        return a[:2, :].T
    raise ValueError(f"Could not interpret position array of shape {a.shape} as XY.")


# -------------------------
# W-track trajectory types
# -------------------------

TRAJECTORY_TYPES: Tuple[str, ...] = (
    "center_to_left",
    "left_to_center",
    "center_to_right",
    "right_to_center",
)

# -------------------------
# Geometry / linearization
# -------------------------


@dataclass(frozen=True)
class WTrackGeometry:
    """Container for W-track geometry parameters used in linearization."""

    total_length: float
    track_graph_from_center: Dict[str, any]
    track_graph_to_center: Dict[str, any]
    edge_order_from_center: List[Tuple[int, int]]
    edge_order_to_center: List[Tuple[int, int]]


def build_wtrack_geometry() -> WTrackGeometry:
    """Build W-track graphs used for linearizing 2D position.

    The node coordinates here match the geometry used in your earlier scripts.
    The track has a central well and left/right outer wells with diagonal arms.

    Returns
    -------
    WTrackGeometry
        Contains `track_graph_from_center` and `track_graph_to_center` dicts,
        keyed by trajectory type.
    """

    # --- Geometry constants (same as your earlier scripts) ---
    dx = 9.5
    dy = 9.0
    diagonal_segment_length = float(np.sqrt(dx**2 + dy**2))

    long_segment_length = 81 - 17 - 2
    short_segment_length = 13.5

    total_length_per_trajectory = (
        long_segment_length * 2 + short_segment_length + 2 * diagonal_segment_length
    )

    # Node positions (right arm)
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

    # Node positions (left arm)
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

    edge_order_from_center = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    edge_order_to_center = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]

    track_graph_from_center: Dict[str, any] = {}
    track_graph_to_center: Dict[str, any] = {}

    for traj in TRAJECTORY_TYPES:
        if traj in ("center_to_right", "right_to_center"):
            track_graph_from_center[traj] = tl.make_track_graph(
                node_positions_right, edges_from_center
            )
            track_graph_to_center[traj] = tl.make_track_graph(
                node_positions_right, edges_to_center
            )
        else:
            track_graph_from_center[traj] = tl.make_track_graph(
                node_positions_left, edges_from_center
            )
            track_graph_to_center[traj] = tl.make_track_graph(
                node_positions_left, edges_to_center
            )

    return WTrackGeometry(
        total_length=float(total_length_per_trajectory),
        track_graph_from_center=track_graph_from_center,
        track_graph_to_center=track_graph_to_center,
        edge_order_from_center=edge_order_from_center,
        edge_order_to_center=edge_order_to_center,
    )


def compute_task_progression_by_traj_wtrack(
    *,
    pos_xy: np.ndarray,  # (Tpos,2)
    pos_t: np.ndarray,  # (Tpos,)
    trajectory_eps: Mapping[str, nap.IntervalSet],
) -> Dict[str, nap.Tsd]:
    """Compute per-trajectory task progression Tsd for a W track.

    Task progression is **linearized path distance fraction** along each
    trajectory, so it always spans ~[0, 1] *within that trajectory*.

    Notes
    -----
    • For trajectories that start at the center (center_to_left/right), the
      linearization uses a graph oriented "from center".
    • For trajectories that end at the center (left/right_to_center), it uses a
      graph oriented "to center".
    • The returned Tsd for each trajectory has `time_support` set to that
      trajectory's IntervalSet.

    Parameters
    ----------
    pos_xy
        Raw 2D position samples, shape (n_samples, 2) for columns (x, y).
    pos_t
        Position timestamps in seconds, shape (n_samples,).
    trajectory_eps
        Dict mapping each trajectory name in `TRAJECTORY_TYPES` to an
        IntervalSet of times when the animal is on that trajectory.

    Returns
    -------
    tp_by_traj
        Dict {traj_name: nap.Tsd} where each Tsd value is in ~[0,1].
    """

    geom = build_wtrack_geometry()

    tp_by_traj: Dict[str, nap.Tsd] = {}
    for traj in TRAJECTORY_TYPES:
        if traj in ("center_to_left", "center_to_right"):
            tg = geom.track_graph_from_center[traj]
            edge_order = geom.edge_order_from_center
        else:
            tg = geom.track_graph_to_center[traj]
            edge_order = geom.edge_order_to_center

        pos_df = tl.get_linearized_position(
            position=pos_xy,
            track_graph=tg,
            edge_order=edge_order,
            edge_spacing=0,
        )

        frac = (pos_df["linear_position"].to_numpy() / geom.total_length).astype(float)
        tp_by_traj[traj] = nap.Tsd(
            t=pos_t,
            d=frac,
            time_support=trajectory_eps[traj],
        )

    return tp_by_traj


# -------------------------
# Utilities
# -------------------------


def get_tsgroup(sorting, timestamps_ephys_all_ptp: np.ndarray) -> nap.TsGroup:
    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
        )
    return nap.TsGroup(data, time_units="s")


def sampling_rate(t: np.ndarray) -> float:
    t = np.asarray(t)
    return (len(t) - 1) / (t[-1] - t[0])


def _coef_feat_by_unit(model: PopulationGLM, n_features: int) -> np.ndarray:
    """Return coefficients as (n_features, n_units) regardless of internal shape."""
    coef = np.asarray(model.coef_)
    if coef.shape[0] != n_features:
        coef = coef.T
    return coef


def bspline_features(
    x: np.ndarray,
    *,
    n_basis: int,
    order: int = 4,
    bounds: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, BSplineEval]:
    """Compute B-spline basis features for a 1D covariate.

    IMPORTANT: we clip x into the basis bounds to avoid NaNs from out-of-range eval.
    """
    x = np.asarray(x).reshape(-1)

    if bounds is None:
        lo = float(np.nanpercentile(x, 1.0))
        hi = float(np.nanpercentile(x, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = map(float, bounds)

    # Clip into bounds to prevent NaNs from basis eval out of range
    x_clip = np.clip(x, lo, hi)

    basis = BSplineEval(n_basis_funcs=int(n_basis), order=int(order), bounds=(lo, hi))
    X = np.asarray(basis.compute_features(x_clip), dtype=float)

    if not np.all(np.isfinite(X)):
        bad = np.where(~np.isfinite(X))
        r0, c0 = int(bad[0][0]), int(bad[1][0])
        raise ValueError(
            f"Non-finite spline features: example at row {r0}, col {c0}, value={X[r0,c0]}. "
            f"bounds=({lo},{hi})"
        )

    return X, basis


def stratified_contiguous_folds(
    labels: np.ndarray, n_folds: int, seed: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified contiguous CV folds for multi-class labels.

    We split indices *within each class* into contiguous chunks and then combine
    one chunk from each class to form each fold's test set.

    This is a multi-class generalization of your light/dark contiguous folding.
    """

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).reshape(-1)
    n = labels.size
    all_idx = np.arange(n, dtype=int)

    classes = np.unique(labels)
    test_lists: List[List[np.ndarray]] = [[] for _ in range(n_folds)]

    for c in classes:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        chunks = np.array_split(idx, n_folds)
        p = rng.permutation(n_folds)
        chunks = np.array(chunks, dtype=object)[p]
        for k in range(n_folds):
            if len(chunks[k]) > 0:
                test_lists[k].append(np.asarray(chunks[k], dtype=int))

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test = (
            np.sort(np.concatenate(test_lists[k]))
            if len(test_lists[k])
            else np.array([], dtype=int)
        )
        is_test = np.zeros(n, dtype=bool)
        is_test[test] = True
        train = all_idx[~is_test]
        folds.append((train, test))

    return folds


def _compute_motor_covariates(
    *,
    position_xy: np.ndarray,  # (Tpos,2)
    body_xy: np.ndarray,  # (Tpos,2)
    pos_t: np.ndarray,  # (Tpos,)
    spike_counts: nap.TsdFrame,
    speed_sigma_s: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Compute motor covariates and interpolate to spike count bin times."""

    # Speed from position
    fs = (len(pos_t) - 1) / (pos_t[-1] - pos_t[0])
    speed = pt.get_speed(
        position=position_xy,
        time=pos_t,
        sampling_frequency=float(fs),
        sigma=float(speed_sigma_s),
    )
    speed_tsd = nap.Tsd(t=pos_t, d=speed)
    speed_bin = speed_tsd.interpolate(spike_counts).to_numpy().reshape(-1)

    # Acceleration
    dt = float(np.median(np.diff(spike_counts.t)))
    accel_bin = np.gradient(speed_bin, dt)

    # Head direction (body->head vector)
    vec = position_xy - body_xy
    hd = np.arctan2(vec[:, 1], vec[:, 0])
    hd_tsd = nap.Tsd(t=pos_t, d=hd)
    hd_bin = hd_tsd.interpolate(spike_counts).to_numpy().reshape(-1)
    sin_hd = np.sin(hd_bin)
    cos_hd = np.cos(hd_bin)

    # Angular velocity (unwrap first)
    hd_unwrap = np.unwrap(hd)
    hd_u_tsd = nap.Tsd(t=pos_t, d=hd_unwrap)
    hd_u_bin = hd_u_tsd.interpolate(spike_counts).to_numpy().reshape(-1)
    hd_vel = np.gradient(hd_u_bin, dt)
    abs_hd_vel = np.abs(hd_vel)

    return {
        "speed": speed_bin,
        "accel": accel_bin,
        "sin_hd": sin_hd,
        "cos_hd": cos_hd,
        "hd_vel": hd_vel,
        "abs_hd_vel": abs_hd_vel,
    }


# -------------------------
# Main fitting function
# -------------------------


def fit_motor_vs_task_progression_wtrack_epoch(
    *,
    spikes: nap.TsGroup,
    position_tsd: nap.TsdFrame,
    body_position_tsd: nap.TsdFrame,
    trajectory_eps: Mapping[str, nap.IntervalSet],
    movement_ep: nap.IntervalSet,
    bin_size_s: float = 0.02,
    # task progression basis
    tp_spline_k: int = 15,
    tp_spline_order: int = 4,
    tp_bounds: Tuple[float, float] = (0.0, 2.0),
    # motor basis
    motor_spline_k: int = 7,
    motor_spline_order: int = 4,
    # regularization / CV
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
) -> Dict[str, object]:
    """Fit motor vs motor+task-progression GLMs (W track, 4 trajectories).

    This is the function you should call.

    What it does
    ------------
    1) Builds a single time-bin dataset by counting spikes across the full time
       range of `position_tsd` and then restricting to bins that are:
         • inside ANY of the 4 trajectory interval sets, AND
         • inside `movement_ep`.

    2) Computes *motor covariates* (speed, accel, head direction components,
       angular velocity) and spline-expands them.

    3) Computes per-trajectory task progression (linearized distance fraction
       along that trajectory). Then shifts task progression so that:
         • left-turn trajectories  (center_to_left, left_to_center):  tp ∈ [0,1]
         • right-turn trajectories (center_to_right, right_to_center): tp ∈ [1,2]
       This makes it natural to use a single spline basis defined over [0,2].

    4) Fits two models with cross-validation:
       - M0: motor-only (plus trajectory one-hot intercepts)
       - M1: motor + trajectory-gated task-progression spline

       Importantly:
         • Motor coefficients are shared across trajectories.
         • Task-progression coefficients are *separate for each trajectory*
           via 0/1 gating (IdentityEval-style).

    5) Reports cross-validated log-likelihood in **bits/spike** per unit:
         bits/spk = (LL_sum / spike_sum) / log(2)
       where LL_sum is the *full* Poisson log-likelihood (includes -log(y!)).

    Parameters
    ----------
    spikes
        Pynapple TsGroup of spike trains.
    position_tsd
        TsdFrame with columns ['x','y'] for head position.
    body_position_tsd
        TsdFrame with columns ['x','y'] for body position (for head direction).
    trajectory_eps
        Dict mapping each trajectory in `TRAJECTORY_TYPES` to an IntervalSet.
        These should not overlap; each time bin should belong to at most one.
    movement_ep
        IntervalSet of movement times (e.g., speed > threshold). We always
        restrict to these bins.
    bin_size_s
        Spike count bin size.
    tp_spline_k, tp_spline_order, tp_bounds
        Task-progression spline basis settings. Default bounds (0,2) match the
        shifting scheme described above.
    motor_spline_k, motor_spline_order
        Spline basis settings for motor covariates.
    ridge
        Ridge strength (L2) used by nemos PopulationGLM.
    n_folds, seed
        Cross-validation settings.
    unit_mask
        Optional boolean mask selecting units from `spikes`.

    Returns
    -------
    results
        A dict with keys:
          • 'unit_ids' : (U,) unit ids
          • 'feature_names_motor'
          • 'feature_names_traj'
          • 'feature_names_tp'
          • 'tp_basis' : metadata about TP basis
          • 'cv' : pooled + per-trajectory CV metrics
          • 'coef' : coefficients from refits on all data for both models
          • 'tp_rate_curves_hz' : model-implied TP curves from the motor+tp model
            (Hz) at mean motor covariates, for each trajectory.
    """

    # ---- Validate trajectory dict ----
    for traj in TRAJECTORY_TYPES:
        if traj not in trajectory_eps:
            raise ValueError(f"trajectory_eps is missing required key: {traj}")

    # ---- Choose units ----
    sp = spikes if unit_mask is None else spikes[unit_mask]

    # ---- Spike counts across the whole recording (then mask) ----
    pos_t = np.asarray(position_tsd.t)
    all_ep = nap.IntervalSet(start=float(pos_t[0]), end=float(pos_t[-1]))
    spike_counts = sp.count(bin_size_s, ep=all_ep)  # TsdFrame

    unit_ids = np.asarray(spike_counts.columns)
    y_full = np.asarray(spike_counts.d, dtype=float)  # (T, U)
    T_full, U = y_full.shape

    # ---- Masks: movement + any trajectory ----
    in_move = np.asarray(spike_counts.in_interval(movement_ep), dtype=bool).reshape(-1)
    in_traj: Dict[str, np.ndarray] = {
        traj: np.asarray(
            spike_counts.in_interval(trajectory_eps[traj]), dtype=bool
        ).reshape(-1)
        for traj in TRAJECTORY_TYPES
    }
    in_any = np.zeros(T_full, dtype=bool)
    for traj in TRAJECTORY_TYPES:
        in_any |= in_traj[traj]

    keep = in_move & in_any
    if not np.any(keep):
        raise ValueError(
            "No time bins remain after restricting to movement & trajectories."
        )

    # ---- Trajectory label per kept bin ----
    traj_label_full = np.full(T_full, -1, dtype=int)
    for k, traj in enumerate(TRAJECTORY_TYPES):
        traj_label_full[in_traj[traj]] = k

    # Check for overlaps (each kept bin must belong to exactly one trajectory)
    overlap_count = np.zeros(T_full, dtype=int)
    for traj in TRAJECTORY_TYPES:
        overlap_count += in_traj[traj].astype(int)
    if np.any(keep & (overlap_count != 1)):
        raise ValueError(
            "Trajectory IntervalSets appear to overlap (or have gaps inside keep mask). "
            "Each kept time bin must belong to exactly one trajectory."
        )

    # ---- Motor covariates ----
    cov = _compute_motor_covariates(
        position_xy=np.asarray(position_tsd.d),
        body_xy=np.asarray(body_position_tsd.d),
        pos_t=np.asarray(position_tsd.t),
        spike_counts=spike_counts,
    )

    # ---- Task progression per trajectory, all in [0,1] (no shifting) ----
    tp_by_traj = compute_task_progression_by_traj_wtrack(
        pos_xy=np.asarray(position_tsd.d),
        pos_t=np.asarray(position_tsd.t),
        trajectory_eps=trajectory_eps,
    )

    tp_full = np.full(T_full, np.nan, dtype=float)
    for traj in TRAJECTORY_TYPES:
        # interpolate() will respect tp_by_traj[traj].time_support, so this returns
        # values only for bins in that trajectory.
        tp_vals = (
            tp_by_traj[traj]
            .interpolate(spike_counts)
            .to_numpy()
            .reshape(-1)
            .astype(float)
        )

        m = in_traj[traj]
        n_mask = int(m.sum())
        if tp_vals.size != n_mask:
            raise RuntimeError(
                f"{traj}: interpolate returned {tp_vals.size} TP samples but "
                f"in_traj mask has {n_mask} bins. "
                "Check trajectory_eps vs spike_counts binning / epoch boundaries."
            )

        # IMPORTANT: tp_vals is already restricted to those bins, so do NOT do tp_vals[m]
        tp_full[m] = tp_vals

    # Force TP bounds to [0,1] since we are not shifting anymore
    tp_lo, tp_hi = 0.0, 1.0
    tp_full = np.clip(tp_full, tp_lo, tp_hi)

    # ---- Final mask: finite covariates + finite tp ----
    finite_cov = np.ones(T_full, dtype=bool)
    for v in cov.values():
        finite_cov &= np.isfinite(v)
    keep = keep & finite_cov & np.isfinite(tp_full)
    if not np.any(keep):
        raise ValueError(
            "No time bins remain after removing NaNs/Infs in covariates/TP."
        )

    # ---- Apply mask ----
    y = y_full[keep]  # (T, U)
    traj_label = traj_label_full[keep]  # (T,)
    tp = tp_full[keep]  # (T,)

    cov_k = {k: np.asarray(v[keep], dtype=float) for k, v in cov.items()}

    # ---- Motor design matrix ----
    motor_blocks: List[np.ndarray] = []
    motor_names: List[str] = []

    for name in ("speed", "accel", "hd_vel", "abs_hd_vel"):
        Xv, _ = bspline_features(
            cov_k[name],
            n_basis=motor_spline_k,
            order=motor_spline_order,
            bounds=None,
        )
        motor_blocks.append(Xv)
        motor_names.extend([f"{name}_bs{i}" for i in range(Xv.shape[1])])

    motor_blocks.append(cov_k["sin_hd"][:, None])
    motor_names.append("sin_hd")
    motor_blocks.append(cov_k["cos_hd"][:, None])
    motor_names.append("cos_hd")

    X_motor = np.concatenate(motor_blocks, axis=1)
    P_motor = X_motor.shape[1]

    # ---- Trajectory-GROUP dummy (1 dummy; group0 is reference) ----
    # Groups are defined by TP_GROUPS:
    #   group0 = TP_GROUPS[0] (LC_CR)
    #   group1 = TP_GROUPS[1] (RC_CL)
    n_traj = len(TRAJECTORY_TYPES)

    traj_to_idx = {name: i for i, name in enumerate(TRAJECTORY_TYPES)}
    traj_idx_to_group = np.full(n_traj, -1, dtype=int)

    for gi, (gname, (ta, tb)) in enumerate(TP_GROUPS):
        traj_idx_to_group[traj_to_idx[ta]] = gi
        traj_idx_to_group[traj_to_idx[tb]] = gi

    if np.any(traj_idx_to_group < 0):
        missing = [TRAJECTORY_TYPES[i] for i in np.where(traj_idx_to_group < 0)[0]]
        raise ValueError(f"Some trajectories are not assigned to a group: {missing}")

    group_label = traj_idx_to_group[traj_label]  # (T,) values 0 or 1

    # one dummy column for group1 ("RC_CL"), group0 is absorbed by the intercept
    X_traj = (group_label == 1).astype(float)[:, None]  # (T,1)
    traj_names = [f"is_{TP_GROUPS[1][0]}"]  # e.g. "is_RC_CL"
    P_traj = X_traj.shape[1]  # =1

    # ---- Task progression basis over [0,1], gated by TP_GROUPS (2 groups) ----
    tp_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(tp_lo, tp_hi),  # (0,1)
    )
    B_tp = np.asarray(tp_basis.compute_features(tp), dtype=float)  # (T, Ktp)
    if not np.all(np.isfinite(B_tp)):
        raise ValueError("Non-finite values in B_tp; TP out of bounds?")

    Ktp = B_tp.shape[1]

    tp_blocks: List[np.ndarray] = []
    tp_names: List[str] = []

    # Use the SAME traj_to_idx and TP_GROUPS used for group_label
    for gi, (gname, (ta, tb)) in enumerate(TP_GROUPS):
        idxs = [traj_to_idx[ta], traj_to_idx[tb]]
        gate = np.isin(traj_label, idxs).astype(float)[:, None]  # (T,1)
        tp_blocks.append(B_tp * gate)
        tp_names.extend([f"tp_{gname}_bs{i}" for i in range(Ktp)])

    X_tp = np.concatenate(tp_blocks, axis=1)  # (T, 2*Ktp)
    P_tp = X_tp.shape[1]

    if np.allclose(X_tp, 0.0):
        raise RuntimeError("X_tp is all zeros. Check traj_label or TP_GROUPS gating.")

    # ---- Design matrices ----
    X0 = np.concatenate([X_motor, X_traj], axis=1)
    X1 = np.concatenate([X_motor, X_traj, X_tp], axis=1)

    def _check_finite(name, X):
        if not np.all(np.isfinite(X)):
            bad = np.where(~np.isfinite(X))
            r0, c0 = int(bad[0][0]), int(bad[1][0])
            raise ValueError(
                f"{name} has non-finite values. Example: ({r0},{c0})={X[r0,c0]}"
            )

    _check_finite("X_motor", X_motor)
    _check_finite("X_traj", X_traj)
    _check_finite("X_tp", X_tp)
    _check_finite("X0", X0)
    _check_finite("X1", X1)

    # ---- Cross-validation ----
    folds = stratified_contiguous_folds(traj_label, n_folds=n_folds, seed=seed)

    # --- diagnostics: how many test bins per trajectory per fold? ---
    fold_test_counts = np.zeros((n_folds, n_traj), dtype=int)
    for i, (_, te) in enumerate(folds):
        for k in range(n_traj):
            fold_test_counts[i, k] = int(np.sum(traj_label[te] == k))

    # Optional: warn if any fold is missing any trajectory type
    import warnings

    for k, traj in enumerate(TRAJECTORY_TYPES):
        if fold_test_counts[:, k].min() == 0:
            warnings.warn(
                f"CV: trajectory '{traj}' has 0 test bins in at least one fold. "
                "Reduce n_folds or ensure enough bins for that trajectory."
            )

    # Optional: check “balanced-ness” span per trajectory (should be 0 or 1 in most cases)
    span = fold_test_counts.max(axis=0) - fold_test_counts.min(axis=0)
    if np.any(span > 1):
        warnings.warn(
            f"CV: fold test counts span > 1 for some trajectories: span={span}. "
            "This is unexpected with array_split; double-check labels."
        )
    agg_sum_per_neuron = lambda arr: jnp.sum(arr, axis=0)

    ll0_sum = np.zeros(U, dtype=float)
    ll1_sum = np.zeros(U, dtype=float)
    spk_sum = np.zeros(U, dtype=float)

    ll0_sum_by_traj = {traj: np.zeros(U, dtype=float) for traj in TRAJECTORY_TYPES}
    ll1_sum_by_traj = {traj: np.zeros(U, dtype=float) for traj in TRAJECTORY_TYPES}
    spk_sum_by_traj = {traj: np.zeros(U, dtype=float) for traj in TRAJECTORY_TYPES}

    for tr, te in folds:
        m0 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m1 = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        m0.fit(X0[tr], y[tr])
        m1.fit(X1[tr], y[tr])

        ll0 = np.asarray(
            m0.score(
                X0[te],
                y[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        ll1 = np.asarray(
            m1.score(
                X1[te],
                y[te],
                score_type="log-likelihood",
                aggregate_sample_scores=agg_sum_per_neuron,
            )
        )
        spk = np.asarray(y[te].sum(axis=0), dtype=float)

        ll0_sum += ll0
        ll1_sum += ll1
        spk_sum += spk

        for k, traj in enumerate(TRAJECTORY_TYPES):
            te_k = te[traj_label[te] == k]
            if te_k.size == 0:
                continue
            ll0_k = np.asarray(
                m0.score(
                    X0[te_k],
                    y[te_k],
                    score_type="log-likelihood",
                    aggregate_sample_scores=agg_sum_per_neuron,
                )
            )
            ll1_k = np.asarray(
                m1.score(
                    X1[te_k],
                    y[te_k],
                    score_type="log-likelihood",
                    aggregate_sample_scores=agg_sum_per_neuron,
                )
            )
            spk_k = np.asarray(y[te_k].sum(axis=0), dtype=float)

            ll0_sum_by_traj[traj] += ll0_k
            ll1_sum_by_traj[traj] += ll1_k
            spk_sum_by_traj[traj] += spk_k

    inv_log2 = 1.0 / np.log(2.0)

    def _per_spike(ll_sum: np.ndarray, spk_sum_: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(spk_sum_ > 0, ll_sum / spk_sum_, np.nan)

    ll0_ps = _per_spike(ll0_sum, spk_sum)
    ll1_ps = _per_spike(ll1_sum, spk_sum)

    cv_pooled = {
        "spike_sum": spk_sum,
        "ll_motor_bits_per_spike": ll0_ps * inv_log2,
        "ll_motor_tp_bits_per_spike": ll1_ps * inv_log2,
        "dll_bits_per_spike": (ll1_ps - ll0_ps) * inv_log2,
    }

    cv_by_traj: Dict[str, Dict[str, np.ndarray]] = {}
    for traj in TRAJECTORY_TYPES:
        ll0_t = ll0_sum_by_traj[traj]
        ll1_t = ll1_sum_by_traj[traj]
        spk_t = spk_sum_by_traj[traj]
        ll0_ps_t = _per_spike(ll0_t, spk_t)
        ll1_ps_t = _per_spike(ll1_t, spk_t)
        cv_by_traj[traj] = {
            "spike_sum": spk_t,
            "ll_motor_bits_per_spike": ll0_ps_t * inv_log2,
            "ll_motor_tp_bits_per_spike": ll1_ps_t * inv_log2,
            "dll_bits_per_spike": (ll1_ps_t - ll0_ps_t) * inv_log2,
        }

    # ---- Refit on all data for coefficients ----
    m0_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m1_all = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
    m0_all.fit(X0, y)
    m1_all.fit(X1, y)

    coef0 = _coef_feat_by_unit(m0_all, n_features=X0.shape[1])
    coef1 = _coef_feat_by_unit(m1_all, n_features=X1.shape[1])
    intercept0 = np.asarray(m0_all.intercept_).reshape(-1)
    intercept1 = np.asarray(m1_all.intercept_).reshape(-1)

    coef0_motor = coef0[:P_motor, :]
    coef0_traj = coef0[P_motor : P_motor + P_traj, :]

    coef1_motor = coef1[:P_motor, :]
    coef1_traj = coef1[P_motor : P_motor + P_traj, :]
    coef1_tp = coef1[P_motor + P_traj :, :]

    coef1_tp = coef1[P_motor + P_traj :, :]  # (2*Ktp, U)

    coef1_tp_by_group: Dict[str, np.ndarray] = {}
    for gi, (gname, _) in enumerate(TP_GROUPS):
        coef1_tp_by_group[gname] = coef1_tp[gi * Ktp : (gi + 1) * Ktp, :]

    coef1_tp_by_traj: Dict[str, np.ndarray] = {}
    for gi, (gname, (ta, tb)) in enumerate(TP_GROUPS):
        coef1_tp_by_traj[ta] = coef1_tp_by_group[gname]
        coef1_tp_by_traj[tb] = coef1_tp_by_group[gname]

    # ---- TP rate curves (Hz) from motor+tp at mean motor covariates ----
    motor_mean = X_motor.mean(axis=0)
    if not np.all(np.isfinite(motor_mean)):
        raise RuntimeError("motor_mean is non-finite; X_motor contains NaNs/Infs.")
    tp_curves_hz: Dict[str, Dict[str, np.ndarray]] = {}

    for k, traj in enumerate(TRAJECTORY_TYPES):
        grid = np.linspace(0.0, 1.0, 200)
        Bgrid = np.asarray(tp_basis.compute_features(grid), dtype=float)

        gi = traj_idx_to_group[k]
        traj_dummy = np.zeros((grid.size, P_traj), dtype=float)
        if gi == 1:
            traj_dummy[:, 0] = 1.0

        tp_block = np.zeros((grid.size, P_tp), dtype=float)
        tp_block[:, gi * Ktp : (gi + 1) * Ktp] = Bgrid

        Xgrid = np.concatenate(
            [np.repeat(motor_mean[None, :], grid.size, axis=0), traj_dummy, tp_block],
            axis=1,
        )
        eta = intercept1[None, :] + Xgrid @ coef1
        rate_hz = np.exp(eta) / float(bin_size_s)
        tp_curves_hz[traj] = {"tp_grid": grid, "rate_hz": rate_hz}

    return {
        "unit_ids": unit_ids,
        "bin_size_s": float(bin_size_s),
        "trajectory_types": TRAJECTORY_TYPES,
        "feature_names_motor": motor_names,
        "feature_names_traj": traj_names,
        "feature_names_tp": tp_names,
        "tp_basis": {
            "n_splines": int(tp_spline_k),
            "order": int(tp_spline_order),
            "bounds": (tp_lo, tp_hi),
        },
        "cv": {
            "pooled": cv_pooled,
            "by_traj": cv_by_traj,
            "fold_test_counts": fold_test_counts,
        },
        "coef": {
            "motor_only": {
                "intercept": intercept0,
                "coef_motor": coef0_motor,
                "coef_traj": coef0_traj,
            },
            "motor_tp": {
                "intercept": intercept1,
                "coef_motor": coef1_motor,
                "coef_traj": coef1_traj,
                "coef_tp_by_traj": coef1_tp_by_traj,
                "coef_tp_by_group": coef1_tp_by_group,
                "coef_tp_by_traj": coef1_tp_by_traj,
            },
        },
        "tp_rate_curves_hz": tp_curves_hz,
    }


def main():
    animal_name = "L14"
    date = "20240611"
    analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

    num_sleep_epochs = 5
    num_run_epochs = 4
    _, run_epoch_list = kyutils.get_epoch_list(
        num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
    )

    region = "v1"
    dark_epoch = run_epoch_list[3]  # your convention

    # ---------- load pickles ----------
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

    # ---------- spikes ----------
    sorting = si.load(analysis_path / f"sorting_{region}")
    spikes = get_tsgroup(sorting, timestamps_ephys_all_ptp)

    # ---------- build nap objects for THIS epoch ----------
    position_offset = 10
    bin_size_s = 0.02
    speed_threshold = 4.0

    # timebase for position/body position (after offset)
    tpos_full = np.asarray(timestamps_position_dict[dark_epoch])
    tpos = tpos_full[position_offset:]

    pos_xy_full = _as_xy(position_dict[dark_epoch])
    body_xy_full = _as_xy(body_position_dict[dark_epoch])

    pos_xy = pos_xy_full[position_offset:]
    body_xy = body_xy_full[position_offset:]

    # TsdFrames
    position_tsd = nap.TsdFrame(t=tpos, d=pos_xy, columns=["x", "y"])
    body_position_tsd = nap.TsdFrame(t=tpos, d=body_xy, columns=["x", "y"])

    # run epoch interval (after offset)
    run_ep = nap.IntervalSet(start=float(tpos[0]), end=float(tpos[-1]))

    # movement interval
    sp = pt.get_speed(
        position=pos_xy,
        time=tpos,
        sampling_frequency=_sampling_rate(tpos),
        sigma=0.1,
    )
    speed_tsd = nap.Tsd(t=tpos, d=sp)
    movement_ep = speed_tsd.threshold(speed_threshold, method="above").time_support
    movement_ep = movement_ep.intersect(run_ep)

    # trajectory intervals (restricted to run_ep)
    trajectory_eps = {}
    for traj in TRAJECTORY_TYPES:
        starts = trajectory_times[dark_epoch][traj][:, 0]
        ends = trajectory_times[dark_epoch][traj][:, -1]
        trajectory_eps[traj] = nap.IntervalSet(start=starts, end=ends).intersect(run_ep)

    fr_during_movement = spikes.restrict(movement_ep).rate.to_numpy()
    fr_threshold = 0.5
    unit_mask = fr_during_movement > fr_threshold

    # ---------- fit ----------
    res = fit_motor_vs_task_progression_wtrack_epoch(
        spikes=spikes,
        position_tsd=position_tsd,
        body_position_tsd=body_position_tsd,
        trajectory_eps=trajectory_eps,
        movement_ep=movement_ep,
        bin_size_s=bin_size_s,
        ridge=1e-3,
        n_folds=5,
        seed=0,
        motor_spline_k=8,
        motor_spline_order=4,
        tp_spline_k=25,
        tp_spline_order=4,
        tp_bounds=(0.0, 1.0),
        unit_mask=unit_mask,
    )

    # ---------- save ----------
    out_dir = analysis_path / "glm_motor_vs_tp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{region}_{dark_epoch}_motor_vs_tp.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(res, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
