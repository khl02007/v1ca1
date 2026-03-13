from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import pynapple as nap
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import scipy
import position_tools as pt
import spikeinterface.full as si
import kyutils
import pandas as pd
import track_linearization as tl
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d

SegmentEndpoints = Sequence[Tuple[float, float]]  # [(0, pt1), (pt1, pt2), (pt2, 1)]


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

encoding_decoding = {
    "center_to_left": "right_to_center",
    "right_to_center": "center_to_left",
    "center_to_right": "left_to_center",
    "left_to_center": "center_to_right",
}


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


def smooth_pf_along_position(pf, pos_dim: str, sigma_bins: float):
    pf = pf.fillna(0)  # Replaces NaN values with 0
    # gaussian_filter1d works on numpy; apply along the position axis
    axis = pf.get_axis_num(pos_dim)
    sm = gaussian_filter1d(pf.values, sigma=sigma_bins, axis=axis, mode="nearest")
    return pf.copy(data=sm)


spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])


all_ep = {}
trajectory_ep = {}

for epoch in run_epoch_list:
    all_ep[epoch] = nap.IntervalSet(
        start=timestamps_position_dict[epoch][position_offset],
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
gap = 20

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


def get_task_progression_tuning_curve_by_trajectory(
    task_progression_by_trajectory, task_progression_by_trajectory_bins
):
    tpf_by_trajectory = {}
    tpf_by_trajectory_smoothed = {}
    for region in regions:
        tpf_by_trajectory[region] = {}
        tpf_by_trajectory_smoothed[region] = {}
        for epoch in run_epoch_list:
            tpf_by_trajectory[region][epoch] = {}
            tpf_by_trajectory_smoothed[region][epoch] = {}
            for trajectory_type in trajectory_types:
                tpf_by_trajectory[region][epoch][trajectory_type] = (
                    nap.compute_tuning_curves(
                        data=spikes[region],
                        features=task_progression_by_trajectory[epoch][trajectory_type],
                        bins=[
                            task_progression_by_trajectory_bins
                        ],  # Use standardized bins
                        epochs=movement[epoch].time_support,
                        feature_names=["tp"],
                    )
                )
                tpf_by_trajectory_smoothed[region][epoch][trajectory_type] = (
                    smooth_pf_along_position(
                        tpf_by_trajectory[region][epoch][trajectory_type],
                        pos_dim="tp",
                        sigma_bins=1.0,
                    )
                )
    return tpf_by_trajectory, tpf_by_trajectory_smoothed


def get_task_progression_tuning_curve_by_trajectory_even_odd(
    task_progression_by_trajectory, task_progression_by_trajectory_bins
):
    tpf_by_trajectory_even = {}
    tpf_by_trajectory_odd = {}
    tpf_by_trajectory_even_smoothed = {}
    tpf_by_trajectory_odd_smoothed = {}
    for region in regions:
        tpf_by_trajectory_even[region] = {}
        tpf_by_trajectory_odd[region] = {}
        tpf_by_trajectory_even_smoothed[region] = {}
        tpf_by_trajectory_odd_smoothed[region] = {}

        for epoch in run_epoch_list:
            tpf_by_trajectory_even[region][epoch] = {}
            tpf_by_trajectory_odd[region][epoch] = {}
            tpf_by_trajectory_even_smoothed[region][epoch] = {}
            tpf_by_trajectory_odd_smoothed[region][epoch] = {}

            for trajectory_type in trajectory_types:
                tpf_by_trajectory_even[region][epoch][trajectory_type] = (
                    nap.compute_tuning_curves(
                        data=spikes[region],
                        features=task_progression_by_trajectory[epoch][trajectory_type],
                        bins=[
                            task_progression_by_trajectory_bins
                        ],  # Use standardized bins
                        epochs=trajectory_ep[epoch][trajectory_type][::2].intersect(
                            movement[epoch].time_support
                        ),
                        feature_names=["tp"],
                    )
                )
                tpf_by_trajectory_even_smoothed[region][epoch][trajectory_type] = (
                    smooth_pf_along_position(
                        tpf_by_trajectory_even[region][epoch][trajectory_type],
                        pos_dim="tp",
                        sigma_bins=1.0,
                    )
                )

                tpf_by_trajectory_odd[region][epoch][trajectory_type] = (
                    nap.compute_tuning_curves(
                        data=spikes[region],
                        features=task_progression_by_trajectory[epoch][trajectory_type],
                        bins=[
                            task_progression_by_trajectory_bins
                        ],  # Use standardized bins
                        epochs=trajectory_ep[epoch][trajectory_type][1::2].intersect(
                            movement[epoch].time_support
                        ),
                        feature_names=["tp"],
                    )
                )
                tpf_by_trajectory_odd_smoothed[region][epoch][trajectory_type] = (
                    smooth_pf_along_position(
                        tpf_by_trajectory_odd[region][epoch][trajectory_type],
                        pos_dim="tp",
                        sigma_bins=1.0,
                    )
                )

    return tpf_by_trajectory_even_smoothed, tpf_by_trajectory_odd_smoothed


def _get_global_bin_edges(tc: xr.DataArray) -> np.ndarray:
    """Get the single global 1D bin edge array from tc.attrs['bin_edges'] (list length 1)."""
    edges_list = tc.attrs.get("bin_edges", None)
    if not isinstance(edges_list, (list, tuple)) or len(edges_list) != 1:
        raise ValueError(
            "Expected tc.attrs['bin_edges'] to be a list/tuple of length 1."
        )
    edges = np.asarray(edges_list[0], dtype=float).squeeze()
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("bin_edges[0] must be a finite 1D array with >= 2 entries.")
    return edges


def _occupancy_1d(tc: xr.DataArray, n_tp: int) -> Optional[np.ndarray]:
    """
    Return occupancy weights as 1D array length n_tp if available; else None.

    Supports:
    - occupancy as 1D array length n_tp
    - occupancy as list/tuple length 1 containing a 1D array length n_tp
    """
    occ = tc.attrs.get("occupancy", None)
    if occ is None:
        return None

    if isinstance(occ, (list, tuple)):
        if len(occ) != 1:
            return None
        w = np.asarray(occ[0], dtype=float).squeeze()
    else:
        w = np.asarray(occ, dtype=float).squeeze()

    if w.ndim == 1 and w.size == n_tp and np.all(np.isfinite(w)):
        return w
    return None


def _choose_weights(
    y_da: xr.DataArray,
    x_da: xr.DataArray,
    *,
    use_occupancy_weights: bool,
    tp_dim: str,
) -> np.ndarray:
    """
    Choose per-bin weights for fitting/evaluation.

    If occupancy is present on y and/or x, uses:
      w = min(occ_y, occ_x) when both available
      w = occ_y or occ_x when only one available
    otherwise uniform weights.
    """
    n_tp = y_da.sizes[tp_dim]
    w = np.ones(n_tp, dtype=float)
    if not use_occupancy_weights:
        return w

    wy = _occupancy_1d(y_da, n_tp=n_tp)
    wx = _occupancy_1d(x_da, n_tp=n_tp)

    if wy is not None and wx is not None:
        return np.minimum(wy, wx)
    if wy is not None:
        return wy
    if wx is not None:
        return wx
    return w


def _align_and_validate(
    pf_y_fit: xr.DataArray,
    pf_x: xr.DataArray,
    pf_y_cv: Optional[xr.DataArray],
    *,
    tp_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Align arrays on ('unit', tp_dim) and validate bin edges.

    Returns
    -------
    y_fit, x, y_eval : xr.DataArray
        y_eval is pf_y_cv if provided, else y_fit.
    """
    if set(pf_y_fit.dims) != {"unit", tp_dim} or set(pf_x.dims) != {"unit", tp_dim}:
        raise ValueError(f"pf_y and pf_x must have dims exactly ('unit','{tp_dim}').")

    if pf_y_cv is not None and set(pf_y_cv.dims) != {"unit", tp_dim}:
        raise ValueError(f"pf_y_cv must have dims exactly ('unit','{tp_dim}').")

    # Align fit first
    y_fit, x = xr.align(pf_y_fit, pf_x, join="inner")

    # Align eval (if given) onto the same coordinate system
    if pf_y_cv is None:
        y_eval = y_fit
    else:
        y_eval, x, y_fit = xr.align(pf_y_cv, x, y_fit, join="inner")

    # Validate identical binning (edges)
    edges_x = _get_global_bin_edges(x)
    edges_y_fit = _get_global_bin_edges(y_fit)
    if edges_x.size != edges_y_fit.size or not np.allclose(
        edges_x, edges_y_fit, equal_nan=False
    ):
        raise ValueError(
            "bin_edges differ between pf_y (fit) and pf_x; align/bin them first."
        )

    edges_y_eval = _get_global_bin_edges(y_eval)
    if edges_x.size != edges_y_eval.size or not np.allclose(
        edges_x, edges_y_eval, equal_nan=False
    ):
        raise ValueError(
            "bin_edges differ between pf_y_cv and pf_x; align/bin them first."
        )

    return y_fit, x, y_eval


def _segment_masks_from_endpoints_using_centers(
    tc: xr.DataArray,
    segment_endpoints: SegmentEndpoints,
    *,
    tp_dim: str = "tp",
) -> list[np.ndarray]:
    """
    Boolean masks over tp bins using bin centers derived from global bin_edges.

    Segment k is [lo, hi) for k < last, and [lo, hi] for last.
    """
    if not isinstance(segment_endpoints, (list, tuple)) or len(segment_endpoints) < 1:
        raise ValueError(
            "segment_endpoints must be a non-empty sequence of (lo, hi) pairs."
        )

    edges = _get_global_bin_edges(tc)
    n_tp = tc.sizes[tp_dim]
    if edges.size != n_tp + 1:
        raise ValueError(f"bin_edges length must be n_tp+1={n_tp+1}, got {edges.size}.")

    centers = 0.5 * (edges[:-1] + edges[1:])

    masks: list[np.ndarray] = []
    for k, (lo, hi) in enumerate(segment_endpoints):
        lo_f, hi_f = float(lo), float(hi)
        if not (hi_f > lo_f):
            raise ValueError(
                f"segment_endpoints[{k}] must satisfy hi > lo; got ({lo}, {hi})."
            )
        if k < len(segment_endpoints) - 1:
            m = (centers >= lo_f) & (centers < hi_f)
        else:
            m = (centers >= lo_f) & (centers <= hi_f)
        masks.append(m)

    return masks


def _weighted_r2_rmse(
    y: np.ndarray,  # (n_unit, n_bin)
    yhat: np.ndarray,  # (n_unit, n_bin)
    w: np.ndarray,  # (n_bin,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted R^2 and RMSE per unit, plus n_eff (# valid bins).

    Notes
    -----
    Valid bins require finite y, finite yhat, and w>0.
    """
    valid = np.isfinite(y) & np.isfinite(yhat) & (w[None, :] > 0)
    w_eff = np.where(valid, w[None, :], 0.0)

    sumw = w_eff.sum(axis=1)
    n_eff = (w_eff > 0).sum(axis=1).astype(int)

    r2 = np.full(y.shape[0], np.nan)
    rmse = np.full(y.shape[0], np.nan)

    ok = sumw > 0
    if not np.any(ok):
        return r2, rmse, n_eff

    ybar = np.full(y.shape[0], np.nan)
    ybar[ok] = (w_eff[ok] * y[ok]).sum(axis=1) / sumw[ok]

    sse = (w_eff * (y - yhat) ** 2).sum(axis=1)
    sst = (w_eff * (y - ybar[:, None]) ** 2).sum(axis=1)

    ok2 = ok & (sst > 0)
    r2[ok2] = 1.0 - (sse[ok2] / sst[ok2])
    rmse[ok] = np.sqrt(sse[ok] / sumw[ok])
    return r2, rmse, n_eff


def _weighted_gain_fit_params(
    x: np.ndarray,  # (n_unit, n_bin)
    y_fit: np.ndarray,  # (n_unit, n_bin)
    w_fit: np.ndarray,  # (n_bin,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted gain-only fit per unit: y_fit ~= a*x (no intercept).

    Returns
    -------
    a : np.ndarray
        Shape (n_unit,).
    n_eff_fit : np.ndarray
        Shape (n_unit,). Number of bins used in fitting.
    """
    valid = np.isfinite(x) & np.isfinite(y_fit) & (w_fit[None, :] > 0)
    w_eff = np.where(valid, w_fit[None, :], 0.0)

    n_eff_fit = (w_eff > 0).sum(axis=1).astype(int)

    num = (w_eff * x * y_fit).sum(axis=1)
    den = (w_eff * x * x).sum(axis=1)

    a = np.full(x.shape[0], np.nan)
    ok = den > 0
    a[ok] = num[ok] / den[ok]
    return a, n_eff_fit


def _weighted_affine_fit_params(
    x: np.ndarray,  # (n_unit, n_bin)
    y_fit: np.ndarray,  # (n_unit, n_bin)
    w_fit: np.ndarray,  # (n_bin,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted affine fit per unit: y_fit ~= a*x + b.

    Returns
    -------
    a, b : np.ndarray
        Shape (n_unit,).
    n_eff_fit : np.ndarray
        Shape (n_unit,). Number of bins used in fitting.
    """
    valid = np.isfinite(x) & np.isfinite(y_fit) & (w_fit[None, :] > 0)
    w_eff = np.where(valid, w_fit[None, :], 0.0)

    sumw = w_eff.sum(axis=1)
    n_eff_fit = (w_eff > 0).sum(axis=1).astype(int)

    safe = sumw > 0
    xbar = np.full(x.shape[0], np.nan)
    ybar = np.full(y_fit.shape[0], np.nan)

    xbar[safe] = (w_eff[safe] * x[safe]).sum(axis=1) / sumw[safe]
    ybar[safe] = (w_eff[safe] * y_fit[safe]).sum(axis=1) / sumw[safe]

    xc = x - xbar[:, None]
    yc = y_fit - ybar[:, None]

    denom = (w_eff * xc**2).sum(axis=1)
    numer = (w_eff * xc * yc).sum(axis=1)

    a = np.full(x.shape[0], np.nan)
    ok = safe & (denom > 0)
    a[ok] = numer[ok] / denom[ok]

    b = ybar - a * xbar
    return a, b, n_eff_fit


# ---------------------------------------------------------------------
# 1) slope=1 global intercept: y ~= x + b
# ---------------------------------------------------------------------
def fit_slope1_global_intercept_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit per unit: y_fit ~= 1*x + b (single global b), i.e. yhat = x + b.

    If pf_y_cv is provided, r2/rmse are computed vs pf_y_cv (held-out).
    Training metrics are also returned as r2_fit/rmse_fit.

    Returns
    -------
    pd.DataFrame
        Columns:
        unit, a, b,
        r2, rmse, n_eff,
        r2_fit, rmse_fit, n_eff_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    n_tp = y_fit_da.sizes[tp_dim]

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)
    unit_ids = np.asarray(y_fit_da["unit"].values)

    # b = weighted mean of residual (y_fit - x) using fit weights
    valid_fit = np.isfinite(x) & np.isfinite(y_fit) & (w_fit[None, :] > 0)
    w_eff = np.where(valid_fit, w_fit[None, :], 0.0)
    sumw = w_eff.sum(axis=1)

    b = np.full(x.shape[0], np.nan)
    ok = sumw > 0
    b[ok] = (w_eff[ok] * (y_fit[ok] - x[ok])).sum(axis=1) / sumw[ok]

    yhat = x + b[:, None]

    # Eval metrics (held-out if provided)
    r2, rmse, n_eff = _weighted_r2_rmse(y=y_eval, yhat=yhat, w=w_eval)
    # Fit metrics
    r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(y=y_fit, yhat=yhat, w=w_fit)

    return pd.DataFrame(
        {
            "unit": unit_ids,
            "a": 1.0,
            "b": b,
            "r2": r2,
            "rmse": rmse,
            "n_eff": n_eff,
            "r2_fit": r2_fit,
            "rmse_fit": rmse_fit,
            "n_eff_fit": n_eff_fit,
        }
    )


# ---------------------------------------------------------------------
# 2) slope=1, segment intercepts: y ~= x + b_s
# ---------------------------------------------------------------------
def fit_slope1_segment_intercepts_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    segment_endpoints: SegmentEndpoints,
    segment_labels: Optional[Sequence[str]] = None,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit per unit: y_fit ~= 1*x + b_s (segment-wise b), i.e. yhat = x + b_s within each segment.

    If pf_y_cv is provided, r2/rmse/total_r2 are computed vs pf_y_cv (held-out).
    Training metrics are also returned as r2_fit/rmse_fit/total_r2_fit.

    Returns
    -------
    pd.DataFrame
        One row per (unit, segment) with columns:
        unit, segment, a, b,
        r2, rmse, n_eff, total_r2,
        r2_fit, rmse_fit, n_eff_fit, total_r2_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    n_unit = y_fit_da.sizes["unit"]
    n_tp = y_fit_da.sizes[tp_dim]
    unit_ids = np.asarray(y_fit_da["unit"].values)

    masks = _segment_masks_from_endpoints_using_centers(
        y_fit_da, segment_endpoints, tp_dim=tp_dim
    )
    n_seg = len(masks)
    if segment_labels is None:
        segment_labels = [f"seg{i}" for i in range(n_seg)]
    if len(segment_labels) != n_seg:
        raise ValueError("segment_labels must match the number of segments.")

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)  # (unit,tp)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    resid_fit = y_fit - x  # fit residuals

    b = np.full((n_unit, n_seg), np.nan)
    yhat = np.full_like(y_fit, np.nan)

    finite_xy_fit = np.isfinite(x) & np.isfinite(y_fit)
    w_pos_fit = w_fit > 0

    # Fit b_s per segment using fit residuals + fit weights
    for j, m in enumerate(masks):
        valid = finite_xy_fit & (m[None, :]) & (w_pos_fit[None, :])
        w_eff = np.where(valid, w_fit[None, :], 0.0)
        sumw = w_eff.sum(axis=1)

        bj = np.full(n_unit, np.nan)
        ok = sumw > 0
        bj[ok] = (w_eff[ok] * resid_fit[ok]).sum(axis=1) / sumw[ok]

        b[:, j] = bj
        yhat[:, m] = x[:, m] + bj[:, None]

    cover = np.zeros(n_tp, dtype=bool)
    for m in masks:
        cover |= m

    # Total metrics
    total_r2, _, _ = _weighted_r2_rmse(
        y=y_eval[:, cover], yhat=yhat[:, cover], w=w_eval[cover]
    )
    total_r2_fit, _, _ = _weighted_r2_rmse(
        y=y_fit[:, cover], yhat=yhat[:, cover], w=w_fit[cover]
    )

    rows: list[dict[str, Any]] = []
    for j, (m, seg_label) in enumerate(zip(masks, segment_labels)):
        r2, rmse, n_eff = _weighted_r2_rmse(
            y=y_eval[:, m], yhat=yhat[:, m], w=w_eval[m]
        )
        r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(
            y=y_fit[:, m], yhat=yhat[:, m], w=w_fit[m]
        )

        for i in range(n_unit):
            rows.append(
                dict(
                    unit=unit_ids[i],
                    segment=seg_label,
                    a=1.0,
                    b=float(b[i, j]) if np.isfinite(b[i, j]) else np.nan,
                    r2=float(r2[i]) if np.isfinite(r2[i]) else np.nan,
                    rmse=float(rmse[i]) if np.isfinite(rmse[i]) else np.nan,
                    n_eff=int(n_eff[i]),
                    total_r2=float(total_r2[i]) if np.isfinite(total_r2[i]) else np.nan,
                    r2_fit=float(r2_fit[i]) if np.isfinite(r2_fit[i]) else np.nan,
                    rmse_fit=float(rmse_fit[i]) if np.isfinite(rmse_fit[i]) else np.nan,
                    n_eff_fit=int(n_eff_fit[i]),
                    total_r2_fit=(
                        float(total_r2_fit[i])
                        if np.isfinite(total_r2_fit[i])
                        else np.nan
                    ),
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3) global gain-only: y ~= a*x
# ---------------------------------------------------------------------
def fit_global_gain_place_fields_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit one global gain-only transform per unit: y_fit ~= a * x.

    If pf_y_cv is provided, global_r2/global_rmse/global_n_eff are computed on held-out y.
    Training metrics are also returned as global_*_fit.

    Returns
    -------
    pd.DataFrame
        Columns:
        unit, global_a,
        global_r2, global_rmse, global_n_eff,
        global_r2_fit, global_rmse_fit, global_n_eff_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    unit_ids = np.asarray(y_fit_da["unit"].values)

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    a, n_eff_fit = _weighted_gain_fit_params(x=x, y_fit=y_fit, w_fit=w_fit)
    yhat = a[:, None] * x

    r2, rmse, n_eff = _weighted_r2_rmse(y=y_eval, yhat=yhat, w=w_eval)
    r2_fit, rmse_fit, n_eff_fit_metrics = _weighted_r2_rmse(y=y_fit, yhat=yhat, w=w_fit)

    # keep both "n_eff_fit" from fit-validity and the metric-validity (they often match)
    return pd.DataFrame(
        {
            "unit": unit_ids,
            "global_a": a,
            "global_r2": r2,
            "global_rmse": rmse,
            "global_n_eff": n_eff,
            "global_r2_fit": r2_fit,
            "global_rmse_fit": rmse_fit,
            "global_n_eff_fit": n_eff_fit_metrics,
            "global_n_eff_fit_params": n_eff_fit,
        }
    )


# ---------------------------------------------------------------------
# 4) piecewise gain-only: y ~= a_s*x per segment
# ---------------------------------------------------------------------
def fit_piecewise_gain_place_fields_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    segment_endpoints: SegmentEndpoints,
    segment_labels: Optional[Sequence[str]] = None,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit gain-only transforms per segment: y_fit ~= a_seg * x (no intercept).

    Metrics (r2/rmse/total_r2) are computed on held-out pf_y_cv if provided.
    Training metrics are also returned as *_fit.

    Returns
    -------
    pd.DataFrame
        One row per (unit, segment) with columns:
        unit, segment, a,
        r2, rmse, n_eff, total_r2,
        r2_fit, rmse_fit, n_eff_fit, total_r2_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    n_unit = y_fit_da.sizes["unit"]
    n_tp = y_fit_da.sizes[tp_dim]
    unit_ids = np.asarray(y_fit_da["unit"].values)

    masks = _segment_masks_from_endpoints_using_centers(
        y_fit_da, segment_endpoints, tp_dim=tp_dim
    )
    n_seg = len(masks)
    if segment_labels is None:
        segment_labels = [f"seg{i}" for i in range(n_seg)]
    if len(segment_labels) != n_seg:
        raise ValueError("segment_labels must match the number of segments.")

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    a_all = np.full((n_unit, n_seg), np.nan)
    rows: list[dict[str, Any]] = []

    for j, (mask, seg_label) in enumerate(zip(masks, segment_labels)):
        a_j, _ = _weighted_gain_fit_params(
            x=x[:, mask], y_fit=y_fit[:, mask], w_fit=w_fit[mask]
        )
        a_all[:, j] = a_j

        yhat_seg = a_j[:, None] * x[:, mask]

        r2, rmse, n_eff = _weighted_r2_rmse(
            y=y_eval[:, mask], yhat=yhat_seg, w=w_eval[mask]
        )
        r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(
            y=y_fit[:, mask], yhat=yhat_seg, w=w_fit[mask]
        )

        for i in range(n_unit):
            rows.append(
                dict(
                    unit=unit_ids[i],
                    segment=seg_label,
                    a=float(a_j[i]) if np.isfinite(a_j[i]) else np.nan,
                    r2=float(r2[i]) if np.isfinite(r2[i]) else np.nan,
                    rmse=float(rmse[i]) if np.isfinite(rmse[i]) else np.nan,
                    n_eff=int(n_eff[i]),
                    r2_fit=float(r2_fit[i]) if np.isfinite(r2_fit[i]) else np.nan,
                    rmse_fit=float(rmse_fit[i]) if np.isfinite(rmse_fit[i]) else np.nan,
                    n_eff_fit=int(n_eff_fit[i]),
                )
            )

    df = pd.DataFrame(rows)

    # Piecewise prediction across all bins for total_r2
    yhat = np.full_like(y_fit, np.nan)
    for j, mask in enumerate(masks):
        yhat[:, mask] = a_all[:, j, None] * x[:, mask]

    cover = np.zeros(n_tp, dtype=bool)
    for m in masks:
        cover |= m

    total_r2, _, _ = _weighted_r2_rmse(
        y=y_eval[:, cover], yhat=yhat[:, cover], w=w_eval[cover]
    )
    total_r2_fit, _, _ = _weighted_r2_rmse(
        y=y_fit[:, cover], yhat=yhat[:, cover], w=w_fit[cover]
    )

    total_df = pd.DataFrame(
        {"unit": unit_ids, "total_r2": total_r2, "total_r2_fit": total_r2_fit}
    )
    return df.merge(total_df, on="unit", how="left")


# ---------------------------------------------------------------------
# 5) global affine: y ~= a*x + b
# ---------------------------------------------------------------------
def fit_global_affine_place_fields_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit one global affine transform per unit: y_fit ~= a * x + b.

    Metrics are computed on held-out pf_y_cv if provided; training metrics returned as *_fit.

    Returns
    -------
    pd.DataFrame
        Columns:
        unit, global_a, global_b,
        global_r2, global_rmse, global_n_eff,
        global_r2_fit, global_rmse_fit, global_n_eff_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    unit_ids = np.asarray(y_fit_da["unit"].values)

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    a, b, _ = _weighted_affine_fit_params(x=x, y_fit=y_fit, w_fit=w_fit)
    yhat = a[:, None] * x + b[:, None]

    r2, rmse, n_eff = _weighted_r2_rmse(y=y_eval, yhat=yhat, w=w_eval)
    r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(y=y_fit, yhat=yhat, w=w_fit)

    return pd.DataFrame(
        {
            "unit": unit_ids,
            "global_a": a,
            "global_b": b,
            "global_r2": r2,
            "global_rmse": rmse,
            "global_n_eff": n_eff,
            "global_r2_fit": r2_fit,
            "global_rmse_fit": rmse_fit,
            "global_n_eff_fit": n_eff_fit,
        }
    )


# ---------------------------------------------------------------------
# 6) per-segment affine: y ~= a_s*x + b_s per segment + total_r2
# ---------------------------------------------------------------------
def fit_affine_place_fields_per_segment_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    segment_endpoints: SegmentEndpoints,
    segment_labels: Optional[Sequence[str]] = None,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit per-unit affine transforms per segment: y_fit ~= a_seg * x + b_seg.

    Per-segment metrics and total_r2 are computed on held-out pf_y_cv if provided.
    Training metrics returned as *_fit.

    Returns
    -------
    pd.DataFrame
        One row per (unit, segment) with columns:
        unit, segment, a, b,
        r2, rmse, n_eff,
        r2_fit, rmse_fit, n_eff_fit,
        total_r2, total_r2_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    n_unit = y_fit_da.sizes["unit"]
    n_tp = y_fit_da.sizes[tp_dim]
    unit_ids = np.asarray(y_fit_da["unit"].values)

    masks = _segment_masks_from_endpoints_using_centers(
        y_fit_da, segment_endpoints, tp_dim=tp_dim
    )
    n_seg = len(masks)
    if segment_labels is None:
        segment_labels = [f"seg{i}" for i in range(n_seg)]
    if len(segment_labels) != n_seg:
        raise ValueError("segment_labels must match the number of segments.")

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    a_all = np.full((n_unit, n_seg), np.nan)
    b_all = np.full((n_unit, n_seg), np.nan)

    rows: list[dict[str, Any]] = []
    for j, (mask, seg_label) in enumerate(zip(masks, segment_labels)):
        a_j, b_j, _ = _weighted_affine_fit_params(
            x=x[:, mask], y_fit=y_fit[:, mask], w_fit=w_fit[mask]
        )
        a_all[:, j] = a_j
        b_all[:, j] = b_j

        yhat_seg = a_j[:, None] * x[:, mask] + b_j[:, None]

        r2, rmse, n_eff = _weighted_r2_rmse(
            y=y_eval[:, mask], yhat=yhat_seg, w=w_eval[mask]
        )
        r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(
            y=y_fit[:, mask], yhat=yhat_seg, w=w_fit[mask]
        )

        for i in range(n_unit):
            rows.append(
                dict(
                    unit=unit_ids[i],
                    segment=seg_label,
                    a=float(a_j[i]) if np.isfinite(a_j[i]) else np.nan,
                    b=float(b_j[i]) if np.isfinite(b_j[i]) else np.nan,
                    r2=float(r2[i]) if np.isfinite(r2[i]) else np.nan,
                    rmse=float(rmse[i]) if np.isfinite(rmse[i]) else np.nan,
                    n_eff=int(n_eff[i]),
                    r2_fit=float(r2_fit[i]) if np.isfinite(r2_fit[i]) else np.nan,
                    rmse_fit=float(rmse_fit[i]) if np.isfinite(rmse_fit[i]) else np.nan,
                    n_eff_fit=int(n_eff_fit[i]),
                )
            )

    df = pd.DataFrame(rows)

    # Piecewise prediction across all bins for total_r2
    yhat = np.full_like(y_fit, np.nan)
    for j, mask in enumerate(masks):
        yhat[:, mask] = a_all[:, j, None] * x[:, mask] + b_all[:, j, None]

    cover = np.zeros(n_tp, dtype=bool)
    for m in masks:
        cover |= m

    total_r2, _, _ = _weighted_r2_rmse(
        y=y_eval[:, cover], yhat=yhat[:, cover], w=w_eval[cover]
    )
    total_r2_fit, _, _ = _weighted_r2_rmse(
        y=y_fit[:, cover], yhat=yhat[:, cover], w=w_fit[cover]
    )

    total_df = pd.DataFrame(
        {"unit": unit_ids, "total_r2": total_r2, "total_r2_fit": total_r2_fit}
    )
    return df.merge(total_df, on="unit", how="left")


# ---------------------------------------------------------------------
# 7) a_s per segment, shared b: y ~= a_s*x + b
# ---------------------------------------------------------------------
def fit_piecewise_gain_shared_intercept_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    segment_endpoints: SegmentEndpoints,
    segment_labels: Optional[Sequence[str]] = None,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit per unit: y_fit ~= a_s * x + b, where a_s differs by segment, but b is shared.

    Metrics are computed on held-out pf_y_cv if provided; training metrics returned as *_fit.

    Returns
    -------
    pd.DataFrame
        One row per (unit, segment) with:
        unit, segment, a, b,
        r2, rmse, n_eff, total_r2,
        r2_fit, rmse_fit, n_eff_fit, total_r2_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    n_unit = y_fit_da.sizes["unit"]
    n_tp = y_fit_da.sizes[tp_dim]
    unit_ids = np.asarray(y_fit_da["unit"].values)

    masks = _segment_masks_from_endpoints_using_centers(
        y_fit_da, segment_endpoints, tp_dim=tp_dim
    )
    n_seg = len(masks)

    if segment_labels is None:
        segment_labels = [f"seg{i}" for i in range(n_seg)]
    if len(segment_labels) != n_seg:
        raise ValueError("segment_labels must match number of segments.")

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)  # (unit,tp)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    finite_xy_fit = np.isfinite(x) & np.isfinite(y_fit)
    w_pos_fit = w_fit > 0

    # Segment-wise sufficient statistics for the fit
    Sw = np.zeros((n_unit, n_seg))
    Sx = np.zeros((n_unit, n_seg))
    Sy = np.zeros((n_unit, n_seg))
    Sxx = np.zeros((n_unit, n_seg))
    Sxy = np.zeros((n_unit, n_seg))

    for j, m in enumerate(masks):
        valid = finite_xy_fit & (m[None, :]) & (w_pos_fit[None, :])
        w_eff = np.where(valid, w_fit[None, :], 0.0)
        Sw[:, j] = w_eff.sum(axis=1)
        Sx[:, j] = (w_eff * x).sum(axis=1)
        Sy[:, j] = (w_eff * y_fit).sum(axis=1)
        Sxx[:, j] = (w_eff * x * x).sum(axis=1)
        Sxy[:, j] = (w_eff * x * y_fit).sum(axis=1)

    Sw_tot = Sw.sum(axis=1)
    Sy_tot = Sy.sum(axis=1)

    inv_Sxx = np.divide(1.0, Sxx, out=np.zeros_like(Sxx), where=Sxx > 0)
    A = np.sum(Sx * Sxy * inv_Sxx, axis=1)
    B = np.sum((Sx * Sx) * inv_Sxx, axis=1)

    denom = Sw_tot - B
    b = np.divide(Sy_tot - A, denom, out=np.full(n_unit, np.nan), where=denom > 0)

    a = np.divide(
        Sxy - b[:, None] * Sx, Sxx, out=np.full_like(Sxy, np.nan), where=Sxx > 0
    )

    # Piecewise predictions across all bins
    yhat = np.full_like(y_fit, np.nan)
    for j, m in enumerate(masks):
        yhat[:, m] = a[:, j, None] * x[:, m] + b[:, None]

    cover = np.zeros(n_tp, dtype=bool)
    for m in masks:
        cover |= m

    total_r2, _, _ = _weighted_r2_rmse(
        y=y_eval[:, cover], yhat=yhat[:, cover], w=w_eval[cover]
    )
    total_r2_fit, _, _ = _weighted_r2_rmse(
        y=y_fit[:, cover], yhat=yhat[:, cover], w=w_fit[cover]
    )

    rows: list[dict[str, Any]] = []
    for j, (m, seg_label) in enumerate(zip(masks, segment_labels)):
        r2, rmse, n_eff = _weighted_r2_rmse(
            y=y_eval[:, m], yhat=yhat[:, m], w=w_eval[m]
        )
        r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(
            y=y_fit[:, m], yhat=yhat[:, m], w=w_fit[m]
        )

        for i in range(n_unit):
            rows.append(
                dict(
                    unit=unit_ids[i],
                    segment=seg_label,
                    a=float(a[i, j]) if np.isfinite(a[i, j]) else np.nan,
                    b=float(b[i]) if np.isfinite(b[i]) else np.nan,
                    r2=float(r2[i]) if np.isfinite(r2[i]) else np.nan,
                    rmse=float(rmse[i]) if np.isfinite(rmse[i]) else np.nan,
                    n_eff=int(n_eff[i]),
                    total_r2=float(total_r2[i]) if np.isfinite(total_r2[i]) else np.nan,
                    r2_fit=float(r2_fit[i]) if np.isfinite(r2_fit[i]) else np.nan,
                    rmse_fit=float(rmse_fit[i]) if np.isfinite(rmse_fit[i]) else np.nan,
                    n_eff_fit=int(n_eff_fit[i]),
                    total_r2_fit=(
                        float(total_r2_fit[i])
                        if np.isfinite(total_r2_fit[i])
                        else np.nan
                    ),
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 8) shared a, segment intercepts: y ~= a*x + b_s
# ---------------------------------------------------------------------
def fit_shared_gain_segment_intercepts_df(
    pf_y: xr.DataArray,
    pf_x: xr.DataArray,
    *,
    segment_endpoints: SegmentEndpoints,
    segment_labels: Optional[Sequence[str]] = None,
    pf_y_cv: Optional[xr.DataArray] = None,
    use_occupancy_weights: bool = True,
    tp_dim: str = "tp",
) -> pd.DataFrame:
    """
    Fit per unit: y_fit ~= a * x + b_s, where a is shared across segments and b_s differs by segment.

    Metrics are computed on held-out pf_y_cv if provided; training metrics returned as *_fit.

    Returns
    -------
    pd.DataFrame
        One row per (unit, segment) with:
        unit, segment, a, b,
        r2, rmse, n_eff, total_r2,
        r2_fit, rmse_fit, n_eff_fit, total_r2_fit
    """
    y_fit_da, x_da, y_eval_da = _align_and_validate(pf_y, pf_x, pf_y_cv, tp_dim=tp_dim)
    n_unit = y_fit_da.sizes["unit"]
    n_tp = y_fit_da.sizes[tp_dim]
    unit_ids = np.asarray(y_fit_da["unit"].values)

    masks = _segment_masks_from_endpoints_using_centers(
        y_fit_da, segment_endpoints, tp_dim=tp_dim
    )
    n_seg = len(masks)

    if segment_labels is None:
        segment_labels = [f"seg{i}" for i in range(n_seg)]
    if len(segment_labels) != n_seg:
        raise ValueError("segment_labels must match number of segments.")

    w_fit = _choose_weights(
        y_fit_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )
    w_eval = _choose_weights(
        y_eval_da, x_da, use_occupancy_weights=use_occupancy_weights, tp_dim=tp_dim
    )

    x = np.asarray(x_da.values, dtype=float)  # (unit,tp)
    y_fit = np.asarray(y_fit_da.values, dtype=float)
    y_eval = np.asarray(y_eval_da.values, dtype=float)

    finite_xy_fit = np.isfinite(x) & np.isfinite(y_fit)
    w_pos_fit = w_fit > 0

    # Per-segment sufficient stats (fit)
    Sw = np.zeros((n_unit, n_seg))
    Sx = np.zeros((n_unit, n_seg))
    Sy = np.zeros((n_unit, n_seg))
    for j, m in enumerate(masks):
        valid = finite_xy_fit & (m[None, :]) & (w_pos_fit[None, :])
        w_eff = np.where(valid, w_fit[None, :], 0.0)
        Sw[:, j] = w_eff.sum(axis=1)
        Sx[:, j] = (w_eff * x).sum(axis=1)
        Sy[:, j] = (w_eff * y_fit).sum(axis=1)

    cover = np.zeros(n_tp, dtype=bool)
    for m in masks:
        cover |= m

    valid_all = finite_xy_fit & (cover[None, :]) & (w_pos_fit[None, :])
    w_all = np.where(valid_all, w_fit[None, :], 0.0)
    Sxx_tot = (w_all * x * x).sum(axis=1)
    Sxy_tot = (w_all * x * y_fit).sum(axis=1)

    inv_Sw = np.divide(1.0, Sw, out=np.zeros_like(Sw), where=Sw > 0)
    corr1 = np.sum(Sx * Sy * inv_Sw, axis=1)
    corr2 = np.sum((Sx * Sx) * inv_Sw, axis=1)

    denom = Sxx_tot - corr2
    a = np.divide(Sxy_tot - corr1, denom, out=np.full(n_unit, np.nan), where=denom > 0)

    b = np.divide(Sy - a[:, None] * Sx, Sw, out=np.full_like(Sy, np.nan), where=Sw > 0)

    # Piecewise predictions
    yhat = np.full_like(y_fit, np.nan)
    for j, m in enumerate(masks):
        yhat[:, m] = a[:, None] * x[:, m] + b[:, j, None]

    total_r2, _, _ = _weighted_r2_rmse(
        y=y_eval[:, cover], yhat=yhat[:, cover], w=w_eval[cover]
    )
    total_r2_fit, _, _ = _weighted_r2_rmse(
        y=y_fit[:, cover], yhat=yhat[:, cover], w=w_fit[cover]
    )

    rows: list[dict[str, Any]] = []
    for j, (m, seg_label) in enumerate(zip(masks, segment_labels)):
        r2, rmse, n_eff = _weighted_r2_rmse(
            y=y_eval[:, m], yhat=yhat[:, m], w=w_eval[m]
        )
        r2_fit, rmse_fit, n_eff_fit = _weighted_r2_rmse(
            y=y_fit[:, m], yhat=yhat[:, m], w=w_fit[m]
        )

        for i in range(n_unit):
            rows.append(
                dict(
                    unit=unit_ids[i],
                    segment=seg_label,
                    a=float(a[i]) if np.isfinite(a[i]) else np.nan,
                    b=float(b[i, j]) if np.isfinite(b[i, j]) else np.nan,
                    r2=float(r2[i]) if np.isfinite(r2[i]) else np.nan,
                    rmse=float(rmse[i]) if np.isfinite(rmse[i]) else np.nan,
                    n_eff=int(n_eff[i]),
                    total_r2=float(total_r2[i]) if np.isfinite(total_r2[i]) else np.nan,
                    r2_fit=float(r2_fit[i]) if np.isfinite(r2_fit[i]) else np.nan,
                    rmse_fit=float(rmse_fit[i]) if np.isfinite(rmse_fit[i]) else np.nan,
                    n_eff_fit=int(n_eff_fit[i]),
                    total_r2_fit=(
                        float(total_r2_fit[i])
                        if np.isfinite(total_r2_fit[i])
                        else np.nan
                    ),
                )
            )

    return pd.DataFrame(rows)


def main():

    place_bin_size = 4  # cm
    place_gap = 0  # cm

    task_progression_bin_size = place_bin_size / total_length_per_trajectory
    task_progression_gap = place_gap / total_length_per_trajectory

    task_progression_by_trajectory, task_progression_by_trajectory_bins = (
        get_task_progression_by_trajectory(
            task_progression_bin_size=task_progression_bin_size
        )
    )
    tpf_by_trajectory, tpf_by_trajectory_smoothed = (
        get_task_progression_tuning_curve_by_trajectory(
            task_progression_by_trajectory, task_progression_by_trajectory_bins
        )
    )

    tpf_by_trajectory_even_smoothed, tpf_by_trajectory_odd_smoothed = (
        get_task_progression_tuning_curve_by_trajectory_even_odd(
            task_progression_by_trajectory, task_progression_by_trajectory_bins
        )
    )

    region = "v1"

    epoch1 = run_epoch_list[0]
    epoch2 = run_epoch_list[3]

    trajectory_type = "center_to_left"

    pf_x = tpf_by_trajectory_smoothed[region][epoch2][trajectory_type]

    pf_y_fit = tpf_by_trajectory_odd_smoothed[region][epoch1][trajectory_type]
    pf_y_cv = tpf_by_trajectory_even_smoothed[region][epoch1][trajectory_type]

    segment_endpoints = [
        (
            0.0,
            (long_segment_length + diagonal_segment_length / 2)
            / total_length_per_trajectory,
        ),
        (
            (long_segment_length + diagonal_segment_length / 2)
            / total_length_per_trajectory,
            (
                long_segment_length
                + short_segment_length
                + diagonal_segment_length
                + diagonal_segment_length / 2
            )
            / total_length_per_trajectory,
        ),
        (
            (
                long_segment_length
                + short_segment_length
                + diagonal_segment_length
                + diagonal_segment_length / 2
            )
            / total_length_per_trajectory,
            1.0,
        ),
    ]
    segment_labels = ["S1", "S2", "S3"]

    # a_global, no b
    df_a_global_b_0 = fit_global_gain_place_fields_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
    )

    # a_segment, no b
    df_a_segment_b_0 = fit_piecewise_gain_place_fields_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
        segment_endpoints=segment_endpoints,
        segment_labels=segment_labels,
    )

    # a_1, b_global
    df_a_1_b_global = fit_slope1_global_intercept_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
    )

    # a_global, b_global
    df_a_global_b_global = fit_global_affine_place_fields_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
    )

    # a_segment, b_global
    df_a_segment_b_global = fit_piecewise_gain_shared_intercept_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
        segment_endpoints=segment_endpoints,
        segment_labels=segment_labels,
    )

    # a_1, b_segment
    df_a_1_b_segment = fit_slope1_segment_intercepts_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
        segment_endpoints=segment_endpoints,
        segment_labels=segment_labels,
    )

    # a_global, b_segment
    df_a_global_b_segment = fit_shared_gain_segment_intercepts_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
        segment_endpoints=segment_endpoints,
        segment_labels=segment_labels,
    )

    # a_segment, b_segment
    df_a_segment_b_segment = fit_affine_place_fields_per_segment_df(
        pf_y=pf_y_fit,
        pf_x=pf_x,
        pf_y_cv=pf_y_cv,
        segment_endpoints=segment_endpoints,
        segment_labels=segment_labels,
    )


if __name__ == "__main__":
    main()
