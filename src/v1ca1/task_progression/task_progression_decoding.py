from __future__ import annotations
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import position_tools as pt
import spikeinterface.full as si
import kyutils
import pandas as pd
import track_linearization as tl
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d


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


def get_linear_position(place_bin_size, gap=0):
    # place field: concatenate all four trajectories
    linear_position = {}
    for epoch in run_epoch_list:
        t_list = []
        d_list = []
        for i, trajectory_type in enumerate(trajectory_types):
            position_df = tl.get_linearized_position(
                position=position_dict[epoch][position_offset:],
                track_graph=track_graph_from_center[trajectory_type],
                edge_order=linear_edge_order_from_center,
                edge_spacing=0,
            )
            lp = nap.Tsd(
                t=timestamps_position_dict[epoch][position_offset:],
                d=position_df["linear_position"]
                + (total_length_per_trajectory + gap) * i,
                time_support=trajectory_ep[epoch][trajectory_type],
            )
            t_list.append(lp.t)
            d_list.append(lp.d)
        t = np.concatenate(t_list)
        d = np.concatenate(d_list)
        sort_idx = np.argsort(t)
        linear_position[epoch] = nap.Tsd(
            t=t[sort_idx],
            d=d[sort_idx],
            time_support=movement[epoch].time_support,
        )
    position_bins = np.arange(
        0, total_length_per_trajectory * 4 + gap * 3 + place_bin_size, place_bin_size
    )
    return linear_position, position_bins


# task progression field: "fold" same turn trajectories on top of one another and concatenate across trajectory groups
def get_task_progression(task_progression_bin_size, gap=0):
    task_progression = {}
    for epoch in run_epoch_list:
        t_list = []
        d_list = []
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
            if trajectory_type in ["center_to_left", "right_to_center"]:
                offset = 0
            else:
                offset = 1 + gap / total_length_per_trajectory

            ltp = nap.Tsd(
                t=timestamps_position_dict[epoch][position_offset:],
                d=position_df["linear_position"] / total_length_per_trajectory + offset,
                time_support=trajectory_ep[epoch][trajectory_type],
            )
            t_list.append(ltp.t)
            d_list.append(ltp.d)

        t = np.concatenate(t_list)
        d = np.concatenate(d_list)
        sort_idx = np.argsort(t)
        task_progression[epoch] = nap.Tsd(
            t=t[sort_idx],
            d=d[sort_idx],
            time_support=movement[epoch].time_support,
        )
    task_progression_bins = np.arange(
        0,
        2 + gap / total_length_per_trajectory + task_progression_bin_size,
        task_progression_bin_size,
    )
    return task_progression, task_progression_bins


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


def get_train_test_splits(epoch, n_folds=5, random_state=47):
    """generates random train-test splits across trajectories in an epoch

    Parameters
    ----------
    epoch : str
        _description_
    n_folds : int, optional
        _description_, by default 5
    random_state : int, optional
        _description_, by default 47

    Returns
    -------
    _type_
        _description_
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    train_ep = {}
    test_ep = {}
    for trajectory_type in trajectory_types:
        train_ep[trajectory_type] = []
        test_ep[trajectory_type] = []
        for fold, (train_idx, test_idx) in enumerate(
            kf.split(trajectory_ep[epoch][trajectory_type])
        ):
            train_ep[trajectory_type].append(train_idx)
            test_ep[trajectory_type].append(test_idx)

    def get_start_end(tr_ep, n_folds):
        start_times = {}
        end_times = {}
        for fold in range(n_folds):
            st = []
            en = []
            for trajectory_type in trajectory_types:
                t_ep = trajectory_ep[epoch][trajectory_type][
                    tr_ep[trajectory_type][fold]
                ]
                st.append(t_ep.start)
                en.append(t_ep.end)
            start_times[fold] = np.concatenate(st)
            end_times[fold] = np.concatenate(en)

        train_eps = {}
        for fold in range(n_folds):
            sort_idx = np.argsort(start_times[fold])
            train_eps[fold] = nap.IntervalSet(
                start=start_times[fold][sort_idx],
                end=end_times[fold][sort_idx],
            )
        return train_eps

    train_eps = get_start_end(train_ep, n_folds=n_folds)
    test_eps = get_start_end(test_ep, n_folds=n_folds)
    return train_eps, test_eps


def decode_cv(region, epoch, feature, bins, n_folds=5):

    decoded_features = []
    true_features = []

    train_eps, test_eps = get_train_test_splits(epoch, n_folds=n_folds, random_state=47)

    # kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # for train_idx, test_idx in kf.split(movement[epoch].time_support):
    for k in range(n_folds):
        tc = nap.compute_tuning_curves(
            data=spikes[region],
            features=feature[epoch],
            bins=[bins],  # Use standardized bins
            # epochs=movement[epoch].time_support[train_idx],
            epochs=train_eps[k].intersect(movement[epoch].time_support),
        )
        decoded, proba_feature = nap.decode_bayes(
            tuning_curves=tc,
            data=spikes[region],
            # epochs=movement[epoch].time_support[test_idx],
            epochs=test_eps[k].intersect(movement[epoch].time_support),
            sliding_window_size=4,
            bin_size=0.02,
        )
        decoded_features.append(decoded)
        true_features.append(
            feature[epoch].restrict(test_eps[k].intersect(movement[epoch].time_support))
        )

    ts = []
    ds = []
    for feature in true_features:
        ts.append(feature.t)
        ds.append(feature.d)
    ts = np.concatenate(ts)
    ds = np.concatenate(ds)
    sort_idx = np.argsort(ts)
    true_tsd = nap.Tsd(
        t=ts[sort_idx],
        d=ds[sort_idx],
        time_support=movement[epoch].time_support,
    )

    ts = []
    ds = []
    for feature in decoded_features:
        ts.append(feature.t)
        ds.append(feature.d)
    ts = np.concatenate(ts)
    ds = np.concatenate(ds)
    sort_idx = np.argsort(ts)
    decoded_tsd = nap.Tsd(
        t=ts[sort_idx],
        d=ds[sort_idx],
        time_support=movement[epoch].time_support,
    )

    return true_tsd, decoded_tsd


def decode_task_cross_trajectory(region, epoch, feature, bins, fr_threshold=0.5):

    fr_dark = fr_during_movement[region][run_epoch_list[3]]
    mask_dark_active = fr_dark > fr_threshold

    encoding_decoding = {
        "center_to_left": "right_to_center",
        "right_to_center": "center_to_left",
        "center_to_right": "left_to_center",
        "left_to_center": "center_to_right",
    }

    true_tsd = {}
    decoded_tsd = {}
    for encoding_trajectory, decoding_trajectory in encoding_decoding.items():
        tc = nap.compute_tuning_curves(
            data=spikes[region],
            features=feature[epoch][encoding_trajectory],
            bins=[bins],
            epochs=trajectory_ep[epoch][encoding_trajectory].intersect(
                movement[epoch].time_support
            ),
        )

        decoded, proba_feature = nap.decode_bayes(
            tuning_curves=tc,
            data=spikes[region],
            epochs=trajectory_ep[epoch][decoding_trajectory].intersect(
                movement[epoch].time_support
            ),
            sliding_window_size=4,
            bin_size=0.02,
        )

        true_tsd[(encoding_trajectory, decoding_trajectory)] = feature[epoch][
            decoding_trajectory
        ].restrict(
            trajectory_ep[epoch][decoding_trajectory].intersect(
                movement[epoch].time_support
            )
        )
        decoded_tsd[(encoding_trajectory, decoding_trajectory)] = decoded

    return true_tsd, decoded_tsd


def plot_decode(true_place, decoded_place, true_tp, decoded_tp, fig_path):

    fig, ax = plt.subplots(figsize=(12, 8), nrows=2)
    ax[0].scatter(
        true_place.times(),
        true_place.values,
        label="True place",
    )
    ax[0].scatter(
        decoded_place.times(),
        decoded_place.values,
        label="Decoded place",
        c="orange",
        s=1,
        alpha=0.7,
    )
    ax[0].legend(
        frameon=False,
        bbox_to_anchor=(1.0, 1.0),
    )
    ax[1].scatter(
        true_tp.times(),
        true_tp.values,
        label="True tp",
    )
    ax[1].scatter(
        decoded_tp.times(),
        decoded_tp.values,
        label="Decoded tp",
        c="orange",
        s=1,
        alpha=0.7,
    )
    ax[1].legend(
        frameon=False,
        bbox_to_anchor=(1.0, 1.0),
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    return None


def plot_error_vs_position(
    decoded_pos: nap.Tsd,
    true_pos: nap.Tsd,
    *,
    n_bins: int = 40,
    min_count: int = 20,
    error_mode: Literal["abs", "sq", "signed"] = "abs",
    yerr_mode: Literal["sem", "sd", "q95"] = "sem",
    q_level: float = 0.95,
    clip_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
    ax: Optional[plt.Axes] = None,
    color="black",
    label="light",
) -> plt.Axes:
    """
    Plot decoding error (y) vs true position (x) with error bars per position bin.

    yerr_mode:
        - "sem": SD/sqrt(n) around the mean
        - "sd" : SD around the mean
        - "q95": central q_level interval around the median (prevents negative yerr)
    """
    if yerr_mode == "q95" and not (0.0 < q_level < 1.0):
        raise ValueError("q_level must be in (0, 1).")

    # --- numpy arrays ---
    t_dec = np.asarray(decoded_pos.t, dtype=np.float64)
    x_hat = np.asarray(decoded_pos.d, dtype=np.float64)

    t_true = np.asarray(true_pos.t, dtype=np.float64)
    x_true = np.asarray(true_pos.d, dtype=np.float64)

    # --- interpolate true position at decoded times (NaN outside range) ---
    ok_true = np.isfinite(t_true) & np.isfinite(x_true)
    if ok_true.sum() < 2:
        raise ValueError("true_pos has <2 finite samples; cannot interpolate.")

    tt = t_true[ok_true]
    xx = x_true[ok_true]

    order = np.argsort(tt)
    tt = tt[order]
    xx = xx[order]

    # remove duplicate timestamps (keep first occurrence)
    uniq = np.concatenate(([True], np.diff(tt) > 0))
    tt = tt[uniq]
    xx = xx[uniq]

    x_true_at_dec = np.interp(t_dec, tt, xx, left=np.nan, right=np.nan)

    # --- per-sample error ---
    if error_mode == "abs":
        err = np.abs(x_hat - x_true_at_dec)
        ylab = "Decoding error |x̂ − x|"
    elif error_mode == "sq":
        err = (x_hat - x_true_at_dec) ** 2
        ylab = "Decoding error (x̂ − x)²"
    else:  # signed
        err = x_hat - x_true_at_dec
        ylab = "Signed error (x̂ − x)"

    pos = x_true_at_dec

    # --- drop NaNs ---
    good = np.isfinite(pos) & np.isfinite(err)
    pos = pos[good]
    err = err[good]
    if pos.size == 0:
        raise ValueError("No finite aligned samples after NaN handling.")

    # --- choose bin range ---
    if clip_quantiles is not None:
        lo, hi = np.quantile(pos, clip_quantiles)
    else:
        lo, hi = float(np.min(pos)), float(np.max(pos))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(pos)), float(np.max(pos))

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # --- assign bins ---
    b = np.digitize(pos, edges) - 1
    in_range = (b >= 0) & (b < n_bins)
    b = b[in_range]
    err = err[in_range]

    # --- summarize each bin ---
    y = np.full(n_bins, np.nan, dtype=np.float64)
    yerr_sym = np.full(n_bins, np.nan, dtype=np.float64)  # sem/sd
    yerr_asym = np.full((2, n_bins), np.nan, dtype=np.float64)  # q95

    alpha = (1.0 - q_level) / 2.0
    q_lo = alpha
    q_hi = 1.0 - alpha

    for i in range(n_bins):
        e = err[b == i]
        n = e.size
        if n < min_count:
            continue

        if yerr_mode in ("sem", "sd"):
            center = float(np.mean(e))
            y[i] = center
            sd = float(np.std(e, ddof=1)) if n > 1 else np.nan
            if yerr_mode == "sem":
                yerr_sym[i] = sd / np.sqrt(n) if np.isfinite(sd) else np.nan
            else:
                yerr_sym[i] = sd
        else:
            # Quantile interval: center at median so distances are nonnegative
            center = float(np.median(e))
            y[i] = center
            lo_q, hi_q = np.quantile(e, [q_lo, q_hi])
            yerr_asym[0, i] = center - lo_q
            yerr_asym[1, i] = hi_q - center

    # --- plot ---
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if yerr_mode in ("sem", "sd"):
        m = np.isfinite(y) & np.isfinite(yerr_sym)
        ax.errorbar(
            centers[m],
            y[m],
            yerr=yerr_sym[m],
            fmt="o-",
            capsize=2,
            lw=1,
            ms=4,
            color=color,
            label=label,
        )
    else:
        m = np.isfinite(y) & np.isfinite(yerr_asym[0]) & np.isfinite(yerr_asym[1])
        ax.errorbar(
            centers[m],
            y[m],
            yerr=yerr_asym[:, m],
            fmt="o-",
            capsize=2,
            lw=1,
            ms=4,
            color=color,
            label=label,
        )

    ax.set_xlabel("Position (true)")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.2)

    if error_mode == "signed":
        ax.axhline(0.0, lw=1, alpha=0.6, color="black")

    return ax


SummaryMode = Literal["mean_std", "median_iqr"]
ErrorMode = Literal["signed", "unsigned"]
AlignTo = Literal["decoded", "true"]


def plot_decoding_error_vs_position(
    true_pos: nap.Tsd,
    decoded_pos: nap.Tsd,
    *,
    # binning
    n_bins: int = 40,
    bin_edges: Optional[np.ndarray] = None,
    pos_range: Optional[Tuple[float, float]] = None,
    clip_quantiles: Optional[Tuple[float, float]] = None,  # e.g. (0.01, 0.99)
    min_count: int = 20,
    # error + summary
    error_mode: ErrorMode = "unsigned",
    summary: SummaryMode = "mean_std",
    # alignment/interpolation
    align_to: AlignTo = "decoded",
    ep: Optional[nap.IntervalSet] = None,
    left: Optional[float] = np.nan,
    right: Optional[float] = np.nan,
    # plotting
    ax: Optional[plt.Axes] = None,
    color: str = "black",
    label: Optional[str] = None,
    show_scatter: bool = False,
    scatter_kws: Optional[Dict[str, Any]] = None,
    line_kws: Optional[Dict[str, Any]] = None,
    # outputs
    return_stats: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, Dict[str, np.ndarray]]]:
    """
    Plot decoding error (decoded - true) as a function of position.

    Steps:
      1) Restrict both series to a common epoch (intersection by default)
      2) Use nap.Tsd.interpolate to align one series onto the other's timestamps
      3) Compute per-sample error
      4) Bin by position and summarize within each bin

    Parameters
    ----------
    true_pos, decoded_pos : nap.Tsd
        True and decoded position time series.
    n_bins : int
        Number of position bins (ignored if bin_edges is provided).
    bin_edges : np.ndarray | None
        Explicit bin edges. Use this when overlaying conditions so bins match.
    pos_range : (float, float) | None
        Force bin range; overrides clip_quantiles if provided.
    clip_quantiles : (float, float) | None
        If provided and pos_range is None, compute bin range from position quantiles.
    min_count : int
        Minimum number of samples required to keep a bin.
    error_mode : {"signed","unsigned"}
        Signed: (decoded - true). Unsigned: abs(decoded - true).
    summary : {"mean_std","median_iqr"}
        mean_std: center=mean, errorbar=std (symmetric)
        median_iqr: center=median, errorbar spans Q1..Q3 (asymmetric)
    align_to : {"decoded","true"}
        "decoded" (default): interpolate true onto decoded timestamps (typical for decoder output).
        "true": interpolate decoded onto true timestamps (if you want error at each true sample).
    ep : nap.IntervalSet | None
        Epoch to use. If None, uses intersection of time_supports (when available).
    left, right : float | None
        Passed to nap.Tsd.interpolate. Using np.nan avoids edge extrapolation artifacts.
    show_scatter : bool
        If True, plot per-sample (pos, error) scatter behind binned summary.

    Returns
    -------
    ax : matplotlib Axes
        The axis with the plot.
    (ax, stats) if return_stats=True
        stats contains bin centers, counts, and summarized values.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    # ---- choose an epoch (intersection by default) ----
    if ep is None:
        if hasattr(true_pos, "time_support") and hasattr(decoded_pos, "time_support"):
            ep = true_pos.time_support.intersect(decoded_pos.time_support)
        else:
            # fallback: intersect the min/max time ranges
            t0 = max(float(true_pos.t[0]), float(decoded_pos.t[0]))
            t1 = min(float(true_pos.t[-1]), float(decoded_pos.t[-1]))
            ep = nap.IntervalSet(start=t0, end=t1)

    # ---- restrict ----
    true_r = true_pos.restrict(ep)
    dec_r = decoded_pos.restrict(ep)

    # ---- align using nap.Tsd.interpolate ----
    if align_to == "decoded":
        # evaluate true position at decoded timestamps
        true_at_dec = true_r.interpolate(dec_r, ep=ep, left=left, right=right)
        pos = np.asarray(true_at_dec.d, dtype=np.float64)
        dec = np.asarray(dec_r.d, dtype=np.float64)
    else:  # align_to == "true"
        # evaluate decoded position at true timestamps
        dec_at_true = dec_r.interpolate(true_r, ep=ep, left=left, right=right)
        pos = np.asarray(true_r.d, dtype=np.float64)
        dec = np.asarray(dec_at_true.d, dtype=np.float64)

    # ---- compute error per sample ----
    err = dec - pos
    if error_mode == "unsigned":
        err = np.abs(err)

    # ---- drop non-finite ----
    good = np.isfinite(pos) & np.isfinite(err)
    pos = pos[good]
    err = err[good]
    if pos.size == 0:
        raise ValueError("No finite aligned samples after interpolation/restriction.")

    # ---- decide bin edges ----
    if bin_edges is None:
        if pos_range is not None:
            lo, hi = map(float, pos_range)
        elif clip_quantiles is not None:
            qlo, qhi = clip_quantiles
            if not (0.0 <= qlo < qhi <= 1.0):
                raise ValueError("clip_quantiles must satisfy 0 <= lo < hi <= 1.")
            lo, hi = np.quantile(pos, [qlo, qhi]).astype(float)
        else:
            lo, hi = float(np.min(pos)), float(np.max(pos))

        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(pos)), float(np.max(pos))

        bin_edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    else:
        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("bin_edges must be a 1D array with length >= 2.")
        if not np.all(np.diff(bin_edges) > 0):
            raise ValueError("bin_edges must be strictly increasing.")

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins_eff = centers.size

    # ---- assign bins ----
    b = np.digitize(pos, bin_edges) - 1
    in_range = (b >= 0) & (b < n_bins_eff)
    b = b[in_range]
    pos_in = pos[in_range]
    err_in = err[in_range]

    # ---- optional scatter for debugging ----
    if show_scatter:
        sk = dict(s=6, alpha=0.15, linewidths=0)
        if scatter_kws:
            sk.update(scatter_kws)
        ax.scatter(pos_in, err_in, color=color, **sk)

    # ---- summarize within bins ----
    y = np.full(n_bins_eff, np.nan, dtype=np.float64)
    yerr_low = np.full(n_bins_eff, np.nan, dtype=np.float64)
    yerr_high = np.full(n_bins_eff, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins_eff, dtype=int)

    for i in range(n_bins_eff):
        e = err_in[b == i]
        n = e.size
        counts[i] = n
        if n < min_count:
            continue

        if summary == "mean_std":
            center = float(np.mean(e))
            spread = float(np.std(e, ddof=1)) if n > 1 else np.nan
            y[i] = center
            yerr_low[i] = spread
            yerr_high[i] = spread
        else:  # "median_iqr"
            center = float(np.median(e))
            q1, q3 = np.quantile(e, [0.25, 0.75]).astype(float)
            y[i] = center
            yerr_low[i] = center - q1
            yerr_high[i] = q3 - center

    valid = np.isfinite(y) & np.isfinite(yerr_low) & np.isfinite(yerr_high)
    if not np.any(valid):
        raise ValueError(
            "No bins had enough samples. Try lowering min_count or reducing n_bins."
        )

    # ---- plot binned summary ----
    lk = dict(fmt="o-", lw=1.0, ms=4, capsize=2)
    if line_kws:
        lk.update(line_kws)

    ax.errorbar(
        centers[valid],
        y[valid],
        yerr=np.vstack([yerr_low[valid], yerr_high[valid]]),
        color=color,
        label=label,
        **lk,
    )

    # ---- labels ----
    ax.set_xlabel("Position (true)")
    if error_mode == "signed":
        ylab = "Decoded − true"
        ax.axhline(0.0, lw=1, alpha=0.5, color="black")
    else:
        ylab = "|Decoded − true|"

    if summary == "mean_std":
        ylab += " (mean ± std)"
    else:
        ylab += " (median, IQR)"

    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.2)

    # ---- return stats if requested ----
    if return_stats:
        stats = dict(
            bin_edges=bin_edges,
            bin_centers=centers,
            n=counts,
            center=y,
            yerr_low=yerr_low,
            yerr_high=yerr_high,
            valid=valid,
        )
        return ax, stats

    return ax


def main():
    save_dir = analysis_path / "task_progression_decoding"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = analysis_path / "figs" / "task_progression_decoding"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_folds = 5

    true_tp = {"v1": {}, "ca1": {}}
    true_place = {"v1": {}, "ca1": {}}
    decoded_tp = {"v1": {}, "ca1": {}}
    decoded_place = {"v1": {}, "ca1": {}}

    true_tp_cross_traj = {"v1": {}, "ca1": {}}
    decoded_tp_cross_traj = {"v1": {}, "ca1": {}}

    place_bin_size = 4  # cm
    place_gap = 0  # cm

    task_progression_bin_size = place_bin_size / total_length_per_trajectory
    task_progression_gap = place_gap / total_length_per_trajectory

    linear_position, position_bins = get_linear_position(
        place_bin_size=place_bin_size, gap=place_gap
    )
    task_progression, task_progression_bins = get_task_progression(
        task_progression_bin_size=task_progression_bin_size,
        gap=task_progression_gap,
    )
    task_progression_by_trajectory, task_progression_by_trajectory_bins = (
        get_task_progression_by_trajectory(
            task_progression_bin_size=task_progression_bin_size
        )
    )

    for region in regions:
        for epoch in run_epoch_list:
            true_place[region][epoch], decoded_place[region][epoch] = decode_cv(
                region,
                epoch,
                feature=linear_position,
                bins=position_bins,
                n_folds=n_folds,
            )
            true_tp[region][epoch], decoded_tp[region][epoch] = decode_cv(
                region,
                epoch,
                feature=task_progression,
                bins=task_progression_bins,
                n_folds=n_folds,
            )

            plot_decode(
                true_place[region][epoch],
                decoded_place[region][epoch],
                true_tp[region][epoch],
                decoded_tp[region][epoch],
                fig_path=fig_dir / f"{region}_{epoch}_place_tp_decoding.png",
            )

            true_tp_cross_traj[region][epoch], decoded_tp_cross_traj[region][epoch] = (
                decode_task_cross_trajectory(
                    region,
                    epoch,
                    task_progression_by_trajectory,
                    task_progression_by_trajectory_bins,
                    fr_threshold=0.5,
                )
            )

    with open(save_dir / f"true_place.pkl", "wb") as f:
        pickle.dump(true_place, f)
    with open(save_dir / f"true_tp.pkl", "wb") as f:
        pickle.dump(true_tp, f)
    with open(save_dir / f"decoded_place.pkl", "wb") as f:
        pickle.dump(decoded_place, f)
    with open(save_dir / f"decoded_tp.pkl", "wb") as f:
        pickle.dump(decoded_tp, f)
    with open(save_dir / f"true_tp_cross_traj.pkl", "wb") as f:
        pickle.dump(true_tp_cross_traj, f)
    with open(save_dir / f"decoded_tp_cross_traj.pkl", "wb") as f:
        pickle.dump(decoded_tp_cross_traj, f)

    # plot comparison (decoding error by position)

    epoch1 = run_epoch_list[0]
    epoch2 = run_epoch_list[3]

    for region in regions:
        fig, ax = plt.subplots(figsize=(24, 3))
        plot_error_vs_position(
            decoded_place[region][epoch1],
            true_place[region][epoch1],
            n_bins=int(total_length_per_trajectory / place_bin_size) * 4,
            min_count=5,
            error_mode="signed",
            yerr_mode="sd",
            q_level=0.5,
            ax=ax,
            label="light",
            color="orange",
        )
        plot_error_vs_position(
            decoded_place[region][epoch2],
            true_place[region][epoch2],
            n_bins=int(total_length_per_trajectory / place_bin_size) * 4,
            min_count=5,
            error_mode="signed",
            yerr_mode="sd",
            q_level=0.5,
            ax=ax,
            label="dark",
            color="gray",
        )
        ax.axvline(total_length_per_trajectory + place_gap / 2, color="black")
        ax.axvline(
            total_length_per_trajectory * 2 + place_gap + place_gap / 2, color="black"
        )
        ax.axvline(
            total_length_per_trajectory * 3 + 2 * place_gap + place_gap / 2,
            color="black",
        )
        ax.set_ylim(
            [
                -total_length_per_trajectory - place_gap / 2,
                total_length_per_trajectory + place_gap / 2,
            ]
        )
        ax.set_xlim([0, 4 * total_length_per_trajectory + 3 * place_gap])

        ax.set_xlabel("Position (cm)")

        plt.tight_layout()
        fig.savefig(fig_dir / f"{region}_{epoch1}_{epoch2}_de_by_position.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 3))
        plot_error_vs_position(
            decoded_tp[region][epoch1],
            true_tp[region][epoch1],
            n_bins=int(1 / task_progression_bin_size) * 2,
            min_count=5,
            error_mode="signed",
            yerr_mode="sd",
            q_level=0.5,
            ax=ax,
            label="light",
            color="orange",
        )
        plot_error_vs_position(
            decoded_tp[region][epoch2],
            true_tp[region][epoch2],
            n_bins=int(1 / task_progression_bin_size) * 2,
            min_count=5,
            error_mode="signed",
            yerr_mode="sd",
            q_level=0.5,
            ax=ax,
            label="dark",
            color="gray",
        )
        ax.axvline(1, color="black")

        ax.set_xlim([0, 2 + task_progression_gap / total_length_per_trajectory])
        ax.set_ylim(
            [
                -1 - task_progression_gap / total_length_per_trajectory / 2,
                1 + task_progression_gap / total_length_per_trajectory / 2,
            ]
        )

        ax.set_xlabel("Task progression")
        plt.tight_layout()
        fig.savefig(fig_dir / f"{region}_{epoch1}_{epoch2}_de_by_tp.png", dpi=300)
        plt.close(fig)

    for region in regions:
        for encoding_trajectory, decoding_trajectory in encoding_decoding.items():
            fig, ax = plt.subplots(figsize=(12, 3))
            plot_error_vs_position(
                decoded_tp_cross_traj[region][epoch1][
                    (encoding_trajectory, decoding_trajectory)
                ],
                true_tp_cross_traj[region][epoch1][
                    (encoding_trajectory, decoding_trajectory)
                ],
                n_bins=int(1 / task_progression_bin_size),
                min_count=5,
                error_mode="signed",
                yerr_mode="sd",
                q_level=0.5,
                ax=ax,
                label="light",
                color="orange",
            )
            plot_error_vs_position(
                decoded_tp_cross_traj[region][epoch2][
                    (encoding_trajectory, decoding_trajectory)
                ],
                true_tp_cross_traj[region][epoch2][
                    (encoding_trajectory, decoding_trajectory)
                ],
                n_bins=int(1 / task_progression_bin_size),
                min_count=5,
                error_mode="signed",
                yerr_mode="sd",
                q_level=0.5,
                ax=ax,
                label="dark",
                color="gray",
            )

            ax.set_xlim([0, 1])
            ax.set_ylim([-1, 1])

            ax.set_xlabel("Task progression")

            plt.tight_layout()
            fig.savefig(
                fig_dir
                / f"{region}_{epoch1}_{epoch2}_{encoding_trajectory}_{decoding_trajectory}_de.png",
                dpi=300,
            )
            plt.close(fig)

    # new function
    for region in regions:
        # place
        fig, ax = plt.subplots(figsize=(24, 3))

        # light
        plot_decoding_error_vs_position(
            true_place[region][epoch1],
            decoded_place[region][epoch1],
            bin_edges=np.linspace(
                0,
                4 * total_length_per_trajectory + 3 * place_gap,
                int(total_length_per_trajectory / place_bin_size) * 4 + 1,
            ),
            error_mode="signed",
            summary="median_iqr",
            min_count=5,
            ax=ax,
            color="orange",
            label="light",
        )

        # dark
        plot_decoding_error_vs_position(
            true_place[region][epoch2],
            decoded_place[region][epoch2],
            bin_edges=np.linspace(
                0,
                4 * total_length_per_trajectory + 3 * place_gap,
                int(total_length_per_trajectory / place_bin_size) * 4 + 1,
            ),
            error_mode="signed",
            summary="median_iqr",
            min_count=5,
            ax=ax,
            color="gray",
            label="dark",
        )

        ax.legend(frameon=False)

        ax.axvline(total_length_per_trajectory + place_gap / 2, color="black")
        ax.axvline(
            total_length_per_trajectory * 2 + place_gap + place_gap / 2, color="black"
        )
        ax.axvline(
            total_length_per_trajectory * 3 + 2 * place_gap + place_gap / 2,
            color="black",
        )
        ax.set_ylim(
            [
                -total_length_per_trajectory - place_gap / 2,
                total_length_per_trajectory + place_gap / 2,
            ]
        )
        ax.set_xlim([0, 4 * total_length_per_trajectory + 3 * place_gap])

        ax.set_xlabel("Position (cm)")

        plt.tight_layout()
        fig.savefig(
            fig_dir / f"{region}_{epoch1}_{epoch2}_de_by_position_new.png", dpi=300
        )
        plt.close(fig)

        # TP
        fig, ax = plt.subplots(figsize=(12, 3))
        plot_decoding_error_vs_position(
            true_tp[region][epoch1],
            decoded_tp[region][epoch1],
            bin_edges=np.linspace(
                0,
                2 * total_length_per_trajectory + 1 * place_gap,
                int(total_length_per_trajectory / place_bin_size) * 2 + 1,
            )
            / total_length_per_trajectory,
            error_mode="signed",
            summary="median_iqr",
            min_count=5,
            ax=ax,
            color="orange",
            label="light",
        )
        plot_decoding_error_vs_position(
            true_tp[region][epoch2],
            decoded_tp[region][epoch2],
            bin_edges=np.linspace(
                0,
                2 * total_length_per_trajectory + 1 * place_gap,
                int(total_length_per_trajectory / place_bin_size) * 2 + 1,
            )
            / total_length_per_trajectory,
            error_mode="signed",
            summary="median_iqr",
            min_count=5,
            ax=ax,
            color="gray",
            label="dark",
        )

        ax.axvline(1, color="black")

        ax.set_xlim([0, 2])
        ax.set_ylim(
            [
                -1 - task_progression_gap / total_length_per_trajectory / 2,
                1 + task_progression_gap / total_length_per_trajectory / 2,
            ]
        )

        ax.set_xlabel("Task progression")
        plt.tight_layout()
        fig.savefig(fig_dir / f"{region}_{epoch1}_{epoch2}_de_by_tp_new.png", dpi=300)
        plt.close(fig)

    # new function cross traj
    for region in regions:
        for encoding_trajectory, decoding_trajectory in encoding_decoding.items():
            fig, ax = plt.subplots(figsize=(12, 3))
            plot_decoding_error_vs_position(
                true_tp_cross_traj[region][epoch1][
                    (encoding_trajectory, decoding_trajectory)
                ],
                decoded_tp_cross_traj[region][epoch1][
                    (encoding_trajectory, decoding_trajectory)
                ],
                bin_edges=np.linspace(
                    0,
                    1 * total_length_per_trajectory + 0 * place_gap,
                    int(total_length_per_trajectory / place_bin_size) * 1 + 1,
                )
                / total_length_per_trajectory,
                error_mode="signed",
                summary="median_iqr",
                min_count=5,
                ax=ax,
                color="orange",
                label="light",
            )
            plot_decoding_error_vs_position(
                true_tp_cross_traj[region][epoch2][
                    (encoding_trajectory, decoding_trajectory)
                ],
                decoded_tp_cross_traj[region][epoch2][
                    (encoding_trajectory, decoding_trajectory)
                ],
                bin_edges=np.linspace(
                    0,
                    1 * total_length_per_trajectory + 0 * place_gap,
                    int(total_length_per_trajectory / place_bin_size) * 1 + 1,
                )
                / total_length_per_trajectory,
                error_mode="signed",
                summary="median_iqr",
                min_count=5,
                ax=ax,
                color="gray",
                label="dark",
            )

            ax.set_xlim([0, 1])
            ax.set_ylim([-1, 1])

            ax.set_xlabel("Task progression")

            plt.tight_layout()
            fig.savefig(
                fig_dir
                / f"{region}_{epoch1}_{epoch2}_{encoding_trajectory}_{decoding_trajectory}_de_new.png",
                dpi=300,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
