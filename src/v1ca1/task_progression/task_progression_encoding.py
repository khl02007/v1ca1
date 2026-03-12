from __future__ import annotations
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
import seaborn as sns
from typing import Optional, Tuple, Any, Iterable, Optional
import numpy.typing as npt
from sklearn.model_selection import KFold

from dataclasses import dataclass
from typing import Dict, List, Tuple


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

speed_threshold = 4  # cm/s
position_offset = 10  # frames
place_bin_size = 4  # cm


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


# turn sorting into pynapple
spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])


# define intervals
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


fig_dir = analysis_path / "figs" / "task_progression_tuning"
fig_dir.mkdir(parents=True, exist_ok=True)


# make track graphs for each trajectory
dx = 9.5
dy = 9
diagonal_segment_length = np.sqrt(dx**2 + dy**2)

long_segment_length = 81 - 17 - 2
short_segment_length = 13.5

total_length_per_trajectory = (
    long_segment_length * 2 + short_segment_length + 2 * diagonal_segment_length
)

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

# make speed and movement pynapple
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


# calculate firing rates (for identifying active cells)
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


from scipy.ndimage import gaussian_filter1d


def smooth_pf_along_position(pf, pos_dim: str, sigma_bins: float):
    pf = pf.fillna(0)  # Replaces NaN values with 0
    # gaussian_filter1d works on numpy; apply along the position axis
    axis = pf.get_axis_num(pos_dim)
    sm = gaussian_filter1d(pf.values, sigma=sigma_bins, axis=axis, mode="nearest")
    return pf.copy(data=sm)


def smooth_pf_along_position_nan_aware(
    pf: Any,
    pos_dim: str,
    sigma_bins: float,
    *,
    eps: float = 1e-12,
    mode: str = "nearest",
) -> Any:
    """
    Smooth a tuning curve along a position dimension without treating NaNs as zeros.

    This performs masked (NaN-aware) Gaussian smoothing:
      smooth(rate * mask) / smooth(mask)

    Parameters
    ----------
    pf
        Tuning curve object with attributes:
          - pf.values: ndarray
          - pf.get_axis_num(pos_dim): int
          - pf.copy(data=ndarray): returns same type
        (e.g., xarray.DataArray-like object returned by pynapple tuning curves.)
    pos_dim
        Name of the position dimension to smooth along (e.g., "linpos", "tp").
    sigma_bins
        Gaussian kernel width in units of bins.
    eps
        Small constant to avoid division by zero.
    mode
        Boundary mode passed to `gaussian_filter1d`.

    Returns
    -------
    pf_smoothed
        Same type as `pf`, with smoothed values. Bins with no support remain NaN.
    """
    axis = pf.get_axis_num(pos_dim)
    x = np.asarray(pf.values, dtype=np.float64)

    mask = np.isfinite(x)
    x_filled = np.where(mask, x, 0.0)

    num = gaussian_filter1d(x_filled, sigma=sigma_bins, axis=axis, mode=mode)
    den = gaussian_filter1d(
        mask.astype(np.float64), sigma=sigma_bins, axis=axis, mode=mode
    )

    sm = num / np.maximum(den, eps)

    # If a bin has effectively zero support even after smoothing, keep it NaN
    sm = np.where(den > eps, sm, np.nan)

    return pf.copy(data=sm)


place_bin_size = 4  # cm
task_progression_bin_size = place_bin_size / total_length_per_trajectory


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
# calculate task progression tuning curve for each trajectory
# task_progression_by_trajectory_bins = np.linspace(0, 1, 41)
task_progression_by_trajectory_bins = np.arange(
    0,
    1 + task_progression_bin_size,
    task_progression_bin_size,
)

tpf_by_trajectory = {}
for region in regions:
    tpf_by_trajectory[region] = {}
    for epoch in run_epoch_list:
        tpf_by_trajectory[region][epoch] = {}
        for trajectory_type in trajectory_types:
            tpf_by_trajectory[region][epoch][trajectory_type] = (
                nap.compute_tuning_curves(
                    data=spikes[region],
                    features=task_progression_by_trajectory[epoch][trajectory_type],
                    bins=[task_progression_by_trajectory_bins],
                    epochs=movement[epoch].time_support,
                    feature_names=["tp"],
                )
            )

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
            d=position_df["linear_position"] + total_length_per_trajectory * i,
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
    0, total_length_per_trajectory * 4 + place_bin_size, place_bin_size
)
pf = {}
smoothed_pf = {}
place_si = {}
for region in regions:
    pf[region] = {}
    smoothed_pf[region] = {}
    place_si[region] = {}
    for epoch in run_epoch_list:
        pf[region][epoch] = nap.compute_tuning_curves(
            data=spikes[region],
            features=linear_position[epoch],
            bins=[position_bins],  # Use standardized bins
            epochs=movement[epoch].time_support,
            feature_names=["linpos"],
        )
        smoothed_pf[region][epoch] = smooth_pf_along_position_nan_aware(
            pf[region][epoch], pos_dim="linpos", sigma_bins=1.0
        )
        place_si[region][epoch] = nap.compute_mutual_information(pf[region][epoch])


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
            offset = 1

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
    0, 2 + task_progression_bin_size, task_progression_bin_size
)
tpf = {}
smoothed_tpf = {}
tp_si = {}
for region in regions:
    tpf[region] = {}
    smoothed_tpf[region] = {}
    tp_si[region] = {}
    for epoch in run_epoch_list:
        tpf[region][epoch] = nap.compute_tuning_curves(
            data=spikes[region],
            features=task_progression[epoch],
            bins=[task_progression_bins],  # Use standardized bins
            epochs=movement[epoch].time_support,
            feature_names=["tp"],
        )
        smoothed_tpf[region][epoch] = smooth_pf_along_position_nan_aware(
            tpf[region][epoch], pos_dim="tp", sigma_bins=1.0
        )
        tp_si[region][epoch] = nap.compute_mutual_information(tpf[region][epoch])


# --- HELPER: Interpolation ---


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


def _intervalset_to_arrays(ep: nap.IntervalSet) -> Tuple[np.ndarray, np.ndarray]:
    """Return (start, end) arrays as float64."""
    start = np.asarray(ep.start, dtype=np.float64).ravel()
    end = np.asarray(ep.end, dtype=np.float64).ravel()
    if start.size == 0:
        return start, end
    order = np.argsort(start)
    return start[order], end[order]


def split_intervalset_by_duration(
    ep: nap.IntervalSet, n_splits: int
) -> List[nap.IntervalSet]:
    """
    Split an IntervalSet into `n_splits` folds with (approximately) equal total duration.
    The folds are *blocked* in time over the concatenated intervals (not random).

    Parameters
    ----------
    ep : nap.IntervalSet
        Disjoint intervals in time (e.g., movement time_support).
    n_splits : int
        Number of folds.

    Returns
    -------
    folds : list[nap.IntervalSet]
        List of IntervalSets, one per fold, ordered in time.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    starts, ends = _intervalset_to_arrays(ep)
    if starts.size == 0:
        return [nap.IntervalSet(start=[], end=[]) for _ in range(n_splits)]

    durations = ends - starts
    total = float(np.sum(durations))
    target = total / n_splits

    folds: List[nap.IntervalSet] = []
    cur_s: List[float] = []
    cur_e: List[float] = []
    acc = 0.0

    for s0, e0 in zip(starts, ends):
        s = float(s0)
        e = float(e0)
        while s < e:
            remaining = target - acc
            seg_len = min(e - s, remaining)
            seg_start = s
            seg_end = s + seg_len

            cur_s.append(seg_start)
            cur_e.append(seg_end)

            acc += seg_len
            s = seg_end

            # close fold once we hit target (with tolerance)
            if acc >= target - 1e-12 and len(folds) < n_splits - 1:
                folds.append(nap.IntervalSet(start=cur_s, end=cur_e))
                cur_s, cur_e = [], []
                acc = 0.0

    # last fold takes remainder
    folds.append(nap.IntervalSet(start=cur_s, end=cur_e))

    # If numerical edge-cases produce fewer/more folds, fix deterministically:
    if len(folds) < n_splits:
        # pad empties (rare)
        for _ in range(n_splits - len(folds)):
            folds.append(nap.IntervalSet(start=[], end=[]))
    elif len(folds) > n_splits:
        # merge extras into last
        extra = folds[n_splits:]
        folds = folds[:n_splits]
        if extra:
            s_last, e_last = _intervalset_to_arrays(folds[-1])
            s_add, e_add = [], []
            for ex in extra:
                s_ex, e_ex = _intervalset_to_arrays(ex)
                s_add.extend(s_ex.tolist())
                e_add.extend(e_ex.tolist())
            if s_add:
                folds[-1] = nap.IntervalSet(
                    start=np.concatenate([s_last, np.asarray(s_add)]),
                    end=np.concatenate([e_last, np.asarray(e_add)]),
                )

    return folds


def intervalset_difference(a: nap.IntervalSet, b: nap.IntervalSet) -> nap.IntervalSet:
    """
    Return a \\ b (subtract b from a) for disjoint interval sets.

    Parameters
    ----------
    a, b : nap.IntervalSet
        Interval sets.

    Returns
    -------
    out : nap.IntervalSet
        The parts of `a` not covered by `b`.
    """
    a_s, a_e = _intervalset_to_arrays(a)
    b_s, b_e = _intervalset_to_arrays(b)

    if a_s.size == 0:
        return nap.IntervalSet(start=[], end=[])
    if b_s.size == 0:
        return nap.IntervalSet(start=a_s, end=a_e)

    out_s: List[float] = []
    out_e: List[float] = []

    j = 0
    nb = b_s.size

    for sa, ea in zip(a_s, a_e):
        cur = float(sa)

        # advance b while it ends before current a interval starts
        while j < nb and b_e[j] <= sa:
            j += 1

        k = j
        while k < nb and b_s[k] < ea:
            # non-overlap segment before b
            if b_s[k] > cur:
                out_s.append(cur)
                out_e.append(min(float(b_s[k]), float(ea)))
            # skip overlapped part
            cur = max(cur, float(b_e[k]))
            if cur >= ea:
                break
            k += 1

        if cur < ea:
            out_s.append(cur)
            out_e.append(float(ea))

    return nap.IntervalSet(start=out_s, end=out_e)


@dataclass
class LLResult:
    """Per-fold additive log-likelihood bookkeeping."""

    ll_model: float
    ll_null: float
    n_spikes: int

    @property
    def ll_model_per_spike(self) -> float:
        return np.nan if self.n_spikes == 0 else self.ll_model / self.n_spikes

    @property
    def ll_null_per_spike(self) -> float:
        return np.nan if self.n_spikes == 0 else self.ll_null / self.n_spikes

    @property
    def info_bits_per_spike(self) -> float:
        return (
            np.nan
            if self.n_spikes == 0
            else (self.ll_model - self.ll_null) / (np.log(2.0) * self.n_spikes)
        )


def poisson_ll_on_epoch(
    *,
    unit_spikes: nap.Ts,
    position: nap.Tsd,
    tuning_curve: npt.NDArray[np.float64],
    bin_edges: npt.NDArray[np.float64],
    epoch: nap.IntervalSet,
    bin_size: float = 0.002,
    epsilon: float = 1e-10,
    # NEW (optional):
    null_rate: Optional[float] = None,
    fill_rate: Optional[float] = None,
) -> LLResult:
    """
    Evaluate Poisson log-likelihood on `epoch` using a fixed tuning curve (fit elsewhere).

    Fixes vs your original version:
      - does NOT clip counts to 1 (so per-spike is really per spike)
      - masks NaN positions (avoids NaN -> last bin artifact)
      - replaces NaN/invalid tuning-curve bins with `fill_rate` (so LL doesn't become NaN)
      - optionally uses a provided `null_rate` (e.g. train-mean) instead of fitting null on test
    """
    # Restrict signals
    pos_ep = position.restrict(epoch)
    spikes_ep = unit_spikes.restrict(epoch)

    # Bin spikes to create master time grid
    binned_spikes = spikes_ep.count(bin_size, epoch)
    pos_at_bins = pos_ep.interpolate(binned_spikes, ep=epoch)

    # Extract arrays
    k = np.asarray(binned_spikes.d, dtype=np.int64).ravel()  # spike counts per bin
    x = np.asarray(pos_at_bins.d, dtype=np.float64).ravel()  # position per bin

    # Drop bins where position is NaN/inf (IMPORTANT)
    valid = np.isfinite(x)
    if not np.all(valid):
        k = k[valid]
        x = x[valid]

    dt = float(bin_size)
    n_spikes = int(k.sum())
    total_time = float(k.size * dt)

    # If the epoch is empty after masking, contribute nothing
    if total_time <= 0.0 or k.size == 0:
        return LLResult(ll_model=0.0, ll_null=0.0, n_spikes=0)

    # Choose null rate:
    # - if you pass null_rate (recommended: train mean), use it
    # - else fall back to test mean (your old behavior)
    if null_rate is None:
        null_rate_eval = (n_spikes / total_time) if n_spikes > 0 else 0.0
    else:
        null_rate_eval = float(null_rate)
    null_rate_eval = max(null_rate_eval, epsilon)

    # Choose fill rate for unsupported (NaN) tuning bins:
    # default: use null rate (safe)
    if fill_rate is None:
        fill_rate_eval = null_rate_eval
    else:
        fill_rate_eval = float(fill_rate)
    fill_rate_eval = max(fill_rate_eval, epsilon)

    # Lookup model rate at each time bin
    tuning_curve = np.asarray(tuning_curve, dtype=np.float64).ravel()
    spatial_idx = np.digitize(x, bin_edges) - 1
    spatial_idx = np.clip(spatial_idx, 0, tuning_curve.size - 1)

    lam = tuning_curve[spatial_idx]

    # Replace NaN/inf model rates with fill_rate_eval (CRITICAL FIX)
    lam = np.where(np.isfinite(lam), lam, fill_rate_eval)

    # Floor at epsilon to avoid log(0)
    lam = np.maximum(lam, epsilon)

    # Log-likelihood under inhomogeneous Poisson (dropping log(k!) term)
    ll_model = float(np.sum(k * np.log(lam * dt) - lam * dt))

    # Null LL under homogeneous Poisson with null_rate_eval
    ll_null = float(np.sum(k * np.log(null_rate_eval * dt) - null_rate_eval * dt))

    return LLResult(ll_model=ll_model, ll_null=ll_null, n_spikes=n_spikes)


def interpolate_nans(y):
    """
    Linearly interpolates NaN values in a 1D numpy array.
    If the entire array is NaN, returns zeros.
    """
    if np.all(np.isnan(y)):
        return np.nan_to_num(y)

    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]

    y_out = y.copy()
    y_out[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y_out


def calculate_bits_per_spike_pynapple(
    unit_spikes, position, tuning_curve, bin_edges, epoch, bin_size=0.002
):
    """
    Calculates spatial information (bits/spike) using Pynapple objects.

    Parameters:
    -----------
    unit_spikes : nap.Ts or nap.Tsd
        Spikes for a single neuron (e.g. spikes[region][unit_id]).
    position : nap.Tsd
        Position data (e.g. linear_position).
    tuning_curve : numpy.ndarray or pandas.Series
        The firing rate values of the tuning curve.
    bin_edges : numpy.ndarray
        The spatial bin edges used to generate the tuning curve.
    epoch : nap.IntervalSet
        The specific time intervals to analyze (e.g. movement & correct_trials).
    bin_size : float
        Time bin size in seconds (default 0.002).

    Returns:
    --------
    bits_per_spike : float
    """

    # 1. RESTRICT DATA TO THE EPOCH
    # This automatically handles gaps/discontinuities in the epoch
    pos_ep = position.restrict(epoch)
    spikes_ep = unit_spikes.restrict(epoch)

    # 2. BIN SPIKES (Create the time grid)
    # .count() returns a Tsd of spike counts in each bin over the epoch
    # The index of 'binned_spikes' becomes our master time clock
    binned_spikes = spikes_ep.count(bin_size, epoch)

    # 3. ALIGN POSITION TO TIME BINS
    # This is the magic of Pynapple: interpolate position to the spike bin timestamps
    # It automatically handles the gaps in the IntervalSet
    pos_at_bins = pos_ep.interpolate(binned_spikes, ep=epoch)

    # Extract numpy arrays for calculation
    # We use .d (data) to get the values
    k = binned_spikes.d.flatten()  # Spike counts (0 or 1 usually)
    x = pos_at_bins.d.flatten()  # Position at those times

    # Clip spike counts to 1 (Bernoulli assumption for small bins)
    k = np.minimum(k, 1)

    # 4. LOOKUP PREDICTED RATES (Model)
    # Digitizing position to find spatial bins
    # bin_edges usually has N+1 items for N bins
    spatial_indices = np.digitize(x, bin_edges) - 1

    # Handle edge cases (literally) where items fall on the last edge or outside
    n_spatial_bins = len(tuning_curve)
    spatial_indices = np.clip(spatial_indices, 0, n_spatial_bins - 1)

    # Get lambda (predicted rate)
    predicted_rates = tuning_curve[spatial_indices]

    # Safety floor for log
    epsilon = 1e-10
    predicted_rates = np.maximum(predicted_rates, epsilon)

    # 5. CALCULATE LOG-LIKELIHOODS
    dt = bin_size

    # --- Model Likelihood ---
    # sum( k * ln(lambda*dt) - lambda*dt )
    term1 = k * np.log(predicted_rates * dt)
    term2 = predicted_rates * dt
    ll_model = np.sum(term1 - term2)

    # --- Null Likelihood (Mean Rate) ---
    # Calculate global mean rate over this specific epoch
    total_spikes = np.sum(k)
    total_time = len(k) * dt

    if total_spikes == 0:
        return 0.0

    mean_rate = total_spikes / total_time

    term1_null = k * np.log(mean_rate * dt)
    term2_null = mean_rate * dt
    ll_null = np.sum(term1_null - term2_null)

    # 6. CONVERT TO BITS/SPIKE
    ll_diff_nats = ll_model - ll_null
    bits_per_spike = (ll_diff_nats / np.log(2)) / total_spikes

    return bits_per_spike


# --- PLOTTING FUNCTIONS ---
def cv_epoch_to_df(
    cv_place: Dict[str, Dict[int, dict]],
    cv_tp: Dict[str, Dict[int, dict]],
    *,
    region: str,
    epoch: str,
) -> pd.DataFrame:
    """
    Convert CV results for a single epoch into a tidy DataFrame (one row per unit).

    Parameters
    ----------
    cv_place, cv_tp
        Nested dicts: cv_place[epoch][unit_id] -> metrics dict.
    region
        Region label.
    epoch
        Epoch key to extract.

    Returns
    -------
    df
        Columns:
          - region, epoch, unit_id
          - n_spikes (pooled held-out spikes across folds)
          - ll_place, ll_tp (per spike)
          - delta_bits = (ll_place - ll_tp)/log(2)
          - delta_info_bits (if available, else NaN)
    """
    rows: list[dict[str, Any]] = []
    units = cv_place.get(epoch, {})

    for unit_id, d_place in units.items():
        d_tp = cv_tp[epoch][unit_id]

        ll_place = float(d_place.get("ll_model_per_spike_cv", np.nan))
        ll_tp = float(d_tp.get("ll_model_per_spike_cv", np.nan))

        n_spikes = int(np.sum(np.asarray(d_place.get("fold_n_spikes", 0))))

        info_place = d_place.get("info_bits_per_spike_cv", np.nan)
        info_tp = d_tp.get("info_bits_per_spike_cv", np.nan)
        delta_info_bits = (
            float(info_place) - float(info_tp)
            if np.isfinite(info_place) and np.isfinite(info_tp)
            else np.nan
        )

        rows.append(
            {
                "region": region,
                "epoch": epoch,
                "unit_id": int(unit_id),
                "n_spikes": n_spikes,
                "ll_place": ll_place,
                "ll_tp": ll_tp,
                "delta_bits": (ll_place - ll_tp) / np.log(2.0),
                "delta_info_bits": delta_info_bits,
            }
        )

    return pd.DataFrame(rows)


def filter_epoch_df(df: pd.DataFrame, *, min_spikes: int = 50) -> pd.DataFrame:
    """
    Filter out units with too few held-out spikes (noisy CV estimates).
    """
    out = df.copy()
    out = out[np.isfinite(out["ll_place"]) & np.isfinite(out["ll_tp"])]
    out = out[out["n_spikes"] >= int(min_spikes)]
    return out


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x, x
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def _bootstrap_ci_fraction_positive(
    x: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for p = P(x > 0). Returns (p_hat, lo, hi).
    """
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan

    p_hat = float(np.mean(x > 0))
    boot = np.empty(int(n_boot), dtype=np.float64)

    for i in range(int(n_boot)):
        samp = rng.choice(x, size=x.size, replace=True)
        boot[i] = np.mean(samp > 0)

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1.0 - alpha / 2))
    return p_hat, lo, hi


def plot_cv_two_epoch_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    epoch_a: str,
    epoch_b: str,
    region: str,
    save_path: Path,
    delta_col: str = "delta_bits",
    title_extra: str = "",
) -> None:
    """
    Save a single multi-panel figure comparing two epochs.

    Panels:
      A) Scatter epoch_a: LL_place vs LL_tp
      B) Scatter epoch_b: LL_place vs LL_tp
      C) Paired slope: delta (epoch_a -> epoch_b) for units present in both
      D) Fraction favoring place (delta>0) for each epoch with bootstrap CI
      E) ECDF overlay of delta for the two epochs

    Parameters
    ----------
    df_a, df_b
        Filtered per-unit tables for epoch_a and epoch_b.
    epoch_a, epoch_b
        Epoch labels to show on plots.
    region
        Region label.
    save_path
        Output path (e.g. .png or .pdf).
    delta_col
        Which delta column to plot: "delta_bits" or "delta_info_bits".
    title_extra
        Extra title string.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # index by unit for pairing
    a = df_a.set_index("unit_id")
    b = df_b.set_index("unit_id")
    common_units = a.index.intersection(b.index)

    mosaic = [
        ["scatter_a", "scatter_b"],
        ["paired", "frac"],
        ["ecdf", "ecdf"],
    ]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(11, 10))

    def _scatter(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        ax.scatter(df["ll_tp"], df["ll_place"], s=12, alpha=0.6)

        if df.shape[0] > 0:
            lo = float(np.nanmin([df["ll_tp"].min(), df["ll_place"].min()]))
            hi = float(np.nanmax([df["ll_tp"].max(), df["ll_place"].max()]))
            ax.plot([lo, hi], [lo, hi])

        ax.set_title(title)
        ax.set_xlabel("LL_tp (per spike)")
        ax.set_ylabel("LL_place (per spike)")

    _scatter(df_a, axd["scatter_a"], f"{epoch_a}: LL_place vs LL_tp")
    _scatter(df_b, axd["scatter_b"], f"{epoch_b}: LL_place vs LL_tp")

    # Paired slope plot
    ax = axd["paired"]
    if common_units.size == 0:
        ax.text(0.5, 0.5, "No paired units", ha="center", va="center")
        ax.set_axis_off()
    else:
        da = a.loc[common_units, delta_col].to_numpy()
        db = b.loc[common_units, delta_col].to_numpy()

        # sort by epoch_a delta for nicer display
        order = np.argsort(da)
        da = da[order]
        db = db[order]

        for y0, y1 in zip(da, db):
            ax.plot([0, 1], [y0, y1], alpha=0.25)

        ax.axhline(0.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([epoch_a, epoch_b])
        ax.set_ylabel(delta_col)
        ax.set_title("Paired change per unit")

        ax.scatter([0, 1], [float(np.median(da)), float(np.median(db))], s=90)

    # Fraction favoring place with CI
    ax = axd["frac"]
    p_a, lo_a, hi_a = _bootstrap_ci_fraction_positive(df_a[delta_col].to_numpy())
    p_b, lo_b, hi_b = _bootstrap_ci_fraction_positive(df_b[delta_col].to_numpy())

    xs = [epoch_a, epoch_b]
    ps = [p_a, p_b]
    yerr = [[p_a - lo_a, p_b - lo_b], [hi_a - p_a, hi_b - p_b]]

    ax.bar(xs, ps, yerr=yerr, capsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"P({delta_col} > 0)")
    ax.set_title("Fraction of units favoring place")

    # ECDF overlay
    ax = axd["ecdf"]
    x_a, y_a = _ecdf(df_a[delta_col].to_numpy())
    x_b, y_b = _ecdf(df_b[delta_col].to_numpy())
    ax.plot(x_a, y_a, label=epoch_a)
    ax.plot(x_b, y_b, label=epoch_b)
    ax.axvline(0.0)
    ax.set_xlabel(delta_col)
    ax.set_ylabel("ECDF")
    ax.set_title("Distribution shift (ECDF)")
    ax.legend()

    fig.suptitle(
        f"{region} CV comparison: {epoch_a} vs {epoch_b} {title_extra}".strip()
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- MAIN ---
def main():
    data_dir = analysis_path / "tp_encoding"
    data_dir.mkdir(parents=True, exist_ok=True)

    n_folds = 5
    sigma_bins = 1.0  # smoothing

    for region in regions:
        # outputs:
        # [epoch][unit_id] -> dict with fold metrics + aggregated CV metric
        cv_place: Dict[str, Dict[int, dict]] = {}
        cv_tp: Dict[str, Dict[int, dict]] = {}

        for epoch in run_epoch_list:
            cv_place[epoch] = {}
            cv_tp[epoch] = {}

            valid_ep = movement[epoch].time_support
            train_eps, test_eps = get_train_test_splits(
                epoch, n_folds=n_folds, random_state=47
            )
            # fold_eps = split_intervalset_by_duration(valid_ep, n_splits=n_folds)

            unit_ids = list(spikes[region].index)

            # initialize containers
            for unit_id in unit_ids:
                cv_place[epoch][unit_id] = {
                    "fold_ll_model": np.zeros(n_folds, dtype=np.float64),
                    "fold_ll_null": np.zeros(n_folds, dtype=np.float64),
                    "fold_n_spikes": np.zeros(n_folds, dtype=np.int64),
                    "ll_model_per_spike_cv": np.nan,
                    "ll_null_per_spike_cv": np.nan,
                    "info_bits_per_spike_cv": np.nan,
                }
                cv_tp[epoch][unit_id] = {
                    "fold_ll_model": np.zeros(n_folds, dtype=np.float64),
                    "fold_ll_null": np.zeros(n_folds, dtype=np.float64),
                    "fold_n_spikes": np.zeros(n_folds, dtype=np.int64),
                    "ll_model_per_spike_cv": np.nan,
                    "ll_null_per_spike_cv": np.nan,
                    "info_bits_per_spike_cv": np.nan,
                }

            for k in range(n_folds):
                # Intersect once (cleaner than intersecting repeatedly)
                test_fold = test_eps[k].intersect(valid_ep)
                train_fold = train_eps[k].intersect(valid_ep)

                # Skip empty folds (prevents weird edge cases)
                train_dur = float(train_fold.tot_length())
                test_dur = float(test_fold.tot_length())
                if train_dur <= 0.0 or test_dur <= 0.0:
                    continue

                # ---- Fit tuning curves on TRAIN fold ----
                pf_train = nap.compute_tuning_curves(
                    data=spikes[region],
                    features=linear_position[epoch],
                    bins=[position_bins],
                    epochs=train_fold,
                    feature_names=["linpos"],
                )
                sm_pf_train = smooth_pf_along_position_nan_aware(
                    pf_train, pos_dim="linpos", sigma_bins=sigma_bins
                )

                tpf_train = nap.compute_tuning_curves(
                    data=spikes[region],
                    features=task_progression[epoch],
                    bins=[task_progression_bins],
                    epochs=train_fold,
                    feature_names=["tp"],
                )
                sm_tpf_train = smooth_pf_along_position_nan_aware(
                    tpf_train, pos_dim="tp", sigma_bins=sigma_bins
                )

                # ---- Evaluate on TEST fold ----
                for unit_id in unit_ids:
                    unit_ts = spikes[region][unit_id]

                    # Train mean rate for this unit (used for fill + optionally for null)
                    # This uses spike count directly (not binned), so it's fast + exact.
                    n_train_spikes = int(len(unit_ts.restrict(train_fold).t))
                    train_rate = max(n_train_spikes / train_dur, 1e-10)

                    # Place model
                    res_p = poisson_ll_on_epoch(
                        unit_spikes=unit_ts,
                        position=linear_position[epoch],
                        tuning_curve=sm_pf_train[unit_id].to_numpy(),
                        bin_edges=sm_pf_train[unit_id].bin_edges[0],
                        epoch=test_fold,
                        bin_size=0.02,
                        epsilon=1e-10,
                        fill_rate=train_rate,  # << NEW (prevents NaN tuning bins from killing LL)
                        null_rate=train_rate,  # << NEW (strict CV null). Remove if you want test-fit null.
                    )
                    cv_place[epoch][unit_id]["fold_ll_model"][k] = res_p.ll_model
                    cv_place[epoch][unit_id]["fold_ll_null"][k] = res_p.ll_null
                    cv_place[epoch][unit_id]["fold_n_spikes"][k] = res_p.n_spikes

                    # Task-progression model
                    res_t = poisson_ll_on_epoch(
                        unit_spikes=unit_ts,
                        position=task_progression[epoch],
                        tuning_curve=sm_tpf_train[unit_id].to_numpy(),
                        bin_edges=sm_tpf_train[unit_id].bin_edges[0],
                        epoch=test_fold,
                        bin_size=0.02,
                        epsilon=1e-10,
                        fill_rate=train_rate,  # << NEW
                        null_rate=train_rate,  # << NEW (optional)
                    )
                    cv_tp[epoch][unit_id]["fold_ll_model"][k] = res_t.ll_model
                    cv_tp[epoch][unit_id]["fold_ll_null"][k] = res_t.ll_null
                    cv_tp[epoch][unit_id]["fold_n_spikes"][k] = res_t.n_spikes
            # for k in range(n_folds):
            #     # test_ep = fold_eps[k]
            #     # train_ep = intervalset_difference(valid_ep, test_ep)
            #     test_ep = test_eps[k]
            #     train_ep = train_eps[k]

            #     # fit tuning curves on TRAIN folds
            #     pf_train = nap.compute_tuning_curves(
            #         data=spikes[region],
            #         features=linear_position[epoch],
            #         bins=[position_bins],
            #         epochs=train_ep.intersect(valid_ep),
            #         feature_names=["linpos"],
            #     )
            #     sm_pf_train = smooth_pf_along_position_nan_aware(
            #         pf_train, pos_dim="linpos", sigma_bins=sigma_bins
            #     )

            #     tpf_train = nap.compute_tuning_curves(
            #         data=spikes[region],
            #         features=task_progression[epoch],
            #         bins=[task_progression_bins],
            #         epochs=train_ep.intersect(valid_ep),
            #         feature_names=["tp"],
            #     )
            #     sm_tpf_train = smooth_pf_along_position_nan_aware(
            #         tpf_train, pos_dim="tp", sigma_bins=sigma_bins
            #     )

            #     # evaluate on TEST fold
            #     for unit_id in unit_ids:
            #         unit_ts = spikes[region][unit_id]

            #         # place model
            #         res_p = poisson_ll_on_epoch(
            #             unit_spikes=unit_ts,
            #             position=linear_position[epoch],
            #             tuning_curve=sm_pf_train[unit_id].to_numpy(),
            #             bin_edges=sm_pf_train[unit_id].bin_edges[0],
            #             epoch=test_ep.intersect(valid_ep),
            #         )
            #         cv_place[epoch][unit_id]["fold_ll_model"][k] = res_p.ll_model
            #         cv_place[epoch][unit_id]["fold_ll_null"][k] = res_p.ll_null
            #         cv_place[epoch][unit_id]["fold_n_spikes"][k] = res_p.n_spikes

            #         # task-progression model
            #         res_t = poisson_ll_on_epoch(
            #             unit_spikes=unit_ts,
            #             position=task_progression[epoch],
            #             tuning_curve=sm_tpf_train[unit_id].to_numpy(),
            #             bin_edges=sm_tpf_train[unit_id].bin_edges[0],
            #             epoch=test_ep.intersect(valid_ep),
            #         )
            #         cv_tp[epoch][unit_id]["fold_ll_model"][k] = res_t.ll_model
            #         cv_tp[epoch][unit_id]["fold_ll_null"][k] = res_t.ll_null
            #         cv_tp[epoch][unit_id]["fold_n_spikes"][k] = res_t.n_spikes

            # aggregate across folds (additive LLs)
            for unit_id in unit_ids:
                # place
                llm = float(np.sum(cv_place[epoch][unit_id]["fold_ll_model"]))
                lln = float(np.sum(cv_place[epoch][unit_id]["fold_ll_null"]))
                ns = int(np.sum(cv_place[epoch][unit_id]["fold_n_spikes"]))

                if ns > 0:
                    cv_place[epoch][unit_id]["ll_model_per_spike_cv"] = llm / ns
                    cv_place[epoch][unit_id]["ll_null_per_spike_cv"] = lln / ns
                    cv_place[epoch][unit_id]["info_bits_per_spike_cv"] = (llm - lln) / (
                        np.log(2.0) * ns
                    )

                # tp
                llm = float(np.sum(cv_tp[epoch][unit_id]["fold_ll_model"]))
                lln = float(np.sum(cv_tp[epoch][unit_id]["fold_ll_null"]))
                ns = int(np.sum(cv_tp[epoch][unit_id]["fold_n_spikes"]))

                if ns > 0:
                    cv_tp[epoch][unit_id]["ll_model_per_spike_cv"] = llm / ns
                    cv_tp[epoch][unit_id]["ll_null_per_spike_cv"] = lln / ns
                    cv_tp[epoch][unit_id]["info_bits_per_spike_cv"] = (llm - lln) / (
                        np.log(2.0) * ns
                    )

        # save
        with open(data_dir / f"{region}_cv{n_folds}_place_ll.pkl", "wb") as f:
            pickle.dump(cv_place, f)

        with open(data_dir / f"{region}_cv{n_folds}_tp_ll.pkl", "wb") as f:
            pickle.dump(cv_tp, f)

        fig_out_dir = analysis_path / "figs" / "tp_encoding_cv"
        fig_out_dir.mkdir(parents=True, exist_ok=True)

        epoch_a = run_epoch_list[0]  # or explicit string you want
        epoch_b = run_epoch_list[3]

        df_a = filter_epoch_df(
            cv_epoch_to_df(cv_place, cv_tp, region=region, epoch=epoch_a),
            min_spikes=50,
        )
        df_b = filter_epoch_df(
            cv_epoch_to_df(cv_place, cv_tp, region=region, epoch=epoch_b),
            min_spikes=50,
        )

        plot_cv_two_epoch_comparison(
            df_a,
            df_b,
            epoch_a=epoch_a,
            epoch_b=epoch_b,
            region=region,
            save_path=fig_out_dir / f"{region}_{epoch_a}_vs_{epoch_b}_cv.png",
            delta_col="delta_bits",  # or "delta_info_bits"
            title_extra="(CV 5-fold)",
        )


if __name__ == "__main__":
    main()
