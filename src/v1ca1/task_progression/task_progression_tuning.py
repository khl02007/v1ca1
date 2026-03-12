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
import scipy.stats as stats
import seaborn as sns
from typing import Optional, Tuple, Any, Iterable, Optional
import numpy.typing as npt
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


def smooth_pf_along_position(pf, pos_dim: str, sigma_bins: float):
    pf = pf.fillna(0)  # Replaces NaN values with 0
    # gaussian_filter1d works on numpy; apply along the position axis
    axis = pf.get_axis_num(pos_dim)
    sm = gaussian_filter1d(pf.values, sigma=sigma_bins, axis=axis, mode="nearest")
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
# calculate task progression tuning curve and mutual info for each trajectory
# task_progression_by_trajectory_bins = np.linspace(0, 1, 41)
task_progression_by_trajectory_bins = np.arange(
    0,
    1 + task_progression_bin_size,
    task_progression_bin_size,
)

tpf_by_trajectory = {}
tp_si_by_trajectory = {}
for region in regions:
    tpf_by_trajectory[region] = {}
    tp_si_by_trajectory[region] = {}
    for epoch in run_epoch_list:
        tpf_by_trajectory[region][epoch] = {}
        tp_si_by_trajectory[region][epoch] = {}
        for trajectory_type in trajectory_types:
            tpf_by_trajectory[region][epoch][trajectory_type] = (
                nap.compute_tuning_curves(
                    data=spikes[region],
                    features=task_progression_by_trajectory[epoch][trajectory_type],
                    bins=[task_progression_by_trajectory_bins],  # Use standardized bins
                    epochs=movement[epoch].time_support,
                    feature_names=["tp"],
                )
            )
            tp_si_by_trajectory[region][epoch][trajectory_type] = (
                nap.compute_mutual_information(
                    tpf_by_trajectory[region][epoch][trajectory_type]
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
        smoothed_pf[region][epoch] = smooth_pf_along_position(
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
        smoothed_tpf[region][epoch] = smooth_pf_along_position(
            tpf[region][epoch], pos_dim="tp", sigma_bins=1.0
        )
        tp_si[region][epoch] = nap.compute_mutual_information(tpf[region][epoch])


# --- HELPER: Interpolation ---
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


def compute_shuffled_si(region, epoch, feature, bins, n_shuffles=50, min_shift=20.0):
    """
    Computes chance-level Spatial Information by circularly shifting spikes
    WITHIN the time support of the epoch to preserve spike counts.
    """
    # Initialize dictionary to store sum of shuffled SI for averaging later
    shuffled_si_accum = pd.DataFrame(
        0.0, index=spikes[region].keys(), columns=["bits/sec", "bits/spike"]
    )
    # Pre-calculate epoch duration for modulo arithmetic
    duration = all_ep[epoch].end - all_ep[epoch].start

    # We only shift if the epoch is long enough
    if duration <= min_shift:
        print(
            f"Warning: Epoch too short for shuffle ({duration:.1f}s). Returning NaNs."
        )
        return pd.Series({unit: np.nan for unit in spikes.keys()})

    spikes_to_shift = spikes[region].restrict(all_ep[epoch])
    for i in range(n_shuffles):
        shifted_spikes = nap.shift_timestamps(
            spikes_to_shift, min_shift=min_shift, max_shift=duration - min_shift
        )

        tc_shuffled = nap.compute_tuning_curves(
            data=shifted_spikes,
            features=feature,
            bins=[bins],
            epochs=movement[epoch].time_support,
        )

        # 4. Compute Mutual Information on Shuffled Data
        si_shuffled = nap.compute_mutual_information(
            tuning_curves=tc_shuffled,
        )

        shuffled_si_accum += si_shuffled.fillna(0)

    return shuffled_si_accum / n_shuffles


def calculate_task_similarity(
    tpf_by_trajectory, region, epoch, fr_map, fr_threshold=0.5, turn_type="left"
):
    """
    Calculates correlation between distinct trajectories for the same turn,
    filtering out neurons that are silent in the dark.

    Parameters
    ----------
    tpf_by_trajectory : dict
        Dict[region][epoch][trajectory_type] -> xarray (unit, tp)
    region : str
        Brain region key.
    epoch : str
        Epoch key.
    fr_map : dict, pd.Series, or np.ndarray
        The average firing rate in the dark.
        - If dict/Series: keys must be unit_ids.
        - If np.ndarray: MUST be aligned with sorted(unit_ids).
    fr_threshold : float
        Exclude units with firing rate < this value in the dark.
    turn_type : str
        'left' or 'right'.
    """

    # 1. Define the Pair
    if turn_type == "left":
        traj_1 = "center_to_left"
        traj_2 = "right_to_center"
    elif turn_type == "right":
        traj_1 = "center_to_right"
        traj_2 = "left_to_center"
    else:
        raise ValueError("turn_type must be 'left' or 'right'")

    # Check data existence
    if (
        traj_1 not in tpf_by_trajectory[region][epoch]
        or traj_2 not in tpf_by_trajectory[region][epoch]
    ):
        print(f"Warning: Missing trajectory data for {epoch}")
        return pd.DataFrame()

    da_1 = tpf_by_trajectory[region][epoch][traj_1]
    da_2 = tpf_by_trajectory[region][epoch][traj_2]

    correlations = []

    # 2. Identify Units (Intersection of Traj 1 and Traj 2)
    if "unit" in da_1.coords:
        units_1 = da_1.coords["unit"].values
        units_2 = da_2.coords["unit"].values
    else:
        units_1 = da_1.unit.values
        units_2 = da_2.unit.values

    common_units = np.intersect1d(units_1, units_2)

    # --- 3. APPLY FIRING RATE FILTER ---
    active_units = []

    for unit in common_units:
        # Determine the firing rate for this unit
        try:
            if isinstance(fr_map, (pd.Series, dict)):
                rate = fr_map[unit]
            elif isinstance(fr_map, (np.ndarray, list)):
                # DANGER: Assumes array index matches unit ID index (0, 1, 2...)
                # Only use this if your units are 0, 1, 2... or perfectly sorted
                rate = fr_map[int(unit)]
            else:
                # Fallback or error
                rate = 0

            # The Filter Condition
            if rate > fr_threshold:
                active_units.append(unit)

        except KeyError:
            # Unit missing from FR map (safe skip)
            continue

    active_units = np.array(active_units)

    if len(active_units) == 0:
        print(f"No units passed FR threshold > {fr_threshold} Hz")
        return pd.DataFrame()

    # 4. Compute Correlation for Active Units
    for unit in active_units:
        # Interpolate NaNs instead of zeroing them
        curve_1 = interpolate_nans(da_1.sel(unit=unit).values)
        curve_2 = interpolate_nans(da_2.sel(unit=unit).values)

        # Secondary Safety Check: Ensure curves aren't flat zeros
        # (Even if avg FR is high, the curve might be empty on this specific trajectory)
        if np.max(curve_1) > 0 and np.max(curve_2) > 0:
            # Check variance to avoid Pearson divide-by-zero
            if np.std(curve_1) > 1e-9 and np.std(curve_2) > 1e-9:
                r, p = stats.pearsonr(curve_1, curve_2)
                correlations.append({"unit": unit, "similarity": r})

    return pd.DataFrame(correlations).set_index("unit")


def fit_gain_only(
    y_light: npt.NDArray[np.float64],
    y_dark: npt.NDArray[np.float64],
    *,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Fit gain-only model: y_dark ≈ g * y_light (zero intercept).

    Parameters
    ----------
    y_light, y_dark
        1D arrays of equal length.
    eps
        Small constant for numerical stability.

    Returns
    -------
    g : float
        Least-squares gain (slope).
    sse : float
        Sum of squared errors.
    r2 : float
        Coefficient of determination (NaN if undefined).
    """
    x = np.asarray(y_light, dtype=np.float64)
    y = np.asarray(y_dark, dtype=np.float64)

    denom = float(np.dot(x, x)) + eps
    g = float(np.dot(x, y) / denom)

    y_hat = g * x
    resid = y - y_hat
    sse = float(np.dot(resid, resid))

    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = float(1.0 - sse / (ss_tot + eps)) if ss_tot > eps else np.nan
    return g, sse, r2


def fit_affine(
    y_light: npt.NDArray[np.float64],
    y_dark: npt.NDArray[np.float64],
    *,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """
    Fit affine model: y_dark ≈ a * y_light + b.

    Parameters
    ----------
    y_light, y_dark
        1D arrays of equal length.
    eps
        Small constant for numerical stability.

    Returns
    -------
    a : float
        Least-squares slope.
    b : float
        Least-squares intercept.
    sse : float
        Sum of squared errors.
    r2 : float
        Coefficient of determination (NaN if undefined).
    """
    x = np.asarray(y_light, dtype=np.float64)
    y = np.asarray(y_dark, dtype=np.float64)

    x_mean = float(x.mean())
    y_mean = float(y.mean())

    var_x = float(np.dot(x - x_mean, x - x_mean))
    if var_x <= eps:
        # Degenerate x: can't fit slope reliably
        return np.nan, np.nan, np.nan, np.nan

    cov_xy = float(np.dot(x - x_mean, y - y_mean))
    a = cov_xy / (var_x + eps)
    b = y_mean - a * x_mean

    y_hat = a * x + b
    resid = y - y_hat
    sse = float(np.dot(resid, resid))

    ss_tot = float(np.dot(y - y_mean, y - y_mean))
    r2 = float(1.0 - sse / (ss_tot + eps)) if ss_tot > eps else np.nan
    return float(a), float(b), sse, r2


def _overlap_by_shift(
    y_light: npt.NDArray[np.float64],
    y_dark: npt.NDArray[np.float64],
    shift: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Align arrays with a non-circular integer shift.

    Convention:
      y_dark[i] ≈ g * y_light[i - shift]

    So:
      shift > 0 means dark is shifted to the RIGHT relative to light.
    """
    n = min(len(y_light), len(y_dark))
    yl = y_light[:n]
    yd = y_dark[:n]

    if shift == 0:
        return yl, yd

    if shift > 0:
        # yd[shift:] aligns with yl[:-shift]
        return yl[: n - shift], yd[shift:]
    # shift < 0
    s = -shift
    # yd[:-s] aligns with yl[s:]
    return yl[s:], yd[: n - s]


def fit_shift_gain(
    y_light: npt.NDArray[np.float64],
    y_dark: npt.NDArray[np.float64],
    *,
    max_shift: int = 10,
    min_bins: int = 8,
    eps: float = 1e-12,
) -> Tuple[int, float, float, float, int]:
    """
    Fit shift+gain model: y_dark[i] ≈ g * y_light[i - shift].

    Parameters
    ----------
    y_light, y_dark
        1D arrays (same binning).
    max_shift
        Max absolute shift in bins to test.
    min_bins
        Minimum number of overlapping bins required to evaluate a shift.
    eps
        Numerical stability constant.

    Returns
    -------
    best_shift : int
        Shift (in bins) that minimizes SSE.
    best_g : float
        Gain slope for best_shift.
    best_sse : float
        SSE for best_shift.
    best_r2 : float
        R^2 for best_shift (NaN if undefined).
    best_n : int
        Number of bins used in best fit.
    """
    best_shift = 0
    best_g = np.nan
    best_sse = np.inf
    best_r2 = np.nan
    best_n = 0

    for shift in range(-max_shift, max_shift + 1):
        x, y = _overlap_by_shift(y_light, y_dark, shift)

        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < min_bins:
            continue

        x = x[valid].astype(np.float64, copy=False)
        y = y[valid].astype(np.float64, copy=False)

        if float(np.sum(x)) <= eps or float(np.sum(y)) <= eps:
            continue

        g, sse, r2 = fit_gain_only(x, y, eps=eps)  # uses your existing helper

        if sse < best_sse:
            best_shift = shift
            best_g = g
            best_sse = sse
            best_r2 = r2
            best_n = int(x.size)

    if not np.isfinite(best_sse):
        return 0, np.nan, np.nan, np.nan, 0

    return best_shift, best_g, float(best_sse), best_r2, best_n


def compute_gain_via_overlap(
    tpf_by_trajectory,
    region,
    epoch_light,
    epoch_dark,
    trajectory="center_to_left",
    fr_map_dark=None,
    fr_threshold=0.5,
    *,
    eps: float = 1e-12,
    max_shift: int = 10,
    min_bins: int = 8,
):
    """
    Computes overlap metrics + fits:
      - gain-only:  y_dark ≈ g * y_light
      - affine:     y_dark ≈ a * y_light + b
      - shift+gain: y_dark[i] ≈ g * y_light[i - shift]
    """
    if (
        trajectory not in tpf_by_trajectory[region][epoch_light]
        or trajectory not in tpf_by_trajectory[region][epoch_dark]
    ):
        print(f"Missing trajectory {trajectory}")
        return pd.DataFrame()

    da_light = tpf_by_trajectory[region][epoch_light][trajectory]
    da_dark = tpf_by_trajectory[region][epoch_dark][trajectory]

    units_light = (
        da_light.coords["unit"].values
        if "unit" in da_light.coords
        else da_light.unit.values
    )
    units_dark = (
        da_dark.coords["unit"].values
        if "unit" in da_dark.coords
        else da_dark.unit.values
    )
    units = np.intersect1d(units_light, units_dark)

    rows = []

    for unit in units:
        # --- Filter by dark FR ---
        if fr_map_dark is not None:
            try:
                rate = (
                    fr_map_dark[unit]
                    if isinstance(fr_map_dark, (dict, pd.Series))
                    else fr_map_dark[int(unit)]
                )
                if float(rate) < fr_threshold:
                    continue
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        y_light = np.asarray(
            interpolate_nans(da_light.sel(unit=unit).values), dtype=np.float64
        )
        y_dark = np.asarray(
            interpolate_nans(da_dark.sel(unit=unit).values), dtype=np.float64
        )

        valid = np.isfinite(y_light) & np.isfinite(y_dark)
        if not np.any(valid):
            continue
        y_light = y_light[valid]
        y_dark = y_dark[valid]

        if float(np.sum(y_light)) <= eps or float(np.sum(y_dark)) <= eps:
            continue

        # --- overlaps ---
        raw_intersection = float(np.minimum(y_light, y_dark).sum())
        raw_union = float(np.maximum(y_light, y_dark).sum())
        raw_overlap = raw_intersection / (raw_union + eps)

        p_light = y_light / (float(np.sum(y_light)) + eps)
        p_dark = y_dark / (float(np.sum(y_dark)) + eps)
        shape_overlap = float(np.minimum(p_light, p_dark).sum())  # in [0, 1]

        peak_light = float(np.max(y_light))
        peak_dark = float(np.max(y_dark))
        gain_ratio_peak = peak_dark / (peak_light + eps)

        # --- model fits ---
        g0, sse_g0, r2_g0 = fit_gain_only(y_light, y_dark, eps=eps)
        a1, b1, sse_a1, r2_a1 = fit_affine(y_light, y_dark, eps=eps)

        shift, gs, sse_sg, r2_sg, n_sg = fit_shift_gain(
            y_light,
            y_dark,
            max_shift=max_shift,
            min_bins=min_bins,
            eps=eps,
        )

        # best model by SSE (simple and robust)
        sse_dict = {"gain": sse_g0, "affine": sse_a1, "shift_gain": sse_sg}
        best_model = min(sse_dict, key=sse_dict.get)

        rows.append(
            {
                "unit": unit,
                "shape_overlap": shape_overlap,
                "raw_overlap": raw_overlap,
                "sag": shape_overlap - raw_overlap,
                "gain_ratio_peak": gain_ratio_peak,
                # gain-only
                "gain_slope": g0,
                "gain_sse": sse_g0,
                "gain_r2": r2_g0,
                # affine
                "affine_slope": a1,
                "affine_intercept": b1,
                "affine_sse": sse_a1,
                "affine_r2": r2_a1,
                # shift+gain
                "shift_bins": shift,
                "shift_gain_slope": gs,
                "shift_gain_sse": sse_sg,
                "shift_gain_r2": r2_sg,
                "shift_gain_nbins": n_sg,
                # comparisons
                "delta_sse_gain_minus_affine": sse_g0 - sse_a1,
                "delta_sse_gain_minus_shiftgain": sse_g0 - sse_sg,
                "best_model_sse": best_model,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("unit")


# def compute_gain_via_overlap(
#     tpf_by_trajectory,
#     region,
#     epoch_light,
#     epoch_dark,
#     trajectory="center_to_left",
#     fr_map_dark=None,
#     fr_threshold=0.5,
# ):
#     """
#     Computes overlap metrics to distinguish Gain Modulation from Remapping.
#     Compares the SAME trajectory between Light and Dark.
#     """

#     # 1. Check Data Availability
#     if (
#         trajectory not in tpf_by_trajectory[region][epoch_light]
#         or trajectory not in tpf_by_trajectory[region][epoch_dark]
#     ):
#         print(f"Missing trajectory {trajectory}")
#         return pd.DataFrame()

#     da_light = tpf_by_trajectory[region][epoch_light][trajectory]
#     da_dark = tpf_by_trajectory[region][epoch_dark][trajectory]

#     # 2. Find Common Units
#     if "unit" in da_light.coords:
#         units = np.intersect1d(
#             da_light.coords["unit"].values, da_dark.coords["unit"].values
#         )
#     else:
#         units = np.intersect1d(da_light.unit.values, da_dark.unit.values)

#     results = []

#     for unit in units:
#         # --- Filter by Firing Rate (Optional but Recommended) ---
#         if fr_map_dark is not None:
#             # Check if unit exists in map and exceeds threshold
#             try:
#                 # Handle dict/series vs array
#                 rate = (
#                     fr_map_dark[unit]
#                     if isinstance(fr_map_dark, (dict, pd.Series))
#                     else fr_map_dark[int(unit)]
#                 )
#                 if rate < fr_threshold:
#                     continue
#             except:
#                 continue

#         # Interpolate NaNs here
#         y_light = interpolate_nans(da_light.sel(unit=unit).values)
#         y_dark = interpolate_nans(da_dark.sel(unit=unit).values)

#         # Skip silent cells (cannot calculate overlap if sum is 0)
#         if np.sum(y_light) == 0 or np.sum(y_dark) == 0:
#             continue

#         # --- Metric A: Raw Overlap (Sensitive to Rate Changes) ---
#         # Formula: Intersection / Union of RAW curves
#         raw_intersection = np.minimum(y_light, y_dark).sum()
#         raw_union = np.maximum(y_light, y_dark).sum()
#         raw_overlap = raw_intersection / raw_union if raw_union > 0 else 0

#         # --- Metric B: Shape Overlap (Normalized) ---
#         # Formula: Intersection of PROBABILITY distributions
#         p_light = y_light / np.sum(y_light)
#         p_dark = y_dark / np.sum(y_dark)

#         # Intersection of probabilities (Sum of mins)
#         # Note: Since sum(p1)=1 and sum(p2)=1, the max overlap is 1.0
#         shape_intersection = np.minimum(p_light, p_dark).sum()
#         # shape_union = np.maximum(p_light, p_dark).sum()

#         # shape_overlap = shape_intersection / shape_union if shape_union > 0 else 0
#         shape_overlap = shape_intersection

#         # --- Metric C: Gain Factor (Ratio of Peaks) ---
#         # To see direction of modulation (Enhancement vs Suppression)
#         peak_light = np.max(y_light)
#         peak_dark = np.max(y_dark)
#         gain_ratio = peak_dark / peak_light if peak_light > 0 else 0

#         results.append(
#             {
#                 "unit": unit,
#                 "shape_overlap": shape_overlap,
#                 "raw_overlap": raw_overlap,
#                 "gain_ratio": gain_ratio,
#                 "sag": shape_overlap - raw_overlap,  # Positive = Gain Modulation
#             }
#         )

#     return pd.DataFrame(results).set_index("unit")


def analyze_multitrajectory_gain(
    tpf_by_trajectory, region, epoch_light, epoch_dark, fr_threshold=0.5
):
    """
    Calculates Gain Modulation Index (GMI) for all 4 trajectories and visualizes consistency.
    """

    trajectories = [
        "center_to_left",
        "right_to_center",
        "center_to_right",
        "left_to_center",
    ]
    gmi_data = {}

    # 1. Calculate GMI for each trajectory separately
    for traj in trajectories:
        if traj not in tpf_by_trajectory[region][epoch_light]:
            continue

        da_light = tpf_by_trajectory[region][epoch_light][traj]
        da_dark = tpf_by_trajectory[region][epoch_dark][traj]

        # Get common units
        if "unit" in da_light.coords:
            units = np.intersect1d(
                da_light.coords["unit"].values, da_dark.coords["unit"].values
            )
        else:
            units = np.intersect1d(da_light.unit.values, da_dark.unit.values)

        traj_gmi = {}

        for unit in units:
            # Interpolate curves before finding peaks
            curve_light = interpolate_nans(da_light.sel(unit=unit).values)
            curve_dark = interpolate_nans(da_dark.sel(unit=unit).values)

            peak_light = np.max(curve_light)
            peak_dark = np.max(curve_dark)

            # FIX: Check for zero denominator explicitly
            if (peak_light + peak_dark) == 0:
                traj_gmi[unit] = np.nan
            elif max(peak_light, peak_dark) < fr_threshold:
                traj_gmi[unit] = np.nan
            else:
                # Calculate GMI (-1 to 1)
                gmi = (peak_dark - peak_light) / (peak_dark + peak_light)
                traj_gmi[unit] = gmi

        gmi_data[traj] = traj_gmi

    # 2. Create Master DataFrame
    df_gmi = pd.DataFrame(gmi_data)

    # Drop rows where ALL trajectories are NaN (silent cells)
    df_gmi.dropna(how="all", inplace=True)

    return df_gmi


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


def plot_task_similarity(comparison, region, turn_type, fig_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        comparison["sim_light"], comparison["sim_dark"], color="purple", alpha=0.6, s=50
    )
    ax.plot([-1, 1], [-1, 1], "k--", label="No Change")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.text(-0.6, 0.6, "Task Coding Emerges", color="green", ha="center")
    ax.text(0.6, -0.6, "Place Coding Dominates", color="gray", ha="center")

    ax.set_xlabel("Light Similarity")
    ax.set_ylabel("Dark Similarity")
    ax.set_title(f"{region} {turn_type} Coding Shift")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def _get_curve_x_and_dim(da: Any) -> tuple[np.ndarray, str]:
    """
    Get x-axis values (bin centers) and the position dimension name from an xarray-like DataArray.

    Parameters
    ----------
    da
        DataArray with dims including 'unit' and one position/bin dimension.

    Returns
    -------
    x : np.ndarray
        Bin coordinate values (or np.arange if unavailable).
    pos_dim : str
        Name of the non-unit dimension.
    """
    dims = list(getattr(da, "dims", []))
    pos_dim = next((d for d in dims if d != "unit"), None)
    if pos_dim is None:
        return np.arange(da.shape[-1]), "pos"

    coords = getattr(da, "coords", {})
    if pos_dim in coords:
        return np.asarray(coords[pos_dim].values), pos_dim
    return np.arange(da.sizes[pos_dim]), pos_dim


def predict_gain_only(
    y_light: np.ndarray,
    gain_slope: float,
) -> np.ndarray:
    """Predict dark curve under gain-only model: y_hat = g * y_light."""
    return gain_slope * np.asarray(y_light, dtype=float)


def predict_affine(
    y_light: np.ndarray,
    affine_slope: float,
    affine_intercept: float,
) -> np.ndarray:
    """Predict dark curve under affine model: y_hat = a * y_light + b."""
    y_light = np.asarray(y_light, dtype=float)
    return affine_slope * y_light + affine_intercept


def predict_shift_gain_non_circular(
    y_light: np.ndarray,
    shift_bins: int,
    gain_slope: float,
) -> np.ndarray:
    """
    Predict dark curve under non-circular shift+gain:
      y_hat[i] = g * y_light[i - shift_bins]
    Undefined edges are set to NaN.
    """
    y_light = np.asarray(y_light, dtype=float)
    y_hat = np.full_like(y_light, np.nan, dtype=float)
    n = y_light.size

    if shift_bins == 0:
        y_hat[:] = gain_slope * y_light
        return y_hat

    if shift_bins > 0:
        # i >= shift: use light[i - shift]
        y_hat[shift_bins:] = gain_slope * y_light[: n - shift_bins]
        return y_hat

    s = -shift_bins
    # i < n - s: use light[i + s]
    y_hat[: n - s] = gain_slope * y_light[s:]
    return y_hat


def plot_fit_summary(
    df_fit: pd.DataFrame,
    *,
    fig_path: Optional[Any] = None,
    title: str = "Light→Dark tuning: gain vs affine vs shift+gain",
) -> None:
    """
    Visualize summary statistics from compute_gain_via_overlap (three fits).

    Expects columns:
      - best_model_sse in {'gain','affine','shift_gain'}
      - gain_r2, affine_r2, shift_gain_r2
      - gain_slope, affine_slope, affine_intercept, shift_bins
      - delta_sse_gain_minus_affine, delta_sse_gain_minus_shiftgain

    Parameters
    ----------
    df_fit
        Output dataframe from compute_gain_via_overlap.
    fig_path
        If provided, saves figure here.
    title
        Figure title.
    """
    if df_fit.empty:
        return

    required = {
        "best_model_sse",
        "gain_r2",
        "affine_r2",
        "shift_gain_r2",
        "shift_bins",
        "delta_sse_gain_minus_affine",
        "delta_sse_gain_minus_shiftgain",
    }
    missing = required - set(df_fit.columns)
    if missing:
        raise ValueError(f"df_fit missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax0, ax1, ax2, ax3 = axes.ravel()

    # (A) Best model counts
    counts = (
        df_fit["best_model_sse"]
        .value_counts()
        .reindex(["gain", "affine", "shift_gain"])
    )
    ax0.bar(counts.index, counts.values)
    ax0.set_ylabel("Count")
    ax0.set_title("Best model by SSE")

    # (B) R2 comparisons (gain vs affine; gain vs shift+gain)
    x = df_fit["gain_r2"].to_numpy()
    y_aff = df_fit["affine_r2"].to_numpy()
    y_sg = df_fit["shift_gain_r2"].to_numpy()

    ax1.scatter(x, y_aff, alpha=0.35, label="affine vs gain")
    ax1.scatter(x, y_sg, alpha=0.35, label="shift+gain vs gain", marker="x")
    lim_lo = np.nanmin([x, y_aff, y_sg])
    lim_hi = np.nanmax([x, y_aff, y_sg])
    ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1)
    ax1.set_xlabel("gain_r2")
    ax1.set_ylabel("affine_r2 / shift_gain_r2")
    ax1.set_title("Fit quality (R²)")
    ax1.legend(frameon=False)

    # (C) SSE deltas: which model improves over gain?
    # Positive delta => gain SSE larger (worse) than comparator.
    ax2.scatter(
        df_fit["delta_sse_gain_minus_affine"],
        df_fit["delta_sse_gain_minus_shiftgain"],
        alpha=0.35,
    )
    ax2.axhline(0, color="k", lw=1, alpha=0.6)
    ax2.axvline(0, color="k", lw=1, alpha=0.6)
    ax2.set_xlabel("SSE(gain) - SSE(affine)")
    ax2.set_ylabel("SSE(gain) - SSE(shift+gain)")
    ax2.set_title("Improvement over gain (SSE deltas)")

    # (D) Shift distribution (only where shift+gain is best)
    df_sg = df_fit[df_fit["best_model_sse"] == "shift_gain"]
    if len(df_sg) > 0:
        ax3.hist(df_sg["shift_bins"].dropna().to_numpy(), bins=25)
        ax3.set_title("Shift bins (only units best-fit by shift+gain)")
        ax3.set_xlabel("shift_bins")
        ax3.set_ylabel("Count")
    else:
        ax3.text(0.5, 0.5, "No shift+gain winners", ha="center", va="center")
        ax3.set_axis_off()

    fig.suptitle(title)
    fig.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _pick_example_units(
    df_fit: pd.DataFrame,
    *,
    n_per_model: int = 3,
    min_r2: float = 0.2,
) -> dict[str, list[Any]]:
    """
    Pick representative units for each best_model_sse category.
    """
    out: dict[str, list[Any]] = {}
    for model in ["gain", "affine", "shift_gain"]:
        sub = df_fit[df_fit["best_model_sse"] == model].copy()
        if sub.empty:
            out[model] = []
            continue

        # choose ranking by the corresponding R2
        r2_col = {
            "gain": "gain_r2",
            "affine": "affine_r2",
            "shift_gain": "shift_gain_r2",
        }[model]
        sub = sub[np.isfinite(sub[r2_col]) & (sub[r2_col] >= min_r2)]
        sub = sub.sort_values(r2_col, ascending=False)
        out[model] = list(sub.head(n_per_model).index.values)
    return out


def plot_example_unit_fits(
    tpf_by_trajectory: dict[str, Any],
    region: str,
    epoch_light: Any,
    epoch_dark: Any,
    trajectory: str,
    df_fit: pd.DataFrame,
    *,
    units: Optional[Iterable[Any]] = None,
    n_per_model: int = 2,
    fig_path: Optional[Any] = None,
    title_prefix: str = "",
) -> None:
    """
    Plot tuning curves for example units, overlaying predictions from:
      - gain-only
      - affine
      - shift+gain (non-circular)

    Parameters
    ----------
    tpf_by_trajectory
        Dict[region][epoch][trajectory] -> DataArray(unit, pos_dim).
    region, epoch_light, epoch_dark, trajectory
        Keys to select tuning curves.
    df_fit
        Output dataframe from compute_gain_via_overlap (must include fit params).
    units
        If provided, plots these units. Otherwise auto-selects per best_model_sse.
    n_per_model
        If auto-selecting, number of units per model type.
    fig_path
        If provided, saves figure here.
    title_prefix
        Optional prefix for the figure title.
    """
    da_light = tpf_by_trajectory[region][epoch_light][trajectory]
    da_dark = tpf_by_trajectory[region][epoch_dark][trajectory]
    x, _ = _get_curve_x_and_dim(da_light)

    if units is None:
        picked = _pick_example_units(df_fit, n_per_model=n_per_model)
        units = [u for model in ["gain", "affine", "shift_gain"] for u in picked[model]]

    units = list(units)
    if len(units) == 0:
        return

    n_rows = len(units)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, unit in zip(axes, units):
        if unit not in df_fit.index:
            continue

        y_light = np.asarray(
            interpolate_nans(da_light.sel(unit=unit).values), dtype=float
        )
        y_dark = np.asarray(
            interpolate_nans(da_dark.sel(unit=unit).values), dtype=float
        )

        # predictions
        g0 = float(df_fit.loc[unit, "gain_slope"])
        a1 = float(df_fit.loc[unit, "affine_slope"])
        b1 = float(df_fit.loc[unit, "affine_intercept"])
        sh = int(df_fit.loc[unit, "shift_bins"])
        gs = float(df_fit.loc[unit, "shift_gain_slope"])

        y_hat_gain = predict_gain_only(y_light, g0)
        y_hat_aff = predict_affine(y_light, a1, b1)
        y_hat_sg = predict_shift_gain_non_circular(y_light, sh, gs)

        best = str(df_fit.loc[unit, "best_model_sse"])

        ax.plot(x, y_light, lw=2, label="light")
        ax.plot(x, y_dark, lw=2, label="dark")
        ax.plot(x, y_hat_gain, lw=1.5, linestyle="--", label="gain-only pred")
        ax.plot(x, y_hat_aff, lw=1.5, linestyle="--", label="affine pred")
        ax.plot(x, y_hat_sg, lw=1.5, linestyle="--", label="shift+gain pred")

        ax.set_ylabel("FR")
        ax.set_title(
            f"unit={unit} | best={best} | "
            f"gain_r2={df_fit.loc[unit,'gain_r2']:.2f}, "
            f"affine_r2={df_fit.loc[unit,'affine_r2']:.2f}, "
            f"shift_r2={df_fit.loc[unit,'shift_gain_r2']:.2f} | "
            f"shift={sh}"
        )
        ax.legend(ncols=3, frameon=False)

    axes[-1].set_xlabel("Task progression bin")
    fig.suptitle(f"{title_prefix}{region} {trajectory}: example unit fits")
    fig.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_gain_overlap(df_overlap, region, trajectory, fig_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot A: Sag
    sns.scatterplot(
        data=df_overlap,
        x="shape_overlap",
        y="raw_overlap",
        color="purple",
        alpha=0.5,
        ax=ax[0],
    )
    ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set_title(f"Gain Modulation: {trajectory}")

    # Plot B: Ratio
    stable_cells = df_overlap[df_overlap["shape_overlap"] > 0.6]
    if not stable_cells.empty:
        sns.histplot(
            stable_cells["gain_ratio_peak"],
            bins=np.logspace(-1, 1, 30),
            ax=ax[1],
            color="teal",
        )
        ax[1].set_xscale("log")
        ax[1].axvline(1, color="k", linestyle="--")
        ax[1].set_title(f"Gain Direction (Stable n={len(stable_cells)})")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_multitrajectory_gain(df_gmi, region, turn_type, trajectory_pair, fig_dir):
    """
    Plots consistency between the two trajectories in the pair (e.g., center_to_left vs right_to_center).
    """
    if df_gmi.empty:
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot A: Consistency Scatter
    # Check if both trajectories exist in the dataframe
    if trajectory_pair[0] in df_gmi.columns and trajectory_pair[1] in df_gmi.columns:
        common = df_gmi.dropna(subset=trajectory_pair)
        if not common.empty:
            sns.scatterplot(
                data=common, x=trajectory_pair[0], y=trajectory_pair[1], ax=ax[0]
            )
            ax[0].plot([-1, 1], [-1, 1], "k--")
            ax[0].set_title(f"Gain Consistency: {turn_type}")
            ax[0].set_xlabel(trajectory_pair[0])
            ax[0].set_ylabel(trajectory_pair[1])

    # Plot B: Heatmap
    if not df_gmi.empty:
        df_sorted = (
            df_gmi.assign(mean_gmi=df_gmi.mean(axis=1))
            .sort_values("mean_gmi", ascending=False)
            .drop(columns="mean_gmi")
        )
        sns.heatmap(
            df_sorted,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            yticklabels=False,
            ax=ax[1],
        )
        ax[1].set_title(f"Population Gain ({turn_type})")

    plt.tight_layout()
    plt.savefig(fig_dir / f"{region}_{turn_type}_multitrajectory_gain.png", dpi=300)
    plt.close(fig)


def plot_MI(
    place_si_corrected,
    tp_si_corrected,
    region,
    epoch1,
    epoch2,
    fr_threshold,
    fig_path,
    mode="bits/spike",
):
    mask_dark_active = fr_during_movement[region][run_epoch_list[3]] > fr_threshold

    fig, ax = plt.subplots()
    for epoch in [epoch1, epoch2]:
        psi_to_plot = place_si_corrected[region][epoch][mode][mask_dark_active]
        tsi_to_plot = tp_si_corrected[region][epoch][mode][mask_dark_active]
        ax.scatter(psi_to_plot, tsi_to_plot, alpha=0.2, label=epoch)

    ax.set_xlabel("Place MI (bits/spike)")
    ax.set_ylabel("Task progression MI (bits/spike)")
    ax.legend()

    ax.plot([-2, 0], [0, 0], "k--")
    ax.plot([0, 0], [-2, 0], "k--")
    ax.plot([0, 3], [0, 3], "k--")

    ax.set_xlim([-0.5, 3])
    ax.set_ylim([-0.5, 3])

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


# --- MAIN ---


def main():
    data_dir = analysis_path / "tp_tuning"
    data_dir.mkdir(parents=True, exist_ok=True)
    fr_threshold = {"v1": 0.5, "ca1": 0}
    light_epoch = run_epoch_list[0]
    dark_epoch = run_epoch_list[3]

    for region in regions:
        # Create FR map for this region
        fr_map_dark = pd.Series(
            fr_during_movement[region][dark_epoch], index=spikes[region].keys()
        )

        # Iterate strictly by Turn Type
        for turn_type in ["left", "right"]:
            print(f"--- Processing {region}: {turn_type} turn ---")

            # Define the specific trajectories for this turn
            if turn_type == "left":
                traj_pair = ["center_to_left", "right_to_center"]
            else:
                traj_pair = ["center_to_right", "left_to_center"]

            # 1. TASK SIMILARITY (Light vs Dark)
            df_light = calculate_task_similarity(
                tpf_by_trajectory,
                region,
                light_epoch,
                fr_map_dark,
                fr_threshold[region],
                turn_type,
            )
            df_light.to_parquet(
                data_dir / f"{region}_{light_epoch}_{turn_type}_tpf_similarity.parquet"
            )

            df_dark = calculate_task_similarity(
                tpf_by_trajectory,
                region,
                dark_epoch,
                fr_map_dark,
                fr_threshold[region],
                turn_type,
            )
            df_dark.to_parquet(
                data_dir / f"{region}_{dark_epoch}_{turn_type}_tpf_similarity.parquet"
            )

            if not df_light.empty and not df_dark.empty:
                comparison = df_light.rename(columns={"similarity": "sim_light"}).join(
                    df_dark.rename(columns={"similarity": "sim_dark"}), how="inner"
                )
                plot_task_similarity(
                    comparison,
                    region,
                    turn_type,
                    fig_path=fig_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{turn_type}_task_similarity.png",
                )

            # 2. GAIN OVERLAP (Per Trajectory in this Turn)
            for traj in traj_pair:
                df_overlap = compute_gain_via_overlap(
                    tpf_by_trajectory,
                    region,
                    light_epoch,
                    dark_epoch,
                    traj,
                    fr_map_dark,
                    fr_threshold[region],
                )
                df_overlap.to_parquet(
                    data_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{traj}_overlap.parquet"
                )

                if not df_overlap.empty:
                    plot_gain_overlap(
                        df_overlap,
                        region,
                        traj,
                        fig_path=fig_dir
                        / f"{region}_{light_epoch}_{dark_epoch}_{traj}_gain_overlap.png",
                    )
                # After df_overlap is computed (per region/trajectory):

                plot_fit_summary(
                    df_overlap,
                    fig_path=fig_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{traj}_fit_summary.png",
                    title=f"{region} {traj}: fit summary ({light_epoch} → {dark_epoch})",
                )

                plot_example_unit_fits(
                    tpf_by_trajectory,
                    region,
                    light_epoch,
                    dark_epoch,
                    traj,
                    df_overlap,
                    n_per_model=2,
                    fig_path=fig_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{traj}_example_fits.png",
                    title_prefix="",
                )

            # 3. MULTI-TRAJECTORY GAIN (Consistency within this Turn)
            # Only analyze the relevant trajectories for this turn
            # df_gmi = analyze_multitrajectory_gain(
            #     tpf_by_trajectory,
            #     region,
            #     light_epoch,
            #     dark_epoch,
            #     fr_threshold=fr_threshold[region],
            # )

            # if not df_gmi.empty:
            #     plot_multitrajectory_gain(df_gmi, region, turn_type, traj_pair, fig_dir)
        ll_bits_per_spike_place = {}
        ll_bits_per_spike_tp = {}

        for epoch in run_epoch_list:
            ll_bits_per_spike_place[epoch] = {}
            ll_bits_per_spike_tp[epoch] = {}

            valid_epoch = movement[epoch].time_support

            for unit_id in spikes[region].index:

                unit_ts = spikes[region][unit_id]

                # Calculate
                ll_bits_per_spike_place[epoch][unit_id] = (
                    calculate_bits_per_spike_pynapple(
                        unit_spikes=unit_ts,
                        position=linear_position[
                            epoch
                        ],  # Pass the full Tsd, the function restricts it
                        tuning_curve=smoothed_pf[region][epoch][unit_id].to_numpy(),
                        bin_edges=smoothed_pf[region][epoch][unit_id].bin_edges[0],
                        epoch=valid_epoch,
                    )
                )

                # print(f"Information: {score:.4f} bits/spike")

                # Calculate
                ll_bits_per_spike_tp[epoch][unit_id] = (
                    calculate_bits_per_spike_pynapple(
                        unit_spikes=unit_ts,
                        position=task_progression[
                            epoch
                        ],  # Pass the full Tsd, the function restricts it
                        tuning_curve=smoothed_tpf[region][epoch][unit_id].to_numpy(),
                        bin_edges=smoothed_tpf[region][epoch][unit_id].bin_edges[0],
                        epoch=valid_epoch,
                    )
                )

        with open(data_dir / f"{region}_ll_bits_per_spike_place.pkl", "wb") as f:
            pickle.dump(ll_bits_per_spike_place, f)

        with open(data_dir / f"{region}_ll_bits_per_spike_tp.pkl", "wb") as f:
            pickle.dump(ll_bits_per_spike_tp, f)

    place_si_corrected = {}

    print("Computing Place SI (Shuffle Corrected)...")
    for region in regions:
        place_si_corrected[region] = {}

        for epoch in run_epoch_list:
            print(f"Region: {region}, Epoch: {epoch}")

            # B. Compute Shuffled SI (Chance Level)
            # We pass the SAME movement epochs so shifts happen within valid run time
            mean_shuffle = compute_shuffled_si(
                region=region,
                epoch=epoch,
                feature=linear_position[epoch],
                bins=position_bins,
                n_shuffles=50,
            )

            # C. Subtract Bias
            # (Raw - Chance)
            place_si_corrected[region][epoch] = place_si[region][epoch] - mean_shuffle

    with open(data_dir / "place_si_corrected.pkl", "wb") as f:
        pickle.dump(place_si_corrected, f)

    tp_si_corrected = {}

    print("Computing Task Progression SI (Shuffle Corrected)...")
    for region in regions:
        tp_si_corrected[region] = {}

        for epoch in run_epoch_list:
            print(f"Region: {region}, Epoch: {epoch}")

            # B. Compute Shuffle
            mean_shuffle = compute_shuffled_si(
                region=region,
                epoch=epoch,
                feature=task_progression[epoch],
                bins=task_progression_bins,
                n_shuffles=50,
            )

            # C. Subtract Bias
            tp_si_corrected[region][epoch] = tp_si[region][epoch] - mean_shuffle

    with open(data_dir / "tp_si_corrected.pkl", "wb") as f:
        pickle.dump(tp_si_corrected, f)

    for region in regions:
        plot_MI(
            place_si_corrected,
            tp_si_corrected,
            region,
            light_epoch,
            dark_epoch,
            fr_threshold=fr_threshold[region],
            fig_path=fig_dir / f"{region}_{light_epoch}_{dark_epoch}_mi_comparison.png",
            mode="bits/spike",
        )


if __name__ == "__main__":
    main()
