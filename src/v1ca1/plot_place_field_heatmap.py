import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import scipy
import position_tools as pt
import spikeinterface.full as si
import kyutils
import numpy as np
import track_linearization as tl


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


import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_pf_along_position(pf, pos_dim: str, sigma_bins: float):
    pf = pf.fillna(0)  # Replaces NaN values with 0
    # gaussian_filter1d works on numpy; apply along the position axis
    axis = pf.get_axis_num(pos_dim)
    sm = gaussian_filter1d(pf.values, sigma=sigma_bins, axis=axis, mode="nearest")
    return pf.copy(data=sm)


def _sampling_rate(t_position: np.ndarray) -> float:
    return (len(t_position) - 1) / (t_position[-1] - t_position[0])


spikes = {}
for region in regions:
    spikes[region] = get_tsgroup(sorting[region])


def plot_place_field_heatmap_trajectories(region, epoch, smoothing_bins=None):

    fig_save_dir = analysis_path / "figs" / "place_field_heatmap"
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    # intervals
    # all_ep = nap.IntervalSet(
    #     start=timestamps_position_dict[epoch][position_offset],
    #     end=timestamps_position_dict[epoch][-1],
    # )
    # ep_duration = (
    #     timestamps_position_dict[epoch][-1]
    #     - timestamps_position_dict[epoch][position_offset]
    # )
    # first_half_ep = nap.IntervalSet(
    #     start=timestamps_position_dict[epoch][position_offset],
    #     end=timestamps_position_dict[epoch][-1] - ep_duration / 2,
    # )
    # second_half_ep = nap.IntervalSet(
    #     start=timestamps_position_dict[epoch][position_offset] + ep_duration / 2,
    #     end=timestamps_position_dict[epoch][-1],
    # )

    trajectory_ep = {}
    for trajectory_type in trajectory_types:
        trajectory_ep[trajectory_type] = nap.IntervalSet(
            start=trajectory_times[epoch][trajectory_type][:, 0],
            end=trajectory_times[epoch][trajectory_type][:, -1],
        )

    # trajectory_first_half_ep = {}
    # for trajectory_type in trajectory_types:
    #     i = int(len(trajectory_times[epoch][trajectory_type]) / 2)
    #     trajectory_first_half_ep[trajectory_type] = nap.IntervalSet(
    #         start=trajectory_times[epoch][trajectory_type][:i, 0],
    #         end=trajectory_times[epoch][trajectory_type][:i, -1],
    #     )

    # trajectory_second_half_ep = {}
    # for trajectory_type in trajectory_types:
    #     i = int(len(trajectory_times[epoch][trajectory_type]) / 2)
    #     trajectory_second_half_ep[trajectory_type] = nap.IntervalSet(
    #         start=trajectory_times[epoch][trajectory_type][i:, 0],
    #         end=trajectory_times[epoch][trajectory_type][i:, -1],
    #     )

    dx = 9.5
    dy = 9
    diagonal_segment_length = np.sqrt(dx**2 + dy**2)

    long_segment_length = 81 - 17 - 2
    short_segment_length = 13.5

    node_positions_left = np.array(
        [
            (55.5, 81),  # center well
            (55.5, 81 - long_segment_length),
            (55.5 - dx, 81 - long_segment_length - dy),
            (55.5 - dx - short_segment_length, 81 - long_segment_length - dy),
            (55.5 - 2 * dx - short_segment_length, 81 - long_segment_length),
            (55.5 - 2 * dx - short_segment_length, 81),
        ]
    )

    node_positions_right = np.array(
        [
            (55.5, 81),  # center well
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

    linear_position = {}
    for trajectory_type in trajectory_types:
        if trajectory_type in ["center_to_right", "right_to_center"]:
            tg = track_graph_left
        else:
            tg = track_graph_right
        position_df = tl.get_linearized_position(
            position=position_dict[epoch][position_offset:],
            track_graph=tg,
            edge_order=linear_edge_order,
            edge_spacing=linear_edge_spacing,
        )
        linear_position[trajectory_type] = nap.Tsd(
            t=timestamps_position_dict[epoch][position_offset:],
            d=position_df["linear_position"],
            time_support=trajectory_ep[trajectory_type],
        )

    speed = pt.get_speed(
        position=position_dict[epoch][position_offset:],
        time=timestamps_position_dict[epoch][position_offset:],
        sampling_frequency=_sampling_rate(
            timestamps_position_dict[epoch][position_offset:]
        ),
        sigma=0.1,
    )
    speed_tsd = nap.Tsd(t=timestamps_position_dict[epoch][position_offset:], d=speed)
    movement = speed_tsd.threshold(speed_threshold, method="above")

    # 4x4 grid:
    # - rows = which trajectory_type defines the ordering (from pf1 of that row's trajectory)
    # - cols = which trajectory_type's pf2 is being plotted
    #
    # Important: we compute pf1/pf2 once per trajectory_type, then reuse.
    n_traj = len(trajectory_types)

    # -------------------------
    # Precompute pf1/pf2 for all trajectory types
    # -------------------------
    pf1_by_traj = {}
    pf2_by_traj = {}

    for traj in trajectory_types:
        if smoothing_bins is None:
            pf1_by_traj[traj] = nap.compute_tuning_curves(
                data=spikes[region],
                features=linear_position[traj],
                epochs=trajectory_ep[traj][::2].intersect(movement.time_support),
                bins=40,
                feature_names=["linear position (cm)"],
            )

            pf2_by_traj[traj] = nap.compute_tuning_curves(
                data=spikes[region],
                features=linear_position[traj],
                epochs=trajectory_ep[traj][1::2].intersect(movement.time_support),
                bins=40,
                feature_names=["linear position (cm)"],
            )
        else:
            pf1_raw = nap.compute_tuning_curves(
                data=spikes[region],
                features=linear_position[traj],
                epochs=trajectory_ep[traj][::2].intersect(movement.time_support),
                bins=80,
                feature_names=["linear position (cm)"],
            )

            pf2_raw = nap.compute_tuning_curves(
                data=spikes[region],
                features=linear_position[traj],
                epochs=trajectory_ep[traj][1::2].intersect(movement.time_support),
                bins=80,
                feature_names=["linear position (cm)"],
            )

            # get dim names once
            unit_dim, pos_dim = pf1_raw.dims

            # ---- smooth pf1 for ORDERING ----
            pf1_by_traj[traj] = smooth_pf_along_position(
                pf1_raw, pos_dim, smoothing_bins
            )

            # ---- smooth pf2 for DISPLAY (optional, comment out if you want raw) ----
            pf2_by_traj[traj] = smooth_pf_along_position(
                pf2_raw, pos_dim, smoothing_bins
            )

    # -------------------------
    # Compute unit ordering per trajectory type (based on pf1 peak position)
    # -------------------------
    order_by_traj = {}
    for traj in trajectory_types:
        pf1 = pf1_by_traj[traj]
        unit_dim, pos_dim = pf1.dims

        pf1_max = pf1.max(dim=pos_dim, skipna=True).where(lambda x: x > 0)
        pf1_norm = pf1 / pf1_max

        peak_pos = pf1_norm.fillna(-np.inf).idxmax(dim=pos_dim)
        order_by_traj[traj] = peak_pos.argsort().data  # plain integer indices

    # -------------------------
    # Plot: each row uses one ordering for all columns
    # -------------------------
    fig, ax = plt.subplots(
        nrows=n_traj,
        ncols=n_traj,
        figsize=(24, 12),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    for r, order_traj in enumerate(trajectory_types):
        pf1_ref = pf1_by_traj[order_traj]
        unit_dim, pos_dim = pf1_ref.dims
        unit_order = order_by_traj[order_traj]

        for c, plot_traj in enumerate(trajectory_types):
            pf2 = pf2_by_traj[plot_traj]

            # align pf2's units to the reference unit labels (pf1_ref) so ordering is consistent
            pf2_aligned = pf2.sel({unit_dim: pf1_ref[unit_dim]})
            pf2_sorted = pf2_aligned.isel({unit_dim: unit_order})

            # normalize per unit (safe)
            pf2_max = pf2_sorted.max(dim=pos_dim, skipna=True).where(lambda x: x > 0)
            pf2_norm = pf2_sorted / pf2_max

            # ensure axes: y=units, x=position
            pf2_norm = pf2_norm.transpose(unit_dim, pos_dim)

            # clean y-axis ticks (0..n_units-1) but keep x coordinate
            pf2_norm = pf2_norm.assign_coords(
                {unit_dim: np.arange(pf2_norm.sizes[unit_dim])}
            )

            pf2_norm.plot.imshow(origin="upper", ax=ax[r, c], add_colorbar=False)

            if r == 0:
                ax[r, c].set_title(f"pf2: {plot_traj}")
            if c == 0:
                ax[r, c].set_ylabel(f"order: {order_traj}")

    for a in ax.flatten():
        a.axvline(
            long_segment_length + diagonal_segment_length * 0.5,
            color="black",
            alpha=0.5,
        )
        a.axvline(
            long_segment_length + diagonal_segment_length * 1.5 + short_segment_length,
            color="black",
            alpha=0.5,
        )

    # one shared colorbar (optional)
    mappable = ax[0, 0].images[0]
    fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.01)

    fig.suptitle(f"{animal_name} {date} {region} {epoch}")

    fig.savefig(
        fig_save_dir / f"{region}_{epoch}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return None


def main():
    for region in regions:
        for epoch in run_epoch_list:
            plot_place_field_heatmap_trajectories(region, epoch, smoothing_bins=1.5)


if __name__ == "__main__":
    main()
