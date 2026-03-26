import replay_trajectory_classification as rtc
import numpy as np
import kyutils
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import position_tools as pt
import track_linearization as tl
import scipy

animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / "20240611"

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

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

speed_threshold = 4  # cm/s


def unwrap_degrees(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap angle sequence in degrees, originally in range [-180, 180].

    Parameters
    ----------
    angles : np.ndarray
        Array of angles (n_time,) in degrees.

    Returns
    -------
    np.ndarray
        Unwrapped angle sequence (n_time,) in degrees.
    """
    angles = np.asarray(angles)
    diffs = np.diff(angles)
    diffs = np.where(diffs > 180, diffs - 360, diffs)
    diffs = np.where(diffs < -180, diffs + 360, diffs)
    return np.concatenate(([angles[0]], angles[0] + np.cumsum(diffs)))


def get_track_graph():
    # track graphs
    long_segment_length = 71
    short_segment_length = 32

    total_length = long_segment_length * 2 + short_segment_length

    x = np.arange(0, total_length, step=1)

    node_positions_left = np.array(
        [
            (55, 81),  # center well
            (55 - short_segment_length, 81),  # left well
            (55, 81 - long_segment_length),  # center junction
            (55 - short_segment_length, 81 - long_segment_length),  # left junction
        ]
    )
    node_positions_right = np.array(
        [
            (55, 81),  # center well
            # (55-short_segment_length, 81),  # left well
            (55 + short_segment_length, 81),  # right well
            (55, 81 - long_segment_length),  # center junction
            # (55-short_segment_length, 81-long_segment_length),  # left junction
            (55 + short_segment_length, 81 - long_segment_length),  # right junction
        ]
    )

    edges = np.array(
        [
            (0, 2),
            (2, 3),
            (3, 1),
        ]
    )

    linear_edge_order = [
        (0, 2),
        (2, 3),
        (3, 1),
    ]
    linear_edge_spacing = 0
    track_graph_left = tl.make_track_graph(node_positions_left, edges)
    track_graph_right = tl.make_track_graph(node_positions_right, edges)

    return (
        track_graph_left,
        track_graph_right,
        linear_edge_order,
        linear_edge_spacing,
        x,
    )


def get_track_graph_new():

    dx = 9.5
    dy = 9

    long_segment_length = 81 - 17 - 2
    short_segment_length = 13.5
    diagonal_segment_length = np.sqrt(dx**2 + dy**2)

    total_length = (
        long_segment_length * 2 + short_segment_length + diagonal_segment_length * 2
    )

    x = np.arange(0, total_length, step=1)

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

    return (
        track_graph_left,
        track_graph_right,
        linear_edge_order,
        linear_edge_spacing,
        x,
    )


def plot_motor_vars_given_place():

    fig_save_path = analysis_path / "figs"
    fig_save_path.mkdir(parents=True, exist_ok=True)

    track_graph_left, track_graph_right, linear_edge_order, linear_edge_spacing, x = (
        get_track_graph_new()
    )

    fig, ax = plt.subplots(
        nrows=len(run_epoch_list) * 2 + 1,
        ncols=4,
        figsize=(16, 16),
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5, 1, 1, 1, 1]},
        squeeze=False,
    )

    for trajectory_type in trajectory_types:
        if trajectory_type == "center_to_left":
            ax_row_offset = 0
            color = "blue"
            track_graph = track_graph_right
        elif trajectory_type == "left_to_center":
            ax_row_offset = len(run_epoch_list) + 1
            track_graph = track_graph_right
            color = "green"
        elif trajectory_type == "center_to_right":
            ax_row_offset = 0
            color = "red"
            track_graph = track_graph_left
        elif trajectory_type == "right_to_center":
            ax_row_offset = len(run_epoch_list) + 1
            track_graph = track_graph_left
            color = "purple"

        for epoch_idx, epoch in enumerate(run_epoch_list):
            timestamps_position = timestamps_position_dict[epoch][position_offset:]

            position = position_dict[epoch][position_offset:]
            position_body = body_position_dict[epoch][position_offset:]

            position_sampling_rate = len(position) / (
                timestamps_position[-1] - timestamps_position[0]
            )
            speed = pt.get_speed(
                position,
                time=timestamps_position,
                sampling_frequency=position_sampling_rate,
                sigma=0.1,
            )

            # head_direction = np.rad2deg(
            #     np.arctan2(
            #         position_body[:, 1] - position[:, 1],
            #         position_body[:, 0] - position[:, 0],
            #     )
            # )
            # head_direction = (head_direction + 90) % 360

            head_direction = np.rad2deg(pt.get_angle(position_body, position))
            head_direction_unwrapped = unwrap_degrees(head_direction)
            head_direction_velocity = np.diff(
                head_direction_unwrapped, prepend=head_direction_unwrapped[0]
            )
            head_direction_velocity = pt.core.gaussian_smooth(
                head_direction_velocity,
                sigma=0.1,
                sampling_frequency=position_sampling_rate,
            )
            head_direction_speed = np.abs(head_direction_velocity)

            speed_list = []
            headdir_list = []
            headdir_velocity_list = []
            headdir_speed_list = []

            for trajectory_ind, (
                trajectory_start_time,
                trajectory_end_time,
            ) in enumerate(trajectory_times[epoch][trajectory_type]):
                mask = (
                    (timestamps_position > trajectory_start_time)
                    & (timestamps_position <= trajectory_end_time)
                    & (speed > speed_threshold)
                )

                if not np.any(mask):
                    continue

                position_df = tl.get_linearized_position(
                    position=position[mask],
                    track_graph=track_graph,
                    edge_order=linear_edge_order,
                    edge_spacing=linear_edge_spacing,
                )
                linear_pos = position_df["linear_position"].to_numpy()

                f_speed = scipy.interpolate.interp1d(
                    linear_pos,
                    speed[mask],
                    axis=0,
                    bounds_error=False,
                    kind="linear",
                )

                f_headdir = scipy.interpolate.interp1d(
                    linear_pos,
                    head_direction[mask],
                    axis=0,
                    bounds_error=False,
                    kind="linear",
                )

                f_headdir_velocity = scipy.interpolate.interp1d(
                    linear_pos,
                    head_direction_velocity[mask],
                    axis=0,
                    bounds_error=False,
                    kind="linear",
                )

                f_headdir_speed = scipy.interpolate.interp1d(
                    linear_pos,
                    head_direction_speed[mask],
                    axis=0,
                    bounds_error=False,
                    kind="linear",
                )

                speed_interp = f_speed(x)
                speed_list.append(speed_interp)

                headdir_interp = f_headdir(x)
                headdir_list.append(headdir_interp)

                headdir_velocity_interp = f_headdir_velocity(x)
                headdir_velocity_list.append(headdir_velocity_interp)

                headdir_speed_interp = f_headdir_speed(x)
                headdir_speed_list.append(headdir_speed_interp)

                # headdir_diff_interp = np.diff(
                #     headdir_interp,
                #     prepend=headdir_interp[0],
                # )
                # headdir_diff_interp = (headdir_diff_interp + 180) % 360 - 180
                # headdir_diff_interp = headdir_diff_interp * position_sampling_rate

                # headdir_diff_list.append(headdir_diff_interp)

                ax[ax_row_offset + epoch_idx, 0].plot(
                    x,
                    speed_interp,
                    linewidth=1.0,
                    color=color,
                    alpha=0.3,
                )

                ax[ax_row_offset + epoch_idx, 1].plot(
                    x,
                    headdir_interp,
                    linewidth=1.0,
                    color=color,
                    alpha=0.3,
                )

                ax[ax_row_offset + epoch_idx, 2].plot(
                    x,
                    headdir_velocity_interp,
                    linewidth=1.0,
                    color=color,
                    alpha=0.3,
                )

                ax[ax_row_offset + epoch_idx, 3].plot(
                    x,
                    headdir_speed_interp,
                    linewidth=1.0,
                    color=color,
                    alpha=0.3,
                )

            ax[ax_row_offset + epoch_idx, 0].plot(
                x,
                np.mean(speed_list, axis=0),
                linewidth=2,
                color=color,
                label=trajectory_type,
            )

            ax[ax_row_offset + epoch_idx, 1].plot(
                x,
                np.mean(headdir_list, axis=0),
                linewidth=2,
                color=color,
                label=trajectory_type,
            )
            ax[ax_row_offset + epoch_idx, 2].plot(
                x,
                np.mean(headdir_velocity_list, axis=0),
                linewidth=2,
                color=color,
                label=trajectory_type,
            )
            ax[ax_row_offset + epoch_idx, 2].plot(
                x,
                np.mean(headdir_speed_list, axis=0),
                linewidth=2,
                color=color,
                label=trajectory_type,
            )

            # ax[ax_row_offset + epoch_idx, ax_col_offset].set_yticklabels("")

            # ax[ax_row_offset + epoch_idx, ax_col_offset].set_xticks([0, 180, 360])

            if epoch_idx != 3:
                ax[ax_row_offset + epoch_idx, 0].set_xticklabels("")
                ax[ax_row_offset + epoch_idx, 1].set_xticklabels("")
                ax[ax_row_offset + epoch_idx, 2].set_xticklabels("")
                ax[ax_row_offset + epoch_idx, 3].set_xticklabels("")

            # else:
            #     ax[ax_row_offset + epoch_idx, ax_col_offset].set_xticklabels(
            #         [0, 180, 360]
            #     )

            ax[ax_row_offset + epoch_idx, 0].set_xlim([x[0], x[-1]])
            ax[ax_row_offset + epoch_idx, 1].set_xlim([x[0], x[-1]])
            ax[ax_row_offset + epoch_idx, 2].set_xlim([x[0], x[-1]])
            ax[ax_row_offset + epoch_idx, 3].set_xlim([x[0], x[-1]])

            ax[ax_row_offset + epoch_idx, 0].set_ylim([0, 75])
            ax[ax_row_offset + epoch_idx, 1].set_ylim([-180, 180])
            ax[ax_row_offset + epoch_idx, 2].set_ylim([-25, 25])
            ax[ax_row_offset + epoch_idx, 3].set_ylim([0, 25])

    ax[0, -1].legend()
    ax[5, -1].legend()

    # ax[0, 1].set_title("Outbound")
    # ax[5, 1].set_title("Inbound")

    ax[0, 0].set_title("Speed (cm/s)")
    ax[0, 1].set_title("Head angle (deg)")
    ax[0, 2].set_title("Head angular velocity (deg/s)")
    ax[0, 3].set_title("Head angular speeed (deg/s)")

    ax[4, 0].axis("off")
    ax[4, 1].axis("off")
    ax[4, 2].axis("off")
    ax[4, 3].axis("off")

    ax[0, 0].set_ylabel("Stim1")
    ax[1, 0].set_ylabel("Stim2")
    ax[2, 0].set_ylabel("Stim3")
    ax[3, 0].set_ylabel("Dark")

    ax[-1, -1].set_xlabel("Position (cm)")

    fig.suptitle(f"{animal_name} {date}")

    fig.savefig(
        fig_save_path / f"motor_position.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return None


def main():
    plot_motor_vars_given_place()


if __name__ == "__main__":
    main()
