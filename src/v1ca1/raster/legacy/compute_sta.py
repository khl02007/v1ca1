import spikeinterface.full as si
import numpy as np
import kyutils
from pathlib import Path
import pickle
import scipy
import position_tools as pt
import pynapple as nap
import matplotlib.pyplot as plt
import time

animal_name = "L14"
data_path = Path("/nimbus/kyu") / animal_name
date = "20240611"
analysis_path = data_path / "singleday_sort" / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)
with open(analysis_path / "body_position.pkl", "rb") as f:
    body_position_dict = pickle.load(f)

nwb_file_base_path = Path("/stelmo/nwb/raw")

(analysis_path / "sta").mkdir(parents=True, exist_ok=True)

position_offset = 10
temporal_bin_size_s = 2e-3
sampling_rate = int(1 / temporal_bin_size_s)


regions = ["v1", "ca1"]
sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)


epoch_intervals = nap.IntervalSet(
    start=[timestamps_ephys[epoch][0] for epoch in run_epoch_list],
    end=[timestamps_ephys[epoch][-1] for epoch in run_epoch_list],
)


def get_tsgroup(sorting):
    "convert spikeinterface sorting object to pynapple tsgroup object"
    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
        )
    tsgroup = nap.TsGroup(data, time_units="s")
    return tsgroup


def get_sta(bin_size=0.01):

    sta = {}
    for region in regions:
        spikes = get_tsgroup(sorting[region])
        sta[region] = {}
        for run_epoch_idx, run_epoch in enumerate(run_epoch_list):

            t_position = timestamps_position[run_epoch][position_offset:]
            start_time = t_position[0]
            end_time = t_position[-1]
            n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

            time = np.linspace(start_time, end_time, n_samples)

            position = position_dict[run_epoch][position_offset:]
            body_position = body_position_dict[run_epoch][position_offset:]

            # interpolate position at the decoding times
            f_pos = scipy.interpolate.interp1d(
                t_position,
                position,
                axis=0,
                bounds_error=False,
                kind="linear",
            )
            position_interp = f_pos(time)

            f_pos_body = scipy.interpolate.interp1d(
                t_position, body_position, axis=0, bounds_error=False, kind="linear"
            )
            position_body_interp = f_pos_body(time)

            head_direction_interp = (
                np.arctan2(
                    position_body_interp[:, 1] - position_interp[:, 1],
                    position_body_interp[:, 0] - position_interp[:, 0],
                )
                + np.pi
            )

            position_sampling_rate = len(position) / (t_position[-1] - t_position[0])
            speed = pt.get_speed(
                position,
                time=t_position,
                sampling_frequency=position_sampling_rate,
                sigma=0.1,
            )
            f_speed = scipy.interpolate.interp1d(
                t_position, speed, axis=0, bounds_error=False, kind="linear"
            )
            speed_interp = f_speed(time)

            features = nap.TsdFrame(
                t=time,
                d=np.vstack((head_direction_interp, speed_interp)).T,
                columns=["head_direction", "speed"],
            )

            spikes_epoch = spikes.restrict(epoch_intervals[run_epoch_idx])
            sta[region][run_epoch] = nap.compute_event_trigger_average(
                group=spikes_epoch,
                feature=features,
                binsize=bin_size,
                windowsize=(-1, 1),
                time_unit="s",
            )

    return sta


def plot_sta(bin_size):
    fig_save_path = analysis_path / "figs" / "sta" / f"bin_size_{bin_size}"
    fig_save_path.mkdir(parents=True, exist_ok=True)
    sta = get_sta(bin_size=bin_size)
    for region in regions:
        for unit_id in sorting[region].get_unit_ids():
            fig, ax = plt.subplots(
                nrows=2,
                ncols=len(run_epoch_list),
                figsize=(len(run_epoch_list) * 3, 2 * 2),
            )
            for run_epoch_idx, run_epoch in enumerate(run_epoch_list):
                ax[0, run_epoch_idx].plot(sta[region][run_epoch][:, unit_id, 0])
                ax[1, run_epoch_idx].plot(sta[region][run_epoch][:, unit_id, 1])
                ax[0, run_epoch_idx].set_title(run_epoch)
                ax[0, run_epoch_idx].set_ylim((0, 2 * np.pi))
                ax[1, run_epoch_idx].set_ylim((0, 90))
            ax[0, 0].set_xlabel("Time from spike (s)")
            ax[0, 0].set_ylabel("Mean head direction | spike (rad)")
            ax[1, 0].set_ylabel("Mean speed | spike (cm/s)")
            fig.suptitle(
                f"{animal_name} {date} {region} {unit_id} spike triggered average head dir and speed"
            )
            fig.savefig(
                fig_save_path / f"{animal_name}_{date}_{region}_{unit_id}_sta.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

    return None


def main():
    start = time.perf_counter()

    for bin_size in [0.01, 0.02, 0.05]:
        plot_sta(bin_size=bin_size)

    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
