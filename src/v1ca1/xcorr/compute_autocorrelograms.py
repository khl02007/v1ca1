import spikeinterface.full as si
import numpy as np
import kyutils
import time
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pynapple as nap
import argparse
import position_tools as pt
import scipy

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

position_offset = 10

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)

with open(analysis_path / "ripple" / "Kay_ripple_detector.pkl", "rb") as f:
    Kay_ripple_detector = pickle.load(f)

with open(analysis_path / "v1_theta_times.pkl", "rb") as f:
    v1_theta_times = pickle.load(f)

sleep_times = {}
for epoch in epoch_list:
    with open(analysis_path / "sleep_times" / f"{epoch}.pkl", "rb") as f:
        sleep_times[epoch] = pickle.load(f)

immobility_times = {}
for epoch in epoch_list:
    with open(analysis_path / "immobility_times" / f"{epoch}.pkl", "rb") as f:
        immobility_times[epoch] = pickle.load(f)

run_times = {}
for epoch in epoch_list:
    with open(analysis_path / "run_times" / f"{epoch}.pkl", "rb") as f:
        run_times[epoch] = pickle.load(f)


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")


time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)
speed_threshold = 4  # cm/s

save_dir = analysis_path / "autocorr"
save_dir.mkdir(parents=True, exist_ok=True)


def get_tsgroup(sorting):
    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
        )
    tsgroup = nap.TsGroup(data, time_units="s")
    return tsgroup


def get_ripple_intervals(epoch):
    ripple_periods = nap.IntervalSet(
        start=Kay_ripple_detector[epoch]["start_time"],
        end=Kay_ripple_detector[epoch]["end_time"],
    )
    return ripple_periods


def get_sleep_intervals(epoch):
    nrem_sleep_periods = nap.IntervalSet(
        start=sleep_times[epoch]["start_time"],
        end=sleep_times[epoch]["end_time"],
    )
    return nrem_sleep_periods


def get_immobility_intervals(epoch):
    immobility_periods = nap.IntervalSet(
        start=immobility_times[epoch]["start_time"],
        end=immobility_times[epoch]["end_time"],
    )
    return immobility_periods


def get_run_intervals(epoch):
    run_periods = nap.IntervalSet(
        start=run_times[epoch]["start_time"],
        end=run_times[epoch]["end_time"],
    )
    return run_periods


def get_v1_theta_intervals(epoch):
    v1_theta_periods = nap.IntervalSet(
        start=v1_theta_times[epoch]["start_time_s"],
        end=v1_theta_times[epoch]["stop_time_s"],
    )
    return v1_theta_periods


def compute_autocorr_pynapple(
    region, bin_size=2e-3, max_lag=500e-3, state="immobility"
):

    if state in ["all", "immobility", "ripple", "run"]:
        epochs = epoch_list
    elif state in ["v1_theta"]:
        epochs = run_epoch_list
    else:
        epochs = sleep_epoch_list

    for region in regions:
        tsgroup = get_tsgroup(sorting[region])

        autocorr = {}
        for epoch in epochs:
            if state == "all":
                ep = nap.IntervalSet(
                    start=timestamps_ephys[epoch][0],
                    end=timestamps_ephys[epoch][-1],
                )
            elif state == "immobility":
                ep = get_immobility_intervals(epoch)
            elif state == "sleep":
                ep = get_sleep_intervals(epoch)
            elif state == "ripple":
                ep = get_ripple_intervals(epoch)
            elif state == "run":
                ep = get_run_intervals(epoch)
            elif state == "v1_theta":
                ep = get_v1_theta_intervals(epoch)
            else:
                ValueError(f"{state} state is not defined")

            autocorr[epoch] = nap.compute_autocorrelogram(
                group=tsgroup,
                binsize=bin_size,
                windowsize=max_lag,
                time_units="s",
                ep=ep,
                norm=False,
            )

        with open(
            save_dir
            / f"autocorr_{region}_bin_size_{bin_size}_max_lag_{max_lag}_state_{state}.pkl",
            "wb",
        ) as f:
            pickle.dump(autocorr, f)

    return autocorr


def plot_autocorr_per_unit(bin_size, max_lag, state):

    fig_save_path = (
        analysis_path
        / "figs"
        / "autocorr"
        / "single_unit"
        / f"bin_size_{bin_size}_max_lag_{max_lag}_state_{state}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    if state in ["all", "immobility", "ripple", "run"]:
        epochs = epoch_list
    elif state in ["v1_theta"]:
        epochs = run_epoch_list
    else:
        epochs = sleep_epoch_list

    for region in regions:

        with open(
            save_dir
            / f"autocorr_{region}_bin_size_{bin_size}_max_lag_{max_lag}_state_{state}.pkl",
            "rb",
        ) as f:
            autocorr = pickle.load(f)

        for unit_id in sorting[region].get_unit_ids():
            fig, ax = plt.subplots(
                ncols=len(epochs),
                figsize=(len(epochs) * 3, 3),
            )

            for epoch_idx, epoch in enumerate(epochs):
                lag_times = autocorr[epoch].index.to_numpy()
                ax[epoch_idx].plot(
                    lag_times,
                    autocorr[epoch][unit_id],
                )
                ax[epoch_idx].set_xlim([lag_times[0], lag_times[-1]])
                ax[epoch_idx].set_title(epoch)

            ax[0].set_ylabel("Firing rate (Hz)")
            ax[0].set_xlabel("Time lag (s)")

            fig.suptitle(
                f"{animal_name} {date} {region} {unit_id} autocorrelogram across epochs"
            )
            fig.savefig(
                fig_save_path / f"autocorr_{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
    return None


def plot_autocorr_heatmap(bin_size, max_lag, state):

    if state in ["all", "immobility", "ripple", "run"]:
        epochs = epoch_list
    elif state in ["v1_theta"]:
        epochs = run_epoch_list
    else:
        epochs = sleep_epoch_list

    fig_save_path = (
        analysis_path
        / "figs"
        / "autocorr"
        / "heatmap"
        / f"bin_size_{bin_size}_max_lag_{max_lag}_state_{state}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    for region in regions:
        with open(
            save_dir
            / f"autocorr_{region}_bin_size_{bin_size}_max_lag_{max_lag}_state_{state}.pkl",
            "rb",
        ) as f:
            autocorr = pickle.load(f)

        fig, ax = plt.subplots(
            ncols=len(epochs),
            figsize=(len(epochs) * 3, 12),
            squeeze=True,
        )

        for epoch_idx, epoch in enumerate(epochs):
            lag_times = autocorr[epoch].index.to_numpy()
            acg_vals = autocorr[epoch].to_numpy()
            acg_vals = acg_vals / np.nanmax(acg_vals, axis=0)
            acg_vals = acg_vals[:, np.argsort(acg_vals[0, :])]
            mid_ind = int(np.floor(len(lag_times) / 2))
            acg_vals = acg_vals[
                :, np.argsort(np.abs(np.argmax(acg_vals[mid_ind:], axis=0)))
            ]
            ax[epoch_idx].imshow(
                acg_vals.T,
                cmap="viridis",
                aspect="auto",
                vmin=0,
                vmax=1,
                extent=[lag_times[0], lag_times[-1], 0, acg_vals.shape[1]],
                origin="lower",
            )
            ax[epoch_idx].set_xlim([lag_times[0], lag_times[-1]])
            ax[epoch_idx].set_title(epoch)

        ax[0].set_ylabel("Neurons")
        ax[0].set_xlabel("Time lag (s)")

        fig.suptitle(f"{animal_name} v1 {date} autocorrelogram heatmap across epochs")
        fig.savefig(
            fig_save_path / f"autocorr_{region}_heatmap_{date}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ms4 on concatenated run epochs")
    parser.add_argument("--overwrite", type=str, help="Probe index")
    # parser.add_argument("--shank_idx", type=int, help="Shank index")
    return parser.parse_args()


def main():
    # args = parse_arguments()

    # if args.overwrite == "True":
    #     overwrite = True
    # else:
    #     overwrite = False

    start = time.perf_counter()

    states = ["v1_theta", "ripple", "sleep", "immobility"]
    bin_size = 0.01
    max_lag = 1

    for state in states:
        compute_autocorr_pynapple(
            region=region,
            bin_size=bin_size,
            max_lag=max_lag,
            state=state,
        )

    for state in states:
        plot_autocorr_heatmap(bin_size=bin_size, max_lag=max_lag, state=state)

    for state in states:
        plot_autocorr_per_unit(bin_size=bin_size, max_lag=max_lag, state=state)

    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
