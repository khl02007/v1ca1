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

save_dir = analysis_path / "xcorr"
save_dir.mkdir(parents=True, exist_ok=True)


def get_tsgroup(sorting):
    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
        )
    tsgroup = nap.TsGroup(data, time_units="s")
    return tsgroup


def get_ripple_intervals(epoch, max_lag=0):
    ripple_periods = nap.IntervalSet(
        start=Kay_ripple_detector[epoch]["start_time"] - max_lag,
        end=Kay_ripple_detector[epoch]["end_time"] + max_lag,
    )
    return ripple_periods


def get_sleep_intervals(epoch, max_lag=0):
    nrem_sleep_periods = nap.IntervalSet(
        start=sleep_times[epoch]["start_time"] - max_lag,
        end=sleep_times[epoch]["end_time"] + max_lag,
    )
    return nrem_sleep_periods


def get_immobility_intervals(epoch, max_lag=0):
    immobility_periods = nap.IntervalSet(
        start=immobility_times[epoch]["start_time"] - max_lag,
        end=immobility_times[epoch]["end_time"] + max_lag,
    )
    return immobility_periods


def get_run_intervals(epoch, max_lag=0):
    run_periods = nap.IntervalSet(
        start=run_times[epoch]["start_time"] - max_lag,
        end=run_times[epoch]["end_time"] + max_lag,
    )
    return run_periods


def compute_xcorr_pynapple(epoch, state="immobility", bin_size=2e-3, max_lag=500e-3):
    print(
        f"computing xcorr between ca1 and v1 in {epoch} state {state} bin_size {bin_size} max_lag {max_lag}"
    )

    v1_tsgroup = get_tsgroup(sorting["v1"])
    ca1_tsgroup = get_tsgroup(sorting["ca1"])

    if state == "all":
        ep = nap.IntervalSet(
            start=timestamps_ephys[epoch][0],
            end=timestamps_ephys[epoch][-1],
        )
    elif state == "immobility":
        ep = get_immobility_intervals(epoch, max_lag=max_lag)
    elif state == "sleep":
        ep = get_sleep_intervals(epoch=epoch, max_lag=max_lag)
    elif state == "ripple":
        ep = get_ripple_intervals(epoch=epoch, max_lag=max_lag)
    elif state == "run":
        ep = get_run_intervals(epoch=epoch, max_lag=max_lag)
    else:
        ValueError(f"{state} state is not defined")

    xcorr = nap.compute_crosscorrelogram(
        group=(ca1_tsgroup, v1_tsgroup),
        binsize=bin_size,
        windowsize=max_lag,
        time_units="s",
        norm=False,
        ep=ep,
    )

    with open(
        save_dir
        / f"xcorr_{epoch}_state_{state}_bin_size_{bin_size}_max_lag_{max_lag}.pkl",
        "wb",
    ) as f:
        pickle.dump(xcorr, f)

    return xcorr


def plot_xcorr_per_unit(state, bin_size, max_lag):

    fig_save_path = (
        analysis_path
        / "figs"
        / "xcorr"
        / "single_unit"
        / f"state_{state}_bin_size_{bin_size}_max_lag_{max_lag}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    xcorrs = {}
    for epoch in epoch_list:
        with open(
            analysis_path
            / "xcorr"
            / f"xcorr_{epoch}_state_{state}_bin_size_{bin_size}_max_lag_{max_lag}.pkl",
            "rb",
        ) as f:
            xcorrs[epoch] = pickle.load(f)

    ca1_unit_ids = np.unique([i[0] for i in xcorrs[epoch_list[0]].keys()])
    v1_unit_ids = np.unique([i[1] for i in xcorrs[epoch_list[0]].keys()])

    for ca1_unit_id in ca1_unit_ids:
        for v1_unit_id in v1_unit_ids:
            fig, ax = plt.subplots(
                ncols=len(epoch_list), figsize=(len(epoch_list) * 3, 3)
            )
            for epoch_idx, epoch in enumerate(epoch_list):
                ax[epoch_idx].plot(xcorrs[epoch][(ca1_unit_id, v1_unit_id)])
                ax[epoch_idx].axvline(0, color="red")
                ax[epoch_idx].set_title(epoch)
                ax[epoch_idx].set_xlim([max_lag * -1, max_lag])
                if epoch_idx == 0:
                    ax[epoch_idx].set_xlabel("Time lag (s)")
            fig.suptitle(
                f"Firing rate of V1 {v1_unit_id} relative to spike timing of CA1 {ca1_unit_id}"
            )
            fig.savefig(
                fig_save_path / f"xcorr_ca1_{ca1_unit_id}_v1_{v1_unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
    return None


def plot_xcorr_heatmap(epoch, state, bin_size, max_lag):
    print(
        f"plotting xcorr heatmap epoch {epoch} state {state} bin_size {bin_size} max_lag {max_lag}"
    )

    with open(
        analysis_path
        / "xcorr"
        / f"xcorr_{epoch}_state_{state}_bin_size_{bin_size}_max_lag_{max_lag}.pkl",
        "rb",
    ) as f:
        xcorr = pickle.load(f)

    fig_save_path = (
        analysis_path
        / "figs"
        / "xcorr"
        / "heatmap"
        / f"{epoch}_state_{state}_bin_size_{bin_size}_max_lag_{max_lag}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    ca1_unit_ids = np.unique([i[0] for i in xcorr.keys()])
    # v1_unit_ids = np.unique([i[1] for i in xcorr.keys()])

    for ca1_unit_id in ca1_unit_ids:
        fig, ax = plt.subplots(
            figsize=(3, 12),
        )
        arr = np.asarray([xcorr[i] for i in xcorr.keys() if i[0] == ca1_unit_id])
        inds = np.argsort(np.max(arr, axis=1))
        ax.imshow(
            arr[inds],
            aspect="auto",
            extent=[xcorr.index[0], xcorr.index[-1], len(arr) - 1, 0],
        )
        ax.axvline(0, color="white")
        ax.set_title(
            f"{animal_name} {date} {epoch} {state} xcorr V1 activity relative to CA1 unit {ca1_unit_id} spiking"
        )
        ax.set_ylabel("V1 units")
        ax.set_xlabel("Time lag (s)")
        fig.savefig(
            fig_save_path / f"ca1_{ca1_unit_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return None


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Run ms4 on concatenated run epochs")
#     parser.add_argument("--overwrite", type=str, help="Probe index")
#     # parser.add_argument("--shank_idx", type=int, help="Shank index")
#     return parser.parse_args()


def main():
    # args = parse_arguments()

    # if args.overwrite == "True":
    #     overwrite = True
    # else:
    #     overwrite = False

    start = time.perf_counter()

    bin_size = 0.005
    max_lag = 1

    for epoch in epoch_list:
        compute_xcorr_pynapple(
            epoch=epoch,
            state="run",
            bin_size=bin_size,
            max_lag=max_lag,
        )

    # for epoch in epoch_list:
    #     for state in ["run", "sleep", "immobility", "ripple", "all"]:
    # compute_xcorr_pynapple(
    #     epoch=epoch,
    #     state=state,
    #     bin_size=bin_size,
    #     max_lag=max_lag,
    # )

    for state in ["run"]:
        plot_xcorr_per_unit(state=state, bin_size=bin_size, max_lag=max_lag)

    #         plot_xcorr_heatmap(
    #             epoch=epoch, state=state, bin_size=bin_size, max_lag=max_lag
    #         )

    # for state in ["ripple", "sleep", "immobility"]:
    #     plot_autocorr_per_unit(bin_size=bin_size, max_lag=max_lag, state=state)

    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
