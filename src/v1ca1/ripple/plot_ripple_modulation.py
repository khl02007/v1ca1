import spikeinterface.full as si
import numpy as np
import kyutils
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import argparse
import scipy
import time
import pynapple as nap


animal_name = "L14"
date = "20240611"

analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

regions = ["v1", "ca1"]
sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")


with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)


position_offset = 10

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "ripple" / "Kay_ripple_detector.pkl", "rb") as f:
    Kay_ripple_times_dict = pickle.load(f)

sigma = 1.0
kernel_size = 5
kernel = scipy.signal.windows.gaussian(kernel_size, std=sigma)
kernel /= np.sum(kernel)


def get_tsgroup(sorting):
    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
        )
    tsgroup = nap.TsGroup(data, time_units="s")
    return tsgroup


def select_ordered_unit_ids(fr: dict, order_date: str, order_epoch: str) -> np.ndarray:
    """
    Return unit_ids sorted by descending max(|FR|) for a reference (date, epoch).
    Falls back to an empty array if keys are missing or no nonzero FR exists.
    """
    if order_date not in fr or order_epoch not in fr[order_date]:
        return np.array([], dtype=object)

    fr_epoch = fr[order_date][order_epoch]
    if not isinstance(fr_epoch, dict) or len(fr_epoch) == 0:
        return np.array([], dtype=object)

    unit_ids_nonzero = [
        uid
        for uid, arr in fr_epoch.items()
        if isinstance(arr, np.ndarray) and arr.size > 0 and np.any(arr != 0)
    ]
    if len(unit_ids_nonzero) == 0:
        return np.array([], dtype=object)

    max_vals = np.array([np.max(np.abs(fr_epoch[uid])) for uid in unit_ids_nonzero])
    order = np.argsort(max_vals)[::-1]  # descending
    return np.asarray(unit_ids_nonzero, dtype=object)[order]


def compute_ripple_triggered_mean_fr(
    region,
    date_list,
    epoch_list,
    bin_width=0.01,
    time_before=0.5,
    time_after=0.5,
    ripple_mean_zscore_threshold=2.0,
):

    bin_edges = np.arange(-time_before, time_after, bin_width)
    t = bin_edges[:-1] + 0.5 * bin_width
    fr = {}
    fr["time"] = t
    for date in date_list:

        fr[date] = {}
        for epoch in epoch_list:
            fr[date][epoch] = {}
            for unit_id in sorting[region].get_unit_ids():
                spike_times_s = timestamps_ephys_all_ptp[
                    sorting[region].get_unit_spike_train(unit_id)
                ]
                spike_times_list = []
                for ripple in Kay_ripple_times_dict[date][epoch].itertuples():
                    if ripple.mean_zscore > ripple_mean_zscore_threshold:
                        spike_times_during_ripple = (
                            spike_times_s[
                                (spike_times_s > (ripple.start_time - time_before))
                                & (spike_times_s <= (ripple.start_time + time_after))
                            ]
                            - ripple.start_time
                        )
                        spike_times_list.append(spike_times_during_ripple)
                if not spike_times_list:
                    spike_times_list = [[]]

                spike_counts, _ = np.histogram(
                    np.concatenate(spike_times_list), bins=bin_edges
                )
                fr[date][epoch][unit_id] = (
                    spike_counts / bin_width / len(spike_times_list)
                )
    fr_save_path = analysis_path / "ripple" / f"ripple_triggered_mean_fr_{region}.pkl"
    with open(
        fr_save_path,
        "wb",
    ) as f:
        pickle.dump(fr, f)

    return fr


def compute_ripple_triggered_mean_fr_pynapple(
    region, ripple_threshold_zscore=3.0, time_before=0.5, time_after=0.5
):
    tsgroup = get_tsgroup(sorting[region])

    print(f"Computing PETH of region {region} with pynapple...")
    peth = {}
    # jittered_peth = {}
    for epoch_idx, epoch in enumerate(epoch_list):
        ripple_start_times = nap.Ts(
            t=Kay_ripple_times_dict[epoch][
                Kay_ripple_times_dict[epoch].mean_zscore > ripple_threshold_zscore
            ].start_time.to_numpy(),
            time_units="s",
            time_support=nap.IntervalSet(
                start=timestamps_ephys[epoch][0],
                end=timestamps_ephys[epoch][-1],
                time_units="s",
            ),
        )
        # jittered_ripple_start_times = nap.jitter_timestamps(
        #     ripple_start_times, max_jitter=3
        # )
        # jittered_peth[epoch] = nap.compute_perievent(
        #     timestamps=tsgroup,
        #     tref=jittered_ripple_start_times,
        #     minmax=(-time_before, time_after),
        #     time_unit="s",
        # )
        peth[epoch] = nap.compute_perievent(
            timestamps=tsgroup,
            tref=ripple_start_times,
            minmax=(-time_before, time_after),
            time_unit="s",
        )
    return peth  # , jittered_peth


def plot_ripple_modulation_single_unit_pynapple_uncertainty(
    ripple_threshold_zscore=4,
    time_before=0.5,
    time_after=0.5,
    bin_size=20e-3,
    n_jitters=100,
    max_jitter=3,
    raster=True,
):

    fig_save_path = (
        analysis_path
        / "figs"
        / "ripple"
        / f"fr_single_unit_threshold_{ripple_threshold_zscore}_bin_size_{bin_size}_time_before_{time_before}_time_after_{time_after}_raster_{raster}_n_jitters_{n_jitters}_max_jitter_{max_jitter}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    response_window = [0, 0.1]
    baseline_window = [-0.5, -0.3]

    for region in regions:
        unit_ids = sorting[region].get_unit_ids()
        tsgroup = get_tsgroup(sorting[region])
        for unit_id in unit_ids:
            fig, ax = plt.subplots(
                ncols=len(epoch_list), figsize=(len(epoch_list) * 4, 2)
            )
            max_across_epochs = []
            for epoch_idx, epoch in enumerate(epoch_list):
                ripple_start_times = nap.Ts(
                    t=Kay_ripple_times_dict[epoch][
                        Kay_ripple_times_dict[epoch].mean_zscore
                        > ripple_threshold_zscore
                    ].start_time.to_numpy(),
                    time_units="s",
                    time_support=nap.IntervalSet(
                        start=timestamps_ephys[epoch][0],
                        end=timestamps_ephys[epoch][-1],
                        time_units="s",
                    ),
                )
                peth = nap.compute_perievent(
                    timestamps=tsgroup[unit_id],
                    tref=ripple_start_times,
                    minmax=(-time_before, time_after),
                    time_unit="s",
                )
                jittered_peth = {}
                for i in range(n_jitters):
                    jittered_ripple_start_times = nap.jitter_timestamps(
                        ripple_start_times, max_jitter=3
                    )
                    jittered_peth[i] = nap.compute_perievent(
                        timestamps=tsgroup[unit_id],
                        tref=jittered_ripple_start_times,
                        minmax=(-time_before, time_after),
                        time_unit="s",
                    )

                jittered_distribution = []
                for i in range(n_jitters):
                    jittered_distribution.append(
                        np.mean(jittered_peth[i].count(bin_size) / bin_size, axis=1)
                    )

                upper = np.percentile(np.asarray(jittered_distribution), q=95, axis=0)
                median = np.percentile(np.asarray(jittered_distribution), q=50, axis=0)
                lower = np.percentile(np.asarray(jittered_distribution), q=5, axis=0)

                rta = np.mean(peth.count(bin_size), 1) / bin_size
                max_across_epochs.append(np.max(rta))
                t = rta.index

                ax[epoch_idx].plot(t, rta, linewidth=3, color="red", zorder=5)
                ax[epoch_idx].plot(t, median, "gray", zorder=2)
                ax[epoch_idx].fill_between(
                    t, upper, lower, alpha=0.3, color="gray", zorder=2
                )
                if raster:
                    aa = ax[epoch_idx].twinx()
                    aa.plot(
                        peth.to_tsd(),
                        "|",
                        markersize=1,
                        color="black",
                        mew=1,
                        zorder=1,
                    )
                    aa.set_yticks([])

                ax[epoch_idx].axvline(0)

                baseline_mean = np.mean(
                    rta[(t >= baseline_window[0]) & (t < baseline_window[-1])]
                )
                baseline_std = np.std(
                    rta[(t >= baseline_window[0]) & (t < baseline_window[-1])]
                )
                rta_zscore = (
                    np.mean(rta[(t >= response_window[0]) & (t < response_window[-1])])
                    - baseline_mean
                ) / baseline_std

                ax[epoch_idx].text(
                    0.01,
                    0.99,
                    f"$z=${np.round(rta_zscore, 2)}",
                    transform=ax[epoch_idx].transAxes,
                    ha="left",
                    va="top",
                )

                ax[epoch_idx].set_title(epoch)
            # for a in ax.flatten():
            #     a.set_ylim([0,np.max(max_across_epochs)])
            ax[0].set_xlabel("Time lag (s)")
            ax[0].set_ylabel("Firing rate (Hz)")

            fig.suptitle(
                f"{animal_name} {date} {region} {unit_id} mean fr triggered by ripples > {ripple_threshold_zscore}sd"
            )
            fig.savefig(
                fig_save_path / f"{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    return None


def plot_ripple_modulation_single_unit_pynapple_zscore(
    ripple_threshold_zscore=4,
    time_before=0.5,
    time_after=0.5,
    bin_size=20e-3,
):

    fig_save_path = (
        analysis_path
        / "figs"
        / "ripple"
        / f"zscore_threshold_{ripple_threshold_zscore}_bin_size_{bin_size}_time_before_{time_before}_time_after_{time_after}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    response_window = [0, 0.1]
    baseline_window = [-0.5, -0.3]

    zscore_dict = {}
    for epoch_idx, epoch in enumerate(epoch_list):
        ripple_start_times = nap.Ts(
            t=Kay_ripple_times_dict[epoch][
                Kay_ripple_times_dict[epoch].mean_zscore > ripple_threshold_zscore
            ].start_time.to_numpy(),
            time_units="s",
            time_support=nap.IntervalSet(
                start=timestamps_ephys[epoch][0],
                end=timestamps_ephys[epoch][-1],
                time_units="s",
            ),
        )
        zscore_dict[epoch] = {}
        for region_idx, region in enumerate(regions):
            unit_ids = sorting[region].get_unit_ids()
            tsgroup = get_tsgroup(sorting[region])
            zscore_dict[epoch][region] = {}
            for unit_id in unit_ids:
                # max_across_epochs = []
                peth = nap.compute_perievent(
                    timestamps=tsgroup[unit_id],
                    tref=ripple_start_times,
                    minmax=(-time_before, time_after),
                    time_unit="s",
                )
                rta = np.mean(peth.count(bin_size), 1) / bin_size
                t = rta.index

                baseline_mean = np.mean(
                    rta[(t >= baseline_window[0]) & (t < baseline_window[-1])]
                )
                baseline_std = np.std(
                    rta[(t >= baseline_window[0]) & (t < baseline_window[-1])]
                )
                rta_zscore = (
                    np.mean(rta[(t >= response_window[0]) & (t < response_window[-1])])
                    - baseline_mean
                ) / baseline_std
                zscore_dict[epoch][region][unit_id] = rta_zscore

    fig, ax = plt.subplots(ncols=len(epoch_list), nrows=len(regions), figsize=(16, 3))
    for epoch_idx, epoch in enumerate(epoch_list):
        for region_idx, region in enumerate(regions):
            ax[region_idx, epoch_idx].hist(zscore_dict[epoch][region].values())

    fig.suptitle(
        f"{animal_name} {date} {region} {unit_id} zscore distribution for ripples > {ripple_threshold_zscore}sd"
    )
    fig.savefig(
        fig_save_path / f"zscore.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return None


def plot_ripple_modulation_single_unit_pynapple(
    region_list=regions,
    ripple_threshold_zscore=4,
    time_before=0.5,
    time_after=0.5,
    bin_size=50e-3,
    raster=True,
):
    fig_save_path = (
        analysis_path
        / "figs"
        / "ripple"
        / f"fr_single_unit_threshold_{ripple_threshold_zscore}_bin_size_{bin_size}_time_before_{time_before}_time_after_{time_after}_raster_{raster}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    for region in region_list:

        unit_ids = sorting[region].get_unit_ids()

        print(f"Computing PETH of region {region} with pynapple...")

        peth = compute_ripple_triggered_mean_fr_pynapple(
            region=region,
            ripple_threshold_zscore=ripple_threshold_zscore,
            time_before=time_before,
            time_after=time_after,
        )

        print("Plotting...")
        for unit_id in unit_ids:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=len(epoch_list),
                figsize=(len(epoch_list) * 3, 1 * 3),
            )
            max_val_list = []
            for epoch_idx, epoch in enumerate(epoch_list):
                ax[epoch_idx].plot(
                    np.mean(peth[epoch][unit_id].count(bin_size), axis=1) / bin_size,
                    linewidth=2,
                    color="red",
                    zorder=3,
                )
                # ax[epoch_idx].plot(
                #     np.mean(jittered_peth[epoch][unit_id].count(bin_size), axis=1)
                #     / bin_size,
                #     linewidth=2,
                #     color="gray",
                #     zorder=2,
                # )
                if raster:
                    aa = ax[epoch_idx].twinx()
                    aa.plot(
                        peth[epoch][unit_id].to_tsd(),
                        "|",
                        markersize=1,
                        color="black",
                        mew=1,
                        zorder=1,
                    )
                    aa.set_yticks([])
                max_val_list.append(
                    np.max(
                        np.mean(peth[epoch][unit_id].count(bin_size), axis=1) / bin_size
                    )
                )
                max_val_list.append(
                    np.max(
                        np.mean(jittered_peth[epoch][unit_id].count(bin_size), axis=1)
                        / bin_size
                    )
                )
                ax[epoch_idx].axvline(0, color="blue")
                ax[epoch_idx].set_xlim([-time_before, time_after])
                ax[epoch_idx].set_title(epoch)

            for a in ax.flatten():
                a.set_ylim([0, np.max(max_val_list)])

            ax[0].set_xlabel("Time from ripple start (s)")
            ax[0].set_ylabel("Firing rate (Hz)")

            fig.suptitle(
                f"{animal_name} {region} {unit_id} mean fr triggered by ripples > {ripple_threshold_zscore}sd"
            )
            fig.savefig(
                fig_save_path / f"{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    return None


def plot_ripple_modulation_pynapple(
    region_list=regions,
    epoch_list=epoch_list,
    order_epoch="01_s1",
    ripple_threshold_zscore=3,
    bin_size=50e-3,
    time_before=0.5,
    time_after=0.5,
    normalize="zscore",
):
    fig_save_path = (
        analysis_path
        / "figs"
        / "ripple"
        / f"fr_all_units_threshold_{ripple_threshold_zscore}_bin_size_{bin_size}_time_before_{time_before}_time_after_{time_after}_normalize_{normalize}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    for region in region_list:
        print(f"Working on region {region}...")
        unit_ids = sorting[region].get_unit_ids()
        tsgroup = get_tsgroup(sorting[region])

        print("Computing PETH with pynapple...")
        peth = {}
        for epoch in epoch_list:
            ripple_start_times = nap.Ts(
                t=Kay_ripple_times_dict[epoch][
                    Kay_ripple_times_dict[epoch].mean_zscore > ripple_threshold_zscore
                ].start_time.to_numpy()
            )
            peth[epoch] = nap.compute_perievent(
                timestamps=tsgroup,
                tref=ripple_start_times,
                minmax=(-time_before, time_after),
                time_unit="s",
            )

        order_mean_fr = np.vstack(
            [
                (
                    np.mean(peth[order_epoch][unit_id].count(bin_size), axis=1)
                    / bin_size
                ).to_numpy()
                for unit_id in unit_ids
            ]
        )
        order_mean_fr = scipy.ndimage.convolve1d(
            order_mean_fr, kernel, mode="reflect", axis=1
        )
        # order_mean_fr = scipy.stats.zscore(order_mean_fr, axis=1)
        order = np.argsort(np.max(order_mean_fr, axis=1))

        fig, ax = plt.subplots(
            ncols=len(epoch_list),
            figsize=(len(epoch_list) * 3, 9),
        )
        t = np.mean(peth[epoch][unit_ids[0]].count(bin_size), axis=1).index
        for epoch_idx, epoch in enumerate(epoch_list):
            fr = np.vstack(
                [
                    (
                        np.mean(peth[epoch][unit_id].count(bin_size), axis=1) / bin_size
                    ).to_numpy()
                    for unit_id in unit_ids
                ]
            )
            fr = scipy.ndimage.convolve1d(fr, kernel, mode="reflect", axis=1)
            if normalize == "max":
                fr = fr / np.max(fr, axis=1, keepdims=True)
                vmin = 0
                vmax = 1
                cmap = "viridis"
            elif normalize == "zscore":
                fr = scipy.stats.zscore(fr, axis=1)
                vmin = -3.0
                vmax = 3.0
                cmap = "RdBu"
            im = ax[epoch_idx].imshow(
                fr[order],
                cmap=cmap,  # diverging to show +/- modulation
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[t[0], t[-1], 0, fr.shape[0]],
                origin="upper",  # put unit 0 at bottom
            )
            ax[epoch_idx].axvline(0, color="red", alpha=0.5, linewidth=1)
            ax[epoch_idx].set_title(epoch)
            ax[epoch_idx].set_xlim([t[0], t[-1]])
            if epoch_idx == 0:
                ax[epoch_idx].set_xlabel("Time from ripple start (s)")
                ax[epoch_idx].set_ylabel(f"{region} units")
            else:
                ax[epoch_idx].set_yticklabels([])
                fig.suptitle(
                    f"{animal_name} {date} {region} modulation by ripples > {ripple_threshold_zscore}sd"
                )
        fig.savefig(
            fig_save_path
            / f"{animal_name}_{date}_{region}_ripple_modulation_population.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return None


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Run ms4 on concatenated run epochs")
#     parser.add_argument(
#         "--overwrite", type=str, help="whether to overwrite existing lfp and detection"
#     )

#     return parser.parse_args()


def main():
    # args = parse_arguments()
    # if args.overwrite == "True":
    #     overwrite = True
    # else:
    #     overwrite = False

    start = time.perf_counter()

    time_before = 0.5
    time_after = 0.5
    # for ripple_threshold_zscore in [4.0, 2.0]:
    #     for normalize in ["max", "zscore"]:
    #         for bin_size in [10e-3, 20e-3]:
    #             plot_ripple_modulation_pynapple(
    #                 region_list=regions,
    #                 epoch_list=epoch_list,
    #                 order_epoch="01_s1",
    #                 ripple_threshold_zscore=ripple_threshold_zscore,
    #                 bin_size=bin_size,
    #                 time_before=time_before,
    #                 time_after=time_after,
    #                 normalize=normalize,
    #             )

    for ripple_threshold_zscore in [4.0, 2.0]:
        for bin_size in [20e-3, 10e-3]:
            # plot_ripple_modulation_single_unit_pynapple_uncertainty(
            #     ripple_threshold_zscore=ripple_threshold_zscore, bin_size=bin_size
            # )

            plot_ripple_modulation_single_unit_pynapple_zscore(
                ripple_threshold_zscore=ripple_threshold_zscore,
                time_before=time_before,
                time_after=time_after,
                bin_size=bin_size,
            )

            # plot_ripple_modulation_single_unit_pynapple(
            #     region_list=regions,
            #     ripple_threshold_zscore=ripple_threshold_zscore,
            #     time_before=time_before,
            #     time_after=time_after,
            #     bin_size=bin_size,
            #     raster=True,
            # )

    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
