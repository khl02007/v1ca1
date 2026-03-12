import nemos as nmo
import pynapple as nap
import numpy as np
import pickle
from pathlib import Path
import position_tools as pt
import spikeinterface.full as si
import time

import kyutils

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

trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]

time_bin_size = 2e-3
speed_threshold = 4  # cm/s
position_offset = 10

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


def _sampling_rate(t_position: np.ndarray) -> float:
    return (len(t_position) - 1) / (t_position[-1] - t_position[0])


def get_start_stop_times_n_fold_cv(epoch, n_folds, ptp=True):

    trajectory_type_list = np.concatenate(
        [
            [trajectory_type] * len(trajectory_times[epoch][trajectory_type])
            for trajectory_type in trajectory_types
        ]
    )
    trajectory_type_list = trajectory_type_list[
        np.argsort(
            np.concatenate(
                [
                    trajectory_times[epoch][trajectory_type]
                    for trajectory_type in trajectory_types
                ]
            )[:, 0]
        )
    ]
    first_trajectory_type = trajectory_type_list[0]
    first_trajectory_type_start_times = np.asarray(
        trajectory_times[epoch][first_trajectory_type]
    )[:, 0]

    folds = np.array_split(first_trajectory_type_start_times, n_folds)
    folds_start_stop_times = np.asarray(
        [[folds[i][0], folds[i + 1][0]] for i in range(len(folds) - 1)]
    )
    folds_start_stop_times = np.vstack(
        (
            folds_start_stop_times,
            [folds_start_stop_times[-1][1], timestamps_ephys[epoch][-1]],
        )
    )
    folds_start_stop_times[0] = [
        timestamps_position_dict[epoch][position_offset:][0],
        folds_start_stop_times[0][1],
    ]
    folds_start_stop_times = np.asarray(folds_start_stop_times)
    if not ptp:
        folds_start_stop_times = folds_start_stop_times - timestamps_ephys[epoch][0]
    return folds_start_stop_times


def calculate_ll_score(region, epoch, n_folds=5):
    print(f"fitting glm to {region} {epoch} with {n_folds}-fold cv")

    glm_fits_save_path = analysis_path / "glm" / "fits"
    glm_results_save_path = analysis_path / "glm" / "results"
    glm_tc_save_path = analysis_path / "glm" / "tc"

    glm_fits_save_path.mkdir(parents=True, exist_ok=True)
    glm_results_save_path.mkdir(parents=True, exist_ok=True)
    glm_tc_save_path.mkdir(parents=True, exist_ok=True)

    # intervals
    all_ep = nap.IntervalSet(
        start=timestamps_position_dict[epoch][position_offset],
        end=timestamps_position_dict[epoch][-1],
    )

    folds_start_stop_times = get_start_stop_times_n_fold_cv(
        epoch=epoch, n_folds=n_folds, ptp=True
    )
    cv_folds_ep = nap.IntervalSet(
        start=folds_start_stop_times[:, 0],
        end=folds_start_stop_times[:, -1],
    )

    trajectory_ep = {}
    for trajectory_type in trajectory_types:
        trajectory_ep[trajectory_type] = nap.IntervalSet(
            start=trajectory_times[epoch][trajectory_type][:, 0],
            end=trajectory_times[epoch][trajectory_type][:, -1],
        )

    # y
    spikes = get_tsgroup(sorting[region])
    spike_counts = spikes.count(bin_size=time_bin_size, ep=all_ep)

    # categorical (trajectory)
    in_trajectory = {}
    for trajectory_type in trajectory_types:
        in_trajectory[trajectory_type] = spike_counts.in_interval(
            trajectory_ep[trajectory_type]
        )

    # motor variables
    position_2d = nap.TsdFrame(
        t=timestamps_position_dict[epoch][position_offset:],
        d=position_dict[epoch][position_offset:],
        columns=["x", "y"],
        time_units="s",
    )
    position_2d = position_2d.interpolate(ts=spike_counts, ep=all_ep)

    speed = pt.get_speed(
        position=position_dict[epoch][position_offset:],
        time=timestamps_position_dict[epoch][position_offset:],
        sampling_frequency=_sampling_rate(
            timestamps_position_dict[epoch][position_offset:]
        ),
        sigma=0.1,
    )
    speed_tsd = nap.Tsd(t=timestamps_position_dict[epoch][position_offset:], d=speed)
    speed_tsd_interp = speed_tsd.interpolate(spike_counts, ep=all_ep)

    acceleration = np.gradient(speed, timestamps_position_dict[epoch][position_offset:])
    acceleration = pt.core.gaussian_smooth(
        acceleration,
        sigma=0.1,
        sampling_frequency=_sampling_rate(
            timestamps_position_dict[epoch][position_offset:]
        ),
    )
    acc_tsd = nap.Tsd(
        t=timestamps_position_dict[epoch][position_offset:], d=acceleration
    )
    acc_tsd_interp = acc_tsd.interpolate(spike_counts, ep=all_ep)

    head_direction_rad = pt.get_angle(
        body_position_dict[epoch][position_offset:],
        position_dict[epoch][position_offset:],
    )
    cos_hd = np.cos(head_direction_rad)
    sin_hd = np.sin(head_direction_rad)
    headdir_tsd = nap.TsdFrame(
        t=timestamps_position_dict[epoch][position_offset:],
        d=np.vstack((cos_hd, sin_hd)).T,
        columns=["cos_hd", "sin_hd"],
    )
    headdir_tsd_interp = headdir_tsd.interpolate(spike_counts, ep=all_ep)
    headdir_tsd_interp = nap.Tsd(
        t=headdir_tsd_interp.t,
        d=np.arctan2(
            headdir_tsd_interp["sin_hd"].to_numpy(),
            headdir_tsd_interp["cos_hd"].to_numpy(),
        ),
        time_support=all_ep,
    )

    head_direction_rad_unwrapped = np.unwrap(head_direction_rad)
    head_direction_rad_velocity = np.gradient(
        head_direction_rad_unwrapped, timestamps_position_dict[epoch][position_offset:]
    )
    head_direction_rad_velocity = pt.core.gaussian_smooth(
        head_direction_rad_velocity,
        sigma=0.1,
        sampling_frequency=_sampling_rate(
            timestamps_position_dict[epoch][position_offset:]
        ),
    )
    headdir_vel_tsd = nap.Tsd(
        t=timestamps_position_dict[epoch][position_offset:],
        d=head_direction_rad_velocity,
    )
    headdir_vel_tsd_interp = headdir_vel_tsd.interpolate(spike_counts, ep=all_ep)

    headdir_speed_tsd_interp = np.abs(headdir_vel_tsd_interp)

    # basis
    position_2d_basis_center_to_left = nmo.basis.BSplineEval(
        n_basis_funcs=10, label="center_to_left_x"
    ) * nmo.basis.BSplineEval(n_basis_funcs=10, label="center_to_left_y")
    position_2d_basis_center_to_left.label = "position_center_to_left"

    position_2d_basis_center_to_right = nmo.basis.BSplineEval(
        n_basis_funcs=10, label="center_to_right_x"
    ) * nmo.basis.BSplineEval(n_basis_funcs=10, label="center_to_right_y")
    position_2d_basis_center_to_right.label = "position_center_to_right"

    position_2d_basis_left_to_center = nmo.basis.BSplineEval(
        n_basis_funcs=10, label="left_to_center_x"
    ) * nmo.basis.BSplineEval(n_basis_funcs=10, label="left_to_center_y")
    position_2d_basis_left_to_center.label = "position_left_to_center"

    position_2d_basis_right_to_center = nmo.basis.BSplineEval(
        n_basis_funcs=10, label="right_to_center_x"
    ) * nmo.basis.BSplineEval(n_basis_funcs=10, label="right_to_center_y")
    position_2d_basis_right_to_center.label = "position_right_to_center"

    path_identity_center_to_left = nmo.basis.IdentityEval(label="center_to_left")
    path_identity_center_to_right = nmo.basis.IdentityEval(label="center_to_right")
    path_identity_left_to_center = nmo.basis.IdentityEval(label="left_to_center")
    path_identity_right_to_center = nmo.basis.IdentityEval(label="right_to_center")

    speed_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="speed")
    acc_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="acc")
    headdir_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="headdir")
    headdir_vel_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="headdir_vel")
    headdir_speed_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="headdir_speed")
    null_basis = nmo.basis.IdentityEval(label="null")

    # models
    models = {
        "place": (
            position_2d_basis_center_to_left * path_identity_center_to_left
            + position_2d_basis_center_to_right * path_identity_center_to_right
            + position_2d_basis_left_to_center * path_identity_left_to_center
            + position_2d_basis_right_to_center * path_identity_right_to_center
        ),
        "motor": (
            speed_basis
            + acc_basis
            + headdir_basis
            + headdir_vel_basis
            + headdir_speed_basis
        ),
        "place_motor": (
            position_2d_basis_center_to_left * path_identity_center_to_left
            + position_2d_basis_center_to_right * path_identity_center_to_right
            + position_2d_basis_left_to_center * path_identity_left_to_center
            + position_2d_basis_right_to_center * path_identity_right_to_center
            + speed_basis
            + acc_basis
            + headdir_basis
            + headdir_vel_basis
            + headdir_speed_basis
        ),
        "null": null_basis,
    }

    features = {
        "place": (
            position_2d["x"],
            position_2d["y"],
            in_trajectory["center_to_left"],
            position_2d["x"],
            position_2d["y"],
            in_trajectory["center_to_right"],
            position_2d["x"],
            position_2d["y"],
            in_trajectory["left_to_center"],
            position_2d["x"],
            position_2d["y"],
            in_trajectory["right_to_center"],
        ),
        "motor": (
            speed_tsd_interp,
            acc_tsd_interp,
            headdir_tsd_interp,
            headdir_vel_tsd_interp,
            headdir_speed_tsd_interp,
        ),
        "place_motor": (
            position_2d["x"],
            position_2d["y"],
            in_trajectory["center_to_left"],
            position_2d["x"],
            position_2d["y"],
            in_trajectory["center_to_right"],
            position_2d["x"],
            position_2d["y"],
            in_trajectory["left_to_center"],
            position_2d["x"],
            position_2d["y"],
            in_trajectory["right_to_center"],
            speed_tsd_interp,
            acc_tsd_interp,
            headdir_tsd_interp,
            headdir_vel_tsd_interp,
            headdir_speed_tsd_interp,
        ),
        "null": (spike_counts.in_interval(all_ep),),
    }

    # cv loop per unit and model
    n_time_bins = {}
    for cv_fold in range(len(cv_folds_ep)):
        test_ep = cv_folds_ep[cv_fold]
        n_time_bins[cv_fold] = len(spike_counts.restrict(test_ep))

    n_spikes = {}
    for unit_id in list(spike_counts.columns):
        n_spikes[unit_id] = {}
        for cv_fold in range(len(cv_folds_ep)):
            test_ep = cv_folds_ep[cv_fold]
            n_spikes[unit_id][cv_fold] = np.sum(
                spike_counts.restrict(test_ep)[:, unit_id]
            )

    scores_ll = {}
    for unit_id in list(spike_counts.columns):
        scores_ll[unit_id] = {}
        for model in models:
            scores_ll[unit_id][model] = {}
            X = models[model].compute_features(*features[model])
            for cv_fold in range(len(cv_folds_ep)):
                test_ep = cv_folds_ep[cv_fold]
                # train_ep = cv_folds_ep.set_diff(test_ep)

                model_param_path = (
                    glm_fits_save_path
                    / f"{epoch}_{region}_{unit_id}_{model}_{cv_fold}_of_{len(cv_folds_ep)}.npz"
                )

                glm = nmo.load_model(model_param_path)
                # glm = nmo.glm.GLM(
                #     solver_kwargs=dict(tol=10**-12),
                # )
                # glm.fit(
                #     X.restrict(train_ep),
                #     spike_counts.restrict(train_ep)[:, unit_id],
                # )

                # glm.save_params(
                #     glm_fits_save_path
                #     / f"{epoch}_{region}_{unit_id}_{model}_{cv_fold}_of_{len(cv_folds_ep)}.npz"
                # )

                # scores[unit_id][model][cv_fold] = glm.score(
                #     X.restrict(test_ep),
                #     spike_counts.restrict(test_ep)[:, unit_id],
                #     score_type="pseudo-r2-McFadden",
                # )

                scores_ll[unit_id][model][cv_fold] = glm.score(
                    X.restrict(test_ep),
                    spike_counts.restrict(test_ep)[:, unit_id],
                    score_type="log-likelihood",
                )

                # predicted_rates[unit_id][model][cv_fold] = (
                #     glm.predict(X.restrict(test_ep)) / time_bin_size
                # )

        # compute place field
        # predicted_rates_concat = np.concatenate(
        #     predicted_rates[unit_id]["place"].values()
        # )
        # glm_place_fields[unit_id] = {}
        # for trajectory_type in trajectory_types:
        #     glm_place_fields[unit_id][trajectory_type] = nap.compute_tuning_curves(
        #         data=predicted_rates_concat.restrict(trajectory_ep[trajectory_type]),
        #         features=position_2d.restrict(trajectory_ep[trajectory_type]),
        #         bins=30,
        #         feature_names=["x", "y"],
        #     )

    with open(glm_results_save_path / f"{epoch}_{region}_scores_ll.pkl", "wb") as f:
        pickle.dump(scores_ll, f)
    with open(glm_results_save_path / f"{epoch}_{region}_time_bins.pkl", "wb") as f:
        pickle.dump(n_time_bins, f)
    with open(glm_results_save_path / f"{epoch}_{region}_n_spikes.pkl", "wb") as f:
        pickle.dump(n_spikes, f)

    # empirical place fields
    # real_place_fields = {}
    # for trajectory_type in trajectory_types:
    #     real_place_fields[trajectory_type] = nap.compute_tuning_curves(
    #         data=spike_counts.smooth(100e-3).restrict(trajectory_ep[trajectory_type])
    #         / time_bin_size,
    #         features=position_2d.restrict(trajectory_ep[trajectory_type]),
    #         bins=30,
    #         feature_names=["x", "y"],
    #     )

    # with open(
    #     glm_tc_save_path / f"{epoch}_{region}_empirical_place_fields.pkl", "wb"
    # ) as f:
    #     pickle.dump(real_place_fields, f)

    return glm


def main():
    start = time.perf_counter()

    for region in regions:
        for epoch in run_epoch_list[::-1]:
            calculate_ll_score(region, epoch, n_folds=5)

    end = time.perf_counter()

    elapsed = end - start

    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
