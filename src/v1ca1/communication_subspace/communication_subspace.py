from typing import Dict, Optional
from numpy.typing import NDArray
import spikeinterface.full as si
import numpy as np
from sklearn.cross_decomposition import PLSRegression, CCA
import kyutils
import pickle
import scipy
import position_tools as pt
from pathlib import Path

animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]
trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]
regions = ["v1", "ca1"]

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

position_offset = 10

speed_threshold = 4  # cm/s


def get_spike_indicator(sorting, timestamps_ephys_all, time):
    spike_indicator = []
    for unit_id in sorting.get_unit_ids():
        spike_times = timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)]
        spike_times = spike_times[(spike_times > time[0]) & (spike_times <= time[-1])]
        spike_indicator.append(
            np.bincount(np.digitize(spike_times, time[1:-1]), minlength=time.shape[0])
        )
    spike_indicator = np.asarray(spike_indicator).T
    return spike_indicator


def smooth_firing_rates(
    data: NDArray[np.floating],
    bin_size_ms: float,
    smoothing_sigma_ms: float,
    zscore: bool = True,
) -> NDArray[np.floating]:
    """
    Preprocess neural firing rates by applying Gaussian smoothing and
    (optionally) z-scoring along the time axis.

    Parameters
    ----------
    data : ndarray of shape (n_neurons, n_timepoints)
        Neural activity matrix. Should contain spike counts or pre-binned
        firing rates. First dimension indexes neurons; second dimension indexes
        time bins.
    bin_size_ms : float
        Duration of each time bin (in milliseconds).
        Used to convert smoothing sigma to units of bins.
    smoothing_sigma_ms : float
        Standard deviation of Gaussian smoothing kernel in milliseconds.
        The actual kernel is defined in units of bins.
    zscore : bool, default=True
        If True, z-score each neuron's time series after smoothing.

    Returns
    -------
    data_out : ndarray of shape (n_neurons, n_timepoints)
        Preprocessed neural activity. Same shape as input.

    Notes
    -----
    - Smoothing is applied along the time axis for each neuron.
    - Z-scoring subtracts the mean and divides by the standard deviation for
      each neuron independently.
    - Convolution is done in 'same' mode, preserving the original time axis.
    """
    n_neurons, n_timepoints = data.shape

    # Convert sigma from milliseconds → bins
    sigma_bins = smoothing_sigma_ms / bin_size_ms

    # Build Gaussian kernel in bins
    half_width = int(np.ceil(4 * sigma_bins))
    t = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-(t**2) / (2 * sigma_bins**2))
    kernel /= kernel.sum()

    # Vectorized convolution: apply the same kernel to each neuron
    data_smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=1,
        arr=data,
    )

    if not zscore:
        return data_smoothed

    # Z-score each neuron across time
    mean = data_smoothed.mean(axis=1, keepdims=True)
    std = data_smoothed.std(axis=1, keepdims=True)

    # Avoid division by extremely small values
    std_safe = np.where(std < 1e-12, 1.0, std)

    data_out = (data_smoothed - mean) / std_safe
    return data_out


def zscore_data(data_smoothed):
    # Z-score each neuron across time
    mean = data_smoothed.mean(axis=1, keepdims=True)
    std = data_smoothed.std(axis=1, keepdims=True)

    # Avoid division by extremely small values
    std_safe = np.where(std < 1e-12, 1.0, std)

    data_out = (data_smoothed - mean) / std_safe
    return data_out


def remove_inactive_neurons(
    data: NDArray[np.floating],
    neuron_ids: Optional[NDArray[np.integer]] = None,
    var_threshold: float = 1e-3,
):
    """
    Remove inactive neurons based on their temporal variance, optionally
    tracking neuron IDs.

    Parameters
    ----------
    data : ndarray of shape (n_neurons, n_timepoints)
        Preprocessed neural activity matrix (smoothed + z-scored).
        Rows represent neurons, columns represent time bins.

    neuron_ids : ndarray of shape (n_neurons,), optional
        List or array of neuron IDs corresponding to each row of `data`.
        If None, neuron indices [0, ..., n_neurons-1] are used internally.

    var_threshold : float, default=1e-3
        Minimum variance across time required to retain a neuron.
        Neurons with variance below this threshold are removed.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "data_active": ndarray of shape (n_active_neurons, n_timepoints)
          Data matrix after neuron filtering.
        - "active_mask": ndarray of shape (n_neurons,)
          Boolean mask indicating retained neurons.
        - "variance": ndarray of shape (n_neurons,)
          Per-neuron variance before filtering.
        - "neuron_ids_active": ndarray of shape (n_active_neurons,)
          IDs of retained neurons.

    Notes
    -----
    - If no neuron_ids are provided, default integer indices are returned.
    - Variance is computed along the time axis for each neuron.
    """
    if data.ndim != 2:
        raise ValueError(
            f"`data` must have shape (n_neurons, n_timepoints); got {data.shape}."
        )

    n_neurons = data.shape[0]

    # If neuron_ids not provided, assign default IDs 0..n_neurons-1
    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)
    else:
        if len(neuron_ids) != n_neurons:
            raise ValueError("Length of neuron_ids must match number of rows in data.")

    # Variance across time
    variance = data.var(axis=1)

    # Neurons above variance threshold
    active_mask = variance >= var_threshold

    data_active = data[active_mask, :]
    neuron_ids_active = neuron_ids[active_mask]

    return data_active, neuron_ids_active


def preprocess_data(
    epoch,
    region,
    state="movement",
    bin_size_ms=5,
    smoothing_sigma_ms=None,
    neuron_mask=None,
):
    assert state in [
        "movement",
        "immobility",
        "sleep",
        "all",
        "ripple",
    ], "`state` must be either 'movement', 'immobility', or 'all'"

    unit_ids = sorting[region].get_unit_ids()

    t_position = timestamps_position[epoch][position_offset:]
    position = position_dict[epoch][position_offset:]
    position_sampling_rate = len(position) / (t_position[-1] - t_position[0])

    start_time = t_position[0]
    end_time = t_position[-1]

    sampling_rate = int(1000.0 / bin_size_ms)

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)
    speed = pt.get_speed(
        position, time=t_position, sampling_frequency=position_sampling_rate, sigma=0.1
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )
    speed_interp = f_speed(time)

    X = get_spike_indicator(sorting[region], timestamps_ephys_all_ptp, time).T

    if smoothing_sigma_ms is not None:
        X = smooth_firing_rates(
            data=X,
            bin_size_ms=bin_size_ms,
            smoothing_sigma_ms=smoothing_sigma_ms,
            zscore=False,
        )

    X = zscore_data(X)

    if state == "movement":
        X = X[:, speed_interp > speed_threshold]
    elif state == "immobility":
        X = X[:, speed_interp <= speed_threshold]
    elif state == "sleep":
        with open(analysis_path / "sleep_times" / f"{epoch}.pkl", "rb") as f:
            sleep_times = pickle.load(f)
        time_mask = np.zeros_like(time, dtype=bool)
        for start_time, stop_time in zip(
            sleep_times["start_time"], sleep_times["end_time"]
        ):
            time_mask[(time > start_time) & (time <= stop_time)] = True
        X = X[:, time_mask]
    elif state == "ripple":
        with open(analysis_path / "ripple" / f"Kay_ripple_detector.pkl", "rb") as f:
            Kay_ripple_detector = pickle.load(f)
        time_mask = np.zeros_like(time, dtype=bool)
        for start_time, stop_time in zip(
            Kay_ripple_detector[epoch]["start_time"],
            Kay_ripple_detector[epoch]["end_time"],
        ):
            time_mask[(time > start_time) & (time <= stop_time)] = True
        X = X[:, time_mask]

    if neuron_mask is None:
        X, active_unit_ids = remove_inactive_neurons(
            data=X,
            neuron_ids=unit_ids,
        )
    else:
        X = X[neuron_mask]
        active_unit_ids = np.where(neuron_mask)[0]

    return X, active_unit_ids


def get_smoothed_fr_per_trial(epoch, bin_size_ms=2, smoothing_sigma_ms=50):
    "smooth firing rate and then sliced data to identify only run times during each trajectory trial"
    t_position = timestamps_position[epoch][position_offset:]
    position = position_dict[epoch][position_offset:]
    position_sampling_rate = len(position) / (t_position[-1] - t_position[0])

    start_time = t_position[0]
    end_time = t_position[-1]
    sampling_rate = int(1000.0 / bin_size_ms)
    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)
    speed = pt.get_speed(
        position, time=t_position, sampling_frequency=position_sampling_rate, sigma=0.1
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )
    speed_interp = f_speed(time)

    smoothed_fr_run = {}
    for region in regions:
        smoothed_fr_run[region] = {}

        X = get_spike_indicator(sorting[region], timestamps_ephys_all_ptp, time).T
        if smoothing_sigma_ms is not None:
            X = smooth_firing_rates(
                data=X,
                bin_size_ms=bin_size_ms,
                smoothing_sigma_ms=smoothing_sigma_ms,
                zscore=False,
            )

        for trajectory_type in trajectory_types:
            smoothed_fr_run[region][trajectory_type] = {}
            for unit_id in sorting[region].get_unit_ids():
                smoothed_fr_list = []
                for traj_start_time, traj_stop_time in trajectory_times[epoch][
                    trajectory_type
                ]:
                    inds_to_keep = (
                        (time > traj_start_time)
                        & (time <= traj_stop_time)
                        & (speed_interp > speed_threshold)
                    )
                    smoothed_fr_list.append(X[unit_id, inds_to_keep])
                smoothed_fr_run[region][trajectory_type][unit_id] = smoothed_fr_list
    return smoothed_fr_run


def get_residual(
    epoch, bin_size_ms=2, smoothing_sigma_ms=50, position_std=2.0, edge_ms=10
):
    "subtract expected firing rate based on place encoding model from smoothed firing rate"
    edge_bins = int(edge_ms / bin_size_ms)
    smoothed_fr_per_trial = get_smoothed_fr_per_trial(
        epoch, bin_size_ms=bin_size_ms, smoothing_sigma_ms=smoothing_sigma_ms
    )
    rate_function_path = (
        analysis_path
        / f"rate_function_trajectory/rate_function_{epoch}_use_half_all_movement_True_position_std_{position_std}_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl"
    )
    with open(rate_function_path, "rb") as f:
        rate_function = pickle.load(f)

    residuals = {}
    for region in regions:
        residuals[region] = {}
        for unit_id in sorting[region].get_unit_ids():
            tmp = []
            for trajectory_type in trajectory_types:
                tmp.append(
                    np.concatenate(
                        [
                            i[edge_bins : int(-1 * edge_bins)]
                            - j[edge_bins : int(-1 * edge_bins)]
                            for (i, j) in zip(
                                smoothed_fr_per_trial[region][trajectory_type][unit_id],
                                rate_function[region][trajectory_type][unit_id],
                            )
                        ]
                    )
                )
            residuals[region][unit_id] = np.concatenate(tmp)

        sorted_keys = sorted(residuals[region].keys())

        # Stack the values in sorted key order
        arr = np.vstack([residuals[region][k] for k in sorted_keys])
        residuals[region] = arr
    return residuals


def fit_pls_covarying_dims(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
):
    """
    Fit PLS to two neural population recordings and extract shared dimensions.

    Parameters
    ----------
    X : ndarray of shape (n_neurons_a, n_timepoints)
        Preprocessed data from region A.
        Each row is a neuron, each column is a time point.
        Data should already be smoothed and (optionally) z-scored.
    Y : ndarray of shape (n_neurons_b, n_timepoints)
        Preprocessed data from region B.
        Same time axis as `X`.
    n_components : int
        Number of PLS components (shared dimensions) to extract.

    Returns
    -------
    results : dict
        Dictionary with the following entries:

        - "x_weights": ndarray of shape (n_neurons_a, n_components)
          Population weight vectors (one per component) in region A.
        - "y_weights": ndarray of shape (n_neurons_b, n_components)
          Population weight vectors in region B.
        - "x_scores": ndarray of shape (n_components, n_timepoints)
          Latent time courses in region A (projected activity).
        - "y_scores": ndarray of shape (n_components, n_timepoints)
          Latent time courses in region B.
        - "component_covariances": ndarray of shape (n_components,)
          Sample covariance between X and Y scores for each component.
          Larger values = stronger shared covariance.
    """

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of timepoints; "
            f"got {X.shape[1]} and {Y.shape[1]}."
        )

    n_timepoints = X.shape[1]

    # sklearn expects shape (n_samples, n_features)
    # Here: samples = timepoints, features = neurons
    X_t = X.T  # shape (n_timepoints, n_neurons_a)
    Y_t = Y.T  # shape (n_timepoints, n_neurons_b)

    # PLSRegression maximizes covariance between X and Y projections.
    # We set scale=False because we assume you've already z-scored.
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
    pls.fit(X_t, Y_t)

    results = {"model": pls}

    return results


def fit_cca_shared_dims(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
):
    """
    Fit Canonical Correlation Analysis (CCA) to two neural population
    recordings and extract shared dimensions.

    Parameters
    ----------
    X : ndarray of shape (n_neurons_a, n_timepoints)
        Preprocessed data from region A.
        Each row is a neuron, each column is a time point.
        Data should already be smoothed and typically z-scored.
    Y : ndarray of shape (n_neurons_b, n_timepoints)
        Preprocessed data from region B.
        Same time axis as `X`.
    n_components : int
        Number of CCA components (canonical dimensions) to extract.
        Must be <= min(n_neurons_a, n_neurons_b, n_timepoints).

    Returns
    -------
    results : dict
        Dictionary with the following entries:

        - "x_weights": ndarray of shape (n_neurons_a, n_components)
          Canonical weight vectors (one per component) in region A.
        - "y_weights": ndarray of shape (n_neurons_b, n_components)
          Canonical weight vectors in region B.
        - "x_scores": ndarray of shape (n_components, n_timepoints)
          Canonical variate time courses in region A.
        - "y_scores": ndarray of shape (n_components, n_timepoints)
          Canonical variate time courses in region B.
        - "canonical_correlations": ndarray of shape (n_components,)
          Pearson correlation between corresponding canonical variates.
          By construction these are non-increasing with component index.

    Notes
    -----
    - CCA in scikit-learn expects input of shape (n_samples, n_features).
      Here we use timepoints as samples and neurons as features, so we
      transpose the input before passing to CCA.
    - CCA internally centers the data across samples. If you later want to
      project new data onto these canonical axes, you must use the same
      centering (i.e., reuse cca.x_mean_ / cca.y_mean_).
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of timepoints; "
            f"got {X.shape[1]} and {Y.shape[1]}."
        )

    n_timepoints = X.shape[1]

    # sklearn expects shape (n_samples, n_features)
    # samples = timepoints, features = neurons
    X_t = X.T  # (n_timepoints, n_neurons_a)
    Y_t = Y.T  # (n_timepoints, n_neurons_b)

    # CCA: maximize correlation between projections
    # scale=False because we usually already z-score neurons
    cca = CCA(n_components=n_components, scale=False, max_iter=1000)
    cca.fit(X_t, Y_t)

    results = {"model": cca}
    return results


def main():
    results_save_dir_pls = analysis_path / "communication_subspace" / "PLS"
    results_save_dir_pls.mkdir(parents=True, exist_ok=True)

    results_save_dir_cca = analysis_path / "communication_subspace" / "CCA"
    results_save_dir_cca.mkdir(parents=True, exist_ok=True)

    n_components = 5
    position_std = 2.0

    for smoothing_sigma_ms in [20, 50, 10, 5, 100]:
        for bin_size_ms in [2]:
            for state in ["immobility", "ripple", "sleep"]:
                for epoch in epoch_list:
                    print(
                        f"signal smoothing_sigma_ms {smoothing_sigma_ms} bin_size_ms {bin_size_ms} state {state} epoch {epoch}"
                    )
                    for region1 in regions:
                        if region1 == "v1":
                            region2 = "ca1"
                        else:
                            region2 = "v1"

                        X = {}
                        active_unit_ids = {}
                        X[region1], active_unit_ids[region1] = preprocess_data(
                            epoch=epoch,
                            region=region1,
                            state=state,
                            bin_size_ms=bin_size_ms,
                            smoothing_sigma_ms=smoothing_sigma_ms,
                        )
                        X[region2], active_unit_ids[region2] = preprocess_data(
                            epoch=epoch,
                            region=region2,
                            state=state,
                            bin_size_ms=bin_size_ms,
                            smoothing_sigma_ms=smoothing_sigma_ms,
                        )

                        # V1 -> CA1
                        results_pls = fit_pls_covarying_dims(
                            X=X[region1], Y=X[region2], n_components=n_components
                        )
                        results_cca = fit_cca_shared_dims(
                            X=X[region1], Y=X[region2], n_components=n_components
                        )

                        results_pls["x"] = region1
                        results_pls["y"] = region2
                        results_pls["active_unit_ids_x"] = active_unit_ids[region1]
                        results_pls["active_unit_ids_y"] = active_unit_ids[region2]

                        results_cca["x"] = region1
                        results_cca["y"] = region2
                        results_cca["active_unit_ids_x"] = active_unit_ids[region1]
                        results_cca["active_unit_ids_y"] = active_unit_ids[region2]

                        results_save_path_pls = (
                            results_save_dir_pls
                            / f"pls_signal_{region1}_{region2}_{epoch}_n_components_{n_components}_state_{state}_bin_size_ms_{bin_size_ms}_smoothing_sigma_ms_{smoothing_sigma_ms}.pkl"
                        )
                        results_save_path_cca = (
                            results_save_dir_cca
                            / f"cca_signal_{region1}_{region2}_{epoch}_n_components_{n_components}_state_{state}_bin_size_ms_{bin_size_ms}_smoothing_sigma_ms_{smoothing_sigma_ms}.pkl"
                        )

                        with open(
                            results_save_path_pls,
                            "wb",
                        ) as f:
                            pickle.dump(results_pls, f)

                        with open(
                            results_save_path_cca,
                            "wb",
                        ) as f:
                            pickle.dump(results_cca, f)

    # for smoothing_sigma_ms in [10, 20, 50, 100]:
    #     for run_epoch in run_epoch_list:
    #         print(f"residual smoothing_sigma_ms {smoothing_sigma_ms} epoch {run_epoch}")

    #         residuals = get_residual(
    #             run_epoch,
    #             bin_size_ms=2,
    #             smoothing_sigma_ms=smoothing_sigma_ms,
    #             position_std=position_std,
    #             edge_ms=5,
    #         )
    #         for region1 in regions:
    #             if region1 == "v1":
    #                 region2 = "ca1"
    #             else:
    #                 region2 = "v1"

    #             X = {}
    #             active_unit_ids = {}
    #             X[region1] = zscore_data(residuals[region1])
    #             X[region1], active_unit_ids[region1] = remove_inactive_neurons(
    #                 data=X[region1],
    #                 neuron_ids=sorting[region1].get_unit_ids(),
    #             )

    #             X[region2] = zscore_data(residuals[region2])
    #             X[region2], active_unit_ids[region2] = remove_inactive_neurons(
    #                 data=X[region2],
    #                 neuron_ids=sorting[region2].get_unit_ids(),
    #             )

    #             results_pls = fit_pls_covarying_dims(
    #                 X=X[region1], Y=X[region2], n_components=n_components
    #             )
    #             results_cca = fit_cca_shared_dims(
    #                 X=X[region1], Y=X[region2], n_components=n_components
    #             )

    #             results_pls["x"] = region1
    #             results_pls["y"] = region2
    #             results_pls["active_unit_ids_x"] = active_unit_ids[region1]
    #             results_pls["active_unit_ids_y"] = active_unit_ids[region2]

    #             results_cca["x"] = region1
    #             results_cca["y"] = region2
    #             results_cca["active_unit_ids_x"] = active_unit_ids[region1]
    #             results_cca["active_unit_ids_y"] = active_unit_ids[region2]

    #             results_save_path_pls = (
    #                 results_save_dir_pls
    #                 / f"pls_residual_{region1}_{region2}_{run_epoch}_n_components_{n_components}_smoothing_sigma_ms_{smoothing_sigma_ms}.pkl"
    #             )
    #             results_save_path_cca = (
    #                 results_save_dir_cca
    #                 / f"cca_residual_{region1}_{region2}_{run_epoch}_n_components_{n_components}_smoothing_sigma_ms_{smoothing_sigma_ms}.pkl"
    #             )

    #             with open(
    #                 results_save_path_pls,
    #                 "wb",
    #             ) as f:
    #                 pickle.dump(results_pls, f)

    #             with open(
    #                 results_save_path_cca,
    #                 "wb",
    #             ) as f:
    #                 pickle.dump(results_cca, f)


if __name__ == "__main__":
    main()
