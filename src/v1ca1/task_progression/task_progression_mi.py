from __future__ import annotations

"""Compute place and task-progression mutual information for one session.

This script loads one session's spike trains, position timestamps, position
samples, and trajectory intervals; rebuilds movement intervals and W-track
coordinates; computes place-field and task-progression tuning curves; estimates
mutual information and shuffle-corrected mutual information; computes
log-likelihood bits-per-spike scores; and plots place-vs-task-progression MI
for one selected light epoch and one selected dark epoch.

Outputs are written under the analysis directory in pickle artifacts and MI
comparison figures. Run metadata is also recorded under
`analysis_path / "v1ca1_log"`.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    build_combined_task_progression_bins,
    build_linear_position_bins,
    compute_movement_firing_rates,
    get_analysis_path,
    prepare_task_progression_session,
)


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}


def get_light_and_dark_epochs(
    run_epochs: list[str],
    light_epoch: str | None,
    dark_epoch: str | None,
) -> tuple[str, str]:
    """Return validated light and dark epoch labels for one session."""
    if not run_epochs:
        raise ValueError("No run epochs were found for this session.")

    selected_light_epoch = light_epoch or run_epochs[0]
    selected_dark_epoch = dark_epoch or run_epochs[-1]
    missing = [
        epoch
        for epoch in (selected_light_epoch, selected_dark_epoch)
        if epoch not in run_epochs
    ]
    if missing:
        raise ValueError(
            f"Requested epochs were not found in run epochs {run_epochs!r}: {missing!r}"
        )
    return selected_light_epoch, selected_dark_epoch


def smooth_tuning_curve_along_position(
    tuning_curve,
    position_dim: str,
    sigma_bins: float,
):
    """Apply Gaussian smoothing along the position axis of one tuning curve."""
    from scipy.ndimage import gaussian_filter1d

    tuning_curve = tuning_curve.fillna(0)
    axis = tuning_curve.get_axis_num(position_dim)
    smoothed = gaussian_filter1d(
        tuning_curve.values,
        sigma=sigma_bins,
        axis=axis,
        mode="nearest",
    )
    return tuning_curve.copy(data=smoothed)


def calculate_bits_per_spike_pynapple(
    unit_spikes,
    position,
    tuning_curve: np.ndarray,
    bin_edges: np.ndarray,
    epoch,
    bin_size_s: float = 0.002,
) -> float:
    """Compute log-likelihood bits/spike for one unit and one tuning curve."""
    position_epoch = position.restrict(epoch)
    spikes_epoch = unit_spikes.restrict(epoch)
    binned_spikes = spikes_epoch.count(bin_size_s, epoch)
    position_at_bins = position_epoch.interpolate(binned_spikes, ep=epoch)

    spike_counts = np.minimum(np.asarray(binned_spikes.d, dtype=float).flatten(), 1)
    positions = np.asarray(position_at_bins.d, dtype=float).flatten()

    if spike_counts.size == 0:
        return 0.0

    spatial_indices = np.digitize(positions, bin_edges) - 1
    spatial_indices = np.clip(spatial_indices, 0, len(tuning_curve) - 1)

    predicted_rates = np.asarray(tuning_curve, dtype=float)[spatial_indices]
    predicted_rates = np.maximum(predicted_rates, 1e-10)

    log_likelihood_model = np.sum(
        spike_counts * np.log(predicted_rates * bin_size_s) - predicted_rates * bin_size_s
    )
    total_spikes = float(np.sum(spike_counts))
    if total_spikes == 0:
        return 0.0

    mean_rate = total_spikes / (len(spike_counts) * bin_size_s)
    log_likelihood_null = np.sum(
        spike_counts * np.log(mean_rate * bin_size_s) - mean_rate * bin_size_s
    )
    return float((log_likelihood_model - log_likelihood_null) / np.log(2) / total_spikes)


def compute_shuffled_si(
    spikes,
    epoch,
    movement_epoch,
    feature,
    bins: np.ndarray,
    n_shuffles: int = 50,
    min_shift_s: float = 20.0,
):
    """Estimate chance-level mutual information via circular timestamp shifts."""
    import pandas as pd
    import pynapple as nap

    shuffled_accumulator = pd.DataFrame(
        0.0,
        index=list(spikes.keys()),
        columns=["bits/sec", "bits/spike"],
    )
    duration = float(epoch.end[0] - epoch.start[0])
    if duration <= min_shift_s:
        raise ValueError(
            "Epoch is too short for shuffle-based MI estimation. "
            f"Epoch duration: {duration:.2f} s, min_shift_s: {min_shift_s:.2f} s."
        )

    spikes_to_shift = spikes.restrict(epoch)
    for _ in range(n_shuffles):
        shifted_spikes = nap.shift_timestamps(
            spikes_to_shift,
            min_shift=min_shift_s,
            max_shift=duration - min_shift_s,
        )
        shuffled_tuning_curve = nap.compute_tuning_curves(
            data=shifted_spikes,
            features=feature,
            bins=[bins],
            epochs=movement_epoch,
        )
        shuffled_si = nap.compute_mutual_information(shuffled_tuning_curve)
        shuffled_accumulator += shuffled_si.fillna(0)
    return shuffled_accumulator / n_shuffles


def plot_mi_comparison(
    place_si_corrected,
    task_progression_si_corrected,
    movement_firing_rates,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    fig_path: Path,
    mode: str = "bits/spike",
) -> None:
    """Save a place-MI versus task-progression-MI comparison scatter plot."""
    import matplotlib.pyplot as plt

    dark_active_mask = (
        movement_firing_rates[region][dark_epoch] > DEFAULT_REGION_FR_THRESHOLDS[region]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    for epoch in (light_epoch, dark_epoch):
        ax.scatter(
            place_si_corrected[region][epoch][mode][dark_active_mask],
            task_progression_si_corrected[region][epoch][mode][dark_active_mask],
            alpha=0.25,
            label=epoch,
        )

    ax.plot([-0.5, 3], [-0.5, 3], "k--", linewidth=1)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_xlabel("Place MI (bits/spike)")
    ax.set_ylabel("Task progression MI (bits/spike)")
    ax.set_title(f"{region.upper()} place vs task progression MI")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the task-progression MI script."""
    parser = argparse.ArgumentParser(
        description="Compute place and task-progression mutual information"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--light-epoch",
        help="Run epoch label to use as the light epoch. Defaults to the first run epoch.",
    )
    parser.add_argument(
        "--dark-epoch",
        help="Run epoch label to use as the dark epoch. Defaults to the last run epoch.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore per epoch. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--place-bin-size-cm",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=f"Spatial bin size in cm for place and task-progression tuning curves. Default: {DEFAULT_PLACE_BIN_SIZE_CM}",
    )
    parser.add_argument(
        "--num-shuffles",
        type=int,
        default=50,
        help="Number of circular-shift shuffles used for MI correction.",
    )
    parser.add_argument(
        "--shuffle-min-shift-s",
        type=float,
        default=20.0,
        help="Minimum circular shift in seconds used during shuffle correction.",
    )
    parser.add_argument(
        "--ll-bin-size-s",
        type=float,
        default=0.002,
        help="Time bin size in seconds used for log-likelihood bits/spike calculations.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the task-progression MI workflow for one session."""
    import pynapple as nap

    args = parse_arguments()
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    light_epoch, dark_epoch = get_light_and_dark_epochs(
        session["run_epochs"],
        args.light_epoch,
        args.dark_epoch,
    )
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = analysis_path / "task_progression_mi"
    fig_dir = analysis_path / "figs" / "task_progression_mi"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    linear_position_bins = build_linear_position_bins(
        args.animal_name,
        args.place_bin_size_cm,
    )
    task_progression_bins = build_combined_task_progression_bins(
        args.animal_name,
        args.place_bin_size_cm,
    )

    place_tuning_curves = {}
    task_progression_tuning_curves = {}
    place_si = {}
    task_progression_si = {}
    smoothed_place_tuning_curves = {}
    smoothed_task_progression_tuning_curves = {}
    ll_bits_per_spike_place = {}
    ll_bits_per_spike_task_progression = {}

    for region in REGIONS:
        place_tuning_curves[region] = {}
        task_progression_tuning_curves[region] = {}
        place_si[region] = {}
        task_progression_si[region] = {}
        smoothed_place_tuning_curves[region] = {}
        smoothed_task_progression_tuning_curves[region] = {}
        ll_bits_per_spike_place[region] = {}
        ll_bits_per_spike_task_progression[region] = {}

        for epoch in session["run_epochs"]:
            place_tuning_curves[region][epoch] = nap.compute_tuning_curves(
                data=session["spikes_by_region"][region],
                features=session["linear_position_by_run"][epoch],
                bins=[linear_position_bins],
                epochs=session["movement_by_run"][epoch],
                feature_names=["linpos"],
            )
            task_progression_tuning_curves[region][epoch] = nap.compute_tuning_curves(
                data=session["spikes_by_region"][region],
                features=session["task_progression_by_run"][epoch],
                bins=[task_progression_bins],
                epochs=session["movement_by_run"][epoch],
                feature_names=["tp"],
            )
            smoothed_place_tuning_curves[region][epoch] = smooth_tuning_curve_along_position(
                place_tuning_curves[region][epoch],
                position_dim="linpos",
                sigma_bins=1.0,
            )
            smoothed_task_progression_tuning_curves[region][epoch] = (
                smooth_tuning_curve_along_position(
                    task_progression_tuning_curves[region][epoch],
                    position_dim="tp",
                    sigma_bins=1.0,
                )
            )
            place_si[region][epoch] = nap.compute_mutual_information(
                place_tuning_curves[region][epoch]
            )
            task_progression_si[region][epoch] = nap.compute_mutual_information(
                task_progression_tuning_curves[region][epoch]
            )

            ll_bits_per_spike_place[region][epoch] = {}
            ll_bits_per_spike_task_progression[region][epoch] = {}
            for unit_id in session["spikes_by_region"][region].keys():
                ll_bits_per_spike_place[region][epoch][unit_id] = (
                    calculate_bits_per_spike_pynapple(
                        unit_spikes=session["spikes_by_region"][region][unit_id],
                        position=session["linear_position_by_run"][epoch],
                        tuning_curve=smoothed_place_tuning_curves[region][epoch]
                        .sel(unit=unit_id)
                        .to_numpy(),
                        bin_edges=smoothed_place_tuning_curves[region][epoch].bin_edges[0],
                        epoch=session["movement_by_run"][epoch],
                        bin_size_s=args.ll_bin_size_s,
                    )
                )
                ll_bits_per_spike_task_progression[region][epoch][unit_id] = (
                    calculate_bits_per_spike_pynapple(
                        unit_spikes=session["spikes_by_region"][region][unit_id],
                        position=session["task_progression_by_run"][epoch],
                        tuning_curve=smoothed_task_progression_tuning_curves[region][epoch]
                        .sel(unit=unit_id)
                        .to_numpy(),
                        bin_edges=smoothed_task_progression_tuning_curves[region][epoch].bin_edges[
                            0
                        ],
                        epoch=session["movement_by_run"][epoch],
                        bin_size_s=args.ll_bin_size_s,
                    )
                )

    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    place_si_corrected = {}
    task_progression_si_corrected = {}
    for region in REGIONS:
        place_si_corrected[region] = {}
        task_progression_si_corrected[region] = {}
        for epoch in session["run_epochs"]:
            place_si_corrected[region][epoch] = place_si[region][epoch] - compute_shuffled_si(
                session["spikes_by_region"][region],
                epoch=session["all_epoch_by_run"][epoch],
                movement_epoch=session["movement_by_run"][epoch],
                feature=session["linear_position_by_run"][epoch],
                bins=linear_position_bins,
                n_shuffles=args.num_shuffles,
                min_shift_s=args.shuffle_min_shift_s,
            )
            task_progression_si_corrected[region][epoch] = task_progression_si[region][
                epoch
            ] - compute_shuffled_si(
                session["spikes_by_region"][region],
                epoch=session["all_epoch_by_run"][epoch],
                movement_epoch=session["movement_by_run"][epoch],
                feature=session["task_progression_by_run"][epoch],
                bins=task_progression_bins,
                n_shuffles=args.num_shuffles,
                min_shift_s=args.shuffle_min_shift_s,
            )

    saved_artifacts = {
        "place_tuning_curves_pickle": data_dir / "place_tuning_curves.pkl",
        "task_progression_tuning_curves_pickle": data_dir / "task_progression_tuning_curves.pkl",
        "place_si_pickle": data_dir / "place_si.pkl",
        "task_progression_si_pickle": data_dir / "task_progression_si.pkl",
        "place_si_corrected_pickle": data_dir / "place_si_corrected.pkl",
        "task_progression_si_corrected_pickle": data_dir / "task_progression_si_corrected.pkl",
        "ll_bits_per_spike_place_pickle": data_dir / "ll_bits_per_spike_place.pkl",
        "ll_bits_per_spike_task_progression_pickle": data_dir
        / "ll_bits_per_spike_task_progression.pkl",
    }
    for path, payload in (
        (saved_artifacts["place_tuning_curves_pickle"], place_tuning_curves),
        (
            saved_artifacts["task_progression_tuning_curves_pickle"],
            task_progression_tuning_curves,
        ),
        (saved_artifacts["place_si_pickle"], place_si),
        (saved_artifacts["task_progression_si_pickle"], task_progression_si),
        (saved_artifacts["place_si_corrected_pickle"], place_si_corrected),
        (
            saved_artifacts["task_progression_si_corrected_pickle"],
            task_progression_si_corrected,
        ),
        (saved_artifacts["ll_bits_per_spike_place_pickle"], ll_bits_per_spike_place),
        (
            saved_artifacts["ll_bits_per_spike_task_progression_pickle"],
            ll_bits_per_spike_task_progression,
        ),
    ):
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    saved_figures: list[Path] = []
    for region in REGIONS:
        fig_path = fig_dir / f"{region}_{light_epoch}_{dark_epoch}_mi_comparison.png"
        plot_mi_comparison(
            place_si_corrected,
            task_progression_si_corrected,
            movement_firing_rates,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            fig_path=fig_path,
        )
        saved_figures.append(fig_path)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.task_progression_mi",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "place_bin_size_cm": args.place_bin_size_cm,
            "num_shuffles": args.num_shuffles,
            "shuffle_min_shift_s": args.shuffle_min_shift_s,
            "ll_bin_size_s": args.ll_bin_size_s,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "saved_artifacts": saved_artifacts,
            "saved_figures": saved_figures,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
