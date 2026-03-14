from __future__ import annotations

"""Compute place and task-progression mutual information for one session.

This script loads one session's spike trains, position timestamps, position
samples, and trajectory intervals; rebuilds movement intervals and W-track
coordinates; computes place-field and task-progression tuning curves; estimates
mutual information and shuffle-corrected mutual information; computes
log-likelihood bits-per-spike scores; and plots place-vs-task-progression MI
for one selected light epoch and one selected dark epoch.

Primary outputs are per-unit parquet summary tables for all run epochs. The
selected light-vs-dark comparison tables that drive the scatter plots are also
saved as parquet. Optional legacy pickle outputs preserve the previous nested
artifact structure for downstream compatibility. Run metadata is recorded under
`analysis_path / "v1ca1_log"`.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
DEFAULT_SIGMA_BINS = 1.0

TuningCurvesByRegion = dict[str, dict[str, Any]]
MetricsByRegion = dict[str, dict[str, pd.DataFrame]]
BitsBySpikeByRegion = dict[str, dict[str, dict[Any, float]]]
SummaryTablesByRegion = dict[str, dict[str, pd.DataFrame]]


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
    tuning_curve: Any,
    position_dim: str,
    sigma_bins: float,
    *,
    eps: float = 1e-12,
    mode: str = "nearest",
) -> Any:
    """Apply NaN-aware Gaussian smoothing along one tuning-curve position axis."""
    from scipy.ndimage import gaussian_filter1d

    if sigma_bins <= 0:
        return tuning_curve.copy()

    axis = tuning_curve.get_axis_num(position_dim)
    values = np.asarray(tuning_curve.values, dtype=np.float64)
    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)

    numerator = gaussian_filter1d(filled, sigma=sigma_bins, axis=axis, mode=mode)
    denominator = gaussian_filter1d(
        mask.astype(np.float64),
        sigma=sigma_bins,
        axis=axis,
        mode=mode,
    )
    smoothed = numerator / np.maximum(denominator, eps)
    smoothed = np.where(denominator > eps, smoothed, np.nan)
    return tuning_curve.copy(data=smoothed)


def calculate_bits_per_spike_pynapple(
    unit_spikes: Any,
    position: Any,
    tuning_curve: np.ndarray,
    bin_edges: np.ndarray,
    epoch: Any,
    bin_size_s: float = 0.002,
    epsilon: float = 1e-10,
) -> float:
    """Compute log-likelihood bits/spike for one unit and one tuning curve."""
    position_epoch = position.restrict(epoch)
    spikes_epoch = unit_spikes.restrict(epoch)
    binned_spikes = spikes_epoch.count(bin_size_s, epoch)
    position_at_bins = position_epoch.interpolate(binned_spikes, ep=epoch)

    spike_counts = np.minimum(np.asarray(binned_spikes.d, dtype=float).ravel(), 1.0)
    positions = np.asarray(position_at_bins.d, dtype=float).ravel()
    valid = np.isfinite(positions)
    if not np.all(valid):
        spike_counts = spike_counts[valid]
        positions = positions[valid]

    if spike_counts.size == 0:
        return 0.0

    total_spikes = float(np.sum(spike_counts))
    if total_spikes == 0:
        return 0.0

    mean_rate = max(total_spikes / (len(spike_counts) * bin_size_s), epsilon)
    spatial_indices = np.digitize(positions, bin_edges) - 1
    spatial_indices = np.clip(spatial_indices, 0, len(tuning_curve) - 1)

    predicted_rates = np.asarray(tuning_curve, dtype=float)[spatial_indices]
    predicted_rates = np.where(np.isfinite(predicted_rates), predicted_rates, mean_rate)
    predicted_rates = np.maximum(predicted_rates, epsilon)

    log_likelihood_model = np.sum(
        spike_counts * np.log(predicted_rates * bin_size_s) - predicted_rates * bin_size_s
    )
    log_likelihood_null = np.sum(
        spike_counts * np.log(mean_rate * bin_size_s) - mean_rate * bin_size_s
    )
    return float((log_likelihood_model - log_likelihood_null) / np.log(2) / total_spikes)


def compute_shuffled_si(
    spikes: Any,
    epoch: Any,
    movement_epoch: Any,
    feature: Any,
    bins: np.ndarray,
    n_shuffles: int = 50,
    min_shift_s: float = 20.0,
) -> pd.DataFrame:
    """Estimate chance-level mutual information via circular timestamp shifts."""
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


def compute_raw_tuning_and_mi(
    session: dict[str, Any],
    linear_position_bins: np.ndarray,
    task_progression_bins: np.ndarray,
) -> tuple[TuningCurvesByRegion, TuningCurvesByRegion, MetricsByRegion, MetricsByRegion]:
    """Compute unsmoothed tuning curves and raw MI for all regions and epochs."""
    import pynapple as nap

    place_tuning_curves: TuningCurvesByRegion = {region: {} for region in REGIONS}
    task_progression_tuning_curves: TuningCurvesByRegion = {region: {} for region in REGIONS}
    place_si: MetricsByRegion = {region: {} for region in REGIONS}
    task_progression_si: MetricsByRegion = {region: {} for region in REGIONS}

    for region in REGIONS:
        spikes = session["spikes_by_region"][region]
        for epoch in session["run_epochs"]:
            movement_epoch = session["movement_by_run"][epoch]
            place_tuning_curve = nap.compute_tuning_curves(
                data=spikes,
                features=session["linear_position_by_run"][epoch],
                bins=[linear_position_bins],
                epochs=movement_epoch,
                feature_names=["linpos"],
            )
            task_progression_tuning_curve = nap.compute_tuning_curves(
                data=spikes,
                features=session["task_progression_by_run"][epoch],
                bins=[task_progression_bins],
                epochs=movement_epoch,
                feature_names=["tp"],
            )

            place_tuning_curves[region][epoch] = place_tuning_curve
            task_progression_tuning_curves[region][epoch] = task_progression_tuning_curve
            place_si[region][epoch] = nap.compute_mutual_information(place_tuning_curve)
            task_progression_si[region][epoch] = nap.compute_mutual_information(
                task_progression_tuning_curve
            )

    return place_tuning_curves, task_progression_tuning_curves, place_si, task_progression_si


def compute_smoothed_tuning_and_ll(
    session: dict[str, Any],
    place_tuning_curves: TuningCurvesByRegion,
    task_progression_tuning_curves: TuningCurvesByRegion,
    *,
    sigma_bins: float,
    ll_bin_size_s: float,
) -> tuple[TuningCurvesByRegion, TuningCurvesByRegion, BitsBySpikeByRegion, BitsBySpikeByRegion]:
    """Compute smoothed tuning curves and LL bits/spike for all regions and epochs."""
    smoothed_place_tuning_curves: TuningCurvesByRegion = {region: {} for region in REGIONS}
    smoothed_task_progression_tuning_curves: TuningCurvesByRegion = {
        region: {} for region in REGIONS
    }
    ll_bits_per_spike_place: BitsBySpikeByRegion = {region: {} for region in REGIONS}
    ll_bits_per_spike_task_progression: BitsBySpikeByRegion = {
        region: {} for region in REGIONS
    }

    for region in REGIONS:
        spikes = session["spikes_by_region"][region]
        for epoch in session["run_epochs"]:
            movement_epoch = session["movement_by_run"][epoch]
            smoothed_place_tuning_curve = smooth_tuning_curve_along_position(
                place_tuning_curves[region][epoch],
                position_dim="linpos",
                sigma_bins=sigma_bins,
            )
            smoothed_task_progression_tuning_curve = smooth_tuning_curve_along_position(
                task_progression_tuning_curves[region][epoch],
                position_dim="tp",
                sigma_bins=sigma_bins,
            )
            smoothed_place_tuning_curves[region][epoch] = smoothed_place_tuning_curve
            smoothed_task_progression_tuning_curves[region][epoch] = (
                smoothed_task_progression_tuning_curve
            )

            place_bin_edges = np.asarray(smoothed_place_tuning_curve.bin_edges[0], dtype=float)
            task_progression_bin_edges = np.asarray(
                smoothed_task_progression_tuning_curve.bin_edges[0],
                dtype=float,
            )

            ll_bits_per_spike_place[region][epoch] = {}
            ll_bits_per_spike_task_progression[region][epoch] = {}
            for unit_id in spikes.keys():
                ll_bits_per_spike_place[region][epoch][unit_id] = (
                    calculate_bits_per_spike_pynapple(
                        unit_spikes=spikes[unit_id],
                        position=session["linear_position_by_run"][epoch],
                        tuning_curve=smoothed_place_tuning_curve.sel(unit=unit_id).to_numpy(),
                        bin_edges=place_bin_edges,
                        epoch=movement_epoch,
                        bin_size_s=ll_bin_size_s,
                    )
                )
                ll_bits_per_spike_task_progression[region][epoch][unit_id] = (
                    calculate_bits_per_spike_pynapple(
                        unit_spikes=spikes[unit_id],
                        position=session["task_progression_by_run"][epoch],
                        tuning_curve=smoothed_task_progression_tuning_curve.sel(
                            unit=unit_id
                        ).to_numpy(),
                        bin_edges=task_progression_bin_edges,
                        epoch=movement_epoch,
                        bin_size_s=ll_bin_size_s,
                    )
                )

    return (
        smoothed_place_tuning_curves,
        smoothed_task_progression_tuning_curves,
        ll_bits_per_spike_place,
        ll_bits_per_spike_task_progression,
    )


def compute_corrected_mi(
    session: dict[str, Any],
    place_si: MetricsByRegion,
    task_progression_si: MetricsByRegion,
    linear_position_bins: np.ndarray,
    task_progression_bins: np.ndarray,
    *,
    n_shuffles: int,
    min_shift_s: float,
) -> tuple[MetricsByRegion, MetricsByRegion]:
    """Compute shuffle-corrected MI for all regions and epochs."""
    place_si_corrected: MetricsByRegion = {region: {} for region in REGIONS}
    task_progression_si_corrected: MetricsByRegion = {region: {} for region in REGIONS}

    for region in REGIONS:
        spikes = session["spikes_by_region"][region]
        for epoch in session["run_epochs"]:
            place_shuffle = compute_shuffled_si(
                spikes,
                epoch=session["all_epoch_by_run"][epoch],
                movement_epoch=session["movement_by_run"][epoch],
                feature=session["linear_position_by_run"][epoch],
                bins=linear_position_bins,
                n_shuffles=n_shuffles,
                min_shift_s=min_shift_s,
            )
            task_progression_shuffle = compute_shuffled_si(
                spikes,
                epoch=session["all_epoch_by_run"][epoch],
                movement_epoch=session["movement_by_run"][epoch],
                feature=session["task_progression_by_run"][epoch],
                bins=task_progression_bins,
                n_shuffles=n_shuffles,
                min_shift_s=min_shift_s,
            )
            place_si_corrected[region][epoch] = place_si[region][epoch] - place_shuffle
            task_progression_si_corrected[region][epoch] = (
                task_progression_si[region][epoch] - task_progression_shuffle
            )

    return place_si_corrected, task_progression_si_corrected


def build_epoch_summary_tables(
    session: dict[str, Any],
    movement_firing_rates: dict[str, dict[str, np.ndarray]],
    place_si: MetricsByRegion,
    task_progression_si: MetricsByRegion,
    place_si_corrected: MetricsByRegion,
    task_progression_si_corrected: MetricsByRegion,
    ll_bits_per_spike_place: BitsBySpikeByRegion,
    ll_bits_per_spike_task_progression: BitsBySpikeByRegion,
) -> SummaryTablesByRegion:
    """Build one per-unit summary table for each region and run epoch."""
    summary_tables: SummaryTablesByRegion = {region: {} for region in REGIONS}

    for region in REGIONS:
        unit_index = pd.Index(list(session["spikes_by_region"][region].keys()), name="unit")
        for epoch in session["run_epochs"]:
            movement_rate_series = pd.Series(
                movement_firing_rates[region][epoch],
                index=unit_index,
                dtype=float,
            )
            place_epoch = place_si[region][epoch].reindex(unit_index)
            task_progression_epoch = task_progression_si[region][epoch].reindex(unit_index)
            place_corrected_epoch = place_si_corrected[region][epoch].reindex(unit_index)
            task_progression_corrected_epoch = task_progression_si_corrected[region][epoch].reindex(
                unit_index
            )
            ll_place_epoch = pd.Series(
                ll_bits_per_spike_place[region][epoch],
                dtype=float,
            ).reindex(unit_index)
            ll_task_progression_epoch = pd.Series(
                ll_bits_per_spike_task_progression[region][epoch],
                dtype=float,
            ).reindex(unit_index)

            summary_table = pd.DataFrame(index=unit_index)
            summary_table["region"] = region
            summary_table["epoch"] = epoch
            summary_table["movement_firing_rate_hz"] = movement_rate_series
            summary_table["place_mi_bits_per_sec"] = place_epoch["bits/sec"]
            summary_table["place_mi_bits_per_spike"] = place_epoch["bits/spike"]
            summary_table["task_progression_mi_bits_per_sec"] = task_progression_epoch["bits/sec"]
            summary_table["task_progression_mi_bits_per_spike"] = task_progression_epoch[
                "bits/spike"
            ]
            summary_table["place_mi_corrected_bits_per_sec"] = place_corrected_epoch["bits/sec"]
            summary_table["place_mi_corrected_bits_per_spike"] = place_corrected_epoch[
                "bits/spike"
            ]
            summary_table["task_progression_mi_corrected_bits_per_sec"] = (
                task_progression_corrected_epoch["bits/sec"]
            )
            summary_table["task_progression_mi_corrected_bits_per_spike"] = (
                task_progression_corrected_epoch["bits/spike"]
            )
            summary_table["place_ll_bits_per_spike"] = ll_place_epoch
            summary_table["task_progression_ll_bits_per_spike"] = ll_task_progression_epoch
            summary_tables[region][epoch] = summary_table.reset_index()

    return summary_tables


def build_light_dark_comparison_tables(
    summary_tables: SummaryTablesByRegion,
    *,
    light_epoch: str,
    dark_epoch: str,
) -> dict[str, pd.DataFrame]:
    """Build selected light-vs-dark comparison tables for plotting."""
    comparison_tables: dict[str, pd.DataFrame] = {}

    for region in REGIONS:
        light_table = summary_tables[region][light_epoch].set_index("unit")
        dark_table = summary_tables[region][dark_epoch].set_index("unit")
        common_units = light_table.index.intersection(dark_table.index)

        comparison_table = pd.DataFrame(index=common_units)
        comparison_table["region"] = region
        comparison_table["light_epoch"] = light_epoch
        comparison_table["dark_epoch"] = dark_epoch
        comparison_table["dark_movement_firing_rate_hz"] = dark_table.loc[
            common_units, "movement_firing_rate_hz"
        ]
        comparison_table["dark_active"] = (
            comparison_table["dark_movement_firing_rate_hz"]
            > DEFAULT_REGION_FR_THRESHOLDS[region]
        )
        comparison_table["light_place_mi_corrected_bits_per_spike"] = light_table.loc[
            common_units, "place_mi_corrected_bits_per_spike"
        ]
        comparison_table["light_task_progression_mi_corrected_bits_per_spike"] = (
            light_table.loc[common_units, "task_progression_mi_corrected_bits_per_spike"]
        )
        comparison_table["dark_place_mi_corrected_bits_per_spike"] = dark_table.loc[
            common_units, "place_mi_corrected_bits_per_spike"
        ]
        comparison_table["dark_task_progression_mi_corrected_bits_per_spike"] = (
            dark_table.loc[common_units, "task_progression_mi_corrected_bits_per_spike"]
        )
        comparison_tables[region] = comparison_table.reset_index()

    return comparison_tables


def save_summary_tables(
    summary_tables: SummaryTablesByRegion,
    data_dir: Path,
) -> list[Path]:
    """Write one parquet summary table per region and run epoch."""
    saved_paths: list[Path] = []
    for region, region_tables in summary_tables.items():
        for epoch, table in region_tables.items():
            path = data_dir / f"{region}_{epoch}_mi_summary.parquet"
            table.to_parquet(path, index=False)
            saved_paths.append(path)
    return saved_paths


def save_comparison_tables(
    comparison_tables: dict[str, pd.DataFrame],
    data_dir: Path,
    *,
    light_epoch: str,
    dark_epoch: str,
) -> list[Path]:
    """Write selected light-vs-dark MI comparison tables as parquet."""
    saved_paths: list[Path] = []
    for region, table in comparison_tables.items():
        path = data_dir / f"{region}_{light_epoch}_{dark_epoch}_mi_comparison.parquet"
        table.to_parquet(path, index=False)
        saved_paths.append(path)
    return saved_paths


def save_legacy_pickles(
    data_dir: Path,
    *,
    place_tuning_curves: TuningCurvesByRegion,
    task_progression_tuning_curves: TuningCurvesByRegion,
    place_si: MetricsByRegion,
    task_progression_si: MetricsByRegion,
    place_si_corrected: MetricsByRegion,
    task_progression_si_corrected: MetricsByRegion,
    ll_bits_per_spike_place: BitsBySpikeByRegion,
    ll_bits_per_spike_task_progression: BitsBySpikeByRegion,
) -> dict[str, Path]:
    """Write the legacy nested pickle artifacts for compatibility."""
    saved_paths = {
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
        (saved_paths["place_tuning_curves_pickle"], place_tuning_curves),
        (
            saved_paths["task_progression_tuning_curves_pickle"],
            task_progression_tuning_curves,
        ),
        (saved_paths["place_si_pickle"], place_si),
        (saved_paths["task_progression_si_pickle"], task_progression_si),
        (saved_paths["place_si_corrected_pickle"], place_si_corrected),
        (
            saved_paths["task_progression_si_corrected_pickle"],
            task_progression_si_corrected,
        ),
        (saved_paths["ll_bits_per_spike_place_pickle"], ll_bits_per_spike_place),
        (
            saved_paths["ll_bits_per_spike_task_progression_pickle"],
            ll_bits_per_spike_task_progression,
        ),
    ):
        with open(path, "wb") as file_pointer:
            pickle.dump(payload, file_pointer)
    return saved_paths


def plot_mi_comparison(
    comparison_table: pd.DataFrame,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    fig_path: Path,
) -> None:
    """Save a place-MI versus task-progression-MI comparison scatter plot."""
    import matplotlib.pyplot as plt

    active_table = comparison_table.loc[comparison_table["dark_active"]].copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        active_table["light_place_mi_corrected_bits_per_spike"],
        active_table["light_task_progression_mi_corrected_bits_per_spike"],
        alpha=0.25,
        label=light_epoch,
    )
    ax.scatter(
        active_table["dark_place_mi_corrected_bits_per_spike"],
        active_table["dark_task_progression_mi_corrected_bits_per_spike"],
        alpha=0.25,
        label=dark_epoch,
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


def save_mi_figures(
    comparison_tables: dict[str, pd.DataFrame],
    fig_dir: Path,
    *,
    light_epoch: str,
    dark_epoch: str,
) -> list[Path]:
    """Write place-vs-task-progression MI comparison figures."""
    saved_paths: list[Path] = []
    for region, comparison_table in comparison_tables.items():
        fig_path = fig_dir / f"{region}_{light_epoch}_{dark_epoch}_mi_comparison.png"
        plot_mi_comparison(
            comparison_table,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            fig_path=fig_path,
        )
        saved_paths.append(fig_path)
    return saved_paths


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
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in tuning-curve bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--save-legacy-pickle",
        action="store_true",
        help="Also write the legacy nested pickle outputs for compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the task-progression MI workflow for one session."""
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

    (
        place_tuning_curves,
        task_progression_tuning_curves,
        place_si,
        task_progression_si,
    ) = compute_raw_tuning_and_mi(
        session,
        linear_position_bins,
        task_progression_bins,
    )
    (
        _smoothed_place_tuning_curves,
        _smoothed_task_progression_tuning_curves,
        ll_bits_per_spike_place,
        ll_bits_per_spike_task_progression,
    ) = compute_smoothed_tuning_and_ll(
        session,
        place_tuning_curves,
        task_progression_tuning_curves,
        sigma_bins=args.sigma_bins,
        ll_bin_size_s=args.ll_bin_size_s,
    )

    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    place_si_corrected, task_progression_si_corrected = compute_corrected_mi(
        session,
        place_si,
        task_progression_si,
        linear_position_bins,
        task_progression_bins,
        n_shuffles=args.num_shuffles,
        min_shift_s=args.shuffle_min_shift_s,
    )

    summary_tables = build_epoch_summary_tables(
        session,
        movement_firing_rates,
        place_si,
        task_progression_si,
        place_si_corrected,
        task_progression_si_corrected,
        ll_bits_per_spike_place,
        ll_bits_per_spike_task_progression,
    )
    comparison_tables = build_light_dark_comparison_tables(
        summary_tables,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

    saved_epoch_tables = save_summary_tables(summary_tables, data_dir=data_dir)
    saved_comparison_tables = save_comparison_tables(
        comparison_tables,
        data_dir=data_dir,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    saved_figures = save_mi_figures(
        comparison_tables,
        fig_dir=fig_dir,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

    saved_legacy_pickles: dict[str, Path] = {}
    if args.save_legacy_pickle:
        saved_legacy_pickles = save_legacy_pickles(
            data_dir,
            place_tuning_curves=place_tuning_curves,
            task_progression_tuning_curves=task_progression_tuning_curves,
            place_si=place_si,
            task_progression_si=task_progression_si,
            place_si_corrected=place_si_corrected,
            task_progression_si_corrected=task_progression_si_corrected,
            ll_bits_per_spike_place=ll_bits_per_spike_place,
            ll_bits_per_spike_task_progression=ll_bits_per_spike_task_progression,
        )

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
            "sigma_bins": args.sigma_bins,
            "save_legacy_pickle": args.save_legacy_pickle,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "saved_epoch_tables": saved_epoch_tables,
            "saved_comparison_tables": saved_comparison_tables,
            "saved_figures": saved_figures,
            "saved_legacy_pickles": saved_legacy_pickles,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
