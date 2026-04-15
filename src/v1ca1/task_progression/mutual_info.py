from __future__ import annotations

"""Compute place and task-progression mutual information for one session.

This script loads one session's spike trains, position timestamps, position
samples, and trajectory intervals; rebuilds movement intervals and W-track
coordinates; computes place-field and task-progression tuning curves; and
estimates raw and shuffle-corrected mutual information for each usable run
epoch.

Primary outputs are per-unit parquet summary tables for all usable run epochs.
Run metadata is recorded under `analysis_path / "v1ca1_log"`.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    get_run_epochs,
    load_clean_dlc_position_data,
    load_epoch_tags,
)
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
    get_task_progression_output_dir,
    prepare_task_progression_session,
)
from v1ca1.task_progression.tuning_analysis import (
    circular_shift_unit_spikes_on_movement_axis,
)
TuningCurvesByRegion = dict[str, dict[str, Any]]
MetricsByRegion = dict[str, dict[str, pd.DataFrame]]
SummaryTablesByRegion = dict[str, dict[str, pd.DataFrame]]


def get_session_regions(session: dict[str, Any]) -> tuple[str, ...]:
    """Return the ordered regions present in one loaded session."""
    return tuple(session["spikes_by_region"].keys())


def has_any_finite_position(position_xy: np.ndarray | None) -> bool:
    """Return whether one XY position array contains at least one finite sample."""
    if position_xy is None:
        return False
    position_array = np.asarray(position_xy, dtype=float)
    return position_array.size > 0 and np.isfinite(position_array).any()


def select_run_epochs(
    run_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return requested run epochs, defaulting to all available run epochs."""
    if not requested_epochs:
        return run_epochs

    selected_epochs = list(dict.fromkeys(requested_epochs))
    missing_epochs = [epoch for epoch in selected_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            "Requested epochs were not found in available run epochs "
            f"{run_epochs!r}: {missing_epochs!r}"
        )
    return selected_epochs


def get_run_epochs_with_usable_head_position(
    analysis_path: Path,
    requested_epochs: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return run epochs whose cleaned DLC head position is present and not all NaN."""
    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    run_epochs = select_run_epochs(get_run_epochs(epoch_tags), requested_epochs)
    epoch_order, head_position_by_epoch, _body_position_by_epoch = load_clean_dlc_position_data(
        analysis_path,
        input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    head_position_epochs = {str(epoch) for epoch in epoch_order}

    usable_epochs: list[str] = []
    skipped_epochs: list[dict[str, str]] = []
    position_source = str(
        analysis_path / DEFAULT_CLEAN_DLC_POSITION_DIRNAME / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    for epoch in run_epochs:
        if epoch not in head_position_epochs:
            skipped_epochs.append(
                {"epoch": epoch, "reason": f"head position missing from {position_source}"}
            )
            continue

        head_position = head_position_by_epoch.get(epoch)
        if not has_any_finite_position(head_position):
            skipped_epochs.append(
                {"epoch": epoch, "reason": f"head position is all NaN in {position_source}"}
            )
            continue

        usable_epochs.append(epoch)

    return usable_epochs, skipped_epochs


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
    movement_duration = float(movement_epoch.tot_length())
    if movement_duration <= 2.0 * float(min_shift_s):
        raise ValueError(
            "Movement epoch is too short for movement-axis shuffle-based MI estimation. "
            f"Movement duration: {movement_duration:.2f} s, min_shift_s: {min_shift_s:.2f} s."
        )

    rng = np.random.default_rng()
    for _ in range(n_shuffles):
        shifted_spikes = circular_shift_spikes_on_movement_axis(
            spikes,
            movement_epoch,
            rng=rng,
            min_shift_s=min_shift_s,
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


def circular_shift_spikes_on_movement_axis(
    spikes: Any,
    movement_epoch: Any,
    *,
    rng: np.random.Generator,
    min_shift_s: float,
) -> Any:
    """Circularly shift each unit independently on the concatenated movement axis."""
    import pynapple as nap

    movement_duration = float(movement_epoch.tot_length())
    if movement_duration <= 2.0 * float(min_shift_s):
        raise ValueError(
            "Movement epoch is too short for movement-axis circular shuffling. "
            f"Movement duration: {movement_duration:.2f} s, min_shift_s: {min_shift_s:.2f} s."
        )

    min_shift_fraction = float(min_shift_s) / movement_duration
    shifted_spikes = {
        unit: circular_shift_unit_spikes_on_movement_axis(
            spikes[unit],
            movement_epoch,
            rng=rng,
            min_shift_fraction=min_shift_fraction,
        )
        for unit in spikes.keys()
    }
    return nap.TsGroup(shifted_spikes, time_units="s")


def compute_raw_tuning_and_mi(
    session: dict[str, Any],
    linear_position_bins: np.ndarray,
    task_progression_bins: np.ndarray,
) -> tuple[TuningCurvesByRegion, TuningCurvesByRegion, MetricsByRegion, MetricsByRegion]:
    """Compute unsmoothed tuning curves and raw MI for all regions and epochs."""
    import pynapple as nap

    regions = get_session_regions(session)
    place_tuning_curves: TuningCurvesByRegion = {region: {} for region in regions}
    task_progression_tuning_curves: TuningCurvesByRegion = {region: {} for region in regions}
    place_si: MetricsByRegion = {region: {} for region in regions}
    task_progression_si: MetricsByRegion = {region: {} for region in regions}

    for region in regions:
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
    regions = get_session_regions(session)
    place_si_corrected: MetricsByRegion = {region: {} for region in regions}
    task_progression_si_corrected: MetricsByRegion = {region: {} for region in regions}

    for region in regions:
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
) -> SummaryTablesByRegion:
    """Build one per-unit summary table for each region and run epoch."""
    regions = get_session_regions(session)
    summary_tables: SummaryTablesByRegion = {region: {} for region in regions}

    for region in regions:
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
            summary_tables[region][epoch] = summary_table.reset_index()

    return summary_tables


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
        "--epochs",
        "--epoch",
        nargs="+",
        default=None,
        help="Run epoch label(s) to analyze. Default: all usable run epochs.",
    )
    parser.add_argument(
        "--regions",
        "--region",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Regions to fit. Default: {' '.join(REGIONS)}",
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
    return parser.parse_args()


def main() -> None:
    """Run the task-progression MI workflow for one session."""
    args = parse_arguments()
    selected_regions = tuple(args.regions)
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    requested_epochs = list(args.epochs) if args.epochs else None
    usable_run_epochs, skipped_epochs = get_run_epochs_with_usable_head_position(
        analysis_path,
        requested_epochs=requested_epochs,
    )
    if not usable_run_epochs:
        raise ValueError(
            "No requested run epochs have usable head position in "
            f"{DEFAULT_CLEAN_DLC_POSITION_NAME!r}."
        )

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        selected_run_epochs=usable_run_epochs,
        position_source="clean_dlc_head",
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        require_npz_timestamps=True,
        load_body_position=False,
    )

    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)

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
    )

    saved_epoch_tables = save_summary_tables(summary_tables, data_dir=data_dir)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.mutual_info",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "epochs": requested_epochs,
            "regions": list(selected_regions),
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "place_bin_size_cm": args.place_bin_size_cm,
            "num_shuffles": args.num_shuffles,
            "shuffle_min_shift_s": args.shuffle_min_shift_s,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "skipped_epochs": skipped_epochs,
            "saved_epoch_tables": saved_epoch_tables,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
