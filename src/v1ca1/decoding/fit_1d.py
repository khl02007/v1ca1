from __future__ import annotations

"""Fit full-W 1D RTC decoders with lap-wise cross-validation.

The script loads one run epoch from the modern analysis directory layout,
linearizes cleaned DLC head position on the animal-specific full W-track,
builds shuffled lap-wise cross-validation folds from trajectory intervals, and
fits one sorted-spikes RTC classifier per requested region and fold. Classifier
objects are written with `SortedSpikesClassifier.save_model`. When requested,
V1 fits use ripple GLM unit filtering by deviance-explained significance or a
direct deviance-explained threshold. Run metadata are recorded under the session
log directory.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.decoding._1d import (
    DEFAULT_BRANCH_GAP_CM,
    DEFAULT_DATA_ROOT,
    DEFAULT_MOVEMENT_VAR,
    DEFAULT_N_FOLDS,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_POSITION_STD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    DEFAULT_TIME_BIN_SIZE_S,
    DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD,
    DISCRETE_VAR_CHOICES,
    REGIONS,
    build_classifier_output_paths,
    build_unit_ids_by_region,
    build_lapwise_cv,
    build_time_grid,
    build_decoder_state_models,
    compute_speed_on_time_grid,
    get_analysis_path_for_session,
    get_fit_output_dir,
    get_spike_indicator,
    get_trajectory_direction,
    get_unit_selection_label,
    interpolate_position_to_time,
    linearize_full_w_position,
    load_required_session_inputs,
    load_sortings,
    make_classifier,
    make_interval_mask,
    preflight_no_existing,
    require_spiking_likelihood_kde_gpu,
    select_regions,
    validate_classifier_place_fields,
    validate_fold_count,
)
from v1ca1.helper.cuda import configure_cuda_visible_devices
from v1ca1.helper.run_logging import write_run_log


SCRIPT_NAME = "v1ca1.decoding.fit_1d"


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments."""
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if args.time_bin_size_s <= 0:
        raise ValueError("--time-bin-size-s must be positive.")
    if args.position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if args.speed_threshold_cm_s < 0:
        raise ValueError("--speed-threshold-cm-s must be non-negative.")
    if args.position_std <= 0:
        raise ValueError("--position-std must be positive.")
    if args.place_bin_size <= 0:
        raise ValueError("--place-bin-size must be positive.")
    if args.movement_var <= 0:
        raise ValueError("--movement-var must be positive.")
    if args.branch_gap_cm < 0:
        raise ValueError("--branch-gap-cm must be non-negative.")
    if args.v1_ripple_glm_devexp_threshold is not None:
        if not np.isfinite(args.v1_ripple_glm_devexp_threshold):
            raise ValueError("--v1-ripple-glm-devexp-threshold must be finite.")


def fit_region_classifiers(
    *,
    region: str,
    sorting: Any,
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids: list[Any],
    linear_position: np.ndarray,
    speed: np.ndarray,
    train_intervals_by_fold: dict[int, np.ndarray],
    output_paths: dict[tuple[str, int], Path],
    n_folds: int,
    movement: bool,
    speed_threshold_cm_s: float,
    direction: bool,
    discrete_var: str,
    movement_var: float,
    place_bin_size: float,
    position_std: float,
    track_graph: Any,
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
) -> list[Path]:
    """Fit and save all fold classifiers for one region."""
    spike_indicator = get_spike_indicator(
        sorting,
        timestamps_ephys_all=timestamps_ephys_all,
        time_grid=time_grid,
        unit_ids=unit_ids,
    )
    if spike_indicator.shape[1] == 0:
        raise ValueError(f"Region {region!r} has no units to fit.")

    encoding_labels = None
    if direction:
        encoding_labels, _is_inbound = get_trajectory_direction(linear_position)

    continuous_transition_types, observation_models = build_decoder_state_models(
        direction=direction,
        discrete_var=discrete_var,
        movement_var=movement_var,
    )

    saved_paths: list[Path] = []
    for fold in range(n_folds):
        train_mask = make_interval_mask(time_grid, train_intervals_by_fold[fold])
        if movement:
            train_mask &= speed > speed_threshold_cm_s
        train_mask &= np.isfinite(linear_position)

        n_training_bins = int(np.sum(train_mask))
        if n_training_bins == 0:
            raise ValueError(
                f"Fold {fold} for region {region!r} has no training bins after "
                "lap, movement, and finite-position filtering."
            )

        print(
            f"Fitting {region} fold {fold + 1}/{n_folds} with "
            f"{n_training_bins} training bins and {spike_indicator.shape[1]} units."
        )
        classifier = make_classifier(
            place_bin_size=place_bin_size,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=edge_spacing,
            continuous_transition_types=continuous_transition_types,
            observation_models=observation_models,
            position_std=position_std,
        )
        classifier.fit(
            linear_position,
            spike_indicator,
            encoding_group_labels=encoding_labels,
            is_training=train_mask,
        )
        validate_classifier_place_fields(classifier)
        output_path = output_paths[(region, fold)]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save_model(output_path)
        saved_paths.append(output_path)
    return saved_paths


def run(args: argparse.Namespace) -> None:
    """Run the 1D decoder fitting workflow."""
    validate_arguments(args)
    configure_cuda_visible_devices(args.cuda_visible_devices)

    analysis_path = get_analysis_path_for_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    selected_regions = select_regions(args.region)
    v1_unit_selection_requested = (
        args.v1_ripple_glm_units
        or args.v1_ripple_glm_devexp_threshold is not None
    )
    unit_selection_label = get_unit_selection_label(
        args.v1_ripple_glm_units and "v1" in selected_regions,
        (
            args.v1_ripple_glm_devexp_threshold
            if "v1" in selected_regions
            else None
        ),
    )
    if v1_unit_selection_requested and "v1" not in selected_regions:
        print(
            "A V1 ripple GLM unit-selection option was passed, but V1 is not selected; "
            "CA1 units are unchanged."
        )
    output_paths = build_classifier_output_paths(
        get_fit_output_dir(analysis_path),
        regions=selected_regions,
        epoch=args.epoch,
        n_folds=args.n_folds,
        random_state=args.random_state,
        direction=args.direction,
        movement=args.movement,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        position_std=args.position_std,
        discrete_var=args.discrete_var,
        place_bin_size=args.place_bin_size,
        movement_var=args.movement_var,
        branch_gap_cm=args.branch_gap_cm,
        unit_selection_label=unit_selection_label,
    )
    preflight_no_existing(
        list(output_paths.values()),
        overwrite=args.overwrite,
        output_kind="classifier",
    )
    require_spiking_likelihood_kde_gpu()

    session = load_required_session_inputs(
        analysis_path=analysis_path,
        animal_name=args.animal_name,
        epoch=args.epoch,
    )
    lap_counts = validate_fold_count(
        session["trajectory_intervals"],
        n_folds=args.n_folds,
    )
    train_intervals_by_fold, test_intervals_by_fold, fold_interval_records = build_lapwise_cv(
        session["trajectory_intervals"],
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    sortings = load_sortings(analysis_path, selected_regions)
    unit_ids_by_region, unit_selection_by_region = build_unit_ids_by_region(
        sortings=sortings,
        regions=selected_regions,
        analysis_path=analysis_path,
        epoch=args.epoch,
        v1_ripple_glm_units=args.v1_ripple_glm_units,
        v1_ripple_glm_devexp_threshold=args.v1_ripple_glm_devexp_threshold,
    )

    time_grid = build_time_grid(
        session["timestamps_position"],
        position_offset=args.position_offset,
        time_bin_size_s=args.time_bin_size_s,
    )
    position_interp = interpolate_position_to_time(
        session["position"],
        session["timestamps_position"],
        time_grid,
        position_offset=args.position_offset,
    )
    linear_position, track_graph, edge_order, edge_spacing = linearize_full_w_position(
        animal_name=args.animal_name,
        position_interp=position_interp,
        branch_gap_cm=args.branch_gap_cm,
    )
    speed = compute_speed_on_time_grid(
        session["position"],
        session["timestamps_position"],
        time_grid,
        position_offset=args.position_offset,
    )

    print(
        f"Fitting 1D decoder for {args.animal_name} {args.date} epoch {args.epoch}; "
        f"regions={list(selected_regions)}, n_folds={args.n_folds}, "
        f"random_state={args.random_state}, direction={args.direction}, "
        f"movement={args.movement}, unit_selection={unit_selection_label}."
    )
    saved_classifier_paths: list[Path] = []
    for region in selected_regions:
        saved_classifier_paths.extend(
            fit_region_classifiers(
                region=region,
                sorting=sortings[region],
                timestamps_ephys_all=session["timestamps_ephys_all"],
                time_grid=time_grid,
                unit_ids=unit_ids_by_region[region],
                linear_position=linear_position,
                speed=speed,
                train_intervals_by_fold=train_intervals_by_fold,
                output_paths=output_paths,
                n_folds=args.n_folds,
                movement=args.movement,
                speed_threshold_cm_s=args.speed_threshold_cm_s,
                direction=args.direction,
                discrete_var=args.discrete_var,
                movement_var=args.movement_var,
                place_bin_size=args.place_bin_size,
                position_std=args.position_std,
                track_graph=track_graph,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
            )
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name=SCRIPT_NAME,
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "epoch": args.epoch,
            "regions": list(selected_regions),
            "data_root": args.data_root,
            "n_folds": args.n_folds,
            "random_state": args.random_state,
            "time_bin_size_s": args.time_bin_size_s,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "position_std": args.position_std,
            "place_bin_size": args.place_bin_size,
            "movement_var": args.movement_var,
            "discrete_var": args.discrete_var,
            "branch_gap_cm": args.branch_gap_cm,
            "direction": args.direction,
            "movement": args.movement,
            "v1_ripple_glm_units": args.v1_ripple_glm_units,
            "v1_ripple_glm_devexp_threshold": args.v1_ripple_glm_devexp_threshold,
            "unit_selection_label": unit_selection_label,
            "v1_ripple_glm_p_value_threshold": DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD,
            "cuda_visible_devices": args.cuda_visible_devices,
            "overwrite": args.overwrite,
        },
        outputs={
            "sources": {
                **session["sources"],
                "sorting": {
                    region: str(analysis_path / f"sorting_{region}")
                    for region in selected_regions
                },
            },
            "run_epochs": session["run_epochs"],
            "lap_counts": lap_counts,
            "edge_order": edge_order,
            "edge_spacing": edge_spacing,
            "unit_selection_by_region": unit_selection_by_region,
            "fold_interval_records": fold_interval_records,
            "train_intervals_by_fold": {
                fold: intervals.tolist()
                for fold, intervals in train_intervals_by_fold.items()
            },
            "test_intervals_by_fold": {
                fold: intervals.tolist()
                for fold, intervals in test_intervals_by_fold.items()
            },
            "saved_classifier_paths": saved_classifier_paths,
        },
    )
    print(f"Saved {len(saved_classifier_paths)} classifier(s).")
    print(f"Saved run metadata to {log_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the 1D decoder fit."""
    parser = argparse.ArgumentParser(
        description="Fit a full-W 1D RTC decoder with lap-wise cross-validation."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name, e.g. L14.")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format.")
    parser.add_argument("--epoch", required=True, help="Run epoch name, e.g. 02_r1.")
    parser.add_argument(
        "--region",
        choices=REGIONS,
        help="Optional single region to fit. Default: fit all regions.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Number of shuffled lap-wise cross-validation folds. Default: {DEFAULT_N_FOLDS}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for shuffled lap-wise folds. Default: {DEFAULT_RANDOM_STATE}",
    )
    parser.add_argument(
        "--time-bin-size-s",
        type=float,
        default=DEFAULT_TIME_BIN_SIZE_S,
        help=f"Spike-count time bin size in seconds. Default: {DEFAULT_TIME_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Leading position samples to drop. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Movement speed threshold used when --movement is enabled. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--position-std",
        type=float,
        default=DEFAULT_POSITION_STD,
        help=f"KDE position standard deviation in cm. Default: {DEFAULT_POSITION_STD}",
    )
    parser.add_argument(
        "--place-bin-size",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=f"RTC environment place bin size in cm. Default: {DEFAULT_PLACE_BIN_SIZE_CM}",
    )
    parser.add_argument(
        "--movement-var",
        type=float,
        default=DEFAULT_MOVEMENT_VAR,
        help=f"Random-walk movement variance. Default: {DEFAULT_MOVEMENT_VAR}",
    )
    parser.add_argument(
        "--discrete-var",
        choices=DISCRETE_VAR_CHOICES,
        default="switching",
        help="Discrete transition model. Default: switching.",
    )
    parser.add_argument(
        "--branch-gap-cm",
        type=float,
        default=DEFAULT_BRANCH_GAP_CM,
        help=(
            "Gap inserted between left and right branches in the full-track "
            f"linear coordinate. Default: {DEFAULT_BRANCH_GAP_CM}"
        ),
    )
    parser.add_argument(
        "--direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit separate inbound/outbound encoding groups. Default: enabled.",
    )
    parser.add_argument(
        "--movement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict training to movement bins. Default: enabled.",
    )
    unit_selection_group = parser.add_mutually_exclusive_group()
    unit_selection_group.add_argument(
        "--v1-ripple-glm-units",
        action="store_true",
        help=(
            "Restrict V1 units to those with ripple GLM deviance-explained "
            "p-value < "
            f"{DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD}. Default: use all units."
        ),
    )
    unit_selection_group.add_argument(
        "--v1-ripple-glm-devexp-threshold",
        type=float,
        help=(
            "Restrict V1 units to those with ripple GLM ripple_devexp_mean "
            "greater than or equal to this value. Default: use all units."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Optional value for CUDA_VISIBLE_DEVICES, for example '0' or '0,1'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing classifier outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run(parse_arguments())


if __name__ == "__main__":
    main()
