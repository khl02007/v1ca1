from __future__ import annotations

"""Predict full-W 1D RTC decoder posteriors across one run epoch.

The script loads fold-specific classifiers produced by `v1ca1.decoding.fit_1d`,
assigns every trimmed epoch time bin to exactly one cross-validation fold, and
saves one combined NetCDF prediction result per requested region. Bins inside
trajectory intervals inherit that trajectory's held-out fold, while bins
between trajectories inherit the earlier trajectory's fold. When requested, V1
prediction uses the same ripple GLM unit subset expected by fit.
By default, RTC computes the acausal smoothed posterior; `--causal-only`
skips that pass and writes the causal posterior output instead.
"""

import argparse
import os
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
    POSTERIOR_KIND_ACAUSAL,
    POSTERIOR_KIND_CAUSAL,
    REGIONS,
    build_classifier_output_paths,
    build_full_epoch_prediction_fold,
    build_lapwise_cv,
    build_prediction_output_paths,
    build_time_grid,
    build_unit_ids_by_region,
    compute_speed_on_time_grid,
    concatenate_fold_results,
    figurl_output_path,
    get_analysis_path_for_session,
    get_fit_output_dir,
    get_predict_output_dir,
    get_spike_indicator,
    get_state_names,
    get_unit_selection_label,
    interpolate_position_to_time,
    linearize_full_w_position,
    load_classifier,
    load_required_session_inputs,
    load_sortings,
    preflight_no_existing,
    require_existing_paths,
    select_regions,
    validate_classifier_place_fields,
    validate_fold_count,
)
from v1ca1.helper.cuda import configure_cuda_visible_devices
from v1ca1.helper.run_logging import write_run_log


SCRIPT_NAME = "v1ca1.decoding.predict_1d"


def get_posterior_kind(*, causal_only: bool) -> str:
    """Return the prediction posterior kind for the requested mode."""
    if causal_only:
        return POSTERIOR_KIND_CAUSAL
    return POSTERIOR_KIND_ACAUSAL


def get_posterior_name(*, causal_only: bool) -> str:
    """Return the RTC posterior variable to save and display."""
    if causal_only:
        return "causal_posterior"
    return "acausal_posterior"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for 1D decoder prediction."""
    parser = argparse.ArgumentParser(
        description="Predict full-W 1D RTC decoder posteriors across one run epoch."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name, e.g. L14.")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format.")
    parser.add_argument("--epoch", required=True, help="Run epoch name, e.g. 02_r1.")
    parser.add_argument(
        "--region",
        choices=REGIONS,
        help="Optional single region to predict. Default: predict all regions.",
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
        help=f"Random seed used for shuffled lap-wise folds. Default: {DEFAULT_RANDOM_STATE}",
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
            "Recorded for consistency with fit settings. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--position-std",
        type=float,
        default=DEFAULT_POSITION_STD,
        help=f"KDE position standard deviation used by fit_1d. Default: {DEFAULT_POSITION_STD}",
    )
    parser.add_argument(
        "--place-bin-size",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=f"RTC environment place bin size used by fit_1d. Default: {DEFAULT_PLACE_BIN_SIZE_CM}",
    )
    parser.add_argument(
        "--movement-var",
        type=float,
        default=DEFAULT_MOVEMENT_VAR,
        help=f"Random-walk movement variance used by fit_1d. Default: {DEFAULT_MOVEMENT_VAR}",
    )
    parser.add_argument(
        "--discrete-var",
        choices=DISCRETE_VAR_CHOICES,
        default="switching",
        help="Discrete transition model used by fit_1d. Default: switching.",
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
        help="Load classifiers fit with inbound/outbound groups. Default: enabled.",
    )
    parser.add_argument(
        "--movement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load classifiers fit with movement-restricted training. Default: enabled.",
    )
    unit_selection_group = parser.add_mutually_exclusive_group()
    unit_selection_group.add_argument(
        "--v1-ripple-glm-units",
        action="store_true",
        help=(
            "Use classifiers and V1 spike indicators restricted to units with "
            "ripple GLM deviance-explained p-value < "
            f"{DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD}. Default: use all units."
        ),
    )
    unit_selection_group.add_argument(
        "--v1-ripple-glm-devexp-threshold",
        type=float,
        help=(
            "Use classifiers and V1 spike indicators restricted to units with "
            "ripple GLM ripple_devexp_mean greater than or equal to this value. "
            "Default: use all units."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Optional value for CUDA_VISIBLE_DEVICES, for example '0' or '0,1'.",
    )
    parser.add_argument(
        "--figurl",
        action="store_true",
        help="Generate one combined figurl text output for the predicted epoch.",
    )
    parser.add_argument(
        "--causal-only",
        action="store_true",
        help=(
            "Skip the acausal smoothing pass and save/display only the causal "
            "posterior. Default: compute acausal posterior."
        ),
    )
    parser.add_argument(
        "--figurl-only",
        action="store_true",
        help=(
            "Regenerate only the figurl from existing prediction NetCDF outputs. "
            "Implies --figurl and skips classifier loading and posterior prediction."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction outputs.",
    )
    return parser.parse_args()


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


def discretize_and_trim(posterior: Any) -> Any:
    """Return a compact uint8 posterior for SortingView display."""
    discretized = np.multiply(posterior, 255).astype(np.uint8)
    stacked = discretized.stack(unified_index=["time", "position"])
    return stacked.where(stacked > 0, drop=True).astype(np.uint8)


def get_observations_per_time(trimmed_posterior: Any, posterior: Any) -> np.ndarray:
    """Return the sparse posterior observation count per time bin."""
    import xarray as xr

    times, counts = np.unique(trimmed_posterior.time.values, return_counts=True)
    indexed_counts = xr.DataArray(counts, coords={"time": times})
    _aligned_time, aligned_counts = xr.align(
        posterior.time,
        indexed_counts,
        join="left",
        fill_value=0,
    )
    return aligned_counts.values.astype(np.uint8)


def get_trimmed_bin_center_index(
    place_bin_centers: np.ndarray,
    trimmed_place_bin_centers: np.ndarray,
) -> np.ndarray:
    """Return original bin-center indices for sparse posterior positions."""
    return np.asarray(
        [
            np.nonzero(np.isclose(place_bin_centers, trimmed_bin_center))[0][0]
            for trimmed_bin_center in trimmed_place_bin_centers
        ],
        dtype=np.uint16,
    )


def create_1d_decode_view(
    *,
    posterior: Any,
    sampling_frequency: float,
    observed_position: np.ndarray,
) -> Any:
    """Return one SortingView decoded-linear-position view."""
    import sortingview.views.franklab as vvf

    trimmed_posterior = discretize_and_trim(posterior)
    observations_per_time = get_observations_per_time(trimmed_posterior, posterior)
    trimmed_bin_center_index = get_trimmed_bin_center_index(
        posterior.position.values,
        trimmed_posterior.position.values,
    )
    return vvf.DecodedLinearPositionData(
        values=trimmed_posterior.values,
        positions=trimmed_bin_center_index,
        frame_bounds=observations_per_time,
        positions_key=posterior.position.values.astype(np.float32),
        observed_positions=np.asarray(observed_position, dtype=np.float32),
        start_time_sec=float(posterior.time.values[0]),
        sampling_frequency=float(sampling_frequency),
    )


def make_speed_view(time_grid: np.ndarray, speed: np.ndarray) -> Any:
    """Return a SortingView speed trace."""
    import sortingview.views as vv

    speed_view = vv.TimeseriesGraph()
    speed_view.add_line_series(
        name="Speed (cm/s)",
        t=np.asarray(time_grid, dtype=np.float32),
        y=np.asarray(speed, dtype=np.float32),
        color="black",
        width=1,
    )
    return speed_view


def make_state_probability_view(results: Any, *, posterior_name: str) -> Any:
    """Return state probability traces for one prediction dataset."""
    import sortingview.views as vv

    color_cycle = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    probability_view = vv.TimeseriesGraph()
    for state, color in zip(results.state.values, color_cycle, strict=False):
        probability_view.add_line_series(
            name=str(state),
            t=np.asarray(results.time.values, dtype=np.float32),
            y=np.asarray(
                results[posterior_name].sel(state=state).sum("position"),
                dtype=np.float32,
            ),
            color=color,
            width=1,
        )
    return probability_view


def make_multiunit_rate_view(
    *,
    time_grid: np.ndarray,
    spike_indicator: np.ndarray,
    time_bin_size_s: float,
    region: str,
) -> Any:
    """Return a simple population spike-rate trace for one region."""
    import sortingview.views as vv

    multiunit_rate_hz = np.sum(spike_indicator, axis=1) / float(time_bin_size_s)
    rate_view = vv.TimeseriesGraph()
    rate_view.add_line_series(
        name=f"{region.upper()} population rate (spikes/s)",
        t=np.asarray(time_grid, dtype=np.float32),
        y=np.asarray(multiunit_rate_hz, dtype=np.float32),
        color="black",
        width=1,
    )
    return rate_view


def make_raster_view(
    *,
    sorting: Any,
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids: list[Any] | None = None,
) -> Any:
    """Return a raster view for spikes inside the decoded time span."""
    import sortingview.views as vv

    plot_items = []
    start_time = float(time_grid[0])
    end_time = float(time_grid[-1])
    all_timestamps = np.asarray(timestamps_ephys_all, dtype=float)
    selected_unit_ids = list(sorting.get_unit_ids()) if unit_ids is None else unit_ids
    for unit_id in selected_unit_ids:
        spike_indices = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=int)
        spike_times = all_timestamps[spike_indices]
        spike_times = spike_times[(spike_times > start_time) & (spike_times <= end_time)]
        if spike_times.size:
            plot_items.append(
                vv.RasterPlotItem(
                    unit_id=str(unit_id),
                    spike_times_sec=spike_times.astype(np.float32),
                )
            )
    return vv.RasterPlot(start_time_sec=start_time, end_time_sec=end_time, plots=plot_items)


def make_region_figurl_panel(
    *,
    region: str,
    results: Any,
    sorting: Any,
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids: list[Any],
    linear_position: np.ndarray,
    speed: np.ndarray,
    spike_indicator: np.ndarray,
    time_bin_size_s: float,
    posterior_name: str,
) -> Any:
    """Return a vertical SortingView panel for one region."""
    import sortingview.views as vv

    decode_view = create_1d_decode_view(
        posterior=results[posterior_name].sum("state"),
        sampling_frequency=1.0 / float(time_bin_size_s),
        observed_position=linear_position,
    )
    return vv.Box(
        direction="vertical",
        show_titles=True,
        items=[
            vv.LayoutItem(decode_view, stretch=3, title=f"Decode {region.upper()}"),
            vv.LayoutItem(
                make_state_probability_view(results, posterior_name=posterior_name),
                stretch=1,
                title=f"State probability {region.upper()}",
            ),
            vv.LayoutItem(
                make_multiunit_rate_view(
                    time_grid=time_grid,
                    spike_indicator=spike_indicator,
                    time_bin_size_s=time_bin_size_s,
                    region=region,
                ),
                stretch=1,
                title=f"Population rate {region.upper()}",
            ),
            vv.LayoutItem(
                make_raster_view(
                    sorting=sorting,
                    timestamps_ephys_all=timestamps_ephys_all,
                    time_grid=time_grid,
                    unit_ids=unit_ids,
                ),
                stretch=1,
                title=f"Raster {region.upper()}",
            ),
            vv.LayoutItem(make_speed_view(time_grid, speed), stretch=1, title="Speed"),
        ],
    )


def make_combined_figurl_view(
    *,
    regions: tuple[str, ...],
    results_by_region: dict[str, Any],
    sortings: dict[str, Any],
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids_by_region: dict[str, list[Any]],
    linear_position: np.ndarray,
    speed: np.ndarray,
    spike_indicator_by_region: dict[str, np.ndarray],
    time_bin_size_s: float,
    posterior_name: str,
) -> Any:
    """Return a one-region or two-region SortingView layout."""
    import sortingview.views as vv

    panels = [
        make_region_figurl_panel(
            region=region,
            results=results_by_region[region],
            sorting=sortings[region],
            timestamps_ephys_all=timestamps_ephys_all,
            time_grid=time_grid,
            unit_ids=unit_ids_by_region[region],
            linear_position=linear_position,
            speed=speed,
            spike_indicator=spike_indicator_by_region[region],
            time_bin_size_s=time_bin_size_s,
            posterior_name=posterior_name,
        )
        for region in regions
    ]
    if len(panels) == 1:
        return vv.Box(
            direction="horizontal",
            show_titles=True,
            height=800,
            items=[vv.LayoutItem(panels[0], stretch=1)],
        )
    return vv.Splitter(
        direction="horizontal",
        item1=vv.LayoutItem(panels[0], stretch=1),
        item2=vv.LayoutItem(panels[1], stretch=1),
    )


def save_figurl(
    *,
    figurl_path: Path,
    regions: tuple[str, ...],
    animal_name: str,
    date: str,
    epoch: str,
    n_folds: int,
    unit_selection_label: str,
    results_by_region: dict[str, Any],
    sortings: dict[str, Any],
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids_by_region: dict[str, list[Any]],
    linear_position: np.ndarray,
    speed: np.ndarray,
    spike_indicator_by_region: dict[str, np.ndarray],
    time_bin_size_s: float,
    posterior_name: str,
) -> Path:
    """Create and save one combined figurl URL."""
    os.environ.setdefault("KACHERY_ZONE", "franklab.default")
    figurl_time_zero = float(time_grid[0])
    figurl_time_grid = np.asarray(time_grid, dtype=float) - figurl_time_zero
    figurl_timestamps_ephys_all = (
        np.asarray(timestamps_ephys_all, dtype=float) - figurl_time_zero
    )
    figurl_results_by_region = {
        region: results.assign_coords(time=figurl_time_grid)
        for region, results in results_by_region.items()
    }
    view = make_combined_figurl_view(
        regions=regions,
        results_by_region=figurl_results_by_region,
        sortings=sortings,
        timestamps_ephys_all=figurl_timestamps_ephys_all,
        time_grid=figurl_time_grid,
        unit_ids_by_region=unit_ids_by_region,
        linear_position=linear_position,
        speed=speed,
        spike_indicator_by_region=spike_indicator_by_region,
        time_bin_size_s=time_bin_size_s,
        posterior_name=posterior_name,
    )
    label = (
        f"{animal_name} {date} {epoch} 1D CV decode "
        f"regions {','.join(regions)} {n_folds} folds "
        f"unit_selection {unit_selection_label} "
        f"posterior {posterior_name} "
        f"figurl_time_zero_s {figurl_time_zero:.6f}"
    )
    figurl_path.parent.mkdir(parents=True, exist_ok=True)
    view_url = view.url(label=label)
    with open(figurl_path, "w", encoding="utf-8") as file:
        file.write(view_url)
        file.write("\n")
    print(view_url)
    return figurl_path


def predict_region(
    *,
    region: str,
    sorting: Any,
    classifier_paths: dict[tuple[str, int], Path],
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids: list[Any],
    fold_by_time: np.ndarray,
    linear_position: np.ndarray,
    speed: np.ndarray,
    n_folds: int,
    state_names: list[str],
    causal_only: bool,
) -> tuple[Any, np.ndarray]:
    """Predict one combined cross-validated posterior dataset for one region."""
    spike_indicator = get_spike_indicator(
        sorting,
        timestamps_ephys_all=timestamps_ephys_all,
        time_grid=time_grid,
        unit_ids=unit_ids,
    )
    if spike_indicator.shape[1] == 0:
        raise ValueError(f"Region {region!r} has no units to predict.")

    fold_results: list[Any] = []
    for fold in range(n_folds):
        fold_mask = fold_by_time == fold
        if not np.any(fold_mask):
            raise ValueError(f"Fold {fold} has no assigned prediction bins.")

        print(
            f"Predicting {region} fold {fold + 1}/{n_folds} with "
            f"{int(np.sum(fold_mask))} time bins."
        )
        classifier_path = classifier_paths[(region, fold)]
        classifier = load_classifier(classifier_path)
        validate_classifier_place_fields(
            classifier,
            classifier_path=classifier_path,
        )
        fold_results.append(
            classifier.predict(
                spike_indicator[fold_mask],
                time=time_grid[fold_mask],
                state_names=state_names,
                is_compute_acausal=not causal_only,
                use_gpu=True,
            )
        )

    combined = concatenate_fold_results(
        fold_results,
        time_grid=time_grid,
        linear_position=linear_position,
        speed=speed,
        fold_by_time=fold_by_time,
    )
    return combined, spike_indicator


def load_saved_prediction_result(
    path: Path,
    *,
    time_grid: np.ndarray,
    posterior_name: str,
) -> Any:
    """Load one existing prediction result and validate its time coordinate."""
    import xarray as xr

    result = xr.open_dataset(path).load()
    predicted_time = np.asarray(result.time, dtype=float)
    expected_time = np.asarray(time_grid, dtype=float)
    if predicted_time.size != expected_time.size:
        raise ValueError(
            f"Saved prediction result {path} has {predicted_time.size} time bins, "
            f"but the current settings expect {expected_time.size}."
        )
    if not np.allclose(predicted_time, expected_time, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Saved prediction result {path} does not match the current time grid. "
            "Check --time-bin-size-s, --position-offset, --animal-name, --date, "
            "and --epoch."
        )
    if posterior_name not in result:
        raise ValueError(
            f"Saved prediction result {path} does not contain {posterior_name!r}. "
            "Check whether the file was produced with matching --causal-only."
        )
    return result


def run(args: argparse.Namespace) -> None:
    """Run the 1D decoder prediction workflow."""
    validate_arguments(args)
    configure_cuda_visible_devices(args.cuda_visible_devices)

    analysis_path = get_analysis_path_for_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    selected_regions = select_regions(args.region)
    should_generate_figurl = args.figurl or args.figurl_only
    posterior_kind = get_posterior_kind(causal_only=args.causal_only)
    posterior_name = get_posterior_name(causal_only=args.causal_only)
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
    classifier_paths = build_classifier_output_paths(
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
    prediction_paths = build_prediction_output_paths(
        get_predict_output_dir(analysis_path),
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
        posterior_kind=posterior_kind,
    )
    figurl_path_value = (
        figurl_output_path(
            get_predict_output_dir(analysis_path),
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
            posterior_kind=posterior_kind,
        )
        if should_generate_figurl
        else None
    )

    output_preflight_paths = (
        [figurl_path_value]
        if args.figurl_only and figurl_path_value is not None
        else list(prediction_paths.values())
    )
    if figurl_path_value is not None:
        if figurl_path_value not in output_preflight_paths:
            output_preflight_paths.append(figurl_path_value)
    preflight_no_existing(
        output_preflight_paths,
        overwrite=args.overwrite,
        output_kind="figurl" if args.figurl_only else "prediction",
    )
    if args.figurl_only:
        require_existing_paths(
            list(prediction_paths.values()),
            input_kind="prediction",
        )
    else:
        require_existing_paths(
            list(classifier_paths.values()),
            input_kind="classifier",
        )

    session = load_required_session_inputs(
        analysis_path=analysis_path,
        animal_name=args.animal_name,
        epoch=args.epoch,
    )
    lap_counts = validate_fold_count(
        session["trajectory_intervals"],
        n_folds=args.n_folds,
    )
    _train_intervals, _test_intervals, fold_interval_records = build_lapwise_cv(
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
    linear_position, _track_graph, edge_order, edge_spacing = linearize_full_w_position(
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
    fold_by_time = build_full_epoch_prediction_fold(time_grid, fold_interval_records)
    state_names = get_state_names(direction=args.direction, discrete_var=args.discrete_var)

    action_label = (
        "Regenerating figurl from existing predictions"
        if args.figurl_only
        else "Predicting"
    )
    print(
        f"{action_label} for {args.animal_name} {args.date} epoch {args.epoch}; "
        f"regions={list(selected_regions)}, n_folds={args.n_folds}, "
        f"random_state={args.random_state}, direction={args.direction}, "
        f"movement={args.movement}, unit_selection={unit_selection_label}."
        f" posterior={posterior_name}."
    )
    results_by_region: dict[str, Any] = {}
    spike_indicator_by_region: dict[str, np.ndarray] = {}
    saved_prediction_paths: list[Path] = []
    for region in selected_regions:
        if args.figurl_only:
            combined_result = load_saved_prediction_result(
                prediction_paths[region],
                time_grid=time_grid,
                posterior_name=posterior_name,
            )
            spike_indicator = get_spike_indicator(
                sortings[region],
                timestamps_ephys_all=session["timestamps_ephys_all"],
                time_grid=time_grid,
                unit_ids=unit_ids_by_region[region],
            )
            results_by_region[region] = combined_result
            spike_indicator_by_region[region] = spike_indicator
            print(
                f"Loaded existing {region} prediction results from "
                f"{prediction_paths[region]}"
            )
            continue

        combined_result, spike_indicator = predict_region(
            region=region,
            sorting=sortings[region],
            classifier_paths=classifier_paths,
            timestamps_ephys_all=session["timestamps_ephys_all"],
            time_grid=time_grid,
            unit_ids=unit_ids_by_region[region],
            fold_by_time=fold_by_time,
            linear_position=linear_position,
            speed=speed,
            n_folds=args.n_folds,
            state_names=state_names,
            causal_only=args.causal_only,
        )
        combined_result.attrs.update(
            {
                "animal_name": args.animal_name,
                "date": args.date,
                "epoch": args.epoch,
                "region": region,
                "n_folds": int(args.n_folds),
                "random_state": int(args.random_state),
                "time_bin_size_s": float(args.time_bin_size_s),
                "position_offset": int(args.position_offset),
                "speed_threshold_cm_s": float(args.speed_threshold_cm_s),
                "position_std": float(args.position_std),
                "place_bin_size": float(args.place_bin_size),
                "movement_var": float(args.movement_var),
                "discrete_var": str(args.discrete_var),
                "branch_gap_cm": float(args.branch_gap_cm),
                "direction": str(args.direction),
                "movement": str(args.movement),
                "v1_ripple_glm_units": str(args.v1_ripple_glm_units),
                "v1_ripple_glm_devexp_threshold": (
                    ""
                    if args.v1_ripple_glm_devexp_threshold is None
                    else str(args.v1_ripple_glm_devexp_threshold)
                ),
                "unit_selection_label": unit_selection_label,
                "unit_selection_method": unit_selection_by_region[region]["method"],
                "n_units_available": int(unit_selection_by_region[region]["n_available"]),
                "n_units_selected": int(unit_selection_by_region[region]["n_selected"]),
                "v1_ripple_glm_p_value_threshold": float(
                    DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD
                ),
                "posterior_kind": posterior_kind,
                "posterior_name": posterior_name,
                "causal_only": str(args.causal_only),
                "is_compute_acausal": str(not args.causal_only),
                "unit_selection_source_path": str(
                    unit_selection_by_region[region].get("source_path", "")
                ),
                "prediction_scope": "full_trimmed_epoch_earlier_gap_fold",
            }
        )
        output_path = prediction_paths[region]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_result.to_netcdf(output_path)
        saved_prediction_paths.append(output_path)
        results_by_region[region] = combined_result
        spike_indicator_by_region[region] = spike_indicator
        print(f"Saved {region} prediction results to {output_path}")

    saved_figurl_path = None
    if figurl_path_value is not None:
        saved_figurl_path = save_figurl(
            figurl_path=figurl_path_value,
            regions=selected_regions,
            animal_name=args.animal_name,
            date=args.date,
            epoch=args.epoch,
            n_folds=args.n_folds,
            unit_selection_label=unit_selection_label,
            results_by_region=results_by_region,
            sortings=sortings,
            timestamps_ephys_all=session["timestamps_ephys_all"],
            time_grid=time_grid,
            unit_ids_by_region=unit_ids_by_region,
            linear_position=linear_position,
            speed=speed,
            spike_indicator_by_region=spike_indicator_by_region,
            time_bin_size_s=args.time_bin_size_s,
            posterior_name=posterior_name,
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
            "figurl": should_generate_figurl,
            "causal_only": args.causal_only,
            "posterior_kind": posterior_kind,
            "posterior_name": posterior_name,
            "figurl_only": args.figurl_only,
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
            "classifier_paths": classifier_paths,
            "prediction_paths": prediction_paths,
            "saved_prediction_paths": saved_prediction_paths,
            "saved_figurl_path": saved_figurl_path,
        },
    )
    if args.figurl_only:
        print("Skipped posterior prediction and reused existing prediction result file(s).")
    else:
        print(f"Saved {len(saved_prediction_paths)} prediction result file(s).")
    print(f"Saved run metadata to {log_path}")


def main() -> None:
    """CLI entry point."""
    run(parse_arguments())


if __name__ == "__main__":
    main()
