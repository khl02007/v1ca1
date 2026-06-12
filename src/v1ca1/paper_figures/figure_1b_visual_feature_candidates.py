from __future__ import annotations

"""Find and plot Fig. 1B-style visual-feature candidate cells."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
)
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSITION_BIN_COUNT,
    FIGURE_FORMATS,
    PANEL_DARK_LIGHT_RASTER_BACKGROUND_ALPHA,
    PANEL_DARK_LIGHT_RASTER_COLOR,
    PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS,
    PANEL_DARK_LIGHT_TRAJECTORY_EPOCH_BACKGROUNDS,
    PANEL_DARK_LIGHT_VISUAL_LABEL_COLORS,
    TASK_PROGRESSION_SEGMENT_BOUNDARIES,
    build_output_path,
)
from v1ca1.paper_figures.figure_3 import (
    PANEL_A_LIGHT_EPOCHS,
    get_dark_epoch,
    get_dark_light_glm_selected_path,
    load_panel_a_example_data,
    parse_dataset_id,
    plot_panel_a_example,
)
from v1ca1.paper_figures.style import apply_paper_style, figure_size, save_figure
from v1ca1.raster.plot_place_field_heatmap import DEFAULT_SIGMA_BINS


DEFAULT_OUTPUT_NAME = "figure_1b_visual_feature_candidates"
DEFAULT_REGION = "v1"
DEFAULT_EXAMPLE_COUNT = 6
DEFAULT_MIN_FEATURE_PEAK_HZ = 2.5
DEFAULT_MAX_DARK_PEAK_HZ = 3.0
DEFAULT_MAX_DARK_MOVEMENT_FIRING_RATE_HZ = 1.25
DEFAULT_MIN_FEATURE_TO_OPPOSITE_RATIO = 1.5
FEATURE_RESPONSE_DENOMINATOR_OFFSET_HZ = 0.25
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_EXAMPLE_HEIGHT_MM = 39.0
LEFT_ARM_TRAJECTORIES = ("center_to_left", "left_to_center")
RIGHT_ARM_TRAJECTORIES = ("center_to_right", "right_to_center")
SIDE_TRAJECTORIES = {
    "left": LEFT_ARM_TRAJECTORIES,
    "right": RIGHT_ARM_TRAJECTORIES,
}
VISUAL_FEATURE_SIDE_BY_EPOCH = {
    "A": {
        "02_r1": "left",
        "06_r3": "right",
    },
    "B": {
        "02_r1": "right",
        "06_r3": "left",
    },
}


def _finite_robust_peak(values: np.ndarray) -> float:
    """Return a finite 95th-percentile peak estimate for one response vector."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.nanpercentile(values, 95))


def _trajectory_terminal_arm_mask(tp_grid: np.ndarray, trajectory_type: str) -> np.ndarray:
    """Return bins covering the terminal arm segment for one trajectory."""
    tp_grid = np.asarray(tp_grid, dtype=float)
    first_boundary, second_boundary = TASK_PROGRESSION_SEGMENT_BOUNDARIES
    if str(trajectory_type).startswith("center_to_"):
        return tp_grid >= float(second_boundary)
    return tp_grid <= float(first_boundary)


def _trajectory_arm_peaks(
    firing_rate_grid: np.ndarray,
    trajectories: Sequence[str],
    tp_grid: np.ndarray,
) -> dict[str, float]:
    """Return terminal-arm response peaks for all trajectory rows."""
    firing_rate_grid = np.asarray(firing_rate_grid, dtype=float)
    return {
        str(trajectory_type): _finite_robust_peak(
            firing_rate_grid[
                trajectory_index,
                _trajectory_terminal_arm_mask(tp_grid, str(trajectory_type)),
            ]
        )
        for trajectory_index, trajectory_type in enumerate(trajectories)
    }


def _side_response_mean(
    trajectory_peaks: Mapping[str, float],
    side_name: str,
) -> float:
    """Return mean terminal-arm peak for one physical side."""
    peaks = np.asarray(
        [trajectory_peaks[trajectory] for trajectory in SIDE_TRAJECTORIES[side_name]],
        dtype=float,
    )
    if not np.isfinite(peaks).any():
        return np.nan
    return float(np.nanmean(peaks))


def _opposite_side(side_name: str) -> str:
    """Return the opposite physical side name."""
    if side_name == "left":
        return "right"
    if side_name == "right":
        return "left"
    raise ValueError(f"Unknown side name {side_name!r}.")


def _unit_index(units: np.ndarray, unit_id: int) -> int:
    """Return the coordinate index for one unit ID."""
    matches = np.flatnonzero(np.asarray(units, dtype=int) == int(unit_id))
    if matches.size != 1:
        raise ValueError(f"Expected exactly one match for unit {unit_id}, found {matches.size}.")
    return int(matches[0])


def _score_visual_feature_candidate(
    *,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str,
    unit_id: int,
    feature_name: str,
    light_peaks: Mapping[str, Mapping[str, float]],
    dark_peaks: Mapping[str, float],
    dark_movement_firing_rate_hz: float,
    light_glm_score: float,
) -> dict[str, Any]:
    """Return one scored feature-matched light response candidate."""
    feature_side_by_epoch = VISUAL_FEATURE_SIDE_BY_EPOCH[feature_name]
    feature_epoch_peaks = [
        _side_response_mean(light_peaks[light_epoch], side_name)
        for light_epoch, side_name in feature_side_by_epoch.items()
    ]
    opposite_epoch_peaks = [
        _side_response_mean(light_peaks[light_epoch], _opposite_side(side_name))
        for light_epoch, side_name in feature_side_by_epoch.items()
    ]
    dark_values = np.asarray(list(dark_peaks.values()), dtype=float)
    feature_min_peak_hz = float(np.nanmin(feature_epoch_peaks))
    feature_mean_peak_hz = float(np.nanmean(feature_epoch_peaks))
    opposite_mean_peak_hz = float(np.nanmean(opposite_epoch_peaks))
    dark_max_peak_hz = float(np.nanmax(dark_values))
    dark_mean_peak_hz = float(np.nanmean(dark_values))
    feature_to_dark_ratio = feature_min_peak_hz / (
        dark_max_peak_hz + FEATURE_RESPONSE_DENOMINATOR_OFFSET_HZ
    )
    feature_to_opposite_ratio = feature_mean_peak_hz / (
        opposite_mean_peak_hz + FEATURE_RESPONSE_DENOMINATOR_OFFSET_HZ
    )
    score = (
        feature_to_dark_ratio
        * feature_to_opposite_ratio
        * np.sqrt(max(feature_min_peak_hz, 0.0))
    )
    return {
        "animal_name": animal_name,
        "date": date,
        "dark_epoch": dark_epoch,
        "region": region,
        "unit_id": int(unit_id),
        "feature_name": feature_name,
        "score": float(score),
        "feature_min_peak_hz": feature_min_peak_hz,
        "feature_mean_peak_hz": feature_mean_peak_hz,
        "feature_to_dark_ratio": float(feature_to_dark_ratio),
        "feature_to_opposite_ratio": float(feature_to_opposite_ratio),
        "opposite_mean_peak_hz": opposite_mean_peak_hz,
        "dark_max_peak_hz": dark_max_peak_hz,
        "dark_mean_peak_hz": dark_mean_peak_hz,
        "dark_movement_firing_rate_hz": float(dark_movement_firing_rate_hz),
        "light_glm_score_bits_per_spike": float(light_glm_score),
    }


def _candidate_passes_filters(
    candidate: Mapping[str, Any],
    *,
    min_feature_peak_hz: float,
    max_dark_peak_hz: float,
    max_dark_movement_firing_rate_hz: float,
    min_feature_to_opposite_ratio: float,
) -> bool:
    """Return whether one candidate passes the visual-feature filters."""
    return (
        float(candidate["feature_min_peak_hz"]) >= float(min_feature_peak_hz)
        and float(candidate["dark_max_peak_hz"]) <= float(max_dark_peak_hz)
        and float(candidate["dark_movement_firing_rate_hz"])
        <= float(max_dark_movement_firing_rate_hz)
        and float(candidate["feature_to_opposite_ratio"])
        >= float(min_feature_to_opposite_ratio)
    )


def scan_visual_feature_candidates(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
) -> list[dict[str, Any]]:
    """Score units by dark-suppressed, feature-matched light-arm responses."""
    import xarray as xr

    candidates: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        dark_epoch = get_dark_epoch(animal_name, date, dataset_dark_epoch)
        light_datasets = {}
        missing_light_paths = []
        for light_epoch in PANEL_A_LIGHT_EPOCHS:
            path = get_dark_light_glm_selected_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                model_name="visual",
            )
            if not path.exists():
                missing_light_paths.append(path)
                continue
            light_datasets[light_epoch] = xr.open_dataset(path)
        if missing_light_paths:
            for dataset_obj in light_datasets.values():
                dataset_obj.close()
            print(
                f"Skipping {animal_name} {date}; missing visual selected files: "
                + ", ".join(str(path) for path in missing_light_paths)
            )
            continue

        unit_sets = [
            set(np.asarray(dataset_obj.coords["unit"].values, dtype=int).tolist())
            for dataset_obj in light_datasets.values()
        ]
        shared_units = sorted(set.intersection(*unit_sets))
        for unit_id in shared_units:
            light_peaks: dict[str, dict[str, float]] = {}
            dark_peaks_by_light: dict[str, dict[str, float]] = {}
            dark_movement_rates = []
            light_scores = []
            for light_epoch, dataset_obj in light_datasets.items():
                unit_index = _unit_index(dataset_obj.coords["unit"].values, unit_id)
                trajectories = [str(value) for value in dataset_obj.coords["trajectory"].values]
                tp_grid = np.asarray(dataset_obj.coords["tp_grid"].values, dtype=float)
                light_grid = np.asarray(
                    dataset_obj["light_hz_grid"].isel(unit=unit_index).values,
                    dtype=float,
                )
                dark_grid = np.asarray(
                    dataset_obj["dark_hz_grid"].isel(unit=unit_index).values,
                    dtype=float,
                )
                light_peaks[light_epoch] = _trajectory_arm_peaks(
                    light_grid,
                    trajectories,
                    tp_grid,
                )
                dark_peaks_by_light[light_epoch] = _trajectory_arm_peaks(
                    dark_grid,
                    trajectories,
                    tp_grid,
                )
                dark_movement_rates.append(
                    float(
                        dataset_obj["dark_movement_firing_rate_hz"]
                        .isel(unit=unit_index)
                        .values
                    )
                )
                light_scores.append(
                    float(
                        np.nanmean(
                            dataset_obj["ll_bits_per_spike_cv_light"]
                            .isel(unit=unit_index)
                            .values
                        )
                    )
                )

            dark_peaks = {
                trajectory: float(
                    np.nanmean(
                        [
                            dark_peaks_by_light[light_epoch][trajectory]
                            for light_epoch in PANEL_A_LIGHT_EPOCHS
                        ]
                    )
                )
                for trajectory in (*LEFT_ARM_TRAJECTORIES, *RIGHT_ARM_TRAJECTORIES)
            }
            for feature_name in VISUAL_FEATURE_SIDE_BY_EPOCH:
                candidates.append(
                    _score_visual_feature_candidate(
                        animal_name=animal_name,
                        date=date,
                        dark_epoch=dark_epoch,
                        region=region,
                        unit_id=unit_id,
                        feature_name=feature_name,
                        light_peaks=light_peaks,
                        dark_peaks=dark_peaks,
                        dark_movement_firing_rate_hz=float(np.nanmean(dark_movement_rates)),
                        light_glm_score=float(np.nanmean(light_scores)),
                    )
                )

        for dataset_obj in light_datasets.values():
            dataset_obj.close()

    candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(float(candidate["score"]))
    ]
    candidates.sort(key=lambda candidate: float(candidate["score"]), reverse=True)
    return candidates


def select_visual_feature_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    example_count: int,
    min_feature_peak_hz: float,
    max_dark_peak_hz: float,
    max_dark_movement_firing_rate_hz: float,
    min_feature_to_opposite_ratio: float,
) -> list[dict[str, Any]]:
    """Return top unique units passing the visual-feature filters."""
    selected: list[dict[str, Any]] = []
    seen_units: set[tuple[str, str, str, int]] = set()
    for candidate in candidates:
        if not _candidate_passes_filters(
            candidate,
            min_feature_peak_hz=min_feature_peak_hz,
            max_dark_peak_hz=max_dark_peak_hz,
            max_dark_movement_firing_rate_hz=max_dark_movement_firing_rate_hz,
            min_feature_to_opposite_ratio=min_feature_to_opposite_ratio,
        ):
            continue
        unit_key = (
            str(candidate["animal_name"]),
            str(candidate["date"]),
            str(candidate["region"]),
            int(candidate["unit_id"]),
        )
        if unit_key in seen_units:
            continue
        selected.append(dict(candidate))
        seen_units.add(unit_key)
        if len(selected) >= int(example_count):
            break
    return selected


def _format_candidate_title(candidate: Mapping[str, Any]) -> str:
    """Return a compact candidate label for one plotted row."""
    return (
        f"{candidate['animal_name']} {candidate['date']} "
        f"{str(candidate['region']).upper()} unit {int(candidate['unit_id'])}, "
        f"feature {candidate['feature_name']}  "
        f"light min {float(candidate['feature_min_peak_hz']):.1f} Hz, "
        f"dark max {float(candidate['dark_max_peak_hz']):.1f} Hz"
    )


def plot_visual_feature_candidate_examples(
    *,
    data_root: Path,
    output_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path | None,
    refresh_panel_example_cache: bool,
    dpi: int,
) -> Path:
    """Plot selected visual-feature candidates in a Fig. 1B-style SVG."""
    import matplotlib.pyplot as plt

    if not candidates:
        raise ValueError("No visual-feature candidates were selected for plotting.")

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            DEFAULT_FIGURE_WIDTH_MM,
            DEFAULT_EXAMPLE_HEIGHT_MM * len(candidates),
        ),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(nrows=len(candidates), ncols=1)
    for row_index, candidate in enumerate(candidates):
        ax = fig.add_subplot(grid[row_index, 0])
        example = load_panel_a_example_data(
            data_root=data_root,
            animal_name=str(candidate["animal_name"]),
            date=str(candidate["date"]),
            region=str(candidate["region"]),
            unit_id=int(candidate["unit_id"]),
            dark_epoch=str(candidate["dark_epoch"]),
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
        plot_panel_a_example(
            ax,
            example,
            visual_label_colors=PANEL_DARK_LIGHT_VISUAL_LABEL_COLORS,
            raster_color=PANEL_DARK_LIGHT_RASTER_COLOR,
            trajectory_epoch_color_overrides=PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS,
            trajectory_epoch_backgrounds=PANEL_DARK_LIGHT_TRAJECTORY_EPOCH_BACKGROUNDS,
            epoch_background_alpha=PANEL_DARK_LIGHT_RASTER_BACKGROUND_ALPHA,
        )
        for child_axis in ax.child_axes:
            if child_axis.get_xlabel():
                child_axis.xaxis.label.set_text("Norm. path progression")
        ax.set_title(_format_candidate_title(candidate), fontsize=7.0, pad=1.0)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved visual-feature candidate examples to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Fig. 1B-style candidate plotting."""
    parser = argparse.ArgumentParser(
        description="Find and plot Fig. 1B-style visual-feature candidate cells."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default="svg",
        help="Output format. Default: svg",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date or animal:date:dark_epoch. "
            "May be repeated. Default: use all processed paper data sets."
        ),
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"Region to scan. Default: {DEFAULT_REGION}",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=DEFAULT_EXAMPLE_COUNT,
        help=f"Number of candidate cells to plot. Default: {DEFAULT_EXAMPLE_COUNT}",
    )
    parser.add_argument(
        "--min-feature-peak-hz",
        type=float,
        default=DEFAULT_MIN_FEATURE_PEAK_HZ,
        help=(
            "Minimum feature-matched light arm peak in each light epoch. "
            f"Default: {DEFAULT_MIN_FEATURE_PEAK_HZ}"
        ),
    )
    parser.add_argument(
        "--max-dark-peak-hz",
        type=float,
        default=DEFAULT_MAX_DARK_PEAK_HZ,
        help=(
            "Maximum allowed dark terminal-arm peak from the selected visual GLM. "
            f"Default: {DEFAULT_MAX_DARK_PEAK_HZ}"
        ),
    )
    parser.add_argument(
        "--max-dark-movement-firing-rate-hz",
        type=float,
        default=DEFAULT_MAX_DARK_MOVEMENT_FIRING_RATE_HZ,
        help=(
            "Maximum allowed full dark movement firing rate. "
            f"Default: {DEFAULT_MAX_DARK_MOVEMENT_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-feature-to-opposite-ratio",
        type=float,
        default=DEFAULT_MIN_FEATURE_TO_OPPOSITE_RATIO,
        help=(
            "Minimum ratio of feature-matched light peaks to opposite-feature peaks. "
            f"Default: {DEFAULT_MIN_FEATURE_TO_OPPOSITE_RATIO}"
        ),
    )
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=DEFAULT_POSITION_BIN_COUNT,
        help=f"Number of normalized-position bins. Default: {DEFAULT_POSITION_BIN_COUNT}",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore. Default: {DEFAULT_POSITION_OFFSET}",
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
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--panel-example-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached candidate raster/rate data. Default: <output-dir>/cache.",
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help="Recompute selected candidate rasters/rates even when cache exists.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Fig. 1B-style candidate scan and plot."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    candidates = scan_visual_feature_candidates(
        data_root=args.data_root,
        datasets=datasets,
        region=args.region,
    )
    selected_candidates = select_visual_feature_candidates(
        candidates,
        example_count=args.example_count,
        min_feature_peak_hz=args.min_feature_peak_hz,
        max_dark_peak_hz=args.max_dark_peak_hz,
        max_dark_movement_firing_rate_hz=args.max_dark_movement_firing_rate_hz,
        min_feature_to_opposite_ratio=args.min_feature_to_opposite_ratio,
    )
    print(f"Scanned {len(candidates)} feature/unit candidates.")
    print(f"Selected {len(selected_candidates)} candidate cells for plotting.")
    for index, candidate in enumerate(selected_candidates, start=1):
        print(
            f"{index}: {_format_candidate_title(candidate)}; "
            f"score {float(candidate['score']):.2f}, "
            f"opposite {float(candidate['opposite_mean_peak_hz']):.1f} Hz, "
            f"dark movement {float(candidate['dark_movement_firing_rate_hz']):.2f} Hz"
        )
    plot_visual_feature_candidate_examples(
        data_root=args.data_root,
        output_path=output_path,
        candidates=selected_candidates,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
