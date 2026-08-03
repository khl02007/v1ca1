"""Generate Figure 2."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm, to_hex

from v1ca1.helper.plot_wtrack_schematic import (
    draw_large_ovals,
    get_w_track_geometry,
)
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
    TRAJECTORY_TYPES,
    get_analysis_path,
    load_trajectory_intervals,
)
from v1ca1.helper.wtrack import get_wtrack_total_length
from v1ca1.paper_figures import _figure_2_base as _figure_2
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DECODING_PERMUTATION_COUNT,
    DECODING_PERMUTATION_SEED,
    DECODING_SIGNIFICANCE_BRACKET_HEIGHT,
    DECODING_SIGNIFICANCE_BRACKET_LINEWIDTH,
    DECODING_SIGNIFICANCE_LABEL_FONTSIZE,
    DECODING_SIGNIFICANCE_LABEL_Y_OFFSET,
    _align_absolute_error_with_times,
    _intervalset_to_arrays,
    significance_stars,
    stratified_median_permutation_test,
)
from v1ca1.paper_figures._dark_light import (
    GLM_EMPIRICAL_COLOR,
    PANEL_E_CROSS_COMPARISONS,
    PANEL_E_PLACE_MODEL_NAME,
    PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM,
    PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT,
    PANEL_G_INDEPENDENT_BASIS_ICON_TOP,
    PANEL_G_INDEPENDENT_BASIS_ICON_WIDTH,
    _load_decoding_tsd,
    _draw_panel_g_basis_icon,
    _draw_panel_h_track,
    build_panel_quant_epoch_specs,
    get_cross_trajectory_decoding_tsd_paths,
    get_within_epoch_decoding_tsd_paths,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_arm_side_outlines


DEFAULT_OUTPUT_NAME = "figure_2"
DEFAULT_FIGURE_WIDTH_MM = _figure_2.DEFAULT_FIGURE_WIDTH_MM
DEFAULT_FIGURE_HEIGHT_MM = (
    _figure_2.PANEL_A_SINGLE_ROW_HEIGHT_MM
    + _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM
    + _figure_2.PANEL_D_ROW_HEIGHT_MM
)
PANEL_B2C2_ROW_WIDTH_RATIOS = (1.0, 1.0)
PANEL_B2C2_ROW_WSPACE = 0.035
PANEL_D2E2_ROW_WIDTH_RATIOS = (1.0, 1.0)
PANEL_D2E2_ROW_WSPACE = 0.050
PANEL_C2_SIGNIFICANCE_BRACKET_X = (1.0, 2.0)
PANEL_C2_SIGNIFICANCE_BRACKET_Y_FRACTION = 0.82
PANEL_C2_RIGHT_SIGNIFICANCE_BRACKET_Y_FRACTION = 0.68
PANEL_C2_ERROR_AXIS_LABEL = "|Norm. error|"
PANEL_E_DECODING_ANALYSES = ("cross_trajectory", "place")
PANEL_E_EXPECTED_MEDIAN_DIFFERENCE_SIGNS = {
    "cross_trajectory": 1.0,
    "place": -1.0,
}
PANEL_E_TRIAL_ERROR_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "epoch_type",
    "epoch",
    "region",
    "analysis",
    "comparison",
    "comparison_label",
    "transfer_family",
    "encoding_trajectory",
    "decoding_trajectory",
    "trial_index",
    "trial_start",
    "trial_end",
    "trial_median_absolute_error",
    "n_samples",
    "true_path",
    "decoded_path",
)
PANEL_E_PERMUTATION_RESULT_COLUMNS = (
    "animal_name",
    "analysis",
    "median_difference",
    "p_two_sided",
    "p_less",
    "p_greater",
    "n_permutations",
)
PANEL_A2_SINGLE_ROW_SCHEMATIC_AXIS_LEFT = -0.055
PANEL_D2_SCHEMATIC_AXIS_BOUNDS = _figure_2.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS
PANEL_D2_RESULT_AXIS_BOUNDS = _figure_2.PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS
PANEL_D2_DARK_TRACK_CENTER_X = 0.18
PANEL_D2_SEGMENT_TRACK_CENTER_X = 0.405
PANEL_D2_LIGHT_TRACK_CENTER_X = 0.615
PANEL_D2_BASIS_ICON_CENTER_X = 0.5 * (
    PANEL_D2_DARK_TRACK_CENTER_X + PANEL_D2_LIGHT_TRACK_CENTER_X
)
PANEL_D2_PREDICT_TRACK_CENTER_X = 0.860
PANEL_D2_SHARED_PLUS_X = 0.295
PANEL_D2_SHARED_ARROW_X = (0.500, 0.545)
PANEL_D2_EQUALS_X = 0.5 * (PANEL_D2_SHARED_ARROW_X[0] + PANEL_D2_SHARED_ARROW_X[1])
PANEL_D2_CUE_SWAP_ARROW_MARGIN = 0.006
PANEL_D2_INDEPENDENT_ROW_Y_OFFSET = 0.0
PANEL_D2_SHARED_ROW_Y_OFFSET = -0.070
PANEL_D2_SEGMENT_LABEL_GAP = 0.095
PANEL_D2_CUE_SWAP_LABEL_Y_OFFSET = 0.080
PANEL_D2_RIGHT_ARM_OUTLINE_COLOR = "#0072B2"
PANEL_D2_PLACE_FIELD_COLORS = ("#221150", "#B73779", "#FCFDBF")
PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP = "inferno"
PANEL_D2_DARK_SCAFFOLD_FIELD_COLOR_VALUES = (0.08, 0.54, 0.92)
PANEL_D2_DARK_SCAFFOLD_FIELD_BASE_COLORS = tuple(
    to_hex(colormaps[PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP](value))
    for value in PANEL_D2_DARK_SCAFFOLD_FIELD_COLOR_VALUES
)
PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_COLORS = (
    PANEL_D2_DARK_SCAFFOLD_FIELD_BASE_COLORS
)
PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_COLORS = PANEL_D2_DARK_SCAFFOLD_FIELD_BASE_COLORS
PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_COLORS = PANEL_D2_DARK_SCAFFOLD_FIELD_BASE_COLORS
PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_RATE_GAIN = 0.25
PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN = 0.65
PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN = 1.0
PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA = 0.65
PANEL_D2_RATE_COLORBAR_BOUNDS = (0.690, 0.405, 0.160, 0.025)
PANEL_D2_RATE_COLORBAR_AXIS_LABEL = "panel_d2_relative_firing_rate_colorbar"
PANEL_D2_RATE_COLORBAR_LABEL = "Relative firing rate"
PANEL_D2_RATE_COLORBAR_ENDPOINT_LABELS = ("Low", "High")
PANEL_D2_RATE_COLORBAR_FONTSIZE = _figure_2.MIN_PUBLICATION_FONTSIZE_PT
PANEL_D2_DARK_FIELD_PLACE_FIELD_ALPHA = 0.5
PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_ALPHA = 1.0
PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA = 1.0
PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_ALPHA = 1.0
PANEL_D2_SEGMENT_ARROW_COLOR = "black"
PANEL_D2_SEGMENT_ARROW_Y_MARGIN = 0.34
PANEL_D2_SEGMENT_ARROW_LINEWIDTH = 0.9
PANEL_D2_SEGMENT_ARROW_MUTATION_SCALE = 6.8
PANEL_D2_SEGMENT_OVAL_REGIONS = ("left_arm", "right_arm")
PANEL_D2_SEGMENT_OVAL_FILL_COLOR = "#8A8A8A"
PANEL_D2_SEGMENT_OVAL_EDGE_COLOR = "black"
PANEL_D2_SEGMENT_OVAL_LINEWIDTH = 0.45
PANEL_D2_SEGMENT_OVAL_ALPHAS = (0.46, 0.16)
PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_COLORS = {
    "left_arm": _figure_2.PANEL_B_VISUAL_ICON_COLORS["A"],
    "right_arm": _figure_2.PANEL_B_VISUAL_ICON_COLORS["B"],
}
PANEL_D2_DARK_SCAFFOLD_PREDICTION_ARM_SIDE_OUTLINE_COLORS = {
    "left_arm": PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_COLORS["right_arm"],
    "right_arm": PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_COLORS["left_arm"],
}
PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_GAP = 0.32
PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_LINEWIDTH = 1.25
PANEL_D2_SEGMENT_OUTLINE_COLORS = {
    **_figure_2.PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS,
    "right_arm": PANEL_D2_RIGHT_ARM_OUTLINE_COLOR,
}
PANEL_D2_SEGMENT_OUTLINE_LINEWIDTHS = {
    **_figure_2.PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS,
    "right_arm": _figure_2.PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS["left_arm"],
}
PANEL_D2_SEGMENT_MODULATION_LABEL = "Stimulus-specific\ngain modulation"
PANEL_D2_BASIS_BOTTOM_LINEWIDTH = 2.6
PANEL_D2_BASIS_LABEL_Y_OFFSET = (
    0.5 - 0.5 * (
        PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM + PANEL_G_INDEPENDENT_BASIS_ICON_TOP
    )
) * (
    PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT
    * _figure_2.PANEL_B_INDEPENDENT_BASIS_ICON_SCALE
)
PANEL_D2_EXAMPLE_SLOT_BOUNDS = (
    (0.080, 0.610, 0.150, 0.240),
    (0.310, 0.610, 0.150, 0.240),
    (0.080, 0.175, 0.150, 0.240),
)
PANEL_D2_TRACE_LEGEND_SLOT_BOUNDS = (0.285, 0.175, 0.175, 0.240)
PANEL_D2_HISTOGRAM_AXIS_BOUNDS = (0.600, 0.165, 0.380, 0.685)
PANEL_D2_EXAMPLE_ICON_BOUNDS = (-0.20, 0.27, 0.15, 0.30)
PANEL_D2_EXAMPLE_HEADER_X = 0.64
PANEL_D2_TRACE_LEGEND_ANCHOR = (0.485, 0.985)


def __getattr__(name: str) -> Any:
    """Delegate unchanged Figure 2 helpers and constants to the base module."""
    return getattr(_figure_2, name)


def _bounds_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> list[float]:
    """Return inset bounds from center coordinates."""
    return [center_x - width / 2.0, center_y - height / 2.0, width, height]


def _align_text_tops_to_reference_display_y(fig: Any, texts: Sequence[Any]) -> None:
    """Align text artists by their rendered top edge in display coordinates."""
    if len(texts) < 2:
        return
    for text in texts:
        axes = getattr(text, "axes", None)
        if axes is not None and text is axes.title:
            axes._autotitlepos = False
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reference_top = max(text.get_window_extent(renderer).y1 for text in texts)
    for text in texts:
        bbox = text.get_window_extent(renderer)
        display_position = text.get_transform().transform(text.get_position())
        adjusted_position = text.get_transform().inverted().transform(
            (display_position[0], display_position[1] + reference_top - bbox.y1)
        )
        text.set_position((text.get_position()[0], float(adjusted_position[1])))


def _align_text_center_to_reference_display_x(
    fig: Any,
    text: Any,
    reference_text: Any,
) -> None:
    """Align one text artist to another by rendered horizontal center."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bbox = text.get_window_extent(renderer)
    reference_bbox = reference_text.get_window_extent(renderer)
    text_center_x = text_bbox.x0 + text_bbox.width / 2.0
    reference_center_x = reference_bbox.x0 + reference_bbox.width / 2.0
    display_position = text.get_transform().transform(text.get_position())
    adjusted_position = text.get_transform().inverted().transform(
        (display_position[0] + reference_center_x - text_center_x, display_position[1])
    )
    text.set_position((float(adjusted_position[0]), text.get_position()[1]))


def _align_panel_b_top_histogram_label_to_scatter(
    fig: Any,
    panel_b_axis: Any,
) -> None:
    """Align Panel B top histogram y-label to the scatter y-label."""
    scatter_parent = next(
        (
            child_axis
            for child_axis in panel_b_axis.child_axes
            if len(child_axis.child_axes) == 3
        ),
        None,
    )
    if scatter_parent is None:
        return
    main_axis = next(
        (
            child_axis
            for child_axis in scatter_parent.child_axes
            if child_axis.get_xlabel() == "Dark DPPI"
        ),
        None,
    )
    top_histogram_axis = next(
        (
            child_axis
            for child_axis in scatter_parent.child_axes
            if child_axis.get_ylabel() == "Frac."
        ),
        None,
    )
    if main_axis is None or top_histogram_axis is None:
        return
    _align_text_center_to_reference_display_x(
        fig,
        top_histogram_axis.yaxis.label,
        main_axis.yaxis.label,
    )


def _append_panel_e_trial_errors(
    records: list[dict[str, Any]],
    *,
    timestamps: np.ndarray,
    absolute_error: np.ndarray,
    intervals: Any,
    animal_name: str,
    date: str,
    epoch_type: str,
    epoch: str,
    region: str,
    analysis: str,
    comparison: str,
    comparison_label: str,
    transfer_family: str,
    encoding_trajectory: str | None,
    decoding_trajectory: str,
    true_path: Path,
    decoded_path: Path,
) -> None:
    """Append one normalized median decoding error per finite lap."""
    starts, ends = _intervalset_to_arrays(intervals)
    for trial_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        in_trial = (timestamps >= start) & (timestamps < end)
        values = absolute_error[in_trial]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        records.append(
            {
                "animal_name": animal_name,
                "date": date,
                "epoch_type": epoch_type,
                "epoch": epoch,
                "region": region,
                "analysis": analysis,
                "comparison": comparison,
                "comparison_label": comparison_label,
                "transfer_family": transfer_family,
                "encoding_trajectory": encoding_trajectory,
                "decoding_trajectory": decoding_trajectory,
                "trial_index": int(trial_index),
                "trial_start": float(start),
                "trial_end": float(end),
                "trial_median_absolute_error": float(np.median(values)),
                "n_samples": int(values.size),
                "true_path": str(true_path),
                "decoded_path": str(decoded_path),
            }
        )


def build_panel_e_decoding_trial_error_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    comparisons: Sequence[
        tuple[str, str, str, Sequence[tuple[str, str]]]
    ] = PANEL_E_CROSS_COMPARISONS,
) -> Any:
    """Build lap-level normalized decoding errors for Figure 2E inference."""
    import pandas as pd

    records: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        epoch_specs = build_panel_quant_epoch_specs(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
        epochs = [epoch for _epoch_type, epoch in epoch_specs]
        if len(set(epochs)) != len(epochs):
            raise ValueError(
                "Figure 2E decoding inference requires distinct Light and Dark "
                f"epochs for {animal_name} {date}; received {epochs!r}."
            )
        analysis_path = get_analysis_path(animal_name, date, Path(data_root))
        trajectory_intervals, _source = load_trajectory_intervals(
            analysis_path,
            epochs,
        )
        place_normalization = float(get_wtrack_total_length(animal_name))
        if not np.isfinite(place_normalization) or place_normalization <= 0.0:
            raise ValueError(
                "W-track length must be positive and finite for Figure 2E "
                f"place decoding; got {place_normalization!r} for {animal_name}."
            )

        for epoch_type, epoch in epoch_specs:
            if epoch not in trajectory_intervals:
                raise ValueError(
                    f"Trajectory intervals do not contain epoch {epoch!r} for "
                    f"{animal_name} {date}."
                )
            epoch_intervals = trajectory_intervals[epoch]

            true_place_path, decoded_place_path = (
                get_within_epoch_decoding_tsd_paths(
                    data_root,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    epoch=epoch,
                    model_name=PANEL_E_PLACE_MODEL_NAME,
                )
            )
            true_place = _load_decoding_tsd(true_place_path)
            decoded_place = _load_decoding_tsd(decoded_place_path)
            place_timestamps, place_absolute_error = (
                _align_absolute_error_with_times(
                    true_place,
                    decoded_place,
                )
            )
            place_absolute_error = place_absolute_error / place_normalization
            for decoding_trajectory in TRAJECTORY_TYPES:
                if decoding_trajectory not in epoch_intervals:
                    raise ValueError(
                        "Trajectory intervals do not contain decoding trajectory "
                        f"{decoding_trajectory!r} for {animal_name} {date} {epoch}."
                    )
                _append_panel_e_trial_errors(
                    records,
                    timestamps=place_timestamps,
                    absolute_error=place_absolute_error,
                    intervals=epoch_intervals[decoding_trajectory],
                    animal_name=animal_name,
                    date=date,
                    epoch_type=epoch_type,
                    epoch=epoch,
                    region=region,
                    analysis="place",
                    comparison="place",
                    comparison_label="Place",
                    transfer_family="within_epoch",
                    encoding_trajectory=None,
                    decoding_trajectory=decoding_trajectory,
                    true_path=true_place_path,
                    decoded_path=decoded_place_path,
                )

            for (
                comparison,
                comparison_label,
                transfer_family,
                trajectory_pairs,
            ) in comparisons:
                for encoding_trajectory, decoding_trajectory in trajectory_pairs:
                    if decoding_trajectory not in epoch_intervals:
                        raise ValueError(
                            "Trajectory intervals do not contain decoding "
                            f"trajectory {decoding_trajectory!r} for "
                            f"{animal_name} {date} {epoch}."
                        )
                    true_cross_path, decoded_cross_path = (
                        get_cross_trajectory_decoding_tsd_paths(
                            data_root,
                            animal_name=animal_name,
                            date=date,
                            region=region,
                            epoch=epoch,
                            transfer_family=transfer_family,
                            encoding_trajectory=encoding_trajectory,
                            decoding_trajectory=decoding_trajectory,
                        )
                    )
                    true_cross = _load_decoding_tsd(true_cross_path)
                    decoded_cross = _load_decoding_tsd(decoded_cross_path)
                    cross_timestamps, cross_absolute_error = (
                        _align_absolute_error_with_times(
                            true_cross,
                            decoded_cross,
                        )
                    )
                    _append_panel_e_trial_errors(
                        records,
                        timestamps=cross_timestamps,
                        absolute_error=cross_absolute_error,
                        intervals=epoch_intervals[decoding_trajectory],
                        animal_name=animal_name,
                        date=date,
                        epoch_type=epoch_type,
                        epoch=epoch,
                        region=region,
                        analysis="cross_trajectory",
                        comparison=comparison,
                        comparison_label=comparison_label,
                        transfer_family=transfer_family,
                        encoding_trajectory=encoding_trajectory,
                        decoding_trajectory=decoding_trajectory,
                        true_path=true_cross_path,
                        decoded_path=decoded_cross_path,
                    )

    return pd.DataFrame.from_records(
        records,
        columns=PANEL_E_TRIAL_ERROR_TABLE_COLUMNS,
    )


def compute_panel_e_decoding_permutation_tests(
    trial_table: Any,
    *,
    n_permutations: int = DECODING_PERMUTATION_COUNT,
    seed: int = DECODING_PERMUTATION_SEED,
) -> Any:
    """Run Figure 1G's stratified median shuffle for each Figure 2E analysis."""
    import pandas as pd

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    required_columns = {
        "animal_name",
        "date",
        "epoch_type",
        "analysis",
        "decoding_trajectory",
        "trial_median_absolute_error",
    }
    missing_columns = required_columns.difference(trial_table.columns)
    if missing_columns:
        raise ValueError(
            "Figure 2E trial-error table is missing required columns: "
            f"{sorted(missing_columns)!r}."
        )

    rng = np.random.default_rng(seed)
    expected_trajectories = set(TRAJECTORY_TYPES)
    records = []
    for animal_name, animal_table in trial_table.groupby(
        "animal_name",
        sort=True,
    ):
        animal_dates = set(animal_table["date"].astype(str))
        if len(animal_dates) != 1:
            raise ValueError(
                "Figure 2E decoding inference requires exactly one session per "
                f"animal; {animal_name} has dates {sorted(animal_dates)!r}."
            )
        for analysis in PANEL_E_DECODING_ANALYSES:
            analysis_table = animal_table.loc[
                animal_table["analysis"].astype(str) == analysis
            ].copy()
            for epoch_type in ("light", "dark"):
                epoch_table = analysis_table.loc[
                    analysis_table["epoch_type"].astype(str) == epoch_type
                ]
                observed_trajectories = set(
                    epoch_table["decoding_trajectory"].astype(str)
                )
                if observed_trajectories != expected_trajectories:
                    raise ValueError(
                        "Incomplete Figure 2E decoding-trajectory coverage for "
                        f"{animal_name} {analysis!r} {epoch_type!r}: expected "
                        f"{sorted(expected_trajectories)!r}, observed "
                        f"{sorted(observed_trajectories)!r}."
                    )

            test_table = analysis_table.copy()
            test_table["comparison"] = test_table["epoch_type"].astype(str)
            result = stratified_median_permutation_test(
                test_table,
                "light",
                "dark",
                n_permutations=n_permutations,
                rng=rng,
            )
            records.append(
                {
                    "animal_name": str(animal_name),
                    "analysis": analysis,
                    **result,
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=PANEL_E_PERMUTATION_RESULT_COLUMNS,
    )


def build_panel_e_decoding_significance_labels(
    per_animal_results: Any,
    *,
    animal_names: Sequence[str],
) -> tuple[str, str]:
    """Return conservative data-derived labels for Figure 2E's two brackets."""
    required_columns = {
        "animal_name",
        "analysis",
        "median_difference",
        "p_two_sided",
    }
    missing_columns = required_columns.difference(per_animal_results.columns)
    if missing_columns:
        raise ValueError(
            "Figure 2E permutation-test table is missing required columns: "
            f"{sorted(missing_columns)!r}."
        )

    expected_animals = tuple(str(animal_name) for animal_name in animal_names)
    if not expected_animals or len(set(expected_animals)) != len(expected_animals):
        raise ValueError("animal_names must contain unique animal identifiers.")
    result_animals = per_animal_results["animal_name"].astype(str)
    observed_animals = set(result_animals)
    if observed_animals != set(expected_animals):
        raise ValueError(
            "Permutation-test animals do not match Figure 2E animals: "
            f"expected {sorted(expected_animals)!r}, "
            f"observed {sorted(observed_animals)!r}."
        )

    result_analyses = per_animal_results["analysis"].astype(str)
    labels = []
    for analysis in PANEL_E_DECODING_ANALYSES:
        analysis_results = per_animal_results.loc[
            (result_analyses == analysis)
            & result_animals.isin(expected_animals)
        ]
        analysis_animals = analysis_results["animal_name"].astype(str)
        counts = analysis_animals.value_counts()
        if (
            set(analysis_animals) != set(expected_animals)
            or len(analysis_results) != len(expected_animals)
            or not np.all(counts.to_numpy(dtype=int) == 1)
        ):
            raise ValueError(
                "Expected exactly one Figure 2E permutation result per animal "
                f"for analysis {analysis!r}."
            )
        p_values = np.asarray(
            analysis_results["p_two_sided"],
            dtype=float,
        )
        median_differences = np.asarray(
            analysis_results["median_difference"],
            dtype=float,
        )
        if not np.all(np.isfinite(p_values)) or np.any(
            (p_values < 0.0) | (p_values > 1.0)
        ):
            raise ValueError(
                "Figure 2E permutation results contain invalid two-sided "
                f"p-values for analysis {analysis!r}."
            )
        expected_sign = PANEL_E_EXPECTED_MEDIAN_DIFFERENCE_SIGNS[analysis]
        if not np.all(np.isfinite(median_differences)) or np.any(
            expected_sign * median_differences <= 0.0
        ):
            expected_direction = "higher" if expected_sign > 0.0 else "lower"
            raise ValueError(
                "Figure 2E expects Light to have "
                f"{expected_direction} median trial errors than Dark for every "
                f"animal in analysis {analysis!r}."
            )
        labels.append(significance_stars(float(np.max(p_values))))

    return labels[0], labels[1]


def _add_panel_c2_light_dark_bracket(
    ax: Any,
    label: str,
    *,
    y_fraction: float = PANEL_C2_SIGNIFICANCE_BRACKET_Y_FRACTION,
) -> None:
    """Draw one data-derived Figure 2E light-dark significance bracket."""
    x_start, x_stop = PANEL_C2_SIGNIFICANCE_BRACKET_X
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    y = y_min + float(y_fraction) * y_span
    y_top = y + DECODING_SIGNIFICANCE_BRACKET_HEIGHT
    if y_top > y_max:
        y_top = y_max
        y = y_top - DECODING_SIGNIFICANCE_BRACKET_HEIGHT
    ax.plot(
        [x_start, x_start, x_stop, x_stop],
        [y, y_top, y_top, y],
        color="black",
        linewidth=DECODING_SIGNIFICANCE_BRACKET_LINEWIDTH,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        (x_start + x_stop) / 2.0,
        y_top + DECODING_SIGNIFICANCE_LABEL_Y_OFFSET,
        str(label),
        ha="center",
        va="bottom",
        fontsize=DECODING_SIGNIFICANCE_LABEL_FONTSIZE,
        color="black",
        clip_on=False,
        zorder=7,
    )


def add_panel_c2_light_dark_brackets(
    panel_c_axis: Any,
    labels: Sequence[str],
) -> None:
    """Add data-derived light-dark brackets to the two Figure 2E axes."""
    if len(labels) != 2:
        raise ValueError("Figure 2E requires exactly two significance labels.")
    y_fractions = (
        PANEL_C2_SIGNIFICANCE_BRACKET_Y_FRACTION,
        PANEL_C2_RIGHT_SIGNIFICANCE_BRACKET_Y_FRACTION,
    )
    for child_axis, label, y_fraction in zip(
        panel_c_axis.child_axes[:2],
        labels,
        y_fractions,
        strict=True,
    ):
        _add_panel_c2_light_dark_bracket(
            child_axis,
            label,
            y_fraction=y_fraction,
        )


def format_panel_c2_decoding_axes(panel_c_axis: Any) -> None:
    """Apply Figure 2-specific cleanup to Panel C decoding axes."""
    child_axes = panel_c_axis.child_axes[:2]
    for child_axis in child_axes:
        for text in list(child_axis.texts):
            if text.get_text().startswith(("Light med.", "Dark med.")):
                text.remove()
    if child_axes:
        child_axes[0].yaxis.label.set_text(PANEL_C2_ERROR_AXIS_LABEL)


def plot_panel_a2_examples_single_row(
    ax: Any,
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot Figure 2 Panel A with W-track icons farther from the rasters."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        return

    column_bounds = _figure_2._equal_width_row_bounds(
        len(examples),
        _figure_2.PANEL_A_SINGLE_ROW_COLUMN_GAP,
    )
    for example_index, (example, (left, column_width)) in enumerate(
        zip(examples, column_bounds, strict=True),
        start=1,
    ):
        example_ax = ax.inset_axes([left, 0.0, column_width, 1.0])
        plot_kwargs: dict[str, Any] = {
            "title": None,
            "dark_epoch_axis_left": _figure_2.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT,
            "light_epoch_axis_left": _figure_2.PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT,
            "epoch_axis_width": _figure_2.PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH,
            "schematic_axis_left": PANEL_A2_SINGLE_ROW_SCHEMATIC_AXIS_LEFT,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        }
        y_max_override = _figure_2.PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(example_index)
        if y_max_override is not None:
            plot_kwargs["y_max"] = y_max_override
        _figure_2.plot_panel_a_example(example_ax, example, **plot_kwargs)
        rate_axes = [
            child_ax for child_ax in example_ax.child_axes if child_ax.get_xlabel()
        ]
        for rate_ax in rate_axes:
            rate_ax.set_xlabel("")
            rate_ax.tick_params(
                axis="x",
                labelsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
                pad=0.4,
            )
        example_ax.text(
            0.5,
            0.985,
            f"Example cell {example_index}",
            ha="center",
            va="top",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
            transform=example_ax.transAxes,
        )
        example_ax.text(
            0.5,
            _figure_2.PANEL_A_SINGLE_ROW_XLABEL_Y,
            "Norm. path progression",
            ha="center",
            va="top",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
            transform=example_ax.transAxes,
            clip_on=False,
        )


def _draw_panel_d2_track(
    ax: Any,
    *,
    center_x: float,
    center_y: float,
    track_kind: str,
    track_size: tuple[float, float],
    **kwargs: Any,
) -> Any:
    """Draw one Figure 2 Panel D W-track icon."""
    track_ax = ax.inset_axes(
        _bounds_from_center(center_x, center_y, track_size[0], track_size[1])
    )
    track_ax.set_zorder(0)
    track_ax.patch.set_visible(False)
    _draw_panel_h_track(track_ax, track_kind=track_kind, **kwargs)
    _remove_panel_d2_stimulus_labels(track_ax)
    return track_ax


def _remove_panel_d2_stimulus_labels(ax: Any) -> None:
    """Remove standalone A/B labels without changing W-track geometry."""
    for text in tuple(ax.texts):
        if text.get_text() in {"A", "B"}:
            text.remove()


def _set_panel_d2_place_field_alpha(ax: Any, alpha: float) -> None:
    """Set a uniform alpha on place-field ellipses in one Panel D2 icon."""
    from matplotlib.patches import Ellipse

    for patch in ax.patches:
        if type(patch) is Ellipse:
            patch.set_alpha(float(alpha))


def _apply_panel_d2_place_field_rate_gain(ax: Any, gain: float) -> None:
    """Recolor Panel D2 place-field ellipses as fixed-scale rate values."""
    from matplotlib.patches import Ellipse

    _outline, _points, dims = get_w_track_geometry()
    field_center_y = dims["y1"] + 1.45
    field_sigma = 0.58
    cmap = colormaps[PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP]
    for patch in ax.patches:
        if type(patch) is not Ellipse:
            continue
        relative_rate = math.exp(
            -0.5 * ((float(patch.center[1]) - field_center_y) / field_sigma) ** 2
        )
        color_value = min(max(float(gain) * relative_rate, 0.0), 1.0)
        color_value = color_value**PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA
        patch.set_facecolor(cmap(color_value))
        patch.set_edgecolor("none")
        patch.set_alpha(1.0)


def _draw_panel_d2_rate_colorbar(ax: Any) -> Any:
    """Draw the schematic relative-rate scale used for Panel D2 fields."""
    colorbar_ax = ax.inset_axes(PANEL_D2_RATE_COLORBAR_BOUNDS)
    colorbar_ax.set_label(PANEL_D2_RATE_COLORBAR_AXIS_LABEL)
    colorbar_ax.set_zorder(0)
    mappable = ScalarMappable(
        norm=PowerNorm(
            gamma=PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA,
            vmin=0.0,
            vmax=1.0,
        ),
        cmap=PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP,
    )
    colorbar = ax.figure.colorbar(
        mappable,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=(0.0, 1.0),
    )
    colorbar.set_ticklabels(PANEL_D2_RATE_COLORBAR_ENDPOINT_LABELS)
    colorbar.ax.tick_params(
        axis="x",
        which="both",
        length=0.0,
        pad=0.6,
        labelsize=PANEL_D2_RATE_COLORBAR_FONTSIZE,
    )
    colorbar.ax.set_title(
        PANEL_D2_RATE_COLORBAR_LABEL,
        fontsize=PANEL_D2_RATE_COLORBAR_FONTSIZE,
        pad=1.0,
    )
    colorbar.outline.set_linewidth(0.4)
    return colorbar_ax


def _draw_panel_d2_segment_ovals(ax: Any) -> None:
    """Draw bilateral gain ovals in the dark-scaffold modulation icon."""
    _outline, _points, dims = get_w_track_geometry()
    draw_large_ovals(
        ax,
        dims,
        oval_regions=list(PANEL_D2_SEGMENT_OVAL_REGIONS),
        oval_styles=[
            {
                "edge_color": PANEL_D2_SEGMENT_OVAL_EDGE_COLOR,
                "fill_color": PANEL_D2_SEGMENT_OVAL_FILL_COLOR,
                "fill_alpha": alpha,
                "linewidth": PANEL_D2_SEGMENT_OVAL_LINEWIDTH,
            }
            for alpha in PANEL_D2_SEGMENT_OVAL_ALPHAS
        ],
    )


def _draw_panel_d2_segment_arrows(ax: Any) -> None:
    """Draw full gain-direction arrows in the segment modulation icon."""
    _outline, _points, dims = get_w_track_geometry()
    y_bottom = dims["y1"] + PANEL_D2_SEGMENT_ARROW_Y_MARGIN
    y_top = dims["y2"] - PANEL_D2_SEGMENT_ARROW_Y_MARGIN
    for center_x, y_start, y_end, label_suffix in (
        ((dims["x0"] + dims["x1"]) / 2.0, y_bottom, y_top, "up"),
        ((dims["x4"] + dims["x5"]) / 2.0, y_top, y_bottom, "down"),
    ):
        ax.annotate(
            "",
            xy=(center_x, y_end),
            xytext=(center_x, y_start),
            xycoords="data",
            textcoords="data",
            arrowprops={
                "arrowstyle": "-|>",
                "color": PANEL_D2_SEGMENT_ARROW_COLOR,
                "lw": PANEL_D2_SEGMENT_ARROW_LINEWIDTH,
                "mutation_scale": PANEL_D2_SEGMENT_ARROW_MUTATION_SCALE,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": "arc3,rad=0",
            },
            annotation_clip=False,
            zorder=6,
        ).arrow_patch.set_label(f"_panel_d2_segment_arrow_{label_suffix}")


def _draw_panel_d2_segment_arm_side_outlines(
    ax: Any,
    *,
    arm_colors: Mapping[str, str] = PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_COLORS,
) -> None:
    """Draw stim-colored side outlines for the arm-specific modulation icon."""
    draw_w_track_arm_side_outlines(
        ax,
        arm_colors=arm_colors,
        gap=PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_GAP,
        linewidth=PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_LINEWIDTH,
        label_prefix="_panel_d2_segment_arm_side_outline",
    )


def _draw_panel_d2_basis_icon(
    ax: Any,
    *,
    center_x: float,
    center_y: float,
    scale: float,
) -> None:
    """Draw the independent-basis icon in the Figure 2 Panel D schematic."""
    width = PANEL_G_INDEPENDENT_BASIS_ICON_WIDTH * float(scale)
    height = PANEL_G_INDEPENDENT_BASIS_ICON_HEIGHT * float(scale)
    visual_center_y = 0.5 * (
        PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM + PANEL_G_INDEPENDENT_BASIS_ICON_TOP
    )
    basis_ax = ax.inset_axes(
        [
            center_x - width / 2.0,
            center_y - height * visual_center_y,
            width,
            height,
        ]
    )
    basis_ax.set_zorder(0)
    basis_ax.patch.set_visible(False)
    _draw_panel_g_basis_icon(basis_ax)
    vertical_span = (
        PANEL_G_INDEPENDENT_BASIS_ICON_TOP - PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM
    )
    horizontal_left = 0.5 - vertical_span / 2.0
    horizontal_right = 0.5 + vertical_span / 2.0
    basis_ax.plot(
        [horizontal_left, horizontal_right],
        [PANEL_G_INDEPENDENT_BASIS_ICON_BOTTOM] * 2,
        color="black",
        linewidth=PANEL_D2_BASIS_BOTTOM_LINEWIDTH,
        solid_capstyle="butt",
        zorder=5,
    )


def _draw_panel_d2_horizontal_arrow(
    ax: Any,
    *,
    start_x: float,
    end_x: float,
    y: float,
) -> None:
    """Draw a short horizontal arrow in Panel D schematic coordinates."""
    ax.annotate(
        "",
        xy=(end_x, y),
        xytext=(start_x, y),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 0.8,
            "mutation_scale": 7.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def draw_panel_d2_architecture_schematic(
    ax: Any,
    *,
    show_dark_track_labels: bool = True,
    track_size: tuple[float, float] = _figure_2.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE,
) -> None:
    """Draw Panel D architecture with adjacent BA prediction icons."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.patch.set_visible(False)
    ax.set_zorder(1)

    schematic_shift = _figure_2.PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
    independent_y = (
        _figure_2.PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y + schematic_shift
        + PANEL_D2_INDEPENDENT_ROW_Y_OFFSET
    )
    shared_y = (
        _figure_2.PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
        + schematic_shift
        + PANEL_D2_SHARED_ROW_Y_OFFSET
    )
    field_label_y = _figure_2.PANEL_B_FIELD_LABEL_Y + schematic_shift
    basis_label_y = (
        _figure_2.PANEL_G_INDEPENDENT_BASIS_LABEL_Y
        + schematic_shift
        + PANEL_D2_BASIS_LABEL_Y_OFFSET
    )
    segment_label_y = shared_y + track_size[1] / 2.0 + PANEL_D2_SEGMENT_LABEL_GAP
    cue_swap_arrow_start_x = (
        PANEL_D2_LIGHT_TRACK_CENTER_X
        + track_size[0] / 2.0
        + PANEL_D2_CUE_SWAP_ARROW_MARGIN
    )
    cue_swap_arrow_end_x = (
        PANEL_D2_PREDICT_TRACK_CENTER_X
        - track_size[0] / 2.0
        - PANEL_D2_CUE_SWAP_ARROW_MARGIN
    )
    text_kwargs = {"transform": ax.transAxes}

    ax.text(
        PANEL_D2_DARK_TRACK_CENTER_X,
        field_label_y,
        "Dark field",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_LIGHT_TRACK_CENTER_X,
        field_label_y,
        "Light field",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_PREDICT_TRACK_CENTER_X,
        field_label_y + PANEL_D2_CUE_SWAP_LABEL_Y_OFFSET,
        "Cue-swap\nprediction",
        ha="center",
        va="top",
        fontsize=5.8,
        **text_kwargs,
    )
    ax.text(
        _figure_2.PANEL_D_MODEL_LABEL_X,
        independent_y,
        "Independent\nmodel",
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_MODEL_LABEL_FONTSIZE,
        fontweight="bold",
        **text_kwargs,
    )
    ax.text(
        _figure_2.PANEL_D_MODEL_LABEL_X,
        shared_y,
        "Dark scaffold\nmodel",
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_MODEL_LABEL_FONTSIZE,
        fontweight="bold",
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_BASIS_ICON_CENTER_X,
        basis_label_y,
        _figure_2.PANEL_B_INDEPENDENT_BASIS_LABEL,
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_COMPONENT_LABEL_FONTSIZE,
        **text_kwargs,
    )
    ax.text(
        PANEL_D2_SEGMENT_TRACK_CENTER_X,
        segment_label_y,
        PANEL_D2_SEGMENT_MODULATION_LABEL,
        ha="center",
        va="center",
        fontsize=_figure_2.PANEL_B_COMPONENT_LABEL_FONTSIZE,
        **text_kwargs,
    )

    independent_dark_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_DARK_TRACK_CENTER_X,
        center_y=independent_y,
        track_size=track_size,
        track_kind="dark",
        show_labels=show_dark_track_labels,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _apply_panel_d2_place_field_rate_gain(
        independent_dark_ax,
        PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN,
    )
    _set_panel_d2_place_field_alpha(
        independent_dark_ax,
        PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_ALPHA,
    )
    _draw_panel_d2_basis_icon(
        ax,
        center_x=PANEL_D2_BASIS_ICON_CENTER_X,
        center_y=independent_y,
        scale=_figure_2.PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
    )
    independent_light_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_LIGHT_TRACK_CENTER_X,
        center_y=independent_y,
        track_size=track_size,
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        highlighted_segments=(3,),
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _apply_panel_d2_place_field_rate_gain(
        independent_light_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN,
    )
    _set_panel_d2_place_field_alpha(
        independent_light_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA,
    )
    _draw_panel_d2_segment_arm_side_outlines(independent_light_ax)
    independent_predict_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_PREDICT_TRACK_CENTER_X,
        center_y=independent_y,
        track_size=track_size,
        track_kind="independent_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        highlighted_segments=(3,),
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="right_arm",
    )
    _apply_panel_d2_place_field_rate_gain(
        independent_predict_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN,
    )
    _set_panel_d2_place_field_alpha(
        independent_predict_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA,
    )
    _draw_panel_d2_segment_arm_side_outlines(
        independent_predict_ax,
        arm_colors=PANEL_D2_DARK_SCAFFOLD_PREDICTION_ARM_SIDE_OUTLINE_COLORS,
    )
    _draw_panel_d2_horizontal_arrow(
        ax,
        start_x=cue_swap_arrow_start_x,
        end_x=cue_swap_arrow_end_x,
        y=independent_y,
    )

    shared_dark_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_DARK_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="dark",
        show_labels=show_dark_track_labels,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="left_arm",
    )
    _apply_panel_d2_place_field_rate_gain(
        shared_dark_ax,
        PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN,
    )
    _set_panel_d2_place_field_alpha(
        shared_dark_ax,
        PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_ALPHA,
    )
    ax.text(
        PANEL_D2_SHARED_PLUS_X,
        shared_y,
        "+",
        ha="center",
        va="center",
        fontsize=8.0,
        **text_kwargs,
    )
    segment_oval_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_SEGMENT_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="segment_modulation",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        segment_outline_colors={},
        segment_outline_linewidths={},
    )
    _draw_panel_d2_segment_ovals(segment_oval_ax)
    _draw_panel_d2_segment_arrows(segment_oval_ax)
    _draw_panel_d2_segment_arm_side_outlines(segment_oval_ax)
    ax.text(
        PANEL_D2_EQUALS_X,
        shared_y,
        "=",
        ha="center",
        va="center",
        fontsize=8.0,
        **text_kwargs,
    )
    shared_light_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_LIGHT_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
    )
    _apply_panel_d2_place_field_rate_gain(
        shared_light_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN,
    )
    _set_panel_d2_place_field_alpha(
        shared_light_ax,
        PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA,
    )
    _draw_panel_d2_segment_arm_side_outlines(shared_light_ax)
    shared_predict_ax = _draw_panel_d2_track(
        ax,
        center_x=PANEL_D2_PREDICT_TRACK_CENTER_X,
        center_y=shared_y,
        track_size=track_size,
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_right",
        stimulus_layout="stim2",
        label_fontsize=_figure_2.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2.PANEL_B_VISUAL_ICON_COLORS,
        show_place_field_blob=True,
        place_field_colors=PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_COLORS,
        place_field_blob_size_scale=_figure_2.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE,
        place_field_arm="left_arm",
    )
    _apply_panel_d2_place_field_rate_gain(
        shared_predict_ax,
        PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_RATE_GAIN,
    )
    _set_panel_d2_place_field_alpha(
        shared_predict_ax,
        PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_ALPHA,
    )
    _draw_panel_d2_segment_arm_side_outlines(
        shared_predict_ax,
        arm_colors=PANEL_D2_DARK_SCAFFOLD_PREDICTION_ARM_SIDE_OUTLINE_COLORS,
    )
    _draw_panel_d2_horizontal_arrow(
        ax,
        start_x=cue_swap_arrow_start_x,
        end_x=cue_swap_arrow_end_x,
        y=shared_y,
    )
    _draw_panel_d2_rate_colorbar(ax)


def _plot_panel_d2_swap_results(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    show_example_xlabel: bool = True,
) -> None:
    """Plot the three swap examples and mean-delta histogram."""
    examples = list(swap_examples.values()) if isinstance(swap_examples, dict) else list(
        swap_examples or []
    )
    for example_index, bounds in enumerate(PANEL_D2_EXAMPLE_SLOT_BOUNDS):
        example_ax = ax.inset_axes(bounds)
        example = examples[example_index] if example_index < len(examples) else None
        _figure_2._plot_panel_h_switched_segment_example(
            example_ax,
            example,
            model_name=model_name,
            model_colors=model_colors,
            model_labels=model_labels,
            example_label=f"Example {example_index + 1}",
            show_xlabel=show_example_xlabel and example_index == 2,
            show_ylabel=example_index != 1,
            show_legend=False,
            show_xticklabels=True,
            icon_bounds=PANEL_D2_EXAMPLE_ICON_BOUNDS,
            legend_loc="center left",
            legend_bbox_to_anchor=None,
        )
        example_ax.tick_params(labelsize=4.3)
        for text in tuple(example_ax.texts):
            if text.get_text().startswith("ΔLL="):
                example_ax.title.set_text(
                    f"Ex. {example_index + 1} ({text.get_text()})"
                )
                example_ax.title.set_x(PANEL_D2_EXAMPLE_HEADER_X)
                text.remove()
        _figure_2._set_nested_legend_fontsize(example_ax, 3.9)
        _figure_2._replace_nested_text(
            example_ax,
            "Norm. path progression",
            "Norm.\npath progression",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
        )

    legend_ax = ax.inset_axes(PANEL_D2_TRACE_LEGEND_SLOT_BOUNDS)
    legend_ax.axis("off")
    _add_panel_d2_trace_legend(
        legend_ax,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1,
        fontsize=4.4,
    )

    histogram_ax = ax.inset_axes(PANEL_D2_HISTOGRAM_AXIS_BOUNDS)
    _figure_2.plot_panel_d_mean_swap_delta_axis(
        histogram_ax,
        swap_delta_table,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def _add_panel_d2_trace_legend(
    ax: Any,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] | None = PANEL_D2_TRACE_LEGEND_ANCHOR,
    ncol: int = 3,
    fontsize: float = 3.6,
) -> Any:
    """Add a compact legend for the Panel D empirical and model traces."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0.0],
            [0.0],
            color=GLM_EMPIRICAL_COLOR,
            linewidth=0.9,
            label="Empirical",
        ),
        Line2D(
            [0.0],
            [0.0],
            color=_figure_2._panel_model_color("visual", model_colors),
            linewidth=0.8,
            label="Independent",
        ),
        Line2D(
            [0.0],
            [0.0],
            color=_figure_2._panel_model_color(model_name, model_colors),
            linewidth=0.8,
            label=_figure_2._panel_model_label(model_name, model_labels),
        ),
    ]
    return ax.legend(
        handles=handles,
        frameon=False,
        fontsize=fontsize,
        handlelength=0.75,
        handletextpad=0.25,
        columnspacing=0.55,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0.0,
        ncol=ncol,
    )


def plot_panel_d2_architecture_with_swap_results(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot Figure 2 Panel D with the BA icons and swap results."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes(PANEL_D2_SCHEMATIC_AXIS_BOUNDS)
    result_ax = ax.inset_axes(PANEL_D2_RESULT_AXIS_BOUNDS)
    result_ax.set_xlim(0.0, 1.0)
    result_ax.set_ylim(0.0, 1.0)
    result_ax.axis("off")

    draw_panel_d2_architecture_schematic(schematic_ax)
    _plot_panel_d2_swap_results(
        result_ax,
        swap_delta_table,
        swap_examples,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def plot_panel_d2_swap_results_panel(
    ax: Any,
    swap_delta_table: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot the Figure 2 cue-swap prediction results as a standalone panel."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    _plot_panel_d2_swap_results(
        ax,
        swap_delta_table,
        swap_examples,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
        show_example_xlabel=True,
    )


def plot_panel_d2_architecture_panel(
    ax: Any,
) -> None:
    """Plot the Figure 2 model schematic as a standalone panel."""
    draw_panel_d2_architecture_schematic(ax)


def plot_panel_e2_decoding_panel(
    ax: Any,
    decoding_error_table: Any,
    *,
    significance_labels: Sequence[str] = (),
) -> None:
    """Plot the Figure 2 dark-light decoding comparison as a standalone panel."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    _figure_2.plot_panel_c_cross_and_place_decoding(
        ax,
        decoding_error_table,
    )
    format_panel_c2_decoding_axes(ax)
    if significance_labels:
        add_panel_c2_light_dark_brackets(ax, significance_labels)


def make_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = _figure_2.DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = _figure_2.DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = _figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = _figure_2.DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    dark_tuning_correlation_threshold: float = (
        _figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
    ),
    high_dark_tuning_correlation_threshold: float = (
        _figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
    ),
    decoding_n_permutations: int = DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = DECODING_PERMUTATION_SEED,
) -> Path:
    """Build and save Figure 2."""
    import matplotlib.pyplot as plt

    if decoding_n_permutations <= 0:
        raise ValueError("decoding_n_permutations must be positive.")
    if decoding_permutation_seed < 0:
        raise ValueError("decoding_permutation_seed must be non-negative.")
    normalized_datasets = [
        normalize_dataset_id(dataset)
        for dataset in datasets
    ]
    decoding_animal_names = tuple(
        animal_name
        for animal_name, _date, _epoch in normalized_datasets
    )
    if (
        not decoding_animal_names
        or len(set(decoding_animal_names)) != len(decoding_animal_names)
    ):
        raise ValueError(
            "Figure 2E decoding inference requires exactly one data set per "
            f"animal; received {normalized_datasets!r}."
        )

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else _figure_2.DEFAULT_REGIONS[0]
    panel_glm_payload = _figure_2.load_panel_glm_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        swap_delta_min_movement_firing_rate_hz=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        swap_delta_min_tuning_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        swap_model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
        swap_example_count=len(_figure_2.PANEL_C_SWAP_EXAMPLES),
        swap_requested_examples=_figure_2.PANEL_C_SWAP_EXAMPLES,
        dark_light_requested_examples=_figure_2.PANEL_C_DARK_LIGHT_EXAMPLES,
    )
    panel_a_examples = [
        _figure_2.load_panel_a_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            unit_id=unit_id,
            trajectories=trajectories,
            dark_epoch=dark_epoch,
            light_epoch=light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
        for animal_name, date, region, unit_id, trajectories in (
            _figure_2.FIGURE_2_PANEL_A_EXAMPLES
        )
    ]
    panel_b_overlap_table = _figure_2.load_panel_b_tuning_overlap_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_b_overlap_table = _figure_2.filter_panel_b_overlap_by_even_odd_stability(
        panel_b_overlap_table,
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_movement_firing_rate_hz=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        min_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    )
    panel_e_decoding_error_table = _figure_2.load_panel_e_decoding_error_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_e_decoding_trial_error_table = (
        build_panel_e_decoding_trial_error_table(
            data_root=data_root,
            datasets=datasets,
            region=quant_region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
    )
    panel_e_permutation_results = compute_panel_e_decoding_permutation_tests(
        panel_e_decoding_trial_error_table,
        n_permutations=decoding_n_permutations,
        seed=decoding_permutation_seed,
    )
    panel_e_significance_labels = build_panel_e_decoding_significance_labels(
        panel_e_permutation_results,
        animal_names=decoding_animal_names,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        **_figure_2.CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS
    )
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[
            _figure_2.PANEL_A_SINGLE_ROW_HEIGHT_MM,
            _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM,
            _figure_2.PANEL_D_ROW_HEIGHT_MM,
        ],
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    quant_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_B2C2_ROW_WIDTH_RATIOS,
        wspace=PANEL_B2C2_ROW_WSPACE,
    )
    panel_b_axis = fig.add_subplot(quant_grid[0, 0])
    panel_c_axis = fig.add_subplot(quant_grid[0, 1])
    bottom_grid = outer_grid[2, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_D2E2_ROW_WIDTH_RATIOS,
        wspace=PANEL_D2E2_ROW_WSPACE,
    )
    panel_d_axis = fig.add_subplot(bottom_grid[0, 0])
    panel_e_axis = fig.add_subplot(bottom_grid[0, 1])

    plot_panel_a2_examples_single_row(panel_a_axis, panel_a_examples)
    _figure_2.plot_panel_b_dpp_overlap_with_schematic(
        panel_b_axis,
        panel_b_overlap_table,
        example=panel_a_examples[0],
        low_threshold=dark_tuning_correlation_threshold,
        high_threshold=high_dark_tuning_correlation_threshold,
        show_grouped=False,
        show_scatter_linear_fit=True,
        show_scatter_r2=True,
        scatter_equal_aspect=True,
    )
    _figure_2._replace_nested_text(
        panel_b_axis,
        "DPP index",
        "DPP index (DPPI)",
        fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    plot_panel_d2_architecture_panel(panel_c_axis)
    plot_panel_d2_swap_results_panel(
        panel_d_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
        model_colors=_figure_2.PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=_figure_2.PANEL_C_SWAP_MODEL_LABELS_2_3,
    )
    plot_panel_e2_decoding_panel(
        panel_e_axis,
        panel_e_decoding_error_table,
        significance_labels=panel_e_significance_labels,
    )

    label_axis(panel_a_axis, "A", x=-0.02, y=_figure_2.PANEL_A_LABEL_Y)
    panel_a_label = panel_a_axis.texts[-1]
    panel_a_axis.set_title(
        "Example DPP cells in dark and light",
        fontsize=8,
        pad=_figure_2.PANEL_A_TITLE_PAD,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=_figure_2.PANEL_B_LABEL_Y, va="baseline")
    panel_b_label = panel_b_axis.texts[-1]
    panel_b_title = panel_b_axis.set_title(
        "Dark and light DPP coding",
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    label_axis(panel_c_axis, "C", x=-0.035, y=_figure_2.PANEL_B_LABEL_Y, va="baseline")
    panel_c_label = panel_c_axis.texts[-1]
    label_axis(panel_d_axis, "D", x=-0.02, y=_figure_2.PANEL_BC_LABEL_Y)
    panel_d_label = panel_d_axis.texts[-1]
    label_axis(panel_e_axis, "E", x=-0.035, y=_figure_2.PANEL_BC_LABEL_Y)
    panel_e_label = panel_e_axis.texts[-1]
    panel_c_title = panel_c_axis.set_title(
        "Two models that relate dark and light activity",
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    panel_d_title = panel_d_axis.set_title(
        "Dark and light cue-swap prediction comparison",
        fontsize=8,
        pad=_figure_2.PANEL_BC_TITLE_PAD,
    )
    panel_e_title = panel_e_axis.set_title(
        "Dark and light decoding comparison",
        fontsize=8,
        pad=_figure_2.PANEL_BC_TITLE_PAD,
    )

    _figure_2._raise_text_to_minimum_fontsize(
        fig,
        _figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _figure_2._set_axis_horizontal_bounds(
        panel_a_axis,
        left=_figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[0],
        width=_figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[1],
    )
    panel_a_axis_height = panel_a_axis.get_position().height
    _figure_2._set_axis_height_preserving_top(panel_b_axis, panel_a_axis_height)
    _figure_2._set_axis_height_preserving_top(panel_c_axis, panel_a_axis_height)
    _figure_2._scale_axis_width_from_left(
        panel_b_axis,
        _figure_2.PANEL_B_HORIZONTAL_WIDTH_SCALE,
    )
    fig.canvas.draw()
    _figure_2._align_text_to_reference_display_x(panel_b_label, panel_a_label)
    _figure_2._align_text_to_reference_display_x(panel_d_label, panel_a_label)
    _figure_2._align_texts_to_reference_display_y(
        (panel_d_title, panel_d_label, panel_e_title, panel_e_label)
    )
    _align_text_tops_to_reference_display_y(
        fig,
        (panel_d_title, panel_d_label, panel_e_title, panel_e_label),
    )
    _figure_2._align_texts_to_reference_display_y(
        (panel_b_title, panel_b_label, panel_c_title, panel_c_label)
    )
    _align_panel_b_top_histogram_label_to_scatter(fig, panel_b_axis)

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 2 generation."""
    parser = argparse.ArgumentParser(description="Generate Figure 2.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_figure_2.DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure output. Default: {_figure_2.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--panel-example-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached example-cell rasters and rate curves. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help="Recompute example-cell data and overwrite matching caches.",
    )
    parser.add_argument(
        "--dark-tuning-correlation-threshold",
        "--dpp-index-threshold",
        dest="dark_tuning_correlation_threshold",
        type=float,
        default=_figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Dark tuning-correlation threshold for Panel B low/high grouping. "
            f"Default: {_figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--high-dark-tuning-correlation-threshold",
        type=float,
        default=_figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Upper dark tuning-correlation threshold for Panel B high group. "
            f"Default: {_figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=_figure_2.FIGURE_FORMATS,
        default=_figure_2.DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {_figure_2.DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_figure_2.parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date. May be repeated. "
            "Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help=(
            "Region to include. May be repeated. "
            f"Default: {', '.join(_figure_2.DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=_figure_2.DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {_figure_2.DEFAULT_POSITION_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=_figure_2.DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples to ignore. "
            f"Default: {_figure_2.DEFAULT_POSITION_OFFSET}"
        ),
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=_figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {_figure_2.DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=_figure_2.DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in bins. Default: {_figure_2.DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--decoding-n-permutations",
        type=int,
        default=DECODING_PERMUTATION_COUNT,
        help=(
            "Label permutations used for Figure 2E decoding inference. "
            f"Default: {DECODING_PERMUTATION_COUNT}"
        ),
    )
    parser.add_argument(
        "--decoding-permutation-seed",
        type=int,
        default=DECODING_PERMUTATION_SEED,
        help=(
            "Random seed used for Figure 2E decoding inference. "
            f"Default: {DECODING_PERMUTATION_SEED}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else _figure_2.DEFAULT_REGIONS
    output_path = _figure_2.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
        dark_tuning_correlation_threshold=args.dark_tuning_correlation_threshold,
        high_dark_tuning_correlation_threshold=(
            args.high_dark_tuning_correlation_threshold
        ),
        decoding_n_permutations=args.decoding_n_permutations,
        decoding_permutation_seed=args.decoding_permutation_seed,
    )


if __name__ == "__main__":
    main()
