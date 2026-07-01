from __future__ import annotations

"""Generate Figure 2 panels moved from the dark-light Figure 3 layout."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import get_stability_table_path
from v1ca1.paper_figures.figure_3 import (
    DARK_MOVEMENT_FR_CACHE_COLUMNS,
    DARK_MOVEMENT_FR_CACHE_VERSION,
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_4_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_4_WIDTH_MM,
    load_dark_movement_firing_rate_cache,
    save_dark_movement_firing_rate_cache,
)
from v1ca1.paper_figures.old_fig3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    FIGURE_FORMATS,
    PANEL_A_EXAMPLES,
    PANEL_GH_WIDTH_RATIOS,
    PANEL_H_INDEPENDENT_TRACK_CENTER_Y,
    PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y,
    PANEL_H_SHARED_DARK_TRACK_CENTER_Y,
    PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y,
    build_panel_c_similarity_pairs,
    build_output_path,
    get_compute_tuning_curve_path,
    get_dark_epoch,
    get_light_epoch,
    get_tuning_similarity_path,
    load_panel_a_example_data,
    load_panel_c_similarity_table,
    load_panel_glm_data,
    parse_dataset_id,
    plot_panel_a_example,
    plot_panel_g_model_architecture,
    plot_panel_h_swap_delta,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "figure_2"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_4_WIDTH_MM
PANEL_A_TO_GH_HEIGHT_RATIOS = (0.637, 1.3)
PANEL_A_EXAMPLE_ROW_HEIGHT_MM = 50.4
PANEL_BC_ROW_HEIGHT_MM = (
    FIGURE_4_HEIGHT_MM
    * 1.3
    * PANEL_A_TO_GH_HEIGHT_RATIOS[1]
    / sum(PANEL_A_TO_GH_HEIGHT_RATIOS)
)
DEFAULT_FIGURE_HEIGHT_MM = PANEL_A_EXAMPLE_ROW_HEIGHT_MM + PANEL_BC_ROW_HEIGHT_MM
FIGURE_2_CONSTRAINED_LAYOUT_PADS = {
    "h_pad": 0.01,
    "w_pad": 0.01,
    "hspace": 0.01,
    "wspace": 0.02,
}
PANEL_AB_WIDTH_RATIOS = (0.64, 0.36)
PANEL_AB_WSPACE = 0.10
PANEL_A_EXAMPLE_COLUMN_GAP = 0.035
PANEL_A_EXAMPLE_ROW_GAP = 0.055
FIGURE_2_PANEL_A_EXAMPLES = (
    *PANEL_A_EXAMPLES,
    ("L12", "20240421", "v1", 37, ("center_to_right", "left_to_center")),
    ("L14", "20240611", "v1", 30, ("center_to_left", "right_to_center")),
)
PANEL_A_EXAMPLE_Y_MAX_OVERRIDES = {
    4: 85.0,
}
PANEL_A_EXAMPLE_CORRELATION_TEXT_OVERRIDES = {
    3: {
        "correlation_text_position": (0.04, 0.92),
        "correlation_text_ha": "left",
    },
}
PANEL_A_LABEL_Y = 1.03
PANEL_A_TITLE_PAD = 0.5
PANEL_B_LABEL_Y = 1.03
PANEL_B_TITLE_PAD = 0.5
PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD = 0.5
PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD = 0.75
PANEL_B_DPP_OVERLAP_METRIC = "absolute_overlap"
PANEL_B_DARK_ACTIVITY_THRESHOLD_HZ = 0.5
PANEL_B_TUNING_CORRELATION_TRAJECTORIES = (
    "center_to_left",
    "right_to_center",
    "center_to_right",
    "left_to_center",
)
PANEL_B_DPP_COMPARISON_LABELS = ("left_turn", "right_turn")
PANEL_B_SINGLE_LIGHT_DPP_AXIS_BOUNDS = (0.12, 0.16, 0.76, 0.70)
PANEL_B_ACTIVITY_DPP_AXIS_BOUNDS = (0.02, 0.16, 0.40, 0.70)
PANEL_B_LIGHT_DPP_AXIS_BOUNDS = (0.54, 0.16, 0.44, 0.70)
PANEL_B_BOX_COLORS = {
    "low_dpp": "#72B7B2",
    "mid_dpp": "#9E9E9E",
    "high_dpp": "#E45756",
}
PANEL_B_ACTIVITY_COLORS = {
    "inactive": "#E69F00",
    "active": "#4D4D4D",
}
PANEL_BC_LABEL_Y = 1.03
PANEL_BC_TITLE_PAD = 0.5
PANEL_B_SCHEMATIC_HEIGHT_FRACTION = 0.72
PANEL_B_SCHEMATIC_TRACK_SIZE = (0.2025, 0.2547)
PANEL_B_INDEPENDENT_BASIS_ICON_SCALE = 0.70
PANEL_B_INDEPENDENT_BASIS_LABEL = "Independent"
PANEL_B_EXAMPLE_AXIS_BOUNDS = (0.0, 0.01, 1.0, 0.44)
PANEL_B_EXAMPLE_FIELD_Y = 0.13
PANEL_B_EXAMPLE_FIELD_HEIGHT = 0.62
PANEL_B_EXAMPLE_ICON_BOUNDS = (0.04, 0.27, 0.09, 0.34)
PANEL_B_EXAMPLE_XLABEL_Y = 0.02
PANEL_B_EXAMPLE_COLUMN_WIDTH = 0.50
PANEL_B_EXAMPLE_COLUMN_GAP = 0.0
PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET = 0.20
PANEL_B_EXAMPLE_FIELD_WIDTH = 0.28
PANEL_B_EXAMPLE_FIELD_GAP = 0.075
PANEL_B_EXAMPLE_LAYOUT = "rows"
PANEL_B_EXAMPLE_ROW_HEIGHT = 0.46
PANEL_B_EXAMPLE_ROW_GAP = 0.05
PANEL_B_MODEL_LABEL_X = 0.03
PANEL_B_MODEL_LABEL_FONTSIZE = 5.8
PANEL_B_COMPONENT_LABEL_FONTSIZE = 5.8
PANEL_B_SEGMENT_MODULATION_LABEL = "Segment-specific\nmodulation"
PANEL_B_SEGMENT_MODULATION_LABEL_Y = 0.545
PANEL_INDEPENDENT_MODEL_COLOR = "#0072B2"
PANEL_SHARED_SCAFFOLD_MODEL_COLOR = "#CC79A7"
PANEL_B_EXAMPLE_SHARED_MODEL_NAME = "task_segment_scalar"
PANEL_B_EXAMPLE_MODEL_COLORS = {
    "visual": PANEL_INDEPENDENT_MODEL_COLOR,
    PANEL_B_EXAMPLE_SHARED_MODEL_NAME: PANEL_SHARED_SCAFFOLD_MODEL_COLOR,
}
PANEL_B_EXAMPLE_MODEL_LABELS = {
    "visual": "Independent",
    PANEL_B_EXAMPLE_SHARED_MODEL_NAME: "Segment scalar",
}
PANEL_C_DARK_LIGHT_EXAMPLES = (
    ("L14", "20240611", "v1", 30, "center_to_left"),
    ("L12", "20240421", "v1", 37, "left_to_center"),
)
PANEL_B_ALIGNMENT_SCHEMATIC_AXIS_BOUNDS = (-0.06, 0.39, 0.40, 0.58)
PANEL_C_SCHEMATIC_AXIS_BOUNDS = (-0.08, 0.25, 0.40, 0.72)
PANEL_C_DELTA_AXIS_BOUNDS = (0.39, 0.35, 0.60, 0.59)
PANEL_B_MIN_TUNING_STABILITY_CORRELATION = 0.5
PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION = (
    PANEL_B_MIN_TUNING_STABILITY_CORRELATION
)
PANEL_C_DELTA_GRID_BOUNDS = (
    (0.035, 0.42, 0.445, 0.50),
    (0.535, 0.42, 0.445, 0.50),
    (0.035, -0.22, 0.445, 0.50),
    (0.535, -0.22, 0.445, 0.50),
)
PANEL_C_DELTA_XLABEL_Y = -0.40
PANEL_C_SWAP_EXAMPLES = (
    ("L15", "20241121", "v1", 27, "center_to_right"),
    ("L19", "20250930", "v1", 4, "center_to_left"),
    ("L15", "20241121", "v1", 146, "center_to_right"),
)
PANEL_C_EXAMPLE_AXIS_BOUNDS = (
    (0.095, -0.18, 0.20, 0.19),
    (0.405, -0.18, 0.20, 0.19),
    (0.715, -0.18, 0.20, 0.19),
)
PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS = (
    (0.96, 0.94),
    (0.96, 0.06),
    (0.96, 0.94),
)
PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS = ("top", "bottom", "top")
PANEL_C_EXAMPLE_ICON_BOUNDS = (-0.46, 0.28, 0.26, 0.38)
PANEL_C_PREDICTION_LABEL_FONTSIZE = 5.8
PANEL_C_SWAP_MODEL_NAME = "task_segment_scalar"
PANEL_C_SWAP_MODEL_COLORS = {
    "visual": PANEL_INDEPENDENT_MODEL_COLOR,
    PANEL_C_SWAP_MODEL_NAME: PANEL_SHARED_SCAFFOLD_MODEL_COLOR,
}
PANEL_C_SWAP_MODEL_LABELS = {
    PANEL_C_SWAP_MODEL_NAME: "Shared-scaffold",
}
PANEL_C_INDEPENDENT_TRACK_CENTER_Y = 0.742
PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y = 0.60
PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y = 0.34
PANEL_C_SHARED_DARK_TRACK_CENTER_Y = 0.0
PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y = 0.17
PANEL_C_SHARED_PREDICTION_LABEL_Y = -0.24
PANEL_C_SCHEMATIC_TRACK_SIZE = (0.628, 0.316)
PANEL_C_HORIZONTAL_SHIFT = -0.025


def _panel_b_schematic_center_y_for_panel_c_center_y(panel_c_center_y: float) -> float:
    """Return a Panel B schematic y-center aligned to one Panel C schematic row."""
    panel_b_schematic_bottom = 1.0 - PANEL_B_SCHEMATIC_HEIGHT_FRACTION
    panel_c_parent_center_y = (
        PANEL_B_ALIGNMENT_SCHEMATIC_AXIS_BOUNDS[1]
        + PANEL_B_ALIGNMENT_SCHEMATIC_AXIS_BOUNDS[3] * float(panel_c_center_y)
    )
    return (
        panel_c_parent_center_y - panel_b_schematic_bottom
    ) / PANEL_B_SCHEMATIC_HEIGHT_FRACTION


PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y = (
    _panel_b_schematic_center_y_for_panel_c_center_y(
        PANEL_H_INDEPENDENT_TRACK_CENTER_Y
    )
)
PANEL_B_FIELD_LABEL_Y = 0.9619
PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y = (
    _panel_b_schematic_center_y_for_panel_c_center_y(
        (
            PANEL_H_SEGMENT_MODULATION_TRACK_CENTER_Y
            + PANEL_H_SHARED_DARK_TRACK_CENTER_Y
            + PANEL_H_SHARED_LIGHT_TRACK_CENTER_Y
        )
        / 3.0
    )
)


def _shift_axis_horizontally(ax: Any, dx_figure_fraction: float) -> None:
    """Shift one axis after constrained layout has selected its size."""
    if dx_figure_fraction == 0.0:
        return
    box = ax.get_position()
    ax.set_axes_locator(None)
    ax.set_position(
        [
            box.x0 + dx_figure_fraction,
            box.y0,
            box.width,
            box.height,
        ]
    )


def _align_texts_to_reference_display_y(texts: Sequence[Any]) -> None:
    """Align text artists to the first text's rendered vertical position."""
    if len(texts) < 2:
        return
    for text in texts:
        axes = getattr(text, "axes", None)
        if axes is not None and text is axes.title:
            axes._autotitlepos = False
    reference_display = texts[0].get_transform().transform(texts[0].get_position())
    for text in texts[1:]:
        x_position, _y_position = text.get_position()
        display_position = text.get_transform().transform(text.get_position())
        adjusted_position = text.get_transform().inverted().transform(
            (display_position[0], reference_display[1])
        )
        text.set_position((x_position, float(adjusted_position[1])))


def _align_text_to_reference_display_x(text: Any, reference_text: Any) -> None:
    """Align one text artist to another text artist's rendered horizontal position."""
    reference_display = reference_text.get_transform().transform(
        reference_text.get_position()
    )
    x_position, _y_position = text.get_position()
    display_position = text.get_transform().transform(text.get_position())
    adjusted_position = text.get_transform().inverted().transform(
        (reference_display[0], display_position[1])
    )
    text.set_position((float(adjusted_position[0]), _y_position))


def _format_panel_b_cache_token(value: Any) -> str:
    """Return a filesystem-safe token for one Panel B cache value."""
    token = "".join(
        character if character.isalnum() else "_"
        for character in str(value).strip()
    ).strip("_")
    return token or "none"


def _format_panel_b_cache_number(value: float) -> str:
    """Return a compact Panel B numeric cache token."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_panel_b_dark_movement_firing_rate_cache_metadata(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str,
    panel: str = "B",
) -> dict[str, Any]:
    """Return metadata for one Panel B dark movement firing-rate cache."""
    return {
        "cache_version": DARK_MOVEMENT_FR_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": str(panel),
        "artifact": "dark_movement_firing_rate",
        "data_root": str(Path(data_root)),
        "animal_name": str(animal_name),
        "date": str(date),
        "dark_epoch": str(dark_epoch),
        "region": str(region),
        "speed_threshold_cm_s": float(DEFAULT_SPEED_THRESHOLD_CM_S),
        "columns": list(DARK_MOVEMENT_FR_CACHE_COLUMNS),
    }


def build_panel_b_dark_movement_firing_rate_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the descriptive cache path for one Panel B dark-rate table."""
    region = _format_panel_b_cache_token(metadata["region"])
    animal_name = _format_panel_b_cache_token(metadata["animal_name"])
    date = _format_panel_b_cache_token(metadata["date"])
    dark_epoch = _format_panel_b_cache_token(metadata["dark_epoch"])
    speed = _format_panel_b_cache_number(float(metadata["speed_threshold_cm_s"]))
    cache_version = int(metadata["cache_version"])
    filename = (
        f"{DEFAULT_OUTPUT_NAME}_dark_movement_firing_rate_"
        f"{region}_{animal_name}_{date}_{dark_epoch}"
        f"_speed{speed}_cachev{cache_version}.parquet"
    )
    return Path(cache_dir) / filename


def load_panel_b_dark_movement_firing_rate_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str,
    cache_dir: Path | None,
    refresh_cache: bool,
) -> Any:
    """Return dark movement firing rates for the Panel B activity split."""
    import pandas as pd

    from v1ca1.helper.session import compute_movement_firing_rates
    from v1ca1.task_progression._session import prepare_task_progression_session

    metadata = build_panel_b_dark_movement_firing_rate_cache_metadata(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        dark_epoch=dark_epoch,
        region=region,
    )
    cache_path = (
        build_panel_b_dark_movement_firing_rate_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached_table = load_dark_movement_firing_rate_cache(cache_path, metadata)
        if cached_table is not None:
            print(f"Loaded Panel B dark movement firing-rate cache from {cache_path}.")
            return cached_table
        historical_metadata = build_panel_b_dark_movement_firing_rate_cache_metadata(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            dark_epoch=dark_epoch,
            region=region,
            panel="D",
        )
        cached_table = load_dark_movement_firing_rate_cache(
            cache_path,
            historical_metadata,
        )
        if cached_table is not None:
            print(
                "Loaded historical Panel D dark movement firing-rate cache "
                f"from {cache_path}."
            )
            return cached_table

    session = prepare_task_progression_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        selected_run_epochs=[dark_epoch],
        load_body_position=False,
        include_generalized_place=False,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    spikes = session["spikes_by_region"][region]
    unit_ids = np.asarray(list(spikes.keys()), dtype=int)
    firing_rates_hz = np.asarray(movement_firing_rates[region][dark_epoch], dtype=float)
    if unit_ids.shape[0] != firing_rates_hz.shape[0]:
        raise ValueError(
            "Panel B dark movement firing-rate table is not aligned with spike unit IDs: "
            f"{unit_ids.shape[0]} unit IDs and {firing_rates_hz.shape[0]} rates."
        )
    table = pd.DataFrame(
        {
            "unit": unit_ids,
            "dark_firing_rate_hz": firing_rates_hz,
        }
    )
    if cache_path is not None:
        save_dark_movement_firing_rate_cache(cache_path, table, metadata)
        print(f"Saved Panel B dark movement firing-rate cache to {cache_path}.")
    return table


def _compute_curve_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the finite Pearson correlation between two tuning curves."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        return float("nan")
    valid = np.isfinite(first) & np.isfinite(second)
    if np.sum(valid) < 2:
        return float("nan")
    first_values = first[valid]
    second_values = second[valid]
    if np.nanstd(first_values) <= 0.0 or np.nanstd(second_values) <= 0.0:
        return float("nan")
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _compute_curve_peak(values: np.ndarray) -> float:
    """Return the finite peak firing rate in one tuning curve."""
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return float("nan")
    return float(np.nanmax(finite_values))


def _select_dark_peak_tuning_correlation(
    records: Sequence[tuple[str, float, float]],
) -> tuple[str, float, float] | None:
    """Return trajectory, correlation, and peak for the largest dark tuning peak."""
    finite_peak_records = [
        (str(trajectory), float(correlation), float(dark_peak))
        for trajectory, correlation, dark_peak in records
        if np.isfinite(dark_peak)
    ]
    if not finite_peak_records:
        return None
    return max(finite_peak_records, key=lambda item: item[2])


def load_panel_b_light_dark_tuning_correlation_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    light_epoch: str,
    dark_epoch: str,
    region: str,
) -> Any:
    """Return per-unit light-vs-dark correlations for the dark peak trajectory."""
    import pandas as pd
    import xarray as xr

    correlations: dict[int, list[tuple[str, float, float]]] = {}
    for trajectory in PANEL_B_TUNING_CORRELATION_TRAJECTORIES:
        light_path = get_compute_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=light_epoch,
            trajectory=trajectory,
        )
        dark_path = get_compute_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=dark_epoch,
            trajectory=trajectory,
        )
        if not light_path.exists():
            raise FileNotFoundError(f"Missing light tuning-curve artifact: {light_path}")
        if not dark_path.exists():
            raise FileNotFoundError(f"Missing dark tuning-curve artifact: {dark_path}")
        with xr.open_dataarray(light_path) as light_curves, xr.open_dataarray(
            dark_path
        ) as dark_curves:
            units = np.intersect1d(
                np.asarray(light_curves.coords["unit"].values, dtype=int),
                np.asarray(dark_curves.coords["unit"].values, dtype=int),
            )
            for unit_id in units:
                light_curve = light_curves.sel(unit=int(unit_id)).values
                dark_curve = dark_curves.sel(unit=int(unit_id)).values
                correlations.setdefault(int(unit_id), []).append(
                    (
                        trajectory,
                        _compute_curve_correlation(light_curve, dark_curve),
                        _compute_curve_peak(dark_curve),
                    )
                )

    rows = []
    for unit_id, unit_correlations in correlations.items():
        selected_record = _select_dark_peak_tuning_correlation(unit_correlations)
        if selected_record is None:
            continue
        best_trajectory, best_correlation, best_dark_peak = selected_record
        if not np.isfinite(best_correlation):
            continue
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "unit": int(unit_id),
                "light_epoch": light_epoch,
                "dark_epoch": dark_epoch,
                "best_trajectory": best_trajectory,
                "best_dark_peak_firing_rate_hz": float(best_dark_peak),
                "light_dark_tuning_correlation": float(best_correlation),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "animal_name",
            "date",
            "unit",
            "light_epoch",
            "dark_epoch",
            "best_trajectory",
            "best_dark_peak_firing_rate_hz",
            "light_dark_tuning_correlation",
        ],
    )


def _load_panel_b_saved_place_tuning_curves(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectories: Sequence[str],
) -> dict[str, Any]:
    """Load cached trajectory tuning curves for DPP-overlap fallback."""
    import xarray as xr

    curves: dict[str, Any] = {}
    for trajectory in trajectories:
        path = get_compute_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory=trajectory,
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing tuning-curve artifact: {path}")
        with xr.open_dataarray(path) as data_array:
            if "unit" not in data_array.dims:
                raise ValueError(f"Tuning curve {path} is missing a unit dimension.")
            data_array = data_array.transpose("unit", ...)
            units = np.asarray(data_array.coords["unit"].values, dtype=int)
            values = np.asarray(data_array.values, dtype=float)
        curves[str(trajectory)] = {
            "units": units,
            "values": values.reshape(values.shape[0], -1),
            "index_by_unit": {int(unit): index for index, unit in enumerate(units)},
        }
    return curves


def _compute_panel_b_similarity_from_saved_curves(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    similarity_metric: str,
) -> Any:
    """Compute one within-epoch similarity table from cached tuning curves."""
    import pandas as pd

    from v1ca1.task_progression.tuning_analysis import (
        DIRECT_COMPARISON_SPECS,
        compute_similarity_score,
    )

    reference_path = get_tuning_similarity_path(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
    )
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing reference tuning-similarity artifact: {reference_path}"
        )
    reference_table = pd.read_parquet(reference_path)
    required_columns = ("unit", "region", "epoch", "comparison_label")
    missing_columns = [
        column for column in required_columns if column not in reference_table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Tuning similarity table {reference_path} is missing columns "
            f"{missing_columns!r}."
        )

    trajectories = sorted(
        {str(spec["trajectory_a"]) for spec in DIRECT_COMPARISON_SPECS}
        | {str(spec["trajectory_b"]) for spec in DIRECT_COMPARISON_SPECS}
    )
    curves = _load_panel_b_saved_place_tuning_curves(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        trajectories=trajectories,
    )

    rows: list[dict[str, Any]] = []
    for spec in DIRECT_COMPARISON_SPECS:
        comparison_label = str(spec["comparison_label"])
        reference_rows = reference_table[
            (reference_table["region"].astype(str) == str(region))
            & (reference_table["epoch"].astype(str) == str(epoch))
            & (reference_table["comparison_label"].astype(str) == comparison_label)
        ].copy()
        if reference_rows.empty:
            continue
        reference_rows["unit"] = pd.to_numeric(reference_rows["unit"], errors="coerce")
        reference_rows = reference_rows[
            np.isfinite(reference_rows["unit"].to_numpy(dtype=float))
        ].copy()
        reference_rows["unit"] = reference_rows["unit"].astype(int)

        trajectory_a = str(spec["trajectory_a"])
        trajectory_b = str(spec["trajectory_b"])
        curve_a = curves[trajectory_a]
        curve_b = curves[trajectory_b]
        common_units = set(
            np.intersect1d(curve_a["units"], curve_b["units"]).tolist()
        )
        for _row_index, reference_row in reference_rows.iterrows():
            unit = int(reference_row["unit"])
            if unit not in common_units:
                continue
            values_a = curve_a["values"][curve_a["index_by_unit"][unit]]
            values_b = curve_b["values"][curve_b["index_by_unit"][unit]]
            if bool(spec["flip_trajectory_b"]):
                values_b = values_b[::-1]
            similarity = compute_similarity_score(
                values_a,
                values_b,
                similarity_metric=similarity_metric,
            )
            if not np.isfinite(similarity):
                continue
            rows.append(
                {
                    "unit": unit,
                    "region": str(region),
                    "epoch": str(epoch),
                    "comparison_label": comparison_label,
                    "similarity": float(similarity),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["unit", "region", "epoch", "comparison_label", "similarity"]
        )
    return pd.DataFrame(rows)


def load_panel_b_dpp_similarity_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    value_column: str,
    similarity_metric: str = "correlation",
) -> Any:
    """Return each unit's same-turn DPP similarities for one epoch."""
    import pandas as pd

    if similarity_metric == "correlation":
        path = get_tuning_similarity_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
        )
    else:
        path = get_tuning_similarity_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            similarity_metric=similarity_metric,
        )
    if path.exists():
        table = pd.read_parquet(path)
    elif similarity_metric != "correlation":
        table = _compute_panel_b_similarity_from_saved_curves(
            data_root,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            region=region,
            similarity_metric=similarity_metric,
        )
    else:
        raise FileNotFoundError(f"Missing DPP similarity artifact: {path}")
    missing_columns = [
        column
        for column in ("unit", "region", "epoch", "comparison_label", "similarity")
        if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"DPP similarity table {path} is missing columns {missing_columns!r}."
        )
    rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & (
            table["comparison_label"]
            .astype(str)
            .isin(PANEL_B_DPP_COMPARISON_LABELS)
        )
    ].copy()
    rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
    rows["similarity"] = pd.to_numeric(rows["similarity"], errors="coerce")
    rows = rows[
        np.isfinite(rows["unit"].to_numpy(dtype=float))
        & np.isfinite(rows["similarity"].to_numpy(dtype=float))
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["unit", "comparison_label", value_column])
    rows["unit"] = rows["unit"].astype(int)
    rows["comparison_label"] = rows["comparison_label"].astype(str)
    return rows.loc[:, ["unit", "comparison_label", "similarity"]].rename(
        columns={"similarity": value_column}
    )


def load_panel_b_dark_dpp_index_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    dark_epoch: str,
    region: str,
    similarity_metric: str = "correlation",
) -> Any:
    """Return each unit's dark-selected same-turn DPP index."""
    import pandas as pd

    rows = load_panel_b_dpp_similarity_table(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=dark_epoch,
        region=region,
        value_column="dark_dpp_index",
        similarity_metric=similarity_metric,
    )
    if rows.empty:
        return pd.DataFrame(
            columns=["unit", "dpp_comparison_label", "dark_dpp_index"]
        )
    rows = rows.sort_values(
        ["unit", "dark_dpp_index", "comparison_label"],
        ascending=[True, False, True],
    )
    return (
        rows.drop_duplicates("unit", keep="first")
        .rename(columns={"comparison_label": "dpp_comparison_label"})
        .reset_index(drop=True)
    )


def load_panel_b_light_dpp_index_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    light_epoch: str,
    region: str,
    similarity_metric: str = "correlation",
) -> Any:
    """Return each unit's light same-turn DPP similarities."""
    return load_panel_b_dpp_similarity_table(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=light_epoch,
        region=region,
        value_column="light_dpp_index",
        similarity_metric=similarity_metric,
    )


def load_panel_b_light_tuning_stability_table(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    light_epoch: str,
    region: str,
    min_tuning_stability_correlation: float = (
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Return units stable in at least one light trajectory for Figure 2B."""
    import pandas as pd

    if min_tuning_stability_correlation < -1.0:
        raise ValueError("min_tuning_stability_correlation must be at least -1.")

    path = get_stability_table_path(data_root, animal_name, date)
    if not path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(path)
    missing_columns = [
        column
        for column in (
            "unit",
            "region",
            "epoch",
            "trajectory_type",
            "stability_correlation",
        )
        if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Tuning stability table {path} is missing columns {missing_columns!r}."
        )
    rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(light_epoch))
        & (
            table["trajectory_type"]
            .astype(str)
            .isin(PANEL_B_TUNING_CORRELATION_TRAJECTORIES)
        )
    ].copy()
    rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
    rows["stability_correlation"] = pd.to_numeric(
        rows["stability_correlation"],
        errors="coerce",
    )
    rows = rows[
        np.isfinite(rows["unit"].to_numpy(dtype=float))
        & np.isfinite(rows["stability_correlation"].to_numpy(dtype=float))
        & (
            rows["stability_correlation"]
            >= float(min_tuning_stability_correlation)
        )
    ].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["unit", "max_light_tuning_stability_correlation"]
        )
    rows["unit"] = rows["unit"].astype(int)
    return (
        rows.groupby("unit", as_index=False, observed=False)[
            "stability_correlation"
        ]
        .max()
        .rename(
            columns={
                "stability_correlation": "max_light_tuning_stability_correlation"
            }
        )
    )


def load_panel_b_tuning_correlation_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
) -> Any:
    """Return the Figure 3B scatter pairs used for the Figure 2B projection."""
    similarity_table = load_panel_c_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    return build_panel_c_similarity_pairs(similarity_table)


def load_panel_b_tuning_overlap_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    similarity_metric: str = PANEL_B_DPP_OVERLAP_METRIC,
) -> Any:
    """Return paired dark/light DPP overlap values for panel B."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        for epoch_type, epoch in (
            ("dark", dataset_dark_epoch),
            ("light", dataset_light_epoch),
        ):
            rows = load_panel_b_dpp_similarity_table(
                data_root,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                region=region,
                value_column="similarity",
                similarity_metric=similarity_metric,
            )
            if rows.empty:
                continue
            rows = rows.assign(
                animal_name=animal_name,
                date=date,
                epoch_type=epoch_type,
                epoch=epoch,
            )
            tables.append(
                rows[
                    [
                        "animal_name",
                        "date",
                        "epoch_type",
                        "epoch",
                        "unit",
                        "comparison_label",
                        "similarity",
                    ]
                ]
            )

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "unit",
                "comparison_label",
                "similarity_light",
                "similarity_dark",
            ]
        )
    return build_panel_c_similarity_pairs(pd.concat(tables, axis=0, ignore_index=True))


def load_panel_b_dark_activity_light_dpp_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    dark_movement_fr_cache_dir: Path | None,
    refresh_dark_movement_fr_cache: bool = False,
    dark_activity_threshold_hz: float = PANEL_B_DARK_ACTIVITY_THRESHOLD_HZ,
    min_tuning_stability_correlation: float = (
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Return stable-light cells with light DPP and dark activity labels."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        light_dpp = load_panel_b_light_dpp_index_table(
            data_root,
            animal_name=animal_name,
            date=date,
            light_epoch=dataset_light_epoch,
            region=region,
        )
        dark_activity = load_panel_b_dark_movement_firing_rate_table(
            data_root,
            animal_name=animal_name,
            date=date,
            dark_epoch=dataset_dark_epoch,
            region=region,
            cache_dir=dark_movement_fr_cache_dir,
            refresh_cache=refresh_dark_movement_fr_cache,
        )
        light_stability = load_panel_b_light_tuning_stability_table(
            data_root,
            animal_name=animal_name,
            date=date,
            light_epoch=dataset_light_epoch,
            region=region,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
        )
        if light_dpp.empty or dark_activity.empty or light_stability.empty:
            continue
        light_dpp = light_dpp.copy()
        dark_activity = dark_activity.copy()
        light_stability = light_stability.copy()
        light_dpp["unit"] = pd.to_numeric(
            light_dpp["unit"],
            errors="coerce",
        )
        light_dpp["light_dpp_index"] = pd.to_numeric(
            light_dpp["light_dpp_index"],
            errors="coerce",
        )
        dark_activity["unit"] = pd.to_numeric(
            dark_activity["unit"],
            errors="coerce",
        )
        light_stability["unit"] = pd.to_numeric(
            light_stability["unit"],
            errors="coerce",
        )
        light_dpp = light_dpp[
            np.isfinite(light_dpp["unit"].to_numpy(dtype=float))
            & np.isfinite(light_dpp["light_dpp_index"].to_numpy(dtype=float))
        ].copy()
        dark_activity = dark_activity[
            np.isfinite(dark_activity["unit"].to_numpy(dtype=float))
        ].copy()
        light_stability = light_stability[
            np.isfinite(light_stability["unit"].to_numpy(dtype=float))
        ].copy()
        light_dpp["unit"] = light_dpp["unit"].astype(int)
        dark_activity["unit"] = dark_activity["unit"].astype(int)
        light_stability["unit"] = light_stability["unit"].astype(int)

        light_dpp = (
            light_dpp.sort_values(
                ["unit", "light_dpp_index", "comparison_label"],
                ascending=[True, False, True],
            )
            .drop_duplicates("unit", keep="first")
            .rename(columns={"comparison_label": "dpp_comparison_label"})
        )
        joined = (
            light_dpp.merge(dark_activity, on="unit", how="inner")
            .merge(light_stability, on="unit", how="inner")
        )
        if joined.empty:
            continue
        tables.append(
            joined.assign(
                animal_name=animal_name,
                date=date,
                light_epoch=dataset_light_epoch,
                dark_epoch=dataset_dark_epoch,
                similarity_light=np.asarray(
                    joined["light_dpp_index"],
                    dtype=float,
                ),
            )
        )

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "unit",
                "light_epoch",
                "dark_epoch",
                "dpp_comparison_label",
                "light_dpp_index",
                "similarity_light",
                "dark_firing_rate_hz",
                "max_light_tuning_stability_correlation",
                "dark_active",
                "dark_activity_group",
            ]
        )

    joined = pd.concat(tables, axis=0, ignore_index=True)
    if joined.empty:
        return joined.assign(
            dark_firing_rate_hz=np.nan,
            max_light_tuning_stability_correlation=np.nan,
            dark_active=False,
            dark_activity_group="Dark inactive",
        )
    dark_rates = pd.to_numeric(joined["dark_firing_rate_hz"], errors="coerce")
    joined = joined[np.isfinite(dark_rates.to_numpy(dtype=float))].copy()
    dark_rates = pd.to_numeric(joined["dark_firing_rate_hz"], errors="coerce")
    dark_active = dark_rates.to_numpy(dtype=float) >= float(dark_activity_threshold_hz)
    return joined.assign(
        dark_active=dark_active,
        dark_activity_group=np.where(dark_active, "Dark active", "Dark inactive"),
    )


def build_panel_b_stable_light_paired_dpp_table(
    paired_table: Any,
    dark_activity_table: Any,
) -> Any:
    """Join paired dark/light DPP rows to stable-light dark-activity metadata."""
    import pandas as pd

    if paired_table is None or dark_activity_table is None:
        return paired_table
    if not len(paired_table) or not len(dark_activity_table):
        return paired_table.iloc[0:0].copy()

    key_columns = ["animal_name", "date", "unit"]
    metadata_columns = [
        *key_columns,
        "dark_firing_rate_hz",
        "max_light_tuning_stability_correlation",
        "dark_active",
        "dark_activity_group",
    ]
    missing_columns = [
        column for column in metadata_columns if column not in dark_activity_table
    ]
    if missing_columns:
        raise ValueError(
            f"Panel B dark-activity table is missing columns {missing_columns!r}."
        )

    paired = paired_table.copy()
    metadata = dark_activity_table.loc[:, metadata_columns].copy()
    for table in (paired, metadata):
        table["unit"] = pd.to_numeric(table["unit"], errors="coerce")
        table.dropna(subset=["unit"], inplace=True)
        table["unit"] = table["unit"].astype(int)
    metadata = metadata.drop_duplicates(key_columns)
    return paired.merge(metadata, on=key_columns, how="inner")


def _plot_panel_b_boxplot(
    ax: Any,
    values_by_group: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    ylabel: str | None = None,
) -> None:
    """Plot one compact box-and-whisker comparison with jittered cells."""
    plot_data = [np.asarray(values, dtype=float) for values in values_by_group]
    plot_data = [values[np.isfinite(values)] for values in plot_data]
    box_positions = [
        position
        for position, values in enumerate(plot_data, start=1)
        if values.size
    ]
    box_data = [values for values in plot_data if values.size]
    if box_data:
        boxplot = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "edgecolor": "0.25",
                "linewidth": 0.65,
            },
            medianprops={
                "color": "black",
                "linewidth": 0.8,
            },
            whiskerprops={
                "color": "0.25",
                "linewidth": 0.65,
            },
            capprops={
                "color": "0.25",
                "linewidth": 0.65,
            },
        )
        for patch, position in zip(boxplot["boxes"], box_positions, strict=False):
            patch.set_facecolor(colors[position - 1])
            patch.set_alpha(0.36)
            patch.set_zorder(2)
        for artist_group in ("medians", "whiskers", "caps"):
            for artist in boxplot[artist_group]:
                artist.set_zorder(4)

        for position in box_positions:
            values = plot_data[position - 1]
            rng = np.random.default_rng(22_000 + position)
            x_values = position + rng.uniform(-0.13, 0.13, size=values.size)
            ax.scatter(
                x_values,
                values,
                s=3.5,
                color=colors[position - 1],
                alpha=0.34,
                edgecolors="none",
                zorder=3,
            )
            if values.size < 2 or np.nanstd(values) <= 0.0:
                ax.hlines(
                    float(np.nanmedian(values)),
                    position - 0.22,
                    position + 0.22,
                    color="black",
                    linewidth=0.7,
                    zorder=4,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No values",
            ha="center",
            va="center",
            fontsize=6.0,
            transform=ax.transAxes,
        )

    ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.55, zorder=0)
    ax.axhline(0.5, color="0.25", linestyle=":", linewidth=0.55, zorder=0)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    tick_labels = [
        f"{label}\nn={values.size}"
        for label, values in zip(labels, plot_data, strict=True)
    ]
    ax.set_xticklabels(tick_labels, fontsize=4.6)
    ax.set_title(title, fontsize=5.8, pad=1.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=5.4, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1)
    ax.tick_params(axis="x", length=0.0, pad=1)


def _fraction_weights(values: np.ndarray) -> np.ndarray:
    """Return histogram weights that sum to one."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return np.full(values.shape, 1.0 / values.size, dtype=float)


def _plot_panel_b_overlap_boxplot(
    ax: Any,
    values_by_group: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    ylabel: str | None = None,
) -> None:
    """Plot compact light-overlap distributions grouped by dark overlap."""
    plot_data = [np.asarray(values, dtype=float) for values in values_by_group]
    plot_data = [values[np.isfinite(values)] for values in plot_data]
    box_positions = [
        position
        for position, values in enumerate(plot_data, start=1)
        if values.size
    ]
    box_data = [values for values in plot_data if values.size]
    if box_data:
        boxplot = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            boxprops={"edgecolor": "0.25", "linewidth": 0.65},
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"color": "0.25", "linewidth": 0.65},
            capprops={"color": "0.25", "linewidth": 0.65},
        )
        for patch, position in zip(boxplot["boxes"], box_positions, strict=False):
            patch.set_facecolor(colors[position - 1])
            patch.set_alpha(0.36)
            patch.set_zorder(2)
        for artist_group in ("medians", "whiskers", "caps"):
            for artist in boxplot[artist_group]:
                artist.set_zorder(4)

        for position in box_positions:
            values = plot_data[position - 1]
            rng = np.random.default_rng(42_000 + position)
            x_values = position + rng.uniform(-0.13, 0.13, size=values.size)
            ax.scatter(
                x_values,
                values,
                s=3.1,
                color=colors[position - 1],
                alpha=0.30,
                edgecolors="none",
                zorder=3,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No values",
            ha="center",
            va="center",
            fontsize=5.0,
            transform=ax.transAxes,
        )

    ax.axhline(0.5, color="0.25", linestyle=":", linewidth=0.55, zorder=0)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    tick_labels = [
        f"{label}\nn={values.size}"
        for label, values in zip(labels, plot_data, strict=True)
    ]
    ax.set_xticklabels(tick_labels, fontsize=4.1)
    ax.set_title(title, fontsize=5.4, pad=1.2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=5.0, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=4.4, length=1.4, pad=1)
    ax.tick_params(axis="x", length=0.0, pad=1)

    summaries = []
    for label, values in zip(("low", "mid", "high"), plot_data, strict=True):
        summaries.append(
            f"{label} {np.mean(values > 0.5):.0%} > 0.5" if values.size else f"{label} n/a"
        )
    ax.text(
        0.98,
        0.04,
        "\n".join(summaries),
        ha="right",
        va="bottom",
        fontsize=3.6,
        transform=ax.transAxes,
    )


def _plot_panel_b_overlap_scatter_with_marginals(
    ax: Any,
    table: Any,
    *,
    title: str | None = "Dark vs light DPP",
) -> None:
    """Plot dark-vs-light DPP overlap with marginal fraction histograms."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    scatter_ax = ax.inset_axes((0.14, 0.13, 0.60, 0.62))
    top_ax = ax.inset_axes((0.14, 0.79, 0.60, 0.15), sharex=scatter_ax)
    right_ax = ax.inset_axes((0.78, 0.13, 0.18, 0.62), sharey=scatter_ax)

    if table is None or not len(table):
        scatter_ax.text(0.5, 0.5, "No paired\noverlap", ha="center", va="center")
        return

    dark_values = np.asarray(table["similarity_dark"], dtype=float)
    light_values = np.asarray(table["similarity_light"], dtype=float)
    valid = np.isfinite(dark_values) & np.isfinite(light_values)
    dark_values = np.clip(dark_values[valid], 0.0, 1.0)
    light_values = np.clip(light_values[valid], 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 24)

    scatter_ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="0.45",
        linestyle="--",
        linewidth=0.55,
        zorder=1,
    )
    scatter_ax.axvline(0.5, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
    scatter_ax.axhline(0.5, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
    scatter_ax.scatter(
        dark_values,
        light_values,
        s=3.5,
        color="0.25",
        alpha=0.26,
        edgecolors="none",
        zorder=2,
    )
    if dark_values.size:
        median_delta = float(np.nanmedian(light_values - dark_values))
        scatter_ax.text(
            0.04,
            0.94,
            f"median Δ={median_delta:.2f}",
            ha="left",
            va="top",
            fontsize=4.0,
            transform=scatter_ax.transAxes,
        )
    scatter_ax.set_xlim(0.0, 1.0)
    scatter_ax.set_ylim(0.0, 1.0)
    scatter_ax.set_xlabel("Dark DPP\noverlap", fontsize=4.7, labelpad=1.0)
    scatter_ax.set_ylabel("Light DPP\noverlap", fontsize=4.7, labelpad=1.0)
    scatter_ax.tick_params(labelsize=3.8, length=1.3, pad=0.8)
    scatter_ax.spines["top"].set_visible(False)
    scatter_ax.spines["right"].set_visible(False)

    top_ax.hist(
        dark_values,
        bins=bins,
        weights=_fraction_weights(dark_values),
        color="0.55",
        edgecolor="none",
        alpha=0.70,
    )
    top_ax.axvline(0.5, color="0.35", linestyle=":", linewidth=0.55)
    top_ax.set_ylim(0.0, 0.13)
    top_ax.set_ylabel("Frac.", fontsize=3.6, labelpad=0.6)
    top_ax.tick_params(axis="x", labelbottom=False, length=0.0)
    top_ax.tick_params(axis="y", labelsize=3.2, length=1.0, pad=0.5)
    top_ax.spines["top"].set_visible(False)
    top_ax.spines["right"].set_visible(False)

    right_ax.hist(
        light_values,
        bins=bins,
        weights=_fraction_weights(light_values),
        color=PANEL_B_ACTIVITY_COLORS["inactive"],
        edgecolor="none",
        alpha=0.55,
        orientation="horizontal",
    )
    right_ax.axhline(0.5, color="0.35", linestyle=":", linewidth=0.55)
    right_ax.set_xlim(0.0, 0.13)
    right_ax.set_xlabel("Frac.", fontsize=3.6, labelpad=0.6)
    right_ax.tick_params(axis="y", labelleft=False, length=0.0)
    right_ax.tick_params(axis="x", labelsize=3.2, length=1.0, pad=0.5)
    right_ax.spines["top"].set_visible(False)
    right_ax.spines["right"].set_visible(False)

    if title is not None:
        ax.text(
            0.52,
            0.99,
            title,
            ha="center",
            va="top",
            fontsize=5.4,
            transform=ax.transAxes,
        )


def _plot_panel_b_violin(
    ax: Any,
    values_by_group: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    ylabel: str | None = None,
) -> None:
    """Plot one compact violin comparison with jittered cells and medians."""
    plot_data = [np.asarray(values, dtype=float) for values in values_by_group]
    plot_data = [values[np.isfinite(values)] for values in plot_data]
    positions = np.arange(1, len(plot_data) + 1)
    nonempty_positions = [
        position
        for position, values in zip(positions, plot_data, strict=True)
        if values.size
    ]
    nonempty_data = [values for values in plot_data if values.size]
    if nonempty_data:
        parts = ax.violinplot(
            nonempty_data,
            positions=nonempty_positions,
            widths=0.58,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, position in zip(parts["bodies"], nonempty_positions, strict=True):
            body.set_facecolor(colors[position - 1])
            body.set_edgecolor("none")
            body.set_alpha(0.34)
            body.set_zorder(1)
        for position in nonempty_positions:
            values = plot_data[position - 1]
            q25, median, q75 = np.nanpercentile(values, [25, 50, 75])
            ax.plot(
                [position, position],
                [q25, q75],
                color="0.25",
                linewidth=0.7,
                zorder=3,
            )
            ax.plot(
                [position - 0.17, position + 0.17],
                [median, median],
                color="black",
                linewidth=0.85,
                zorder=4,
            )
            rng = np.random.default_rng(32_000 + int(position))
            x_values = position + rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                x_values,
                values,
                s=3.2,
                color=colors[position - 1],
                alpha=0.30,
                edgecolors="none",
                zorder=2,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No values",
            ha="center",
            va="center",
            fontsize=6.0,
            transform=ax.transAxes,
        )

    ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.55, zorder=0)
    ax.axhline(0.5, color="0.25", linestyle=":", linewidth=0.55, zorder=0)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(positions)
    tick_labels = [
        f"{label}\nn={values.size}"
        for label, values in zip(labels, plot_data, strict=True)
    ]
    ax.set_xticklabels(tick_labels, fontsize=4.5)
    ax.set_title(title, fontsize=5.8, pad=1.5)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=5.4, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1)
    ax.tick_params(axis="x", length=0.0, pad=1)


def _get_panel_b_light_similarity_by_dark_similarity_values(
    table: Any,
    *,
    low_threshold: float,
    high_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Fig3B x-values split into low, middle, and high dark-corr bins."""
    if high_threshold <= low_threshold:
        raise ValueError("high_threshold must be greater than low_threshold.")
    if table is None or not len(table):
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    dark_values = np.asarray(table["similarity_dark"], dtype=float)
    light_values = np.asarray(table["similarity_light"], dtype=float)
    valid = np.isfinite(dark_values) & np.isfinite(light_values)
    dark_values = dark_values[valid]
    light_values = light_values[valid]
    return (
        light_values[dark_values < float(low_threshold)],
        light_values[
            (dark_values >= float(low_threshold))
            & (dark_values < float(high_threshold))
        ],
        light_values[dark_values >= float(high_threshold)],
    )


def _get_panel_b_light_dpp_by_dark_activity_values(
    table: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return light DPP split by dark movement activity group."""
    if table is None or not len(table):
        empty = np.asarray([], dtype=float)
        return empty, empty
    light_values = np.asarray(table["similarity_light"], dtype=float)
    dark_active = np.asarray(table["dark_active"], dtype=bool)
    valid = np.isfinite(light_values)
    return (
        light_values[valid & ~dark_active],
        light_values[valid & dark_active],
    )


def _get_panel_b_light_dpp_by_dark_dpp_threshold_values(
    table: Any,
    *,
    threshold: float,
    dark_active_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return light DPP split by low/high dark DPP in dark-active cells."""
    if table is None or not len(table):
        empty = np.asarray([], dtype=float)
        return empty, empty
    dark_values = np.asarray(table["similarity_dark"], dtype=float)
    light_values = np.asarray(table["similarity_light"], dtype=float)
    valid = np.isfinite(dark_values) & np.isfinite(light_values)
    if dark_active_only and "dark_active" in table:
        valid &= np.asarray(table["dark_active"], dtype=bool)
    return (
        light_values[valid & (dark_values < float(threshold))],
        light_values[valid & (dark_values > float(threshold))],
    )


def plot_panel_b_dpp_overlap_grouped(
    ax: Any,
    table: Any,
    *,
    low_threshold: float = PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
    high_threshold: float = PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
) -> None:
    """Plot light DPP overlap grouped by dark DPP overlap."""
    low_values, middle_values, high_values = (
        _get_panel_b_light_similarity_by_dark_similarity_values(
            table,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
    )
    low_threshold_label = f"{float(low_threshold):g}"
    high_threshold_label = f"{float(high_threshold):g}"
    _plot_panel_b_overlap_boxplot(
        ax,
        (low_values, middle_values, high_values),
        labels=(
            f"Dark DPP\n<{low_threshold_label}",
            f"Dark DPP\n{low_threshold_label}-{high_threshold_label}",
            f"Dark DPP\n>={high_threshold_label}",
        ),
        colors=(
            PANEL_B_BOX_COLORS["low_dpp"],
            PANEL_B_BOX_COLORS["mid_dpp"],
            PANEL_B_BOX_COLORS["high_dpp"],
        ),
        title="Grouped by dark DPP",
        ylabel="Light DPP\noverlap",
    )


def plot_panel_b_dpp_overlap_scatter(
    ax: Any,
    table: Any,
    *,
    title: str | None = "Dark vs light DPP",
) -> None:
    """Plot dark-vs-light DPP overlap with marginal distributions."""
    _plot_panel_b_overlap_scatter_with_marginals(ax, table, title=title)


def _add_panel_b_activity_light_dpp_annotation(
    ax: Any,
    inactive_values: np.ndarray,
    active_values: np.ndarray,
) -> None:
    """Annotate Fig2B light-DPP fractions above the reference line."""
    inactive = np.asarray(inactive_values, dtype=float)
    active = np.asarray(active_values, dtype=float)
    inactive = inactive[np.isfinite(inactive)]
    active = active[np.isfinite(active)]
    if not inactive.size and not active.size:
        return
    inactive_text = (
        f"inactive {np.mean(inactive > 0.5):.0%} > 0.5"
        if inactive.size
        else "inactive n/a"
    )
    active_text = (
        f"active {np.mean(active > 0.5):.0%} > 0.5"
        if active.size
        else "active n/a"
    )
    ax.text(
        0.97,
        0.04,
        inactive_text + "\n" + active_text,
        ha="right",
        va="bottom",
        fontsize=4.3,
        transform=ax.transAxes,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 0.6,
        },
    )


def plot_panel_b_light_similarity_by_dark_similarity(
    ax: Any,
    table: Any,
    *,
    dark_activity_table: Any | None = None,
    low_threshold: float = PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
    high_threshold: float = PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
) -> None:
    """Plot Fig3B light correlations split by dark activity and dark DPP."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if dark_activity_table is None:
        dpp_ax = ax.inset_axes(PANEL_B_SINGLE_LIGHT_DPP_AXIS_BOUNDS)
    else:
        activity_ax = ax.inset_axes(PANEL_B_ACTIVITY_DPP_AXIS_BOUNDS)
        inactive_values, active_values = _get_panel_b_light_dpp_by_dark_activity_values(
            dark_activity_table
        )
        _plot_panel_b_violin(
            activity_ax,
            (inactive_values, active_values),
            labels=(
                f"Dark-inactive\n<{PANEL_B_DARK_ACTIVITY_THRESHOLD_HZ:g} Hz",
                f"Dark active\n>={PANEL_B_DARK_ACTIVITY_THRESHOLD_HZ:g} Hz",
            ),
            colors=(
                PANEL_B_ACTIVITY_COLORS["inactive"],
                PANEL_B_ACTIVITY_COLORS["active"],
            ),
            title="Stable light cells",
            ylabel="Light DPP\ncorr.",
        )
        _add_panel_b_activity_light_dpp_annotation(
            activity_ax,
            inactive_values,
            active_values,
        )
        dpp_ax = ax.inset_axes(PANEL_B_LIGHT_DPP_AXIS_BOUNDS)

    low_threshold_label = f"{float(low_threshold):g}"
    high_threshold_label = f"{float(high_threshold):g}"
    if dark_activity_table is None:
        (
            low_values,
            middle_values,
            high_values,
        ) = _get_panel_b_light_similarity_by_dark_similarity_values(
            table,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        dpp_labels = (
            f"Dark DPP\n<{low_threshold_label}",
            f"Dark DPP\n{low_threshold_label}-{high_threshold_label}",
            f"Dark DPP\n>={high_threshold_label}",
        )
        dpp_title = "Grouped by dark DPP"
        _plot_panel_b_boxplot(
            dpp_ax,
            (low_values, middle_values, high_values),
            labels=dpp_labels,
            colors=(
                PANEL_B_BOX_COLORS["low_dpp"],
                PANEL_B_BOX_COLORS["mid_dpp"],
                PANEL_B_BOX_COLORS["high_dpp"],
            ),
            title=dpp_title,
            ylabel="Light DPP\ncorr.",
        )
        if low_values.size or middle_values.size or high_values.size:
            low_summary = (
                f"low {np.mean(low_values > 0.5):.0%} > 0.5"
                if low_values.size
                else "low n/a"
            )
            middle_summary = (
                f"mid {np.mean(middle_values > 0.5):.0%} > 0.5"
                if middle_values.size
                else "mid n/a"
            )
            high_summary = (
                f"high {np.mean(high_values > 0.5):.0%} > 0.5"
                if high_values.size
                else "high n/a"
            )
            dpp_ax.text(
                0.98,
                0.04,
                low_summary + "\n" + middle_summary + "\n" + high_summary,
                ha="right",
                va="bottom",
                fontsize=4.5,
                transform=dpp_ax.transAxes,
            )
    else:
        low_values, high_values = _get_panel_b_light_dpp_by_dark_dpp_threshold_values(
            table,
            threshold=low_threshold,
            dark_active_only=True,
        )
        dpp_labels = (
            f"Dark DPP\n<{low_threshold_label}",
            f"Dark DPP\n>{low_threshold_label}",
        )
        _plot_panel_b_violin(
            dpp_ax,
            (low_values, high_values),
            labels=dpp_labels,
            colors=(
                PANEL_B_BOX_COLORS["low_dpp"],
                PANEL_B_BOX_COLORS["high_dpp"],
            ),
            title="Dark active cells",
            ylabel=None,
        )
        if low_values.size or high_values.size:
            low_summary = (
                f"low {np.mean(low_values > 0.5):.0%} > 0.5"
                if low_values.size
                else "low n/a"
            )
            high_summary = (
                f"high {np.mean(high_values > 0.5):.0%} > 0.5"
                if high_values.size
                else "high n/a"
            )
            dpp_ax.text(
                0.98,
                0.04,
                low_summary + "\n" + high_summary,
                ha="right",
                va="bottom",
                fontsize=4.5,
                transform=dpp_ax.transAxes,
            )


def plot_panel_a_examples_row(
    ax: Any,
    examples: Sequence[dict[str, Any]],
    *,
    similarity_annotation: str = "correlation",
    show_similarity_annotation: bool = True,
) -> None:
    """Plot the moved Figure 3A examples and added Figure 2A examples."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(
            0.5,
            0.5,
            "No examples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    column_gap = PANEL_A_EXAMPLE_COLUMN_GAP
    row_gap = PANEL_A_EXAMPLE_ROW_GAP
    column_count = min(2, len(examples))
    row_count = (len(examples) + column_count - 1) // column_count
    column_width = (1.0 - column_gap * (column_count - 1)) / column_count
    row_height = (1.0 - row_gap * (row_count - 1)) / row_count
    for example_index, example in enumerate(examples, start=1):
        row_index = (example_index - 1) // column_count
        column_index = (example_index - 1) % column_count
        left = column_index * (column_width + column_gap)
        bottom = 1.0 - (row_index + 1) * row_height - row_index * row_gap
        example_ax = ax.inset_axes([left, bottom, column_width, row_height])
        correlation_text_options = (
            PANEL_A_EXAMPLE_CORRELATION_TEXT_OVERRIDES.get(example_index, {})
        )
        plot_kwargs = dict(
            title=None,
            show_correlation=show_similarity_annotation,
            **correlation_text_options,
        )
        y_max_override = PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(example_index)
        if y_max_override is not None:
            plot_kwargs["y_max"] = y_max_override
        if similarity_annotation != "correlation":
            plot_kwargs["similarity_annotation"] = similarity_annotation
        plot_panel_a_example(example_ax, example, **plot_kwargs)
        example_ax.text(
            0.5,
            0.985,
            f"Example cell {example_index}",
            ha="center",
            va="top",
            fontsize=5.3,
            transform=example_ax.transAxes,
        )


def make_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    dark_tuning_correlation_threshold: float = (
        PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
    ),
    high_dark_tuning_correlation_threshold: float = (
        PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
    ),
) -> Path:
    """Build and save Figure 2."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_glm_payload = load_panel_glm_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        swap_delta_min_tuning_stability_correlation=(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        swap_model_name=PANEL_C_SWAP_MODEL_NAME,
        swap_example_count=len(PANEL_C_SWAP_EXAMPLES),
        swap_requested_examples=PANEL_C_SWAP_EXAMPLES,
        dark_light_requested_examples=PANEL_C_DARK_LIGHT_EXAMPLES,
    )
    panel_a_examples = [
        load_panel_a_example_data(
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
            FIGURE_2_PANEL_A_EXAMPLES
        )
    ]
    panel_b_table = load_panel_b_tuning_correlation_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_b_activity_table = load_panel_b_dark_activity_light_dpp_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        dark_movement_fr_cache_dir=panel_example_cache_dir,
    )
    panel_b_stable_light_paired_table = build_panel_b_stable_light_paired_dpp_table(
        panel_b_table,
        panel_b_activity_table,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(**FIGURE_2_CONSTRAINED_LAYOUT_PADS)
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[
            PANEL_A_EXAMPLE_ROW_HEIGHT_MM,
            PANEL_BC_ROW_HEIGHT_MM,
        ],
    )
    top_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_AB_WIDTH_RATIOS,
        wspace=PANEL_AB_WSPACE,
    )
    panel_a_axis = fig.add_subplot(top_grid[0, 0])
    panel_b_axis = fig.add_subplot(top_grid[0, 1])
    glm_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_GH_WIDTH_RATIOS,
    )
    panel_c_axis = fig.add_subplot(glm_grid[0, 0])
    panel_d_axis = fig.add_subplot(glm_grid[0, 1])

    plot_panel_a_examples_row(panel_a_axis, panel_a_examples)
    plot_panel_b_light_similarity_by_dark_similarity(
        panel_b_axis,
        panel_b_stable_light_paired_table,
        dark_activity_table=panel_b_activity_table,
        low_threshold=dark_tuning_correlation_threshold,
        high_threshold=high_dark_tuning_correlation_threshold,
    )
    plot_panel_g_model_architecture(
        panel_c_axis,
        panel_glm_payload["dark_light_examples"],
        independent_track_center_y=PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y,
        shared_track_center_y=PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y,
        schematic_height_fraction=PANEL_B_SCHEMATIC_HEIGHT_FRACTION,
        schematic_track_size=PANEL_B_SCHEMATIC_TRACK_SIZE,
        independent_basis_icon_scale=PANEL_B_INDEPENDENT_BASIS_ICON_SCALE,
        independent_basis_label=PANEL_B_INDEPENDENT_BASIS_LABEL,
        show_dark_track_labels=True,
        field_label_y=PANEL_B_FIELD_LABEL_Y,
        model_label_x=PANEL_B_MODEL_LABEL_X,
        model_label_fontsize=PANEL_B_MODEL_LABEL_FONTSIZE,
        shared_model_label="Segment scalar\nmodel",
        component_label_fontsize=PANEL_B_COMPONENT_LABEL_FONTSIZE,
        segment_modulation_label_y=PANEL_B_SEGMENT_MODULATION_LABEL_Y,
        segment_modulation_label=PANEL_B_SEGMENT_MODULATION_LABEL,
        example_axis_bounds=PANEL_B_EXAMPLE_AXIS_BOUNDS,
        example_field_y=PANEL_B_EXAMPLE_FIELD_Y,
        example_field_height=PANEL_B_EXAMPLE_FIELD_HEIGHT,
        example_icon_bounds=PANEL_B_EXAMPLE_ICON_BOUNDS,
        example_xlabel_y=PANEL_B_EXAMPLE_XLABEL_Y,
        example_column_width=PANEL_B_EXAMPLE_COLUMN_WIDTH,
        example_column_gap=PANEL_B_EXAMPLE_COLUMN_GAP,
        example_plot_left_offset=PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET,
        example_field_width=PANEL_B_EXAMPLE_FIELD_WIDTH,
        example_field_gap=PANEL_B_EXAMPLE_FIELD_GAP,
        example_layout=PANEL_B_EXAMPLE_LAYOUT,
        example_row_height=PANEL_B_EXAMPLE_ROW_HEIGHT,
        example_row_gap=PANEL_B_EXAMPLE_ROW_GAP,
        model_colors=PANEL_B_EXAMPLE_MODEL_COLORS,
        model_labels=PANEL_B_EXAMPLE_MODEL_LABELS,
    )
    plot_panel_h_swap_delta(
        panel_d_axis,
        panel_glm_payload["swap_delta"],
        panel_glm_payload["swap_examples"],
        model_name=PANEL_C_SWAP_MODEL_NAME,
        model_colors=PANEL_C_SWAP_MODEL_COLORS,
        model_labels=PANEL_C_SWAP_MODEL_LABELS,
        schematic_axis_bounds=PANEL_C_SCHEMATIC_AXIS_BOUNDS,
        delta_axis_bounds=PANEL_C_DELTA_AXIS_BOUNDS,
        example_axis_bounds=PANEL_C_EXAMPLE_AXIS_BOUNDS,
        schematic_track_size=PANEL_C_SCHEMATIC_TRACK_SIZE,
        show_dark_track_labels=True,
        show_model_labels=False,
        prediction_label_fontsize=PANEL_C_PREDICTION_LABEL_FONTSIZE,
        independent_track_center_y=PANEL_C_INDEPENDENT_TRACK_CENTER_Y,
        independent_prediction_label_y=PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y,
        segment_modulation_track_center_y=PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y,
        shared_dark_track_center_y=PANEL_C_SHARED_DARK_TRACK_CENTER_Y,
        shared_light_track_center_y=PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y,
        shared_prediction_label_y=PANEL_C_SHARED_PREDICTION_LABEL_Y,
        delta_grid_bounds=PANEL_C_DELTA_GRID_BOUNDS,
        delta_xlabel_y=PANEL_C_DELTA_XLABEL_Y,
        example_delta_label_positions=PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS,
        example_delta_label_vertical_alignments=(
            PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS
        ),
        example_icon_bounds=PANEL_C_EXAMPLE_ICON_BOUNDS,
    )

    label_axis(panel_a_axis, "A", x=-0.02, y=PANEL_A_LABEL_Y)
    panel_a_label = panel_a_axis.texts[-1]
    panel_a_axis.set_title(
        "Example DPP cells in dark and light",
        fontsize=8,
        pad=PANEL_A_TITLE_PAD,
    )
    label_axis(panel_b_axis, "B", x=-0.035, y=PANEL_B_LABEL_Y)
    panel_b_label = panel_b_axis.texts[-1]
    label_axis(panel_c_axis, "C", x=-0.035, y=PANEL_BC_LABEL_Y)
    panel_c_label = panel_c_axis.texts[-1]
    label_axis(panel_d_axis, "D", x=-0.035, y=PANEL_BC_LABEL_Y)
    panel_d_label = panel_d_axis.texts[-1]
    panel_b_axis.set_title(
        "Light DPP across dark-defined groups",
        fontsize=8,
        pad=PANEL_B_TITLE_PAD,
    )
    panel_c_title = panel_c_axis.set_title(
        "Two models that relate dark and light activity",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )
    panel_d_title = panel_d_axis.set_title(
        "Predicting activity in held-out light epoch",
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )

    fig.canvas.draw()
    _shift_axis_horizontally(panel_d_axis, PANEL_C_HORIZONTAL_SHIFT)
    fig.canvas.draw()
    _align_texts_to_reference_display_y((panel_c_label, panel_d_label))
    _align_texts_to_reference_display_y((panel_c_title, panel_d_title))
    _align_text_to_reference_display_x(panel_c_label, panel_a_label)
    _align_texts_to_reference_display_y((panel_c_title, panel_c_label))
    _align_texts_to_reference_display_y((panel_c_label, panel_d_label))
    fig.set_layout_engine(None)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 2 generation."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 2 dark-light example and GLM panels."
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
        default=PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Dark tuning-correlation threshold for Panel B low/high grouping. "
            f"Default: {PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--high-dark-tuning-correlation-threshold",
        type=float,
        default=PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Upper dark tuning-correlation threshold for Panel B high group. "
            f"Default: {PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_id,
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
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument(
        "--light-epoch",
        default=None,
        help=(
            "Light run epoch for GLM panels. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help=(
            "Dark run epoch. "
            f"Default: registry value, currently {DEFAULT_DARK_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {DEFAULT_POSITION_BIN_COUNT}"
        ),
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
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
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
    )


if __name__ == "__main__":
    main()
