from __future__ import annotations

"""Generate Supplementary Figure 3 heatmap and per-animal Figure 3C-F panels."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
    HEATMAP_COLORBAR_LABELPAD,
    HEATMAP_COLORBAR_LABEL_FONTSIZE,
    PANEL_D_HEATMAP_CMAP,
    align_panel_values_to_unit_order,
    build_normalized_position_bins,
    build_unit_keys,
    compute_unit_movement_firing_rates,
    extract_tuning_curve_arrays,
    get_stability_table_path,
    load_or_compute_panel_d_heatmap_payload,
    normalize_linear_position_by_trajectory,
    normalize_panel_values_per_trajectory,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_3_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_3_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PANEL_BC_HEIGHT_MM,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    FIGURE_FORMATS,
    PANEL_B_LINEAR_POSITION_ORIENTATION,
    PANEL_B_TRAJECTORY_TYPES,
    PANEL_DEF_WIDTH_RATIOS,
    PANEL_C_COLORBAR_PAD,
    PANEL_C_NEURON_SCALE_BAR_X,
    add_centered_axis_text,
    add_segment_boundary_lines,
    build_output_path,
    draw_neuron_scale_bar,
    get_dark_epoch,
    get_light_epoch,
    load_panel_d_similarity_table,
    load_panel_e_encoding_delta_table,
    load_panel_f_decoding_error_table,
    parse_dataset_id,
    plot_panel_d_similarity,
    plot_panel_e_encoding_delta_histogram,
    plot_panel_f_decoding_error,
    plot_pooled_heatmap_grid,
    setup_light_heatmap_panel,
)
from v1ca1.raster.plot_place_field_heatmap import (
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    prepare_heatmap_session,
    smooth_tuning_curve_nan_aware,
)
from v1ca1.paper_figures.style import (
    COMPACT_HISTOGRAM_KWARGS,
    NEUTRAL_COLORS,
    TRAJECTORY_COLORS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "supplementary_figure_3"
DEFAULT_FIGURE_WIDTH_MM = FIGURE_3_WIDTH_MM
DEFAULT_SECTION_SPACER_MM = 3.0
DEFAULT_BOTTOM_SECTION_SPACER_MM = 14.0
DEFAULT_REORDERED_HEATMAP_HEIGHT_MM = DEFAULT_PANEL_BC_HEIGHT_MM
DEFAULT_PER_ANIMAL_GRID_HEIGHT_MM = FIGURE_3_HEIGHT_MM
DEFAULT_DARK_LIGHT_CORRELATION_HEIGHT_MM = 22.0
DEFAULT_STABILITY_OVERLAY_HEIGHT_MM = 22.0
DEFAULT_FIGURE_HEIGHT_MM = (
    DEFAULT_REORDERED_HEATMAP_HEIGHT_MM
    + DEFAULT_SECTION_SPACER_MM
    + DEFAULT_PER_ANIMAL_GRID_HEIGHT_MM
    + DEFAULT_BOTTOM_SECTION_SPACER_MM
    + DEFAULT_DARK_LIGHT_CORRELATION_HEIGHT_MM
    + DEFAULT_STABILITY_OVERLAY_HEIGHT_MM
)
ANIMAL_ROW_LABEL_FONTSIZE = 5.2
PER_ANIMAL_COLUMN_WIDTH_RATIOS = (
    PANEL_DEF_WIDTH_RATIOS[0],
    PANEL_DEF_WIDTH_RATIOS[0],
    PANEL_DEF_WIDTH_RATIOS[1],
    PANEL_DEF_WIDTH_RATIOS[2],
)
PER_ANIMAL_GRID_HSPACE = 0.22
PER_ANIMAL_GRID_WSPACE = 0.20
PANEL_TITLE_FONTSIZE = 8.0
STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION = 0.5
DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ = 0.5
DARK_LIGHT_CORRELATION_BINS = np.linspace(-1.0, 1.0, 21)
DARK_LIGHT_CORRELATION_GRID_WSPACE = 0.24
DARK_LIGHT_CORRELATION_TUNING_CURVE_RELATIVE_DIR = (
    Path("task_progression") / "compute_tuning_curves"
)
DARK_LIGHT_CORRELATION_RATE_RELATIVE_DIR = (
    Path("task_progression") / "dark_light_glm" / "selected"
)
DARK_LIGHT_CORRELATION_RATE_MODEL = "visual"
DARK_LIGHT_CORRELATION_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_epoch",
    "trajectory_type",
    "unit",
    "dark_movement_firing_rate_hz",
    "light_movement_firing_rate_hz",
    "correlation",
)
LIGHT_TUNING_STABILITY_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "light_epoch",
    "trajectory_type",
    "unit",
    "stability_correlation",
)
PANEL_A_SCATTER_ALPHA = 0.30
PANEL_A_GRID_LEFT = 0.045
PANEL_A_GRID_RIGHT = 0.965
PANEL_A_GRID_TOP = 0.965
PANEL_A_GRID_BOTTOM = 0.055
PANEL_A_FIGURE_1D_WIDTH_CORRECTION = 0.995515695
PANEL_A_HEATMAP_WIDTH_FRACTION = (
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION * PANEL_A_FIGURE_1D_WIDTH_CORRECTION
)
PANEL_A_HEATMAP_SIDE_SPACER_FRACTION = (
    (PANEL_A_GRID_RIGHT - PANEL_A_GRID_LEFT - PANEL_A_HEATMAP_WIDTH_FRACTION) / 2.0
)
REORDERED_HEATMAP_TITLE = "Fig. 1D cells in light"
REORDERED_HEATMAP_CMAP = PANEL_D_HEATMAP_CMAP
REORDERED_HEATMAP_VMAX = 1.0
REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION = 0.5
PANEL_A_FIGURE_1D_ORDER_MODE = "figure_1d_order"
PANEL_A_ORDER_MODES = (
    PANEL_A_FIGURE_1D_ORDER_MODE,
)
PANEL_A_CACHE_PREFIX = "supplementary_figure_3_panel_a"
PANEL_A_CACHE_VERSION = 2
PANEL_A_CACHE_METADATA_KEY = "__metadata__"
PANEL_A_CACHE_DATASET_TOKEN_LIMIT = 120


def group_datasets_by_animal(datasets: Sequence[DatasetId]) -> dict[str, list[DatasetId]]:
    """Return normalized data sets grouped by animal in input order."""
    grouped: dict[str, list[DatasetId]] = {}
    for dataset in datasets:
        normalized = normalize_dataset_id(dataset)
        animal_name = str(normalized[0])
        grouped.setdefault(animal_name, []).append(normalized)
    return grouped


def format_animal_row_label(animal_name: str, datasets: Sequence[DatasetId]) -> str:
    """Return a compact label for one per-animal row."""
    dates = []
    for dataset in datasets:
        _animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        if str(date) not in dates:
            dates.append(str(date))
    if not dates:
        return str(animal_name)
    return f"{animal_name}\n{', '.join(dates)}"


def hide_x_axis_labels(ax: object) -> None:
    """Hide x-axis label text and tick labels for repeated row panels."""
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelbottom=False)


def set_panel_a_dot_alpha(ax: object) -> None:
    """Make the per-animal Figure 3C scatter points more visible."""
    for collection in ax.collections:
        collection.set_alpha(PANEL_A_SCATTER_ALPHA)


def _require_table_columns(table: Any, path: Path, columns: Sequence[str]) -> None:
    """Validate that a loaded table has the required columns."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Table {path} is missing columns {missing!r}.")


def load_epoch_stable_units_by_tuning_stability(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_correlation: float = STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION,
) -> set[int]:
    """Return units with odd/even stability above threshold in any trajectory."""
    if min_correlation < -1.0:
        raise ValueError("min_correlation must be at least -1.")

    import pandas as pd

    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(table_path)
    _require_table_columns(
        table,
        table_path,
        ("unit", "region", "epoch", "trajectory_type", "stability_correlation"),
    )
    correlations = pd.to_numeric(table["stability_correlation"], errors="coerce")
    stable_rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & (table["trajectory_type"].astype(str).isin(PANEL_B_TRAJECTORY_TYPES))
        & np.isfinite(correlations.to_numpy(dtype=float))
        & (correlations.to_numpy(dtype=float) > float(min_correlation))
    ]
    return set(
        pd.to_numeric(stable_rows["unit"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )


def filter_panel_d_similarity_table_by_tuning_stability(
    similarity_table: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_correlation: float = STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION,
) -> Any:
    """Keep similarity rows for units stable in at least one trajectory in both epochs."""
    import pandas as pd

    required_columns = ("animal_name", "date", "unit")
    _require_table_columns(similarity_table, Path("similarity_table"), required_columns)
    stable_units_by_session: dict[tuple[str, str], set[int]] = {}
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        selected_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        light_stable_units = load_epoch_stable_units_by_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_light_epoch,
            min_correlation=min_correlation,
        )
        dark_stable_units = load_epoch_stable_units_by_tuning_stability(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_dark_epoch,
            min_correlation=min_correlation,
        )
        stable_units_by_session[(str(animal_name), str(date))] = (
            light_stable_units & dark_stable_units
        )

    table = similarity_table.copy()
    unit_values = pd.to_numeric(table["unit"], errors="coerce")
    keep_mask = []
    for animal_name, date, unit in zip(
        table["animal_name"].astype(str),
        table["date"].astype(str),
        unit_values,
        strict=True,
    ):
        keep_mask.append(
            np.isfinite(unit)
            and int(unit) in stable_units_by_session.get((animal_name, date), set())
        )
    return table.loc[np.asarray(keep_mask, dtype=bool)].copy()


def concatenate_unit_parts(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate unit-key chunks for pooled heatmap ordering."""
    if not parts:
        return np.asarray([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def concatenate_value_parts(
    parts: list[np.ndarray],
    position_bin_count: int,
) -> np.ndarray:
    """Concatenate tuning-curve chunks for one pooled heatmap panel."""
    if not parts:
        return np.empty((0, position_bin_count), dtype=float)
    return np.vstack(parts)


def collect_curve_arrays(
    curve_sets: Sequence[dict[str, Any]],
    *,
    curve_key: str,
    trajectory_type: str,
    position_bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled unit keys and tuning values for one trajectory."""
    unit_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for curve_set in curve_sets:
        curve = curve_set[curve_key].get(trajectory_type)
        if curve is None:
            continue
        units, values = extract_tuning_curve_arrays(curve)
        unit_parts.append(
            build_unit_keys(
                animal_name=str(curve_set["animal_name"]),
                date=str(curve_set["date"]),
                region=str(curve_set["region"]),
                units=units,
            )
        )
        value_parts.append(values)
    return (
        concatenate_unit_parts(unit_parts),
        concatenate_value_parts(value_parts, position_bin_count),
    )


def compute_epoch_all_trial_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    epoch: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
) -> dict[str, Any]:
    """Compute all-trial movement tuning curves and movement firing rates."""
    session = prepare_heatmap_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=(region,),
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )
    selected_epoch = session["run_epochs"][0]
    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        session["position_by_epoch"][selected_epoch],
        session["timestamps_position"][selected_epoch],
        session["trajectory_intervals"][selected_epoch],
        position_offset=position_offset,
        use_trajectory_direction=use_trajectory_direction,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    bin_edges = build_normalized_position_bins(position_bin_count)
    curves = {}
    for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
        epochs = session["trajectory_intervals"][selected_epoch][
            trajectory_type
        ].intersect(session["movement_by_run"][selected_epoch])
        curves[trajectory_type] = compute_place_tuning_curve(
            session["spikes_by_region"][region],
            normalized_position_by_trajectory[trajectory_type],
            epochs,
            bin_edges=bin_edges,
            sigma_bins=sigma_bins,
        )
    movement_firing_rates = compute_unit_movement_firing_rates(
        session["spikes_by_region"][region],
        session["movement_by_run"][selected_epoch],
    )
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": selected_epoch,
        "all_curves": curves,
        "movement_firing_rates_hz": movement_firing_rates,
    }


def compute_light_epoch_all_trial_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
) -> dict[str, Any]:
    """Compute all-trial light movement tuning curves."""
    return compute_epoch_all_trial_tuning_curves(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        region=region,
        epoch=get_light_epoch(animal_name, date, light_epoch),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        use_trajectory_direction=use_trajectory_direction,
    )


def compute_dark_epoch_all_trial_tuning_curves(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    region: str,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    use_trajectory_direction: bool = False,
) -> dict[str, Any]:
    """Compute all-trial dark movement tuning curves."""
    return compute_epoch_all_trial_tuning_curves(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        region=region,
        epoch=get_dark_epoch(animal_name, date, dark_epoch),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        use_trajectory_direction=use_trajectory_direction,
    )


def get_saved_task_progression_tuning_curve_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
) -> Path:
    """Return one saved trajectory tuning-curve artifact path."""
    return (
        Path(data_root)
        / animal_name
        / date
        / DARK_LIGHT_CORRELATION_TUNING_CURVE_RELATIVE_DIR
        / f"{region}_{epoch}_place_{trajectory_type}_tuning_curves.nc"
    )


def get_dark_light_movement_rate_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_name: str = DARK_LIGHT_CORRELATION_RATE_MODEL,
) -> Path:
    """Return the selected dark/light artifact that stores movement rates."""
    return (
        Path(data_root)
        / animal_name
        / date
        / DARK_LIGHT_CORRELATION_RATE_RELATIVE_DIR
        / f"{region}_{light_epoch}_vs_{dark_epoch}_{model_name}_selected.nc"
    )


def load_saved_epoch_tuning_curve_set(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    sigma_bins: float,
) -> dict[str, Any]:
    """Load saved empirical trajectory tuning curves for one epoch."""
    import xarray as xr

    curves = {}
    for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
        path = get_saved_task_progression_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory_type=trajectory_type,
        )
        if not path.exists():
            raise FileNotFoundError(
                "Missing saved task-progression tuning curve. Expected "
                f"{path}. Run `python -m v1ca1.task_progression.compute_tuning_curves` "
                "for this session first."
            )
        dataset = xr.open_dataset(path)
        try:
            if "firing_rate_hz" not in dataset:
                raise ValueError(f"Tuning curve artifact {path} lacks firing_rate_hz.")
            curve = dataset["firing_rate_hz"].load()
            if float(sigma_bins) > 0.0:
                curve = smooth_tuning_curve_nan_aware(
                    curve,
                    sigma_bins=float(sigma_bins),
                )
            curves[trajectory_type] = curve
        finally:
            dataset.close()
    return {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
        "all_curves": curves,
    }


def saved_epoch_tuning_curve_set_exists(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> bool:
    """Return whether all saved trajectory tuning curves exist for one epoch."""
    return all(
        get_saved_task_progression_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory_type=trajectory_type,
        ).exists()
        for trajectory_type in PANEL_B_TRAJECTORY_TYPES
    )


def load_dark_light_movement_firing_rates(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
) -> tuple[dict[int, float], dict[int, float]]:
    """Load saved dark and light movement firing rates keyed by unit."""
    import xarray as xr

    path = get_dark_light_movement_rate_path(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing selected dark/light model artifact with movement firing rates. "
            f"Expected {path}. Run `python -m v1ca1.task_progression.dark_light_glm` "
            "for this session first."
        )
    dataset = xr.open_dataset(path)
    try:
        for variable in (
            "dark_movement_firing_rate_hz",
            "light_movement_firing_rate_hz",
        ):
            if variable not in dataset:
                raise ValueError(f"Dark/light artifact {path} lacks {variable}.")
        units = np.asarray(dataset.coords["unit"].values, dtype=int)
        dark_rates = np.asarray(dataset["dark_movement_firing_rate_hz"].values, dtype=float)
        light_rates = np.asarray(
            dataset["light_movement_firing_rate_hz"].values,
            dtype=float,
        )
    finally:
        dataset.close()
    return (
        {int(unit): float(rate) for unit, rate in zip(units, dark_rates, strict=True)},
        {int(unit): float(rate) for unit, rate in zip(units, light_rates, strict=True)},
    )


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return weights that normalize one histogram to a fraction of cells."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def compute_tuning_curve_correlation(
    dark_values: np.ndarray,
    light_values: np.ndarray,
) -> float:
    """Return Pearson correlation between matched dark and light tuning curves."""
    dark_values = np.asarray(dark_values, dtype=float).reshape(-1)
    light_values = np.asarray(light_values, dtype=float).reshape(-1)
    if dark_values.shape != light_values.shape:
        raise ValueError(
            "Dark and light tuning curves must have matching shapes. "
            f"Got {dark_values.shape} and {light_values.shape}."
        )
    finite_mask = np.isfinite(dark_values) & np.isfinite(light_values)
    if int(np.count_nonzero(finite_mask)) < 2:
        return float("nan")
    dark_finite = dark_values[finite_mask]
    light_finite = light_values[finite_mask]
    if np.nanstd(dark_finite) <= 0.0 or np.nanstd(light_finite) <= 0.0:
        return float("nan")
    return float(np.corrcoef(dark_finite, light_finite)[0, 1])


def _movement_rate_lookup(movement_firing_rates: dict[Any, float]) -> dict[int, float]:
    """Return movement firing rates keyed by integer unit id."""
    return {
        int(unit_id): float(rate)
        for unit_id, rate in movement_firing_rates.items()
    }


def build_dark_light_tuning_correlation_table(
    dark_curve_sets: Sequence[dict[str, Any]],
    light_curve_sets: Sequence[dict[str, Any]],
    *,
    min_movement_firing_rate_hz: float = (
        DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return per-unit dark/light tuning-curve correlations by trajectory."""
    if min_movement_firing_rate_hz < 0.0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")

    import pandas as pd

    rows: list[dict[str, Any]] = []
    for dark_set, light_set in zip(dark_curve_sets, light_curve_sets, strict=True):
        dark_rates = _movement_rate_lookup(dark_set["movement_firing_rates_hz"])
        light_rates = _movement_rate_lookup(light_set["movement_firing_rates_hz"])
        for trajectory_type in PANEL_B_TRAJECTORY_TYPES:
            dark_curve = dark_set["all_curves"].get(trajectory_type)
            light_curve = light_set["all_curves"].get(trajectory_type)
            if dark_curve is None or light_curve is None:
                continue
            dark_units, dark_values = extract_tuning_curve_arrays(dark_curve)
            light_units, light_values = extract_tuning_curve_arrays(light_curve)
            dark_rows = {int(unit): index for index, unit in enumerate(dark_units)}
            light_rows = {int(unit): index for index, unit in enumerate(light_units)}
            for unit_id in sorted(set(dark_rows).intersection(light_rows)):
                dark_rate = dark_rates.get(unit_id, 0.0)
                light_rate = light_rates.get(unit_id, 0.0)
                if (
                    dark_rate < float(min_movement_firing_rate_hz)
                    or light_rate < float(min_movement_firing_rate_hz)
                ):
                    continue
                correlation = compute_tuning_curve_correlation(
                    dark_values[dark_rows[unit_id]],
                    light_values[light_rows[unit_id]],
                )
                if not np.isfinite(correlation):
                    continue
                rows.append(
                    {
                        "animal_name": str(dark_set["animal_name"]),
                        "date": str(dark_set["date"]),
                        "region": str(dark_set["region"]),
                        "dark_epoch": str(dark_set["epoch"]),
                        "light_epoch": str(light_set["epoch"]),
                        "trajectory_type": trajectory_type,
                        "unit": int(unit_id),
                        "dark_movement_firing_rate_hz": float(dark_rate),
                        "light_movement_firing_rate_hz": float(light_rate),
                        "correlation": float(correlation),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=DARK_LIGHT_CORRELATION_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=DARK_LIGHT_CORRELATION_TABLE_COLUMNS)


def load_dark_light_tuning_correlation_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    min_movement_firing_rate_hz: float = (
        DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Compute pooled dark/light tuning-curve correlations for V1 cells."""

    dark_curve_sets = []
    light_curve_sets = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        selected_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        dark_rates, light_rates = load_dark_light_movement_firing_rates(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=selected_light_epoch,
            dark_epoch=selected_dark_epoch,
        )
        if saved_epoch_tuning_curve_set_exists(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_dark_epoch,
        ) and saved_epoch_tuning_curve_set_exists(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_light_epoch,
        ):
            dark_curve_set = load_saved_epoch_tuning_curve_set(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=selected_dark_epoch,
                sigma_bins=sigma_bins,
            )
            light_curve_set = load_saved_epoch_tuning_curve_set(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=selected_light_epoch,
                sigma_bins=sigma_bins,
            )
        else:
            dark_curve_set = compute_dark_epoch_all_trial_tuning_curves(
                animal_name=animal_name,
                date=date,
                data_root=data_root,
                region=region,
                dark_epoch=selected_dark_epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                use_trajectory_direction=True,
            )
            light_curve_set = compute_light_epoch_all_trial_tuning_curves(
                animal_name=animal_name,
                date=date,
                data_root=data_root,
                region=region,
                light_epoch=selected_light_epoch,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                use_trajectory_direction=True,
            )
        dark_curve_set["movement_firing_rates_hz"] = dark_rates
        light_curve_set["movement_firing_rates_hz"] = light_rates
        dark_curve_sets.append(dark_curve_set)
        light_curve_sets.append(light_curve_set)
    return build_dark_light_tuning_correlation_table(
        dark_curve_sets,
        light_curve_sets,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
    )


def load_light_tuning_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
) -> Any:
    """Load pooled light-epoch odd/even tuning-stability rows by trajectory."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        table_path = get_stability_table_path(data_root, animal_name, date)
        if not table_path.exists():
            raise FileNotFoundError(
                "Missing task-progression stability table. Expected "
                f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
                "for this session first."
            )
        table = pd.read_parquet(table_path)
        _require_table_columns(
            table,
            table_path,
            ("unit", "region", "epoch", "trajectory_type", "stability_correlation"),
        )
        filtered = table[
            (table["region"].astype(str) == str(region))
            & (table["epoch"].astype(str) == str(selected_light_epoch))
            & (table["trajectory_type"].astype(str).isin(PANEL_B_TRAJECTORY_TYPES))
        ].copy()
        filtered["stability_correlation"] = pd.to_numeric(
            filtered["stability_correlation"],
            errors="coerce",
        )
        filtered["unit"] = pd.to_numeric(filtered["unit"], errors="coerce")
        filtered = filtered[
            np.isfinite(filtered["stability_correlation"].to_numpy(dtype=float))
            & np.isfinite(filtered["unit"].to_numpy(dtype=float))
        ].copy()
        if filtered.empty:
            continue
        filtered = filtered.assign(
            animal_name=animal_name,
            date=date,
            light_epoch=selected_light_epoch,
        )
        filtered["unit"] = filtered["unit"].astype(int)
        tables.append(
            filtered[
                [
                    "animal_name",
                    "date",
                    "region",
                    "light_epoch",
                    "trajectory_type",
                    "unit",
                    "stability_correlation",
                ]
            ]
        )

    if not tables:
        return pd.DataFrame(columns=LIGHT_TUNING_STABILITY_TABLE_COLUMNS)
    return pd.concat(tables, axis=0, ignore_index=True)


def _format_trajectory_label(trajectory_type: str) -> str:
    """Return compact trajectory labels for histogram titles."""
    labels = {
        "center_to_left": "C-L",
        "center_to_right": "C-R",
        "right_to_center": "R-C",
        "left_to_center": "L-C",
    }
    return labels.get(trajectory_type, trajectory_type)


def plot_dark_light_tuning_correlation_histograms(
    axes: Sequence["Axes"],
    correlation_table: Any,
) -> None:
    """Plot dark/light tuning-curve correlation histograms by trajectory."""
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    if axes_array.size != len(PANEL_B_TRAJECTORY_TYPES):
        raise ValueError(
            "Expected one axis per trajectory type. "
            f"Got {axes_array.size} axes for {len(PANEL_B_TRAJECTORY_TYPES)} trajectories."
        )

    for axis_index, (ax, trajectory_type) in enumerate(
        zip(axes_array, PANEL_B_TRAJECTORY_TYPES, strict=True)
    ):
        rows = correlation_table[
            correlation_table["trajectory_type"].astype(str) == trajectory_type
        ]
        values = np.asarray(rows["correlation"], dtype=float)
        values = values[np.isfinite(values)]
        ax.axvspan(
            -1.0,
            0.0,
            color=NEUTRAL_COLORS["dark_epoch_background"],
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
        if values.size:
            ax.hist(
                values,
                bins=DARK_LIGHT_CORRELATION_BINS,
                weights=_fraction_histogram_weights(values),
                color=TRAJECTORY_COLORS[trajectory_type],
                **COMPACT_HISTOGRAM_KWARGS,
                zorder=2,
            )
            summary = f"n = {values.size}\nmed. {np.median(values):.2f}"
        else:
            summary = "n = 0\nmed. n/a"
        ax.text(
            0.04,
            0.94,
            summary,
            ha="left",
            va="top",
            fontsize=5.2,
            transform=ax.transAxes,
            color="0.25",
        )
        ax.set_title(
            _format_trajectory_label(trajectory_type),
            fontsize=6.2,
            pad=1.5,
            color=TRAJECTORY_COLORS[trajectory_type],
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Dark-light tuning corr.", fontsize=6.2, labelpad=1.5)
        if axis_index == 0:
            ax.set_ylabel("Frac.", fontsize=6.4, labelpad=1.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.8, length=1.8, pad=1)


def plot_dark_light_with_light_stability_histograms(
    axes: Sequence["Axes"],
    correlation_table: Any,
    stability_table: Any,
) -> None:
    """Plot dark/light tuning similarity with light odd/even stability overlays."""
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    if axes_array.size != len(PANEL_B_TRAJECTORY_TYPES):
        raise ValueError(
            "Expected one axis per trajectory type. "
            f"Got {axes_array.size} axes for {len(PANEL_B_TRAJECTORY_TYPES)} trajectories."
        )

    for axis_index, (ax, trajectory_type) in enumerate(
        zip(axes_array, PANEL_B_TRAJECTORY_TYPES, strict=True)
    ):
        correlation_rows = correlation_table[
            correlation_table["trajectory_type"].astype(str) == trajectory_type
        ]
        correlation_values = np.asarray(correlation_rows["correlation"], dtype=float)
        correlation_values = correlation_values[np.isfinite(correlation_values)]
        stability_rows = stability_table[
            stability_table["trajectory_type"].astype(str) == trajectory_type
        ]
        stability_values = np.asarray(
            stability_rows["stability_correlation"],
            dtype=float,
        )
        stability_values = stability_values[np.isfinite(stability_values)]

        ax.axvspan(
            -1.0,
            0.0,
            color=NEUTRAL_COLORS["dark_epoch_background"],
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
        if correlation_values.size:
            ax.hist(
                correlation_values,
                bins=DARK_LIGHT_CORRELATION_BINS,
                weights=_fraction_histogram_weights(correlation_values),
                color=TRAJECTORY_COLORS[trajectory_type],
                label="Dark-light",
                **COMPACT_HISTOGRAM_KWARGS,
                zorder=2,
            )
        if stability_values.size:
            ax.hist(
                stability_values,
                bins=DARK_LIGHT_CORRELATION_BINS,
                weights=_fraction_histogram_weights(stability_values),
                histtype="step",
                color="black",
                linewidth=0.85,
                label="Light odd/even",
                zorder=3,
            )
        if axis_index == 0:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(
                    handles,
                    labels,
                    frameon=False,
                    fontsize=4.8,
                    loc="upper left",
                    handlelength=1.4,
                )
        ax.set_title(
            _format_trajectory_label(trajectory_type),
            fontsize=6.2,
            pad=1.5,
            color=TRAJECTORY_COLORS[trajectory_type],
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Correlation", fontsize=6.2, labelpad=1.5)
        if axis_index == 0:
            ax.set_ylabel("Frac.", fontsize=6.4, labelpad=1.5)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.8, length=1.8, pad=1)


def select_unit_keys_by_light_tuning_stability(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    min_stability_correlation: float = (
        REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION
    ),
) -> set[str]:
    """Return unit keys stable in at least one light-epoch trajectory."""
    if min_stability_correlation < -1.0:
        raise ValueError("min_stability_correlation must be at least -1.")

    import pandas as pd

    included_unit_keys: set[str] = set()
    for dataset in datasets:
        animal_name, date, _dark_epoch = normalize_dataset_id(dataset)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        table_path = get_stability_table_path(data_root, animal_name, date)
        if not table_path.exists():
            raise FileNotFoundError(
                "Missing task-progression stability table. Expected "
                f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
                "for this session first."
            )
        table = pd.read_parquet(table_path)
        table = table[
            (table["epoch"].astype(str) == str(selected_light_epoch))
            & (table["region"].astype(str) == str(region))
            & (table["trajectory_type"].astype(str).isin(PANEL_B_TRAJECTORY_TYPES))
        ]
        correlations = np.asarray(table["stability_correlation"], dtype=float)
        stable_rows = table[
            np.isfinite(correlations)
            & (correlations > float(min_stability_correlation))
        ]
        for unit_id in stable_rows["unit"].drop_duplicates().to_numpy():
            included_unit_keys.add(f"{animal_name}:{date}:{region}:{int(unit_id)}")
    return included_unit_keys


def filter_ordered_unit_keys_by_unit_set(
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    included_unit_keys: set[str],
) -> dict[str, np.ndarray]:
    """Return cached row unit keys restricted to an included unit-key set."""
    return {
        trajectory_type: np.asarray(
            [
                str(unit_key)
                for unit_key in ordered_unit_keys_by_trajectory[trajectory_type]
                if str(unit_key) in included_unit_keys
            ],
            dtype=object,
        )
        for trajectory_type in PANEL_B_TRAJECTORY_TYPES
    }


def build_figure_1d_ordered_light_panel_values(
    *,
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    light_curve_sets: Sequence[dict[str, Any]],
    position_bin_count: int,
    curve_key: str = "all_curves",
) -> dict[tuple[str, str], np.ndarray]:
    """Return light heatmaps aligned to a fixed unit-key row order."""
    panels: dict[tuple[str, str], np.ndarray] = {}
    for order_trajectory in PANEL_B_TRAJECTORY_TYPES:
        reference_units = np.asarray(
            ordered_unit_keys_by_trajectory[order_trajectory],
            dtype=object,
        )
        unit_order = np.arange(reference_units.size, dtype=int)
        sorted_values_by_plot_trajectory: dict[str, np.ndarray] = {}
        for plot_trajectory in PANEL_B_TRAJECTORY_TYPES:
            display_units, display_values = collect_curve_arrays(
                light_curve_sets,
                curve_key=curve_key,
                trajectory_type=plot_trajectory,
                position_bin_count=position_bin_count,
            )
            if reference_units.size == 0 or display_units.size == 0:
                sorted_values_by_plot_trajectory[plot_trajectory] = np.full(
                    (reference_units.size, position_bin_count),
                    np.nan,
                    dtype=float,
                )
                continue
            sorted_values_by_plot_trajectory[plot_trajectory] = (
                align_panel_values_to_unit_order(
                    display_values,
                    display_units,
                    reference_units,
                    unit_order,
                )
            )
        for plot_trajectory, values in sorted_values_by_plot_trajectory.items():
            panels[(order_trajectory, plot_trajectory)] = (
                normalize_panel_values_per_trajectory(values)
            )
    return panels


def _format_panel_a_cache_token(value: object) -> str:
    """Return a filesystem-safe token for Supplementary Figure 3A caches."""
    text = str(value).strip()
    cleaned = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            cleaned.append(character)
        elif character == ".":
            cleaned.append("p")
        else:
            cleaned.append("-")
    token = "".join(cleaned).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "none"


def _format_panel_a_cache_number(value: float | int) -> str:
    """Return a compact numeric token for Supplementary Figure 3A caches."""
    return _format_panel_a_cache_token(f"{float(value):g}")


def _build_panel_a_dataset_cache_token(
    dataset_metadata: Sequence[dict[str, str]],
) -> str:
    """Return a descriptive cache token for the Supplementary Figure 3A data sets."""
    dataset_tokens = [
        _format_panel_a_cache_token(
            f"{dataset['animal_name']}-{dataset['date']}-"
            f"{dataset['dark_epoch']}-{dataset['light_epoch']}"
        )
        for dataset in dataset_metadata
    ]
    token = "_".join(dataset_tokens) or "none"
    if len(token) <= PANEL_A_CACHE_DATASET_TOKEN_LIMIT:
        return token

    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    prefix = "_".join(dataset_tokens[:2])
    return _format_panel_a_cache_token(
        f"{prefix}_{len(dataset_tokens)}datasets_{digest}"
    )


def build_panel_a_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    order_mode: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Return metadata identifying one Supplementary Figure 3A heatmap cache."""
    if order_mode not in PANEL_A_ORDER_MODES:
        raise ValueError(f"Unknown panel A order_mode {order_mode!r}.")
    dataset_metadata = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_metadata.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": dark_epoch if dark_epoch is not None else dataset_dark_epoch,
                "light_epoch": get_light_epoch(animal_name, date, light_epoch),
            }
        )
    return {
        "cache_version": PANEL_A_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "A",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "light_epoch_argument": light_epoch,
        "dark_epoch_argument": dark_epoch,
        "datasets": dataset_metadata,
        "trajectory_types": list(PANEL_B_TRAJECTORY_TYPES),
        "linear_position_orientation": PANEL_B_LINEAR_POSITION_ORIENTATION,
        "order_mode": order_mode,
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "min_light_stability_correlation": float(
            REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION
        ),
        "firing_rate_normalization": "unit_max_per_trajectory",
        "order_trials": "figure_1d_dark_odd",
        "display_trials": "light_all",
        "source_unit_set": "figure_1d_cache_v6",
    }


def build_panel_a_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the cache path for one Supplementary Figure 3A heatmap payload."""
    dataset_token = _build_panel_a_dataset_cache_token(metadata["datasets"])
    region_token = _format_panel_a_cache_token(metadata["region"])
    light_epochs = [
        _format_panel_a_cache_token(dataset["light_epoch"])
        for dataset in metadata["datasets"]
    ]
    unique_light_epochs = list(dict.fromkeys(light_epochs))
    light_epoch_token = (
        unique_light_epochs[0]
        if len(unique_light_epochs) == 1
        else "mixed-" + "_".join(unique_light_epochs)
    )
    order_token = _format_panel_a_cache_token(metadata["order_mode"])
    filename = (
        f"{PANEL_A_CACHE_PREFIX}_{region_token}_light{light_epoch_token}"
        f"_datasets-{dataset_token}"
        f"_order{order_token}"
        f"_minlightstab"
        f"{_format_panel_a_cache_number(metadata['min_light_stability_correlation'])}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_panel_a_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_panel_a_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _panel_a_cache_array_name(order_trajectory: str, plot_trajectory: str) -> str:
    """Return the cache array name for one Supplementary Figure 3A panel."""
    return f"{order_trajectory}__{plot_trajectory}"


def _panel_a_cache_unit_order_array_name(order_trajectory: str) -> str:
    """Return the cache array name for one Supplementary Figure 3A unit order."""
    return f"unit_order__{order_trajectory}"


def save_panel_a_cache(
    cache_path: Path,
    panels: dict[tuple[str, str], np.ndarray],
    ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Write one Supplementary Figure 3A heatmap cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        PANEL_A_CACHE_METADATA_KEY: np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    trajectory_types = tuple(str(trajectory) for trajectory in metadata["trajectory_types"])
    for order_trajectory in trajectory_types:
        payload[_panel_a_cache_unit_order_array_name(order_trajectory)] = np.asarray(
            ordered_unit_keys_by_trajectory[order_trajectory],
            dtype=str,
        )
        for plot_trajectory in trajectory_types:
            payload[_panel_a_cache_array_name(order_trajectory, plot_trajectory)] = (
                np.asarray(panels[(order_trajectory, plot_trajectory)], dtype=float)
            )
    np.savez_compressed(cache_path, **payload)


def load_panel_a_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]] | None:
    """Return cached Supplementary Figure 3A heatmaps when metadata matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_A_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Supplementary Figure 3A cache at {cache_path}.")
                return None

            trajectory_types = tuple(
                str(trajectory) for trajectory in expected_metadata["trajectory_types"]
            )
            panels: dict[tuple[str, str], np.ndarray] = {}
            ordered_unit_keys_by_trajectory: dict[str, np.ndarray] = {}
            for order_trajectory in trajectory_types:
                ordered_unit_keys_by_trajectory[order_trajectory] = np.asarray(
                    data[_panel_a_cache_unit_order_array_name(order_trajectory)],
                    dtype=str,
                )
                for plot_trajectory in trajectory_types:
                    panels[(order_trajectory, plot_trajectory)] = np.asarray(
                        data[_panel_a_cache_array_name(order_trajectory, plot_trajectory)],
                        dtype=float,
                    )
            return panels, ordered_unit_keys_by_trajectory
    except Exception as exc:
        print(f"Ignoring unreadable Supplementary Figure 3A cache at {cache_path}: {exc}")
        return None


def build_panel_a_heatmap_payloads(
    *,
    figure_1d_ordered_unit_keys_by_trajectory: dict[str, np.ndarray],
    light_curve_sets: Sequence[dict[str, Any]],
    position_bin_count: int,
) -> dict[str, tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]]:
    """Return the Supplementary Figure 3A Fig. 1D-order heatmap payload."""
    figure_1d_panels = build_figure_1d_ordered_light_panel_values(
        ordered_unit_keys_by_trajectory=figure_1d_ordered_unit_keys_by_trajectory,
        light_curve_sets=light_curve_sets,
        position_bin_count=position_bin_count,
        curve_key="all_curves",
    )
    return {
        PANEL_A_FIGURE_1D_ORDER_MODE: (
            figure_1d_panels,
            figure_1d_ordered_unit_keys_by_trajectory,
        ),
    }


def load_dark_ordered_light_panel_values(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    figure_1d_cache_dir: Path | None = None,
    panel_a_cache_dir: Path | None = None,
    refresh_panel_a_cache: bool = False,
) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """Load Fig. 1D cells in light, cached for the displayed row order."""
    figure_1d_cache_dir = (
        DEFAULT_OUTPUT_DIR / "cache"
        if figure_1d_cache_dir is None
        else Path(figure_1d_cache_dir)
    )
    panel_a_cache_dir = (
        DEFAULT_OUTPUT_DIR / "cache"
        if panel_a_cache_dir is None
        else Path(panel_a_cache_dir)
    )
    figure_1d_datasets = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        figure_1d_datasets.append(
            (
                animal_name,
                date,
                dark_epoch if dark_epoch is not None else dataset_dark_epoch,
            )
        )

    metadata_by_order_mode = {
        order_mode: build_panel_a_cache_metadata(
            data_root=data_root,
            datasets=figure_1d_datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            order_mode=order_mode,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        )
        for order_mode in PANEL_A_ORDER_MODES
    }
    cache_paths_by_order_mode = {
        order_mode: build_panel_a_cache_path(panel_a_cache_dir, metadata)
        for order_mode, metadata in metadata_by_order_mode.items()
    }
    loaded_payloads = {}
    if not refresh_panel_a_cache:
        for order_mode, cache_path in cache_paths_by_order_mode.items():
            cached_payload = load_panel_a_cache(
                cache_path,
                metadata_by_order_mode[order_mode],
            )
            if cached_payload is not None:
                print(f"Loaded Supplementary Figure 3A cache from {cache_path}.")
                loaded_payloads[order_mode] = cached_payload

    missing_order_modes = [
        order_mode
        for order_mode in PANEL_A_ORDER_MODES
        if order_mode not in loaded_payloads
    ]
    if missing_order_modes:
        light_curve_sets = []
        for dataset in datasets:
            animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
            light_curve_sets.append(
                compute_light_epoch_all_trial_tuning_curves(
                    animal_name=animal_name,
                    date=date,
                    data_root=data_root,
                    region=region,
                    light_epoch=light_epoch,
                    position_bin_count=position_bin_count,
                    position_offset=position_offset,
                    speed_threshold_cm_s=speed_threshold_cm_s,
                    sigma_bins=sigma_bins,
                    use_trajectory_direction=True,
                )
            )
        _figure_1d_panels, ordered_unit_keys_by_trajectory = (
            load_or_compute_panel_d_heatmap_payload(
                data_root=data_root,
                datasets=figure_1d_datasets,
                region=region,
                position_bin_count=position_bin_count,
                position_offset=position_offset,
                speed_threshold_cm_s=speed_threshold_cm_s,
                sigma_bins=sigma_bins,
                panel_d_cache_dir=figure_1d_cache_dir,
                refresh_panel_d_cache=False,
                require_ordered_unit_keys=True,
            )
        )
        light_stable_unit_keys = select_unit_keys_by_light_tuning_stability(
            data_root=data_root,
            datasets=figure_1d_datasets,
            region=region,
            light_epoch=light_epoch,
        )
        ordered_unit_keys_by_trajectory = filter_ordered_unit_keys_by_unit_set(
            ordered_unit_keys_by_trajectory,
            light_stable_unit_keys,
        )
        computed_payloads = build_panel_a_heatmap_payloads(
            figure_1d_ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
            light_curve_sets=light_curve_sets,
            position_bin_count=position_bin_count,
        )
        for order_mode in missing_order_modes:
            panels, ordered_unit_keys = computed_payloads[order_mode]
            cache_path = cache_paths_by_order_mode[order_mode]
            save_panel_a_cache(
                cache_path,
                panels,
                ordered_unit_keys,
                metadata_by_order_mode[order_mode],
            )
            print(f"Saved Supplementary Figure 3A cache to {cache_path}.")
            loaded_payloads[order_mode] = panels, ordered_unit_keys

    return {
        order_mode: loaded_payloads[order_mode][0]
        for order_mode in PANEL_A_ORDER_MODES
    }


def set_heatmap_display_style(
    heatmap_axes: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Set the colormap and display range for all images in one heatmap grid."""
    for ax in np.asarray(heatmap_axes, dtype=object).ravel():
        for image in ax.images:
            image.set_cmap(cmap)
            image.set_clim(vmin=float(vmin), vmax=float(vmax))


def plot_dark_ordered_light_heatmap_regions(
    heatmap_axes: np.ndarray,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    figure_1d_cache_dir: Path | None = None,
    panel_a_cache_dir: Path | None = None,
    refresh_panel_a_cache: bool = False,
    order_mode: str = PANEL_A_FIGURE_1D_ORDER_MODE,
) -> "AxesImage | None":
    """Plot light-epoch heatmaps with rows sorted by Figure 1D dark order."""
    if order_mode not in PANEL_A_ORDER_MODES:
        raise ValueError(f"Unknown panel A order_mode {order_mode!r}.")
    color_image = None
    for region_index, region in enumerate(regions):
        panels_by_order_mode = load_dark_ordered_light_panel_values(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            figure_1d_cache_dir=figure_1d_cache_dir,
            panel_a_cache_dir=panel_a_cache_dir,
            refresh_panel_a_cache=refresh_panel_a_cache,
        )
        panels = panels_by_order_mode[order_mode]
        start_row = region_index * len(PANEL_B_TRAJECTORY_TYPES)
        stop_row = start_row + len(PANEL_B_TRAJECTORY_TYPES)
        image = plot_pooled_heatmap_grid(
            heatmap_axes[start_row:stop_row, :],
            panels,
            trajectory_types=PANEL_B_TRAJECTORY_TYPES,
            axis_orientation=PANEL_B_LINEAR_POSITION_ORIENTATION,
            cmap=REORDERED_HEATMAP_CMAP,
        )
        set_heatmap_display_style(
            heatmap_axes[start_row:stop_row, :],
            cmap=REORDERED_HEATMAP_CMAP,
            vmin=0.0,
            vmax=REORDERED_HEATMAP_VMAX,
        )
        for heatmap_ax in heatmap_axes[start_row:stop_row, :].ravel():
            add_segment_boundary_lines(heatmap_ax)
        if color_image is None and image is not None:
            color_image = image
    return color_image


def make_supplementary_figure_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    figure_1d_cache_dir: Path | None = None,
    panel_a_cache_dir: Path | None = None,
    refresh_panel_a_cache: bool = False,
) -> Path:
    """Build and save Supplementary Figure 3."""
    import matplotlib.pyplot as plt

    datasets = [normalize_dataset_id(dataset) for dataset in datasets]
    animal_groups = group_datasets_by_animal(datasets)
    figure_1d_cache_dir = (
        Path(output_path).parent / "cache"
        if figure_1d_cache_dir is None
        else Path(figure_1d_cache_dir)
    )
    panel_a_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_a_cache_dir is None
        else Path(panel_a_cache_dir)
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=False,
    )
    if not datasets:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved Supplementary Figure 3 to {output_path}")
        return output_path

    outer_grid = fig.add_gridspec(
        nrows=6,
        ncols=1,
        height_ratios=[
            DEFAULT_REORDERED_HEATMAP_HEIGHT_MM,
            DEFAULT_SECTION_SPACER_MM,
            DEFAULT_PER_ANIMAL_GRID_HEIGHT_MM,
            DEFAULT_BOTTOM_SECTION_SPACER_MM,
            DEFAULT_DARK_LIGHT_CORRELATION_HEIGHT_MM,
            DEFAULT_STABILITY_OVERLAY_HEIGHT_MM,
        ],
        hspace=0.04,
        left=PANEL_A_GRID_LEFT,
        right=PANEL_A_GRID_RIGHT,
        top=PANEL_A_GRID_TOP,
        bottom=PANEL_A_GRID_BOTTOM,
    )
    panel_a_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            PANEL_A_HEATMAP_SIDE_SPACER_FRACTION,
            PANEL_A_HEATMAP_WIDTH_FRACTION,
            PANEL_A_HEATMAP_SIDE_SPACER_FRACTION,
        ],
        wspace=0.0,
    )

    heatmap_panels = {
        PANEL_A_FIGURE_1D_ORDER_MODE: setup_light_heatmap_panel(
            fig,
            panel_a_grid[0, 1],
            regions=(region,),
        ),
    }
    color_image = None
    colorbar_axes = []
    for order_mode, heatmap_panel in heatmap_panels.items():
        image = plot_dark_ordered_light_heatmap_regions(
            heatmap_panel["heatmap_axes"],
            data_root=data_root,
            datasets=datasets,
            regions=(region,),
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            figure_1d_cache_dir=figure_1d_cache_dir,
            panel_a_cache_dir=panel_a_cache_dir,
            refresh_panel_a_cache=refresh_panel_a_cache,
            order_mode=order_mode,
        )
        if color_image is None and image is not None:
            color_image = image
        colorbar_axes.extend(heatmap_panel["heatmap_axes"].ravel().tolist())
    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=colorbar_axes,
            shrink=0.24,
            pad=PANEL_C_COLORBAR_PAD,
            aspect=7,
            ticks=[0.0, 1.0],
        )
        colorbar.ax.set_yticklabels(["0", "1"])
        colorbar.ax.tick_params(length=2)
        colorbar.set_label(
            "Norm. FR",
            rotation=90,
            labelpad=HEATMAP_COLORBAR_LABELPAD,
            fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
        )
    draw_neuron_scale_bar(
        heatmap_panels[PANEL_A_FIGURE_1D_ORDER_MODE]["heatmap_axes"][-1, -1],
        x=PANEL_C_NEURON_SCALE_BAR_X,
    )

    spacer_axis = fig.add_subplot(outer_grid[1, 0])
    spacer_axis.axis("off")

    per_animal_grid = outer_grid[2, 0].subgridspec(
        nrows=len(animal_groups),
        ncols=len(PER_ANIMAL_COLUMN_WIDTH_RATIOS),
        width_ratios=PER_ANIMAL_COLUMN_WIDTH_RATIOS,
        hspace=PER_ANIMAL_GRID_HSPACE,
        wspace=PER_ANIMAL_GRID_WSPACE,
    )
    for row_index, (animal_name, animal_datasets) in enumerate(animal_groups.items()):
        panel_d_axis = fig.add_subplot(per_animal_grid[row_index, 0])
        similarity_table = load_panel_d_similarity_table(
            data_root=data_root,
            datasets=animal_datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
        plot_panel_d_similarity(panel_d_axis, similarity_table)
        set_panel_a_dot_alpha(panel_d_axis)
        panel_d_axis.text(
            0.04,
            0.13,
            format_animal_row_label(animal_name, animal_datasets),
            ha="left",
            va="bottom",
            fontsize=ANIMAL_ROW_LABEL_FONTSIZE,
            transform=panel_d_axis.transAxes,
            color="0.25",
        )

        stable_similarity_axis = fig.add_subplot(per_animal_grid[row_index, 1])
        stable_similarity_table = filter_panel_d_similarity_table_by_tuning_stability(
            similarity_table,
            data_root=data_root,
            datasets=animal_datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
        plot_panel_d_similarity(stable_similarity_axis, stable_similarity_table)
        set_panel_a_dot_alpha(stable_similarity_axis)

        panel_e_axis = fig.add_subplot(per_animal_grid[row_index, 2])
        encoding_delta_table = load_panel_e_encoding_delta_table(
            data_root=data_root,
            datasets=animal_datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
        plot_panel_e_encoding_delta_histogram(panel_e_axis, encoding_delta_table)

        panel_f_axis = fig.add_subplot(per_animal_grid[row_index, 3])
        decoding_error_table = load_panel_f_decoding_error_table(
            data_root=data_root,
            datasets=animal_datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
        plot_panel_f_decoding_error(panel_f_axis, decoding_error_table)

        if row_index < len(animal_groups) - 1:
            hide_x_axis_labels(panel_d_axis)
            hide_x_axis_labels(stable_similarity_axis)
            hide_x_axis_labels(panel_e_axis)
            for child_axis in panel_f_axis.child_axes:
                hide_x_axis_labels(child_axis)

        if row_index == 0:
            panel_d_axis.set_title(
                "Fig. 3C similarity",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=2,
            )
            stable_similarity_axis.set_title(
                "Fig. 3C stable similarity",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=2,
            )
            panel_e_axis.set_title(
                "Fig. 3D encoding",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=2,
            )
            panel_f_axis.set_title(
                "Fig. 3E decoding",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=2,
            )
            label_axis(panel_d_axis, "B", x=-0.48, y=1.05)
            label_axis(stable_similarity_axis, "C", x=-0.48, y=1.05)
            label_axis(panel_e_axis, "D", x=-0.22, y=1.05)
            label_axis(panel_f_axis, "E", x=-0.10, y=1.05)

    bottom_spacer_axis = fig.add_subplot(outer_grid[3, 0])
    bottom_spacer_axis.axis("off")
    correlation_grid = outer_grid[4, 0].subgridspec(
        nrows=1,
        ncols=len(PANEL_B_TRAJECTORY_TYPES),
        wspace=DARK_LIGHT_CORRELATION_GRID_WSPACE,
    )
    correlation_axes = [
        fig.add_subplot(correlation_grid[0, column_index])
        for column_index in range(len(PANEL_B_TRAJECTORY_TYPES))
    ]
    dark_light_correlation_table = load_dark_light_tuning_correlation_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    plot_dark_light_tuning_correlation_histograms(
        correlation_axes,
        dark_light_correlation_table,
    )

    overlay_grid = outer_grid[5, 0].subgridspec(
        nrows=1,
        ncols=len(PANEL_B_TRAJECTORY_TYPES),
        wspace=DARK_LIGHT_CORRELATION_GRID_WSPACE,
    )
    overlay_axes = [
        fig.add_subplot(overlay_grid[0, column_index])
        for column_index in range(len(PANEL_B_TRAJECTORY_TYPES))
    ]
    light_stability_table = load_light_tuning_stability_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
    )
    plot_dark_light_with_light_stability_histograms(
        overlay_axes,
        dark_light_correlation_table,
        light_stability_table,
    )

    fig.canvas.draw()
    add_centered_axis_text(
        fig,
        correlation_axes,
        "Dark-light tuning similarity",
        y_offset=0.025,
        fontsize=PANEL_TITLE_FONTSIZE,
    )
    label_axis(correlation_axes[0], "F", x=-0.30, y=1.05)
    add_centered_axis_text(
        fig,
        overlay_axes,
        "Dark-light similarity and light odd/even stability",
        y_offset=0.025,
        fontsize=PANEL_TITLE_FONTSIZE,
    )
    label_axis(overlay_axes[0], "G", x=-0.30, y=1.05)
    for heatmap_panel in heatmap_panels.values():
        add_centered_axis_text(
            fig,
            heatmap_panel["tuning_schematic_axes"],
            REORDERED_HEATMAP_TITLE,
            y_offset=0.006,
            fontsize=PANEL_TITLE_FONTSIZE,
        )
        add_centered_axis_text(
            fig,
            heatmap_panel["tuning_schematic_axes"],
            "Tuning",
            y_offset=-0.026,
            fontsize=8.0,
        )
        add_centered_axis_text(
            fig,
            heatmap_panel["order_schematic_axes"],
            "Order",
            y_offset=-0.006,
            rotation=90,
        )
    label_axis(
        heatmap_panels[PANEL_A_FIGURE_1D_ORDER_MODE]["corner_axis"],
        "A",
        x=-0.12,
        y=0.52,
    )

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Supplementary Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 3 generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Supplementary Figure 3 reordered heatmap and per-animal "
            "Figure 3C-E panels."
        )
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
        choices=REGIONS,
        default=DEFAULT_REGIONS[0],
        help=f"Region to include. Default: {DEFAULT_REGIONS[0]}.",
    )
    parser.add_argument(
        "--light-epoch",
        default=None,
        help=(
            "Light run epoch for the reordered heatmap and Figure 3D-F panels. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help="Dark run epoch. Default: registry value for each animal.",
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
        help=(
            "Number of leading position samples to ignore. "
            f"Default: {DEFAULT_POSITION_OFFSET}"
        ),
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
        "--panel-a-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Supplementary Figure 3A heatmap matrices. "
            "Default: output directory/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-a-cache",
        action="store_true",
        help="Recompute Supplementary Figure 3A and overwrite matching caches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Supplementary Figure 3 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_a_cache_dir=args.panel_a_cache_dir,
        refresh_panel_a_cache=args.refresh_panel_a_cache,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
