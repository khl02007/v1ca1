from __future__ import annotations

"""Shared helpers for full-W 1D decoding workflows."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy import interpolate
from sklearn.model_selection import KFold

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    get_analysis_path,
    get_run_epochs,
    load_clean_dlc_position_data,
    load_ephys_timestamps_all,
    load_epoch_tags,
    load_position_timestamps,
    load_trajectory_intervals,
)
from v1ca1.helper.wtrack import get_wtrack_full_graph
from v1ca1.ripple.ripple_glm import (
    DEFAULT_RIDGE_STRENGTH as DEFAULT_RIPPLE_GLM_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S as DEFAULT_RIPPLE_GLM_WINDOW_OFFSET_S,
    DEFAULT_RIPPLE_WINDOW_S as DEFAULT_RIPPLE_GLM_WINDOW_S,
    RIPPLE_SELECTION_MODE_ALL,
    format_ridge_strength_suffix,
    format_ripple_selection_suffix,
    format_ripple_window_suffix,
)


DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_STATE = 47
DEFAULT_TIME_BIN_SIZE_S = 0.002
DEFAULT_POSITION_STD = 4.0
DEFAULT_PLACE_BIN_SIZE_CM = 2.0
DEFAULT_MOVEMENT_VAR = 6.0
DEFAULT_BRANCH_GAP_CM = 15.0
DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD = 0.05
DISCRETE_VAR_CHOICES = ("switching", "random_walk", "uniform")
UNIT_SELECTION_ALL = "all_units"
UNIT_SELECTION_V1_RIPPLE_DEVEXP = "v1_units_ripple_devexp_p_lt_0p05"


def get_unit_selection_label(v1_ripple_glm_units: bool) -> str:
    """Return the filename/log label for the requested unit-selection mode."""
    if v1_ripple_glm_units:
        return UNIT_SELECTION_V1_RIPPLE_DEVEXP
    return UNIT_SELECTION_ALL


def format_unit_selection_path_suffix(unit_selection_label: str) -> str:
    """Return the filename suffix for one unit-selection label."""
    if unit_selection_label == UNIT_SELECTION_ALL:
        return ""
    return f"_{unit_selection_label}"


def select_regions(region: str | None) -> tuple[str, ...]:
    """Return the requested region tuple."""
    if region is None:
        return REGIONS
    return (region,)


def get_fit_output_dir(analysis_path: Path) -> Path:
    """Return the output directory for fitted 1D decoder classifiers."""
    return analysis_path / "decoding" / "fit" / "1d"


def get_predict_output_dir(analysis_path: Path) -> Path:
    """Return the output directory for predicted 1D decoder results."""
    return analysis_path / "decoding" / "predict" / "1d"


def classifier_output_path(
    output_dir: Path,
    *,
    region: str,
    epoch: str,
    fold: int,
    n_folds: int,
    random_state: int,
    direction: bool,
    movement: bool,
    speed_threshold_cm_s: float,
    position_std: float,
    discrete_var: str,
    place_bin_size: float,
    movement_var: float,
    branch_gap_cm: float,
    unit_selection_label: str = UNIT_SELECTION_ALL,
) -> Path:
    """Return one classifier output path for the requested fit settings."""
    return output_dir / (
        f"classifier_{region}_{epoch}_1d"
        f"_fold_{fold}_of_{n_folds}_cv"
        f"_random_state_{random_state}"
        f"_direction_{direction}"
        f"_movement_{movement}"
        f"_speed_threshold_cm_s_{speed_threshold_cm_s}"
        f"_position_std_{position_std}"
        f"_discrete_var_{discrete_var}"
        f"_place_bin_size_{place_bin_size}"
        f"_movement_var_{movement_var}"
        f"_branch_gap_cm_{branch_gap_cm}"
        f"{format_unit_selection_path_suffix(unit_selection_label)}"
        ".pkl"
    )


def build_classifier_output_paths(
    output_dir: Path,
    *,
    regions: tuple[str, ...],
    epoch: str,
    n_folds: int,
    random_state: int,
    direction: bool,
    movement: bool,
    speed_threshold_cm_s: float,
    position_std: float,
    discrete_var: str,
    place_bin_size: float,
    movement_var: float,
    branch_gap_cm: float,
    unit_selection_label: str = UNIT_SELECTION_ALL,
) -> dict[tuple[str, int], Path]:
    """Return all expected classifier output paths."""
    return {
        (region, fold): classifier_output_path(
            output_dir,
            region=region,
            epoch=epoch,
            fold=fold,
            n_folds=n_folds,
            random_state=random_state,
            direction=direction,
            movement=movement,
            speed_threshold_cm_s=speed_threshold_cm_s,
            position_std=position_std,
            discrete_var=discrete_var,
            place_bin_size=place_bin_size,
            movement_var=movement_var,
            branch_gap_cm=branch_gap_cm,
            unit_selection_label=unit_selection_label,
        )
        for region in regions
        for fold in range(n_folds)
    }


def prediction_output_path(
    output_dir: Path,
    *,
    region: str,
    epoch: str,
    n_folds: int,
    random_state: int,
    direction: bool,
    movement: bool,
    speed_threshold_cm_s: float,
    position_std: float,
    discrete_var: str,
    place_bin_size: float,
    movement_var: float,
    branch_gap_cm: float,
    unit_selection_label: str = UNIT_SELECTION_ALL,
) -> Path:
    """Return the combined prediction result path for one region."""
    return output_dir / (
        f"results_{region}_{epoch}_1d_cv"
        f"_folds_{n_folds}"
        f"_random_state_{random_state}"
        f"_direction_{direction}"
        f"_movement_{movement}"
        f"_speed_threshold_cm_s_{speed_threshold_cm_s}"
        f"_position_std_{position_std}"
        f"_discrete_var_{discrete_var}"
        f"_place_bin_size_{place_bin_size}"
        f"_movement_var_{movement_var}"
        f"_branch_gap_cm_{branch_gap_cm}"
        f"{format_unit_selection_path_suffix(unit_selection_label)}"
        ".nc"
    )


def build_prediction_output_paths(
    output_dir: Path,
    *,
    regions: tuple[str, ...],
    epoch: str,
    n_folds: int,
    random_state: int,
    direction: bool,
    movement: bool,
    speed_threshold_cm_s: float,
    position_std: float,
    discrete_var: str,
    place_bin_size: float,
    movement_var: float,
    branch_gap_cm: float,
    unit_selection_label: str = UNIT_SELECTION_ALL,
) -> dict[str, Path]:
    """Return combined prediction output paths for all requested regions."""
    return {
        region: prediction_output_path(
            output_dir,
            region=region,
            epoch=epoch,
            n_folds=n_folds,
            random_state=random_state,
            direction=direction,
            movement=movement,
            speed_threshold_cm_s=speed_threshold_cm_s,
            position_std=position_std,
            discrete_var=discrete_var,
            place_bin_size=place_bin_size,
            movement_var=movement_var,
            branch_gap_cm=branch_gap_cm,
            unit_selection_label=unit_selection_label,
        )
        for region in regions
    }


def figurl_output_path(
    output_dir: Path,
    *,
    regions: tuple[str, ...],
    epoch: str,
    n_folds: int,
    random_state: int,
    direction: bool,
    movement: bool,
    speed_threshold_cm_s: float,
    position_std: float,
    discrete_var: str,
    place_bin_size: float,
    movement_var: float,
    branch_gap_cm: float,
    unit_selection_label: str = UNIT_SELECTION_ALL,
) -> Path:
    """Return the combined figurl text output path."""
    region_label = "_".join(regions)
    return output_dir / "figurl" / (
        f"view_url_{region_label}_{epoch}_1d_cv"
        f"_folds_{n_folds}"
        f"_random_state_{random_state}"
        f"_direction_{direction}"
        f"_movement_{movement}"
        f"_speed_threshold_cm_s_{speed_threshold_cm_s}"
        f"_position_std_{position_std}"
        f"_discrete_var_{discrete_var}"
        f"_place_bin_size_{place_bin_size}"
        f"_movement_var_{movement_var}"
        f"_branch_gap_cm_{branch_gap_cm}"
        f"{format_unit_selection_path_suffix(unit_selection_label)}"
        ".txt"
    )


def preflight_no_existing(
    paths: list[Path],
    *,
    overwrite: bool,
    output_kind: str,
) -> None:
    """Fail before running if outputs exist and overwrite is disabled."""
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return

    if not overwrite:
        print(f"Found existing {output_kind} output(s). Refusing to run without --overwrite.")
        print("Use --overwrite to replace these files, or remove them before rerunning.")
        for path in existing_paths[:20]:
            print(f"  {path}")
        if len(existing_paths) > 20:
            print(f"  ... and {len(existing_paths) - 20} more")
        raise SystemExit(1)

    print(f"Overwriting existing {output_kind} output(s) because --overwrite was passed:")
    for path in existing_paths[:20]:
        print(f"  {path}")
    if len(existing_paths) > 20:
        print(f"  ... and {len(existing_paths) - 20} more")


def require_existing_paths(paths: list[Path], *, input_kind: str) -> None:
    """Fail before running if required input paths are missing."""
    missing_paths = [path for path in paths if not path.exists()]
    if not missing_paths:
        return

    print(f"Missing required {input_kind} input(s).")
    for path in missing_paths[:20]:
        print(f"  {path}")
    if len(missing_paths) > 20:
        print(f"  ... and {len(missing_paths) - 20} more")
    raise SystemExit(1)


def require_modern_source(source: str, artifact_name: str) -> None:
    """Require a modern artifact source and reject pickle fallbacks."""
    if source != "pynapple":
        raise ValueError(
            f"This workflow requires modern {artifact_name} artifacts. "
            f"Found source {source!r}, expected 'pynapple'."
        )


def load_required_session_inputs(
    *,
    analysis_path: Path,
    animal_name: str,
    epoch: str,
) -> dict[str, Any]:
    """Load the required modern session artifacts for one epoch."""
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, epoch_source = load_epoch_tags(analysis_path)
    require_modern_source(epoch_source, "timestamps_ephys")
    run_epochs = get_run_epochs(epoch_tags)
    if epoch not in run_epochs:
        raise ValueError(
            f"--epoch must be one of the run epochs {run_epochs!r}. Got {epoch!r}."
        )

    position_epoch_tags, timestamps_position, position_timestamp_source = (
        load_position_timestamps(analysis_path)
    )
    require_modern_source(position_timestamp_source, "timestamps_position")
    if position_epoch_tags != epoch_tags:
        raise ValueError(
            "Saved position timestamp epochs do not match saved ephys epochs. "
            f"Ephys epochs: {epoch_tags!r}; position epochs: {position_epoch_tags!r}"
        )

    timestamps_ephys_all, ephys_all_source = load_ephys_timestamps_all(analysis_path)
    require_modern_source(ephys_all_source, "timestamps_ephys_all")

    clean_dlc_path = (
        analysis_path / DEFAULT_CLEAN_DLC_POSITION_DIRNAME / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    parquet_epoch_order, position_by_epoch, _body_position_by_epoch = (
        load_clean_dlc_position_data(analysis_path, validate_timestamps=True)
    )
    if epoch not in parquet_epoch_order:
        raise ValueError(
            f"Cleaned DLC position parquet does not contain epoch {epoch!r}. "
            f"Available epochs: {parquet_epoch_order!r}"
        )

    trajectory_intervals_by_epoch, trajectory_source = load_trajectory_intervals(
        analysis_path,
        [epoch],
    )
    if trajectory_source != "parquet":
        raise ValueError(
            f"This workflow requires trajectory_times.parquet. Found {trajectory_source!r}."
        )

    return {
        "animal_name": animal_name,
        "analysis_path": analysis_path,
        "epoch_tags": epoch_tags,
        "run_epochs": run_epochs,
        "epoch": epoch,
        "timestamps_position": timestamps_position[epoch],
        "timestamps_ephys_all": timestamps_ephys_all,
        "position": position_by_epoch[epoch],
        "trajectory_intervals": trajectory_intervals_by_epoch[epoch],
        "sources": {
            "epoch_tags": epoch_source,
            "timestamps_position": position_timestamp_source,
            "timestamps_ephys_all": ephys_all_source,
            "trajectory_intervals": trajectory_source,
            "position": str(clean_dlc_path),
            "track_geometry": animal_name,
        },
    }


def load_sortings(analysis_path: Path, regions: tuple[str, ...]) -> dict[str, Any]:
    """Load SpikeInterface sorting extractors for the requested regions."""
    import spikeinterface.full as si

    sortings: dict[str, Any] = {}
    for region in regions:
        sorting_path = analysis_path / f"sorting_{region}"
        if not sorting_path.exists():
            raise FileNotFoundError(f"Sorting output not found: {sorting_path}")
        sortings[region] = si.load(sorting_path)
    return sortings


def get_default_v1_ripple_glm_path(analysis_path: Path, epoch: str) -> Path:
    """Return the canonical ripple GLM artifact used for V1 unit selection."""
    ripple_window_suffix = format_ripple_window_suffix(
        DEFAULT_RIPPLE_GLM_WINDOW_S,
        ripple_window_offset_s=DEFAULT_RIPPLE_GLM_WINDOW_OFFSET_S,
    )
    ripple_selection_suffix = format_ripple_selection_suffix(RIPPLE_SELECTION_MODE_ALL)
    ridge_strength_suffix = format_ridge_strength_suffix(DEFAULT_RIPPLE_GLM_RIDGE_STRENGTH)
    return analysis_path / "ripple_glm" / (
        f"{epoch}_{ripple_window_suffix}_{ripple_selection_suffix}_"
        f"{ridge_strength_suffix}_samplewise_ripple_glm.nc"
    )


def normalize_unit_id(unit_id: Any) -> Any:
    """Return a Python scalar unit ID when possible."""
    if isinstance(unit_id, np.generic):
        unit_id = unit_id.item()
    if isinstance(unit_id, bytes):
        return unit_id.decode("utf-8")
    return unit_id


def unit_ids_for_log(unit_ids: list[Any]) -> list[str]:
    """Return unit IDs as JSON-safe strings for run logs."""
    return [str(normalize_unit_id(unit_id)) for unit_id in unit_ids]


def align_unit_ids_to_sorting(
    unit_ids: list[Any],
    sorting_unit_ids: list[Any],
    *,
    region: str,
    source_path: Path,
) -> list[Any]:
    """Return selected unit IDs aligned to the sorting extractor's ID objects."""
    normalized_sorting_unit_ids = [
        normalize_unit_id(unit_id) for unit_id in sorting_unit_ids
    ]
    sorting_by_value = {
        unit_id: sorting_unit_ids[index]
        for index, unit_id in enumerate(normalized_sorting_unit_ids)
    }
    sorting_by_text: dict[str, Any] = {}
    duplicate_text_ids: set[str] = set()
    for unit_id, original_unit_id in zip(
        normalized_sorting_unit_ids,
        sorting_unit_ids,
        strict=True,
    ):
        unit_id_text = str(unit_id)
        if unit_id_text in sorting_by_text:
            duplicate_text_ids.add(unit_id_text)
        sorting_by_text[unit_id_text] = original_unit_id

    aligned_unit_ids: list[Any] = []
    missing_unit_ids: list[Any] = []
    for unit_id in unit_ids:
        normalized_unit_id = normalize_unit_id(unit_id)
        if normalized_unit_id in sorting_by_value:
            aligned_unit_ids.append(sorting_by_value[normalized_unit_id])
            continue

        unit_id_text = str(normalized_unit_id)
        if unit_id_text in sorting_by_text and unit_id_text not in duplicate_text_ids:
            aligned_unit_ids.append(sorting_by_text[unit_id_text])
            continue

        missing_unit_ids.append(normalized_unit_id)

    if missing_unit_ids:
        raise ValueError(
            f"Ripple GLM unit-selection artifact for region {region!r} contains "
            "unit IDs that are not present in the sorting extractor. "
            f"Missing unit IDs: {missing_unit_ids[:20]!r}; artifact: {source_path}"
        )
    return aligned_unit_ids


def load_v1_ripple_glm_significant_unit_ids(
    *,
    analysis_path: Path,
    epoch: str,
    p_value_threshold: float = DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD,
) -> tuple[list[Any], dict[str, Any]]:
    """Load V1 units with significant ripple deviance explained."""
    import xarray as xr

    artifact_path = get_default_v1_ripple_glm_path(analysis_path, epoch)
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Required V1 ripple GLM unit-selection artifact not found. "
            f"Expected the default same-epoch ripple GLM output at {artifact_path}"
        )

    with xr.open_dataset(artifact_path) as dataset:
        if "ripple_devexp_p_value" not in dataset:
            raise ValueError(
                "Ripple GLM artifact is missing required variable "
                f"'ripple_devexp_p_value': {artifact_path}"
            )
        if "unit" not in dataset.coords:
            raise ValueError(
                f"Ripple GLM artifact is missing required 'unit' coordinate: {artifact_path}"
            )

        p_values = np.asarray(dataset["ripple_devexp_p_value"].values, dtype=float).ravel()
        unit_ids = np.asarray(dataset.coords["unit"].values).ravel()
        if p_values.shape[0] != unit_ids.shape[0]:
            raise ValueError(
                "Ripple GLM artifact has mismatched p-value and unit dimensions: "
                f"{p_values.shape[0]} p-values vs {unit_ids.shape[0]} units in {artifact_path}"
            )
        keep = np.isfinite(p_values) & (p_values < float(p_value_threshold))
        selected_unit_ids = [
            normalize_unit_id(unit_id)
            for unit_id in unit_ids[keep].tolist()
        ]

    if not selected_unit_ids:
        raise ValueError(
            "No V1 units passed ripple GLM deviance-explained selection "
            f"(ripple_devexp_p_value < {p_value_threshold}) in {artifact_path}"
        )

    return selected_unit_ids, {
        "method": UNIT_SELECTION_V1_RIPPLE_DEVEXP,
        "source_path": artifact_path,
        "p_value_threshold": float(p_value_threshold),
        "n_available": int(unit_ids.size),
        "n_selected": int(len(selected_unit_ids)),
    }


def build_unit_ids_by_region(
    *,
    sortings: dict[str, Any],
    regions: tuple[str, ...],
    analysis_path: Path,
    epoch: str,
    v1_ripple_glm_units: bool,
) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]]]:
    """Return decoding unit IDs and selection metadata for each region."""
    unit_ids_by_region: dict[str, list[Any]] = {}
    unit_selection_by_region: dict[str, dict[str, Any]] = {}

    for region in regions:
        sorting_unit_ids = list(sortings[region].get_unit_ids())
        if region == "v1" and v1_ripple_glm_units:
            selected_unit_ids, selection_info = load_v1_ripple_glm_significant_unit_ids(
                analysis_path=analysis_path,
                epoch=epoch,
            )
            aligned_unit_ids = align_unit_ids_to_sorting(
                selected_unit_ids,
                sorting_unit_ids,
                region=region,
                source_path=selection_info["source_path"],
            )
            unit_ids_by_region[region] = aligned_unit_ids
            unit_selection_by_region[region] = {
                **selection_info,
                "selected_unit_ids": unit_ids_for_log(aligned_unit_ids),
            }
            print(
                "Restricting V1 decoding units to "
                f"{len(aligned_unit_ids)}/{selection_info['n_available']} units with "
                "ripple_devexp_p_value "
                f"< {selection_info['p_value_threshold']} from "
                f"{selection_info['source_path']}."
            )
            continue

        unit_ids_by_region[region] = sorting_unit_ids
        unit_selection_by_region[region] = {
            "method": UNIT_SELECTION_ALL,
            "n_available": int(len(sorting_unit_ids)),
            "n_selected": int(len(sorting_unit_ids)),
            "selected_unit_ids": unit_ids_for_log(sorting_unit_ids),
        }

    return unit_ids_by_region, unit_selection_by_region


def get_analysis_path_for_session(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the analysis path for one animal/date session."""
    return get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)


def build_time_grid(
    timestamps_position: np.ndarray,
    *,
    position_offset: int,
    time_bin_size_s: float,
) -> np.ndarray:
    """Return the uniform decoder time grid for one epoch."""
    if timestamps_position.size <= position_offset:
        raise ValueError(
            "Position offset removes all timestamp samples. "
            f"timestamp count: {timestamps_position.size}, position_offset: {position_offset}"
        )

    trimmed_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    start_time = float(trimmed_timestamps[0])
    end_time = float(trimmed_timestamps[-1])
    if end_time <= start_time:
        raise ValueError("Trimmed position timestamps must span a positive duration.")

    sampling_rate = 1.0 / float(time_bin_size_s)
    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1
    return np.linspace(start_time, end_time, n_samples)


def get_spike_indicator(
    sorting: Any,
    *,
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
    unit_ids: list[Any] | None = None,
) -> np.ndarray:
    """Bin one sorting extractor's spike trains onto the requested time grid."""
    spike_indicator: list[np.ndarray] = []
    all_timestamps = np.asarray(timestamps_ephys_all, dtype=float)
    time_array = np.asarray(time_grid, dtype=float)

    selected_unit_ids = list(sorting.get_unit_ids()) if unit_ids is None else unit_ids
    for unit_id in selected_unit_ids:
        spike_indices = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=int)
        spike_times = all_timestamps[spike_indices]
        spike_times = spike_times[(spike_times > time_array[0]) & (spike_times <= time_array[-1])]
        spike_indicator.append(
            np.bincount(
                np.digitize(spike_times, time_array[1:-1]),
                minlength=time_array.shape[0],
            )
        )
    if not spike_indicator:
        return np.zeros((time_array.shape[0], 0), dtype=float)
    return np.asarray(spike_indicator, dtype=float).T


def interpolate_position_to_time(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    time_grid: np.ndarray,
    *,
    position_offset: int,
) -> np.ndarray:
    """Interpolate trimmed XY position onto the decoder time grid."""
    if position.shape[0] != timestamps_position.size:
        raise ValueError(
            "Position samples and position timestamps must have matching lengths. "
            f"Got {position.shape[0]} and {timestamps_position.size}."
        )
    trimmed_position = np.asarray(position[position_offset:], dtype=float)
    trimmed_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    interpolator = interpolate.interp1d(
        trimmed_timestamps,
        trimmed_position,
        axis=0,
        bounds_error=False,
        kind="linear",
    )
    return np.asarray(interpolator(time_grid), dtype=float)


def linearize_full_w_position(
    *,
    animal_name: str,
    position_interp: np.ndarray,
    branch_gap_cm: float,
) -> tuple[np.ndarray, Any, list[tuple[int, int]], list[float]]:
    """Linearize interpolated XY position on the animal-specific full W-track."""
    import track_linearization as tl

    track_graph, edge_order, edge_spacing = get_wtrack_full_graph(
        animal_name,
        branch_gap_cm=branch_gap_cm,
    )
    position_df = tl.get_linearized_position(
        position=position_interp,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )
    linear_position = np.asarray(position_df["linear_position"], dtype=float)
    return linear_position, track_graph, edge_order, edge_spacing


def compute_speed_on_time_grid(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    time_grid: np.ndarray,
    *,
    position_offset: int,
) -> np.ndarray:
    """Compute speed from trimmed position and interpolate onto the decoder grid."""
    import position_tools as pt

    trimmed_position = np.asarray(position[position_offset:], dtype=float)
    trimmed_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    position_sampling_rate = len(trimmed_position) / (
        trimmed_timestamps[-1] - trimmed_timestamps[0]
    )
    speed = np.asarray(
        pt.get_speed(
            trimmed_position,
            time=trimmed_timestamps,
            sampling_frequency=position_sampling_rate,
            sigma=DEFAULT_SPEED_SIGMA_S,
        ),
        dtype=float,
    )
    speed_interpolator = interpolate.interp1d(
        trimmed_timestamps,
        speed,
        axis=0,
        bounds_error=False,
        kind="linear",
    )
    return np.asarray(speed_interpolator(time_grid), dtype=float)


def intervalset_to_arrays(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted start and end arrays for one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            f"IntervalSet starts and ends have mismatched shapes: {starts.shape} vs {ends.shape}."
        )
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts)
    return starts[order], ends[order]


def validate_fold_count(
    trajectory_intervals: dict[str, Any],
    *,
    n_folds: int,
) -> dict[str, int]:
    """Validate that every trajectory type supports the requested fold count."""
    lap_counts: dict[str, int] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        starts, _ends = intervalset_to_arrays(trajectory_intervals[trajectory_type])
        lap_counts[trajectory_type] = int(starts.size)
        if starts.size < n_folds:
            raise ValueError(
                f"Trajectory {trajectory_type!r} has {starts.size} laps, but "
                f"--n-folds={n_folds} requires at least {n_folds} laps per trajectory."
            )
    return lap_counts


def combine_interval_chunks(
    starts: list[np.ndarray],
    ends: list[np.ndarray],
) -> np.ndarray:
    """Combine interval chunks into one sorted `(n, 2)` array."""
    start_chunks = [np.asarray(chunk, dtype=float) for chunk in starts if chunk.size]
    end_chunks = [np.asarray(chunk, dtype=float) for chunk in ends if chunk.size]
    if not start_chunks:
        return np.empty((0, 2), dtype=float)

    start_array = np.concatenate(start_chunks)
    end_array = np.concatenate(end_chunks)
    order = np.argsort(start_array)
    return np.column_stack((start_array[order], end_array[order]))


def build_lapwise_cv(
    trajectory_intervals: dict[str, Any],
    *,
    n_folds: int,
    random_state: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict[str, Any]]]:
    """Build shuffled lap-wise CV intervals and per-lap fold ownership records."""
    validate_fold_count(trajectory_intervals, n_folds=n_folds)

    train_starts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    train_ends: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    test_starts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    test_ends: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    fold_interval_records: list[dict[str, Any]] = []

    for trajectory_type in TRAJECTORY_TYPES:
        starts, ends = intervalset_to_arrays(trajectory_intervals[trajectory_type])
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(starts.size))):
            train_starts[fold].append(starts[train_idx])
            train_ends[fold].append(ends[train_idx])
            test_starts[fold].append(starts[test_idx])
            test_ends[fold].append(ends[test_idx])
            for lap_index in test_idx:
                fold_interval_records.append(
                    {
                        "start": float(starts[lap_index]),
                        "end": float(ends[lap_index]),
                        "trajectory_type": trajectory_type,
                        "fold": int(fold),
                    }
                )

    train_intervals = {
        fold: combine_interval_chunks(train_starts[fold], train_ends[fold])
        for fold in range(n_folds)
    }
    test_intervals = {
        fold: combine_interval_chunks(test_starts[fold], test_ends[fold])
        for fold in range(n_folds)
    }
    sorted_records = validate_non_overlapping_fold_intervals(fold_interval_records)
    return train_intervals, test_intervals, sorted_records


def build_lapwise_train_test_intervals(
    trajectory_intervals: dict[str, Any],
    *,
    n_folds: int,
    random_state: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build shuffled lap-wise train/test intervals pooled across trajectories."""
    train_intervals, test_intervals, _records = build_lapwise_cv(
        trajectory_intervals,
        n_folds=n_folds,
        random_state=random_state,
    )
    return train_intervals, test_intervals


def validate_non_overlapping_fold_intervals(
    fold_interval_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return chronologically sorted fold records, failing on invalid overlaps."""
    if not fold_interval_records:
        raise ValueError("No trajectory intervals were available for fold assignment.")

    sorted_records = sorted(fold_interval_records, key=lambda record: record["start"])
    previous_record: dict[str, Any] | None = None
    for record in sorted_records:
        if record["end"] <= record["start"]:
            raise ValueError(
                "Trajectory interval end time must be after start time. "
                f"Got {record!r}."
            )
        if previous_record is not None and record["start"] < previous_record["end"]:
            raise ValueError(
                "Trajectory intervals overlap, so prediction fold ownership is ambiguous. "
                f"Previous interval: {previous_record!r}; current interval: {record!r}."
            )
        previous_record = record
    return sorted_records


def make_interval_mask(time_grid: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Return a boolean mask selecting time bins inside any interval."""
    mask = np.zeros(time_grid.shape, dtype=bool)
    for start_time, end_time in np.asarray(intervals, dtype=float):
        mask |= (time_grid > start_time) & (time_grid <= end_time)
    return mask


def build_full_epoch_prediction_fold(
    time_grid: np.ndarray,
    fold_interval_records: list[dict[str, Any]],
) -> np.ndarray:
    """Assign every trimmed epoch time bin to exactly one prediction fold.

    Bins inside a trajectory interval inherit that trajectory interval's fold.
    Bins between trajectories inherit the earlier trajectory's fold. Bins
    before the first trajectory inherit the first fold, and bins after the last
    trajectory inherit the last fold.
    """
    sorted_records = validate_non_overlapping_fold_intervals(fold_interval_records)
    time_array = np.asarray(time_grid, dtype=float)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    if time_array.size > 1 and np.any(np.diff(time_array) <= 0):
        raise ValueError("time_grid must be strictly increasing.")

    fold_by_time = np.full(time_array.shape, -1, dtype=int)
    for index, record in enumerate(sorted_records):
        fold = int(record["fold"])
        start_time = float(record["start"])
        end_time = float(record["end"])
        if index == 0:
            fold_by_time[time_array <= start_time] = fold
        fold_by_time[(time_array > start_time) & (time_array <= end_time)] = fold

        if index + 1 < len(sorted_records):
            next_start_time = float(sorted_records[index + 1]["start"])
            fold_by_time[(time_array > end_time) & (time_array <= next_start_time)] = fold
        else:
            fold_by_time[time_array > end_time] = fold

    if np.any(fold_by_time < 0):
        raise ValueError("Could not assign every time bin to a prediction fold.")
    return fold_by_time


def get_trajectory_direction(linear_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inbound/outbound labels from linear-position derivatives."""
    is_inbound = np.insert(np.diff(linear_position) < 0, 0, False)
    return np.where(is_inbound, "Inbound", "Outbound"), is_inbound


def get_state_names(*, direction: bool, discrete_var: str) -> list[str]:
    """Return RTC state labels for one model configuration."""
    if direction and discrete_var == "switching":
        return [
            "Inbound-Continuous",
            "Inbound-Fragmented",
            "Outbound-Continuous",
            "Outbound-Fragmented",
        ]
    if direction and discrete_var in {"random_walk", "uniform"}:
        return ["Inbound", "Outbound"]
    if not direction and discrete_var == "switching":
        return ["Continuous", "Fragmented"]
    if not direction and discrete_var == "random_walk":
        return ["Continuous1", "Continuous2"]
    if not direction and discrete_var == "uniform":
        return ["Fragmented1", "Fragmented2"]
    raise ValueError(f"Unsupported discrete_var={discrete_var!r}.")


def build_decoder_state_models(
    *,
    direction: bool,
    discrete_var: str,
    movement_var: float,
) -> tuple[list[list[Any]], list[Any] | None]:
    """Build RTC transition and observation model objects."""
    import replay_trajectory_classification as rtc

    random_walk = rtc.RandomWalk(movement_var=movement_var)
    uniform = rtc.Uniform()

    if direction and discrete_var == "switching":
        return (
            [
                [random_walk, uniform, uniform, uniform],
                [uniform, uniform, uniform, uniform],
                [uniform, uniform, random_walk, uniform],
                [uniform, uniform, uniform, uniform],
            ],
            [
                rtc.ObservationModel(encoding_group="Inbound"),
                rtc.ObservationModel(encoding_group="Inbound"),
                rtc.ObservationModel(encoding_group="Outbound"),
                rtc.ObservationModel(encoding_group="Outbound"),
            ],
        )
    if direction and discrete_var == "random_walk":
        return (
            [[random_walk, random_walk], [random_walk, random_walk]],
            [
                rtc.ObservationModel(encoding_group="Inbound"),
                rtc.ObservationModel(encoding_group="Outbound"),
            ],
        )
    if direction and discrete_var == "uniform":
        return (
            [[uniform, uniform], [uniform, uniform]],
            [
                rtc.ObservationModel(encoding_group="Inbound"),
                rtc.ObservationModel(encoding_group="Outbound"),
            ],
        )
    if not direction and discrete_var == "switching":
        return [[random_walk, uniform], [uniform, uniform]], None
    if not direction and discrete_var == "random_walk":
        return [[random_walk, random_walk], [random_walk, random_walk]], None
    if not direction and discrete_var == "uniform":
        return [[uniform, uniform], [uniform, uniform]], None

    raise ValueError(f"Unsupported discrete_var={discrete_var!r}.")


def make_classifier(
    *,
    place_bin_size: float,
    track_graph: Any,
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
    continuous_transition_types: list[list[Any]],
    observation_models: list[Any] | None,
    position_std: float,
) -> Any:
    """Construct one RTC SortedSpikesClassifier."""
    import replay_trajectory_classification as rtc

    environment = rtc.Environment(
        place_bin_size=place_bin_size,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )
    return rtc.SortedSpikesClassifier(
        environments=environment,
        continuous_transition_types=continuous_transition_types,
        observation_models=observation_models,
        sorted_spikes_algorithm="spiking_likelihood_kde_gpu",
        sorted_spikes_algorithm_params={
            "position_std": position_std,
            "use_diffusion": False,
            "block_size": int(2**11),
        },
    )


def load_classifier(path: Path) -> Any:
    """Load one saved RTC SortedSpikesClassifier."""
    import replay_trajectory_classification as rtc

    return rtc.SortedSpikesClassifier.load_model(filename=path)


def concatenate_fold_results(
    fold_results: list[Any],
    *,
    time_grid: np.ndarray,
    linear_position: np.ndarray,
    speed: np.ndarray,
    fold_by_time: np.ndarray,
) -> Any:
    """Concatenate fold prediction datasets and add aligned metadata variables."""
    import xarray as xr

    if not fold_results:
        raise ValueError("No fold prediction results were provided.")

    combined = xr.concat(
        fold_results,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
    ).sortby("time")
    predicted_time = np.asarray(combined.time, dtype=float)
    expected_time = np.asarray(time_grid, dtype=float)
    if predicted_time.size != expected_time.size:
        raise ValueError(
            "Combined predictions do not cover the full trimmed epoch. "
            f"Predicted {predicted_time.size} bins; expected {expected_time.size}."
        )
    if predicted_time.size > 1 and np.any(np.diff(predicted_time) <= 0):
        raise ValueError("Combined prediction time coordinate is not strictly increasing.")
    if not np.allclose(predicted_time, expected_time, rtol=0.0, atol=1e-9):
        raise ValueError("Combined prediction times do not match the expected time grid.")

    return combined.assign(
        true_linear_position=("time", np.asarray(linear_position, dtype=float)),
        speed_cm_s=("time", np.asarray(speed, dtype=float)),
        cv_fold=("time", np.asarray(fold_by_time, dtype=int)),
    )
