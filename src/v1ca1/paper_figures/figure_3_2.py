from __future__ import annotations

"""Generate Figure 3_2 with dark-light example cells and shuffle controls."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import v1ca1.paper_figures.figure_3 as figure_3
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DEFAULT_LIGHT_EPOCH,
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    build_normalized_position_bins,
    compute_unit_movement_firing_rates,
    extract_tuning_curve_arrays,
    get_stability_table_path,
    get_unit_spike_times,
    normalize_linear_position_by_trajectory,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PANEL_AB_HEIGHT_MM,
    DEFAULT_PANEL_DEF_HEIGHT_MM,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    FIGURE_FORMATS,
    PANEL_A_EXAMPLES,
    PANEL_DEF_WIDTH_RATIOS,
    PANEL_DEF_WSPACE,
    PANEL_TRAJECTORY_COLORS,
    build_output_path,
    get_dark_epoch,
    get_light_epoch,
    load_panel_a_example_data,
    load_panel_quantification_data,
    parse_dataset_id,
    plot_panel_c_vision_tuning_panel,
    plot_panel_d_route_place_panel,
)
from v1ca1.paper_figures.style import (
    NEUTRAL_COLORS,
    apply_paper_style,
    figure_size,
    save_figure,
)
from v1ca1.paper_figures.supplementary_figure_3 import (
    DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ,
    compute_tuning_curve_correlation,
)
from v1ca1.raster.plot_place_field_heatmap import (
    build_linear_position_by_trajectory,
    compute_place_tuning_curve,
    prepare_heatmap_session,
    smooth_values_nan_aware,
)


DEFAULT_OUTPUT_NAME = "figure_3_2"
DEFAULT_FIGURE_HEIGHT_MM = figure_3.DEFAULT_FIGURE_HEIGHT_MM
PANEL_AB_WIDTH_RATIOS = (0.58, 0.42)
PANEL_AB_HEADER_Y_OFFSET = 0.018
PANEL_A_ADDITIONAL_EXAMPLES = (
    ("L15", "20241121", "v1", 426, ("center_to_right", "left_to_center")),
    ("L15", "20241121", "v1", 169, ("center_to_left", "right_to_center")),
)
PANEL_A_EXAMPLE_GRID_COLUMNS = 2
PANEL_A_EXAMPLE_GRID_WSPACE = 0.00
PANEL_A_EXAMPLE_GRID_HSPACE = 0.04
PANEL_B_CORRELATION_GRID_BOUNDS = (0.06, 0.08, 0.91, 0.75)
PANEL_B_CORRELATION_GRID_WSPACE = 0.30
PANEL_B_CORRELATION_GRID_HSPACE = 0.20
PANEL_B_N_SHUFFLES = 200
PANEL_B_SHUFFLE_SEED = 20240611
PANEL_B_MIN_TUNING_STABILITY_CORRELATION = 0.5
PANEL_B_SHIFT_FRACTION_BOUNDS = (0.30, 0.60)
PANEL_B_PERCENTILE_BINS = np.linspace(0.0, 100.0, 21)
PANEL_B_SIGNIFICANT_PERCENTILE = 95.0
PANEL_B_OBSERVED_COLOR = "black"
PANEL_B_SHUFFLE_COLOR = NEUTRAL_COLORS["nonsignificant"]
PANEL_B_CACHE_VERSION = 1
PANEL_B_CACHE_PREFIX = "figure_3_2_panel_b_circular_shuffle"
PANEL_B_CACHE_METADATA_KEY = "__metadata__"
PANEL_B_CACHE_DATASET_TOKEN_LIMIT = 120
PANEL_B_OBSERVED_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_epoch",
    "trajectory_type",
    "unit",
    "dark_movement_firing_rate_hz",
    "light_movement_firing_rate_hz",
    "dark_stability_correlation",
    "light_stability_correlation",
    "correlation",
)
PANEL_B_SHUFFLE_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_epoch",
    "trajectory_type",
    "shuffle",
    "unit",
    "shift_fraction_min",
    "shift_fraction_max",
    "correlation",
)


def _fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return histogram weights that sum to one."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _movement_rate_lookup(movement_firing_rates: dict[Any, float]) -> dict[int, float]:
    """Return movement firing rates keyed by integer unit id."""
    return {
        int(unit_id): float(rate)
        for unit_id, rate in movement_firing_rates.items()
    }


def _interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned interval starts and ends from an IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).reshape(-1)
    ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    if starts.shape != ends.shape:
        raise ValueError(
            "IntervalSet start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts, ends


def _make_interval_set(starts: Sequence[float], ends: Sequence[float]) -> Any:
    """Return a pynapple IntervalSet from aligned starts and ends."""
    import pynapple as nap

    return nap.IntervalSet(
        start=np.asarray(starts, dtype=float),
        end=np.asarray(ends, dtype=float),
        time_units="s",
    )


def _format_cache_token(value: object) -> str:
    """Return a filesystem-safe token for Figure 3_2 cache paths."""
    return figure_3._format_panel_b_cache_token(value)


def _format_cache_number(value: float | int) -> str:
    """Return a compact numeric token for Figure 3_2 cache paths."""
    return figure_3._format_panel_b_cache_number(value)


def _build_dataset_cache_token(dataset_metadata: Sequence[dict[str, str]]) -> str:
    """Return a descriptive token for the cached data-set list."""
    dataset_tokens = [
        _format_cache_token(
            (
                f"{dataset['animal_name']}-{dataset['date']}-"
                f"dark{dataset['dark_epoch']}-light{dataset['light_epoch']}"
            )
        )
        for dataset in dataset_metadata
    ]
    token = "_".join(dataset_tokens) or "none"
    if len(token) <= PANEL_B_CACHE_DATASET_TOKEN_LIMIT:
        return token

    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    prefix = "_".join(dataset_tokens[:2])
    return _format_cache_token(f"{prefix}_{len(dataset_tokens)}datasets_{digest}")


def build_panel_b_cache_metadata(
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
    n_shuffles: int,
    shuffle_seed: int,
    min_movement_firing_rate_hz: float,
    min_tuning_stability_correlation: float,
    shift_fraction_bounds: tuple[float, float],
) -> dict[str, Any]:
    """Return metadata that identifies one Panel B circular-shuffle cache."""
    dataset_metadata = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_metadata.append(
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": get_dark_epoch(animal_name, date, dark_epoch),
                "light_epoch": get_light_epoch(animal_name, date, light_epoch),
            }
        )

    return {
        "cache_version": PANEL_B_CACHE_VERSION,
        "figure": DEFAULT_OUTPUT_NAME,
        "panel": "B",
        "payload": "observed_and_circular_light_spike_shuffle_correlations",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "light_epoch_argument": light_epoch,
        "dark_epoch_argument": dark_epoch,
        "datasets": dataset_metadata,
        "trajectory_types": list(figure_3.PANEL_B_TRAJECTORY_TYPES),
        "position_bin_count": int(position_bin_count),
        "position_offset": int(position_offset),
        "speed_threshold_cm_s": float(speed_threshold_cm_s),
        "sigma_bins": float(sigma_bins),
        "n_shuffles": int(n_shuffles),
        "shuffle_seed": int(shuffle_seed),
        "min_movement_firing_rate_hz": float(min_movement_firing_rate_hz),
        "min_tuning_stability_correlation": float(
            min_tuning_stability_correlation
        ),
        "shift_fraction_bounds": [
            float(shift_fraction_bounds[0]),
            float(shift_fraction_bounds[1]),
        ],
        "shuffle_unit": "light_spike_times",
        "shuffle_scope": "per_unit_per_trajectory_per_trial",
    }


def build_panel_b_cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    """Return the descriptive cache path for Panel B circular-shuffle payload."""
    dataset_token = _build_dataset_cache_token(metadata["datasets"])
    shift_bounds = metadata["shift_fraction_bounds"]
    filename = (
        f"{PANEL_B_CACHE_PREFIX}_{_format_cache_token(metadata['region'])}"
        f"_datasets-{dataset_token}"
        f"_shuff{int(metadata['n_shuffles'])}"
        f"_seed{int(metadata['shuffle_seed'])}"
        f"_minmovefr{_format_cache_number(metadata['min_movement_firing_rate_hz'])}"
        f"_minstab{_format_cache_number(metadata['min_tuning_stability_correlation'])}"
        f"_shift{_format_cache_number(shift_bounds[0])}-"
        f"{_format_cache_number(shift_bounds[1])}"
        f"_posbins{int(metadata['position_bin_count'])}"
        f"_offset{int(metadata['position_offset'])}"
        f"_speed{_format_cache_number(metadata['speed_threshold_cm_s'])}"
        f"_sigma{_format_cache_number(metadata['sigma_bins'])}"
        f"_cachev{int(metadata['cache_version'])}.npz"
    )
    return Path(cache_dir) / filename


def _table_to_json_payload(table: Any) -> str:
    """Return a JSON table payload preserving columns and row records."""
    return json.dumps(
        {
            "columns": [str(column) for column in table.columns],
            "records": table.to_dict("records"),
        },
        sort_keys=True,
    )


def _table_from_json_payload(payload: str) -> Any:
    """Return a pandas table from a JSON cache payload."""
    import pandas as pd

    decoded = json.loads(payload)
    return pd.DataFrame(decoded["records"], columns=decoded["columns"])


def save_panel_b_cache(
    cache_path: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write one Panel B circular-shuffle correlation cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{
            PANEL_B_CACHE_METADATA_KEY: np.asarray(
                json.dumps(metadata, sort_keys=True)
            ),
            "observed_table": np.asarray(
                _table_to_json_payload(payload["observed"])
            ),
            "shuffle_table": np.asarray(_table_to_json_payload(payload["shuffle"])),
        },
    )


def load_panel_b_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return cached Panel B correlations when metadata still matches."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data[PANEL_B_CACHE_METADATA_KEY].item()))
            if cached_metadata != expected_metadata:
                print(f"Ignoring stale Figure 3_2 Panel B cache at {cache_path}.")
                return None
            return {
                "observed": _table_from_json_payload(
                    str(data["observed_table"].item())
                ),
                "shuffle": _table_from_json_payload(
                    str(data["shuffle_table"].item())
                ),
            }
    except Exception as exc:
        print(f"Ignoring unreadable Figure 3_2 Panel B cache at {cache_path}: {exc}")
        return None


def load_epoch_stability_by_trajectory(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_tuning_stability_correlation: float,
) -> dict[str, dict[int, float]]:
    """Return stable unit IDs and correlations for one epoch by trajectory."""
    import pandas as pd

    if min_tuning_stability_correlation < -1.0:
        raise ValueError("min_tuning_stability_correlation must be at least -1.")
    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(table_path)
    required = ("unit", "region", "epoch", "trajectory_type", "stability_correlation")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Stability table {table_path} is missing columns {missing!r}.")

    correlations = pd.to_numeric(table["stability_correlation"], errors="coerce")
    units = pd.to_numeric(table["unit"], errors="coerce")
    filtered = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & (table["trajectory_type"].astype(str).isin(figure_3.PANEL_B_TRAJECTORY_TYPES))
        & np.isfinite(correlations.to_numpy(dtype=float))
        & np.isfinite(units.to_numpy(dtype=float))
        & (correlations.to_numpy(dtype=float) >= float(min_tuning_stability_correlation))
    ].copy()
    if filtered.empty:
        return {trajectory: {} for trajectory in figure_3.PANEL_B_TRAJECTORY_TYPES}

    filtered["unit"] = units.loc[filtered.index].astype(int)
    filtered["stability_correlation"] = correlations.loc[filtered.index].astype(float)
    result: dict[str, dict[int, float]] = {}
    for trajectory_type in figure_3.PANEL_B_TRAJECTORY_TYPES:
        rows = filtered[filtered["trajectory_type"].astype(str) == trajectory_type]
        grouped = rows.groupby("unit")["stability_correlation"].max()
        result[trajectory_type] = {
            int(unit): float(correlation)
            for unit, correlation in grouped.items()
        }
    return result


def compute_epoch_tuning_curve_set(
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
) -> dict[str, Any]:
    """Compute normalized all-trial tuning curves for one epoch."""
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
        use_trajectory_direction=True,
    )
    normalized_position_by_trajectory = normalize_linear_position_by_trajectory(
        animal_name,
        linear_position_by_trajectory,
    )
    bin_edges = build_normalized_position_bins(position_bin_count)
    curves = {}
    for trajectory_type in figure_3.PANEL_B_TRAJECTORY_TYPES:
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
        "session": session,
        "position_by_trajectory": normalized_position_by_trajectory,
        "bin_edges": bin_edges,
        "all_curves": curves,
        "movement_firing_rates_hz": movement_firing_rates,
    }


def _eligible_units_for_trajectory(
    *,
    trajectory_type: str,
    dark_curve: Any,
    light_curve: Any,
    dark_rates: dict[int, float],
    light_rates: dict[int, float],
    dark_stability: dict[str, dict[int, float]],
    light_stability: dict[str, dict[int, float]],
    min_movement_firing_rate_hz: float,
) -> np.ndarray:
    """Return units passing curve, movement-rate, and stability filters."""
    if dark_curve is None or light_curve is None:
        return np.asarray([], dtype=int)
    dark_units, _dark_values = extract_tuning_curve_arrays(dark_curve)
    light_units, _light_values = extract_tuning_curve_arrays(light_curve)
    dark_stable_units = set(dark_stability.get(trajectory_type, {}))
    light_stable_units = set(light_stability.get(trajectory_type, {}))
    units = []
    for unit in sorted(set(map(int, dark_units)).intersection(map(int, light_units))):
        if unit not in dark_stable_units or unit not in light_stable_units:
            continue
        if dark_rates.get(unit, 0.0) < float(min_movement_firing_rate_hz):
            continue
        if light_rates.get(unit, 0.0) < float(min_movement_firing_rate_hz):
            continue
        units.append(int(unit))
    return np.asarray(units, dtype=int)


def build_observed_dark_light_correlation_table(
    *,
    dark_set: dict[str, Any],
    light_set: dict[str, Any],
    dark_stability: dict[str, dict[int, float]],
    light_stability: dict[str, dict[int, float]],
    min_movement_firing_rate_hz: float,
) -> Any:
    """Return observed dark/light correlations for eligible units by trajectory."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    dark_rates = _movement_rate_lookup(dark_set["movement_firing_rates_hz"])
    light_rates = _movement_rate_lookup(light_set["movement_firing_rates_hz"])
    for trajectory_type in figure_3.PANEL_B_TRAJECTORY_TYPES:
        dark_curve = dark_set["all_curves"].get(trajectory_type)
        light_curve = light_set["all_curves"].get(trajectory_type)
        eligible_units = _eligible_units_for_trajectory(
            trajectory_type=trajectory_type,
            dark_curve=dark_curve,
            light_curve=light_curve,
            dark_rates=dark_rates,
            light_rates=light_rates,
            dark_stability=dark_stability,
            light_stability=light_stability,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        )
        if eligible_units.size == 0:
            continue
        dark_units, dark_values = extract_tuning_curve_arrays(dark_curve)
        light_units, light_values = extract_tuning_curve_arrays(light_curve)
        dark_rows = {int(unit): index for index, unit in enumerate(dark_units)}
        light_rows = {int(unit): index for index, unit in enumerate(light_units)}
        for unit_id in eligible_units:
            correlation = compute_tuning_curve_correlation(
                dark_values[dark_rows[int(unit_id)]],
                light_values[light_rows[int(unit_id)]],
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
                    "dark_movement_firing_rate_hz": float(dark_rates[int(unit_id)]),
                    "light_movement_firing_rate_hz": float(light_rates[int(unit_id)]),
                    "dark_stability_correlation": float(
                        dark_stability[trajectory_type][int(unit_id)]
                    ),
                    "light_stability_correlation": float(
                        light_stability[trajectory_type][int(unit_id)]
                    ),
                    "correlation": float(correlation),
                }
            )

    if not rows:
        return pd.DataFrame(columns=PANEL_B_OBSERVED_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=PANEL_B_OBSERVED_TABLE_COLUMNS)


def build_trial_movement_chunks(
    trajectory_interval: Any,
    movement_interval: Any,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return movement chunks grouped by trajectory trial."""
    trial_starts, trial_ends = _interval_bounds(trajectory_interval)
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(trial_starts, trial_ends, strict=True):
        trial_interval = _make_interval_set([float(start)], [float(end)])
        trial_movement = trial_interval.intersect(movement_interval)
        movement_starts, movement_ends = _interval_bounds(trial_movement)
        valid = movement_ends > movement_starts
        chunks.append((movement_starts[valid], movement_ends[valid]))
    return chunks


def _spike_times_in_chunks(
    spike_times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Return spike times that fall within any interval chunk."""
    if starts.size == 0:
        return np.asarray([], dtype=float)
    mask = np.zeros(np.asarray(spike_times).shape, dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        mask |= (spike_times >= float(start)) & (spike_times < float(end))
    return np.asarray(spike_times, dtype=float)[mask]


def circular_shift_spikes_within_trial_chunks(
    spike_times: np.ndarray,
    trial_chunks: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    rng: np.random.Generator,
    shift_fraction_bounds: tuple[float, float],
) -> np.ndarray:
    """Circularly shift spikes within each trial's movement chunks."""
    min_fraction, max_fraction = map(float, shift_fraction_bounds)
    if not (0.0 <= min_fraction <= max_fraction <= 1.0):
        raise ValueError("shift_fraction_bounds must satisfy 0 <= min <= max <= 1.")

    shifted_trials: list[np.ndarray] = []
    spike_times = np.asarray(spike_times, dtype=float).reshape(-1)
    for starts, ends in trial_chunks:
        starts = np.asarray(starts, dtype=float).reshape(-1)
        ends = np.asarray(ends, dtype=float).reshape(-1)
        durations = ends - starts
        valid = durations > 0.0
        starts = starts[valid]
        ends = ends[valid]
        durations = durations[valid]
        total_duration = float(np.sum(durations))
        if total_duration <= 0.0:
            continue

        trial_spikes = _spike_times_in_chunks(spike_times, starts, ends)
        if trial_spikes.size == 0:
            continue

        chunk_offsets = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
        warped_parts = []
        for chunk_start, chunk_end, chunk_offset in zip(
            starts,
            ends,
            chunk_offsets,
            strict=True,
        ):
            in_chunk = trial_spikes[
                (trial_spikes >= chunk_start) & (trial_spikes < chunk_end)
            ]
            if in_chunk.size:
                warped_parts.append(chunk_offset + (in_chunk - chunk_start))
        if not warped_parts:
            continue

        shift_fraction = float(rng.uniform(min_fraction, max_fraction))
        direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        shifted_warped = (
            np.concatenate(warped_parts) + direction * shift_fraction * total_duration
        ) % total_duration

        cumulative_ends = np.cumsum(durations)
        chunk_indices = np.searchsorted(cumulative_ends, shifted_warped, side="right")
        chunk_indices = np.minimum(chunk_indices, starts.size - 1)
        chunk_starts_on_axis = cumulative_ends[chunk_indices] - durations[chunk_indices]
        shifted_times = (
            starts[chunk_indices] + shifted_warped - chunk_starts_on_axis
        )
        shifted_trials.append(shifted_times)

    if not shifted_trials:
        return np.asarray([], dtype=float)
    return np.sort(np.concatenate(shifted_trials).astype(float, copy=False))


def prepare_trial_chunk_maps(
    trial_chunks: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, np.ndarray | float]]:
    """Return reusable movement-time maps for trajectory trials."""
    maps: list[dict[str, np.ndarray | float]] = []
    for starts, ends in trial_chunks:
        starts = np.asarray(starts, dtype=float).reshape(-1)
        ends = np.asarray(ends, dtype=float).reshape(-1)
        durations = ends - starts
        valid = durations > 0.0
        starts = starts[valid]
        durations = durations[valid]
        if starts.size == 0:
            maps.append(
                {
                    "starts": starts,
                    "durations": durations,
                    "cumulative_ends": np.asarray([], dtype=float),
                    "total_duration": 0.0,
                }
            )
            continue
        maps.append(
            {
                "starts": starts,
                "durations": durations,
                "cumulative_ends": np.cumsum(durations),
                "total_duration": float(np.sum(durations)),
            }
        )
    return maps


def warp_spikes_to_trial_movement_axis(
    spike_times: np.ndarray,
    trial_chunks: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    """Return per-trial spike coordinates on concatenated movement-time axes."""
    spike_times = np.asarray(spike_times, dtype=float).reshape(-1)
    warped_trials: list[np.ndarray] = []
    for starts, ends in trial_chunks:
        starts = np.asarray(starts, dtype=float).reshape(-1)
        ends = np.asarray(ends, dtype=float).reshape(-1)
        durations = ends - starts
        valid = durations > 0.0
        starts = starts[valid]
        ends = ends[valid]
        durations = durations[valid]
        if starts.size == 0:
            warped_trials.append(np.asarray([], dtype=float))
            continue
        chunk_offsets = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
        warped_parts = []
        for chunk_start, chunk_end, chunk_offset in zip(
            starts,
            ends,
            chunk_offsets,
            strict=True,
        ):
            in_chunk = spike_times[
                (spike_times >= chunk_start) & (spike_times < chunk_end)
            ]
            if in_chunk.size:
                warped_parts.append(chunk_offset + (in_chunk - chunk_start))
        warped_trials.append(
            np.concatenate(warped_parts).astype(float, copy=False)
            if warped_parts
            else np.asarray([], dtype=float)
        )
    return warped_trials


def sample_shifted_trial_spike_times(
    warped_trials: Sequence[np.ndarray],
    trial_maps: Sequence[dict[str, np.ndarray | float]],
    *,
    rng: np.random.Generator,
    shift_fraction_bounds: tuple[float, float],
) -> np.ndarray:
    """Sample shifted spike times from prewarped per-trial spike coordinates."""
    min_fraction, max_fraction = map(float, shift_fraction_bounds)
    shifted_trials: list[np.ndarray] = []
    for warped, trial_map in zip(warped_trials, trial_maps, strict=True):
        warped = np.asarray(warped, dtype=float).reshape(-1)
        if warped.size == 0:
            continue
        total_duration = float(trial_map["total_duration"])
        if total_duration <= 0.0:
            continue
        starts = np.asarray(trial_map["starts"], dtype=float)
        durations = np.asarray(trial_map["durations"], dtype=float)
        cumulative_ends = np.asarray(trial_map["cumulative_ends"], dtype=float)
        shift_fraction = float(rng.uniform(min_fraction, max_fraction))
        direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        shifted_warped = (
            warped + direction * shift_fraction * total_duration
        ) % total_duration
        chunk_indices = np.searchsorted(cumulative_ends, shifted_warped, side="right")
        chunk_indices = np.minimum(chunk_indices, starts.size - 1)
        chunk_starts_on_axis = cumulative_ends[chunk_indices] - durations[chunk_indices]
        shifted_trials.append(
            starts[chunk_indices] + shifted_warped - chunk_starts_on_axis
        )
    if not shifted_trials:
        return np.asarray([], dtype=float)
    return np.sort(np.concatenate(shifted_trials).astype(float, copy=False))


def _interval_mask(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return a boolean mask for values inside any interval."""
    values = np.asarray(values, dtype=float)
    mask = np.zeros(values.shape, dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        mask |= (values >= float(start)) & (values < float(end))
    return mask


def _position_samples_in_epoch(position_tsd: Any, epoch: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return position sample times and values inside an epoch."""
    times = np.asarray(position_tsd.t, dtype=float)
    values = np.asarray(position_tsd.d, dtype=float)
    starts, ends = _interval_bounds(epoch)
    if starts.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    mask = _interval_mask(times, starts, ends)
    return times[mask], values[mask]


def compute_position_bin_occupancy_s(
    position_tsd: Any,
    epoch: Any,
    *,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Return approximate occupancy seconds per position bin."""
    times, values = _position_samples_in_epoch(position_tsd, epoch)
    if times.size < 2:
        return np.zeros(np.asarray(bin_edges).size - 1, dtype=float)
    finite = np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if times.size < 2:
        return np.zeros(np.asarray(bin_edges).size - 1, dtype=float)
    sample_dt = float(np.nanmedian(np.diff(np.sort(times))))
    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        sample_dt = 1.0
    occupancy, _edges = np.histogram(
        values,
        bins=np.asarray(bin_edges, dtype=float),
        weights=np.full(values.shape, sample_dt, dtype=float),
    )
    return np.asarray(occupancy, dtype=float)


def interpolate_position_at_spikes(
    spike_times: np.ndarray,
    position_tsd: Any,
    epoch: Any,
) -> np.ndarray:
    """Return interpolated position values for spikes inside an epoch."""
    spike_times = np.asarray(spike_times, dtype=float)
    starts, ends = _interval_bounds(epoch)
    if starts.size == 0 or spike_times.size == 0:
        return np.asarray([], dtype=float)
    spike_times = spike_times[_interval_mask(spike_times, starts, ends)]
    if spike_times.size == 0:
        return np.asarray([], dtype=float)

    position_times, position_values = _position_samples_in_epoch(position_tsd, epoch)
    finite = np.isfinite(position_values)
    position_times = position_times[finite]
    position_values = position_values[finite]
    if position_times.size < 2:
        return np.asarray([], dtype=float)
    order = np.argsort(position_times)
    position_times = position_times[order]
    position_values = position_values[order]
    return np.interp(spike_times, position_times, position_values)


def compute_shuffled_tuning_matrix(
    *,
    unit_ids: np.ndarray,
    warped_spikes_by_unit: dict[int, list[np.ndarray]],
    trial_maps: Sequence[dict[str, np.ndarray | float]],
    position_times: np.ndarray,
    position_values: np.ndarray,
    occupancy_s: np.ndarray,
    bin_edges: np.ndarray,
    rng: np.random.Generator,
    shift_fraction_bounds: tuple[float, float],
    sigma_bins: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit IDs and tuning matrix for one circular-shuffle sample."""
    unit_ids = np.asarray(unit_ids, dtype=int)
    values = np.full((unit_ids.size, occupancy_s.size), np.nan, dtype=float)
    positive_occupancy = occupancy_s > 0.0
    for row_index, unit_id in enumerate(unit_ids):
        shifted_spikes = sample_shifted_trial_spike_times(
            warped_spikes_by_unit[int(unit_id)],
            trial_maps,
            rng=rng,
            shift_fraction_bounds=shift_fraction_bounds,
        )
        if shifted_spikes.size and position_times.size >= 2:
            spike_positions = np.interp(shifted_spikes, position_times, position_values)
        else:
            spike_positions = np.asarray([], dtype=float)
        counts, _edges = np.histogram(
            spike_positions,
            bins=np.asarray(bin_edges, dtype=float),
        )
        unit_values = np.full(occupancy_s.shape, np.nan, dtype=float)
        unit_values[positive_occupancy] = (
            counts[positive_occupancy] / occupancy_s[positive_occupancy]
        )
        values[row_index] = unit_values
    if float(sigma_bins) > 0.0:
        values = smooth_values_nan_aware(values, sigma_bins=float(sigma_bins), axis=1)
    return unit_ids, values


def build_circular_light_shuffle_correlation_table(
    *,
    dark_set: dict[str, Any],
    light_set: dict[str, Any],
    dark_stability: dict[str, dict[int, float]],
    light_stability: dict[str, dict[int, float]],
    n_shuffles: int,
    shuffle_seed: int,
    min_movement_firing_rate_hz: float,
    shift_fraction_bounds: tuple[float, float],
    sigma_bins: float,
) -> Any:
    """Return light-spike circular-shuffle correlations by trajectory."""
    if n_shuffles < 0:
        raise ValueError("n_shuffles must be non-negative.")

    import pandas as pd

    rng = np.random.default_rng(shuffle_seed)
    rows: list[dict[str, Any]] = []
    dark_rates = _movement_rate_lookup(dark_set["movement_firing_rates_hz"])
    light_rates = _movement_rate_lookup(light_set["movement_firing_rates_hz"])
    light_session = light_set["session"]
    light_epoch = str(light_set["epoch"])
    light_spikes = light_session["spikes_by_region"][str(light_set["region"])]
    movement_interval = light_session["movement_by_run"][light_epoch]
    trajectory_intervals = light_session["trajectory_intervals"][light_epoch]
    shift_min, shift_max = map(float, shift_fraction_bounds)

    for trajectory_type in figure_3.PANEL_B_TRAJECTORY_TYPES:
        dark_curve = dark_set["all_curves"].get(trajectory_type)
        light_curve = light_set["all_curves"].get(trajectory_type)
        eligible_units = _eligible_units_for_trajectory(
            trajectory_type=trajectory_type,
            dark_curve=dark_curve,
            light_curve=light_curve,
            dark_rates=dark_rates,
            light_rates=light_rates,
            dark_stability=dark_stability,
            light_stability=light_stability,
            min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        )
        if eligible_units.size == 0:
            continue

        dark_units, dark_values = extract_tuning_curve_arrays(dark_curve)
        dark_rows = {int(unit): index for index, unit in enumerate(dark_units)}
        trial_chunks = build_trial_movement_chunks(
            trajectory_intervals[trajectory_type],
            movement_interval,
        )
        trajectory_epoch = trajectory_intervals[trajectory_type].intersect(
            movement_interval
        )
        position_tsd = light_set["position_by_trajectory"][trajectory_type]
        occupancy_s = compute_position_bin_occupancy_s(
            position_tsd,
            trajectory_epoch,
            bin_edges=light_set["bin_edges"],
        )
        position_times, position_values = _position_samples_in_epoch(
            position_tsd,
            trajectory_epoch,
        )
        finite_position = np.isfinite(position_values)
        position_times = position_times[finite_position]
        position_values = position_values[finite_position]
        position_order = np.argsort(position_times)
        position_times = position_times[position_order]
        position_values = position_values[position_order]
        trial_maps = prepare_trial_chunk_maps(trial_chunks)
        warped_spikes_by_unit = {
            int(unit_id): warp_spikes_to_trial_movement_axis(
                get_unit_spike_times(light_spikes, int(unit_id)),
                trial_chunks,
            )
            for unit_id in eligible_units
        }
        for shuffle_index in range(int(n_shuffles)):
            shuffled_units, shuffled_values = compute_shuffled_tuning_matrix(
                unit_ids=eligible_units,
                warped_spikes_by_unit=warped_spikes_by_unit,
                trial_maps=trial_maps,
                position_times=position_times,
                position_values=position_values,
                occupancy_s=occupancy_s,
                bin_edges=light_set["bin_edges"],
                rng=rng,
                shift_fraction_bounds=shift_fraction_bounds,
                sigma_bins=sigma_bins,
            )
            shuffled_rows = {
                int(unit): index for index, unit in enumerate(shuffled_units)
            }
            for unit_id in eligible_units:
                unit_id = int(unit_id)
                if unit_id not in dark_rows or unit_id not in shuffled_rows:
                    continue
                correlation = compute_tuning_curve_correlation(
                    dark_values[dark_rows[unit_id]],
                    shuffled_values[shuffled_rows[unit_id]],
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
                        "shuffle": int(shuffle_index),
                        "unit": unit_id,
                        "shift_fraction_min": shift_min,
                        "shift_fraction_max": shift_max,
                        "correlation": float(correlation),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=PANEL_B_SHUFFLE_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=PANEL_B_SHUFFLE_TABLE_COLUMNS)


def build_panel_b_tuning_correlation_data(
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
    n_shuffles: int,
    shuffle_seed: int,
    min_movement_firing_rate_hz: float,
    min_tuning_stability_correlation: float,
    shift_fraction_bounds: tuple[float, float],
) -> dict[str, Any]:
    """Build observed and circular-shuffled dark/light tuning correlations."""
    import pandas as pd

    observed_tables = []
    shuffle_tables = []
    for dataset_index, dataset in enumerate(datasets):
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        selected_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        selected_light_epoch = get_light_epoch(animal_name, date, light_epoch)
        dark_set = compute_epoch_tuning_curve_set(
            animal_name=animal_name,
            date=date,
            data_root=data_root,
            region=region,
            epoch=selected_dark_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        )
        light_set = compute_epoch_tuning_curve_set(
            animal_name=animal_name,
            date=date,
            data_root=data_root,
            region=region,
            epoch=selected_light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
        )
        dark_stability = load_epoch_stability_by_trajectory(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_dark_epoch,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
        )
        light_stability = load_epoch_stability_by_trajectory(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=selected_light_epoch,
            min_tuning_stability_correlation=min_tuning_stability_correlation,
        )
        observed_tables.append(
            build_observed_dark_light_correlation_table(
                dark_set=dark_set,
                light_set=light_set,
                dark_stability=dark_stability,
                light_stability=light_stability,
                min_movement_firing_rate_hz=min_movement_firing_rate_hz,
            )
        )
        shuffle_tables.append(
            build_circular_light_shuffle_correlation_table(
                dark_set=dark_set,
                light_set=light_set,
                dark_stability=dark_stability,
                light_stability=light_stability,
                n_shuffles=n_shuffles,
                shuffle_seed=int(shuffle_seed) + dataset_index,
                min_movement_firing_rate_hz=min_movement_firing_rate_hz,
                shift_fraction_bounds=shift_fraction_bounds,
                sigma_bins=sigma_bins,
            )
        )

    observed = (
        pd.concat(observed_tables, ignore_index=True)
        if observed_tables
        else pd.DataFrame(columns=PANEL_B_OBSERVED_TABLE_COLUMNS)
    )
    shuffle = (
        pd.concat(shuffle_tables, ignore_index=True)
        if shuffle_tables
        else pd.DataFrame(columns=PANEL_B_SHUFFLE_TABLE_COLUMNS)
    )
    return {"observed": observed, "shuffle": shuffle}


def load_panel_b_tuning_correlation_data(
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
    n_shuffles: int = PANEL_B_N_SHUFFLES,
    shuffle_seed: int = PANEL_B_SHUFFLE_SEED,
    min_movement_firing_rate_hz: float = (
        DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_tuning_stability_correlation: float = (
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    ),
    shift_fraction_bounds: tuple[float, float] = PANEL_B_SHIFT_FRACTION_BOUNDS,
    panel_b_cache_dir: Path | None = None,
    refresh_panel_b_cache: bool = False,
) -> dict[str, Any]:
    """Load or build observed and circular-shuffled dark/light correlations."""
    metadata = build_panel_b_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        n_shuffles=n_shuffles,
        shuffle_seed=shuffle_seed,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
        shift_fraction_bounds=shift_fraction_bounds,
    )
    cache_path = (
        build_panel_b_cache_path(panel_b_cache_dir, metadata)
        if panel_b_cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_panel_b_cache:
        cached = load_panel_b_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Figure 3_2 Panel B cache from {cache_path}.")
            return cached

    payload = build_panel_b_tuning_correlation_data(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        n_shuffles=n_shuffles,
        shuffle_seed=shuffle_seed,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
        shift_fraction_bounds=shift_fraction_bounds,
    )
    if cache_path is not None:
        save_panel_b_cache(cache_path, payload, metadata)
        print(f"Saved Figure 3_2 Panel B cache to {cache_path}.")
    return payload


PANEL_B_PERCENTILE_KEY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_epoch",
    "trajectory_type",
    "unit",
)
PANEL_B_PERCENTILE_TABLE_COLUMNS = (
    *PANEL_B_PERCENTILE_KEY_COLUMNS,
    "correlation",
    "n_shuffles",
    "percentile",
)


def build_panel_b_percentile_table(
    observed_table: Any,
    shuffle_table: Any,
) -> Any:
    """Return observed-correlation percentiles within each cell's shuffle null."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    if observed_table is None or shuffle_table is None:
        return pd.DataFrame(columns=PANEL_B_PERCENTILE_TABLE_COLUMNS)
    if len(observed_table) == 0 or len(shuffle_table) == 0:
        return pd.DataFrame(columns=PANEL_B_PERCENTILE_TABLE_COLUMNS)

    shuffle_groups = {
        key: group["correlation"].to_numpy(dtype=float)
        for key, group in shuffle_table.groupby(
            list(PANEL_B_PERCENTILE_KEY_COLUMNS),
            sort=False,
        )
    }
    for _row_index, row in observed_table.iterrows():
        key = tuple(row[column] for column in PANEL_B_PERCENTILE_KEY_COLUMNS)
        null_values = np.asarray(shuffle_groups.get(key, []), dtype=float)
        null_values = null_values[np.isfinite(null_values)]
        observed = float(row["correlation"])
        if not np.isfinite(observed) or null_values.size == 0:
            continue
        percentile = (
            (np.count_nonzero(null_values <= observed) + 1)
            / (null_values.size + 1)
            * 100.0
        )
        rows.append(
            {
                **{
                    column: row[column]
                    for column in PANEL_B_PERCENTILE_KEY_COLUMNS
                },
                "correlation": observed,
                "n_shuffles": int(null_values.size),
                "percentile": float(percentile),
            }
        )
    return pd.DataFrame(rows, columns=PANEL_B_PERCENTILE_TABLE_COLUMNS)


def _finite_percentile_values(table: Any) -> np.ndarray:
    """Return finite percentile values from a table-like object."""
    if table is None or "percentile" not in table:
        return np.asarray([], dtype=float)
    values = np.asarray(table["percentile"], dtype=float)
    return values[np.isfinite(values)]


def _percentile_histogram_fractions(values: np.ndarray) -> np.ndarray:
    """Return fractional histogram bin heights for percentile values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=float)
    counts, _bin_edges = np.histogram(
        values,
        bins=PANEL_B_PERCENTILE_BINS,
        weights=_fraction_histogram_weights(values),
    )
    return np.asarray(counts, dtype=float)


def _panel_b_shared_percentile_y_limit(percentile_table: Any) -> float:
    """Return one y-axis limit for all Panel B percentile histograms."""
    max_fraction = 0.0
    for trajectory_type in figure_3.PANEL_B_TRAJECTORY_TYPES:
        trajectory_table = percentile_table[
            percentile_table["trajectory_type"].astype(str) == str(trajectory_type)
        ]
        bin_fractions = _percentile_histogram_fractions(
            _finite_percentile_values(trajectory_table)
        )
        if bin_fractions.size:
            max_fraction = max(max_fraction, float(np.nanmax(bin_fractions)))
    if not np.isfinite(max_fraction) or max_fraction <= 0.0:
        return 0.1
    return max(0.1, max_fraction * 1.15)


def plot_panel_b_percentile_distribution(
    ax: Any,
    percentile_table: Any,
    *,
    trajectory_type: str | None = None,
    show_ylabel: bool = True,
    y_limit: float | None = None,
) -> None:
    """Plot observed correlation percentiles within each cell's shuffle null."""
    if trajectory_type is not None:
        percentile_table = percentile_table[
            percentile_table["trajectory_type"].astype(str) == str(trajectory_type)
        ]
    percentile_values = _finite_percentile_values(percentile_table)
    significant_fraction = (
        float(np.mean(percentile_values >= PANEL_B_SIGNIFICANT_PERCENTILE))
        if percentile_values.size
        else float("nan")
    )
    ax.axvline(
        PANEL_B_SIGNIFICANT_PERCENTILE,
        color=PANEL_B_SHUFFLE_COLOR,
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )
    if percentile_values.size:
        ax.hist(
            percentile_values,
            bins=PANEL_B_PERCENTILE_BINS,
            weights=_fraction_histogram_weights(percentile_values),
            histtype="step",
            color=PANEL_B_OBSERVED_COLOR,
            linewidth=1.0,
            zorder=2,
        )

    median_percentile = (
        float(np.nanmedian(percentile_values))
        if percentile_values.size
        else float("nan")
    )
    median_text = (
        f"med. {median_percentile:.0f}%"
        if np.isfinite(median_percentile)
        else "med. n/a"
    )
    if trajectory_type is None:
        title = "Light-dark tuning similarity"
        title_color = "0.20"
    else:
        title = figure_3.PANEL_TRAJECTORY_LABELS.get(
            trajectory_type,
            trajectory_type,
        )
        title_color = PANEL_TRAJECTORY_COLORS.get(trajectory_type, "0.20")
    ax.text(
        0.0,
        1.095,
        title,
        ha="left",
        va="bottom",
        fontsize=5.8,
        transform=ax.transAxes,
        color=title_color,
        clip_on=False,
    )
    ax.text(
        0.0,
        1.015,
        f"{median_text}, n = {percentile_values.size}",
        ha="left",
        va="bottom",
        fontsize=4.6,
        transform=ax.transAxes,
        color="0.25",
        clip_on=False,
    )
    fraction_text = (
        f">=95%: {significant_fraction:.0%}"
        if np.isfinite(significant_fraction)
        else ">=95%: n/a"
    )
    ax.text(
        0.04,
        0.92,
        fraction_text,
        ha="left",
        va="top",
        fontsize=4.8,
        transform=ax.transAxes,
        color="0.25",
    )
    ax.set_xlim(0.0, 100.0)
    if y_limit is not None and np.isfinite(float(y_limit)):
        ax.set_ylim(0.0, float(y_limit))
    ax.set_xlabel("Observed corr. percentile", fontsize=6.2, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Frac. cells", fontsize=6.4, labelpad=1.5)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.8, length=1.8, pad=1)


def plot_panel_b_percentile_grid(
    ax: Any,
    percentile_table: Any,
) -> None:
    """Plot trajectory-wise within-cell percentile distributions."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    n_cols = 2
    n_rows = 2
    cell_width = (1.0 - PANEL_B_CORRELATION_GRID_WSPACE) / n_cols
    cell_height = (1.0 - PANEL_B_CORRELATION_GRID_HSPACE) / n_rows
    y_limit = _panel_b_shared_percentile_y_limit(percentile_table)
    axes = []
    for index, trajectory_type in enumerate(figure_3.PANEL_B_TRAJECTORY_TYPES):
        row = index // n_cols
        col = index % n_cols
        x0 = col * (cell_width + PANEL_B_CORRELATION_GRID_WSPACE)
        y0 = 1.0 - (row + 1) * cell_height - row * PANEL_B_CORRELATION_GRID_HSPACE
        child_ax = ax.inset_axes([x0, y0, cell_width, cell_height])
        plot_panel_b_percentile_distribution(
            child_ax,
            percentile_table,
            trajectory_type=trajectory_type,
            show_ylabel=col == 0,
            y_limit=y_limit,
        )
        axes.append(child_ax)
    return None


def add_panel_b_similarity_legend(ax: Any) -> None:
    """Add the shared Panel B percentile-distribution legend."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0.0],
            [0.0],
            color=PANEL_B_OBSERVED_COLOR,
            linewidth=1.0,
        ),
        Line2D(
            [0.0],
            [0.0],
            color=PANEL_B_SHUFFLE_COLOR,
            linestyle="--",
            linewidth=0.8,
        ),
    ]
    ax.legend(
        handles,
        ["Cells", "p <= 0.05"],
        frameon=False,
        fontsize=4.8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        handlelength=1.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )


def plot_panel_a_example_grid(
    ax: Any,
    examples: Sequence[dict[str, Any]],
    *,
    n_columns: int = PANEL_A_EXAMPLE_GRID_COLUMNS,
) -> None:
    """Plot Figure 3_2 single-unit examples in a compact grid."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center", transform=ax.transAxes)
        return

    n_columns = max(1, int(n_columns))
    n_rows = int(np.ceil(len(examples) / n_columns))
    cell_width = (1.0 - PANEL_A_EXAMPLE_GRID_WSPACE * (n_columns - 1)) / n_columns
    cell_height = (1.0 - PANEL_A_EXAMPLE_GRID_HSPACE * (n_rows - 1)) / n_rows
    for index, example in enumerate(examples):
        row = index // n_columns
        col = index % n_columns
        x0 = col * (cell_width + PANEL_A_EXAMPLE_GRID_WSPACE)
        y0 = 1.0 - (row + 1) * cell_height - row * PANEL_A_EXAMPLE_GRID_HSPACE
        child_ax = ax.inset_axes([x0, y0, cell_width, cell_height])
        figure_3.plot_panel_a_example(
            child_ax,
            example,
            title=f"Cell {index + 1}",
            y_shift=0.0,
        )


def plot_panel_b_similarity_panel(
    ax: Any,
    *,
    observed_table: Any,
    shuffle_table: Any,
) -> None:
    """Plot Panel B per-cell light/dark tuning similarity percentile."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    percentile_table = build_panel_b_percentile_table(
        observed_table,
        shuffle_table,
    )
    correlation_ax = ax.inset_axes(PANEL_B_CORRELATION_GRID_BOUNDS)
    plot_panel_b_percentile_grid(
        correlation_ax,
        percentile_table,
    )
    add_panel_b_similarity_legend(ax)


def load_panel_a_additional_examples(
    *,
    data_root: Path,
    dark_epoch: str | None,
    light_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path | None,
    refresh_panel_example_cache: bool,
) -> list[dict[str, Any]]:
    """Load the additional Figure 3_2 Panel A example cells."""
    return [
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
        for animal_name, date, region, unit_id, trajectories in PANEL_A_ADDITIONAL_EXAMPLES
    ]


def make_figure_3_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    dpi: int,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    panel_b_cache_dir: Path | None = None,
    refresh_panel_b_cache: bool = False,
    panel_b_n_shuffles: int = PANEL_B_N_SHUFFLES,
    panel_b_shuffle_seed: int = PANEL_B_SHUFFLE_SEED,
) -> Path:
    """Build and save Figure 3_2."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    panel_b_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_b_cache_dir is None
        else Path(panel_b_cache_dir)
    )
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    panel_quant_payload = load_panel_quantification_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    panel_b_correlation_data = load_panel_b_tuning_correlation_data(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        n_shuffles=panel_b_n_shuffles,
        shuffle_seed=panel_b_shuffle_seed,
        panel_b_cache_dir=panel_b_cache_dir,
        refresh_panel_b_cache=refresh_panel_b_cache,
    )

    apply_paper_style()
    fig_height_mm = (
        DEFAULT_PANEL_AB_HEIGHT_MM * max(len(regions), 1)
    ) + DEFAULT_PANEL_DEF_HEIGHT_MM
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[
            DEFAULT_PANEL_AB_HEIGHT_MM * max(len(regions), 1),
            DEFAULT_PANEL_DEF_HEIGHT_MM,
        ],
    )
    middle_grid = outer_grid[0, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_AB_WIDTH_RATIOS,
    )
    panel_a_axis = fig.add_subplot(middle_grid[0, 0])
    panel_b_axis = fig.add_subplot(middle_grid[0, 1])
    bottom_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_DEF_WIDTH_RATIOS,
        wspace=PANEL_DEF_WSPACE,
    )
    panel_c_container_axis = fig.add_subplot(bottom_grid[0, 0])
    panel_d_container_axis = fig.add_subplot(bottom_grid[0, 1])

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
        for animal_name, date, region, unit_id, trajectories in PANEL_A_EXAMPLES
    ]
    panel_a_examples.extend(
        load_panel_a_additional_examples(
            data_root=data_root,
            dark_epoch=dark_epoch,
            light_epoch=light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
    )

    plot_panel_a_example_grid(panel_a_axis, panel_a_examples)
    plot_panel_b_similarity_panel(
        panel_b_axis,
        observed_table=panel_b_correlation_data["observed"],
        shuffle_table=panel_b_correlation_data["shuffle"],
    )
    plot_panel_c_vision_tuning_panel(
        panel_c_container_axis,
        panel_quant_payload["similarity"],
        panel_quant_payload["decoding_error"],
    )
    plot_panel_d_route_place_panel(
        panel_d_container_axis,
        panel_quant_payload["encoding_delta"],
        panel_quant_payload["decoding_error"],
    )

    fig.canvas.draw()
    panel_ab_header_y = (
        figure_3._axis_group_top_y([panel_a_axis, panel_b_axis])
        + PANEL_AB_HEADER_Y_OFFSET
    )
    figure_3._add_panel_label_at_figure_y(
        fig,
        panel_a_axis,
        "A",
        x=-0.07,
        y=panel_ab_header_y,
    )
    figure_3._add_panel_label_at_figure_y(
        fig,
        panel_b_axis,
        "B",
        x=-0.04,
        y=panel_ab_header_y,
    )
    figure_3._add_panel_cd_label(panel_c_container_axis, "C")
    figure_3._add_panel_cd_label(panel_d_container_axis, "D")
    panel_a_title_x, _panel_a_title_y = figure_3._axis_to_figure_coordinates(
        fig,
        panel_a_axis,
        0.5,
        0.0,
    )
    panel_b_title_x, _panel_b_title_y = figure_3._axis_to_figure_coordinates(
        fig,
        panel_b_axis,
        0.5,
        0.0,
    )
    fig.text(
        panel_a_title_x,
        panel_ab_header_y,
        "Example DPP cells in dark and light",
        ha="center",
        va="center",
        fontsize=7.2,
        linespacing=1.0,
    )
    fig.text(
        panel_b_title_x,
        panel_ab_header_y,
        "Light-dark tuning similarity",
        ha="center",
        va="center",
        fontsize=7.2,
        linespacing=1.0,
    )
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 3_2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 3_2 generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure 3_2 dark/light example cells and tuning-shuffle control."
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
        "--panel-b-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached Panel B circular-shuffle correlations. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-b-cache",
        action="store_true",
        help=(
            "Recompute Panel B circular-shuffle correlations and overwrite "
            "matching caches."
        ),
    )
    parser.add_argument(
        "--panel-b-n-shuffles",
        type=int,
        default=PANEL_B_N_SHUFFLES,
        help=(
            "Number of circular light-spike shuffles for Panel B. "
            f"Default: {PANEL_B_N_SHUFFLES}"
        ),
    )
    parser.add_argument(
        "--panel-b-shuffle-seed",
        type=int,
        default=PANEL_B_SHUFFLE_SEED,
        help=(
            "Random seed for Panel B circular light-spike shuffles. "
            f"Default: {PANEL_B_SHUFFLE_SEED}"
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
            "Light run epoch for example and correlation panels. "
            f"Default: registry value, currently {DEFAULT_LIGHT_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help=(
            "Dark run epoch for example and correlation panels. "
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
    """Run Figure 3_2 generation."""
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
    panel_b_cache_dir = (
        args.panel_b_cache_dir
        if args.panel_b_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_3_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        dpi=args.dpi,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
        panel_b_cache_dir=panel_b_cache_dir,
        refresh_panel_b_cache=args.refresh_panel_b_cache,
        panel_b_n_shuffles=args.panel_b_n_shuffles,
        panel_b_shuffle_seed=args.panel_b_shuffle_seed,
    )


if __name__ == "__main__":
    main()
