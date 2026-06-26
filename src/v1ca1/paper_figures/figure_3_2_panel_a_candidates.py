from __future__ import annotations

"""Find and plot Figure 3_2 Panel A-like dark/light candidate cells."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import v1ca1.paper_figures.figure_2 as figure_2
import v1ca1.paper_figures.figure_3 as figure_3
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
)
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_POSITION_BIN_COUNT,
    PANEL_D_TRAJECTORY_TYPES,
    PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION,
    get_stability_table_path,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIGMA_BINS,
    FIGURE_FORMATS,
    PANEL_C_SIMILARITY_COMPARISON_LABELS,
    build_output_path,
    load_panel_a_example_data,
    parse_dataset_id,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    save_figure,
)
from v1ca1.task_progression.tuning_analysis import LABEL_TO_SPEC


DEFAULT_OUTPUT_NAME = "figure_3_2_panel_a_candidates"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_CANDIDATE_COUNT = 8
DEFAULT_CANDIDATE_COLUMNS = 2
DEFAULT_MIN_DARK_SIMILARITY = 0.6
DEFAULT_MAX_LIGHT_SIMILARITY = 0.25
DEFAULT_MIN_LIGHT_SIMILARITY = 0.4
DEFAULT_MIN_DARK_LIGHT_DELTA = 0.0
DEFAULT_MIN_MOVEMENT_FIRING_RATE_HZ = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ
DEFAULT_MIN_TUNING_STABILITY_CORRELATION = (
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION
)
DEFAULT_MIN_SHARED_SCAFFOLD_DELTA = None
DEFAULT_SWAP_MODEL_NAME = figure_2.PANEL_C_SWAP_MODEL_NAME
DEFAULT_SWAP_DELTA_MIN_TUNING_STABILITY_CORRELATION = (
    figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
)
DEFAULT_REQUIRE_EXISTING_EXAMPLE_CACHE = False
DEFAULT_PLOT_MODE = "panel-a"
PLOT_MODES = ("panel-a", "curves")
DEFAULT_SELECTION_MODE = "dark_high_light_low"
SELECTION_MODES = ("dark_high_light_low", "dark_high_light_high")
DEFAULT_DIVERSIFY_BY_PEAK = False
DEFAULT_PLOT_BOTH_TURN_DIRECTIONS = False
DEFAULT_SPLIT_TURN_DIRECTIONS = False
DEFAULT_PEAK_BIN_COUNT = 10
CANDIDATE_TABLE_SUFFIX = "_candidates.parquet"
CANDIDATE_FIGURE_WIDTH_MM = 165.0
CANDIDATE_ROW_HEIGHT_MM = 45.0
CANDIDATE_MIN_FIGURE_HEIGHT_MM = 55.0
CANDIDATE_GRID_WSPACE = 0.025
CANDIDATE_GRID_HSPACE = 0.030
CURVE_GALLERY_ROW_HEIGHT_MM = 25.0
CURVE_GALLERY_MIN_FIGURE_HEIGHT_MM = 45.0
CURVE_GALLERY_TITLE_FONTSIZE = 4.9
CURVE_GALLERY_AXIS_TITLE_FONTSIZE = 4.8
CURVE_GALLERY_AXIS_LABEL_FONTSIZE = 4.2
CANDIDATE_TURN_DIRECTION_TRAJECTORIES = (
    ("Left turn", ("center_to_left", "right_to_center")),
    ("Right turn", ("center_to_right", "left_to_center")),
)
CANDIDATE_TABLE_COLUMNS = (
    "rank",
    "animal_name",
    "date",
    "region",
    "unit",
    "comparison_label",
    "trajectory_a",
    "trajectory_b",
    "similarity_dark",
    "similarity_light",
    "dark_light_delta",
    "dark_peak_position",
    "dark_peak_bin",
)
CANDIDATE_OPTIONAL_TABLE_COLUMNS = (
    "dark_movement_firing_rate_hz",
    "max_dark_tuning_stability_correlation",
    "pair_delta_ll_bits_per_spike",
    "n_pair_delta_trajectories",
)
EXCLUDE_CANDIDATE_KEY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "comparison_label",
)


def get_candidate_trajectories(comparison_label: str) -> tuple[str, str]:
    """Return the two same-turn trajectories for one comparison label."""
    if comparison_label not in PANEL_C_SIMILARITY_COMPARISON_LABELS:
        raise ValueError(
            f"Expected a same-turn comparison label, got {comparison_label!r}."
        )
    spec = LABEL_TO_SPEC[str(comparison_label)]
    return str(spec["trajectory_a"]), str(spec["trajectory_b"])


def build_candidate_similarity_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
) -> Any:
    """Return paired same-turn dark/light similarity rows with a drop score."""
    similarity_table = figure_3.load_panel_c_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    pairs = figure_3.build_panel_c_similarity_pairs(similarity_table)
    if pairs.empty:
        return pairs.assign(dark_light_delta=np.asarray([], dtype=float))

    pairs = pairs.copy()
    pairs["region"] = str(region)
    pairs["similarity_dark"] = pairs["similarity_dark"].astype(float)
    pairs["similarity_light"] = pairs["similarity_light"].astype(float)
    pairs["dark_light_delta"] = (
        pairs["similarity_dark"] - pairs["similarity_light"]
    )
    pairs["trajectory_a"] = [
        get_candidate_trajectories(str(label))[0]
        for label in pairs["comparison_label"]
    ]
    pairs["trajectory_b"] = [
        get_candidate_trajectories(str(label))[1]
        for label in pairs["comparison_label"]
    ]
    return pairs


def load_dark_movement_rate_metric_table(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    cache_dir: Path | None,
) -> Any:
    """Return per-unit dark movement firing rates with candidate keys."""
    import pandas as pd

    table = figure_2.load_panel_b_dark_movement_firing_rate_table(
        data_root,
        animal_name=animal_name,
        date=date,
        dark_epoch=dark_epoch,
        region=region,
        cache_dir=cache_dir,
        refresh_cache=False,
    )
    missing_columns = [
        column for column in ("unit", "dark_firing_rate_hz") if column not in table
    ]
    if missing_columns:
        raise ValueError(
            "Dark movement firing-rate table is missing columns "
            f"{missing_columns!r}."
        )

    rows = table.loc[:, ["unit", "dark_firing_rate_hz"]].copy()
    rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
    rows["dark_firing_rate_hz"] = pd.to_numeric(
        rows["dark_firing_rate_hz"],
        errors="coerce",
    )
    rows = rows[
        np.isfinite(rows["unit"].to_numpy(dtype=float))
        & np.isfinite(rows["dark_firing_rate_hz"].to_numpy(dtype=float))
    ].copy()
    rows["unit"] = rows["unit"].astype(int)
    rows["animal_name"] = str(animal_name)
    rows["date"] = str(date)
    rows["region"] = str(region)
    return rows.rename(
        columns={"dark_firing_rate_hz": "dark_movement_firing_rate_hz"}
    ).loc[
        :,
        [
            "animal_name",
            "date",
            "region",
            "unit",
            "dark_movement_firing_rate_hz",
        ],
    ]


def load_dark_tuning_stability_metric_table(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
) -> Any:
    """Return each unit's maximum dark odd/even stability across trajectories."""
    import pandas as pd

    table_path = get_stability_table_path(data_root, animal_name, date)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            "for this session first."
        )
    table = pd.read_parquet(table_path)
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
            f"Tuning stability table {table_path} is missing columns "
            f"{missing_columns!r}."
        )

    rows = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(dark_epoch))
        & (table["trajectory_type"].astype(str).isin(PANEL_D_TRAJECTORY_TYPES))
    ].copy()
    rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
    rows["stability_correlation"] = pd.to_numeric(
        rows["stability_correlation"],
        errors="coerce",
    )
    rows = rows[
        np.isfinite(rows["unit"].to_numpy(dtype=float))
        & np.isfinite(rows["stability_correlation"].to_numpy(dtype=float))
    ].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "region",
                "unit",
                "max_dark_tuning_stability_correlation",
            ]
        )

    rows["unit"] = rows["unit"].astype(int)
    rows = (
        rows.groupby("unit", as_index=False, observed=False)[
            "stability_correlation"
        ]
        .max()
        .rename(
            columns={
                "stability_correlation": (
                    "max_dark_tuning_stability_correlation"
                )
            }
        )
    )
    rows["animal_name"] = str(animal_name)
    rows["date"] = str(date)
    rows["region"] = str(region)
    return rows.loc[
        :,
        [
            "animal_name",
            "date",
            "region",
            "unit",
            "max_dark_tuning_stability_correlation",
        ],
    ]


def load_candidate_unit_metric_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    cache_dir: Path | None,
) -> Any:
    """Return pooled dark movement-rate and stability metrics for candidates."""
    import pandas as pd

    tables = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        selected_dark_epoch = (
            str(dataset_dark_epoch)
            if dark_epoch is None
            else figure_3.get_dark_epoch(animal_name, date, dark_epoch)
        )
        movement_table = load_dark_movement_rate_metric_table(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=selected_dark_epoch,
            cache_dir=cache_dir,
        )
        stability_table = load_dark_tuning_stability_metric_table(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=selected_dark_epoch,
        )
        tables.append(
            movement_table.merge(
                stability_table,
                on=["animal_name", "date", "region", "unit"],
                how="outer",
            )
        )

    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "region",
                "unit",
                "dark_movement_firing_rate_hz",
                "max_dark_tuning_stability_correlation",
            ]
        )
    return pd.concat(tables, axis=0, ignore_index=True, sort=False)


def filter_candidate_rows_by_unit_metrics(
    candidate_table: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    min_movement_firing_rate_hz: float | None,
    min_tuning_stability_correlation: float | None,
    cache_dir: Path | None,
) -> Any:
    """Keep candidates whose dark movement rate and stability pass thresholds."""
    if min_movement_firing_rate_hz is None and min_tuning_stability_correlation is None:
        return candidate_table.copy()
    if (
        min_movement_firing_rate_hz is not None
        and min_movement_firing_rate_hz < 0.0
    ):
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")
    if (
        min_tuning_stability_correlation is not None
        and min_tuning_stability_correlation < -1.0
    ):
        raise ValueError("min_tuning_stability_correlation must be at least -1.")

    if candidate_table.empty:
        table = candidate_table.copy()
        table["dark_movement_firing_rate_hz"] = np.asarray([], dtype=float)
        table["max_dark_tuning_stability_correlation"] = np.asarray([], dtype=float)
        return table

    metric_table = load_candidate_unit_metric_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
    )
    table = candidate_table.merge(
        metric_table,
        on=["animal_name", "date", "region", "unit"],
        how="left",
    )
    keep = np.ones(len(table), dtype=bool)
    if min_movement_firing_rate_hz is not None:
        movement_rates = np.asarray(table["dark_movement_firing_rate_hz"], dtype=float)
        keep &= (
            np.isfinite(movement_rates)
            & (movement_rates > float(min_movement_firing_rate_hz))
        )
    if min_tuning_stability_correlation is not None:
        stability_values = np.asarray(
            table["max_dark_tuning_stability_correlation"],
            dtype=float,
        )
        keep &= (
            np.isfinite(stability_values)
            & (stability_values > float(min_tuning_stability_correlation))
        )
    return table.loc[keep].copy()


def normalize_curve_position(position: np.ndarray) -> np.ndarray:
    """Return curve coordinates scaled onto the plotted 0-1 progression axis."""
    position = np.asarray(position, dtype=float)
    finite = position[np.isfinite(position)]
    if finite.size == 0:
        return position
    min_position = float(np.nanmin(finite))
    max_position = float(np.nanmax(finite))
    span = max_position - min_position
    if span <= 0.0:
        return np.zeros_like(position, dtype=float)
    if min_position < 0.0 or max_position > 1.0:
        return (position - min_position) / span
    return position


def _load_curve_dataarray(
    curve_cache: dict[tuple[str, str, str, str, str], Any],
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory: str,
) -> Any:
    """Load one saved empirical trajectory tuning curve, caching by session key."""
    key = (
        str(animal_name),
        str(date),
        str(region),
        str(epoch),
        str(trajectory),
    )
    if key in curve_cache:
        return curve_cache[key]

    import xarray as xr

    path = figure_3.get_compute_tuning_curve_path(
        data_root,
        animal_name=str(animal_name),
        date=str(date),
        region=str(region),
        epoch=str(epoch),
        trajectory=str(trajectory),
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing saved empirical tuning curve. Expected "
            f"{path}. Run `python -m v1ca1.task_progression.compute_tuning_curves` "
            "for this session first."
        )
    curve_cache[key] = xr.load_dataarray(path)
    return curve_cache[key]


def extract_unit_curve(curve: Any, unit_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized position and firing rate for one unit from one curve."""
    units = np.asarray(curve.coords["unit"].values, dtype=int)
    if int(unit_id) not in set(units):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    position_dim = next(dim for dim in curve.dims if dim != "unit")
    position = normalize_curve_position(
        np.asarray(curve.coords[position_dim].values, dtype=float)
    )
    values = np.asarray(curve.sel(unit=int(unit_id)).values, dtype=float)
    return position, values


def _candidate_epoch_curve(
    row: Any,
    *,
    data_root: Path,
    epoch: str,
    trajectory: str,
    curve_cache: dict[tuple[str, str, str, str, str], Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one candidate's unit curve for one epoch and trajectory."""
    curve = _load_curve_dataarray(
        curve_cache,
        data_root=data_root,
        animal_name=str(row["animal_name"]),
        date=str(row["date"]),
        region=str(row["region"]),
        epoch=epoch,
        trajectory=trajectory,
    )
    return extract_unit_curve(curve, int(row["unit"]))


def compute_dark_peak_position_for_row(
    row: Any,
    *,
    data_root: Path,
    dark_epoch: str | None,
    curve_cache: dict[tuple[str, str, str, str, str], Any],
) -> float:
    """Return the peak of the mean dark same-turn tuning curve for one row."""
    epoch = figure_3.get_dark_epoch(
        str(row["animal_name"]),
        str(row["date"]),
        dark_epoch,
    )
    curves: list[np.ndarray] = []
    position_reference: np.ndarray | None = None
    for trajectory in (str(row["trajectory_a"]), str(row["trajectory_b"])):
        try:
            position, values = _candidate_epoch_curve(
                row,
                data_root=data_root,
                epoch=epoch,
                trajectory=trajectory,
                curve_cache=curve_cache,
            )
        except FileNotFoundError:
            return float("nan")
        if position.size == 0 or values.size == 0:
            continue
        if position_reference is None:
            position_reference = position
        elif position.shape != position_reference.shape or not np.allclose(
            position,
            position_reference,
            equal_nan=True,
        ):
            continue
        curves.append(values)

    if position_reference is None or not curves:
        return float("nan")
    mean_curve = np.nanmean(np.vstack(curves), axis=0)
    finite = np.isfinite(mean_curve) & np.isfinite(position_reference)
    if not np.any(finite):
        return float("nan")
    peak_index = np.flatnonzero(finite)[int(np.nanargmax(mean_curve[finite]))]
    return float(position_reference[peak_index])


def add_dark_peak_positions(
    candidate_table: Any,
    *,
    data_root: Path,
    dark_epoch: str | None,
    peak_bin_count: int,
) -> Any:
    """Add dark peak position and peak-bin annotations to candidate rows."""
    if candidate_table.empty:
        table = candidate_table.copy()
        table["dark_peak_position"] = np.asarray([], dtype=float)
        table["dark_peak_bin"] = np.asarray([], dtype=int)
        return table

    table = candidate_table.copy()
    curve_cache: dict[tuple[str, str, str, str, str], Any] = {}
    table["dark_peak_position"] = [
        compute_dark_peak_position_for_row(
            row,
            data_root=data_root,
            dark_epoch=dark_epoch,
            curve_cache=curve_cache,
        )
        for _index, row in table.iterrows()
    ]
    peak_bin_count = max(1, int(peak_bin_count))
    peak_positions = np.asarray(table["dark_peak_position"], dtype=float)
    peak_bins = np.floor(
        np.clip(peak_positions, 0.0, 1.0 - np.finfo(float).eps)
        * peak_bin_count
    )
    peak_bins[~np.isfinite(peak_positions)] = -1
    table["dark_peak_bin"] = peak_bins.astype(int)
    return table


def _sort_candidate_table_for_selection(
    table: Any,
    *,
    selection_mode: str,
) -> Any:
    """Return candidates sorted for the requested dark/light similarity pattern."""
    if selection_mode == "dark_high_light_low":
        sort_columns = [
            "dark_light_delta",
            "similarity_dark",
            "similarity_light",
            "animal_name",
            "date",
            "unit",
        ]
        ascending = [False, False, True, True, True, True]
        if "pair_delta_ll_bits_per_spike" in table.columns:
            sort_columns = ["pair_delta_ll_bits_per_spike", *sort_columns]
            ascending = [False, *ascending]
        return table.sort_values(
            sort_columns,
            ascending=ascending,
            kind="stable",
        )
    if selection_mode == "dark_high_light_high":
        table = table.copy()
        table["_joint_similarity"] = np.minimum(
            np.asarray(table["similarity_dark"], dtype=float),
            np.asarray(table["similarity_light"], dtype=float),
        )
        return table.sort_values(
            [
                "_joint_similarity",
                "similarity_dark",
                "similarity_light",
                "animal_name",
                "date",
                "unit",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        ).drop(columns="_joint_similarity")
    raise ValueError(f"selection_mode must be one of {SELECTION_MODES!r}.")


def add_pair_swap_delta(
    candidate_table: Any,
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    model_name: str,
    min_tuning_stability_correlation: float | None,
) -> Any:
    """Add Fig. 2D pair-mean model-minus-independent LL delta to candidates."""
    import pandas as pd

    if candidate_table.empty:
        table = candidate_table.copy()
        table["pair_delta_ll_bits_per_spike"] = np.asarray([], dtype=float)
        table["n_pair_delta_trajectories"] = np.asarray([], dtype=int)
        return table

    swap_delta = figure_3.load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
        model_name=model_name,
    )
    swap_delta = figure_3._filter_panel_h_heldout_delta(swap_delta)
    if swap_delta.empty:
        table = candidate_table.copy()
        table["pair_delta_ll_bits_per_spike"] = np.nan
        table["n_pair_delta_trajectories"] = 0
        return table

    swap_delta = swap_delta.copy()
    swap_delta["unit"] = pd.to_numeric(swap_delta["unit"], errors="coerce")
    swap_delta["delta_ll_bits_per_spike"] = pd.to_numeric(
        swap_delta["delta_ll_bits_per_spike"],
        errors="coerce",
    )
    swap_delta = swap_delta[
        np.isfinite(swap_delta["unit"].to_numpy(dtype=float))
        & np.isfinite(
            swap_delta["delta_ll_bits_per_spike"].to_numpy(dtype=float)
        )
    ].copy()
    swap_delta["unit"] = swap_delta["unit"].astype(int)

    pair_delta_values: list[float] = []
    pair_delta_counts: list[int] = []
    for _index, row in candidate_table.iterrows():
        trajectories = get_candidate_trajectories(str(row["comparison_label"]))
        rows = swap_delta[
            (swap_delta["animal_name"].astype(str) == str(row["animal_name"]))
            & (swap_delta["date"].astype(str) == str(row["date"]))
            & (swap_delta["region"].astype(str) == str(row["region"]))
            & (swap_delta["unit"].astype(int) == int(row["unit"]))
            & (swap_delta["trajectory"].astype(str).isin(trajectories))
        ]
        values = np.asarray(rows["delta_ll_bits_per_spike"], dtype=float)
        values = values[np.isfinite(values)]
        pair_delta_values.append(float(np.nanmean(values)) if values.size else np.nan)
        pair_delta_counts.append(int(values.size))

    table = candidate_table.copy()
    table["pair_delta_ll_bits_per_spike"] = pair_delta_values
    table["n_pair_delta_trajectories"] = pair_delta_counts
    return table


def exclude_candidate_rows(candidate_table: Any, exclude_paths: Sequence[Path]) -> Any:
    """Remove rows whose candidate keys appear in prior candidate tables."""
    if candidate_table.empty or not exclude_paths:
        return candidate_table.copy()

    import pandas as pd

    excluded_tables = []
    for path in exclude_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate exclusion table: {path}")
        table = pd.read_parquet(path)
        missing = [
            column
            for column in EXCLUDE_CANDIDATE_KEY_COLUMNS
            if column not in table.columns
        ]
        if missing:
            raise ValueError(f"Exclusion table {path} is missing columns {missing!r}.")
        excluded_tables.append(table.loc[:, EXCLUDE_CANDIDATE_KEY_COLUMNS].copy())

    if not excluded_tables:
        return candidate_table.copy()

    excluded = pd.concat(excluded_tables, axis=0, ignore_index=True).drop_duplicates()
    merged = candidate_table.merge(
        excluded.assign(_exclude_candidate=True),
        on=list(EXCLUDE_CANDIDATE_KEY_COLUMNS),
        how="left",
    )
    return merged[merged["_exclude_candidate"].isna()].drop(
        columns="_exclude_candidate"
    )


def _select_peak_diverse_rows(
    table: Any,
    *,
    candidate_count: int,
    peak_bin_count: int,
    selection_mode: str,
) -> Any:
    """Return rows selected round-robin across dark peak-position bins."""
    if "dark_peak_bin" not in table.columns:
        raise ValueError("Peak-diverse selection requires dark_peak_bin.")

    table = table[
        np.isfinite(np.asarray(table["dark_peak_position"], dtype=float))
        & (np.asarray(table["dark_peak_bin"], dtype=int) >= 0)
    ].copy()
    if table.empty:
        return table

    peak_bin_count = max(1, int(peak_bin_count))
    rows_by_bin = [
        _sort_candidate_table_for_selection(
            table[table["dark_peak_bin"].astype(int) == peak_bin],
            selection_mode=selection_mode,
        )
        for peak_bin in range(peak_bin_count)
    ]
    selected_indices: list[Any] = []
    offsets = [0 for _peak_bin in range(peak_bin_count)]
    target_count = max(int(candidate_count), 0)
    while len(selected_indices) < target_count:
        selected_this_pass = False
        for peak_bin, bin_rows in enumerate(rows_by_bin):
            offset = offsets[peak_bin]
            if offset >= len(bin_rows):
                continue
            selected_indices.append(bin_rows.index[offset])
            offsets[peak_bin] += 1
            selected_this_pass = True
            if len(selected_indices) >= target_count:
                break
        if not selected_this_pass:
            break

    return table.loc[selected_indices].sort_values(
        ["dark_peak_position", "similarity_dark", "similarity_light"],
        ascending=[True, False, False],
        kind="stable",
    )


def select_candidate_rows(
    candidate_table: Any,
    *,
    candidate_count: int,
    min_dark_similarity: float,
    max_light_similarity: float,
    min_light_similarity: float,
    min_dark_light_delta: float,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    diversify_by_peak: bool = DEFAULT_DIVERSIFY_BY_PEAK,
    peak_bin_count: int = DEFAULT_PEAK_BIN_COUNT,
) -> Any:
    """Return the top cells matching the requested dark/light similarity pattern."""
    import pandas as pd

    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES!r}.")
    if candidate_table.empty:
        return pd.DataFrame(columns=CANDIDATE_TABLE_COLUMNS)

    table = candidate_table.copy()
    if selection_mode == "dark_high_light_low":
        table = table[
            (table["similarity_dark"] >= float(min_dark_similarity))
            & (table["similarity_light"] <= float(max_light_similarity))
            & (table["dark_light_delta"] >= float(min_dark_light_delta))
        ].copy()
    else:
        table = table[
            (table["similarity_dark"] >= float(min_dark_similarity))
            & (table["similarity_light"] >= float(min_light_similarity))
        ].copy()
    if table.empty:
        return pd.DataFrame(columns=CANDIDATE_TABLE_COLUMNS)

    if "dark_peak_position" not in table.columns:
        table["dark_peak_position"] = np.nan
    if "dark_peak_bin" not in table.columns:
        table["dark_peak_bin"] = -1

    if diversify_by_peak:
        table = _select_peak_diverse_rows(
            table,
            candidate_count=candidate_count,
            peak_bin_count=peak_bin_count,
            selection_mode=selection_mode,
        )
    else:
        table = _sort_candidate_table_for_selection(
            table,
            selection_mode=selection_mode,
        ).head(max(int(candidate_count), 0))
    table = table.reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1, dtype=int)
    output_columns = list(CANDIDATE_TABLE_COLUMNS)
    for column in CANDIDATE_OPTIONAL_TABLE_COLUMNS:
        if column in table.columns:
            output_columns.append(column)
    return table.loc[:, output_columns]


def _panel_example_cache_path_for_row(
    row: Any,
    *,
    data_root: Path,
    epoch: str,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path,
) -> Path:
    """Return the expected cached example path for one candidate and epoch."""
    metadata = figure_3.build_panel_example_cache_metadata(
        data_root=data_root,
        panel_name=figure_3.PANEL_A_EXAMPLE_CACHE_PANEL_NAME,
        animal_name=str(row["animal_name"]),
        date=str(row["date"]),
        epoch=epoch,
        region=str(row["region"]),
        unit_id=int(row["unit"]),
        trajectories=(str(row["trajectory_a"]), str(row["trajectory_b"])),
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
    )
    return figure_3.build_panel_example_cache_path(
        panel_example_cache_dir,
        metadata,
    )


def filter_rows_with_existing_example_cache(
    candidate_table: Any,
    *,
    data_root: Path,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path,
) -> Any:
    """Keep candidate rows that already have matching dark and light example caches."""
    if candidate_table.empty:
        return candidate_table.copy()

    keep_rows: list[bool] = []
    for _index, row in candidate_table.iterrows():
        dark_epoch_id = figure_3.get_dark_epoch(
            str(row["animal_name"]),
            str(row["date"]),
            dark_epoch,
        )
        light_epoch_id = figure_3.get_light_epoch(
            str(row["animal_name"]),
            str(row["date"]),
            light_epoch,
        )
        dark_cache_path = _panel_example_cache_path_for_row(
            row,
            data_root=data_root,
            epoch=dark_epoch_id,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
        )
        light_cache_path = _panel_example_cache_path_for_row(
            row,
            data_root=data_root,
            epoch=light_epoch_id,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
        )
        keep_rows.append(dark_cache_path.exists() and light_cache_path.exists())

    return candidate_table.loc[keep_rows].copy()


def _format_candidate_title(row: Any) -> str:
    """Return a compact title for one plotted candidate cell."""
    turn_label = str(row["comparison_label"]).replace("_", " ")
    peak_position = float(row.get("dark_peak_position", np.nan))
    peak_text = f"pk={peak_position:.2f}" if np.isfinite(peak_position) else ""
    delta_ll = float(row.get("pair_delta_ll_bits_per_spike", np.nan))
    delta_text = f"ΔLL={delta_ll:.2f}" if np.isfinite(delta_ll) else ""
    final_line = "  ".join(
        text for text in (peak_text, delta_text) if text
    )
    final_line = f"\n{final_line}" if final_line else ""
    return (
        f"{int(row['rank'])}. {row['animal_name']} {row['date']} u{int(row['unit'])}\n"
        f"{turn_label}  d={float(row['similarity_dark']):.2f} "
        f"l={float(row['similarity_light']):.2f} "
        f"diff={float(row['dark_light_delta']):.2f}{final_line}"
    )


def load_candidate_examples(
    candidate_rows: Any,
    *,
    data_root: Path,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    panel_example_cache_dir: Path | None,
    refresh_panel_example_cache: bool,
) -> list[dict[str, Any]]:
    """Load Panel A-style dark/light example payloads for candidate cells."""
    examples: list[dict[str, Any]] = []
    for _index, row in candidate_rows.iterrows():
        trajectories = (str(row["trajectory_a"]), str(row["trajectory_b"]))
        examples.append(
            load_panel_a_example_data(
                data_root=data_root,
                animal_name=str(row["animal_name"]),
                date=str(row["date"]),
                region=str(row["region"]),
                unit_id=int(row["unit"]),
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
        )
    return examples


def plot_candidate_examples(
    *,
    candidate_rows: Any,
    examples: Sequence[dict[str, Any]],
    output_path: Path,
    n_columns: int,
    dpi: int,
) -> Path:
    """Plot candidate cells using the Figure 3 Panel A example layout."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    n_examples = len(examples)
    n_columns = max(1, int(n_columns))
    n_rows = max(1, int(np.ceil(n_examples / n_columns)))
    fig_height_mm = max(
        CANDIDATE_MIN_FIGURE_HEIGHT_MM,
        CANDIDATE_ROW_HEIGHT_MM * n_rows,
    )
    fig, ax = plt.subplots(
        figsize=figure_size(CANDIDATE_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    if n_examples == 0:
        ax.text(0.5, 0.5, "No candidate cells", ha="center", va="center")
    else:
        cell_width = (
            1.0 - CANDIDATE_GRID_WSPACE * (n_columns - 1)
        ) / n_columns
        cell_height = (
            1.0 - CANDIDATE_GRID_HSPACE * (n_rows - 1)
        ) / n_rows
        for index, (example, (_row_index, row)) in enumerate(
            zip(examples, candidate_rows.iterrows(), strict=True)
        ):
            row_number = index // n_columns
            column_number = index % n_columns
            x0 = column_number * (cell_width + CANDIDATE_GRID_WSPACE)
            y0 = (
                1.0
                - (row_number + 1) * cell_height
                - row_number * CANDIDATE_GRID_HSPACE
            )
            child_ax = ax.inset_axes([x0, y0, cell_width, cell_height])
            figure_3.plot_panel_a_example(
                child_ax,
                example,
                title=_format_candidate_title(row),
                y_shift=0.0,
            )

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _load_candidate_curve_payload(
    row: Any,
    *,
    data_root: Path,
    light_epoch: str | None,
    dark_epoch: str | None,
    curve_cache: dict[tuple[str, str, str, str, str], Any],
    trajectories: Sequence[str] | None = None,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return dark and light empirical curves for one candidate row."""
    epochs = {
        "dark": figure_3.get_dark_epoch(
            str(row["animal_name"]),
            str(row["date"]),
            dark_epoch,
        ),
        "light": figure_3.get_light_epoch(
            str(row["animal_name"]),
            str(row["date"]),
            light_epoch,
        ),
    }
    plot_trajectories = (
        tuple(trajectories)
        if trajectories is not None
        else (str(row["trajectory_a"]), str(row["trajectory_b"]))
    )
    payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for epoch_key, epoch in epochs.items():
        payload[epoch_key] = {}
        for trajectory in plot_trajectories:
            try:
                payload[epoch_key][trajectory] = _candidate_epoch_curve(
                    row,
                    data_root=data_root,
                    epoch=epoch,
                    trajectory=trajectory,
                    curve_cache=curve_cache,
                )
            except FileNotFoundError:
                payload[epoch_key][trajectory] = (
                    np.asarray([], dtype=float),
                    np.asarray([], dtype=float),
                )
    return payload


def _candidate_curve_y_max(
    payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> float:
    """Return a shared y-limit for one candidate's dark/light curves."""
    values: list[np.ndarray] = []
    for epoch_payload in payload.values():
        for _position, curve_values in epoch_payload.values():
            finite = np.asarray(curve_values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                values.append(finite)
    if not values:
        return 1.0
    return max(1.0, float(np.ceil(np.nanmax(np.concatenate(values)))))


def _plot_candidate_curve_axis(
    ax: Any,
    *,
    epoch_key: str,
    epoch_payload: dict[str, tuple[np.ndarray, np.ndarray]],
    y_max: float,
    show_ylabel: bool,
    show_xlabel: bool,
    peak_position: float,
    show_title: bool = True,
) -> None:
    """Plot one compact dark or light trajectory-pair tuning axis."""
    for trajectory, (position, values) in epoch_payload.items():
        ax.plot(
            position,
            values,
            color=figure_3.PANEL_TRAJECTORY_COLORS[trajectory],
            linewidth=0.75,
            label=figure_3.PANEL_TRAJECTORY_LABELS.get(trajectory, trajectory),
        )
    figure_3.add_segment_boundary_lines(ax)
    if epoch_key == "dark" and np.isfinite(peak_position):
        ax.axvline(
            peak_position,
            color="0.20",
            linewidth=0.45,
            linestyle=":",
            zorder=1,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    if show_title:
        ax.set_title(
            figure_3.PANEL_A_EPOCH_LABELS[epoch_key],
            fontsize=CURVE_GALLERY_AXIS_TITLE_FONTSIZE,
            pad=0.8,
        )
    if show_xlabel:
        ax.set_xlabel(
            "Norm. path progression",
            fontsize=CURVE_GALLERY_AXIS_LABEL_FONTSIZE,
            labelpad=0.7,
        )
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("FR", fontsize=CURVE_GALLERY_AXIS_LABEL_FONTSIZE, labelpad=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=3.9, length=1.1, pad=0.7)


def _subset_candidate_curve_payload(
    payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    epoch_key: str,
    trajectories: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return epoch curves restricted to one same-turn trajectory pair."""
    empty = (np.asarray([], dtype=float), np.asarray([], dtype=float))
    return {
        trajectory: payload.get(epoch_key, {}).get(trajectory, empty)
        for trajectory in trajectories
    }


def _plot_split_turn_curve_cell(
    cell_ax: Any,
    *,
    payload: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    y_max: float,
    peak_position: float,
    show_xlabel: bool,
) -> None:
    """Plot left- and right-turn trajectory pairs in separate dark/light panels."""
    row_specs = (
        (CANDIDATE_TURN_DIRECTION_TRAJECTORIES[0], 0.49),
        (CANDIDATE_TURN_DIRECTION_TRAJECTORIES[1], 0.13),
    )
    for row_number, ((turn_label, trajectories), y0) in enumerate(row_specs):
        show_row_xlabel = show_xlabel and row_number == len(row_specs) - 1
        show_title = row_number == 0
        cell_ax.text(
            0.045,
            y0 + 0.13,
            turn_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=CURVE_GALLERY_AXIS_LABEL_FONTSIZE,
            transform=cell_ax.transAxes,
        )
        dark_ax = cell_ax.inset_axes([0.14, y0, 0.35, 0.25])
        light_ax = cell_ax.inset_axes([0.58, y0, 0.35, 0.25])
        dark_ax.set_facecolor(figure_3.PANEL_A_DARK_EPOCH_BACKGROUND)
        _plot_candidate_curve_axis(
            dark_ax,
            epoch_key="dark",
            epoch_payload=_subset_candidate_curve_payload(
                payload,
                epoch_key="dark",
                trajectories=trajectories,
            ),
            y_max=y_max,
            show_ylabel=True,
            show_xlabel=show_row_xlabel,
            peak_position=peak_position,
            show_title=show_title,
        )
        _plot_candidate_curve_axis(
            light_ax,
            epoch_key="light",
            epoch_payload=_subset_candidate_curve_payload(
                payload,
                epoch_key="light",
                trajectories=trajectories,
            ),
            y_max=y_max,
            show_ylabel=False,
            show_xlabel=show_row_xlabel,
            peak_position=peak_position,
            show_title=show_title,
        )


def plot_candidate_curve_gallery(
    *,
    candidate_rows: Any,
    data_root: Path,
    light_epoch: str | None,
    dark_epoch: str | None,
    output_path: Path,
    n_columns: int,
    dpi: int,
    plot_both_turn_directions: bool = DEFAULT_PLOT_BOTH_TURN_DIRECTIONS,
    split_turn_directions: bool = DEFAULT_SPLIT_TURN_DIRECTIONS,
) -> Path:
    """Plot a compact dark/light tuning-curve gallery for many candidates."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    n_examples = len(candidate_rows)
    n_columns = max(1, int(n_columns))
    n_rows = max(1, int(np.ceil(max(n_examples, 1) / n_columns)))
    row_height_mm = (
        CURVE_GALLERY_ROW_HEIGHT_MM * 1.35
        if split_turn_directions
        else CURVE_GALLERY_ROW_HEIGHT_MM
    )
    fig_height_mm = max(
        CURVE_GALLERY_MIN_FIGURE_HEIGHT_MM,
        row_height_mm * n_rows,
    )
    fig, ax = plt.subplots(
        figsize=figure_size(CANDIDATE_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=True,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    if n_examples == 0:
        ax.text(0.5, 0.5, "No candidate cells", ha="center", va="center")
        save_figure(fig, output_path, dpi=dpi)
        plt.close(fig)
        return output_path

    curve_cache: dict[tuple[str, str, str, str, str], Any] = {}
    cell_width = (1.0 - CANDIDATE_GRID_WSPACE * (n_columns - 1)) / n_columns
    cell_height = (1.0 - CANDIDATE_GRID_HSPACE * (n_rows - 1)) / n_rows
    for index, (_row_index, row) in enumerate(candidate_rows.iterrows()):
        row_number = index // n_columns
        column_number = index % n_columns
        x0 = column_number * (cell_width + CANDIDATE_GRID_WSPACE)
        y0 = (
            1.0
            - (row_number + 1) * cell_height
            - row_number * CANDIDATE_GRID_HSPACE
        )
        cell_ax = ax.inset_axes([x0, y0, cell_width, cell_height])
        cell_ax.set_xlim(0.0, 1.0)
        cell_ax.set_ylim(0.0, 1.0)
        cell_ax.axis("off")
        cell_ax.text(
            0.5,
            0.98,
            _format_candidate_title(row),
            ha="center",
            va="top",
            fontsize=CURVE_GALLERY_TITLE_FONTSIZE,
            linespacing=1.0,
            transform=cell_ax.transAxes,
        )

        payload = _load_candidate_curve_payload(
            row,
            data_root=data_root,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            curve_cache=curve_cache,
            trajectories=(
                figure_2.PANEL_B_TUNING_CORRELATION_TRAJECTORIES
                if plot_both_turn_directions or split_turn_directions
                else None
            ),
        )
        y_max = _candidate_curve_y_max(payload)
        peak_position = float(row.get("dark_peak_position", np.nan))
        show_xlabel = row_number == n_rows - 1
        if split_turn_directions:
            _plot_split_turn_curve_cell(
                cell_ax,
                payload=payload,
                y_max=y_max,
                peak_position=peak_position,
                show_xlabel=show_xlabel,
            )
            continue

        dark_ax = cell_ax.inset_axes([0.08, 0.15, 0.40, 0.58])
        light_ax = cell_ax.inset_axes([0.56, 0.15, 0.40, 0.58])
        dark_ax.set_facecolor(figure_3.PANEL_A_DARK_EPOCH_BACKGROUND)
        _plot_candidate_curve_axis(
            dark_ax,
            epoch_key="dark",
            epoch_payload=payload["dark"],
            y_max=y_max,
            show_ylabel=True,
            show_xlabel=show_xlabel,
            peak_position=peak_position,
        )
        _plot_candidate_curve_axis(
            light_ax,
            epoch_key="light",
            epoch_payload=payload["light"],
            y_max=y_max,
            show_ylabel=False,
            show_xlabel=show_xlabel,
            peak_position=peak_position,
        )

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def get_candidate_table_path(output_path: Path) -> Path:
    """Return the parquet path paired with one candidate figure output."""
    return Path(output_path).with_name(Path(output_path).stem + CANDIDATE_TABLE_SUFFIX)


def make_panel_a_candidate_figure(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    position_bin_count: int,
    position_offset: int,
    speed_threshold_cm_s: float,
    sigma_bins: float,
    candidate_count: int,
    min_dark_similarity: float,
    max_light_similarity: float,
    min_light_similarity: float,
    min_dark_light_delta: float,
    min_movement_firing_rate_hz: float | None,
    min_tuning_stability_correlation: float | None,
    min_shared_scaffold_delta: float | None,
    swap_model_name: str,
    swap_delta_min_tuning_stability_correlation: float | None,
    n_columns: int,
    dpi: int,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    plot_mode: str = DEFAULT_PLOT_MODE,
    diversify_by_peak: bool = DEFAULT_DIVERSIFY_BY_PEAK,
    peak_bin_count: int = DEFAULT_PEAK_BIN_COUNT,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    require_existing_example_cache: bool = (
        DEFAULT_REQUIRE_EXISTING_EXAMPLE_CACHE
    ),
    exclude_candidate_tables: Sequence[Path] = (),
    plot_both_turn_directions: bool = DEFAULT_PLOT_BOTH_TURN_DIRECTIONS,
    split_turn_directions: bool = DEFAULT_SPLIT_TURN_DIRECTIONS,
) -> tuple[Path, Path]:
    """Rank candidate cells, save a table, and plot a Panel A-style SVG."""
    if plot_mode not in PLOT_MODES:
        raise ValueError(f"plot_mode must be one of {PLOT_MODES!r}.")
    output_path = Path(output_path)
    panel_example_cache_dir = (
        output_path.parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    candidate_table = build_candidate_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    candidate_table = filter_candidate_rows_by_unit_metrics(
        candidate_table,
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        min_movement_firing_rate_hz=min_movement_firing_rate_hz,
        min_tuning_stability_correlation=min_tuning_stability_correlation,
        cache_dir=panel_example_cache_dir,
    )
    if diversify_by_peak or plot_mode == "curves":
        candidate_table = add_dark_peak_positions(
            candidate_table,
            data_root=data_root,
            dark_epoch=dark_epoch,
            peak_bin_count=peak_bin_count,
        )
    if min_shared_scaffold_delta is not None:
        candidate_table = add_pair_swap_delta(
            candidate_table,
            data_root=data_root,
            datasets=datasets,
            region=region,
            dark_epoch=dark_epoch,
            model_name=swap_model_name,
            min_tuning_stability_correlation=(
                swap_delta_min_tuning_stability_correlation
            ),
        )
        candidate_table = candidate_table[
            candidate_table["pair_delta_ll_bits_per_spike"]
            >= float(min_shared_scaffold_delta)
        ].copy()
    candidate_table = exclude_candidate_rows(
        candidate_table,
        exclude_candidate_tables,
    )
    if require_existing_example_cache and plot_mode == "panel-a":
        candidate_table = filter_rows_with_existing_example_cache(
            candidate_table,
            data_root=data_root,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
        )
    candidate_rows = select_candidate_rows(
        candidate_table,
        candidate_count=candidate_count,
        min_dark_similarity=min_dark_similarity,
        max_light_similarity=max_light_similarity,
        min_light_similarity=min_light_similarity,
        min_dark_light_delta=min_dark_light_delta,
        selection_mode=selection_mode,
        diversify_by_peak=diversify_by_peak,
        peak_bin_count=peak_bin_count,
    )
    table_path = get_candidate_table_path(output_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows.to_parquet(table_path)

    if plot_mode == "panel-a":
        examples = load_candidate_examples(
            candidate_rows,
            data_root=data_root,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
        plot_candidate_examples(
            candidate_rows=candidate_rows,
            examples=examples,
            output_path=output_path,
            n_columns=n_columns,
            dpi=dpi,
        )
    else:
        plot_candidate_curve_gallery(
            candidate_rows=candidate_rows,
            data_root=data_root,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            output_path=output_path,
            n_columns=n_columns,
            dpi=dpi,
            plot_both_turn_directions=plot_both_turn_directions,
            split_turn_directions=split_turn_directions,
        )
    return output_path, table_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 3_2 Panel A candidate plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank cells by dark/light same-turn trajectory tuning similarity, "
            "then plot Panel A-style candidates."
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
            "Directory for cached dark/light example rasters and curves. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help="Recompute example cells even when matching caches exist.",
    )
    parser.add_argument(
        "--require-existing-example-cache",
        action="store_true",
        default=DEFAULT_REQUIRE_EXISTING_EXAMPLE_CACHE,
        help=(
            "Only plot candidates that already have matching dark and light "
            "Panel A example caches."
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
        "--plot-mode",
        choices=PLOT_MODES,
        default=DEFAULT_PLOT_MODE,
        help=(
            "Plot full Panel A-style raster examples or a compact curve-only gallery. "
            f"Default: {DEFAULT_PLOT_MODE}"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=DEFAULT_SELECTION_MODE,
        help=(
            "Candidate pattern to select. The default keeps high-dark/low-light "
            "cells; dark_high_light_high keeps cells with high similarity in both "
            f"epochs. Default: {DEFAULT_SELECTION_MODE}"
        ),
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
        "--exclude-candidate-table",
        action="append",
        type=Path,
        default=[],
        help=(
            "Prior candidate parquet table to exclude by animal/date/region/unit/"
            "comparison label. May be repeated."
        ),
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default="v1",
        help="Region to include. Default: v1.",
    )
    parser.add_argument(
        "--light-epoch",
        default=None,
        help="Light run epoch. Default: registry value.",
    )
    parser.add_argument(
        "--dark-epoch",
        default=None,
        help="Dark run epoch. Default: registry value.",
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
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
        help=f"Number of candidates to plot. Default: {DEFAULT_CANDIDATE_COUNT}",
    )
    parser.add_argument(
        "--min-dark-similarity",
        type=float,
        default=DEFAULT_MIN_DARK_SIMILARITY,
        help=(
            "Minimum same-turn dark tuning correlation for candidate cells. "
            f"Default: {DEFAULT_MIN_DARK_SIMILARITY}"
        ),
    )
    parser.add_argument(
        "--max-light-similarity",
        type=float,
        default=DEFAULT_MAX_LIGHT_SIMILARITY,
        help=(
            "Maximum same-turn light tuning correlation for dark_high_light_low "
            "candidate cells. "
            f"Default: {DEFAULT_MAX_LIGHT_SIMILARITY}"
        ),
    )
    parser.add_argument(
        "--min-light-similarity",
        type=float,
        default=DEFAULT_MIN_LIGHT_SIMILARITY,
        help=(
            "Minimum same-turn light tuning correlation for dark_high_light_high "
            "candidate cells. "
            f"Default: {DEFAULT_MIN_LIGHT_SIMILARITY}"
        ),
    )
    parser.add_argument(
        "--min-dark-light-delta",
        type=float,
        default=DEFAULT_MIN_DARK_LIGHT_DELTA,
        help=(
            "Minimum dark-minus-light same-turn correlation drop. "
            f"Default: {DEFAULT_MIN_DARK_LIGHT_DELTA}"
        ),
    )
    parser.add_argument(
        "--min-movement-firing-rate-hz",
        type=float,
        default=DEFAULT_MIN_MOVEMENT_FIRING_RATE_HZ,
        help=(
            "Minimum dark movement firing rate for plotted units. "
            f"Default: {DEFAULT_MIN_MOVEMENT_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-tuning-stability-correlation",
        type=float,
        default=DEFAULT_MIN_TUNING_STABILITY_CORRELATION,
        help=(
            "Minimum maximum dark odd/even stability correlation across "
            "trajectories for plotted units. "
            f"Default: {DEFAULT_MIN_TUNING_STABILITY_CORRELATION}"
        ),
    )
    parser.add_argument(
        "--min-shared-scaffold-delta",
        type=float,
        default=DEFAULT_MIN_SHARED_SCAFFOLD_DELTA,
        help=(
            "Minimum Fig. 2D pair-mean model-minus-independent held-out LL "
            "delta. Omit to disable this filter."
        ),
    )
    parser.add_argument(
        "--swap-model-name",
        default=DEFAULT_SWAP_MODEL_NAME,
        help=(
            "Model name used for the Fig. 2D model-minus-independent filter. "
            f"Default: {DEFAULT_SWAP_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--swap-delta-min-tuning-stability-correlation",
        type=float,
        default=DEFAULT_SWAP_DELTA_MIN_TUNING_STABILITY_CORRELATION,
        help=(
            "Dark odd/even stability threshold used when loading Fig. 2D swap "
            "deltas. Default matches Figure 2."
        ),
    )
    parser.add_argument(
        "--plot-both-turn-directions",
        action="store_true",
        default=DEFAULT_PLOT_BOTH_TURN_DIRECTIONS,
        help="In curve mode, plot both same-turn direction pairs for each unit.",
    )
    parser.add_argument(
        "--split-turn-directions",
        action="store_true",
        default=DEFAULT_SPLIT_TURN_DIRECTIONS,
        help=(
            "In curve mode, plot left- and right-turn trajectory pairs in "
            "separate dark/light panels for each unit."
        ),
    )
    parser.add_argument(
        "--diversify-by-peak",
        action="store_true",
        default=DEFAULT_DIVERSIFY_BY_PEAK,
        help="Select candidates round-robin across dark peak-position bins.",
    )
    parser.add_argument(
        "--peak-bin-count",
        type=int,
        default=DEFAULT_PEAK_BIN_COUNT,
        help=(
            "Number of dark peak-position bins for diversified selection. "
            f"Default: {DEFAULT_PEAK_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=DEFAULT_CANDIDATE_COLUMNS,
        help=f"Number of candidate columns in the figure. Default: {DEFAULT_CANDIDATE_COLUMNS}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run candidate selection and plotting."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    figure_path, table_path = make_panel_a_candidate_figure(
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
        candidate_count=args.candidate_count,
        min_dark_similarity=args.min_dark_similarity,
        max_light_similarity=args.max_light_similarity,
        min_light_similarity=args.min_light_similarity,
        min_dark_light_delta=args.min_dark_light_delta,
        min_movement_firing_rate_hz=args.min_movement_firing_rate_hz,
        min_tuning_stability_correlation=args.min_tuning_stability_correlation,
        min_shared_scaffold_delta=args.min_shared_scaffold_delta,
        swap_model_name=args.swap_model_name,
        swap_delta_min_tuning_stability_correlation=(
            args.swap_delta_min_tuning_stability_correlation
        ),
        n_columns=args.n_columns,
        dpi=args.dpi,
        selection_mode=args.selection_mode,
        plot_mode=args.plot_mode,
        diversify_by_peak=args.diversify_by_peak,
        peak_bin_count=args.peak_bin_count,
        panel_example_cache_dir=args.panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
        require_existing_example_cache=args.require_existing_example_cache,
        exclude_candidate_tables=tuple(args.exclude_candidate_table),
        plot_both_turn_directions=args.plot_both_turn_directions,
        split_turn_directions=args.split_turn_directions,
    )
    print(f"Saved Panel A candidate figure to {figure_path}")
    print(f"Saved Panel A candidate table to {table_path}")


if __name__ == "__main__":
    main()
