from __future__ import annotations

"""Compare selected segment coefficients across two light epochs.

This module consumes selected NetCDF outputs from
`v1ca1.task_progression.dark_light_glm`, aligns matching units between two
light-training epochs, and compares the learned segment light-modulation
coefficients for each trajectory and segment. By default it compares only the
segment-local coefficient. With `--include-light-offset`, it adds the selected
trajectory's scalar light-offset coefficient to each segment coefficient before
comparison.
"""

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
)


SUPPORTED_MODEL_NAMES = (
    "task_segment_bump",
    "task_segment_scalar",
)
DEPRECATED_MODEL_NAME_ALIASES = {
    "segment_bump_gain": "task_segment_bump",
    "segment_scalar_gain": "task_segment_scalar",
}
DEPRECATED_DARK_LIGHT_ONLY_MODEL_NAMES = (
    "overlapping_segment_bump_gain",
)
MODEL_NAME_CHOICES = (
    *SUPPORTED_MODEL_NAMES,
    *DEPRECATED_MODEL_NAME_ALIASES,
    *DEPRECATED_DARK_LIGHT_ONLY_MODEL_NAMES,
)
TRAJECTORIES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
TRAJECTORY_GROUPS = {
    "outbound": ("center_to_left", "center_to_right"),
    "inbound": ("left_to_center", "right_to_center"),
}
SWITCHED_SEGMENT_BY_TRAJECTORY = {
    "center_to_left": 2,
    "center_to_right": 2,
    "left_to_center": 0,
    "right_to_center": 0,
}
TRAJECTORY_COLORS = {
    "center_to_left": "#1b9e77",
    "center_to_right": "#d95f02",
    "left_to_center": "#7570b3",
    "right_to_center": "#e7298a",
}


def _dataset_path(
    input_dir: Path,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_name: str,
) -> Path:
    """Return the expected selected NetCDF path for one dark/light fit."""
    return input_dir / f"{region}_{light_epoch}_vs_{dark_epoch}_{model_name}_selected.nc"


def _normalize_model_names(model_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return canonical selected model names and any deprecation messages."""
    normalized: list[str] = []
    messages: list[str] = []
    for requested_name in model_names:
        model_name = str(requested_name)
        if model_name in DEPRECATED_DARK_LIGHT_ONLY_MODEL_NAMES:
            raise ValueError(
                f"Model {model_name!r} is deprecated for the selected-fit workflow. "
                f"Use one of {SUPPORTED_MODEL_NAMES!r}."
            )
        if model_name in DEPRECATED_MODEL_NAME_ALIASES:
            replacement = DEPRECATED_MODEL_NAME_ALIASES[model_name]
            messages.append(
                f"Model {model_name!r} is deprecated; using {replacement!r} instead."
            )
            model_name = replacement
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model {model_name!r}. Expected one of "
                f"{SUPPORTED_MODEL_NAMES!r}."
            )
        if model_name not in normalized:
            normalized.append(model_name)
    return normalized, messages


def _resolve_model_names(
    input_dir: Path,
    *,
    region: str,
    light_epoch1: str,
    light_epoch2: str,
    dark_epoch: str,
    requested_model_names: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Return requested models or all available selected segment models."""
    if requested_model_names is not None:
        model_names, messages = _normalize_model_names(requested_model_names)
        return model_names, messages

    available_model_names = []
    for model_name in SUPPORTED_MODEL_NAMES:
        dataset1_path = _dataset_path(
            input_dir,
            region=region,
            light_epoch=light_epoch1,
            dark_epoch=dark_epoch,
            model_name=model_name,
        )
        dataset2_path = _dataset_path(
            input_dir,
            region=region,
            light_epoch=light_epoch2,
            dark_epoch=dark_epoch,
            model_name=model_name,
        )
        if dataset1_path.exists() and dataset2_path.exists():
            available_model_names.append(model_name)

    if not available_model_names:
        raise FileNotFoundError(
            "No supported selected segment models were found for both requested "
            f"light epochs in {input_dir}. Checked models: {list(SUPPORTED_MODEL_NAMES)!r}."
        )
    return available_model_names, []


def _attr_epoch(dataset, *names: str) -> str:
    """Return the first non-empty epoch attribute from a selected dataset."""
    for name in names:
        value = str(dataset.attrs.get(name, ""))
        if value:
            return value
    return ""


def _load_selected_dataset(
    path: Path,
    *,
    expected_region: str,
    expected_light_epoch: str,
    expected_dark_epoch: str,
    expected_model_name: str,
):
    """Load one selected dark/light fit dataset and validate core metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Selected dark/light fit dataset not found: {path}")

    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to load dark_light_glm NetCDF files."
        ) from exc

    dataset = xr.load_dataset(path)

    fit_stage = str(dataset.attrs.get("fit_stage", ""))
    if fit_stage != "selected":
        raise ValueError(
            f"Dataset {path} has fit_stage={fit_stage!r}; expected 'selected'. "
            "Regenerate or select dark_light_glm outputs before running this script."
        )

    region = str(dataset.attrs.get("region", ""))
    model_name = str(dataset.attrs.get("model_name", ""))
    light_epoch = _attr_epoch(dataset, "light_train_epoch", "light_epoch")
    dark_epoch = _attr_epoch(dataset, "dark_train_epoch", "dark_epoch")
    if region and region != expected_region:
        raise ValueError(
            f"Dataset region mismatch for {path}: expected {expected_region!r}, "
            f"found {region!r}."
        )
    if model_name != expected_model_name:
        raise ValueError(
            f"Dataset model_name mismatch for {path}: expected "
            f"{expected_model_name!r}, found {model_name!r}."
        )
    if light_epoch != expected_light_epoch:
        raise ValueError(
            f"Dataset light_train_epoch mismatch for {path}: expected "
            f"{expected_light_epoch!r}, found {light_epoch!r}."
        )
    if dark_epoch != expected_dark_epoch:
        raise ValueError(
            f"Dataset dark_train_epoch mismatch for {path}: expected "
            f"{expected_dark_epoch!r}, found {dark_epoch!r}."
        )

    if "segment_edges" not in dataset:
        raise ValueError(f"Dataset {path} does not contain segment_edges.")
    if _segment_gain_var_name(expected_model_name) not in dataset:
        raise ValueError(
            f"Dataset {path} is missing required variable "
            f"{_segment_gain_var_name(expected_model_name)!r}."
        )

    _validate_segment_layout(dataset, path=path, model_name=expected_model_name)
    _validate_supported_trajectories(dataset, path=path)
    return dataset


def _validate_segment_layout(dataset, *, path: Path, model_name: str) -> None:
    """Require exactly three saved task-progression segments."""
    segment_edges = _segment_edges(dataset)
    n_segments = segment_edges.size - 1
    if n_segments != 3:
        raise ValueError(
            f"Dataset {path} must use exactly 3 segments for this script. "
            f"Found {n_segments} segments from segment_edges={segment_edges.tolist()}."
        )

    coef = np.asarray(dataset[_segment_gain_var_name(model_name)].values, dtype=float)
    if coef.ndim != 3:
        raise ValueError(
            f"Expected selected segment coefficients with shape "
            f"(trajectory, segment_basis, unit), got {coef.shape} in {path}."
        )
    if coef.shape[1] != n_segments:
        raise ValueError(
            f"Expected one coefficient row per segment in {path}. "
            f"Found {coef.shape[1]} coefficient rows and {n_segments} segments."
        )


def _validate_supported_trajectories(dataset, *, path: Path) -> None:
    """Ensure the dataset contains all four expected trajectory types."""
    dataset_trajectories = {
        str(value) for value in np.asarray(dataset.coords["trajectory"].values)
    }
    missing = [trajectory for trajectory in TRAJECTORIES if trajectory not in dataset_trajectories]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required trajectories: {missing!r}. "
            f"Found {sorted(dataset_trajectories)!r}."
        )


def _segment_edges(dataset) -> np.ndarray:
    """Return validated segment edges from one selected dataset."""
    segment_edges = np.asarray(dataset["segment_edges"].values, dtype=float).reshape(-1)
    if segment_edges.ndim != 1 or segment_edges.size < 2:
        raise ValueError(
            f"segment_edges must be a 1D array with len>=2. Got {segment_edges.shape}."
        )
    if np.any(np.diff(segment_edges) <= 0):
        raise ValueError("segment_edges must be strictly increasing.")
    return segment_edges


def _segment_gain_var_name(model_name: str) -> str:
    """Return the selected segment-gain coefficient variable name."""
    if model_name == "task_segment_bump":
        return "coef_segment_bump_gain"
    if model_name == "task_segment_scalar":
        return "coef_segment_scalar_gain"
    raise ValueError(f"Unsupported selected segment model {model_name!r}.")


def _align_by_unit_ids(
    ids_a: np.ndarray,
    values_a: np.ndarray,
    ids_b: np.ndarray,
    values_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two unit-indexed arrays on their shared unit ids."""
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    common, idx_a, idx_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    if common.size == 0:
        raise ValueError("No overlapping unit_ids were found.")
    return common, np.asarray(values_a)[idx_a], np.asarray(values_b)[idx_b]


def _corr_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a Pearson correlation between two 1D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def _python_scalar(value: Any) -> Any:
    """Return a plain Python scalar when possible for parquet friendliness."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float_attr(dataset, name: str) -> float:
    """Return a numeric dataset attribute or NaN when absent."""
    value = dataset.attrs.get(name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _int_attr(dataset, name: str) -> int | float:
    """Return an integer dataset attribute or NaN when absent."""
    value = dataset.attrs.get(name, np.nan)
    try:
        return int(value)
    except (TypeError, ValueError):
        return np.nan


def _dataset_selection_metadata(dataset) -> dict[str, Any]:
    """Return selected-fit metadata useful for output tables."""
    return {
        "selected_ridge": _float_attr(dataset, "selected_ridge"),
        "selected_bin_size_s": _float_attr(dataset, "selected_bin_size_s"),
        "selected_n_splines": _int_attr(dataset, "selected_n_splines"),
        "selection_score": _float_attr(dataset, "selection_score"),
        "selection_metric": str(dataset.attrs.get("selection_metric", "")),
        "selection_model_name": str(dataset.attrs.get("selection_model_name", "")),
        "selection_visual_ridge": _float_attr(dataset, "selection_visual_ridge"),
        "selection_visual_score": _float_attr(dataset, "selection_visual_score"),
    }


def _coefficient_mode(include_light_offset: bool) -> str:
    """Return the output label for the compared coefficient value."""
    return "segment_plus_light_offset" if include_light_offset else "segment_only"


def _coefficient_axis_label(include_light_offset: bool) -> str:
    """Return a concise axis label for plotted coefficient values."""
    if include_light_offset:
        return "segment coefficient + light offset"
    return "segment coefficient"


def _segment_coefficients(
    dataset,
    *,
    model_name: str,
    trajectory: str,
    segment_index: int,
    include_light_offset: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one segment coefficient vector for one trajectory."""
    segment_edges = _segment_edges(dataset)
    n_segments = segment_edges.size - 1
    if segment_index < 0 or segment_index >= n_segments:
        raise ValueError(
            f"segment_index={segment_index} out of range for n_segments={n_segments}."
        )

    gamma = np.asarray(
        dataset[_segment_gain_var_name(model_name)].sel(trajectory=trajectory).values,
        dtype=float,
    )
    if gamma.ndim != 2:
        raise ValueError(
            f"Expected segment coefficients with shape (segment_basis, unit), "
            f"got {gamma.shape}."
        )
    if gamma.shape[0] != n_segments:
        raise ValueError(
            "This script expects one saved coefficient row per segment. "
            f"Got gamma.shape[0]={gamma.shape[0]} and n_segments={n_segments}."
        )

    values = np.asarray(gamma[segment_index, :], dtype=float)
    if include_light_offset:
        if "coef_light_offset" not in dataset:
            raise ValueError(
                "Cannot include light offset because selected dataset is missing "
                "'coef_light_offset'."
            )
        light_offset = np.asarray(
            dataset["coef_light_offset"].sel(trajectory=trajectory).values,
            dtype=float,
        ).reshape(-1)
        if light_offset.shape != values.shape:
            raise ValueError(
                "coef_light_offset shape does not match segment coefficient shape. "
                f"Got {light_offset.shape} and {values.shape}."
            )
        values = values + light_offset

    unit_ids = np.asarray(dataset.coords["unit"].values)
    return unit_ids, values, segment_edges


def _base_row_metadata(
    *,
    args: argparse.Namespace,
    model_name: str,
    meta1: dict[str, Any],
    meta2: dict[str, Any],
) -> dict[str, Any]:
    """Return output metadata shared by point and summary rows."""
    return {
        "animal_name": args.animal_name,
        "date": args.date,
        "region": args.region,
        "model_name": model_name,
        "model_family": model_name,
        "light_epoch1": args.light_epoch1,
        "light_epoch2": args.light_epoch2,
        "dark_epoch": args.dark_epoch,
        "coefficient_mode": _coefficient_mode(args.include_light_offset),
        "include_light_offset": bool(args.include_light_offset),
        "selected_ridge_light_epoch1": float(meta1["selected_ridge"]),
        "selected_ridge_light_epoch2": float(meta2["selected_ridge"]),
        "selected_bin_size_s_light_epoch1": float(meta1["selected_bin_size_s"]),
        "selected_bin_size_s_light_epoch2": float(meta2["selected_bin_size_s"]),
        "selected_n_splines_light_epoch1": meta1["selected_n_splines"],
        "selected_n_splines_light_epoch2": meta2["selected_n_splines"],
        "selection_score_light_epoch1": float(meta1["selection_score"]),
        "selection_score_light_epoch2": float(meta2["selection_score"]),
        "selection_metric_light_epoch1": meta1["selection_metric"],
        "selection_metric_light_epoch2": meta2["selection_metric"],
        "selection_model_name_light_epoch1": meta1["selection_model_name"],
        "selection_model_name_light_epoch2": meta2["selection_model_name"],
        "selection_visual_ridge_light_epoch1": float(meta1["selection_visual_ridge"]),
        "selection_visual_ridge_light_epoch2": float(meta2["selection_visual_ridge"]),
        "selection_visual_score_light_epoch1": float(meta1["selection_visual_score"]),
        "selection_visual_score_light_epoch2": float(meta2["selection_visual_score"]),
    }


def build_comparison_tables(
    dataset1,
    dataset2,
    *,
    args: argparse.Namespace,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-unit and per-panel comparison tables for two light epochs."""
    segment_edges1 = _segment_edges(dataset1)
    segment_edges2 = _segment_edges(dataset2)
    if not np.allclose(segment_edges1, segment_edges2):
        raise ValueError("Compared datasets use different segment_edges and cannot be aligned.")

    meta1 = _dataset_selection_metadata(dataset1)
    meta2 = _dataset_selection_metadata(dataset2)
    shared_metadata = _base_row_metadata(
        args=args,
        model_name=model_name,
        meta1=meta1,
        meta2=meta2,
    )

    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for trajectory in TRAJECTORIES:
        switched_segment = SWITCHED_SEGMENT_BY_TRAJECTORY[trajectory]
        for segment_index in range(segment_edges1.size - 1):
            unit_ids1, coeff1, _ = _segment_coefficients(
                dataset1,
                model_name=model_name,
                trajectory=trajectory,
                segment_index=segment_index,
                include_light_offset=args.include_light_offset,
            )
            unit_ids2, coeff2, _ = _segment_coefficients(
                dataset2,
                model_name=model_name,
                trajectory=trajectory,
                segment_index=segment_index,
                include_light_offset=args.include_light_offset,
            )
            common_units, aligned1, aligned2 = _align_by_unit_ids(
                unit_ids1,
                coeff1,
                unit_ids2,
                coeff2,
            )
            valid = np.isfinite(aligned1) & np.isfinite(aligned2)
            valid_units = common_units[valid]
            valid_coeff1 = aligned1[valid]
            valid_coeff2 = aligned2[valid]
            correlation = _corr_fast(valid_coeff1, valid_coeff2)

            segment_start = float(segment_edges1[segment_index])
            segment_end = float(segment_edges1[segment_index + 1])
            segment_center = float(0.5 * (segment_start + segment_end))
            is_switched_segment = bool(segment_index == switched_segment)

            summary_rows.append(
                {
                    **shared_metadata,
                    "trajectory": trajectory,
                    "segment_index": int(segment_index),
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "segment_center": segment_center,
                    "is_switched_segment": is_switched_segment,
                    "correlation_coefficient": float(correlation),
                    "n_units": int(valid_units.size),
                    "n_common_units": int(common_units.size),
                }
            )

            for unit_id, coefficient1, coefficient2 in zip(
                valid_units,
                valid_coeff1,
                valid_coeff2,
                strict=True,
            ):
                point_rows.append(
                    {
                        **shared_metadata,
                        "trajectory": trajectory,
                        "segment_index": int(segment_index),
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "segment_center": segment_center,
                        "is_switched_segment": is_switched_segment,
                        "unit_id": _python_scalar(unit_id),
                        "coef_light_epoch1": float(coefficient1),
                        "coef_light_epoch2": float(coefficient2),
                    }
                )

    point_table = pd.DataFrame(point_rows)
    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table.sort_values(
        ["trajectory", "segment_index"], ignore_index=True
    )
    if not point_table.empty:
        point_table = point_table.sort_values(
            ["trajectory", "segment_index", "unit_id"], ignore_index=True
        )
    return point_table, summary_table


def _trajectory_limits(points: pd.DataFrame) -> tuple[float, float]:
    """Return shared x/y limits for one trajectory's scatter figure."""
    if points.empty:
        return -1.0, 1.0
    values = np.concatenate(
        [
            points["coef_light_epoch1"].to_numpy(dtype=float),
            points["coef_light_epoch2"].to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        pad = max(abs(lower), 1.0) * 0.05
        return lower - pad, upper + pad
    pad = 0.05 * (upper - lower)
    return lower - pad, upper + pad


def plot_trajectory_scatter(
    point_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    trajectory: str,
    light_epoch1: str,
    light_epoch2: str,
    region: str,
    model_name: str,
    dark_epoch: str,
    include_light_offset: bool,
):
    """Return the three-panel scatter figure for one trajectory."""
    import matplotlib.pyplot as plt

    point_subset = point_table.loc[point_table["trajectory"] == trajectory].copy()
    summary_subset = summary_table.loc[summary_table["trajectory"] == trajectory].copy()
    summary_subset = summary_subset.sort_values("segment_index", ignore_index=True)
    limits = _trajectory_limits(point_subset)
    color = TRAJECTORY_COLORS[trajectory]
    coefficient_label = _coefficient_axis_label(include_light_offset)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for axis, segment_index in zip(axes, range(3), strict=True):
        panel_points = point_subset.loc[point_subset["segment_index"] == segment_index].copy()
        panel_summary = summary_subset.loc[summary_subset["segment_index"] == segment_index]
        if panel_summary.empty:
            raise ValueError(
                f"Missing summary row for trajectory={trajectory!r}, "
                f"segment_index={segment_index}."
            )
        row = panel_summary.iloc[0]

        if bool(row["is_switched_segment"]):
            axis.set_facecolor("#f3f3f3")

        if not panel_points.empty:
            axis.scatter(
                panel_points["coef_light_epoch1"].to_numpy(dtype=float),
                panel_points["coef_light_epoch2"].to_numpy(dtype=float),
                s=18,
                color=color,
                alpha=0.85,
                edgecolors="none",
            )

        axis.plot(limits, limits, linestyle="--", color="0.35", linewidth=1.2)
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)

        title = (
            f"Segment {segment_index} "
            f"({float(row['segment_start']):.2f}-{float(row['segment_end']):.2f})"
        )
        if bool(row["is_switched_segment"]):
            title += " (switched)"
        axis.set_title(title)
        axis.text(
            0.04,
            0.96,
            f"r = {float(row['correlation_coefficient']):.2f}\n"
            f"n = {int(row['n_units'])}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
        )
        axis.set_xlabel(f"{light_epoch1} {coefficient_label}")

    axes[0].set_ylabel(f"{light_epoch2} {coefficient_label}")
    fig.suptitle(
        "Selected segment coefficient comparison\n"
        f"{region.upper()} | {trajectory} | {light_epoch1} vs {light_epoch2} | "
        f"dark ref {dark_epoch} | {model_name}"
    )
    fig.tight_layout()
    return fig


def plot_correlation_summary(
    summary_table: pd.DataFrame,
    *,
    region: str,
    model_name: str,
    light_epoch1: str,
    light_epoch2: str,
    dark_epoch: str,
    include_light_offset: bool,
):
    """Return the outbound/inbound summary figure of segment-wise correlations."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    correlation_values = summary_table["correlation_coefficient"].to_numpy(dtype=float)
    correlation_values = correlation_values[np.isfinite(correlation_values)]
    if correlation_values.size == 0:
        y_limits = (-1.0, 1.0)
    else:
        y_min = float(np.min(correlation_values))
        y_max = float(np.max(correlation_values))
        if np.isclose(y_min, y_max):
            pad = max(abs(y_min), 0.1) * 0.1
        else:
            pad = 0.08 * (y_max - y_min)
        y_limits = (y_min - pad, y_max + pad)

    for axis, (group_name, trajectories) in zip(axes, TRAJECTORY_GROUPS.items(), strict=True):
        switched_segment = SWITCHED_SEGMENT_BY_TRAJECTORY[trajectories[0]]
        if any(
            SWITCHED_SEGMENT_BY_TRAJECTORY[trajectory] != switched_segment
            for trajectory in trajectories
        ):
            raise ValueError(f"Trajectories in group {group_name!r} do not share a switched segment.")

        axis.axvspan(
            switched_segment - 0.18,
            switched_segment + 0.18,
            color="#f1f1f1",
            zorder=0,
        )
        for trajectory in trajectories:
            subset = summary_table.loc[summary_table["trajectory"] == trajectory].copy()
            subset = subset.sort_values("segment_index", ignore_index=True)
            color = TRAJECTORY_COLORS[trajectory]
            x = subset["segment_index"].to_numpy(dtype=int)
            y = subset["correlation_coefficient"].to_numpy(dtype=float)
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                color=color,
                label=trajectory,
            )
            switched_mask = subset["is_switched_segment"].to_numpy(dtype=bool)
            if np.any(switched_mask):
                switched_x = x[switched_mask]
                switched_y = y[switched_mask]
                axis.scatter(
                    switched_x,
                    switched_y,
                    marker="*",
                    s=180,
                    color=color,
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=4,
                )

        label_rows = (
            summary_table.loc[
                summary_table["trajectory"] == trajectories[0],
                ["segment_index", "segment_start", "segment_end"],
            ]
            .drop_duplicates()
            .sort_values("segment_index", ignore_index=True)
        )
        axis.set_title(f"{group_name.capitalize()} (switched: segment {switched_segment})")
        axis.set_xlabel("Segment")
        axis.set_xticks(label_rows["segment_index"].to_numpy(dtype=int))
        axis.set_xticklabels(
            [
                f"{int(row.segment_index)}\n({float(row.segment_start):.2f}-{float(row.segment_end):.2f})"
                for row in label_rows.itertuples(index=False)
            ]
        )
        axis.set_ylim(*y_limits)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

    axes[0].set_ylabel("Correlation coefficient")
    fig.suptitle(
        "Selected segment coefficient comparison summary\n"
        f"{region.upper()} | {model_name} | {light_epoch1} vs {light_epoch2} | "
        f"dark ref {dark_epoch} | {_coefficient_axis_label(include_light_offset)}"
    )
    fig.tight_layout()
    return fig


def _output_stem(args: argparse.Namespace, *, model_name: str) -> str:
    """Build the shared output filename stem for this comparison."""
    offset_tag = "with_light_offset" if args.include_light_offset else "segment_only"
    return (
        f"{args.region}_{model_name}_{args.light_epoch1}_vs_{args.light_epoch2}"
        f"_ref_{args.dark_epoch}_{offset_tag}_selected_segment_coefficients"
    )


def _run_one_model_name(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    fig_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    """Run the selected comparison workflow for one model and save outputs."""
    dataset1_path = _dataset_path(
        input_dir,
        region=args.region,
        light_epoch=args.light_epoch1,
        dark_epoch=args.dark_epoch,
        model_name=model_name,
    )
    dataset2_path = _dataset_path(
        input_dir,
        region=args.region,
        light_epoch=args.light_epoch2,
        dark_epoch=args.dark_epoch,
        model_name=model_name,
    )

    dataset1 = _load_selected_dataset(
        dataset1_path,
        expected_region=args.region,
        expected_light_epoch=args.light_epoch1,
        expected_dark_epoch=args.dark_epoch,
        expected_model_name=model_name,
    )
    dataset2 = _load_selected_dataset(
        dataset2_path,
        expected_region=args.region,
        expected_light_epoch=args.light_epoch2,
        expected_dark_epoch=args.dark_epoch,
        expected_model_name=model_name,
    )

    try:
        point_table, summary_table = build_comparison_tables(
            dataset1,
            dataset2,
            args=args,
            model_name=model_name,
        )

        stem = _output_stem(args, model_name=model_name)
        points_parquet_path = output_dir / f"{stem}_points.parquet"
        summary_parquet_path = output_dir / f"{stem}_summary.parquet"
        point_table.to_parquet(points_parquet_path, index=False)
        summary_table.to_parquet(summary_parquet_path, index=False)

        figure_paths: list[Path] = []
        figures = []
        for trajectory in TRAJECTORIES:
            fig = plot_trajectory_scatter(
                point_table,
                summary_table,
                trajectory=trajectory,
                light_epoch1=args.light_epoch1,
                light_epoch2=args.light_epoch2,
                region=args.region,
                model_name=model_name,
                dark_epoch=args.dark_epoch,
                include_light_offset=args.include_light_offset,
            )
            figure_path = fig_dir / f"{stem}_{trajectory}.png"
            fig.savefig(figure_path, dpi=200, bbox_inches="tight")
            figure_paths.append(figure_path)
            figures.append(fig)

        summary_figure = plot_correlation_summary(
            summary_table,
            region=args.region,
            model_name=model_name,
            light_epoch1=args.light_epoch1,
            light_epoch2=args.light_epoch2,
            dark_epoch=args.dark_epoch,
            include_light_offset=args.include_light_offset,
        )
        summary_figure_path = fig_dir / f"{stem}_correlation_summary.png"
        summary_figure.savefig(summary_figure_path, dpi=200, bbox_inches="tight")
        figure_paths.append(summary_figure_path)
        figures.append(summary_figure)

        if args.show:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            for fig in figures:
                plt.close(fig)

        return {
            "model_name": model_name,
            "dataset_paths": {
                "light_epoch1": dataset1_path,
                "light_epoch2": dataset2_path,
            },
            "selected_parameters": {
                "light_epoch1": _dataset_selection_metadata(dataset1),
                "light_epoch2": _dataset_selection_metadata(dataset2),
            },
            "coefficient_mode": _coefficient_mode(args.include_light_offset),
            "points_parquet": points_parquet_path,
            "summary_parquet": summary_parquet_path,
            "figure_paths": figure_paths,
        }
    finally:
        dataset1.close()
        dataset2.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for selected coefficient comparisons."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected task-segment coefficients across two light epochs "
            "using NetCDF outputs from dark_light_glm."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal name.")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format.")
    parser.add_argument(
        "--region",
        required=True,
        choices=("v1", "ca1"),
        help="Region to analyze.",
    )
    parser.add_argument(
        "--models",
        "--model-families",
        "--model-family",
        dest="models",
        nargs="+",
        choices=MODEL_NAME_CHOICES,
        help=(
            "Selected segment models to analyze. If omitted, the script runs "
            "all supported selected segment models available for both requested "
            "light epochs. Compatible deprecated dark_light_glm aliases are "
            "normalized."
        ),
    )
    parser.add_argument(
        "--light-epoch1",
        "--light-train-epoch1",
        dest="light_epoch1",
        required=True,
        help="First light-training epoch to compare.",
    )
    parser.add_argument(
        "--light-epoch2",
        "--light-train-epoch2",
        dest="light_epoch2",
        required=True,
        help="Second light-training epoch to compare.",
    )
    parser.add_argument(
        "--dark-epoch",
        "--dark-train-epoch",
        dest="dark_epoch",
        required=True,
        help="Shared dark-training epoch used in the selected dark/light fits.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base analysis directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--input-dir",
        "--dark-light-glm-dir",
        dest="input_dir",
        type=Path,
        help=(
            "Directory containing selected dark_light_glm NetCDF files. "
            "Default: analysis_path / 'task_progression' / 'dark_light_glm' / 'selected'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for saved parquet summaries. "
            "Default: analysis_path / 'task_progression' / 'segment_coefficient_comparison'"
        ),
    )
    parser.add_argument(
        "--include-light-offset",
        "--swap-light-offset",
        dest="include_light_offset",
        action="store_true",
        help=(
            "Add the selected trajectory's scalar light-offset coefficient to "
            "each segment coefficient before comparing epochs. Default: compare "
            "segment-local coefficients only."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the saved figures interactively after writing them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected segment-coefficient comparison workflow."""
    args = parse_arguments()
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    input_dir = (
        args.input_dir
        if args.input_dir is not None
        else get_task_progression_output_dir(analysis_path, "dark_light_glm") / "selected"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_names, model_messages = _resolve_model_names(
        input_dir,
        region=args.region,
        light_epoch1=args.light_epoch1,
        light_epoch2=args.light_epoch2,
        dark_epoch=args.dark_epoch,
        requested_model_names=args.models,
    )
    for message in model_messages:
        print(message)

    model_results = []
    for model_name in model_names:
        model_results.append(
            _run_one_model_name(
                args=args,
                input_dir=input_dir,
                output_dir=output_dir,
                fig_dir=fig_dir,
                model_name=model_name,
            )
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.segment_coefficient_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "region": args.region,
            "models_requested": args.models,
            "models_run": model_names,
            "light_epoch1": args.light_epoch1,
            "light_epoch2": args.light_epoch2,
            "dark_epoch": args.dark_epoch,
            "data_root": args.data_root,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "fig_dir": fig_dir,
            "include_light_offset": bool(args.include_light_offset),
            "coefficient_mode": _coefficient_mode(args.include_light_offset),
        },
        outputs={
            "model_outputs": {
                result["model_name"]: {
                    "model_name": result["model_name"],
                    "source_datasets": result["dataset_paths"],
                    "selected_parameters": result["selected_parameters"],
                    "coefficient_mode": result["coefficient_mode"],
                    "saved_parquets": {
                        "point_table": result["points_parquet"],
                        "summary_table": result["summary_parquet"],
                    },
                    "saved_figures": result["figure_paths"],
                }
                for result in model_results
            },
        },
    )

    for result in model_results:
        print(f"[{result['model_name']}] Saved point table to {result['points_parquet']}")
        print(f"[{result['model_name']}] Saved summary table to {result['summary_parquet']}")
        print(
            f"[{result['model_name']}] Saved figures to "
            f"{[str(path) for path in result['figure_paths']]}"
        )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
