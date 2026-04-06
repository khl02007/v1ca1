from __future__ import annotations

"""Compare raw segment coefficients across two light epochs.

This module loads segment-based dark/light GLM fits from
`v1ca1.task_progression.dark_light_glm`, aligns matching units between two
light epochs, and compares the raw segment interaction coefficients for each
trajectory and segment. It saves a long-form parquet table with one row per
matched unit, a compact summary parquet with one row per trajectory/segment,
trajectory-specific scatter figures, and an outbound/inbound correlation
summary figure.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import DEFAULT_DATA_ROOT, get_analysis_path


SUPPORTED_MODEL_FAMILIES = (
    "segment_bump_gain",
    "segment_scalar_gain",
    "overlapping_segment_bump_gain",
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for segment-coefficient comparisons."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw segment coefficients across two light epochs using "
            "NetCDF outputs from dark_light_glm."
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
        "--model-family",
        choices=SUPPORTED_MODEL_FAMILIES,
        help=(
            "Optional segment-based dark/light model family to analyze. "
            "If omitted, the script runs all supported families available for "
            "both requested light epochs."
        ),
    )
    parser.add_argument(
        "--light-epoch1",
        required=True,
        help="First light epoch to compare.",
    )
    parser.add_argument(
        "--light-epoch2",
        required=True,
        help="Second light epoch to compare.",
    )
    parser.add_argument(
        "--dark-epoch",
        required=True,
        help="Shared dark epoch used in the underlying dark/light fits.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base analysis directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing dark_light_glm NetCDF files. "
            "Default: analysis_path / 'task_progression_dark_light'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for saved parquet summaries. "
            "Default: analysis_path / 'segment_coefficient_comparison'"
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        help=(
            "Optional preferred ridge value. If provided, the nearest saved ridge "
            "is used if needed. If omitted, the script runs all ridge values saved "
            "in both compared datasets."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the saved figures interactively after writing them.",
    )
    return parser.parse_args()


def _dataset_path(
    input_dir: Path,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_family: str,
) -> Path:
    """Return the expected NetCDF path for one dark/light fit dataset."""
    return input_dir / f"{region}_{light_epoch}_vs_{dark_epoch}_{model_family}.nc"


def _resolve_model_families(
    input_dir: Path,
    *,
    region: str,
    light_epoch1: str,
    light_epoch2: str,
    dark_epoch: str,
    requested_model_family: str | None,
) -> list[str]:
    """Return the requested model family or all supported families available for both epochs."""
    if requested_model_family is not None:
        return [requested_model_family]

    available_families = []
    for model_family in SUPPORTED_MODEL_FAMILIES:
        dataset1_path = _dataset_path(
            input_dir,
            region=region,
            light_epoch=light_epoch1,
            dark_epoch=dark_epoch,
            model_family=model_family,
        )
        dataset2_path = _dataset_path(
            input_dir,
            region=region,
            light_epoch=light_epoch2,
            dark_epoch=dark_epoch,
            model_family=model_family,
        )
        if dataset1_path.exists() and dataset2_path.exists():
            available_families.append(model_family)

    if not available_families:
        raise FileNotFoundError(
            "No supported model families were found for both requested light epochs in "
            f"{input_dir}. Checked families: {list(SUPPORTED_MODEL_FAMILIES)!r}."
        )
    return available_families


def _load_fit_dataset(
    path: Path,
    *,
    expected_region: str,
    expected_light_epoch: str,
    expected_dark_epoch: str,
    expected_model_family: str,
):
    """Load one dark/light fit dataset and validate its core metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Dark/light fit dataset not found: {path}")

    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to load dark_light_glm NetCDF files."
        ) from exc

    dataset = xr.load_dataset(path)

    region = str(dataset.attrs.get("region", ""))
    light_epoch = str(dataset.attrs.get("light_epoch", ""))
    dark_epoch = str(dataset.attrs.get("dark_epoch", ""))
    model_family = str(dataset.attrs.get("model_family", ""))
    if region and region != expected_region:
        raise ValueError(
            f"Dataset region mismatch for {path}: expected {expected_region!r}, "
            f"found {region!r}."
        )
    if light_epoch and light_epoch != expected_light_epoch:
        raise ValueError(
            f"Dataset light_epoch mismatch for {path}: expected "
            f"{expected_light_epoch!r}, found {light_epoch!r}."
        )
    if dark_epoch and dark_epoch != expected_dark_epoch:
        raise ValueError(
            f"Dataset dark_epoch mismatch for {path}: expected {expected_dark_epoch!r}, "
            f"found {dark_epoch!r}."
        )
    if model_family and model_family != expected_model_family:
        raise ValueError(
            f"Dataset model_family mismatch for {path}: expected "
            f"{expected_model_family!r}, found {model_family!r}."
        )

    if "segment_edges" not in dataset:
        raise ValueError(
            f"Dataset {path} does not contain segment_edges and is not supported by "
            "this script."
        )

    gain_basis = str(dataset.attrs.get("gain_basis", ""))
    if gain_basis not in {"segment_raised_cosine", "segment_scalar"}:
        raise ValueError(
            f"Dataset {path} uses gain_basis={gain_basis!r}, which is not supported by "
            "this script."
        )

    required_vars = [_segment_gain_var_name(dataset)]
    missing_vars = [name for name in required_vars if name not in dataset]
    if missing_vars:
        raise ValueError(f"Dataset {path} is missing required variables: {missing_vars}")

    _validate_segment_layout(dataset, path=path)
    _validate_supported_trajectories(dataset, path=path)
    return dataset


def _validate_segment_layout(dataset, *, path: Path) -> None:
    """Require the saved fit to use exactly three task-progression segments."""
    segment_edges, _ = _segment_metadata(dataset)
    n_segments = segment_edges.size - 1
    if n_segments != 3:
        raise ValueError(
            f"Dataset {path} must use exactly 3 segments for this script. "
            f"Found {n_segments} segments from segment_edges={segment_edges.tolist()}."
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


def _select_ridge(dataset, ridge: float) -> float:
    """Return the exact or nearest saved ridge value from one fit dataset."""
    ridge_values = np.asarray(dataset.coords["ridge"].values, dtype=float).reshape(-1)
    if ridge_values.size == 0:
        raise ValueError("The fit dataset does not contain any ridge values.")
    if np.any(ridge_values <= 0):
        raise ValueError(
            f"Saved ridge values must be positive. Got {ridge_values.tolist()}."
        )
    if ridge in ridge_values:
        return float(ridge)
    ridge_index = int(
        np.argmin(np.abs(np.log10(ridge_values) - np.log10(float(ridge))))
    )
    return float(ridge_values[ridge_index])


def _common_saved_ridges(dataset1, dataset2) -> list[float]:
    """Return the exact saved ridge values shared by two datasets."""
    ridge_values1 = np.asarray(dataset1.coords["ridge"].values, dtype=float).reshape(-1)
    ridge_values2 = np.asarray(dataset2.coords["ridge"].values, dtype=float).reshape(-1)
    common = np.intersect1d(ridge_values1, ridge_values2)
    common = np.asarray(common, dtype=float)
    common = common[np.isfinite(common)]
    common = common[common > 0]
    if common.size == 0:
        raise ValueError("The compared datasets do not share any saved positive ridge values.")
    return [float(value) for value in common.tolist()]


def _requested_ridge_value(args: argparse.Namespace) -> float:
    """Return a parquet-friendly requested ridge value."""
    return np.nan if args.ridge is None else float(args.ridge)


def _format_ridge_tag(value: float) -> str:
    """Return a compact filesystem-safe string for one ridge value."""
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _ridge_output_tag(*, ridge_used1: float, ridge_used2: float) -> str:
    """Return the filename tag encoding the ridge values used for one comparison."""
    tag1 = _format_ridge_tag(ridge_used1)
    tag2 = _format_ridge_tag(ridge_used2)
    if np.isclose(ridge_used1, ridge_used2):
        return f"ridge_{tag1}"
    return f"ridge1_{tag1}_ridge2_{tag2}"


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


def _segment_metadata(dataset) -> tuple[np.ndarray, str]:
    """Return validated segment edges and gain basis metadata."""
    segment_edges = np.asarray(dataset["segment_edges"].values, dtype=float).reshape(-1)
    if segment_edges.ndim != 1 or segment_edges.size < 2:
        raise ValueError(
            f"segment_edges must be a 1D array with len>=2. Got {segment_edges.shape}."
        )
    if np.any(np.diff(segment_edges) <= 0):
        raise ValueError("segment_edges must be strictly increasing.")
    gain_basis = str(dataset.attrs.get("gain_basis", ""))
    return segment_edges, gain_basis


def _segment_gain_var_name(dataset) -> str:
    """Return the saved segment-gain coefficient variable name for one dataset."""
    _, gain_basis = _segment_metadata(dataset)
    if gain_basis == "segment_raised_cosine":
        return "coef_segment_bump_gain_full_all"
    if gain_basis == "segment_scalar":
        return "coef_segment_scalar_gain_full_all"
    raise ValueError(f"Unsupported gain_basis {gain_basis!r}.")


def _segment_coefficients(
    dataset,
    *,
    trajectory: str,
    ridge_used: float,
    segment_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one raw segment coefficient vector for one trajectory."""
    segment_edges, _ = _segment_metadata(dataset)
    n_segments = segment_edges.size - 1
    if segment_index < 0 or segment_index >= n_segments:
        raise ValueError(
            f"segment_index={segment_index} out of range for n_segments={n_segments}."
        )

    gamma = np.asarray(
        dataset[_segment_gain_var_name(dataset)]
        .sel(trajectory=trajectory, ridge=ridge_used)
        .values,
        dtype=float,
    )
    if gamma.ndim != 2:
        raise ValueError(
            f"Expected segment coefficients with shape (basis, unit), got {gamma.shape}."
        )
    if gamma.shape[0] != n_segments:
        raise ValueError(
            "This script expects one saved coefficient row per segment. "
            f"Got gamma.shape[0]={gamma.shape[0]} and n_segments={n_segments}."
        )
    unit_ids = np.asarray(dataset.coords["unit"].values)
    return unit_ids, np.asarray(gamma[segment_index, :], dtype=float), segment_edges


def _python_scalar(value: Any) -> Any:
    """Return a plain Python scalar when possible for parquet friendliness."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_comparison_tables(
    dataset1,
    dataset2,
    *,
    args: argparse.Namespace,
    ridge_used1: float,
    ridge_used2: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-unit and per-panel comparison tables for two light epochs."""
    segment_edges1, _ = _segment_metadata(dataset1)
    segment_edges2, _ = _segment_metadata(dataset2)
    if not np.allclose(segment_edges1, segment_edges2):
        raise ValueError("Compared datasets use different segment_edges and cannot be aligned.")

    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for trajectory in TRAJECTORIES:
        switched_segment = SWITCHED_SEGMENT_BY_TRAJECTORY[trajectory]
        for segment_index in range(segment_edges1.size - 1):
            unit_ids1, coeff1, _ = _segment_coefficients(
                dataset1,
                trajectory=trajectory,
                ridge_used=ridge_used1,
                segment_index=segment_index,
            )
            unit_ids2, coeff2, _ = _segment_coefficients(
                dataset2,
                trajectory=trajectory,
                ridge_used=ridge_used2,
                segment_index=segment_index,
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
                    "animal_name": args.animal_name,
                    "date": args.date,
                    "region": args.region,
                    "model_family": args.model_family,
                    "light_epoch1": args.light_epoch1,
                    "light_epoch2": args.light_epoch2,
                    "dark_epoch": args.dark_epoch,
                    "ridge_requested": _requested_ridge_value(args),
                    "ridge_used_light_epoch1": float(ridge_used1),
                    "ridge_used_light_epoch2": float(ridge_used2),
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
                        "animal_name": args.animal_name,
                        "date": args.date,
                        "region": args.region,
                        "model_family": args.model_family,
                        "light_epoch1": args.light_epoch1,
                        "light_epoch2": args.light_epoch2,
                        "dark_epoch": args.dark_epoch,
                        "ridge_requested": _requested_ridge_value(args),
                        "ridge_used_light_epoch1": float(ridge_used1),
                        "ridge_used_light_epoch2": float(ridge_used2),
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
    model_family: str,
    dark_epoch: str,
):
    """Return the three-panel scatter figure for one trajectory."""
    import matplotlib.pyplot as plt

    point_subset = point_table.loc[point_table["trajectory"] == trajectory].copy()
    summary_subset = summary_table.loc[summary_table["trajectory"] == trajectory].copy()
    summary_subset = summary_subset.sort_values("segment_index", ignore_index=True)
    limits = _trajectory_limits(point_subset)
    color = TRAJECTORY_COLORS[trajectory]

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
        axis.set_xlabel(f"{light_epoch1} coefficient")

    axes[0].set_ylabel(f"{light_epoch2} coefficient")
    fig.suptitle(
        "Segment coefficient comparison\n"
        f"{region.upper()} | {trajectory} | {light_epoch1} vs {light_epoch2} | "
        f"dark ref {dark_epoch} | {model_family}"
    )
    fig.tight_layout()
    return fig


def plot_correlation_summary(summary_table: pd.DataFrame):
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
    fig.suptitle("Segment coefficient comparison summary")
    fig.tight_layout()
    return fig


def _output_stem(
    args: argparse.Namespace,
    *,
    ridge_used1: float,
    ridge_used2: float,
) -> str:
    """Build the shared output filename stem for this comparison."""
    return (
        f"{args.region}_{args.model_family}_{args.light_epoch1}_vs_{args.light_epoch2}"
        f"_ref_{args.dark_epoch}_{_ridge_output_tag(ridge_used1=ridge_used1, ridge_used2=ridge_used2)}"
        f"_segment_coefficients"
    )


def _run_one_model_family(
    *,
    args: argparse.Namespace,
    analysis_path: Path,
    input_dir: Path,
    output_dir: Path,
    fig_dir: Path,
    model_family: str,
) -> list[dict[str, Any]]:
    """Run the comparison workflow for one model family and save its outputs."""
    dataset1_path = _dataset_path(
        input_dir,
        region=args.region,
        light_epoch=args.light_epoch1,
        dark_epoch=args.dark_epoch,
        model_family=model_family,
    )
    dataset2_path = _dataset_path(
        input_dir,
        region=args.region,
        light_epoch=args.light_epoch2,
        dark_epoch=args.dark_epoch,
        model_family=model_family,
    )

    dataset1 = _load_fit_dataset(
        dataset1_path,
        expected_region=args.region,
        expected_light_epoch=args.light_epoch1,
        expected_dark_epoch=args.dark_epoch,
        expected_model_family=model_family,
    )
    dataset2 = _load_fit_dataset(
        dataset2_path,
        expected_region=args.region,
        expected_light_epoch=args.light_epoch2,
        expected_dark_epoch=args.dark_epoch,
        expected_model_family=model_family,
    )

    try:
        requested_ridges = (
            [_select_ridge(dataset1, args.ridge), _select_ridge(dataset2, args.ridge)]
            if args.ridge is not None
            else None
        )
        ridge_pairs = (
            [(float(requested_ridges[0]), float(requested_ridges[1]))]
            if requested_ridges is not None
            else [(ridge_value, ridge_value) for ridge_value in _common_saved_ridges(dataset1, dataset2)]
        )

        family_results: list[dict[str, Any]] = []
        for ridge_used1, ridge_used2 in ridge_pairs:
            point_table, summary_table = build_comparison_tables(
                dataset1,
                dataset2,
                args=args,
                ridge_used1=ridge_used1,
                ridge_used2=ridge_used2,
            )

            family_args = argparse.Namespace(**vars(args))
            family_args.model_family = model_family
            stem = _output_stem(
                family_args,
                ridge_used1=ridge_used1,
                ridge_used2=ridge_used2,
            )
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
                    model_family=model_family,
                    dark_epoch=args.dark_epoch,
                )
                figure_path = fig_dir / f"{stem}_{trajectory}.png"
                fig.savefig(figure_path, dpi=200, bbox_inches="tight")
                figure_paths.append(figure_path)
                figures.append(fig)

            summary_figure = plot_correlation_summary(summary_table)
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

            family_results.append(
                {
                    "model_family": model_family,
                    "dataset_paths": {
                        "light_epoch1": dataset1_path,
                        "light_epoch2": dataset2_path,
                    },
                    "ridge_selection": {
                        "requested": _requested_ridge_value(args),
                        "light_epoch1": float(ridge_used1),
                        "light_epoch2": float(ridge_used2),
                    },
                    "points_parquet": points_parquet_path,
                    "summary_parquet": summary_parquet_path,
                    "figure_paths": figure_paths,
                }
            )

        return family_results
    finally:
        dataset1.close()
        dataset2.close()


def main() -> None:
    """Run the segment-coefficient comparison workflow."""
    args = parse_arguments()
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    input_dir = (
        args.input_dir
        if args.input_dir is not None
        else analysis_path / "task_progression_dark_light"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else analysis_path / "segment_coefficient_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = analysis_path / "figs" / "segment_coefficient_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_families = _resolve_model_families(
        input_dir,
        region=args.region,
        light_epoch1=args.light_epoch1,
        light_epoch2=args.light_epoch2,
        dark_epoch=args.dark_epoch,
        requested_model_family=args.model_family,
    )
    family_results = []
    for model_family in model_families:
        family_results.extend(
            _run_one_model_family(
                args=args,
                analysis_path=analysis_path,
                input_dir=input_dir,
                output_dir=output_dir,
                fig_dir=fig_dir,
                model_family=model_family,
            )
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.segment_coefficient_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "region": args.region,
            "model_family": args.model_family,
            "model_families_run": model_families,
            "light_epoch1": args.light_epoch1,
            "light_epoch2": args.light_epoch2,
            "dark_epoch": args.dark_epoch,
            "data_root": args.data_root,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "fig_dir": fig_dir,
            "ridge_requested": _requested_ridge_value(args),
        },
        outputs={
            "family_outputs": {
                result["model_family"]
                + "_"
                + _ridge_output_tag(
                    ridge_used1=result["ridge_selection"]["light_epoch1"],
                    ridge_used2=result["ridge_selection"]["light_epoch2"],
                ): {
                    "model_family": result["model_family"],
                    "source_datasets": result["dataset_paths"],
                    "ridge_selection": result["ridge_selection"],
                    "saved_parquets": {
                        "point_table": result["points_parquet"],
                        "summary_table": result["summary_parquet"],
                    },
                    "saved_figures": result["figure_paths"],
                }
                for result in family_results
            },
        },
    )

    for result in family_results:
        print(f"[{result['model_family']}] Saved point table to {result['points_parquet']}")
        print(f"[{result['model_family']}] Saved summary table to {result['summary_parquet']}")
        print(
            f"[{result['model_family']}] Saved figures to "
            f"{[str(path) for path in result['figure_paths']]}"
        )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
