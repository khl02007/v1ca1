from __future__ import annotations

"""Compare segment-wise visual-gain swap consistency across light epochs.

This module modernizes the legacy visual-gain field correlation workflow to use
NetCDF outputs from `v1ca1.task_progression.dark_light_glm`. It
loads one session's dark/light fit datasets for two light epochs and one shared
dark epoch, reconstructs per-segment visual-gain effects for matching units,
computes legacy left-vs-right swap-consistency statistics for outbound and
inbound trajectory pairs, and saves both a parquet summary table and a figure.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import DEFAULT_DATA_ROOT, get_analysis_path


SUPPORTED_MODEL_FAMILIES = (
    "mult_per_segment",
    "mult_per_segment_scalar",
    "mult_per_segment_overlap",
)
TRAJECTORIES = {
    "outbound": ("center_to_left", "center_to_right"),
    "inbound": ("left_to_center", "right_to_center"),
}
MODE_FULL_LIGHT = "full_light"
MODE_INTERACTION_ONLY = "interaction_only"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for visual-gain swap consistency plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare task-progression visual-gain swap consistency across two "
            "light epochs using NetCDF outputs from dark_light_glm."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--region",
        required=True,
        choices=("v1", "ca1"),
        help="Region to analyze.",
    )
    parser.add_argument(
        "--model-family",
        required=True,
        choices=SUPPORTED_MODEL_FAMILIES,
        help="Segment-based dark/light model family to analyze.",
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
        "--baseline-light-epoch",
        help=(
            "Optional baseline light epoch. Its light-vs-dark effect is "
            "subtracted from both compared light epochs before computing "
            "swap consistency."
        ),
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
            "Directory for saved parquet summaries and figures. "
            "Default: analysis_path / 'task_progression_visual_gain_correlation'"
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Preferred ridge value. The nearest saved ridge is used if needed.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Number of bootstrap resamples across neurons for each segment.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence interval width in percent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for bootstrapping.",
    )
    parser.add_argument(
        "--exclude-global-light-offset",
        action="store_true",
        help=(
            "Use only the segment interaction term. By default the full "
            "segment-level light effect includes the global light offset."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
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

    required_vars = [
        "coef_light_full_all",
        _segment_gain_var_name(dataset),
    ]
    missing_vars = [name for name in required_vars if name not in dataset]
    if missing_vars:
        raise ValueError(f"Dataset {path} is missing required variables: {missing_vars}")

    return dataset


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


def _align_by_unit_ids(ids_a, values_a, ids_b, values_b):
    """Align two unit-indexed arrays on their shared unit ids."""
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    common, idx_a, idx_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    if common.size == 0:
        raise ValueError("No overlapping unit_ids were found.")
    return common, np.asarray(values_a)[idx_a], np.asarray(values_b)[idx_b]


def _corr_fast(x, y) -> float:
    """Compute a Pearson correlation between two 1D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def _bootstrap_r(x, y, *, n_boot: int, ci: float, seed: int) -> tuple[float, float, float]:
    """Bootstrap one correlation estimate across neurons."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must match, got {x.shape} and {y.shape}.")

    rng = np.random.default_rng(seed)
    n_units = x.size
    r_obs = _corr_fast(x, y)
    bootstrap = np.full(n_boot, np.nan, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, n_units, size=n_units)
        bootstrap[index] = _corr_fast(x[sample], y[sample])
    bootstrap = bootstrap[np.isfinite(bootstrap)]

    alpha = (100.0 - float(ci)) / 2.0
    if bootstrap.size == 0:
        return float(r_obs), np.nan, np.nan
    ci_low, ci_high = np.percentile(bootstrap, [alpha, 100.0 - alpha])
    return float(r_obs), float(ci_low), float(ci_high)


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


def _segment_effect(
    dataset,
    *,
    trajectory: str,
    ridge_used: float,
    seg_idx: int,
    include_global_light: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one segment-level visual-gain effect vector for one trajectory."""
    segment_edges, gain_basis = _segment_metadata(dataset)
    unit_ids = np.asarray(dataset.coords["unit"].values)
    global_light = np.asarray(
        dataset["coef_light_full_all"].sel(trajectory=trajectory, ridge=ridge_used).values,
        dtype=float,
    )
    gamma = np.asarray(
        dataset[_segment_gain_var_name(dataset)]
        .sel(trajectory=trajectory, ridge=ridge_used)
        .values,
        dtype=float,
    )
    if gamma.ndim != 2:
        raise ValueError(
            f"Expected light modulation coefficients with shape (basis, unit), got {gamma.shape}."
        )

    if gain_basis == "segment_scalar":
        n_segments = segment_edges.size - 1
        if gamma.shape[0] != n_segments:
            raise ValueError(
                "segment_scalar datasets must store one interaction coefficient per "
                f"segment. Got gamma.shape[0]={gamma.shape[0]} and {n_segments} "
                "segments."
            )
        if seg_idx < 0 or seg_idx >= n_segments:
            raise ValueError(f"seg_idx={seg_idx} out of range for n_segments={n_segments}.")
        if include_global_light:
            effect = global_light + gamma[seg_idx, :]
        else:
            effect = gamma[seg_idx, :]
        return unit_ids, np.asarray(effect, dtype=float), segment_edges

    n_segments = segment_edges.size - 1
    if gamma.shape[0] != n_segments:
        raise ValueError(
            "This script only supports segment-based gain layouts with one "
            "interaction vector per segment. "
            f"Got gamma.shape[0]={gamma.shape[0]} and {n_segments} segments."
        )
    if seg_idx < 0 or seg_idx >= n_segments:
        raise ValueError(f"seg_idx={seg_idx} out of range for n_segments={n_segments}.")
    effect = gamma[seg_idx, :]
    if include_global_light and global_light.shape == effect.shape:
        effect = global_light + effect
    return unit_ids, np.asarray(effect, dtype=float), segment_edges


def _effect_segment_with_baseline(
    dataset,
    *,
    trajectory: str,
    ridge_used: float,
    seg_idx: int,
    include_global_light: bool,
    baseline_dataset=None,
    baseline_ridge_used: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one segment effect, optionally baseline-subtracted."""
    unit_ids, effect, segment_edges = _segment_effect(
        dataset,
        trajectory=trajectory,
        ridge_used=ridge_used,
        seg_idx=seg_idx,
        include_global_light=include_global_light,
    )
    if baseline_dataset is None:
        return unit_ids, effect, segment_edges

    if baseline_ridge_used is None:
        raise ValueError("baseline_ridge_used is required when baseline_dataset is given.")
    base_unit_ids, base_effect, _ = _segment_effect(
        baseline_dataset,
        trajectory=trajectory,
        ridge_used=baseline_ridge_used,
        seg_idx=seg_idx,
        include_global_light=include_global_light,
    )
    common_units, aligned_effect, aligned_baseline = _align_by_unit_ids(
        unit_ids,
        effect,
        base_unit_ids,
        base_effect,
    )
    return common_units, aligned_effect - aligned_baseline, segment_edges


def _left_right_contrast(
    dataset,
    *,
    direction: str,
    ridge_used: float,
    seg_idx: int,
    include_global_light: bool,
    baseline_dataset=None,
    baseline_ridge_used: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-unit left-minus-right visual-gain contrast for one direction."""
    try:
        traj_left, traj_right = TRAJECTORIES[direction]
    except KeyError as exc:
        raise ValueError("direction must be 'outbound' or 'inbound'.") from exc

    ids_left, effect_left, segment_edges = _effect_segment_with_baseline(
        dataset,
        trajectory=traj_left,
        ridge_used=ridge_used,
        seg_idx=seg_idx,
        include_global_light=include_global_light,
        baseline_dataset=baseline_dataset,
        baseline_ridge_used=baseline_ridge_used,
    )
    ids_right, effect_right, _ = _effect_segment_with_baseline(
        dataset,
        trajectory=traj_right,
        ridge_used=ridge_used,
        seg_idx=seg_idx,
        include_global_light=include_global_light,
        baseline_dataset=baseline_dataset,
        baseline_ridge_used=baseline_ridge_used,
    )
    common_units, left_aligned, right_aligned = _align_by_unit_ids(
        ids_left,
        effect_left,
        ids_right,
        effect_right,
    )
    return common_units, left_aligned - right_aligned, segment_edges


def _segment_edges_for_direction(dataset, *, direction: str) -> np.ndarray:
    """Return validated segment edges for the requested direction."""
    trajectory = TRAJECTORIES[direction][0]
    segment_edges, gain_basis = _segment_metadata(dataset)
    if gain_basis not in {"segment_raised_cosine", "segment_scalar"}:
        raise ValueError(f"Unsupported gain_basis {gain_basis!r}.")

    first_ridge = float(np.asarray(dataset.coords["ridge"].values, dtype=float)[0])
    gamma = np.asarray(
        dataset[_segment_gain_var_name(dataset)]
        .sel(trajectory=trajectory, ridge=first_ridge)
        .values,
        dtype=float,
    )
    n_segments = segment_edges.size - 1
    expected_basis = n_segments
    if gamma.shape[0] != expected_basis:
        raise ValueError(
            f"Dataset layout for trajectory {trajectory!r} is not compatible with "
            f"segment-wise visual-gain analysis: expected {expected_basis} basis rows, "
            f"found {gamma.shape[0]}."
        )
    return segment_edges


def _swap_consistency_by_segment(
    dataset1,
    dataset2,
    *,
    direction: str,
    ridge_used1: float,
    ridge_used2: float,
    include_global_light: bool,
    n_boot: int,
    ci: float,
    seed: int,
    baseline_dataset=None,
    baseline_ridge_used: float | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute per-segment swap-consistency results for one direction."""
    segment_edges = _segment_edges_for_direction(dataset1, direction=direction)
    segment_edges_other = _segment_edges_for_direction(dataset2, direction=direction)
    if not np.allclose(segment_edges, segment_edges_other):
        raise ValueError(
            "Compared datasets use different segment_edges and cannot be aligned."
        )
    if baseline_dataset is not None:
        baseline_edges = _segment_edges_for_direction(baseline_dataset, direction=direction)
        if not np.allclose(segment_edges, baseline_edges):
            raise ValueError(
                "Baseline dataset uses different segment_edges and cannot be aligned."
            )

    results: list[dict[str, float | int | str]] = []
    for seg_idx in range(segment_edges.size - 1):
        ids1, contrast1, _ = _left_right_contrast(
            dataset1,
            direction=direction,
            ridge_used=ridge_used1,
            seg_idx=seg_idx,
            include_global_light=include_global_light,
            baseline_dataset=baseline_dataset,
            baseline_ridge_used=baseline_ridge_used,
        )
        ids2, contrast2, _ = _left_right_contrast(
            dataset2,
            direction=direction,
            ridge_used=ridge_used2,
            seg_idx=seg_idx,
            include_global_light=include_global_light,
            baseline_dataset=baseline_dataset,
            baseline_ridge_used=baseline_ridge_used,
        )
        common_units, aligned1, aligned2 = _align_by_unit_ids(ids1, contrast1, ids2, contrast2)
        valid = np.isfinite(aligned1) & np.isfinite(aligned2)
        aligned1 = aligned1[valid]
        aligned2 = aligned2[valid]
        n_neurons = int(aligned1.size)

        if n_neurons < 3:
            r_swap = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            r_swap, ci_low, ci_high = _bootstrap_r(
                aligned1,
                -aligned2,
                n_boot=n_boot,
                ci=ci,
                seed=seed + (1000 * seg_idx),
            )

        results.append(
            {
                "direction": direction,
                "traj_left": TRAJECTORIES[direction][0],
                "traj_right": TRAJECTORIES[direction][1],
                "segment_index": int(seg_idx),
                "segment_start": float(segment_edges[seg_idx]),
                "segment_end": float(segment_edges[seg_idx + 1]),
                "segment_center": float(
                    0.5 * (segment_edges[seg_idx] + segment_edges[seg_idx + 1])
                ),
                "r_swap": float(r_swap),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_neurons": n_neurons,
                "n_common_units": int(common_units.size),
            }
        )
    return results


def _plot_direction_panel(ax, results: pd.DataFrame, title: str) -> None:
    """Draw one segment-wise r_swap panel."""
    x = results["segment_center"].to_numpy(dtype=float)
    labels = [
        f"{start:.2f}-{end:.2f}"
        for start, end in zip(
            results["segment_start"].to_numpy(dtype=float),
            results["segment_end"].to_numpy(dtype=float),
            strict=True,
        )
    ]
    r_swap = results["r_swap"].to_numpy(dtype=float)
    ci_low = results["ci_low"].to_numpy(dtype=float)
    ci_high = results["ci_high"].to_numpy(dtype=float)
    finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)

    ax.vlines(x[finite_ci], ci_low[finite_ci], ci_high[finite_ci], color="0.25", linewidth=1.5)
    ax.scatter(x, r_swap, s=28, color="black", zorder=3)
    ax.axhline(0.0, linewidth=1.0, color="0.35", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel(r"$r_{\mathrm{swap}} = \mathrm{corr}(d_1,\,-d_2)$")
    ax.set_xlabel("Segment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.2)


def plot_rswap_by_segment(
    summary_table: pd.DataFrame,
    *,
    figure_title: str,
):
    """Return the two-panel outbound/inbound swap-consistency figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    outbound = summary_table.loc[summary_table["direction"] == "outbound"].copy()
    inbound = summary_table.loc[summary_table["direction"] == "inbound"].copy()
    _plot_direction_panel(axes[0], outbound, "Outbound")
    _plot_direction_panel(axes[1], inbound, "Inbound")
    fig.suptitle(figure_title)
    fig.tight_layout()
    return fig


def _output_stem(args: argparse.Namespace, *, include_global_light: bool) -> str:
    """Build the shared output filename stem for this comparison."""
    mode_tag = MODE_FULL_LIGHT if include_global_light else MODE_INTERACTION_ONLY
    baseline_tag = (
        ""
        if args.baseline_light_epoch is None
        else f"_minus_{args.baseline_light_epoch}"
    )
    return (
        f"{args.region}_{args.model_family}_{args.light_epoch1}_vs_{args.light_epoch2}"
        f"_ref_{args.dark_epoch}{baseline_tag}_{mode_tag}_rswap"
    )


def _figure_title(args: argparse.Namespace, *, include_global_light: bool) -> str:
    """Return the figure title for one comparison."""
    stat_label = (
        "full visual gain field"
        if include_global_light
        else "segment interaction term"
    )
    title = (
        "Swap-consistency across segments\n"
        f"{args.region.upper()} | {args.light_epoch1} vs {args.light_epoch2} | "
        f"dark ref {args.dark_epoch} | {args.model_family} | {stat_label}"
    )
    if args.baseline_light_epoch is not None:
        title += f" | baseline-subtracted: {args.baseline_light_epoch}"
    return title


def main() -> None:
    """Run the modernized visual-gain correlation workflow."""
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
        else analysis_path / "task_progression_visual_gain_correlation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    include_global_light = not args.exclude_global_light_offset
    dataset1_path = _dataset_path(
        input_dir,
        region=args.region,
        light_epoch=args.light_epoch1,
        dark_epoch=args.dark_epoch,
        model_family=args.model_family,
    )
    dataset2_path = _dataset_path(
        input_dir,
        region=args.region,
        light_epoch=args.light_epoch2,
        dark_epoch=args.dark_epoch,
        model_family=args.model_family,
    )
    baseline_path = None
    if args.baseline_light_epoch is not None:
        baseline_path = _dataset_path(
            input_dir,
            region=args.region,
            light_epoch=args.baseline_light_epoch,
            dark_epoch=args.dark_epoch,
            model_family=args.model_family,
        )

    dataset1 = _load_fit_dataset(
        dataset1_path,
        expected_region=args.region,
        expected_light_epoch=args.light_epoch1,
        expected_dark_epoch=args.dark_epoch,
        expected_model_family=args.model_family,
    )
    dataset2 = _load_fit_dataset(
        dataset2_path,
        expected_region=args.region,
        expected_light_epoch=args.light_epoch2,
        expected_dark_epoch=args.dark_epoch,
        expected_model_family=args.model_family,
    )
    baseline_dataset = None
    if baseline_path is not None:
        baseline_dataset = _load_fit_dataset(
            baseline_path,
            expected_region=args.region,
            expected_light_epoch=args.baseline_light_epoch,
            expected_dark_epoch=args.dark_epoch,
            expected_model_family=args.model_family,
        )

    ridge_used1 = _select_ridge(dataset1, args.ridge)
    ridge_used2 = _select_ridge(dataset2, args.ridge)
    baseline_ridge_used = (
        None if baseline_dataset is None else _select_ridge(baseline_dataset, args.ridge)
    )

    results: list[dict[str, float | int | str | bool | None]] = []
    for direction, seed_offset in (("outbound", 0), ("inbound", 1)):
        direction_results = _swap_consistency_by_segment(
            dataset1,
            dataset2,
            direction=direction,
            ridge_used1=ridge_used1,
            ridge_used2=ridge_used2,
            include_global_light=include_global_light,
            n_boot=args.n_boot,
            ci=args.ci,
            seed=args.seed + seed_offset,
            baseline_dataset=baseline_dataset,
            baseline_ridge_used=baseline_ridge_used,
        )
        for row in direction_results:
            row.update(
                {
                    "animal_name": args.animal_name,
                    "date": args.date,
                    "region": args.region,
                    "model_family": args.model_family,
                    "light_epoch1": args.light_epoch1,
                    "light_epoch2": args.light_epoch2,
                    "dark_epoch": args.dark_epoch,
                    "baseline_light_epoch": args.baseline_light_epoch,
                    "include_global_light_offset": bool(include_global_light),
                    "mode": (
                        MODE_FULL_LIGHT
                        if include_global_light
                        else MODE_INTERACTION_ONLY
                    ),
                    "ridge_requested": float(args.ridge),
                    "ridge_used_light_epoch1": float(ridge_used1),
                    "ridge_used_light_epoch2": float(ridge_used2),
                    "ridge_used_baseline": (
                        np.nan
                        if baseline_ridge_used is None
                        else float(baseline_ridge_used)
                    ),
                    "n_boot": int(args.n_boot),
                    "ci_percent": float(args.ci),
                }
            )
            results.append(row)

    summary_table = pd.DataFrame(results)
    summary_table = summary_table[
        [
            "animal_name",
            "date",
            "region",
            "model_family",
            "light_epoch1",
            "light_epoch2",
            "dark_epoch",
            "baseline_light_epoch",
            "mode",
            "include_global_light_offset",
            "ridge_requested",
            "ridge_used_light_epoch1",
            "ridge_used_light_epoch2",
            "ridge_used_baseline",
            "direction",
            "traj_left",
            "traj_right",
            "segment_index",
            "segment_start",
            "segment_end",
            "segment_center",
            "r_swap",
            "ci_low",
            "ci_high",
            "n_neurons",
            "n_common_units",
            "n_boot",
            "ci_percent",
        ]
    ].sort_values(["direction", "segment_index"], ignore_index=True)

    stem = _output_stem(args, include_global_light=include_global_light)
    parquet_path = output_dir / f"{stem}.parquet"
    figure_path = output_dir / f"{stem}.png"
    summary_table.to_parquet(parquet_path, index=False)

    fig = plot_rswap_by_segment(
        summary_table,
        figure_title=_figure_title(args, include_global_light=include_global_light),
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.swap_gain_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "region": args.region,
            "model_family": args.model_family,
            "light_epoch1": args.light_epoch1,
            "light_epoch2": args.light_epoch2,
            "dark_epoch": args.dark_epoch,
            "baseline_light_epoch": args.baseline_light_epoch,
            "data_root": args.data_root,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "ridge_requested": args.ridge,
            "n_boot": args.n_boot,
            "ci": args.ci,
            "seed": args.seed,
            "include_global_light_offset": include_global_light,
        },
        outputs={
            "source_datasets": {
                "light_epoch1": dataset1_path,
                "light_epoch2": dataset2_path,
                "baseline_light_epoch": baseline_path,
            },
            "ridge_selection": {
                "requested": float(args.ridge),
                "light_epoch1": float(ridge_used1),
                "light_epoch2": float(ridge_used2),
                "baseline_light_epoch": baseline_ridge_used,
            },
            "saved_parquet": parquet_path,
            "saved_figure": figure_path,
        },
    )

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)

    dataset1.close()
    dataset2.close()
    if baseline_dataset is not None:
        baseline_dataset.close()

    print(f"Saved summary table to {parquet_path}")
    print(f"Saved figure to {figure_path}")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
