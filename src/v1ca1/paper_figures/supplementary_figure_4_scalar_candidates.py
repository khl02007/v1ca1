from __future__ import annotations

"""Plot candidate segment-scalar examples for Supplementary Figure 4."""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures.datasets import (
    DEFAULT_DARK_EPOCH,
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    FIGURE_FORMATS,
    PANEL_H_HELDOUT_LIGHT_EPOCH,
    PANEL_H_SWAP_DELTA_VARIABLE,
    PANEL_H_TRAIN_LIGHT_EPOCH,
    get_dark_epoch,
    get_swap_glm_selected_comparison_path,
    parse_dataset_id,
)
from v1ca1.paper_figures.style import (
    MODEL_CLASS_COLORS,
    NEUTRAL_COLORS,
    apply_paper_style,
    figure_size,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "supplementary_figure_4_scalar_similarity_top40"
DEFAULT_OUTPUT_FORMAT = "svg"
SCALAR_MODEL_NAME = "task_segment_scalar"
DEFAULT_TOP_N = 40
DEFAULT_N_COLUMNS = 4
DEFAULT_FIGURE_WIDTH_MM = 180.0
PANEL_HEIGHT_MM = 33.0
MIN_SIMILARITY_POINTS = 3


def compute_segment_curve_similarity(
    observed_position: np.ndarray,
    observed_rate_hz: np.ndarray,
    model_position: np.ndarray,
    model_rate_hz: np.ndarray,
    segment_start: float,
    segment_end: float,
    *,
    min_points: int = MIN_SIMILARITY_POINTS,
) -> dict[str, float] | None:
    """Return empirical-vs-model curve similarity over one switched segment."""
    observed_position = np.asarray(observed_position, dtype=float).reshape(-1)
    observed_rate_hz = np.asarray(observed_rate_hz, dtype=float).reshape(-1)
    model_position = np.asarray(model_position, dtype=float).reshape(-1)
    model_rate_hz = np.asarray(model_rate_hz, dtype=float).reshape(-1)
    segment_start = float(segment_start)
    segment_end = float(segment_end)

    observed_mask = (
        np.isfinite(observed_position)
        & np.isfinite(observed_rate_hz)
        & (observed_position >= segment_start)
        & (observed_position <= segment_end)
    )
    model_mask = (
        np.isfinite(model_position)
        & np.isfinite(model_rate_hz)
        & (model_position >= segment_start)
        & (model_position <= segment_end)
    )
    if int(np.sum(observed_mask)) < int(min_points) or int(np.sum(model_mask)) < 2:
        return None

    observed_x = observed_position[observed_mask]
    observed_y = observed_rate_hz[observed_mask]
    model_x = model_position[model_mask]
    model_y = model_rate_hz[model_mask]
    order = np.argsort(model_x)
    predicted_y = np.interp(observed_x, model_x[order], model_y[order])
    finite_mask = np.isfinite(observed_y) & np.isfinite(predicted_y)
    if int(np.sum(finite_mask)) < int(min_points):
        return None

    observed_y = observed_y[finite_mask]
    predicted_y = predicted_y[finite_mask]
    if float(np.std(observed_y)) == 0.0 or float(np.std(predicted_y)) == 0.0:
        return None

    correlation = float(np.corrcoef(observed_y, predicted_y)[0, 1])
    if not np.isfinite(correlation):
        return None
    rmse_hz = float(np.sqrt(np.mean((observed_y - predicted_y) ** 2)))
    return {
        "similarity_r": correlation,
        "rmse_hz": rmse_hz,
        "n_similarity_points": float(np.sum(finite_mask)),
    }


def collect_scalar_similarity_candidates(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = PANEL_H_TRAIN_LIGHT_EPOCH,
    light_test_epoch: str = PANEL_H_HELDOUT_LIGHT_EPOCH,
    model_name: str = SCALAR_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Load and score all segment-scalar switched-segment candidates."""
    import xarray as xr

    candidates: list[dict[str, Any]] = []
    model_name = str(model_name)
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        source_path = get_swap_glm_selected_comparison_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        if not source_path.exists():
            continue
        with xr.open_dataset(source_path) as dataset_obj:
            available_models = [str(value) for value in dataset_obj.coords["model"].values]
            if model_name not in available_models:
                continue
            if PANEL_H_SWAP_DELTA_VARIABLE not in dataset_obj:
                continue
            tp_grid = np.asarray(dataset_obj.coords["tp_grid"].values, dtype=float)
            observed_position = np.asarray(
                dataset_obj.coords["tp_observed_bin"].values,
                dtype=float,
            )
            trajectories = [str(value) for value in dataset_obj.coords["trajectory"].values]
            units = np.asarray(dataset_obj.coords["unit"].values)
            delta = np.asarray(
                dataset_obj[PANEL_H_SWAP_DELTA_VARIABLE].sel(model=model_name).values,
                dtype=float,
            )
            for trajectory_index, trajectory in enumerate(trajectories):
                segment_start = float(
                    dataset_obj["swap_segment_start"].isel(
                        trajectory=trajectory_index
                    ).values
                )
                segment_end = float(
                    dataset_obj["swap_segment_end"].isel(
                        trajectory=trajectory_index
                    ).values
                )
                observed_rates = np.asarray(
                    dataset_obj["test_light_observed_rate_hz"]
                    .isel(trajectory=trajectory_index)
                    .values,
                    dtype=float,
                )
                model_rates = np.asarray(
                    dataset_obj["test_light_swapped_hz_grid"]
                    .sel(model=model_name)
                    .isel(trajectory=trajectory_index)
                    .values,
                    dtype=float,
                )
                for unit_index, unit_id in enumerate(units):
                    observed_rate = observed_rates[:, unit_index]
                    model_rate = model_rates[:, unit_index]
                    score = compute_segment_curve_similarity(
                        observed_position,
                        observed_rate,
                        tp_grid,
                        model_rate,
                        segment_start,
                        segment_end,
                    )
                    if score is None:
                        continue
                    candidates.append(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "region": region,
                            "dark_epoch": dataset_dark_epoch,
                            "light_train_epoch": light_train_epoch,
                            "light_test_epoch": light_test_epoch,
                            "model_name": model_name,
                            "trajectory": trajectory,
                            "unit_id": int(unit_id),
                            "segment_start": segment_start,
                            "segment_end": segment_end,
                            "tp_grid": tp_grid.copy(),
                            "observed_position": observed_position.copy(),
                            "observed_rate_hz": observed_rate.copy(),
                            "model_rate_hz": model_rate.copy(),
                            "delta_ll_bits_per_spike": float(
                                delta[trajectory_index, unit_index]
                            ),
                            "source_path": str(source_path),
                            **score,
                        }
                    )
    return candidates


def select_top_similarity_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    top_n: int = DEFAULT_TOP_N,
    unique_cells: bool = True,
) -> list[dict[str, Any]]:
    """Return the best segment-scalar candidates by curve correlation."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["similarity_r"]),
            -float(candidate["rmse_hz"]),
            float(candidate.get("delta_ll_bits_per_spike", np.nan)),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, int]] = set()
    for candidate in ordered:
        cell_key = (
            str(candidate["animal_name"]),
            str(candidate["date"]),
            int(candidate["unit_id"]),
        )
        if unique_cells and cell_key in seen_cells:
            continue
        selected.append(dict(candidate))
        seen_cells.add(cell_key)
        if len(selected) >= int(top_n):
            break
    return selected


def _trajectory_label(trajectory: str) -> str:
    """Return a compact trajectory label for subplot titles."""
    return str(trajectory).replace("_to_", " to ").replace("_", " ")


def _plot_candidate_axis(
    ax: Any,
    candidate: dict[str, Any],
    *,
    rank: int,
    show_legend: bool,
) -> None:
    """Plot one empirical and segment-scalar switched-segment candidate."""
    segment_start = float(candidate["segment_start"])
    segment_end = float(candidate["segment_end"])
    observed_position = np.asarray(candidate["observed_position"], dtype=float)
    observed_rate = np.asarray(candidate["observed_rate_hz"], dtype=float)
    model_position = np.asarray(candidate["tp_grid"], dtype=float)
    model_rate = np.asarray(candidate["model_rate_hz"], dtype=float)
    observed_mask = (
        np.isfinite(observed_position)
        & np.isfinite(observed_rate)
        & (observed_position >= segment_start)
        & (observed_position <= segment_end)
    )
    model_mask = (
        np.isfinite(model_position)
        & np.isfinite(model_rate)
        & (model_position >= segment_start)
        & (model_position <= segment_end)
    )

    ax.plot(
        observed_position[observed_mask],
        observed_rate[observed_mask],
        color=NEUTRAL_COLORS["empirical"],
        linewidth=0.85,
        label="Empirical",
    )
    ax.plot(
        model_position[model_mask],
        model_rate[model_mask],
        color=MODEL_CLASS_COLORS[SCALAR_MODEL_NAME],
        linewidth=0.85,
        label="Segment scalar",
    )
    ax.axvspan(segment_start, segment_end, color=MODEL_CLASS_COLORS[SCALAR_MODEL_NAME], alpha=0.06)
    ax.set_xlim(segment_start, segment_end)
    finite_rates = np.concatenate(
        (
            observed_rate[observed_mask & np.isfinite(observed_rate)],
            model_rate[model_mask & np.isfinite(model_rate)],
        )
    )
    y_max = 1.0 if finite_rates.size == 0 else max(1.0, float(np.nanmax(finite_rates)) * 1.08)
    ax.set_ylim(0.0, y_max)
    ax.set_title(
        (
            f"{rank}. {candidate['animal_name']} {candidate['date']} "
            f"cell {candidate['unit_id']}\n"
            f"{_trajectory_label(candidate['trajectory'])}, "
            f"r={float(candidate['similarity_r']):.2f}, "
            f"dLL={float(candidate['delta_ll_bits_per_spike']):.2f}"
        ),
        fontsize=5.1,
        pad=1.2,
    )
    ax.tick_params(labelsize=4.4, length=1.2, pad=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=4.5, handlelength=1.2, loc="upper right")


def plot_scalar_similarity_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    output_path: Path,
    dpi: int,
    n_columns: int = DEFAULT_N_COLUMNS,
) -> Path:
    """Save a grid of top empirical-vs-segment-scalar example candidates."""
    import matplotlib.pyplot as plt

    candidates = list(candidates)
    if not candidates:
        raise ValueError("No segment-scalar similarity candidates to plot.")
    n_columns = max(int(n_columns), 1)
    n_rows = int(math.ceil(len(candidates) / n_columns))

    apply_paper_style()
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, PANEL_HEIGHT_MM * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis_index, ax in enumerate(axes.reshape(-1)):
        if axis_index >= len(candidates):
            ax.axis("off")
            continue
        _plot_candidate_axis(
            ax,
            candidates[axis_index],
            rank=axis_index + 1,
            show_legend=axis_index == 0,
        )
        row_index = axis_index // n_columns
        column_index = axis_index % n_columns
        if row_index == n_rows - 1:
            ax.set_xlabel("Switched segment", fontsize=5.0, labelpad=0.8)
        if column_index == 0:
            ax.set_ylabel("FR (Hz)", fontsize=5.0, labelpad=0.8)

    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def make_scalar_similarity_candidate_figure(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str,
    light_test_epoch: str,
    top_n: int,
    n_columns: int,
    unique_cells: bool,
    dpi: int,
) -> Path:
    """Build and save the segment-scalar top-similarity candidate figure."""
    candidates = collect_scalar_similarity_candidates(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
    )
    selected = select_top_similarity_candidates(
        candidates,
        top_n=top_n,
        unique_cells=unique_cells,
    )
    output_path = plot_scalar_similarity_candidates(
        selected,
        output_path=output_path,
        dpi=dpi,
        n_columns=n_columns,
    )
    print(f"Saved scalar similarity candidates to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for scalar similarity candidate plotting."""
    parser = argparse.ArgumentParser(
        description="Plot top segment-scalar empirical similarity candidates."
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
        default="v1",
        help="Region to include. Default: v1.",
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
        "--light-train-epoch",
        default=PANEL_H_TRAIN_LIGHT_EPOCH,
        help=f"Light train epoch. Default: {PANEL_H_TRAIN_LIGHT_EPOCH}.",
    )
    parser.add_argument(
        "--light-test-epoch",
        default=PANEL_H_HELDOUT_LIGHT_EPOCH,
        help=f"Held-out light test epoch. Default: {PANEL_H_HELDOUT_LIGHT_EPOCH}.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of candidates to plot. Default: {DEFAULT_TOP_N}.",
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=DEFAULT_N_COLUMNS,
        help=f"Number of subplot columns. Default: {DEFAULT_N_COLUMNS}.",
    )
    parser.add_argument(
        "--allow-duplicate-units",
        action="store_true",
        help="Allow the same unit to appear on more than one trajectory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run scalar similarity candidate plotting."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = Path(args.output_dir) / f"{args.output_name}.{args.output_format}"
    make_scalar_similarity_candidate_figure(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        dark_epoch=args.dark_epoch,
        light_train_epoch=args.light_train_epoch,
        light_test_epoch=args.light_test_epoch,
        top_n=args.top_n,
        n_columns=args.n_columns,
        unique_cells=not bool(args.allow_duplicate_units),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
