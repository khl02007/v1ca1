"""Generate standalone Figure 3 Panel B prediction examples."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    make_figure_3_epoch_ids,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    PANEL_C_EPOCH_ORDER,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    DEFAULT_RIPPLE_WINDOW_S,
    FIGURE_FORMATS,
    MODEL_COLOR,
    SOURCE_PREDICTOR_MODE_CHOICES,
    SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
    build_output_path,
    get_ripple_glm_path,
    parse_dataset_id,
)
from v1ca1.paper_figures.style import apply_paper_style, figure_size, save_figure


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "figure_3_prediction_examples"
DEFAULT_OUTPUT_FORMAT = "pdf"
DEFAULT_FIGURE_WIDTH_MM = 165.0
DEFAULT_FIGURE_HEIGHT_MM = 150.0
DEFAULT_TOP_N_UNITS = 30
DEFAULT_N_COLUMNS = 5


def get_default_prediction_datasets() -> list[DatasetId]:
    """Return the pooled Figure 3 Panel B GLM epochs."""
    datasets = []
    for animal_name, date, _dark_epoch in get_processed_datasets():
        epoch_ids = make_figure_3_epoch_ids(animal_name=animal_name, date=date)
        datasets.extend(
            normalize_dataset_id(epoch_ids[epoch_type])
            for epoch_type in PANEL_C_EPOCH_ORDER
            if epoch_type in epoch_ids
        )
    return datasets


def _prediction_matrix(data_array: Any, *, variable_name: str, path: Path) -> np.ndarray:
    """Return one prediction variable as samples by units."""
    if "unit" not in data_array.dims:
        raise ValueError(f"{variable_name} in {path} lacks a unit dimension.")
    values = np.asarray(data_array.values, dtype=float)
    if values.ndim < 2:
        raise ValueError(f"{variable_name} in {path} must include sample and unit dimensions.")
    unit_axis = int(data_array.get_axis_num("unit"))
    values = np.moveaxis(values, unit_axis, -1)
    return values.reshape(-1, values.shape[-1])


def load_dataset_prediction_examples(
    data_root: Path,
    *,
    dataset: DatasetId,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    source_predictor_mode: str = SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
) -> list[dict[str, Any]]:
    """Load all finite held-out prediction examples for one session."""
    import xarray as xr

    animal_name, date, epoch = normalize_dataset_id(dataset)
    path = get_ripple_glm_path(
        data_root,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        source_predictor_mode=source_predictor_mode,
    )
    if not path.exists():
        raise FileNotFoundError(f"Ripple-GLM NetCDF not found: {path}")

    dataset_obj = xr.load_dataset(path)
    try:
        required_variables = (
            "ripple_devexp_mean",
            "ripple_devexp_p_value",
            "ripple_observed_count_oof",
            "ripple_predicted_count_oof",
        )
        missing_variables = [
            variable for variable in required_variables if variable not in dataset_obj
        ]
        if missing_variables:
            raise ValueError(
                f"Ripple-GLM output {path} is missing variables {missing_variables!r}."
            )
        if "unit" not in dataset_obj.coords:
            raise ValueError(f"Ripple-GLM output {path} lacks unit coordinates.")

        unit_ids = np.asarray(dataset_obj.coords["unit"].values)
        devexp = np.asarray(dataset_obj["ripple_devexp_mean"].values, dtype=float).reshape(-1)
        p_values = np.asarray(dataset_obj["ripple_devexp_p_value"].values, dtype=float).reshape(-1)
        observed = _prediction_matrix(
            dataset_obj["ripple_observed_count_oof"],
            variable_name="ripple_observed_count_oof",
            path=path,
        )
        predicted = _prediction_matrix(
            dataset_obj["ripple_predicted_count_oof"],
            variable_name="ripple_predicted_count_oof",
            path=path,
        )
        if not (
            unit_ids.shape[0]
            == devexp.shape[0]
            == p_values.shape[0]
            == observed.shape[1]
            == predicted.shape[1]
        ):
            raise ValueError(f"Ripple-GLM output {path} has inconsistent unit dimensions.")

        selected_indices = np.flatnonzero(np.isfinite(devexp))
        if selected_indices.size == 0:
            raise ValueError(f"Ripple-GLM output has no finite deviance values: {path}")
        examples = []
        for unit_index in selected_indices:
            examples.append(
                {
                    "rank": 0,
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": epoch,
                    "unit_id": unit_ids[unit_index],
                    "observed": observed[:, unit_index],
                    "predicted": predicted[:, unit_index],
                    "ripple_devexp_mean": float(devexp[unit_index]),
                    "ripple_devexp_p_value": float(p_values[unit_index]),
                    "source_path": str(path),
                }
            )
    finally:
        dataset_obj.close()

    return examples


def _rank_prediction_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    top_n_units: int,
    rank_offset: int = 0,
) -> list[dict[str, Any]]:
    """Return examples globally ranked by deviance explained."""
    finite_examples = [
        dict(example)
        for example in examples
        if np.isfinite(float(example["ripple_devexp_mean"]))
    ]
    finite_examples.sort(
        key=lambda example: (
            float(example["ripple_devexp_mean"]),
            str(example["animal_name"]),
            str(example["date"]),
            str(example["epoch"]),
            int(example["unit_id"]),
        ),
        reverse=True,
    )
    start_index = max(0, int(rank_offset))
    stop_index = start_index + max(0, int(top_n_units))
    selected_examples = finite_examples[start_index:stop_index]
    for rank, example in enumerate(selected_examples, start=start_index + 1):
        example["rank"] = rank
    return selected_examples


def load_top_prediction_examples(
    data_root: Path,
    *,
    datasets: Sequence[DatasetId] | None = None,
    top_n_units: int = DEFAULT_TOP_N_UNITS,
    rank_offset: int = 0,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    source_predictor_mode: str = SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
) -> list[dict[str, Any]]:
    """Load top held-out prediction examples globally ranked across sessions."""
    selected_datasets = get_default_prediction_datasets() if datasets is None else list(datasets)
    all_examples: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    for dataset in selected_datasets:
        try:
            all_examples.extend(
                load_dataset_prediction_examples(
                    data_root,
                    dataset=dataset,
                    ripple_window_s=ripple_window_s,
                    ripple_window_offset_s=ripple_window_offset_s,
                    ripple_selection=ripple_selection,
                    ridge_strength=ridge_strength,
                    source_predictor_mode=source_predictor_mode,
                )
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            animal_name, date, epoch = normalize_dataset_id(dataset)
            missing_artifacts.append(f"{animal_name} {date} {epoch}: {exc}")

    examples = _rank_prediction_examples(
        all_examples,
        top_n_units=top_n_units,
        rank_offset=rank_offset,
    )
    if not examples and missing_artifacts:
        raise FileNotFoundError(
            "Could not load prediction examples from any dataset:\n"
            + "\n".join(missing_artifacts)
        )
    for missing_artifact in missing_artifacts:
        print(f"Skipping prediction examples for {missing_artifact}")
    return examples


def _format_p_value(value: float) -> str:
    """Return a compact p-value string for small plot annotations."""
    if not np.isfinite(value):
        return "nan"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _plot_prediction_example_axis(ax: Any, example: Mapping[str, Any]) -> None:
    """Plot one observed-versus-predicted unit example."""
    observed = np.asarray(example["observed"], dtype=float)
    predicted = np.asarray(example["predicted"], dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if np.any(valid):
        ax.scatter(
            observed[valid],
            predicted[valid],
            s=6.0,
            color=MODEL_COLOR,
            alpha=0.42,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        max_value = float(
            max(
                1.0,
                np.nanmax(observed[valid]),
                np.nanmax(predicted[valid]),
            )
        )
        ax.plot(
            [0.0, max_value],
            [0.0, max_value],
            color="0.20",
            linestyle="--",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xlim(0.0, max_value)
        ax.set_ylim(0.0, max_value)
    else:
        ax.text(0.5, 0.5, "No finite\nsamples", ha="center", va="center", fontsize=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        (
            f"#{int(example['rank'])} {example['animal_name']} "
            f"{example['epoch']} unit {example['unit_id']}"
        ),
        fontsize=5.6,
        pad=1.4,
    )
    ax.text(
        0.05,
        0.95,
        "devexp="
        f"{float(example['ripple_devexp_mean']):.2f}\n"
        f"p={_format_p_value(float(example['ripple_devexp_p_value']))}",
        ha="left",
        va="top",
        fontsize=4.5,
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=4.8, length=1.5, pad=1)


def plot_prediction_example_grid(
    axes: Sequence[Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    n_columns: int = DEFAULT_N_COLUMNS,
) -> None:
    """Plot top prediction examples into a flat sequence of axes."""
    for axis_index, ax in enumerate(axes):
        if axis_index >= len(examples):
            ax.axis("off")
            continue
        _plot_prediction_example_axis(ax, examples[axis_index])
        row_index = axis_index // int(n_columns)
        column_index = axis_index % int(n_columns)
        if row_index == (len(axes) - 1) // int(n_columns):
            ax.set_xlabel("Actual count", fontsize=5.4, labelpad=1.0)
        else:
            ax.set_xticklabels([])
        if column_index == 0:
            ax.set_ylabel("Predicted count", fontsize=5.4, labelpad=1.0)
        else:
            ax.set_yticklabels([])


def make_prediction_examples_figure(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId] | None = None,
    top_n_units: int = DEFAULT_TOP_N_UNITS,
    rank_offset: int = 0,
    n_columns: int = DEFAULT_N_COLUMNS,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    ripple_selection: str = DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    source_predictor_mode: str = SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
    dpi: int = 300,
) -> Path:
    """Build and save a standalone grid of top prediction examples."""
    import matplotlib.pyplot as plt

    examples = load_top_prediction_examples(
        data_root,
        datasets=datasets,
        top_n_units=top_n_units,
        rank_offset=rank_offset,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
        ripple_selection=ripple_selection,
        ridge_strength=ridge_strength,
        source_predictor_mode=source_predictor_mode,
    )
    if not examples:
        raise ValueError("No prediction examples were selected.")

    n_columns = max(1, int(n_columns))
    n_rows = int(np.ceil(len(examples) / n_columns))
    apply_paper_style()
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = tuple(axes.reshape(-1))
    plot_prediction_example_grid(flat_axes, examples, n_columns=n_columns)
    dataset_count = len(get_default_prediction_datasets()) if datasets is None else len(datasets)
    rank_start = int(examples[0]["rank"])
    rank_stop = int(examples[-1]["rank"])
    fig.suptitle(
        (
            f"V1 ripple-GLM prediction examples #{rank_start}-{rank_stop} "
            f"across {dataset_count} session"
            f"{'' if dataset_count == 1 else 's'}"
        ),
        fontsize=8.0,
    )
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 3 prediction examples to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the prediction-example figure."""
    parser = argparse.ArgumentParser(
        description="Generate observed-vs-predicted examples for Figure 3 Panel B GLMs."
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
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename stem. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--format",
        choices=FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset_id,
        action="append",
        default=None,
        help=(
            "Data set to include in the prediction-example pool. "
            "May be repeated. Format: animal:date:epoch. "
            "Default: all processed Figure 3 datasets."
        ),
    )
    parser.add_argument(
        "--top-n-units",
        type=int,
        default=DEFAULT_TOP_N_UNITS,
        help=f"Number of highest-devexp target units to plot. Default: {DEFAULT_TOP_N_UNITS}",
    )
    parser.add_argument(
        "--rank-offset",
        type=int,
        default=0,
        help=(
            "Number of highest-ranked units to skip before plotting. "
            "Use 30 with --top-n-units 30 to plot ranks 31-60. Default: 0"
        ),
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=DEFAULT_N_COLUMNS,
        help=f"Number of subplot columns. Default: {DEFAULT_N_COLUMNS}",
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_S,
        help=f"Ripple-GLM window length in seconds. Default: {DEFAULT_RIPPLE_WINDOW_S}",
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        help=(
            "Ripple-GLM window offset in seconds. "
            f"Default: {DEFAULT_RIPPLE_WINDOW_OFFSET_S}"
        ),
    )
    parser.add_argument(
        "--ripple-selection",
        choices=("allripples", "deduped", "single"),
        default=DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
        help=(
            "Ripple-selection suffix for the GLM output. "
            f"Default: {DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION}"
        ),
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Ripple-GLM ridge strength. Default: {DEFAULT_RIDGE_STRENGTH:g}",
    )
    parser.add_argument(
        "--source-predictor-mode",
        choices=SOURCE_PREDICTOR_MODE_CHOICES,
        default=SOURCE_PREDICTOR_MODE_UNIT_VECTOR,
        help=f"CA1 predictor mode. Default: {SOURCE_PREDICTOR_MODE_UNIT_VECTOR}.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run standalone prediction-example generation."""
    args = parse_arguments(argv)
    output_path = build_output_path(args.output_dir, args.output_name, args.format)
    make_prediction_examples_figure(
        data_root=args.data_root,
        output_path=output_path,
        datasets=args.dataset,
        top_n_units=args.top_n_units,
        rank_offset=args.rank_offset,
        n_columns=args.n_columns,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection=args.ripple_selection,
        ridge_strength=args.ridge_strength,
        source_predictor_mode=args.source_predictor_mode,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
