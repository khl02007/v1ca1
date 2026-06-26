from __future__ import annotations

"""Generate Supplementary Figure 2 scalar controls and per-animal summaries."""

import argparse
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
from v1ca1.paper_figures.figure_1 import get_stability_table_path
from v1ca1.paper_figures.figure_4 import load_dark_movement_firing_rate_table
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_REGIONS,
    FIGURE_FORMATS,
    PANEL_TRAJECTORY_COLORS,
    PANEL_H_DELTA_TRAJECTORIES,
    PANEL_H_DELTA_X_LIMITS,
    PANEL_TRAJECTORY_LABELS,
    _filter_panel_h_heldout_delta,
    build_output_path,
    get_dark_epoch,
    get_dataset_analysis_path,
    get_dark_light_glm_selected_path,
    get_swap_glm_selected_comparison_path,
    load_panel_h_swap_delta_table,
    parse_dataset_id,
)
from v1ca1.paper_figures.figure_2 import (
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_NAME = "supplementary_figure_2"
SCALAR_MODEL_NAME = "task_segment_scalar"
FIGURE_2B_DELTA_BOX_WIDTH = 0.13
FIGURE_2B_DELTA_SUBPANEL_BOUNDS = (
    (0.03, 0.18, 0.22, 0.68),
    (0.27, 0.18, 0.22, 0.68),
    (0.51, 0.18, 0.22, 0.68),
    (0.75, 0.18, 0.22, 0.68),
)
MIXED_GLM_FULL_DELTA_AXIS_BOUNDS = (0.115, 0.14, 0.54, 0.70)
MIXED_GLM_FULL_BEST_AXIS_BOUNDS = (0.695, 0.32, 0.19, 0.32)
LETTER_PAPER_WIDTH_IN = 8.5
LETTER_HORIZONTAL_MARGIN_IN = 1.0
DEFAULT_FIGURE_WIDTH_MM = (
    LETTER_PAPER_WIDTH_IN - 2.0 * LETTER_HORIZONTAL_MARGIN_IN
) * 25.4
SCALAR_PANEL_HEIGHT_MM = 62.0
DEFAULT_ANIMAL_ROW_HEIGHT_MM = 35.0
MIXED_GLM_EMPIRICAL_PANEL_HEIGHT_MM = 40.0
PER_ANIMAL_GRID_HSPACE = 0.42
PANEL_TITLE_FONTSIZE = 8.0
SWAP_TUNING_CURVE_COMPARISON_RELATIVE_DIR = (
    Path("task_progression") / "swap_tuning_curve_comparison"
)
EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH = "02_r1"
EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH = "06_r3"
MULTIPLICATIVE_SEGMENT_LABEL = "Multiplicative segment"
MULTIPLICATIVE_SEGMENT_SHORT_LABEL = "MS"
ADDITIVE_SEGMENT_SHORT_LABEL = "AS"
ADDITIVE_LABEL = "Additive"
ADDITIVE_SHORT_LABEL = "A"
EMPIRICAL_PAIRWISE_MODEL_NAMES = {
    "V": "empirical_visual",
    MULTIPLICATIVE_SEGMENT_SHORT_LABEL: "empirical_segment_multiplicative_ratio",
    ADDITIVE_SEGMENT_SHORT_LABEL: "empirical_segment_additive_delta",
}
EMPIRICAL_PAIRWISE_DELTA_PAIRS = (
    (
        f"V-{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}",
        f"delta_V_minus_{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_bits_per_spike",
    ),
    (
        f"V-{ADDITIVE_SEGMENT_SHORT_LABEL}",
        f"delta_V_minus_{ADDITIVE_SEGMENT_SHORT_LABEL}_bits_per_spike",
    ),
)
EMPIRICAL_BEST_MODEL_COLORS = {
    "V": "#4D4D4D",
    MULTIPLICATIVE_SEGMENT_SHORT_LABEL: "#1A9850",
    ADDITIVE_SEGMENT_SHORT_LABEL: "#7570B3",
    "tie": "#BDBDBD",
}
GLM_SCALAR_MODEL_LABEL = MULTIPLICATIVE_SEGMENT_SHORT_LABEL
GLM_SCALAR_BEST_MODEL_COLORS = {
    "V": "#4D4D4D",
    GLM_SCALAR_MODEL_LABEL: "#1A9850",
    "tie": "#BDBDBD",
}
MIXED_GLM_TASK_LABEL = MULTIPLICATIVE_SEGMENT_SHORT_LABEL
MIXED_EMPIRICAL_SA_LABEL = ADDITIVE_SEGMENT_SHORT_LABEL
MIXED_EMPIRICAL_AD_LABEL = ADDITIVE_SHORT_LABEL
FULL_ADDITIVE_INDEPENDENT_LABEL = "Independent"
FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL = "Shared-scaffold"
FULL_ADDITIVE_ADDITIVE_LABEL = ADDITIVE_LABEL
FULL_ADDITIVE_BEST_MODEL_DISPLAY_LABELS = {
    "V": FULL_ADDITIVE_INDEPENDENT_LABEL,
    MIXED_GLM_TASK_LABEL: FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL,
    MIXED_EMPIRICAL_AD_LABEL: FULL_ADDITIVE_ADDITIVE_LABEL,
    "tie": "tie",
}
HYBRID_GLM_EMPIRICAL_LABEL = "H"
REVERSE_HYBRID_GLM_EMPIRICAL_LABEL = "H2"
MIXED_EMPIRICAL_SA_MODEL_NAME = "empirical_segment_additive_delta"
MIXED_EMPIRICAL_AD_MODEL_NAME = "empirical_pointwise_additive_delta"
MIXED_GLM_EMPIRICAL_BEST_MODEL_COLORS = {
    "V": "#4D4D4D",
    MIXED_GLM_TASK_LABEL: "#1A9850",
    MIXED_EMPIRICAL_SA_LABEL: "#7570B3",
    "tie": "#BDBDBD",
}
MIXED_GLM_FULL_ADDITIVE_BEST_MODEL_COLORS = {
    "V": "#4D4D4D",
    MIXED_GLM_TASK_LABEL: "#1A9850",
    MIXED_EMPIRICAL_AD_LABEL: "#D95F02",
    "tie": "#BDBDBD",
}
HYBRID_GLM_EMPIRICAL_BEST_MODEL_COLORS = {
    "V": "#4D4D4D",
    MIXED_GLM_TASK_LABEL: "#1A9850",
    HYBRID_GLM_EMPIRICAL_LABEL: "#D95F02",
    REVERSE_HYBRID_GLM_EMPIRICAL_LABEL: "#7570B3",
    "tie": "#BDBDBD",
}
EMPIRICAL_PAIRWISE_BOX_WIDTH = 0.16
EMPIRICAL_PAIRWISE_BOX_OFFSETS = (-0.27, -0.09, 0.09, 0.27)
MULTIPLIER_COMPARISON_SPECS = (
    (
        "log(G_emp) vs\nlog(G_glm_segment)",
        "log_empirical_ms_gain",
        "log_glm_segment_gain",
        "log(G_emp)",
        "log(G_glm_segment)",
        "Greens",
    ),
    (
        "log(G_emp) vs\nlog(G_glm_full)",
        "log_empirical_ms_gain",
        "log_glm_full_gain",
        "log(G_emp)",
        "log(G_glm_full)",
        "Oranges",
    ),
)
MULTIPLIER_EPSILON = 1e-10
SCALAR_BASELINE_SCORE_VARIABLE = "ll_bits_per_spike_cv_combined"
SCALAR_BASELINE_SCORE_COLUMN = "scalar_vs_baseline_bits_per_spike"
FULL_SEGMENT_LOG_GAIN_THRESHOLD = float(np.log(1.5))
FULL_SEGMENT_GAIN_MIN_TUNING_STABILITY_CORRELATION = 0.5
FULL_SEGMENT_GAIN_COLORS = ("#4D4D4D", "#1A9850", "#D95F02")
NESTED_DARK_ACTIVE_FR_THRESHOLD_HZ = 0.5
NESTED_TUNING_STABILITY_CORRELATION_THRESHOLD = 0.5
NESTED_SCALAR_BASELINE_SCORE_THRESHOLD = 0.0
NESTED_MODULATION_COLORS = {
    "dark_inactive": "#D9D9D9",
    "dark_active": "#4D4D4D",
    "unstable": "#BDBDBD",
    "stable": "#1A9850",
    "no_scalar_fit": "#F0F0F0",
    "not_modulated": "#91BFDB",
    "modulated": "#D95F02",
}
NESTED_MODULATION_BAR_AXIS_BOUNDS = (0.09, 0.30, 0.25, 0.52)
NESTED_MODULATION_LEGEND_AXIS_BOUNDS = (0.43, 0.18, 0.28, 0.68)
FULL_SEGMENT_GAIN_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_train_epoch",
    "light_train_epoch",
    "reliability_epoch",
    "trajectory",
    "segment_index_1based",
    "segment_basis",
    "unit",
    "full_segment_log_gain",
    "segment_specific_log_gain",
    "light_offset_log_gain",
    SCALAR_BASELINE_SCORE_COLUMN,
    "stability_correlation",
    "glm_source_path",
    "stability_source_path",
)
NESTED_MODULATION_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_train_epoch",
    "total_cell_count",
    "dark_inactive_count",
    "dark_active_count",
    "dark_active_unstable_count",
    "dark_active_stable_count",
    "dark_active_stable_no_scalar_fit_count",
    "dark_active_stable_missing_scalar_fit_count",
    "dark_active_stable_scalar_below_baseline_count",
    "dark_active_stable_unmodulated_count",
    "dark_active_stable_modulated_count",
    "dark_active_fr_threshold_hz",
    "tuning_stability_correlation_threshold",
    "scalar_baseline_score_threshold",
    "full_segment_log_gain_threshold",
    "dark_firing_rate_source_path",
    "stability_source_path",
)


def group_datasets_by_animal(
    datasets: Sequence[DatasetId],
) -> dict[str, list[DatasetId]]:
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


def plot_figure_2b_delta_ll_boxplots(
    ax: Any,
    swap_delta_table: Any,
    *,
    animal_names: Sequence[str] | None = None,
) -> None:
    """Plot Figure 2B delta LL distributions in animal subpanels."""
    trajectory_types = tuple(PANEL_H_DELTA_TRAJECTORIES)
    table = _filter_panel_h_heldout_delta(swap_delta_table)
    if animal_names is None:
        if table is None or "animal_name" not in table:
            selected_animals: tuple[str, ...] = ()
        else:
            selected_animals = tuple(dict.fromkeys(table["animal_name"].astype(str)))
    else:
        selected_animals = tuple(str(animal_name) for animal_name in animal_names)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    trajectory_positions = {
        trajectory_type: float(len(trajectory_types) - trajectory_index - 1)
        for trajectory_index, trajectory_type in enumerate(trajectory_types)
    }
    required_columns = {"animal_name", "trajectory", "delta_ll_bits_per_spike"}
    has_required_columns = table is not None and required_columns.issubset(table.columns)
    any_plotted = False

    for animal_index, animal_name in enumerate(selected_animals):
        if animal_index >= len(FIGURE_2B_DELTA_SUBPANEL_BOUNDS):
            break
        child_ax = ax.inset_axes(FIGURE_2B_DELTA_SUBPANEL_BOUNDS[animal_index])
        child_ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
        plotted_values = []
        plotted_positions = []
        plotted_colors = []
        plotted_positive_fractions = []
        if has_required_columns:
            animal_rows = table[
                table["animal_name"].astype(str) == animal_name
            ]
            for trajectory_type in trajectory_types:
                trajectory_rows = animal_rows[
                    animal_rows["trajectory"].astype(str) == str(trajectory_type)
                ]
                values = np.asarray(
                    trajectory_rows["delta_ll_bits_per_spike"],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                plotted_values.append(values)
                plotted_positions.append(trajectory_positions[trajectory_type])
                plotted_colors.append(PANEL_TRAJECTORY_COLORS[trajectory_type])
                plotted_positive_fractions.append(float(np.mean(values > 0.0)))

        if plotted_values:
            any_plotted = True
            boxplot = child_ax.boxplot(
                plotted_values,
                positions=plotted_positions,
                widths=FIGURE_2B_DELTA_BOX_WIDTH,
                orientation="horizontal",
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 0.75},
                whiskerprops={"color": "0.30", "linewidth": 0.55},
                capprops={"color": "0.30", "linewidth": 0.55},
                boxprops={"linewidth": 0.55},
            )
            for patch, color in zip(boxplot["boxes"], plotted_colors, strict=True):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.68)
            for position, color, fraction in zip(
                plotted_positions,
                plotted_colors,
                plotted_positive_fractions,
                strict=True,
            ):
                child_ax.text(
                    1.02,
                    position,
                    f"{fraction:.0%} >0",
                    ha="left",
                    va="center",
                    fontsize=3.6,
                    color=color,
                    clip_on=False,
                    transform=child_ax.get_yaxis_transform(),
                )
        else:
            child_ax.text(
                0.5,
                0.5,
                "No finite\nvalues",
                ha="center",
                va="center",
                fontsize=4.8,
                transform=child_ax.transAxes,
            )

        child_ax.set_xlim(*PANEL_H_DELTA_X_LIMITS)
        child_ax.set_ylim(-0.55, max(len(trajectory_types) - 1, 0) + 0.55)
        child_ax.set_yticks(list(trajectory_positions.values()))
        if animal_index == 0:
            child_ax.set_yticklabels(
                [
                    PANEL_TRAJECTORY_LABELS.get(trajectory_type, trajectory_type)
                    for trajectory_type in trajectory_types
                ]
            )
            child_ax.set_ylabel("Trajectory", fontsize=5.2)
        else:
            child_ax.set_yticklabels([])
            child_ax.tick_params(axis="y", length=0)
        child_ax.set_title(animal_name, fontsize=5.6, pad=1.2)
        child_ax.spines["top"].set_visible(False)
        child_ax.spines["right"].set_visible(False)
        child_ax.tick_params(labelsize=4.5, length=1.2, pad=0.8)

    if not any_plotted:
        ax.text(
            0.5,
            0.5,
            "No finite Figure 2B\nDelta LL values",
            ha="center",
            va="center",
            fontsize=5.0,
            transform=ax.transAxes,
        )

    ax.text(
        0.5,
        0.055,
        "\N{GREEK CAPITAL LETTER DELTA}LL (bits/spike)",
        ha="center",
        va="top",
        fontsize=5.2,
        transform=ax.transAxes,
    )


def get_figure_height_mm(n_animal_rows: int) -> float:
    """Return the Supplementary Figure 2 height for the requested row count."""
    if int(n_animal_rows) <= 0:
        return DEFAULT_ANIMAL_ROW_HEIGHT_MM
    return (
        SCALAR_PANEL_HEIGHT_MM
        + MIXED_GLM_EMPIRICAL_PANEL_HEIGHT_MM
    )


def get_swap_tuning_curve_comparison_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Path:
    """Return one empirical swap tuning-curve comparison parquet path."""
    return (
        get_dataset_analysis_path(data_root, animal_name, date)
        / SWAP_TUNING_CURVE_COMPARISON_RELATIVE_DIR
        / (
            f"{region}_{dark_epoch}_traindark_"
            f"{light_train_epoch}_trainlight_"
            f"{light_test_epoch}_testlight_swap_tuning_curve_comparison.parquet"
        )
    )


def get_swap_tuning_curve_comparison_dataset_path(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Path:
    """Return one empirical swap tuning-curve comparison NetCDF path."""
    return get_swap_tuning_curve_comparison_path(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
    ).with_suffix(".nc")


def _flatten_pivot_columns(table: Any) -> Any:
    """Return a table with flattened metric/model pivot columns."""
    table = table.copy()
    table.columns = [
        f"{metric}__{model_name}" if model_name else str(metric)
        for metric, model_name in table.columns
    ]
    return table


def _segment_overlap_weights(
    bin_edges: np.ndarray,
    segment_edges: np.ndarray,
    segment_index: int,
) -> np.ndarray:
    """Return per-bin fractional overlap with one TP segment."""
    observed_edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    if observed_edges.size < 2:
        raise ValueError("bin_edges must contain at least two edges.")
    left = observed_edges[:-1]
    right = observed_edges[1:]
    width = right - left
    if np.any(width <= 0.0):
        raise ValueError("bin_edges must be strictly increasing.")
    edges = np.asarray(segment_edges, dtype=float).reshape(-1)
    start = float(edges[int(segment_index)])
    end = float(edges[int(segment_index) + 1])
    overlap = np.maximum(0.0, np.minimum(right, end) - np.maximum(left, start))
    return np.clip(overlap / width, 0.0, 1.0)


def _interpolate_rate_matrix(
    source_x: np.ndarray,
    rates_hz: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Interpolate a `(tp_bin, unit)` rate matrix onto target TP bins."""
    source = np.asarray(source_x, dtype=float).reshape(-1)
    target = np.asarray(target_x, dtype=float).reshape(-1)
    matrix = np.asarray(rates_hz, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != source.size:
        raise ValueError(
            "rates_hz must be shaped `(tp_bin, unit)` and match source_x. "
            f"Got source_x={source.size}, rates_hz={matrix.shape}."
        )
    output = np.full((target.size, matrix.shape[1]), np.nan, dtype=float)
    for unit_index in range(matrix.shape[1]):
        unit_rates = matrix[:, unit_index]
        finite = np.isfinite(source) & np.isfinite(unit_rates)
        if not np.any(finite):
            continue
        output[:, unit_index] = np.interp(
            target,
            source[finite],
            unit_rates[finite],
            left=unit_rates[finite][0],
            right=unit_rates[finite][-1],
        )
    return output


def _aggregate_poisson_ll_bits_per_spike(
    *,
    spike_counts: np.ndarray,
    occupancy_s: np.ndarray,
    rates_hz: np.ndarray,
    epsilon: float = MULTIPLIER_EPSILON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score aggregate TP-bin counts; model-independent constants are omitted."""
    counts = np.asarray(spike_counts, dtype=float)
    occupancy = np.asarray(occupancy_s, dtype=float).reshape(-1)
    rates = np.asarray(rates_hz, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    if rates.ndim == 1:
        rates = rates[:, None]
    if counts.shape != rates.shape or counts.shape[0] != occupancy.size:
        raise ValueError(
            "spike_counts, occupancy_s, and rates_hz must describe the same "
            "TP bins and units. "
            f"Got counts={counts.shape}, occupancy={occupancy.shape}, "
            f"rates={rates.shape}."
        )
    expected_counts = np.maximum(
        np.clip(rates, float(epsilon), None) * occupancy[:, None],
        float(epsilon),
    )
    ll_sum = np.sum(counts * np.log(expected_counts) - expected_counts, axis=0)
    spike_sum = np.sum(counts, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_bits_per_spike = np.where(
            spike_sum > 0.0,
            ll_sum / (np.log(2.0) * spike_sum),
            np.nan,
        )
    return ll_sum, ll_bits_per_spike, spike_sum


def load_empirical_pairwise_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Any:
    """Load V/MS/AS empirical swap-model pairwise delta LL values."""
    import pandas as pd

    tables = []
    missing_paths: list[Path] = []
    model_names = tuple(EMPIRICAL_PAIRWISE_MODEL_NAMES.values())
    labels = np.asarray(list(EMPIRICAL_PAIRWISE_MODEL_NAMES), dtype=object)
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        path = get_swap_tuning_curve_comparison_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        if not path.exists():
            missing_paths.append(path)
            continue

        table = pd.read_parquet(path)
        required_columns = {
            "animal_name",
            "date",
            "region",
            "dark_train_epoch",
            "light_train_epoch",
            "light_test_epoch",
            "trajectory",
            "unit",
            "model",
            "ll_sum",
            "ll_bits_per_s",
            "ll_bits_per_spike",
        }
        missing_columns = sorted(required_columns.difference(table.columns))
        if missing_columns:
            raise KeyError(
                f"{path} is missing required column(s): "
                + ", ".join(missing_columns)
            )
        table = table[table["model"].astype(str).isin(model_names)].copy()
        available_models = set(table["model"].astype(str))
        missing_models = [
            model_name
            for model_name in model_names
            if model_name not in available_models
        ]
        if missing_models:
            raise KeyError(
                f"{path} is missing empirical model(s): "
                + ", ".join(missing_models)
            )
        table = table[
            np.isfinite(table["ll_sum"].to_numpy(dtype=float))
            & np.isfinite(table["ll_bits_per_s"].to_numpy(dtype=float))
            & np.isfinite(table["ll_bits_per_spike"].to_numpy(dtype=float))
        ].copy()
        if table.empty:
            continue

        index_columns = [
            "animal_name",
            "date",
            "region",
            "dark_train_epoch",
            "light_train_epoch",
            "light_test_epoch",
            "trajectory",
            "unit",
        ]
        wide = table.pivot_table(
            index=index_columns,
            columns="model",
            values=["ll_sum", "ll_bits_per_s", "ll_bits_per_spike"],
            aggfunc="first",
        )
        wide = _flatten_pivot_columns(wide).reset_index()
        ll_columns = [
            f"ll_sum__{EMPIRICAL_PAIRWISE_MODEL_NAMES[label]}" for label in labels
        ]
        ll_values = wide[ll_columns].to_numpy(dtype=float)
        max_ll = np.nanmax(ll_values, axis=1)
        tie_mask = (
            np.isclose(ll_values, max_ll[:, None], rtol=0.0, atol=1e-12).sum(axis=1)
            > 1
        )
        winner_index = np.nanargmax(ll_values, axis=1)
        wide["winner"] = labels[winner_index]
        wide.loc[tie_mask, "winner"] = "tie"
        for label_a, label_b in (
            ("V", MULTIPLICATIVE_SEGMENT_SHORT_LABEL),
            ("V", ADDITIVE_SEGMENT_SHORT_LABEL),
            (
                MULTIPLICATIVE_SEGMENT_SHORT_LABEL,
                ADDITIVE_SEGMENT_SHORT_LABEL,
            ),
        ):
            model_a = EMPIRICAL_PAIRWISE_MODEL_NAMES[label_a]
            model_b = EMPIRICAL_PAIRWISE_MODEL_NAMES[label_b]
            wide[f"delta_{label_a}_minus_{label_b}_bits_per_s"] = (
                wide[f"ll_bits_per_s__{model_a}"]
                - wide[f"ll_bits_per_s__{model_b}"]
            )
            wide[f"delta_{label_a}_minus_{label_b}_bits_per_spike"] = (
                wide[f"ll_bits_per_spike__{model_a}"]
                - wide[f"ll_bits_per_spike__{model_b}"]
            )
            wide[f"delta_{label_a}_minus_{label_b}_ll_sum"] = (
                wide[f"ll_sum__{model_a}"] - wide[f"ll_sum__{model_b}"]
            )
        wide["source_path"] = str(path)
        wide["winner_model_name"] = [
            EMPIRICAL_PAIRWISE_MODEL_NAMES.get(str(label), "tie")
            for label in wide["winner"]
        ]
        tables.append(wide)

    if tables:
        return pd.concat(tables, axis=0, ignore_index=True)

    columns = [
        "animal_name",
        "date",
        "region",
        "dark_train_epoch",
        "light_train_epoch",
        "light_test_epoch",
        "trajectory",
        "unit",
        "winner",
        "winner_model_name",
        f"delta_V_minus_{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_bits_per_s",
        f"delta_V_minus_{ADDITIVE_SEGMENT_SHORT_LABEL}_bits_per_s",
        (
            "delta_"
            f"{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_minus_"
            f"{ADDITIVE_SEGMENT_SHORT_LABEL}_bits_per_s"
        ),
        f"delta_V_minus_{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_bits_per_spike",
        f"delta_V_minus_{ADDITIVE_SEGMENT_SHORT_LABEL}_bits_per_spike",
        (
            "delta_"
            f"{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_minus_"
            f"{ADDITIVE_SEGMENT_SHORT_LABEL}_bits_per_spike"
        ),
        "source_path",
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No empirical swap tuning-curve comparison artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    return pd.DataFrame(columns=columns)


def load_glm_scalar_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Any:
    """Load GLM visual-minus-MS swap delta LL values."""
    table = load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        light_epoch_pairs=((light_train_epoch, light_test_epoch),),
        model_name=SCALAR_MODEL_NAME,
    )
    if table is None or len(table) == 0:
        return table
    table = table.copy()
    delta_model_minus_visual = np.asarray(
        table["delta_ll_bits_per_spike"],
        dtype=float,
    )
    table["delta_V_minus_scalar_bits_per_spike"] = -delta_model_minus_visual
    table["winner"] = np.where(
        table["delta_V_minus_scalar_bits_per_spike"] > 0.0,
        "V",
        np.where(
            table["delta_V_minus_scalar_bits_per_spike"] < 0.0,
            GLM_SCALAR_MODEL_LABEL,
            "tie",
        ),
    )
    return table


def load_mixed_glm_empirical_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
    empirical_model_name: str = MIXED_EMPIRICAL_SA_MODEL_NAME,
    empirical_label: str = MIXED_EMPIRICAL_SA_LABEL,
) -> Any:
    """Load matched GLM visual/MS and one empirical additive swap score."""
    import pandas as pd
    import xarray as xr

    rows = []
    missing_paths: list[Path] = []
    glm_models = {
        "V": "visual",
        MIXED_GLM_TASK_LABEL: SCALAR_MODEL_NAME,
    }
    raw_ll_sum_var = "test_light_swapped_segment_swapped_raw_ll_sum"
    raw_ll_bits_var = "test_light_swapped_segment_swapped_raw_ll_bits_per_spike"
    n_bins_var = "test_light_swapped_segment_n_bins"
    swap_segment_var = "swap_segment_index_1based"
    join_columns = [
        "animal_name",
        "date",
        "region",
        "dark_train_epoch",
        "light_train_epoch",
        "light_test_epoch",
        "trajectory",
        "unit",
    ]
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        glm_path = get_swap_glm_selected_comparison_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        empirical_path = get_swap_tuning_curve_comparison_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        if not glm_path.exists() or not empirical_path.exists():
            missing_paths.extend(
                path for path in (glm_path, empirical_path) if not path.exists()
            )
            continue

        with xr.open_dataset(glm_path) as glm_dataset:
            for variable_name in (
                raw_ll_sum_var,
                raw_ll_bits_var,
                n_bins_var,
                swap_segment_var,
            ):
                if variable_name not in glm_dataset:
                    raise KeyError(f"{glm_path} is missing {variable_name!r}.")
            available_models = {str(value) for value in glm_dataset.coords["model"].values}
            missing_models = [
                model_name
                for model_name in glm_models.values()
                if model_name not in available_models
            ]
            if missing_models:
                raise KeyError(
                    f"{glm_path} is missing GLM model(s): "
                    + ", ".join(missing_models)
                )
            trajectories = [str(value) for value in glm_dataset.coords["trajectory"].values]
            units = np.asarray(glm_dataset.coords["unit"].values, dtype=int)
            trajectory_grid, unit_grid = np.meshgrid(
                np.asarray(trajectories, dtype=object),
                units,
                indexing="ij",
            )
            glm_table = pd.DataFrame(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "region": region,
                    "dark_train_epoch": dataset_dark_epoch,
                    "light_train_epoch": light_train_epoch,
                    "light_test_epoch": light_test_epoch,
                    "trajectory": trajectory_grid.ravel(),
                    "unit": unit_grid.ravel(),
                    "glm_test_light_bin_count": np.repeat(
                        np.asarray(glm_dataset[n_bins_var].values, dtype=float),
                        units.size,
                    ),
                    "swap_segment_index_1based": np.repeat(
                        np.asarray(glm_dataset[swap_segment_var].values, dtype=int),
                        units.size,
                    ),
                    "glm_source_path": str(glm_path),
                }
            )
            for label, model_name in glm_models.items():
                glm_table[f"{label}_ll_sum"] = np.asarray(
                    glm_dataset[raw_ll_sum_var].sel(model=model_name).values,
                    dtype=float,
                ).ravel()
                glm_table[f"{label}_bits_per_spike"] = np.asarray(
                    glm_dataset[raw_ll_bits_var].sel(model=model_name).values,
                    dtype=float,
                ).ravel()

        empirical_table = pd.read_parquet(empirical_path)
        empirical_table = empirical_table[
            empirical_table["model"].astype(str) == str(empirical_model_name)
        ].copy()
        empirical_required_columns = set(join_columns).union(
            {
                "ll_sum",
                "ll_bits_per_spike",
                "test_light_bin_count",
            }
        )
        missing_columns = sorted(
            empirical_required_columns.difference(empirical_table.columns)
        )
        if missing_columns:
            raise KeyError(
                f"{empirical_path} is missing required column(s): "
                + ", ".join(missing_columns)
            )
        empirical_table = empirical_table[
            [
                *join_columns,
                "ll_sum",
                "ll_bits_per_spike",
                "test_light_bin_count",
            ]
        ].rename(
            columns={
                "ll_sum": f"{empirical_label}_ll_sum",
                "ll_bits_per_spike": (
                    f"{empirical_label}_bits_per_spike"
                ),
                "test_light_bin_count": "empirical_test_light_bin_count",
            }
        )
        empirical_table["empirical_source_path"] = str(empirical_path)
        merged = pd.merge(
            glm_table,
            empirical_table,
            on=join_columns,
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue
        if not np.allclose(
            merged["glm_test_light_bin_count"].to_numpy(dtype=float),
            merged["empirical_test_light_bin_count"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "GLM and empirical swap comparison bin counts differ for "
                f"{animal_name} {date}."
            )
        score_columns = [
            "V_ll_sum",
            f"{MIXED_GLM_TASK_LABEL}_ll_sum",
            f"{empirical_label}_ll_sum",
            "V_bits_per_spike",
            f"{MIXED_GLM_TASK_LABEL}_bits_per_spike",
            f"{empirical_label}_bits_per_spike",
        ]
        finite_mask = np.ones(len(merged), dtype=bool)
        for column in score_columns:
            finite_mask &= np.isfinite(merged[column].to_numpy(dtype=float))
        merged = merged[finite_mask].copy()
        if merged.empty:
            continue
        labels = np.asarray(
            ["V", MIXED_GLM_TASK_LABEL, empirical_label],
            dtype=object,
        )
        ll_values = merged[
            [
                "V_ll_sum",
                f"{MIXED_GLM_TASK_LABEL}_ll_sum",
                f"{empirical_label}_ll_sum",
            ]
        ].to_numpy(dtype=float)
        max_ll = np.nanmax(ll_values, axis=1)
        tie_mask = (
            np.isclose(ll_values, max_ll[:, None], rtol=0.0, atol=1e-12).sum(axis=1)
            > 1
        )
        winner_index = np.nanargmax(ll_values, axis=1)
        merged["winner"] = labels[winner_index]
        merged.loc[tie_mask, "winner"] = "tie"
        merged["delta_V_minus_task_bits_per_spike"] = (
            merged["V_bits_per_spike"]
            - merged[f"{MIXED_GLM_TASK_LABEL}_bits_per_spike"]
        )
        merged[f"delta_V_minus_{empirical_label}_bits_per_spike"] = (
            merged["V_bits_per_spike"]
            - merged[f"{empirical_label}_bits_per_spike"]
        )
        rows.append(merged)

    if rows:
        return pd.concat(rows, axis=0, ignore_index=True)

    columns = [
        *join_columns,
        "swap_segment_index_1based",
        "winner",
        "delta_V_minus_task_bits_per_spike",
        f"delta_V_minus_{empirical_label}_bits_per_spike",
        "glm_source_path",
        "empirical_source_path",
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No matched GLM/empirical swap comparison artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    return pd.DataFrame(columns=columns)


def load_mixed_glm_full_additive_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Any:
    """Load matched GLM visual/MS and empirical additive swap scores."""
    return load_mixed_glm_empirical_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
        empirical_model_name=MIXED_EMPIRICAL_AD_MODEL_NAME,
        empirical_label=MIXED_EMPIRICAL_AD_LABEL,
    )


def load_hybrid_glm_empirical_delta_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Any:
    """Score GLM V/MS and empirical-dark times GLM multiplier on swap bins."""
    import pandas as pd
    import xarray as xr

    rows: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    required_swap_vars = {
        "swap_source_trajectory",
        "swap_segment_index_1based",
        "test_light_occupancy_s",
        "test_light_spike_count",
        "dark_hz_grid",
        "test_light_swapped_hz_grid",
        "test_light_swapped_segment_n_bins",
        "test_light_swapped_segment_swapped_spike_sum",
    }
    required_empirical_vars = {
        "same_dark_train_tuning_hz",
        "other_dark_train_tuning_hz",
        "other_light_train_tuning_hz",
        "segment_bin_mask",
    }
    required_glm_vars = {
        "coef_segment_scalar_gain",
        "coef_light_offset",
    }
    glm_models = {
        "V": "visual",
        MIXED_GLM_TASK_LABEL: SCALAR_MODEL_NAME,
    }

    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        swap_path = get_swap_glm_selected_comparison_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        empirical_path = get_swap_tuning_curve_comparison_dataset_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        scalar_glm_path = get_dark_light_glm_selected_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=light_train_epoch,
            dark_epoch=dataset_dark_epoch,
            model_name=SCALAR_MODEL_NAME,
        )
        if (
            not swap_path.exists()
            or not empirical_path.exists()
            or not scalar_glm_path.exists()
        ):
            missing_paths.extend(
                path
                for path in (swap_path, empirical_path, scalar_glm_path)
                if not path.exists()
            )
            continue

        with xr.open_dataset(swap_path) as swap_dataset, xr.open_dataset(
            empirical_path
        ) as empirical_dataset, xr.open_dataset(scalar_glm_path) as scalar_dataset:
            missing_swap_vars = sorted(required_swap_vars.difference(swap_dataset.data_vars))
            if missing_swap_vars:
                raise KeyError(
                    f"{swap_path} is missing variable(s): "
                    + ", ".join(missing_swap_vars)
                )
            missing_empirical_vars = sorted(
                required_empirical_vars.difference(empirical_dataset.data_vars)
            )
            if missing_empirical_vars:
                raise KeyError(
                    f"{empirical_path} is missing variable(s): "
                    + ", ".join(missing_empirical_vars)
                )
            missing_glm_vars = sorted(required_glm_vars.difference(scalar_dataset.data_vars))
            if missing_glm_vars:
                raise KeyError(
                    f"{scalar_glm_path} is missing variable(s): "
                    + ", ".join(missing_glm_vars)
                )

            available_models = {str(value) for value in swap_dataset.coords["model"].values}
            missing_models = [
                model_name
                for model_name in glm_models.values()
                if model_name not in available_models
            ]
            if missing_models:
                raise KeyError(
                    f"{swap_path} is missing GLM model(s): "
                    + ", ".join(missing_models)
                )

            swap_units = [
                int(unit_id) for unit_id in np.asarray(swap_dataset.coords["unit"].values)
            ]
            empirical_units = [
                int(unit_id)
                for unit_id in np.asarray(empirical_dataset.coords["unit"].values)
            ]
            scalar_units = [
                int(unit_id)
                for unit_id in np.asarray(scalar_dataset.coords["unit"].values)
            ]
            empirical_unit_index = {
                unit_id: index for index, unit_id in enumerate(empirical_units)
            }
            scalar_unit_index = {
                unit_id: index for index, unit_id in enumerate(scalar_units)
            }
            common_units = [
                unit_id
                for unit_id in swap_units
                if unit_id in empirical_unit_index and unit_id in scalar_unit_index
            ]
            if not common_units:
                continue
            swap_unit_indices = [
                swap_units.index(unit_id) for unit_id in common_units
            ]
            empirical_unit_indices = [
                empirical_unit_index[unit_id] for unit_id in common_units
            ]
            scalar_unit_indices = [
                scalar_unit_index[unit_id] for unit_id in common_units
            ]

            observed_tp = np.asarray(
                swap_dataset.coords["tp_observed_bin"].values,
                dtype=float,
            )
            observed_edges = np.asarray(
                swap_dataset.coords["tp_observed_edge"].values,
                dtype=float,
            )
            glm_tp_grid = np.asarray(swap_dataset.coords["tp_grid"].values, dtype=float)
            empirical_tp_grid = np.asarray(
                empirical_dataset.coords["tp_bin"].values,
                dtype=float,
            )
            segment_edges = np.asarray(
                swap_dataset.coords["segment_edge"].values,
                dtype=float,
            )
            bin_size_s = float(swap_dataset.attrs.get("bin_size_s", np.nan))
            if not np.isfinite(bin_size_s) or bin_size_s <= 0.0:
                raise ValueError(f"{swap_path} has invalid bin_size_s={bin_size_s!r}.")

            for trajectory in np.asarray(
                swap_dataset.coords["trajectory"].values,
                dtype=str,
            ):
                source_trajectory = str(
                    np.asarray(
                        swap_dataset["swap_source_trajectory"].sel(
                            trajectory=trajectory
                        ).values
                    ).item()
                )
                segment_index = int(
                    np.asarray(
                        swap_dataset["swap_segment_index_1based"].sel(
                            trajectory=trajectory
                        ).values
                    ).item()
                ) - 1
                segment_weights = _segment_overlap_weights(
                    observed_edges,
                    segment_edges,
                    segment_index,
                )
                scored_bins = segment_weights > 0.0
                if not np.any(scored_bins):
                    continue

                occupancy = np.asarray(
                    swap_dataset["test_light_occupancy_s"]
                    .sel(trajectory=trajectory)
                    .values,
                    dtype=float,
                )
                counts = np.asarray(
                    swap_dataset["test_light_spike_count"]
                    .sel(trajectory=trajectory)
                    .transpose("tp_observed_bin", "unit")
                    .values,
                    dtype=float,
                )[:, swap_unit_indices]
                segment_occupancy = (
                    occupancy[scored_bins] * segment_weights[scored_bins]
                )
                segment_counts = (
                    counts[scored_bins] * segment_weights[scored_bins, None]
                )
                saved_bin_count = float(
                    np.asarray(
                        swap_dataset["test_light_swapped_segment_n_bins"].sel(
                            trajectory=trajectory
                        ).values
                    ).item()
                )
                observed_bin_count = float(np.sum(segment_occupancy) / bin_size_s)
                if observed_bin_count <= 0.0:
                    raise ValueError(
                        "Hybrid scoring found no occupancy in the saved swap "
                        f"segment for {animal_name} {date} {trajectory}."
                    )
                segment_occupancy = segment_occupancy * (
                    saved_bin_count / observed_bin_count
                )
                saved_spike_sum = np.asarray(
                    swap_dataset["test_light_swapped_segment_swapped_spike_sum"]
                    .sel(model="visual", trajectory=trajectory)
                    .values,
                    dtype=float,
                )[swap_unit_indices]
                weighted_spike_sum = np.sum(segment_counts, axis=0)
                scalable = (
                    np.isfinite(saved_spike_sum)
                    & (saved_spike_sum >= 0.0)
                    & (weighted_spike_sum > 0.0)
                )
                segment_counts[:, scalable] *= (
                    saved_spike_sum[scalable] / weighted_spike_sum[scalable]
                )[None, :]
                impossible = (
                    np.isfinite(saved_spike_sum)
                    & (saved_spike_sum > 0.0)
                    & (weighted_spike_sum <= 0.0)
                )
                if np.any(impossible):
                    raise ValueError(
                        "Hybrid scoring could not distribute saved swapped-segment "
                        "spikes from the aggregate TP-bin summary for "
                        f"{animal_name} {date} {trajectory}."
                    )

                score_values: dict[str, dict[str, np.ndarray]] = {}
                for label, model_name in glm_models.items():
                    rate_grid = np.asarray(
                        swap_dataset["test_light_swapped_hz_grid"]
                        .sel(model=model_name, trajectory=trajectory)
                        .transpose("tp_grid", "unit")
                        .values,
                        dtype=float,
                    )[:, swap_unit_indices]
                    rates_observed = _interpolate_rate_matrix(
                        glm_tp_grid,
                        rate_grid,
                        observed_tp,
                    )
                    ll_sum, bits_per_spike, spike_sum = (
                        _aggregate_poisson_ll_bits_per_spike(
                            spike_counts=segment_counts,
                            occupancy_s=segment_occupancy,
                            rates_hz=rates_observed[scored_bins],
                        )
                    )
                    score_values[label] = {
                        "ll_sum": ll_sum,
                        "bits_per_spike": bits_per_spike,
                        "spike_sum": spike_sum,
                    }

                same_dark = np.asarray(
                    empirical_dataset["same_dark_train_tuning_hz"]
                    .sel(trajectory=trajectory)
                    .transpose("tp_bin", "unit")
                    .values,
                    dtype=float,
                )[:, empirical_unit_indices]
                source_segment_gain = np.asarray(
                    scalar_dataset["coef_segment_scalar_gain"]
                    .sel(trajectory=source_trajectory)
                    .isel(segment_basis=segment_index)
                    .values,
                    dtype=float,
                )[scalar_unit_indices]
                target_light_offset = np.asarray(
                    scalar_dataset["coef_light_offset"]
                    .sel(trajectory=trajectory)
                    .values,
                    dtype=float,
                )[scalar_unit_indices]
                hybrid_rates = np.clip(
                    same_dark
                    * np.exp(source_segment_gain + target_light_offset)[None, :],
                    MULTIPLIER_EPSILON,
                    None,
                )
                hybrid_observed = _interpolate_rate_matrix(
                    empirical_tp_grid,
                    hybrid_rates,
                    observed_tp,
                )
                hybrid_ll_sum, hybrid_bits_per_spike, hybrid_spike_sum = (
                    _aggregate_poisson_ll_bits_per_spike(
                        spike_counts=segment_counts,
                        occupancy_s=segment_occupancy,
                        rates_hz=hybrid_observed[scored_bins],
                    )
                )
                score_values[HYBRID_GLM_EMPIRICAL_LABEL] = {
                    "ll_sum": hybrid_ll_sum,
                    "bits_per_spike": hybrid_bits_per_spike,
                    "spike_sum": hybrid_spike_sum,
                }

                empirical_segment_mask = np.asarray(
                    empirical_dataset["segment_bin_mask"].sel(
                        trajectory=trajectory
                    ).values,
                    dtype=bool,
                )
                if not np.any(empirical_segment_mask):
                    continue
                other_dark = np.asarray(
                    empirical_dataset["other_dark_train_tuning_hz"]
                    .sel(trajectory=trajectory)
                    .transpose("tp_bin", "unit")
                    .values,
                    dtype=float,
                )[:, empirical_unit_indices]
                other_light = np.asarray(
                    empirical_dataset["other_light_train_tuning_hz"]
                    .sel(trajectory=trajectory)
                    .transpose("tp_bin", "unit")
                    .values,
                    dtype=float,
                )[:, empirical_unit_indices]
                empirical_gain = np.sum(
                    other_light[empirical_segment_mask],
                    axis=0,
                ) / np.clip(
                    np.sum(other_dark[empirical_segment_mask], axis=0),
                    MULTIPLIER_EPSILON,
                    None,
                )
                glm_dark_rate_grid = np.asarray(
                    swap_dataset["dark_hz_grid"]
                    .sel(model=SCALAR_MODEL_NAME, trajectory=trajectory)
                    .transpose("tp_grid", "unit")
                    .values,
                    dtype=float,
                )[:, swap_unit_indices]
                reverse_hybrid_rates = np.clip(
                    glm_dark_rate_grid * empirical_gain[None, :],
                    MULTIPLIER_EPSILON,
                    None,
                )
                reverse_hybrid_observed = _interpolate_rate_matrix(
                    glm_tp_grid,
                    reverse_hybrid_rates,
                    observed_tp,
                )
                reverse_hybrid_ll_sum, reverse_hybrid_bits_per_spike, _ = (
                    _aggregate_poisson_ll_bits_per_spike(
                        spike_counts=segment_counts,
                        occupancy_s=segment_occupancy,
                        rates_hz=reverse_hybrid_observed[scored_bins],
                    )
                )
                score_values[REVERSE_HYBRID_GLM_EMPIRICAL_LABEL] = {
                    "ll_sum": reverse_hybrid_ll_sum,
                    "bits_per_spike": reverse_hybrid_bits_per_spike,
                    "spike_sum": hybrid_spike_sum,
                }

                labels = np.asarray(
                    [
                        "V",
                        MIXED_GLM_TASK_LABEL,
                        HYBRID_GLM_EMPIRICAL_LABEL,
                        REVERSE_HYBRID_GLM_EMPIRICAL_LABEL,
                    ],
                    dtype=object,
                )
                ll_values = np.column_stack(
                    [score_values[str(label)]["ll_sum"] for label in labels]
                )
                bits_values = np.column_stack(
                    [score_values[str(label)]["bits_per_spike"] for label in labels]
                )
                finite = np.all(np.isfinite(ll_values), axis=1) & np.all(
                    np.isfinite(bits_values),
                    axis=1,
                )
                if not np.any(finite):
                    continue
                ll_values = ll_values[finite]
                bits_values = bits_values[finite]
                finite_units = [
                    unit_id
                    for unit_id, keep in zip(common_units, finite, strict=True)
                    if bool(keep)
                ]
                spike_sum = score_values["V"]["spike_sum"][finite]
                max_ll = np.nanmax(ll_values, axis=1)
                tie_mask = (
                    np.isclose(ll_values, max_ll[:, None], rtol=0.0, atol=1e-12).sum(
                        axis=1
                    )
                    > 1
                )
                winner_index = np.nanargmax(ll_values, axis=1)
                winners = labels[winner_index]
                winners[tie_mask] = "tie"

                for row_index, unit_id in enumerate(finite_units):
                    rows.append(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "region": region,
                            "dark_train_epoch": dataset_dark_epoch,
                            "light_train_epoch": light_train_epoch,
                            "light_test_epoch": light_test_epoch,
                            "trajectory": trajectory,
                            "source_trajectory": source_trajectory,
                            "swap_segment_index_1based": segment_index + 1,
                            "unit": int(unit_id),
                            "test_light_spike_sum": float(spike_sum[row_index]),
                            "test_light_bin_count": saved_bin_count,
                            "winner": str(winners[row_index]),
                            "V_ll_sum": float(ll_values[row_index, 0]),
                            f"{MIXED_GLM_TASK_LABEL}_ll_sum": float(
                                ll_values[row_index, 1]
                            ),
                            f"{HYBRID_GLM_EMPIRICAL_LABEL}_ll_sum": float(
                                ll_values[row_index, 2]
                            ),
                            f"{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_ll_sum": float(
                                ll_values[row_index, 3]
                            ),
                            "V_bits_per_spike": float(bits_values[row_index, 0]),
                            f"{MIXED_GLM_TASK_LABEL}_bits_per_spike": float(
                                bits_values[row_index, 1]
                            ),
                            f"{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike": float(
                                bits_values[row_index, 2]
                            ),
                            (
                                f"{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}"
                                "_bits_per_spike"
                            ): float(bits_values[row_index, 3]),
                            "delta_V_minus_task_bits_per_spike": float(
                                bits_values[row_index, 0] - bits_values[row_index, 1]
                            ),
                            (
                                "delta_V_minus_"
                                f"{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike"
                            ): float(
                                bits_values[row_index, 0] - bits_values[row_index, 2]
                            ),
                            (
                                "delta_V_minus_"
                                f"{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike"
                            ): float(
                                bits_values[row_index, 0] - bits_values[row_index, 3]
                            ),
                            "glm_swap_source_path": str(swap_path),
                            "empirical_source_path": str(empirical_path),
                            "glm_scalar_source_path": str(scalar_glm_path),
                        }
                    )

    columns = [
        "animal_name",
        "date",
        "region",
        "dark_train_epoch",
        "light_train_epoch",
        "light_test_epoch",
        "trajectory",
        "source_trajectory",
        "swap_segment_index_1based",
        "unit",
        "test_light_spike_sum",
        "test_light_bin_count",
        "winner",
        "V_ll_sum",
        f"{MIXED_GLM_TASK_LABEL}_ll_sum",
        f"{HYBRID_GLM_EMPIRICAL_LABEL}_ll_sum",
        f"{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_ll_sum",
        "V_bits_per_spike",
        f"{MIXED_GLM_TASK_LABEL}_bits_per_spike",
        f"{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike",
        f"{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike",
        "delta_V_minus_task_bits_per_spike",
        f"delta_V_minus_{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike",
        f"delta_V_minus_{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike",
        "glm_swap_source_path",
        "empirical_source_path",
        "glm_scalar_source_path",
    ]
    if rows:
        return pd.DataFrame(rows, columns=columns)
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No matched hybrid GLM/empirical swap artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    return pd.DataFrame(columns=columns)


def _require_table_columns(table: Any, path: Path, columns: Sequence[str]) -> None:
    """Raise if a loaded table is missing required columns."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Table {path} is missing columns {missing!r}.")


def _load_reliable_tuning_keys(
    *,
    table_path: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_correlation: float,
) -> dict[tuple[str, int], float]:
    """Return reliable `(trajectory, unit)` keys from one stability table."""
    import pandas as pd

    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run `python -m v1ca1.task_progression.stability` "
            f"for {animal_name} {date} first."
        )
    table = pd.read_parquet(table_path)
    _require_table_columns(
        table,
        table_path,
        ("unit", "region", "epoch", "trajectory_type", "stability_correlation"),
    )
    correlations = pd.to_numeric(table["stability_correlation"], errors="coerce")
    units = pd.to_numeric(table["unit"], errors="coerce")
    reliable = table[
        (table["region"].astype(str) == str(region))
        & (table["epoch"].astype(str) == str(epoch))
        & np.isfinite(correlations.to_numpy(dtype=float))
        & np.isfinite(units.to_numpy(dtype=float))
        & (correlations.to_numpy(dtype=float) >= float(min_correlation))
    ].copy()
    if reliable.empty:
        return {}
    reliable["unit"] = units.loc[reliable.index].astype(int)
    reliable["stability_correlation"] = correlations.loc[reliable.index].astype(float)
    grouped = reliable.groupby(["trajectory_type", "unit"], sort=False)[
        "stability_correlation"
    ].max()
    return {
        (str(trajectory), int(unit)): float(correlation)
        for (trajectory, unit), correlation in grouped.items()
    }


def load_full_segment_log_gain_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    min_tuning_stability_correlation: float = (
        FULL_SEGMENT_GAIN_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load full scalar segment log gains for reliable trajectory-unit fits."""
    import pandas as pd
    import xarray as xr

    rows: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    if float(min_tuning_stability_correlation) < -1.0:
        raise ValueError("min_tuning_stability_correlation must be at least -1.")

    required_glm_vars = {
        "coef_segment_scalar_gain",
        "coef_light_offset",
        SCALAR_BASELINE_SCORE_VARIABLE,
    }
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        reliability_epoch = dataset_dark_epoch
        glm_path = get_dark_light_glm_selected_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=light_train_epoch,
            dark_epoch=dataset_dark_epoch,
            model_name=SCALAR_MODEL_NAME,
        )
        stability_path = get_stability_table_path(data_root, animal_name, date)
        if not glm_path.exists() or not stability_path.exists():
            missing_paths.extend(
                path for path in (glm_path, stability_path) if not path.exists()
            )
            continue

        reliable_keys = _load_reliable_tuning_keys(
            table_path=stability_path,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=reliability_epoch,
            min_correlation=float(min_tuning_stability_correlation),
        )
        if not reliable_keys:
            continue

        with xr.open_dataset(glm_path) as glm_dataset:
            missing_glm_vars = sorted(required_glm_vars.difference(glm_dataset.data_vars))
            if missing_glm_vars:
                raise KeyError(
                    f"{glm_path} is missing variable(s): "
                    + ", ".join(missing_glm_vars)
                )

            glm_units = [
                int(unit_id) for unit_id in np.asarray(glm_dataset.coords["unit"].values)
            ]
            unit_index = {unit_id: index for index, unit_id in enumerate(glm_units)}
            trajectories = [
                str(trajectory)
                for trajectory in np.asarray(glm_dataset.coords["trajectory"].values)
            ]
            if "segment_basis" in glm_dataset.coords:
                segment_basis_values = np.asarray(
                    glm_dataset.coords["segment_basis"].values,
                    dtype=int,
                )
            else:
                segment_basis_values = np.arange(
                    int(glm_dataset["coef_segment_scalar_gain"].shape[1]),
                    dtype=int,
                )

            for trajectory in trajectories:
                common_units = [
                    unit_id
                    for unit_id in glm_units
                    if (trajectory, unit_id) in reliable_keys
                ]
                if not common_units:
                    continue
                common_unit_indices = [unit_index[unit_id] for unit_id in common_units]
                segment_gain = np.asarray(
                    glm_dataset["coef_segment_scalar_gain"]
                    .sel(trajectory=trajectory)
                    .transpose("segment_basis", "unit")
                    .values,
                    dtype=float,
                )[:, common_unit_indices]
                light_offset = np.asarray(
                    glm_dataset["coef_light_offset"].sel(trajectory=trajectory).values,
                    dtype=float,
                )[common_unit_indices]
                baseline_score = np.asarray(
                    glm_dataset[SCALAR_BASELINE_SCORE_VARIABLE]
                    .sel(trajectory=trajectory)
                    .values,
                    dtype=float,
                )[common_unit_indices]
                full_gain = segment_gain + light_offset[None, :]

                for segment_index, segment_basis in enumerate(segment_basis_values):
                    for unit_position, unit_id in enumerate(common_units):
                        value = float(full_gain[segment_index, unit_position])
                        segment_value = float(segment_gain[segment_index, unit_position])
                        offset_value = float(light_offset[unit_position])
                        baseline_value = float(baseline_score[unit_position])
                        stability = float(reliable_keys[(trajectory, unit_id)])
                        if not (
                            np.isfinite(value)
                            and np.isfinite(segment_value)
                            and np.isfinite(offset_value)
                            and np.isfinite(baseline_value)
                            and np.isfinite(stability)
                        ):
                            continue
                        rows.append(
                            {
                                "animal_name": animal_name,
                                "date": date,
                                "region": region,
                                "dark_train_epoch": dataset_dark_epoch,
                                "light_train_epoch": light_train_epoch,
                                "reliability_epoch": reliability_epoch,
                                "trajectory": trajectory,
                                "segment_index_1based": int(segment_index + 1),
                                "segment_basis": int(segment_basis),
                                "unit": int(unit_id),
                                "full_segment_log_gain": value,
                                "segment_specific_log_gain": segment_value,
                                "light_offset_log_gain": offset_value,
                                SCALAR_BASELINE_SCORE_COLUMN: baseline_value,
                                "stability_correlation": stability,
                                "glm_source_path": str(glm_path),
                                "stability_source_path": str(stability_path),
                            }
                        )

    if rows:
        return pd.DataFrame(rows, columns=FULL_SEGMENT_GAIN_TABLE_COLUMNS)
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No full segment log-gain artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    return pd.DataFrame(columns=FULL_SEGMENT_GAIN_TABLE_COLUMNS)


def load_nested_vision_modulation_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    dark_active_fr_threshold_hz: float = NESTED_DARK_ACTIVE_FR_THRESHOLD_HZ,
    min_tuning_stability_correlation: float = (
        NESTED_TUNING_STABILITY_CORRELATION_THRESHOLD
    ),
    full_segment_log_gain_threshold: float = FULL_SEGMENT_LOG_GAIN_THRESHOLD,
    min_scalar_baseline_score: float = NESTED_SCALAR_BASELINE_SCORE_THRESHOLD,
    full_gain_table: Any | None = None,
) -> Any:
    """Return nested dark activity, stability, and vision-modulation counts."""
    import pandas as pd

    if float(dark_active_fr_threshold_hz) < 0.0:
        raise ValueError("dark_active_fr_threshold_hz must be non-negative.")
    if float(min_tuning_stability_correlation) < -1.0:
        raise ValueError("min_tuning_stability_correlation must be at least -1.")
    if float(full_segment_log_gain_threshold) < 0.0:
        raise ValueError("full_segment_log_gain_threshold must be non-negative.")
    if not np.isfinite(float(min_scalar_baseline_score)):
        raise ValueError("min_scalar_baseline_score must be finite.")

    if full_gain_table is None:
        full_gain_table = load_full_segment_log_gain_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            dark_epoch=dark_epoch,
            light_train_epoch=light_train_epoch,
            min_tuning_stability_correlation=float(min_tuning_stability_correlation),
        )

    full_gain_table = (
        pd.DataFrame(columns=FULL_SEGMENT_GAIN_TABLE_COLUMNS)
        if full_gain_table is None
        else full_gain_table
    )
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        rate_table = load_dark_movement_firing_rate_table(
            data_root,
            animal_name=animal_name,
            date=date,
            dark_epoch=dataset_dark_epoch,
            region=region,
        )
        _require_table_columns(
            rate_table,
            Path("<dark_movement_firing_rate_table>"),
            ("unit", "dark_firing_rate_hz"),
        )
        rate_units = pd.to_numeric(rate_table["unit"], errors="coerce")
        dark_rates = pd.to_numeric(rate_table["dark_firing_rate_hz"], errors="coerce")
        finite_rates = np.isfinite(rate_units.to_numpy(dtype=float)) & np.isfinite(
            dark_rates.to_numpy(dtype=float)
        )
        unit_values = rate_units.loc[finite_rates].astype(int).to_numpy(dtype=int)
        rate_values = dark_rates.loc[finite_rates].to_numpy(dtype=float)
        all_units = {int(unit_id) for unit_id in unit_values}
        active_units = {
            int(unit_id)
            for unit_id, rate in zip(unit_values, rate_values, strict=True)
            if float(rate) >= float(dark_active_fr_threshold_hz)
        }

        stability_path = get_stability_table_path(data_root, animal_name, date)
        reliable_keys = _load_reliable_tuning_keys(
            table_path=stability_path,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=dataset_dark_epoch,
            min_correlation=float(min_tuning_stability_correlation),
        )
        stable_units = {int(unit_id) for _trajectory, unit_id in reliable_keys}
        active_stable_units = active_units & stable_units

        gain_rows = full_gain_table[
            (full_gain_table["animal_name"].astype(str) == str(animal_name))
            & (full_gain_table["date"].astype(str) == str(date))
            & (full_gain_table["region"].astype(str) == str(region))
            & (full_gain_table["dark_train_epoch"].astype(str) == str(dataset_dark_epoch))
            & (full_gain_table["light_train_epoch"].astype(str) == str(light_train_epoch))
        ]
        assessed_units = {
            int(unit_id)
            for unit_id in pd.to_numeric(gain_rows["unit"], errors="coerce").dropna()
        }
        if SCALAR_BASELINE_SCORE_COLUMN in gain_rows.columns:
            score_units = pd.to_numeric(gain_rows["unit"], errors="coerce")
            baseline_scores = pd.to_numeric(
                gain_rows[SCALAR_BASELINE_SCORE_COLUMN],
                errors="coerce",
            )
            score_table = pd.DataFrame(
                {
                    "unit": score_units,
                    "baseline_score": baseline_scores,
                }
            )
            finite_scores = np.isfinite(score_table["unit"].to_numpy(dtype=float)) & (
                np.isfinite(score_table["baseline_score"].to_numpy(dtype=float))
            )
            score_table = score_table.loc[finite_scores].copy()
            if score_table.empty:
                baseline_qualified_units: set[int] = set()
            else:
                score_table["unit"] = score_table["unit"].astype(int)
                score_by_unit = score_table.groupby("unit", sort=False)[
                    "baseline_score"
                ].median()
                baseline_qualified_units = {
                    int(unit_id)
                    for unit_id, score in score_by_unit.items()
                    if float(score) >= float(min_scalar_baseline_score)
                }
        else:
            baseline_qualified_units = set(assessed_units)
        gain_values = pd.to_numeric(
            gain_rows["full_segment_log_gain"],
            errors="coerce",
        )
        modulated_rows = gain_rows[
            np.isfinite(gain_values.to_numpy(dtype=float))
            & (
                np.abs(gain_values.to_numpy(dtype=float))
                >= float(full_segment_log_gain_threshold)
            )
        ]
        modulated_units = {
            int(unit_id)
            for unit_id in pd.to_numeric(modulated_rows["unit"], errors="coerce").dropna()
        }

        assessed_active_stable_units = (
            active_stable_units & assessed_units & baseline_qualified_units
        )
        modulated_active_stable_units = assessed_active_stable_units & modulated_units
        unmodulated_active_stable_units = (
            assessed_active_stable_units - modulated_active_stable_units
        )
        missing_scalar_fit_units = active_stable_units - assessed_units
        scalar_below_baseline_units = (
            active_stable_units & assessed_units
        ) - baseline_qualified_units
        no_scalar_fit_units = missing_scalar_fit_units | scalar_below_baseline_units

        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "dark_epoch": dataset_dark_epoch,
                "light_train_epoch": light_train_epoch,
                "total_cell_count": int(len(all_units)),
                "dark_inactive_count": int(len(all_units - active_units)),
                "dark_active_count": int(len(active_units)),
                "dark_active_unstable_count": int(len(active_units - stable_units)),
                "dark_active_stable_count": int(len(active_stable_units)),
                "dark_active_stable_no_scalar_fit_count": int(len(no_scalar_fit_units)),
                "dark_active_stable_missing_scalar_fit_count": int(
                    len(missing_scalar_fit_units)
                ),
                "dark_active_stable_scalar_below_baseline_count": int(
                    len(scalar_below_baseline_units)
                ),
                "dark_active_stable_unmodulated_count": int(
                    len(unmodulated_active_stable_units)
                ),
                "dark_active_stable_modulated_count": int(
                    len(modulated_active_stable_units)
                ),
                "dark_active_fr_threshold_hz": float(dark_active_fr_threshold_hz),
                "tuning_stability_correlation_threshold": float(
                    min_tuning_stability_correlation
                ),
                "scalar_baseline_score_threshold": float(min_scalar_baseline_score),
                "full_segment_log_gain_threshold": float(
                    full_segment_log_gain_threshold
                ),
                "dark_firing_rate_source_path": "",
                "stability_source_path": str(stability_path),
            }
        )

    if rows:
        return pd.DataFrame(rows, columns=NESTED_MODULATION_TABLE_COLUMNS)
    return pd.DataFrame(columns=NESTED_MODULATION_TABLE_COLUMNS)


def load_scalar_multiplier_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    light_train_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TRAIN_EPOCH,
    light_test_epoch: str = EMPIRICAL_PAIRWISE_LIGHT_TEST_EPOCH,
) -> Any:
    """Load matched empirical and GLM scalar multipliers for swapped segments."""
    import pandas as pd
    import xarray as xr

    rows: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    required_empirical_vars = {
        "other_dark_train_tuning_hz",
        "other_light_train_tuning_hz",
        "segment_bin_mask",
        "swap_source_trajectory",
        "swap_segment_index_1based",
    }
    required_glm_vars = {
        "coef_segment_scalar_gain",
        "coef_light_offset",
    }
    for dataset in datasets:
        animal_name, date, _dataset_dark_epoch = normalize_dataset_id(dataset)
        dataset_dark_epoch = get_dark_epoch(animal_name, date, dark_epoch)
        empirical_path = get_swap_tuning_curve_comparison_dataset_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            dark_epoch=dataset_dark_epoch,
            light_train_epoch=light_train_epoch,
            light_test_epoch=light_test_epoch,
        )
        glm_path = get_dark_light_glm_selected_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            light_epoch=light_train_epoch,
            dark_epoch=dataset_dark_epoch,
            model_name=SCALAR_MODEL_NAME,
        )
        if not empirical_path.exists() or not glm_path.exists():
            missing_paths.extend(
                path for path in (empirical_path, glm_path) if not path.exists()
            )
            continue

        with xr.open_dataset(empirical_path) as empirical_dataset, xr.open_dataset(
            glm_path
        ) as glm_dataset:
            missing_empirical_vars = sorted(
                required_empirical_vars.difference(empirical_dataset.data_vars)
            )
            if missing_empirical_vars:
                raise KeyError(
                    f"{empirical_path} is missing variable(s): "
                    + ", ".join(missing_empirical_vars)
                )
            missing_glm_vars = sorted(required_glm_vars.difference(glm_dataset.data_vars))
            if missing_glm_vars:
                raise KeyError(
                    f"{glm_path} is missing variable(s): "
                    + ", ".join(missing_glm_vars)
                )

            empirical_units = [
                int(unit_id)
                for unit_id in np.asarray(empirical_dataset.coords["unit"].values)
            ]
            empirical_unit_index = {
                int(unit_id): index for index, unit_id in enumerate(empirical_units)
            }
            glm_units = [
                int(unit_id) for unit_id in np.asarray(glm_dataset.coords["unit"].values)
            ]
            glm_unit_index = {
                int(unit_id): index for index, unit_id in enumerate(glm_units)
            }
            common_units = [
                unit_id for unit_id in empirical_units if unit_id in glm_unit_index
            ]
            if not common_units:
                continue
            empirical_unit_indices = [
                empirical_unit_index[unit_id] for unit_id in common_units
            ]
            glm_unit_indices = [glm_unit_index[unit_id] for unit_id in common_units]

            for trajectory in np.asarray(
                empirical_dataset.coords["trajectory"].values,
                dtype=str,
            ):
                source_trajectory = str(
                    np.asarray(
                        empirical_dataset["swap_source_trajectory"].sel(
                            trajectory=trajectory
                        ).values
                    ).item()
                )
                segment_index = int(
                    np.asarray(
                        empirical_dataset["swap_segment_index_1based"].sel(
                            trajectory=trajectory
                        ).values
                    ).item()
                ) - 1
                segment_mask = np.asarray(
                    empirical_dataset["segment_bin_mask"].sel(
                        trajectory=trajectory
                    ).values,
                    dtype=bool,
                )
                if not np.any(segment_mask):
                    continue

                other_dark = np.asarray(
                    empirical_dataset["other_dark_train_tuning_hz"]
                    .sel(trajectory=trajectory)
                    .transpose("tp_bin", "unit")
                    .values,
                    dtype=float,
                )[:, empirical_unit_indices]
                other_light = np.asarray(
                    empirical_dataset["other_light_train_tuning_hz"]
                    .sel(trajectory=trajectory)
                    .transpose("tp_bin", "unit")
                    .values,
                    dtype=float,
                )[:, empirical_unit_indices]
                empirical_numerator = np.sum(other_light[segment_mask], axis=0)
                empirical_denominator = np.sum(other_dark[segment_mask], axis=0)
                log_empirical_ms_gain = np.log(
                    np.clip(empirical_numerator, MULTIPLIER_EPSILON, None)
                ) - np.log(
                    np.clip(empirical_denominator, MULTIPLIER_EPSILON, None)
                )

                source_segment_gain = np.asarray(
                    glm_dataset["coef_segment_scalar_gain"]
                    .sel(trajectory=source_trajectory)
                    .isel(segment_basis=segment_index)
                    .values,
                    dtype=float,
                )[glm_unit_indices]
                target_light_offset = np.asarray(
                    glm_dataset["coef_light_offset"]
                    .sel(trajectory=trajectory)
                    .values,
                    dtype=float,
                )[glm_unit_indices]
                log_glm_segment_gain = source_segment_gain
                log_glm_full_gain = source_segment_gain + target_light_offset

                for unit_id, log_empirical, log_segment, log_full in zip(
                    common_units,
                    log_empirical_ms_gain,
                    log_glm_segment_gain,
                    log_glm_full_gain,
                    strict=True,
                ):
                    if not (
                        np.isfinite(log_empirical)
                        and np.isfinite(log_segment)
                        and np.isfinite(log_full)
                    ):
                        continue
                    rows.append(
                        {
                            "animal_name": animal_name,
                            "date": date,
                            "region": region,
                            "dark_train_epoch": dataset_dark_epoch,
                            "light_train_epoch": light_train_epoch,
                            "light_test_epoch": light_test_epoch,
                            "trajectory": trajectory,
                            "source_trajectory": source_trajectory,
                            "swap_segment_index_1based": segment_index + 1,
                            "unit": int(unit_id),
                            "log_empirical_ms_gain": float(log_empirical),
                            "log_glm_segment_gain": float(log_segment),
                            "log_glm_full_gain": float(log_full),
                            "delta_log_glm_segment_minus_empirical": float(
                                log_segment - log_empirical
                            ),
                            "delta_log_glm_full_minus_empirical": float(
                                log_full - log_empirical
                            ),
                            "empirical_source_path": str(empirical_path),
                            "glm_source_path": str(glm_path),
                        }
                    )

    columns = [
        "animal_name",
        "date",
        "region",
        "dark_train_epoch",
        "light_train_epoch",
        "light_test_epoch",
        "trajectory",
        "source_trajectory",
        "swap_segment_index_1based",
        "unit",
        "log_empirical_ms_gain",
        "log_glm_segment_gain",
        "log_glm_full_gain",
        "delta_log_glm_segment_minus_empirical",
        "delta_log_glm_full_minus_empirical",
        "empirical_source_path",
        "glm_source_path",
    ]
    if rows:
        return pd.DataFrame(rows, columns=columns)
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "No matched GLM/empirical scalar multiplier artifacts were available. "
            f"Missing paths included:\n{missing_text}"
        )
    return pd.DataFrame(columns=columns)


def plot_scalar_multiplier_histograms(ax: Any, multiplier_table: Any) -> None:
    """Plot pooled empirical-vs-GLM scalar multiplier comparisons."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    inset_bounds = (
        (0.08, 0.30, 0.25, 0.56),
        (0.38, 0.30, 0.25, 0.56),
        (0.68, 0.30, 0.25, 0.56),
    )
    histogram_axes = [ax.inset_axes(bounds) for bounds in inset_bounds]
    if multiplier_table is None or len(multiplier_table) == 0:
        ax.text(0.5, 0.5, "No matched scalar\nmultiplier values", ha="center", va="center")
        return

    pooled_values = []
    for _title, x_column, y_column, _xlabel, _ylabel, _cmap in MULTIPLIER_COMPARISON_SPECS:
        for column in (x_column, y_column):
            values = np.asarray(multiplier_table[column], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                pooled_values.append(values)
    delta_values = np.asarray(
        multiplier_table["delta_log_glm_full_minus_empirical"],
        dtype=float,
    )
    delta_values = delta_values[np.isfinite(delta_values)]
    if delta_values.size:
        pooled_values.append(delta_values)
    if not pooled_values:
        ax.text(0.5, 0.5, "No finite scalar\nmultiplier values", ha="center", va="center")
        return

    combined_xy_values = np.concatenate(
        [
            np.asarray(multiplier_table[column], dtype=float)[
                np.isfinite(np.asarray(multiplier_table[column], dtype=float))
            ]
            for _title, x_column, y_column, _xlabel, _ylabel, _cmap in (
                MULTIPLIER_COMPARISON_SPECS
            )
            for column in (x_column, y_column)
        ]
    )
    q_low, q_high = np.nanpercentile(combined_xy_values, [1.0, 99.0])
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        q_low = float(np.nanmin(combined_xy_values))
        q_high = float(np.nanmax(combined_xy_values))
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        q_low, q_high = -1.0, 1.0
    xy_pad = max(0.10, 0.08 * (q_high - q_low))
    xy_min = q_low - xy_pad
    xy_max = q_high + xy_pad

    for hist_axis, (title, x_column, y_column, xlabel, ylabel, cmap) in zip(
        histogram_axes[:2],
        MULTIPLIER_COMPARISON_SPECS,
        strict=True,
    ):
        x_values = np.asarray(multiplier_table[x_column], dtype=float)
        y_values = np.asarray(multiplier_table[y_column], dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite]
        y_values = y_values[finite]
        if x_values.size == 0:
            hist_axis.text(0.5, 0.5, "No values", ha="center", va="center")
            hist_axis.axis("off")
            continue
        hist_axis.hist2d(
            x_values,
            y_values,
            bins=34,
            range=((xy_min, xy_max), (xy_min, xy_max)),
            cmap=cmap,
            cmin=1,
        )
        hist_axis.plot(
            [xy_min, xy_max],
            [xy_min, xy_max],
            color="0.25",
            linestyle="--",
            linewidth=0.7,
        )
        hist_axis.axvline(0.0, color="0.70", linewidth=0.45, zorder=0)
        hist_axis.axhline(0.0, color="0.70", linewidth=0.45, zorder=0)
        median_delta = float(np.nanmedian(y_values - x_values))
        hist_axis.text(
            0.98,
            0.04,
            f"med Δ {median_delta:.2f}\nn={x_values.size}",
            ha="right",
            va="bottom",
            fontsize=4.6,
            color="0.20",
            transform=hist_axis.transAxes,
        )
        hist_axis.set_xlim(xy_min, xy_max)
        hist_axis.set_ylim(xy_min, xy_max)
        hist_axis.set_aspect("equal", adjustable="box")
        hist_axis.set_title(title, fontsize=5.5, pad=1.5)
        hist_axis.set_xlabel(xlabel, labelpad=1.0)
        hist_axis.set_ylabel(ylabel, labelpad=1.0)
        hist_axis.spines["top"].set_visible(False)
        hist_axis.spines["right"].set_visible(False)
        hist_axis.tick_params(labelsize=5.0, length=1.5, pad=1.0)

    delta_axis = histogram_axes[2]
    if delta_values.size == 0:
        delta_axis.text(0.5, 0.5, "No values", ha="center", va="center")
        delta_axis.axis("off")
    else:
        delta_q_low, delta_q_high = np.nanpercentile(delta_values, [1.0, 99.0])
        if (
            not np.isfinite(delta_q_low)
            or not np.isfinite(delta_q_high)
            or delta_q_high <= delta_q_low
        ):
            delta_q_low = float(np.nanmin(delta_values))
            delta_q_high = float(np.nanmax(delta_values))
        if (
            not np.isfinite(delta_q_low)
            or not np.isfinite(delta_q_high)
            or delta_q_high <= delta_q_low
        ):
            delta_q_low, delta_q_high = -1.0, 1.0
        delta_pad = max(0.10, 0.08 * (delta_q_high - delta_q_low))
        delta_min = delta_q_low - delta_pad
        delta_max = delta_q_high + delta_pad
        bins = np.linspace(delta_min, delta_max, 30)
        delta_axis.hist(
            delta_values,
            bins=bins,
            weights=np.ones(delta_values.size, dtype=float) / float(delta_values.size),
            color="#D95F02",
            alpha=0.74,
            edgecolor="white",
            linewidth=0.25,
        )
        median_delta = float(np.nanmedian(delta_values))
        delta_axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.7)
        delta_axis.axvline(median_delta, color="black", linewidth=0.7)
        delta_axis.text(
            0.98,
            0.92,
            f"med {median_delta:.2f}\nn={delta_values.size}",
            ha="right",
            va="top",
            fontsize=4.6,
            color="0.20",
            transform=delta_axis.transAxes,
        )
        delta_axis.set_xlim(delta_min, delta_max)
        delta_axis.set_title("GLM full - empirical", fontsize=5.5, pad=1.5)
        delta_axis.set_xlabel("Δ log gain", labelpad=1.0)
        delta_axis.set_ylabel("Fraction", labelpad=1.0)
        delta_axis.spines["top"].set_visible(False)
        delta_axis.spines["right"].set_visible(False)
        delta_axis.tick_params(labelsize=5.0, length=1.5, pad=1.0)

    ax.text(
        0.50,
        0.015,
        "Dashed diagonal/vertical line = equal multipliers",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="0.25",
        transform=ax.transAxes,
    )


def filter_swapped_segment_shared_scaffold_gain_table(
    gain_table: Any,
    comparison_table: Any,
) -> Any:
    """Return swapped-segment gains where shared-scaffold beats additive."""
    import pandas as pd

    gain_columns = list(FULL_SEGMENT_GAIN_TABLE_COLUMNS)
    if gain_table is None or comparison_table is None:
        return pd.DataFrame(columns=gain_columns)

    gain_table = pd.DataFrame(gain_table).copy()
    comparison_table = pd.DataFrame(comparison_table).copy()
    if gain_table.empty or comparison_table.empty:
        return pd.DataFrame(columns=gain_columns)

    join_columns = [
        "animal_name",
        "date",
        "region",
        "dark_train_epoch",
        "light_train_epoch",
        "trajectory",
        "unit",
    ]
    required_gain_columns = set(join_columns).union(
        {
            "segment_index_1based",
            "segment_specific_log_gain",
        }
    )
    required_comparison_columns = set(join_columns).union(
        {
            "swap_segment_index_1based",
            f"{MIXED_GLM_TASK_LABEL}_bits_per_spike",
            f"{MIXED_EMPIRICAL_AD_LABEL}_bits_per_spike",
        }
    )
    missing_gain_columns = sorted(required_gain_columns.difference(gain_table.columns))
    missing_comparison_columns = sorted(
        required_comparison_columns.difference(comparison_table.columns)
    )
    if missing_gain_columns or missing_comparison_columns:
        missing_text = []
        if missing_gain_columns:
            missing_text.append(f"gain table: {missing_gain_columns}")
        if missing_comparison_columns:
            missing_text.append(f"comparison table: {missing_comparison_columns}")
        raise KeyError("; ".join(missing_text))

    shared_scores = pd.to_numeric(
        comparison_table[f"{MIXED_GLM_TASK_LABEL}_bits_per_spike"],
        errors="coerce",
    )
    additive_scores = pd.to_numeric(
        comparison_table[f"{MIXED_EMPIRICAL_AD_LABEL}_bits_per_spike"],
        errors="coerce",
    )
    swap_segments = pd.to_numeric(
        comparison_table["swap_segment_index_1based"],
        errors="coerce",
    )
    better_rows = comparison_table[
        np.isfinite(shared_scores.to_numpy(dtype=float))
        & np.isfinite(additive_scores.to_numpy(dtype=float))
        & np.isfinite(swap_segments.to_numpy(dtype=float))
        & (shared_scores.to_numpy(dtype=float) > additive_scores.to_numpy(dtype=float))
    ].copy()
    if better_rows.empty:
        return gain_table.iloc[0:0].copy()
    better_rows["swap_segment_index_1based"] = (
        swap_segments.loc[better_rows.index].astype(int)
    )
    better_rows = better_rows[
        [*join_columns, "swap_segment_index_1based"]
    ].drop_duplicates()

    merged = pd.merge(
        gain_table,
        better_rows,
        on=join_columns,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return gain_table.iloc[0:0].copy()
    segment_indices = pd.to_numeric(
        merged["segment_index_1based"],
        errors="coerce",
    )
    swap_segment_indices = pd.to_numeric(
        merged["swap_segment_index_1based"],
        errors="coerce",
    )
    segment_values = segment_indices.to_numpy(dtype=float)
    swap_segment_values = swap_segment_indices.to_numpy(dtype=float)
    finite_segments = np.isfinite(segment_values) & np.isfinite(swap_segment_values)
    return merged[
        finite_segments & (segment_values == swap_segment_values)
    ].copy()


def plot_swapped_segment_shared_scaffold_gain_histograms(
    ax: Any,
    gain_table: Any,
    comparison_table: Any,
    *,
    threshold: float = FULL_SEGMENT_LOG_GAIN_THRESHOLD,
) -> None:
    """Plot swapped-segment coefficients where shared-scaffold beats additive."""
    trajectory_types = tuple(PANEL_H_DELTA_TRAJECTORIES)
    filtered_table = filter_swapped_segment_shared_scaffold_gain_table(
        gain_table,
        comparison_table,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    inset_bounds = (
        (0.07, 0.34, 0.20, 0.51),
        (0.305, 0.34, 0.20, 0.51),
        (0.54, 0.34, 0.20, 0.51),
        (0.775, 0.34, 0.20, 0.51),
    )
    histogram_axes = [ax.inset_axes(bounds) for bounds in inset_bounds]
    if filtered_table.empty:
        ax.text(
            0.5,
            0.5,
            "No swapped-segment\nShared-scaffold > Additive values",
            ha="center",
            va="center",
        )
        return

    all_values = np.asarray(filtered_table["segment_specific_log_gain"], dtype=float)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite swapped-segment\ncoefficients",
            ha="center",
            va="center",
        )
        return

    q_abs = float(np.nanpercentile(np.abs(all_values), 99.0))
    axis_abs = max(float(threshold) * 1.35, 0.25, q_abs * 1.08)
    if not np.isfinite(axis_abs) or axis_abs <= 0.0:
        axis_abs = 1.0
    bins = np.linspace(-axis_abs, axis_abs, 31)

    for trajectory_index, (axis, trajectory_type) in enumerate(
        zip(histogram_axes, trajectory_types, strict=True)
    ):
        rows = filtered_table[
            filtered_table["trajectory"].astype(str) == str(trajectory_type)
        ]
        values = np.asarray(rows["segment_specific_log_gain"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            axis.text(0.5, 0.5, "No values", ha="center", va="center")
            axis.axis("off")
            continue
        weights = np.ones(values.size, dtype=float) / float(values.size)
        axis.hist(
            values,
            bins=bins,
            weights=weights,
            color=PANEL_TRAJECTORY_COLORS[trajectory_type],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.25,
        )
        median_value = float(np.nanmedian(values))
        large_fraction = float(np.mean(np.abs(values) >= float(threshold)))
        axis.axvline(0.0, color="0.35", linewidth=0.65)
        axis.axvline(float(threshold), color="0.35", linestyle="--", linewidth=0.65)
        axis.axvline(-float(threshold), color="0.35", linestyle="--", linewidth=0.65)
        axis.axvline(median_value, color="black", linewidth=0.75)
        axis.text(
            0.98,
            0.92,
            f"med {median_value:.2f}\n|g|≥thr {large_fraction:.2f}\nn={values.size}",
            ha="right",
            va="top",
            fontsize=4.6,
            color="0.20",
            transform=axis.transAxes,
        )
        axis.set_xlim(-axis_abs, axis_abs)
        axis.set_title(
            PANEL_TRAJECTORY_LABELS.get(trajectory_type, trajectory_type),
            fontsize=5.5,
            pad=1.5,
        )
        axis.set_xlabel("")
        if trajectory_index == 0:
            axis.set_ylabel("Fraction", labelpad=1.0)
        else:
            axis.set_ylabel("")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=5.0, length=1.5, pad=1.0)

    unique_cells = filtered_table[
        ["animal_name", "date", "region", "unit"]
    ].drop_duplicates()
    unique_fits = filtered_table[
        ["animal_name", "date", "region", "trajectory", "unit"]
    ].drop_duplicates()
    ax.text(
        0.50,
        0.16,
        "Swapped-segment coefficient",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color="black",
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.055,
        (
            "Swapped segment only; Shared-scaffold > Additive; "
            f"cells={len(unique_cells)}, trajectory-unit fits={len(unique_fits)}"
        ),
        ha="center",
        va="bottom",
        fontsize=4.9,
        color="0.25",
        transform=ax.transAxes,
    )


def plot_combined_full_segment_log_gain_histogram(
    ax: Any,
    gain_table: Any,
    *,
    threshold: float = FULL_SEGMENT_LOG_GAIN_THRESHOLD,
) -> None:
    """Plot full scalar segment log gains pooled across all segments."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    histogram_axis = ax.inset_axes((0.14, 0.24, 0.72, 0.62))
    if gain_table is None or len(gain_table) == 0:
        ax.text(0.5, 0.5, "No reliable scalar\ngain values", ha="center", va="center")
        return

    values = np.asarray(gain_table["full_segment_log_gain"], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.text(0.5, 0.5, "No finite scalar\ngain values", ha="center", va="center")
        return

    q_abs = float(np.nanpercentile(np.abs(values), 99.0))
    axis_abs = max(float(threshold) * 1.35, 0.25, q_abs * 1.08)
    if not np.isfinite(axis_abs) or axis_abs <= 0.0:
        axis_abs = 1.0
    bins = np.linspace(-axis_abs, axis_abs, 37)
    weights = np.ones(values.size, dtype=float) / float(values.size)
    histogram_axis.hist(
        values,
        bins=bins,
        weights=weights,
        color="#4D4D4D",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    median_value = float(np.nanmedian(values))
    large_fraction = float(np.mean(np.abs(values) >= float(threshold)))
    positive_fraction = float(np.mean(values >= float(threshold)))
    negative_fraction = float(np.mean(values <= -float(threshold)))
    histogram_axis.axvline(0.0, color="0.35", linewidth=0.7)
    histogram_axis.axvline(float(threshold), color="0.35", linestyle="--", linewidth=0.7)
    histogram_axis.axvline(-float(threshold), color="0.35", linestyle="--", linewidth=0.7)
    histogram_axis.axvline(median_value, color="black", linewidth=0.8)
    histogram_axis.text(
        0.98,
        0.92,
        (
            f"med {median_value:.2f}\n"
            f"|g|>=thr {large_fraction:.2f}\n"
            f"+ {positive_fraction:.2f}, - {negative_fraction:.2f}\n"
            f"n={values.size}"
        ),
        ha="right",
        va="top",
        fontsize=4.8,
        color="0.20",
        transform=histogram_axis.transAxes,
    )
    histogram_axis.set_xlim(-axis_abs, axis_abs)
    histogram_axis.set_xlabel("Full segment log gain, pooled segments", labelpad=1.0)
    histogram_axis.set_ylabel("Fraction", labelpad=1.0)
    histogram_axis.spines["top"].set_visible(False)
    histogram_axis.spines["right"].set_visible(False)
    histogram_axis.tick_params(labelsize=5.0, length=1.5, pad=1.0)

    unique_cells = gain_table[
        ["animal_name", "date", "region", "unit"]
    ].drop_duplicates()
    unique_fits = gain_table[
        ["animal_name", "date", "region", "trajectory", "unit"]
    ].drop_duplicates()
    ax.text(
        0.50,
        0.035,
        (
            f"Pooled segments 1-3; threshold = +/-log(1.5); "
            f"cells={len(unique_cells)}, trajectory-unit fits={len(unique_fits)}"
        ),
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="0.25",
        transform=ax.transAxes,
    )


def plot_nested_vision_modulation_bar(
    ax: Any,
    nested_table: Any,
    *,
    legend_axis: Any | None = None,
    show_note: bool = True,
) -> None:
    """Plot nested dark activity, tuning stability, and vision modulation fractions."""
    from matplotlib.patches import Patch

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, 2.55)
    ax.axvline(1.0, color="0.85", linewidth=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=5.0, length=1.5, pad=1.0)
    ax.set_xlabel("Conditional fraction", labelpad=1.0)

    if nested_table is None or len(nested_table) == 0:
        ax.text(0.5, 0.5, "No nested modulation\nsummary values", ha="center", va="center")
        ax.set_yticks([])
        return

    def _sum_count(column: str) -> int:
        return int(np.nansum(np.asarray(nested_table[column], dtype=float)))

    rows = (
        (
            2.0,
            "All V1 cells",
            _sum_count("total_cell_count"),
            (
                ("dark_inactive", "Dark inactive", _sum_count("dark_inactive_count")),
                ("dark_active", "Dark active", _sum_count("dark_active_count")),
            ),
        ),
        (
            1.0,
            "Dark active",
            _sum_count("dark_active_count"),
            (
                (
                    "unstable",
                    "Unstable",
                    _sum_count("dark_active_unstable_count"),
                ),
                ("stable", "Stable", _sum_count("dark_active_stable_count")),
            ),
        ),
        (
            0.0,
            "Dark active + stable",
            _sum_count("dark_active_stable_count"),
            (
                (
                    "no_scalar_fit",
                    "No/poor scalar fit",
                    _sum_count("dark_active_stable_no_scalar_fit_count"),
                ),
                (
                    "not_modulated",
                    "Not modulated",
                    _sum_count("dark_active_stable_unmodulated_count"),
                ),
                (
                    "modulated",
                    "Modulated",
                    _sum_count("dark_active_stable_modulated_count"),
                ),
            ),
        ),
    )

    for y_position, _label, denominator, segments in rows:
        left = 0.0
        if denominator <= 0:
            ax.text(
                0.02,
                y_position,
                "n=0",
                ha="left",
                va="center",
                fontsize=5.2,
                color="0.35",
            )
            continue
        for key, _segment_label, count in segments:
            width = float(count) / float(denominator)
            ax.barh(
                y_position,
                width,
                left=left,
                height=0.42,
                color=NESTED_MODULATION_COLORS[key],
                edgecolor="white",
                linewidth=0.45,
            )
            if width >= 0.08:
                text_color = "white" if key in {"dark_active", "stable", "modulated"} else "0.15"
                ax.text(
                    left + width / 2.0,
                    y_position,
                    f"{count}\n{width:.0%}",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color=text_color,
                )
            left += width
        ax.text(
            1.015,
            y_position,
            f"n={denominator}",
            ha="left",
            va="center",
            fontsize=5.2,
            color="0.25",
            transform=ax.get_yaxis_transform(),
        )

    ax.set_yticks([row[0] for row in rows])
    ax.set_yticklabels([row[1] for row in rows], fontsize=5.4)

    handles = [
        Patch(facecolor=NESTED_MODULATION_COLORS[key], edgecolor="none", label=label)
        for key, label in (
            ("dark_inactive", "Dark inactive"),
            ("dark_active", "Dark active"),
            ("unstable", "Unstable"),
            ("stable", "Stable"),
            ("no_scalar_fit", "No/poor scalar fit"),
            ("not_modulated", "Not modulated"),
            ("modulated", "Modulated"),
        )
    ]
    if legend_axis is None:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncols=4,
            frameon=False,
            fontsize=4.8,
            handlelength=0.9,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
    else:
        legend_axis.set_xlim(0.0, 1.0)
        legend_axis.set_ylim(0.0, 1.0)
        legend_axis.axis("off")
        legend_axis.legend(
            handles=handles,
            loc="center left",
            ncols=2,
            frameon=False,
            fontsize=4.8,
            handlelength=0.9,
            columnspacing=0.9,
            labelspacing=0.8,
            borderaxespad=0.0,
        )

    if show_note:
        ax.text(
            0.0,
            -0.35,
            (
                f"Dark active: FR >= {NESTED_DARK_ACTIVE_FR_THRESHOLD_HZ:.1f} Hz; "
                f"stable: odd/even r >= {NESTED_TUNING_STABILITY_CORRELATION_THRESHOLD:.1f}; "
                "usable scalar: CV gain >= 0 bits/spike; "
                "modulated: any segment |g| >= log(1.5)"
            ),
            ha="left",
            va="center",
            fontsize=5.0,
            color="0.25",
        )


def plot_empirical_pairwise_delta(ax: Any, delta_table: Any) -> None:
    """Plot pooled V-MS and V-AS delta LL boxes split by trajectory."""
    from matplotlib.patches import Patch

    trajectory_types = tuple(PANEL_H_DELTA_TRAJECTORIES)
    pair_labels = tuple(label for label, _column in EMPIRICAL_PAIRWISE_DELTA_PAIRS)
    pair_centers = np.asarray(
        [len(pair_labels) - pair_index - 1 for pair_index in range(len(pair_labels))],
        dtype=float,
    )
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    if delta_table is None or len(delta_table) == 0:
        ax.text(0.5, 0.5, "No empirical\nswap values", ha="center", va="center")
    else:
        plotted_values = []
        plotted_positions = []
        plotted_colors = []
        comparison_fractions: dict[str, tuple[float, float]] = {}
        for pair_index, (_pair_label, column) in enumerate(
            EMPIRICAL_PAIRWISE_DELTA_PAIRS
        ):
            all_values = np.asarray(delta_table[column], dtype=float)
            all_values = all_values[np.isfinite(all_values)]
            if all_values.size:
                comparison_fractions[_pair_label] = (
                    float(np.mean(all_values < 0.0)),
                    float(np.mean(all_values > 0.0)),
                )
            for trajectory_index, trajectory_type in enumerate(trajectory_types):
                rows = delta_table[
                    delta_table["trajectory"].astype(str) == str(trajectory_type)
                ]
                values = np.asarray(rows[column], dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                plotted_values.append(values)
                plotted_positions.append(
                    pair_centers[pair_index]
                    + EMPIRICAL_PAIRWISE_BOX_OFFSETS[trajectory_index]
                )
                plotted_colors.append(PANEL_TRAJECTORY_COLORS[trajectory_type])

        if not plotted_values:
            ax.text(
                0.5,
                0.5,
                "No finite empirical\nswap values",
                ha="center",
                va="center",
            )
        else:
            boxplot = ax.boxplot(
                plotted_values,
                positions=plotted_positions,
                widths=EMPIRICAL_PAIRWISE_BOX_WIDTH,
                vert=False,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 0.75},
                whiskerprops={"color": "0.30", "linewidth": 0.55},
                capprops={"color": "0.30", "linewidth": 0.55},
                boxprops={"linewidth": 0.55},
            )
            for patch, color in zip(boxplot["boxes"], plotted_colors, strict=True):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.62)

            legend_handles = [
                Patch(
                    facecolor=PANEL_TRAJECTORY_COLORS[trajectory_type],
                    edgecolor="0.25",
                    alpha=0.62,
                    label=PANEL_TRAJECTORY_LABELS.get(
                        trajectory_type,
                        trajectory_type,
                    ),
                )
                for trajectory_type in trajectory_types
            ]
            ax.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=4.6,
                handlelength=0.9,
                ncols=2,
                borderaxespad=0.0,
                loc="lower right",
            )

            whisker_values = [
                value
                for whisker in boxplot["whiskers"]
                for value in whisker.get_xdata()
                if np.isfinite(value)
            ]
            if whisker_values:
                x_min = min(whisker_values)
                x_max = max(whisker_values)
                x_pad = max(0.05, 0.08 * (x_max - x_min))
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
            x_left, x_right = ax.get_xlim()
            x_span = x_right - x_left
            for pair_index, (pair_label, _column) in enumerate(
                EMPIRICAL_PAIRWISE_DELTA_PAIRS
            ):
                other_label = pair_label.split("-", maxsplit=1)[1]
                other_fraction, visual_fraction = comparison_fractions.get(
                    pair_label,
                    (float("nan"), float("nan")),
                )
                y_text = pair_centers[pair_index] + 0.40
                if np.isfinite(other_fraction):
                    ax.text(
                        x_left + 0.02 * x_span,
                        y_text,
                        f"{other_label} {other_fraction:.0%}",
                        ha="left",
                        va="center",
                        fontsize=4.5,
                        color="0.25",
                    )
                    ax.text(
                        x_right - 0.02 * x_span,
                        y_text,
                        f"V {visual_fraction:.0%}",
                        ha="right",
                        va="center",
                        fontsize=4.5,
                        color="0.25",
                    )
    ax.set_ylim(-0.50, len(pair_labels) - 0.50)
    if not ax.get_xlim()[0] < 0.0 < ax.get_xlim()[1]:
        ax.set_xlim(min(ax.get_xlim()[0], -0.05), max(ax.get_xlim()[1], 0.05))
    ax.set_yticks(pair_centers)
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("ΔLL (bits/spike)", labelpad=1.2)
    ax.text(
        0.02,
        1.02,
        (
            f"{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}/"
            f"{ADDITIVE_SEGMENT_SHORT_LABEL} better"
        ),
        ha="left",
        va="bottom",
        fontsize=4.6,
        color="0.25",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        0.98,
        1.02,
        "Visual better",
        ha="right",
        va="bottom",
        fontsize=4.6,
        color="0.25",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.2, length=1.5, pad=1.0)


def plot_glm_scalar_pairwise_delta(
    ax: Any,
    delta_table: Any,
    *,
    delta_column: str = "delta_V_minus_scalar_bits_per_spike",
    model_label: str = GLM_SCALAR_MODEL_LABEL,
    no_data_label: str = "No GLM\nswap values",
    no_finite_label: str = "No finite GLM\nswap values",
    show_legend: bool = True,
) -> None:
    """Plot one visual-minus-model delta LL box row by trajectory."""
    from matplotlib.patches import Patch

    trajectory_types = tuple(PANEL_H_DELTA_TRAJECTORIES)
    center = 0.0
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    if delta_table is None or len(delta_table) == 0:
        ax.text(0.5, 0.5, no_data_label, ha="center", va="center")
    else:
        plotted_values = []
        plotted_positions = []
        plotted_colors = []
        all_values = np.asarray(
            delta_table[delta_column],
            dtype=float,
        )
        all_values = all_values[np.isfinite(all_values)]
        model_fraction = float(np.mean(all_values < 0.0)) if all_values.size else np.nan
        visual_fraction = float(np.mean(all_values > 0.0)) if all_values.size else np.nan
        for trajectory_index, trajectory_type in enumerate(trajectory_types):
            rows = delta_table[
                delta_table["trajectory"].astype(str) == str(trajectory_type)
            ]
            values = np.asarray(
                rows[delta_column],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            plotted_values.append(values)
            plotted_positions.append(
                center + EMPIRICAL_PAIRWISE_BOX_OFFSETS[trajectory_index]
            )
            plotted_colors.append(PANEL_TRAJECTORY_COLORS[trajectory_type])

        if not plotted_values:
            ax.text(0.5, 0.5, no_finite_label, ha="center", va="center")
        else:
            boxplot = ax.boxplot(
                plotted_values,
                positions=plotted_positions,
                widths=EMPIRICAL_PAIRWISE_BOX_WIDTH,
                vert=False,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 0.75},
                whiskerprops={"color": "0.30", "linewidth": 0.55},
                capprops={"color": "0.30", "linewidth": 0.55},
                boxprops={"linewidth": 0.55},
            )
            for patch, color in zip(boxplot["boxes"], plotted_colors, strict=True):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.62)

            legend_handles = [
                Patch(
                    facecolor=PANEL_TRAJECTORY_COLORS[trajectory_type],
                    edgecolor="0.25",
                    alpha=0.62,
                    label=PANEL_TRAJECTORY_LABELS.get(
                        trajectory_type,
                        trajectory_type,
                    ),
                )
                for trajectory_type in trajectory_types
            ]
            if show_legend:
                ax.legend(
                    handles=legend_handles,
                    frameon=False,
                    fontsize=4.6,
                    handlelength=0.9,
                    ncols=2,
                    borderaxespad=0.0,
                    loc="lower right",
                )

            whisker_values = [
                value
                for whisker in boxplot["whiskers"]
                for value in whisker.get_xdata()
                if np.isfinite(value)
            ]
            if whisker_values:
                x_min = min(whisker_values)
                x_max = max(whisker_values)
                x_pad = max(0.05, 0.08 * (x_max - x_min))
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
            x_left, x_right = ax.get_xlim()
            x_span = x_right - x_left
            if np.isfinite(model_fraction):
                ax.text(
                    x_left + 0.02 * x_span,
                    center + 0.40,
                    f"{model_label} {model_fraction:.0%}",
                    ha="left",
                    va="center",
                    fontsize=4.5,
                    color="0.25",
                )
                ax.text(
                    x_right - 0.02 * x_span,
                    center + 0.40,
                    f"V {visual_fraction:.0%}",
                    ha="right",
                    va="center",
                    fontsize=4.5,
                    color="0.25",
                )

    ax.set_ylim(-0.50, 0.50)
    if not ax.get_xlim()[0] < 0.0 < ax.get_xlim()[1]:
        ax.set_xlim(min(ax.get_xlim()[0], -0.05), max(ax.get_xlim()[1], 0.05))
    ax.set_yticks([center])
    ax.set_yticklabels([f"V-{model_label}"])
    ax.set_xlabel("ΔLL (bits/spike)", labelpad=1.2)
    ax.text(
        0.02,
        1.02,
        f"{model_label} better",
        ha="left",
        va="bottom",
        fontsize=4.6,
        color="0.25",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        0.98,
        1.02,
        "Visual better",
        ha="right",
        va="bottom",
        fontsize=4.6,
        color="0.25",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.2, length=1.5, pad=1.0)


def plot_mixed_glm_empirical_pairwise_delta(
    ax: Any,
    delta_table: Any,
    *,
    empirical_label: str = MIXED_EMPIRICAL_SA_LABEL,
    show_legend: bool = True,
) -> None:
    """Plot matched GLM/empirical V-MS and V-additive delta LL boxes."""
    from matplotlib.patches import Patch

    trajectory_types = tuple(PANEL_H_DELTA_TRAJECTORIES)
    delta_pairs = (
        (f"V-{MIXED_GLM_TASK_LABEL}", "delta_V_minus_task_bits_per_spike"),
        (
            f"V-{empirical_label}",
            f"delta_V_minus_{empirical_label}_bits_per_spike",
        ),
    )
    pair_labels = tuple(label for label, _column in delta_pairs)
    pair_centers = np.asarray(
        [len(pair_labels) - pair_index - 1 for pair_index in range(len(pair_labels))],
        dtype=float,
    )
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    if delta_table is None or len(delta_table) == 0:
        ax.text(0.5, 0.5, "No matched\nswap values", ha="center", va="center")
    else:
        plotted_values = []
        plotted_positions = []
        plotted_colors = []
        comparison_fractions: dict[str, tuple[float, float]] = {}
        for pair_index, (pair_label, column) in enumerate(delta_pairs):
            all_values = np.asarray(delta_table[column], dtype=float)
            all_values = all_values[np.isfinite(all_values)]
            if all_values.size:
                comparison_fractions[pair_label] = (
                    float(np.mean(all_values < 0.0)),
                    float(np.mean(all_values > 0.0)),
                )
            for trajectory_index, trajectory_type in enumerate(trajectory_types):
                rows = delta_table[
                    delta_table["trajectory"].astype(str) == str(trajectory_type)
                ]
                values = np.asarray(rows[column], dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                plotted_values.append(values)
                plotted_positions.append(
                    pair_centers[pair_index]
                    + EMPIRICAL_PAIRWISE_BOX_OFFSETS[trajectory_index]
                )
                plotted_colors.append(PANEL_TRAJECTORY_COLORS[trajectory_type])

        if not plotted_values:
            ax.text(0.5, 0.5, "No finite matched\nswap values", ha="center", va="center")
        else:
            boxplot = ax.boxplot(
                plotted_values,
                positions=plotted_positions,
                widths=EMPIRICAL_PAIRWISE_BOX_WIDTH,
                vert=False,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 0.75},
                whiskerprops={"color": "0.30", "linewidth": 0.55},
                capprops={"color": "0.30", "linewidth": 0.55},
                boxprops={"linewidth": 0.55},
            )
            for patch, color in zip(boxplot["boxes"], plotted_colors, strict=True):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.62)

            legend_handles = [
                Patch(
                    facecolor=PANEL_TRAJECTORY_COLORS[trajectory_type],
                    edgecolor="0.25",
                    alpha=0.62,
                    label=PANEL_TRAJECTORY_LABELS.get(
                        trajectory_type,
                        trajectory_type,
                    ),
                )
                for trajectory_type in trajectory_types
            ]
            if show_legend:
                ax.legend(
                    handles=legend_handles,
                    frameon=False,
                    fontsize=4.6,
                    handlelength=0.9,
                    ncols=2,
                    borderaxespad=0.0,
                    loc="lower right",
                )

            whisker_values = [
                value
                for whisker in boxplot["whiskers"]
                for value in whisker.get_xdata()
                if np.isfinite(value)
            ]
            if whisker_values:
                x_min = min(whisker_values)
                x_max = max(whisker_values)
                x_pad = max(0.05, 0.08 * (x_max - x_min))
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
            x_left, x_right = ax.get_xlim()
            x_span = x_right - x_left
            for pair_index, (pair_label, _column) in enumerate(delta_pairs):
                other_label = pair_label.split("-", maxsplit=1)[1]
                other_fraction, visual_fraction = comparison_fractions.get(
                    pair_label,
                    (float("nan"), float("nan")),
                )
                y_text = pair_centers[pair_index] + 0.40
                if np.isfinite(other_fraction):
                    ax.text(
                        x_left + 0.02 * x_span,
                        y_text,
                        f"{other_label} {other_fraction:.0%}",
                        ha="left",
                        va="center",
                        fontsize=4.5,
                        color="0.25",
                    )
                    ax.text(
                        x_right - 0.02 * x_span,
                        y_text,
                        f"V {visual_fraction:.0%}",
                        ha="right",
                        va="center",
                        fontsize=4.5,
                        color="0.25",
                    )

    ax.set_ylim(-0.50, len(pair_labels) - 0.50)
    if not ax.get_xlim()[0] < 0.0 < ax.get_xlim()[1]:
        ax.set_xlim(min(ax.get_xlim()[0], -0.05), max(ax.get_xlim()[1], 0.05))
    ax.set_yticks(pair_centers)
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("ΔLL (bits/spike)", labelpad=1.2)
    ax.text(
        0.02,
        1.02,
        f"{MIXED_GLM_TASK_LABEL}/{empirical_label} better",
        ha="left",
        va="bottom",
        fontsize=4.6,
        color="0.25",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        0.98,
        1.02,
        "Visual better",
        ha="right",
        va="bottom",
        fontsize=4.6,
        color="0.25",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.2, length=1.5, pad=1.0)


def plot_empirical_best_fraction_bar(ax: Any, delta_table: Any) -> None:
    """Plot pooled winner fractions for empirical V/MS/AS models."""
    plot_best_fraction_bar(
        ax,
        delta_table,
        labels=(
            "V",
            MULTIPLICATIVE_SEGMENT_SHORT_LABEL,
            ADDITIVE_SEGMENT_SHORT_LABEL,
            "tie",
        ),
        colors=EMPIRICAL_BEST_MODEL_COLORS,
    )


def plot_best_fraction_bar(
    ax: Any,
    delta_table: Any,
    *,
    labels: Sequence[str],
    colors: dict[str, str],
    display_labels: dict[str, str] | None = None,
) -> None:
    """Plot pooled winner fractions for a model set."""
    if delta_table is None or len(delta_table) == 0 or "winner" not in delta_table:
        ax.text(0.5, 0.5, "No best-model\nvalues", ha="center", va="center")
        ax.axis("off")
        return

    winners = np.asarray(delta_table["winner"], dtype=object)
    valid = np.isin(winners, labels)
    winners = winners[valid]
    if winners.size == 0:
        ax.text(0.5, 0.5, "No finite\nwinners", ha="center", va="center")
        ax.axis("off")
        return

    bottom = 0.0
    display_labels = {} if display_labels is None else dict(display_labels)
    for label in labels:
        fraction = float(np.mean(winners == label))
        if fraction <= 0.0:
            continue
        ax.bar(
            [0.0],
            [fraction],
            bottom=[bottom],
            width=0.48,
            color=colors[label],
            edgecolor="white",
            linewidth=0.35,
        )
        if label != "tie" and fraction >= 0.08:
            display_label = display_labels.get(label, str(label))
            if "\n" not in display_label and len(display_label) > 12:
                display_label = display_label.replace("-", "-\n", 1)
            ax.text(
                0.0,
                bottom + fraction / 2.0,
                f"{display_label}\n{fraction:.0%}",
                ha="center",
                va="center",
                fontsize=3.6 if len(display_label.replace("\n", "")) > 8 else 4.5,
                color="white",
            )
        bottom += fraction

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Frac. cells", fontsize=4.8, labelpad=0.8)
    ax.set_title("Best model", fontsize=5.2, pad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=4.5, length=1.2, pad=0.8)
    ax.tick_params(axis="x", length=0)


def plot_glm_scalar_best_fraction_bar(ax: Any, delta_table: Any) -> None:
    """Plot pooled winner fractions for GLM visual and MS."""
    plot_best_fraction_bar(
        ax,
        delta_table,
        labels=("V", GLM_SCALAR_MODEL_LABEL, "tie"),
        colors=GLM_SCALAR_BEST_MODEL_COLORS,
    )


def plot_mixed_glm_empirical_best_fraction_bar(ax: Any, delta_table: Any) -> None:
    """Plot pooled winner fractions for matched GLM/empirical models."""
    plot_best_fraction_bar(
        ax,
        delta_table,
        labels=("V", MIXED_GLM_TASK_LABEL, MIXED_EMPIRICAL_SA_LABEL, "tie"),
        colors=MIXED_GLM_EMPIRICAL_BEST_MODEL_COLORS,
    )


def plot_mixed_glm_full_additive_pairwise_delta(
    ax: Any,
    delta_table: Any,
    *,
    show_legend: bool = True,
) -> None:
    """Plot matched GLM/empirical full-additive delta LL boxes."""
    from matplotlib.patches import Patch

    trajectory_types = tuple(PANEL_H_DELTA_TRAJECTORIES)
    model_columns = {
        FULL_ADDITIVE_INDEPENDENT_LABEL: "V_bits_per_spike",
        FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL: (
            f"{MIXED_GLM_TASK_LABEL}_bits_per_spike"
        ),
        FULL_ADDITIVE_ADDITIVE_LABEL: (
            f"{MIXED_EMPIRICAL_AD_LABEL}_bits_per_spike"
        ),
    }
    delta_pairs = (
        (
            (
                f"{FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL} - "
                f"{FULL_ADDITIVE_INDEPENDENT_LABEL}"
            ),
            FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL,
            FULL_ADDITIVE_INDEPENDENT_LABEL,
        ),
        (
            (
                f"{FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL} - "
                f"{FULL_ADDITIVE_ADDITIVE_LABEL}"
            ),
            FULL_ADDITIVE_SHARED_SCAFFOLD_LABEL,
            FULL_ADDITIVE_ADDITIVE_LABEL,
        ),
        (
            (
                f"{FULL_ADDITIVE_INDEPENDENT_LABEL} - "
                f"{FULL_ADDITIVE_ADDITIVE_LABEL}"
            ),
            FULL_ADDITIVE_INDEPENDENT_LABEL,
            FULL_ADDITIVE_ADDITIVE_LABEL,
        ),
    )
    pair_labels = tuple(label for label, _left_label, _right_label in delta_pairs)
    pair_centers = np.asarray(
        [len(pair_labels) - pair_index - 1 for pair_index in range(len(pair_labels))],
        dtype=float,
    )
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.6, zorder=1)
    required_columns = {"trajectory", *model_columns.values()}
    if delta_table is None or len(delta_table) == 0:
        ax.text(0.5, 0.5, "No matched\nswap values", ha="center", va="center")
    elif not required_columns.issubset(delta_table.columns):
        missing_columns = sorted(required_columns.difference(delta_table.columns))
        ax.text(
            0.5,
            0.5,
            "Missing columns:\n" + "\n".join(missing_columns[:3]),
            ha="center",
            va="center",
            fontsize=5.0,
        )
    else:
        plotted_values = []
        plotted_positions = []
        plotted_colors = []
        comparison_fractions: dict[str, tuple[float, float]] = {}
        for pair_index, (pair_label, left_label, right_label) in enumerate(delta_pairs):
            left_column = model_columns[left_label]
            right_column = model_columns[right_label]
            all_values = (
                np.asarray(delta_table[left_column], dtype=float)
                - np.asarray(delta_table[right_column], dtype=float)
            )
            all_values = all_values[np.isfinite(all_values)]
            if all_values.size:
                comparison_fractions[pair_label] = (
                    float(np.mean(all_values < 0.0)),
                    float(np.mean(all_values > 0.0)),
                )
            for trajectory_index, trajectory_type in enumerate(trajectory_types):
                rows = delta_table[
                    delta_table["trajectory"].astype(str) == str(trajectory_type)
                ]
                values = (
                    np.asarray(rows[left_column], dtype=float)
                    - np.asarray(rows[right_column], dtype=float)
                )
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                plotted_values.append(values)
                plotted_positions.append(
                    pair_centers[pair_index]
                    + EMPIRICAL_PAIRWISE_BOX_OFFSETS[trajectory_index]
                )
                plotted_colors.append(PANEL_TRAJECTORY_COLORS[trajectory_type])

        if not plotted_values:
            ax.text(0.5, 0.5, "No finite matched\nswap values", ha="center", va="center")
        else:
            boxplot = ax.boxplot(
                plotted_values,
                positions=plotted_positions,
                widths=EMPIRICAL_PAIRWISE_BOX_WIDTH,
                vert=False,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "black", "linewidth": 0.75},
                whiskerprops={"color": "0.30", "linewidth": 0.55},
                capprops={"color": "0.30", "linewidth": 0.55},
                boxprops={"linewidth": 0.55},
            )
            for patch, color in zip(boxplot["boxes"], plotted_colors, strict=True):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_alpha(0.62)

            legend_handles = [
                Patch(
                    facecolor=PANEL_TRAJECTORY_COLORS[trajectory_type],
                    edgecolor="0.25",
                    alpha=0.62,
                    label=PANEL_TRAJECTORY_LABELS.get(
                        trajectory_type,
                        trajectory_type,
                    ),
                )
                for trajectory_type in trajectory_types
            ]
            if show_legend:
                ax.legend(
                    handles=legend_handles,
                    frameon=False,
                    fontsize=4.6,
                    handlelength=0.9,
                    ncols=2,
                    borderaxespad=0.0,
                    loc="lower right",
                )

            whisker_values = [
                value
                for whisker in boxplot["whiskers"]
                for value in whisker.get_xdata()
                if np.isfinite(value)
            ]
            if whisker_values:
                x_min = min(whisker_values)
                x_max = max(whisker_values)
                x_pad = max(0.05, 0.08 * (x_max - x_min))
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
            x_left, x_right = ax.get_xlim()
            x_span = x_right - x_left
            for pair_index, (pair_label, left_label, right_label) in enumerate(
                delta_pairs
            ):
                right_fraction, left_fraction = comparison_fractions.get(
                    pair_label,
                    (float("nan"), float("nan")),
                )
                y_text = pair_centers[pair_index] + 0.40
                if np.isfinite(right_fraction):
                    ax.text(
                        x_left + 0.02 * x_span,
                        y_text,
                        f"{right_label} {right_fraction:.0%}",
                        ha="left",
                        va="center",
                        fontsize=4.5,
                        color="0.25",
                    )
                    ax.text(
                        x_right - 0.02 * x_span,
                        y_text,
                        f"{left_label} {left_fraction:.0%}",
                        ha="right",
                        va="center",
                        fontsize=4.5,
                        color="0.25",
                    )

    ax.set_ylim(-0.50, len(pair_labels) - 0.50)
    if not ax.get_xlim()[0] < 0.0 < ax.get_xlim()[1]:
        ax.set_xlim(min(ax.get_xlim()[0], -0.05), max(ax.get_xlim()[1], 0.05))
    ax.set_yticks(pair_centers)
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("ΔLL (bits/spike)", labelpad=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.2, length=1.5, pad=1.0)


def plot_mixed_glm_full_additive_best_fraction_bar(ax: Any, delta_table: Any) -> None:
    """Plot pooled winner fractions for matched GLM/additive models."""
    labels = ("V", MIXED_GLM_TASK_LABEL, MIXED_EMPIRICAL_AD_LABEL, "tie")
    if delta_table is None or len(delta_table) == 0 or "winner" not in delta_table:
        ax.text(0.5, 0.5, "No best-model\nvalues", ha="center", va="center")
        ax.axis("off")
        return

    winners = np.asarray(delta_table["winner"], dtype=object)
    valid = np.isin(winners, labels)
    winners = winners[valid]
    if winners.size == 0:
        ax.text(0.5, 0.5, "No finite\nwinners", ha="center", va="center")
        ax.axis("off")
        return

    fractions = [float(np.mean(winners == label)) for label in labels]
    display_labels = [
        FULL_ADDITIVE_BEST_MODEL_DISPLAY_LABELS.get(label, str(label)).replace(
            "-",
            "-\n",
            1,
        )
        for label in labels
    ]
    left = 0.0
    for label, display_label, fraction in zip(
        labels,
        display_labels,
        fractions,
        strict=True,
    ):
        if fraction <= 0.0:
            continue
        ax.barh(
            [0.0],
            [fraction],
            left=[left],
            height=0.32,
            color=MIXED_GLM_FULL_ADDITIVE_BEST_MODEL_COLORS[label],
            edgecolor="white",
            linewidth=0.35,
        )
        if fraction >= 0.08:
            ax.text(
                left + fraction / 2.0,
                0.28,
                f"{display_label}\n{fraction:.0%}",
                ha="center",
                va="bottom",
                fontsize=3.7,
                color="0.20",
                linespacing=0.90,
            )
        left += fraction

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.28, 0.80)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Frac. cells", fontsize=4.8, labelpad=0.8)
    ax.set_title("Best model", fontsize=5.2, pad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=4.5, length=1.2, pad=0.8)


def plot_empirical_multiplicative_pairwise_delta(ax: Any, delta_table: Any) -> None:
    """Plot empirical visual-vs-MS delta LL boxes."""
    plot_glm_scalar_pairwise_delta(
        ax,
        delta_table,
        delta_column=(
            f"delta_V_minus_{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_bits_per_spike"
        ),
        model_label=MULTIPLICATIVE_SEGMENT_SHORT_LABEL,
        no_data_label="No empirical\nswap values",
        no_finite_label="No finite empirical\nswap values",
        show_legend=False,
    )


def plot_empirical_multiplicative_best_fraction_bar(
    ax: Any,
    delta_table: Any,
) -> None:
    """Plot pooled pairwise winner fractions for empirical V and MS."""
    column = f"delta_V_minus_{MULTIPLICATIVE_SEGMENT_SHORT_LABEL}_bits_per_spike"
    if delta_table is not None and len(delta_table) > 0 and column in delta_table:
        delta_table = delta_table.copy()
        delta = np.asarray(delta_table[column], dtype=float)
        delta_table["winner"] = np.where(
            delta > 0.0,
            "V",
            np.where(delta < 0.0, MULTIPLICATIVE_SEGMENT_SHORT_LABEL, "tie"),
        )
    plot_best_fraction_bar(
        ax,
        delta_table,
        labels=("V", MULTIPLICATIVE_SEGMENT_SHORT_LABEL, "tie"),
        colors=EMPIRICAL_BEST_MODEL_COLORS,
    )


def plot_hybrid_glm_empirical_pairwise_delta(ax: Any, delta_table: Any) -> None:
    """Plot matched GLM and hybrid empirical-dark delta LL boxes."""
    plot_glm_scalar_pairwise_delta(
        ax,
        delta_table,
        delta_column=(
            f"delta_V_minus_{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike"
        ),
        model_label=HYBRID_GLM_EMPIRICAL_LABEL,
        no_data_label="No hybrid\nswap values",
        no_finite_label="No finite hybrid\nswap values",
        show_legend=False,
    )


def plot_hybrid_glm_empirical_best_fraction_bar(ax: Any, delta_table: Any) -> None:
    """Plot pooled winner fractions for matched GLM/hybrid models."""
    if (
        delta_table is not None
        and len(delta_table) > 0
        and f"delta_V_minus_{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike" in delta_table
    ):
        delta_table = delta_table.copy()
        delta = np.asarray(
            delta_table[
                f"delta_V_minus_{HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike"
            ],
            dtype=float,
        )
        delta_table["winner"] = np.where(
            delta > 0.0,
            "V",
            np.where(delta < 0.0, HYBRID_GLM_EMPIRICAL_LABEL, "tie"),
        )
    plot_best_fraction_bar(
        ax,
        delta_table,
        labels=("V", HYBRID_GLM_EMPIRICAL_LABEL, "tie"),
        colors=HYBRID_GLM_EMPIRICAL_BEST_MODEL_COLORS,
    )


def plot_reverse_hybrid_glm_empirical_pairwise_delta(
    ax: Any,
    delta_table: Any,
) -> None:
    """Plot visual-vs-GLM-dark empirical-multiplier hybrid delta LL boxes."""
    plot_glm_scalar_pairwise_delta(
        ax,
        delta_table,
        delta_column=(
            f"delta_V_minus_{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike"
        ),
        model_label=REVERSE_HYBRID_GLM_EMPIRICAL_LABEL,
        no_data_label="No reverse hybrid\nswap values",
        no_finite_label="No finite reverse hybrid\nswap values",
        show_legend=False,
    )


def plot_reverse_hybrid_glm_empirical_best_fraction_bar(
    ax: Any,
    delta_table: Any,
) -> None:
    """Plot pooled pairwise winner fractions for visual and reverse hybrid."""
    column = (
        f"delta_V_minus_{REVERSE_HYBRID_GLM_EMPIRICAL_LABEL}_bits_per_spike"
    )
    if delta_table is not None and len(delta_table) > 0 and column in delta_table:
        delta_table = delta_table.copy()
        delta = np.asarray(delta_table[column], dtype=float)
        delta_table["winner"] = np.where(
            delta > 0.0,
            "V",
            np.where(delta < 0.0, REVERSE_HYBRID_GLM_EMPIRICAL_LABEL, "tie"),
        )
    plot_best_fraction_bar(
        ax,
        delta_table,
        labels=("V", REVERSE_HYBRID_GLM_EMPIRICAL_LABEL, "tie"),
        colors=HYBRID_GLM_EMPIRICAL_BEST_MODEL_COLORS,
    )


def make_supplementary_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    region: str,
    dark_epoch: str | None,
    dpi: int,
) -> Path:
    """Build and save Supplementary Figure 2."""
    import matplotlib.pyplot as plt

    datasets = [normalize_dataset_id(dataset) for dataset in datasets]
    animal_groups = group_datasets_by_animal(datasets)

    apply_paper_style()
    fig_height_mm = get_figure_height_mm(len(animal_groups))
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, fig_height_mm),
        constrained_layout=False,
    )
    if not datasets:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", fontsize=6.0)
        ax.axis("off")
        save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        print(f"Saved Supplementary Figure 2 to {output_path}")
        return output_path

    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=(
            [SCALAR_PANEL_HEIGHT_MM]
            + [MIXED_GLM_EMPIRICAL_PANEL_HEIGHT_MM]
        ),
        hspace=PER_ANIMAL_GRID_HSPACE,
        left=0.125,
        right=0.985,
        top=0.94,
        bottom=0.06,
    )
    scalar_axis = fig.add_subplot(outer_grid[0, 0])
    scalar_swap_delta_table = load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
        min_tuning_stability_correlation=(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        model_name=SCALAR_MODEL_NAME,
    )
    plot_figure_2b_delta_ll_boxplots(
        scalar_axis,
        scalar_swap_delta_table,
        animal_names=tuple(animal_groups),
    )
    scalar_axis.set_title(
        "Shared scaffold - Independent \N{GREEK CAPITAL LETTER DELTA} LL by animal and trajectory",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=2,
    )
    label_axis(scalar_axis, "A", x=-0.115, y=1.02)

    mixed_axis = fig.add_subplot(outer_grid[1, 0])
    mixed_full_additive_table = load_mixed_glm_full_additive_delta_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        dark_epoch=dark_epoch,
    )
    mixed_axis.set_xlim(0.0, 1.0)
    mixed_axis.set_ylim(0.0, 1.0)
    mixed_axis.axis("off")
    mixed_full_delta_axis = mixed_axis.inset_axes(MIXED_GLM_FULL_DELTA_AXIS_BOUNDS)
    plot_mixed_glm_full_additive_pairwise_delta(
        mixed_full_delta_axis,
        mixed_full_additive_table,
        show_legend=False,
    )
    mixed_full_best_axis = mixed_axis.inset_axes(MIXED_GLM_FULL_BEST_AXIS_BOUNDS)
    plot_mixed_glm_full_additive_best_fraction_bar(
        mixed_full_best_axis,
        mixed_full_additive_table,
    )
    mixed_axis.set_title(
        "Comparison between shared-scaffold, independent, and additive model",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=2,
    )
    label_axis(mixed_axis, "B", x=-0.115, y=1.02)

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Supplementary Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Supplementary Figure 2 generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Supplementary Figure 2 scalar controls and model summaries."
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
        "--dark-epoch",
        default=None,
        help=(
            "Dark run epoch. "
            f"Default: registry value, currently {DEFAULT_DARK_EPOCH} unless overridden."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Supplementary Figure 2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_supplementary_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        region=args.region,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
