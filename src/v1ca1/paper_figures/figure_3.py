"""Generate Figure 3 with a three-model architecture schematic."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures import _figure_2_base
from v1ca1.paper_figures import figure_2_old as _figure_2
from v1ca1.paper_figures._dark_light import (
    PANEL_H_HELDOUT_LIGHT_EPOCH,
    PANEL_H_TRAIN_LIGHT_EPOCH,
    load_panel_h_swap_delta_table,
    load_panel_h_swap_examples,
)
from v1ca1.paper_figures.datasets import DatasetId, get_processed_datasets
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_DIR = _figure_2.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "figure_3"
DEFAULT_OUTPUT_FORMAT = _figure_2.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = _figure_2.FIGURE_FORMATS
DEFAULT_REGIONS = _figure_2.DEFAULT_REGIONS
DEFAULT_FIGURE_WIDTH_MM = _figure_2.DEFAULT_FIGURE_WIDTH_MM / 2.0

PANEL_A_SOURCE_INDEPENDENT_ROW_Y = (
    _figure_2_base.PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y
    + _figure_2_base.PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
    + _figure_2.PANEL_D2_INDEPENDENT_ROW_Y_OFFSET
)
PANEL_A_SOURCE_SCAFFOLD_ROW_Y = (
    _figure_2_base.PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
    + _figure_2_base.PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
    + _figure_2.PANEL_D2_SHARED_ROW_Y_OFFSET
)
PANEL_A_MODEL_ROW_PITCH = (
    PANEL_A_SOURCE_INDEPENDENT_ROW_Y - PANEL_A_SOURCE_SCAFFOLD_ROW_Y
)
PANEL_A_SOURCE_AXIS_HEIGHT = 1.0 / (1.0 + PANEL_A_MODEL_ROW_PITCH)
PANEL_A_UPPER_SOURCE_AXIS_BOTTOM = 1.0 - PANEL_A_SOURCE_AXIS_HEIGHT

TOP_ROW_HEIGHT_MM = _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM * (
    1.0 + PANEL_A_MODEL_ROW_PITCH
)
PANEL_B_ROW_HEIGHT_MM = 0.9 * _figure_2.PANEL_D_ROW_HEIGHT_MM
PANEL_C_ROW_HEIGHT_MM = 0.9 * _figure_2.PANEL_D_ROW_HEIGHT_MM
DEFAULT_FIGURE_HEIGHT_MM = (
    TOP_ROW_HEIGHT_MM + PANEL_B_ROW_HEIGHT_MM + PANEL_C_ROW_HEIGHT_MM
)
PANEL_TITLES = (
    "Three models that relate dark and light activity",
    "Dark and light stimulus-swap prediction comparison",
)
PANEL_LABEL_FONTSIZE = 8.0
PANEL_A_B_LABEL_X = -0.035
PANEL_B_LABEL_X = PANEL_A_B_LABEL_X
PANEL_C_LABEL_X = PANEL_A_B_LABEL_X
PANEL_ROW_LABEL_Y = _figure_2.PANEL_B_LABEL_Y
PANEL_BC_TITLE_PAD = _figure_2.PANEL_BC_TITLE_PAD + 2.5

PANEL_B_EXAMPLE_SLOT_BOUNDS = (
    (0.065, 0.170, 0.180, 0.600),
    (0.305, 0.170, 0.180, 0.600),
    (0.545, 0.170, 0.180, 0.600),
    (0.785, 0.170, 0.180, 0.600),
)
PANEL_B_TRACE_LEGEND_SLOT_BOUNDS = (0.180, 0.840, 0.640, 0.100)
PANEL_B_EXAMPLE_ICON_BOUNDS = (-0.235, 0.255, 0.190, 0.360)
PANEL_B_EXAMPLE_HEADER_X = 0.5
PANEL_B_SHARED_XLABEL = "Norm. path progression (switched segment only)"
PANEL_B_SHARED_XLABEL_POSITION = (0.515, -0.035)
PANEL_B_SWAP_EXAMPLES = (
    ("L12", "20240421", "v1", 53, "left_to_center"),
    ("L12", "20240421", "v1", 270, "center_to_left"),
    ("L19", "20250930", "v1", 66, "right_to_center"),
    ("L19", "20250930", "v1", 31, "center_to_right"),
)
PANEL_C_MULTIPLICATIVE_VS_INDEPENDENT_HISTOGRAM_AXIS_BOUNDS = (
    0.070,
    0.290,
    0.400,
    0.610,
)
PANEL_C_MULTIPLICATIVE_VS_ADDITIVE_HISTOGRAM_AXIS_BOUNDS = (
    0.560,
    0.290,
    0.400,
    0.610,
)
PANEL_C_HISTOGRAM_TITLES = (
    "Multiplicative vs. independent",
    "Multiplicative vs. additive",
)
PANEL_C_SHARED_XLABEL = _figure_2.DELTA_LOG_LIKELIHOOD_AXIS_LABEL.replace(
    "\n",
    " ",
)
PANEL_C_SHARED_XLABEL_POSITION = (0.515, 0.084)
PANEL_C_BOTTOM_ANNOTATION_Y = 0.14
PANEL_C_DELTA_JOIN_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "trajectory",
    "unit",
)

PANEL_B_ADDITIVE_MODEL_NAME = "visual_additive_delta"
PANEL_B_ADDITIVE_MODEL_COLOR = "#D95F02"
PANEL_B_ADDITIVE_MODEL_LABEL = "Additive"
PANEL_C_MODEL_LABELS = {
    **_figure_2.PANEL_C_SWAP_MODEL_LABELS_2_3,
    _figure_2.PANEL_C_SWAP_MODEL_NAME: "Multiplicative",
}
PANEL_B_MODEL_COLORS = {
    **_figure_2.PANEL_C_SWAP_MODEL_COLORS_2_3,
    PANEL_B_ADDITIVE_MODEL_NAME: PANEL_B_ADDITIVE_MODEL_COLOR,
}
PANEL_B_MODEL_LABELS = {
    **PANEL_C_MODEL_LABELS,
    PANEL_B_ADDITIVE_MODEL_NAME: PANEL_B_ADDITIVE_MODEL_LABEL,
}
PANEL_B_EMPIRICAL_COLOR = "black"
PANEL_B_EMPIRICAL_LINEWIDTH = 1.15
PANEL_B_MODEL_LINESTYLES = {
    "visual": (0.0, (1.2, 1.1)),
    _figure_2.PANEL_C_SWAP_MODEL_NAME: "solid",
    PANEL_B_ADDITIVE_MODEL_NAME: (0.0, (3.2, 1.4)),
}

INDEPENDENT_MODEL_LABEL = "Independent\nmodel"
MULTIPLICATIVE_MODEL_LABEL = "Multiplicative\nmodel"
ADDITIVE_MODEL_LABEL = "Additive\nmodel"
ADDITIVE_COMPONENT_LABEL = "Stimulus-specific\nadditive component"
PANEL_A_STIMULUS_SWAP_PREDICTION_LABEL = "Stimulus-swap\nprediction"
PANEL_A_MULTIPLICATION_SYMBOL = "\N{MULTIPLICATION SIGN}"
PANEL_A_ADDITIVE_FIELD_RATE_GAIN = (
    _figure_2.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN
    - _figure_2.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN
)
PANEL_A_RATE_SCALE_BOUNDS = (0.735, 0.415, 0.095, 0.012)
PANEL_A_RATE_SCALE_LABEL = "FR"
PANEL_A_RATE_SCALE_COLOR = "0.45"
PANEL_A_RATE_SCALE_OUTLINE_COLOR = "0.55"
PANEL_A_RATE_SCALE_OUTLINE_LINEWIDTH = 0.25
_SOURCE_SCAFFOLD_MODEL_LABEL = "Dark scaffold\nmodel"
_SOURCE_CUE_SWAP_PREDICTION_LABEL = "Cue-swap\nprediction"


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the Figure 3 output path for a supported format."""
    return _figure_2.build_output_path(output_dir, output_name, output_format)


def load_figure_3_panel_data(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    dark_epoch: str | None,
) -> dict[str, Any]:
    """Load the multiplicative and additive data used by Figure 3."""
    dataset_ids = tuple(datasets)
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    example_loader_kwargs = {
        "data_root": data_root,
        "datasets": dataset_ids,
        "region": quant_region,
        "dark_epoch": dark_epoch,
        "example_count": len(PANEL_B_SWAP_EXAMPLES),
        "requested_examples": PANEL_B_SWAP_EXAMPLES,
    }
    multiplicative_examples = load_panel_h_swap_examples(
        **example_loader_kwargs,
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
    )
    additive_examples = load_panel_h_swap_examples(
        **example_loader_kwargs,
        model_name=PANEL_B_ADDITIVE_MODEL_NAME,
    )
    multiplicative_delta_table = load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=quant_region,
        dark_epoch=dark_epoch,
        min_movement_firing_rate_hz=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        min_tuning_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        model_name=_figure_2.PANEL_C_SWAP_MODEL_NAME,
    )
    additive_delta_table = load_panel_h_swap_delta_table(
        data_root=data_root,
        datasets=dataset_ids,
        region=quant_region,
        dark_epoch=dark_epoch,
        light_epoch_pairs=(
            (PANEL_H_TRAIN_LIGHT_EPOCH, PANEL_H_HELDOUT_LIGHT_EPOCH),
        ),
        min_movement_firing_rate_hz=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        min_tuning_stability_correlation=(
            _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        model_name=PANEL_B_ADDITIVE_MODEL_NAME,
    )
    return {
        "swap_delta": multiplicative_delta_table,
        "swap_additive_delta": additive_delta_table,
        "swap_examples": _merge_additive_predictions(
            multiplicative_examples,
            additive_examples,
        ),
    }


def _swap_example_key(example: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    """Return the stable identity fields for one stimulus-swap example."""
    return (
        str(example["animal_name"]),
        str(example["date"]),
        str(example["region"]),
        int(example["unit_id"]),
        str(example["trajectory"]),
    )


def _merge_additive_predictions(
    swap_examples: Sequence[Mapping[str, Any]],
    additive_examples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add matched full-additive curves without changing example selection."""
    additive_by_key = {
        _swap_example_key(example): example for example in additive_examples
    }
    merged_examples: list[dict[str, Any]] = []
    for example in swap_examples:
        example_key = _swap_example_key(example)
        if example_key not in additive_by_key:
            raise ValueError(
                "Missing full-additive prediction for stimulus-swap example "
                f"{example_key!r}."
            )
        additive_example = additive_by_key[example_key]
        additive_models = additive_example.get("models", {})
        if PANEL_B_ADDITIVE_MODEL_NAME not in additive_models:
            raise KeyError(
                "Additive stimulus-swap example is missing model "
                f"{PANEL_B_ADDITIVE_MODEL_NAME!r}."
            )
        merged_examples.append(
            {
                **example,
                "models": {
                    **example.get("models", {}),
                    PANEL_B_ADDITIVE_MODEL_NAME: additive_models[
                        PANEL_B_ADDITIVE_MODEL_NAME
                    ],
                },
            }
        )
    return merged_examples


def _rename_scaffold_model(source_axis: Any, model_label: str) -> None:
    """Rename the one scaffold-model label drawn on a source axis."""
    labels = [
        text
        for text in source_axis.texts
        if text.get_text() == _SOURCE_SCAFFOLD_MODEL_LABEL
    ]
    if len(labels) != 1:
        raise RuntimeError(
            "Expected one dark-scaffold model label in the architecture schematic."
        )
    labels[0].set_text(model_label)


def _draw_quiet_rate_scale(source_axis: Any) -> None:
    """Replace the inherited rate colorbar with a compact, muted scale."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import PowerNorm

    inherited_axes = [
        child
        for child in source_axis.child_axes
        if child.get_label() == _figure_2.PANEL_D2_RATE_COLORBAR_AXIS_LABEL
    ]
    if len(inherited_axes) != 1:
        raise RuntimeError("Expected one inherited field-rate colorbar axis.")
    inherited_axes[0].remove()

    rate_axis = source_axis.inset_axes(PANEL_A_RATE_SCALE_BOUNDS)
    rate_axis.set_label(_figure_2.PANEL_D2_RATE_COLORBAR_AXIS_LABEL)
    rate_axis.set_zorder(0)
    mappable = ScalarMappable(
        norm=PowerNorm(
            gamma=_figure_2.PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA,
            vmin=0.0,
            vmax=1.0,
        ),
        cmap=_figure_2.PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP,
    )
    colorbar = source_axis.figure.colorbar(
        mappable,
        cax=rate_axis,
        orientation="horizontal",
        ticks=(),
    )
    colorbar.ax.set_title(
        PANEL_A_RATE_SCALE_LABEL,
        fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
        color=PANEL_A_RATE_SCALE_COLOR,
        pad=0.5,
    )
    colorbar.outline.set_edgecolor(PANEL_A_RATE_SCALE_OUTLINE_COLOR)
    colorbar.outline.set_linewidth(PANEL_A_RATE_SCALE_OUTLINE_LINEWIDTH)


def _configure_multiplicative_model_row(source_axis: Any) -> None:
    """Apply Figure 3 labels and operators to the multiplicative row."""
    _rename_scaffold_model(source_axis, MULTIPLICATIVE_MODEL_LABEL)

    prediction_labels = [
        text
        for text in source_axis.texts
        if text.get_text() == _SOURCE_CUE_SWAP_PREDICTION_LABEL
    ]
    if len(prediction_labels) != 1:
        raise RuntimeError("Expected one cue-swap prediction label.")
    prediction_labels[0].set_text(PANEL_A_STIMULUS_SWAP_PREDICTION_LABEL)

    multiplication_operators = [
        text
        for text in source_axis.texts
        if text.get_text() == "+"
        and math.isclose(
            float(text.get_position()[1]),
            PANEL_A_SOURCE_SCAFFOLD_ROW_Y,
        )
    ]
    if len(multiplication_operators) != 1:
        raise RuntimeError("Expected one multiplicative-row plus sign.")
    multiplication_operators[0].set_text(PANEL_A_MULTIPLICATION_SYMBOL)
    _draw_quiet_rate_scale(source_axis)


def _show_only_scaffold_model_row(source_axis: Any, model_label: str) -> None:
    """Hide the duplicate headers and independent row on a source axis."""
    track_axes = [
        child
        for child in source_axis.child_axes
        if child.get_label() == "inset_axes"
    ]
    if len(track_axes) != 8:
        raise RuntimeError(
            "Expected eight architecture inset axes before selecting a scaffold row."
        )
    shared_track_axes = set(track_axes[-4:])
    for child in source_axis.child_axes:
        child.set_visible(child in shared_track_axes)

    keep_text = {
        _SOURCE_SCAFFOLD_MODEL_LABEL,
        _figure_2.PANEL_D2_SEGMENT_MODULATION_LABEL,
        "+",
        "=",
    }
    for text in source_axis.texts:
        text_value = text.get_text()
        is_shared_arrow = text_value == "" and math.isclose(
            float(text.get_position()[1]),
            PANEL_A_SOURCE_SCAFFOLD_ROW_Y,
        )
        text.set_visible(text_value in keep_text or is_shared_arrow)
    _rename_scaffold_model(source_axis, model_label)


def _copy_place_fields(
    source_axis: Any,
    target_axis: Any,
    *,
    target_arm: str,
) -> None:
    """Copy one field strip to an arm of another W-track axis."""
    from matplotlib.patches import Ellipse

    _outline, _points, dims = _figure_2.get_w_track_geometry()
    arm_centers = {
        "left_arm": (dims["x0"] + dims["x1"]) / 2.0,
        "right_arm": (dims["x4"] + dims["x5"]) / 2.0,
    }
    if target_arm not in arm_centers:
        raise ValueError(f"Unknown target W-track arm {target_arm!r}.")
    target_x = arm_centers[target_arm]

    for source_patch in source_axis.patches:
        if type(source_patch) is not Ellipse:
            continue
        target_axis.add_patch(
            Ellipse(
                (target_x, float(source_patch.center[1])),
                float(source_patch.width),
                float(source_patch.height),
                angle=float(source_patch.angle),
                facecolor=source_patch.get_facecolor(),
                edgecolor="none",
                linewidth=float(source_patch.get_linewidth()),
                alpha=source_patch.get_alpha(),
                zorder=source_patch.get_zorder(),
            )
        )


def _configure_additive_model_row(source_axis: Any) -> None:
    """Replace the gain icon with the residual field in the additive model."""
    track_axes = [
        child
        for child in source_axis.child_axes
        if child.get_visible() and child.get_label() == "inset_axes"
    ]
    if len(track_axes) != 4:
        raise RuntimeError(
            "Expected four visible W-track axes in the additive model row."
        )

    additive_dark_axis = track_axes[0]
    additive_component_axis = track_axes[1]
    additive_component_axis.clear()
    additive_component_axis.set_zorder(0)
    additive_component_axis.patch.set_visible(False)
    _figure_2._draw_panel_h_track(
        additive_component_axis,
        track_kind="shared_light",
        show_labels=True,
        trajectory_name="center_to_left",
        stimulus_layout="stim1",
        label_fontsize=_figure_2_base.PANEL_D_SCHEMATIC_LABEL_FONTSIZE,
        label_colors=_figure_2_base.PANEL_B_VISUAL_ICON_COLORS,
        show_place_field_blob=True,
        place_field_colors=(
            _figure_2.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_COLORS
        ),
        place_field_blob_size_scale=(
            _figure_2_base.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE
        ),
    )
    _figure_2._remove_panel_d2_stimulus_labels(additive_component_axis)
    _figure_2._apply_panel_d2_place_field_rate_gain(
        additive_component_axis,
        PANEL_A_ADDITIVE_FIELD_RATE_GAIN,
    )
    _figure_2._set_panel_d2_place_field_alpha(
        additive_component_axis,
        _figure_2.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA,
    )
    _figure_2._draw_panel_d2_segment_arm_side_outlines(
        additive_component_axis
    )

    additive_light_axis = track_axes[2]
    cue_swap_prediction_axis = track_axes[3]
    from matplotlib.patches import Ellipse

    for patch in tuple(cue_swap_prediction_axis.patches):
        if type(patch) is Ellipse:
            patch.remove()
    _copy_place_fields(
        additive_dark_axis,
        cue_swap_prediction_axis,
        target_arm="left_arm",
    )
    _copy_place_fields(
        additive_component_axis,
        cue_swap_prediction_axis,
        target_arm="right_arm",
    )

    component_labels = [
        text
        for text in source_axis.texts
        if text.get_visible()
        and text.get_text() == _figure_2.PANEL_D2_SEGMENT_MODULATION_LABEL
    ]
    if len(component_labels) != 1:
        raise RuntimeError(
            "Expected one component label in the additive model row."
        )
    component_labels[0].set_text(ADDITIVE_COMPONENT_LABEL)


def plot_panel_a_three_model_architecture(ax: Any) -> None:
    """Draw independent, multiplicative, and additive model schematics."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.patch.set_visible(False)

    upper_source_axis = ax.inset_axes(
        (
            0.0,
            PANEL_A_UPPER_SOURCE_AXIS_BOTTOM,
            1.0,
            PANEL_A_SOURCE_AXIS_HEIGHT,
        )
    )
    _figure_2.plot_panel_d2_architecture_panel(upper_source_axis)
    _configure_multiplicative_model_row(upper_source_axis)

    lower_source_axis = ax.inset_axes(
        (0.0, 0.0, 1.0, PANEL_A_SOURCE_AXIS_HEIGHT)
    )
    _figure_2.plot_panel_d2_architecture_panel(lower_source_axis)
    _show_only_scaffold_model_row(lower_source_axis, ADDITIVE_MODEL_LABEL)
    _configure_additive_model_row(lower_source_axis)


def _plot_panel_b_additive_prediction(
    ax: Any,
    swap_example: Mapping[str, Any] | None,
    *,
    model_colors: Mapping[str, str] | None,
    model_labels: Mapping[str, str] | None,
) -> None:
    """Overlay the full-additive prediction and include it in the y scale."""
    if swap_example is None:
        return

    import numpy as np

    models = swap_example.get("models", {})
    if PANEL_B_ADDITIVE_MODEL_NAME not in models:
        raise KeyError(
            "Stimulus-swap example is missing additive prediction "
            f"{PANEL_B_ADDITIVE_MODEL_NAME!r}."
        )
    start = float(swap_example["segment_start"])
    end = float(swap_example["segment_end"])
    tp_grid = np.asarray(swap_example["tp_grid"], dtype=float)
    additive_rate = np.asarray(
        models[PANEL_B_ADDITIVE_MODEL_NAME],
        dtype=float,
    )
    if additive_rate.shape != tp_grid.shape:
        raise ValueError(
            "Additive prediction and task-progression grid must have matching "
            f"shapes. Got {additive_rate.shape} and {tp_grid.shape}."
        )
    grid_mask = (tp_grid >= start) & (tp_grid <= end)
    ax.plot(
        tp_grid[grid_mask],
        additive_rate[grid_mask],
        color=_figure_2._panel_model_color(
            PANEL_B_ADDITIVE_MODEL_NAME,
            model_colors,
        ),
        linewidth=0.8,
        label=_figure_2._panel_model_label(
            PANEL_B_ADDITIVE_MODEL_NAME,
            model_labels,
        ),
        zorder=3,
    )

    finite_values = [
        np.asarray(line.get_ydata(), dtype=float)
        for line in ax.lines
        if np.asarray(line.get_ydata()).size
    ]
    finite_values = [
        values[np.isfinite(values)] for values in finite_values if values.size
    ]
    y_max = 1.0
    if finite_values:
        y_max = max(
            y_max,
            float(np.ceil(np.nanmax(np.concatenate(finite_values)))),
        )
    ax.set_ylim(0.0, y_max)
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])


def _style_panel_b_comparison_lines(
    ax: Any,
    *,
    model_name: str,
    model_labels: Mapping[str, str] | None,
) -> None:
    """Distinguish empirical samples and the three prediction classes."""
    lines_by_label = {line.get_label(): line for line in ax.lines}
    empirical_line = lines_by_label.get("Empirical")
    if empirical_line is not None:
        empirical_line.set_color(PANEL_B_EMPIRICAL_COLOR)
        empirical_line.set_linewidth(PANEL_B_EMPIRICAL_LINEWIDTH)
        empirical_line.set_linestyle("solid")
        empirical_line.set_marker("")
        empirical_line.set_zorder(6)

    for plotted_model_name in (
        "visual",
        model_name,
        PANEL_B_ADDITIVE_MODEL_NAME,
    ):
        line_label = _figure_2._panel_model_label(
            plotted_model_name,
            model_labels,
        )
        prediction_line = lines_by_label.get(line_label)
        if prediction_line is None:
            continue
        prediction_line.set_linestyle(
            PANEL_B_MODEL_LINESTYLES[plotted_model_name]
        )
        prediction_line.set_linewidth(0.8)
        prediction_line.set_marker("")
        prediction_line.set_zorder(3)


def _add_panel_b_trace_legend(
    ax: Any,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None,
    model_labels: Mapping[str, str] | None,
) -> Any:
    """Add the empirical, independent, multiplicative, and additive key."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0.0],
            [0.0],
            color=PANEL_B_EMPIRICAL_COLOR,
            linewidth=PANEL_B_EMPIRICAL_LINEWIDTH,
            linestyle="solid",
            label="Empirical",
        ),
        Line2D(
            [0.0],
            [0.0],
            color=_figure_2._panel_model_color("visual", model_colors),
            linewidth=0.8,
            linestyle=PANEL_B_MODEL_LINESTYLES["visual"],
            label="Independent",
        ),
        Line2D(
            [0.0],
            [0.0],
            color=_figure_2._panel_model_color(model_name, model_colors),
            linewidth=0.8,
            linestyle=PANEL_B_MODEL_LINESTYLES[model_name],
            label=_figure_2._panel_model_label(model_name, model_labels),
        ),
        Line2D(
            [0.0],
            [0.0],
            color=_figure_2._panel_model_color(
                PANEL_B_ADDITIVE_MODEL_NAME,
                model_colors,
            ),
            linewidth=0.8,
            linestyle=PANEL_B_MODEL_LINESTYLES[
                PANEL_B_ADDITIVE_MODEL_NAME
            ],
            label=_figure_2._panel_model_label(
                PANEL_B_ADDITIVE_MODEL_NAME,
                model_labels,
            ),
        ),
    ]
    return ax.legend(
        handles=handles,
        frameon=False,
        handlelength=0.9,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=4,
        fontsize=4.8,
    )


def plot_panel_b_swap_examples_panel(
    ax: Any,
    swap_examples: dict[str, Any] | Sequence[dict[str, Any]] | None,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot four stimulus-swap examples across a full-width row."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    examples = (
        list(swap_examples.values())
        if isinstance(swap_examples, dict)
        else list(swap_examples or [])
    )

    for example_index, bounds in enumerate(PANEL_B_EXAMPLE_SLOT_BOUNDS):
        example_axis = ax.inset_axes(bounds)
        example = examples[example_index] if example_index < len(examples) else None
        existing_patches = {id(patch) for patch in example_axis.patches}
        _figure_2._plot_panel_h_switched_segment_example(
            example_axis,
            example,
            model_name=model_name,
            model_colors=model_colors,
            model_labels=model_labels,
            example_label=f"Example {example_index + 1}",
            show_xlabel=False,
            show_ylabel=example_index == 0,
            show_legend=False,
            show_xticklabels=True,
            icon_bounds=PANEL_B_EXAMPLE_ICON_BOUNDS,
            legend_loc="center left",
            legend_bbox_to_anchor=None,
        )
        for patch in tuple(example_axis.patches):
            if id(patch) not in existing_patches:
                patch.remove()
        _plot_panel_b_additive_prediction(
            example_axis,
            example,
            model_colors=model_colors,
            model_labels=model_labels,
        )
        _style_panel_b_comparison_lines(
            example_axis,
            model_name=model_name,
            model_labels=model_labels,
        )
        example_axis.tick_params(labelsize=4.3)
        for text in tuple(example_axis.texts):
            if not text.get_text().startswith("ΔLL="):
                continue
            text.remove()
        example_axis.title.set_text(f"Ex. {example_index + 1}")
        example_axis.title.set_x(PANEL_B_EXAMPLE_HEADER_X)
        _figure_2._set_nested_legend_fontsize(example_axis, 3.9)
        _figure_2._replace_nested_text(
            example_axis,
            "Norm. path progression",
            "Norm.\npath progression",
            fontsize=_figure_2.MIN_PUBLICATION_FONTSIZE_PT,
        )

    ax.text(
        *PANEL_B_SHARED_XLABEL_POSITION,
        PANEL_B_SHARED_XLABEL,
        ha="center",
        va="bottom",
        fontsize=7.0,
        transform=ax.transAxes,
        clip_on=False,
    )

    legend_axis = ax.inset_axes(PANEL_B_TRACE_LEGEND_SLOT_BOUNDS)
    legend_axis.axis("off")
    _add_panel_b_trace_legend(
        legend_axis,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )


def _build_panel_c_delta_tables(
    multiplicative_delta_table: Any,
    additive_delta_table: Any,
) -> tuple[Any, Any]:
    """Return matched multiplicative-vs-independent and -additive rows."""
    import numpy as np
    import pandas as pd

    def _prepare_delta_table(table: Any, label: str) -> Any:
        heldout_table = _figure_2._filter_panel_h_heldout_delta(table)
        required_columns = {
            *PANEL_C_DELTA_JOIN_COLUMNS,
            "delta_ll_bits_per_spike",
        }
        if heldout_table is None:
            heldout_table = pd.DataFrame(columns=sorted(required_columns))
        missing_columns = sorted(
            required_columns.difference(heldout_table.columns)
        )
        if missing_columns:
            raise KeyError(
                f"{label} delta table is missing columns: "
                + ", ".join(missing_columns)
            )
        prepared = heldout_table[
            [*PANEL_C_DELTA_JOIN_COLUMNS, "delta_ll_bits_per_spike"]
        ].copy()
        prepared["delta_ll_bits_per_spike"] = pd.to_numeric(
            prepared["delta_ll_bits_per_spike"],
            errors="coerce",
        )
        return prepared[
            np.isfinite(
                prepared["delta_ll_bits_per_spike"].to_numpy(dtype=float)
            )
        ].copy()

    multiplicative = _prepare_delta_table(
        multiplicative_delta_table,
        "Multiplicative",
    ).rename(
        columns={
            "delta_ll_bits_per_spike": (
                "multiplicative_minus_independent_bits_per_spike"
            )
        }
    )
    additive = _prepare_delta_table(
        additive_delta_table,
        "Additive",
    ).rename(
        columns={
            "delta_ll_bits_per_spike": (
                "additive_minus_independent_bits_per_spike"
            )
        }
    )
    paired = multiplicative.merge(
        additive,
        on=list(PANEL_C_DELTA_JOIN_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    trajectory_names = {
        str(value) for value in _figure_2.PANEL_H_DELTA_TRAJECTORIES
    }
    paired = paired[
        paired["trajectory"].astype(str).isin(trajectory_names)
    ].copy()
    unit_columns = [
        column
        for column in PANEL_C_DELTA_JOIN_COLUMNS
        if column != "trajectory"
    ]
    trajectory_count = paired.groupby(
        unit_columns,
        dropna=False,
    )["trajectory"].transform("nunique")
    paired = paired[trajectory_count >= len(trajectory_names)].copy()
    common_columns = [*PANEL_C_DELTA_JOIN_COLUMNS]
    multiplicative_vs_independent = paired[
        [
            *common_columns,
            "multiplicative_minus_independent_bits_per_spike",
        ]
    ].rename(
        columns={
            "multiplicative_minus_independent_bits_per_spike": (
                "delta_ll_bits_per_spike"
            )
        }
    )
    multiplicative_vs_additive = paired[common_columns].copy()
    multiplicative_vs_additive["delta_ll_bits_per_spike"] = (
        paired["multiplicative_minus_independent_bits_per_spike"]
        - paired["additive_minus_independent_bits_per_spike"]
    )
    for comparison_table in (
        multiplicative_vs_independent,
        multiplicative_vs_additive,
    ):
        comparison_table["model_name"] = _figure_2.PANEL_C_SWAP_MODEL_NAME
    return multiplicative_vs_independent, multiplicative_vs_additive


def plot_panel_c_swap_histogram_panel(
    ax: Any,
    swap_delta_table: Any,
    swap_additive_delta_table: Any,
    *,
    model_name: str,
    model_colors: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
) -> None:
    """Plot multiplicative comparisons with independent and additive models."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    (
        multiplicative_vs_independent_table,
        multiplicative_vs_additive_table,
    ) = _build_panel_c_delta_tables(
        swap_delta_table,
        swap_additive_delta_table,
    )
    multiplicative_vs_independent_axis = ax.inset_axes(
        PANEL_C_MULTIPLICATIVE_VS_INDEPENDENT_HISTOGRAM_AXIS_BOUNDS
    )
    multiplicative_vs_additive_axis = ax.inset_axes(
        PANEL_C_MULTIPLICATIVE_VS_ADDITIVE_HISTOGRAM_AXIS_BOUNDS
    )
    _figure_2.plot_panel_d_mean_swap_delta_axis(
        multiplicative_vs_independent_axis,
        multiplicative_vs_independent_table,
        model_name=model_name,
        model_colors=model_colors,
        model_labels=model_labels,
    )
    multiplicative_vs_additive_colors = dict(model_colors or {})
    multiplicative_vs_additive_colors["visual"] = (
        PANEL_B_ADDITIVE_MODEL_COLOR
    )
    _figure_2.plot_panel_d_mean_swap_delta_axis(
        multiplicative_vs_additive_axis,
        multiplicative_vs_additive_table,
        model_name=model_name,
        model_colors=multiplicative_vs_additive_colors,
        model_labels=model_labels,
    )
    count_labels = []
    for histogram_axis, comparison_label in (
        (multiplicative_vs_independent_axis, "Indep.\nbetter"),
        (multiplicative_vs_additive_axis, "Additive\nbetter"),
    ):
        for text in tuple(histogram_axis.texts):
            if text.get_text() == "Indep. better":
                text.set_text(comparison_label)
            elif text.get_text().startswith("n = "):
                cell_count_label = text.get_text().splitlines()[0]
                text.set_text(cell_count_label)
                text.set_position(
                    (text.get_position()[0], PANEL_C_BOTTOM_ANNOTATION_Y)
                )
                count_labels.append(cell_count_label)
            elif text.get_text().endswith(">0"):
                text.set_position(
                    (text.get_position()[0], PANEL_C_BOTTOM_ANNOTATION_Y)
                )
    if count_labels:
        if len(set(count_labels)) != 1:
            raise ValueError(
                "Panel C comparison histogram cohorts do not match."
            )
    multiplicative_vs_independent_axis.set_title(
        PANEL_C_HISTOGRAM_TITLES[0],
        fontsize=6.0,
        pad=1.5,
    )
    multiplicative_vs_additive_axis.set_title(
        PANEL_C_HISTOGRAM_TITLES[1],
        fontsize=6.0,
        pad=1.5,
    )
    multiplicative_vs_independent_axis.set_xlabel("")
    multiplicative_vs_additive_axis.set_xlabel("")
    multiplicative_vs_additive_axis.set_ylabel("")
    ax.text(
        *PANEL_C_SHARED_XLABEL_POSITION,
        PANEL_C_SHARED_XLABEL,
        ha="center",
        va="bottom",
        fontsize=7.0,
        transform=ax.transAxes,
        clip_on=False,
    )


def make_figure_3(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    dark_epoch: str | None,
    dpi: int,
) -> Path:
    """Build and save Figure 3."""
    import matplotlib.pyplot as plt

    panel_data = load_figure_3_panel_data(
        data_root=data_root,
        datasets=datasets,
        regions=regions,
        dark_epoch=dark_epoch,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        **_figure_2.CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS
    )
    outer_grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=(
            TOP_ROW_HEIGHT_MM,
            PANEL_B_ROW_HEIGHT_MM,
            PANEL_C_ROW_HEIGHT_MM,
        ),
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    panel_b_axis = fig.add_subplot(outer_grid[1, 0])
    panel_c_axis = fig.add_subplot(outer_grid[2, 0])

    plot_panel_a_three_model_architecture(panel_a_axis)
    panel_b_model_kwargs = {
        "model_name": _figure_2.PANEL_C_SWAP_MODEL_NAME,
        "model_colors": PANEL_B_MODEL_COLORS,
        "model_labels": PANEL_B_MODEL_LABELS,
    }
    panel_c_model_kwargs = {
        "model_name": _figure_2.PANEL_C_SWAP_MODEL_NAME,
        "model_colors": _figure_2.PANEL_C_SWAP_MODEL_COLORS_2_3,
        "model_labels": PANEL_C_MODEL_LABELS,
    }
    plot_panel_b_swap_examples_panel(
        panel_b_axis,
        panel_data["swap_examples"],
        **panel_b_model_kwargs,
    )
    plot_panel_c_swap_histogram_panel(
        panel_c_axis,
        panel_data["swap_delta"],
        panel_data["swap_additive_delta"],
        **panel_c_model_kwargs,
    )

    label_axis(
        panel_a_axis,
        "A",
        x=PANEL_A_B_LABEL_X,
        y=_figure_2.PANEL_B_LABEL_Y,
        va="baseline",
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    panel_a_label = panel_a_axis.texts[-1]
    label_axis(
        panel_b_axis,
        "B",
        x=PANEL_B_LABEL_X,
        y=PANEL_ROW_LABEL_Y,
        va="baseline",
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    panel_b_label = panel_b_axis.texts[-1]
    label_axis(
        panel_c_axis,
        "C",
        x=PANEL_C_LABEL_X,
        y=PANEL_ROW_LABEL_Y,
        va="baseline",
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    panel_c_label = panel_c_axis.texts[-1]
    panel_a_title = panel_a_axis.set_title(
        PANEL_TITLES[0],
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    panel_b_axis.set_title(
        PANEL_TITLES[1],
        fontsize=8,
        pad=PANEL_BC_TITLE_PAD,
    )

    _figure_2._raise_text_to_minimum_fontsize(
        fig,
        _figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _figure_2._align_text_to_reference_display_x(panel_b_label, panel_a_label)
    _figure_2._align_text_to_reference_display_x(panel_c_label, panel_a_label)
    _figure_2._align_text_tops_to_reference_display_y(
        fig,
        (panel_a_title, panel_a_label),
    )

    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 3 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 3 generation."""
    parser = argparse.ArgumentParser(description="Generate Figure 3.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Base directory containing analysis outputs. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
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
        type=_figure_2.parse_dataset_id,
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
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 3 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_figure_3(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
