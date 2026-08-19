"""Tests for the three-model Figure 3."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from v1ca1.paper_figures import figure_2_old as figure_2_module
from v1ca1.paper_figures import figure_3 as figure_3_module


def _relative_bounds(parent_ax: Any, child_ax: Any) -> tuple[float, ...]:
    """Return one child axis's bounds in its parent axis coordinates."""
    parent_bounds = parent_ax.get_position().bounds
    child_bounds = child_ax.get_position().bounds
    parent_x, parent_y, parent_width, parent_height = parent_bounds
    child_x, child_y, child_width, child_height = child_bounds
    return (
        (child_x - parent_x) / parent_width,
        (child_y - parent_y) / parent_height,
        child_width / parent_width,
        child_height / parent_height,
    )


def _axis_artist_signature(ax: Any) -> tuple[Any, ...]:
    """Return a compact signature for comparing duplicated schematic axes."""
    return (
        len(ax.lines),
        len(ax.patches),
        len(ax.collections),
        len(ax.images),
        tuple(text.get_text() for text in ax.texts),
    )


def _descendant_axes(ax: Any) -> list[Any]:
    """Return all recursively nested axes below one parent axis."""
    descendants: list[Any] = []
    for child_ax in ax.child_axes:
        descendants.append(child_ax)
        descendants.extend(_descendant_axes(child_ax))
    return descendants


def _text_position_in_axes(ax: Any, text: Any) -> tuple[float, float]:
    """Return a text artist's anchor position in another axis's coordinates."""
    display_position = text.get_transform().transform(text.get_position())
    x_position, y_position = ax.transAxes.inverted().transform(display_position)
    return float(x_position), float(y_position)


def _shared_xlabel_tick_gap_points(
    fig: Any,
    panel_ax: Any,
    shared_xlabel: str,
) -> float:
    """Return the rendered gap between x ticks and a shared label in points."""
    renderer = fig.canvas.get_renderer()
    label = next(
        text for text in panel_ax.texts if text.get_text() == shared_xlabel
    )
    visible_tick_labels = [
        tick_label
        for child_ax in panel_ax.child_axes
        for tick_label in child_ax.get_xticklabels()
        if tick_label.get_visible() and tick_label.get_text()
    ]
    assert visible_tick_labels
    label_top = label.get_window_extent(renderer).y1
    tick_bottom = min(
        tick_label.get_window_extent(renderer).y0
        for tick_label in visible_tick_labels
    )
    return float((tick_bottom - label_top) * 72.0 / fig.dpi)


def _make_panel_b_example(example_index: int) -> dict[str, object]:
    """Return one synthetic example containing all Figure 3 traces."""
    additive_offset = 8.0 + example_index
    return {
        "delta_ll_bits_per_spike": 0.11 + 0.10 * example_index,
        "segment_start": 0.0,
        "segment_end": 1.0,
        "observed_position": [0.0, 0.5, 1.0],
        "observed_rate_hz": [1.0, 2.0, 1.0],
        "tp_grid": [0.0, 0.5, 1.0],
        "trajectory": "center_to_left",
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "models": {
            "visual": [1.0, 1.5, 1.0],
            figure_2_module.PANEL_C_SWAP_MODEL_NAME: [1.0, 3.0, 1.0],
            figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME: [
                additive_offset,
                additive_offset + 1.0,
                additive_offset,
            ],
        },
    }


def test_defaults_and_output_path_match_figure_3_conventions() -> None:
    args = figure_3_module.parse_arguments([])

    assert figure_3_module.DEFAULT_OUTPUT_NAME == "figure_3"
    assert args.output_name == "figure_3"
    assert args.output_dir == figure_2_module.DEFAULT_OUTPUT_DIR
    assert args.output_format == figure_2_module.DEFAULT_OUTPUT_FORMAT
    assert figure_3_module.DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        figure_2_module.DEFAULT_FIGURE_WIDTH_MM / 2.0
    )
    assert figure_3_module.DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        figure_3_module.TOP_ROW_HEIGHT_MM
        + figure_3_module.PANEL_B_ROW_HEIGHT_MM
        + figure_3_module.PANEL_C_ROW_HEIGHT_MM
    )
    assert figure_3_module.PANEL_B_ROW_HEIGHT_MM == pytest.approx(
        0.9 * figure_2_module.PANEL_D_ROW_HEIGHT_MM
    )
    assert figure_3_module.PANEL_C_ROW_HEIGHT_MM == pytest.approx(
        0.9 * figure_2_module.PANEL_D_ROW_HEIGHT_MM
    )
    assert figure_3_module.PANEL_TITLES == (
        "Three models that relate dark and light activity",
        "Dark and light stimulus-swap prediction comparison",
    )
    assert args.dataset is None
    assert args.region is None
    assert args.dark_epoch is None
    assert args.dpi == 300
    comparison_model = figure_2_module.PANEL_C_SWAP_MODEL_NAME
    assert comparison_model == "task_segment_scalar"
    assert (
        figure_3_module.PANEL_C_MODEL_LABELS[comparison_model]
        == "Multiplicative"
    )
    assert (
        figure_3_module.PANEL_B_MODEL_LABELS[
            figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME
        ]
        == "Additive"
    )
    assert figure_3_module.PANEL_B_SWAP_EXAMPLES == (
        ("L12", "20240421", "v1", 53, "left_to_center"),
        ("L12", "20240421", "v1", 270, "center_to_left"),
        ("L19", "20250930", "v1", 66, "right_to_center"),
        ("L19", "20250930", "v1", 31, "center_to_right"),
    )
    assert figure_2_module.PANEL_C_SWAP_EXAMPLES == (
        ("L15", "20241121", "v1", 27, "center_to_right"),
        ("L19", "20250930", "v1", 4, "center_to_left"),
        ("L15", "20241121", "v1", 146, "center_to_right"),
    )
    assert (
        figure_3_module.PANEL_B_SWAP_EXAMPLES
        != figure_2_module.PANEL_C_SWAP_EXAMPLES
    )
    assert figure_3_module.PANEL_C_HISTOGRAM_TITLES == (
        "Multiplicative vs. independent",
        "Multiplicative vs. additive",
    )
    independent_comparison_bounds = (
        figure_3_module.PANEL_C_MULTIPLICATIVE_VS_INDEPENDENT_HISTOGRAM_AXIS_BOUNDS
    )
    additive_comparison_bounds = (
        figure_3_module.PANEL_C_MULTIPLICATIVE_VS_ADDITIVE_HISTOGRAM_AXIS_BOUNDS
    )
    assert (
        independent_comparison_bounds[0] + independent_comparison_bounds[2]
        < additive_comparison_bounds[0]
    )
    assert "Dark scaffold" not in (
        figure_3_module.PANEL_B_MODEL_LABELS.values()
    )
    assert "Dark scaffold" not in (
        figure_3_module.PANEL_C_MODEL_LABELS.values()
    )
    assert (
        figure_2_module.PANEL_C_SWAP_MODEL_LABELS_2_3[comparison_model]
        == "Dark scaffold"
    )
    assert figure_3_module.build_output_path(
        args.output_dir,
        args.output_name,
        "svg",
    ) == Path("paper_figures/output/figure_3.svg")


def test_canonical_figure_3_is_promoted_and_independent() -> None:
    code = (
        "import importlib.util, sys; "
        "from v1ca1.paper_figures import figure_3; "
        "assert figure_3.DEFAULT_OUTPUT_NAME == 'figure_3'; "
        "assert figure_3.PANEL_TITLES[0] == "
        "'Three models that relate dark and light activity'; "
        "assert not hasattr(figure_3, 'PANEL_BC_SPLIT_LABEL_Y'); "
        "assert 'v1ca1.paper_figures.figure_3_old' not in sys.modules; "
        "assert importlib.util.find_spec("
        "'v1ca1.paper_figures.figure_3_2') is None"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_panel_data_loader_merges_additive_predictions_by_example_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    comparison_model = figure_2_module.PANEL_C_SWAP_MODEL_NAME
    additive_model = figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME
    requested_examples = figure_3_module.PANEL_B_SWAP_EXAMPLES
    swap_delta = object()
    swap_additive_delta = object()

    def make_example(
        example_spec: tuple[str, str, str, int, str],
        example_index: int,
        *,
        model_name: str,
    ) -> dict[str, object]:
        animal_name, date, region, unit_id, trajectory = example_spec
        return {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "unit_id": unit_id,
            "trajectory": trajectory,
            "model_name": model_name,
            "delta_ll_bits_per_spike": 0.1 + example_index,
            "models": {
                "visual": [1.0, 2.0],
                model_name: [10.0 + example_index, 11.0 + example_index],
            },
        }

    comparison_examples = [
        make_example(spec, index, model_name=comparison_model)
        for index, spec in enumerate(requested_examples)
    ]
    additive_examples = [
        make_example(spec, index, model_name=additive_model)
        for index, spec in enumerate(requested_examples)
    ]
    def fake_load_examples(**kwargs: object) -> list[dict[str, object]]:
        calls.setdefault("example_kwargs", []).append(kwargs)
        if kwargs["model_name"] == comparison_model:
            return comparison_examples
        if kwargs["model_name"] == additive_model:
            return list(reversed(additive_examples))
        raise AssertionError(f"Unexpected model {kwargs['model_name']!r}")

    def fake_load_delta(**kwargs: object) -> object:
        calls.setdefault("delta_kwargs", []).append(kwargs)
        if kwargs["model_name"] == comparison_model:
            return swap_delta
        if kwargs["model_name"] == additive_model:
            return swap_additive_delta
        raise AssertionError(f"Unexpected model {kwargs['model_name']!r}")

    monkeypatch.setattr(
        figure_3_module,
        "load_panel_h_swap_examples",
        fake_load_examples,
    )
    monkeypatch.setattr(
        figure_3_module,
        "load_panel_h_swap_delta_table",
        fake_load_delta,
    )
    result = figure_3_module.load_figure_3_panel_data(
        data_root=Path("/analysis"),
        datasets=(("L14", "20240611", "08_r4"),),
        regions=("v1",),
        dark_epoch="08_r4",
    )

    assert result["swap_delta"] is swap_delta
    assert result["swap_additive_delta"] is swap_additive_delta
    assert [
        figure_3_module._swap_example_key(example)
        for example in result["swap_examples"]
    ] == [
        figure_3_module._swap_example_key(example)
        for example in comparison_examples
    ]
    for example_index, example in enumerate(result["swap_examples"]):
        assert example["model_name"] == comparison_model
        assert example["delta_ll_bits_per_spike"] == pytest.approx(
            0.1 + example_index
        )
        assert example["models"][comparison_model] == [
            10.0 + example_index,
            11.0 + example_index,
        ]
        assert example["models"][additive_model] == [
            10.0 + example_index,
            11.0 + example_index,
        ]
    assert calls["example_kwargs"] == [
        {
            "data_root": Path("/analysis"),
            "datasets": (("L14", "20240611", "08_r4"),),
            "region": "v1",
            "dark_epoch": "08_r4",
            "model_name": comparison_model,
            "example_count": len(requested_examples),
            "requested_examples": requested_examples,
        },
        {
            "data_root": Path("/analysis"),
            "datasets": (("L14", "20240611", "08_r4"),),
            "region": "v1",
            "dark_epoch": "08_r4",
            "model_name": additive_model,
            "example_count": len(requested_examples),
            "requested_examples": requested_examples,
        },
    ]
    common_delta_kwargs = {
        "data_root": Path("/analysis"),
        "datasets": (("L14", "20240611", "08_r4"),),
        "region": "v1",
        "dark_epoch": "08_r4",
        "min_movement_firing_rate_hz": (
            figure_2_module.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        "min_tuning_stability_correlation": (
            figure_2_module.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    }
    assert calls["delta_kwargs"] == [
        {
            **common_delta_kwargs,
            "model_name": comparison_model,
        },
        {
            **common_delta_kwargs,
            "light_epoch_pairs": (
                (
                    figure_3_module.PANEL_H_TRAIN_LIGHT_EPOCH,
                    figure_3_module.PANEL_H_HELDOUT_LIGHT_EPOCH,
                ),
            ),
            "model_name": additive_model,
        },
    ]


def test_panel_a_additive_component_completes_the_light_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Ellipse

    original_apply_rate_gain = (
        figure_2_module._apply_panel_d2_place_field_rate_gain
    )

    def record_rate_gain(ax: Any, gain: float) -> None:
        ax._figure_3_rate_gain = float(gain)
        original_apply_rate_gain(ax, gain)

    monkeypatch.setattr(
        figure_2_module,
        "_apply_panel_d2_place_field_rate_gain",
        record_rate_gain,
    )

    fig, ax = plt.subplots()
    figure_3_module.plot_panel_a_three_model_architecture(ax)
    fig.canvas.draw()

    nested_axes = _descendant_axes(ax)
    visible_texts = [
        text
        for nested_ax in nested_axes
        if nested_ax.get_visible()
        for text in nested_ax.texts
        if text.get_visible()
    ]
    labels = {
        text.get_text(): text
        for text in visible_texts
        if text.get_text()
        in {
            "Independent\nmodel",
            "Multiplicative\nmodel",
            "Additive\nmodel",
            "Dark scaffold\nmodel",
        }
    }
    assert set(labels) == {
        "Independent\nmodel",
        "Multiplicative\nmodel",
        "Additive\nmodel",
    }
    independent_y = _text_position_in_axes(
        ax,
        labels["Independent\nmodel"],
    )[1]
    multiplicative_y = _text_position_in_axes(
        ax,
        labels["Multiplicative\nmodel"],
    )[1]
    additive_y = _text_position_in_axes(ax, labels["Additive\nmodel"])[1]
    assert independent_y > multiplicative_y > additive_y
    assert all(text.get_fontweight() == "bold" for text in labels.values())

    rate_colorbar_axes = [
        child_ax
        for child_ax in nested_axes
        if child_ax.get_visible()
        if child_ax.get_label()
        == figure_2_module.PANEL_D2_RATE_COLORBAR_AXIS_LABEL
    ]
    assert len(rate_colorbar_axes) == 1
    rate_colorbar_axis = rate_colorbar_axes[0]
    rate_colorbar_parent = next(
        child_ax
        for child_ax in ax.child_axes
        if rate_colorbar_axis in child_ax.child_axes
    )
    assert _relative_bounds(
        rate_colorbar_parent,
        rate_colorbar_axis,
    ) == pytest.approx(figure_3_module.PANEL_A_RATE_SCALE_BOUNDS)
    assert (
        figure_3_module.PANEL_A_RATE_SCALE_BOUNDS[2]
        < figure_2_module.PANEL_D2_RATE_COLORBAR_BOUNDS[2]
    )
    assert (
        figure_3_module.PANEL_A_RATE_SCALE_BOUNDS[3]
        < figure_2_module.PANEL_D2_RATE_COLORBAR_BOUNDS[3]
    )
    assert rate_colorbar_axis.get_title() == (
        figure_3_module.PANEL_A_RATE_SCALE_LABEL
    )
    assert figure_3_module.PANEL_A_RATE_SCALE_LABEL == "FR"
    assert rate_colorbar_axis.title.get_color() == (
        figure_3_module.PANEL_A_RATE_SCALE_COLOR
    )
    assert not any(
        tick_label.get_text()
        for tick_label in rate_colorbar_axis.get_xticklabels()
    )
    assert rate_colorbar_axis.spines["outline"].get_edgecolor() == (
        pytest.approx(
            to_rgba(figure_3_module.PANEL_A_RATE_SCALE_OUTLINE_COLOR)
        )
    )
    assert rate_colorbar_axis.spines["outline"].get_linewidth() == (
        pytest.approx(
            figure_3_module.PANEL_A_RATE_SCALE_OUTLINE_LINEWIDTH
        )
    )

    schematic_axes = [
        child_ax
        for child_ax in nested_axes
        if child_ax.get_visible()
        and not child_ax.child_axes
        and child_ax not in rate_colorbar_axes
    ]

    def row_axes(row_y: float) -> list[Any]:
        return sorted(
            (
                child_ax
                for child_ax in schematic_axes
                if abs(
                    (
                        _relative_bounds(ax, child_ax)[1]
                        + _relative_bounds(ax, child_ax)[3] / 2.0
                    )
                    - row_y
                )
                < 0.02
            ),
            key=lambda child_ax: _relative_bounds(ax, child_ax)[0],
        )

    multiplicative_axes = row_axes(multiplicative_y)
    additive_axes = row_axes(additive_y)
    assert len(multiplicative_axes) == len(additive_axes) == 4
    for column_index in (0, 2):
        multiplicative_ax = multiplicative_axes[column_index]
        additive_ax = additive_axes[column_index]
        multiplicative_bounds = _relative_bounds(ax, multiplicative_ax)
        additive_bounds = _relative_bounds(ax, additive_ax)
        assert additive_bounds[0] == pytest.approx(multiplicative_bounds[0])
        assert additive_bounds[2:] == pytest.approx(multiplicative_bounds[2:])
        assert _axis_artist_signature(additive_ax) == _axis_artist_signature(
            multiplicative_ax
        )

    additive_dark_ax = additive_axes[0]
    additive_component_ax = additive_axes[1]
    additive_light_ax = additive_axes[2]
    component_bounds = _relative_bounds(ax, additive_component_ax)
    light_bounds = _relative_bounds(ax, additive_light_ax)
    assert component_bounds[2:] == pytest.approx(light_bounds[2:])
    assert additive_component_ax.get_xlim() == pytest.approx(
        additive_light_ax.get_xlim()
    )
    assert additive_component_ax.get_ylim() == pytest.approx(
        additive_light_ax.get_ylim()
    )
    assert _axis_artist_signature(additive_component_ax) == (
        _axis_artist_signature(additive_light_ax)
    )

    component_fields = [
        patch
        for patch in additive_component_ax.patches
        if type(patch) is Ellipse
    ]
    dark_fields = [
        patch for patch in additive_dark_ax.patches if type(patch) is Ellipse
    ]
    light_fields = [
        patch for patch in additive_light_ax.patches if type(patch) is Ellipse
    ]
    assert len(dark_fields) == len(component_fields) == len(light_fields) > 0
    for component_field, light_field in zip(
        component_fields,
        light_fields,
        strict=True,
    ):
        assert component_field.center == pytest.approx(light_field.center)
        assert component_field.width == pytest.approx(light_field.width)
        assert component_field.height == pytest.approx(light_field.height)
        assert component_field.angle == pytest.approx(light_field.angle)

    dark_gain = figure_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN
    light_gain = figure_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN
    component_gain = figure_3_module.PANEL_A_ADDITIVE_FIELD_RATE_GAIN
    assert component_gain == pytest.approx(light_gain - dark_gain)
    assert dark_gain + component_gain == pytest.approx(light_gain)
    assert additive_axes[0]._figure_3_rate_gain == pytest.approx(dark_gain)
    assert additive_component_ax._figure_3_rate_gain == pytest.approx(
        component_gain
    )
    assert additive_light_ax._figure_3_rate_gain == pytest.approx(light_gain)

    _outline, _points, dims = figure_2_module.get_w_track_geometry()
    additive_prediction_ax = additive_axes[3]
    prediction_fields = [
        patch
        for patch in additive_prediction_ax.patches
        if type(patch) is Ellipse
    ]
    left_arm_x = (dims["x0"] + dims["x1"]) / 2.0
    right_arm_x = (dims["x4"] + dims["x5"]) / 2.0
    left_prediction_fields = sorted(
        (
            patch
            for patch in prediction_fields
            if float(patch.center[0]) == pytest.approx(left_arm_x)
        ),
        key=lambda patch: float(patch.center[1]),
    )
    right_prediction_fields = sorted(
        (
            patch
            for patch in prediction_fields
            if float(patch.center[0]) == pytest.approx(right_arm_x)
        ),
        key=lambda patch: float(patch.center[1]),
    )
    assert len(prediction_fields) == 14
    assert len(left_prediction_fields) == len(dark_fields) == 7
    assert len(right_prediction_fields) == len(component_fields) == 7

    def assert_field_copies_match(
        source_fields: list[Any],
        copied_fields: list[Any],
    ) -> None:
        for source_field, copied_field in zip(
            sorted(source_fields, key=lambda patch: float(patch.center[1])),
            copied_fields,
            strict=True,
        ):
            assert copied_field.center[1] == pytest.approx(
                source_field.center[1]
            )
            assert copied_field.width == pytest.approx(source_field.width)
            assert copied_field.height == pytest.approx(source_field.height)
            assert copied_field.angle == pytest.approx(source_field.angle)
            assert copied_field.get_facecolor() == pytest.approx(
                source_field.get_facecolor()
            )
            assert copied_field.get_edgecolor() == pytest.approx(
                source_field.get_edgecolor()
            )
            assert copied_field.get_alpha() == pytest.approx(
                source_field.get_alpha()
            )
            assert copied_field.get_zorder() == pytest.approx(
                source_field.get_zorder()
            )

    assert_field_copies_match(dark_fields, left_prediction_fields)
    assert_field_copies_match(component_fields, right_prediction_fields)
    assert [
        type(patch)
        for patch in additive_prediction_ax.patches
        if type(patch) is not Ellipse
    ] == [
        type(patch)
        for patch in multiplicative_axes[3].patches
        if type(patch) is not Ellipse
    ]
    assert len(additive_prediction_ax.lines) == len(
        multiplicative_axes[3].lines
    )

    field_center_y = dims["y1"] + 1.45
    field_sigma = 0.58
    gamma = figure_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA
    cmap = colormaps[
        figure_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP
    ]
    for component_field in component_fields:
        relative_rate = math.exp(
            -0.5
            * (
                (float(component_field.center[1]) - field_center_y)
                / field_sigma
            )
            ** 2
        )
        expected_color = cmap((component_gain * relative_rate) ** gamma)
        assert component_field.get_facecolor()[:3] == pytest.approx(
            expected_color[:3],
            abs=0.01,
        )

    visible_text_values = [text.get_text() for text in visible_texts]
    assert figure_3_module.ADDITIVE_COMPONENT_LABEL in visible_text_values
    assert visible_text_values.count(
        figure_3_module.PANEL_A_STIMULUS_SWAP_PREDICTION_LABEL
    ) == 1
    assert not any("Cue-swap" in value for value in visible_text_values)
    assert visible_text_values.count(
        figure_2_module.PANEL_D2_SEGMENT_MODULATION_LABEL
    ) == 1

    operator_rows = {
        row_y: sorted(
            text.get_text()
            for text in visible_texts
            if text.get_text()
            in {"+", figure_3_module.PANEL_A_MULTIPLICATION_SYMBOL, "="}
            and abs(_text_position_in_axes(ax, text)[1] - row_y) < 1e-9
        )
        for row_y in (multiplicative_y, additive_y)
    }
    assert operator_rows == {
        multiplicative_y: [
            "=",
            figure_3_module.PANEL_A_MULTIPLICATION_SYMBOL,
        ],
        additive_y: ["+", "="],
    }
    plt.close(fig)


def test_panel_b_swap_examples_span_the_full_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls: dict[str, Any] = {"examples": []}

    def fake_plot_example(
        ax: Any,
        example: object,
        **kwargs: object,
    ) -> None:
        calls["examples"].append((ax, example, kwargs))
        ax.set_title(str(kwargs["example_label"]))
        ax.text(
            0.96,
            0.94,
            f"ΔLL={float(example['delta_ll_bits_per_spike']):.2f}",
        )

    monkeypatch.setattr(
        figure_2_module,
        "_plot_panel_h_switched_segment_example",
        fake_plot_example,
    )

    examples = [
        _make_panel_b_example(index)
        for index in range(len(figure_3_module.PANEL_B_SWAP_EXAMPLES))
    ]
    model_colors = figure_3_module.PANEL_B_MODEL_COLORS
    model_labels = figure_3_module.PANEL_B_MODEL_LABELS
    fig, ax = plt.subplots()
    figure_3_module.plot_panel_b_swap_examples_panel(
        ax,
        examples,
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        model_colors=model_colors,
        model_labels=model_labels,
    )
    fig.canvas.draw()

    example_calls = calls["examples"]
    example_axes = [
        example_axis
        for example_axis, _example, _kwargs in example_calls
    ]
    assert [example for _axis, example, _kwargs in example_calls] == examples
    assert len(example_axes) == 4
    for example_axis, expected_bounds in zip(
        example_axes,
        figure_3_module.PANEL_B_EXAMPLE_SLOT_BOUNDS,
        strict=True,
    ):
        assert _relative_bounds(ax, example_axis) == pytest.approx(expected_bounds)

    example_bounds = [_relative_bounds(ax, child_ax) for child_ax in example_axes]
    assert [bounds[1] for bounds in example_bounds] == pytest.approx(
        [example_bounds[0][1]] * 4
    )
    assert [bounds[2] for bounds in example_bounds] == pytest.approx(
        [example_bounds[0][2]] * 4
    )
    assert [bounds[3] for bounds in example_bounds] == pytest.approx(
        [example_bounds[0][3]] * 4
    )
    assert all(
        left_bounds[0] + left_bounds[2] < right_bounds[0]
        for left_bounds, right_bounds in zip(
            example_bounds[:-1],
            example_bounds[1:],
            strict=True,
        )
    )
    assert example_bounds[0][0] < 0.1
    assert example_bounds[-1][0] + example_bounds[-1][2] > 0.9

    example_kwargs = [kwargs for _axis, _example, kwargs in example_calls]
    assert [kwargs["show_xlabel"] for kwargs in example_kwargs] == [
        False,
        False,
        False,
        False,
    ]
    assert [kwargs["show_ylabel"] for kwargs in example_kwargs] == [
        True,
        False,
        False,
        False,
    ]
    for kwargs in example_kwargs:
        assert kwargs["model_name"] == figure_2_module.PANEL_C_SWAP_MODEL_NAME
        assert kwargs["model_colors"] is model_colors
        assert kwargs["model_labels"] is model_labels
        assert kwargs["show_legend"] is False
        assert kwargs["show_xticklabels"] is True
        assert kwargs["icon_bounds"] == (
            figure_3_module.PANEL_B_EXAMPLE_ICON_BOUNDS
        )
    icon_x, icon_y, icon_width, icon_height = (
        figure_3_module.PANEL_B_EXAMPLE_ICON_BOUNDS
    )
    previous_icon_bounds = (-0.180, 0.300, 0.140, 0.270)
    assert icon_width > previous_icon_bounds[2]
    assert icon_height > previous_icon_bounds[3]
    assert icon_x + icon_width <= 0.0
    assert icon_y >= 0.0
    assert icon_y + icon_height <= 1.0
    for slot_x, slot_y, slot_width, slot_height in example_bounds:
        icon_panel_bounds = (
            slot_x + icon_x * slot_width,
            slot_y + icon_y * slot_height,
            icon_width * slot_width,
            icon_height * slot_height,
        )
        assert icon_panel_bounds[0] >= 0.0
        assert icon_panel_bounds[1] >= 0.0
        assert icon_panel_bounds[0] + icon_panel_bounds[2] <= slot_x
        assert icon_panel_bounds[1] + icon_panel_bounds[3] <= 1.0
    assert [example_axis.get_title() for example_axis in example_axes] == [
        "Ex. 1",
        "Ex. 2",
        "Ex. 3",
        "Ex. 4",
    ]
    shared_xlabels = [
        text
        for text in ax.texts
        if text.get_text() == figure_3_module.PANEL_B_SHARED_XLABEL
    ]
    assert figure_3_module.PANEL_B_SHARED_XLABEL == (
        "Norm. path progression (switched segment only)"
    )
    assert len(shared_xlabels) == 1
    assert shared_xlabels[0].get_position() == pytest.approx(
        figure_3_module.PANEL_B_SHARED_XLABEL_POSITION
    )
    assert shared_xlabels[0].get_position()[1] < 0.020
    renderer = fig.canvas.get_renderer()
    shared_xlabel_top = shared_xlabels[0].get_window_extent(renderer).y1
    tick_label_bottoms = [
        tick_label.get_window_extent(renderer).y0
        for example_axis in example_axes
        for tick_label in example_axis.get_xticklabels()
        if tick_label.get_visible() and tick_label.get_text()
    ]
    assert tick_label_bottoms
    assert min(tick_label_bottoms) > shared_xlabel_top
    assert all(not example_axis.get_xlabel() for example_axis in example_axes)
    assert not [
        text
        for example_axis in example_axes
        for text in example_axis.texts
        if text.get_text().startswith("ΔLL=")
    ]

    legend_axes = [
        child_axis
        for child_axis in ax.child_axes
        if _relative_bounds(ax, child_axis)
        == pytest.approx(figure_3_module.PANEL_B_TRACE_LEGEND_SLOT_BOUNDS)
    ]
    assert len(legend_axes) == 1
    legend_axis = legend_axes[0]
    assert legend_axis not in example_axes
    assert _relative_bounds(ax, legend_axis) == pytest.approx(
        figure_3_module.PANEL_B_TRACE_LEGEND_SLOT_BOUNDS
    )
    legend = legend_axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Empirical",
        "Independent",
        "Multiplicative",
        "Additive",
    ]
    assert legend._ncols == 4
    plt.close(fig)


def test_panel_b_plots_the_corresponding_additive_prediction_for_each_example(
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    examples = [
        _make_panel_b_example(index)
        for index in range(len(figure_3_module.PANEL_B_SWAP_EXAMPLES))
    ]
    fig, ax = plt.subplots()
    figure_3_module.plot_panel_b_swap_examples_panel(
        ax,
        examples,
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        model_colors=figure_3_module.PANEL_B_MODEL_COLORS,
        model_labels=figure_3_module.PANEL_B_MODEL_LABELS,
    )
    fig.canvas.draw()

    example_axes = sorted(
        (
            child_axis
            for child_axis in ax.child_axes
            if any(
                _relative_bounds(ax, child_axis) == pytest.approx(bounds)
                for bounds in figure_3_module.PANEL_B_EXAMPLE_SLOT_BOUNDS
            )
        ),
        key=lambda child_axis: _relative_bounds(ax, child_axis)[0],
    )
    assert len(example_axes) == len(examples) == 4
    expected_trace_labels = {
        "Empirical",
        "Independent",
        "Multiplicative",
        "Additive",
    }
    model_names_by_label = {
        "Independent": "visual",
        "Multiplicative": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "Additive": figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME,
    }
    example_lines: list[dict[str, Any]] = []
    for example_index, (example_axis, example) in enumerate(
        zip(example_axes, examples, strict=True),
        start=1,
    ):
        lines_by_label = {
            line.get_label(): line
            for line in example_axis.lines
            if line.get_label() in expected_trace_labels
        }
        example_lines.append(lines_by_label)
        assert set(lines_by_label) == expected_trace_labels
        additive_values = example["models"][
            figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME
        ]
        assert list(lines_by_label["Additive"].get_ydata()) == pytest.approx(
            additive_values
        )
        assert example_axis.get_ylim()[1] >= max(additive_values)
        assert "Dark scaffold" not in {
            line.get_label() for line in example_axis.lines
        }
        assert example_axis.get_title() == f"Ex. {example_index}"
        assert not any(
            text.get_text().startswith("ΔLL=")
            for text in example_axis.texts
        )
        assert example_axis.get_facecolor() == pytest.approx(
            to_rgba("white")
        )
        assert not any(
            patch.get_visible() and patch.get_facecolor()[3] > 0.0
            for patch in example_axis.patches
        )

        empirical_line = lines_by_label["Empirical"]
        assert to_rgba(empirical_line.get_color()) == pytest.approx(
            to_rgba("black")
        )
        assert empirical_line.get_linewidth() == pytest.approx(
            figure_3_module.PANEL_B_EMPIRICAL_LINEWIDTH
        )
        assert empirical_line.get_linestyle() == "-"
        assert empirical_line.get_marker() in ("", "None", None)
        assert empirical_line.get_zorder() > max(
            lines_by_label[label].get_zorder()
            for label in model_names_by_label
        )

        for trace_label, model_name in model_names_by_label.items():
            prediction_line = lines_by_label[trace_label]
            expected_style = Line2D(
                [],
                [],
                linestyle=(
                    figure_3_module.PANEL_B_MODEL_LINESTYLES[model_name]
                ),
            )
            assert prediction_line._unscaled_dash_pattern == (
                expected_style._unscaled_dash_pattern
            )
            assert prediction_line.get_linewidth() == pytest.approx(0.8)
            assert prediction_line.get_marker() in ("", "None", None)

    legend_axes = [
        child_axis
        for child_axis in ax.child_axes
        if _relative_bounds(ax, child_axis)
        == pytest.approx(figure_3_module.PANEL_B_TRACE_LEGEND_SLOT_BOUNDS)
    ]
    assert len(legend_axes) == 1
    legend = legend_axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Empirical",
        "Independent",
        "Multiplicative",
        "Additive",
    ]
    legend_handles = {
        handle.get_label(): handle for handle in legend.legend_handles
    }
    assert set(legend_handles) == expected_trace_labels
    for trace_label in expected_trace_labels:
        plotted_line = example_lines[0][trace_label]
        legend_handle = legend_handles[trace_label]
        assert to_rgba(legend_handle.get_color()) == pytest.approx(
            to_rgba(plotted_line.get_color())
        )
        assert legend_handle.get_linewidth() == pytest.approx(
            plotted_line.get_linewidth()
        )
        if trace_label == "Empirical":
            assert to_rgba(legend_handle.get_color()) == pytest.approx(
                to_rgba("black")
            )
            assert legend_handle.get_linestyle() == "-"
            assert legend_handle.get_marker() in ("", "None", None)
            assert plotted_line.get_marker() in ("", "None", None)
        else:
            assert legend_handle.get_marker() in ("", "None", None)
            assert plotted_line.get_marker() in ("", "None", None)
        assert legend_handle._unscaled_dash_pattern == (
            plotted_line._unscaled_dash_pattern
        )
    plt.close(fig)


def test_panel_c_delta_tables_use_the_same_matched_forward_heldout_cohort() -> None:
    pd = pytest.importorskip("pandas")

    def make_row(
        unit: int,
        delta: float,
        model_name: str,
        *,
        trajectory: str,
        train_epoch: str = "02_r1",
        test_epoch: str = "06_r3",
    ) -> dict[str, object]:
        return {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "dark_epoch": "08_r4",
            "light_train_epoch": train_epoch,
            "light_test_epoch": test_epoch,
            "trajectory": trajectory,
            "unit": unit,
            "model_name": model_name,
            "delta_ll_bits_per_spike": delta,
            "source_path": "/analysis/L14/selected.nc",
        }

    multiplicative_model = figure_2_module.PANEL_C_SWAP_MODEL_NAME
    additive_model = figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME
    trajectories = tuple(figure_2_module.PANEL_H_DELTA_TRAJECTORIES)
    multiplicative_delta = pd.DataFrame(
        [
            make_row(
                unit,
                delta,
                multiplicative_model,
                trajectory=trajectory,
            )
            for unit, delta in ((11, 0.8), (22, -0.1), (44, 0.7))
            for trajectory in trajectories
        ]
        + [
            make_row(
                33,
                0.9,
                multiplicative_model,
                trajectory=trajectory,
                train_epoch="06_r3",
                test_epoch="02_r1",
            )
            for trajectory in trajectories
        ]
    )
    additive_delta = pd.DataFrame(
        [
            make_row(
                unit,
                delta,
                additive_model,
                trajectory=trajectory,
            )
            for unit, delta in ((22, -0.4), (11, 0.3), (55, 0.1))
            for trajectory in reversed(trajectories)
        ]
        + [
            make_row(
                33,
                0.2,
                additive_model,
                trajectory=trajectory,
                train_epoch="06_r3",
                test_epoch="02_r1",
            )
            for trajectory in trajectories
        ]
    )

    paired_independent, paired_additive = (
        figure_3_module._build_panel_c_delta_tables(
            multiplicative_delta,
            additive_delta,
        )
    )

    assert len(paired_independent) == len(paired_additive) == 8
    assert sorted(paired_independent["unit"].unique()) == [11, 22]
    assert sorted(paired_additive["unit"].unique()) == [11, 22]
    assert paired_independent.groupby("unit")[
        "delta_ll_bits_per_spike"
    ].mean().to_dict() == pytest.approx({11: 0.8, 22: -0.1})
    assert paired_additive.groupby("unit")[
        "delta_ll_bits_per_spike"
    ].mean().to_dict() == pytest.approx({11: 0.5, 22: 0.3})
    assert set(paired_independent["model_name"]) == {multiplicative_model}
    assert set(paired_additive["model_name"]) == {multiplicative_model}


def test_panel_c_compares_multiplicative_with_independent_then_additive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    pd = pytest.importorskip("pandas")

    calls: dict[str, Any] = {"histograms": []}
    identity_rows = [
        {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "dark_epoch": "08_r4",
            "light_train_epoch": "02_r1",
            "light_test_epoch": "06_r3",
            "unit": 11,
            "source_path": "/analysis/L14/selected.nc",
        },
        {
            "animal_name": "L15",
            "date": "20241121",
            "region": "v1",
            "dark_epoch": "08_r4",
            "light_train_epoch": "02_r1",
            "light_test_epoch": "06_r3",
            "unit": 22,
            "source_path": "/analysis/L15/selected.nc",
        },
    ]
    trajectories = tuple(figure_2_module.PANEL_H_DELTA_TRAJECTORIES)
    multiplicative_delta = pd.DataFrame(
        [
            {
                **identity_row,
                "trajectory": trajectory,
                "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
                "delta_ll_bits_per_spike": delta,
            }
            for identity_row, delta in zip(
                identity_rows,
                (0.8, -0.1),
                strict=True,
            )
            for trajectory in trajectories
        ]
    )
    additive_delta = pd.DataFrame(
        [
            {
                **identity_rows[1],
                "trajectory": trajectory,
                "model_name": figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME,
                "delta_ll_bits_per_spike": -0.4,
            }
            for trajectory in reversed(trajectories)
        ]
        + [
            {
                **identity_rows[0],
                "trajectory": trajectory,
                "model_name": figure_3_module.PANEL_B_ADDITIVE_MODEL_NAME,
                "delta_ll_bits_per_spike": 0.3,
            }
            for trajectory in reversed(trajectories)
        ]
    )

    def fake_plot_histogram(
        ax: Any,
        swap_delta_table: object,
        **kwargs: object,
    ) -> None:
        calls["histograms"].append((ax, swap_delta_table, kwargs))
        ax.set_ylim(0.0, 0.2 + 0.1 * len(calls["histograms"]))
        ax.set_ylabel("Fraction")
        model_name = str(kwargs["model_name"])
        model_colors = kwargs["model_colors"]
        model_labels = kwargs["model_labels"]
        ax.text(
            0.03,
            0.97,
            "Indep. better",
            color=model_colors["visual"],
        )
        ax.text(
            0.70,
            0.97,
            f"{model_labels[model_name]}\nbetter",
            color=model_colors[model_name],
        )
        ax.text(
            0.97,
            0.06,
            "75% >0",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            0.03,
            0.06,
            "n = 12 cells\n2 animals",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.set_xlabel(figure_2_module.DELTA_LOG_LIKELIHOOD_AXIS_LABEL)

    def fail_if_example_is_plotted(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Panel C plotted a stimulus-swap example")

    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d_mean_swap_delta_axis",
        fake_plot_histogram,
    )
    monkeypatch.setattr(
        figure_2_module,
        "_plot_panel_h_switched_segment_example",
        fail_if_example_is_plotted,
    )
    model_colors = figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3
    model_labels = figure_3_module.PANEL_C_MODEL_LABELS
    fig, ax = plt.subplots()
    figure_3_module.plot_panel_c_swap_histogram_panel(
        ax,
        multiplicative_delta,
        additive_delta,
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        model_colors=model_colors,
        model_labels=model_labels,
    )
    fig.canvas.draw()

    histogram_calls = calls["histograms"]
    assert len(histogram_calls) == 2
    independent_axis, independent_table, independent_kwargs = histogram_calls[0]
    additive_axis, multiplicative_vs_additive, additive_kwargs = (
        histogram_calls[1]
    )
    assert independent_table is not multiplicative_delta
    assert len(independent_table) == 8
    assert sorted(independent_table["unit"].unique()) == [11, 22]
    assert independent_table.groupby("unit")[
        "delta_ll_bits_per_spike"
    ].mean().to_dict() == pytest.approx({11: 0.8, 22: -0.1})
    assert independent_kwargs == {
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "model_colors": model_colors,
        "model_labels": model_labels,
    }
    assert multiplicative_vs_additive is not multiplicative_delta
    assert multiplicative_vs_additive is not additive_delta
    assert len(multiplicative_vs_additive) == 8
    assert sorted(multiplicative_vs_additive["unit"].unique()) == [11, 22]
    assert set(multiplicative_vs_additive["model_name"]) == {
        figure_2_module.PANEL_C_SWAP_MODEL_NAME
    }
    assert multiplicative_vs_additive.groupby("unit")[
        "delta_ll_bits_per_spike"
    ].mean().to_dict() == pytest.approx({11: 0.5, 22: 0.3})
    assert additive_kwargs["model_name"] == (
        figure_2_module.PANEL_C_SWAP_MODEL_NAME
    )
    assert additive_kwargs["model_labels"] == model_labels
    assert to_rgba(additive_kwargs["model_colors"]["visual"]) == pytest.approx(
        to_rgba(figure_3_module.PANEL_B_ADDITIVE_MODEL_COLOR)
    )
    assert additive_kwargs["model_colors"][figure_2_module.PANEL_C_SWAP_MODEL_NAME] == (
        model_colors[figure_2_module.PANEL_C_SWAP_MODEL_NAME]
    )
    assert model_colors["visual"] == (
        figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3["visual"]
    )
    assert model_colors[figure_2_module.PANEL_C_SWAP_MODEL_NAME] == (
        figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3[
            figure_2_module.PANEL_C_SWAP_MODEL_NAME
        ]
    )
    assert model_labels[figure_2_module.PANEL_C_SWAP_MODEL_NAME] == (
        "Multiplicative"
    )
    assert ax.child_axes == [independent_axis, additive_axis]
    assert _relative_bounds(ax, independent_axis) == pytest.approx(
        figure_3_module.PANEL_C_MULTIPLICATIVE_VS_INDEPENDENT_HISTOGRAM_AXIS_BOUNDS
    )
    assert _relative_bounds(ax, additive_axis) == pytest.approx(
        figure_3_module.PANEL_C_MULTIPLICATIVE_VS_ADDITIVE_HISTOGRAM_AXIS_BOUNDS
    )
    assert independent_axis.get_title() == (
        figure_3_module.PANEL_C_HISTOGRAM_TITLES[0]
    )
    assert additive_axis.get_title() == (
        figure_3_module.PANEL_C_HISTOGRAM_TITLES[1]
    )
    assert independent_axis.get_ylim() == pytest.approx((0.0, 0.3))
    assert additive_axis.get_ylim() == pytest.approx((0.0, 0.4))
    assert additive_axis.get_ylabel() == ""
    assert all(
        tick_label.get_visible()
        for tick_label in additive_axis.get_yticklabels()
    )
    assert [text.get_text() for text in independent_axis.texts] == [
        "Indep.\nbetter",
        "Multiplicative\nbetter",
        "75% >0",
        "n = 12 cells",
    ]
    assert [text.get_text() for text in additive_axis.texts] == [
        "Additive\nbetter",
        "Multiplicative\nbetter",
        "75% >0",
        "n = 12 cells",
    ]
    assert to_rgba(additive_axis.texts[0].get_color()) == pytest.approx(
        to_rgba(figure_3_module.PANEL_B_ADDITIVE_MODEL_COLOR)
    )
    assert independent_axis.get_xlabel() == ""
    assert additive_axis.get_xlabel() == ""
    shared_xlabels = [
        text
        for text in ax.texts
        if text.get_text() == figure_3_module.PANEL_C_SHARED_XLABEL
    ]
    assert len(shared_xlabels) == 1
    assert figure_3_module.PANEL_C_SHARED_XLABEL == (
        "Δ log likelihood (bits/spike)"
    )
    assert "\n" not in shared_xlabels[0].get_text()
    assert shared_xlabels[0].get_transform() is ax.transAxes
    assert shared_xlabels[0].get_position() == pytest.approx(
        figure_3_module.PANEL_C_SHARED_XLABEL_POSITION
    )
    for histogram_axis in (independent_axis, additive_axis):
        count_labels = [
            text
            for text in histogram_axis.texts
            if text.get_text().startswith("n = ")
        ]
        assert len(count_labels) == 1
        assert count_labels[0].get_text() == "n = 12 cells"
        assert count_labels[0].get_position()[0] == pytest.approx(0.03)
        assert count_labels[0].get_position()[1] == pytest.approx(
            figure_3_module.PANEL_C_BOTTOM_ANNOTATION_Y
        )
        assert count_labels[0].get_transform() is histogram_axis.transAxes
        assert count_labels[0].get_ha() == "left"
        assert count_labels[0].get_va() == "bottom"
        positive_labels = [
            text
            for text in histogram_axis.texts
            if text.get_text().endswith("% >0")
        ]
        assert len(positive_labels) == 1
        assert positive_labels[0].get_position()[0] == pytest.approx(0.97)
        assert positive_labels[0].get_position()[1] == pytest.approx(
            figure_3_module.PANEL_C_BOTTOM_ANNOTATION_Y
        )
        assert positive_labels[0].get_transform() is histogram_axis.transAxes
    visible_text = [
        text.get_text()
        for axis in (ax, independent_axis, additive_axis)
        for text in axis.texts
    ]
    assert not any("animal" in text.lower() for text in visible_text)
    assert not any(
        tail_label in text
        for text in visible_text
        for tail_label in ("<−1", "<-1", ">1")
    )
    plt.close(fig)


def test_panel_c_shared_xlabel_matches_panel_b_tick_label_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")

    calls: dict[str, Any] = {}
    examples = [
        _make_panel_b_example(index)
        for index in range(len(figure_3_module.PANEL_B_SWAP_EXAMPLES))
    ]
    comparison_table = pd.DataFrame(
        {"delta_ll_bits_per_spike": [0.15, -0.05, 0.20]}
    )
    original_plot_panel_b = figure_3_module.plot_panel_b_swap_examples_panel
    original_plot_panel_c = figure_3_module.plot_panel_c_swap_histogram_panel

    def fake_load_panel_data(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "swap_delta": object(),
            "swap_additive_delta": object(),
            "swap_examples": examples,
        }

    def fake_build_panel_c_delta_tables(
        multiplicative_delta: object,
        additive_delta: object,
    ) -> tuple[Any, Any]:
        del multiplicative_delta, additive_delta
        return comparison_table, comparison_table.copy()

    def capture_panel_b(ax: Any, *args: object, **kwargs: object) -> None:
        calls["panel_b_axis"] = ax
        original_plot_panel_b(ax, *args, **kwargs)

    def capture_panel_c(ax: Any, *args: object, **kwargs: object) -> None:
        calls["panel_c_axis"] = ax
        original_plot_panel_c(ax, *args, **kwargs)

    def fake_save_figure(
        fig: Any,
        output_path: Path,
        dpi: int,
        **kwargs: object,
    ) -> Path:
        del dpi, kwargs
        fig.canvas.draw()
        calls["panel_b_gap_points"] = _shared_xlabel_tick_gap_points(
            fig,
            calls["panel_b_axis"],
            figure_3_module.PANEL_B_SHARED_XLABEL,
        )
        calls["panel_c_gap_points"] = _shared_xlabel_tick_gap_points(
            fig,
            calls["panel_c_axis"],
            figure_3_module.PANEL_C_SHARED_XLABEL,
        )
        return output_path

    monkeypatch.setattr(
        figure_3_module,
        "load_figure_3_panel_data",
        fake_load_panel_data,
    )
    monkeypatch.setattr(
        figure_3_module,
        "_build_panel_c_delta_tables",
        fake_build_panel_c_delta_tables,
    )
    monkeypatch.setattr(
        figure_3_module,
        "plot_panel_b_swap_examples_panel",
        capture_panel_b,
    )
    monkeypatch.setattr(
        figure_3_module,
        "plot_panel_c_swap_histogram_panel",
        capture_panel_c,
    )
    monkeypatch.setattr(
        figure_3_module,
        "save_figure",
        fake_save_figure,
    )

    figure_3_module.make_figure_3(
        data_root=Path("/analysis"),
        output_path=tmp_path / "figure_3.svg",
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        dark_epoch=None,
        dpi=144,
    )

    assert calls["panel_b_gap_points"] > 0.0
    assert calls["panel_c_gap_points"] == pytest.approx(
        calls["panel_b_gap_points"],
        abs=0.25,
    )


def test_make_figure_3_uses_three_full_width_rows_and_split_swap_panels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, Any] = {}

    def fake_load_panel_data(**kwargs: object) -> dict[str, object]:
        calls["loader_kwargs"] = kwargs
        return {
            "swap_delta": "swap-delta",
            "swap_additive_delta": "swap-additive-delta",
            "swap_examples": ["swap-example"],
        }

    def fake_plot_architecture(ax: object) -> None:
        calls["architecture_axis"] = ax

    def fake_plot_examples(
        ax: object,
        swap_examples: object,
        **kwargs: object,
    ) -> None:
        calls["panel_b_axis"] = ax
        calls["swap_examples"] = swap_examples
        calls["panel_b_kwargs"] = kwargs

    def fake_plot_histogram(
        ax: object,
        swap_delta: object,
        swap_additive_delta: object,
        **kwargs: object,
    ) -> None:
        calls["panel_c_axis"] = ax
        calls["swap_delta"] = swap_delta
        calls["swap_additive_delta"] = swap_additive_delta
        calls["panel_c_kwargs"] = kwargs

    def fail_if_old_architecture_is_plotted(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Figure 3 used the two-model architecture renderer")

    def fake_save_figure(
        figure: object,
        output_path: Path,
        dpi: int,
        **kwargs: object,
    ) -> Path:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        titled_axes = {ax.get_title(): ax for ax in figure.axes if ax.get_title()}
        calls["figsize"] = tuple(figure.get_size_inches())
        calls["titles"] = tuple(titled_axes)
        calls["bounds"] = {
            title: ax.get_position().bounds for title, ax in titled_axes.items()
        }
        calls["labels"] = {
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["label_font_sizes"] = {
            text.get_text(): text.get_fontsize()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["label_display_positions"] = {
            text.get_text(): text.get_transform().transform(text.get_position())
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        panel_a_label = next(
            text
            for text in calls["architecture_axis"].texts
            if text.get_text() == "A"
        )
        calls["panel_a_header_tops"] = (
            panel_a_label.get_window_extent(renderer).y1,
            calls["architecture_axis"].title.get_window_extent(renderer).y1,
        )
        calls["label_positions"] = {
            text.get_text(): text.get_position()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["panel_bounds"] = {
            "A": calls["architecture_axis"].get_position().bounds,
            "B": calls["panel_b_axis"].get_position().bounds,
            "C": calls["panel_c_axis"].get_position().bounds,
        }
        calls["height_ratios"] = (
            calls["architecture_axis"]
            .get_subplotspec()
            .get_gridspec()
            .get_height_ratios()
        )
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        return output_path

    monkeypatch.setattr(
        figure_3_module,
        "load_figure_3_panel_data",
        fake_load_panel_data,
    )
    monkeypatch.setattr(
        figure_3_module,
        "plot_panel_a_three_model_architecture",
        fake_plot_architecture,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d2_architecture_panel",
        fail_if_old_architecture_is_plotted,
    )
    monkeypatch.setattr(
        figure_3_module,
        "plot_panel_b_swap_examples_panel",
        fake_plot_examples,
    )
    monkeypatch.setattr(
        figure_3_module,
        "plot_panel_c_swap_histogram_panel",
        fake_plot_histogram,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d2_swap_results_panel",
        fail_if_old_architecture_is_plotted,
    )
    monkeypatch.setattr(figure_3_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_3.svg"
    saved_path = figure_3_module.make_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        dark_epoch=None,
        dpi=144,
    )

    assert saved_path == output_path
    assert calls["figsize"] == pytest.approx(
        (
            figure_3_module.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            figure_3_module.DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    assert calls["titles"] == figure_3_module.PANEL_TITLES
    assert calls["labels"] == {"A", "B", "C"}
    assert set(calls["label_font_sizes"].values()) == {
        figure_3_module.PANEL_LABEL_FONTSIZE
    }
    assert calls["label_positions"]["A"] == pytest.approx(
        (
            figure_3_module.PANEL_A_B_LABEL_X,
            figure_2_module.PANEL_B_LABEL_Y,
        )
    )
    assert calls["label_positions"]["B"] == pytest.approx(
        (
            figure_3_module.PANEL_B_LABEL_X,
            figure_3_module.PANEL_ROW_LABEL_Y,
        )
    )
    assert calls["label_positions"]["C"] == pytest.approx(
        (
            figure_3_module.PANEL_C_LABEL_X,
            figure_3_module.PANEL_ROW_LABEL_Y,
        )
    )
    assert calls["label_display_positions"]["A"][0] == pytest.approx(
        calls["label_display_positions"]["B"][0]
    )
    assert calls["label_display_positions"]["A"][0] == pytest.approx(
        calls["label_display_positions"]["C"][0]
    )
    assert calls["panel_a_header_tops"][0] == pytest.approx(
        calls["panel_a_header_tops"][1]
    )
    assert calls["label_display_positions"]["A"][1] > (
        calls["label_display_positions"]["B"][1]
    )
    assert calls["label_display_positions"]["B"][1] > (
        calls["label_display_positions"]["C"][1]
    )
    panel_bounds = calls["panel_bounds"]
    assert panel_bounds["A"][1] > panel_bounds["B"][1] > panel_bounds["C"][1]
    assert calls["height_ratios"] == pytest.approx(
        (
            figure_3_module.TOP_ROW_HEIGHT_MM,
            0.9 * figure_2_module.PANEL_D_ROW_HEIGHT_MM,
            0.9 * figure_2_module.PANEL_D_ROW_HEIGHT_MM,
        )
    )
    assert [panel_bounds[label][0] for label in ("A", "B", "C")] == (
        pytest.approx([panel_bounds["A"][0]] * 3)
    )
    assert [panel_bounds[label][2] for label in ("A", "B", "C")] == (
        pytest.approx([panel_bounds["A"][2]] * 3)
    )
    assert calls["swap_delta"] == "swap-delta"
    assert calls["swap_additive_delta"] == "swap-additive-delta"
    assert calls["swap_examples"] == ["swap-example"]
    expected_panel_b_model_kwargs = {
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "model_colors": figure_3_module.PANEL_B_MODEL_COLORS,
        "model_labels": figure_3_module.PANEL_B_MODEL_LABELS,
    }
    expected_panel_c_model_kwargs = {
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "model_colors": figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3,
        "model_labels": figure_3_module.PANEL_C_MODEL_LABELS,
    }
    assert calls["panel_b_kwargs"] == expected_panel_b_model_kwargs
    assert calls["panel_c_kwargs"] == expected_panel_c_model_kwargs
    assert calls["loader_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "regions": ("v1",),
        "dark_epoch": None,
    }
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 144
    assert calls["save_kwargs"] == {"bbox_inches": None}


def test_main_builds_figure_3_output_and_forwards_cli_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_make_figure_3(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        figure_3_module,
        "make_figure_3",
        fake_make_figure_3,
    )
    figure_3_module.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            str(tmp_path),
            "--format",
            "svg",
            "--dataset",
            "L14:20240611:08_r4",
            "--region",
            "v1",
            "--dark-epoch",
            "08_r4",
            "--dpi",
            "144",
        ]
    )

    assert calls == {
        "data_root": Path("/analysis"),
        "output_path": tmp_path / "figure_3.svg",
        "datasets": [("L14", "20240611", "08_r4")],
        "regions": ("v1",),
        "dark_epoch": "08_r4",
        "dpi": 144,
    }
