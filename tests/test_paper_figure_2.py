from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_2 as figure


def test_default_height_uses_one_shared_row_for_panels_b_and_c() -> None:
    args = figure.parse_arguments([])

    assert figure.DEFAULT_OUTPUT_NAME == "figure_2"
    assert args.output_dir == Path("paper_figures/output")
    assert args.output_format == figure._figure_2.DEFAULT_OUTPUT_FORMAT
    assert not hasattr(args, "decoding_n_permutations")
    assert not hasattr(args, "dark_tuning_correlation_threshold")
    assert args.min_epoch_movement_firing_rate_hz == pytest.approx(0.5)
    assert args.min_path_movement_firing_rate_hz == pytest.approx(0.5)
    assert args.min_segment_mean_firing_rate_hz == pytest.approx(0.5)
    assert not hasattr(args, "min_movement_firing_rate_hz")
    assert args.min_stability_correlation == pytest.approx(0.5)
    assert figure.DEFAULT_FIGURE_WIDTH_MM == 165.0
    assert figure.DEFAULT_FIGURE_WIDTH_MM == (
        figure._figure_2.DEFAULT_FIGURE_WIDTH_MM
    )
    assert figure.DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        figure.PANEL_A_HEIGHT_MM
        + figure._figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM
    )


def test_panel_c_schematic_uses_raw_path_colored_curves_and_gray_overlap() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legacy_figure = figure._figure_2._figure_2
    trajectories = ("center_to_left", "right_to_center")
    fig, axis = plt.subplots()
    legacy_figure.plot_panel_b_dppi_schematic(
        axis,
        {"trajectories": trajectories},
        style="path_colored_gray_overlap",
    )
    fig.canvas.draw()

    curve_axis = axis.child_axes[0]
    curve_lines = [
        line
        for line in curve_axis.lines
        if np.asarray(line.get_xdata()).size == 121
    ]
    assert len(curve_lines) == 2
    assert [line.get_color() for line in curve_lines] == [
        legacy_figure.PANEL_TRAJECTORY_COLORS[trajectory]
        for trajectory in trajectories
    ]
    curve_heights = [
        float(np.max(np.asarray(line.get_ydata(), dtype=float)))
        for line in curve_lines
    ]
    assert curve_heights[1] / curve_heights[0] == pytest.approx(
        legacy_figure.PANEL_B_DPPI_PATH_COLORED_SECOND_RATE_SCALE,
        abs=1e-3,
    )
    assert len(curve_axis.collections) == 1
    overlap_fill = curve_axis.collections[0]
    expected_fill = matplotlib.colors.to_rgba(
        legacy_figure.PANEL_B_DPPI_GRAY_OVERLAP_COLOR
    )
    assert overlap_fill.get_facecolor()[0, :3] == pytest.approx(
        expected_fill[:3]
    )
    assert overlap_fill.get_alpha() == pytest.approx(
        legacy_figure.PANEL_B_DPPI_GRAY_OVERLAP_ALPHA
    )
    plt.close(fig)


def test_panel_a_example_plotter_wraps_eight_cells_into_two_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted_axes = []
    plotted_kwargs = []
    ylabel_axes = []

    def fake_plot_panel_a_example(
        axis: object,
        example: object,
        **kwargs: object,
    ) -> None:
        plotted_axes.append(axis)
        plotted_kwargs.append(kwargs)
        raster_axis = axis.inset_axes((0.05, 0.40, 0.37, 0.40))
        raster_axis.set_ylabel("Trials")
        rate_axis = axis.inset_axes((0.05, 0.05, 0.37, 0.25))
        rate_axis.set_ylabel("FR (Hz)")
        ylabel_axes.extend((raster_axis, rate_axis))

    monkeypatch.setattr(
        figure._figure_2._figure_2,
        "plot_panel_a_example",
        fake_plot_panel_a_example,
    )
    fig, axis = plt.subplots()
    figure._figure_2.plot_panel_a2_examples_single_row(
        axis,
        [{"unit_id": unit_id} for unit_id in range(8)],
        y_max_overrides={3: 85.0},
        schematic_scale=1.5,
        ylabel_x=-0.44,
    )
    fig.canvas.draw()

    assert len(plotted_axes) == 8
    bounds = [child_axis.get_position().bounds for child_axis in plotted_axes]
    assert len({round(bound[0], 6) for bound in bounds}) == 4
    assert len({round(bound[1], 6) for bound in bounds}) == 2
    assert all(bound[2] == pytest.approx(bounds[0][2]) for bound in bounds)
    assert all(bound[3] == pytest.approx(bounds[0][3]) for bound in bounds)
    parent_bounds = axis.get_position().bounds
    horizontal_gap = bounds[1][0] - (bounds[0][0] + bounds[0][2])
    vertical_gap = bounds[0][1] - (bounds[4][1] + bounds[4][3])
    assert horizontal_gap / parent_bounds[2] == pytest.approx(
        figure._figure_2.PANEL_A2_MULTIROW_COLUMN_GAP
    )
    assert vertical_gap / parent_bounds[3] == pytest.approx(
        figure._figure_2.PANEL_A2_MULTIROW_ROW_GAP
    )
    headings = [
        text.get_text()
        for child_axis in plotted_axes
        for text in child_axis.texts
        if text.get_text().startswith("Example cell")
    ]
    assert headings == [f"Example cell {index}" for index in range(1, 9)]
    assert [
        text.get_fontweight()
        for child_axis in plotted_axes
        for text in child_axis.texts
        if text.get_text().startswith("Example cell")
    ] == ["semibold"] * 8
    assert plotted_kwargs[2]["y_max"] == pytest.approx(85.0)
    assert all(
        kwargs["schematic_axis_width"]
        == pytest.approx(1.5 * figure._figure_2.PANEL_A2_SCHEMATIC_AXIS_WIDTH)
        for kwargs in plotted_kwargs
    )
    assert all(
        kwargs["schematic_axis_height"]
        == pytest.approx(1.5 * figure._figure_2.PANEL_A2_SCHEMATIC_AXIS_HEIGHT)
        for kwargs in plotted_kwargs
    )
    original_schematic_right = (
        figure._figure_2.PANEL_A2_SINGLE_ROW_SCHEMATIC_AXIS_LEFT
        + figure._figure_2.PANEL_A2_SCHEMATIC_AXIS_WIDTH
    )
    assert all(
        kwargs["schematic_axis_left"] + kwargs["schematic_axis_width"]
        == pytest.approx(original_schematic_right)
        for kwargs in plotted_kwargs
    )
    assert all(
        ylabel_axis.yaxis.label.get_position() == pytest.approx((-0.44, 0.50))
        for ylabel_axis in ylabel_axes
    )
    assert all(
        "y_max" not in kwargs
        for index, kwargs in enumerate(plotted_kwargs)
        if index != 2
    )
    plt.close(fig)


def test_make_figure_2_renders_only_a_through_c_with_preserved_b_c_flows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}
    panel_h_shift_profile_table = object()
    legacy_panel_b_raw_table = object()
    legacy_panel_b_filtered_table = object()

    def fail_unrequested_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("An unrelated Figure 2 loader was called")

    def fake_load_panel_a_example_data(**kwargs: object) -> dict[str, object]:
        calls.setdefault("panel_a_loads", []).append(kwargs)
        return {"unit_id": kwargs["unit_id"]}

    def fake_load_panel_h_shift_profile_table(**kwargs: object) -> object:
        calls.setdefault("panel_h_shift_profile_loads", []).append(kwargs)
        calls["panel_h_shift_profile_load"] = kwargs
        return panel_h_shift_profile_table

    def fake_load_legacy_panel_b_table(**kwargs: object) -> object:
        calls["legacy_panel_b_load"] = kwargs
        return legacy_panel_b_raw_table

    def fake_filter_legacy_panel_b_table(
        table: object,
        **kwargs: object,
    ) -> object:
        calls["legacy_panel_b_filter_table"] = table
        calls["legacy_panel_b_filter_kwargs"] = kwargs
        return legacy_panel_b_filtered_table

    def fake_plot_panel_a(
        axis: object,
        examples: object,
        **kwargs: object,
    ) -> None:
        calls["panel_a_axis"] = axis
        calls["panel_a_examples"] = examples
        calls["panel_a_plot_kwargs"] = kwargs

    def fake_plot_circular_shift_schematic(axis: object) -> None:
        calls["panel_b_schematic_axis"] = axis

    def fake_plot_legacy_panel_c(
        axis: object,
        table: object,
        **kwargs: object,
    ) -> None:
        calls["legacy_panel_c_axis"] = axis
        calls["legacy_panel_c_table"] = table
        calls["legacy_panel_c_plot_kwargs"] = kwargs
        for index, legacy_text in enumerate(
            (
                "DPP index",
                "DPPI = max(left overlap,\nright overlap)",
                "Dark DPPI",
                "Light DPPI",
            )
        ):
            axis.text(
                0.5,
                0.65 - 0.10 * index,
                legacy_text,
                ha="center",
                va="center",
            )

    def fake_align_legacy_panel_d_marginal(
        mpl_figure: object,
        axis: object,
    ) -> None:
        calls["legacy_panel_d_marginal_figure"] = mpl_figure
        calls["legacy_panel_d_marginal_axis"] = axis

    def fake_align_panel_b_and_c_axes(
        mpl_figure: object,
        panel_b_axis: object,
        panel_c_axis: object,
    ) -> None:
        calls["panel_b_c_alignment"] = (
            mpl_figure,
            panel_b_axis,
            panel_c_axis,
        )

    def fake_plot_population_shift_profile(
        axis: object,
        table: object,
    ) -> None:
        calls.setdefault("population_shift_profile_calls", []).append(
            (axis, table)
        )
        calls["population_shift_profile_axis"] = axis
        calls["population_shift_profile_table"] = table
        axis.set_xlabel("Circular shift")
        axis.set_ylabel("Norm. overlap")

    def fake_save_figure(
        mpl_figure: object,
        output_path: Path,
        dpi: int,
        **kwargs: object,
    ) -> Path:
        mpl_figure.canvas.draw()
        calls["figsize"] = tuple(mpl_figure.get_size_inches())
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["axis_bounds"] = {
            axis.get_title(): axis.get_position().bounds
            for axis in mpl_figure.axes
            if axis.get_title()
        }
        calls["panel_labels"] = {
            text.get_text()
            for axis in mpl_figure.axes
            for text in axis.texts
            if text.get_text()
            in {
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
            }
        }
        calls["panel_label_by_title"] = {
            axis.get_title(): panel_labels[0]
            for axis in mpl_figure.axes
            if axis.get_title()
            and (
                panel_labels := [
                    text.get_text()
                    for text in axis.texts
                    if text.get_text()
                    in {
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "K",
                        "L",
                    }
                ]
            )
        }
        calls["all_text"] = {
            text.get_text()
            for axis in mpl_figure.axes
            for text in axis.texts
        }
        return output_path

    monkeypatch.setattr(
        figure._figure_2,
        "load_panel_a_example_data",
        fake_load_panel_a_example_data,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_panel_b_tuning_average_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_panel_b_split_half_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_panel_b_circular_null_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_panel_c_path_invariance_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_panel_d_achievable_stability_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_panel_h_shift_profile_table",
        fake_load_panel_h_shift_profile_table,
    )
    monkeypatch.setattr(
        figure._figure_2._figure_2,
        "load_panel_b_tuning_overlap_table",
        fake_load_legacy_panel_b_table,
    )
    monkeypatch.setattr(
        figure._figure_2._figure_2,
        "filter_panel_b_overlap_by_even_odd_stability",
        fake_filter_legacy_panel_b_table,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_segment_overlap_response_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_segment_stability_reference_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_full_path_achievable_stability_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "load_or_compute_segment_matched_achievable_stability_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure._figure_2,
        "plot_panel_a2_examples_single_row",
        fake_plot_panel_a,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_b_dark_same_turn_schematic",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_b_location_gain_schematic",
        fail_unrequested_loader,
        raising=False,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_b_tuning_similarity_with_schematic",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_circular_shift_schematic",
        fake_plot_circular_shift_schematic,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_c_dark_selected_path_invariance",
        fail_unrequested_loader,
        raising=False,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_c_path_invariance",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_d_achievable_stability",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_full_path_achievable_stability",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_segment_matched_achievable_stability",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_h_shift_profiles",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_population_shift_profile",
        fake_plot_population_shift_profile,
    )
    monkeypatch.setattr(
        figure._figure_2._figure_2,
        "plot_panel_b_dpp_overlap_with_schematic",
        fake_plot_legacy_panel_c,
    )
    monkeypatch.setattr(
        figure._figure_2,
        "_align_panel_b_top_histogram_label_to_scatter",
        fake_align_legacy_panel_d_marginal,
    )
    monkeypatch.setattr(
        figure,
        "_align_panel_b_profile_with_panel_c_scatter",
        fake_align_panel_b_and_c_axes,
    )
    monkeypatch.setattr(
        figure,
        "plot_raw_circular_shift_profiles",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_panel_c_raw_circular_shift_analysis",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_segment_overlap_response",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_segment_stability_references",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure,
        "plot_achievable_segment_stability",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure._figure_2,
        "load_panel_glm_data",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(
        figure._figure_2,
        "load_panel_e_decoding_error_table",
        fail_unrequested_loader,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_2.svg"
    saved_path = figure.make_figure_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        dark_epoch=None,
        dpi=200,
    )

    assert saved_path == output_path
    assert len(calls["panel_a_loads"]) == len(
        figure.FIGURE_2_PANEL_A_EXAMPLES
    )
    assert [
        (
            call["animal_name"],
            call["date"],
            call["region"],
            call["unit_id"],
            call["trajectories"],
        )
        for call in calls["panel_a_loads"]
    ] == [
        (
            "L15",
            "20241121",
            "v1",
            409,
            ("center_to_left", "right_to_center"),
        ),
        (
            "L14",
            "20240611",
            "v1",
            34,
            ("center_to_left", "right_to_center"),
        ),
        (
            "L14",
            "20240611",
            "v1",
            30,
            ("center_to_left", "right_to_center"),
        ),
        (
            "L15",
            "20241121",
            "v1",
            418,
            ("center_to_left", "right_to_center"),
        ),
        (
            "L14",
            "20240611",
            "v1",
            172,
            ("center_to_right", "left_to_center"),
        ),
        (
            "L15",
            "20241121",
            "v1",
            473,
            ("center_to_right", "left_to_center"),
        ),
        (
            "L15",
            "20241121",
            "v1",
            70,
            ("center_to_right", "left_to_center"),
        ),
        (
            "L12",
            "20240421",
            "v1",
            37,
            ("center_to_right", "left_to_center"),
        ),
    ]
    assert calls["panel_a_plot_kwargs"] == {
        "y_max_overrides": figure.FIGURE_2_PANEL_A_Y_MAX_OVERRIDES,
        "schematic_scale": figure.FIGURE_2_PANEL_A_WTRACK_SCALE,
        "ylabel_x": figure.FIGURE_2_PANEL_A_YLABEL_X,
    }
    assert calls["population_shift_profile_table"] is panel_h_shift_profile_table
    assert len(calls["population_shift_profile_calls"]) == 1
    assert calls["legacy_panel_b_filter_table"] is legacy_panel_b_raw_table
    assert calls["legacy_panel_c_table"] is legacy_panel_b_filtered_table
    assert calls["legacy_panel_d_marginal_axis"] is calls["legacy_panel_c_axis"]
    assert (
        calls["legacy_panel_d_marginal_figure"]
        is calls["legacy_panel_c_axis"].get_figure(root=False)
    )
    alignment_figure, alignment_panel_b, alignment_panel_c = calls[
        "panel_b_c_alignment"
    ]
    assert alignment_figure is calls["legacy_panel_c_axis"].get_figure(root=False)
    assert alignment_panel_b.get_title() == "Tuning shift across dark and light"
    assert alignment_panel_c is calls["legacy_panel_c_axis"]
    expected_path_load = {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "v1",
        "light_epoch": None,
        "dark_epoch": None,
        "cache_dir": tmp_path / "cache",
        "refresh_cache": False,
        "min_epoch_movement_firing_rate_hz": 0.5,
        "min_path_movement_firing_rate_hz": 0.5,
        "min_segment_mean_firing_rate_hz": 0.5,
        "min_stability_correlation": 0.5,
    }
    assert calls["panel_h_shift_profile_load"] == expected_path_load
    assert calls["panel_h_shift_profile_loads"] == [expected_path_load]
    legacy_figure = figure._figure_2._figure_2
    expected_legacy_load = {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "v1",
        "light_epoch": None,
        "dark_epoch": None,
    }
    assert calls["legacy_panel_b_load"] == expected_legacy_load
    assert calls["legacy_panel_b_filter_kwargs"] == {
        **expected_legacy_load,
        "min_movement_firing_rate_hz": (
            legacy_figure.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        "min_stability_correlation": (
            legacy_figure.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    }
    legacy_example_spec = figure._figure_2.FIGURE_2_PANEL_A_EXAMPLES[0]
    legacy_example_index = figure.FIGURE_2_PANEL_A_EXAMPLES.index(
        legacy_example_spec
    )
    assert legacy_example_index == 1
    assert calls["legacy_panel_c_plot_kwargs"] == {
        "example": calls["panel_a_examples"][legacy_example_index],
        "low_threshold": (
            legacy_figure.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
        ),
        "high_threshold": (
            legacy_figure.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
        ),
        "show_grouped": False,
        "show_scatter_linear_fit": True,
        "show_scatter_r2": True,
        "scatter_equal_aspect": True,
        "schematic_style": "path_colored_gray_overlap",
    }
    assert calls["figsize"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            figure.DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 200
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == {
        "A",
        "B",
        "C",
    }
    assert calls["panel_label_by_title"] == {
        "Path-invariant progression tuning across dark and light": "A",
        "Tuning shift across dark and light": "B",
        "Path-invariance across dark and light": "C",
    }
    panel_a_axis = calls["panel_a_axis"]
    panel_a_label_artist = next(
        text for text in panel_a_axis.texts if text.get_text() == "A"
    )
    panel_a_title_display_y = panel_a_axis.title.get_transform().transform(
        panel_a_axis.title.get_position()
    )[1]
    panel_a_label_display_y = panel_a_label_artist.get_transform().transform(
        panel_a_label_artist.get_position()
    )[1]
    assert panel_a_title_display_y == pytest.approx(panel_a_label_display_y)
    renderer = panel_a_axis.get_figure(root=False).canvas.get_renderer()
    assert panel_a_axis.title.get_window_extent(renderer).y0 == pytest.approx(
        panel_a_label_artist.get_window_extent(renderer).y0
    )
    assert "Split-half-rescaled circular-shift profiles (median and IQR)" not in (
        calls["all_text"]
    )
    assert not any("K=" in text for text in calls["all_text"])
    expected_pii_text = {
        "Path-invariance\nindex (PII)",
        "PII = max(left overlap,\nright overlap)",
        "Dark PII",
        "Light PII",
    }
    assert expected_pii_text <= calls["all_text"]
    assert not any(
        "DPP index" in text or "DPPI" in text
        for text in calls["all_text"]
    )
    assert "Unit-area overlap" not in calls["all_text"]
    population_axis = calls["population_shift_profile_axis"]
    assert calls["panel_b_schematic_axis"].get_title() == ""
    assert population_axis.get_title() == ""
    assert population_axis.get_xlabel() == "Circular shift"
    assert population_axis.get_ylabel() == "Norm. overlap"
    axis_bounds = calls["axis_bounds"]
    panel_a_bounds = axis_bounds[
        "Path-invariant progression tuning across dark and light"
    ]
    panel_b_bounds = axis_bounds["Tuning shift across dark and light"]
    panel_c_bounds = axis_bounds["Path-invariance across dark and light"]
    assert panel_c_bounds == pytest.approx(
        calls["legacy_panel_c_axis"].get_position().bounds
    )
    panel_b_profile_bounds = population_axis.get_position().bounds
    panel_b_schematic_bounds = calls[
        "panel_b_schematic_axis"
    ].get_position().bounds
    downstream_titles = {
        "Achievable tuning stability",
        "Full-path stability (split-half r > 0.5)",
        "Full-path stability (no stability filter)",
        "Segment-matched achievable stability",
        "Segment tuning stability and response ratio",
        "Unaveraged segment tuning stability",
        "Achievable segment tuning stability",
    }
    assert downstream_titles.isdisjoint(axis_bounds)
    assert panel_a_bounds[2] == pytest.approx(
        figure._figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[1]
    )
    assert figure.PANEL_BC_ROW_WIDTH_RATIOS == (1.0, 1.0)
    assert figure.PANEL_B_ROW_WSPACE == pytest.approx(0.035)
    assert panel_a_bounds[1] > panel_b_bounds[1]
    assert panel_b_bounds[1] + panel_b_bounds[3] < panel_a_bounds[1]
    assert panel_b_bounds[1] == pytest.approx(panel_c_bounds[1])
    assert panel_b_bounds[3] == pytest.approx(panel_c_bounds[3])
    assert panel_b_bounds[0] + panel_b_bounds[2] < panel_c_bounds[0]
    assert panel_b_bounds[2] == pytest.approx(panel_c_bounds[2])
    assert panel_b_bounds[0] == pytest.approx(panel_a_bounds[0])
    assert panel_c_bounds[0] + panel_c_bounds[2] == pytest.approx(
        panel_a_bounds[0] + panel_a_bounds[2]
    )
    assert (
        panel_b_schematic_bounds[0] + panel_b_schematic_bounds[2]
        < panel_b_profile_bounds[0]
    )
    assert panel_b_profile_bounds[2] / panel_b_schematic_bounds[2] == (
        pytest.approx(0.280 / 0.2624)
    )
    panel_b_child_gutter = (
        panel_b_profile_bounds[0]
        - panel_b_schematic_bounds[0]
        - panel_b_schematic_bounds[2]
    ) / panel_b_bounds[2]
    assert panel_b_child_gutter == pytest.approx(0.170, abs=1e-12)
    assert panel_b_schematic_bounds[1] < panel_b_profile_bounds[1]
    assert panel_b_schematic_bounds[3] > panel_b_profile_bounds[3]
    panel_b_children_center = 0.5 * (
        panel_b_schematic_bounds[0]
        + panel_b_profile_bounds[0]
        + panel_b_profile_bounds[2]
    )
    assert panel_b_children_center == pytest.approx(
        panel_b_bounds[0] + 0.5 * panel_b_bounds[2]
    )
    for child_bounds in [panel_b_schematic_bounds, panel_b_profile_bounds]:
        assert child_bounds[0] >= panel_b_bounds[0]
        assert child_bounds[1] >= panel_b_bounds[1]
        assert (
            child_bounds[0] + child_bounds[2]
            <= panel_b_bounds[0] + panel_b_bounds[2] + 1e-9
        )
        assert (
            child_bounds[1] + child_bounds[3]
            <= panel_b_bounds[1] + panel_b_bounds[3] + 1e-9
        )


def test_main_uses_non_colliding_default_output_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    datasets = [("L14", "20240611", "08_r4")]

    monkeypatch.setattr(figure, "get_processed_datasets", lambda: datasets)

    def fake_make_figure_2(**kwargs: object) -> Path:
        calls.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(figure, "make_figure_2", fake_make_figure_2)

    figure.main([])

    assert calls["output_path"] == Path(
        f"paper_figures/output/figure_2.{figure._figure_2.DEFAULT_OUTPUT_FORMAT}"
    )
    assert calls["datasets"] is datasets
    assert calls["panel_example_cache_dir"] == Path("paper_figures/output/cache")
    assert calls["panel_tuning_similarity_cache_dir"] == Path(
        "paper_figures/output/cache"
    )
    assert calls["min_epoch_movement_firing_rate_hz"] == pytest.approx(0.5)
    assert calls["min_segment_mean_firing_rate_hz"] == pytest.approx(0.5)
