from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_1 as figure_1_module
from v1ca1.paper_figures.figure_1 import (
    CYCLE_ARROW_LINEWIDTH,
    CYCLE_ARROW_MUTATION_SCALE,
    CYCLE_ARROW_SPECS,
    CYCLE_TRAJECTORY_LAYOUT,
    BOTTOM_ROW_PANEL_WSPACE,
    DEFAULT_ASSET_DIR,
    DEFAULT_BEHAVIOR_ASSET_NAME,
    DEFAULT_DARK_EPOCH,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_BOTTOM_ROW_HEIGHT_MM,
    DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION,
    DEFAULT_HEATMAP_HEIGHT_MM,
    DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM,
    DEFAULT_PANEL_E_WIDTH_FRACTION,
    DEFAULT_PANEL_F_WIDTH_FRACTION,
    DEFAULT_PANEL_G_WIDTH_FRACTION,
    DEFAULT_PANEL_H_WIDTH_FRACTION,
    DEFAULT_REGIONS,
    DEFAULT_TOP_ROW_HEIGHT_MM,
    DECODING_COMPARISON_RELATIVE_DIR,
    DECODING_CROSS_TRAJECTORY_COMPARISONS,
    DECODING_EXAMPLE_TEST_TRAJECTORIES,
    DECODING_EXAMPLE_TRAIN_TRAJECTORY,
    DECODING_MEDIAN_LABEL_FONTSIZE,
    DECODING_MEDIAN_LABEL_X_OFFSET,
    DECODING_SCHEMATIC_HEIGHT,
    DECODING_SCHEMATIC_WIDTH,
    DECODING_SCHEMATIC_Y,
    DECODING_TRAIN_LABEL_Y,
    DECODING_TRAIN_SCHEMATIC_CENTER_X,
    DECODING_XTICK_LABEL_FONTSIZE,
    DECODING_YLABEL_FONTSIZE,
    DELTA_LOG_LIKELIHOOD_AXIS_LABEL,
    ENCODING_COMPARISON_RELATIVE_DIR,
    ENCODING_COMPARISON_MIN_SPIKES,
    ENCODING_MIN_TUNING_STABILITY_CORRELATION,
    ENCODING_DPP_COMPARISON_COLORS,
    ENCODING_DPP_COMPARISONS,
    HEATMAP_COLORBAR_LABELPAD,
    HEATMAP_COLORBAR_LABEL_FONTSIZE,
    HEATMAP_COLORBAR_PAD,
    HEATMAP_ORDER_LABEL_OFFSET,
    HEATMAP_PATH_LABEL_OFFSET,
    HEATMAP_TUNING_LABEL_OFFSET,
    MOVEMENT_AXIS_ARROW_MARGIN,
    MOVEMENT_AXIS_Y,
    MOTOR_DELTA_METRIC,
    MOTOR_MIN_TUNING_STABILITY_CORRELATION,
    MOTOR_NESTED_CV_RELATIVE_DIR,
    NEURON_SCALE_BAR_COUNT,
    PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS,
    PANEL_DARK_LIGHT_VISUAL_LABEL_COLORS,
    PANEL_E_EXAMPLES,
    PANEL_E_AXIS_LABEL_FONTSIZE,
    PANEL_E_CACHE_VERSION,
    PANEL_E_FR_TRAJECTORY_PAIRS,
    PANEL_E_RASTER_TICK_MARKEREDGEWIDTH,
    PANEL_E_RASTER_TICK_MARKERSIZE,
    PANEL_E_RASTER_TRAJECTORY_LAYOUT,
    PANEL_E_TICK_LABEL_FONTSIZE,
    PANEL_E_TRAJECTORY_COLORS,
    PANEL_H_DECODING_ANIMALS,
    PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    PANEL_D_CACHE_VERSION,
    PANEL_D_FIRING_RATE_NORMALIZATION,
    PANEL_D_HEATMAP_CMAP,
    PANEL_D_LINEAR_POSITION_ORIENTATION,
    PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_D_TRAJECTORY_TYPES,
    SCHEMATIC_COLORS,
    STABILITY_AXIS_LABEL_FONTSIZE,
    STABILITY_LEGEND_FONTSIZE,
    STABILITY_TICK_LABEL_FONTSIZE,
    STABILITY_TRAJECTORY_LAYOUT,
    STABILITY_TABLE_RELATIVE_PATH,
    TASK_PROGRESSION_SEGMENT_BOUNDARIES,
    TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR,
    TASK_PROGRESSION_SEGMENT_BOUNDARY_LINEWIDTH,
    TASK_PROGRESSION_XLABEL,
    TRAJECTORY_TYPES,
    add_centered_axis_text,
    add_centered_below_axis_text,
    build_normalized_position_bins,
    build_pooled_panel_values,
    build_pooled_panel_values_and_unit_order,
    build_panel_d_cache_metadata,
    build_panel_d_cache_path,
    build_panel_e_cache_metadata,
    build_panel_e_cache_path,
    build_output_path,
    build_unit_keys,
    build_zero_including_histogram_bins,
    draw_panel_a_assets,
    draw_behavior_task_design_panel,
    draw_panel_a_anatomy_assets,
    draw_neuron_scale_bar,
    draw_visual_stimuli_schematic,
    draw_w_track_cycle_panel,
    find_encoding_summary_path,
    find_motor_nested_cv_path,
    filter_datasets_by_animals,
    filter_tuning_curve_units,
    format_place_bin_size_token,
    get_cross_trajectory_decoding_tsd_paths,
    get_decoding_comparison_dir,
    get_dark_epoch,
    get_encoding_comparison_dir,
    get_figure_1_asset_path,
    get_movement_arrow_points,
    get_motor_nested_cv_dir,
    get_stability_table_path,
    load_panel_asset_image,
    load_dark_epoch_stability_table,
    load_decoding_absolute_error_table,
    load_encoding_delta_table,
    load_motor_delta_table,
    load_or_compute_panel_e_example_data,
    orient_panel_e_task_progression,
    plot_decoding_error_panel,
    plot_dark_heatmap_regions,
    plot_dark_light_example_panel,
    plot_encoding_delta_panel,
    plot_panel_e_examples,
    plot_pooled_heatmap_grid,
    plot_motor_delta_panel,
    plot_stability_panel,
    save_panel_d_cache,
    load_panel_d_cache,
    load_panel_d_cache_payload,
    load_panel_d_ordered_unit_keys,
    save_panel_e_cache,
    load_panel_e_cache,
    get_trajectory_endpoint_labels,
    parse_arguments,
    parse_dataset_id,
    select_units_by_movement_firing_rate,
    select_units_by_tuning_stability,
)


def test_get_dark_epoch_uses_l15_override() -> None:
    assert get_dark_epoch("L14", "20240611") == DEFAULT_DARK_EPOCH
    assert get_dark_epoch("L15", "20241121") == "10_r5"
    assert get_dark_epoch("L14", "20240611", "10_r5") == "10_r5"


def test_parse_dataset_id_requires_animal_and_date() -> None:
    assert parse_dataset_id("L14:20240611") == ("L14", "20240611", "08_r4")
    assert parse_dataset_id("L15:20241121:10_r5") == ("L15", "20241121", "10_r5")

    with pytest.raises(argparse.ArgumentTypeError, match="animal:date"):
        parse_dataset_id("L14")


def test_filter_datasets_by_animals_normalizes_and_keeps_requested_animals() -> None:
    datasets = [
        ("L12", "20240421"),
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "02_r1", "10_r5", "07_s4"),
        ("L19", "20250930", "08_r4"),
    ]

    filtered = filter_datasets_by_animals(datasets, PANEL_H_DECODING_ANIMALS)

    assert filtered == [
        ("L12", "20240421", "08_r4"),
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L19", "20250930", "08_r4"),
    ]


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "figure_1", "svg") == Path(
        "paper_figures/figure_1.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "figure_1", "jpg")


def test_get_figure_1_asset_path_uses_asset_directory() -> None:
    assert get_figure_1_asset_path(Path("assets"), "probe.jpg") == Path(
        "assets/probe.jpg"
    )


def test_svg_asset_loader_uses_same_stem_raster_export(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    svg_path = tmp_path / "histology.svg"
    png_path = tmp_path / "histology.png"
    svg_path.write_text("<svg viewBox='0 0 1 1'></svg>", encoding="utf-8")
    mpimg.imsave(png_path, np.ones((2, 2, 3)))

    image = load_panel_asset_image(svg_path)

    assert image.shape[0] == 2
    assert image.shape[1] == 2


def test_build_normalized_position_bins_spans_zero_to_one() -> None:
    bins = build_normalized_position_bins(4)

    assert np.allclose(bins, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_panel_d_cache_path_is_descriptive() -> None:
    metadata = build_panel_d_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        position_bin_count=100,
        position_offset=5,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    cache_path = build_panel_d_cache_path(Path("paper_figures/output/cache"), metadata)

    assert metadata["cache_version"] == PANEL_D_CACHE_VERSION
    assert metadata["data_root"] == "/analysis"
    assert metadata["datasets"] == [
        {
            "animal_name": "L14",
            "date": "20240611",
            "dark_epoch": "08_r4",
        }
    ]
    assert tuple(metadata["trajectory_types"]) == PANEL_D_TRAJECTORY_TYPES
    assert metadata["linear_position_orientation"] == "task_progression"
    assert PANEL_D_FIRING_RATE_NORMALIZATION == "unit_max_per_trajectory"
    assert metadata["min_movement_firing_rate_hz"] == pytest.approx(
        PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ
    )
    assert metadata["min_tuning_stability_correlation"] == pytest.approx(
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    )
    assert "firing_rate_normalization" not in metadata
    assert cache_path == Path(
        "paper_figures/output/cache/"
        "figure_1_panel_d_v1_dark08_r4_datasets-L14-20240611-08_r4"
        "_orienttask_progression_minmovefr0p5_minstab0p5"
        "_posbins100_offset5_speed4_sigma1p5_cachev6.npz"
    )


def test_panel_d_cache_roundtrip_validates_metadata(tmp_path: Path) -> None:
    metadata = build_panel_d_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    panels = {}
    for index, order_trajectory in enumerate(PANEL_D_TRAJECTORY_TYPES):
        for plot_trajectory in PANEL_D_TRAJECTORY_TYPES:
            panels[(order_trajectory, plot_trajectory)] = np.full(
                (index + 1, 3),
                index,
                dtype=float,
            )
    ordered_unit_keys_by_trajectory = {
        order_trajectory: np.asarray([f"unit-{index}"], dtype=str)
        for index, order_trajectory in enumerate(PANEL_D_TRAJECTORY_TYPES)
    }
    cache_path = build_panel_d_cache_path(tmp_path, metadata)

    save_panel_d_cache(
        cache_path,
        panels,
        metadata,
        ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
    )
    loaded = load_panel_d_cache(cache_path, metadata)
    loaded_payload = load_panel_d_cache_payload(cache_path, metadata)
    loaded_unit_keys = load_panel_d_ordered_unit_keys(cache_path, metadata)

    assert loaded is not None
    assert loaded_payload is not None
    assert loaded_unit_keys is not None
    for key, expected in panels.items():
        assert np.array_equal(loaded[key], expected)
    for key, expected in ordered_unit_keys_by_trajectory.items():
        assert np.array_equal(loaded_unit_keys[key], expected)

    stale_metadata = dict(metadata)
    stale_metadata["position_bin_count"] = 4
    assert load_panel_d_cache(cache_path, stale_metadata) is None


def test_panel_d_selects_units_by_minimum_movement_firing_rate() -> None:
    movement_firing_rates = {
        1: 0.49,
        2: 0.5,
        3: 2.0,
    }

    assert np.array_equal(
        select_units_by_movement_firing_rate(movement_firing_rates, 0.5),
        [2, 3],
    )
    assert np.array_equal(
        select_units_by_movement_firing_rate(movement_firing_rates, None),
        [1, 2, 3],
    )


def test_panel_d_selects_units_stable_on_at_least_one_trajectory() -> None:
    pd = pytest.importorskip("pandas")
    stability_table = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3],
            "stability_correlation": [0.1, 0.5, 0.49, 0.2, np.nan],
        }
    )

    assert np.array_equal(
        select_units_by_tuning_stability(stability_table, 0.5),
        [1],
    )
    assert np.array_equal(
        select_units_by_tuning_stability(stability_table, None),
        [1, 2, 3],
    )


def test_panel_d_filters_tuning_curve_units_by_inclusion_list() -> None:
    xr = pytest.importorskip("xarray")
    tuning_curve = xr.DataArray(
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        dims=("unit", "linpos"),
        coords={
            "unit": np.asarray([10, 20, 30]),
            "linpos": np.asarray([0.25, 0.75]),
        },
    )

    filtered = filter_tuning_curve_units(tuning_curve, np.asarray([30, 10]))

    assert np.array_equal(filtered.coords["unit"].values, [10, 30])
    assert np.array_equal(filtered.values, [[1.0, 2.0], [5.0, 6.0]])


def test_panel_d_supports_per_and_across_trajectory_normalization() -> None:
    xr = pytest.importorskip("xarray")

    def make_curve(values: list[list[float]]):
        value_array = np.asarray(values, dtype=float)
        return xr.DataArray(
            value_array,
            dims=("unit", "linpos"),
            coords={
                "unit": np.asarray([1], dtype=int),
                "linpos": np.arange(value_array.shape[1], dtype=float),
            },
        )

    even_values = {
        PANEL_D_TRAJECTORY_TYPES[0]: [[1.0, 2.0]],
        PANEL_D_TRAJECTORY_TYPES[1]: [[4.0, 0.0]],
        PANEL_D_TRAJECTORY_TYPES[2]: [[0.0, 8.0]],
        PANEL_D_TRAJECTORY_TYPES[3]: [[2.0, 6.0]],
    }
    curve_set = {
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "odd_curves": {
            trajectory_type: make_curve([[1.0, 0.0]])
            for trajectory_type in PANEL_D_TRAJECTORY_TYPES
        },
        "even_curves": {
            trajectory_type: make_curve(values)
            for trajectory_type, values in even_values.items()
        },
    }

    panels = build_pooled_panel_values(
        [curve_set],
        position_bin_count=2,
        trajectory_types=PANEL_D_TRAJECTORY_TYPES,
    )
    panels_with_order, ordered_unit_keys = build_pooled_panel_values_and_unit_order(
        [curve_set],
        position_bin_count=2,
        trajectory_types=PANEL_D_TRAJECTORY_TYPES,
    )
    order_trajectory = PANEL_D_TRAJECTORY_TYPES[0]

    assert np.array_equal(
        panels_with_order[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[0])],
        panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[0])],
    )
    assert np.array_equal(
        ordered_unit_keys[order_trajectory],
        np.asarray(["L14:20240611:v1:1"], dtype=object),
    )
    assert np.allclose(
        panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[0])],
        [[1.0 / 2.0, 1.0]],
    )
    assert np.allclose(
        panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[1])],
        [[1.0, 0.0]],
    )
    assert np.allclose(
        panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[2])],
        [[0.0, 1.0]],
    )
    assert np.allclose(
        panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[3])],
        [[2.0 / 6.0, 1.0]],
    )

    across_panels = build_pooled_panel_values(
        [curve_set],
        position_bin_count=2,
        trajectory_types=PANEL_D_TRAJECTORY_TYPES,
        firing_rate_normalization=PANEL_D_ACROSS_TRAJECTORY_FIRING_RATE_NORMALIZATION,
    )

    assert np.allclose(
        across_panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[0])],
        [[1.0 / 8.0, 2.0 / 8.0]],
    )
    assert np.allclose(
        across_panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[1])],
        [[4.0 / 8.0, 0.0]],
    )
    assert np.allclose(
        across_panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[2])],
        [[0.0, 1.0]],
    )
    assert np.allclose(
        across_panels[(order_trajectory, PANEL_D_TRAJECTORY_TYPES[3])],
        [[2.0 / 8.0, 6.0 / 8.0]],
    )


def test_panel_e_cache_path_is_descriptive() -> None:
    metadata = build_panel_e_cache_metadata(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region="v1",
        unit_id=34,
        position_bin_count=50,
        position_offset=5,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    cache_path = build_panel_e_cache_path(Path("paper_figures/output/cache"), metadata)

    assert metadata["cache_version"] == PANEL_E_CACHE_VERSION
    assert metadata["data_root"] == "/analysis"
    assert metadata["unit_id"] == 34
    assert cache_path == Path(
        "paper_figures/output/cache/"
        "figure_1_panel_e_L14-20240611-08_r4-v1-unit34"
        "_posbins50_offset5_speed4_sigma1p5_cachev1.npz"
    )


def test_panel_e_cache_roundtrip_validates_metadata(tmp_path: Path) -> None:
    metadata = build_panel_e_cache_metadata(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region="v1",
        unit_id=34,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    example_data = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "region": "v1",
        "unit_id": 34,
        "raster_positions": {
            trajectory_type: [np.asarray([0.1, 0.2]), np.asarray([], dtype=float)]
            for trajectory_type in TRAJECTORY_TYPES
        },
        "firing_rates": {
            trajectory_type: (
                np.asarray([0.0, 0.5, 1.0]),
                np.asarray([0.0, 1.0, 0.5]),
            )
            for trajectory_type in TRAJECTORY_TYPES
        },
    }
    cache_path = build_panel_e_cache_path(tmp_path, metadata)

    save_panel_e_cache(cache_path, example_data, metadata)
    loaded = load_panel_e_cache(cache_path, metadata)

    assert loaded is not None
    assert loaded["animal_name"] == "L14"
    assert loaded["unit_id"] == 34
    for trajectory_type in TRAJECTORY_TYPES:
        assert len(loaded["raster_positions"][trajectory_type]) == 2
        assert loaded["raster_positions"][trajectory_type][0].tolist() == pytest.approx(
            [0.1, 0.2]
        )
        assert loaded["raster_positions"][trajectory_type][1].size == 0
        position, firing_rate = loaded["firing_rates"][trajectory_type]
        assert position.tolist() == pytest.approx([0.0, 0.5, 1.0])
        assert firing_rate.tolist() == pytest.approx([0.0, 1.0, 0.5])

    stale_metadata = dict(metadata)
    stale_metadata["unit_id"] = 35
    assert load_panel_e_cache(cache_path, stale_metadata) is None


def test_load_or_compute_panel_e_example_data_uses_matching_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = build_panel_e_cache_metadata(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region="v1",
        unit_id=34,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    example_data = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "region": "v1",
        "unit_id": 34,
        "raster_positions": {
            trajectory_type: [np.asarray([0.1])]
            for trajectory_type in TRAJECTORY_TYPES
        },
        "firing_rates": {
            trajectory_type: (
                np.asarray([0.0, 1.0]),
                np.asarray([0.5, 0.25]),
            )
            for trajectory_type in TRAJECTORY_TYPES
        },
    }
    save_panel_e_cache(build_panel_e_cache_path(tmp_path, metadata), example_data, metadata)
    monkeypatch.setattr(
        figure_1_module,
        "load_panel_e_example_data",
        lambda **_kwargs: pytest.fail("Panel E cache was not used."),
    )

    loaded = load_or_compute_panel_e_example_data(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region="v1",
        unit_id=34,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_e_cache_dir=tmp_path,
        refresh_panel_e_cache=False,
    )

    assert loaded["raster_positions"]["center_to_left"][0].tolist() == pytest.approx(
        [0.1]
    )
    position, firing_rate = loaded["firing_rates"]["center_to_left"]
    assert position.tolist() == pytest.approx([0.0, 1.0])
    assert firing_rate.tolist() == pytest.approx([0.5, 0.25])


def test_get_stability_table_path_uses_task_progression_output_location() -> None:
    path = get_stability_table_path(Path("/analysis"), "L14", "20240611")

    assert path == Path("/analysis") / "L14" / "20240611" / STABILITY_TABLE_RELATIVE_PATH


def test_get_motor_nested_cv_dir_uses_task_progression_motor_location() -> None:
    path = get_motor_nested_cv_dir(Path("/analysis"), "L14", "20240611")

    assert path == Path("/analysis") / "L14" / "20240611" / MOTOR_NESTED_CV_RELATIVE_DIR


def test_get_encoding_comparison_dir_uses_task_progression_output_location() -> None:
    path = get_encoding_comparison_dir(Path("/analysis"), "L14", "20240611")

    assert path == (
        Path("/analysis") / "L14" / "20240611" / ENCODING_COMPARISON_RELATIVE_DIR
    )


def test_get_decoding_comparison_dir_uses_task_progression_output_location() -> None:
    path = get_decoding_comparison_dir(Path("/analysis"), "L14", "20240611")

    assert path == (
        Path("/analysis") / "L14" / "20240611" / DECODING_COMPARISON_RELATIVE_DIR
    )


def test_get_cross_trajectory_decoding_tsd_paths_match_decoder_outputs() -> None:
    true_path, decoded_path = get_cross_trajectory_decoding_tsd_paths(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        transfer_family="same_turn_cross_arm",
        encoding_trajectory="center_to_left",
        decoding_trajectory="right_to_center",
    )

    expected_prefix = (
        Path("/analysis")
        / "L14"
        / "20240611"
        / DECODING_COMPARISON_RELATIVE_DIR
        / "v1_08_r4_same_turn_cross_arm_center_to_left_to_right_to_center"
    )
    assert true_path == expected_prefix.with_name(
        f"{expected_prefix.name}_true_tp_cross_traj.npz"
    )
    assert decoded_path == expected_prefix.with_name(
        f"{expected_prefix.name}_decoded_tp_cross_traj.npz"
    )


def test_load_dark_epoch_stability_table_filters_dark_epoch_and_regions(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    table_path = tmp_path / "L15" / "20241121" / STABILITY_TABLE_RELATIVE_PATH
    table_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "animal_name": ["L15", "L15", "L15"],
            "date": ["20241121", "20241121", "20241121"],
            "unit": [1, 2, 3],
            "region": ["v1", "ca1", "s1"],
            "epoch": ["10_r5", "02_r1", "10_r5"],
            "trajectory_type": ["center_to_left", "center_to_left", "center_to_left"],
            "stability_correlation": [0.5, 0.6, 0.7],
        }
    ).to_parquet(table_path)

    table = load_dark_epoch_stability_table(
        data_root=tmp_path,
        datasets=[("L15", "20241121", "10_r5")],
        regions=("v1", "ca1"),
    )

    assert table["unit"].tolist() == [1]
    assert table["epoch"].tolist() == ["10_r5"]


def test_find_motor_nested_cv_path_prefers_zscore_output(tmp_path: Path) -> None:
    data_dir = get_motor_nested_cv_dir(tmp_path, "L14", "20240611")
    data_dir.mkdir(parents=True)
    spline_path = data_dir / "v1_08_r4_nested_lapcv_bin0p05s_spline.nc"
    zscore_path = data_dir / "v1_08_r4_nested_lapcv_bin0p05s_zscore.nc"
    spline_path.write_text("spline", encoding="utf-8")
    zscore_path.write_text("zscore", encoding="utf-8")

    path = find_motor_nested_cv_path(
        data_root=tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
    )

    assert path == zscore_path


def test_load_motor_delta_table_filters_registered_dark_epoch_by_stability(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    xr = pytest.importorskip("xarray")

    data_dir = get_motor_nested_cv_dir(tmp_path, "L14", "20240611")
    data_dir.mkdir(parents=True)
    path = data_dir / "v1_08_r4_nested_lapcv_bin0p05s_zscore.nc"
    xr.Dataset(
        data_vars={
            "pooled_delta_bits_per_spike": (
                ("delta_metric", "unit"),
                np.asarray([[0.1, -0.2]], dtype=float),
            )
        },
        coords={
            "delta_metric": np.asarray([MOTOR_DELTA_METRIC], dtype=str),
            "unit": np.asarray([11, 12], dtype=int),
        },
    ).to_netcdf(path)
    stability_path = get_stability_table_path(tmp_path, "L14", "20240611")
    stability_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L14", "L14"],
            "date": ["20240611", "20240611", "20240611", "20240611"],
            "unit": [11, 11, 12, 13],
            "region": ["v1", "v1", "v1", "v1"],
            "epoch": ["08_r4", "08_r4", "08_r4", "10_r5"],
            "trajectory_type": [
                "right_to_center",
                "center_to_left",
                "center_to_right",
                "center_to_left",
            ],
            "stability_correlation": [0.2, 0.5, 0.49, 0.9],
        }
    ).to_parquet(stability_path)

    table = load_motor_delta_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
    )

    assert MOTOR_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)
    assert table["animal_name"].tolist() == ["L14"]
    assert table["epoch"].tolist() == ["08_r4"]
    assert table["unit"].tolist() == [11]
    assert table["delta_log_likelihood_bits_per_spike"].tolist() == pytest.approx(
        [0.1]
    )


def test_format_place_bin_size_token_matches_encoding_filename_token() -> None:
    assert format_place_bin_size_token(4.0) == "placebin4cm"
    assert format_place_bin_size_token(2.5) == "placebin2p5cm"


def test_find_encoding_summary_path_prefers_placebin_output(tmp_path: Path) -> None:
    data_dir = get_encoding_comparison_dir(tmp_path, "L14", "20240611")
    data_dir.mkdir(parents=True)
    legacy_path = data_dir / "v1_08_r4_cv5_encoding_summary.parquet"
    placebin_path = data_dir / "v1_08_r4_cv5_placebin4cm_encoding_summary.parquet"
    legacy_path.write_text("legacy", encoding="utf-8")
    placebin_path.write_text("placebin", encoding="utf-8")

    path = find_encoding_summary_path(
        data_root=tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        place_bin_size_cm=4.0,
    )

    assert path == placebin_path


def test_load_encoding_delta_table_filters_by_stability_and_negates_saved_columns(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    data_dir = get_encoding_comparison_dir(tmp_path, "L14", "20240611")
    data_dir.mkdir(parents=True)
    path = data_dir / "v1_08_r4_cv5_placebin4cm_encoding_summary.parquet"
    pd.DataFrame(
        {
            "n_spikes": [100, 100],
            "delta_bits_generalized_place_vs_tp": [0.25, -0.5],
            "delta_bits_gtp_vs_tp": [-0.1, 0.2],
        },
        index=pd.Index([11, 12], name="unit"),
    ).to_parquet(path)
    stability_path = get_stability_table_path(tmp_path, "L14", "20240611")
    stability_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L14", "L14"],
            "date": ["20240611", "20240611", "20240611", "20240611"],
            "unit": [11, 11, 12, 13],
            "region": ["v1", "v1", "v1", "v1"],
            "epoch": ["08_r4", "08_r4", "08_r4", "10_r5"],
            "trajectory_type": [
                "right_to_center",
                "center_to_left",
                "center_to_right",
                "center_to_left",
            ],
            "stability_correlation": [0.2, 0.5, 0.49, 0.9],
        }
    ).to_parquet(stability_path)

    table = load_encoding_delta_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        place_bin_size_cm=4.0,
    )

    by_comparison = {
        comparison: rows["delta_log_likelihood_bits_per_spike"].tolist()
        for comparison, rows in table.groupby("comparison", sort=False)
    }
    assert by_comparison["dpp_vs_absolute_place"] == pytest.approx([-0.25])
    assert by_comparison["dpp_vs_absolute_task_progression"] == pytest.approx(
        [0.1]
    )
    assert ENCODING_COMPARISON_MIN_SPIKES == 0
    assert ENCODING_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)
    assert table["unit"].tolist() == [11, 11]
    assert [comparison for comparison, _label, _column in ENCODING_DPP_COMPARISONS] == [
        "dpp_vs_absolute_place",
        "dpp_vs_absolute_task_progression",
    ]


def test_load_decoding_absolute_error_table_reads_sample_level_npz(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    nap = pytest.importorskip("pynapple")
    del pd

    comparison = (
        "same_turn_cross_arm",
        "Same turn\ncross arm",
        "same_turn_cross_arm",
        (("center_to_left", "right_to_center"),),
    )
    true_path, decoded_path = get_cross_trajectory_decoding_tsd_paths(
        data_root=tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        transfer_family="same_turn_cross_arm",
        encoding_trajectory="center_to_left",
        decoding_trajectory="right_to_center",
    )
    true_path.parent.mkdir(parents=True)
    nap.Tsd(
        t=np.asarray([0.0, 1.0, 2.0]),
        d=np.asarray([0.0, 0.5, 1.0]),
        time_units="s",
    ).save(true_path)
    nap.Tsd(
        t=np.asarray([0.0, 1.0, 2.0]),
        d=np.asarray([0.1, 0.25, 0.75]),
        time_units="s",
    ).save(decoded_path)

    table = load_decoding_absolute_error_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        comparisons=(comparison,),
    )

    assert table["comparison"].tolist() == ["same_turn_cross_arm"] * 3
    assert table["absolute_error"].tolist() == pytest.approx([0.1, 0.25, 0.25])
    assert table["encoding_trajectory"].tolist() == ["center_to_left"] * 3
    assert table["decoding_trajectory"].tolist() == ["right_to_center"] * 3


def test_build_unit_keys_disambiguates_sessions_and_regions() -> None:
    unit_keys = build_unit_keys(
        animal_name="L14",
        date="20240611",
        region="ca1",
        units=np.asarray([1, 2]),
    )

    assert unit_keys.tolist() == ["L14:20240611:ca1:1", "L14:20240611:ca1:2"]


def test_default_region_is_v1() -> None:
    args = parse_arguments([])

    assert DEFAULT_REGIONS == ("v1",)
    assert args.region is None
    assert args.output_dir == Path("paper_figures") / "output"
    assert args.asset_dir == DEFAULT_ASSET_DIR
    assert args.panel_d_cache_dir is None
    assert args.refresh_panel_d_cache is False
    assert args.panel_e_cache_dir is None
    assert args.refresh_panel_e_cache is False


def test_default_figure_width_fits_letter_page_with_one_inch_margins() -> None:
    letter_width_with_one_inch_margins_mm = 6.5 * 25.4

    assert DEFAULT_FIGURE_WIDTH_MM <= letter_width_with_one_inch_margins_mm


def test_bottom_row_width_fractions_reserve_panel_e_space() -> None:
    assert DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION == pytest.approx(0.7)
    assert DEFAULT_PANEL_E_WIDTH_FRACTION == pytest.approx(0.3)
    assert BOTTOM_ROW_PANEL_WSPACE == pytest.approx(0.05)
    assert (
        DEFAULT_HEATMAP_PANEL_WIDTH_FRACTION + DEFAULT_PANEL_E_WIDTH_FRACTION
    ) == pytest.approx(1.0)


def test_final_row_width_fractions_make_panels_f_g_h_equal() -> None:
    assert DEFAULT_PANEL_F_WIDTH_FRACTION == pytest.approx(1.0 / 3.0)
    assert DEFAULT_PANEL_G_WIDTH_FRACTION == pytest.approx(1.0 / 3.0)
    assert DEFAULT_PANEL_H_WIDTH_FRACTION == pytest.approx(1.0 / 3.0)
    assert (
        DEFAULT_PANEL_F_WIDTH_FRACTION
        + DEFAULT_PANEL_G_WIDTH_FRACTION
        + DEFAULT_PANEL_H_WIDTH_FRACTION
    ) == pytest.approx(1.0)


def test_panel_d_colorbar_label_size_uses_requested_scale() -> None:
    assert HEATMAP_COLORBAR_LABEL_FONTSIZE == pytest.approx(4.9)
    assert HEATMAP_COLORBAR_PAD == pytest.approx(0.001)
    assert HEATMAP_COLORBAR_LABELPAD == pytest.approx(0)
    assert HEATMAP_TUNING_LABEL_OFFSET == pytest.approx(-0.004)
    assert HEATMAP_ORDER_LABEL_OFFSET == pytest.approx(0.007)
    assert HEATMAP_PATH_LABEL_OFFSET == pytest.approx(0.020)


def test_dark_light_visual_color_configuration_swaps_right_arm_trajectories() -> None:
    assert PANEL_DARK_LIGHT_VISUAL_LABEL_COLORS == {
        "A": figure_1_module.VISUAL_CONDITION_COLORS["02_r1"],
        "B": figure_1_module.VISUAL_CONDITION_COLORS["06_r3"],
    }
    assert PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS == {
        "center_to_right": {
            "02_r1": figure_1_module.VISUAL_CONDITION_COLORS["06_r3"],
            "06_r3": figure_1_module.VISUAL_CONDITION_COLORS["02_r1"],
        },
        "right_to_center": {
            "02_r1": figure_1_module.VISUAL_CONDITION_COLORS["06_r3"],
            "06_r3": figure_1_module.VISUAL_CONDITION_COLORS["02_r1"],
        },
    }


def test_plot_dark_light_example_panel_uses_visual_color_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import v1ca1.paper_figures.figure_3 as figure_3_source

    observed: dict[str, object] = {}

    def fake_load_panel_a_example_data(**kwargs: object) -> dict[str, object]:
        observed["load_kwargs"] = kwargs
        return {"example": True}

    def fake_plot_panel_a_example(
        _ax: object,
        example: dict[str, object],
        **kwargs: object,
    ) -> None:
        observed["example"] = example
        observed["plot_kwargs"] = kwargs

    monkeypatch.setattr(
        figure_3_source,
        "load_panel_a_example_data",
        fake_load_panel_a_example_data,
    )
    monkeypatch.setattr(
        figure_3_source,
        "plot_panel_a_example",
        fake_plot_panel_a_example,
    )

    fig, ax = plt.subplots()
    plot_dark_light_example_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_example_cache_dir=None,
        refresh_panel_example_cache=False,
    )

    assert observed["example"] == {"example": True}
    assert observed["plot_kwargs"] == {
        "visual_label_colors": PANEL_DARK_LIGHT_VISUAL_LABEL_COLORS,
        "trajectory_epoch_color_overrides": PANEL_DARK_LIGHT_RIGHT_ARM_EPOCH_COLORS,
    }
    plt.close(fig)


def test_panel_e_example_configuration_uses_requested_cells_and_layout() -> None:
    assert PANEL_E_EXAMPLES == (
        ("L14", "20240611", "08_r4", "v1", 34),
        ("L15", "20241121", "10_r5", "v1", 473),
    )
    assert PANEL_E_RASTER_TRAJECTORY_LAYOUT == (
        ("center_to_left", "center_to_right"),
        ("right_to_center", "left_to_center"),
    )
    assert PANEL_E_FR_TRAJECTORY_PAIRS == (
        ("center_to_left", "right_to_center"),
        ("center_to_right", "left_to_center"),
    )
    assert STABILITY_TRAJECTORY_LAYOUT == PANEL_E_RASTER_TRAJECTORY_LAYOUT


def test_orient_panel_e_task_progression_flips_inbound_trajectories() -> None:
    nap = pytest.importorskip("pynapple")

    outbound = nap.Tsd(
        t=np.asarray([0.0, 1.0, 2.0]),
        d=np.asarray([0.0, 0.25, 1.0]),
        time_units="s",
    )
    inbound = nap.Tsd(
        t=np.asarray([0.0, 1.0, 2.0]),
        d=np.asarray([0.0, 0.25, 1.0]),
        time_units="s",
    )

    outbound_oriented = orient_panel_e_task_progression(outbound, "center_to_left")
    inbound_oriented = orient_panel_e_task_progression(inbound, "left_to_center")

    assert np.asarray(outbound_oriented.d).tolist() == pytest.approx([0.0, 0.25, 1.0])
    assert np.asarray(inbound_oriented.d).tolist() == pytest.approx([1.0, 0.75, 0.0])


def test_default_figure_height_is_shorter_than_previous_layout() -> None:
    previous_height_mm = 182.0
    figure_height_mm = (
        DEFAULT_TOP_ROW_HEIGHT_MM
        + DEFAULT_HEATMAP_HEIGHT_MM
        + DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM
        + DEFAULT_BOTTOM_ROW_HEIGHT_MM
    )

    assert DEFAULT_MIDDLE_TO_FINAL_ROW_SPACER_MM == pytest.approx(0.5)
    assert figure_height_mm == pytest.approx(154.5)
    assert figure_height_mm < previous_height_mm


def test_cycle_panel_uses_square_task_trajectory_layout() -> None:
    assert CYCLE_TRAJECTORY_LAYOUT == (
        ("left_to_center", (0.58, 0.70, 0.26, 0.27)),
        ("center_to_right", (0.58, 0.28, 0.26, 0.27)),
        ("right_to_center", (0.16, 0.28, 0.26, 0.27)),
        ("center_to_left", (0.16, 0.70, 0.26, 0.27)),
    )


def test_cycle_panel_arrows_close_square_sides() -> None:
    assert CYCLE_ARROW_SPECS == (
        ((0.71, 0.70), (0.71, 0.55), 0.0),
        ((0.58, 0.415), (0.42, 0.415), 0.0),
        ((0.29, 0.55), (0.29, 0.70), 0.0),
        ((0.42, 0.835), (0.58, 0.835), 0.0),
    )
    assert CYCLE_ARROW_LINEWIDTH == pytest.approx(1.08)
    assert CYCLE_ARROW_MUTATION_SCALE == pytest.approx(12.6)


def test_panel_d_trajectory_order_uses_requested_sequence() -> None:
    assert PANEL_D_TRAJECTORY_TYPES == (
        "right_to_center",
        "center_to_left",
        "left_to_center",
        "center_to_right",
    )
    assert PANEL_D_LINEAR_POSITION_ORIENTATION == "task_progression"
    assert PANEL_D_HEATMAP_CMAP == "viridis"
    assert PANEL_D_FIRING_RATE_NORMALIZATION == "unit_max_per_trajectory"
    assert PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ == pytest.approx(0.5)
    assert PANEL_D_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)


def test_trajectory_axis_labels_mark_center_and_side() -> None:
    assert get_trajectory_endpoint_labels("center_to_left") == ("C", "L")
    assert get_trajectory_endpoint_labels("left_to_center") == ("C", "L")
    assert get_trajectory_endpoint_labels("center_to_right") == ("C", "R")
    assert get_trajectory_endpoint_labels("right_to_center") == ("C", "R")


def test_task_progression_axis_labels_mark_start_and_end() -> None:
    assert get_trajectory_endpoint_labels(
        "right_to_center",
        axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
    ) == ("R", "C")
    assert get_trajectory_endpoint_labels(
        "center_to_left",
        axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
    ) == ("C", "L")
    assert get_trajectory_endpoint_labels(
        "left_to_center",
        axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
    ) == ("L", "C")
    assert get_trajectory_endpoint_labels(
        "center_to_right",
        axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
    ) == ("C", "R")


def test_movement_axis_arrow_points_follow_behavior_direction() -> None:
    outbound_start, outbound_end = get_movement_arrow_points("center_to_left")
    inbound_start, inbound_end = get_movement_arrow_points("left_to_center")

    assert outbound_start == pytest.approx((MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y))
    assert outbound_end == pytest.approx((1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y))
    assert inbound_start == pytest.approx((1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y))
    assert inbound_end == pytest.approx((MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y))
    assert outbound_start[0] < outbound_end[0]
    assert inbound_start[0] > inbound_end[0]


def test_task_progression_movement_axis_is_left_to_right() -> None:
    start, end = get_movement_arrow_points(
        "right_to_center",
        axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
    )

    assert start == pytest.approx((MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y))
    assert end == pytest.approx((1.0 - MOVEMENT_AXIS_ARROW_MARGIN, MOVEMENT_AXIS_Y))
    assert start[0] < end[0]


def test_draw_neuron_scale_bar_uses_data_height() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_ylim(250, 0)
    draw_neuron_scale_bar(ax)

    assert NEURON_SCALE_BAR_COUNT == 100
    assert ax.lines[0].get_ydata().tolist() == [180.0, 80.0]
    assert len(ax.lines) == 1
    assert ax.texts[-1].get_text() == "100 neurons"
    plt.close(fig)


def test_add_centered_axis_text_places_figure_text() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(ncols=2)
    fig.canvas.draw()
    text = add_centered_axis_text(fig, axes, "Tuning")

    assert text.get_text() == "Tuning"
    assert text.get_fontsize() == 9.0
    assert text.figure is fig
    plt.close(fig)


def test_add_centered_below_axis_text_places_figure_text() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(ncols=2)
    fig.canvas.draw()
    text = add_centered_below_axis_text(fig, axes, TASK_PROGRESSION_XLABEL)

    boxes = [ax.get_position() for ax in axes]
    expected_x = (min(box.x0 for box in boxes) + max(box.x1 for box in boxes)) / 2

    assert text.get_text() == TASK_PROGRESSION_XLABEL
    assert text.get_position()[0] == pytest.approx(expected_x)
    assert text.figure is fig
    plt.close(fig)


def test_plot_pooled_heatmap_grid_adds_segment_boundary_lines() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(TRAJECTORY_TYPES),
        ncols=len(TRAJECTORY_TYPES),
    )
    panels = {
        (order_trajectory, plot_trajectory): np.ones((3, 5), dtype=float)
        for order_trajectory in TRAJECTORY_TYPES
        for plot_trajectory in TRAJECTORY_TYPES
    }

    image = plot_pooled_heatmap_grid(axes, panels)

    assert image is not None
    assert image.get_cmap().name == "viridis"
    for ax in axes.ravel():
        boundary_lines = ax.lines[-len(TASK_PROGRESSION_SEGMENT_BOUNDARIES) :]
        assert [line.get_xdata()[0] for line in boundary_lines] == pytest.approx(
            TASK_PROGRESSION_SEGMENT_BOUNDARIES
        )
        assert [line.get_color() for line in boundary_lines] == [
            TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR
        ] * len(TASK_PROGRESSION_SEGMENT_BOUNDARIES)
        assert [line.get_linewidth() for line in boundary_lines] == pytest.approx(
            [TASK_PROGRESSION_SEGMENT_BOUNDARY_LINEWIDTH]
            * len(TASK_PROGRESSION_SEGMENT_BOUNDARIES)
        )
    for ax in axes[-1, :]:
        assert ax.get_xticks() == pytest.approx([0.0, 1.0])
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1"]
        assert ax.get_xlabel() == ""
    for ax in axes[:-1, :].ravel():
        assert ax.get_xticks().size == 0
        assert ax.get_xlabel() == ""
    plt.close(fig)


def test_plot_dark_heatmap_regions_uses_matching_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = build_panel_d_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    panels = {
        (order_trajectory, plot_trajectory): np.ones((2, 3), dtype=float)
        for order_trajectory in PANEL_D_TRAJECTORY_TYPES
        for plot_trajectory in PANEL_D_TRAJECTORY_TYPES
    }
    save_panel_d_cache(build_panel_d_cache_path(tmp_path, metadata), panels, metadata)

    monkeypatch.setattr(
        figure_1_module,
        "compute_dark_epoch_tuning_curves",
        lambda **_kwargs: pytest.fail("Panel D cache was not used."),
    )
    observed = {}

    def _fake_plot_pooled_heatmap_grid(_axes, cached_panels, **kwargs):
        observed["panels"] = cached_panels
        observed["kwargs"] = kwargs
        return None

    monkeypatch.setattr(
        figure_1_module,
        "plot_pooled_heatmap_grid",
        _fake_plot_pooled_heatmap_grid,
    )

    fig, axes = plt.subplots(nrows=4, ncols=4)
    plot_dark_heatmap_regions(
        np.asarray(axes, dtype=object),
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_d_cache_dir=tmp_path,
    )

    assert observed["panels"] is not None
    assert observed["kwargs"]["trajectory_types"] == PANEL_D_TRAJECTORY_TYPES
    assert observed["kwargs"]["axis_orientation"] == PANEL_D_LINEAR_POSITION_ORIENTATION
    assert observed["kwargs"]["cmap"] == PANEL_D_HEATMAP_CMAP
    for key in panels:
        assert np.array_equal(observed["panels"][key], panels[key])
    plt.close(fig)


def test_draw_panel_a_assets_places_probe_and_behavior_left_of_histology(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    probe_path = tmp_path / "probe.png"
    behavior_path = tmp_path / "behavior.png"
    histology_path = tmp_path / "histology.png"
    mpimg.imsave(probe_path, np.ones((4, 6, 3)))
    mpimg.imsave(behavior_path, np.ones((5, 4, 3)))
    mpimg.imsave(histology_path, np.ones((5, 6, 3)))

    fig, ax = plt.subplots()
    draw_panel_a_assets(
        ax,
        asset_dir=tmp_path,
        probe_asset_name="probe.png",
        behavior_asset_name="behavior.png",
        histology_asset_name="histology.png",
    )

    assert DEFAULT_BEHAVIOR_ASSET_NAME == "behavior.png"
    assert len(ax.child_axes) == 3
    probe_ax, behavior_ax, histology_ax = ax.child_axes
    assert probe_ax.get_position().x0 < histology_ax.get_position().x0
    assert behavior_ax.get_position().x0 < histology_ax.get_position().x0
    assert behavior_ax.get_position().y0 < probe_ax.get_position().y0
    probe_center = probe_ax.get_position().x0 + probe_ax.get_position().width / 2
    behavior_center = (
        behavior_ax.get_position().x0 + behavior_ax.get_position().width / 2
    )
    assert probe_center == pytest.approx(behavior_center)
    assert behavior_ax.get_position().y0 == pytest.approx(
        histology_ax.get_position().y0
    )
    assert behavior_ax.get_position().width == pytest.approx(
        0.48 * ax.get_position().width
    )
    assert behavior_ax.get_position().height == pytest.approx(
        0.76 * ax.get_position().height
    )
    assert histology_ax.get_position().height > probe_ax.get_position().height
    assert histology_ax.get_position().width > probe_ax.get_position().width
    assert histology_ax.get_position().height > ax.get_position().height
    assert histology_ax.get_position().width > 0.85 * ax.get_position().width
    assert histology_ax.get_position().x0 - ax.get_position().x0 > (
        0.15 * ax.get_position().width
    )
    probe_image = ax.child_axes[0].images[0].get_array()
    assert probe_image.shape[:2] == (6, 4)
    plt.close(fig)


def test_draw_panel_a_anatomy_assets_places_probe_left_of_histology(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    probe_path = tmp_path / "probe.png"
    histology_path = tmp_path / "histology.png"
    mpimg.imsave(probe_path, np.ones((4, 6, 3)))
    mpimg.imsave(histology_path, np.ones((5, 6, 3)))

    fig, ax = plt.subplots()
    draw_panel_a_anatomy_assets(
        ax,
        asset_dir=tmp_path,
        probe_asset_name="probe.png",
        histology_asset_name="histology.png",
    )

    assert len(ax.child_axes) == 2
    probe_ax, histology_ax = ax.child_axes
    assert probe_ax.get_position().x0 < histology_ax.get_position().x0
    assert histology_ax.get_position().height > probe_ax.get_position().height
    assert histology_ax.get_position().width > probe_ax.get_position().width
    probe_image = probe_ax.images[0].get_array()
    assert probe_image.shape[:2] == (6, 4)
    plt.close(fig)


def test_draw_behavior_task_design_panel_places_behavior_left_of_cycle(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    behavior_path = tmp_path / "behavior.png"
    mpimg.imsave(behavior_path, np.ones((5, 4, 3)))

    fig, ax = plt.subplots()
    draw_behavior_task_design_panel(
        ax,
        asset_dir=tmp_path,
        behavior_asset_name="behavior.png",
    )

    assert len(ax.child_axes) == 2
    behavior_ax, cycle_ax = ax.child_axes
    assert behavior_ax.get_position().x0 < cycle_ax.get_position().x0
    assert behavior_ax.images
    assert len(cycle_ax.child_axes) == 4
    plt.close(fig)


def test_draw_w_track_cycle_panel_adds_four_inset_schematics() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_w_track_cycle_panel(ax)

    assert len(ax.child_axes) == 4
    arrow_patches = [
        text.arrow_patch
        for text in ax.texts
        if getattr(text, "arrow_patch", None) is not None
    ]
    assert len(arrow_patches) == 4
    assert all(
        patch.get_mutation_scale() == pytest.approx(CYCLE_ARROW_MUTATION_SCALE)
        for patch in arrow_patches
    )
    assert all(
        patch.get_linewidth() == pytest.approx(CYCLE_ARROW_LINEWIDTH)
        for patch in arrow_patches
    )
    visible_text = [text.get_text() for text in ax.texts if text.get_text()]
    assert visible_text == ["L", "C", "R", "Visual stimuli"]
    plt.close(fig)


def test_draw_visual_stimuli_schematic_matches_reference_layout() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    fig, ax = plt.subplots()
    draw_visual_stimuli_schematic(ax)

    rectangles = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    visual_stimulus_color = to_rgba(SCHEMATIC_COLORS["visual_stimulus"])[:3]
    yellow_monitor_rectangles = [
        patch
        for patch in rectangles
        if patch.get_facecolor()[:3] == pytest.approx(visual_stimulus_color)
    ]
    screen_rectangles = [
        patch
        for patch in rectangles
        if patch.get_y() == pytest.approx(0.015)
        and patch.get_height() == pytest.approx(0.105)
        and patch.get_edgecolor()[3] > 0.0
    ]
    black_screen_rectangles = [
        patch
        for patch in screen_rectangles
        if patch.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
    ]
    dot_screen_rectangles = [
        patch
        for patch in screen_rectangles
        if patch.get_facecolor()[:3] == pytest.approx((0.65, 0.65, 0.65))
    ]
    track_polygons = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    dot_ellipses = [patch for patch in ax.patches if isinstance(patch, Ellipse)]
    assert len(yellow_monitor_rectangles) == 6
    assert len(screen_rectangles) == 3
    assert all(patch.get_width() < patch.get_height() for patch in screen_rectangles)
    assert track_polygons
    track_vertices = track_polygons[0].get_xy()
    axes_aspect = (ax.get_position().width * fig.get_figwidth()) / (
        ax.get_position().height * fig.get_figheight()
    )
    assert np.ptp(track_vertices[:, 1]) == pytest.approx(0.188)
    assert (
        np.ptp(track_vertices[:, 0]) / np.ptp(track_vertices[:, 1]) * axes_aspect
    ) == pytest.approx(4.1 / 4.0)
    bottom_monitor_rectangles = [
        patch
        for patch in yellow_monitor_rectangles
        if patch.get_width() == pytest.approx(np.ptp(track_vertices[:, 0]))
    ]
    assert bottom_monitor_rectangles
    assert (
        bottom_monitor_rectangles[0].get_y() + bottom_monitor_rectangles[0].get_height()
        < np.min(track_vertices[:, 1])
    )
    assert max(patch.get_width() for patch in yellow_monitor_rectangles) == pytest.approx(
        np.ptp(track_vertices[:, 0])
    )
    assert black_screen_rectangles
    assert dot_screen_rectangles
    assert len(dot_ellipses) == 11
    assert all(patch.width < patch.height for patch in dot_ellipses)
    assert [text.get_text() for text in ax.texts] == ["L", "C", "R", "Visual stimuli"]
    assert ax.texts[-1].get_fontsize() == pytest.approx(8.5)
    assert ax.texts[-1].get_position()[1] == pytest.approx(0.135)
    center_wall_lines = [
        line
        for line in ax.lines
        if line.get_xdata()[0] == pytest.approx(line.get_xdata()[1])
    ]
    horizontal_wall_lines = [
        line
        for line in ax.lines
        if line.get_ydata()[0] == pytest.approx(line.get_ydata()[1])
    ]
    assert len(ax.lines) == 3
    assert len(center_wall_lines) == 2
    assert len(horizontal_wall_lines) == 1
    assert all(
        text.get_position()[1] > horizontal_wall_lines[0].get_ydata()[0]
        for text in ax.texts[:3]
    )
    plt.close(fig)


def test_cycle_panel_labels_bottom_left_linearization_schematic() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_w_track_cycle_panel(ax)

    inset_label_texts = [
        [text.get_text() for text in inset.texts if text.get_text()]
        for inset in ax.child_axes
    ]
    assert inset_label_texts == [[], [], [], []]
    visible_text = [text.get_text() for text in ax.texts if text.get_text()]
    assert visible_text[:3] == ["L", "C", "R"]
    plt.close(fig)


def test_plot_stability_panel_draws_histograms_and_schematics() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for region in ("v1", "ca1"):
        for trajectory_type in (
            "center_to_left",
            "left_to_center",
            "center_to_right",
            "right_to_center",
        ):
            rows.append(
                {
                    "region": region,
                    "trajectory_type": trajectory_type,
                    "stability_correlation": 0.5,
                }
            )
    fig, ax = plt.subplots()
    plot_stability_panel(ax, pd.DataFrame(rows))

    assert len(ax.child_axes) == 8
    schematic_axes = ax.child_axes[::2]
    hist_axes = ax.child_axes[1::2]
    expected_trajectories = [
        trajectory
        for trajectory_row in STABILITY_TRAJECTORY_LAYOUT
        for trajectory in trajectory_row
    ]
    assert [
        schematic_ax.lines[0].get_color() for schematic_ax in schematic_axes
    ] == [PANEL_E_TRAJECTORY_COLORS[trajectory] for trajectory in expected_trajectories]
    legend = ax.get_legend()
    assert legend is not None
    assert all(
        text.get_fontsize() == pytest.approx(STABILITY_LEGEND_FONTSIZE)
        for text in legend.get_texts()
    )
    assert all(
        hist_axis.xaxis.label.get_fontsize()
        == pytest.approx(STABILITY_AXIS_LABEL_FONTSIZE)
        for hist_axis in hist_axes
        if hist_axis.get_xlabel()
    )
    assert all(
        hist_axis.yaxis.label.get_fontsize()
        == pytest.approx(STABILITY_AXIS_LABEL_FONTSIZE)
        for hist_axis in hist_axes
        if hist_axis.get_ylabel()
    )
    assert all(
        tick.get_fontsize() == pytest.approx(STABILITY_TICK_LABEL_FONTSIZE)
        for hist_axis in hist_axes
        for tick in [*hist_axis.get_xticklabels(), *hist_axis.get_yticklabels()]
        if tick.get_text()
    )
    plt.close(fig)


def _fake_panel_e_example(animal_name: str, unit_id: int) -> dict[str, object]:
    trajectories = (
        "center_to_left",
        "center_to_right",
        "right_to_center",
        "left_to_center",
    )
    return {
        "animal_name": animal_name,
        "date": "20240611",
        "epoch": "08_r4",
        "region": "v1",
        "unit_id": unit_id,
        "raster_positions": {
            trajectory: [np.asarray([0.1, 0.4]), np.asarray([0.7])]
            for trajectory in trajectories
        },
        "firing_rates": {
            trajectory: (
                np.asarray([0.0, 0.5, 1.0]),
                np.asarray([0.0, 1.0, 0.5]),
            )
            for trajectory in trajectories
        },
    }


def test_plot_panel_e_examples_stacks_two_example_blocks() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_panel_e_examples(
        ax,
        [
            _fake_panel_e_example("L14", 34),
            _fake_panel_e_example("L15", 38),
        ],
    )

    assert len(ax.lines) == 0
    assert len(ax.child_axes) == 2
    for example_index, example_ax in enumerate(ax.child_axes, start=1):
        assert len(example_ax.child_axes) == 10
        assert [text.get_text() for text in example_ax.texts] == [
            f"Example cell {example_index}"
        ]
        assert example_ax.texts[0].get_position()[0] == pytest.approx(0.50)
        assert example_ax.texts[0].get_horizontalalignment() == "center"
        assert all(child.get_title() == "" for child in example_ax.child_axes)
        schematic_width = example_ax.child_axes[0].get_position().width
        raster_width = example_ax.child_axes[1].get_position().width
        assert schematic_width < 0.25 * raster_width
        assert example_ax.child_axes[4].get_position().y1 < (
            example_ax.child_axes[1].get_position().y0
        )
        assert example_ax.child_axes[6].get_position().y1 < (
            example_ax.child_axes[3].get_position().y0
        )
        raster_axes = [example_ax.child_axes[index] for index in (1, 3, 5, 7)]
        raster_trajectories = [
            "center_to_left",
            "center_to_right",
            "right_to_center",
            "left_to_center",
        ]
        rate_axes = example_ax.child_axes[8:]
        assert all(
            rate_axis.get_xlabel() == "Norm. path progression"
            for rate_axis in rate_axes
        )
        assert all(
            rate_axis.xaxis.label.get_fontsize()
            == pytest.approx(PANEL_E_AXIS_LABEL_FONTSIZE)
            for rate_axis in rate_axes
        )
        assert all(
            panel_e_axis.yaxis.label.get_fontsize()
            == pytest.approx(PANEL_E_AXIS_LABEL_FONTSIZE)
            for panel_e_axis in [*raster_axes, *rate_axes]
            if panel_e_axis.get_ylabel()
        )
        for panel_e_axis in [*raster_axes, *rate_axes]:
            boundary_lines = panel_e_axis.lines[
                -len(TASK_PROGRESSION_SEGMENT_BOUNDARIES) :
            ]
            assert [line.get_xdata()[0] for line in boundary_lines] == pytest.approx(
                TASK_PROGRESSION_SEGMENT_BOUNDARIES
            )
            assert [line.get_color() for line in boundary_lines] == [
                TASK_PROGRESSION_SEGMENT_BOUNDARY_COLOR
            ] * len(TASK_PROGRESSION_SEGMENT_BOUNDARIES)
        for raster_axis, trajectory_type in zip(
            raster_axes,
            raster_trajectories,
            strict=True,
        ):
            raster_tick_lines = raster_axis.lines[
                : -len(TASK_PROGRESSION_SEGMENT_BOUNDARIES)
            ]
            assert raster_tick_lines
            assert all(
                line.get_color() == PANEL_E_TRAJECTORY_COLORS[trajectory_type]
                for line in raster_tick_lines
            )
            assert all(
                line.get_markersize() == pytest.approx(PANEL_E_RASTER_TICK_MARKERSIZE)
                for line in raster_tick_lines
            )
            assert all(
                line.get_markeredgewidth()
                == pytest.approx(PANEL_E_RASTER_TICK_MARKEREDGEWIDTH)
                for line in raster_tick_lines
            )
        for panel_e_axis in [*raster_axes, *rate_axes]:
            tick_lines = [
                tick.tick1line
                for axis in (panel_e_axis.xaxis, panel_e_axis.yaxis)
                for tick in axis.majorTicks
            ]
            assert all(
                tick_line.get_markersize() == pytest.approx(0.9)
                for tick_line in tick_lines
            )
            assert all(
                tick_line.get_markeredgewidth() == pytest.approx(0.35)
                for tick_line in tick_lines
            )
        assert all(
            tick.get_fontsize() == pytest.approx(PANEL_E_TICK_LABEL_FONTSIZE)
            for rate_axis in rate_axes
            for tick in [*rate_axis.get_xticklabels(), *rate_axis.get_yticklabels()]
            if tick.get_text()
        )
    plt.close(fig)


def test_build_zero_including_histogram_bins_includes_zero() -> None:
    bins = build_zero_including_histogram_bins(np.asarray([0.2, 0.3, 0.4]))

    assert np.any(np.isclose(bins, 0.0))


def test_build_zero_including_histogram_bins_uses_requested_bin_count() -> None:
    bins = build_zero_including_histogram_bins(
        np.asarray([-2.0, -0.1, 0.1, 5.0]),
        n_bins=20,
    )

    assert len(bins) < 30
    assert np.any(np.isclose(bins, 0.0))


def test_plot_motor_delta_panel_draws_fraction_histogram() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15", "L16"],
            "date": [
                "20240611",
                "20240611",
                "20241121",
                "20241121",
                "20250302",
            ],
            "unit": [1, 2, 1, 2, 1],
            "delta_log_likelihood_bits_per_spike": [-0.1, 0.0, 0.2, 0.3, np.nan],
        }
    )
    fig, ax = plt.subplots()
    plot_motor_delta_panel(ax, table)

    assert ax.get_ylabel() == "Frac."
    assert ax.get_xlabel() == DELTA_LOG_LIKELIHOOD_AXIS_LABEL
    assert ax.xaxis.label.get_fontsize() == pytest.approx(7.0)
    assert ax.get_title() == ""
    assert ax.get_xlim() == pytest.approx((-1.0, 1.0))
    text_labels = [text.get_text() for text in ax.texts]
    assert "Motor only better" in text_labels
    assert "Motor+DPP better" in text_labels
    assert "50% >0, med. 0.10" in text_labels
    assert "n = 4 cells\n2 animals" in text_labels
    assert ax.texts[0].get_horizontalalignment() == "left"
    assert ax.texts[0].get_position()[0] == pytest.approx(0.03)
    assert ax.texts[1].get_horizontalalignment() == "left"
    assert ax.texts[1].get_position()[0] == pytest.approx(0.68)
    assert ax.texts[2].get_horizontalalignment() == "left"
    assert ax.texts[2].get_position()[0] == pytest.approx(0.68)
    assert ax.texts[2].get_bbox_patch() is None
    assert len(ax.patches) == 21
    assert ax.patches[0].get_extents().x1 <= ax.transData.transform((0.0, 0.0))[0]
    assert ax.lines[0].get_color() == "black"
    plt.close(fig)


def test_plot_encoding_delta_panel_draws_two_model_comparisons() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L15", "L14", "L15", "L16"],
            "date": [
                "20240611",
                "20241121",
                "20240611",
                "20241121",
                "20250302",
            ],
            "unit": [1, 2, 1, 2, 3],
            "comparison": [
                "dpp_vs_absolute_place",
                "dpp_vs_absolute_place",
                "dpp_vs_absolute_task_progression",
                "dpp_vs_absolute_task_progression",
                "dpp_vs_absolute_place",
            ],
            "comparison_label": [
                "DPP - absolute place",
                "DPP - absolute place",
                "DPP - distance-to-reward",
                "DPP - distance-to-reward",
                "DPP - absolute place",
            ],
            "delta_log_likelihood_bits_per_spike": [-0.1, 0.2, -0.3, 0.1, np.nan],
        }
    )
    fig, ax = plt.subplots()
    plot_encoding_delta_panel(ax, table)

    assert ax.get_ylabel() == "Frac."
    assert ax.get_xlabel() == DELTA_LOG_LIKELIHOOD_AXIS_LABEL
    assert ax.xaxis.label.get_fontsize() == pytest.approx(7.0)
    assert ax.get_title() == ""
    assert ax.get_xlim() == pytest.approx((-1.0, 1.0))
    text_labels = [text.get_text() for text in ax.texts]
    assert "Abs place better" in text_labels
    assert "Distance-to-reward\nbetter" in text_labels
    assert text_labels.count("DPP better") == 1
    assert "DPP > abs place\n50% >0, med. 0.05" in text_labels
    assert "DPP > dist.-to-reward\n50% >0, med. -0.10" in text_labels
    assert "n = 2 cells\n2 animals" in text_labels
    assert ax.texts[0].get_color() == ENCODING_DPP_COMPARISON_COLORS[
        "dpp_vs_absolute_place"
    ]
    assert ax.texts[1].get_color() == ENCODING_DPP_COMPARISON_COLORS[
        "dpp_vs_absolute_task_progression"
    ]
    assert ax.texts[0].get_horizontalalignment() == "left"
    assert ax.texts[1].get_horizontalalignment() == "left"
    assert ax.texts[2].get_color() == "black"
    assert ax.texts[2].get_horizontalalignment() == "left"
    assert ax.texts[2].get_position()[0] == pytest.approx(0.67)
    assert all(text.get_bbox_patch() is None for text in ax.texts)
    assert ax.texts[3].get_color() == ENCODING_DPP_COMPARISON_COLORS[
        "dpp_vs_absolute_place"
    ]
    assert ax.texts[4].get_color() == ENCODING_DPP_COMPARISON_COLORS[
        "dpp_vs_absolute_task_progression"
    ]
    assert ax.texts[3].get_horizontalalignment() == "left"
    assert ax.texts[4].get_horizontalalignment() == "left"
    assert ax.texts[3].get_position()[0] == pytest.approx(0.67)
    assert ax.texts[4].get_position()[0] == pytest.approx(0.67)
    assert ax.lines[0].get_color() == "black"
    assert ax.get_legend() is None
    assert len(ax.patches) == 41
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_decoding_error_panel_draws_median_iqr_and_example_schematics() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt

    rows = []
    animals = ("L14", "L15", "L16", "L19")
    for animal_name in animals:
        for comparison, label, transfer_family, _pairs in DECODING_CROSS_TRAJECTORY_COMPARISONS:
            rows.extend(
                {
                    "animal_name": animal_name,
                    "comparison": comparison,
                    "comparison_label": label,
                    "transfer_family": transfer_family,
                    "absolute_error": value,
                }
                for value in (0.1, 0.2, 0.4)
            )
    fig, ax = plt.subplots()
    plot_decoding_error_panel(ax, pd.DataFrame(rows))

    plot_ax = ax.child_axes[0]
    schematic_axes = ax.child_axes[1:]
    assert plot_ax.get_ylabel() == "Abs. norm. error"
    assert plot_ax.yaxis.label.get_fontsize() == pytest.approx(
        DECODING_YLABEL_FONTSIZE
    )
    assert plot_ax.get_ylim() == pytest.approx((0.0, 0.5))
    assert plot_ax.get_position().bounds == pytest.approx(ax.get_position().bounds)
    assert [text.get_text() for text in plot_ax.get_xticklabels()] == [
        label for _comparison, label, _family, _pairs in DECODING_CROSS_TRAJECTORY_COMPARISONS
    ]
    assert "Opposite turn\nsame arm" in [
        text.get_text() for text in plot_ax.get_xticklabels()
    ]
    assert all(
        text.get_fontsize() == pytest.approx(DECODING_XTICK_LABEL_FONTSIZE)
        for text in plot_ax.get_xticklabels()
    )
    assert len(plot_ax.lines) == 0
    assert len(plot_ax.collections) == 2
    assert [text.get_text() for text in plot_ax.texts] == [
        "med. 0.20",
        "med. 0.20",
        "med. 0.20",
    ]
    assert all(
        text.get_fontsize() == pytest.approx(DECODING_MEDIAN_LABEL_FONTSIZE)
        for text in plot_ax.texts
    )
    assert [text.get_horizontalalignment() for text in plot_ax.texts] == [
        "left",
        "left",
        "left",
    ]
    assert [text.get_verticalalignment() for text in plot_ax.texts] == [
        "center",
        "center",
        "center",
    ]
    assert [text.get_position()[0] for text in plot_ax.texts] == pytest.approx(
        [
            position + DECODING_MEDIAN_LABEL_X_OFFSET
            for position in (1.0, 2.0, 3.0)
        ]
    )
    assert [text.get_position()[1] for text in plot_ax.texts] == pytest.approx(
        [0.2, 0.2, 0.2]
    )
    legend = plot_ax.get_legend()
    assert legend is None
    scatter = plot_ax.collections[1]
    assert tuple(scatter.get_facecolors()[0]) == pytest.approx(
        to_rgba(figure_1_module.REGION_COLORS["v1"])
    )
    assert len(ax.patches) == 0
    assert len(schematic_axes) == 1 + len(DECODING_CROSS_TRAJECTORY_COMPARISONS)
    assert all(
        schematic_ax.get_position().y1 < plot_ax.get_position().y0
        for schematic_ax in schematic_axes
    )
    assert [text.get_text() for text in ax.texts] == ["Train"]
    assert ax.texts[0].get_position()[0] == pytest.approx(
        DECODING_TRAIN_SCHEMATIC_CENTER_X
    )
    assert DECODING_SCHEMATIC_Y == pytest.approx(-0.55)
    assert DECODING_TRAIN_LABEL_Y == pytest.approx(-0.32)
    assert ax.texts[0].get_position()[1] == pytest.approx(DECODING_TRAIN_LABEL_Y)
    assert DECODING_TRAIN_LABEL_Y > (
        DECODING_SCHEMATIC_Y + DECODING_SCHEMATIC_HEIGHT
    )
    assert schematic_axes[0].get_position().x1 < plot_ax.get_position().x0
    assert schematic_axes[0].get_position().width == pytest.approx(
        DECODING_SCHEMATIC_WIDTH * ax.get_position().width
    )
    assert schematic_axes[0].get_position().height == pytest.approx(
        DECODING_SCHEMATIC_HEIGHT * ax.get_position().height
    )
    assert DECODING_EXAMPLE_TRAIN_TRAJECTORY == "center_to_left"
    assert DECODING_EXAMPLE_TEST_TRAJECTORIES == {
        "same_turn_cross_arm": "right_to_center",
        "opposite_turn_same_arm": "left_to_center",
        "same_inbound_outbound_cross_arm": "center_to_right",
    }
    plt.close(fig)
