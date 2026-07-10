from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from v1ca1.paper_figures.figure_3 import (
    DEFAULT_EXAMPLE_DATASET,
    DEFAULT_FIGURE_CACHE_DIR,
    DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PANEL_B_PREDICTION_EXAMPLES,
    DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
    DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
    DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    DEFAULT_RIPPLE_WINDOW_S,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    NEURON_SCALE_BAR_COUNT,
    PANEL_ABC_HEADER_LABEL_X_OFFSETS,
    PANEL_D_DARK_ACTIVITY_COLORS,
    add_aligned_panel_headers,
    build_output_path,
    build_glm_dark_activity_devexp_table,
    build_dark_movement_firing_rate_cache_metadata,
    build_dark_movement_firing_rate_cache_path,
    build_peri_ripple_heatmap_payload,
    build_panel_b_schematic_cache_metadata,
    build_panel_b_schematic_cache_path,
    build_ripple_modulation_output_stem,
    compute_significance_distribution_comparison,
    draw_neuron_scale_bar,
    draw_ripple_glm_schematic,
    format_glm_model_window_suffix,
    format_ridge_strength_suffix,
    format_ripple_window_suffix,
    HEATMAP_EPOCH_LABELS,
    HEATMAP_EPOCH_ORDER,
    get_ripple_event_path,
    get_ripple_glm_path,
    get_ripple_glm_model_window_path,
    get_ripple_lfp_path,
    get_ripple_modulation_paths,
    get_place_tuning_curve_path,
    get_ripple_decoding_comparison_summary_path,
    get_screen_xcorr_paths,
    get_tuning_similarity_path,
    load_glm_behavior_association_tables,
    load_glm_source_predictor_comparison_tables,
    load_dark_movement_firing_rate_cache,
    load_glm_prediction_for_unit,
    load_panel_b_schematic_cache,
    load_panel_b_prediction_examples,
    load_glm_offset_panel_tables,
    load_example_glm_prediction,
    load_example_ripple_lfp_trace,
    load_first_available_glm_prediction,
    load_dark_same_turn_tuning_similarity_table,
    load_glm_epoch_summary_tables,
    load_modulation_summary_table,
    load_pooled_ripple_heatmap_epoch_tables,
    load_ripple_decoding_comparison_panel_tables,
    load_ripple_heatmap_epoch_tables,
    load_ripple_count_table,
    load_ripple_glm_summary_table,
    load_top_ca1_xcorr_panel_data,
    save_dark_movement_firing_rate_cache,
    save_panel_b_schematic_cache,
    parse_arguments,
    parse_dataset_id,
    prepare_xcorr_payload_for_display,
    plot_glm_summary_panel,
    plot_epoch_modulation_histogram_panel,
    plot_epoch_ripple_heatmap_panel,
    plot_glm_behavior_association_panel,
    plot_glm_analysis_panel,
    plot_glm_source_predictor_comparison_panel,
    plot_glm_offset_panel,
    plot_modulation_index_panel,
    plot_observed_predicted_panel,
    plot_peri_ripple_heatmap_panel,
    plot_ripple_decoding_comparison_panel,
    plot_ripple_lfp_panel,
    plot_top_ca1_xcorr_panel,
    _prediction_example_axis_limit,
)


def test_parse_dataset_id_requires_animal_and_date() -> None:
    assert parse_dataset_id("L14:20240611") == ("L14", "20240611", "08_r4")
    assert parse_dataset_id("L15:20241121:10_r5") == ("L15", "20241121", "10_r5")

    with pytest.raises(argparse.ArgumentTypeError, match="animal:date"):
        parse_dataset_id("L14")


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "figure_3", "svg") == Path(
        "paper_figures/figure_3.svg"
    )


def test_panel_c_prediction_examples_use_selected_low_bias_cells() -> None:
    assert DEFAULT_PANEL_B_PREDICTION_EXAMPLES == (
        ("L12", "20240421", "02_r1", 24),
        ("L12", "20240421", "02_r1", 32),
        ("L12", "20240421", "02_r1", 110),
    )


@pytest.mark.parametrize(
    ("max_value", "expected_limit"),
    ((11.0, 12.0), (17.7, 20.0), (24.0, 25.0), (37.0, 40.0)),
)
def test_prediction_example_axis_limit_contains_all_values(
    max_value: float,
    expected_limit: float,
) -> None:
    example = {
        "observed": np.array([0.0, max_value]),
        "predicted": np.array([0.0, max_value - 1.0]),
    }

    assert _prediction_example_axis_limit(example) == pytest.approx(expected_limit)

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "figure_3", "jpg")


def test_add_aligned_panel_headers_uses_shared_vertical_position() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 2.0, 1.0],
    )
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[0, 2]),
    ]
    titles = (
        "Ripple-triggered\nmean firing rates",
        "Predicting V1 activity during ripples\nwith CA1 activity",
        "CA1 spike vector vs.\nmean CA1 activity",
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, fontsize=7.2, pad=2)
    fig.canvas.draw()

    add_aligned_panel_headers(
        fig,
        axes,
        labels=("A", "B", "C"),
        titles=titles,
        label_x_offsets=PANEL_ABC_HEADER_LABEL_X_OFFSETS,
        fontsize=7.2,
    )

    assert [axis.get_title() for axis in axes] == ["", "", ""]
    header_texts = fig.texts[-6:]
    assert [text.get_text() for text in header_texts] == [
        "A",
        "Ripple-triggered\nmean firing rates",
        "B",
        "Predicting V1 activity during ripples\nwith CA1 activity",
        "C",
        "CA1 spike vector vs.\nmean CA1 activity",
    ]
    assert {text.get_position()[1] for text in header_texts} == {
        header_texts[0].get_position()[1]
    }
    assert {text.get_va() for text in header_texts} == {"top"}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label_text, title_text in zip(header_texts[::2], header_texts[1::2], strict=True):
        assert not label_text.get_window_extent(renderer).overlaps(
            title_text.get_window_extent(renderer)
        )
    plt.close(fig)


def test_dark_movement_firing_rate_cache_path_preserves_legacy_stem() -> None:
    metadata = build_dark_movement_firing_rate_cache_metadata(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        dark_epoch="08_r4",
        region="v1",
    )
    cache_path = build_dark_movement_firing_rate_cache_path(
        Path("paper_figures/output/cache"),
        metadata,
    )

    assert metadata["cache_version"] == 1
    assert metadata["data_root"] == "/analysis"
    assert metadata["animal_name"] == "L14"
    assert metadata["dark_epoch"] == "08_r4"
    assert cache_path == Path(
        "paper_figures/output/cache/"
        "figure_4_dark_movement_firing_rate_v1_L14_20240611_08_r4_speed4_cachev1.parquet"
    )


def test_dark_movement_firing_rate_cache_roundtrip_validates_metadata(
    tmp_path: Path,
) -> None:
    metadata = build_dark_movement_firing_rate_cache_metadata(
        data_root=Path("/analysis"),
        animal_name="L14",
        date="20240611",
        dark_epoch="08_r4",
        region="v1",
    )
    table = pd.DataFrame(
        {
            "unit": [11, 12],
            "dark_firing_rate_hz": [0.1, 0.6],
        }
    )
    cache_path = build_dark_movement_firing_rate_cache_path(tmp_path, metadata)

    pytest.importorskip("pyarrow")
    save_dark_movement_firing_rate_cache(cache_path, table, metadata)
    loaded = load_dark_movement_firing_rate_cache(cache_path, metadata)

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, table)
    stale_metadata = dict(metadata)
    stale_metadata["dark_epoch"] = "10_r5"
    assert load_dark_movement_firing_rate_cache(cache_path, stale_metadata) is None


def test_panel_b_schematic_cache_roundtrip_preserves_legacy_stem(
    tmp_path: Path,
) -> None:
    metadata = build_panel_b_schematic_cache_metadata(
        data_root=Path("/analysis"),
        animal_name="L15",
        date="20241121",
        epoch="02_r1",
    )
    cache_path = build_panel_b_schematic_cache_path(tmp_path, metadata)
    payload = {
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "02_r1",
        "time_s": np.asarray([-0.08, 0.0, 0.22]),
        "filtered_lfp": np.asarray([0.1, -0.2, 0.3]),
        "ripple_start_s": 10.0,
        "ripple_end_s": 10.06,
        "ripple_duration_s": 0.06,
        "mean_zscore": 4.5,
        "channel": 12,
        "n_ripples": 7,
        "time_before_s": DEFAULT_PANEL_B_SCHEMATIC_TIME_BEFORE_S,
        "time_after_s": DEFAULT_PANEL_B_SCHEMATIC_TIME_AFTER_S,
        "n_units_per_region": DEFAULT_PANEL_B_SCHEMATIC_N_UNITS_PER_REGION,
        "ca1_unit_ids": np.asarray([101, 102]),
        "v1_unit_ids": np.asarray([201, 202]),
        "ca1_spike_times_s": (np.asarray([-0.01, 0.03]), np.asarray([0.12])),
        "v1_spike_times_s": (np.asarray([0.02]), np.asarray([0.09, 0.18])),
        "selection_score": np.asarray([1.0, 2.0, 6.0, 4.5, 0.0]),
    }

    save_panel_b_schematic_cache(cache_path, payload, metadata)
    loaded = load_panel_b_schematic_cache(cache_path, metadata)

    assert cache_path.name == (
        "figure_4_panel_b_schematic_L15_20241121_02_r1"
        "_thr2_tb0p08_ta0p22_n5_dur0p15_cachev4.npz"
    )
    assert loaded is not None
    assert loaded["animal_name"] == "L15"
    assert loaded["channel"] == 12
    np.testing.assert_array_equal(loaded["ca1_unit_ids"], np.asarray([101, 102]))
    np.testing.assert_allclose(loaded["ca1_spike_times_s"][0], np.asarray([-0.01, 0.03]))
    np.testing.assert_allclose(loaded["v1_spike_times_s"][1], np.asarray([0.09, 0.18]))

    stale_metadata = dict(metadata)
    stale_metadata["time_after_s"] = 0.2
    assert load_panel_b_schematic_cache(cache_path, stale_metadata) is None


def test_ripple_modulation_paths_match_cached_output_stem(tmp_path: Path) -> None:
    stem = build_ripple_modulation_output_stem(
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region_label="all_regions",
        ripple_threshold_zscore=2.0,
        bin_size_s=0.02,
        time_before_s=0.5,
        time_after_s=0.5,
        response_window=(0.0, 0.1),
        baseline_window=(-0.5, -0.3),
        heatmap_normalize="max",
    )

    assert stem == (
        "L14_20240611_08_r4_all_regions_thr_2_bin_0p02_tb_0p5_ta_0p5_"
        "resp_0_0p1_base_neg0p5_neg0p3_norm_max"
    )

    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )

    assert paths["summary"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple"
        / "ripple_modulation"
        / f"{stem}_summary.parquet"
    )
    assert paths["peri_ripple_firing_rate"].name == f"{stem}_peri_ripple_firing_rate.parquet"


def test_ripple_glm_path_matches_samplewise_output_name(tmp_path: Path) -> None:
    assert format_ripple_window_suffix(0.2) == "rw_0p2s"
    assert format_ripple_window_suffix(0.2, ripple_window_offset_s=-0.2) == "rw_0p2s_off_m0p2s"
    assert format_ridge_strength_suffix(1e-1) == "ridge_1e-1"

    path = get_ripple_glm_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_window_s=0.2,
        ripple_selection="allripples",
        ridge_strength=1e-1,
    )

    assert path == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple_glm"
        / "08_r4_rw_0p2s_allripples_ridge_1e-1_samplewise_ripple_glm.nc"
    )
    mean_path = get_ripple_glm_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_window_s=0.2,
        ripple_selection="single",
        ridge_strength=1e-1,
        source_predictor_mode="mean_activity",
    )

    assert mean_path == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple_glm"
        / "08_r4_rw_0p2s_single_mean_ca1_ridge_1e-1_samplewise_ripple_glm.nc"
    )


def test_ripple_glm_model_window_path_matches_asymmetric_output_name(
    tmp_path: Path,
) -> None:
    assert (
        format_glm_model_window_suffix(
            source_window_s=0.2,
            source_window_offset_s=0.0,
            target_window_s=0.2,
            target_window_offset_s=0.2,
        )
        == "src_rw_0p2s_tgt_rw_0p2s_off_0p2s"
    )

    path = get_ripple_glm_model_window_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        source_window_s=0.2,
        source_window_offset_s=0.0,
        target_window_s=0.2,
        target_window_offset_s=0.2,
        ripple_selection="allripples",
        ridge_strength=1e-1,
    )

    assert path == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple_glm"
        / "02_r1_src_rw_0p2s_tgt_rw_0p2s_off_0p2s_allripples_ridge_1e-1_samplewise_ripple_glm.nc"
    )


def _write_screen_xcorr_cache(tmp_path: Path) -> tuple[Path, np.ndarray]:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("pyarrow")
    output_dir = tmp_path / "RatA" / "20240101" / "xcorr" / "screen_pairs" / "ripple" / "02_r1"
    output_dir.mkdir(parents=True)

    ca1_units = np.array([10, 11, 12], dtype=int)
    v1_units = np.array([101, 102, 103], dtype=int)
    lag_s = np.array([-0.01, 0.0, 0.01], dtype=float)
    xcorr = np.arange(ca1_units.size * v1_units.size * lag_s.size, dtype=float).reshape(
        ca1_units.size,
        v1_units.size,
        lag_s.size,
    )
    xr.Dataset(
        data_vars={"xcorr": (("ca1_unit", "v1_unit", "lag_s"), xcorr)},
        coords={"ca1_unit": ca1_units, "v1_unit": v1_units, "lag_s": lag_s},
    ).to_netcdf(output_dir / "xcorr.nc")

    rows = []
    peak_values = {
        10: {101: 5.0, 102: 4.0, 103: 3.0},
        11: {101: 6.0, 102: 7.0, 103: 2.0},
        12: {101: 1.0, 102: 1.5, 103: 2.0},
    }
    peak_lags = {
        10: {101: 0.0, 102: 0.01, 103: -0.01},
        11: {101: -0.01, 102: 0.01, 103: 0.0},
        12: {101: 0.0, 102: 0.01, 103: -0.01},
    }
    for ca1_unit in ca1_units:
        for v1_unit in v1_units:
            rows.append(
                {
                    "ca1_unit_id": ca1_unit,
                    "v1_unit_id": v1_unit,
                    "n_ca1_state_spikes": 50,
                    "n_v1_state_spikes": 60,
                    "peak_lag_s": peak_lags[int(ca1_unit)][int(v1_unit)],
                    "peak_norm_xcorr": peak_values[int(ca1_unit)][int(v1_unit)],
                    "status": "valid",
                }
            )
    pd.DataFrame(rows).to_parquet(output_dir / "xcorr_summary.parquet", index=False)
    return output_dir, xcorr


def test_load_top_ca1_xcorr_panel_data_uses_shared_v1_order(tmp_path: Path) -> None:
    output_dir, xcorr = _write_screen_xcorr_cache(tmp_path)

    paths = get_screen_xcorr_paths(
        tmp_path,
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
    )
    payload = load_top_ca1_xcorr_panel_data(
        tmp_path,
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        top_n_ca1_units=2,
    )

    assert paths["dataset"] == output_dir / "xcorr.nc"
    assert payload["ca1_unit_ids"].tolist() == [11, 10]
    assert payload["v1_unit_ids"].tolist() == [102, 101, 103]
    assert payload["v1_order_reference_ca1_unit"] == 11
    assert payload["xcorr"].shape == (2, 3, 3)
    assert np.allclose(payload["xcorr"][0, 0], xcorr[1, 1])


def test_prepare_xcorr_payload_for_display_keeps_all_v1_cells() -> None:
    xcorr = np.ones((4, 6, 5), dtype=float)
    xcorr[0, 0, 2] = 9.0
    xcorr[0, 1, 2] = 8.0
    xcorr[1, 2, 2] = 9.0
    xcorr[1, 3, 2] = 8.0
    xcorr[2, 4, 2] = 9.0
    xcorr[2, 5, 2] = 8.0
    payload = {
        "ca1_unit_ids": np.array([11, 10, 9, 8]),
        "v1_unit_ids": np.array([101, 102, 103, 104, 105, 106]),
        "lag_s": np.array([-0.02, -0.01, 0.0, 0.01, 0.02]),
        "xcorr": xcorr,
    }

    cropped = prepare_xcorr_payload_for_display(payload)

    assert cropped["ca1_unit_ids"].tolist() == [11, 10, 9]
    assert cropped["v1_unit_ids"].tolist() == [101, 102, 103, 104, 105, 106]
    assert cropped["v1_group_ca1_indices"].tolist() == [0, 0, 1, 1, 2, 2]
    assert cropped["v1_group_boundaries"].tolist() == [2, 4]
    assert cropped["v1_ordering"] == "shared_multi_example_partner_rank"
    assert cropped["xcorr"].shape == (3, 6, 5)
    np.testing.assert_allclose(cropped["xcorr"], xcorr[:3])


def test_plot_top_ca1_xcorr_panel_smooths_only_lag_and_marks_v1_groups() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xcorr = np.zeros((2, 4, 7), dtype=float)
    xcorr[0, 0, 3] = 5.0
    payload = {
        "ca1_unit_ids": np.array([11, 10]),
        "v1_unit_ids": np.array([101, 102, 103, 104]),
        "lag_s": np.linspace(-0.03, 0.03, 7),
        "xcorr": xcorr,
        "display_vmax": 5.0,
        "v1_group_ca1_indices": np.array([0, 0, 1, 1]),
        "v1_group_boundaries": np.array([2]),
    }

    fig, ax = plt.subplots()
    plot_top_ca1_xcorr_panel(ax, payload)

    plotted = np.asarray(ax.child_axes[0].images[0].get_array(), dtype=float)
    assert 0.0 < plotted[0, 3] < 5.0
    assert plotted[0, 2] > 0.0
    assert np.allclose(plotted[1:], 0.0)
    assert xcorr[0, 0, 2] == pytest.approx(0.0)
    horizontal_lines = [
        line
        for line in ax.child_axes[0].lines
        if np.ptp(np.asarray(line.get_ydata(), dtype=float)) == pytest.approx(0.0)
    ]
    assert len(horizontal_lines) == 1
    assert horizontal_lines[0].get_ydata()[0] == pytest.approx(2.0)
    assert {text.get_text() for text in ax.texts} >= {"V1 sets", "1", "2"}
    plt.close(fig)


def _write_ripple_events(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    path = get_ripple_event_path(tmp_path, "L14", "20240611")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": ["08_r4", "08_r4", "06_r3"],
            "start": [1.0, 1.3, 2.0],
            "end": [1.05, 1.36, 2.04],
            "mean_zscore": [2.1, 5.0, 6.0],
        }
    ).to_parquet(path, index=False)
    return path


def test_load_ripple_count_table_filters_epoch_and_threshold(tmp_path: Path) -> None:
    _write_ripple_events(tmp_path)

    table = load_ripple_count_table(
        tmp_path,
        [("L14", "20240611", "08_r4")],
        ripple_threshold_zscore=3.0,
    )

    assert table["animal_name"].tolist() == ["L14"]
    assert table["epoch"].tolist() == ["08_r4"]
    assert table["n_ripples"].tolist() == [1]


def test_load_example_ripple_lfp_trace_uses_largest_thresholded_ripple(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")
    _write_ripple_events(tmp_path)
    lfp_path = get_ripple_lfp_path(tmp_path, "L14", "20240611", "08_r4")
    lfp_path.parent.mkdir(parents=True)
    time = np.linspace(1.1, 1.45, 36)
    filtered_lfp = np.column_stack([np.sin(time * 30.0), np.cos(time * 30.0)])
    xr.Dataset(
        data_vars={"filtered_lfp": (("sample", "channel"), filtered_lfp)},
        coords={"time": ("sample", time), "channel": ("channel", [101, 102])},
    ).to_netcdf(lfp_path)

    trace = load_example_ripple_lfp_trace(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_threshold_zscore=2.0,
    )

    assert trace["ripple_start_s"] == pytest.approx(1.3)
    assert trace["mean_zscore"] == pytest.approx(5.0)
    assert trace["channel"] == 101
    assert np.min(trace["time_s"]) < 0.0
    assert np.max(trace["time_s"]) > 0.0


def test_load_modulation_summary_table_reads_cached_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )
    paths["summary"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "epoch": ["08_r4", "08_r4"],
            "region": ["v1", "ca1"],
            "unit_id": [11, 101],
            "ripple_modulation_index": [0.2, -0.1],
            "response_zscore": [2.0, -0.5],
        }
    ).to_parquet(paths["summary"], index=False)

    table = load_modulation_summary_table(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert table["region"].tolist() == ["v1", "ca1"]
    assert table["source_path"].tolist() == [str(paths["summary"])] * 2


def test_build_peri_ripple_heatmap_payload_preserves_time_matrix() -> None:
    table = pd.DataFrame(
        {
            "region": ["v1", "v1", "v1", "v1", "ca1", "ca1"],
            "unit_id": [11, 11, 12, 12, 101, 101],
            "time_s": [-0.02, 0.0, -0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )

    payload = build_peri_ripple_heatmap_payload(table, region="v1")

    assert payload["unit_ids"].tolist() == [11, 12]
    assert np.allclose(payload["time_s"], [-0.02, 0.0])
    assert np.allclose(payload["mean_rate_hz"], [[1.0, 2.0], [4.0, 3.0]])


def test_build_peri_ripple_heatmap_payload_keeps_sessions_separate() -> None:
    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15"],
            "date": ["20240611", "20240611", "20241121", "20241121"],
            "epoch": ["06_r3", "06_r3", "06_r3", "06_r3"],
            "region": ["v1", "v1", "v1", "v1"],
            "unit_id": [11, 11, 11, 11],
            "time_s": [-0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 4.0, 3.0],
        }
    )

    payload = build_peri_ripple_heatmap_payload(table, region="v1")

    assert payload["mean_rate_hz"].shape == (2, 2)
    assert np.allclose(payload["mean_rate_hz"], [[1.0, 2.0], [4.0, 3.0]])


def _write_peri_ripple_table(
    tmp_path: Path,
    epoch: str,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
) -> Path:
    pytest.importorskip("pyarrow")
    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    paths["peri_ripple_firing_rate"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "animal_name": [animal_name] * 4,
            "date": [date] * 4,
            "epoch": [epoch] * 4,
            "region": ["v1", "v1", "ca1", "ca1"],
            "unit_id": [11, 11, 101, 101],
            "n_ripples": [3] * 4,
            "bin_size_s": [0.02] * 4,
            "time_before_s": [0.5] * 4,
            "time_after_s": [0.5] * 4,
            "time_s": [-0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(paths["peri_ripple_firing_rate"], index=False)
    return paths["peri_ripple_firing_rate"]


def _write_modulation_summary_table(
    tmp_path: Path,
    epoch: str,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
) -> Path:
    pytest.importorskip("pyarrow")
    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "animal_name": [animal_name, animal_name],
            "date": [date, date],
            "epoch": [epoch, epoch],
            "region": ["v1", "ca1"],
            "unit_id": [11, 101],
            "ripple_modulation_index": [0.2, -0.1],
            "response_zscore": [2.0, -0.5],
        }
    ).to_parquet(paths["summary"], index=False)
    return paths["summary"]


def test_load_ripple_heatmap_epoch_tables_uses_registered_order(tmp_path: Path) -> None:
    for epoch in ("02_r1", "08_r4", "07_s4"):
        _write_peri_ripple_table(tmp_path, epoch)

    epoch_tables = load_ripple_heatmap_epoch_tables(
        tmp_path,
        {
            "light": ("L14", "20240611", "02_r1"),
            "dark": ("L14", "20240611", "08_r4"),
            "sleep": ("L14", "20240611", "07_s4"),
        },
    )

    assert HEATMAP_EPOCH_ORDER == ("light", "dark", "sleep")
    assert HEATMAP_EPOCH_LABELS["light"] == "Light run"
    assert [payload["epoch"] for payload in epoch_tables] == ["02_r1", "08_r4", "07_s4"]
    assert [payload["label"] for payload in epoch_tables] == ["Light run", "Dark run", "Sleep"]


def test_load_pooled_ripple_heatmap_epoch_tables_uses_all_animals(tmp_path: Path) -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    for animal_name, date, dark_epoch in datasets:
        for epoch in ("02_r1", dark_epoch, "07_s4"):
            _write_peri_ripple_table(
                tmp_path,
                epoch,
                animal_name=animal_name,
                date=date,
            )
            _write_modulation_summary_table(
                tmp_path,
                epoch,
                animal_name=animal_name,
                date=date,
            )

    epoch_tables = load_pooled_ripple_heatmap_epoch_tables(tmp_path, datasets)

    assert [payload["epoch"] for payload in epoch_tables] == ["02_r1", "registered", "07_s4"]
    assert [payload["n_datasets"] for payload in epoch_tables] == [2, 2, 2]
    assert set(epoch_tables[1]["firing_rate_table"]["epoch"]) == {"08_r4", "10_r5"}
    assert len(epoch_tables[2]["summary_table"]) == 4


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


def test_plot_epoch_ripple_heatmap_panel_scales_region_height_by_unit_count() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {
            "region": ["v1"] * 6 + ["ca1"] * 2,
            "unit_id": [1, 1, 2, 2, 3, 3, 101, 101],
            "time_s": [-0.02, 0.0] * 4,
            "mean_rate_hz": [1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 1.5, 2.5],
        }
    )
    epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "epoch": "02_r1",
            "firing_rate_table": table,
            "summary_table": pd.DataFrame(
                {
                    "region": ["v1", "ca1"],
                    "ripple_modulation_index": [0.2, -0.1],
                }
            ),
        }
    ]

    fig, ax = plt.subplots()
    plot_epoch_ripple_heatmap_panel(ax, epoch_tables, regions=("v1", "ca1"))
    fig.canvas.draw()

    v1_height = ax.child_axes[0].get_position().height
    ca1_height = ax.child_axes[1].get_position().height
    assert v1_height / ca1_height == pytest.approx(3.0)
    plt.close(fig)


def test_plot_epoch_ripple_heatmap_panel_can_expand_vertically() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {
            "region": ["v1"] * 2 + ["ca1"] * 2,
            "unit_id": [1, 1, 101, 101],
            "time_s": [-0.02, 0.0] * 2,
            "mean_rate_hz": [1.0, 2.0, 1.5, 2.5],
        }
    )
    epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "epoch": "02_r1",
            "firing_rate_table": table,
            "summary_table": None,
        }
    ]

    default_fig, default_ax = plt.subplots()
    plot_epoch_ripple_heatmap_panel(default_ax, epoch_tables, regions=("v1", "ca1"))
    default_fig.canvas.draw()
    default_height = sum(
        child_axis.get_position().height for child_axis in default_ax.child_axes[:2]
    )

    expanded_fig, expanded_ax = plt.subplots()
    plot_epoch_ripple_heatmap_panel(
        expanded_ax,
        epoch_tables,
        regions=("v1", "ca1"),
        expand_heatmaps_vertically=True,
        show_modulation_histogram=False,
    )
    expanded_fig.canvas.draw()
    expanded_height = sum(
        child_axis.get_position().height for child_axis in expanded_ax.child_axes[:2]
    )

    assert expanded_height > default_height
    assert len(default_ax.child_axes) == 4
    assert len(expanded_ax.child_axes) == 3
    plt.close(default_fig)
    plt.close(expanded_fig)


def test_plot_epoch_modulation_histogram_panel_splits_from_heatmap() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "epoch": "02_r1",
            "summary_table": pd.DataFrame(
                {
                    "region": ["v1", "v1", "ca1"],
                    "ripple_modulation_index": [0.2, -0.1, 0.4],
                }
            ),
        }
    ]

    fig, ax = plt.subplots()
    child_axes = plot_epoch_modulation_histogram_panel(
        ax,
        epoch_tables,
        regions=("v1", "ca1"),
    )

    assert len(child_axes) == 1
    assert child_axes[0].get_xlabel() == "Mod. index"
    assert child_axes[0].get_ylabel() == "Frac."
    assert child_axes[0].patches
    assert not child_axes[0].images
    plt.close(fig)


def _write_ripple_glm_dataset(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
    ripple_selection: str = "allripples",
    source_predictor_mode: str = "unit_vector",
    devexp: np.ndarray | None = None,
    p_values: np.ndarray | None = None,
) -> Path:
    xr = pytest.importorskip("xarray")
    path = get_ripple_glm_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_selection=ripple_selection,
        source_predictor_mode=source_predictor_mode,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if devexp is None:
        devexp = np.array([0.1, 0.4])
    if p_values is None:
        p_values = np.array([0.2, 0.01])
    dataset = xr.Dataset(
        data_vars={
            "ripple_devexp_mean": (("unit",), np.asarray(devexp, dtype=float)),
            "ripple_devexp_p_value": (("unit",), np.asarray(p_values, dtype=float)),
            "ripple_bits_per_spike_mean": (("unit",), np.array([0.03, 0.08])),
            "ripple_observed_count_oof": (
                ("sample", "unit"),
                np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]]),
            ),
            "ripple_predicted_count_oof": (
                ("sample", "unit"),
                np.array([[0.8, 0.2], [1.7, 1.3], [0.2, 2.5]]),
            ),
        },
        coords={"sample": np.arange(3), "unit": np.array([11, 12])},
        attrs={"n_ripples_after_selection": 3},
    )
    dataset.to_netcdf(path)
    return path


def _write_ripple_glm_offset_dataset(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str,
    target_window_offset_s: float,
) -> Path:
    xr = pytest.importorskip("xarray")
    path = get_ripple_glm_model_window_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        source_window_s=0.2,
        source_window_offset_s=0.0,
        target_window_s=0.2,
        target_window_offset_s=target_window_offset_s,
        ripple_selection="allripples",
        ridge_strength=1e-1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={
            "ripple_devexp_mean": (("unit",), np.array([0.1, 0.2, -0.1])),
            "ripple_devexp_p_value": (("unit",), np.array([0.01, 0.2, 0.03])),
        },
        coords={"unit": np.array([11, 12, 13])},
        attrs={"n_ripples_after_selection": 5},
    ).to_netcdf(path)
    return path


def test_load_ripple_glm_summary_table_reads_per_unit_metrics(tmp_path: Path) -> None:
    path = _write_ripple_glm_dataset(tmp_path)

    table = load_ripple_glm_summary_table(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert table["unit_id"].tolist() == [11, 12]
    assert np.allclose(table["ripple_devexp_mean"], [0.1, 0.4])
    assert np.allclose(table["ripple_devexp_p_value"], [0.2, 0.01])
    assert table["n_ripples"].tolist() == [3, 3]
    assert table["source_path"].tolist() == [str(path), str(path)]


def test_load_glm_source_predictor_comparison_tables_pairs_vector_and_mean(
    tmp_path: Path,
) -> None:
    _write_ripple_glm_dataset(
        tmp_path,
        epoch="02_r1",
        source_predictor_mode="unit_vector",
        devexp=np.array([0.1, 0.4]),
        p_values=np.array([0.2, 0.01]),
    )
    _write_ripple_glm_dataset(
        tmp_path,
        epoch="02_r1",
        source_predictor_mode="mean_activity",
        devexp=np.array([0.05, 0.2]),
        p_values=np.array([0.3, 0.02]),
    )

    payload = load_glm_source_predictor_comparison_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    table = payload["comparison_table"]
    assert payload["missing_artifacts"] == []
    assert table["unit_id"].tolist() == [11, 12]
    assert table["epoch_type"].tolist() == ["light", "light"]
    assert np.allclose(table["vector_devexp_mean"], [0.1, 0.4])
    assert np.allclose(table["mean_activity_devexp_mean"], [0.05, 0.2])
    assert np.allclose(table["devexp_delta_vector_minus_mean"], [0.05, 0.2])


def test_load_glm_offset_panel_tables_uses_complete_offset_sets(
    tmp_path: Path,
) -> None:
    for epoch in ("02_r1", "08_r4", "07_s4"):
        for target_offset_s in (-0.4, -0.2, 0.0, 0.2):
            _write_ripple_glm_offset_dataset(
                tmp_path,
                epoch=epoch,
                target_window_offset_s=target_offset_s,
            )

    payload = load_glm_offset_panel_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    summary_table = payload["summary_table"]
    unit_table = payload["unit_table"]
    assert payload["missing_artifacts"] == []
    assert payload["skipped_comparisons"] == []
    assert len(summary_table) == 12
    assert len(unit_table) == 36
    assert set(summary_table["epoch_type"]) == {"light", "dark", "sleep"}
    assert set(summary_table["target_window_offset_s"]) == {-0.4, -0.2, 0.0, 0.2}
    assert np.allclose(summary_table["fraction_significant_positive"], 1.0 / 3.0)
    assert np.allclose(summary_table["median_devexp_significant"], 0.1)


def test_load_glm_offset_panel_tables_can_select_light_epoch_only(
    tmp_path: Path,
) -> None:
    for target_offset_s in (-0.4, -0.2, 0.0, 0.2):
        _write_ripple_glm_offset_dataset(
            tmp_path,
            epoch="02_r1",
            target_window_offset_s=target_offset_s,
        )

    payload = load_glm_offset_panel_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
        epoch_types=("light",),
    )

    summary_table = payload["summary_table"]
    assert payload["missing_artifacts"] == []
    assert payload["skipped_comparisons"] == []
    assert len(summary_table) == 4
    assert set(summary_table["epoch_type"]) == {"light"}


def test_load_glm_offset_panel_tables_skips_incomplete_offset_sets(
    tmp_path: Path,
) -> None:
    for target_offset_s in (-0.4, -0.2, 0.0):
        _write_ripple_glm_offset_dataset(
            tmp_path,
            epoch="02_r1",
            target_window_offset_s=target_offset_s,
        )

    payload = load_glm_offset_panel_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert payload["summary_table"].empty
    assert payload["unit_table"].empty
    assert payload["missing_artifacts"]
    assert payload["skipped_comparisons"][0]["epoch_type"] == "light"


def test_load_glm_epoch_summary_tables_reads_light_dark_sleep(tmp_path: Path) -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    for animal_name, date, dark_epoch in datasets:
        for epoch in ("02_r1", dark_epoch, "07_s4"):
            _write_ripple_glm_dataset(
                tmp_path,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
            )

    epoch_tables = load_glm_epoch_summary_tables(tmp_path, datasets)

    assert [payload["epoch"] for payload in epoch_tables] == ["02_r1", "registered", "07_s4"]
    assert [payload["n_datasets"] for payload in epoch_tables] == [2, 2, 2]
    assert set(epoch_tables[1]["summary_table"]["epoch"]) == {"08_r4", "10_r5"}
    assert len(epoch_tables[2]["summary_table"]) == 4


def _write_tuning_similarity_table(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
) -> Path:
    pytest.importorskip("pyarrow")
    path = get_tuning_similarity_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        region="v1",
        epoch=epoch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "unit": [11, 12, 11],
            "region": ["v1", "v1", "v1"],
            "epoch": [epoch, epoch, epoch],
            "comparison_label": ["pooled_same_turn", "pooled_same_turn", "pooled_same_arm"],
            "similarity": [0.3, 0.6, 0.1],
            "firing_rate_hz": [1.0, 2.0, 1.0],
        }
    ).to_parquet(path, index=False)
    return path


def _write_place_tuning_curve(
    tmp_path: Path,
    *,
    trajectory: str,
    values: np.ndarray,
    units: np.ndarray,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
) -> Path:
    xr = pytest.importorskip("xarray")
    path = get_place_tuning_curve_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        region="v1",
        epoch=epoch,
        trajectory=trajectory,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data_array = xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("unit", "position_bin"),
        coords={
            "unit": np.asarray(units, dtype=int),
            "position_bin": np.arange(np.asarray(values).shape[1]),
        },
    )
    data_array.to_netcdf(path)
    return path


def test_load_glm_behavior_association_tables_joins_dark_epoch_metrics(
    tmp_path: Path,
) -> None:
    for epoch in ("02_r1", "08_r4", "07_s4"):
        _write_ripple_glm_dataset(tmp_path, epoch=epoch)
    _write_tuning_similarity_table(tmp_path)

    payload = load_glm_behavior_association_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    similarity_table = payload["similarity_table"]
    assert payload["missing_artifacts"] == []
    assert len(similarity_table) == 6
    assert set(similarity_table["epoch_type"]) == {"light", "dark", "sleep"}
    assert set(similarity_table["tuning_epoch"]) == {"08_r4"}
    assert sorted(similarity_table["unit"].unique().tolist()) == [11, 12]
    assert sorted(similarity_table["same_turn_tuning_similarity"].unique().tolist()) == [0.3, 0.6]


def test_load_dark_same_turn_tuning_similarity_table_computes_missing_metric_from_curves(
    tmp_path: Path,
) -> None:
    units = np.array([11, 12])
    tuning_path = get_tuning_similarity_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        similarity_metric="absolute_overlap",
    )
    assert not tuning_path.exists()

    curves_by_trajectory = {
        "center_to_left": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "right_to_center": np.array([[1.0, 0.0], [1.0, 0.0]]),
        "center_to_right": np.array([[0.0, 1.0], [1.0, 1.0]]),
        "left_to_center": np.array([[0.0, 1.0], [0.0, 1.0]]),
    }
    for trajectory, values in curves_by_trajectory.items():
        _write_place_tuning_curve(
            tmp_path,
            trajectory=trajectory,
            values=values,
            units=units,
        )

    table = load_dark_same_turn_tuning_similarity_table(
        tmp_path,
        animal_name="L14",
        date="20240611",
        dark_epoch="08_r4",
        region="v1",
        tuning_similarity_metric="absolute_overlap",
    )

    assert table["unit"].tolist() == [11, 12]
    assert np.allclose(table["same_turn_tuning_similarity"], [1.0, 0.5])
    assert table["tuning_source_path"].tolist() == [str(tuning_path), str(tuning_path)]


def test_load_glm_behavior_association_tables_reports_missing_tuning(
    tmp_path: Path,
) -> None:
    _write_ripple_glm_dataset(tmp_path)

    payload = load_glm_behavior_association_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert payload["similarity_table"].empty
    assert payload["missing_artifacts"][0]["artifact"] == "tuning_analysis"


def test_build_glm_dark_activity_devexp_table_splits_dark_activity(
    tmp_path: Path,
) -> None:
    _write_ripple_glm_dataset(tmp_path)
    glm_table = load_ripple_glm_summary_table(tmp_path, [("L14", "20240611", "08_r4")])
    dark_activity_table = pd.DataFrame(
        {
            "unit": [11, 12],
            "dark_firing_rate_hz": [0.2, 0.8],
        }
    )

    table = build_glm_dark_activity_devexp_table(
        glm_table,
        dark_activity_table,
        animal_name="L14",
        date="20240611",
        glm_epoch="08_r4",
        epoch_type="light",
        dark_epoch="08_r4",
    )

    assert table["dark_activity_group"].tolist() == ["Dark inactive", "Dark active"]
    assert table["dark_active"].tolist() == [False, True]
    assert np.allclose(table["dark_firing_rate_hz"], [0.2, 0.8])


def _write_decoding_comparison_summary_table(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    train_epoch: str = "08_r4",
    decode_epoch: str = "08_r4",
    representation: str = "place",
    turn_group_match_rate: float = 0.6,
    arm_identity_match_rate: float = 0.4,
) -> Path:
    pytest.importorskip("pyarrow")
    path = get_ripple_decoding_comparison_summary_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        representation=representation,
        train_epoch=train_epoch,
        decode_epoch=decode_epoch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "representation": [representation],
            "train_epoch": [train_epoch],
            "decode_epoch": [decode_epoch],
            "n_ripples": [10],
            "n_ripple_bins": [100],
            "n_effective_shuffles": [100],
            "turn_group_scheme_applicable": [True],
            "turn_group_scheme_reason": ["ok"],
            "turn_group_n_valid_ripples": [10],
            "turn_group_match_rate": [turn_group_match_rate],
            "turn_group_match_rate_shuffle_mean": [0.5],
            "turn_group_match_rate_shuffle_sd": [0.03],
            "turn_group_match_rate_p_value": [0.03],
            "arm_identity_scheme_applicable": [True],
            "arm_identity_scheme_reason": ["ok"],
            "arm_identity_n_valid_ripples": [10],
            "arm_identity_match_rate": [arm_identity_match_rate],
            "arm_identity_match_rate_shuffle_mean": [1.0 / 3.0],
            "arm_identity_match_rate_shuffle_sd": [0.02],
            "arm_identity_match_rate_p_value": [0.04],
        }
    ).to_parquet(path, index=False)
    return path


def test_load_ripple_decoding_comparison_panel_tables_reads_light_dark_metrics(
    tmp_path: Path,
) -> None:
    _write_decoding_comparison_summary_table(
        tmp_path,
        train_epoch="02_r1",
        decode_epoch="02_r1",
        turn_group_match_rate=0.4,
        arm_identity_match_rate=0.5,
    )
    _write_decoding_comparison_summary_table(
        tmp_path,
        train_epoch="08_r4",
        decode_epoch="08_r4",
        turn_group_match_rate=0.7,
        arm_identity_match_rate=0.8,
    )

    payload = load_ripple_decoding_comparison_panel_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    summary_table = payload["summary_table"]
    assert payload["missing_artifacts"] == []
    assert len(summary_table) == 4
    assert set(summary_table["epoch_type"]) == {"light", "dark"}
    assert set(summary_table["label_scheme"]) == {"turn_group", "arm_identity"}
    assert sorted(summary_table["categorical_match_rate"].unique().tolist()) == [
        0.4,
        0.5,
        0.7,
        0.8,
    ]


def test_compute_significance_distribution_comparison_uses_session_strata() -> None:
    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15"],
            "date": ["20240611", "20240611", "20241121", "20241121"],
            "epoch": ["08_r4", "08_r4", "10_r5", "10_r5"],
            "same_turn_tuning_similarity": [0.2, 0.8, 0.4, 0.9],
            "ripple_devexp_p_value": [0.2, 0.01, 0.2, 0.01],
        }
    )

    stats = compute_significance_distribution_comparison(
        table,
        metric_column="same_turn_tuning_similarity",
        n_permutations=100,
        random_seed=1,
    )

    assert stats["n_significant"] == 2
    assert stats["n_nonsignificant"] == 2
    assert stats["median_difference"] == pytest.approx(0.55)
    assert 0.0 < float(stats["p_value"]) <= 1.0


def test_load_example_glm_prediction_selects_top_devexp_unit(tmp_path: Path) -> None:
    _write_ripple_glm_dataset(tmp_path)

    example = load_example_glm_prediction(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )

    assert example["unit_id"] == 12
    assert example["ripple_devexp_mean"] == pytest.approx(0.4)
    assert np.allclose(example["observed"], [0.0, 1.0, 3.0])
    assert np.allclose(example["predicted"], [0.2, 1.3, 2.5])


def test_load_glm_prediction_for_unit_selects_requested_unit(tmp_path: Path) -> None:
    _write_ripple_glm_dataset(tmp_path)

    example = load_glm_prediction_for_unit(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        unit_id=11,
    )

    assert example["unit_id"] == 11
    assert example["ripple_devexp_mean"] == pytest.approx(0.1)
    assert np.allclose(example["observed"], [1.0, 2.0, 0.0])
    assert np.allclose(example["predicted"], [0.8, 1.7, 0.2])


def test_load_panel_b_prediction_examples_preserves_selected_order(tmp_path: Path) -> None:
    _write_ripple_glm_dataset(tmp_path)

    examples = load_panel_b_prediction_examples(
        tmp_path,
        examples=(
            ("L14", "20240611", "08_r4", 12),
            ("L14", "20240611", "08_r4", 11),
        ),
    )

    assert [example["unit_id"] for example in examples] == [12, 11]


def test_load_first_available_glm_prediction_skips_old_schema(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")
    old_path = get_ripple_glm_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )
    old_path.parent.mkdir(parents=True)
    xr.Dataset(
        data_vars={
            "ripple_devexp_mean": (("unit",), np.array([0.1])),
            "ripple_devexp_p_value": (("unit",), np.array([0.5])),
            "ripple_bits_per_spike_mean": (("unit",), np.array([0.01])),
        },
        coords={"unit": np.array([11])},
    ).to_netcdf(old_path)
    _write_ripple_glm_dataset(
        tmp_path,
        animal_name="L19",
        date="20250930",
        epoch="08_r4",
    )

    example = load_first_available_glm_prediction(
        tmp_path,
        preferred_dataset=("L14", "20240611", "08_r4"),
        candidate_datasets=[("L19", "20250930", "08_r4")],
    )

    assert example["unit_id"] == 12
    assert example["ripple_devexp_mean"] == pytest.approx(0.4)


def test_plot_helpers_draw_expected_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peri_table = pd.DataFrame(
        {
            "region": ["v1", "v1", "ca1", "ca1"],
            "unit_id": [11, 11, 101, 101],
            "time_s": [-0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 3.0, 4.0],
        }
    )
    summary_table = pd.DataFrame(
        {
            "region": ["v1", "v1", "ca1", "ca1"],
            "ripple_modulation_index": [0.2, -0.1, 0.3, 0.4],
        }
    )
    glm_table = pd.DataFrame(
        {
            "ripple_devexp_mean": [0.1, 0.4],
            "ripple_devexp_p_value": [0.2, 0.01],
        }
    )
    glm_epoch_table = pd.DataFrame(
        {
            "ripple_devexp_mean": [-2.0, 0.4],
            "ripple_devexp_p_value": [0.2, 0.001],
        }
    )
    glm_epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "epoch": "02_r1",
            "summary_table": glm_epoch_table,
        },
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "epoch": "registered",
            "summary_table": glm_epoch_table,
        },
        {
            "epoch_type": "sleep",
            "label": "Sleep",
            "epoch": "07_s4",
            "summary_table": glm_epoch_table,
        },
    ]
    prediction = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "unit_id": 12,
        "observed": np.array([0.0, 1.0, 3.0]),
        "predicted": np.array([0.2, 1.3, 2.5]),
        "ripple_devexp_mean": 0.4,
        "ripple_devexp_p_value": 0.01,
    }
    trace = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "channel": 101,
        "time_s": np.array([-0.01, 0.0, 0.01]),
        "filtered_lfp": np.array([0.0, 1.0, 0.0]),
        "ripple_duration_s": 0.05,
        "mean_zscore": 5.0,
        "n_ripples": 2,
    }
    xcorr_payload = {
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "02_r1",
        "v1_order_reference_ca1_unit": 11,
        "display_vmax": 5.0,
        "ca1_unit_ids": np.array([11, 10]),
        "v1_unit_ids": np.array([101, 102, 103]),
        "lag_s": np.array([-0.01, 0.0, 0.01]),
        "xcorr": np.ones((2, 3, 3), dtype=float),
    }
    association_payload = {
        "devexp_table": pd.DataFrame(
            {
                "epoch_type": [
                    "light",
                    "light",
                    "light",
                    "light",
                    "dark",
                    "dark",
                    "sleep",
                    "sleep",
                ],
                "dark_firing_rate_hz": [0.2, 0.4, 0.8, 9.0, 0.2, 8.0, 0.3, 12.0],
                "same_turn_tuning_similarity": [
                    0.2,
                    0.4,
                    0.7,
                    0.9,
                    0.6,
                    0.8,
                    0.3,
                    0.85,
                ],
                "ripple_devexp_mean": [0.05, 0.2, 0.3, 0.4, 0.1, 0.3, 0.15, 0.25],
                "ripple_devexp_p_value": [
                    0.001,
                    0.2,
                    0.003,
                    0.004,
                    0.003,
                    0.004,
                    0.002,
                    0.6,
                ],
                "animal_name": [
                    "RatA",
                    "RatB",
                    "RatA",
                    "RatB",
                    "RatA",
                    "RatB",
                    "RatA",
                    "RatB",
                ],
                "date": [
                    "20240101",
                    "20240102",
                    "20240101",
                    "20240102",
                    "20240101",
                    "20240102",
                    "20240101",
                    "20240102",
                ],
            }
        ),
        "missing_artifacts": [],
        "dark_activity_threshold_hz": 0.5,
    }
    source_comparison_payload = {
        "comparison_table": pd.DataFrame(
            {
                "animal_name": ["RatA", "RatA", "RatB", "RatB"],
                "date": ["20240101", "20240101", "20240102", "20240102"],
                "epoch": ["02_r1", "02_r1", "02_r1", "02_r1"],
                "epoch_type": ["light", "light", "light", "light"],
                "unit_id": [1, 2, 3, 4],
                "mean_activity_devexp_mean": [0.0, 0.1, 0.02, 0.05],
                "vector_devexp_mean": [0.1, 0.2, 0.04, 0.15],
                "vector_devexp_p_value": [0.2, 0.01, 0.03, 0.4],
                "mean_activity_devexp_p_value": [0.3, 0.02, 0.2, 0.5],
            }
        ),
        "missing_artifacts": [],
        "ripple_selection": "single",
    }
    decoding_payload = {
        "summary_table": pd.DataFrame(
            {
                "animal_name": ["L14", "L14", "L14", "L14"],
                "date": ["20240611", "20240611", "20240611", "20240611"],
                "representation": ["place", "place", "place", "place"],
                "decode_epoch": ["02_r1", "08_r4", "02_r1", "08_r4"],
                "epoch_type": ["light", "dark", "light", "dark"],
                "label_scheme": [
                    "turn_group",
                    "turn_group",
                    "arm_identity",
                    "arm_identity",
                ],
                "categorical_match_rate": [0.6, 0.7, 0.4, 0.8],
                "categorical_match_rate_shuffle_mean": [0.5, 0.52, 0.35, 0.36],
                "categorical_match_rate_p_value": [0.04, 0.2, 0.03, 0.01],
                "chance_level": [0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0],
            }
        ),
        "categorical_metrics": (("place", "turn_group"), ("place", "arm_identity")),
        "missing_artifacts": [],
    }
    offset_payload = {
        "summary_table": pd.DataFrame(
            {
                "animal_name": ["L14"] * 8,
                "date": ["20240611"] * 8,
                "epoch": ["02_r1"] * 4 + ["08_r4"] * 4,
                "epoch_type": ["light"] * 4 + ["dark"] * 4,
                "epoch_label": ["Light run"] * 4 + ["Dark run"] * 4,
                "target_window_offset_s": [-0.4, -0.2, 0.0, 0.2] * 2,
                "target_window_label": [
                    "-400 to -200",
                    "-200 to 0",
                    "0 to 200",
                    "200 to 400",
                ]
                * 2,
                "fraction_significant_positive": [0.1, 0.2, 0.3, 0.25, 0.05, 0.1, 0.2, 0.15],
                "median_devexp_significant": [0.03, 0.04, 0.06, 0.05, 0.02, 0.03, 0.04, 0.035],
                "n_units": [10] * 8,
                "n_significant_positive": [1, 2, 3, 2, 1, 1, 2, 2],
            }
        ),
        "unit_table": pd.DataFrame(),
        "missing_artifacts": [],
        "skipped_comparisons": [],
        "target_window_offsets_s": (-0.4, -0.2, 0.0, 0.2),
        "source_window_s": 0.2,
        "source_window_offset_s": 0.0,
        "target_window_s": 0.2,
    }

    epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "02_r1",
            "firing_rate_table": peri_table,
            "summary_table": summary_table,
        },
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "08_r4",
            "firing_rate_table": peri_table,
            "summary_table": summary_table,
        },
        {
            "epoch_type": "sleep",
            "label": "Sleep",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "07_s4",
            "firing_rate_table": peri_table,
            "summary_table": summary_table,
        },
    ]

    fig, axes = plt.subplots(3, 3)
    plot_ripple_lfp_panel(axes[0, 0], trace)
    plot_peri_ripple_heatmap_panel(axes[0, 1], peri_table, regions=("v1", "ca1"))
    plot_modulation_index_panel(axes[0, 2], summary_table, regions=("v1", "ca1"))
    draw_ripple_glm_schematic(axes[1, 0])
    plot_glm_summary_panel(axes[1, 1], glm_table)
    plot_observed_predicted_panel(axes[1, 2], prediction)
    plot_epoch_ripple_heatmap_panel(axes[2, 0], epoch_tables, regions=("v1", "ca1"))
    plot_top_ca1_xcorr_panel(axes[2, 1], xcorr_payload)
    plot_glm_analysis_panel(axes[2, 2], glm_epoch_tables)

    assert len(axes[0, 1].child_axes) == 3
    assert len(axes[2, 0].child_axes) == 10
    assert len(axes[2, 1].images) == 0
    assert len(axes[2, 1].child_axes) == 3
    assert all(len(child_axis.images) == 1 for child_axis in axes[2, 1].child_axes[:2])
    assert all(child_axis.get_xlabel() == "" for child_axis in axes[2, 1].child_axes[:2])
    assert "Lag (s)" in [text.get_text() for text in axes[2, 1].texts]
    lag_label = next(text for text in axes[2, 1].texts if text.get_text() == "Lag (s)")
    assert lag_label.get_position()[1] == pytest.approx(0.035)
    panel_b_y_label = next(text for text in axes[2, 1].texts if text.get_text() == "V1")
    assert panel_b_y_label.get_fontsize() == pytest.approx(6.0)
    assert len(axes[1, 0].patches) >= 3
    assert len(axes[1, 1].collections) == 1
    assert len(axes[1, 2].collections) == 1
    assert len(axes[2, 2].child_axes) == 7
    assert axes[2, 2].child_axes[1].get_xlim()[0] == pytest.approx(-0.1)
    assert axes[2, 2].child_axes[1].get_xlim()[1] == pytest.approx(0.5)
    assert len(axes[2, 2].child_axes[1].collections) == 2
    assert axes[2, 2].child_axes[2].get_xlim()[0] == pytest.approx(-0.1)
    assert axes[2, 2].child_axes[2].get_xlim()[1] == pytest.approx(0.5)
    assert len(axes[2, 2].child_axes[2].patches) == 2
    assert [tick.get_text() for tick in axes[2, 2].child_axes[2].get_yticklabels()] == [
        "n.s.",
        "",
    ]
    assert (
        axes[2, 2].child_axes[1].get_ylabel()
        == "-log10 p from shuffle"
    )
    assert any(
        "frac sig=" in text.get_text()
        for text in axes[2, 2].child_axes[1].texts
    )
    plt.close(fig)

    prediction_examples = [
        {
            **prediction,
            "animal_name": animal_name,
            "unit_id": unit_id,
            "ripple_devexp_mean": devexp,
        }
        for animal_name, unit_id, devexp in (
            ("L12", 24, 0.33),
            ("L19", 4, 0.24),
            ("L15", 476, 0.27),
        )
    ]
    fig, ax = plt.subplots()
    plot_glm_analysis_panel(
        ax,
        glm_epoch_tables,
        prediction_examples=prediction_examples,
    )
    assert len(ax.child_axes) == 10
    assert [child.get_title() for child in ax.child_axes[1:4]] == [
        "Example cell 1\n(Dev. exp. 0.33)",
        "Example cell 2\n(Dev. exp. 0.24)",
        "Example cell 3\n(Dev. exp. 0.27)",
    ]
    assert ax.child_axes[3].get_xlabel() == "Actual count"
    assert all(len(child.artists) == 0 for child in ax.child_axes[1:4])
    assert all(len(child.patches) == 0 for child in ax.child_axes[1:4])
    assert all(len(child.collections) == 1 for child in ax.child_axes[1:4])
    assert all(
        child.collections[0].get_sizes()[0] == pytest.approx(2.2)
        for child in ax.child_axes[1:4]
    )
    assert ax.child_axes[1].get_xlim()[0] == pytest.approx(0.0)
    assert ax.child_axes[1].get_xlim()[1] == pytest.approx(4.0)
    assert all(
        child.get_xlim() == pytest.approx(ax.child_axes[1].get_xlim())
        for child in ax.child_axes[1:4]
    )
    assert all(
        child.get_ylim() == pytest.approx(ax.child_axes[1].get_xlim())
        for child in ax.child_axes[1:4]
    )
    for child in ax.child_axes[1:4]:
        np.testing.assert_allclose(child.get_xticks(), [0.0, 2.0, 4.0])
        np.testing.assert_allclose(child.get_yticks(), [0.0, 2.0, 4.0])
    assert [child.get_ylabel() for child in ax.child_axes[1:4]] == [
        "",
        "Predicted count",
        "",
    ]
    assert all(len(child.texts) == 0 for child in ax.child_axes[1:4])
    fig.canvas.draw()
    parent_position = ax.get_position()
    example_positions = [child.get_position() for child in ax.child_axes[1:4]]
    example_heights = [
        position.height / parent_position.height
        for position in example_positions
    ]
    for child in ax.child_axes[1:4]:
        origin = child.transData.transform((0.0, 0.0))
        x_unit = child.transData.transform((1.0, 0.0))
        y_unit = child.transData.transform((0.0, 1.0))
        x_distance = float(np.linalg.norm(x_unit - origin))
        y_distance = float(np.linalg.norm(y_unit - origin))
        assert x_distance == pytest.approx(y_distance, rel=1e-6)
    example_gaps = [
        (example_positions[index].y0 - example_positions[index + 1].y1)
        / parent_position.height
        for index in range(len(example_positions) - 1)
    ]
    assert max(example_heights) < 0.19
    assert min(example_gaps) > 0.08
    summary_positions = [child.get_position() for child in ax.child_axes[4:]]
    example_bottom = min(position.y0 for position in example_positions)
    summary_box_bottoms = [
        position.y0 for position in summary_positions[1::2]
    ]
    assert all(
        summary_bottom == pytest.approx(example_bottom, abs=0.01)
        for summary_bottom in summary_box_bottoms
    )
    example_top = max(position.y1 for position in example_positions)
    summary_scatter_tops = [
        position.y1 for position in summary_positions[::2]
    ]
    assert all(
        summary_top == pytest.approx(example_top, abs=0.01)
        for summary_top in summary_scatter_tops
    )
    assert ax.child_axes[4].get_ylabel() == "-log10 p from shuffle"
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_glm_behavior_association_panel(ax, association_payload)
    assert len(ax.child_axes) == 3
    fraction_ax, box_ax, similarity_ax = ax.child_axes
    assert fraction_ax.get_xlabel() == "$p$<0.05\nfrac."
    assert len(fraction_ax.artists) == 0
    assert fraction_ax.get_ylabel() == ""
    assert fraction_ax.get_title() == ""
    assert box_ax.get_ylabel() == ""
    assert box_ax.get_xlabel() == "Dev.\nexplained"
    assert box_ax.get_title() == ""
    assert similarity_ax.get_xlabel() == "Dark\nDPPI"
    assert similarity_ax.get_ylabel() == "Fraction"
    assert similarity_ax.get_title() == ""
    label_y_values = [
        child_axis.xaxis.label.get_position()[1]
        for child_axis in (fraction_ax, box_ax, similarity_ax)
    ]
    assert len(set(label_y_values)) == 1
    fig.canvas.draw()
    box_position = box_ax.get_position()
    fraction_position = fraction_ax.get_position()
    similarity_position = similarity_ax.get_position()
    parent_position = ax.get_position()
    fraction_width = fraction_position.width / parent_position.width
    box_width = box_position.width / parent_position.width
    assert fraction_width > 0.23
    assert box_width > 0.32
    assert similarity_position.x0 - box_position.x1 > 0.18
    assert ax.texts[-1].get_text() == (
        "Plots show p<0.05 units; dark active split uses 0.5 Hz"
    )
    assert len(fraction_ax.patches) == 2
    assert len(box_ax.patches) == 2
    assert len(box_ax.collections) == 2
    assert len(box_ax.lines) >= 3
    assert len(fraction_ax.collections) == 3
    assert len(fraction_ax.lines) >= 2
    assert len(similarity_ax.patches) == 10
    assert len(similarity_ax.collections) == 0
    assert len(similarity_ax.lines) == 2
    similarity_bin_edges = np.asarray(
        [patch.get_x() for patch in similarity_ax.patches]
        + [
            similarity_ax.patches[-1].get_x()
            + similarity_ax.patches[-1].get_width()
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.diff(similarity_bin_edges), 0.1, atol=1e-9)
    assert np.any(np.isclose(similarity_bin_edges, 0.0))
    assert similarity_ax.patches[0].get_edgecolor()[3] == pytest.approx(0.0)
    np.testing.assert_allclose(
        fraction_ax.patches[0].get_facecolor()[:3],
        to_rgba(PANEL_D_DARK_ACTIVITY_COLORS["inactive"])[:3],
    )
    np.testing.assert_allclose(
        fraction_ax.patches[1].get_facecolor()[:3],
        to_rgba(PANEL_D_DARK_ACTIVITY_COLORS["active"])[:3],
    )
    np.testing.assert_allclose(
        box_ax.patches[0].get_facecolor()[:3],
        to_rgba(PANEL_D_DARK_ACTIVITY_COLORS["inactive"])[:3],
    )
    np.testing.assert_allclose(
        box_ax.patches[1].get_facecolor()[:3],
        to_rgba(PANEL_D_DARK_ACTIVITY_COLORS["active"])[:3],
    )
    np.testing.assert_allclose(
        similarity_ax.patches[0].get_facecolor()[:3],
        to_rgba(PANEL_D_DARK_ACTIVITY_COLORS["active"])[:3],
    )
    assert fraction_ax.get_xlim()[0] == pytest.approx(0.0)
    assert fraction_ax.get_xlim()[1] == pytest.approx(1.0)
    assert [tick.get_text() for tick in fraction_ax.get_yticklabels()] == [
        "Dark\ninactive",
        "Dark\nactive",
    ]
    assert sum(
        len(collection.get_offsets())
        for collection in box_ax.collections
    ) == 3
    assert box_ax.get_xlim()[0] == pytest.approx(-0.1)
    assert box_ax.get_xlim()[1] == pytest.approx(0.5)
    assert similarity_ax.get_xlim()[0] == pytest.approx(0.0)
    assert similarity_ax.get_xlim()[1] == pytest.approx(1.0)
    devexp_box_line_max = max(
        np.nanmax(np.asarray(line.get_xdata(), dtype=float))
        for line in box_ax.lines
        if len(line.get_xdata()) and len(line.get_xdata()) != 4
    )
    assert devexp_box_line_max == pytest.approx(0.4)
    box_significance_markers = [
        text for text in box_ax.texts if text.get_text() == "*"
    ]
    assert len(box_significance_markers) == 1
    assert box_significance_markers[0].get_position()[0] > devexp_box_line_max
    assert box_significance_markers[0].get_position()[1] == pytest.approx(1.5)
    box_bracket_lines = [
        line for line in box_ax.lines if len(line.get_xdata()) == 4
    ]
    assert box_bracket_lines
    assert min(box_bracket_lines[-1].get_xdata()) > devexp_box_line_max
    assert [text.get_text() for text in similarity_ax.texts] == ["median=0.80"]
    assert any(text.get_text() == "0.33\nn=1" for text in fraction_ax.texts)
    assert any(text.get_text() == "0.67\nn=2" for text in fraction_ax.texts)
    significance_markers = [
        text for text in fraction_ax.texts if text.get_text() == "*"
    ]
    assert len(significance_markers) == 1
    assert significance_markers[0].get_position()[0] == pytest.approx(1.05)
    assert significance_markers[0].get_position()[1] == pytest.approx(1.5)
    bracket_lines = [
        line for line in fraction_ax.lines if len(line.get_xdata()) == 4
    ]
    assert bracket_lines
    np.testing.assert_allclose(
        bracket_lines[-1].get_xdata(),
        [0.975, 1.02, 1.02, 0.975],
    )
    np.testing.assert_allclose(bracket_lines[-1].get_ydata(), [1.0, 1.0, 2.0, 2.0])
    assert all(tick.get_text() == "" for tick in box_ax.get_yticklabels())
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_glm_source_predictor_comparison_panel(ax, source_comparison_payload)
    assert len(ax.child_axes) == 3
    assert [child.get_title() for child in ax.child_axes] == ["RatA", "RatB", "Pooled"]
    assert ax.child_axes[0].get_xlim()[0] == pytest.approx(-0.2)
    assert ax.child_axes[0].get_xlim()[1] == pytest.approx(0.5)
    assert ax.child_axes[0].get_ylim()[0] == pytest.approx(-0.2)
    assert ax.child_axes[0].get_ylim()[1] == pytest.approx(0.5)
    assert len(ax.child_axes[0].collections) == 1
    assert len(ax.child_axes[2].collections) == 2
    assert len(ax.child_axes[0].lines) == 1
    assert any(
        text.get_text() == "Mean CA1 activity\ndev. explained"
        for text in ax.texts
    )
    assert any(
        text.get_text() == "CA1 spike vector\ndev. explained"
        for text in ax.texts
    )
    assert any(
        text.get_text() == "Showing vector p<0.05 units"
        for text in ax.texts
    )
    assert any(
        text.get_text() == "n=1\nfrac vector>mean=1.00"
        for text in ax.child_axes[0].texts
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_glm_source_predictor_comparison_panel(
        ax,
        source_comparison_payload,
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
    )
    fig.canvas.draw()
    parent_position = ax.get_position()
    child_position = ax.child_axes[0].get_position()
    compact_child_top = (child_position.y1 - parent_position.y0) / parent_position.height
    assert compact_child_top < 0.75
    assert ax.child_axes[0].get_xlabel() == "Mean CA1\ndev. explained"
    assert ax.child_axes[0].get_ylabel() == "CA1 vector\ndev. explained"
    np.testing.assert_allclose(ax.child_axes[0].get_xticks(), ax.child_axes[0].get_yticks())
    assert [
        tick.get_text() for tick in ax.child_axes[0].get_xticklabels()
    ] == [
        tick.get_text() for tick in ax.child_axes[0].get_yticklabels()
    ]
    assert not ax.texts
    compact_summary_text = next(text for text in ax.child_axes[0].texts if text.get_text() == "n=2")
    assert compact_summary_text.get_position() == pytest.approx((0.97, 0.05))
    assert compact_summary_text.get_ha() == "right"
    assert compact_summary_text.get_va() == "bottom"
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_ripple_decoding_comparison_panel(ax, decoding_payload)
    assert len(ax.child_axes) == 2
    assert [child_axis.get_title() for child_axis in ax.child_axes] == [
        "Turn group",
        "Arm",
    ]
    assert ax.child_axes[1].get_xlabel() == "Decode epoch"
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_glm_offset_panel(ax, offset_payload)
    assert len(ax.child_axes) == 2
    assert ax.child_axes[0].get_title() == "CA1 0-200 ms -> V1 target window"
    assert ax.child_axes[1].get_xlabel() == "V1 target window (ms)"
    plt.close(fig)


def test_parse_arguments_defaults_match_figure_3_cli() -> None:
    args = parse_arguments([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == "figure_3"
    assert args.output_format == "pdf"
    assert args.example_dataset == DEFAULT_EXAMPLE_DATASET
    assert args.light_epoch is None
    assert args.dark_epoch is None
    assert args.sleep_epoch is None
    assert args.dark_movement_fr_cache_dir == DEFAULT_FIGURE_CACHE_DIR
    assert args.refresh_dark_movement_fr_cache is False
    assert args.refresh_panel_b_schematic_cache is False
    assert args.ripple_threshold_zscore == DEFAULT_RIPPLE_THRESHOLD_ZSCORE
    assert args.ripple_window_s == DEFAULT_RIPPLE_WINDOW_S
    assert args.ripple_window_offset_s == DEFAULT_RIPPLE_WINDOW_OFFSET_S
    assert args.ripple_selection == DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION
    assert args.ridge_strength == DEFAULT_RIDGE_STRENGTH
