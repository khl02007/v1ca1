from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.old_fig3 as old_fig3_module
from v1ca1.paper_figures.old_fig3 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_LIGHT_EPOCH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PANEL_AB_HEIGHT_MM,
    DEFAULT_PANEL_A_WIDTH_FRACTION,
    DEFAULT_PANEL_B_WIDTH_FRACTION,
    DEFAULT_PANEL_DEF_HEIGHT_MM,
    DEFAULT_REGIONS,
    PANEL_B_CACHE_VERSION,
    PANEL_B_FIRING_RATE_NORMALIZATION,
    PANEL_A_EXAMPLE_TOP,
    PANEL_A_FIRST_EXAMPLE_Y_SHIFT,
    PANEL_B_HEATMAP_CMAP,
    PANEL_B_LINEAR_POSITION_ORIENTATION,
    PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_B_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_TRAJECTORY_TYPES,
    PANEL_A_DARK_EPOCH_BACKGROUND,
    PANEL_A_EXAMPLES,
    PANEL_TRAJECTORY_COLORS,
    PANEL_C_SCATTER_ALPHA,
    PANEL_C_SCATTER_SIZE,
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_D_ENCODING_X_LIMITS,
    PANEL_E_CROSS_COMPARISONS,
    PANEL_E_NORM_ERROR_YLIM,
    PANEL_E_PLACE_ERROR_YLIM,
    PANEL_QUANT_SUMMARY_TEXT_FONTSIZE,
    GLM_TRAJECTORY_ARROW_COLOR,
    GLM_BASIS_DARK_COLOR,
    GLM_MODEL_COLORS,
    PANEL_H_DELTA_TRAJECTORIES,
    PANEL_H_EXAMPLES,
    PANEL_H_DELTA_AXIS_BOUNDS,
    PANEL_H_SCHEMATIC_AXIS_BOUNDS,
    PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
    PANEL_H_SWAP_DELTA_VARIABLE,
    PANEL_QUANT_EPOCH_COLORS,
    PANEL_EXAMPLE_CACHE_VERSION,
    SEGMENT_BOUNDARIES,
    build_panel_b_cache_metadata,
    build_panel_b_cache_path,
    build_panel_example_cache_metadata,
    build_panel_example_cache_path,
    build_panel_c_similarity_pairs,
    build_output_path,
    get_decoding_summary_path,
    get_dark_light_glm_selected_path,
    get_encoding_summary_candidate_paths,
    get_dark_epoch,
    get_light_epoch,
    get_swap_glm_selected_comparison_path,
    get_stability_table_path,
    get_tuning_similarity_path,
    load_panel_h_swap_delta_table,
    load_panel_d_encoding_delta_table,
    load_panel_e_decoding_error_table,
    load_panel_quantification_data,
    load_or_compute_panel_example_data,
    make_light_epoch_dataset_ids,
    parse_arguments,
    parse_dataset_id,
    plot_panel_c_similarity,
    plot_panel_d_encoding_delta_histogram,
    plot_panel_e_decoding_error,
    plot_panel_h_swap_delta,
    plot_epoch_path_rate_axis,
    plot_light_heatmap_regions,
    plot_panel_a_examples,
    plot_panel_c_vision_tuning_panel,
    plot_panel_d_route_place_panel,
    save_panel_b_cache,
    save_panel_example_cache,
    load_panel_b_cache,
    load_panel_example_cache,
    setup_light_heatmap_panel,
    validate_panel_a_trajectories,
)


def test_parse_dataset_id_requires_animal_and_date() -> None:
    assert parse_dataset_id("L14:20240611") == ("L14", "20240611", "08_r4")
    assert parse_dataset_id("L15:20241121:10_r5") == ("L15", "20241121", "10_r5")

    with pytest.raises(argparse.ArgumentTypeError, match="animal:date"):
        parse_dataset_id("L14")


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "old_fig3", "svg") == Path(
        "paper_figures/old_fig3.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "old_fig3", "jpg")


def test_quantification_artifact_paths_match_task_progression_scripts() -> None:
    data_root = Path("/analysis")
    assert get_tuning_similarity_path(
        data_root,
        animal_name="L15",
        date="20241121",
        region="v1",
        epoch="10_r5",
    ) == Path(
        "/analysis/L15/20241121/task_progression/tuning_analysis/"
        "v1_10_r5_correlation_within_epoch_similarity.parquet"
    )
    assert get_encoding_summary_candidate_paths(
        data_root,
        animal_name="L15",
        date="20241121",
        region="v1",
        epoch="10_r5",
    )[0] == Path(
        "/analysis/L15/20241121/task_progression/encoding_comparison/"
        "v1_10_r5_cv5_placebin4cm_encoding_summary.parquet"
    )
    assert get_decoding_summary_path(
        data_root,
        animal_name="L15",
        date="20241121",
        region="v1",
        epoch="10_r5",
    ) == Path(
        "/analysis/L15/20241121/task_progression/decoding_comparison/"
        "v1_10_r5_decoding_summary.parquet"
    )
    assert get_dark_light_glm_selected_path(
        data_root,
        animal_name="L15",
        date="20241121",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="10_r5",
        model_name="visual",
    ) == Path(
        "/analysis/L15/20241121/task_progression/dark_light_glm/selected/"
        "v1_02_r1_vs_10_r5_visual_selected.nc"
    )
    assert get_swap_glm_selected_comparison_path(
        data_root,
        animal_name="L15",
        date="20241121",
        region="v1",
        dark_epoch="10_r5",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
    ) == Path(
        "/analysis/L15/20241121/task_progression/swap_glm_comparison/"
        "v1_10_r5_traindark_02_r1_trainlight_06_r3_testlight_"
        "dark_light_selected_swap.nc"
    )


def test_load_panel_quantification_data_reports_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tuning_analysis"):
        load_panel_quantification_data(
            data_root=tmp_path,
            datasets=[("L15", "20241121", "10_r5")],
            region="v1",
            light_epoch=None,
            dark_epoch=None,
        )


def test_load_panel_h_swap_delta_table_filters_by_dark_tuning_stability(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    xr = pytest.importorskip("xarray")

    swap_path = get_swap_glm_selected_comparison_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
    )
    swap_path.parent.mkdir(parents=True, exist_ok=True)
    delta = np.full((2, 2, 3), np.nan, dtype=float)
    delta[1] = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=float,
    )
    xr.Dataset(
        {
            PANEL_H_SWAP_DELTA_VARIABLE: (
                ("model", "trajectory", "unit"),
                delta,
            )
        },
        coords={
            "model": ["visual", "task_segment_bump"],
            "trajectory": ["center_to_left", "center_to_right"],
            "unit": [11, 12, 13],
        },
    ).to_netcdf(swap_path)

    stability_path = get_stability_table_path(tmp_path, "L14", "20240611")
    stability_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "animal_name": ["L14"] * 6,
            "date": ["20240611"] * 6,
            "unit": [11, 12, 13, 13, 11, 11],
            "region": ["v1", "v1", "v1", "v1", "v1", "ca1"],
            "epoch": ["08_r4", "08_r4", "08_r4", "08_r4", "02_r1", "08_r4"],
            "trajectory_type": [
                "center_to_left",
                "center_to_left",
                "center_to_left",
                "center_to_right",
                "center_to_left",
                "center_to_left",
            ],
            "stability_correlation": [0.51, 0.50, 0.2, 0.7, 0.9, 0.99],
        }
    ).to_parquet(stability_path)

    table = load_panel_h_swap_delta_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        dark_epoch=None,
        light_epoch_pairs=(("02_r1", "06_r3"),),
        min_tuning_stability_correlation=0.5,
    )

    assert table["unit"].tolist() == [11, 13, 11, 13]
    assert table["delta_ll_bits_per_spike"].tolist() == pytest.approx(
        [0.1, 0.3, 0.4, 0.6]
    )


def test_load_panel_h_swap_delta_table_can_select_segment_scalar_model(
    tmp_path: Path,
) -> None:
    xr = pytest.importorskip("xarray")

    swap_path = get_swap_glm_selected_comparison_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
    )
    swap_path.parent.mkdir(parents=True, exist_ok=True)
    delta = np.asarray(
        [
            [[0.0, 0.0]],
            [[0.1, 0.2]],
            [[0.3, 0.4]],
        ],
        dtype=float,
    )
    xr.Dataset(
        {
            PANEL_H_SWAP_DELTA_VARIABLE: (
                ("model", "trajectory", "unit"),
                delta,
            )
        },
        coords={
            "model": ["visual", "task_segment_bump", "task_segment_scalar"],
            "trajectory": ["center_to_left"],
            "unit": [11, 12],
        },
    ).to_netcdf(swap_path)

    table = load_panel_h_swap_delta_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        dark_epoch=None,
        light_epoch_pairs=(("02_r1", "06_r3"),),
        model_name="task_segment_scalar",
    )

    assert table["model_name"].tolist() == [
        "task_segment_scalar",
        "task_segment_scalar",
    ]
    assert table["delta_ll_bits_per_spike"].tolist() == pytest.approx([0.3, 0.4])


def test_load_panel_d_encoding_delta_table_filters_by_tuning_stability(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")

    for epoch, values in {
        "02_r1": [-0.2, 0.3, -0.9],
        "08_r4": [0.1, -0.4, -0.5],
    }.items():
        path = get_encoding_summary_candidate_paths(
            tmp_path,
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch=epoch,
        )[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "n_spikes": [100, 100, 100],
                "delta_bits_place_vs_tp": values,
            },
            index=pd.Index([11, 12, 13], name="unit"),
        ).to_parquet(path)

    stability_path = get_stability_table_path(tmp_path, "L14", "20240611")
    stability_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "animal_name": ["L14"] * 7,
            "date": ["20240611"] * 7,
            "unit": [11, 12, 13, 11, 12, 13, 13],
            "region": ["v1", "v1", "v1", "v1", "v1", "v1", "ca1"],
            "epoch": [
                "02_r1",
                "02_r1",
                "02_r1",
                "08_r4",
                "08_r4",
                "08_r4",
                "08_r4",
            ],
            "trajectory_type": [
                "center_to_left",
                "center_to_right",
                "not_a_trajectory",
                "center_to_left",
                "left_to_center",
                "right_to_center",
                "right_to_center",
            ],
            "stability_correlation": [0.51, 0.50, 0.99, np.nan, 0.7, 0.6, 0.95],
        }
    ).to_parquet(stability_path)

    table = load_panel_d_encoding_delta_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
    )

    assert PANEL_D_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)
    assert table[["epoch_type", "unit"]].to_dict("records") == [
        {"epoch_type": "light", "unit": 11},
        {"epoch_type": "dark", "unit": 12},
        {"epoch_type": "dark", "unit": 13},
    ]
    assert table["delta_bits_tp_vs_place"].tolist() == pytest.approx(
        [0.2, 0.4, 0.5]
    )


def test_light_and_dark_epoch_helpers_use_registered_defaults() -> None:
    assert get_light_epoch("L14", "20240611") == DEFAULT_LIGHT_EPOCH
    assert get_dark_epoch("L15", "20241121") == "10_r5"
    assert get_light_epoch("L14", "20240611", "04_r2") == "04_r2"
    assert get_dark_epoch("L14", "20240611", "12_r6") == "12_r6"


def test_panel_b_trajectory_order_uses_task_progression_orientation() -> None:
    assert PANEL_B_TRAJECTORY_TYPES == (
        "right_to_center",
        "center_to_left",
        "left_to_center",
        "center_to_right",
    )
    assert PANEL_B_LINEAR_POSITION_ORIENTATION == "task_progression"


def test_make_light_epoch_dataset_ids_keeps_registered_sessions() -> None:
    assert make_light_epoch_dataset_ids(
        [("L14", "20240611", "08_r4"), ("L15", "20241121", "10_r5")]
    ) == [
        ("L14", "20240611", "02_r1"),
        ("L15", "20241121", "02_r1"),
    ]
    assert make_light_epoch_dataset_ids(
        [("L14", "20240611", "08_r4")],
        light_epoch="04_r2",
    ) == [("L14", "20240611", "04_r2")]


def test_panel_b_cache_path_is_descriptive() -> None:
    metadata = build_panel_b_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        position_bin_count=100,
        position_offset=5,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    cache_path = build_panel_b_cache_path(Path("paper_figures/output/cache"), metadata)

    assert metadata["cache_version"] == PANEL_B_CACHE_VERSION
    assert metadata["data_root"] == "/analysis"
    assert tuple(metadata["trajectory_types"]) == PANEL_B_TRAJECTORY_TYPES
    assert metadata["linear_position_orientation"] == "task_progression"
    assert metadata["min_movement_firing_rate_hz"] == pytest.approx(
        PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ
    )
    assert metadata["min_tuning_stability_correlation"] == pytest.approx(
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    )
    assert metadata["firing_rate_normalization"] == PANEL_B_FIRING_RATE_NORMALIZATION
    assert metadata["datasets"] == [
        {
            "animal_name": "L14",
            "date": "20240611",
            "dark_epoch": "08_r4",
            "light_epoch": "02_r1",
        }
    ]
    assert cache_path == Path(
        "paper_figures/output/cache/"
        "figure_3_panel_b_v1_light02_r1_datasets-L14-20240611-02_r1"
        "_orienttask_progression_minmovefr0p5_minstab0p5"
        "_normunit_max_per_trajectory"
        "_posbins100_offset5_speed4_sigma1p5_cachev3.npz"
    )


def test_panel_b_cache_roundtrip_validates_metadata(tmp_path: Path) -> None:
    metadata = build_panel_b_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    panels = {}
    for index, order_trajectory in enumerate(PANEL_B_TRAJECTORY_TYPES):
        for plot_trajectory in PANEL_B_TRAJECTORY_TYPES:
            panels[(order_trajectory, plot_trajectory)] = np.full(
                (index + 1, 3),
                index,
                dtype=float,
            )
    cache_path = build_panel_b_cache_path(tmp_path, metadata)

    save_panel_b_cache(cache_path, panels, metadata)
    loaded = load_panel_b_cache(cache_path, metadata)

    assert loaded is not None
    for key, expected in panels.items():
        assert np.array_equal(loaded[key], expected)

    stale_metadata = dict(metadata)
    stale_metadata["position_bin_count"] = 4
    assert load_panel_b_cache(cache_path, stale_metadata) is None


def test_panel_example_cache_path_is_descriptive() -> None:
    metadata = build_panel_example_cache_metadata(
        data_root=Path("/analysis"),
        panel_name="C",
        animal_name="L15",
        date="20241121",
        epoch="10_r5",
        region="v1",
        unit_id=473,
        trajectories=("center_to_right", "left_to_center"),
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    cache_path = build_panel_example_cache_path(
        Path("paper_figures/output/cache"),
        metadata,
    )

    assert metadata["cache_version"] == PANEL_EXAMPLE_CACHE_VERSION
    assert metadata["payload"] == "raster_positions_and_firing_rates"
    assert metadata["trajectory_types"] == ["center_to_right", "left_to_center"]
    assert cache_path == Path(
        "paper_figures/output/cache/"
        "figure_3_panel_example_c_L15-20241121-10_r5-v1-unit473"
        "_traj-center_to_right-left_to_center"
        "_posbins50_offset10_speed4_sigma1p5_cachev1.npz"
    )


def test_panel_example_cache_roundtrip_validates_metadata(tmp_path: Path) -> None:
    trajectories = ("center_to_right", "left_to_center")
    metadata = build_panel_example_cache_metadata(
        data_root=Path("/analysis"),
        panel_name="C",
        animal_name="L15",
        date="20241121",
        epoch="10_r5",
        region="v1",
        unit_id=473,
        trajectories=trajectories,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    example = {
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "10_r5",
        "region": "v1",
        "unit_id": 473,
        "raster_positions": {
            "center_to_right": [np.asarray([0.1, 0.2]), np.asarray([0.4])],
            "left_to_center": [np.asarray([], dtype=float), np.asarray([0.7])],
        },
        "firing_rates": {
            "center_to_right": (np.asarray([0.0, 0.5, 1.0]), np.asarray([1.0, 2.0, 3.0])),
            "left_to_center": (np.asarray([0.0, 0.5, 1.0]), np.asarray([4.0, 5.0, 6.0])),
        },
    }
    cache_path = build_panel_example_cache_path(tmp_path, metadata)

    save_panel_example_cache(cache_path, example, metadata)
    loaded = load_panel_example_cache(cache_path, metadata)

    assert loaded is not None
    assert loaded["animal_name"] == "L15"
    assert loaded["unit_id"] == 473
    for trajectory in trajectories:
        assert len(loaded["raster_positions"][trajectory]) == len(
            example["raster_positions"][trajectory]
        )
        for loaded_trial, expected_trial in zip(
            loaded["raster_positions"][trajectory],
            example["raster_positions"][trajectory],
        ):
            assert np.array_equal(loaded_trial, expected_trial)
        loaded_position, loaded_rate = loaded["firing_rates"][trajectory]
        expected_position, expected_rate = example["firing_rates"][trajectory]
        assert np.array_equal(loaded_position, expected_position)
        assert np.array_equal(loaded_rate, expected_rate)

    stale_metadata = dict(metadata)
    stale_metadata["unit_id"] = 474
    assert load_panel_example_cache(cache_path, stale_metadata) is None


def test_load_or_compute_panel_example_data_uses_matching_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trajectories = ("center_to_right", "left_to_center")
    metadata = build_panel_example_cache_metadata(
        data_root=Path("/analysis"),
        panel_name="C",
        animal_name="L15",
        date="20241121",
        epoch="10_r5",
        region="v1",
        unit_id=473,
        trajectories=trajectories,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    example = {
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "10_r5",
        "region": "v1",
        "unit_id": 473,
        "raster_positions": {
            "center_to_right": [np.asarray([0.2])],
            "left_to_center": [np.asarray([0.8])],
        },
        "firing_rates": {
            "center_to_right": (np.asarray([0.5]), np.asarray([2.0])),
            "left_to_center": (np.asarray([0.5]), np.asarray([3.0])),
        },
    }
    save_panel_example_cache(
        build_panel_example_cache_path(tmp_path, metadata),
        example,
        metadata,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "load_epoch_unit_rate_curves",
        lambda **_kwargs: pytest.fail("Panel example cache was not used."),
    )

    loaded = load_or_compute_panel_example_data(
        data_root=Path("/analysis"),
        panel_name="C",
        animal_name="L15",
        date="20241121",
        epoch="10_r5",
        region="v1",
        unit_id=473,
        trajectories=trajectories,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_example_cache_dir=tmp_path,
        refresh_panel_example_cache=False,
    )

    assert loaded["unit_id"] == 473
    assert np.array_equal(loaded["firing_rates"]["left_to_center"][1], np.asarray([3.0]))



def test_panel_a_example_configuration_uses_requested_trajectory_pairs() -> None:
    assert PANEL_A_EXAMPLES == (
        ("L14", "20240611", "v1", 34, ("center_to_left", "right_to_center")),
        ("L15", "20241121", "v1", 473, ("center_to_right", "left_to_center")),
    )


def test_panel_h_example_configuration_uses_requested_cells() -> None:
    assert PANEL_H_EXAMPLES == (
        ("L15", "20241121", "v1", 27, "center_to_right"),
        ("L14", "20240611", "v1", 368, "right_to_center"),
    )


def test_validate_panel_a_trajectories_rejects_unknown_names() -> None:
    assert validate_panel_a_trajectories(["center_to_left", "right_to_center"]) == (
        "center_to_left",
        "right_to_center",
    )
    with pytest.raises(ValueError, match="Unknown panel A trajectory"):
        validate_panel_a_trajectories(["center_to_left", "bad"])



def _fake_panel_a_example(trajectories: tuple[str, ...]) -> dict[str, object]:
    position = np.asarray([0.0, 0.5, 1.0], dtype=float)
    return {
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "unit_id": 34,
        "trajectories": trajectories,
        "epoch_rates": {
            "dark": {
                "epoch": "08_r4",
                "raster_positions": {
                    trajectory: [np.asarray([0.1, 0.4]), np.asarray([0.7])]
                    for trajectory in trajectories
                },
                "firing_rates": {
                    trajectory: (position, np.asarray([0.0, 1.0, 0.5], dtype=float))
                    for trajectory in trajectories
                },
            },
            "light": {
                "epoch": "02_r1",
                "raster_positions": {
                    trajectory: [np.asarray([0.2]), np.asarray([0.6, 0.9])]
                    for trajectory in trajectories
                },
                "firing_rates": {
                    trajectory: (position, np.asarray([0.5, 0.2, 1.5], dtype=float))
                    for trajectory in trajectories
                },
            },
        },
    }


def test_plot_epoch_path_rate_axis_overlays_path_type_curves() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = _fake_panel_a_example(("center_to_left", "right_to_center"))
    fig, ax = plt.subplots()
    plot_epoch_path_rate_axis(
        ax,
        example,
        "dark",
        y_max=2.0,
        show_ylabel=True,
        show_legend=True,
    )

    assert len(ax.lines) == 4
    assert [line.get_color() for line in ax.lines[:2]] == [
        PANEL_TRAJECTORY_COLORS["center_to_left"],
        PANEL_TRAJECTORY_COLORS["right_to_center"],
    ]
    assert [line.get_xdata()[0] for line in ax.lines[2:]] == pytest.approx(
        list(SEGMENT_BOUNDARIES)
    )
    assert ax.get_ylabel() == "FR (Hz)"
    assert ax.get_xlabel() == old_fig3_module.TASK_PROGRESSION_XLABEL
    assert ax.get_title() == "Dark"
    assert ax.get_legend() is not None
    plt.close(fig)



def test_plot_panel_a_examples_stacks_two_curve_blocks() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    examples = [
        _fake_panel_a_example(("center_to_left", "right_to_center")),
        _fake_panel_a_example(("center_to_right", "left_to_center")),
    ]
    plot_panel_a_examples(
        ax,
        examples,
    )

    assert len(ax.child_axes) == 2
    parent_position = ax.get_position()
    assert ax.child_axes[0].get_position().y1 == pytest.approx(
        parent_position.y0 + parent_position.height * PANEL_A_EXAMPLE_TOP
    )
    for example_index, (example_ax, example) in enumerate(zip(ax.child_axes, examples), start=1):
        assert [text.get_text() for text in example_ax.texts] == [
            f"Example cell {example_index}"
        ]
        assert example_ax.texts[0].get_position()[0] == pytest.approx(0.50)
        expected_title_y = (
            0.885 + PANEL_A_FIRST_EXAMPLE_Y_SHIFT
            if example_index == 1
            else 0.885
        )
        assert example_ax.texts[0].get_position()[1] == pytest.approx(expected_title_y)
        assert example_ax.texts[0].get_horizontalalignment() == "center"
        assert example_ax.texts[0].get_fontsize() == pytest.approx(5.8)
        assert len(example_ax.child_axes) == 6
        schematic_axes = example_ax.child_axes[:2]
        schematic_patches = [
            patch
            for schematic_ax in schematic_axes
            for patch in schematic_ax.patches
            if isinstance(patch, Polygon)
        ]
        assert schematic_patches
        assert all(patch.get_facecolor()[3] == pytest.approx(0.0) for patch in schematic_patches)
        raster_axes = example_ax.child_axes[2:4]
        rate_axes = example_ax.child_axes[4:]
        assert [schematic_ax.lines[0].get_color() for schematic_ax in schematic_axes] == [
            PANEL_TRAJECTORY_COLORS[example["trajectories"][1]],
            PANEL_TRAJECTORY_COLORS[example["trajectories"][0]],
        ]
        assert all(len(raster_ax.lines) == 6 for raster_ax in raster_axes)
        raster_position = raster_axes[0].get_position()
        schematic_centers = [
            schematic_ax.get_position().y0 + schematic_ax.get_position().height / 2.0
            for schematic_ax in schematic_axes
        ]
        assert schematic_centers == pytest.approx(
            [
                raster_position.y0 + raster_position.height * (4.5 / 7.0),
                raster_position.y0 + raster_position.height * (1.5 / 7.0),
            ]
        )
        assert all(
            schematic_ax.get_position().x1 < raster_axes[0].get_position().x0
            for schematic_ax in schematic_axes
        )
        assert raster_axes[0].yaxis.label.get_position() == pytest.approx((-0.32, 0.5))
        assert raster_axes[0].get_position().width > rate_axes[0].get_position().height
        assert raster_axes[0].get_position().width == pytest.approx(
            rate_axes[0].get_position().width
        )
        assert [line.get_color() for line in raster_axes[0].lines[:2]] == [
            PANEL_TRAJECTORY_COLORS[example["trajectories"][0]],
            PANEL_TRAJECTORY_COLORS[example["trajectories"][0]],
        ]
        assert raster_axes[0].lines[0].get_markersize() == pytest.approx(0.55)
        assert raster_axes[0].lines[0].get_markeredgewidth() == pytest.approx(0.21)
        assert [line.get_xdata()[0] for line in raster_axes[0].lines[-2:]] == pytest.approx(
            list(SEGMENT_BOUNDARIES)
        )
        assert all(len(rate_ax.lines) == 4 for rate_ax in rate_axes)
        assert all(rate_ax.get_legend() is None for rate_ax in rate_axes)
        assert all(not rate_ax.texts for rate_ax in rate_axes)
        assert [
            rate_axes[0].lines[index].get_xdata()[0]
            for index in (2, 3)
        ] == pytest.approx(list(SEGMENT_BOUNDARIES))
        assert [raster_ax.get_title() for raster_ax in raster_axes] == ["Dark", "Light"]
        assert [rate_ax.get_title() for rate_ax in rate_axes] == ["", ""]
        assert raster_axes[0].get_facecolor() == pytest.approx(
            to_rgba(PANEL_A_DARK_EPOCH_BACKGROUND)
        )
        assert rate_axes[0].get_facecolor() == pytest.approx(
            to_rgba(PANEL_A_DARK_EPOCH_BACKGROUND)
        )
        assert raster_axes[1].get_facecolor() == pytest.approx(to_rgba("white"))
        assert rate_axes[1].get_facecolor() == pytest.approx(to_rgba("white"))
        assert rate_axes[0].get_position().x0 < rate_axes[1].get_position().x0
        assert raster_axes[0].get_position().y0 > rate_axes[0].get_position().y0
        assert raster_axes[0].get_position().height > rate_axes[0].get_position().height
        assert all(
            rate_ax.get_xlabel() == old_fig3_module.TASK_PROGRESSION_XLABEL
            for rate_ax in rate_axes
        )
    plt.close(fig)


def test_plot_light_heatmap_regions_adds_segment_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compute_calls = []

    def fake_compute_light_epoch_tuning_curves(**kwargs: object):
        compute_calls.append(kwargs)
        return {}

    pooled_calls = []

    def fake_build_pooled_panel_values(
        _curve_sets,
        *,
        position_bin_count: int,
        trajectory_types: tuple[str, ...],
        firing_rate_normalization: str,
    ):
        pooled_calls.append(
            {
                "position_bin_count": position_bin_count,
                "trajectory_types": trajectory_types,
                "firing_rate_normalization": firing_rate_normalization,
            }
        )
        return {}

    plot_calls = []

    def fake_plot_pooled_heatmap_grid(_axes, _panels, **kwargs: object):
        plot_calls.append(kwargs)
        return None

    monkeypatch.setattr(
        old_fig3_module,
        "compute_light_epoch_tuning_curves",
        fake_compute_light_epoch_tuning_curves,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "build_pooled_panel_values",
        fake_build_pooled_panel_values,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "plot_pooled_heatmap_grid",
        fake_plot_pooled_heatmap_grid,
    )

    fig, axes = plt.subplots(nrows=4, ncols=4)
    plot_light_heatmap_regions(
        np.asarray(axes, dtype=object),
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )

    assert all(len(ax.lines) == 2 for ax in axes.ravel())
    assert [line.get_xdata()[0] for line in axes[0, 0].lines] == pytest.approx(
        list(SEGMENT_BOUNDARIES)
    )
    assert compute_calls[0]["use_trajectory_direction"] is True
    assert compute_calls[0]["min_movement_firing_rate_hz"] == pytest.approx(
        PANEL_B_MIN_MOVEMENT_FIRING_RATE_HZ
    )
    assert compute_calls[0]["min_tuning_stability_correlation"] == pytest.approx(
        PANEL_B_MIN_TUNING_STABILITY_CORRELATION
    )
    assert pooled_calls[0]["trajectory_types"] == PANEL_B_TRAJECTORY_TYPES
    assert (
        pooled_calls[0]["firing_rate_normalization"]
        == PANEL_B_FIRING_RATE_NORMALIZATION
    )
    assert plot_calls[0]["trajectory_types"] == PANEL_B_TRAJECTORY_TYPES
    assert plot_calls[0]["axis_orientation"] == PANEL_B_LINEAR_POSITION_ORIENTATION
    assert plot_calls[0]["cmap"] == PANEL_B_HEATMAP_CMAP
    plt.close(fig)


def test_plot_light_heatmap_regions_uses_matching_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = build_panel_b_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    panels = {
        (order_trajectory, plot_trajectory): np.ones((2, 3), dtype=float)
        for order_trajectory in PANEL_B_TRAJECTORY_TYPES
        for plot_trajectory in PANEL_B_TRAJECTORY_TYPES
    }
    save_panel_b_cache(build_panel_b_cache_path(tmp_path, metadata), panels, metadata)

    monkeypatch.setattr(
        old_fig3_module,
        "compute_light_epoch_tuning_curves",
        lambda **_kwargs: pytest.fail("Panel B cache was not used."),
    )
    observed = {}

    def _fake_plot_pooled_heatmap_grid(_axes, cached_panels, **kwargs):
        observed["panels"] = cached_panels
        observed["kwargs"] = kwargs
        return None

    monkeypatch.setattr(
        old_fig3_module,
        "plot_pooled_heatmap_grid",
        _fake_plot_pooled_heatmap_grid,
    )

    fig, axes = plt.subplots(nrows=4, ncols=4)
    plot_light_heatmap_regions(
        np.asarray(axes, dtype=object),
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_b_cache_dir=tmp_path,
    )

    assert observed["panels"] is not None
    assert observed["kwargs"]["trajectory_types"] == PANEL_B_TRAJECTORY_TYPES
    assert observed["kwargs"]["axis_orientation"] == PANEL_B_LINEAR_POSITION_ORIENTATION
    assert observed["kwargs"]["cmap"] == PANEL_B_HEATMAP_CMAP
    for key in panels:
        assert np.array_equal(observed["panels"][key], panels[key])
    plt.close(fig)


def test_plot_light_heatmap_regions_matches_figure_1d_path_ticks(
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = build_panel_b_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    panels = {
        (order_trajectory, plot_trajectory): np.ones((2, 3), dtype=float)
        for order_trajectory in PANEL_B_TRAJECTORY_TYPES
        for plot_trajectory in PANEL_B_TRAJECTORY_TYPES
    }
    save_panel_b_cache(build_panel_b_cache_path(tmp_path, metadata), panels, metadata)

    fig, axes = plt.subplots(nrows=4, ncols=4)
    plot_light_heatmap_regions(
        np.asarray(axes, dtype=object),
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_b_cache_dir=tmp_path,
    )

    for ax in axes[-1, :]:
        assert ax.get_xticks().tolist() == pytest.approx([0.0, 1.0])
        assert [label.get_text() for label in ax.get_xticklabels()] == ["0", "1"]
        assert ax.get_xlabel() == ""
    for ax in axes[:-1, :].ravel():
        assert ax.get_xticks().tolist() == []
        assert [label.get_text() for label in ax.get_xticklabels()] == []
        assert ax.get_xlabel() == ""
    plt.close(fig)


def test_add_panel_b_path_progression_label_matches_figure_1d_label() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=4, ncols=4)
    text = old_fig3_module.add_panel_b_path_progression_label(
        fig,
        np.asarray(axes, dtype=object),
    )
    bottom_boxes = [ax.get_position() for ax in axes[-1, :]]
    expected_x = (
        min(box.x0 for box in bottom_boxes)
        + max(box.x1 for box in bottom_boxes)
    ) / 2
    expected_y = (
        min(box.y0 for box in bottom_boxes)
        - old_fig3_module.HEATMAP_PATH_LABEL_OFFSET
    )

    assert text.get_text() == old_fig3_module.TASK_PROGRESSION_XLABEL
    assert text.get_position() == pytest.approx((expected_x, expected_y))
    assert text.get_fontsize() == pytest.approx(
        old_fig3_module.PANEL_E_AXIS_LABEL_FONTSIZE
    )
    plt.close(fig)


def test_panel_ab_header_text_uses_one_figure_y_coordinate() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    panel_a_axis = fig.add_axes([0.10, 0.20, 0.25, 0.55])
    corner_axis = fig.add_axes([0.42, 0.70, 0.10, 0.10])
    tuning_axes = [
        fig.add_axes([0.55, 0.72, 0.12, 0.10]),
        fig.add_axes([0.72, 0.72, 0.12, 0.10]),
    ]
    header_y = (
        old_fig3_module._axis_group_top_y(tuning_axes)
        + old_fig3_module.PANEL_AB_HEADER_Y_OFFSET
    )

    label_a = old_fig3_module._add_panel_label_at_figure_y(
        fig,
        panel_a_axis,
        "A",
        x=-0.07,
        y=header_y,
    )
    label_b = old_fig3_module._add_panel_label_at_figure_y(
        fig,
        corner_axis,
        "B",
        x=-0.12,
        y=header_y,
    )
    title_a = fig.text(
        0.20,
        header_y,
        "Example DPP cells in dark and light",
        va="center",
    )
    tuning = old_fig3_module._add_centered_axis_group_text_at_y(
        fig,
        tuning_axes,
        "Tuning",
        y=header_y,
        fontsize=8.0,
    )

    assert [
        label_a.get_position()[1],
        title_a.get_position()[1],
        label_b.get_position()[1],
        tuning.get_position()[1],
    ] == pytest.approx([header_y] * 4)
    assert {
        text.get_verticalalignment()
        for text in (label_a, title_a, label_b, tuning)
    } == {"center"}
    assert tuning.get_position()[0] == pytest.approx(
        old_fig3_module._axis_group_center_x(tuning_axes)
    )
    plt.close(fig)


def test_panel_cd_label_and_group_title_share_vertical_position() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    old_fig3_module._add_panel_cd_group_title(ax, "Vision changes DPP tuning")
    old_fig3_module._add_panel_cd_label(ax, "C")

    title_text = next(
        text for text in ax.texts if text.get_text().startswith("Vision")
    )
    label_text = next(text for text in ax.texts if text.get_text() == "C")
    assert title_text.get_position()[1] == pytest.approx(
        old_fig3_module.PANEL_CD_GROUP_TITLE_Y
    )
    assert label_text.get_position()[1] == pytest.approx(
        old_fig3_module.PANEL_CD_GROUP_TITLE_Y
    )
    assert title_text.get_verticalalignment() == "top"
    assert label_text.get_verticalalignment() == "top"
    plt.close(fig)


def test_plot_quantification_panels_use_light_and_dark_artifacts() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    similarity_table = pandas.DataFrame(
        {
            "animal_name": ["L12"] * 8,
            "date": ["20240421"] * 8,
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "comparison_label": [
                "left_turn",
                "right_turn",
                "left_turn",
                "right_turn",
                "left_turn",
                "right_turn",
                "left_turn",
                "right_turn",
            ],
            "epoch_type": [
                "light",
                "light",
                "dark",
                "dark",
                "light",
                "light",
                "dark",
                "dark",
            ],
            "similarity": [0.2, 0.6, -0.1, 0.3, 0.1, 0.05, 0.2, 0.4],
        }
    )
    delta_table = pandas.DataFrame(
        {
            "animal_name": ["L12", "L12", "L12", "L12"],
            "date": ["20240421", "20240421", "20240421", "20240421"],
            "unit": [1, 2, 1, 2],
            "epoch_type": ["light", "light", "dark", "dark"],
            "delta_bits_tp_vs_place": [0.1, 0.2, -0.2, 0.05],
        }
    )
    decoding_rows = []
    for epoch_index, epoch_type in enumerate(("light", "dark"), start=0):
        median_error = 0.015 + 0.008 * epoch_index
        decoding_rows.append(
            {
                "animal_name": "pooled",
                "date": "pooled",
                "epoch_type": epoch_type,
                "epoch": "02_r1" if epoch_type == "light" else "08_r4",
                "analysis": "place",
                "comparison": "place",
                "comparison_label": "Place",
                "q25_error": median_error - 0.004,
                "median_error": median_error,
                "q75_error": median_error + 0.006,
                "n_samples": 30,
            }
        )
        for comparison, label, _family, _pairs in PANEL_E_CROSS_COMPARISONS:
            decoding_rows.append(
                {
                    "animal_name": "pooled",
                    "date": "pooled",
                    "epoch_type": epoch_type,
                    "epoch": "02_r1" if epoch_type == "light" else "08_r4",
                    "analysis": "cross_trajectory",
                    "comparison": comparison,
                    "comparison_label": label,
                    "q25_error": 0.18 + 0.03 * epoch_index,
                    "median_error": 0.22 + 0.03 * epoch_index,
                    "q75_error": 0.27 + 0.03 * epoch_index,
                    "n_samples": 60,
                }
            )
    decoding_table = pandas.DataFrame(decoding_rows)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    plot_panel_c_similarity(axes[0], similarity_table)
    plot_panel_d_encoding_delta_histogram(axes[1], delta_table)
    plot_panel_e_decoding_error(axes[2], decoding_table)

    paired = build_panel_c_similarity_pairs(similarity_table)
    assert paired["comparison_label"].astype(str).tolist() == [
        "right_turn",
        "right_turn",
    ]
    assert paired[["similarity_light", "similarity_dark"]].to_numpy().ravel() == pytest.approx(
        [0.6, 0.3, 0.05, 0.4]
    )
    assert axes[0].get_xlabel() == "Light tuning corr."
    assert axes[0].get_ylabel() == "Dark tuning corr."
    assert axes[0].get_aspect() == pytest.approx(1.0)
    assert axes[0].lines[0].get_linestyle() == "--"
    assert axes[0].lines[0].get_xdata().tolist() == [-1.0, 1.0]
    assert axes[0].lines[0].get_ydata().tolist() == [-1.0, 1.0]
    assert [text.get_text() for text in axes[0].texts] == ["n=2"]
    assert axes[0].texts[0].get_position() == pytest.approx((0.96, 0.04))
    assert axes[0].texts[0].get_horizontalalignment() == "right"
    assert axes[0].texts[0].get_verticalalignment() == "bottom"
    assert len(axes[0].collections) == 1
    assert axes[0].collections[0].get_alpha() == pytest.approx(PANEL_C_SCATTER_ALPHA)
    assert axes[0].collections[0].get_sizes().tolist() == pytest.approx(
        [PANEL_C_SCATTER_SIZE]
    )
    assert axes[1].get_xlabel() == "Δ log likelihood (bits/spike)"
    assert axes[1].get_ylabel() == "Frac."
    assert "Route-specific\nplace better" in [
        text.get_text() for text in axes[1].texts
    ]
    assert "DPP better" in [text.get_text() for text in axes[1].texts]
    dpp_text = next(text for text in axes[1].texts if text.get_text() == "DPP better")
    light_summary_text = next(
        text for text in axes[1].texts if text.get_text() == "Light: 100% >0\nmed. 0.15"
    )
    dark_summary_text = next(
        text for text in axes[1].texts if text.get_text() == "Dark: 50% >0\nmed. -0.08"
    )
    assert dpp_text.get_horizontalalignment() == "left"
    assert light_summary_text.get_horizontalalignment() == "left"
    assert dark_summary_text.get_horizontalalignment() == "left"
    assert light_summary_text.get_fontsize() == pytest.approx(
        PANEL_QUANT_SUMMARY_TEXT_FONTSIZE
    )
    assert dark_summary_text.get_fontsize() == pytest.approx(
        PANEL_QUANT_SUMMARY_TEXT_FONTSIZE
    )
    assert light_summary_text.get_position()[0] == pytest.approx(
        dpp_text.get_position()[0]
    )
    assert dark_summary_text.get_position()[0] == pytest.approx(
        dpp_text.get_position()[0]
    )
    assert light_summary_text.get_color() == PANEL_QUANT_EPOCH_COLORS["light"]
    assert dark_summary_text.get_color() == PANEL_QUANT_EPOCH_COLORS["dark"]
    light_count_text = next(
        text
        for text in axes[1].texts
        if text.get_text() == "Light: n = 2 cells"
    )
    dark_count_text = next(
        text
        for text in axes[1].texts
        if text.get_text() == "Dark: n = 2 cells"
    )
    animal_count_text = next(
        text for text in axes[1].texts if text.get_text() == "1 animal"
    )
    assert light_count_text.get_position() == pytest.approx((0.03, 0.40))
    assert dark_count_text.get_position() == pytest.approx((0.03, 0.24))
    assert animal_count_text.get_position() == pytest.approx((0.03, 0.08))
    assert light_count_text.get_horizontalalignment() == "left"
    assert dark_count_text.get_horizontalalignment() == "left"
    assert animal_count_text.get_horizontalalignment() == "left"
    assert light_count_text.get_verticalalignment() == "bottom"
    assert dark_count_text.get_verticalalignment() == "bottom"
    assert animal_count_text.get_verticalalignment() == "bottom"
    assert light_count_text.get_fontsize() == pytest.approx(
        PANEL_QUANT_SUMMARY_TEXT_FONTSIZE
    )
    assert dark_count_text.get_fontsize() == pytest.approx(
        PANEL_QUANT_SUMMARY_TEXT_FONTSIZE
    )
    assert animal_count_text.get_fontsize() == pytest.approx(
        PANEL_QUANT_SUMMARY_TEXT_FONTSIZE
    )
    assert light_count_text.get_color() == PANEL_QUANT_EPOCH_COLORS["light"]
    assert dark_count_text.get_color() == PANEL_QUANT_EPOCH_COLORS["dark"]
    assert animal_count_text.get_color() == "0.25"
    assert axes[1].get_legend() is None
    assert axes[1].get_xlim() == pytest.approx(PANEL_D_ENCODING_X_LIMITS)
    assert len(axes[1].lines) == 1
    assert axes[1].lines[0].get_color() == "black"
    assert axes[1].lines[0].get_linestyle() == "--"
    assert axes[1].patches[0].get_edgecolor()[3] == pytest.approx(0.0)
    assert len(axes[1].patches) == 52
    assert len(axes[2].child_axes) == 2
    cross_ax, place_ax = axes[2].child_axes
    assert [child_ax.get_title() for child_ax in axes[2].child_axes] == [
        "Cross-route\ndecoding",
        "Route-specific\nplace decoding",
    ]
    assert cross_ax.get_ylabel() == "Abs. norm. error"
    assert cross_ax.yaxis.label.get_size() == pytest.approx(5.8)
    assert place_ax.get_ylabel() == ""
    assert place_ax.get_ylim() == pytest.approx(PANEL_E_PLACE_ERROR_YLIM)
    assert cross_ax.get_ylim() == pytest.approx(PANEL_E_NORM_ERROR_YLIM)
    assert [text.get_text() for text in place_ax.get_xticklabels()] == ["Light", "Dark"]
    assert [text.get_text() for text in cross_ax.get_xticklabels()] == ["Light", "Dark"]
    assert [text.get_text() for text in cross_ax.texts] == [
        "Light med. 0.22",
        "Dark med. 0.25",
    ]
    assert [text.get_position()[0] for text in cross_ax.texts] == pytest.approx(
        [old_fig3_module.PANEL_E_SUMMARY_TEXT_X] * 2
    )
    assert all(text.get_horizontalalignment() == "right" for text in cross_ax.texts)
    assert [text.get_text() for text in place_ax.texts] == [
        "Light med. 0.01",
        "Dark med. 0.02",
    ]
    assert [text.get_position()[0] for text in place_ax.texts] == pytest.approx(
        [old_fig3_module.PANEL_E_PLACE_SUMMARY_TEXT_X] * 2
    )
    assert all(text.get_horizontalalignment() == "left" for text in place_ax.texts)
    assert place_ax.get_legend() is None
    assert cross_ax.get_legend() is None
    assert len(cross_ax.get_xticklabels()) == 2
    assert len(place_ax.collections) == 4
    assert len(cross_ax.collections) >= 4
    plt.close(fig)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_panel_c_vision_tuning_panel(axes[0], similarity_table, decoding_table)
    plot_panel_d_route_place_panel(axes[1], delta_table, decoding_table)

    assert [text.get_text() for text in axes[0].texts] == [
        "Vision changes DPP tuning"
    ]
    assert [text.get_text() for text in axes[1].texts] == [
        "Shift toward route-specific place coding"
    ]
    assert len(axes[0].child_axes) == 2
    assert len(axes[1].child_axes) == 2
    assert [child_ax.get_title() for child_ax in axes[0].child_axes] == [
        "Same-turn route\ntuning similarity",
        "Cross-route\ndecoding",
    ]
    assert [child_ax.get_title() for child_ax in axes[1].child_axes] == [
        "Encoding comparison",
        "Route-specific\nplace decoding",
    ]
    assert axes[0].child_axes[1].get_ylabel() == "Abs. norm. error"
    assert axes[1].child_axes[1].get_ylabel() == "Abs. norm. error"
    plt.close(fig)


def test_load_panel_e_decoding_error_table_pools_registered_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    del pandas

    calls = []

    def _fake_error_values(
        true_path: Path,
        _decoded_path: Path,
        *,
        normalization: float,
    ) -> np.ndarray:
        assert normalization > 0.0
        path_text = str(true_path)
        calls.append(path_text)
        if true_path.name.endswith("true_place.npz"):
            assert normalization == pytest.approx(100.0)
            return np.asarray([0.01, 0.03])
        assert normalization == pytest.approx(1.0)
        return np.asarray([0.1, 0.2])

    monkeypatch.setattr(
        old_fig3_module,
        "_load_absolute_normalized_decoding_errors",
        _fake_error_values,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "get_wtrack_total_length",
        lambda _animal_name: 100.0,
    )

    table = load_panel_e_decoding_error_table(
        data_root=Path("/analysis"),
        datasets=[
            ("L12", "20240421", "08_r4"),
            ("L14", "20240611", "08_r4"),
            ("L15", "20241121", "10_r5"),
            ("L19", "20250930", "08_r4"),
        ],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
    )

    assert set(table["animal_name"]) == {"pooled"}
    assert set(table["date"]) == {"pooled"}
    assert len(table[table["analysis"] == "place"]) == 2
    assert len(table[table["analysis"] == "cross_trajectory"]) == (
        2 * len(PANEL_E_CROSS_COMPARISONS)
    )
    light_place = table[
        (table["analysis"] == "place") & (table["epoch_type"] == "light")
    ].iloc[0]
    light_cross = table[
        (table["analysis"] == "cross_trajectory")
        & (table["epoch_type"] == "light")
    ].iloc[0]
    assert light_place["n_samples"] == 4 * 2
    assert light_cross["n_samples"] == (
        4 * len(PANEL_E_CROSS_COMPARISONS[0][3]) * 2
    )
    assert any("L19" in call for call in calls)


def test_dark_glm_schematic_tracks_can_reserve_white_stimulus_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    swap_delta_table = pandas.DataFrame(
        {
            "delta_ll_bits_per_spike": [0.0] * len(PANEL_H_DELTA_TRAJECTORIES),
            "light_train_epoch": ["02_r1"] * len(PANEL_H_DELTA_TRAJECTORIES),
            "light_test_epoch": ["06_r3"] * len(PANEL_H_DELTA_TRAJECTORIES),
            "trajectory": list(PANEL_H_DELTA_TRAJECTORIES),
        }
    )

    fig, axis = plt.subplots()
    plot_panel_h_swap_delta(
        axis,
        swap_delta_table,
        [],
        show_dark_track_labels=True,
    )

    panel_h_schematic_ax = axis.child_axes[0]
    dark_track_axes = [panel_h_schematic_ax.child_axes[3]]
    for track_ax in dark_track_axes:
        visible_labels = [text for text in track_ax.texts if text.get_text() in {"A", "B"}]
        assert sorted(text.get_text() for text in visible_labels) == ["A", "B"]
        assert all(text.get_color() == "white" for text in visible_labels)
        assert "C" not in [text.get_text() for text in track_ax.texts]
    plt.close(fig)


def test_plot_panel_h_swap_delta_can_use_segment_scalar_model() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {"visual": "#D73027", "task_segment_scalar": "#1A9850"}
    swap_delta_table = pandas.DataFrame(
        {
            "delta_ll_bits_per_spike": [0.2] * len(PANEL_H_DELTA_TRAJECTORIES),
            "light_train_epoch": ["02_r1"] * len(PANEL_H_DELTA_TRAJECTORIES),
            "light_test_epoch": ["06_r3"] * len(PANEL_H_DELTA_TRAJECTORIES),
            "trajectory": list(PANEL_H_DELTA_TRAJECTORIES),
            "model_name": ["task_segment_scalar"] * len(PANEL_H_DELTA_TRAJECTORIES),
        }
    )
    swap_example = {
        "animal_name": "L15",
        "region": "v1",
        "unit_id": 40,
        "trajectory": "center_to_left",
        "model_name": "task_segment_scalar",
        "delta_ll_bits_per_spike": 0.25,
        "segment_start": 0.4,
        "segment_end": 0.6,
        "tp_grid": np.asarray([0.4, 0.5, 0.6]),
        "observed_position": np.asarray([0.4, 0.5, 0.6]),
        "observed_rate_hz": np.asarray([0.2, 1.5, 0.4]),
        "models": {
            "visual": np.asarray([0.1, 0.8, 0.2]),
            "task_segment_scalar": np.asarray([0.2, 1.2, 0.3]),
        },
    }
    second_swap_example = dict(swap_example)
    second_swap_example["unit_id"] = 41
    second_swap_example["observed_rate_hz"] = np.asarray([0.3, 1.1, 0.4])

    fig, axis = plt.subplots()
    plot_panel_h_swap_delta(
        axis,
        swap_delta_table,
        [swap_example, second_swap_example],
        model_name="task_segment_scalar",
        model_colors=model_colors,
    )

    schematic_ax, delta_ax, _first_example_ax, second_example_ax = axis.child_axes
    assert any(
        text.get_text() == "Segment scalar\nmodel" for text in schematic_ax.texts
    )
    hist_axes = delta_ax.child_axes[1::2]
    assert all(
        "Segment scalar\nbetter" in [text.get_text() for text in hist_ax.texts]
        for hist_ax in hist_axes
    )
    first_hist_text_colors = {
        text.get_text(): text.get_color() for text in hist_axes[0].texts
    }
    assert first_hist_text_colors["Independent\nbetter"] == model_colors["visual"]
    assert first_hist_text_colors["Segment scalar\nbetter"] == model_colors[
        "task_segment_scalar"
    ]
    assert [line.get_color() for line in second_example_ax.lines[1:]] == [
        model_colors["visual"],
        model_colors["task_segment_scalar"],
    ]
    assert second_example_ax.get_legend() is not None
    assert [
        text.get_text() for text in second_example_ax.get_legend().get_texts()
    ] == [
        "Empirical",
        "Independent",
        "Segment scalar",
    ]
    plt.close(fig)


def test_plot_panel_h_swap_delta_uses_schematic_histograms_and_examples() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Ellipse, Polygon

    swap_delta_table = pandas.DataFrame(
        {
            "delta_ll_bits_per_spike": [-0.1, 0.0, 0.2, 0.3],
            "light_train_epoch": ["02_r1", "02_r1", "02_r1", "02_r1"],
            "light_test_epoch": ["06_r3", "06_r3", "06_r3", "06_r3"],
            "trajectory": list(PANEL_H_DELTA_TRAJECTORIES),
        }
    )
    swap_example = {
        "animal_name": "L15",
        "region": "v1",
        "unit_id": 40,
        "trajectory": "center_to_left",
        "delta_ll_bits_per_spike": 0.25,
        "segment_start": 0.4,
        "segment_end": 0.6,
        "tp_grid": np.asarray([0.4, 0.5, 0.6]),
        "observed_position": np.asarray([0.4, 0.5, 0.6]),
        "observed_rate_hz": np.asarray([0.2, 1.5, 0.4]),
        "models": {
            "visual": np.asarray([0.1, 0.8, 0.2]),
            "task_segment_bump": np.asarray([0.2, 1.4, 0.3]),
        },
    }
    second_swap_example = dict(swap_example)
    second_swap_example["unit_id"] = 41
    second_swap_example["delta_ll_bits_per_spike"] = -0.1
    second_swap_example["observed_rate_hz"] = np.asarray([0.3, 1.2, 0.5])
    second_swap_example["models"] = {
        "visual": np.asarray([0.2, 0.6, 0.3]),
        "task_segment_bump": np.asarray([0.3, 1.1, 0.4]),
    }
    fig, axis = plt.subplots()
    plot_panel_h_swap_delta(axis, swap_delta_table, [swap_example, second_swap_example])
    fig.canvas.draw()

    def _axis_center_y(axis):
        position = axis.get_position()
        return position.y0 + position.height / 2.0

    assert len(axis.child_axes) == 4
    schematic_h_ax, delta_h_ax, first_example_h_ax, second_example_h_ax = axis.child_axes
    parent_h_position = axis.get_position()
    schematic_h_position = schematic_h_ax.get_position()
    delta_h_position = delta_h_ax.get_position()
    assert schematic_h_position.width / parent_h_position.width == pytest.approx(
        PANEL_H_SCHEMATIC_AXIS_BOUNDS[2]
    )
    assert schematic_h_position.height / parent_h_position.height == pytest.approx(
        PANEL_H_SCHEMATIC_AXIS_BOUNDS[3]
    )
    assert delta_h_position.x0 == pytest.approx(
        parent_h_position.x0 + parent_h_position.width * PANEL_H_DELTA_AXIS_BOUNDS[0]
    )
    assert schematic_h_ax.get_position().x1 < delta_h_ax.get_position().x0
    assert first_example_h_ax.get_position().y1 < schematic_h_ax.get_position().y0
    assert second_example_h_ax.get_position().y1 < delta_h_ax.get_position().y0
    assert first_example_h_ax.get_position().y0 == pytest.approx(
        second_example_h_ax.get_position().y0
    )
    assert first_example_h_ax.get_position().x1 < second_example_h_ax.get_position().x0
    assert schematic_h_ax.texts[0].get_text() == "Train: AB"
    assert schematic_h_ax.texts[0].get_fontsize() == pytest.approx(5.8)
    assert schematic_h_ax.texts[1].get_fontsize() == pytest.approx(5.8)
    assert schematic_h_ax.texts[2].get_fontsize() == pytest.approx(4.1)
    assert schematic_h_ax.texts[3].get_fontsize() == pytest.approx(3.8)
    assert schematic_h_ax.texts[2].get_position()[0] == pytest.approx(0.045)
    assert schematic_h_ax.texts[3].get_position()[0] == pytest.approx(0.045)
    train_predict_midpoint_x = 0.5 * (
        schematic_h_ax.texts[0].get_position()[0]
        + schematic_h_ax.texts[1].get_position()[0]
    )
    independent_prediction_text = next(
        text
        for text in schematic_h_ax.texts
        if text.get_text().startswith('"Light activity is like the other arm')
    )
    shared_prediction_text = next(
        text
        for text in schematic_h_ax.texts
        if text.get_text().startswith('"Light activity is like the same arm')
    )
    assert independent_prediction_text.get_position()[0] == pytest.approx(
        train_predict_midpoint_x
    )
    assert independent_prediction_text.get_position()[1] == pytest.approx(0.61)
    assert shared_prediction_text.get_position()[0] == pytest.approx(
        train_predict_midpoint_x
    )
    assert shared_prediction_text.get_position()[1] == pytest.approx(0.02)
    panel_h_track_patches = [
        patch
        for track_ax in schematic_h_ax.child_axes
        for patch in track_ax.patches
        if isinstance(patch, Polygon)
    ]
    assert len(panel_h_track_patches) == 5
    assert [
        patch.get_linewidth()
        for patch in panel_h_track_patches
    ] == pytest.approx([PANEL_H_SCHEMATIC_TRACK_LINEWIDTH] * 5)
    assert all(
        any(line.get_color() == GLM_TRAJECTORY_ARROW_COLOR for line in track_ax.lines)
        for track_ax in schematic_h_ax.child_axes
    )
    panel_h_independent_train_ax = schematic_h_ax.child_axes[0]
    panel_h_segment_modulation_ax = schematic_h_ax.child_axes[2]
    panel_h_dark_ax = schematic_h_ax.child_axes[3]
    panel_h_shared_light_ax = schematic_h_ax.child_axes[4]
    assert all(
        text.get_text() != "C"
        for track_ax in schematic_h_ax.child_axes
        for text in track_ax.texts
    )
    panel_h_dark_basis_patches = [
        patch
        for patch in panel_h_dark_ax.patches
        if isinstance(patch, Circle)
    ]
    assert len(panel_h_dark_basis_patches) > 0
    assert all(
        patch.get_facecolor() == pytest.approx(to_rgba(GLM_BASIS_DARK_COLOR, 0.7))
        for patch in panel_h_dark_basis_patches
    )
    panel_h_shared_light_basis_patches = [
        patch
        for patch in panel_h_shared_light_ax.patches
        if isinstance(patch, Circle)
    ]
    assert len(panel_h_shared_light_basis_patches) > 0
    shared_light_filled_basis_patches = [
        patch
        for patch in panel_h_shared_light_basis_patches
        if patch.get_facecolor() == pytest.approx(
            to_rgba(GLM_BASIS_DARK_COLOR, 0.7)
        )
    ]
    shared_light_unfilled_basis_patches = [
        patch
        for patch in panel_h_shared_light_basis_patches
        if patch.get_facecolor() == pytest.approx(to_rgba("none"))
    ]
    assert len(shared_light_filled_basis_patches) > 0
    assert len(shared_light_unfilled_basis_patches) > 0
    assert all(patch.center[0] > 3.0 for patch in shared_light_filled_basis_patches)
    assert all(
        patch.get_edgecolor() == pytest.approx(to_rgba("black"))
        for patch in [*panel_h_dark_basis_patches, *panel_h_shared_light_basis_patches]
    )
    assert _axis_center_y(panel_h_segment_modulation_ax) < (
        _axis_center_y(panel_h_independent_train_ax)
        - schematic_h_position.height * 0.36
    )
    assert _axis_center_y(panel_h_dark_ax) < _axis_center_y(panel_h_segment_modulation_ax)
    assert _axis_center_y(panel_h_shared_light_ax) < _axis_center_y(
        panel_h_segment_modulation_ax
    )
    assert delta_h_ax.get_title() == ""
    assert len(delta_h_ax.child_axes) == 8
    icon_axes = delta_h_ax.child_axes[::2]
    hist_axes = delta_h_ax.child_axes[1::2]
    assert all(icon_ax.get_title() == "" for icon_ax in icon_axes)
    assert all(hist_ax.get_title() == "" for hist_ax in hist_axes)
    assert all(len(icon_ax.lines) > 0 for icon_ax in icon_axes)
    panel_h_delta_xlabel = next(
        text
        for text in delta_h_ax.texts
        if text.get_text() == "Δ log likelihood (bits/spike)"
    )
    panel_h_delta_ylabel = next(
        text for text in delta_h_ax.texts if text.get_text() == "Frac."
    )
    assert panel_h_delta_xlabel.get_position() == pytest.approx((0.53, -0.055))
    assert panel_h_delta_ylabel.get_position() == pytest.approx((-0.055, 0.49))
    for child_delta_ax in hist_axes:
        child_texts = child_delta_ax.texts
        child_text_labels = [text.get_text() for text in child_texts]
        assert child_delta_ax.lines[0].get_linestyle() == "--"
        assert child_delta_ax.lines[0].get_color() == "black"
        assert len(child_delta_ax.lines) == 1
        assert "Independent\nbetter" in child_text_labels
        assert "Shared scaffold\nbetter" in child_text_labels
        independent_better_text = next(
            text for text in child_texts if text.get_text() == "Independent\nbetter"
        )
        shared_better_text = next(
            text for text in child_texts if text.get_text() == "Shared scaffold\nbetter"
        )
        assert independent_better_text.get_horizontalalignment() == "left"
        assert shared_better_text.get_horizontalalignment() == "right"
        assert independent_better_text.get_fontsize() == pytest.approx(4.0)
        assert shared_better_text.get_fontsize() == pytest.approx(4.0)
        assert independent_better_text.get_color() == GLM_MODEL_COLORS["visual"]
        assert shared_better_text.get_color() == GLM_MODEL_COLORS[
            "task_segment_bump"
        ]
        summary_text = next(text for text in child_texts if "% >0\nmed." in text.get_text())
        assert summary_text.get_position() == pytest.approx((0.97, 0.56))
        assert summary_text.get_horizontalalignment() == "right"
        assert summary_text.get_fontsize() == pytest.approx(4.8)
        assert all("n=" not in text.get_text() for text in child_texts)
        assert all(">0=" not in text.get_text() for text in child_texts)
        assert len(child_delta_ax.patches) > 0
        assert child_delta_ax.patches[0].get_edgecolor()[3] == pytest.approx(0.0)
        assert child_delta_ax.patches[0].get_linewidth() == pytest.approx(0.0)
    assert first_example_h_ax.get_xlabel() == "Switched segment"
    assert second_example_h_ax.get_xlabel() == "Switched segment"
    assert first_example_h_ax.get_title() == "Example 1"
    assert second_example_h_ax.get_title() == "Example 2"
    assert first_example_h_ax.get_ylabel() == "FR (Hz)"
    assert second_example_h_ax.get_ylabel() == "FR (Hz)"
    first_delta_text = next(
        text for text in first_example_h_ax.texts if text.get_text() == "ΔLL=0.25"
    )
    second_delta_text = next(
        text for text in second_example_h_ax.texts if text.get_text() == "ΔLL=-0.10"
    )
    assert first_delta_text.get_position() == pytest.approx((0.96, 0.94))
    assert second_delta_text.get_position() == pytest.approx((0.96, 0.94))
    assert first_delta_text.get_horizontalalignment() == "right"
    assert second_delta_text.get_horizontalalignment() == "right"
    assert first_delta_text.get_verticalalignment() == "top"
    assert second_delta_text.get_verticalalignment() == "top"
    assert first_example_h_ax.get_legend() is None
    assert second_example_h_ax.get_legend() is not None
    assert [text.get_text() for text in second_example_h_ax.get_legend().get_texts()] == [
        "Empirical",
        "Independent",
        "Shared scaffold",
    ]
    assert second_example_h_ax.get_legend()._loc == 6
    assert len(first_example_h_ax.child_axes) == 1
    assert len(second_example_h_ax.child_axes) == 1
    assert first_example_h_ax.child_axes[0].get_position().x1 < first_example_h_ax.get_position().x0
    assert second_example_h_ax.child_axes[0].get_position().x1 < second_example_h_ax.get_position().x0
    assert first_example_h_ax.child_axes[0].get_position().y1 < (
        first_example_h_ax.get_position().y0
        + first_example_h_ax.get_position().height * 0.5
    )
    assert second_example_h_ax.child_axes[0].get_position().y1 < (
        second_example_h_ax.get_position().y0
        + second_example_h_ax.get_position().height * 0.5
    )
    assert all(
        len(icon_ax.lines) > 0
        for icon_ax in (
            first_example_h_ax.child_axes[0],
            second_example_h_ax.child_axes[0],
        )
    )
    assert len(first_example_h_ax.lines) == 3
    assert len(second_example_h_ax.lines) == 3
    plt.close(fig)


def test_setup_light_heatmap_panel_uses_figure_1_heatmap_geometry() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig = plt.figure()
    grid = fig.add_gridspec(1, 1)
    panel = setup_light_heatmap_panel(fig, grid[0, 0], regions=("v1",))

    assert panel["heatmap_axes"].shape == (4, 4)
    assert len(panel["tuning_schematic_axes"]) == 4
    assert len(panel["order_schematic_axes"]) == 4
    assert panel["corner_axis"].axison is False
    tuning_patches = [
        patch
        for schematic_ax in panel["tuning_schematic_axes"]
        for patch in schematic_ax.patches
        if isinstance(patch, Polygon)
    ]
    order_patches = [
        patch
        for order_ax in panel["order_schematic_axes"]
        for schematic_ax in order_ax.child_axes
        for patch in schematic_ax.patches
        if isinstance(patch, Polygon)
    ]
    assert tuning_patches
    assert order_patches
    assert all(patch.get_facecolor()[3] == pytest.approx(0.0) for patch in tuning_patches)
    assert all(patch.get_facecolor()[3] == pytest.approx(0.0) for patch in order_patches)
    plt.close(fig)


def test_default_cli_matches_manuscript_figure_format() -> None:
    args = parse_arguments([])

    assert DEFAULT_REGIONS == ("v1",)
    assert args.region is None
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.panel_b_cache_dir is None
    assert args.refresh_panel_b_cache is False
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(165.0)
    assert DEFAULT_PANEL_AB_HEIGHT_MM == pytest.approx(
        old_fig3_module.DEFAULT_HEATMAP_HEIGHT_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        DEFAULT_PANEL_AB_HEIGHT_MM + DEFAULT_PANEL_DEF_HEIGHT_MM
    )
    assert DEFAULT_PANEL_DEF_HEIGHT_MM == pytest.approx(34.0)
    assert DEFAULT_PANEL_B_WIDTH_FRACTION == pytest.approx(0.7)
    assert DEFAULT_PANEL_A_WIDTH_FRACTION == pytest.approx(0.3)
    assert PANEL_B_HEATMAP_CMAP == "viridis"
