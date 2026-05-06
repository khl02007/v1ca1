from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_3 as figure_3_module
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_LIGHT_EPOCH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PANEL_A_HEIGHT_MM,
    DEFAULT_PANEL_B_WIDTH_FRACTION,
    DEFAULT_PANEL_C_WIDTH_FRACTION,
    DEFAULT_PANEL_DEF_HEIGHT_MM,
    DEFAULT_PANEL_GH_HEIGHT_MM,
    DEFAULT_REGIONS,
    PANEL_A_EPOCH_COLORS,
    PANEL_A_EXAMPLE,
    PANEL_A_LIGHT_EPOCHS,
    PANEL_A_TRAJECTORIES,
    PANEL_B_CACHE_VERSION,
    PANEL_C_DARK_EPOCH_BACKGROUND,
    PANEL_C_EXAMPLES,
    PANEL_C_TRAJECTORY_COLORS,
    PANEL_D_SCATTER_ALPHA,
    PANEL_D_SCATTER_SIZE,
    PANEL_E_X_LIMITS,
    PANEL_F_CROSS_COMPARISONS,
    PANEL_F_NORM_ERROR_YLIM,
    PANEL_F_PLACE_ERROR_YLIM,
    PANEL_G_ARROW_COLOR,
    PANEL_G_EXAMPLE_HEIGHT_FRACTION,
    PANEL_G_EXAMPLE_WIDTH_FRACTION,
    PANEL_G_SCHEMATIC_HEIGHT_FRACTION,
    PANEL_G_SCHEMATIC_WIDTH_FRACTION,
    PANEL_H_SCHEMATIC_TRACK_LINEWIDTH,
    PANEL_H_SWAP_DELTA_VARIABLE,
    PANEL_EXAMPLE_CACHE_VERSION,
    SEGMENT_BOUNDARIES,
    build_panel_b_cache_metadata,
    build_panel_b_cache_path,
    build_panel_example_cache_metadata,
    build_panel_example_cache_path,
    build_panel_d_similarity_pairs,
    build_panel_a_epoch_specs,
    build_output_path,
    get_decoding_summary_path,
    get_dark_light_glm_selected_path,
    get_encoding_summary_candidate_paths,
    get_dark_epoch,
    get_light_epoch,
    get_swap_glm_selected_comparison_path,
    get_tuning_similarity_path,
    load_panel_quantification_data,
    load_or_compute_panel_example_data,
    make_light_epoch_dataset_ids,
    parse_arguments,
    parse_dataset_id,
    plot_panel_d_similarity,
    plot_panel_e_encoding_delta_histogram,
    plot_panel_f_decoding_error,
    plot_panel_g_model_architecture,
    plot_panel_h_swap_delta,
    plot_panel_a_example,
    plot_epoch_path_rate_axis,
    plot_light_heatmap_regions,
    plot_panel_c_examples,
    save_panel_b_cache,
    save_panel_example_cache,
    load_panel_b_cache,
    load_panel_example_cache,
    setup_light_heatmap_panel,
    validate_panel_c_trajectories,
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

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "figure_3", "jpg")


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


def test_light_and_dark_epoch_helpers_use_registered_defaults() -> None:
    assert get_light_epoch("L14", "20240611") == DEFAULT_LIGHT_EPOCH
    assert get_dark_epoch("L15", "20241121") == "10_r5"
    assert get_light_epoch("L14", "20240611", "04_r2") == "04_r2"
    assert get_dark_epoch("L14", "20240611", "12_r6") == "12_r6"


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
        "_posbins100_offset5_speed4_sigma1p5_cachev1.npz"
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
    for index, order_trajectory in enumerate(figure_3_module.TRAJECTORY_TYPES):
        for plot_trajectory in figure_3_module.TRAJECTORY_TYPES:
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
        figure_3_module,
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


def test_panel_a_example_configuration_uses_requested_cell_and_epochs() -> None:
    assert PANEL_A_EXAMPLE == ("L14", "20240611", "v1", 229)
    assert PANEL_A_TRAJECTORIES == (
        "center_to_left",
        "center_to_right",
        "left_to_center",
        "right_to_center",
    )
    assert PANEL_A_LIGHT_EPOCHS == ("02_r1", "06_r3")
    assert build_panel_a_epoch_specs("L14", "20240611", dark_epoch=None) == (
        ("02_r1", "02_r1", "02_r1"),
        ("06_r3", "06_r3", "06_r3"),
        ("dark", "Dark", "08_r4"),
    )


def test_panel_c_example_configuration_uses_requested_trajectory_pairs() -> None:
    assert PANEL_C_EXAMPLES == (
        ("L14", "20240611", "v1", 34, ("center_to_left", "right_to_center")),
        ("L15", "20241121", "v1", 473, ("center_to_right", "left_to_center")),
    )


def test_validate_panel_c_trajectories_rejects_unknown_names() -> None:
    assert validate_panel_c_trajectories(["center_to_left", "right_to_center"]) == (
        "center_to_left",
        "right_to_center",
    )
    with pytest.raises(ValueError, match="Unknown panel C trajectory"):
        validate_panel_c_trajectories(["center_to_left", "bad"])


def _fake_panel_a_example() -> dict[str, object]:
    trajectories = (
        "center_to_left",
        "center_to_right",
        "left_to_center",
        "right_to_center",
    )
    position = np.asarray([0.0, 0.5, 1.0], dtype=float)
    epoch_order = ("02_r1", "06_r3", "dark")
    return {
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "unit_id": 229,
        "trajectories": trajectories,
        "epoch_order": epoch_order,
        "epoch_labels": {"02_r1": "02_r1", "06_r3": "06_r3", "dark": "Dark"},
        "epoch_examples": {
            epoch: {
                "raster_positions": {
                    trajectory: [np.asarray([0.1, 0.4]), np.asarray([0.7])]
                    for trajectory in trajectories
                },
                "firing_rates": {
                    trajectory: (
                        position,
                        np.asarray([0.0, 1.0 + index, 0.5], dtype=float),
                    )
                    for index, trajectory in enumerate(trajectories)
                },
            }
            for epoch in epoch_order
        },
    }


def _fake_panel_b_example(trajectories: tuple[str, ...]) -> dict[str, object]:
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


def _fake_panel_g_examples() -> list[dict[str, object]]:
    grid = np.linspace(0.0, 1.0, 5)
    return [
        {
            "animal_name": "L15",
            "date": "20241121",
            "region": "v1",
            "light_epoch": "02_r1",
            "dark_epoch": "10_r5",
            "trajectory": "center_to_left",
            "unit_id": 40,
            "score": 0.3,
            "tp_grid": grid,
            "segment_edges": np.asarray([0.0, 0.4, 0.6, 1.0], dtype=float),
            "empirical": {
                "dark": (grid, np.asarray([0.0, 1.1, 2.2, 0.8, 0.0])),
                "light": (grid, np.asarray([0.0, 0.4, 1.1, 1.9, 0.4])),
            },
            "models": {
                "visual": {
                    "dark_hz": np.asarray([0.0, 1.0, 2.0, 1.0, 0.0]),
                    "light_hz": np.asarray([0.0, 0.5, 1.0, 2.0, 0.5]),
                    "score": 0.4,
                },
                "task_segment_bump": {
                    "dark_hz": np.asarray([0.0, 1.0, 2.0, 1.0, 0.0]),
                    "light_hz": np.asarray([0.0, 1.0, 2.4, 1.4, 0.0]),
                    "score": 0.3,
                },
            },
        },
        {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "light_epoch": "02_r1",
            "dark_epoch": "08_r4",
            "trajectory": "right_to_center",
            "unit_id": 22,
            "score": 0.2,
            "tp_grid": grid,
            "segment_edges": np.asarray([0.0, 0.4, 0.6, 1.0], dtype=float),
            "empirical": {
                "dark": (grid, np.asarray([0.0, 0.7, 1.2, 1.4, 0.3])),
                "light": (grid, np.asarray([0.0, 0.9, 1.5, 1.2, 0.1])),
            },
            "models": {
                "visual": {
                    "dark_hz": np.asarray([0.0, 0.5, 1.0, 1.5, 0.5]),
                    "light_hz": np.asarray([0.0, 1.0, 1.7, 1.0, 0.2]),
                    "score": 0.25,
                },
                "task_segment_bump": {
                    "dark_hz": np.asarray([0.0, 0.5, 1.0, 1.5, 0.5]),
                    "light_hz": np.asarray([0.0, 0.6, 1.3, 1.8, 0.5]),
                    "score": 0.2,
                },
            },
        },
    ]


def test_plot_epoch_path_rate_axis_overlays_path_type_curves() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = _fake_panel_b_example(("center_to_left", "right_to_center"))
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
        PANEL_C_TRAJECTORY_COLORS["center_to_left"],
        PANEL_C_TRAJECTORY_COLORS["right_to_center"],
    ]
    assert [line.get_xdata()[0] for line in ax.lines[2:]] == pytest.approx(
        list(SEGMENT_BOUNDARIES)
    )
    assert ax.get_ylabel() == "FR (Hz)"
    assert ax.get_xlabel() == "Norm. path progression"
    assert ax.get_title() == "Dark"
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_panel_a_example_draws_epoch_rasters_and_bottom_rate_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    plot_panel_a_example(ax, _fake_panel_a_example())

    assert len(ax.child_axes) == 15
    condition_axes = ax.child_axes[:3]
    schematic_axes = ax.child_axes[3::3]
    raster_axes = ax.child_axes[4::3]
    first_trajectory_raster_ax = raster_axes[0]
    rate_axes = ax.child_axes[5::3]
    condition_text = [
        [text.get_text() for text in condition_ax.texts]
        for condition_ax in condition_axes
    ]
    assert condition_text == [["A", "B"], ["B", "A"], []]
    condition_patches = [
        patch
        for condition_ax in condition_axes
        for patch in condition_ax.patches
        if isinstance(patch, Polygon)
    ]
    assert condition_patches[0].get_facecolor()[3] == pytest.approx(0.0)
    assert condition_patches[1].get_facecolor()[3] == pytest.approx(0.0)
    assert condition_patches[2].get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
    track_x = condition_patches[0].get_xy()[:, 0]
    track_y = condition_patches[0].get_xy()[:, 1]
    assert condition_axes[0].texts[0].get_position()[0] < np.min(track_x)
    assert condition_axes[0].texts[1].get_position()[0] > np.max(track_x)
    assert condition_axes[0].texts[0].get_position()[1] == pytest.approx(
        0.5 * (np.min(track_y) + np.max(track_y))
    )
    assert len(ax.texts) == 0
    assert len(schematic_axes) == 4
    assert len(raster_axes) == 4
    assert len(rate_axes) == 4
    assert all(len(rate_ax.lines) == 5 for rate_ax in rate_axes)
    assert [line.get_color() for line in rate_axes[-1].lines[:3]] == [
        PANEL_A_EPOCH_COLORS["02_r1"],
        PANEL_A_EPOCH_COLORS["06_r3"],
        PANEL_A_EPOCH_COLORS["dark"],
    ]
    assert [line.get_xdata()[0] for line in rate_axes[-1].lines[3:]] == pytest.approx(
        [0.4, 0.6]
    )
    assert all(rate_ax.get_xlabel() == "Norm. path progression" for rate_ax in rate_axes)
    assert len(first_trajectory_raster_ax.lines) == 10
    assert [line.get_color() for line in first_trajectory_raster_ax.lines[:2]] == [
        PANEL_A_EPOCH_COLORS["02_r1"],
        PANEL_A_EPOCH_COLORS["02_r1"],
    ]
    assert [line.get_color() for line in first_trajectory_raster_ax.lines[2:4]] == [
        PANEL_A_EPOCH_COLORS["06_r3"],
        PANEL_A_EPOCH_COLORS["06_r3"],
    ]
    assert [line.get_color() for line in first_trajectory_raster_ax.lines[4:6]] == [
        PANEL_A_EPOCH_COLORS["dark"],
        PANEL_A_EPOCH_COLORS["dark"],
    ]
    assert first_trajectory_raster_ax.lines[0].get_markersize() == pytest.approx(0.55)
    assert first_trajectory_raster_ax.lines[0].get_markeredgewidth() == pytest.approx(0.21)
    assert [
        first_trajectory_raster_ax.lines[index].get_xdata()[0]
        for index in (-2, -1)
    ] == pytest.approx([0.4, 0.6])
    assert first_trajectory_raster_ax.get_facecolor() == pytest.approx(to_rgba("white"))
    assert any(
        patch.get_facecolor() == pytest.approx(to_rgba(PANEL_C_DARK_EPOCH_BACKGROUND))
        for patch in first_trajectory_raster_ax.patches
    )
    assert all(rate_ax.get_legend() is None for rate_ax in rate_axes)
    plt.close(fig)


def test_plot_panel_c_examples_stacks_two_curve_blocks() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    examples = [
        _fake_panel_b_example(("center_to_left", "right_to_center")),
        _fake_panel_b_example(("center_to_right", "left_to_center")),
    ]
    plot_panel_c_examples(
        ax,
        examples,
    )

    assert len(ax.child_axes) == 2
    assert ax.child_axes[0].get_position().y1 == pytest.approx(ax.get_position().y1)
    for example_index, (example_ax, example) in enumerate(zip(ax.child_axes, examples), start=1):
        assert [text.get_text() for text in example_ax.texts] == [
            f"Example cell {example_index}"
        ]
        assert example_ax.texts[0].get_position()[0] == pytest.approx(0.50)
        assert example_ax.texts[0].get_position()[1] == pytest.approx(0.995)
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
        assert all(len(raster_ax.lines) == 6 for raster_ax in raster_axes)
        assert all(
            schematic_ax.get_position().x1 < raster_axes[0].get_position().x0
            for schematic_ax in schematic_axes
        )
        assert raster_axes[0].get_position().width > rate_axes[0].get_position().height
        assert raster_axes[0].get_position().width == pytest.approx(
            rate_axes[0].get_position().width
        )
        assert [line.get_color() for line in raster_axes[0].lines[:2]] == [
            PANEL_C_TRAJECTORY_COLORS[example["trajectories"][0]],
            PANEL_C_TRAJECTORY_COLORS[example["trajectories"][0]],
        ]
        assert raster_axes[0].lines[0].get_markersize() == pytest.approx(0.55)
        assert raster_axes[0].lines[0].get_markeredgewidth() == pytest.approx(0.21)
        assert [line.get_xdata()[0] for line in raster_axes[0].lines[-2:]] == pytest.approx(
            list(SEGMENT_BOUNDARIES)
        )
        assert all(len(rate_ax.lines) == 4 for rate_ax in rate_axes)
        assert all(rate_ax.get_legend() is None for rate_ax in rate_axes)
        assert [
            [text.get_text() for text in rate_ax.texts]
            for rate_ax in rate_axes
        ] == [["r=1.00"], ["r=1.00"]]
        assert [
            rate_axes[0].lines[index].get_xdata()[0]
            for index in (2, 3)
        ] == pytest.approx(list(SEGMENT_BOUNDARIES))
        assert [raster_ax.get_title() for raster_ax in raster_axes] == ["Dark", "Light"]
        assert [rate_ax.get_title() for rate_ax in rate_axes] == ["", ""]
        assert raster_axes[0].get_facecolor() == pytest.approx(
            to_rgba(PANEL_C_DARK_EPOCH_BACKGROUND)
        )
        assert rate_axes[0].get_facecolor() == pytest.approx(
            to_rgba(PANEL_C_DARK_EPOCH_BACKGROUND)
        )
        assert raster_axes[1].get_facecolor() == pytest.approx(to_rgba("white"))
        assert rate_axes[1].get_facecolor() == pytest.approx(to_rgba("white"))
        assert rate_axes[0].get_position().x0 < rate_axes[1].get_position().x0
        assert raster_axes[0].get_position().y0 > rate_axes[0].get_position().y0
        assert raster_axes[0].get_position().height > rate_axes[0].get_position().height
        assert all(rate_ax.get_xlabel() == "Norm. path progression" for rate_ax in rate_axes)
    plt.close(fig)


def test_plot_light_heatmap_regions_adds_segment_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        figure_3_module,
        "compute_light_epoch_tuning_curves",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        figure_3_module,
        "build_pooled_panel_values",
        lambda _curve_sets, *, position_bin_count: {},
    )
    monkeypatch.setattr(
        figure_3_module,
        "plot_pooled_heatmap_grid",
        lambda _axes, _panels: None,
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
        for order_trajectory in figure_3_module.TRAJECTORY_TYPES
        for plot_trajectory in figure_3_module.TRAJECTORY_TYPES
    }
    save_panel_b_cache(build_panel_b_cache_path(tmp_path, metadata), panels, metadata)

    monkeypatch.setattr(
        figure_3_module,
        "compute_light_epoch_tuning_curves",
        lambda **_kwargs: pytest.fail("Panel B cache was not used."),
    )
    observed = {}

    def _fake_plot_pooled_heatmap_grid(_axes, cached_panels):
        observed["panels"] = cached_panels
        return None

    monkeypatch.setattr(
        figure_3_module,
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
    for key in panels:
        assert np.array_equal(observed["panels"][key], panels[key])
    plt.close(fig)


def test_plot_quantification_panels_use_light_and_dark_artifacts() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            "epoch_type": ["light", "light", "dark", "dark"],
            "delta_bits_tp_vs_place": [0.1, 0.2, -0.2, 0.05],
        }
    )
    decoding_rows = []
    for animal_index, animal_name in enumerate(("L12", "L14"), start=1):
        for epoch_index, epoch_type in enumerate(("light", "dark"), start=1):
            median_error = 0.05 * animal_index + 0.02 * epoch_index
            decoding_rows.append(
                {
                    "animal_name": animal_name,
                    "date": "20240421",
                    "epoch_type": epoch_type,
                    "epoch": "02_r1" if epoch_type == "light" else "08_r4",
                    "analysis": "place",
                    "comparison": "place",
                    "comparison_label": "Place",
                    "q25_error": median_error - 0.01,
                    "median_error": median_error,
                    "q75_error": median_error + 0.02,
                    "n_samples": 10,
                }
            )
            for comparison, label, _family, _pairs in PANEL_F_CROSS_COMPARISONS:
                decoding_rows.append(
                    {
                        "animal_name": animal_name,
                        "date": "20240421",
                        "epoch_type": epoch_type,
                        "epoch": "02_r1" if epoch_type == "light" else "08_r4",
                        "analysis": "cross_trajectory",
                        "comparison": comparison,
                        "comparison_label": label,
                        "q25_error": median_error - 0.02,
                        "median_error": median_error + 0.03,
                        "q75_error": median_error + 0.04,
                        "n_samples": 20,
                    }
                )
    decoding_table = pandas.DataFrame(decoding_rows)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    plot_panel_d_similarity(axes[0], similarity_table)
    plot_panel_e_encoding_delta_histogram(axes[1], delta_table)
    plot_panel_f_decoding_error(axes[2], decoding_table)

    paired = build_panel_d_similarity_pairs(similarity_table)
    assert paired[["similarity_light", "similarity_dark"]].to_numpy().ravel() == pytest.approx(
        [0.6, 0.3, 0.1, 0.4]
    )
    assert axes[0].get_xlabel() == "Light same-turn\ntuning corr."
    assert axes[0].get_ylabel() == "Dark same-turn\ntuning corr."
    assert axes[0].get_aspect() == "auto"
    assert axes[0].lines[0].get_linestyle() == "--"
    assert axes[0].lines[0].get_xdata().tolist() == [-1.0, 1.0]
    assert axes[0].lines[0].get_ydata().tolist() == [-1.0, 1.0]
    assert [text.get_text() for text in axes[0].texts] == ["n=2"]
    assert len(axes[0].collections) == 1
    assert axes[0].collections[0].get_alpha() == pytest.approx(PANEL_D_SCATTER_ALPHA)
    assert axes[0].collections[0].get_sizes().tolist() == pytest.approx(
        [PANEL_D_SCATTER_SIZE]
    )
    assert axes[1].get_xlabel() == "Delta log likelihood (bits/spike)"
    assert "Trajectory-specific\nplace better" in [
        text.get_text() for text in axes[1].texts
    ]
    assert "DPP better" in [text.get_text() for text in axes[1].texts]
    assert any(text.get_text().startswith("Light: n=2") for text in axes[1].texts)
    assert axes[1].get_legend() is None
    assert axes[1].get_xlim() == pytest.approx(PANEL_E_X_LIMITS)
    assert len(axes[2].child_axes) == 2
    place_ax, cross_ax = axes[2].child_axes
    assert [child_ax.get_title() for child_ax in axes[2].child_axes] == [
        "Within-epoch place",
        "Same-turn cross-arm DPP",
    ]
    assert place_ax.get_ylabel() == "Abs. norm. error"
    assert cross_ax.get_ylabel() == ""
    assert place_ax.get_ylim() == pytest.approx(PANEL_F_PLACE_ERROR_YLIM)
    assert cross_ax.get_ylim() == pytest.approx(PANEL_F_NORM_ERROR_YLIM)
    assert [text.get_text() for text in place_ax.get_xticklabels()] == ["Light", "Dark"]
    assert [text.get_text() for text in cross_ax.get_xticklabels()] == [
        label for _comparison, label, _family, _pairs in PANEL_F_CROSS_COMPARISONS
    ]
    assert [text.get_text() for text in place_ax.get_legend().get_texts()] == [
        "L12",
        "L14",
    ]
    assert [text.get_text() for text in cross_ax.get_legend().get_texts()] == [
        "Light",
        "Dark",
    ]
    assert len(cross_ax.get_xticklabels()) == 1
    assert len(place_ax.collections) == 8
    assert len(cross_ax.collections) == 2 * 2 * len(PANEL_F_CROSS_COMPARISONS) * 2
    plt.close(fig)


def test_plot_glm_panels_use_model_schematic_and_swap_delta() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pandas = pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    swap_delta_table = pandas.DataFrame(
        {
            "delta_ll_bits_per_spike": [-0.1, 0.0, 0.2, 0.3],
            "light_train_epoch": ["02_r1", "02_r1", "02_r1", "02_r1"],
            "light_test_epoch": ["06_r3", "06_r3", "06_r3", "06_r3"],
        }
    )
    swap_example = {
        "animal_name": "L15",
        "region": "v1",
        "unit_id": 40,
        "trajectory": "center_to_left",
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
    second_swap_example["observed_rate_hz"] = np.asarray([0.3, 1.2, 0.5])
    second_swap_example["models"] = {
        "visual": np.asarray([0.2, 0.6, 0.3]),
        "task_segment_bump": np.asarray([0.3, 1.1, 0.4]),
    }
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_panel_g_model_architecture(axes[0], _fake_panel_g_examples())
    plot_panel_h_swap_delta(axes[1], swap_delta_table, [swap_example, second_swap_example])
    fig.canvas.draw()

    assert len(axes[0].child_axes) == 2
    schematic_ax, example_ax = axes[0].child_axes
    assert example_ax.get_position().width / axes[0].get_position().width == pytest.approx(
        PANEL_G_EXAMPLE_WIDTH_FRACTION
    )
    assert schematic_ax.get_position().width / axes[0].get_position().width == pytest.approx(
        PANEL_G_SCHEMATIC_WIDTH_FRACTION
    )
    assert schematic_ax.get_position().height / axes[0].get_position().height == pytest.approx(
        PANEL_G_SCHEMATIC_HEIGHT_FRACTION
    )
    assert example_ax.get_position().height / axes[0].get_position().height == pytest.approx(
        PANEL_G_EXAMPLE_HEIGHT_FRACTION
    )
    assert schematic_ax.get_position().y0 > example_ax.get_position().y0
    assert len(example_ax.child_axes) == 4
    assert all(len(field_ax.lines) == 5 for field_ax in example_ax.child_axes)
    assert [field_ax.get_title() for field_ax in example_ax.child_axes[:2]] == [
        "Dark",
        "Light",
    ]
    assert [
        text.get_text()
        for text in schematic_ax.texts
        if text.get_text()
    ] == [
        "Dark field",
        "Light field",
        "Independent\nmodel",
        "Shared-scaffold\nmodel",
        "Independent\nbasis functions",
        "Segment-specific modulation",
        "+",
    ]
    assert len(schematic_ax.child_axes) == 6
    track_patches = [
        patch
        for track_ax in schematic_ax.child_axes
        for patch in track_ax.patches
        if isinstance(patch, Polygon)
    ]
    assert len(track_patches) == 5
    panel_g_track_axes = [
        schematic_ax.child_axes[index]
        for index in (0, 2, 3, 4, 5)
    ]
    assert all(
        any(line.get_color() == PANEL_G_ARROW_COLOR for line in track_ax.lines)
        for track_ax in panel_g_track_axes
    )
    assert len(axes[1].child_axes) == 4
    schematic_h_ax, delta_h_ax, first_example_h_ax, second_example_h_ax = axes[1].child_axes
    assert schematic_h_ax.get_position().x1 < delta_h_ax.get_position().x0
    assert delta_h_ax.get_position().x1 < first_example_h_ax.get_position().x0
    assert first_example_h_ax.get_position().x0 == pytest.approx(
        second_example_h_ax.get_position().x0
    )
    assert first_example_h_ax.get_position().y0 > second_example_h_ax.get_position().y0
    assert schematic_h_ax.texts[0].get_text() == "Train: AB"
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
        any(line.get_color() == PANEL_G_ARROW_COLOR for line in track_ax.lines)
        for track_ax in schematic_h_ax.child_axes
    )
    assert delta_h_ax.get_xlabel() == "Segment bump - independent LL\n(bits/spike)"
    assert delta_h_ax.get_title() == "Held-out 06_r3"
    assert delta_h_ax.lines[0].get_linestyle() == "--"
    assert any("frac>0=0.50" in text.get_text() for text in delta_h_ax.texts)
    assert len(delta_h_ax.patches) > 0
    assert first_example_h_ax.get_xlabel() == ""
    assert second_example_h_ax.get_xlabel() == "Switched segment"
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
    assert args.panel_example_cache_dir is None
    assert args.refresh_panel_example_cache is False
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(165.0)
    assert DEFAULT_FIGURE_HEIGHT_MM > 0
    assert DEFAULT_PANEL_A_HEIGHT_MM == pytest.approx(44.8)
    assert DEFAULT_PANEL_DEF_HEIGHT_MM == pytest.approx(30.0)
    assert DEFAULT_PANEL_GH_HEIGHT_MM == pytest.approx(42.0)
    assert DEFAULT_PANEL_B_WIDTH_FRACTION == pytest.approx(0.7)
    assert DEFAULT_PANEL_C_WIDTH_FRACTION == pytest.approx(0.3)
