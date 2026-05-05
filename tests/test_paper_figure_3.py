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
    DEFAULT_REGIONS,
    PANEL_A_EPOCH_COLORS,
    PANEL_A_EXAMPLE,
    PANEL_A_LIGHT_EPOCHS,
    PANEL_A_TRAJECTORIES,
    PANEL_C_EXAMPLES,
    PANEL_C_TRAJECTORY_COLORS,
    SEGMENT_BOUNDARIES,
    build_panel_d_similarity_pairs,
    build_panel_a_epoch_specs,
    build_output_path,
    get_decoding_summary_path,
    get_encoding_summary_candidate_paths,
    get_dark_epoch,
    get_light_epoch,
    get_tuning_similarity_path,
    load_panel_quantification_data,
    make_light_epoch_dataset_ids,
    parse_arguments,
    parse_dataset_id,
    plot_panel_d_similarity,
    plot_panel_e_encoding_delta_histogram,
    plot_panel_f_decoding_error,
    plot_panel_a_example,
    plot_epoch_path_rate_axis,
    plot_light_heatmap_regions,
    plot_panel_c_examples,
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
                "firing_rates": {
                    trajectory: (position, np.asarray([0.0, 1.0, 0.5], dtype=float))
                    for trajectory in trajectories
                },
            },
            "light": {
                "epoch": "02_r1",
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
    assert ax.get_xlabel() == "Norm. task progression"
    assert ax.get_title() == "Dark"
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_panel_a_example_draws_epoch_rasters_and_bottom_rate_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    plot_panel_a_example(ax, _fake_panel_a_example())

    assert len(ax.child_axes) == 23
    condition_axes = ax.child_axes[:3]
    schematic_axes = ax.child_axes[3::5]
    first_trajectory_raster_axes = ax.child_axes[4:7]
    rate_axes = ax.child_axes[7::5]
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
    assert all(text.get_text() != "C to L" for text in ax.texts)
    assert len(schematic_axes) == 4
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
    assert [line.get_color() for line in first_trajectory_raster_axes[0].lines[:2]] == [
        PANEL_A_EPOCH_COLORS["02_r1"],
        PANEL_A_EPOCH_COLORS["02_r1"],
    ]
    assert [
        first_trajectory_raster_axes[0].lines[index].get_xdata()[0]
        for index in (2, 3)
    ] == pytest.approx([0.4, 0.6])
    assert rate_axes[-1].get_legend() is not None
    plt.close(fig)


def test_plot_panel_c_examples_stacks_two_curve_blocks() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    plot_panel_c_examples(
        ax,
        [
            _fake_panel_b_example(("center_to_left", "right_to_center")),
            _fake_panel_b_example(("center_to_right", "left_to_center")),
        ],
    )

    assert len(ax.child_axes) == 2
    for example_index, example_ax in enumerate(ax.child_axes, start=1):
        assert [text.get_text() for text in example_ax.texts] == [
            f"Example cell {example_index}"
        ]
        assert len(example_ax.child_axes) == 4
        schematic_axes = example_ax.child_axes[:2]
        schematic_patches = [
            patch
            for schematic_ax in schematic_axes
            for patch in schematic_ax.patches
            if isinstance(patch, Polygon)
        ]
        assert schematic_patches
        assert all(patch.get_facecolor()[3] == pytest.approx(0.0) for patch in schematic_patches)
        rate_axes = example_ax.child_axes[2:]
        assert all(len(rate_ax.lines) == 4 for rate_ax in rate_axes)
        assert [
            rate_axes[0].lines[index].get_xdata()[0]
            for index in (2, 3)
        ] == pytest.approx(list(SEGMENT_BOUNDARIES))
        assert [rate_ax.get_title() for rate_ax in rate_axes] == ["Dark", "Light"]
        assert rate_axes[0].get_position().x0 < rate_axes[1].get_position().x0
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
    decoding_table = pandas.DataFrame(
        {
            "animal_name": ["L12", "L12", "L12", "L12", "L14", "L14", "L14", "L14"],
            "epoch_type": [
                "light",
                "dark",
                "light",
                "dark",
                "light",
                "dark",
                "light",
                "dark",
            ],
            "model": [
                "task_progression",
                "task_progression",
                "place",
                "place",
                "task_progression",
                "task_progression",
                "place",
                "place",
            ],
            "median_abs_error": [0.02, 0.04, 2.0, 3.0, 0.03, 0.05, 2.5, 3.5],
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=3)
    plot_panel_d_similarity(axes[0], similarity_table)
    plot_panel_e_encoding_delta_histogram(axes[1], delta_table)
    plot_panel_f_decoding_error(axes[2], decoding_table)

    paired = build_panel_d_similarity_pairs(similarity_table)
    assert paired[["similarity_light", "similarity_dark"]].to_numpy().ravel() == pytest.approx(
        [0.6, 0.3, 0.1, 0.4]
    )
    assert axes[0].get_xlabel() == "Light same-turn corr."
    assert axes[0].get_ylabel() == "Dark same-turn corr."
    assert axes[0].lines[0].get_linestyle() == "--"
    assert axes[0].lines[0].get_xdata().tolist() == [-1.0, 1.0]
    assert axes[0].lines[0].get_ydata().tolist() == [-1.0, 1.0]
    assert len(axes[0].collections) == 1
    assert axes[0].collections[0].get_alpha() == pytest.approx(0.18)
    assert axes[1].get_xlabel() == "TP - place LL\n(bits/spike)"
    assert len(axes[2].child_axes) == 2
    assert [child_ax.get_title() for child_ax in axes[2].child_axes] == ["TP", "Place"]
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
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(165.0)
    assert DEFAULT_FIGURE_HEIGHT_MM > 0
    assert DEFAULT_PANEL_A_HEIGHT_MM == pytest.approx(44.8)
    assert DEFAULT_PANEL_DEF_HEIGHT_MM == pytest.approx(30.0)
    assert DEFAULT_PANEL_B_WIDTH_FRACTION == pytest.approx(0.7)
    assert DEFAULT_PANEL_C_WIDTH_FRACTION == pytest.approx(0.3)
