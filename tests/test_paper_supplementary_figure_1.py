from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_1 as figure_1_module
import v1ca1.paper_figures.supplementary_figure_1 as supp_figure_1_module
from v1ca1.paper_figures.supplementary_figure_1 import (
    CA1_DECODING_REGION,
    CA1_HEATMAP_REGION,
    CA1_HEATMAP_TITLE,
    DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ,
    DECODING_COMPARISON_REGIONS,
    DECODING_COMPARISON_TITLE,
    DEFAULT_ASSET_DIR,
    DEFAULT_CA1_HEATMAP_ROW_HEIGHT_MM,
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    ORDER_PRESERVATION_REGIONS,
    ORDER_PRESERVATION_TITLE,
    TUNING_SIMILARITY_TITLE,
    TURN_PAIR_SPECS,
    build_ca1_decoding_significance_brackets,
    build_output_path,
    build_turn_tuning_similarity_table,
    compute_curve_peak_positions,
    compute_equal_animal_weights,
    compute_peak_order_preservation_score,
    compute_rowwise_tuning_curve_correlations,
    load_pooled_same_turn_trial_error_table,
    load_pooled_dark_movement_firing_rate_table,
    make_supplementary_figure_1,
    parse_arguments,
    plot_cross_path_decoding_region_comparison_panel,
    plot_order_preservation_panel,
    plot_pooled_ca1_dark_heatmap_panel,
    plot_pooled_dark_movement_firing_rate_histogram,
    plot_pooled_stability_panel,
    plot_turn_region_distribution_panel,
    summarize_order_preservation_scores,
    weighted_quantile,
)


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "supplementary_figure_1", "svg") == Path(
        "paper_figures/supplementary_figure_1.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "supplementary_figure_1", "jpg")


def test_default_cli_matches_supplementary_figure_format() -> None:
    args = parse_arguments([])

    assert DEFAULT_FIGURE_WIDTH_MM == figure_1_module.DEFAULT_FIGURE_WIDTH_MM
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert args.asset_dir == DEFAULT_ASSET_DIR
    assert args.dark_movement_fr_cache_dir is None
    assert args.refresh_dark_movement_fr_cache is False
    assert not hasattr(args, "panel_d_cache_dir")
    assert not hasattr(args, "refresh_panel_d_cache")
    assert not hasattr(args, "position_bin_count")
    assert not hasattr(args, "position_offset")
    assert not hasattr(args, "speed_threshold_cm_s")
    assert not hasattr(args, "sigma_bins")
    assert not hasattr(args, "decoding_n_permutations")
    assert not hasattr(args, "decoding_permutation_seed")
    assert DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM == pytest.approx(40.0)
    assert DEFAULT_CA1_HEATMAP_ROW_HEIGHT_MM == pytest.approx(
        figure_1_module.DEFAULT_HEATMAP_HEIGHT_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM
    )
    assert DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ == pytest.approx(0.5)
    assert CA1_HEATMAP_REGION == "ca1"
    assert CA1_DECODING_REGION == "ca1"
    assert DECODING_COMPARISON_REGIONS == ("v1", "ca1")
    assert not hasattr(args, "region")
    assert not hasattr(args, "encoding_place_bin_size_cm")
    assert not hasattr(args, "panel_heatmap_cache_dir")
    assert not hasattr(args, "refresh_panel_heatmap_cache")


def test_main_forwards_abc_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_make_supplementary_figure_1(**kwargs):
        calls.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(
        supp_figure_1_module,
        "make_supplementary_figure_1",
        fake_make_supplementary_figure_1,
    )

    supp_figure_1_module.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            str(tmp_path),
            "--output-name",
            "ca1_supp",
            "--format",
            "svg",
            "--dataset",
            "L14:20240611:08_r4",
            "--dark-movement-fr-cache-dir",
            "/cache",
            "--refresh-dark-movement-fr-cache",
        ]
    )

    assert calls["data_root"] == Path("/analysis")
    assert calls["output_path"] == tmp_path / "ca1_supp.svg"
    assert calls["datasets"] == [("L14", "20240611", "08_r4")]
    assert calls["dark_movement_fr_cache_dir"] == Path("/cache")
    assert calls["refresh_dark_movement_fr_cache"] is True


def test_plot_pooled_stability_panel_uses_all_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    calls: dict[str, object] = {}
    table = pd.DataFrame(
        {
            "trajectory_type": ["center_to_left"],
            "region": ["v1"],
            "stability_correlation": [0.5],
        }
    )

    def fake_load_dark_epoch_stability_table(**kwargs):
        calls["load_kwargs"] = kwargs
        return table

    def fake_plot_stability_panel(ax, stability_table):
        calls["plot_table"] = stability_table
        ax.text(0.5, 0.5, "pooled")

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_dark_epoch_stability_table",
        fake_load_dark_epoch_stability_table,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_stability_panel",
        fake_plot_stability_panel,
    )

    fig, ax = plt.subplots()
    plot_pooled_stability_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=datasets,
    )

    assert calls["load_kwargs"]["datasets"] == datasets
    assert calls["load_kwargs"]["regions"] == figure_1_module.STABILITY_REGIONS
    assert calls["plot_table"] is table
    plt.close(fig)


def test_load_pooled_dark_movement_firing_rate_table_uses_dataset_dark_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")

    calls = []

    def fake_load_dark_movement_firing_rate_table(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "unit": [len(calls)],
                "dark_firing_rate_hz": [0.25 * len(calls)],
            }
        )

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_dark_movement_firing_rate_table",
        fake_load_dark_movement_firing_rate_table,
    )

    table = load_pooled_dark_movement_firing_rate_table(
        Path("/analysis"),
        [
            ("L14", "20240611", "08_r4"),
            ("L15", "20241121", "10_r5"),
        ],
        cache_dir=Path("/cache"),
        refresh_cache=True,
    )

    assert [call["animal_name"] for call in calls] == ["L14", "L15"]
    assert [call["dark_epoch"] for call in calls] == ["08_r4", "10_r5"]
    assert all(call["region"] == "v1" for call in calls)
    assert all(call["cache_dir"] == Path("/cache") for call in calls)
    assert all(call["refresh_cache"] is True for call in calls)
    assert table["animal_name"].tolist() == ["L14", "L15"]
    assert table["date"].tolist() == ["20240611", "20241121"]
    assert table["dark_epoch"].tolist() == ["08_r4", "10_r5"]
    assert table["dark_firing_rate_hz"].tolist() == [0.25, 0.5]


def test_plot_pooled_dark_movement_firing_rate_histogram_uses_log_threshold() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_pooled_dark_movement_firing_rate_histogram(
        ax,
        {
            "dark_firing_rate_hz": [
                0.0,
                0.1,
                0.5,
                0.6,
                1.2,
                np.nan,
            ]
        },
    )

    assert ax.get_xscale() == "log"
    threshold_lines = [
        line
        for line in ax.lines
        if line.get_linestyle() == "--"
        and np.allclose(line.get_xdata(), [DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ] * 2)
    ]
    assert len(threshold_lines) == 1
    assert any("40% > 0.5 Hz" in text.get_text() for text in ax.texts)
    assert not any("at 0 Hz" in text.get_text() for text in ax.texts)
    assert ax.get_xlabel() == "Mean firing rate (Hz)"
    assert ax.get_ylabel() == "Frac."
    plt.close(fig)


def test_plot_pooled_ca1_dark_heatmap_panel_uses_figure_1d_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 5)
    panel = {
        "corner_axis": axes[0, 0],
        "tuning_schematic_axes": axes[0, 1:],
        "order_schematic_axes": axes[1:, 0],
        "heatmap_axes": axes[1:, 1:],
    }
    calls: dict[str, object] = {}

    def fake_setup_heatmap_comparison_panel(fig_arg, grid_spec, **kwargs):
        calls["setup_fig"] = fig_arg
        calls["setup_grid_spec"] = grid_spec
        calls["setup_kwargs"] = kwargs
        return panel

    def fake_plot_dark_heatmap_regions(heatmap_axes, **kwargs):
        calls["heatmap_axes"] = heatmap_axes
        calls["heatmap_kwargs"] = kwargs
        return None

    def fake_draw_neuron_scale_bar(ax):
        calls["scale_axis"] = ax

    monkeypatch.setattr(
        supp_figure_1_module,
        "setup_heatmap_comparison_panel",
        fake_setup_heatmap_comparison_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_dark_heatmap_regions",
        fake_plot_dark_heatmap_regions,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "draw_neuron_scale_bar",
        fake_draw_neuron_scale_bar,
    )

    grid_spec = object()
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    returned_panel = plot_pooled_ca1_dark_heatmap_panel(
        fig,
        grid_spec,
        data_root=Path("/analysis"),
        datasets=datasets,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_d_cache_dir=Path("/cache"),
        refresh_panel_d_cache=True,
    )

    assert returned_panel is panel
    assert calls["setup_fig"] is fig
    assert calls["setup_grid_spec"] is grid_spec
    assert calls["setup_kwargs"] == {
        "trajectory_types": figure_1_module.PANEL_D_TRAJECTORY_TYPES,
        "fill_track": True,
    }
    assert calls["heatmap_axes"] is panel["heatmap_axes"]
    assert calls["heatmap_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": datasets,
        "regions": (CA1_HEATMAP_REGION,),
        "position_bin_count": 50,
        "position_offset": 10,
        "speed_threshold_cm_s": 4.0,
        "sigma_bins": 1.5,
        "panel_d_cache_dir": Path("/cache"),
        "refresh_panel_d_cache": True,
    }
    assert calls["scale_axis"] is panel["heatmap_axes"][-1, -1]
    plt.close(fig)


def test_rowwise_tuning_curve_correlations_excludes_invalid_curves() -> None:
    correlations = compute_rowwise_tuning_curve_correlations(
        np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
                [np.nan, 1.0, np.nan],
            ]
        ),
        np.asarray(
            [
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        ),
    )

    assert correlations[:2] == pytest.approx([1.0, -1.0])
    assert np.isnan(correlations[2:]).all()
    with pytest.raises(ValueError, match="matching shapes"):
        compute_rowwise_tuning_curve_correlations(
            np.zeros((2, 3)),
            np.zeros((2, 4)),
        )


def test_build_turn_tuning_similarity_table_keeps_both_turn_pairs() -> None:
    pd = pytest.importorskip("pandas")
    del pd

    panels_by_region = {}
    ordered_keys_by_region = {}
    for region in DECODING_COMPARISON_REGIONS:
        panels_by_region[region] = {
            ("center_to_left", "center_to_left"): np.asarray(
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
            ),
            ("center_to_left", "right_to_center"): np.asarray(
                [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
            ),
            ("center_to_right", "center_to_right"): np.asarray(
                [[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]]
            ),
            ("center_to_right", "left_to_center"): np.asarray(
                [[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
            ),
        }
        ordered_keys_by_region[region] = {
            "center_to_left": np.asarray(
                [f"L12:date:{region}:1", f"L14:date:{region}:2"]
            ),
            "center_to_right": np.asarray(
                [f"L12:date:{region}:1", f"L14:date:{region}:2"]
            ),
        }

    table = build_turn_tuning_similarity_table(
        panels_by_region,
        ordered_keys_by_region,
    )

    assert len(table) == 8
    assert set(table["turn_type"]) == {
        turn_type for turn_type, _label, _first, _second in TURN_PAIR_SPECS
    }
    assert set(table["region"]) == set(DECODING_COMPARISON_REGIONS)
    assert set(table["animal_name"]) == {"L12", "L14"}
    left_values = table.loc[
        (table["region"] == "v1") & (table["turn_type"] == "left_turn"),
        "tuning_correlation",
    ]
    assert left_values.tolist() == pytest.approx([1.0, -1.0])


def test_equal_animal_weights_do_not_favor_animals_with_more_observations() -> None:
    weights = compute_equal_animal_weights(["L12", "L12", "L14"])

    assert weights == pytest.approx([0.25, 0.25, 0.50])
    assert np.sum(weights[:2]) == pytest.approx(weights[2])
    assert compute_equal_animal_weights(
        ["L12", "L12", "L12", "L14", "L14"],
        ["a", "a", "b", "a", "b"],
    ) == pytest.approx([0.125, 0.125, 0.25, 0.25, 0.25])
    assert weighted_quantile(
        np.asarray([0.0, 1.0, 2.0]),
        (0.25, 0.50, 0.75),
        np.ones(3),
    ) == pytest.approx([0.25, 1.0, 1.75])


def test_plot_turn_region_distribution_panel_draws_four_weighted_groups() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    for turn_type, _label, _first, _second in TURN_PAIR_SPECS:
        for region_index, region in enumerate(DECODING_COMPARISON_REGIONS):
            for animal_name, value in (
                ("L12", 0.2 + 0.2 * region_index),
                ("L12", 0.3 + 0.2 * region_index),
                ("L14", 0.5 + 0.2 * region_index),
            ):
                records.append(
                    {
                        "region": region,
                        "animal_name": animal_name,
                        "turn_type": turn_type,
                        "value": value,
                    }
                )

    fig, ax = plt.subplots()
    plot_turn_region_distribution_panel(
        ax,
        pd.DataFrame.from_records(records),
        value_column="value",
        ylabel="Similarity",
        y_limits=(-1.0, 1.0),
        show_legend=True,
        show_zero_line=True,
    )

    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "Left-turn\npair",
        "Right-turn\npair",
    ]
    assert ax.get_ylabel() == "Similarity"
    assert ax.get_ylim() == pytest.approx((-1.0, 1.0))
    assert len(ax.collections) >= 8
    assert ax.get_legend() is not None
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    plt.close(fig)


def test_load_pooled_same_turn_trial_error_table_assigns_turn_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    calls = []

    def fake_build_decoding_trial_error_table(**kwargs):
        calls.append(kwargs)
        region = kwargs["region"]
        return pd.DataFrame(
            {
                "region": [region] * 5,
                "animal_name": ["L12"] * 5,
                "date": ["20240421"] * 5,
                "epoch": ["08_r4"] * 5,
                "comparison": [
                    "same_turn_cross_arm",
                    "same_turn_cross_arm",
                    "same_turn_cross_arm",
                    "same_turn_cross_arm",
                    "opposite_turn_same_arm",
                ],
                "encoding_trajectory": [
                    "center_to_left",
                    "right_to_center",
                    "center_to_right",
                    "left_to_center",
                    "center_to_left",
                ],
                "decoding_trajectory": [
                    "right_to_center",
                    "center_to_left",
                    "left_to_center",
                    "center_to_right",
                    "left_to_center",
                ],
                "trial_index": [0, 0, 0, 0, 0],
                "trial_median_absolute_error": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    monkeypatch.setattr(
        supp_figure_1_module,
        "build_decoding_trial_error_table",
        fake_build_decoding_trial_error_table,
    )

    datasets = [("L12", "20240421", "08_r4")]
    table = load_pooled_same_turn_trial_error_table(
        data_root=Path("/analysis"),
        datasets=datasets,
    )

    assert calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": region,
            "comparisons": supp_figure_1_module.SAME_TURN_DECODING_COMPARISONS,
        }
        for region in DECODING_COMPARISON_REGIONS
    ]
    assert len(table) == 8
    assert table["turn_type"].tolist() == [
        "left_turn",
        "left_turn",
        "right_turn",
        "right_turn",
        "left_turn",
        "left_turn",
        "right_turn",
        "right_turn",
    ]
    assert set(table["region"]) == set(DECODING_COMPARISON_REGIONS)


def test_compute_curve_peak_positions_rejects_invalid_and_flat_curves() -> None:
    peaks = compute_curve_peak_positions(
        np.asarray(
            [
                [0.0, 1.0, 1.0, 0.0],
                [np.nan, 2.0, np.nan, np.nan],
                [1.0, 1.0, 1.0, 1.0],
                [np.nan, 0.0, 2.0, 1.0],
            ]
        )
    )

    assert peaks[[0, 3]] == pytest.approx([1.5, 2.0])
    assert np.isnan(peaks[[1, 2]]).all()


def test_peak_order_preservation_score_uses_animal_rows_and_odd_peaks() -> None:
    ordered_unit_keys = np.asarray(
        [
            "L12:20240421:v1:1",
            "L14:20240611:v1:1",
            "L12:20240421:v1:2",
            "L12:20240421:v1:3",
        ]
    )
    panel_values = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    ordered_peak_positions = np.asarray([0.0, 1.0, 1.0, 2.0])

    score, unit_count = compute_peak_order_preservation_score(
        panel_values,
        ordered_unit_keys,
        ordered_peak_positions,
        "L12",
    )

    assert score == pytest.approx(1.0)
    assert unit_count == 3


def test_order_preservation_summary_weights_animals_equally() -> None:
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame(
        {
            "region": ["v1", "v1"],
            "animal_name": ["L12", "L14"],
            "order_trajectory": ["a", "a"],
            "plot_trajectory": ["a", "a"],
            "spearman_rho": [1.0, 0.0],
            "n_units": [100, 3],
        }
    )

    matrices = summarize_order_preservation_scores(
        table,
        ("L12", "L14"),
        regions=("v1",),
        trajectory_types=("a", "b"),
    )

    assert matrices["v1"][0, 0] == pytest.approx(0.5)
    assert np.isnan(matrices["v1"][0, 1])


def test_plot_order_preservation_panel_uses_shared_scale() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = {
        region: np.arange(16, dtype=float).reshape(4, 4) / 7.5 - 1.0
        for region in ORDER_PRESERVATION_REGIONS
    }
    fig, ax = plt.subplots()
    plot_order_preservation_panel(ax, matrices)

    matrix_axes = ax.child_axes[:2]
    colorbar_axis = ax.child_axes[2]
    assert [matrix_ax.get_title() for matrix_ax in matrix_axes] == ["V1", "CA1"]
    assert all(matrix_ax.images[0].norm.vmin == -1.0 for matrix_ax in matrix_axes)
    assert all(matrix_ax.images[0].norm.vmax == 1.0 for matrix_ax in matrix_axes)
    assert all(len(matrix_ax.patches) == 2 for matrix_ax in matrix_axes)
    assert colorbar_axis.get_ylabel() == "Mean ρ"
    plt.close(fig)


def test_ca1_decoding_brackets_require_consistent_same_turn_advantage() -> None:
    pd = pytest.importorskip("pandas")
    animals = figure_1_module.PANEL_H_DECODING_ANIMALS
    records = []
    for contrast_index, (comparison_a, comparison_b, _y) in enumerate(
        figure_1_module.DECODING_SIGNIFICANCE_CONTRASTS
    ):
        for animal_index, animal_name in enumerate(animals):
            median_difference = -0.1
            if contrast_index == 1 and animal_index == 0:
                median_difference = 0.1
            records.append(
                {
                    "animal_name": animal_name,
                    "comparison_a": comparison_a,
                    "comparison_b": comparison_b,
                    "median_difference": median_difference,
                    "p_two_sided": 0.004,
                }
            )

    brackets = build_ca1_decoding_significance_brackets(
        pd.DataFrame.from_records(records)
    )

    assert len(brackets) == 1
    assert brackets[0][-1] == "**"


def test_plot_cross_path_decoding_region_comparison_uses_figure_1g_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls: dict[str, object] = {}
    datasets = [
        ("L12", "20240421", "08_r4"),
    ]
    error_tables = {
        region: pd.DataFrame(
            {
                "region": [region],
                "comparison": ["same_turn_cross_arm"],
                "absolute_error": [0.2],
            }
        )
        for region in DECODING_COMPARISON_REGIONS
    }
    trial_tables = {
        region: pd.DataFrame(
            {
                "region": [region],
                "animal_name": ["L12"],
            }
        )
        for region in DECODING_COMPARISON_REGIONS
    }
    permutation_tables = {
        region: pd.DataFrame(
            {
                "region": [region],
                "animal_name": ["L12"],
            }
        )
        for region in DECODING_COMPARISON_REGIONS
    }

    def fake_load_decoding_absolute_error_table(**kwargs):
        calls.setdefault("load_kwargs", []).append(kwargs)
        return error_tables[kwargs["region"]]

    def fake_build_decoding_trial_error_table(**kwargs):
        calls.setdefault("trial_kwargs", []).append(kwargs)
        return trial_tables[kwargs["region"]]

    def fake_compute_decoding_permutation_tests(table, **kwargs):
        region = str(table["region"].iloc[0])
        calls.setdefault("permutation_calls", []).append(
            (region, table, kwargs)
        )
        return permutation_tables[region]

    def fake_plot_decoding_error_panel(ax, table, **kwargs):
        calls["plot_axis"] = ax
        calls["plot_table"] = table
        calls["plot_kwargs"] = kwargs

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_decoding_absolute_error_table",
        fake_load_decoding_absolute_error_table,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "build_decoding_trial_error_table",
        fake_build_decoding_trial_error_table,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "compute_decoding_permutation_tests",
        fake_compute_decoding_permutation_tests,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_decoding_error_panel",
        fake_plot_decoding_error_panel,
    )

    fig, ax = plt.subplots()
    returned = plot_cross_path_decoding_region_comparison_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=datasets,
        n_permutations=101,
        permutation_seed=7,
    )

    assert set(returned) == set(permutation_tables)
    assert all(
        returned[region] is permutation_tables[region]
        for region in DECODING_COMPARISON_REGIONS
    )
    assert calls["load_kwargs"] == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": region,
        }
        for region in DECODING_COMPARISON_REGIONS
    ]
    assert calls["trial_kwargs"] == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": region,
        }
        for region in DECODING_COMPARISON_REGIONS
    ]
    assert [
        region
        for region, _table, _kwargs in calls["permutation_calls"]
    ] == list(DECODING_COMPARISON_REGIONS)
    assert all(
        table is trial_tables[region]
        for region, table, _kwargs in calls["permutation_calls"]
    )
    assert all(
        kwargs == {
            "n_permutations": 101,
            "seed": 7,
        }
        for _region, _table, kwargs in calls["permutation_calls"]
    )
    assert set(calls["plot_table"]["region"]) == set(DECODING_COMPARISON_REGIONS)
    assert calls["plot_kwargs"] == {
        "comparisons": supp_figure_1_module.CA1_DECODING_COMPARISONS,
        "significance_brackets": (),
        "regions": DECODING_COMPARISON_REGIONS,
        "show_region_legend": True,
        "show_median_labels": False,
        "xtick_label_fontsize": (
            supp_figure_1_module.CA1_DECODING_XTICK_LABEL_FONTSIZE
        ),
    }
    plt.close(fig)


def test_make_supplementary_figure_1_uses_paper_style_and_figure_1_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_apply_paper_style() -> None:
        calls["styled"] = True

    def fake_draw_panel_a_anatomy_assets(ax, **kwargs: object):
        calls["anatomy_kwargs"] = kwargs
        calls["anatomy_axis"] = ax
        ax.text(0.5, 0.5, "anatomy")

    def fake_plot_pooled_stability_panel(ax, **kwargs: object):
        calls["pooled_stability_kwargs"] = kwargs
        calls["pooled_stability_col"] = ax.get_subplotspec().colspan.start
        ax.text(0.5, 0.5, "pooled stability")

    def fake_plot_pooled_dark_movement_firing_rate_panel(ax, **kwargs: object):
        calls["dark_movement_fr_kwargs"] = kwargs
        calls["dark_movement_fr_col"] = ax.get_subplotspec().colspan.start
        ax.text(0.5, 0.5, "dark fr")

    def fail_removed_panel(*args: object, **kwargs: object) -> None:
        raise AssertionError("Removed Supplementary Figure 1 panel was plotted.")

    def fake_save_figure(figure, output_path: Path, dpi: int):
        figure.canvas.draw()
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        calls["axis_titles"] = [ax.get_title() for ax in figure.axes]
        calls["figure_texts"] = [text.get_text() for text in figure.texts]
        return output_path

    monkeypatch.setattr(supp_figure_1_module, "apply_paper_style", fake_apply_paper_style)
    monkeypatch.setattr(
        supp_figure_1_module,
        "draw_panel_a_anatomy_assets",
        fake_draw_panel_a_anatomy_assets,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_pooled_stability_panel",
        fake_plot_pooled_stability_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_pooled_dark_movement_firing_rate_panel",
        fake_plot_pooled_dark_movement_firing_rate_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_pooled_ca1_dark_heatmap_panel",
        fail_removed_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_pooled_turn_tuning_similarity_panel",
        fail_removed_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_pooled_same_turn_decoding_panel",
        fail_removed_panel,
    )
    monkeypatch.setattr(supp_figure_1_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_1.svg"
    asset_dir = tmp_path / "assets"
    datasets = [("L14", "20240611", "08_r4")]
    saved_path = make_supplementary_figure_1(
        data_root=Path("/analysis"),
        asset_dir=asset_dir,
        output_path=output_path,
        datasets=datasets,
        dpi=300,
    )

    figure_width_in = calls["figsize"][0]
    assert saved_path == output_path
    assert calls["styled"] is True
    assert figure_width_in == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(
        DEFAULT_FIGURE_HEIGHT_MM / 25.4
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert sorted(calls["panel_labels"]) == ["A", "B", "C"]
    assert "Probe and histology" in calls["axis_titles"]
    assert "V1 firing rate in darkness" in calls["axis_titles"]
    assert "Tuning stability" in calls["axis_titles"]
    assert TUNING_SIMILARITY_TITLE not in calls["axis_titles"]
    assert DECODING_COMPARISON_TITLE not in calls["axis_titles"]
    assert CA1_HEATMAP_TITLE not in calls["figure_texts"]
    assert "Order" not in calls["figure_texts"]
    assert figure_1_module.TASK_PROGRESSION_XLABEL not in calls["figure_texts"]
    assert calls["anatomy_kwargs"]["asset_dir"] == asset_dir
    assert calls["pooled_stability_kwargs"]["datasets"] == datasets
    assert calls["pooled_stability_col"] == 2
    assert calls["dark_movement_fr_kwargs"]["datasets"] == datasets
    assert calls["dark_movement_fr_col"] == 1
    assert calls["dark_movement_fr_kwargs"]["cache_dir"] == output_path.parent / "cache"
    assert calls["dark_movement_fr_kwargs"]["refresh_cache"] is False
