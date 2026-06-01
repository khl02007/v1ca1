from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_1 as figure_1_module
import v1ca1.paper_figures.supplementary_figure_1 as supp_figure_1_module
from v1ca1.paper_figures.supplementary_figure_1 import (
    DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ,
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    MOTOR_DATASET_LABEL_X,
    MOTOR_DATASET_STACK_AXIS_BOUNDS,
    MODEL_COMPARISON_GRID_WSPACE,
    MODEL_COMPARISON_SECOND_COLUMN_SHIFT_PT,
    MODEL_COMPARISON_THIRD_COLUMN_SHIFT_PT,
    PANEL_D_NORMALIZATION_HEATMAP_HSPACE,
    PANEL_D_NORMALIZATION_HEATMAP_WSPACE,
    PANEL_D_NORMALIZATION_REGION,
    PANEL_D_PER_TRAJECTORY_CACHE_VERSION,
    build_panel_d_per_trajectory_cache_metadata,
    build_output_path,
    keep_only_bottom_x_axis_labels,
    load_panel_d_per_trajectory_panels,
    load_pooled_dark_movement_firing_rate_table,
    make_supplementary_figure_1,
    parse_arguments,
    plot_decoding_error_dataset_stack_panel,
    plot_dataset_stack_panel,
    plot_motor_delta_dataset_stack_panel,
    plot_pooled_dark_movement_firing_rate_histogram,
    plot_pooled_stability_panel,
    shift_model_comparison_columns,
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
    assert args.encoding_place_bin_size_cm == pytest.approx(
        figure_1_module.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
    )
    assert DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM == pytest.approx(40.0)
    assert PANEL_D_NORMALIZATION_HEATMAP_WSPACE == pytest.approx(0.0)
    assert PANEL_D_NORMALIZATION_HEATMAP_HSPACE == pytest.approx(0.0)
    assert DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ == pytest.approx(0.5)
    assert MODEL_COMPARISON_GRID_WSPACE == pytest.approx(-0.10)
    assert MODEL_COMPARISON_SECOND_COLUMN_SHIFT_PT == pytest.approx(-10.0)
    assert MODEL_COMPARISON_THIRD_COLUMN_SHIFT_PT == pytest.approx(-33.0)
    assert not hasattr(args, "region")
    assert not hasattr(args, "panel_d_cache_dir")
    assert not hasattr(args, "panel_heatmap_cache_dir")
    assert not hasattr(args, "refresh_panel_heatmap_cache")


def test_plot_dataset_stack_panel_arranges_one_axis_per_dataset() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    plotted = []

    def plot_dataset(ax, dataset):
        plotted.append(dataset)
        ax.plot([0.0, 1.0], [0.0, 1.0])

    fig, ax = plt.subplots()
    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
    )

    assert plotted == datasets
    assert len(row_axes) == 2
    assert len(ax.child_axes) == 2
    assert [text.get_text() for text in ax.texts] == [
        "L14\n20240611\n08_r4",
        "L15\n20241121\n10_r5",
    ]
    assert row_axes[0].get_position().y0 > row_axes[1].get_position().y0
    plt.close(fig)


def test_shift_model_comparison_columns_moves_later_columns_left() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=3)
    original_positions = [ax.get_position().x0 for ax in axes]

    shift_model_comparison_columns(fig, axes)

    shifted_positions = [ax.get_position().x0 for ax in axes]
    assert shifted_positions[0] == pytest.approx(original_positions[0])
    assert shifted_positions[1] < original_positions[1]
    assert shifted_positions[2] < original_positions[2]
    assert (shifted_positions[2] - original_positions[2]) < (
        shifted_positions[1] - original_positions[1]
    )
    plt.close(fig)


def test_plot_dataset_stack_panel_can_hide_repeated_dataset_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=[("L14", "20240611", "08_r4")],
        plot_dataset=lambda row_ax, _dataset: row_ax.plot([0.0, 1.0], [0.0, 1.0]),
        show_dataset_labels=False,
    )

    assert len(row_axes) == 1
    assert len(ax.texts) == 0
    assert row_axes[0].get_position().x0 < ax.get_position().x0 + 0.05
    plt.close(fig)


def test_keep_only_bottom_x_axis_labels_hides_upper_rows() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=1)
    for ax in axes:
        ax.set_xlabel("Shared x")
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["zero", "one"])
    child_ax = axes[0].inset_axes([0.2, 0.2, 0.5, 0.5])
    child_ax.set_xlabel("Child x")
    child_ax.set_xticks([0.0, 1.0])
    child_ax.set_xticklabels(["left", "right"])

    keep_only_bottom_x_axis_labels(axes)

    assert axes[0].get_xlabel() == ""
    assert child_ax.get_xlabel() == ""
    assert not any(label.get_visible() for label in axes[0].get_xticklabels())
    assert not any(label.get_visible() for label in child_ax.get_xticklabels())
    assert axes[1].get_xlabel() == "Shared x"
    assert all(label.get_visible() for label in axes[1].get_xticklabels())
    plt.close(fig)


def test_plot_motor_delta_dataset_stack_panel_uses_animal_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    loaded = []

    def fake_load_motor_delta_table(**kwargs):
        loaded.append(kwargs["datasets"][0])
        return kwargs["datasets"][0]

    def fake_plot_motor_delta_panel(ax, _table):
        ax.set_ylabel("Frac.")
        ax.set_xlabel("Delta")
        ax.plot([0.0, 1.0], [0.0, 1.0])

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_motor_delta_table",
        fake_load_motor_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_motor_delta_panel",
        fake_plot_motor_delta_panel,
    )

    fig, ax = plt.subplots()
    row_axes = plot_motor_delta_dataset_stack_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=datasets,
    )

    assert loaded == datasets
    assert [text.get_text() for text in ax.texts] == ["L14", "L15"]
    assert all(text.get_position()[0] == pytest.approx(MOTOR_DATASET_LABEL_X) for text in ax.texts)
    assert all(text.get_ha() == "right" for text in ax.texts)
    parent_box = ax.get_position()
    first_row_box = row_axes[0].get_position()
    row_left = parent_box.x0 + MOTOR_DATASET_STACK_AXIS_BOUNDS[0] * parent_box.width
    assert first_row_box.x0 == pytest.approx(row_left)
    plt.close(fig)


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
    assert ax.get_xlabel() == "Movement firing rate (Hz)"
    assert ax.get_ylabel() == "Frac. V1 cells"
    plt.close(fig)


def test_plot_decoding_error_dataset_stack_panel_removes_w_track_icons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for comparison, label, transfer_family, _pairs in (
        figure_1_module.DECODING_CROSS_TRAJECTORY_COMPARISONS
    ):
        rows.extend(
            {
                "comparison": comparison,
                "comparison_label": label,
                "transfer_family": transfer_family,
                "absolute_error": value,
            }
            for value in (0.1, 0.2, 0.4)
        )
    monkeypatch.setattr(
        supp_figure_1_module,
        "load_decoding_absolute_error_table",
        lambda **_kwargs: pd.DataFrame(rows),
    )

    fig, ax = plt.subplots()
    row_axes = plot_decoding_error_dataset_stack_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
    )

    assert len(row_axes) == 1
    assert len(row_axes[0].child_axes) == 1
    assert row_axes[0].child_axes[0].get_ylabel() == "Abs. norm. error"
    assert "Train" not in [text.get_text() for text in row_axes[0].texts]
    plt.close(fig)


def test_panel_d_per_trajectory_cache_metadata_matches_legacy_cache() -> None:
    metadata = build_panel_d_per_trajectory_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region=PANEL_D_NORMALIZATION_REGION,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )

    assert metadata["cache_version"] == PANEL_D_PER_TRAJECTORY_CACHE_VERSION
    assert tuple(metadata["trajectory_types"]) == figure_1_module.PANEL_D_TRAJECTORY_TYPES
    assert metadata["linear_position_orientation"] == "task_progression"
    assert "firing_rate_normalization" not in metadata


def test_load_panel_d_per_trajectory_panels_uses_matching_legacy_cache(
    tmp_path: Path,
) -> None:
    metadata = build_panel_d_per_trajectory_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region=PANEL_D_NORMALIZATION_REGION,
        position_bin_count=2,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    payload = {
        figure_1_module.PANEL_D_CACHE_METADATA_KEY: np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    }
    expected_panels = {}
    for row_index, order_trajectory in enumerate(figure_1_module.PANEL_D_TRAJECTORY_TYPES):
        for col_index, plot_trajectory in enumerate(figure_1_module.PANEL_D_TRAJECTORY_TYPES):
            values = np.full((1, 2), row_index + col_index, dtype=float)
            expected_panels[(order_trajectory, plot_trajectory)] = values
            payload[f"{order_trajectory}__{plot_trajectory}"] = values
    np.savez_compressed(
        tmp_path / f"{figure_1_module.PANEL_D_CACHE_PREFIX}_test_cachev2.npz",
        **payload,
    )
    stale_payload = dict(payload)
    stale_metadata = dict(metadata)
    stale_metadata["position_bin_count"] = 3
    stale_payload[figure_1_module.PANEL_D_CACHE_METADATA_KEY] = np.asarray(
        json.dumps(stale_metadata, sort_keys=True)
    )
    np.savez_compressed(
        tmp_path / f"{figure_1_module.PANEL_D_CACHE_PREFIX}_stale_cachev2.npz",
        **stale_payload,
    )

    loaded = load_panel_d_per_trajectory_panels(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region=PANEL_D_NORMALIZATION_REGION,
        position_bin_count=2,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_d_cache_dir=tmp_path,
    )

    for key, expected in expected_panels.items():
        assert np.array_equal(loaded[key], expected)


def test_make_supplementary_figure_1_uses_paper_style_and_figure_1_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_apply_paper_style() -> None:
        calls["styled"] = True

    def fake_plot_motor_delta_dataset_stack_panel(ax, **kwargs: object):
        calls["motor_kwargs"] = kwargs
        ax.text(0.5, 0.5, "motor")
        return []

    def fake_plot_encoding_delta_dataset_stack_panel(ax, **kwargs: object):
        calls["encoding_kwargs"] = kwargs
        ax.text(0.5, 0.5, "encoding")
        return []

    def fake_plot_decoding_error_dataset_stack_panel(ax, **kwargs: object):
        calls["decoding_kwargs"] = kwargs
        ax.text(0.5, 0.5, "decoding")
        return []

    def fake_draw_panel_a_anatomy_assets(ax, **kwargs: object):
        calls["anatomy_kwargs"] = kwargs
        ax.text(0.5, 0.5, "anatomy")

    def fake_plot_pooled_stability_panel(ax, **kwargs: object):
        calls["pooled_stability_kwargs"] = kwargs
        calls["pooled_stability_col"] = ax.get_subplotspec().colspan.start
        ax.text(0.5, 0.5, "pooled stability")

    def fake_plot_pooled_dark_movement_firing_rate_panel(ax, **kwargs: object):
        calls["dark_movement_fr_kwargs"] = kwargs
        calls["dark_movement_fr_col"] = ax.get_subplotspec().colspan.start
        ax.text(0.5, 0.5, "dark fr")

    def fake_save_figure(figure, output_path: Path, dpi: int):
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        return output_path

    monkeypatch.setattr(supp_figure_1_module, "apply_paper_style", fake_apply_paper_style)
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_motor_delta_dataset_stack_panel",
        fake_plot_motor_delta_dataset_stack_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_encoding_delta_dataset_stack_panel",
        fake_plot_encoding_delta_dataset_stack_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_decoding_error_dataset_stack_panel",
        fake_plot_decoding_error_dataset_stack_panel,
    )
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
    monkeypatch.setattr(supp_figure_1_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_1.svg"
    asset_dir = tmp_path / "assets"
    datasets = [("L14", "20240611", "08_r4")]
    saved_path = make_supplementary_figure_1(
        data_root=Path("/analysis"),
        asset_dir=asset_dir,
        output_path=output_path,
        datasets=datasets,
        encoding_place_bin_size_cm=4.0,
        dpi=300,
    )

    figure_width_in = calls["figsize"][0]
    assert saved_path == output_path
    assert calls["styled"] is True
    assert figure_width_in == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(
        (
            DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM
            + supp_figure_1_module.DEFAULT_SECTION_SPACER_MM
            + supp_figure_1_module.DEFAULT_DATASET_PANEL_HEIGHT_MM * len(datasets)
        )
        / 25.4
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["A", "B", "C", "D", "E", "F"]
    assert calls["anatomy_kwargs"]["asset_dir"] == asset_dir
    assert calls["pooled_stability_kwargs"]["datasets"] == datasets
    assert calls["pooled_stability_col"] == 2
    assert calls["dark_movement_fr_kwargs"]["datasets"] == datasets
    assert calls["dark_movement_fr_col"] == 1
    assert calls["dark_movement_fr_kwargs"]["cache_dir"] == output_path.parent / "cache"
    assert calls["dark_movement_fr_kwargs"]["refresh_cache"] is False
    assert calls["motor_kwargs"]["datasets"] == datasets
    assert calls["motor_kwargs"].get("show_dataset_labels", True) is True
    assert calls["encoding_kwargs"]["datasets"] == datasets
    assert calls["encoding_kwargs"]["show_dataset_labels"] is False
    assert calls["encoding_kwargs"]["place_bin_size_cm"] == pytest.approx(4.0)
    assert calls["decoding_kwargs"]["datasets"] == datasets
    assert calls["decoding_kwargs"]["show_dataset_labels"] is False
