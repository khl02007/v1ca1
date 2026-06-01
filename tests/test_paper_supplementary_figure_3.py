from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_3 as figure_3_module
import v1ca1.paper_figures.supplementary_figure_3 as supp_figure_3_module
from v1ca1.paper_figures.supplementary_figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_BOTTOM_SECTION_SPACER_MM,
    DEFAULT_DARK_LIGHT_CORRELATION_HEIGHT_MM,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PER_ANIMAL_GRID_HEIGHT_MM,
    DEFAULT_REORDERED_HEATMAP_HEIGHT_MM,
    DEFAULT_SECTION_SPACER_MM,
    DEFAULT_STABILITY_OVERLAY_HEIGHT_MM,
    DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_A_CACHE_VERSION,
    PANEL_A_FIGURE_1D_ORDER_MODE,
    REORDERED_HEATMAP_CMAP,
    REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION,
    REORDERED_HEATMAP_VMAX,
    STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION,
    build_dark_light_tuning_correlation_table,
    build_figure_1d_ordered_light_panel_values,
    build_panel_a_cache_metadata,
    build_panel_a_cache_path,
    compute_tuning_curve_correlation,
    filter_panel_d_similarity_table_by_tuning_stability,
    filter_ordered_unit_keys_by_unit_set,
    group_datasets_by_animal,
    load_dark_light_tuning_correlation_table,
    load_light_tuning_stability_table,
    load_panel_a_cache,
    load_dark_ordered_light_panel_values,
    make_supplementary_figure_3,
    parse_arguments,
    plot_dark_ordered_light_heatmap_regions,
    plot_dark_light_tuning_correlation_histograms,
    plot_dark_light_with_light_stability_histograms,
    save_panel_a_cache,
    set_heatmap_display_style,
)


def test_default_cli_matches_figure_3_size_and_region() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "supplementary_figure_3"
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        figure_3_module.DEFAULT_FIGURE_WIDTH_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        DEFAULT_REORDERED_HEATMAP_HEIGHT_MM
        + DEFAULT_SECTION_SPACER_MM
        + DEFAULT_PER_ANIMAL_GRID_HEIGHT_MM
        + DEFAULT_BOTTOM_SECTION_SPACER_MM
        + DEFAULT_DARK_LIGHT_CORRELATION_HEIGHT_MM
        + DEFAULT_STABILITY_OVERLAY_HEIGHT_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM > figure_3_module.DEFAULT_FIGURE_HEIGHT_MM
    assert args.region == figure_3_module.DEFAULT_REGIONS[0]
    assert args.light_epoch is None
    assert args.dark_epoch is None
    assert args.position_bin_count == figure_3_module.DEFAULT_POSITION_BIN_COUNT
    assert args.sigma_bins == figure_3_module.DEFAULT_SIGMA_BINS
    assert args.panel_a_cache_dir is None
    assert args.refresh_panel_a_cache is False
    assert REORDERED_HEATMAP_CMAP == supp_figure_3_module.PANEL_D_HEATMAP_CMAP
    assert REORDERED_HEATMAP_VMAX == pytest.approx(1.0)
    assert REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION == pytest.approx(0.5)


def test_panel_a_cache_path_and_roundtrip(tmp_path: Path) -> None:
    metadata = build_panel_a_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        order_mode=PANEL_A_FIGURE_1D_ORDER_MODE,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )
    panels = {}
    ordered_unit_keys = {}
    for index, order_trajectory in enumerate(figure_3_module.PANEL_B_TRAJECTORY_TYPES):
        ordered_unit_keys[order_trajectory] = np.asarray([f"unit-{index}"], dtype=str)
        for plot_trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES:
            panels[(order_trajectory, plot_trajectory)] = np.full(
                (index + 1, 3),
                index,
                dtype=float,
            )
    cache_path = build_panel_a_cache_path(tmp_path, metadata)

    assert metadata["cache_version"] == PANEL_A_CACHE_VERSION
    assert metadata["order_mode"] == PANEL_A_FIGURE_1D_ORDER_MODE
    assert cache_path.name == (
        "supplementary_figure_3_panel_a_v1_light02_r1_"
        "datasets-L14-20240611-08_r4-02_r1_orderfigure_1d_order_"
        "minlightstab0p5_posbins3_offset0_speed4_sigma1p5_cachev2.npz"
    )

    save_panel_a_cache(cache_path, panels, ordered_unit_keys, metadata)
    loaded = load_panel_a_cache(cache_path, metadata)

    assert loaded is not None
    loaded_panels, loaded_unit_keys = loaded
    for key, expected in panels.items():
        assert np.array_equal(loaded_panels[key], expected)
    for key, expected in ordered_unit_keys.items():
        assert np.array_equal(loaded_unit_keys[key], expected)

    stale_metadata = dict(metadata)
    stale_metadata["position_bin_count"] = 4
    assert load_panel_a_cache(cache_path, stale_metadata) is None


def test_group_datasets_by_animal_preserves_input_order() -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L14", "20240612", "08_r4"),
    ]

    grouped = group_datasets_by_animal(datasets)

    assert list(grouped) == ["L14", "L15"]
    assert grouped["L14"] == [
        ("L14", "20240611", "08_r4"),
        ("L14", "20240612", "08_r4"),
    ]
    assert grouped["L15"] == [("L15", "20241121", "10_r5")]


def test_filter_panel_d_similarity_table_by_tuning_stability_requires_both_epochs(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")

    similarity_table = pandas.DataFrame(
        {
            "animal_name": ["L14", "L14", "L14", "L14", "L14", "L14"],
            "date": ["20240611"] * 6,
            "unit": [1, 1, 2, 2, 3, 3],
            "comparison_label": ["left_turn"] * 6,
            "epoch_type": ["light", "dark"] * 3,
            "similarity": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )
    stability_path = supp_figure_3_module.get_stability_table_path(
        tmp_path,
        "L14",
        "20240611",
    )
    stability_path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3, 3],
            "region": ["v1", "v1", "v1", "v1", "v1", "v1", "ca1"],
            "epoch": ["02_r1", "08_r4", "02_r1", "08_r4", "02_r1", "08_r4", "08_r4"],
            "trajectory_type": [
                "right_to_center",
                "center_to_left",
                "center_to_right",
                "left_to_center",
                "not_a_trajectory",
                "right_to_center",
                "right_to_center",
            ],
            "stability_correlation": [0.51, 0.8, 0.9, 0.5, 0.9, 0.9, 0.95],
        }
    ).to_parquet(stability_path)

    filtered = filter_panel_d_similarity_table_by_tuning_stability(
        similarity_table,
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
    )

    assert STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION == pytest.approx(0.5)
    assert filtered["unit"].tolist() == [1, 1]
    assert filtered["epoch_type"].tolist() == ["light", "dark"]


def test_dark_ordered_light_heatmap_uses_figure_1d_unit_order() -> None:
    xarray = pytest.importorskip("xarray")
    position = np.asarray([0.0, 0.5, 1.0], dtype=float)
    units = np.asarray([1, 2], dtype=int)
    light_values = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=float,
    )

    def _curve(values: np.ndarray):
        return xarray.DataArray(
            values,
            dims=("unit", "position"),
            coords={"unit": units, "position": position},
        )

    ordered_unit_keys_by_trajectory = {
        trajectory: np.asarray(
            ["L14:20240611:v1:2", "L14:20240611:v1:1"],
            dtype=object,
        )
        for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
    }
    light_curve_sets = [
        {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "all_curves": {
                trajectory: _curve(light_values)
                for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
            },
        }
    ]

    panels = build_figure_1d_ordered_light_panel_values(
        ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
        light_curve_sets=light_curve_sets,
        position_bin_count=3,
    )

    assert list(dict.fromkeys(key[0] for key in panels)) == list(
        figure_3_module.PANEL_B_TRAJECTORY_TYPES
    )
    assert list(dict.fromkeys(key[1] for key in panels)) == list(
        figure_3_module.PANEL_B_TRAJECTORY_TYPES
    )
    panel = panels[("center_to_left", "center_to_left")]
    assert np.allclose(
        panel,
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_filter_ordered_unit_keys_preserves_trajectory_order() -> None:
    ordered_unit_keys_by_trajectory = {
        trajectory: np.asarray(
            [
                "L14:20240611:v1:3",
                "L14:20240611:v1:1",
                "L14:20240611:v1:2",
            ],
            dtype=object,
        )
        for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
    }

    filtered = filter_ordered_unit_keys_by_unit_set(
        ordered_unit_keys_by_trajectory,
        {"L14:20240611:v1:1", "L14:20240611:v1:2"},
    )

    for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES:
        assert np.array_equal(
            filtered[trajectory],
            np.asarray(["L14:20240611:v1:1", "L14:20240611:v1:2"], dtype=object),
        )


def test_dark_ordered_light_heatmap_loads_task_progression_curves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_calls = []
    light_calls = []
    stability_calls = []
    build_calls = []

    def fake_load_or_compute_panel_d_heatmap_payload(**kwargs: object):
        cache_calls.append(kwargs)
        return {}, {
            trajectory: np.asarray(
                ["L14:20240611:v1:1", "L14:20240611:v1:2"],
                dtype=object,
            )
            for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
        }

    def fake_compute_light_epoch_all_trial_tuning_curves(**kwargs: object):
        light_calls.append(kwargs)
        return {"animal_name": "L14", "date": "20240611", "region": "v1"}

    def fake_select_unit_keys_by_light_tuning_stability(**kwargs: object):
        stability_calls.append(kwargs)
        return {"L14:20240611:v1:2"}

    def fake_build_panel_a_heatmap_payloads(**kwargs: object):
        build_calls.append(kwargs)
        panels = {
            (order_trajectory, plot_trajectory): np.ones((1, 2), dtype=float)
            for order_trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
            for plot_trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
        }
        ordered_unit_keys = kwargs["figure_1d_ordered_unit_keys_by_trajectory"]
        return {
            PANEL_A_FIGURE_1D_ORDER_MODE: (panels, ordered_unit_keys),
        }

    monkeypatch.setattr(
        supp_figure_3_module,
        "load_or_compute_panel_d_heatmap_payload",
        fake_load_or_compute_panel_d_heatmap_payload,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "compute_light_epoch_all_trial_tuning_curves",
        fake_compute_light_epoch_all_trial_tuning_curves,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "select_unit_keys_by_light_tuning_stability",
        fake_select_unit_keys_by_light_tuning_stability,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "build_panel_a_heatmap_payloads",
        fake_build_panel_a_heatmap_payloads,
    )

    panels = load_dark_ordered_light_panel_values(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        figure_1d_cache_dir=Path("/cache"),
        panel_a_cache_dir=tmp_path,
    )

    assert set(panels) == {PANEL_A_FIGURE_1D_ORDER_MODE}
    assert cache_calls[0]["panel_d_cache_dir"] == Path("/cache")
    assert cache_calls[0]["require_ordered_unit_keys"] is True
    assert cache_calls[0]["datasets"] == [("L14", "20240611", "08_r4")]
    assert light_calls[0]["use_trajectory_direction"] is True
    assert stability_calls[0]["datasets"] == [("L14", "20240611", "08_r4")]
    assert stability_calls[0]["light_epoch"] is None
    filtered_order = build_calls[0]["figure_1d_ordered_unit_keys_by_trajectory"]
    assert np.array_equal(
        filtered_order["center_to_left"],
        np.asarray(["L14:20240611:v1:2"], dtype=object),
    )


def test_plot_dark_ordered_light_heatmap_uses_panel_b_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        supp_figure_3_module,
        "load_dark_ordered_light_panel_values",
        lambda **_kwargs: {
            PANEL_A_FIGURE_1D_ORDER_MODE: {},
        },
    )
    plot_calls = []

    def fake_plot_pooled_heatmap_grid(_axes, _panels, **kwargs: object):
        plot_calls.append(kwargs)
        return None

    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_pooled_heatmap_grid",
        fake_plot_pooled_heatmap_grid,
    )

    fig, axes = plt.subplots(nrows=4, ncols=4)
    plot_dark_ordered_light_heatmap_regions(
        np.asarray(axes, dtype=object),
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )

    assert plot_calls[0]["trajectory_types"] == figure_3_module.PANEL_B_TRAJECTORY_TYPES
    assert (
        plot_calls[0]["axis_orientation"]
        == figure_3_module.PANEL_B_LINEAR_POSITION_ORIENTATION
    )
    assert plot_calls[0]["cmap"] == REORDERED_HEATMAP_CMAP
    plt.close(fig)


def test_reordered_heatmap_display_style_uses_figure_1d_colormap() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2)
    for ax in axes:
        ax.imshow(np.asarray([[0.0, 0.5, 1.0]], dtype=float), vmin=0.0, vmax=1.0)

    set_heatmap_display_style(
        np.asarray(axes, dtype=object),
        cmap=REORDERED_HEATMAP_CMAP,
        vmin=0.0,
        vmax=REORDERED_HEATMAP_VMAX,
    )

    for ax in axes:
        assert ax.images[0].get_cmap().name == REORDERED_HEATMAP_CMAP
        assert ax.images[0].get_clim() == pytest.approx((0.0, 1.0))
    plt.close(fig)


def test_dark_light_tuning_correlation_filters_by_movement_rate() -> None:
    xarray = pytest.importorskip("xarray")
    position = np.asarray([0.0, 0.5, 1.0], dtype=float)
    units = np.asarray([1, 2, 3], dtype=int)

    def _curve(values: list[list[float]]):
        return xarray.DataArray(
            np.asarray(values, dtype=float),
            dims=("unit", "position"),
            coords={"unit": units, "position": position},
        )

    dark_curve_sets = [
        {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "08_r4",
            "movement_firing_rates_hz": {1: 0.5, 2: 0.49, 3: 0.7},
            "all_curves": {
                trajectory: _curve(
                    [
                        [0.0, 1.0, 2.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 1.0, 0.0],
                    ]
                )
                for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
            },
        }
    ]
    light_curve_sets = [
        {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "02_r1",
            "movement_firing_rates_hz": {1: 0.5, 2: 0.8, 3: 0.4},
            "all_curves": {
                trajectory: _curve(
                    [
                        [0.0, 1.0, 2.0],
                        [3.0, 2.0, 1.0],
                        [2.0, 1.0, 0.0],
                    ]
                )
                for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES
            },
        }
    ]

    table = build_dark_light_tuning_correlation_table(
        dark_curve_sets,
        light_curve_sets,
    )

    assert DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ == pytest.approx(0.5)
    assert table["unit"].tolist() == [1] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    assert table["trajectory_type"].tolist() == list(
        figure_3_module.PANEL_B_TRAJECTORY_TYPES
    )
    assert table["correlation"].tolist() == pytest.approx(
        [1.0] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    )
    assert table["dark_movement_firing_rate_hz"].tolist() == pytest.approx(
        [0.5] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    )
    assert table["light_movement_firing_rate_hz"].tolist() == pytest.approx(
        [0.5] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    )


def test_load_dark_light_tuning_correlation_table_reads_saved_artifacts(
    tmp_path: Path,
) -> None:
    xarray = pytest.importorskip("xarray")
    position = np.asarray([0.0, 0.5, 1.0], dtype=float)
    units = np.asarray([1, 2], dtype=int)

    def _write_curve(epoch: str, trajectory: str, values: list[list[float]]) -> None:
        path = supp_figure_3_module.get_saved_task_progression_tuning_curve_path(
            tmp_path,
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch=epoch,
            trajectory_type=trajectory,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        xarray.Dataset(
            {
                "firing_rate_hz": (
                    ("unit", "linpos"),
                    np.asarray(values, dtype=float),
                )
            },
            coords={"unit": units, "linpos": position},
        ).to_netcdf(path)

    for trajectory in figure_3_module.PANEL_B_TRAJECTORY_TYPES:
        _write_curve("08_r4", trajectory, [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        _write_curve("02_r1", trajectory, [[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]])

    rate_path = supp_figure_3_module.get_dark_light_movement_rate_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
    )
    rate_path.parent.mkdir(parents=True, exist_ok=True)
    xarray.Dataset(
        {
            "dark_movement_firing_rate_hz": ("unit", np.asarray([0.5, 0.49])),
            "light_movement_firing_rate_hz": ("unit", np.asarray([0.5, 0.8])),
        },
        coords={"unit": units},
    ).to_netcdf(rate_path)

    table = load_dark_light_tuning_correlation_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
    )

    assert table["unit"].tolist() == [1] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    assert table["correlation"].tolist() == pytest.approx(
        [1.0] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    )


def test_load_light_tuning_stability_table_filters_light_epoch(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")

    stability_path = supp_figure_3_module.get_stability_table_path(
        tmp_path,
        "L14",
        "20240611",
    )
    stability_path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(
        {
            "unit": [1, 2, 3, 4, 5],
            "region": ["v1", "v1", "v1", "ca1", "v1"],
            "epoch": ["02_r1", "02_r1", "08_r4", "02_r1", "02_r1"],
            "trajectory_type": [
                "center_to_left",
                "right_to_center",
                "center_to_left",
                "center_to_left",
                "not_a_trajectory",
            ],
            "stability_correlation": [0.2, 0.6, 0.7, 0.8, 0.9],
        }
    ).to_parquet(stability_path)

    table = load_light_tuning_stability_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch="02_r1",
    )

    assert table["unit"].tolist() == [1, 2]
    assert table["trajectory_type"].tolist() == ["center_to_left", "right_to_center"]
    assert table["light_epoch"].tolist() == ["02_r1", "02_r1"]
    assert table["stability_correlation"].tolist() == pytest.approx([0.2, 0.6])


def test_compute_tuning_curve_correlation_rejects_flat_curves() -> None:
    assert compute_tuning_curve_correlation(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([2.0, 1.0, 0.0]),
    ) == pytest.approx(-1.0)
    assert np.isnan(
        compute_tuning_curve_correlation(
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([0.0, 1.0, 2.0]),
        )
    )


def test_plot_dark_light_tuning_correlation_histograms_draws_four_routes() -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for index, trajectory in enumerate(figure_3_module.PANEL_B_TRAJECTORY_TYPES):
        rows.append(
            {
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "dark_epoch": "08_r4",
                "light_epoch": "02_r1",
                "trajectory_type": trajectory,
                "unit": index + 1,
                "dark_movement_firing_rate_hz": 0.5,
                "light_movement_firing_rate_hz": 0.5,
                "correlation": 0.1 * index,
            }
        )
    fig, axes = plt.subplots(ncols=4)

    plot_dark_light_tuning_correlation_histograms(
        axes,
        pandas.DataFrame(rows),
    )

    assert axes[0].get_ylabel() == "Frac."
    assert all(axis.get_xlim() == pytest.approx((-1.0, 1.0)) for axis in axes)
    assert all(axis.patches for axis in axes)
    assert all("n = 1" in axis.texts[0].get_text() for axis in axes)
    plt.close(fig)


def test_plot_dark_light_with_light_stability_histograms_overlays_step() -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correlation_rows = []
    stability_rows = []
    for index, trajectory in enumerate(figure_3_module.PANEL_B_TRAJECTORY_TYPES):
        correlation_rows.append(
            {
                "trajectory_type": trajectory,
                "correlation": -0.2 + 0.1 * index,
            }
        )
        stability_rows.append(
            {
                "trajectory_type": trajectory,
                "stability_correlation": 0.3 + 0.1 * index,
            }
        )
    fig, axes = plt.subplots(ncols=4)

    plot_dark_light_with_light_stability_histograms(
        axes,
        pandas.DataFrame(correlation_rows),
        pandas.DataFrame(stability_rows),
    )

    assert axes[0].get_xlabel() == "Correlation"
    assert axes[0].get_ylabel() == "Frac."
    assert axes[0].get_legend() is not None
    assert [text.get_text() for text in axes[0].get_legend().texts] == [
        "Dark-light",
        "Light odd/even",
    ]
    assert all(axis.patches for axis in axes)
    assert all(axis.get_xlim() == pytest.approx((-1.0, 1.0)) for axis in axes)
    plt.close(fig)


def test_make_supplementary_figure_3_plots_per_animal_figure_3def(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}
    similarity_calls = []
    encoding_calls = []
    decoding_calls = []
    heatmap_calls = []
    dark_light_correlation_calls = []
    light_stability_calls = []
    overlay_plot_calls = []
    stability_filter_calls = []

    def fake_load_panel_d_similarity_table(**kwargs: object):
        similarity_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "unit": [1],
                "comparison_label": ["left_turn"],
                "epoch_type": ["light"],
                "similarity": [0.5],
            }
        )

    def fake_load_panel_e_encoding_delta_table(**kwargs: object):
        encoding_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "unit": [1],
                "epoch_type": ["light"],
                "delta_bits_tp_vs_place": [0.2],
            }
        )

    def fake_load_panel_f_decoding_error_table(**kwargs: object):
        decoding_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "epoch_type": ["light"],
                "epoch": ["02_r1"],
                "analysis": ["place"],
                "comparison": ["place"],
                "comparison_label": ["Place"],
                "q25_error": [0.01],
                "median_error": [0.02],
                "q75_error": [0.03],
                "n_samples": [20],
            }
        )

    def fake_plot_panel_d_similarity(ax, table):
        calls.setdefault("similarity_tables", []).append(table)
        ax.text(0.5, 0.5, "D")

    def fake_plot_panel_e_encoding_delta_histogram(ax, table):
        calls.setdefault("encoding_tables", []).append(table)
        ax.text(0.5, 0.5, "E")

    def fake_plot_panel_f_decoding_error(ax, table):
        calls.setdefault("decoding_tables", []).append(table)
        ax.text(0.5, 0.5, "F")

    def fake_plot_dark_ordered_light_heatmap_regions(_axes, **kwargs: object):
        heatmap_calls.append(kwargs)
        return None

    def fake_load_dark_light_tuning_correlation_table(**kwargs: object):
        dark_light_correlation_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "region": ["v1"],
                "dark_epoch": ["08_r4"],
                "light_epoch": ["02_r1"],
                "trajectory_type": ["center_to_left"],
                "unit": [1],
                "dark_movement_firing_rate_hz": [0.5],
                "light_movement_firing_rate_hz": [0.5],
                "correlation": [0.4],
            }
        )

    def fake_plot_dark_light_tuning_correlation_histograms(axes, table):
        calls["dark_light_correlation_table"] = table
        axes[0].text(0.5, 0.5, "E")

    def fake_load_light_tuning_stability_table(**kwargs: object):
        light_stability_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "region": ["v1"],
                "light_epoch": ["02_r1"],
                "trajectory_type": ["center_to_left"],
                "unit": [1],
                "stability_correlation": [0.6],
            }
        )

    def fake_plot_dark_light_with_light_stability_histograms(
        axes,
        correlation_table,
        stability_table,
    ):
        overlay_plot_calls.append(
            {
                "correlation_table": correlation_table,
                "stability_table": stability_table,
            }
        )
        axes[0].text(0.5, 0.5, "G")

    def fake_filter_panel_d_similarity_table_by_tuning_stability(table, **kwargs):
        stability_filter_calls.append({"table": table, **kwargs})
        return table.assign(stability_filtered=True)

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

    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_d_similarity_table",
        fake_load_panel_d_similarity_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_e_encoding_delta_table",
        fake_load_panel_e_encoding_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_f_decoding_error_table",
        fake_load_panel_f_decoding_error_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_d_similarity",
        fake_plot_panel_d_similarity,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_e_encoding_delta_histogram",
        fake_plot_panel_e_encoding_delta_histogram,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_f_decoding_error",
        fake_plot_panel_f_decoding_error,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_dark_ordered_light_heatmap_regions",
        fake_plot_dark_ordered_light_heatmap_regions,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_dark_light_tuning_correlation_table",
        fake_load_dark_light_tuning_correlation_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_dark_light_tuning_correlation_histograms",
        fake_plot_dark_light_tuning_correlation_histograms,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_light_tuning_stability_table",
        fake_load_light_tuning_stability_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_dark_light_with_light_stability_histograms",
        fake_plot_dark_light_with_light_stability_histograms,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "filter_panel_d_similarity_table_by_tuning_stability",
        fake_filter_panel_d_similarity_table_by_tuning_stability,
    )
    monkeypatch.setattr(supp_figure_3_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_3.svg"
    datasets = [("L14", "20240611", "08_r4"), ("L15", "20241121", "10_r5")]
    saved_path = make_supplementary_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["A", "B", "C", "D", "E", "F", "G"]
    assert len(heatmap_calls) == 1
    assert heatmap_calls[0]["order_mode"] == PANEL_A_FIGURE_1D_ORDER_MODE
    assert all(call["datasets"] == datasets for call in heatmap_calls)
    assert all(call["regions"] == ("v1",) for call in heatmap_calls)
    assert all(
        call["position_bin_count"] == figure_3_module.DEFAULT_POSITION_BIN_COUNT
        for call in heatmap_calls
    )
    assert [call["datasets"] for call in similarity_calls] == [[datasets[0]], [datasets[1]]]
    assert [call["datasets"] for call in encoding_calls] == [[datasets[0]], [datasets[1]]]
    assert [call["datasets"] for call in decoding_calls] == [[datasets[0]], [datasets[1]]]
    assert [call["datasets"] for call in dark_light_correlation_calls] == [datasets]
    assert [call["datasets"] for call in light_stability_calls] == [datasets]
    assert [call["datasets"] for call in stability_filter_calls] == [
        [datasets[0]],
        [datasets[1]],
    ]
    assert calls["dark_light_correlation_table"]["correlation"].tolist() == [0.4]
    assert overlay_plot_calls[0]["correlation_table"]["correlation"].tolist() == [0.4]
    assert overlay_plot_calls[0]["stability_table"]["stability_correlation"].tolist() == [
        0.6
    ]
    assert all(call["region"] == "v1" for call in similarity_calls)
    assert len(calls["similarity_tables"]) == 4
    similarity_stability_flags = [
        bool(table["stability_filtered"].any())
        if "stability_filtered" in table
        else False
        for table in calls["similarity_tables"]
    ]
    assert similarity_stability_flags == [
        False,
        True,
        False,
        True,
    ]
    assert calls["encoding_tables"]
    assert calls["decoding_tables"]
