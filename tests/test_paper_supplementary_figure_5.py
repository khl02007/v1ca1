from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures._dark_light as dark_light_module
import v1ca1.paper_figures.figure_1 as figure_1_module
import v1ca1.paper_figures.supplementary_figure_5 as supp_figure_3_module
from v1ca1.paper_figures.supplementary_figure_5 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_BOTTOM_SECTION_SPACER_MM,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_MOTOR_GRID_HEIGHT_MM,
    DEFAULT_MOTOR_SUMMARY_HEIGHT_MM,
    DEFAULT_REORDERED_HEATMAP_HEIGHT_MM,
    DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ,
    MOTOR_PANEL_ANIMAL_NAME,
    MOTOR_PANEL_LIGHT_EPOCH,
    MOTOR_SUMMARY_ANIMAL_COLORS,
    MOTOR_VARIABLES,
    PANEL_A_CACHE_VERSION,
    PANEL_A_CV_PCA_LIGHT_EPOCH,
    PANEL_A_CV_PCA_REGION,
    PANEL_A_CV_PCA_SIZE_FRACTION,
    PANEL_A_CV_PCA_TITLE,
    PANEL_A_FIGURE_1D_ORDER_MODE,
    REORDERED_HEATMAP_CMAP,
    REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION,
    REORDERED_HEATMAP_VMAX,
    STABILITY_FILTERED_SIMILARITY_MIN_CORRELATION,
    build_dark_light_tuning_correlation_table,
    build_figure_1d_ordered_light_panel_values,
    build_panel_a_cache_metadata,
    build_panel_a_cache_path,
    build_panel_a_cv_pca_summary_path,
    build_panel_c_motor_profile_correlation_table,
    build_motor_progression_summary_path,
    compute_motor_profile_correlation,
    compute_tuning_curve_correlation,
    filter_panel_d_similarity_table_by_tuning_stability,
    filter_ordered_unit_keys_by_unit_set,
    load_dark_light_tuning_correlation_table,
    load_light_tuning_stability_table,
    load_panel_a_cache,
    load_panel_a_cv_pca_participation_ratio_table,
    load_panel_b_motor_progression_table,
    load_dark_ordered_light_panel_values,
    make_supplementary_figure_5,
    parse_arguments,
    plot_panel_a_cv_pca_participation_ratios,
    plot_panel_b_motor_progression_grid,
    plot_panel_c_motor_profile_correlations,
    plot_dark_ordered_light_heatmap_regions,
    plot_dark_light_tuning_correlation_histograms,
    plot_dark_light_with_light_stability_histograms,
    save_panel_a_cache,
    set_heatmap_display_style,
)


def test_default_cli_keeps_only_motor_control_panels() -> None:
    args = parse_arguments([])

    assert args.output_dir == supp_figure_3_module.DEFAULT_OUTPUT_DIR
    assert DEFAULT_OUTPUT_NAME == "supplementary_figure_5"
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert args.output_format == supp_figure_3_module.DEFAULT_OUTPUT_FORMAT
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        figure_1_module.DEFAULT_FIGURE_WIDTH_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        DEFAULT_MOTOR_GRID_HEIGHT_MM
        + DEFAULT_BOTTOM_SECTION_SPACER_MM
        + DEFAULT_MOTOR_SUMMARY_HEIGHT_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(178.1)
    assert not hasattr(args, "region")
    assert not hasattr(args, "light_epoch")
    assert not hasattr(args, "dark_epoch")
    assert not hasattr(args, "position_bin_count")
    assert not hasattr(args, "panel_a_cache_dir")
    assert not hasattr(args, "refresh_panel_a_cache")
    assert REORDERED_HEATMAP_CMAP == supp_figure_3_module.PANEL_D_HEATMAP_CMAP
    assert REORDERED_HEATMAP_VMAX == pytest.approx(1.0)
    assert REORDERED_HEATMAP_MIN_LIGHT_STABILITY_CORRELATION == pytest.approx(0.5)
    assert PANEL_A_CV_PCA_SIZE_FRACTION == pytest.approx(0.40)
    assert DEFAULT_REORDERED_HEATMAP_HEIGHT_MM == pytest.approx(
        figure_1_module.DEFAULT_HEATMAP_HEIGHT_MM * PANEL_A_CV_PCA_SIZE_FRACTION
    )


def test_main_builds_named_output_and_forwards_relevant_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_make_supplementary_figure_5(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        supp_figure_3_module,
        "make_supplementary_figure_5",
        fake_make_supplementary_figure_5,
    )
    supp_figure_3_module.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            str(tmp_path),
            "--format",
            "svg",
            "--dataset",
            "L14:20240611:08_r4",
            "--dpi",
            "144",
        ]
    )

    assert calls == {
        "data_root": Path("/analysis"),
        "output_path": tmp_path / "supplementary_figure_5.svg",
        "datasets": [("L14", "20240611", "08_r4")],
        "dpi": 144,
    }


def test_canonical_supplementary_figure_5_is_promoted_and_independent() -> None:
    code = (
        "import importlib.util; "
        "from v1ca1.paper_figures import supplementary_figure_5; "
        "assert supplementary_figure_5.DEFAULT_OUTPUT_NAME == "
        "'supplementary_figure_5'; "
        "assert supplementary_figure_5.DEFAULT_FIGURE_HEIGHT_MM == 178.1; "
        "assert importlib.util.find_spec("
        "'v1ca1.paper_figures.supplementary_figure_5_3') is None"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
    for index, order_trajectory in enumerate(dark_light_module.PANEL_B_TRAJECTORY_TYPES):
        ordered_unit_keys[order_trajectory] = np.asarray([f"unit-{index}"], dtype=str)
        for plot_trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES:
            panels[(order_trajectory, plot_trajectory)] = np.full(
                (index + 1, 3),
                index,
                dtype=float,
            )
    cache_path = build_panel_a_cache_path(tmp_path, metadata)

    assert metadata["cache_version"] == PANEL_A_CACHE_VERSION
    assert metadata["order_mode"] == PANEL_A_FIGURE_1D_ORDER_MODE
    assert cache_path.name == (
        "supplementary_figure_5_panel_a_v1_light02_r1_"
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


def test_panel_a_cv_pca_loader_reads_hardcoded_v1_light_summary(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    summary_path = build_panel_a_cv_pca_summary_path(
        data_root=tmp_path,
        animal_name="L14",
        date="20240611",
        dark_epoch="08_r4",
    )
    summary_path.parent.mkdir(parents=True)
    pandas.DataFrame(
        {
            "animal_name": ["L14", "L14", "L14", "L14"],
            "date": ["20240611"] * 4,
            "region": [PANEL_A_CV_PCA_REGION] * 4,
            "dark_epoch": ["08_r4"] * 4,
            "light_epoch": [PANEL_A_CV_PCA_LIGHT_EPOCH] * 4,
            "source_condition": ["dark", "dark", "light", "light"],
            "target_condition": ["dark", "light", "dark", "light"],
            "n_units": [37] * 4,
            "source_cv_participation_ratio": [5.0, 5.0, 8.0, 8.0],
        }
    ).to_parquet(summary_path, index=False)

    table = load_panel_a_cv_pca_participation_ratio_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
    )

    assert summary_path.name == "v1_02_r1_vs_08_r4_cv_pca_summary.parquet"
    assert table["condition"].tolist() == ["dark", "light"]
    assert table["participation_ratio"].tolist() == [5.0, 8.0]
    assert table["n_units"].tolist() == [37, 37]
    assert table["source_path"].tolist() == [summary_path, summary_path]


def test_panel_a_cv_pca_loader_raises_for_missing_summary(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Missing Supplementary Figure 5A cvPCA summary files",
    ):
        load_panel_a_cv_pca_participation_ratio_table(
            data_root=tmp_path,
            datasets=[("L14", "20240611", "08_r4")],
        )


def test_plot_panel_a_cv_pca_participation_ratios_shows_paired_sessions() -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt

    table = pandas.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15"],
            "date": ["20240611", "20240611", "20241121", "20241121"],
            "condition": ["dark", "light", "dark", "light"],
            "participation_ratio": [4.0, 6.0, 5.0, 7.0],
        }
    )
    fig, ax = plt.subplots()

    plot_panel_a_cv_pca_participation_ratios(ax, table)

    assert ax.get_title() == PANEL_A_CV_PCA_TITLE
    assert ax.get_ylabel() == "Participation ratio"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Dark", "Light"]
    assert len(ax.lines) == 2
    assert len(ax.collections) == 2
    assert np.allclose(
        ax.collections[0].get_facecolors(),
        np.asarray([to_rgba(MOTOR_SUMMARY_ANIMAL_COLORS["L14"])] * 2),
    )
    assert np.allclose(
        ax.collections[1].get_facecolors(),
        np.asarray([to_rgba(MOTOR_SUMMARY_ANIMAL_COLORS["L15"])] * 2),
    )
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["L14", "L15"]
    assert legend._loc == 6
    y_min, y_max = ax.get_ylim()
    assert 0.0 < y_min < 4.0
    assert y_max > 7.0
    plt.close(fig)


def _make_motor_progression_rows(
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    dark_epoch: str = "08_r4",
    light_epoch: str = MOTOR_PANEL_LIGHT_EPOCH,
) -> list[dict[str, object]]:
    rows = []
    for epoch_index, epoch in enumerate((dark_epoch, light_epoch)):
        for trajectory_index, trajectory_type in enumerate(
            dark_light_module.PANEL_B_TRAJECTORY_TYPES
        ):
            for variable_index, variable_name in enumerate(MOTOR_VARIABLES):
                for bin_index, bin_center in enumerate((0.25, 0.75)):
                    rows.append(
                        {
                            "epoch": epoch,
                            "trajectory_type": trajectory_type,
                            "variable": variable_name,
                            "progression_bin_index": bin_index,
                            "progression_bin_start": bin_index * 0.5,
                            "progression_bin_end": (bin_index + 1) * 0.5,
                            "progression_bin_center": bin_center,
                            "sample_count": 10,
                            "median": (
                                variable_index
                                + trajectory_index * 0.1
                                + epoch_index * 0.2
                                + bin_index * 0.05
                            ),
                            "q25": 0.0,
                            "q75": 1.0,
                        }
                    )
    return rows


def test_panel_b_motor_loader_reads_light_and_registered_dark(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    path = build_motor_progression_summary_path(
        data_root=tmp_path,
        animal_name="L14",
        date="20240611",
    )
    path.parent.mkdir(parents=True)
    pandas.DataFrame(_make_motor_progression_rows()).to_parquet(path, index=False)

    table = load_panel_b_motor_progression_table(
        data_root=tmp_path,
        datasets=[("L14", "20240611", "08_r4")],
    )

    assert set(table["epoch"].astype(str)) == {"08_r4", MOTOR_PANEL_LIGHT_EPOCH}
    assert set(table["epoch_type"].astype(str)) == {"dark", "light"}
    assert table["animal_name"].unique().tolist() == ["L14"]
    assert table["source_path"].unique().tolist() == [path]


def test_plot_panel_b_motor_progression_grid_uses_example_epoch_medians_and_iqr() -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for animal_name, date, dark_epoch in (
        (MOTOR_PANEL_ANIMAL_NAME, "20240222", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ):
        for row in _make_motor_progression_rows(
            animal_name=animal_name,
            date=date,
            dark_epoch=dark_epoch,
        ):
            row["animal_name"] = animal_name
            row["date"] = date
            row["dark_epoch"] = dark_epoch
            row["light_epoch"] = MOTOR_PANEL_LIGHT_EPOCH
            row["dataset_label"] = f"{animal_name} {date}"
            row["epoch_type"] = "dark" if row["epoch"] == dark_epoch else "light"
            rows.append(row)
    table = pandas.DataFrame(rows)
    fig, axes = plt.subplots(
        nrows=len(MOTOR_VARIABLES),
        ncols=len(dark_light_module.PANEL_B_TRAJECTORY_TYPES),
    )

    plot_panel_b_motor_progression_grid(
        axes,
        table,
        datasets=[
            (MOTOR_PANEL_ANIMAL_NAME, "20240222", "08_r4"),
            ("L15", "20241121", "10_r5"),
        ],
    )

    assert axes[0, 0].get_title() != ""
    assert axes[-1, 0].get_xlabel() == "Norm. position"
    assert axes[0, 0].get_ylabel()
    assert len(axes[0, 0].lines) == 2
    assert len(axes[0, 0].collections) == 2
    assert all(line.get_linestyle() == "-" for line in axes[0, 0].lines)
    assert axes[0, 0].get_legend() is not None
    assert axes[0, -1].get_legend() is None
    plt.close(fig)


def test_build_panel_c_motor_profile_correlation_table_uses_paired_bins() -> None:
    pandas = pytest.importorskip("pandas")

    rows = []
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    for animal_name, date, dark_epoch in datasets:
        for row in _make_motor_progression_rows(
            animal_name=animal_name,
            date=date,
            dark_epoch=dark_epoch,
        ):
            row["animal_name"] = animal_name
            row["date"] = date
            row["dark_epoch"] = dark_epoch
            row["light_epoch"] = MOTOR_PANEL_LIGHT_EPOCH
            row["epoch_type"] = "dark" if row["epoch"] == dark_epoch else "light"
            rows.append(row)

    table = build_panel_c_motor_profile_correlation_table(
        pandas.DataFrame(rows),
        datasets=datasets,
    )

    assert len(table) == (
        len(datasets)
        * len(MOTOR_VARIABLES)
        * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    )
    assert set(table["animal_name"]) == {"L14", "L15"}
    assert table["n_bins"].unique().tolist() == [2]
    assert np.allclose(table["correlation"].to_numpy(dtype=float), 1.0)
    assert compute_motor_profile_correlation(
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
    ) == pytest.approx(-1.0)


def test_plot_panel_c_motor_profile_correlations_draws_animal_dots() -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    rows = []
    for animal_index, (animal_name, date, dark_epoch) in enumerate(datasets):
        for trajectory_type in dark_light_module.PANEL_B_TRAJECTORY_TYPES:
            for variable_index, variable_name in enumerate(MOTOR_VARIABLES):
                rows.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "dark_epoch": dark_epoch,
                        "light_epoch": MOTOR_PANEL_LIGHT_EPOCH,
                        "trajectory_type": trajectory_type,
                        "variable": variable_name,
                        "correlation": 0.8 + animal_index * 0.05 - variable_index * 0.01,
                        "n_bins": 20,
                    }
                )
    fig, axes = plt.subplots(ncols=len(dark_light_module.PANEL_B_TRAJECTORY_TYPES))

    plot_panel_c_motor_profile_correlations(
        axes,
        pandas.DataFrame(rows),
        datasets=datasets,
    )

    assert axes[0].get_title() != ""
    assert axes[0].get_xlabel() == "Profile corr."
    assert axes[0].get_ylabel() == "Motor variable"
    assert axes[0].get_xlim() == pytest.approx((-1.05, 1.05))
    assert axes[-1].get_legend() is not None
    assert any(axis.collections for axis in axes)
    plt.close(fig)


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
        for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
    }
    light_curve_sets = [
        {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "all_curves": {
                trajectory: _curve(light_values)
                for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
            },
        }
    ]

    panels = build_figure_1d_ordered_light_panel_values(
        ordered_unit_keys_by_trajectory=ordered_unit_keys_by_trajectory,
        light_curve_sets=light_curve_sets,
        position_bin_count=3,
    )

    assert list(dict.fromkeys(key[0] for key in panels)) == list(
        dark_light_module.PANEL_B_TRAJECTORY_TYPES
    )
    assert list(dict.fromkeys(key[1] for key in panels)) == list(
        dark_light_module.PANEL_B_TRAJECTORY_TYPES
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
        for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
    }

    filtered = filter_ordered_unit_keys_by_unit_set(
        ordered_unit_keys_by_trajectory,
        {"L14:20240611:v1:1", "L14:20240611:v1:2"},
    )

    for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES:
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
            for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
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
            for order_trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
            for plot_trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
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

    assert plot_calls[0]["trajectory_types"] == dark_light_module.PANEL_B_TRAJECTORY_TYPES
    assert (
        plot_calls[0]["axis_orientation"]
        == dark_light_module.PANEL_B_LINEAR_POSITION_ORIENTATION
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
                for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
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
                for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES
            },
        }
    ]

    table = build_dark_light_tuning_correlation_table(
        dark_curve_sets,
        light_curve_sets,
    )

    assert DARK_LIGHT_CORRELATION_MIN_MOVEMENT_FIRING_RATE_HZ == pytest.approx(0.5)
    assert table["unit"].tolist() == [1] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    assert table["trajectory_type"].tolist() == list(
        dark_light_module.PANEL_B_TRAJECTORY_TYPES
    )
    assert table["correlation"].tolist() == pytest.approx(
        [1.0] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    )
    assert table["dark_movement_firing_rate_hz"].tolist() == pytest.approx(
        [0.5] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    )
    assert table["light_movement_firing_rate_hz"].tolist() == pytest.approx(
        [0.5] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
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

    for trajectory in dark_light_module.PANEL_B_TRAJECTORY_TYPES:
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

    assert table["unit"].tolist() == [1] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    assert table["correlation"].tolist() == pytest.approx(
        [1.0] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
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
    for index, trajectory in enumerate(dark_light_module.PANEL_B_TRAJECTORY_TYPES):
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
    for index, trajectory in enumerate(dark_light_module.PANEL_B_TRAJECTORY_TYPES):
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


def test_make_supplementary_figure_5_relabels_motor_control_panels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}
    motor_load_calls = []
    motor_plot_calls = []
    motor_summary_build_calls = []
    motor_summary_plot_calls = []

    def fail_if_panel_a_is_used(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Supplementary Figure 5 used the omitted cvPCA panel")

    def fake_load_panel_b_motor_progression_table(**kwargs: object):
        motor_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14", "L14"],
                "date": ["20240611", "20240611"],
                "trajectory_type": [dark_light_module.PANEL_B_TRAJECTORY_TYPES[0]] * 2,
                "variable": [MOTOR_VARIABLES[0]] * 2,
                "epoch": ["08_r4", MOTOR_PANEL_LIGHT_EPOCH],
                "epoch_type": ["dark", "light"],
                "progression_bin_index": [0, 0],
                "progression_bin_center": [0.5, 0.5],
                "median": [1.0, 1.1],
                "q25": [0.8, 0.9],
                "q75": [1.2, 1.3],
            }
        )

    def fake_plot_panel_b_motor_progression_grid(axes, table, **kwargs: object):
        motor_plot_calls.append({"axes": axes, "table": table, **kwargs})
        axes[0, 0].text(0.5, 0.5, "B")

    def fake_build_panel_c_motor_profile_correlation_table(table, **kwargs: object):
        motor_summary_build_calls.append({"table": table, **kwargs})
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "dark_epoch": ["08_r4"],
                "light_epoch": [MOTOR_PANEL_LIGHT_EPOCH],
                "trajectory_type": ["center_to_left"],
                "variable": [MOTOR_VARIABLES[0]],
                "correlation": [0.95],
                "n_bins": [20],
            }
        )

    def fake_plot_panel_c_motor_profile_correlations(axes, table, **kwargs: object):
        motor_summary_plot_calls.append({"axes": axes, "table": table, **kwargs})
        axes[0].text(0.5, 0.5, "E")

    def fake_save_figure(figure, output_path: Path, dpi: int, **kwargs: object):
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        calls["figure_titles"] = [text.get_text() for text in figure.texts]
        calls["axis_count"] = len(figure.axes)
        calls["axis_off_count"] = sum(not ax.axison for ax in figure.axes)
        return output_path

    monkeypatch.setattr(supp_figure_3_module, "apply_paper_style", lambda: None)
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_a_cv_pca_participation_ratio_table",
        fail_if_panel_a_is_used,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_a_cv_pca_participation_ratios",
        fail_if_panel_a_is_used,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_b_motor_progression_table",
        fake_load_panel_b_motor_progression_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_b_motor_progression_grid",
        fake_plot_panel_b_motor_progression_grid,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "build_panel_c_motor_profile_correlation_table",
        fake_build_panel_c_motor_profile_correlation_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_c_motor_profile_correlations",
        fake_plot_panel_c_motor_profile_correlations,
    )
    monkeypatch.setattr(supp_figure_3_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_5.svg"
    datasets = [("L14", "20240611", "08_r4"), ("L15", "20241121", "10_r5")]
    saved_path = make_supplementary_figure_5(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == ["A", "B"]
    assert calls["figure_titles"] == [
        supp_figure_3_module.PANEL_C_TITLE,
        supp_figure_3_module.PANEL_B_TITLE,
    ]
    assert calls["axis_count"] == (
        len(MOTOR_VARIABLES) * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
        + 1
        + len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    )
    assert calls["axis_off_count"] == 1
    assert motor_load_calls == [
        {"data_root": Path("/analysis"), "datasets": datasets}
    ]
    assert motor_plot_calls[0]["axes"].shape == (
        len(MOTOR_VARIABLES),
        len(dark_light_module.PANEL_B_TRAJECTORY_TYPES),
    )
    assert motor_plot_calls[0]["datasets"] == datasets
    assert motor_summary_build_calls[0]["datasets"] == datasets
    assert motor_summary_build_calls[0]["table"]["median"].tolist() == [1.0, 1.1]
    assert len(motor_summary_plot_calls[0]["axes"]) == len(
        dark_light_module.PANEL_B_TRAJECTORY_TYPES
    )
    assert motor_summary_plot_calls[0]["datasets"] == datasets
    assert motor_summary_plot_calls[0]["table"]["correlation"].tolist() == [0.95]

    motor_axes = motor_plot_calls[0]["axes"]
    summary_axes = motor_summary_plot_calls[0]["axes"]
    assert [ax.get_xlabel() for ax in motor_axes[-1, :]] == [
        "Norm. path progression"
    ] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    assert all(not ax.get_xlabel() for ax in motor_axes[:-1, :].ravel())
    assert [ax.get_xlabel() for ax in summary_axes] == [
        "Dark-light correlation"
    ] * len(dark_light_module.PANEL_B_TRAJECTORY_TYPES)
    assert motor_axes[0, 0].texts[-1].get_position() == pytest.approx(
        (-0.28, 1.05)
    )
    assert summary_axes[0].texts[-1].get_position() == pytest.approx(
        (-0.30, 1.05)
    )

    motor_grid = motor_axes[0, 0].get_subplotspec().get_gridspec()
    summary_grid = summary_axes[0].get_subplotspec().get_gridspec()
    motor_grid_params = motor_grid.get_subplot_params()
    summary_grid_params = summary_grid.get_subplot_params()
    assert motor_grid.get_geometry() == (
        len(MOTOR_VARIABLES),
        len(dark_light_module.PANEL_B_TRAJECTORY_TYPES),
    )
    assert motor_grid_params.hspace == pytest.approx(
        supp_figure_3_module.MOTOR_GRID_HSPACE
    )
    assert motor_grid_params.wspace == pytest.approx(
        supp_figure_3_module.MOTOR_GRID_WSPACE
    )
    assert summary_grid.get_geometry() == (
        1,
        len(dark_light_module.PANEL_B_TRAJECTORY_TYPES),
    )
    assert summary_grid_params.wspace == pytest.approx(
        supp_figure_3_module.MOTOR_SUMMARY_GRID_WSPACE
    )

    motor_outer_spec = motor_axes[0, 0].get_subplotspec().get_topmost_subplotspec()
    summary_outer_spec = summary_axes[0].get_subplotspec().get_topmost_subplotspec()
    outer_grid = motor_outer_spec.get_gridspec()
    outer_grid_params = outer_grid.get_subplot_params()
    assert motor_outer_spec.rowspan.start == 0
    assert summary_outer_spec.rowspan.start == 2
    assert outer_grid.get_height_ratios() == pytest.approx(
        [
            DEFAULT_MOTOR_GRID_HEIGHT_MM,
            DEFAULT_BOTTOM_SECTION_SPACER_MM,
            DEFAULT_MOTOR_SUMMARY_HEIGHT_MM,
        ]
    )
    assert outer_grid_params.hspace == pytest.approx(
        supp_figure_3_module.PANEL_GRID_HSPACE
    )
    assert outer_grid_params.left == pytest.approx(
        supp_figure_3_module.PANEL_A_GRID_LEFT
    )
    assert outer_grid_params.right == pytest.approx(
        supp_figure_3_module.PANEL_A_GRID_RIGHT
    )
    assert outer_grid_params.top == pytest.approx(
        supp_figure_3_module.PANEL_A_GRID_TOP
    )
    assert outer_grid_params.bottom == pytest.approx(
        supp_figure_3_module.PANEL_A_GRID_BOTTOM
    )
