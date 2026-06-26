from __future__ import annotations

from math import log10
from pathlib import Path

import pytest

import v1ca1.paper_figures.supplementary_figure_4 as supp_figure_4_module
from v1ca1.paper_figures.supplementary_figure_4 import (
    DEFAULT_EPOCH_TYPES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PER_ANIMAL_SIGNIFICANCE_P_VALUE,
    DEFAULT_RIPPLE_SELECTION_MODES,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    SUPPLEMENTARY_FIGURE_4_SIGNIFICANCE_P_VALUE,
    build_output_path,
    get_available_offset_glm_artifacts,
    get_epoch_type_color,
    load_available_glm_scatter_payload,
    load_per_animal_glm_scatter_payload,
    make_supplementary_figure_4,
    parse_arguments,
    plot_dark_firing_rate_devexp_grid,
    plot_glm_scatter_box_panel,
    plot_per_animal_behavior_association_grid,
    plot_per_animal_glm_scatter_grid,
    plot_selection_scatter_grid,
)


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "supplementary_figure_4", "svg") == Path(
        "paper_figures/supplementary_figure_4.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "supplementary_figure_4", "jpg")


def test_default_cli_matches_supplementary_figure_4_defaults() -> None:
    args = parse_arguments([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert tuple(args.ripple_selection) == DEFAULT_RIPPLE_SELECTION_MODES
    assert DEFAULT_RIPPLE_SELECTION_MODES == ("single",)
    assert DEFAULT_PER_ANIMAL_SIGNIFICANCE_P_VALUE == pytest.approx(0.05)
    assert args.dataset is None
    assert args.ripple_window_offset_s == DEFAULT_RIPPLE_WINDOW_OFFSET_S
    assert DEFAULT_EPOCH_TYPES == ("light", "dark", "sleep")
    assert SUPPLEMENTARY_FIGURE_4_SIGNIFICANCE_P_VALUE == pytest.approx(0.005)


def test_get_available_offset_glm_artifacts_uses_target_offset_paths(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "L14" / "20240611" / "ripple_glm"
    data_dir.mkdir(parents=True)
    zero_offset = data_dir / "02_r1_rw_0p2s_single_ridge_1e-1_samplewise_ripple_glm.nc"
    negative_offset = data_dir / (
        "02_r1_src_rw_0p2s_tgt_rw_0p2s_off_m0p2s_"
        "single_ridge_1e-1_samplewise_ripple_glm.nc"
    )
    legacy_same_shift = data_dir / (
        "02_r1_rw_0p2s_off_m0p2s_single_ridge_1e-1_"
        "samplewise_ripple_glm.nc"
    )
    zero_offset.touch()
    negative_offset.touch()
    legacy_same_shift.touch()

    artifacts = get_available_offset_glm_artifacts(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_selection="single",
        target_window_offsets_s=(-0.2, 0.0, 0.2),
    )

    assert [artifact["path"] for artifact in artifacts] == [negative_offset, zero_offset]
    assert [artifact["target_window_offset_s"] for artifact in artifacts] == [-0.2, 0.0]
    assert [artifact["target_window_label"] for artifact in artifacts] == [
        "-200 to 0 ms",
        "0 to 200 ms",
    ]


def test_load_available_glm_scatter_payload_reads_epoch_offset_artifacts(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")

    data_dir = tmp_path / "L14" / "20240611" / "ripple_glm"
    data_dir.mkdir(parents=True)
    path = data_dir / (
        "02_r1_src_rw_0p2s_tgt_rw_0p2s_off_m0p2s_"
        "allripples_ridge_1e-1_samplewise_ripple_glm.nc"
    )
    dataset = xr.Dataset(
        {
            "ripple_devexp_mean": ("unit", np.array([0.1, 0.2])),
            "ripple_devexp_p_value": ("unit", np.array([0.01, 0.2])),
        },
        coords={"unit": np.array([11, 12]), "shuffle": np.arange(3)},
        attrs={"n_ripples_after_selection": 5, "schema_version": "7"},
    )
    dataset.to_netcdf(path)

    payload = load_available_glm_scatter_payload(
        tmp_path,
        [("L14", "20240611", "08_r4")],
        ripple_selection_modes=("allripples",),
    )

    rows = payload["rows_by_selection"]["allripples"]
    assert len(rows) == 3
    assert rows[0]["dataset"] == ("L14", "20240611", "08_r4")
    assert [row["epoch_type"] for row in rows] == ["light", "dark", "sleep"]
    assert [row["epoch"] for row in rows] == ["02_r1", "08_r4", "07_s4"]
    assert [artifact["epoch"] for artifact in rows[0]["artifacts"]] == ["02_r1"]
    assert [artifact["target_window_offset_s"] for artifact in rows[0]["artifacts"]] == [-0.2]
    assert rows[1]["artifacts"] == []
    assert rows[2]["artifacts"] == []
    table = rows[0]["artifacts"][0]["summary_table"]
    assert table["unit_id"].tolist() == [11, 12]
    assert table["n_ripples"].tolist() == [5, 5]
    assert table["n_shuffles"].tolist() == [3, 3]


def test_plot_selection_scatter_grid_uses_one_axis_per_available_offset() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt

    from v1ca1.paper_figures.style import EPOCH_TYPE_COLORS

    selection_rows = [
        {
            "dataset": ("L14", "20240611", "08_r4"),
            "animal_name": "L14",
            "date": "20240611",
            "epoch_type": "light",
            "epoch": "02_r1",
            "artifacts": [
                {
                    "epoch": "02_r1",
                    "target_window_offset_s": -0.2,
                    "target_window_label": "-200 to 0 ms",
                    "summary_table": pd.DataFrame(
                        {
                            "ripple_devexp_mean": [0.1, -0.02],
                            "ripple_devexp_p_value": [0.001, 0.2],
                        }
                    ),
                },
                {
                    "epoch": "02_r1",
                    "target_window_offset_s": 0.0,
                    "target_window_label": "0 to 200 ms",
                    "summary_table": pd.DataFrame(
                        {
                            "ripple_devexp_mean": [0.2],
                            "ripple_devexp_p_value": [0.03],
                        }
                    ),
                },
            ],
        }
    ]

    fig, ax = plt.subplots()
    child_axes = plot_selection_scatter_grid(
        ax,
        selection_rows,
        selection_label="All ripples",
        x_limits=(-0.1, 0.5),
        y_limit=2.2,
    )

    assert len(child_axes) == 2
    assert [child_axis.get_title() for child_axis in child_axes] == [
        "-200 to 0 ms",
        "0 to 200 ms",
    ]
    assert [text.get_text() for text in ax.texts[:2]] == [
        "All ripples (02_r1)",
        "L14\n20240611\nLight 02_r1",
    ]
    assert child_axes[0].get_ylabel() == ""
    assert len(child_axes[0].collections) == 2
    assert child_axes[0].lines[1].get_ydata()[0] == pytest.approx(
        -log10(SUPPLEMENTARY_FIGURE_4_SIGNIFICANCE_P_VALUE)
    )
    assert get_epoch_type_color("light") == EPOCH_TYPE_COLORS["light"]
    assert tuple(child_axes[0].collections[1].get_facecolors()[0]) == pytest.approx(
        to_rgba(EPOCH_TYPE_COLORS["light"], alpha=0.52)
    )
    plt.close(fig)


def test_load_per_animal_glm_scatter_payload_keeps_run_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")

    epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "summary_table": pd.DataFrame({"animal_name": ["L14"]}),
        },
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "summary_table": pd.DataFrame({"animal_name": ["L14"]}),
        },
        {
            "epoch_type": "sleep",
            "label": "Sleep",
            "summary_table": pd.DataFrame({"animal_name": ["L14"]}),
        },
    ]
    calls: dict[str, object] = {}

    def fake_load_glm_epoch_summary_tables(*_args, **kwargs):
        calls.update(kwargs)
        return epoch_tables

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_glm_epoch_summary_tables",
        fake_load_glm_epoch_summary_tables,
    )

    payload = load_per_animal_glm_scatter_payload(
        Path("/analysis"),
        [("L14", "20240611", "08_r4")],
    )

    assert [table["epoch_type"] for table in payload["epoch_tables"]] == ["light"]
    assert tuple(calls["epoch_types"]) == ("light",)
    assert calls["ripple_selection"] == "single"
    assert payload["ripple_selection"] == "single"


def test_plot_per_animal_glm_scatter_grid_plots_run_only() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_rows = {
        "animal_name": ["L14", "L14", "L15", "L15"],
        "date": ["20240611", "20240611", "20241121", "20241121"],
        "ripple_devexp_mean": [0.01, 0.12, -0.02, 0.18],
        "ripple_devexp_p_value": [0.2, 0.001, 0.5, 0.0005],
    }
    payload = {
        "epoch_tables": [
            {
                "epoch_type": "light",
                "label": "Light run",
                "summary_table": pd.DataFrame(base_rows),
            },
        ],
        "ripple_selection": "single",
    }

    fig, ax = plt.subplots()
    child_axes = plot_per_animal_glm_scatter_grid(ax, payload, y_limit=3.0)

    assert len(child_axes) == 2
    assert [child_axis.get_title() for child_axis in child_axes] == [
        "L14",
        "L15",
    ]
    assert child_axes[0].get_position().x0 < child_axes[1].get_position().x0
    assert child_axes[0].get_position().y0 == pytest.approx(
        child_axes[1].get_position().y0
    )
    scatter_axis_ids = {id(child_axis) for child_axis in child_axes}
    box_axes = [
        child_axis
        for child_axis in ax.child_axes
        if id(child_axis) not in scatter_axis_ids
    ]
    assert len(box_axes) == 2
    assert box_axes[0].get_position().y0 < child_axes[0].get_position().y0
    assert [label.get_text() for label in box_axes[0].get_yticklabels()] == [
        "n.s.",
        "p<0.05",
    ]
    assert all(child_axis.get_xlim()[0] == pytest.approx(-0.05) for child_axis in child_axes)
    assert all(child_axis.get_xlim()[1] == pytest.approx(0.40) for child_axis in child_axes)
    assert all(len(child_axis.collections) == 2 for child_axis in child_axes)
    assert child_axes[0].lines[1].get_ydata()[0] == pytest.approx(
        -log10(DEFAULT_PER_ANIMAL_SIGNIFICANCE_P_VALUE)
    )
    plt.close(fig)


def test_plot_dark_firing_rate_devexp_grid_splits_datasets() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15"],
            "date": ["20240611", "20240611", "20241121", "20241121"],
            "epoch_type": ["light", "sleep", "light", "sleep"],
            "dark_firing_rate_hz": [0.5, 1.5, 2.0, 4.0],
            "ripple_devexp_mean": [0.01, 0.12, -0.02, 0.18],
            "ripple_devexp_p_value": [0.2, 0.001, 0.5, 0.0005],
        }
    )
    payload = {"devexp_table": table, "missing_artifacts": []}

    fig, ax = plt.subplots()
    child_axes = plot_dark_firing_rate_devexp_grid(ax, payload)

    assert len(child_axes) == 4
    assert [child_axis.get_title() for child_axis in child_axes[:2]] == ["Run", "Sleep"]
    assert all(child_axis.get_yscale() == "log" for child_axis in child_axes)
    assert all(len(child_axis.collections) == 1 for child_axis in child_axes)
    first_offset = child_axes[0].collections[0].get_offsets()[0]
    assert first_offset[0] == pytest.approx(0.01)
    assert first_offset[1] == pytest.approx(0.5)
    assert child_axes[1].get_xlim()[1] >= 0.40
    plt.close(fig)


def test_plot_per_animal_behavior_association_grid_splits_animal_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L15"],
            "date": ["20240611", "20241121"],
            "epoch_type": ["light", "light"],
            "ripple_devexp_mean": [0.1, 0.2],
            "ripple_devexp_p_value": [0.001, 0.01],
            "dark_firing_rate_hz": [0.2, 1.0],
            "same_turn_tuning_similarity": [0.1, 0.4],
        }
    )
    payload = {
        "devexp_table": table,
        "missing_artifacts": [],
        "dark_activity_threshold_hz": 0.5,
    }
    row_animals: list[list[str]] = []

    def fake_plot_glm_behavior_association_panel(row_ax, row_payload, **_kwargs):
        row_animals.append(
            row_payload["devexp_table"]["animal_name"].astype(str).tolist()
        )
        for column_index, label in enumerate(
            ("p<0.05 frac.", "Dev. explained", "Dark DPP corr.")
        ):
            child_ax = row_ax.inset_axes([0.05 + 0.3 * column_index, 0.2, 0.22, 0.6])
            child_ax.set_xlabel(label)

    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_glm_behavior_association_panel",
        fake_plot_glm_behavior_association_panel,
    )

    fig, ax = plt.subplots()
    row_axes = plot_per_animal_behavior_association_grid(ax, payload)

    assert len(row_axes) == 2
    assert row_animals == [["L14"], ["L15"]]
    assert row_axes[0].get_position().y0 > row_axes[1].get_position().y0
    assert {"L14", "L15"}.issubset({text.get_text() for text in ax.texts})
    assert "Relationship to dark-active DGP cells" not in {
        text.get_text() for text in ax.texts
    }
    assert [child_ax.get_xlabel() for child_ax in row_axes[0].child_axes] == [
        "",
        "",
        "",
    ]
    assert [child_ax.get_xlabel() for child_ax in row_axes[1].child_axes] == [
        "p<0.05 frac.",
        "Dev. explained",
        "Dark DPP corr.",
    ]
    plt.close(fig)


def test_plot_glm_scatter_box_panel_omits_schematic() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_table = pd.DataFrame(
        {
            "ripple_devexp_mean": [0.10, -0.02, 0.18],
            "ripple_devexp_p_value": [0.001, 0.2, 0.02],
        }
    )
    epoch_tables = [
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "summary_table": summary_table,
        }
    ]

    fig, ax = plt.subplots()
    plot_glm_scatter_box_panel(ax, epoch_tables)

    assert len(ax.child_axes) == 2
    scatter_ax, box_ax = ax.child_axes
    assert scatter_ax.get_title() == ""
    assert len(scatter_ax.collections) == 2
    assert scatter_ax.get_ylabel() == r"-log10 $\mathit{p}$ from shuffle"
    assert box_ax.get_xlabel() == "Deviance explained"
    assert [label.get_text() for label in box_ax.get_yticklabels()][0] == "n.s."
    plt.close(fig)


def test_make_supplementary_figure_4_saves_glm_summary_panels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    summary_table = pd.DataFrame(
        {
            "animal_name": ["L14"],
            "date": ["20240611"],
            "ripple_devexp_mean": [0.1],
            "ripple_devexp_p_value": [0.001],
        }
    )
    glm_epoch_tables = [
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "summary_table": summary_table,
        }
    ]
    source_comparison_payload = {
        "comparison_table": pd.DataFrame(
            {
                "animal_name": ["L14", "L14"],
                "date": ["20240611", "20240611"],
                "mean_activity_devexp_mean": [0.01, 0.02],
                "vector_devexp_mean": [0.05, 0.08],
                "vector_devexp_p_value": [0.001, 0.2],
            }
        ),
        "missing_artifacts": [],
    }
    behavior_payload = {
        "devexp_table": pd.DataFrame(
            {
                "animal_name": ["L14", "L14"],
                "date": ["20240611", "20240611"],
                "epoch_type": ["dark", "dark"],
                "ripple_devexp_mean": [0.1, 0.2],
                "ripple_devexp_p_value": [0.001, 0.2],
                "dark_firing_rate_hz": [0.2, 1.0],
                "same_turn_tuning_similarity": [0.1, 0.4],
            }
        ),
        "missing_artifacts": [],
        "dark_activity_threshold_hz": 0.5,
    }
    xcorr_payload = {
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "02_r1",
        "v1_order_reference_ca1_unit": 11,
        "display_vmax": 5.0,
        "ca1_unit_ids": np.array([11]),
        "v1_unit_ids": np.array([101, 102]),
        "lag_s": np.array([-0.01, 0.0, 0.01]),
        "xcorr": np.ones((1, 2, 3), dtype=float),
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_top_ca1_xcorr_panel_data",
        lambda *_args, **_kwargs: xcorr_payload,
    )

    def fake_load_glm_epoch_summary_tables(*_args, **kwargs):
        calls["glm_epoch_kwargs"] = kwargs
        return glm_epoch_tables

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_glm_epoch_summary_tables",
        fake_load_glm_epoch_summary_tables,
    )

    def fake_load_source_comparison_tables(*_args, **kwargs):
        calls["source_kwargs"] = kwargs
        return source_comparison_payload

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_glm_source_predictor_comparison_tables",
        fake_load_source_comparison_tables,
    )

    def fake_load_behavior_tables(*_args, **kwargs):
        calls["behavior_kwargs"] = kwargs
        return behavior_payload

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_glm_dark_activity_devexp_tables",
        fake_load_behavior_tables,
    )

    def fake_save_figure(figure, output_path: Path, dpi: int):
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        calls["axis_titles"] = [ax.get_title() for ax in figure.axes]
        calls["lag_label_y"] = [
            text.get_position()[1]
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() == "Lag (s)"
        ]
        return output_path

    monkeypatch.setattr(supp_figure_4_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_4.svg"
    saved_path = make_supplementary_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        xcorr_dataset=("L15", "20241121", "02_r1"),
        xcorr_state="ripple",
        xcorr_top_ca1_units=1,
        xcorr_bin_size_s=0.005,
        xcorr_max_lag_s=0.5,
        xcorr_display_vmax=5.0,
        ripple_selection_modes=("single",),
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ridge_strength=0.1,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["glm_epoch_kwargs"]["epoch_types"] == ("dark",)
    assert calls["glm_epoch_kwargs"]["ripple_selection"] == "single"
    assert calls["glm_epoch_kwargs"]["ripple_window_offset_s"] == 0.0
    assert calls["source_kwargs"]["ripple_selection"] == "single"
    assert calls["behavior_kwargs"]["ripple_selection"] == "single"
    assert set(calls["panel_labels"]) >= {
        "A",
        "B",
        "C",
        "D",
    }
    assert calls["lag_label_y"] == [pytest.approx(-0.025)]
    assert "Figure 4C run scatter by animal (single)" not in calls["panel_labels"]
    assert (
        "Predicting V1 activity during ripples\nwith CA1 activity"
        in calls["axis_titles"]
    )
    assert (
        "CA1 spike vector vs. mean CA1 activity"
        in calls["axis_titles"]
    )
    assert (
        "Relationship to dark-active DPP cells"
        in calls["axis_titles"]
    )
    assert all("(single)" not in title for title in calls["axis_titles"])
    assert (
        "Dark movement firing rate versus deviance explained"
        not in calls["panel_labels"]
    )
