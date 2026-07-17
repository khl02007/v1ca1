from __future__ import annotations

from math import log10
from pathlib import Path

import pytest

import v1ca1.paper_figures.supplementary_figure_4 as supp_figure_4_module
from v1ca1.paper_figures.supplementary_figure_4 import (
    DEFAULT_EPOCH_TYPES,
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_REGIONS,
    DEFAULT_PER_ANIMAL_SIGNIFICANCE_P_VALUE,
    DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
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
    assert args.region is None
    assert DEFAULT_REGIONS == ("v1", "ca1")
    assert args.ripple_threshold_zscore == DEFAULT_RIPPLE_THRESHOLD_ZSCORE
    assert DEFAULT_PER_ANIMAL_SIGNIFICANCE_P_VALUE == pytest.approx(0.05)
    assert args.dataset is None
    assert args.ripple_window_offset_s == DEFAULT_RIPPLE_WINDOW_OFFSET_S
    assert DEFAULT_EPOCH_TYPES == ("light", "dark", "sleep")
    assert SUPPLEMENTARY_FIGURE_4_SIGNIFICANCE_P_VALUE == pytest.approx(0.005)
    assert not hasattr(args, "xcorr_dataset")

    with pytest.raises(SystemExit):
        parse_arguments(["--xcorr-state", "ripple"])


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
            ("p<0.05 frac.", "Dev. explained", "Dark DPP overlap")
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
        "Dark DPP overlap",
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
                "epoch_type": ["dark", "dark"],
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
                "ripple_devexp_p_value": [0.001, 0.001],
                "dark_firing_rate_hz": [0.2, 1.0],
                "same_turn_tuning_similarity": [0.1, 0.4],
            }
        ),
        "missing_artifacts": [],
        "dark_activity_threshold_hz": 0.5,
    }
    heatmap_epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "02_r1",
            "firing_rate_table": pd.DataFrame(
                {
                    "animal_name": ["L14", "L14", "L14", "L14"],
                    "date": ["20240611", "20240611", "20240611", "20240611"],
                    "epoch": ["02_r1", "02_r1", "02_r1", "02_r1"],
                    "region": ["ca1", "ca1", "v1", "v1"],
                    "unit_id": [101, 101, 11, 11],
                    "time_s": [-0.02, 0.0, -0.02, 0.0],
                    "mean_rate_hz": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "summary_table": pd.DataFrame(
                {
                    "region": ["ca1", "v1"],
                    "ripple_modulation_index": [0.1, 0.2],
                }
            ),
        }
    ]
    calls: dict[str, object] = {}

    def fake_load_heatmap_tables(*_args, **kwargs):
        calls["heatmap_kwargs"] = kwargs
        return heatmap_epoch_tables

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_pooled_ripple_heatmap_epoch_tables",
        fake_load_heatmap_tables,
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
        figure.canvas.draw()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["figure_size_inches"] = tuple(figure.get_size_inches())
        parent_axes = figure.axes[:4]
        calls["parent_bounds"] = [axis.get_position().bounds for axis in parent_axes]
        calls["child_bounds"] = [
            [child_axis.get_position().bounds for child_axis in axis.child_axes]
            for axis in parent_axes
        ]
        renderer = figure.canvas.get_renderer()
        calls["child_window_bounds"] = [
            [
                child_axis.get_window_extent(renderer).bounds
                for child_axis in axis.child_axes
            ]
            for axis in parent_axes
        ]
        calls["child_tight_bounds"] = [
            [child_axis.get_tightbbox(renderer).bounds for child_axis in axis.child_axes]
            for axis in parent_axes
        ]
        calls["child_titles"] = [
            [child_axis.get_title() for child_axis in axis.child_axes]
            for axis in parent_axes
        ]
        calls["figure_dpi"] = figure.dpi
        figure_and_axis_texts = [
            *figure.texts,
            *(text for ax in figure.axes for text in ax.texts),
        ]
        calls["panel_labels"] = [
            text.get_text()
            for text in figure_and_axis_texts
            if text.get_fontweight() == "bold"
        ]
        calls["titles"] = [
            *(text.get_text() for text in figure.texts),
            *(ax.get_title() for ax in figure.axes),
        ]
        calls["header_bounds"] = {
            text.get_text(): text.get_window_extent().bounds
            for text in figure.texts
            if text.get_text()
            in {
                "A",
                "B",
                "C",
                "D",
                "Ripple modulation index",
                "Predicting V1 activity during ripples with CA1 activity",
                "CA1 spike vector vs.\nmean CA1 activity",
                "Relationship to dark-active DPP cells",
            }
        }
        source_child_axis = parent_axes[2].child_axes[0]
        calls["source_summary_texts"] = [
            (
                text.get_text(),
                text.get_position(),
                text.get_horizontalalignment(),
                text.get_verticalalignment(),
            )
            for text in source_child_axis.texts
            if text.get_text().startswith("n=")
        ]
        calls["source_x_label_texts"] = [
            (text.get_position(), text.get_horizontalalignment())
            for text in parent_axes[2].texts
            if text.get_text() == "Mean CA1 activity dev. explained"
        ]
        composition_axis, _devexp_axis, similarity_axis = parent_axes[3].child_axes
        calls["behavior_x_labels"] = [
            (child_axis.get_xlabel(), child_axis.xaxis.label.get_position())
            for child_axis in parent_axes[3].child_axes
        ]
        calls["composition_label_texts"] = [
            (
                text.get_position(),
                text.get_horizontalalignment(),
            )
            for text in composition_axis.texts
            if "\nn=" in text.get_text()
        ]
        calls["similarity_median_texts"] = [
            (
                text.get_position(),
                text.get_horizontalalignment(),
                text.get_verticalalignment(),
            )
            for text in similarity_axis.texts
            if text.get_text().startswith("median=")
        ]
        return output_path

    monkeypatch.setattr(supp_figure_4_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_4.svg"
    saved_path = make_supplementary_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("ca1", "v1"),
        light_epoch="02_override",
        dark_epoch="08_override",
        sleep_epoch="07_override",
        ripple_threshold_zscore=2.0,
        ripple_selection_modes=("single",),
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ridge_strength=0.1,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["heatmap_kwargs"]["ripple_threshold_zscore"] == 2.0
    assert calls["heatmap_kwargs"]["light_epoch"] == "02_override"
    assert calls["heatmap_kwargs"]["dark_epoch"] == "08_override"
    assert calls["heatmap_kwargs"]["sleep_epoch"] == "07_override"
    assert calls["glm_epoch_kwargs"]["epoch_types"] == ("dark",)
    assert calls["glm_epoch_kwargs"]["light_epoch"] == "02_override"
    assert calls["glm_epoch_kwargs"]["dark_epoch"] == "08_override"
    assert calls["glm_epoch_kwargs"]["sleep_epoch"] == "07_override"
    assert calls["glm_epoch_kwargs"]["ripple_selection"] == "single"
    assert calls["glm_epoch_kwargs"]["ripple_window_offset_s"] == 0.0
    assert calls["source_kwargs"]["ripple_selection"] == "single"
    assert calls["source_kwargs"]["epoch_types"] == ("dark",)
    assert calls["source_kwargs"]["light_epoch"] == "02_override"
    assert calls["source_kwargs"]["dark_epoch"] == "08_override"
    assert calls["source_kwargs"]["sleep_epoch"] == "07_override"
    assert calls["behavior_kwargs"]["ripple_selection"] == "single"
    assert calls["behavior_kwargs"]["epoch_types"] == ("dark",)
    assert calls["behavior_kwargs"]["light_epoch"] == "02_override"
    assert calls["behavior_kwargs"]["dark_epoch"] == "08_override"
    assert calls["behavior_kwargs"]["sleep_epoch"] == "07_override"
    assert calls["figure_size_inches"] == pytest.approx(
        (DEFAULT_FIGURE_WIDTH_MM / 25.4, DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    )
    panel_a_bounds, panel_b_bounds, panel_c_bounds, panel_d_bounds = calls[
        "parent_bounds"
    ]
    assert panel_a_bounds[1] > panel_c_bounds[1]
    assert panel_b_bounds[1] > panel_d_bounds[1]
    assert panel_a_bounds[0] == pytest.approx(panel_c_bounds[0])
    assert panel_b_bounds[0] == pytest.approx(panel_d_bounds[0])
    assert panel_b_bounds[2] > 2.0 * panel_a_bounds[2]
    assert panel_d_bounds[2] > 2.0 * panel_c_bounds[2]

    top_row_height = min(panel_a_bounds[3], panel_b_bounds[3])
    bottom_row_height = min(panel_c_bounds[3], panel_d_bounds[3])
    assert panel_a_bounds[3] == pytest.approx(panel_b_bounds[3])
    assert panel_c_bounds[3] == pytest.approx(panel_d_bounds[3])
    assert top_row_height == pytest.approx(bottom_row_height)
    top_row_bottom = min(panel_a_bounds[1], panel_b_bounds[1])
    bottom_row_top = max(
        panel_c_bounds[1] + panel_c_bounds[3],
        panel_d_bounds[1] + panel_d_bounds[3],
    )
    parent_row_gap_mm = (top_row_bottom - bottom_row_top) * DEFAULT_FIGURE_HEIGHT_MM
    assert 0.0 <= parent_row_gap_mm <= 14.0

    def vertical_envelope(
        bounds: list[tuple[float, float, float, float]],
    ) -> tuple[float, float]:
        bottom = min(child_bottom for _left, child_bottom, _width, _height in bounds)
        top = max(
            child_bottom + child_height
            for _left, child_bottom, _width, child_height in bounds
        )
        return bottom, top

    panel_data_envelopes = [
        vertical_envelope(child_bounds) for child_bounds in calls["child_bounds"]
    ]
    one_mm_figure_fraction = 1.0 / DEFAULT_FIGURE_HEIGHT_MM
    panel_a_data_height = panel_data_envelopes[0][1] - panel_data_envelopes[0][0]
    panel_b_data_height = panel_data_envelopes[1][1] - panel_data_envelopes[1][0]
    panel_c_data_height = panel_data_envelopes[2][1] - panel_data_envelopes[2][0]
    panel_d_data_height = panel_data_envelopes[3][1] - panel_data_envelopes[3][0]
    assert panel_a_data_height == pytest.approx(
        panel_c_data_height,
        abs=one_mm_figure_fraction,
    )
    assert panel_b_data_height == pytest.approx(
        panel_d_data_height,
        abs=one_mm_figure_fraction,
    )

    source_bounds = calls["child_bounds"][2]
    behavior_bounds = calls["child_bounds"][3]
    assert len(source_bounds) == 1
    assert len(behavior_bounds) == 3
    source_width_mm = source_bounds[0][2] * DEFAULT_FIGURE_WIDTH_MM
    source_height_mm = source_bounds[0][3] * DEFAULT_FIGURE_HEIGHT_MM
    assert source_width_mm >= 25.0
    assert source_height_mm >= 25.0
    assert calls["child_titles"][2] == [""]
    for child_bounds in behavior_bounds:
        child_left, _child_bottom, child_width, child_height = child_bounds
        assert child_left >= panel_d_bounds[0]
        assert child_left + child_width <= panel_d_bounds[0] + panel_d_bounds[2]
        assert child_width * DEFAULT_FIGURE_WIDTH_MM >= 20.0
        assert child_height * DEFAULT_FIGURE_HEIGHT_MM >= 25.0

    minimum_horizontal_gap_px = 2.0 * calls["figure_dpi"] / 25.4
    behavior_tight_bounds = calls["child_tight_bounds"][3]
    for left_bounds, right_bounds in zip(
        behavior_tight_bounds[:-1],
        behavior_tight_bounds[1:],
        strict=True,
    ):
        assert right_bounds[0] - (left_bounds[0] + left_bounds[2]) >= (
            minimum_horizontal_gap_px
        )

    minimum_vertical_gap_px = 1.5 * calls["figure_dpi"] / 25.4
    scatter_tight_bounds, box_tight_bounds = calls["child_tight_bounds"][1]
    assert scatter_tight_bounds[1] - (
        box_tight_bounds[1] + box_tight_bounds[3]
    ) >= minimum_vertical_gap_px

    header_bounds = calls["header_bounds"]
    minimum_header_gap_px = calls["figure_dpi"] / 25.4
    header_pairs = (
        ("A", "Ripple modulation index"),
        ("B", "Predicting V1 activity during ripples with CA1 activity"),
        ("C", "CA1 spike vector vs.\nmean CA1 activity"),
        ("D", "Relationship to dark-active DPP cells"),
    )
    for label, title in header_pairs:
        label_left, _label_bottom, label_width, _label_height = header_bounds[label]
        assert label_left >= 0.0
        title_left = header_bounds[title][0]
        assert label_left + label_width + minimum_header_gap_px <= title_left
    for panel_index, (label, title) in enumerate(header_pairs):
        header_bottom = min(header_bounds[label][1], header_bounds[title][1])
        child_data_top = max(
            child_bottom + child_height
            for _child_left, child_bottom, _child_width, child_height in calls[
                "child_window_bounds"
            ][panel_index]
        )
        assert header_bottom - child_data_top >= minimum_header_gap_px

    assert calls["source_summary_texts"] == [
        ("n=1\nfrac vector>mean=1.00", (0.97, 0.05), "right", "bottom")
    ]
    assert calls["source_x_label_texts"] == [((0.52, 0.0), "center")]
    assert [label for label, _position in calls["behavior_x_labels"]] == [
        r"$p$<0.05 frac.",
        "Dev. explained",
        "Dark DPPI",
    ]
    assert all("\n" not in label for label, _position in calls["behavior_x_labels"])
    assert len(calls["composition_label_texts"]) == 2
    for (label_x, _label_y), horizontal_alignment in calls[
        "composition_label_texts"
    ]:
        assert label_x == pytest.approx(1.02)
        assert horizontal_alignment == "left"
    assert len(calls["similarity_median_texts"]) == 1
    median_position, median_horizontal_alignment, median_vertical_alignment = calls[
        "similarity_median_texts"
    ][0]
    assert 0.0 <= median_position[0] <= 0.1
    assert 0.9 <= median_position[1] <= 1.0
    assert median_horizontal_alignment == "left"
    assert median_vertical_alignment == "top"
    assert set(calls["panel_labels"]) >= {"A", "B", "C", "D"}
    assert "E" not in calls["panel_labels"]
    assert "Figure 4C run scatter by animal (single)" not in calls["panel_labels"]
    assert "Ripple modulation index" in calls["titles"]
    assert "CA1-V1 cross correlation during ripples" not in calls["titles"]
    assert (
        "Predicting V1 activity during ripples with CA1 activity"
        in calls["titles"]
    )
    assert (
        "CA1 spike vector vs.\nmean CA1 activity"
        in calls["titles"]
    )
    assert (
        "Relationship to dark-active DPP cells"
        in calls["titles"]
    )
    assert "Pooled" not in calls["titles"]
    assert all("(single)" not in title for title in calls["titles"])
    assert (
        "Dark movement firing rate versus deviance explained"
        not in calls["panel_labels"]
    )
