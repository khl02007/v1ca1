from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.supplementary_figure_2 as supp_figure_2_module
from v1ca1.paper_figures.supplementary_figure_2 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_RIPPLE_SELECTION_MODES,
    build_output_path,
    get_available_offset_glm_artifacts,
    load_available_glm_scatter_payload,
    make_supplementary_figure_2,
    parse_arguments,
    plot_selection_scatter_grid,
)


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "supplementary_figure_2", "svg") == Path(
        "paper_figures/supplementary_figure_2.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "supplementary_figure_2", "jpg")


def test_default_cli_matches_supplementary_figure_2_defaults() -> None:
    args = parse_arguments([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert tuple(args.ripple_selection) == DEFAULT_RIPPLE_SELECTION_MODES
    assert args.dataset is None


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


def test_load_available_glm_scatter_payload_reads_light_offset_artifacts(
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
    assert len(rows) == 1
    assert rows[0]["dataset"] == ("L14", "20240611", "08_r4")
    assert [artifact["epoch"] for artifact in rows[0]["artifacts"]] == ["02_r1"]
    assert [artifact["target_window_offset_s"] for artifact in rows[0]["artifacts"]] == [-0.2]
    table = rows[0]["artifacts"][0]["summary_table"]
    assert table["unit_id"].tolist() == [11, 12]
    assert table["n_ripples"].tolist() == [5, 5]
    assert table["n_shuffles"].tolist() == [3, 3]


def test_plot_selection_scatter_grid_uses_one_axis_per_available_offset() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection_rows = [
        {
            "dataset": ("L14", "20240611", "08_r4"),
            "artifacts": [
                {
                    "epoch": "02_r1",
                    "target_window_offset_s": -0.2,
                    "target_window_label": "-200 to 0 ms",
                    "summary_table": pd.DataFrame(
                        {
                            "ripple_devexp_mean": [0.1, -0.02],
                            "ripple_devexp_p_value": [0.01, 0.2],
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
        color="black",
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
        "L14\n20240611",
    ]
    assert child_axes[0].get_ylabel() == ""
    assert len(child_axes[0].collections) == 2
    plt.close(fig)


def test_make_supplementary_figure_2_saves_two_selection_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    payload = {
        "rows_by_selection": {
            "allripples": [
                {
                    "dataset": ("L14", "20240611", "08_r4"),
                    "artifacts": [
                        {
                            "epoch": "02_r1",
                            "target_window_offset_s": 0.0,
                            "target_window_label": "0 to 200 ms",
                            "summary_table": pd.DataFrame(
                                {
                                    "ripple_devexp_mean": [0.1],
                                    "ripple_devexp_p_value": [0.01],
                                }
                            ),
                        }
                    ],
                }
            ],
            "single": [
                {
                    "dataset": ("L14", "20240611", "08_r4"),
                    "artifacts": [
                        {
                            "epoch": "02_r1",
                            "target_window_offset_s": 0.0,
                            "target_window_label": "0 to 200 ms",
                            "summary_table": pd.DataFrame(
                                {
                                    "ripple_devexp_mean": [0.2],
                                    "ripple_devexp_p_value": [0.02],
                                }
                            ),
                        }
                    ],
                }
            ],
        },
        "ripple_selection_modes": ("allripples", "single"),
        "ripple_window_s": 0.2,
        "ridge_strength": 0.1,
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        supp_figure_2_module,
        "load_available_glm_scatter_payload",
        lambda *_args, **_kwargs: payload,
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
        return output_path

    monkeypatch.setattr(supp_figure_2_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_2.svg"
    saved_path = make_supplementary_figure_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        ripple_selection_modes=("allripples", "single"),
        ripple_window_s=0.2,
        ridge_strength=0.1,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert set(calls["panel_labels"]) >= {
        "A",
        "B",
        "All ripples (02_r1)",
        "Single-ripple windows (02_r1)",
    }
