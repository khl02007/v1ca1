from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import v1ca1.paper_figures.supplementary_figure_8 as figure


def test_defaults_and_main_forward_figure_4_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_make_figure(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        figure,
        "get_processed_datasets",
        lambda: pytest.fail("explicit --dataset values should be used"),
    )
    monkeypatch.setattr(figure, "make_supplementary_figure_8", fake_make_figure)

    figure.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            "/figures",
            "--output-name",
            "individual_figure_4def",
            "--format",
            "png",
            "--dataset",
            "L14:20240611:08_r4",
            "--dataset",
            "L15:20241121",
            "--light-epoch",
            "03_r2",
            "--dark-epoch",
            "11_r6",
            "--sleep-epoch",
            "09_s5",
            "--ripple-window-s",
            "0.3",
            "--ripple-window-offset-s",
            "-0.1",
            "--ripple-selection",
            "deduped",
            "--ridge-strength",
            "0.2",
            "--dark-movement-fr-cache-dir",
            "/cache",
            "--refresh-dark-movement-fr-cache",
            "--dpi",
            "144",
        ]
    )

    assert figure.DEFAULT_OUTPUT_NAME == "supplementary_figure_8"
    assert figure.PANEL_LABELS == ("A", "B", "C")
    assert calls == [
        {
            "data_root": Path("/analysis"),
            "output_path": Path("/figures/individual_figure_4def.png"),
            "datasets": [
                ("L14", "20240611", "08_r4"),
                ("L15", "20241121", "10_r5"),
            ],
            "light_epoch": "03_r2",
            "dark_epoch": "11_r6",
            "sleep_epoch": "09_s5",
            "ripple_window_s": 0.3,
            "ripple_window_offset_s": -0.1,
            "ripple_selection": "deduped",
            "ridge_strength": 0.2,
            "dark_movement_fr_cache_dir": Path("/cache"),
            "refresh_dark_movement_fr_cache": True,
            "dpi": 144,
        }
    ]


def test_grouping_and_filters_keep_all_sessions_for_each_animal() -> None:
    pd = pytest.importorskip("pandas")
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L14", "20240612", "08_r4"),
    ]
    grouped = figure.group_datasets_by_animal(datasets)

    assert list(grouped) == ["L14", "L15"]
    assert grouped["L14"] == [datasets[0], datasets[2]]

    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L15", "L14"],
            "token": ["14-a", "15", "14-b"],
        }
    )
    epoch_payloads = [
        {
            "epoch_type": "light",
            "datasets": tuple(datasets),
            "n_datasets": 3,
            "summary_table": table,
        }
    ]
    filtered_epochs = figure.filter_epoch_tables_by_animal(
        epoch_payloads,
        "L14",
    )
    assert filtered_epochs[0]["datasets"] == (datasets[0], datasets[2])
    assert filtered_epochs[0]["n_datasets"] == 2
    assert filtered_epochs[0]["summary_table"]["token"].tolist() == [
        "14-a",
        "14-b",
    ]

    behavior_payload = {
        "devexp_table": table,
        "dark_activity_reference_table": table,
        "dark_active_dppi_reference_table": table,
        "missing_artifacts": [
            {"animal_name": "L14", "artifact": "a"},
            {"animal_name": "L15", "artifact": "b"},
        ],
        "region": "v1",
    }
    filtered_behavior = figure.filter_behavior_payload_by_animal(
        behavior_payload,
        "L15",
    )
    for key in (
        "devexp_table",
        "dark_activity_reference_table",
        "dark_active_dppi_reference_table",
    ):
        assert filtered_behavior[key]["token"].tolist() == ["15"]
    assert filtered_behavior["missing_artifacts"] == [
        {"animal_name": "L15", "artifact": "b"}
    ]
    assert filtered_behavior["region"] == "v1"


def test_loader_uses_only_figure_4_d_to_f_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    epoch_tables = object()
    behavior_payload = object()

    def fake_epoch_loader(*args: Any, **kwargs: Any) -> Any:
        calls["epoch"] = (args, kwargs)
        return epoch_tables

    def fake_behavior_loader(*args: Any, **kwargs: Any) -> Any:
        calls["behavior"] = (args, kwargs)
        return behavior_payload

    monkeypatch.setattr(
        figure._figure_3,
        "load_glm_epoch_summary_tables",
        fake_epoch_loader,
    )
    monkeypatch.setattr(
        figure._figure_3,
        "load_glm_dark_activity_devexp_tables",
        fake_behavior_loader,
    )

    payload = figure.load_supplementary_figure_8_data(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        light_epoch="02_r1",
        dark_epoch=None,
        sleep_epoch="07_s4",
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection="single",
        ridge_strength=0.1,
        dark_movement_fr_cache_dir=Path("/cache"),
        refresh_dark_movement_fr_cache=True,
    )

    assert payload == {
        "panel_d_epoch_tables": epoch_tables,
        "panel_ef_behavior_payload": behavior_payload,
    }
    expected_datasets = (("L14", "20240611", "08_r4"),)
    assert calls["epoch"][0] == (Path("/analysis"), expected_datasets)
    assert calls["epoch"][1] == {
        "light_epoch": "02_r1",
        "dark_epoch": None,
        "sleep_epoch": "07_s4",
        "epoch_types": figure._figure_3.PANEL_C_EPOCH_ORDER,
        "ripple_window_s": 0.2,
        "ripple_window_offset_s": 0.0,
        "ripple_selection": "single",
        "ridge_strength": 0.1,
    }
    assert calls["behavior"][0] == (Path("/analysis"), expected_datasets)
    assert calls["behavior"][1] == {
        "light_epoch": "02_r1",
        "dark_epoch": None,
        "sleep_epoch": "07_s4",
        "ripple_window_s": 0.2,
        "ripple_window_offset_s": 0.0,
        "ripple_selection": "single",
        "ridge_strength": 0.1,
        "dark_movement_fr_cache_dir": Path("/cache"),
        "refresh_dark_movement_fr_cache": True,
        "tuning_similarity_metric": (
            figure._figure_3.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
        ),
    }


def test_panel_d_reuses_only_the_original_summary_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fake_plot(ax: Any, epoch_tables: Any, **kwargs: Any) -> None:
        assert epoch_tables == [{"epoch_type": "light"}]
        assert kwargs == {"ripple_trace": None, "prediction_examples": ()}
        ax.inset_axes((0.0, 0.0, 0.3, 0.8)).set_label("schematic")
        ax.inset_axes((0.7, 0.4, 0.3, 0.4)).set_ylabel(
            "-log10 p from shuffle"
        )
        box_ax = ax.inset_axes((0.7, 0.0, 0.3, 0.3))
        box_ax.set_xlabel("Deviance explained")
        box_ax.set_yticks([1, 2], ["n.s.", "$p$<0.05"])

    monkeypatch.setattr(
        figure._figure_3,
        "plot_glm_analysis_panel",
        fake_plot,
    )
    fig, ax = plt.subplots()
    scatter_ax, box_ax = figure.plot_figure_4_panel_d(
        ax,
        [{"epoch_type": "light"}],
    )

    assert tuple(ax.child_axes) == (scatter_ax, box_ax)
    assert [label.get_text() for label in box_ax.get_yticklabels()] == [
        "",
        "$p$<0.05",
    ]
    parent_box = ax.get_position()
    for child_ax, bounds in (
        (scatter_ax, figure.PANEL_D_SCATTER_AXIS_BOUNDS),
        (box_ax, figure.PANEL_D_BOX_AXIS_BOUNDS),
    ):
        child_box = child_ax.get_position()
        assert child_box.x0 == pytest.approx(
            parent_box.x0 + bounds[0] * parent_box.width
        )
        assert child_box.y0 == pytest.approx(
            parent_box.y0 + bounds[1] * parent_box.height
        )
        assert child_box.width == pytest.approx(bounds[2] * parent_box.width)
        assert child_box.height == pytest.approx(bounds[3] * parent_box.height)
    plt.close(fig)


@pytest.mark.parametrize(
    "p_value",
    [
        0.219739,
        0.009,
    ],
)
def test_panel_f_omits_significance_labels(
    monkeypatch: pytest.MonkeyPatch,
    p_value: float,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        figure._figure_3,
        "compute_dark_active_dppi_mean_rank_permutation",
        lambda *_args, **_kwargs: {
            "selected_values": [0.42, 0.55, 0.63],
            "monte_carlo_p_value": p_value,
        },
    )
    fig, ax = plt.subplots()
    histogram_ax = figure.plot_figure_4_panel_f(
        ax,
        {},
        n_permutations=20,
    )

    assert [text.get_text() for text in histogram_ax.texts] == []
    plt.close(fig)


def test_lightweight_render_writes_one_row_per_animal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")
    import matplotlib.pyplot as plt

    animals = ("L15", "L14")
    summary_rows = []
    devexp_rows = []
    reference_rows = []
    dppi_reference_rows = []
    for animal_index, animal_name in enumerate(animals):
        date = "20241121" if animal_name == "L15" else "20240611"
        for unit in range(1, 5):
            p_value = 0.01 if unit <= 3 else 0.4
            devexp = 0.04 * unit + 0.01 * animal_index
            dark_rate = 0.8 if unit in (1, 3, 4) else 0.2
            similarity = 0.15 + 0.18 * unit
            summary_rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": "02_r1",
                    "unit_id": unit,
                    "ripple_devexp_mean": devexp,
                    "ripple_devexp_p_value": p_value,
                }
            )
            devexp_rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "unit": unit,
                    "epoch_type": "light",
                    "ripple_devexp_mean": devexp,
                    "ripple_devexp_p_value": p_value,
                    "dark_firing_rate_hz": dark_rate,
                    "same_turn_tuning_similarity": similarity,
                }
            )
            reference_rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "unit": unit,
                    "dark_firing_rate_hz": dark_rate,
                }
            )
            if dark_rate >= 0.5:
                dppi_reference_rows.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "unit": unit,
                        "same_turn_tuning_similarity": similarity,
                    }
                )

    panel_data = {
        "panel_d_epoch_tables": [
            {
                "epoch_type": "light",
                "label": "Light run",
                "epoch": "02_r1",
                "datasets": (
                    ("L15", "20241121", "02_r1"),
                    ("L14", "20240611", "02_r1"),
                ),
                "n_datasets": 2,
                "summary_table": pd.DataFrame.from_records(summary_rows),
            }
        ],
        "panel_ef_behavior_payload": {
            "devexp_table": pd.DataFrame.from_records(devexp_rows),
            "dark_activity_reference_table": pd.DataFrame.from_records(
                reference_rows
            ),
            "dark_active_dppi_reference_table": pd.DataFrame.from_records(
                dppi_reference_rows
            ),
            "missing_artifacts": [],
            "region": "v1",
            "dark_activity_threshold_hz": 0.5,
            "tuning_comparison_label": "pooled_same_turn",
            "tuning_similarity_metric": "absolute_overlap",
        },
    }
    monkeypatch.setattr(
        figure,
        "load_supplementary_figure_8_data",
        lambda **_kwargs: panel_data,
    )

    output_path = tmp_path / "nested" / "supplementary_figure_8.png"
    open_figures = set(plt.get_fignums())
    saved_path = figure.make_supplementary_figure_8(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[
            ("L15", "20241121", "10_r5"),
            ("L14", "20240611", "08_r4"),
        ],
        light_epoch=None,
        dark_epoch=None,
        sleep_epoch=None,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection="single",
        ridge_strength=0.1,
        dark_movement_fr_cache_dir=Path("/cache"),
        refresh_dark_movement_fr_cache=False,
        dpi=60,
        dppi_n_permutations=20,
        activity_n_permutations=20,
        activity_devexp_batch_size=5,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert set(plt.get_fignums()) == open_figures
