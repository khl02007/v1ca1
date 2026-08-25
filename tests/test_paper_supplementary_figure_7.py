from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import v1ca1.paper_figures.supplementary_figure_7 as figure


def test_defaults_and_output_path() -> None:
    args = figure.parse_arguments([])

    assert figure.DEFAULT_OUTPUT_NAME == "supplementary_figure_7"
    assert args.output_dir == Path("paper_figures/output")
    assert args.output_name == figure.DEFAULT_OUTPUT_NAME
    assert args.output_format == "pdf"
    assert args.ripple_selection == "single"
    assert args.region is None
    assert (
        args.panel_c_permutations
        == figure.PANEL_C_SOURCE_MIXED_MODEL_PERMUTATIONS
    )
    assert (
        args.panel_c_permutation_seed
        == figure.PANEL_C_SOURCE_MIXED_MODEL_RANDOM_SEED
    )
    assert figure.ANIMALS_PER_ROW == 2
    assert figure.POOLED_PANEL_LABELS == ("A", "B", "C")
    assert figure.INDIVIDUAL_PANEL_LABEL == "D"
    assert figure.get_animal_row_count(4) == 2
    assert figure.get_animal_row_count(3) == 2
    assert figure.get_figure_height_mm(4) == pytest.approx(
        figure.DEFAULT_FIGURE_HEIGHT_MM
    )
    assert figure.get_figure_height_mm(2) == pytest.approx(
        figure.DEFAULT_PANEL_A_ROW_HEIGHT_MM
        + figure.DEFAULT_ANIMAL_ROW_HEIGHT_MM
    )
    assert figure.PANEL_A_HISTOGRAM_BOTTOM == pytest.approx(0.20)
    assert figure.PANEL_A_HISTOGRAM_HEIGHT == pytest.approx(0.70)
    assert figure.build_output_path(
        Path("paper_figures/output"),
        "supplementary_figure_7",
        "svg",
    ) == Path("paper_figures/output/supplementary_figure_7.svg")
    with pytest.raises(ValueError, match="Unknown output format"):
        figure.build_output_path(Path("output"), "figure", "jpg")


def test_panel_c_paired_mixed_model_uses_animal_random_intercept_and_permutation(
) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(7)
    rows = []
    for animal_index, animal_name in enumerate(("L12", "L14", "L15", "L19")):
        mean_activity = rng.normal(0.05, 0.02, 30)
        animal_offset = (-0.01, 0.0, 0.01, 0.02)[animal_index]
        paired_delta = (
            0.03
            + animal_offset
            + rng.normal(0.0, 0.012, mean_activity.size)
        )
        rows.extend(
            {
                "animal_name": animal_name,
                "date": "20200101",
                "unit_id": unit_id,
                "mean_activity_devexp_mean": mean_value,
                "vector_devexp_mean": mean_value + delta_value,
            }
            for unit_id, (mean_value, delta_value) in enumerate(
                zip(mean_activity, paired_delta, strict=True)
            )
        )
    table = pd.DataFrame(rows)

    statistics = figure.compute_source_predictor_paired_mixed_model_permutation(
        table,
        n_permutations=999,
        random_seed=11,
        permutation_batch_size=100,
    )

    assert statistics["test_name"] == (
        "paired_delta_random_intercept_mixed_model_permutation"
    )
    assert statistics["formula"] == (
        "vector_minus_mean_deviance_explained ~ 1 + (1 | animal_name)"
    )
    assert statistics["alternative"] == (
        "vector_greater_than_mean_activity"
    )
    assert statistics["permutation_scheme"] == (
        "source_model_labels_swapped_independently_within_v1_unit"
    )
    assert statistics["n_finite_pairs"] == 120
    assert statistics["n_animals"] == 4
    assert statistics["converged"] is True
    assert statistics["coefficient"] == pytest.approx(
        statistics["permutation_observed_coefficient"],
        rel=5e-5,
    )
    assert statistics["coefficient"] > 0.0
    assert statistics["p_value"] <= 0.01
    assert abs(statistics["permutation_null_mean"]) < 0.002
    assert len(statistics["null_coefficients"]) == 999


def test_panel_c_plot_uses_mixed_model_statistics_without_pooled_sign_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd = pytest.importorskip("pandas")
    from v1ca1.paper_figures import _ripple_panels

    table = pd.DataFrame(
        {
            "animal_name": ["L12", "L14", "L15", "L19"],
            "date": ["d"] * 4,
            "mean_activity_devexp_mean": [0.01, 0.02, 0.03, 0.04],
            "vector_devexp_mean": [0.05, 0.06, 0.07, 0.08],
            "vector_devexp_p_value": [0.01] * 4,
        }
    )
    mixed_model_statistics = {
        "test_name": "paired_delta_random_intercept_mixed_model_permutation",
        "n_finite_pairs": 4,
        "fraction_vector_greater": 1.0,
        "p_value": 0.004,
    }

    def fail_pooled_sign_test(_table: object) -> object:
        raise AssertionError("Panel C calculated the removed pooled sign test")

    monkeypatch.setattr(
        _ripple_panels,
        "compute_source_predictor_paired_sign_test",
        fail_pooled_sign_test,
    )
    fig, ax = plt.subplots()
    returned = _ripple_panels.plot_glm_source_predictor_comparison_panel(
        ax,
        {"comparison_table": table},
        include_per_animal=False,
        include_pooled=True,
        compact_labels=True,
        show_color_note=False,
        annotate_pooled_inference=True,
        pooled_statistics=mixed_model_statistics,
    )

    assert returned is not None
    assert returned["test_name"] == mixed_model_statistics["test_name"]
    assert "**" in {
        text.get_text()
        for child_axis in ax.child_axes
        for text in child_axis.texts
    }
    plt.close(fig)


def test_grouping_and_filters_preserve_order_and_select_one_animal() -> None:
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
            "epochs": ("08_r4", "10_r5", "08_r4"),
            "firing_rate_table": table,
            "summary_table": table,
        }
    ]
    filtered_epochs = figure.filter_epoch_tables_by_animal(
        epoch_payloads,
        "L14",
    )
    assert filtered_epochs[0]["datasets"] == (datasets[0], datasets[2])
    assert filtered_epochs[0]["n_datasets"] == 2
    assert filtered_epochs[0]["epochs"] == ("08_r4", "08_r4")
    for key in ("firing_rate_table", "summary_table"):
        assert filtered_epochs[0][key]["token"].tolist() == [
            "14-a",
            "14-b",
        ]

    source_payload = {
        "comparison_table": table,
        "missing_artifacts": [
            {"animal_name": "L14", "artifact": "a"},
            {"animal_name": "L15", "artifact": "b"},
        ],
        "ripple_selection": "single",
    }
    filtered_source = figure.filter_source_payload_by_animal(
        source_payload,
        "L15",
    )
    assert filtered_source["comparison_table"]["token"].tolist() == ["15"]
    assert filtered_source["missing_artifacts"] == [
        {"animal_name": "L15", "artifact": "b"}
    ]
    assert filtered_source["ripple_selection"] == "single"


def test_make_figure_draws_pooled_and_per_animal_b_and_c(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")

    datasets = [
        ("L12", "20240421", "08_r4"),
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L19", "20250930", "08_r4"),
    ]
    animal_names = [dataset[0] for dataset in datasets]
    table = pd.DataFrame(
        {
            "animal_name": animal_names,
            "date": [dataset[1] for dataset in datasets],
            "token": [f"{animal}-row" for animal in animal_names],
        }
    )
    heatmap_tables = [
        {
            "epoch_type": "light",
            "label": "Light",
            "datasets": tuple(datasets),
            "n_datasets": len(datasets),
            "epochs": tuple("02_r1" for _dataset in datasets),
            "firing_rate_table": table,
            "summary_table": table,
        }
    ]
    panel_a_tables = list(heatmap_tables)
    source_payload = {
        "comparison_table": table,
        "missing_artifacts": [],
        "ripple_selection": "single",
    }
    calls: dict[str, Any] = {}

    def fake_load_heatmaps(*args: object, **kwargs: object) -> object:
        calls["heatmap_load"] = (args, kwargs)
        return heatmap_tables

    def fake_filter(payload: object, order: object) -> object:
        calls["filter"] = (payload, order)
        return panel_a_tables

    def fake_load_source(*args: object, **kwargs: object) -> object:
        calls["source_load"] = (args, kwargs)
        return source_payload

    pooled_mixed_model_statistics = {
        "test_name": "paired_delta_random_intercept_mixed_model_permutation",
        "p_value": 0.02,
        "coefficient": 0.03,
        "n_finite_pairs": 4,
        "n_animals": 4,
    }

    def fake_compute_mixed_model(
        comparison_table: object,
        **kwargs: object,
    ) -> dict[str, object]:
        calls["mixed_model"] = (comparison_table, kwargs)
        return pooled_mixed_model_statistics

    def fake_plot_heatmap(
        axis: Any,
        tables: object,
        **kwargs: object,
    ) -> None:
        calls["heatmap_plot"] = (axis, tables, kwargs)

    def fake_plot_modulation(
        axis: Any,
        tables: object,
        **kwargs: object,
    ) -> list[object]:
        calls.setdefault("modulation_plots", []).append(
            (axis, tables, kwargs)
        )
        return []

    def fake_plot_source(
        axis: Any,
        payload: object,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.setdefault("source_plots", []).append((axis, payload, kwargs))
        return {
            "p_value": 0.01,
            "n_vector_greater": 9,
            "n_tested": 10,
            "n_ties": 1,
        }

    def fake_save(
        mpl_figure: Any,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        mpl_figure.canvas.draw()
        calls["output_path"] = output_path
        calls["save_kwargs"] = kwargs
        calls["figsize"] = tuple(mpl_figure.get_size_inches())
        calls["figure_text_list"] = [
            text.get_text() for text in mpl_figure.texts
        ]
        calls["figure_text"] = set(calls["figure_text_list"])
        animal_label_artists = [
            text
            for axis in mpl_figure.axes
            for text in axis.texts
            if text.get_text() in set(animal_names)
        ]
        calls["animal_labels"] = [
            text.get_text() for text in animal_label_artists
        ]
        calls["animal_label_positions"] = [
            text.get_position() for text in animal_label_artists
        ]
        calls["animal_label_axis_bounds"] = [
            text.axes.get_position().bounds for text in animal_label_artists
        ]
        calls["animal_label_weights"] = [
            text.get_fontweight() for text in animal_label_artists
        ]
        calls["animal_label_boxes"] = [
            text.get_bbox_patch() for text in animal_label_artists
        ]
        return output_path

    monkeypatch.setattr(
        figure,
        "load_pooled_ripple_heatmap_epoch_tables",
        fake_load_heatmaps,
    )
    monkeypatch.setattr(figure, "filter_epoch_payloads", fake_filter)
    monkeypatch.setattr(
        figure,
        "load_glm_source_predictor_comparison_tables",
        fake_load_source,
    )
    monkeypatch.setattr(
        figure,
        "compute_source_predictor_paired_mixed_model_permutation",
        fake_compute_mixed_model,
    )
    monkeypatch.setattr(
        figure,
        "plot_epoch_ripple_heatmap_panel",
        fake_plot_heatmap,
    )
    monkeypatch.setattr(
        figure,
        "plot_epoch_modulation_histogram_panel",
        fake_plot_modulation,
    )
    monkeypatch.setattr(
        figure,
        "plot_glm_source_predictor_comparison_panel",
        fake_plot_source,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save)

    output_path = tmp_path / "supplementary_figure_7.svg"
    saved_path = figure.make_supplementary_figure_7(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        regions=("v1", "ca1"),
        light_epoch=None,
        dark_epoch=None,
        sleep_epoch=None,
        ripple_threshold_zscore=None,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection="single",
        ridge_strength=0.1,
        dpi=200,
        source_comparison_n_permutations=321,
        source_comparison_permutation_seed=17,
        source_comparison_permutation_batch_size=23,
    )

    assert saved_path == output_path
    assert calls["filter"] == (heatmap_tables, figure.PANEL_A_EPOCH_ORDER)
    heatmap_axis, heatmap_input, heatmap_kwargs = calls["heatmap_plot"]
    assert heatmap_input is panel_a_tables
    assert heatmap_kwargs == {
        "regions": ("v1", "ca1"),
        "expand_heatmaps_vertically": True,
        "show_modulation_histogram": False,
        "heatmap_vertical_bounds": figure.PANEL_A_HEATMAP_VERTICAL_BOUNDS,
    }

    modulation_plots = calls["modulation_plots"]
    source_plots = calls["source_plots"]
    assert len(modulation_plots) == 1 + len(datasets)
    assert len(source_plots) == 1 + len(datasets)
    pooled_modulation_axis, pooled_tables, pooled_modulation_kwargs = (
        modulation_plots[0]
    )
    pooled_source_axis, pooled_payload, pooled_source_kwargs = source_plots[0]
    assert pooled_tables is panel_a_tables
    assert pooled_payload is source_payload
    assert pooled_modulation_kwargs == {
        "regions": ("v1", "ca1"),
        "bottom": figure.PANEL_A_HISTOGRAM_BOTTOM,
        "height": figure.PANEL_A_HISTOGRAM_HEIGHT,
    }
    assert pooled_source_kwargs == {
        "include_per_animal": False,
        "include_pooled": True,
        "compact_labels": True,
        "show_color_note": False,
        "annotate_pooled_inference": True,
        "pooled_statistics": pooled_mixed_model_statistics,
    }
    assert calls["mixed_model"] == (
        table,
        {
            "n_permutations": 321,
            "random_seed": 17,
            "permutation_batch_size": 23,
        },
    )
    for animal_name, modulation_call, source_call in zip(
        animal_names,
        modulation_plots[1:],
        source_plots[1:],
        strict=True,
    ):
        _modulation_axis, animal_tables, modulation_kwargs = modulation_call
        _source_axis, animal_payload, source_kwargs = source_call
        for payload in animal_tables:
            assert set(payload["summary_table"]["animal_name"]) == {
                animal_name
            }
        assert set(animal_payload["comparison_table"]["animal_name"]) == {
            animal_name
        }
        assert modulation_kwargs == {
            "regions": ("v1", "ca1"),
            "bottom": figure.PANEL_A_HISTOGRAM_BOTTOM,
            "height": figure.PANEL_A_HISTOGRAM_HEIGHT,
        }
        assert source_kwargs == {
            "include_per_animal": False,
            "include_pooled": True,
            "compact_labels": True,
            "show_color_note": False,
            "annotate_pooled_sign_test": False,
            "axis_limits": (
                figure.L14_PANEL_C_AXIS_LIMITS
                if animal_name == "L14"
                else None
            ),
        }

    block_bounds = [
        (
            modulation_call[0].get_position().bounds,
            source_call[0].get_position().bounds,
        )
        for modulation_call, source_call in zip(
            modulation_plots[1:],
            source_plots[1:],
            strict=True,
        )
    ]
    for panel_b_bounds, panel_c_bounds in block_bounds:
        assert panel_b_bounds[0] + panel_b_bounds[2] < panel_c_bounds[0]
        assert panel_b_bounds[1] == pytest.approx(panel_c_bounds[1])
    assert block_bounds[0][0][1] == pytest.approx(block_bounds[1][0][1])
    assert block_bounds[2][0][1] == pytest.approx(block_bounds[3][0][1])
    assert block_bounds[0][0][1] > block_bounds[2][0][1]
    assert block_bounds[0][1][0] < block_bounds[1][0][0]
    assert block_bounds[2][1][0] < block_bounds[3][0][0]
    assert block_bounds[0][0][0] == pytest.approx(
        block_bounds[2][0][0],
        abs=0.002,
    )
    assert block_bounds[1][0][0] == pytest.approx(
        block_bounds[3][0][0],
        abs=0.002,
    )
    pooled_bounds = [
        axis.get_position().bounds
        for axis in (heatmap_axis, pooled_modulation_axis, pooled_source_axis)
    ]
    assert pooled_bounds[0][0] + pooled_bounds[0][2] < pooled_bounds[1][0]
    assert pooled_bounds[1][0] + pooled_bounds[1][2] < pooled_bounds[2][0]
    assert pooled_bounds[0][1] == pytest.approx(pooled_bounds[1][1])
    assert pooled_bounds[1][1] == pytest.approx(pooled_bounds[2][1])
    assert all(bounds[1] > block_bounds[0][0][1] for bounds in pooled_bounds)

    normalized_datasets = tuple(datasets)
    heatmap_load_args, heatmap_load_kwargs = calls["heatmap_load"]
    assert heatmap_load_args == (Path("/analysis"), normalized_datasets)
    assert heatmap_load_kwargs["ripple_threshold_zscore"] is None
    source_load_args, source_load_kwargs = calls["source_load"]
    assert source_load_args == (Path("/analysis"), normalized_datasets)
    assert source_load_kwargs["epoch_types"] == figure.PANEL_E_GLM_EPOCH_ORDER
    assert source_load_kwargs["ripple_selection"] == "single"
    assert calls["output_path"] == output_path
    assert calls["save_kwargs"]["dpi"] == 200
    output_bbox = calls["save_kwargs"]["bbox_inches"]
    assert output_bbox.bounds == pytest.approx(
        (
            0.0,
            figure.OUTPUT_BOTTOM_CROP_MM / 25.4,
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            (
                figure.DEFAULT_FIGURE_HEIGHT_MM
                - figure.OUTPUT_BOTTOM_CROP_MM
            )
            / 25.4,
        )
    )
    assert calls["figsize"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            figure.DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    assert calls["animal_labels"] == animal_names
    assert calls["animal_label_positions"] == [
        (figure.ANIMAL_LABEL_X, figure.ANIMAL_LABEL_Y)
    ] * len(animal_names)
    for label_bounds, (panel_b_bounds, _panel_c_bounds) in zip(
        calls["animal_label_axis_bounds"],
        block_bounds,
        strict=True,
    ):
        assert label_bounds[0] + label_bounds[2] < panel_b_bounds[0]
        assert label_bounds[1] == pytest.approx(panel_b_bounds[1])
    assert calls["animal_label_weights"] == ["bold"] * len(animal_names)
    assert all(box is None for box in calls["animal_label_boxes"])
    assert {"A", "B", "C", "D", *figure.PANEL_TITLES} <= calls[
        "figure_text"
    ]
    assert calls["figure_text_list"].count(figure.PANEL_TITLES[1]) == 1
    assert calls["figure_text_list"].count(figure.PANEL_TITLES[2]) == 1
    assert calls["figure_text_list"].count(
        figure.INDIVIDUAL_PANEL_LABEL
    ) == 1
