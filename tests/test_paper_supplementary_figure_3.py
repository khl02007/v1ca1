from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.supplementary_figure_3 as figure


def test_main_forwards_cli_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_make_figure(**kwargs: object) -> Path:
        calls.append(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        figure,
        "get_processed_datasets",
        lambda: pytest.fail("explicit --dataset values should be used"),
    )
    monkeypatch.setattr(figure, "make_supplementary_figure_3", fake_make_figure)

    figure.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            "/figures",
            "--output-name",
            "individual_figure_2bc",
            "--format",
            "png",
            "--dataset",
            "L14:20240611:09_r5",
            "--dataset",
            "L15:20241121",
            "--region",
            "ca1",
            "--light-epoch",
            "03_r2",
            "--dark-epoch",
            "11_r6",
            "--panel-tuning-similarity-cache-dir",
            "/cache",
            "--refresh-panel-tuning-similarity-cache",
            "--min-epoch-movement-firing-rate-hz",
            "0.11",
            "--min-path-movement-firing-rate-hz",
            "0.22",
            "--min-segment-mean-firing-rate-hz",
            "0.33",
            "--min-stability-correlation",
            "0.44",
            "--dpi",
            "144",
        ]
    )

    assert calls == [
        {
            "data_root": Path("/analysis"),
            "output_path": Path("/figures/individual_figure_2bc.png"),
            "datasets": [
                ("L14", "20240611", "09_r5"),
                ("L15", "20241121", "10_r5"),
            ],
            "region": "ca1",
            "light_epoch": "03_r2",
            "dark_epoch": "11_r6",
            "dpi": 144,
            "panel_tuning_similarity_cache_dir": Path("/cache"),
            "refresh_panel_tuning_similarity_cache": True,
            "min_epoch_movement_firing_rate_hz": 0.11,
            "min_path_movement_firing_rate_hz": 0.22,
            "min_segment_mean_firing_rate_hz": 0.33,
            "min_stability_correlation": 0.44,
        }
    ]


def test_duplicate_animals_are_rejected_and_labels_are_animal_only() -> None:
    with pytest.raises(ValueError, match="exactly one data set per animal"):
        figure.normalize_individual_animal_datasets(
            [
                ("L14", "20240611", "08_r4"),
                ("L14", "20240612", "09_r5"),
            ]
        )

    assert figure.PANEL_LABELS == ("A", "B")
    assert figure.format_animal_row_label(
        ("L12", "20240421", "08_r4")
    ) == "L12"


def test_panel_data_loader_uses_exact_figure_2_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = Path("/analysis")
    cache_dir = Path("/cache")
    datasets = [
        ("L15", "20241121"),
        ("L14", "20240611", "09_r5"),
    ]
    normalized_datasets = (
        ("L15", "20241121", "10_r5"),
        ("L14", "20240611", "09_r5"),
    )
    shift_table = object()
    raw_overlap_table = object()
    filtered_overlap_table = object()
    calls: dict[str, object] = {}
    legacy_figure_2 = figure.figure_2._figure_2._figure_2

    def fake_shift_loader(**kwargs: object) -> object:
        calls["shift"] = kwargs
        return shift_table

    def fake_overlap_loader(**kwargs: object) -> object:
        calls["overlap"] = kwargs
        return raw_overlap_table

    def fake_overlap_filter(table: object, **kwargs: object) -> object:
        calls["filter"] = (table, kwargs)
        return filtered_overlap_table

    monkeypatch.setattr(
        figure.figure_2,
        "load_or_compute_panel_h_shift_profile_table",
        fake_shift_loader,
    )
    monkeypatch.setattr(
        legacy_figure_2,
        "load_panel_b_tuning_overlap_table",
        fake_overlap_loader,
    )
    monkeypatch.setattr(
        legacy_figure_2,
        "filter_panel_b_overlap_by_even_odd_stability",
        fake_overlap_filter,
    )

    panel_data = figure.load_individual_animal_panel_data(
        data_root=data_root,
        datasets=datasets,
        region="ca1",
        light_epoch="03_r2",
        dark_epoch="11_r6",
        panel_tuning_similarity_cache_dir=cache_dir,
        refresh_panel_tuning_similarity_cache=True,
        min_epoch_movement_firing_rate_hz=0.11,
        min_path_movement_firing_rate_hz=0.22,
        min_segment_mean_firing_rate_hz=0.33,
        min_stability_correlation=0.44,
    )

    assert panel_data == {
        "shift_profile": shift_table,
        "path_invariance": filtered_overlap_table,
    }
    assert calls["shift"] == {
        "data_root": data_root,
        "datasets": normalized_datasets,
        "region": "ca1",
        "light_epoch": "03_r2",
        "dark_epoch": "11_r6",
        "cache_dir": cache_dir,
        "refresh_cache": True,
        "min_epoch_movement_firing_rate_hz": 0.11,
        "min_path_movement_firing_rate_hz": 0.22,
        "min_segment_mean_firing_rate_hz": 0.33,
        "min_stability_correlation": 0.44,
    }
    assert calls["overlap"] == {
        "data_root": data_root,
        "datasets": normalized_datasets,
        "region": "ca1",
        "light_epoch": "03_r2",
        "dark_epoch": "11_r6",
    }
    assert calls["filter"] == (
        raw_overlap_table,
        {
            "data_root": data_root,
            "datasets": normalized_datasets,
            "region": "ca1",
            "light_epoch": "03_r2",
            "dark_epoch": "11_r6",
            "min_movement_firing_rate_hz": (
                legacy_figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
            ),
            "min_stability_correlation": (
                legacy_figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
            ),
        },
    )


def test_make_figure_loads_once_and_filters_rows_in_dataset_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")
    import matplotlib.pyplot as plt

    datasets = [
        ("L15", "20241121", "10_r5"),
        ("L14", "20240611", "08_r4"),
    ]
    panel_data = {
        "shift_profile": pd.DataFrame(
            {
                "animal_name": ["L14", "L15", "L14"],
                "token": ["shift-14-a", "shift-15", "shift-14-b"],
            }
        ),
        "path_invariance": pd.DataFrame(
            {
                "animal_name": ["L14", "L15", "L14"],
                "token": ["pii-14-a", "pii-15", "pii-14-b"],
            }
        ),
    }
    load_calls: list[dict[str, object]] = []
    shift_calls: list[list[str]] = []
    path_invariance_calls: list[list[str]] = []
    events: list[str] = []
    saved: dict[str, object] = {}
    original_close = plt.close

    def fake_load(**kwargs: object) -> dict[str, object]:
        load_calls.append(kwargs)
        return panel_data

    def fake_shift_plot(_axis: object, table: object) -> None:
        shift_calls.append(table["token"].tolist())

    def fake_path_invariance_plot(_axis: object, table: object) -> None:
        path_invariance_calls.append(table["token"].tolist())

    def fail_schematic(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a schematic plotter must not be called")

    def fake_save(
        mpl_figure: object,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        events.append("save")
        mpl_figure.canvas.draw()
        saved["figure"] = mpl_figure
        saved["path"] = output_path
        saved["kwargs"] = kwargs
        saved["size"] = tuple(mpl_figure.get_size_inches())
        saved["texts"] = [
            text.get_text()
            for axis in mpl_figure.axes
            for text in axis.texts
        ]
        saved["titles"] = [
            axis.get_title() for axis in mpl_figure.axes if axis.get_title()
        ]
        return output_path

    def recording_close(mpl_figure: object) -> None:
        events.append("close")
        assert mpl_figure is saved["figure"]
        original_close(mpl_figure)

    monkeypatch.setattr(figure, "load_individual_animal_panel_data", fake_load)
    monkeypatch.setattr(
        figure.figure_2,
        "plot_population_shift_profile",
        fake_shift_plot,
    )
    monkeypatch.setattr(
        figure,
        "plot_path_invariance_data",
        fake_path_invariance_plot,
    )
    monkeypatch.setattr(
        figure.figure_2,
        "plot_panel_b_circular_shift_analysis",
        fail_schematic,
    )
    monkeypatch.setattr(
        figure.figure_2,
        "plot_circular_shift_schematic",
        fail_schematic,
    )
    monkeypatch.setattr(
        figure.figure_2._figure_2._figure_2,
        "plot_panel_b_dpp_overlap_with_schematic",
        fail_schematic,
    )
    monkeypatch.setattr(
        figure.figure_2._figure_2._figure_2,
        "plot_panel_b_dppi_schematic",
        fail_schematic,
    )
    monkeypatch.setattr(
        figure.figure_2._figure_2,
        "_align_panel_b_top_histogram_label_to_scatter",
        lambda *_args: None,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save)
    monkeypatch.setattr(plt, "close", recording_close)

    output_path = tmp_path / "supplementary_figure_3.svg"
    open_figures = set(plt.get_fignums())
    saved_path = figure.make_supplementary_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        dpi=180,
    )

    assert saved_path == output_path
    assert len(load_calls) == 1
    assert load_calls[0]["datasets"] == datasets
    assert load_calls[0]["panel_tuning_similarity_cache_dir"] == (
        tmp_path / "cache"
    )
    assert shift_calls == [["shift-15"], ["shift-14-a", "shift-14-b"]]
    assert path_invariance_calls == [
        ["pii-15"],
        ["pii-14-a", "pii-14-b"],
    ]
    assert saved["path"] == output_path
    assert saved["kwargs"] == {"dpi": 180, "bbox_inches": None}
    assert saved["size"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            2 * figure.DEFAULT_ANIMAL_ROW_HEIGHT_MM / 25.4,
        )
    )
    assert saved["titles"] == list(figure.PANEL_TITLES)
    assert saved["texts"].count("A") == 1
    assert saved["texts"].count("B") == 1
    assert saved["texts"].count("L15") == 1
    assert saved["texts"].count("L14") == 1
    assert "20241121" not in saved["texts"]
    assert "20240611" not in saved["texts"]
    assert "10_r5" not in saved["texts"]
    assert "08_r4" not in saved["texts"]
    assert events == ["save", "close"]
    assert set(plt.get_fignums()) == open_figures


def test_path_invariance_plotter_is_data_only_and_shares_marginal_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legacy_figure_2 = figure.figure_2._figure_2._figure_2
    scatter_calls: list[tuple[object, dict[str, object]]] = []
    format_calls: list[object] = []
    real_formatter = legacy_figure_2._format_panel_b_dppi_scatter_axes

    def fake_scatter(
        axis: object,
        table: object,
        **kwargs: object,
    ) -> None:
        scatter_calls.append((table, kwargs))
        scatter_axis = axis.inset_axes((0.10, 0.10, 0.60, 0.60))
        top_axis = axis.inset_axes((0.10, 0.75, 0.60, 0.20))
        right_axis = axis.inset_axes((0.75, 0.10, 0.20, 0.60))
        scatter_axis.set_xlabel("Dark DPP\noverlap")
        scatter_axis.set_ylabel("Light DPP\noverlap")
        top_axis.set_ylabel("Frac.")
        right_axis.set_xlabel("Frac.")

    def recording_formatter(axis: object) -> None:
        format_calls.append(axis)
        real_formatter(axis)

    def fail_schematic(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a schematic plotter must not be called")

    monkeypatch.setattr(
        legacy_figure_2,
        "plot_panel_b_dpp_overlap_scatter",
        fake_scatter,
    )
    monkeypatch.setattr(
        legacy_figure_2,
        "_format_panel_b_dppi_scatter_axes",
        recording_formatter,
    )
    monkeypatch.setattr(
        legacy_figure_2,
        "plot_panel_b_dpp_overlap_with_schematic",
        fail_schematic,
    )
    monkeypatch.setattr(
        legacy_figure_2,
        "plot_panel_b_dppi_schematic",
        fail_schematic,
    )

    fig, axes = plt.subplots(1, 2)
    tables = (object(), object())
    for axis, table in zip(axes, tables, strict=True):
        figure.plot_path_invariance_data(axis, table)

    assert scatter_calls == [
        (
            table,
            {
                "title": None,
                "show_linear_fit": True,
                "show_r2_annotation": True,
                "equal_aspect": True,
            },
        )
        for table in tables
    ]
    assert format_calls == list(axes)
    assert figure.PATH_INVARIANCE_MARGINAL_FRACTION_LIMIT == 0.21
    for axis in axes:
        scatter_axis = next(
            child for child in axis.child_axes if child.get_xlabel() == "Dark PII"
        )
        top_axis = next(
            child for child in axis.child_axes if child.get_ylabel() == "Frac."
        )
        right_axis = next(
            child for child in axis.child_axes if child.get_xlabel() == "Frac."
        )
        assert scatter_axis.get_ylabel() == "Light PII"
        assert top_axis.get_ylim() == pytest.approx((0.0, 0.21))
        assert top_axis.get_yticks() == pytest.approx((0.0, 0.1, 0.2))
        assert right_axis.get_xlim() == pytest.approx((0.0, 0.21))
        assert right_axis.get_xticks() == pytest.approx((0.0, 0.1, 0.2))
    plt.close(fig)


def test_lightweight_agg_render_saves_nonempty_output_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")
    import matplotlib.pyplot as plt

    animal_name = "L14"
    shift_records = []
    for unit, profile in (
        (1, (0.20, 0.90, 0.25)),
        (2, (0.15, 0.80, 0.30)),
    ):
        for normalized_shift, rescaled_overlap in zip(
            (-0.5, 0.0, 0.5),
            profile,
            strict=True,
        ):
            shift_records.append(
                {
                    "animal_name": animal_name,
                    "date": "20240611",
                    "region": "v1",
                    "unit": unit,
                    "dark_epoch": "08_r4",
                    "light_epoch": "02_r1",
                    "path": "center_to_left",
                    "normalized_shift": normalized_shift,
                    "rescaled_overlap": rescaled_overlap,
                    "rescaling_status": "valid",
                }
            )
    panel_data = {
        "shift_profile": pd.DataFrame.from_records(shift_records),
        "path_invariance": pd.DataFrame(
            {
                "animal_name": [animal_name] * 4,
                "similarity_dark": [0.10, 0.35, 0.60, 0.85],
                "similarity_light": [0.15, 0.30, 0.65, 0.80],
            }
        ),
    }
    monkeypatch.setattr(
        figure,
        "load_individual_animal_panel_data",
        lambda **_kwargs: panel_data,
    )

    open_figures = set(plt.get_fignums())
    output_path = tmp_path / "nested" / "supplementary_figure_3.png"
    saved_path = figure.make_supplementary_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[(animal_name, "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        dpi=60,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert set(plt.get_fignums()) == open_figures
