from __future__ import annotations

from pathlib import Path

import pytest

from v1ca1.paper_figures import _figure_2_panels as figure_2
from v1ca1.paper_figures import supplementary_figure_5
from v1ca1.paper_figures import supplementary_figure_4 as figure


def _make_decoding_summary_table(
    *,
    animal_name: str = "pooled",
    date: str = "pooled",
    medians: dict[tuple[str, str], float] | None = None,
):
    """Return a compact synthetic cross-path/place decoding summary."""
    pandas = pytest.importorskip("pandas")

    if medians is None:
        medians = {
            ("cross_trajectory", "light"): 0.30,
            ("cross_trajectory", "dark"): 0.20,
            ("place", "light"): 0.12,
            ("place", "dark"): 0.08,
        }
    cross_comparison = figure_2.PANEL_E_CROSS_COMPARISONS[0][0]
    rows = []
    for analysis in ("cross_trajectory", "place"):
        comparison = cross_comparison if analysis == "cross_trajectory" else "place"
        for epoch_type in ("light", "dark"):
            median = float(medians[(analysis, epoch_type)])
            rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch_type": epoch_type,
                    "epoch": "02_r1" if epoch_type == "light" else "08_r4",
                    "analysis": analysis,
                    "comparison": comparison,
                    "comparison_label": (
                        "Cross" if analysis == "cross_trajectory" else "Place"
                    ),
                    "q25_error": max(0.0, median - 0.01),
                    "median_error": median,
                    "q75_error": median + 0.01,
                    "n_samples": 50,
                }
            )
    return pandas.DataFrame(rows)


def test_default_cli_matches_requested_source_panels() -> None:
    args = figure.parse_arguments([])

    assert args.output_dir == figure.DEFAULT_OUTPUT_DIR
    assert args.output_name == "supplementary_figure_4"
    assert args.output_format == figure.DEFAULT_OUTPUT_FORMAT
    assert args.region == "v1"
    assert args.light_epoch is None
    assert args.dark_epoch is None
    assert (
        args.decoding_n_permutations
        == figure_2.DECODING_PERMUTATION_COUNT
    )
    assert args.decoding_permutation_seed == figure_2.DECODING_PERMUTATION_SEED
    assert figure.DEFAULT_FIGURE_WIDTH_MM == figure_2.DEFAULT_FIGURE_WIDTH_MM
    assert figure.DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        (
            1.0
            + figure.DEFAULT_ANIMAL_COUNT
            / figure.PANEL_C_ANIMALS_PER_ROW
        )
        * figure_2.PANEL_D_ROW_HEIGHT_MM
    )
    assert figure.PANEL_C_ANIMALS_PER_ROW == 2
    assert figure.get_panel_c_row_count(4) == 2
    assert figure.get_panel_c_row_count(3) == 2
    assert figure.get_figure_height_mm(2) == pytest.approx(
        2.0 * figure_2.PANEL_D_ROW_HEIGHT_MM
    )
    assert figure.PANEL_TITLES[:2] == (
        figure.DECODING_PANEL_TITLE,
        supplementary_figure_5.PANEL_A_CV_PCA_TITLE,
    )
    assert len(figure.PANEL_TITLES) == 3
    assert figure.PANEL_TITLES[2]
    assert not hasattr(args, "panel_a_cache_dir")
    assert not hasattr(args, "position_bin_count")


@pytest.mark.parametrize(
    "datasets",
    [
        [
            ("L12", "20240421", "08_r4"),
            ("L14", "20240611", "08_r4"),
            ("L15", "20241121", "10_r5"),
            ("L19", "20250930", "08_r4"),
        ],
        [
            ("L14", "20240611", "08_r4"),
            ("L19", "20250930", "08_r4"),
        ],
    ],
)
def test_load_panel_a_decoding_data_includes_each_requested_animal(
    monkeypatch: pytest.MonkeyPatch,
    datasets: list[tuple[str, str, str]],
) -> None:
    calls: dict[str, object] = {}

    def fake_load_decoding_error_table(**kwargs: object):
        calls.setdefault("decoding_error_kwargs", []).append(kwargs)
        table = _make_decoding_summary_table()
        calls.setdefault("decoding_error_tables", []).append(table)
        return table

    def fake_build_trial_error_table(**kwargs: object) -> str:
        calls["trial_error_kwargs"] = kwargs
        return "trial-error"

    def fake_compute_permutation_tests(table: object, **kwargs: object):
        pandas = pytest.importorskip("pandas")
        calls["permutation_table"] = table
        calls["permutation_kwargs"] = kwargs
        return pandas.DataFrame(
            {
                "animal_name": [
                    animal_name
                    for animal_name, _date, _epoch in datasets
                    for _analysis in ("cross_trajectory", "place")
                ],
                "analysis": [
                    analysis
                    for _dataset in datasets
                    for analysis in ("cross_trajectory", "place")
                ],
            }
        )

    def fake_build_significance_labels(
        table: object,
        **kwargs: object,
    ) -> tuple[str, str]:
        animal_names = tuple(kwargs["animal_names"])
        calls.setdefault("significance_calls", []).append(
            (table.copy(), {"animal_names": animal_names})
        )
        if len(animal_names) == 1:
            animal_name = animal_names[0]
            return f"{animal_name}-cross", f"{animal_name}-place"
        return "pooled-cross", "pooled-place"

    monkeypatch.setattr(
        figure,
        "load_panel_e_decoding_error_table",
        fake_load_decoding_error_table,
    )
    monkeypatch.setattr(
        figure_2,
        "build_panel_e_decoding_trial_error_table",
        fake_build_trial_error_table,
    )
    monkeypatch.setattr(
        figure_2,
        "compute_panel_e_decoding_permutation_tests",
        fake_compute_permutation_tests,
    )
    monkeypatch.setattr(
        figure_2,
        "build_panel_e_decoding_significance_labels",
        fake_build_significance_labels,
    )

    result = figure.load_panel_a_decoding_data(
        data_root=Path("/analysis"),
        datasets=datasets,
        region="v1",
        light_epoch="02_r1",
        dark_epoch=None,
        decoding_n_permutations=17,
        decoding_permutation_seed=29,
    )

    common_loader_kwargs = {
        "data_root": Path("/analysis"),
        "region": "v1",
        "light_epoch": "02_r1",
        "dark_epoch": None,
    }
    expected_decoding_calls = [
        {**common_loader_kwargs, "datasets": tuple(datasets)},
        *(
            {**common_loader_kwargs, "datasets": (dataset,)}
            for dataset in datasets
        ),
    ]
    assert result["decoding_error"].equals(calls["decoding_error_tables"][0])
    assert result["decoding_significance_labels"] == (
        "pooled-cross",
        "pooled-place",
    )
    assert calls["decoding_error_kwargs"] == expected_decoding_calls
    assert calls["trial_error_kwargs"] == {
        **common_loader_kwargs,
        "datasets": tuple(datasets),
    }
    individual_table = result["individual_decoding_error"]
    assert individual_table["animal_name"].tolist() == [
        animal_name
        for animal_name, _date, _epoch in datasets
        for _row in range(4)
    ]
    assert individual_table["date"].tolist() == [
        date for _animal_name, date, _epoch in datasets for _row in range(4)
    ]
    assert calls["permutation_table"] == "trial-error"
    assert calls["permutation_kwargs"] == {
        "n_permutations": 17,
        "seed": 29,
    }
    expected_animal_names = tuple(dataset[0] for dataset in datasets)
    significance_calls = calls["significance_calls"]
    assert significance_calls[0][1] == {
        "animal_names": expected_animal_names
    }
    assert set(significance_calls[0][0]["animal_name"]) == set(
        expected_animal_names
    )
    assert len(significance_calls) == 1 + len(datasets)
    for (table, kwargs), animal_name in zip(
        significance_calls[1:],
        expected_animal_names,
        strict=True,
    ):
        assert kwargs == {"animal_names": (animal_name,)}
        assert set(table["animal_name"]) == {animal_name}
    assert result["individual_decoding_significance_labels"] == {
        animal_name: (f"{animal_name}-cross", f"{animal_name}-place")
        for animal_name in expected_animal_names
    }


@pytest.mark.parametrize(
    ("datasets", "n_permutations", "seed", "message"),
    [
        ([("L14", "20240611", "08_r4")], 0, 1, "must be positive"),
        ([("L14", "20240611", "08_r4")], 1, -1, "must be non-negative"),
        ([], 1, 1, "exactly one data set per animal"),
        (
            [
                ("L14", "20240611", "08_r4"),
                ("L14", "20240612", "08_r4"),
            ],
            1,
            1,
            "exactly one data set per animal",
        ),
    ],
)
def test_load_panel_a_decoding_data_validates_inference_inputs(
    datasets: list[tuple[str, str, str]],
    n_permutations: int,
    seed: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        figure.load_panel_a_decoding_data(
            data_root=Path("/analysis"),
            datasets=datasets,
            region="v1",
            light_epoch=None,
            dark_epoch=None,
            decoding_n_permutations=n_permutations,
            decoding_permutation_seed=seed,
        )


def test_shared_panel_a_plotter_retains_pooled_summary_and_brackets() -> None:
    np = pytest.importorskip("numpy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection, PathCollection
    import matplotlib.pyplot as plt

    pooled_table = _make_decoding_summary_table()
    fig, ax = plt.subplots()

    figure_2.plot_panel_e2_decoding_panel(
        ax,
        pooled_table,
        significance_labels=("**", "****"),
    )

    assert len(ax.child_axes) == 2
    assert ax.child_axes[0].get_ylim() == pytest.approx((0.0, 0.5))
    assert ax.child_axes[1].get_ylim() == pytest.approx((0.0, 0.2))
    for child_axis, bracket_label in zip(
        ax.child_axes,
        ("**", "****"),
        strict=True,
    ):
        assert bracket_label in [text.get_text() for text in child_axis.texts]
        assert child_axis.get_yscale() == "linear"
        bracket_lines = [
            line
            for line in child_axis.lines
            if len(line.get_xdata()) == 4
            and np.allclose(line.get_xdata(), [1.0, 1.0, 2.0, 2.0])
        ]
        assert len(bracket_lines) == 1
        assert not [
            line for line in child_axis.lines if len(line.get_xdata()) == 2
        ]

        markers = [
            collection
            for collection in child_axis.collections
            if isinstance(collection, PathCollection)
            and len(collection.get_offsets()) == 1
        ]
        assert len(markers) == 2
        assert sorted(
            float(collection.get_offsets()[0, 0]) for collection in markers
        ) == pytest.approx([1.0, 2.0])

        intervals = [
            np.asarray(segment, dtype=float)
            for collection in child_axis.collections
            if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
            if np.asarray(segment).shape == (2, 2)
            and np.allclose(segment[0][0], segment[1][0])
        ]
        assert len(intervals) == 2
    plt.close(fig)


def test_shared_decoding_plotter_applies_custom_linear_limits_before_iqrs() -> None:
    np = pytest.importorskip("numpy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    table = _make_decoding_summary_table()
    table.loc[
        table["analysis"].astype(str) == "cross_trajectory",
        "q75_error",
    ] = [0.398, 0.515]
    table.loc[
        table["analysis"].astype(str) == "place",
        ["q25_error", "median_error", "q75_error"],
    ] = [
        [0.009, 0.024, 1.861],
        [0.038, 0.928, 1.984],
    ]
    fig, ax = plt.subplots()

    figure_2.plot_panel_e2_decoding_panel(
        ax,
        table,
        significance_labels=("**", "****"),
        cross_ylim=(0.0, 0.6),
        place_ylim=(0.0, 2.2),
    )

    assert len(ax.child_axes) == 2
    assert ax.child_axes[0].get_ylim() == pytest.approx((0.0, 0.6))
    assert ax.child_axes[1].get_ylim() == pytest.approx((0.0, 2.2))
    assert all(child.get_yscale() == "linear" for child in ax.child_axes)
    expected_q75 = (0.515, 1.984)
    for child_axis, expected_upper in zip(
        ax.child_axes,
        expected_q75,
        strict=True,
    ):
        interval_endpoints = [
            float(np.asarray(segment, dtype=float)[:, 1].max())
            for collection in child_axis.collections
            if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
            if np.asarray(segment).shape == (2, 2)
            and np.allclose(segment[0][0], segment[1][0])
        ]
        assert max(interval_endpoints) == pytest.approx(expected_upper)
    plt.close(fig)


@pytest.mark.parametrize(
    (
        "animal_name",
        "cross_q75",
        "place_q75",
        "expected_cross",
        "expected_place",
    ),
    [
        ("L12", 0.398, 1.154, (0.0, 0.5), (0.0, 0.2)),
        ("L14", 0.515, 1.411, (0.0, 0.6), (0.0, 0.2)),
        ("L15", 0.494, 1.484, (0.0, 0.55), (0.0, 0.2)),
        ("L19", 0.326, 1.984, (0.0, 0.5), (0.0, 1.5)),
    ],
)
def test_panel_c_ylims_use_requested_place_ranges(
    animal_name: str,
    cross_q75: float,
    place_q75: float,
    expected_cross: tuple[float, float],
    expected_place: tuple[float, float],
) -> None:
    table = _make_decoding_summary_table(animal_name=animal_name)
    table.loc[
        table["analysis"].astype(str) == "cross_trajectory",
        "q75_error",
    ] = cross_q75
    table.loc[
        table["analysis"].astype(str) == "place",
        "q75_error",
    ] = place_q75

    cross_ylim, place_ylim = figure.get_panel_c_decoding_ylims(table)

    assert cross_ylim == pytest.approx(expected_cross)
    assert place_ylim == pytest.approx(expected_place)
    assert cross_ylim[1] > cross_q75


def test_main_builds_named_output_and_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_make_supplementary_figure_4(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        figure,
        "make_supplementary_figure_4",
        fake_make_supplementary_figure_4,
    )
    figure.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            str(tmp_path),
            "--format",
            "svg",
            "--dataset",
            "L14:20240611:08_r4",
            "--region",
            "ca1",
            "--light-epoch",
            "02_r1",
            "--dark-epoch",
            "08_r4",
            "--decoding-n-permutations",
            "19",
            "--decoding-permutation-seed",
            "23",
            "--dpi",
            "144",
        ]
    )

    assert calls == {
        "data_root": Path("/analysis"),
        "output_path": tmp_path / "supplementary_figure_4.svg",
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "ca1",
        "light_epoch": "02_r1",
        "dark_epoch": "08_r4",
        "dpi": 144,
        "decoding_n_permutations": 19,
        "decoding_permutation_seed": 23,
    }


def test_make_supplementary_figure_4_draws_requested_panels_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pandas = pytest.importorskip("pandas")

    calls: dict[str, object] = {}
    datasets = [
        ("L12", "20240421", "08_r4"),
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L19", "20250930", "08_r4"),
    ]
    individual_table = pandas.concat(
        [
            _make_decoding_summary_table(
                animal_name=animal_name,
                date=date,
            )
            for animal_name, date, _epoch in datasets
        ],
        ignore_index=True,
    )
    cross_q75_by_animal = {
        "L12": 0.398,
        "L14": 0.515,
        "L15": 0.494,
        "L19": 0.326,
    }
    place_q75_by_animal = {
        "L12": 1.154,
        "L14": 1.411,
        "L15": 1.484,
        "L19": 1.984,
    }
    for animal_name, q75 in cross_q75_by_animal.items():
        mask = (
            individual_table["animal_name"].astype(str) == animal_name
        ) & (
            individual_table["analysis"].astype(str) == "cross_trajectory"
        )
        individual_table.loc[mask, "q75_error"] = q75
    for animal_name, q75 in place_q75_by_animal.items():
        mask = (
            individual_table["animal_name"].astype(str) == animal_name
        ) & (individual_table["analysis"].astype(str) == "place")
        individual_table.loc[mask, "q75_error"] = q75
    individual_labels = {
        animal_name: (f"{animal_name}-cross", f"{animal_name}-place")
        for animal_name, _date, _epoch in datasets
    }

    def fake_apply_paper_style() -> None:
        calls["styled"] = True

    def fake_load_panel_a_decoding_data(**kwargs: object) -> dict[str, object]:
        calls["panel_a_load_kwargs"] = kwargs
        return {
            "decoding_error": "pooled-decoding-error",
            "individual_decoding_error": individual_table,
            "decoding_significance_labels": ("**", "****"),
            "individual_decoding_significance_labels": individual_labels,
        }

    def fake_plot_decoding_panel(ax, table: object, **kwargs: object) -> None:
        calls.setdefault("decoding_plot_calls", []).append(
            {"axis": ax, "table": table, "kwargs": kwargs}
        )
        ax.text(0.5, 0.5, "decoding")

    def fake_load_cv_pca_table(**kwargs: object) -> str:
        calls["panel_b_load_kwargs"] = kwargs
        return "cv-pca-table"

    def fake_plot_cv_pca_panel(ax, table: object) -> None:
        calls["panel_b_axis"] = ax
        calls["panel_b_table"] = table
        ax.text(0.5, 0.5, "cvPCA")

    def fail_if_omitted_panel_is_loaded(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Supplementary Figure 4 loaded an omitted panel")

    def fake_save_figure(fig, output_path: Path, dpi: int, **kwargs: object) -> Path:
        calls["figsize"] = fig.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["panel_labels"] = [
            text.get_text()
            for ax in fig.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        animal_label_texts = [
            text
            for ax in fig.axes
            for text in ax.texts
            if text.get_text() in {dataset[0] for dataset in datasets}
        ]
        calls["animal_labels"] = [
            text.get_text() for text in animal_label_texts
        ]
        calls["animal_label_positions"] = [
            text.get_position() for text in animal_label_texts
        ]
        calls["animal_label_colors"] = [
            text.get_color() for text in animal_label_texts
        ]
        calls["titles"] = [
            ax.get_title() for ax in fig.axes if ax.get_title()
        ]
        panel_c_title = next(
            ax.title
            for ax in fig.axes
            if ax.get_title() == figure.PANEL_TITLES[2]
        )
        title_bbox = panel_c_title.get_window_extent(
            fig.canvas.get_renderer()
        )
        calls["panel_c_title_center_x"] = (
            title_bbox.x0 + title_bbox.width / 2.0
        ) / fig.bbox.width
        calls["bounds"] = [ax.get_position().bounds for ax in fig.axes]
        return output_path

    monkeypatch.setattr(figure, "apply_paper_style", fake_apply_paper_style)
    monkeypatch.setattr(
        figure,
        "load_panel_a_decoding_data",
        fake_load_panel_a_decoding_data,
    )
    monkeypatch.setattr(
        figure_2,
        "plot_panel_e2_decoding_panel",
        fake_plot_decoding_panel,
    )
    monkeypatch.setattr(
        supplementary_figure_5,
        "load_panel_a_cv_pca_participation_ratio_table",
        fake_load_cv_pca_table,
    )
    monkeypatch.setattr(
        supplementary_figure_5,
        "plot_panel_a_cv_pca_participation_ratios",
        fake_plot_cv_pca_panel,
    )
    monkeypatch.setattr(
        supplementary_figure_5,
        "load_panel_b_motor_progression_table",
        fail_if_omitted_panel_is_loaded,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_4.svg"
    saved_path = figure.make_supplementary_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        dpi=300,
        decoding_n_permutations=17,
        decoding_permutation_seed=29,
    )

    assert saved_path == output_path
    assert calls["styled"] is True
    assert calls["figsize"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            figure.DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == ["A", "B", "C"]
    assert calls["titles"] == list(figure.PANEL_TITLES)
    assert calls["animal_labels"] == [dataset[0] for dataset in datasets]
    assert calls["animal_label_positions"] == [
        (figure.PANEL_C_ANIMAL_LABEL_X, 0.5)
    ] * len(datasets)
    assert calls["animal_label_colors"] == [
        figure.PANEL_C_ANIMAL_LABEL_COLOR
    ] * len(datasets)
    panel_a_bounds, panel_b_bounds, *panel_c_bounds = calls["bounds"]
    assert panel_a_bounds[0] < panel_b_bounds[0]
    assert panel_a_bounds[1] == pytest.approx(panel_b_bounds[1])
    assert panel_c_bounds[0][1] + panel_c_bounds[0][3] < min(
        panel_a_bounds[1],
        panel_b_bounds[1],
    )
    assert panel_c_bounds[0][1] == pytest.approx(panel_c_bounds[1][1])
    assert panel_c_bounds[2][1] == pytest.approx(panel_c_bounds[3][1])
    assert panel_c_bounds[0][1] > panel_c_bounds[2][1]
    assert panel_c_bounds[0][0] < panel_c_bounds[1][0]
    assert panel_c_bounds[2][0] < panel_c_bounds[3][0]
    assert panel_c_bounds[0][0] == pytest.approx(panel_c_bounds[2][0])
    assert panel_c_bounds[1][0] == pytest.approx(panel_c_bounds[3][0])
    assert panel_c_bounds[0][0] > panel_a_bounds[0]
    assert panel_c_bounds[1][0] > panel_b_bounds[0]
    panel_c_row_center = 0.5 * (
        panel_c_bounds[0][0]
        + panel_c_bounds[1][0]
        + panel_c_bounds[1][2]
    )
    assert calls["panel_c_title_center_x"] == pytest.approx(
        panel_c_row_center,
        abs=0.002,
    )
    decoding_plot_calls = calls["decoding_plot_calls"]
    assert len(decoding_plot_calls) == 1 + len(datasets)
    assert decoding_plot_calls[0]["table"] == "pooled-decoding-error"
    assert decoding_plot_calls[0]["kwargs"] == {
        "significance_labels": ("**", "****")
    }
    expected_ylims = {
        "L12": ((0.0, 0.5), (0.0, 0.2)),
        "L14": ((0.0, 0.6), (0.0, 0.2)),
        "L15": ((0.0, 0.55), (0.0, 0.2)),
        "L19": ((0.0, 0.5), (0.0, 1.5)),
    }
    for plot_call, (animal_name, _date, _epoch) in zip(
        decoding_plot_calls[1:],
        datasets,
        strict=True,
    ):
        assert set(plot_call["table"]["animal_name"]) == {animal_name}
        cross_ylim, place_ylim = expected_ylims[animal_name]
        assert plot_call["kwargs"] == {
            "significance_labels": individual_labels[animal_name],
            "cross_ylim": cross_ylim,
            "place_ylim": place_ylim,
        }
    assert calls["panel_a_load_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": tuple(datasets),
        "region": "v1",
        "light_epoch": None,
        "dark_epoch": None,
        "decoding_n_permutations": 17,
        "decoding_permutation_seed": 29,
    }
    assert calls["panel_b_load_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": tuple(datasets),
    }
    assert calls["panel_b_table"] == "cv-pca-table"
