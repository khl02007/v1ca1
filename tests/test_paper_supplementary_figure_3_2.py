from __future__ import annotations

from pathlib import Path

import pytest

from v1ca1.paper_figures import figure_2_old as figure_2
from v1ca1.paper_figures import figure_3
from v1ca1.paper_figures import supplementary_figure_3
from v1ca1.paper_figures import supplementary_figure_3_2 as figure


def test_default_cli_matches_requested_source_panels() -> None:
    args = figure.parse_arguments([])

    assert args.output_dir == figure.DEFAULT_OUTPUT_DIR
    assert args.output_name == "supplementary_figure_3_2"
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
    assert figure.DEFAULT_FIGURE_HEIGHT_MM == figure_2.PANEL_D_ROW_HEIGHT_MM
    assert figure.PANEL_TITLES == (
        figure.DECODING_PANEL_TITLE,
        supplementary_figure_3.PANEL_A_CV_PCA_TITLE,
    )
    assert not hasattr(args, "panel_a_cache_dir")
    assert not hasattr(args, "position_bin_count")


def test_load_panel_a_decoding_data_uses_only_decoding_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_load_decoding_error_table(**kwargs: object) -> str:
        calls["decoding_error_kwargs"] = kwargs
        return "decoding-error"

    def fake_build_trial_error_table(**kwargs: object) -> str:
        calls["trial_error_kwargs"] = kwargs
        return "trial-error"

    def fake_compute_permutation_tests(table: object, **kwargs: object) -> str:
        calls["permutation_table"] = table
        calls["permutation_kwargs"] = kwargs
        return "permutation-results"

    def fake_build_significance_labels(
        table: object,
        **kwargs: object,
    ) -> tuple[str, str]:
        calls["significance_table"] = table
        calls["significance_kwargs"] = kwargs
        return "**", "****"

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

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    result = figure.load_panel_a_decoding_data(
        data_root=Path("/analysis"),
        datasets=datasets,
        region="v1",
        light_epoch="02_r1",
        dark_epoch=None,
        decoding_n_permutations=17,
        decoding_permutation_seed=29,
    )

    expected_loader_kwargs = {
        "data_root": Path("/analysis"),
        "datasets": tuple(datasets),
        "region": "v1",
        "light_epoch": "02_r1",
        "dark_epoch": None,
    }
    assert result == {
        "decoding_error": "decoding-error",
        "decoding_significance_labels": ("**", "****"),
    }
    assert calls["decoding_error_kwargs"] == expected_loader_kwargs
    assert calls["trial_error_kwargs"] == expected_loader_kwargs
    assert calls["permutation_table"] == "trial-error"
    assert calls["permutation_kwargs"] == {
        "n_permutations": 17,
        "seed": 29,
    }
    assert calls["significance_table"] == "permutation-results"
    assert calls["significance_kwargs"] == {
        "animal_names": ("L14", "L15")
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


def test_main_builds_named_output_and_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_make_supplementary_figure_3_2(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        figure,
        "make_supplementary_figure_3_2",
        fake_make_supplementary_figure_3_2,
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
        "output_path": tmp_path / "supplementary_figure_3_2.svg",
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "ca1",
        "light_epoch": "02_r1",
        "dark_epoch": "08_r4",
        "dpi": 144,
        "decoding_n_permutations": 19,
        "decoding_permutation_seed": 23,
    }


def test_make_supplementary_figure_3_2_draws_requested_panels_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_apply_paper_style() -> None:
        calls["styled"] = True

    def fake_load_panel_a_decoding_data(**kwargs: object) -> dict[str, object]:
        calls["panel_a_load_kwargs"] = kwargs
        return {
            "decoding_error": "decoding-error",
            "decoding_significance_labels": ("**", "****"),
        }

    def fake_plot_decoding_panel(ax, table: object, **kwargs: object) -> None:
        calls["panel_a_axis"] = ax
        calls["panel_a_table"] = table
        calls["panel_a_plot_kwargs"] = kwargs
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
        raise AssertionError("Supplementary Figure 3_2 loaded an omitted panel")

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
        calls["titles"] = [ax.get_title() for ax in fig.axes]
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
        supplementary_figure_3,
        "load_panel_a_cv_pca_participation_ratio_table",
        fake_load_cv_pca_table,
    )
    monkeypatch.setattr(
        supplementary_figure_3,
        "plot_panel_a_cv_pca_participation_ratios",
        fake_plot_cv_pca_panel,
    )
    monkeypatch.setattr(
        figure_3,
        "load_figure_3_panel_data",
        fail_if_omitted_panel_is_loaded,
    )
    monkeypatch.setattr(
        supplementary_figure_3,
        "load_panel_b_motor_progression_table",
        fail_if_omitted_panel_is_loaded,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_3_2.svg"
    datasets = [("L14", "20240611", "08_r4")]
    saved_path = figure.make_supplementary_figure_3_2(
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
    assert calls["panel_labels"] == ["A", "B"]
    assert calls["titles"] == list(figure.PANEL_TITLES)
    assert calls["bounds"][0][0] < calls["bounds"][1][0]
    assert calls["panel_a_table"] == "decoding-error"
    assert calls["panel_a_plot_kwargs"] == {
        "significance_labels": ("**", "****")
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
