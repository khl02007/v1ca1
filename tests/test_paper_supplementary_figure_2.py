from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.supplementary_figure_2 as figure


def test_cli_defaults_build_expected_output_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    calls: dict[str, object] = {}

    def fake_make_figure(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(figure, "get_processed_datasets", lambda: datasets)
    monkeypatch.setattr(
        figure,
        "make_supplementary_figure_2",
        fake_make_figure,
    )

    args = figure.parse_arguments([])
    figure.main([])

    assert figure.DEFAULT_OUTPUT_NAME == "supplementary_figure_2"
    assert args.data_root == figure.DEFAULT_DATA_ROOT
    assert args.output_dir == Path("paper_figures/output")
    assert args.output_name == "supplementary_figure_2"
    assert args.output_format == "pdf"
    assert calls == {
        "data_root": figure.DEFAULT_DATA_ROOT,
        "output_path": Path(
            "paper_figures/output/supplementary_figure_2.pdf"
        ),
        "datasets": datasets,
        "dpi": 300,
        "encoding_bin_size_s": figure.figure_1.ENCODING_COMPARISON_BIN_SIZE_S,
        "encoding_place_bin_size_cm": (
            figure.figure_1.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
        ),
        "decoding_n_permutations": figure.figure_1.DECODING_PERMUTATION_COUNT,
        "decoding_permutation_seed": figure.figure_1.DECODING_PERMUTATION_SEED,
    }


def test_duplicate_animal_datasets_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one data set per animal"):
        figure.normalize_individual_animal_datasets(
            [
                ("L14", "20240611", "08_r4"),
                ("L14", "20240612", "08_r4"),
            ]
        )


def test_figure_uses_supplementary_panel_and_animal_only_labels() -> None:
    assert figure.PANEL_LABELS == ("A", "B", "C")
    assert figure.format_animal_row_label(("L12", "20240421", "08_r4")) == "L12"


def test_panel_data_loader_forwards_normalized_datasets_and_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = Path("/analysis")
    datasets = [
        ("L15", "20241121"),
        ("L14", "20240611", "09_r5"),
    ]
    normalized_datasets = (
        ("L15", "20241121", "10_r5"),
        ("L14", "20240611", "09_r5"),
    )
    motor_table = object()
    encoding_table = object()
    decoding_table = object()
    trial_table = object()
    permutation_table = object()
    calls: dict[str, object] = {}

    def fake_motor_loader(**kwargs: object) -> object:
        calls["motor"] = kwargs
        return motor_table

    def fake_encoding_loader(**kwargs: object) -> object:
        calls["encoding"] = kwargs
        return encoding_table

    def fake_decoding_loader(**kwargs: object) -> object:
        calls["decoding"] = kwargs
        return decoding_table

    def fake_trial_loader(**kwargs: object) -> object:
        calls["trial"] = kwargs
        return trial_table

    def fake_permutation_test(table: object, **kwargs: object) -> object:
        calls["permutation"] = (table, kwargs)
        return permutation_table

    monkeypatch.setattr(figure.figure_1, "load_motor_delta_table", fake_motor_loader)
    monkeypatch.setattr(
        figure.figure_1,
        "load_encoding_delta_table",
        fake_encoding_loader,
    )
    monkeypatch.setattr(
        figure.figure_1,
        "load_decoding_absolute_error_table",
        fake_decoding_loader,
    )
    monkeypatch.setattr(
        figure.figure_1,
        "build_decoding_trial_error_table",
        fake_trial_loader,
    )
    monkeypatch.setattr(
        figure.figure_1,
        "compute_decoding_permutation_tests",
        fake_permutation_test,
    )

    panel_data = figure.load_individual_animal_panel_data(
        data_root=data_root,
        datasets=datasets,
        encoding_bin_size_s=0.1,
        encoding_place_bin_size_cm=4.0,
        decoding_n_permutations=321,
        decoding_permutation_seed=17,
    )

    assert panel_data == {
        "motor_delta": motor_table,
        "encoding_delta": encoding_table,
        "decoding_error": decoding_table,
        "decoding_permutation": permutation_table,
    }
    assert calls["motor"] == {
        "data_root": data_root,
        "datasets": normalized_datasets,
        "region": figure.figure_1.MOTOR_DELTA_REGION,
    }
    assert calls["encoding"] == {
        "data_root": data_root,
        "datasets": normalized_datasets,
        "region": figure.figure_1.ENCODING_COMPARISON_REGION,
        "bin_size_s": 0.1,
        "place_bin_size_cm": 4.0,
    }
    assert calls["decoding"] == {
        "data_root": data_root,
        "datasets": normalized_datasets,
        "region": figure.figure_1.DECODING_COMPARISON_REGION,
    }
    assert calls["trial"] == {
        "data_root": data_root,
        "datasets": normalized_datasets,
        "region": figure.figure_1.DECODING_COMPARISON_REGION,
    }
    assert calls["permutation"] == (
        trial_table,
        {"n_permutations": 321, "seed": 17},
    )


def test_figure_filters_each_panel_in_dataset_order_and_builds_singleton_brackets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")

    datasets = [
        ("L15", "20241121", "10_r5"),
        ("L14", "20240611", "08_r4"),
    ]
    panel_data = {
        "motor_delta": pd.DataFrame(
            {
                "animal_name": ["L14", "L15", "L14"],
                "token": ["motor-14-a", "motor-15", "motor-14-b"],
            }
        ),
        "encoding_delta": pd.DataFrame(
            {
                "animal_name": ["L14", "L15"],
                "token": ["encoding-14", "encoding-15"],
            }
        ),
        "decoding_error": pd.DataFrame(
            {
                "animal_name": ["L15", "L14"],
                "token": ["decoding-15", "decoding-14"],
            }
        ),
        "decoding_permutation": pd.DataFrame(
            {
                "animal_name": ["L14", "L15"],
                "token": ["permutation-14", "permutation-15"],
            }
        ),
    }
    motor_calls: list[list[str]] = []
    encoding_calls: list[list[str]] = []
    decoding_calls: list[tuple[list[str], object]] = []
    bracket_calls: list[tuple[list[str], tuple[str, ...]]] = []
    save_calls: dict[str, object] = {}

    monkeypatch.setattr(
        figure,
        "load_individual_animal_panel_data",
        lambda **_kwargs: panel_data,
    )
    monkeypatch.setattr(
        figure.figure_1,
        "plot_motor_delta_panel",
        lambda _axis, table: motor_calls.append(table["token"].tolist()),
    )
    monkeypatch.setattr(
        figure.figure_1,
        "plot_encoding_delta_panel",
        lambda _axis, table: encoding_calls.append(table["token"].tolist()),
    )

    def fake_build_brackets(
        table: object,
        *,
        animal_names: tuple[str, ...],
    ) -> tuple[tuple[float, float, float, str], ...]:
        tokens = table["token"].tolist()
        bracket_calls.append((tokens, animal_names))
        return ((1.0, 2.0, 0.5, animal_names[0]),)

    def fake_plot_decoding(
        _axis: object,
        table: object,
        *,
        significance_brackets: object,
    ) -> None:
        decoding_calls.append((table["token"].tolist(), significance_brackets))

    def fake_save(
        mpl_figure: object,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        mpl_figure.canvas.draw()
        save_calls["output_path"] = output_path
        save_calls["kwargs"] = kwargs
        save_calls["size_inches"] = tuple(mpl_figure.get_size_inches())
        return output_path

    monkeypatch.setattr(
        figure.figure_1,
        "build_decoding_significance_brackets",
        fake_build_brackets,
    )
    monkeypatch.setattr(
        figure.figure_1,
        "plot_decoding_error_panel",
        fake_plot_decoding,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save)

    output_path = tmp_path / "supplementary_figure_2.svg"
    saved_path = figure.make_supplementary_figure_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        dpi=180,
    )

    assert saved_path == output_path
    assert motor_calls == [["motor-15"], ["motor-14-a", "motor-14-b"]]
    assert encoding_calls == [["encoding-15"], ["encoding-14"]]
    assert bracket_calls == [
        (["permutation-15"], ("L15",)),
        (["permutation-14"], ("L14",)),
    ]
    assert decoding_calls == [
        (["decoding-15"], ((1.0, 2.0, 0.5, "L15"),)),
        (["decoding-14"], ((1.0, 2.0, 0.5, "L14"),)),
    ]
    assert save_calls["output_path"] == output_path
    assert save_calls["kwargs"] == {"dpi": 180, "bbox_inches": None}
    assert save_calls["size_inches"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            2 * figure.DEFAULT_ANIMAL_ROW_HEIGHT_MM / 25.4,
        )
    )


def test_lightweight_agg_render_saves_nonempty_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pd = pytest.importorskip("pandas")
    import matplotlib.pyplot as plt

    animal_name = "L14"
    panel_data = {
        "motor_delta": pd.DataFrame(
            {
                "animal_name": [animal_name],
                "date": ["20240611"],
                "unit": [1],
                "delta_log_likelihood_bits_per_spike": [0.2],
            }
        ),
        "encoding_delta": pd.DataFrame(
            {
                "animal_name": [animal_name, animal_name],
                "date": ["20240611", "20240611"],
                "unit": [1, 1],
                "comparison": [
                    "dpp_vs_absolute_place",
                    "dpp_vs_absolute_task_progression",
                ],
                "delta_log_likelihood_bits_per_spike": [0.1, -0.1],
            }
        ),
        "decoding_error": pd.DataFrame(
            {
                "animal_name": [animal_name] * 3,
                "region": ["v1"] * 3,
                "comparison": [
                    "same_turn_cross_arm",
                    "opposite_turn_same_arm",
                    "same_inbound_outbound_cross_arm",
                ],
                "absolute_error": [0.2, 0.3, 0.4],
            }
        ),
        "decoding_permutation": pd.DataFrame(
            {
                "animal_name": [animal_name, animal_name],
                "comparison_a": ["same_turn_cross_arm", "same_turn_cross_arm"],
                "comparison_b": [
                    "opposite_turn_same_arm",
                    "same_inbound_outbound_cross_arm",
                ],
                "median_difference": [-0.1, -0.2],
                "p_two_sided": [0.04, 0.01],
            }
        ),
    }
    monkeypatch.setattr(
        figure,
        "load_individual_animal_panel_data",
        lambda **_kwargs: panel_data,
    )

    open_figures = set(plt.get_fignums())
    output_path = tmp_path / "nested" / "supplementary_figure_2.png"
    saved_path = figure.make_supplementary_figure_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[(animal_name, "20240611", "08_r4")],
        dpi=60,
        decoding_n_permutations=10,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert set(plt.get_fignums()) == open_figures
