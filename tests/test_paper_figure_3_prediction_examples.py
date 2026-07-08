from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v1ca1.paper_figures.figure_3 import get_ripple_glm_path
from v1ca1.paper_figures.figure_3_prediction_examples import (
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_TOP_N_UNITS,
    get_default_prediction_datasets,
    load_dataset_prediction_examples,
    load_top_prediction_examples,
    make_prediction_examples_figure,
    parse_arguments,
)


def _write_prediction_glm_dataset(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
    devexp: np.ndarray | None = None,
) -> Path:
    xr = pytest.importorskip("xarray")
    path = get_ripple_glm_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_selection="single",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if devexp is None:
        devexp = np.array([0.10, np.nan, 0.50, 0.30], dtype=float)
    xr.Dataset(
        data_vars={
            "ripple_devexp_mean": (
                ("unit",),
                np.asarray(devexp, dtype=float),
            ),
            "ripple_devexp_p_value": (
                ("unit",),
                np.array([0.40, 0.20, 0.01, 0.05], dtype=float),
            ),
            "ripple_observed_count_oof": (
                ("sample", "unit"),
                np.array(
                    [
                        [1.0, 2.0, 0.0, 3.0],
                        [0.0, 1.0, 2.0, 1.0],
                        [2.0, 0.0, 4.0, 2.0],
                    ]
                ),
            ),
            "ripple_predicted_count_oof": (
                ("sample", "unit"),
                np.array(
                    [
                        [0.8, 1.9, 0.2, 2.7],
                        [0.1, 1.2, 1.8, 1.2],
                        [1.7, 0.3, 3.7, 2.4],
                    ]
                ),
            ),
        },
        coords={"sample": np.arange(3), "unit": np.array([11, 12, 13, 14])},
    ).to_netcdf(path)
    return path


def test_default_cli_matches_prediction_example_defaults() -> None:
    args = parse_arguments([])

    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert args.format == DEFAULT_OUTPUT_FORMAT
    assert args.top_n_units == DEFAULT_TOP_N_UNITS
    assert args.rank_offset == 0
    assert DEFAULT_TOP_N_UNITS == 30
    assert args.ripple_selection == "single"
    assert args.dataset is None


def test_default_prediction_datasets_match_figure_3_panel_b_epochs() -> None:
    assert get_default_prediction_datasets() == [
        ("L12", "20240421", "02_r1"),
        ("L14", "20240611", "02_r1"),
        ("L15", "20241121", "02_r1"),
        ("L19", "20250930", "02_r1"),
    ]


def test_load_dataset_prediction_examples_reads_one_session(
    tmp_path: Path,
) -> None:
    path = _write_prediction_glm_dataset(tmp_path)

    examples = load_dataset_prediction_examples(
        tmp_path,
        dataset=("L14", "20240611", "08_r4"),
        ripple_selection="single",
    )

    assert [example["unit_id"] for example in examples] == [11, 13, 14]
    assert [example["source_path"] for example in examples] == [str(path)] * 3
    assert examples[1]["ripple_devexp_mean"] == pytest.approx(0.50)
    assert np.allclose(examples[1]["observed"], [0.0, 2.0, 4.0])
    assert np.allclose(examples[1]["predicted"], [0.2, 1.8, 3.7])


def test_load_top_prediction_examples_orders_across_sessions(
    tmp_path: Path,
) -> None:
    _write_prediction_glm_dataset(tmp_path)
    _write_prediction_glm_dataset(
        tmp_path,
        animal_name="L15",
        date="20241121",
        epoch="10_r5",
        devexp=np.array([0.60, 0.05, 0.40, 0.20]),
    )

    examples = load_top_prediction_examples(
        tmp_path,
        datasets=[
            ("L14", "20240611", "08_r4"),
            ("L15", "20241121", "10_r5"),
        ],
        top_n_units=4,
        ripple_selection="single",
    )

    assert [example["rank"] for example in examples] == [1, 2, 3, 4]
    assert [
        (example["animal_name"], example["date"], example["epoch"], example["unit_id"])
        for example in examples
    ] == [
        ("L15", "20241121", "10_r5", 11),
        ("L14", "20240611", "08_r4", 13),
        ("L15", "20241121", "10_r5", 13),
        ("L14", "20240611", "08_r4", 14),
    ]


def test_load_top_prediction_examples_skips_rank_offset(
    tmp_path: Path,
) -> None:
    _write_prediction_glm_dataset(tmp_path)
    _write_prediction_glm_dataset(
        tmp_path,
        animal_name="L15",
        date="20241121",
        epoch="10_r5",
        devexp=np.array([0.60, 0.05, 0.40, 0.20]),
    )

    examples = load_top_prediction_examples(
        tmp_path,
        datasets=[
            ("L14", "20240611", "08_r4"),
            ("L15", "20241121", "10_r5"),
        ],
        top_n_units=2,
        rank_offset=2,
        ripple_selection="single",
    )

    assert [example["rank"] for example in examples] == [3, 4]
    assert [
        (example["animal_name"], example["date"], example["epoch"], example["unit_id"])
        for example in examples
    ] == [
        ("L15", "20241121", "10_r5", 13),
        ("L14", "20240611", "08_r4", 14),
    ]


def test_make_prediction_examples_figure_saves_grid(tmp_path: Path) -> None:
    _write_prediction_glm_dataset(tmp_path)
    output_path = tmp_path / "prediction_examples.svg"

    saved_path = make_prediction_examples_figure(
        data_root=tmp_path,
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        top_n_units=3,
        n_columns=2,
        ripple_selection="single",
    )

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
