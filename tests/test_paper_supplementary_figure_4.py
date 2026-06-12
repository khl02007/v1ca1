from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_3 as figure_3_module
import v1ca1.paper_figures.supplementary_figure_4 as supp_figure_4_module
from v1ca1.paper_figures.supplementary_figure_4 import (
    DEFAULT_ANIMAL_ROW_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_NAME,
    COMBINED_FULL_SEGMENT_GAIN_PANEL_HEIGHT_MM,
    EMPIRICAL_PAIRWISE_MODEL_NAMES,
    EMPIRICAL_PAIRWISE_PANEL_HEIGHT_MM,
    FULL_SEGMENT_GAIN_MIN_TUNING_STABILITY_CORRELATION,
    FULL_SEGMENT_GAIN_PANEL_HEIGHT_MM,
    GLM_SCALAR_PANEL_HEIGHT_MM,
    HYBRID_GLM_EMPIRICAL_PANEL_HEIGHT_MM,
    LETTER_HORIZONTAL_MARGIN_IN,
    LETTER_PAPER_WIDTH_IN,
    MIXED_GLM_EMPIRICAL_PANEL_HEIGHT_MM,
    MULTIPLIER_PANEL_HEIGHT_MM,
    NESTED_DARK_ACTIVE_FR_THRESHOLD_HZ,
    NESTED_MODULATION_PANEL_HEIGHT_MM,
    NESTED_TUNING_STABILITY_CORRELATION_THRESHOLD,
    SCALAR_MODEL_NAME,
    SCALAR_BASELINE_SCORE_COLUMN,
    SCALAR_BASELINE_SCORE_VARIABLE,
    SCALAR_PANEL_MODEL_COLORS,
    SCALAR_PANEL_EXAMPLE_AXIS_BOUNDS,
    SCALAR_PANEL_EXAMPLE_DELTA_LABEL_POSITIONS,
    SCALAR_PANEL_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS,
    SCALAR_PANEL_EXAMPLES,
    SCALAR_PANEL_HEIGHT_MM,
    get_figure_height_mm,
    get_swap_tuning_curve_comparison_path,
    get_swap_tuning_curve_comparison_dataset_path,
    group_datasets_by_animal,
    load_empirical_pairwise_delta_table,
    load_full_segment_log_gain_table,
    load_hybrid_glm_empirical_delta_table,
    load_nested_vision_modulation_table,
    load_scalar_multiplier_table,
    make_supplementary_figure_4,
    parse_arguments,
)


def test_default_cli_matches_letter_width_with_one_inch_margins() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "supplementary_figure_4"
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        (LETTER_PAPER_WIDTH_IN - 2.0 * LETTER_HORIZONTAL_MARGIN_IN) * 25.4
    )
    assert args.output_dir == figure_3_module.DEFAULT_OUTPUT_DIR
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert args.output_format == figure_3_module.DEFAULT_OUTPUT_FORMAT
    assert args.region == figure_3_module.DEFAULT_REGIONS[0]
    assert args.dataset is None
    assert args.dark_epoch is None
    assert get_figure_height_mm(0) == pytest.approx(DEFAULT_ANIMAL_ROW_HEIGHT_MM)
    assert get_figure_height_mm(3) == pytest.approx(
        SCALAR_PANEL_HEIGHT_MM
        + 3 * DEFAULT_ANIMAL_ROW_HEIGHT_MM
        + EMPIRICAL_PAIRWISE_PANEL_HEIGHT_MM
        + GLM_SCALAR_PANEL_HEIGHT_MM
        + MIXED_GLM_EMPIRICAL_PANEL_HEIGHT_MM
        + HYBRID_GLM_EMPIRICAL_PANEL_HEIGHT_MM
        + MULTIPLIER_PANEL_HEIGHT_MM
        + FULL_SEGMENT_GAIN_PANEL_HEIGHT_MM
        + COMBINED_FULL_SEGMENT_GAIN_PANEL_HEIGHT_MM
        + NESTED_MODULATION_PANEL_HEIGHT_MM
    )


def test_group_datasets_by_animal_preserves_input_order() -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L14", "20240612", "08_r4"),
    ]

    grouped = group_datasets_by_animal(datasets)

    assert list(grouped) == ["L14", "L15"]
    assert grouped["L14"] == [
        ("L14", "20240611", "08_r4"),
        ("L14", "20240612", "08_r4"),
    ]
    assert grouped["L15"] == [("L15", "20241121", "10_r5")]


def test_load_empirical_pairwise_delta_table_builds_v_ms_as_deltas(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pandas = pytest.importorskip("pandas")

    path = get_swap_tuning_curve_comparison_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        dark_epoch="08_r4",
    )
    path.parent.mkdir(parents=True)
    rows = []
    for unit, values in {
        1: {
            "empirical_visual": (10.0, 1.0, 0.40),
            "empirical_segment_multiplicative_ratio": (8.0, 0.3, 0.10),
            "empirical_segment_additive_delta": (9.0, 0.2, 0.25),
        },
        2: {
            "empirical_visual": (4.0, 0.1, 0.05),
            "empirical_segment_multiplicative_ratio": (6.0, 0.4, 0.30),
            "empirical_segment_additive_delta": (5.0, 0.2, 0.20),
        },
    }.items():
        for model_name, (ll_sum, ll_bits_per_s, ll_bits_per_spike) in values.items():
            rows.append(
                {
                    "animal_name": "L00",
                    "date": "20000101",
                    "region": "v1",
                    "dark_train_epoch": "08_r4",
                    "light_train_epoch": "02_r1",
                    "light_test_epoch": "06_r3",
                    "trajectory": "center_to_left",
                    "unit": unit,
                    "model": model_name,
                    "ll_sum": ll_sum,
                    "ll_bits_per_s": ll_bits_per_s,
                    "ll_bits_per_spike": ll_bits_per_spike,
                }
            )
    pandas.DataFrame(rows).to_parquet(path, index=False)

    table = load_empirical_pairwise_delta_table(
        data_root=tmp_path,
        datasets=[("L00", "20000101", "08_r4")],
        region="v1",
        dark_epoch=None,
    )

    assert len(table) == 2
    unit_1 = table[table["unit"] == 1].iloc[0]
    assert unit_1["winner"] == "V"
    assert unit_1["winner_model_name"] == EMPIRICAL_PAIRWISE_MODEL_NAMES["V"]
    assert unit_1["delta_V_minus_MS_bits_per_s"] == pytest.approx(0.7)
    assert unit_1["delta_V_minus_AS_bits_per_s"] == pytest.approx(0.8)
    assert unit_1["delta_V_minus_MS_bits_per_spike"] == pytest.approx(0.3)
    assert unit_1["delta_V_minus_AS_bits_per_spike"] == pytest.approx(0.15)
    unit_2 = table[table["unit"] == 2].iloc[0]
    assert unit_2["winner"] == "MS"
    assert unit_2["winner_model_name"] == EMPIRICAL_PAIRWISE_MODEL_NAMES["MS"]
    assert unit_2["delta_V_minus_MS_bits_per_s"] == pytest.approx(-0.3)
    assert unit_2["delta_V_minus_AS_bits_per_s"] == pytest.approx(-0.1)
    assert unit_2["delta_MS_minus_AS_bits_per_s"] == pytest.approx(0.2)
    assert unit_2["delta_V_minus_MS_bits_per_spike"] == pytest.approx(-0.25)
    assert unit_2["delta_V_minus_AS_bits_per_spike"] == pytest.approx(-0.15)
    assert unit_2["delta_MS_minus_AS_bits_per_spike"] == pytest.approx(0.10)


def test_load_scalar_multiplier_table_builds_matched_log_gains(
    tmp_path: Path,
) -> None:
    xarray = pytest.importorskip("xarray")

    empirical_path = get_swap_tuning_curve_comparison_dataset_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        dark_epoch="08_r4",
    )
    empirical_path.parent.mkdir(parents=True)
    empirical_dataset = xarray.Dataset(
        data_vars={
            "other_dark_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray([[[1.0, 2.0], [10.0, 10.0], [10.0, 10.0]]]),
            ),
            "other_light_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray([[[1.0, 2.0], [20.0, 5.0], [30.0, 5.0]]]),
            ),
            "segment_bin_mask": (
                ("trajectory", "tp_bin"),
                np.asarray([[False, True, True]], dtype=bool),
            ),
            "swap_source_trajectory": (
                "trajectory",
                np.asarray(["center_to_right"], dtype=str),
            ),
            "swap_segment_index_1based": ("trajectory", np.asarray([3], dtype=int)),
        },
        coords={
            "trajectory": np.asarray(["center_to_left"], dtype=str),
            "tp_bin": np.asarray([0.1, 0.5, 0.9], dtype=float),
            "unit": np.asarray([1, 2], dtype=int),
        },
    )
    empirical_dataset.to_netcdf(empirical_path)

    glm_path = figure_3_module.get_dark_light_glm_selected_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        model_name=SCALAR_MODEL_NAME,
    )
    glm_path.parent.mkdir(parents=True)
    coef_segment_scalar_gain = np.zeros((2, 3, 2), dtype=float)
    coef_segment_scalar_gain[1, 2, :] = np.asarray([0.40, -0.20])
    coef_light_offset = np.asarray([[0.10, 0.05], [0.0, 0.0]], dtype=float)
    glm_dataset = xarray.Dataset(
        data_vars={
            "coef_segment_scalar_gain": (
                ("trajectory", "segment_basis", "unit"),
                coef_segment_scalar_gain,
            ),
            "coef_light_offset": (
                ("trajectory", "unit"),
                coef_light_offset,
            ),
        },
        coords={
            "trajectory": np.asarray(["center_to_left", "center_to_right"], dtype=str),
            "segment_basis": np.asarray([0, 1, 2], dtype=int),
            "unit": np.asarray([1, 2], dtype=int),
        },
    )
    glm_dataset.to_netcdf(glm_path)

    table = load_scalar_multiplier_table(
        data_root=tmp_path,
        datasets=[("L00", "20000101", "08_r4")],
        region="v1",
        dark_epoch=None,
    )

    assert len(table) == 2
    unit_1 = table[table["unit"] == 1].iloc[0]
    assert unit_1["source_trajectory"] == "center_to_right"
    assert unit_1["swap_segment_index_1based"] == 3
    assert unit_1["log_empirical_ms_gain"] == pytest.approx(np.log(50.0 / 20.0))
    assert unit_1["log_glm_segment_gain"] == pytest.approx(0.40)
    assert unit_1["log_glm_full_gain"] == pytest.approx(0.50)
    unit_2 = table[table["unit"] == 2].iloc[0]
    assert unit_2["log_empirical_ms_gain"] == pytest.approx(np.log(10.0 / 20.0))
    assert unit_2["log_glm_segment_gain"] == pytest.approx(-0.20)
    assert unit_2["log_glm_full_gain"] == pytest.approx(-0.15)


def test_load_full_segment_log_gain_table_filters_reliable_trajectory_units(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pandas = pytest.importorskip("pandas")
    xarray = pytest.importorskip("xarray")

    glm_path = figure_3_module.get_dark_light_glm_selected_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        model_name=SCALAR_MODEL_NAME,
    )
    glm_path.parent.mkdir(parents=True)
    coef_segment_scalar_gain = np.asarray(
        [
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60], [0.70, 0.80, 0.90]],
            [[-0.10, -0.20, -0.30], [-0.40, -0.50, -0.60], [-0.70, -0.80, -0.90]],
        ],
        dtype=float,
    )
    coef_light_offset = np.asarray(
        [[1.00, 2.00, 3.00], [0.05, 0.15, 0.25]],
        dtype=float,
    )
    baseline_score = np.asarray(
        [[0.10, -0.20, 0.30], [0.40, 0.50, 0.60]],
        dtype=float,
    )
    xarray.Dataset(
        data_vars={
            "coef_segment_scalar_gain": (
                ("trajectory", "segment_basis", "unit"),
                coef_segment_scalar_gain,
            ),
            "coef_light_offset": (
                ("trajectory", "unit"),
                coef_light_offset,
            ),
            SCALAR_BASELINE_SCORE_VARIABLE: (
                ("trajectory", "unit"),
                baseline_score,
            ),
        },
        coords={
            "trajectory": np.asarray(["center_to_left", "center_to_right"], dtype=str),
            "segment_basis": np.asarray([0, 1, 2], dtype=int),
            "unit": np.asarray([1, 2, 3], dtype=int),
        },
    ).to_netcdf(glm_path)

    stability_path = figure_3_module.get_stability_table_path(
        tmp_path,
        "L00",
        "20000101",
    )
    stability_path.parent.mkdir(parents=True)
    pandas.DataFrame(
        {
            "unit": [1, 2, 2, 3, 1],
            "region": ["v1", "v1", "v1", "ca1", "v1"],
            "epoch": ["08_r4", "08_r4", "08_r4", "08_r4", "02_r1"],
            "trajectory_type": [
                "center_to_left",
                "center_to_left",
                "center_to_right",
                "center_to_right",
                "center_to_left",
            ],
            "stability_correlation": [0.50, 0.49, 0.75, 0.95, 0.90],
        }
    ).to_parquet(stability_path, index=False)

    table = load_full_segment_log_gain_table(
        data_root=tmp_path,
        datasets=[("L00", "20000101", "08_r4")],
        region="v1",
        dark_epoch=None,
    )

    assert len(table) == 6
    assert set(table["unit"]) == {1, 2}
    assert set(table["trajectory"]) == {"center_to_left", "center_to_right"}
    assert set(table["segment_index_1based"]) == {1, 2, 3}
    left_unit_1 = table[
        (table["trajectory"] == "center_to_left")
        & (table["unit"] == 1)
        & (table["segment_index_1based"] == 2)
    ].iloc[0]
    assert left_unit_1["full_segment_log_gain"] == pytest.approx(1.40)
    assert left_unit_1["segment_specific_log_gain"] == pytest.approx(0.40)
    assert left_unit_1["light_offset_log_gain"] == pytest.approx(1.00)
    assert left_unit_1[SCALAR_BASELINE_SCORE_COLUMN] == pytest.approx(0.10)
    assert left_unit_1["stability_correlation"] == pytest.approx(0.50)
    right_unit_2 = table[
        (table["trajectory"] == "center_to_right")
        & (table["unit"] == 2)
        & (table["segment_index_1based"] == 3)
    ].iloc[0]
    assert right_unit_2["full_segment_log_gain"] == pytest.approx(-0.65)
    assert right_unit_2["reliability_epoch"] == "08_r4"


def test_load_nested_vision_modulation_table_builds_nested_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pandas = pytest.importorskip("pandas")

    rate_calls = []

    def fake_load_dark_movement_firing_rate_table(*args: object, **kwargs: object):
        rate_calls.append({"args": args, "kwargs": kwargs})
        return pandas.DataFrame(
            {
                "unit": [1, 2, 3, 4, 5],
                "dark_firing_rate_hz": [0.2, 0.5, 0.6, 1.0, 0.8],
            }
        )

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_dark_movement_firing_rate_table",
        fake_load_dark_movement_firing_rate_table,
    )

    stability_path = figure_3_module.get_stability_table_path(
        tmp_path,
        "L00",
        "20000101",
    )
    stability_path.parent.mkdir(parents=True)
    pandas.DataFrame(
        {
            "unit": [2, 3, 4, 5, 1],
            "region": ["v1", "v1", "v1", "v1", "v1"],
            "epoch": ["08_r4", "08_r4", "08_r4", "08_r4", "02_r1"],
            "trajectory_type": [
                "center_to_left",
                "center_to_left",
                "center_to_right",
                "right_to_center",
                "center_to_left",
            ],
            "stability_correlation": [0.49, 0.50, 0.75, 0.80, 1.0],
        }
    ).to_parquet(stability_path, index=False)

    full_gain_table = pandas.DataFrame(
        {
            "animal_name": ["L00", "L00", "L00", "L00"],
            "date": ["20000101", "20000101", "20000101", "20000101"],
            "region": ["v1", "v1", "v1", "v1"],
            "dark_train_epoch": ["08_r4", "08_r4", "08_r4", "08_r4"],
            "light_train_epoch": ["02_r1", "02_r1", "02_r1", "02_r1"],
            "trajectory": [
                "center_to_left",
                "center_to_left",
                "center_to_right",
                "center_to_right",
            ],
            "segment_index_1based": [1, 2, 1, 2],
            "unit": [3, 3, 4, 4],
            "full_segment_log_gain": [0.1, 0.2, 0.1, 0.5],
            SCALAR_BASELINE_SCORE_COLUMN: [0.1, 0.1, -0.1, -0.1],
        }
    )

    table = load_nested_vision_modulation_table(
        data_root=tmp_path,
        datasets=[("L00", "20000101", "08_r4")],
        region="v1",
        dark_epoch=None,
        full_gain_table=full_gain_table,
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row["total_cell_count"] == 5
    assert row["dark_inactive_count"] == 1
    assert row["dark_active_count"] == 4
    assert row["dark_active_unstable_count"] == 1
    assert row["dark_active_stable_count"] == 3
    assert row["dark_active_stable_no_scalar_fit_count"] == 2
    assert row["dark_active_stable_missing_scalar_fit_count"] == 1
    assert row["dark_active_stable_scalar_below_baseline_count"] == 1
    assert row["dark_active_stable_unmodulated_count"] == 1
    assert row["dark_active_stable_modulated_count"] == 0
    assert row["dark_active_fr_threshold_hz"] == pytest.approx(
        NESTED_DARK_ACTIVE_FR_THRESHOLD_HZ
    )
    assert row["tuning_stability_correlation_threshold"] == pytest.approx(
        NESTED_TUNING_STABILITY_CORRELATION_THRESHOLD
    )
    assert rate_calls[0]["kwargs"] == {
        "animal_name": "L00",
        "date": "20000101",
        "dark_epoch": "08_r4",
        "region": "v1",
    }


def test_load_hybrid_glm_empirical_delta_table_scores_same_swap_bins(
    tmp_path: Path,
) -> None:
    xarray = pytest.importorskip("xarray")

    swap_path = figure_3_module.get_swap_glm_selected_comparison_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
    )
    swap_path.parent.mkdir(parents=True)
    swap_dataset = xarray.Dataset(
        data_vars={
            "swap_source_trajectory": (
                "trajectory",
                np.asarray(["center_to_right"], dtype=str),
            ),
            "swap_segment_index_1based": ("trajectory", np.asarray([2], dtype=int)),
            "test_light_occupancy_s": (
                ("trajectory", "tp_observed_bin"),
                np.asarray([[0.0, 1.0]], dtype=float),
            ),
            "test_light_spike_count": (
                ("trajectory", "tp_observed_bin", "unit"),
                np.asarray([[[0.0, 0.0], [4.0, 1.0]]], dtype=float),
            ),
            "dark_hz_grid": (
                ("model", "trajectory", "tp_grid", "unit"),
                np.asarray(
                    [
                        [[[1.0, 1.0], [2.0, 4.0]]],
                        [[[1.0, 1.0], [3.0, 3.0]]],
                    ],
                    dtype=float,
                ),
            ),
            "test_light_swapped_hz_grid": (
                ("model", "trajectory", "tp_grid", "unit"),
                np.asarray(
                    [
                        [[[1.0, 1.0], [2.0, 4.0]]],
                        [[[1.0, 1.0], [5.0, 1.0]]],
                    ],
                    dtype=float,
                ),
            ),
            "test_light_swapped_segment_n_bins": (
                "trajectory",
                np.asarray([2.0], dtype=float),
            ),
            "test_light_swapped_segment_swapped_spike_sum": (
                ("model", "trajectory", "unit"),
                np.asarray([[[4.0, 1.0]], [[4.0, 1.0]]], dtype=float),
            ),
        },
        coords={
            "model": np.asarray(["visual", SCALAR_MODEL_NAME], dtype=str),
            "trajectory": np.asarray(["center_to_left"], dtype=str),
            "tp_grid": np.asarray([0.25, 0.75], dtype=float),
            "tp_observed_bin": np.asarray([0.25, 0.75], dtype=float),
            "tp_observed_edge": np.asarray([0.0, 0.5, 1.0], dtype=float),
            "segment_edge": np.asarray([0.0, 0.5, 1.0], dtype=float),
            "unit": np.asarray([1, 2], dtype=int),
        },
        attrs={"bin_size_s": 0.5},
    )
    swap_dataset.to_netcdf(swap_path)

    empirical_path = get_swap_tuning_curve_comparison_dataset_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        dark_epoch="08_r4",
    )
    empirical_path.parent.mkdir(parents=True)
    empirical_dataset = xarray.Dataset(
        data_vars={
            "same_dark_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray([[[1.0, 1.0], [2.0, 2.0]]], dtype=float),
            ),
            "other_dark_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray([[[1.0, 1.0], [2.0, 2.0]]], dtype=float),
            ),
            "other_light_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray([[[1.0, 1.0], [4.0, 1.0]]], dtype=float),
            ),
            "segment_bin_mask": (
                ("trajectory", "tp_bin"),
                np.asarray([[False, True]], dtype=bool),
            ),
        },
        coords={
            "trajectory": np.asarray(["center_to_left"], dtype=str),
            "tp_bin": np.asarray([0.25, 0.75], dtype=float),
            "unit": np.asarray([1, 2], dtype=int),
        },
    )
    empirical_dataset.to_netcdf(empirical_path)

    glm_path = figure_3_module.get_dark_light_glm_selected_path(
        tmp_path,
        animal_name="L00",
        date="20000101",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        model_name=SCALAR_MODEL_NAME,
    )
    glm_path.parent.mkdir(parents=True)
    coef_segment_scalar_gain = np.zeros((2, 2, 2), dtype=float)
    coef_segment_scalar_gain[1, 1, :] = np.log(2.0)
    coef_light_offset = np.asarray(
        [[0.0, -np.log(2.0)], [0.0, 0.0]],
        dtype=float,
    )
    glm_dataset = xarray.Dataset(
        data_vars={
            "coef_segment_scalar_gain": (
                ("trajectory", "segment_basis", "unit"),
                coef_segment_scalar_gain,
            ),
            "coef_light_offset": (
                ("trajectory", "unit"),
                coef_light_offset,
            ),
        },
        coords={
            "trajectory": np.asarray(["center_to_left", "center_to_right"], dtype=str),
            "segment_basis": np.asarray([0, 1], dtype=int),
            "unit": np.asarray([1, 2], dtype=int),
        },
    )
    glm_dataset.to_netcdf(glm_path)

    table = load_hybrid_glm_empirical_delta_table(
        data_root=tmp_path,
        datasets=[("L00", "20000101", "08_r4")],
        region="v1",
        dark_epoch=None,
    )

    assert len(table) == 2
    by_unit = table.set_index("unit")
    visual_unit_1 = 4.0 * np.log(2.0) - 2.0
    hybrid_unit_1 = 4.0 * np.log(4.0) - 4.0
    reverse_hybrid_unit_1 = 4.0 * np.log(6.0) - 6.0
    visual_unit_2 = np.log(4.0) - 4.0
    hybrid_unit_2 = np.log(2.0) - 2.0
    reverse_hybrid_unit_2 = np.log(1.5) - 1.5
    assert by_unit.loc[1, "winner"] == "H"
    assert by_unit.loc[2, "winner"] == "MS"
    assert by_unit.loc[1, "delta_V_minus_H_bits_per_spike"] == pytest.approx(
        (visual_unit_1 - hybrid_unit_1) / (np.log(2.0) * 4.0)
    )
    assert by_unit.loc[2, "delta_V_minus_H_bits_per_spike"] == pytest.approx(
        (visual_unit_2 - hybrid_unit_2) / np.log(2.0)
    )
    assert by_unit.loc[1, "delta_V_minus_H2_bits_per_spike"] == pytest.approx(
        (visual_unit_1 - reverse_hybrid_unit_1) / (np.log(2.0) * 4.0)
    )
    assert by_unit.loc[2, "delta_V_minus_H2_bits_per_spike"] == pytest.approx(
        (visual_unit_2 - reverse_hybrid_unit_2) / np.log(2.0)
    )
    assert by_unit.loc[1, "test_light_bin_count"] == pytest.approx(2.0)
    assert by_unit.loc[1, "test_light_spike_sum"] == pytest.approx(4.0)


def test_load_mixed_glm_full_additive_delta_table_uses_pointwise_additive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    sentinel = object()

    def fake_load_mixed_glm_empirical_delta_table(**kwargs: object):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_mixed_glm_empirical_delta_table",
        fake_load_mixed_glm_empirical_delta_table,
    )

    result = supp_figure_4_module.load_mixed_glm_full_additive_delta_table(
        data_root=tmp_path,
        datasets=[("L00", "20000101", "08_r4")],
        region="v1",
        dark_epoch=None,
    )

    assert result is sentinel
    assert calls == [
        {
            "data_root": tmp_path,
            "datasets": [("L00", "20000101", "08_r4")],
            "region": "v1",
            "dark_epoch": None,
            "light_train_epoch": "02_r1",
            "light_test_epoch": "06_r3",
            "empirical_model_name": "empirical_pointwise_additive_delta",
            "empirical_label": "A",
        }
    ]


def test_plot_hybrid_best_fraction_bar_uses_pairwise_v_h_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    calls = []

    def fake_plot_best_fraction_bar(ax, delta_table, *, labels, colors):
        calls.append(
            {
                "ax": ax,
                "winners": list(delta_table["winner"]),
                "labels": labels,
                "colors": colors,
            }
        )

    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_best_fraction_bar",
        fake_plot_best_fraction_bar,
    )

    table = pandas.DataFrame(
        {
            "winner": ["MS", "MS", "V"],
            "delta_V_minus_H_bits_per_spike": [-0.2, 0.0, 0.3],
        }
    )
    axis = object()
    supp_figure_4_module.plot_hybrid_glm_empirical_best_fraction_bar(axis, table)

    assert calls[0]["ax"] is axis
    assert calls[0]["winners"] == ["H", "tie", "V"]
    assert calls[0]["labels"] == ("V", "H", "tie")


def test_plot_reverse_hybrid_best_fraction_bar_uses_pairwise_v_h2_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    calls = []

    def fake_plot_best_fraction_bar(ax, delta_table, *, labels, colors):
        calls.append(
            {
                "ax": ax,
                "winners": list(delta_table["winner"]),
                "labels": labels,
                "colors": colors,
            }
        )

    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_best_fraction_bar",
        fake_plot_best_fraction_bar,
    )

    table = pandas.DataFrame(
        {
            "winner": ["H", "MS", "V"],
            "delta_V_minus_H2_bits_per_spike": [-0.1, 0.0, 0.2],
        }
    )
    axis = object()
    supp_figure_4_module.plot_reverse_hybrid_glm_empirical_best_fraction_bar(
        axis,
        table,
    )

    assert calls[0]["ax"] is axis
    assert calls[0]["winners"] == ["H2", "tie", "V"]
    assert calls[0]["labels"] == ("V", "H2", "tie")


def test_make_supplementary_figure_4_plots_figure_4c_histograms_per_animal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}
    load_calls = []
    example_calls = []
    scalar_plot_calls = []
    plot_tables = []
    empirical_load_calls = []
    empirical_plot_tables = []
    empirical_best_tables = []
    glm_load_calls = []
    glm_plot_tables = []
    glm_best_tables = []
    mixed_load_calls = []
    mixed_plot_tables = []
    mixed_best_tables = []
    mixed_full_load_calls = []
    mixed_full_plot_tables = []
    mixed_full_best_tables = []
    empirical_multiplicative_plot_tables = []
    empirical_multiplicative_best_tables = []
    hybrid_load_calls = []
    hybrid_plot_tables = []
    hybrid_best_tables = []
    reverse_hybrid_plot_tables = []
    reverse_hybrid_best_tables = []
    multiplier_load_calls = []
    multiplier_plot_tables = []
    full_gain_load_calls = []
    full_gain_plot_tables = []
    combined_gain_plot_tables = []
    nested_modulation_load_calls = []
    nested_modulation_plot_tables = []

    def fake_load_panel_h_swap_delta_table(**kwargs: object):
        load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "light_train_epoch": ["02_r1"],
                "light_test_epoch": ["06_r3"],
                "delta_ll_bits_per_spike": [0.2],
            }
        )

    def fake_plot_figure_4c_histogram_grid(ax, table):
        plot_tables.append(table)
        ax.text(0.5, 0.5, "4C")

    def fake_load_panel_h_swap_examples(**kwargs: object):
        example_calls.append(kwargs)
        return [{"model_name": SCALAR_MODEL_NAME}]

    def fake_plot_panel_h_swap_delta(
        ax,
        swap_delta_table,
        swap_examples,
        **kwargs: object,
    ):
        scalar_plot_calls.append(
            {
                "table": swap_delta_table,
                "examples": swap_examples,
                "kwargs": kwargs,
            }
        )
        ax.text(0.5, 0.5, "scalar")

    def fake_load_empirical_pairwise_delta_table(**kwargs: object):
        empirical_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "unit": [1],
                "winner": ["V"],
                "delta_V_minus_MS_bits_per_s": [0.4],
                "delta_V_minus_AS_bits_per_s": [0.6],
                "delta_V_minus_MS_bits_per_spike": [0.04],
                "delta_V_minus_AS_bits_per_spike": [0.06],
            }
        )

    def fake_plot_empirical_pairwise_delta(ax, table):
        empirical_plot_tables.append(table)
        ax.text(0.5, 0.5, "empirical")

    def fake_plot_empirical_best_fraction_bar(ax, table):
        empirical_best_tables.append(table)
        ax.text(0.5, 0.5, "best")

    def fake_load_glm_scalar_delta_table(**kwargs: object):
        glm_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "unit": [1],
                "winner": ["V"],
                "delta_V_minus_scalar_bits_per_spike": [0.03],
            }
        )

    def fake_plot_glm_scalar_pairwise_delta(ax, table):
        glm_plot_tables.append(table)
        ax.text(0.5, 0.5, "glm")

    def fake_plot_glm_scalar_best_fraction_bar(ax, table):
        glm_best_tables.append(table)
        ax.text(0.5, 0.5, "glm best")

    def fake_load_mixed_glm_empirical_delta_table(**kwargs: object):
        mixed_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "unit": [1],
                "winner": ["V"],
                "delta_V_minus_task_bits_per_spike": [0.04],
                "delta_V_minus_AS_bits_per_spike": [0.06],
            }
        )

    def fake_plot_mixed_glm_empirical_pairwise_delta(ax, table, **kwargs: object):
        mixed_plot_tables.append((table, kwargs))
        ax.text(0.5, 0.5, "mixed")

    def fake_plot_mixed_glm_empirical_best_fraction_bar(ax, table):
        mixed_best_tables.append(table)
        ax.text(0.5, 0.5, "mixed best")

    def fake_load_mixed_glm_full_additive_delta_table(**kwargs: object):
        mixed_full_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "unit": [1],
                "winner": ["A"],
                "delta_V_minus_task_bits_per_spike": [0.04],
                "delta_V_minus_A_bits_per_spike": [-0.02],
            }
        )

    def fake_plot_mixed_glm_full_additive_pairwise_delta(
        ax,
        table,
        **kwargs: object,
    ):
        mixed_full_plot_tables.append((table, kwargs))
        ax.text(0.5, 0.5, "mixed full")

    def fake_plot_mixed_glm_full_additive_best_fraction_bar(ax, table):
        mixed_full_best_tables.append(table)
        ax.text(0.5, 0.5, "mixed full best")

    def fake_plot_empirical_multiplicative_pairwise_delta(ax, table):
        empirical_multiplicative_plot_tables.append(table)
        ax.text(0.5, 0.5, "empirical ms")

    def fake_plot_empirical_multiplicative_best_fraction_bar(ax, table):
        empirical_multiplicative_best_tables.append(table)
        ax.text(0.5, 0.5, "empirical ms best")

    def fake_load_hybrid_glm_empirical_delta_table(**kwargs: object):
        hybrid_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "unit": [1],
                "winner": ["H"],
                "delta_V_minus_task_bits_per_spike": [0.04],
                "delta_V_minus_H_bits_per_spike": [-0.02],
                "delta_V_minus_H2_bits_per_spike": [0.03],
            }
        )

    def fake_plot_hybrid_glm_empirical_pairwise_delta(ax, table):
        hybrid_plot_tables.append(table)
        ax.text(0.5, 0.5, "hybrid")

    def fake_plot_hybrid_glm_empirical_best_fraction_bar(ax, table):
        hybrid_best_tables.append(table)
        ax.text(0.5, 0.5, "hybrid best")

    def fake_plot_reverse_hybrid_glm_empirical_pairwise_delta(ax, table):
        reverse_hybrid_plot_tables.append(table)
        ax.text(0.5, 0.5, "reverse hybrid")

    def fake_plot_reverse_hybrid_glm_empirical_best_fraction_bar(ax, table):
        reverse_hybrid_best_tables.append(table)
        ax.text(0.5, 0.5, "reverse hybrid best")

    def fake_load_scalar_multiplier_table(**kwargs: object):
        multiplier_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "unit": [1],
                "log_empirical_ms_gain": [0.20],
                "log_glm_segment_gain": [0.10],
                "log_glm_full_gain": [0.25],
            }
        )

    def fake_plot_scalar_multiplier_histograms(ax, table):
        multiplier_plot_tables.append(table)
        ax.text(0.5, 0.5, "multipliers")

    def fake_load_full_segment_log_gain_table(**kwargs: object):
        full_gain_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "region": ["v1"],
                "trajectory": ["center_to_left"],
                "segment_index_1based": [1],
                "unit": [1],
                "full_segment_log_gain": [0.35],
            }
        )

    def fake_plot_full_segment_log_gain_histograms(ax, table):
        full_gain_plot_tables.append(table)
        ax.text(0.5, 0.5, "full gains")

    def fake_plot_combined_full_segment_log_gain_histogram(ax, table):
        combined_gain_plot_tables.append(table)
        ax.text(0.5, 0.5, "combined gains")

    def fake_load_nested_vision_modulation_table(**kwargs: object):
        nested_modulation_load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "total_cell_count": [10],
                "dark_inactive_count": [2],
                "dark_active_count": [8],
                "dark_active_unstable_count": [3],
                "dark_active_stable_count": [5],
                "dark_active_stable_no_scalar_fit_count": [1],
                "dark_active_stable_unmodulated_count": [2],
                "dark_active_stable_modulated_count": [2],
            }
        )

    def fake_plot_nested_vision_modulation_bar(ax, table, **kwargs: object):
        nested_modulation_plot_tables.append((table, kwargs))
        ax.text(0.5, 0.5, "nested")

    def fake_save_figure(figure, output_path: Path, dpi: int, **kwargs: object):
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        return output_path

    monkeypatch.setattr(
        supp_figure_4_module,
        "load_panel_h_swap_delta_table",
        fake_load_panel_h_swap_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_figure_4c_histogram_grid",
        fake_plot_figure_4c_histogram_grid,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_panel_h_swap_examples",
        fake_load_panel_h_swap_examples,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_panel_h_swap_delta",
        fake_plot_panel_h_swap_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_empirical_pairwise_delta_table",
        fake_load_empirical_pairwise_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_empirical_pairwise_delta",
        fake_plot_empirical_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_empirical_best_fraction_bar",
        fake_plot_empirical_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_glm_scalar_delta_table",
        fake_load_glm_scalar_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_glm_scalar_pairwise_delta",
        fake_plot_glm_scalar_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_glm_scalar_best_fraction_bar",
        fake_plot_glm_scalar_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_mixed_glm_empirical_delta_table",
        fake_load_mixed_glm_empirical_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_mixed_glm_empirical_pairwise_delta",
        fake_plot_mixed_glm_empirical_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_mixed_glm_empirical_best_fraction_bar",
        fake_plot_mixed_glm_empirical_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_mixed_glm_full_additive_delta_table",
        fake_load_mixed_glm_full_additive_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_mixed_glm_full_additive_pairwise_delta",
        fake_plot_mixed_glm_full_additive_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_mixed_glm_full_additive_best_fraction_bar",
        fake_plot_mixed_glm_full_additive_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_empirical_multiplicative_pairwise_delta",
        fake_plot_empirical_multiplicative_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_empirical_multiplicative_best_fraction_bar",
        fake_plot_empirical_multiplicative_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_hybrid_glm_empirical_delta_table",
        fake_load_hybrid_glm_empirical_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_hybrid_glm_empirical_pairwise_delta",
        fake_plot_hybrid_glm_empirical_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_hybrid_glm_empirical_best_fraction_bar",
        fake_plot_hybrid_glm_empirical_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_reverse_hybrid_glm_empirical_pairwise_delta",
        fake_plot_reverse_hybrid_glm_empirical_pairwise_delta,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_reverse_hybrid_glm_empirical_best_fraction_bar",
        fake_plot_reverse_hybrid_glm_empirical_best_fraction_bar,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_scalar_multiplier_table",
        fake_load_scalar_multiplier_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_scalar_multiplier_histograms",
        fake_plot_scalar_multiplier_histograms,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_full_segment_log_gain_table",
        fake_load_full_segment_log_gain_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_full_segment_log_gain_histograms",
        fake_plot_full_segment_log_gain_histograms,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_combined_full_segment_log_gain_histogram",
        fake_plot_combined_full_segment_log_gain_histogram,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "load_nested_vision_modulation_table",
        fake_load_nested_vision_modulation_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_nested_vision_modulation_bar",
        fake_plot_nested_vision_modulation_bar,
    )
    monkeypatch.setattr(supp_figure_4_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_4.svg"
    datasets = [("L14", "20240611", "08_r4"), ("L15", "20241121", "10_r5")]
    saved_path = make_supplementary_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        region="v1",
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(
        (
            SCALAR_PANEL_HEIGHT_MM
            + 2 * DEFAULT_ANIMAL_ROW_HEIGHT_MM
            + EMPIRICAL_PAIRWISE_PANEL_HEIGHT_MM
            + GLM_SCALAR_PANEL_HEIGHT_MM
            + MIXED_GLM_EMPIRICAL_PANEL_HEIGHT_MM
            + HYBRID_GLM_EMPIRICAL_PANEL_HEIGHT_MM
            + MULTIPLIER_PANEL_HEIGHT_MM
            + FULL_SEGMENT_GAIN_PANEL_HEIGHT_MM
            + COMBINED_FULL_SEGMENT_GAIN_PANEL_HEIGHT_MM
            + NESTED_MODULATION_PANEL_HEIGHT_MM
        )
        / 25.4
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
    ]
    assert [call["datasets"] for call in load_calls] == [
        datasets,
        [datasets[0]],
        [datasets[1]],
    ]
    assert all(call["region"] == "v1" for call in load_calls)
    assert load_calls[0]["model_name"] == SCALAR_MODEL_NAME
    assert load_calls[0]["min_tuning_stability_correlation"] == pytest.approx(
        figure_3_module.PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    )
    assert "model_name" not in load_calls[1]
    assert example_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
            "model_name": SCALAR_MODEL_NAME,
            "example_count": len(SCALAR_PANEL_EXAMPLES),
            "requested_examples": SCALAR_PANEL_EXAMPLES,
        }
    ]
    assert len(scalar_plot_calls) == 1
    assert scalar_plot_calls[0]["examples"] == [{"model_name": SCALAR_MODEL_NAME}]
    assert scalar_plot_calls[0]["kwargs"]["model_name"] == SCALAR_MODEL_NAME
    assert scalar_plot_calls[0]["kwargs"]["model_colors"] == SCALAR_PANEL_MODEL_COLORS
    assert scalar_plot_calls[0]["kwargs"]["example_axis_bounds"] == (
        SCALAR_PANEL_EXAMPLE_AXIS_BOUNDS
    )
    assert scalar_plot_calls[0]["kwargs"]["example_delta_label_positions"] == (
        SCALAR_PANEL_EXAMPLE_DELTA_LABEL_POSITIONS
    )
    assert scalar_plot_calls[0]["kwargs"][
        "example_delta_label_vertical_alignments"
    ] == SCALAR_PANEL_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS
    assert len(plot_tables) == 2
    assert empirical_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
        }
    ]
    assert len(empirical_plot_tables) == 1
    assert len(empirical_best_tables) == 1
    assert glm_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
        }
    ]
    assert len(glm_plot_tables) == 1
    assert len(glm_best_tables) == 1
    assert mixed_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
        }
    ]
    assert len(mixed_plot_tables) == 1
    assert mixed_plot_tables[0][1] == {"show_legend": False}
    assert len(mixed_best_tables) == 1
    assert mixed_full_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
        }
    ]
    assert len(mixed_full_plot_tables) == 1
    assert mixed_full_plot_tables[0][1] == {"show_legend": False}
    assert len(mixed_full_best_tables) == 1
    assert len(empirical_multiplicative_plot_tables) == 1
    assert empirical_multiplicative_plot_tables[0] is empirical_plot_tables[0]
    assert len(empirical_multiplicative_best_tables) == 1
    assert empirical_multiplicative_best_tables[0] is empirical_plot_tables[0]
    assert hybrid_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
        }
    ]
    assert len(hybrid_plot_tables) == 1
    assert len(hybrid_best_tables) == 1
    assert len(reverse_hybrid_plot_tables) == 1
    assert reverse_hybrid_plot_tables[0] is hybrid_plot_tables[0]
    assert len(reverse_hybrid_best_tables) == 1
    assert reverse_hybrid_best_tables[0] is hybrid_plot_tables[0]
    assert multiplier_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
        }
    ]
    assert len(multiplier_plot_tables) == 1
    assert full_gain_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
            "min_tuning_stability_correlation": (
                FULL_SEGMENT_GAIN_MIN_TUNING_STABILITY_CORRELATION
            ),
        }
    ]
    assert len(full_gain_plot_tables) == 1
    assert len(combined_gain_plot_tables) == 1
    assert combined_gain_plot_tables[0] is full_gain_plot_tables[0]
    assert nested_modulation_load_calls == [
        {
            "data_root": Path("/analysis"),
            "datasets": datasets,
            "region": "v1",
            "dark_epoch": None,
            "full_gain_table": full_gain_plot_tables[0],
        }
    ]
    assert len(nested_modulation_plot_tables) == 1
    assert nested_modulation_plot_tables[0][0].equals(
        pandas.DataFrame(
            {
                "total_cell_count": [10],
                "dark_inactive_count": [2],
                "dark_active_count": [8],
                "dark_active_unstable_count": [3],
                "dark_active_stable_count": [5],
                "dark_active_stable_no_scalar_fit_count": [1],
                "dark_active_stable_unmodulated_count": [2],
                "dark_active_stable_modulated_count": [2],
            }
        )
    )
    assert nested_modulation_plot_tables[0][1]["show_note"] is False
    assert nested_modulation_plot_tables[0][1]["legend_axis"] is not None
