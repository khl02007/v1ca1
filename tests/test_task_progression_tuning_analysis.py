from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


MODULE_NAME = "v1ca1.task_progression.tuning_analysis"


def _reload_tuning_analysis_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _make_curve(values_by_unit: dict[int, list[float]]) -> xr.DataArray:
    units = np.asarray(sorted(values_by_unit), dtype=int)
    values = np.asarray([values_by_unit[unit] for unit in units], dtype=float)
    return xr.DataArray(
        values,
        dims=("unit", "tp"),
        coords={
            "unit": units,
            "tp": np.arange(values.shape[1], dtype=float),
        },
    )


def _make_all_unit_curves() -> dict[str, xr.DataArray]:
    """Return four curves with valid, low-rate, and constant units."""
    return {
        "center_to_left": _make_curve(
            {
                11: [6.0, 4.0, 2.0],
                12: [6.0, 4.0, 2.0],
                13: [1.0, 1.0, 1.0],
            }
        ),
        "left_to_center": _make_curve(
            {
                11: [2.0, 4.0, 6.0],
                12: [2.0, 4.0, 6.0],
                13: [1.0, 1.0, 1.0],
            }
        ),
        "center_to_right": _make_curve(
            {
                11: [2.0, 4.0, 6.0],
                12: [2.0, 4.0, 6.0],
                13: [1.0, 1.0, 1.0],
            }
        ),
        "right_to_center": _make_curve(
            {
                11: [6.0, 4.0, 2.0],
                12: [6.0, 4.0, 2.0],
                13: [1.0, 1.0, 1.0],
            }
        ),
    }


def test_shared_similarity_core_remains_available_from_tuning_analysis() -> None:
    module = _reload_tuning_analysis_module()
    similarity = importlib.import_module("v1ca1.task_progression.similarity")

    assert module.DIRECT_COMPARISON_SPECS is similarity.DIRECT_COMPARISON_SPECS
    assert module.SIMILARITY_QC_COLUMNS is similarity.SIMILARITY_QC_COLUMNS
    assert module.compute_similarity_score is similarity.compute_similarity_score
    assert (
        module.compute_similarity_score_with_qc
        is similarity.compute_similarity_score_with_qc
    )
    assert np.array_equal(
        module._flip_curve_if_requested(
            np.asarray([1.0, 2.0, 3.0]),
            should_flip=True,
        ),
        np.asarray([3.0, 2.0, 1.0]),
    )


def test_compute_epoch_similarity_table_adds_same_turn_and_flipped_same_arm_rows() -> None:
    module = _reload_tuning_analysis_module()

    tuning_curves_by_trajectory = {
        "center_to_left": _make_curve(
            {
                11: [6.0, 4.0, 2.0],
                12: [6.0, 4.0, 2.0],
            }
        ),
        "left_to_center": _make_curve(
            {
                11: [2.0, 4.0, 6.0],
                12: [2.0, 4.0, 6.0],
            }
        ),
        "center_to_right": _make_curve(
            {
                11: [2.0, 4.0, 6.0],
                12: [2.0, 4.0, 6.0],
            }
        ),
        "right_to_center": _make_curve(
            {
                11: [6.0, 4.0, 2.0],
                12: [6.0, 4.0, 2.0],
            }
        ),
    }
    epoch_firing_rates = pd.Series({11: 2.0, 12: 0.1}, dtype=float)

    table = module.compute_epoch_similarity_table(
        region="v1",
        epoch="02_r1",
        tuning_curves_by_trajectory=tuning_curves_by_trajectory,
        epoch_firing_rates=epoch_firing_rates,
        firing_rate_threshold_hz=0.5,
        similarity_metric="correlation",
    )

    assert list(table["comparison_label"].astype(str)) == [
        "left_turn",
        "right_turn",
        "left_arm",
        "right_arm",
    ]
    assert table["unit"].tolist() == [11, 11, 11, 11]
    assert np.allclose(table["similarity"], 1.0)

    same_arm = table[table["comparison_family"] == "same_arm"].reset_index(drop=True)
    assert same_arm["comparison_label"].astype(str).tolist() == ["left_arm", "right_arm"]
    assert same_arm["flip_trajectory_b"].tolist() == [True, True]
    assert same_arm["trajectory_b"].tolist() == ["left_to_center", "right_to_center"]


def test_retain_all_units_preserves_scores_and_adds_qc() -> None:
    module = _reload_tuning_analysis_module()
    tuning_curves = _make_all_unit_curves()
    firing_rates = pd.Series({11: 2.0, 12: 0.1, 13: 0.0}, dtype=float)

    legacy = module.compute_epoch_similarity_table(
        region="v1",
        epoch="02_r1",
        tuning_curves_by_trajectory=tuning_curves,
        epoch_firing_rates=firing_rates,
        firing_rate_threshold_hz=0.5,
        similarity_metric="correlation",
    )
    complete = module.compute_epoch_similarity_table(
        region="v1",
        epoch="02_r1",
        tuning_curves_by_trajectory=tuning_curves,
        epoch_firing_rates=firing_rates,
        firing_rate_threshold_hz=0.5,
        similarity_metric="correlation",
        retain_all_units=True,
    )

    assert len(complete) == 3 * len(module.DIRECT_COMPARISON_LABELS)
    assert complete.groupby("unit", observed=False).size().to_dict() == {
        11: 4,
        12: 4,
        13: 4,
    }
    assert set(complete["comparison_label"].astype(str)) == set(
        module.DIRECT_COMPARISON_LABELS
    )
    assert not complete["comparison_label"].astype(str).isin(
        module.POOLED_COMPARISON_LABELS
    ).any()
    assert set(module.SIMILARITY_QC_COLUMNS).issubset(complete.columns)

    low_rate = complete[complete["unit"] == 12]
    assert np.allclose(low_rate["firing_rate_hz"], 0.1)
    assert np.isfinite(low_rate["similarity"]).all()
    assert low_rate["similarity_status"].eq("valid").all()

    constant = complete[complete["unit"] == 13]
    assert constant["similarity"].isna().all()
    assert constant["similarity_status"].eq("nonfinite_similarity").all()
    assert constant["n_trajectory_a_finite_bins"].eq(3).all()
    assert constant["n_trajectory_b_finite_bins"].eq(3).all()
    assert constant["n_paired_finite_bins"].eq(3).all()

    shared = legacy.merge(
        complete,
        on=["unit", "comparison_label"],
        suffixes=("_legacy", "_complete"),
        validate="one_to_one",
    )
    assert np.array_equal(
        shared["similarity_legacy"].to_numpy(dtype=float),
        shared["similarity_complete"].to_numpy(dtype=float),
    )

    legacy_final = module.finalize_epoch_similarity_table(
        legacy,
        retain_all_units=False,
    )
    complete_final = module.finalize_epoch_similarity_table(
        complete,
        retain_all_units=True,
    )
    assert legacy_final["comparison_label"].astype(str).isin(
        module.POOLED_COMPARISON_LABELS
    ).any()
    assert not complete_final["comparison_label"].astype(str).isin(
        module.POOLED_COMPARISON_LABELS
    ).any()

    comparison = module.build_epoch_comparison_table(
        complete,
        complete.assign(epoch="08_r4"),
        region="v1",
        epoch_a="02_r1",
        epoch_b="08_r4",
    )
    assert len(comparison) == len(complete)
    assert "similarity_status_epoch_a" in comparison
    assert "similarity_status_epoch_b" in comparison
    invalid_comparison = comparison[comparison["unit"] == 13]
    assert invalid_comparison["delta_similarity"].isna().all()
    assert invalid_comparison["similarity_status_epoch_a"].eq(
        "nonfinite_similarity"
    ).all()

    with pytest.raises(ValueError, match="complete all-unit QC schema"):
        module.build_epoch_comparison_table(
            complete.drop(columns=["similarity_status"]),
            complete.assign(epoch="08_r4"),
            region="v1",
            epoch_a="02_r1",
            epoch_b="08_r4",
        )


def test_similarity_qc_counts_raw_support_without_changing_interpolation() -> None:
    module = _reload_tuning_analysis_module()
    curve_a = np.asarray([1.0, np.nan, 3.0])
    curve_b = np.asarray([1.0, 2.0, np.nan])

    result = module.compute_similarity_score_with_qc(
        curve_a,
        curve_b,
        similarity_metric="correlation",
    )

    assert result["similarity"] == module.compute_similarity_score(
        curve_a,
        curve_b,
        similarity_metric="correlation",
    )
    assert result["similarity_status"] == "valid"
    assert result["n_trajectory_a_finite_bins"] == 2
    assert result["n_trajectory_b_finite_bins"] == 2
    assert result["n_paired_finite_bins"] == 1


def test_retain_all_units_requires_matching_curve_unit_sets() -> None:
    module = _reload_tuning_analysis_module()
    tuning_curves = _make_all_unit_curves()
    tuning_curves["right_to_center"] = tuning_curves["right_to_center"].sel(
        unit=[11, 12]
    )

    with pytest.raises(ValueError, match="right_to_center"):
        module.compute_epoch_similarity_table(
            region="v1",
            epoch="02_r1",
            tuning_curves_by_trajectory=tuning_curves,
            epoch_firing_rates=pd.Series(
                {11: 2.0, 12: 0.1, 13: 0.0},
                dtype=float,
            ),
            firing_rate_threshold_hz=0.5,
            similarity_metric="correlation",
            retain_all_units=True,
        )


def test_append_pooled_similarity_rows_uses_within_family_max() -> None:
    module = _reload_tuning_analysis_module()

    direct_rows = pd.DataFrame(
        [
            {
                "unit": 11,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_turn",
                "comparison_label": "left_turn",
                "side": "left",
                "trajectory_a": "center_to_left",
                "trajectory_b": "right_to_center",
                "flip_trajectory_b": False,
                "firing_rate_hz": 2.0,
                "similarity": 0.2,
            },
            {
                "unit": 11,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_turn",
                "comparison_label": "right_turn",
                "side": "right",
                "trajectory_a": "center_to_right",
                "trajectory_b": "left_to_center",
                "flip_trajectory_b": False,
                "firing_rate_hz": 2.0,
                "similarity": 0.6,
            },
            {
                "unit": 11,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_arm",
                "comparison_label": "left_arm",
                "side": "left",
                "trajectory_a": "center_to_left",
                "trajectory_b": "left_to_center",
                "flip_trajectory_b": True,
                "firing_rate_hz": 2.0,
                "similarity": 0.4,
            },
            {
                "unit": 11,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_arm",
                "comparison_label": "right_arm",
                "side": "right",
                "trajectory_a": "center_to_right",
                "trajectory_b": "right_to_center",
                "flip_trajectory_b": True,
                "firing_rate_hz": 2.0,
                "similarity": 0.1,
            },
        ]
    )

    table = module.append_pooled_similarity_rows(direct_rows)
    pooled = table[table["comparison_label"].astype(str).isin(module.POOLED_COMPARISON_LABELS)]
    pooled = pooled.sort_values("comparison_label").reset_index(drop=True)

    assert pooled["comparison_label"].astype(str).tolist() == [
        "pooled_same_turn",
        "pooled_same_arm",
    ]
    assert np.allclose(pooled["similarity"], [0.6, 0.4])
    assert pooled["trajectory_a"].isna().all()
    assert pooled["trajectory_b"].isna().all()
    assert pooled["flip_trajectory_b"].isna().all()


def test_build_epoch_comparison_table_intersects_units_and_matching_labels() -> None:
    module = _reload_tuning_analysis_module()

    epoch_a = pd.DataFrame(
        [
            {
                "unit": 11,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_turn",
                "comparison_label": "left_turn",
                "side": "left",
                "trajectory_a": "center_to_left",
                "trajectory_b": "right_to_center",
                "flip_trajectory_b": False,
                "firing_rate_hz": 1.0,
                "similarity": 0.2,
                "p_value": 0.05,
            },
            {
                "unit": 11,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_turn",
                "comparison_label": "pooled_same_turn",
                "side": None,
                "trajectory_a": None,
                "trajectory_b": None,
                "flip_trajectory_b": None,
                "firing_rate_hz": 1.0,
                "similarity": 0.7,
                "p_value": np.nan,
            },
            {
                "unit": 12,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_family": "same_arm",
                "comparison_label": "left_arm",
                "side": "left",
                "trajectory_a": "center_to_left",
                "trajectory_b": "left_to_center",
                "flip_trajectory_b": True,
                "firing_rate_hz": 1.0,
                "similarity": 0.9,
                "p_value": 0.01,
            },
        ]
    )
    epoch_b = pd.DataFrame(
        [
            {
                "unit": 11,
                "region": "v1",
                "epoch": "08_r4",
                "comparison_family": "same_turn",
                "comparison_label": "left_turn",
                "side": "left",
                "trajectory_a": "center_to_left",
                "trajectory_b": "right_to_center",
                "flip_trajectory_b": False,
                "firing_rate_hz": 1.2,
                "similarity": 0.5,
                "p_value": 0.02,
            },
            {
                "unit": 11,
                "region": "v1",
                "epoch": "08_r4",
                "comparison_family": "same_turn",
                "comparison_label": "pooled_same_turn",
                "side": None,
                "trajectory_a": None,
                "trajectory_b": None,
                "flip_trajectory_b": None,
                "firing_rate_hz": 1.2,
                "similarity": 0.4,
                "p_value": np.nan,
            },
            {
                "unit": 13,
                "region": "v1",
                "epoch": "08_r4",
                "comparison_family": "same_arm",
                "comparison_label": "left_arm",
                "side": "left",
                "trajectory_a": "center_to_left",
                "trajectory_b": "left_to_center",
                "flip_trajectory_b": True,
                "firing_rate_hz": 1.2,
                "similarity": 0.8,
                "p_value": 0.03,
            },
        ]
    )

    comparison = module.build_epoch_comparison_table(
        epoch_a,
        epoch_b,
        region="v1",
        epoch_a="02_r1",
        epoch_b="08_r4",
    )

    assert comparison["unit"].tolist() == [11, 11]
    assert comparison["comparison_label"].astype(str).tolist() == [
        "left_turn",
        "pooled_same_turn",
    ]
    assert np.allclose(comparison["similarity_epoch_a"], [0.2, 0.7])
    assert np.allclose(comparison["similarity_epoch_b"], [0.5, 0.4])
    assert np.allclose(comparison["delta_similarity"], [0.3, -0.3])
    assert comparison.loc[0, "trajectory_b"] == "right_to_center"
    assert np.isnan(comparison.loc[1, "p_value_epoch_a"])


def test_resolve_compare_epochs_requires_membership_in_analyzed_set() -> None:
    module = _reload_tuning_analysis_module()

    assert module.resolve_compare_epochs(
        ["02_r1", "08_r4"],
        ["02_r1", "04_r2", "08_r4"],
    ) == ("02_r1", "08_r4")

    with pytest.raises(ValueError, match="Analyzed epochs"):
        module.resolve_compare_epochs(
            ["02_r1", "10_r5"],
            ["02_r1", "04_r2", "08_r4"],
        )


def test_all_unit_outputs_use_collision_safe_filenames(
    tmp_path,
    monkeypatch,
) -> None:
    module = _reload_tuning_analysis_module()
    parquet_paths = []
    figure_paths = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path: parquet_paths.append(path),
    )
    monkeypatch.setattr(
        module,
        "plot_epoch_similarity_distributions",
        lambda *args, **kwargs: figure_paths.append(kwargs["fig_path"]),
    )
    monkeypatch.setattr(
        module,
        "plot_epoch_comparison",
        lambda *args, **kwargs: figure_paths.append(kwargs["fig_path"]),
    )
    table = pd.DataFrame()

    legacy_table_path = module.save_epoch_similarity_table(
        table,
        data_dir=tmp_path,
        region="v1",
        epoch="02_r1",
        similarity_metric="absolute_overlap",
    )
    complete_table_path = module.save_epoch_similarity_table(
        table,
        data_dir=tmp_path,
        region="v1",
        epoch="02_r1",
        similarity_metric="absolute_overlap",
        retain_all_units=True,
    )
    complete_figure_paths = module.save_epoch_similarity_figures(
        table,
        fig_dir=tmp_path,
        region="v1",
        epoch="02_r1",
        similarity_metric="absolute_overlap",
        compute_significance=False,
        retain_all_units=True,
    )
    complete_comparison_path = module.save_epoch_comparison_table(
        table,
        data_dir=tmp_path,
        region="v1",
        epoch_a="02_r1",
        epoch_b="08_r4",
        similarity_metric="absolute_overlap",
        retain_all_units=True,
    )
    complete_comparison_figure = module.save_epoch_comparison_figure(
        table,
        fig_dir=tmp_path,
        region="v1",
        epoch_a="02_r1",
        epoch_b="08_r4",
        similarity_metric="absolute_overlap",
        retain_all_units=True,
    )

    assert legacy_table_path.name == (
        "v1_02_r1_absolute_overlap_within_epoch_similarity.parquet"
    )
    assert complete_table_path.name == (
        "v1_02_r1_absolute_overlap_within_epoch_similarity_all_units.parquet"
    )
    assert complete_figure_paths[0].name == (
        "v1_02_r1_absolute_overlap_similarity_distributions_all_units.png"
    )
    assert complete_comparison_path.name == (
        "v1_02_r1_08_r4_absolute_overlap_similarity_comparison_all_units.parquet"
    )
    assert complete_comparison_figure.name == (
        "v1_02_r1_08_r4_absolute_overlap_similarity_comparison_all_units.png"
    )
    assert parquet_paths == [
        legacy_table_path,
        complete_table_path,
        complete_comparison_path,
    ]
    assert figure_paths == [
        complete_figure_paths[0],
        complete_comparison_figure,
    ]


def test_all_unit_mode_rejects_significance() -> None:
    module = _reload_tuning_analysis_module()

    with pytest.raises(ValueError, match="cannot be combined"):
        module.validate_run_options(
            retain_all_units=True,
            compute_significance=True,
            n_shuffles=100,
        )
