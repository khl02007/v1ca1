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
