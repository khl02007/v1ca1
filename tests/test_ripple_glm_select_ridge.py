from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v1ca1.ripple.ripple_glm_select_ridge import (
    collect_saved_output_records,
    compute_epoch_median_devexp,
    load_epoch_score_row,
    parse_saved_output_path,
    summarize_setup_group,
)


def test_parse_saved_output_path_extracts_setup_metadata() -> None:
    parsed = parse_saved_output_path(
        Path("01_s1_rw_0p2s_deduped_ridge_1e-1_samplewise_ripple_glm.nc")
    )

    assert parsed["epoch"] == "01_s1"
    assert parsed["ripple_window_s"] == pytest.approx(0.2)
    assert parsed["ripple_window_offset_s"] == pytest.approx(0.0)
    assert parsed["ripple_window_suffix"] == "rw_0p2s"
    assert parsed["ripple_selection_mode"] == "deduped"
    assert parsed["ridge_strength"] == pytest.approx(1e-1)
    assert parsed["ridge_strength_suffix"] == "ridge_1e-1"
    assert parsed["setup_label"] == "rw_0p2s_deduped"


def test_parse_saved_output_path_extracts_optional_offset_metadata() -> None:
    parsed = parse_saved_output_path(
        Path("01_s1_rw_0p2s_off_m0p05s_single_ridge_1e-1_samplewise_ripple_glm.nc")
    )

    assert parsed["epoch"] == "01_s1"
    assert parsed["ripple_window_s"] == pytest.approx(0.2)
    assert parsed["ripple_window_offset_s"] == pytest.approx(-0.05)
    assert parsed["ripple_window_suffix"] == "rw_0p2s_off_m0p05s"
    assert parsed["ripple_selection_mode"] == "single"
    assert parsed["setup_label"] == "rw_0p2s_off_m0p05s_single"


def test_collect_saved_output_records_filters_requested_setup(tmp_path) -> None:
    filenames = [
        "01_s1_rw_0p2s_deduped_ridge_1e-1_samplewise_ripple_glm.nc",
        "02_r1_rw_0p2s_deduped_ridge_1e-2_samplewise_ripple_glm.nc",
        "01_s1_rw_0p2s_off_m0p05s_deduped_ridge_1e-1_samplewise_ripple_glm.nc",
        "01_s1_rw_0p1s_allripples_ridge_1e-1_samplewise_ripple_glm.nc",
    ]
    for name in filenames:
        (tmp_path / name).touch()

    records = collect_saved_output_records(
        tmp_path,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection_mode="deduped",
    )

    assert set(records["ripple_window_suffix"]) == {"rw_0p2s"}
    assert set(records["ripple_window_offset_s"]) == {0.0}
    assert set(records["ripple_selection_mode"]) == {"deduped"}
    assert set(records["epoch"]) == {"01_s1", "02_r1"}


def test_compute_epoch_median_devexp_averages_folds_then_median_units() -> None:
    scores = compute_epoch_median_devexp(
        np.array(
            [
                [0.1, 0.2, np.nan],
                [0.3, 0.4, np.nan],
            ],
            dtype=float,
        ),
        min_finite_units=2,
    )

    assert scores["epoch_score_valid"] is True
    assert scores["n_finite_units"] == 2
    assert scores["epoch_median_devexp"] == pytest.approx(0.25)


def test_summarize_setup_group_prefers_larger_ridge_when_scores_tie() -> None:
    group = pd.DataFrame(
        [
            {
                "epoch": "01_s1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "deduped",
                "ridge_strength": 1e-1,
                "ridge_strength_suffix": "ridge_1e-1",
                "output_path": Path("r1_e1.nc"),
                "setup_label": "rw_0p2s_deduped",
            },
            {
                "epoch": "02_r1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "deduped",
                "ridge_strength": 1e-1,
                "ridge_strength_suffix": "ridge_1e-1",
                "output_path": Path("r1_e2.nc"),
                "setup_label": "rw_0p2s_deduped",
            },
            {
                "epoch": "01_s1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "deduped",
                "ridge_strength": 1e-2,
                "ridge_strength_suffix": "ridge_1e-2",
                "output_path": Path("r2_e1.nc"),
                "setup_label": "rw_0p2s_deduped",
            },
            {
                "epoch": "02_r1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "deduped",
                "ridge_strength": 1e-2,
                "ridge_strength_suffix": "ridge_1e-2",
                "output_path": Path("r2_e2.nc"),
                "setup_label": "rw_0p2s_deduped",
            },
        ]
    )
    score_map = {
        "r1_e1.nc": 0.20,
        "r1_e2.nc": 0.20,
        "r2_e1.nc": 0.205,
        "r2_e2.nc": 0.205,
    }

    def fake_loader(output_path: Path, *, min_finite_units: int) -> dict[str, float | int | bool]:
        return {
            "epoch_median_devexp": score_map[output_path.name],
            "n_finite_units": min_finite_units,
            "n_units_total": min_finite_units,
            "n_folds": 5,
            "n_ripples": 10,
            "epoch_score_valid": True,
        }

    summary_rows, epoch_rows = summarize_setup_group(
        group,
        tie_tol=0.01,
        min_common_epochs=2,
        min_finite_units=5,
        score_loader=fake_loader,
    )

    summary_table = pd.DataFrame(summary_rows)
    epoch_table = pd.DataFrame(epoch_rows)
    selected_row = summary_table.loc[summary_table["selected"]].iloc[0]

    assert selected_row["ridge_strength"] == pytest.approx(1e-1)
    assert selected_row["epoch_win_count"] == 2
    assert epoch_table.loc[
        np.isclose(epoch_table["ridge_strength"].to_numpy(dtype=float), 1e-1),
        "epoch_is_winner",
    ].all()


def test_summarize_setup_group_requires_valid_epoch_intersection() -> None:
    group = pd.DataFrame(
        [
            {
                "epoch": "01_s1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "single",
                "ridge_strength": 1e-1,
                "ridge_strength_suffix": "ridge_1e-1",
                "output_path": Path("r1_e1.nc"),
                "setup_label": "rw_0p2s_single",
            },
            {
                "epoch": "02_r1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "single",
                "ridge_strength": 1e-1,
                "ridge_strength_suffix": "ridge_1e-1",
                "output_path": Path("r1_e2.nc"),
                "setup_label": "rw_0p2s_single",
            },
            {
                "epoch": "01_s1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "single",
                "ridge_strength": 1e-2,
                "ridge_strength_suffix": "ridge_1e-2",
                "output_path": Path("r2_e1.nc"),
                "setup_label": "rw_0p2s_single",
            },
            {
                "epoch": "02_r1",
                "ripple_window_s": 0.2,
                "ripple_window_suffix": "rw_0p2s",
                "ripple_selection_mode": "single",
                "ridge_strength": 1e-2,
                "ridge_strength_suffix": "ridge_1e-2",
                "output_path": Path("r2_e2.nc"),
                "setup_label": "rw_0p2s_single",
            },
        ]
    )
    score_map = {
        "r1_e1.nc": {"score": 0.2, "finite_units": 5, "valid": True},
        "r1_e2.nc": {"score": 0.3, "finite_units": 5, "valid": True},
        "r2_e1.nc": {"score": 0.1, "finite_units": 5, "valid": True},
        "r2_e2.nc": {"score": np.nan, "finite_units": 4, "valid": False},
    }

    def fake_loader(output_path: Path, *, min_finite_units: int) -> dict[str, float | int | bool]:
        payload = score_map[output_path.name]
        return {
            "epoch_median_devexp": payload["score"],
            "n_finite_units": payload["finite_units"],
            "n_units_total": min_finite_units,
            "n_folds": 5,
            "n_ripples": 10,
            "epoch_score_valid": payload["valid"],
        }

    summary_rows, epoch_rows = summarize_setup_group(
        group,
        tie_tol=0.01,
        min_common_epochs=2,
        min_finite_units=5,
        score_loader=fake_loader,
    )

    summary_table = pd.DataFrame(summary_rows)
    epoch_table = pd.DataFrame(epoch_rows)

    assert set(summary_table["selection_status"]) == {"insufficient_common_valid_epochs"}
    assert not summary_table["selected"].any()
    assert epoch_table["in_common_valid_epoch_set"].sum() == 2


def test_load_epoch_score_row_reads_saved_netcdf(tmp_path) -> None:
    xr = pytest.importorskip("xarray")

    dataset = xr.Dataset(
        {
            "devexp_ripple_folds": (
                ("fold", "unit"),
                np.array(
                    [
                        [0.1, 0.2, np.nan],
                        [0.3, 0.4, np.nan],
                    ],
                    dtype=float,
                ),
            )
        },
        attrs={"n_ripples": 7},
    )
    output_path = tmp_path / "01_s1_rw_0p2s_deduped_ridge_1e-1_samplewise_ripple_glm.nc"
    dataset.to_netcdf(output_path)

    score_row = load_epoch_score_row(output_path, min_finite_units=2)

    assert score_row["epoch_score_valid"] is True
    assert score_row["n_ripples"] == 7
    assert score_row["epoch_median_devexp"] == pytest.approx(0.25)
