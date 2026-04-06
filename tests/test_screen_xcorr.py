from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import v1ca1.xcorr.screen_xcorr as screen_xcorr


def test_parse_arguments_uses_modern_defaults() -> None:
    args = screen_xcorr.parse_arguments(
        [
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
            "--state",
            "ripple",
        ]
    )

    assert args.data_root == screen_xcorr.DEFAULT_DATA_ROOT
    assert args.bin_size_s == screen_xcorr.DEFAULT_BIN_SIZE_S
    assert args.max_lag_s == screen_xcorr.DEFAULT_MAX_LAG_S
    assert args.min_state_spikes == screen_xcorr.DEFAULT_MIN_STATE_SPIKES
    assert args.extremum_half_width_bins == screen_xcorr.DEFAULT_EXTREMUM_HALF_WIDTH_BINS
    assert args.display_vmax == screen_xcorr.DEFAULT_DISPLAY_VMAX
    assert args.show is False


def test_validate_arguments_rejects_ripple_window_offset_without_window() -> None:
    args = screen_xcorr.parse_arguments(
        [
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
            "--state",
            "ripple",
            "--ripple-window-offset-s",
            "0.05",
        ]
    )

    with pytest.raises(ValueError, match="--ripple-window-offset-s requires --ripple-window-s"):
        screen_xcorr.validate_arguments(args)

def test_summarize_pair_curve_uses_neighborhood_averaged_extremum() -> None:
    lag_times = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=float)
    xcorr_curve = np.array([1.0, 3.0, 3.0, 10.0, 4.0, 12.0, 3.0], dtype=float)

    summary = screen_xcorr.summarize_pair_curve(
        xcorr_curve=xcorr_curve,
        lag_times=lag_times,
        extremum_half_width_bins=1,
    )

    assert summary["status"] == screen_xcorr.PAIR_STATUS_VALID
    assert summary["peak_lag_s"] == pytest.approx(0.2)
    assert summary["peak_norm_xcorr"] == pytest.approx(np.mean([4.0, 12.0, 3.0]))


def test_summarize_pair_curve_marks_missing_finite_bins_invalid() -> None:
    lag_times = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=float)
    xcorr_curve = np.full(7, np.nan, dtype=float)

    summary = screen_xcorr.summarize_pair_curve(
        xcorr_curve=xcorr_curve,
        lag_times=lag_times,
        extremum_half_width_bins=1,
    )

    assert summary["status"] == screen_xcorr.PAIR_STATUS_NO_FINITE_BINS
    assert np.isnan(summary["peak_norm_xcorr"])


def test_order_v1_partners_sorts_by_lag_then_strength() -> None:
    pair_summary = pd.DataFrame(
        {
            "v1_unit_id": [101, 102, 103, 104],
            "peak_lag_s": [0.02, 0.03, -0.01, -0.02],
            "peak_norm_xcorr": [5.0, 3.0, 4.0, 6.0],
            "status": [screen_xcorr.PAIR_STATUS_VALID] * 4,
        },
        index=[10, 11, 12, 13],
    )

    order = screen_xcorr.order_v1_partners_for_ca1(pair_summary)

    assert np.array_equal(order, np.array([13, 12, 10, 11], dtype=int))


def test_order_ca1_units_by_best_partner_sorts_descending() -> None:
    pair_summary = pd.DataFrame(
        {
            "ca1_unit_id": [1, 1, 2, 2, 3],
            "peak_norm_xcorr": [2.0, 3.0, 1.0, 5.5, 4.0],
            "status": [
                screen_xcorr.PAIR_STATUS_VALID,
                screen_xcorr.PAIR_STATUS_VALID,
                screen_xcorr.PAIR_STATUS_VALID,
                screen_xcorr.PAIR_STATUS_VALID,
                screen_xcorr.PAIR_STATUS_VALID,
            ],
        }
    )

    ordered_units = screen_xcorr.order_ca1_units_by_best_partner(pair_summary)

    assert np.array_equal(ordered_units, np.array([2, 3, 1]))


def test_merge_interval_rows_merges_overlaps() -> None:
    merged = screen_xcorr.merge_interval_rows(
        pd.DataFrame(
            {
                "start": [0.30, 0.00, 0.10, 0.60],
                "end": [0.40, 0.20, 0.50, 0.70],
            }
        )
    )

    assert np.allclose(merged["start"], [0.0, 0.6])
    assert np.allclose(merged["end"], [0.5, 0.7])


def test_build_fixed_ripple_window_tables_clips_and_merges(monkeypatch) -> None:
    analysis_path = Path("/tmp/fake_session")
    ripple_tables = {
        "01_s1": pd.DataFrame(
            {
                "start_time": [0.10, 0.22, 0.90],
                "end_time": [0.15, 0.27, 0.95],
            }
        )
    }
    timestamps_by_epoch = {"01_s1": np.array([0.0, 0.5, 1.0], dtype=float)}

    monkeypatch.setattr(screen_xcorr, "load_ripple_tables", lambda _: (ripple_tables, "pickle"))

    intervals_by_epoch, source, metadata_by_epoch = screen_xcorr.build_fixed_ripple_window_tables(
        analysis_path,
        ripple_window_s=0.2,
        ripple_window_offset_s=-0.1,
        timestamps_by_epoch=timestamps_by_epoch,
    )

    assert source == "pickle"
    assert np.allclose(intervals_by_epoch["01_s1"]["start"], [0.0, 0.8])
    assert np.allclose(intervals_by_epoch["01_s1"]["end"], [0.32, 1.0])
    assert metadata_by_epoch["01_s1"] == {
        "original_ripple_count": 3,
        "clipped_window_count": 3,
        "merged_interval_count": 2,
    }


def test_get_output_dir_adds_ripple_window_suffix_when_requested(tmp_path) -> None:
    output_dir = screen_xcorr.get_output_dir(
        tmp_path,
        "ripple",
        "01_s1",
        max_lag_s=0.5,
        bin_size_s=0.005,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.05,
    )

    assert output_dir == (
        tmp_path
        / "xcorr"
        / "screen_pairs"
        / "ripple"
        / "rw_0p2s_off_0p05s"
        / "ml_0p5s_bs_0p005s"
        / "01_s1"
    )


def test_get_output_dir_adds_xcorr_settings_suffix_without_ripple_window(tmp_path) -> None:
    output_dir = screen_xcorr.get_output_dir(
        tmp_path,
        "ripple",
        "02_r1",
        max_lag_s=0.1,
        bin_size_s=0.01,
        ripple_window_s=None,
        ripple_window_offset_s=0.0,
    )

    assert output_dir == (
        tmp_path
        / "xcorr"
        / "screen_pairs"
        / "ripple"
        / "ml_0p1s_bs_0p01s"
        / "02_r1"
    )


def test_load_state_interval_tables_prefers_run_parquet(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "start": [1.0, 3.0],
            "end": [2.0, 4.0],
            "epoch": ["02_r1", "04_r2"],
        }
    ).to_parquet(analysis_path / "run_times.parquet", index=False)

    intervals_by_epoch, source = screen_xcorr.load_state_interval_tables(
        analysis_path,
        "run",
    )

    assert source == "parquet"
    assert sorted(intervals_by_epoch) == ["02_r1", "04_r2"]
    assert np.allclose(intervals_by_epoch["02_r1"]["start"], [1.0])
    assert np.allclose(intervals_by_epoch["04_r2"]["end"], [4.0])


def test_resolve_epoch_groups_defaults_to_epoch_by_epoch() -> None:
    epoch_groups = screen_xcorr.resolve_epoch_groups(
        saved_epoch_tags=["01_s1", "02_r1", "04_r2"],
        intervals_by_epoch={
            "02_r1": pd.DataFrame({"start": [1.0], "end": [2.0]}),
            "04_r2": pd.DataFrame({"start": [3.0], "end": [4.0]}),
        },
        requested_epochs=None,
    )

    assert epoch_groups == [("02_r1", ["02_r1"]), ("04_r2", ["04_r2"])]


def test_resolve_epoch_groups_supports_pooled_keyword() -> None:
    epoch_groups = screen_xcorr.resolve_epoch_groups(
        saved_epoch_tags=["01_s1", "02_r1", "04_r2"],
        intervals_by_epoch={
            "02_r1": pd.DataFrame({"start": [1.0], "end": [2.0]}),
            "04_r2": pd.DataFrame({"start": [3.0], "end": [4.0]}),
        },
        requested_epochs=["pooled"],
    )

    assert epoch_groups == [("pooled", ["02_r1", "04_r2"])]


def test_resolve_epoch_groups_rejects_pooled_plus_specific_epochs() -> None:
    with pytest.raises(ValueError, match="do not pass additional epoch labels"):
        screen_xcorr.resolve_epoch_groups(
            saved_epoch_tags=["02_r1", "04_r2"],
            intervals_by_epoch={
                "02_r1": pd.DataFrame({"start": [1.0], "end": [2.0]}),
                "04_r2": pd.DataFrame({"start": [3.0], "end": [4.0]}),
            },
            requested_epochs=["pooled", "02_r1"],
        )
