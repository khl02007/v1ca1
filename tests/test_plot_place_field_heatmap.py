from __future__ import annotations

import pickle

import numpy as np
import pytest

import v1ca1.raster.plot_place_field_heatmap as heatmap
from v1ca1.raster.plot_place_field_heatmap import (
    align_and_normalize_panel_values,
    compute_unit_order,
    load_heatmap_trajectory_intervals,
    load_position_data_with_fallback,
    parse_arguments,
    smooth_values_nan_aware,
    split_odd_even_bounds,
)


def test_split_odd_even_bounds_uses_legacy_alternating_traversals() -> None:
    starts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ends = starts + 0.25

    odd_starts, odd_ends, even_starts, even_ends = split_odd_even_bounds(starts, ends)

    assert np.allclose(odd_starts, [1.0, 3.0, 5.0])
    assert np.allclose(odd_ends, [1.25, 3.25, 5.25])
    assert np.allclose(even_starts, [2.0, 4.0])
    assert np.allclose(even_ends, [2.25, 4.25])


def test_compute_unit_order_sorts_by_peak_bin_and_pushes_invalid_rows_last() -> None:
    values = np.array(
        [
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    order = compute_unit_order(values)

    assert np.array_equal(order, [1, 0, 2, 3])


def test_align_and_normalize_panel_values_handles_missing_units() -> None:
    display_values = np.array(
        [
            [2.0, 4.0, 0.0],
            [1.0, 3.0, 0.0],
        ]
    )
    display_units = np.array([30, 10])
    reference_units = np.array([10, 20, 30])
    unit_order = np.array([2, 0, 1])

    panel_values = align_and_normalize_panel_values(
        display_values,
        display_units,
        reference_units,
        unit_order,
    )

    assert panel_values.shape == (3, 3)
    assert np.allclose(panel_values[0], [0.5, 1.0, 0.0])
    assert np.allclose(panel_values[1], [1.0 / 3.0, 1.0, 0.0])
    assert np.isnan(panel_values[2]).all()


def test_smooth_values_nan_aware_preserves_fully_unsupported_rows() -> None:
    values = np.array(
        [
            [0.0, 1.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    smoothed = smooth_values_nan_aware(values, sigma_bins=1.0, axis=1)

    assert np.isnan(smoothed[1]).all()
    assert np.isfinite(smoothed[0, :3]).all()


def test_parse_arguments_does_not_expose_position_source_flags() -> None:
    args = parse_arguments(
        [
            "--animal-name",
            "L14",
            "--date",
            "20240611",
        ]
    )

    assert not hasattr(args, "position_source")
    assert not hasattr(args, "clean_dlc_input_dirname")
    assert not hasattr(args, "clean_dlc_input_name")


def test_load_heatmap_trajectory_intervals_accepts_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trajectory_intervals = {"02_r1": {"left_to_center": object()}}

    def fake_load_trajectory_intervals(analysis_path, run_epochs):
        assert analysis_path == tmp_path
        assert run_epochs == ["02_r1"]
        return trajectory_intervals, "parquet"

    monkeypatch.setattr(heatmap, "load_trajectory_intervals", fake_load_trajectory_intervals)

    loaded = load_heatmap_trajectory_intervals(tmp_path, ["02_r1"])

    assert loaded is trajectory_intervals


def test_load_heatmap_trajectory_intervals_rejects_pickle_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_load_trajectory_intervals(analysis_path, run_epochs):
        assert analysis_path == tmp_path
        assert run_epochs == ["02_r1"]
        return {"02_r1": {}}, "pickle"

    monkeypatch.setattr(heatmap, "load_trajectory_intervals", fake_load_trajectory_intervals)

    with pytest.raises(
        FileNotFoundError,
        match="trajectory_times.parquet.*get_trajectory_times",
    ):
        load_heatmap_trajectory_intervals(tmp_path, ["02_r1"])


def test_load_position_data_with_fallback_prefers_combined_clean_dlc_head(tmp_path) -> None:
    pd = pytest.importorskip("pandas")

    analysis_path = tmp_path / "L14" / "20240611"
    analysis_path.mkdir(parents=True)

    timestamps_position = {
        "02_r1": np.array([0.0, 0.1, 0.2], dtype=float),
        "04_r2": np.array([1.0, 1.1], dtype=float),
    }
    with open(analysis_path / "timestamps_position.pkl", "wb") as file:
        pickle.dump(timestamps_position, file)

    clean_dir = analysis_path / "dlc_position_cleaned"
    clean_dir.mkdir()
    table = pd.DataFrame(
        {
            "epoch": ["02_r1", "02_r1", "02_r1", "04_r2", "04_r2"],
            "frame": [0, 1, 2, 0, 1],
            "frame_time_s": [0.0, 0.1, 0.2, 1.0, 1.1],
            "head_x_cm": [1.0, 2.0, 3.0, 4.0, 5.0],
            "head_y_cm": [10.0, 20.0, 30.0, 40.0, 50.0],
            "body_x_cm": [6.0, 7.0, 8.0, 9.0, 10.0],
            "body_y_cm": [60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )
    table.to_parquet(clean_dir / "position.parquet", index=False)

    head_position, head_source = load_position_data_with_fallback(analysis_path)

    assert head_source.endswith("dlc_position_cleaned/position.parquet")
    assert np.allclose(head_position["02_r1"], [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


def test_load_position_data_with_fallback_uses_position_pkl_when_clean_dlc_missing(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_path = tmp_path / "L14" / "20240611"
    analysis_path.mkdir(parents=True)

    legacy_position = {
        "02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
    }
    with open(analysis_path / "position.pkl", "wb") as file:
        pickle.dump(legacy_position, file)

    position_by_epoch, source = load_position_data_with_fallback(analysis_path)
    captured = capsys.readouterr()

    assert source.endswith("position.pkl")
    assert "position.pkl is being used for position data" in captured.out
    assert np.allclose(position_by_epoch["02_r1"], [[1.0, 2.0], [3.0, 4.0]])


def test_load_position_data_with_fallback_fails_when_no_position_source_exists(
    tmp_path,
) -> None:
    analysis_path = tmp_path / "L14" / "20240611"
    analysis_path.mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match="Could not find cleaned DLC position, position.pkl, or body_position.pkl",
    ):
        load_position_data_with_fallback(analysis_path)
