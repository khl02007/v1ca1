from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_2 as figure


def test_path_tuning_similarity_is_pathwise_gain_invariant() -> None:
    """The index should remove one multiplicative gain for one path."""
    dark_curve = np.asarray([1.0, 2.0, 1.0, 0.0])
    light_curve = 3.0 * dark_curve

    result = figure.compute_path_tuning_similarity(dark_curve, light_curve)

    assert result["similarity_status"] == "valid"
    assert result["tuning_similarity_index"] == pytest.approx(1.0)
    assert result["dark_area"] == pytest.approx(4.0)
    assert result["light_area"] == pytest.approx(12.0)
    assert result["light_dark_gain_ratio"] == pytest.approx(3.0)
    assert result["dark_n_finite_bins"] == 4
    assert result["light_n_finite_bins"] == 4
    assert result["n_paired_finite_bins"] == 4


def test_pooled_path_rate_weights_odd_even_counts_by_their_durations() -> None:
    result = figure.compute_pooled_path_movement_firing_rate(
        n_odd_spikes=4,
        n_even_spikes=8,
        odd_duration_s=2.0,
        even_duration_s=8.0,
    )

    assert result["path_movement_firing_rate_status"] == "valid"
    assert result["path_movement_firing_rate_hz"] == pytest.approx(1.2)
    assert result["path_movement_firing_rate_hz"] != pytest.approx(1.5)


def test_pooled_path_rate_keeps_silent_cells_with_positive_support() -> None:
    result = figure.compute_pooled_path_movement_firing_rate(
        n_odd_spikes=0,
        n_even_spikes=0,
        odd_duration_s=3.0,
        even_duration_s=7.0,
    )

    assert result["path_movement_firing_rate_status"] == "valid"
    assert result["path_movement_firing_rate_hz"] == pytest.approx(0.0)


def test_pooled_path_rate_allows_one_empty_split_with_pooled_support() -> None:
    result = figure.compute_pooled_path_movement_firing_rate(
        n_odd_spikes=0,
        n_even_spikes=3,
        odd_duration_s=0.0,
        even_duration_s=6.0,
    )

    assert result["path_movement_firing_rate_status"] == "valid"
    assert result["path_movement_firing_rate_hz"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("odd_duration_s", "even_duration_s"),
    [
        (0.0, 0.0),
        (-1.0, 2.0),
        (np.nan, 1.0),
        (1.0, np.inf),
    ],
)
def test_pooled_path_rate_rejects_invalid_or_nonpositive_total_support(
    odd_duration_s: float,
    even_duration_s: float,
) -> None:
    result = figure.compute_pooled_path_movement_firing_rate(
        n_odd_spikes=1,
        n_even_spikes=1,
        odd_duration_s=odd_duration_s,
        even_duration_s=even_duration_s,
    )

    assert result["path_movement_firing_rate_status"] == "invalid_duration"
    assert np.isnan(result["path_movement_firing_rate_hz"])


@pytest.mark.parametrize(
    ("n_odd_spikes", "n_even_spikes"),
    [(-1, 1), (1, -1), (np.nan, 1), (1, np.inf), (1.5, 1)],
)
def test_pooled_path_rate_rejects_invalid_spike_counts(
    n_odd_spikes: float,
    n_even_spikes: float,
) -> None:
    result = figure.compute_pooled_path_movement_firing_rate(
        n_odd_spikes=n_odd_spikes,
        n_even_spikes=n_even_spikes,
        odd_duration_s=2.0,
        even_duration_s=3.0,
    )

    assert result["path_movement_firing_rate_status"] == "invalid_spike_count"
    assert np.isnan(result["path_movement_firing_rate_hz"])


def test_path_tuning_similarity_matches_normalized_overlap_identity() -> None:
    """TSI should equal one minus half the normalized L1 distance."""
    dark_curve = np.asarray([1.0, 3.0])
    light_curve = np.asarray([3.0, 1.0])

    result = figure.compute_path_tuning_similarity(dark_curve, light_curve)
    normalized_dark = dark_curve / dark_curve.sum()
    normalized_light = light_curve / light_curve.sum()
    expected = 1.0 - 0.5 * np.abs(
        normalized_dark - normalized_light
    ).sum()

    assert result["similarity_status"] == "valid"
    assert result["tuning_similarity_index"] == pytest.approx(expected)
    assert result["tuning_similarity_index"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("dark_curve", "light_curve", "expected_status"),
    [
        ([0.0, 0.0], [1.0, 2.0], "nonpositive_dark_area"),
        ([1.0, 2.0], [0.0, 0.0], "nonpositive_light_area"),
        ([np.nan, np.nan], [1.0, 2.0], "no_finite_dark_bins"),
        ([1.0, 2.0], [np.nan, np.nan], "no_finite_light_bins"),
        ([1.0, -0.1], [1.0, 2.0], "negative_firing_rate"),
    ],
)
def test_path_tuning_similarity_keeps_invalid_curves_out_of_histograms(
    dark_curve: list[float],
    light_curve: list[float],
    expected_status: str,
) -> None:
    result = figure.compute_path_tuning_similarity(dark_curve, light_curve)

    assert result["similarity_status"] == expected_status
    assert np.isnan(result["tuning_similarity_index"])


def test_path_tuning_similarity_interpolates_partial_missing_bins_with_qc() -> None:
    """A missing endpoint may be repaired while raw finite support is retained."""
    dark_curve = np.asarray([1.0, 2.0, 2.0])
    light_curve = np.asarray([2.0, 4.0, np.nan])

    result = figure.compute_path_tuning_similarity(dark_curve, light_curve)

    assert result["similarity_status"] == "valid"
    assert result["tuning_similarity_index"] == pytest.approx(1.0)
    assert result["dark_n_finite_bins"] == 3
    assert result["light_n_finite_bins"] == 2
    assert result["n_paired_finite_bins"] == 2


def test_path_tuning_similarity_reports_shape_mismatch_without_broadcasting() -> None:
    result = figure.compute_path_tuning_similarity(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 2.0, 3.0]),
    )

    assert result["similarity_status"] == "shape_mismatch"
    assert np.isnan(result["tuning_similarity_index"])


def test_tuning_curve_alignment_requires_identical_units_and_position_grid(
    tmp_path,
) -> None:
    xarray = pytest.importorskip("xarray")
    dark = xarray.DataArray(
        np.ones((2, 3)),
        dims=("unit", "linpos"),
        coords={"unit": [1, 2], "linpos": [0.0, 0.5, 1.0]},
    )
    aligned_light = dark.copy()

    figure._require_aligned_tuning_curves(
        dark,
        aligned_light,
        dark_path=tmp_path / "dark.nc",
        light_path=tmp_path / "light.nc",
    )

    with pytest.raises(ValueError, match="'unit' coordinates"):
        figure._require_aligned_tuning_curves(
            dark,
            aligned_light.assign_coords(unit=[1, 3]),
            dark_path=tmp_path / "dark.nc",
            light_path=tmp_path / "different_units.nc",
        )

    with pytest.raises(ValueError, match="'linpos' coordinates"):
        figure._require_aligned_tuning_curves(
            dark,
            aligned_light.assign_coords(linpos=[0.0, 0.4, 1.0]),
            dark_path=tmp_path / "dark.nc",
            light_path=tmp_path / "different_grid.nc",
        )


def test_session_table_applies_strict_path_and_epoch_qc_in_both_conditions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    xarray = pytest.importorskip("xarray")
    from v1ca1.helper.wtrack import get_wtrack_total_length

    positions = get_wtrack_total_length("L14") * np.asarray([0.2, 0.5, 0.8])
    paths: dict[tuple[str, str], Path] = {}
    for epoch in ("02_r1", "08_r4"):
        for path_name in figure.PANEL_B_PATH_ORDER:
            path = tmp_path / f"{epoch}_{path_name}.nc"
            xarray.DataArray(
                np.asarray([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]),
                dims=("unit", "linpos"),
                coords={"unit": [1, 2], "linpos": positions},
            ).to_netcdf(path)
            paths[(epoch, path_name)] = path

    stability_rows = []
    for unit in (1, 2):
        for epoch in ("02_r1", "08_r4"):
            for path_name in figure.PANEL_B_PATH_ORDER:
                correlation = 0.49
                n_odd_spikes = 2
                n_even_spikes = 2
                if unit == 1 and path_name == "center_to_left":
                    n_even_spikes = 4
                    correlation = 0.51
                if unit == 2 and path_name == "center_to_left":
                    if epoch == "08_r4":
                        n_even_spikes = 3
                        correlation = 0.5
                if unit == 2 and path_name == "right_to_center":
                    if epoch == "02_r1":
                        n_even_spikes = 3
                        correlation = 0.5
                stability_rows.append(
                    {
                        "unit": unit,
                        "epoch": epoch,
                        "trajectory_type": path_name,
                        "n_odd_spikes": n_odd_spikes,
                        "n_even_spikes": n_even_spikes,
                        "odd_duration_s": 4.0,
                        "even_duration_s": 6.0,
                        "stability_correlation": correlation,
                        "stability_shape_overlap": 0.75,
                        "shape_overlap_status": "valid",
                        "segment_stability_shape_overlaps": json.dumps(
                            [0.75, 0.75, 0.75]
                        ),
                        "segment_shape_overlap_statuses": json.dumps(
                            ["valid", "valid", "valid"]
                        ),
                        "odd_tuning_curve_area": 2.0,
                        "even_tuning_curve_area": 3.0,
                        # This full-epoch movement rate is deliberately not
                        # reconstructed from the path spike counts above.
                        "firing_rate_hz": 0.7 if unit == 1 else 0.5,
                    }
                )
    stability = pandas.DataFrame(stability_rows).set_index(
        ["unit", "epoch", "trajectory_type"]
    )

    monkeypatch.setattr(
        figure._figure_2,
        "get_compute_tuning_curve_path",
        lambda _data_root, *, epoch, trajectory, **_kwargs: paths[
            (epoch, trajectory)
        ],
    )
    monkeypatch.setattr(
        figure,
        "_load_session_stability_rows",
        lambda *args, **kwargs: (stability, tmp_path / "stability.parquet"),
    )

    table = figure._build_session_tuning_similarity_table(
        data_root=tmp_path,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_epoch="02_r1",
        min_epoch_movement_firing_rate_hz=0.5,
        min_path_movement_firing_rate_hz=0.5,
        min_stability_correlation=0.5,
    )

    assert len(table) == 2 * len(figure.PANEL_B_PATH_ORDER)
    eligible = table.loc[table["passes_qc"], ["unit", "path"]]
    assert eligible.to_records(index=False).tolist() == [
        (1, "center_to_left")
    ]
    crossed_path_unit = table[table["unit"] == 2]
    assert not crossed_path_unit["passes_qc"].any()
    assert not crossed_path_unit["passes_unit_qc"].any()
    exact_threshold_row = crossed_path_unit[
        crossed_path_unit["path"] == "center_to_left"
    ].iloc[0]
    assert exact_threshold_row["dark_path_movement_firing_rate_hz"] == (
        pytest.approx(0.5)
    )
    assert exact_threshold_row["dark_stability_correlation"] == (
        pytest.approx(0.5)
    )
    assert not exact_threshold_row["passes_dark_path_rate_qc"]
    assert not exact_threshold_row["passes_dark_stability_qc"]
    assert table.loc[table["unit"] == 1, "passes_unit_qc"].all()
    assert set(
        table.loc[
            table["unit"] == 1,
            "dark_epoch_movement_firing_rate_hz",
        ]
    ) == {0.7}
    assert set(
        table.loc[
            table["unit"] == 2,
            "dark_epoch_movement_firing_rate_hz",
        ]
    ) == {0.5}
    assert not table.loc[
        table["unit"] == 2,
        "passes_light_epoch_rate_qc",
    ].any()
    assert set(crossed_path_unit["qc_status"]) != {"valid"}
    assert {
        "dark_path_movement_firing_rate_hz",
        "light_path_movement_firing_rate_hz",
        "dark_path_movement_firing_rate_status",
        "light_path_movement_firing_rate_status",
        "dark_n_odd_spikes",
        "dark_n_even_spikes",
        "dark_odd_duration_s",
        "dark_even_duration_s",
        "light_n_odd_spikes",
        "light_n_even_spikes",
        "light_odd_duration_s",
        "light_even_duration_s",
        "passes_dark_path_rate_qc",
        "passes_light_path_rate_qc",
        "passes_dark_epoch_rate_qc",
        "passes_light_epoch_rate_qc",
        "passes_unit_qc",
        "dark_stability_shape_overlap",
        "light_stability_shape_overlap",
        "dark_shape_overlap_status",
        "light_shape_overlap_status",
        "dark_odd_tuning_curve_area",
        "dark_even_tuning_curve_area",
        "light_odd_tuning_curve_area",
        "light_even_tuning_curve_area",
    }.issubset(table.columns)
    assert set(table["dark_stability_shape_overlap"]) == {0.75}
    assert set(table["light_stability_shape_overlap"]) == {0.75}
    assert set(table["dark_shape_overlap_status"]) == {"valid"}
    assert set(table["light_shape_overlap_status"]) == {"valid"}
    assert "dark_movement_firing_rate_hz" not in table
    assert "light_movement_firing_rate_hz" not in table


def test_cli_and_cache_use_path_specific_rate_terminology(tmp_path: Path) -> None:
    args = figure.parse_arguments(
        ["--min-path-movement-firing-rate-hz", "0.75"]
    )

    assert args.min_path_movement_firing_rate_hz == pytest.approx(0.75)
    assert not hasattr(args, "min_movement_firing_rate_hz")
    metadata = figure.build_panel_b_tuning_similarity_cache_metadata(
        data_root=tmp_path,
        datasets=[],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        min_epoch_movement_firing_rate_hz=0.5,
        min_path_movement_firing_rate_hz=0.75,
        min_segment_mean_firing_rate_hz=0.5,
        min_stability_correlation=0.5,
    )
    assert metadata["min_epoch_movement_firing_rate_hz"] == pytest.approx(0.5)
    assert metadata["min_path_movement_firing_rate_hz"] == pytest.approx(0.75)
    assert metadata["min_segment_mean_firing_rate_hz"] == pytest.approx(0.5)
    assert metadata["threshold_comparison"] == "strict_greater_than"
    assert metadata["epoch_movement_firing_rate_source"].endswith(
        ".firing_rate_hz"
    )
    assert "min_movement_firing_rate_hz" not in metadata


def test_tuning_similarity_cache_requires_metadata_and_complete_schema(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    cache_path = tmp_path / "tsi.parquet"
    metadata = {"cache_version": 2, "metric": "unit_area_minimum_overlap"}
    table = pandas.DataFrame(columns=figure.PANEL_B_TUNING_SIMILARITY_COLUMNS)

    figure.save_panel_b_tuning_similarity_cache(cache_path, table, metadata)

    loaded = figure.load_panel_b_tuning_similarity_cache(cache_path, metadata)
    assert loaded is not None
    assert loaded.columns.tolist() == list(
        figure.PANEL_B_TUNING_SIMILARITY_COLUMNS
    )
    assert figure.load_panel_b_tuning_similarity_cache(
        cache_path,
        {**metadata, "cache_version": 3},
    ) is None

    table.drop(columns=["passes_qc"]).to_parquet(cache_path, index=False)
    assert figure.load_panel_b_tuning_similarity_cache(
        cache_path,
        metadata,
    ) is None

