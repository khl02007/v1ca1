from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v1ca1.paper_figures.figure_2 import (
    DEFAULT_EXAMPLE_DATASET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_SELECTION,
    DEFAULT_RIPPLE_THRESHOLD_ZSCORE,
    DEFAULT_RIPPLE_WINDOW_S,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    DEFAULT_XCORR_BIN_SIZE_S,
    DEFAULT_XCORR_DATASET,
    DEFAULT_XCORR_DISPLAY_VMAX,
    DEFAULT_XCORR_MAX_LAG_S,
    DEFAULT_XCORR_STATE,
    DEFAULT_XCORR_TOP_CA1_UNITS,
    build_output_path,
    build_peri_ripple_heatmap_payload,
    build_ripple_modulation_output_stem,
    compute_significance_distribution_comparison,
    draw_ripple_glm_schematic,
    format_ridge_strength_suffix,
    format_ripple_window_suffix,
    get_encoding_comparison_summary_path,
    HEATMAP_EPOCH_LABELS,
    HEATMAP_EPOCH_ORDER,
    get_ripple_event_path,
    get_ripple_glm_path,
    get_ripple_lfp_path,
    get_ripple_modulation_paths,
    get_screen_xcorr_paths,
    get_tuning_similarity_path,
    load_glm_behavior_association_tables,
    load_example_glm_prediction,
    load_example_ripple_lfp_trace,
    load_first_available_glm_prediction,
    load_glm_epoch_summary_tables,
    load_modulation_summary_table,
    load_pooled_ripple_heatmap_epoch_tables,
    load_ripple_heatmap_epoch_tables,
    load_ripple_count_table,
    load_ripple_glm_summary_table,
    load_top_ca1_xcorr_panel_data,
    parse_arguments,
    parse_dataset_id,
    plot_glm_summary_panel,
    plot_epoch_ripple_heatmap_panel,
    plot_glm_behavior_association_panel,
    plot_glm_analysis_panel,
    plot_modulation_index_panel,
    plot_observed_predicted_panel,
    plot_peri_ripple_heatmap_panel,
    plot_ripple_lfp_panel,
    plot_top_ca1_xcorr_panel,
)


def test_parse_dataset_id_requires_animal_and_date() -> None:
    assert parse_dataset_id("L14:20240611") == ("L14", "20240611", "08_r4")
    assert parse_dataset_id("L15:20241121:10_r5") == ("L15", "20241121", "10_r5")

    with pytest.raises(argparse.ArgumentTypeError, match="animal:date"):
        parse_dataset_id("L14")


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "figure_2", "svg") == Path(
        "paper_figures/figure_2.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "figure_2", "jpg")


def test_ripple_modulation_paths_match_cached_output_stem(tmp_path: Path) -> None:
    stem = build_ripple_modulation_output_stem(
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region_label="all_regions",
        ripple_threshold_zscore=2.0,
        bin_size_s=0.02,
        time_before_s=0.5,
        time_after_s=0.5,
        response_window=(0.0, 0.1),
        baseline_window=(-0.5, -0.3),
        heatmap_normalize="max",
    )

    assert stem == (
        "L14_20240611_08_r4_all_regions_thr_2_bin_0p02_tb_0p5_ta_0p5_"
        "resp_0_0p1_base_neg0p5_neg0p3_norm_max"
    )

    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )

    assert paths["summary"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple"
        / "ripple_modulation"
        / f"{stem}_summary.parquet"
    )
    assert paths["peri_ripple_firing_rate"].name == f"{stem}_peri_ripple_firing_rate.parquet"


def test_ripple_glm_path_matches_samplewise_output_name(tmp_path: Path) -> None:
    assert format_ripple_window_suffix(0.2) == "rw_0p2s"
    assert format_ripple_window_suffix(0.2, ripple_window_offset_s=-0.2) == "rw_0p2s_off_m0p2s"
    assert format_ridge_strength_suffix(1e-1) == "ridge_1e-1"

    path = get_ripple_glm_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_window_s=0.2,
        ripple_selection="allripples",
        ridge_strength=1e-1,
    )

    assert path == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple_glm"
        / "08_r4_rw_0p2s_allripples_ridge_1e-1_samplewise_ripple_glm.nc"
    )


def _write_screen_xcorr_cache(tmp_path: Path) -> tuple[Path, np.ndarray]:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("pyarrow")
    output_dir = tmp_path / "RatA" / "20240101" / "xcorr" / "screen_pairs" / "ripple" / "02_r1"
    output_dir.mkdir(parents=True)

    ca1_units = np.array([10, 11, 12], dtype=int)
    v1_units = np.array([101, 102, 103], dtype=int)
    lag_s = np.array([-0.01, 0.0, 0.01], dtype=float)
    xcorr = np.arange(ca1_units.size * v1_units.size * lag_s.size, dtype=float).reshape(
        ca1_units.size,
        v1_units.size,
        lag_s.size,
    )
    xr.Dataset(
        data_vars={"xcorr": (("ca1_unit", "v1_unit", "lag_s"), xcorr)},
        coords={"ca1_unit": ca1_units, "v1_unit": v1_units, "lag_s": lag_s},
    ).to_netcdf(output_dir / "xcorr.nc")

    rows = []
    peak_values = {
        10: {101: 5.0, 102: 4.0, 103: 3.0},
        11: {101: 6.0, 102: 7.0, 103: 2.0},
        12: {101: 1.0, 102: 1.5, 103: 2.0},
    }
    peak_lags = {
        10: {101: 0.0, 102: 0.01, 103: -0.01},
        11: {101: -0.01, 102: 0.01, 103: 0.0},
        12: {101: 0.0, 102: 0.01, 103: -0.01},
    }
    for ca1_unit in ca1_units:
        for v1_unit in v1_units:
            rows.append(
                {
                    "ca1_unit_id": ca1_unit,
                    "v1_unit_id": v1_unit,
                    "n_ca1_state_spikes": 50,
                    "n_v1_state_spikes": 60,
                    "peak_lag_s": peak_lags[int(ca1_unit)][int(v1_unit)],
                    "peak_norm_xcorr": peak_values[int(ca1_unit)][int(v1_unit)],
                    "status": "valid",
                }
            )
    pd.DataFrame(rows).to_parquet(output_dir / "xcorr_summary.parquet", index=False)
    return output_dir, xcorr


def test_load_top_ca1_xcorr_panel_data_uses_shared_v1_order(tmp_path: Path) -> None:
    output_dir, xcorr = _write_screen_xcorr_cache(tmp_path)

    paths = get_screen_xcorr_paths(
        tmp_path,
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
    )
    payload = load_top_ca1_xcorr_panel_data(
        tmp_path,
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        top_n_ca1_units=2,
    )

    assert paths["dataset"] == output_dir / "xcorr.nc"
    assert payload["ca1_unit_ids"].tolist() == [11, 10]
    assert payload["v1_unit_ids"].tolist() == [102, 101, 103]
    assert payload["v1_order_reference_ca1_unit"] == 11
    assert payload["xcorr"].shape == (2, 3, 3)
    assert np.allclose(payload["xcorr"][0, 0], xcorr[1, 1])


def _write_ripple_events(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    path = get_ripple_event_path(tmp_path, "L14", "20240611")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": ["08_r4", "08_r4", "06_r3"],
            "start": [1.0, 1.3, 2.0],
            "end": [1.05, 1.36, 2.04],
            "mean_zscore": [2.1, 5.0, 6.0],
        }
    ).to_parquet(path, index=False)
    return path


def test_load_ripple_count_table_filters_epoch_and_threshold(tmp_path: Path) -> None:
    _write_ripple_events(tmp_path)

    table = load_ripple_count_table(
        tmp_path,
        [("L14", "20240611", "08_r4")],
        ripple_threshold_zscore=3.0,
    )

    assert table["animal_name"].tolist() == ["L14"]
    assert table["epoch"].tolist() == ["08_r4"]
    assert table["n_ripples"].tolist() == [1]


def test_load_example_ripple_lfp_trace_uses_largest_thresholded_ripple(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")
    _write_ripple_events(tmp_path)
    lfp_path = get_ripple_lfp_path(tmp_path, "L14", "20240611", "08_r4")
    lfp_path.parent.mkdir(parents=True)
    time = np.linspace(1.1, 1.45, 36)
    filtered_lfp = np.column_stack([np.sin(time * 30.0), np.cos(time * 30.0)])
    xr.Dataset(
        data_vars={"filtered_lfp": (("sample", "channel"), filtered_lfp)},
        coords={"time": ("sample", time), "channel": ("channel", [101, 102])},
    ).to_netcdf(lfp_path)

    trace = load_example_ripple_lfp_trace(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_threshold_zscore=2.0,
    )

    assert trace["ripple_start_s"] == pytest.approx(1.3)
    assert trace["mean_zscore"] == pytest.approx(5.0)
    assert trace["channel"] == 101
    assert np.min(trace["time_s"]) < 0.0
    assert np.max(trace["time_s"]) > 0.0


def test_load_modulation_summary_table_reads_cached_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )
    paths["summary"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "epoch": ["08_r4", "08_r4"],
            "region": ["v1", "ca1"],
            "unit_id": [11, 101],
            "ripple_modulation_index": [0.2, -0.1],
            "response_zscore": [2.0, -0.5],
        }
    ).to_parquet(paths["summary"], index=False)

    table = load_modulation_summary_table(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert table["region"].tolist() == ["v1", "ca1"]
    assert table["source_path"].tolist() == [str(paths["summary"])] * 2


def test_build_peri_ripple_heatmap_payload_preserves_time_matrix() -> None:
    table = pd.DataFrame(
        {
            "region": ["v1", "v1", "v1", "v1", "ca1", "ca1"],
            "unit_id": [11, 11, 12, 12, 101, 101],
            "time_s": [-0.02, 0.0, -0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )

    payload = build_peri_ripple_heatmap_payload(table, region="v1")

    assert payload["unit_ids"].tolist() == [11, 12]
    assert np.allclose(payload["time_s"], [-0.02, 0.0])
    assert np.allclose(payload["mean_rate_hz"], [[1.0, 2.0], [4.0, 3.0]])


def test_build_peri_ripple_heatmap_payload_keeps_sessions_separate() -> None:
    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15"],
            "date": ["20240611", "20240611", "20241121", "20241121"],
            "epoch": ["06_r3", "06_r3", "06_r3", "06_r3"],
            "region": ["v1", "v1", "v1", "v1"],
            "unit_id": [11, 11, 11, 11],
            "time_s": [-0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 4.0, 3.0],
        }
    )

    payload = build_peri_ripple_heatmap_payload(table, region="v1")

    assert payload["mean_rate_hz"].shape == (2, 2)
    assert np.allclose(payload["mean_rate_hz"], [[1.0, 2.0], [4.0, 3.0]])


def _write_peri_ripple_table(
    tmp_path: Path,
    epoch: str,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
) -> Path:
    pytest.importorskip("pyarrow")
    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    paths["peri_ripple_firing_rate"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "animal_name": [animal_name] * 4,
            "date": [date] * 4,
            "epoch": [epoch] * 4,
            "region": ["v1", "v1", "ca1", "ca1"],
            "unit_id": [11, 11, 101, 101],
            "n_ripples": [3] * 4,
            "bin_size_s": [0.02] * 4,
            "time_before_s": [0.5] * 4,
            "time_after_s": [0.5] * 4,
            "time_s": [-0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(paths["peri_ripple_firing_rate"], index=False)
    return paths["peri_ripple_firing_rate"]


def _write_modulation_summary_table(
    tmp_path: Path,
    epoch: str,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
) -> Path:
    pytest.importorskip("pyarrow")
    paths = get_ripple_modulation_paths(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "animal_name": [animal_name, animal_name],
            "date": [date, date],
            "epoch": [epoch, epoch],
            "region": ["v1", "ca1"],
            "unit_id": [11, 101],
            "ripple_modulation_index": [0.2, -0.1],
            "response_zscore": [2.0, -0.5],
        }
    ).to_parquet(paths["summary"], index=False)
    return paths["summary"]


def test_load_ripple_heatmap_epoch_tables_uses_registered_order(tmp_path: Path) -> None:
    for epoch in ("02_r1", "08_r4", "07_s4"):
        _write_peri_ripple_table(tmp_path, epoch)

    epoch_tables = load_ripple_heatmap_epoch_tables(
        tmp_path,
        {
            "light": ("L14", "20240611", "02_r1"),
            "dark": ("L14", "20240611", "08_r4"),
            "sleep": ("L14", "20240611", "07_s4"),
        },
    )

    assert HEATMAP_EPOCH_ORDER == ("light", "dark", "sleep")
    assert HEATMAP_EPOCH_LABELS["light"] == "Light run"
    assert [payload["epoch"] for payload in epoch_tables] == ["02_r1", "08_r4", "07_s4"]
    assert [payload["label"] for payload in epoch_tables] == ["Light run", "Dark run", "Sleep"]


def test_load_pooled_ripple_heatmap_epoch_tables_uses_all_animals(tmp_path: Path) -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    for animal_name, date, dark_epoch in datasets:
        for epoch in ("02_r1", dark_epoch, "07_s4"):
            _write_peri_ripple_table(
                tmp_path,
                epoch,
                animal_name=animal_name,
                date=date,
            )
            _write_modulation_summary_table(
                tmp_path,
                epoch,
                animal_name=animal_name,
                date=date,
            )

    epoch_tables = load_pooled_ripple_heatmap_epoch_tables(tmp_path, datasets)

    assert [payload["epoch"] for payload in epoch_tables] == ["02_r1", "registered", "07_s4"]
    assert [payload["n_datasets"] for payload in epoch_tables] == [2, 2, 2]
    assert set(epoch_tables[1]["firing_rate_table"]["epoch"]) == {"08_r4", "10_r5"}
    assert len(epoch_tables[2]["summary_table"]) == 4


def _write_ripple_glm_dataset(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
) -> Path:
    xr = pytest.importorskip("xarray")
    path = get_ripple_glm_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "ripple_devexp_mean": (("unit",), np.array([0.1, 0.4])),
            "ripple_devexp_p_value": (("unit",), np.array([0.2, 0.01])),
            "ripple_bits_per_spike_mean": (("unit",), np.array([0.03, 0.08])),
            "ripple_observed_count_oof": (
                ("sample", "unit"),
                np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]]),
            ),
            "ripple_predicted_count_oof": (
                ("sample", "unit"),
                np.array([[0.8, 0.2], [1.7, 1.3], [0.2, 2.5]]),
            ),
        },
        coords={"sample": np.arange(3), "unit": np.array([11, 12])},
        attrs={"n_ripples_after_selection": 3},
    )
    dataset.to_netcdf(path)
    return path


def test_load_ripple_glm_summary_table_reads_per_unit_metrics(tmp_path: Path) -> None:
    path = _write_ripple_glm_dataset(tmp_path)

    table = load_ripple_glm_summary_table(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert table["unit_id"].tolist() == [11, 12]
    assert np.allclose(table["ripple_devexp_mean"], [0.1, 0.4])
    assert np.allclose(table["ripple_devexp_p_value"], [0.2, 0.01])
    assert table["n_ripples"].tolist() == [3, 3]
    assert table["source_path"].tolist() == [str(path), str(path)]


def test_load_glm_epoch_summary_tables_reads_light_dark_sleep(tmp_path: Path) -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    for animal_name, date, dark_epoch in datasets:
        for epoch in ("02_r1", dark_epoch, "07_s4"):
            _write_ripple_glm_dataset(
                tmp_path,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
            )

    epoch_tables = load_glm_epoch_summary_tables(tmp_path, datasets)

    assert [payload["epoch"] for payload in epoch_tables] == ["02_r1", "registered", "07_s4"]
    assert [payload["n_datasets"] for payload in epoch_tables] == [2, 2, 2]
    assert set(epoch_tables[1]["summary_table"]["epoch"]) == {"08_r4", "10_r5"}
    assert len(epoch_tables[2]["summary_table"]) == 4


def _write_encoding_summary_table(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
) -> Path:
    pytest.importorskip("pyarrow")
    path = get_encoding_comparison_summary_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        region="v1",
        epoch=epoch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "n_spikes": [100, 200],
            "delta_bits_generalized_place_vs_tp": [-0.2, 0.1],
        },
        index=pd.Index([11, 12], name="unit"),
    )
    table.to_parquet(path)
    return path


def _write_tuning_similarity_table(
    tmp_path: Path,
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    epoch: str = "08_r4",
) -> Path:
    pytest.importorskip("pyarrow")
    path = get_tuning_similarity_path(
        tmp_path,
        animal_name=animal_name,
        date=date,
        region="v1",
        epoch=epoch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "unit": [11, 12, 11],
            "region": ["v1", "v1", "v1"],
            "epoch": [epoch, epoch, epoch],
            "comparison_label": ["pooled_same_turn", "pooled_same_turn", "pooled_same_arm"],
            "similarity": [0.3, 0.6, 0.1],
            "firing_rate_hz": [1.0, 2.0, 1.0],
        }
    ).to_parquet(path, index=False)
    return path


def test_load_glm_behavior_association_tables_joins_dark_epoch_metrics(
    tmp_path: Path,
) -> None:
    for epoch in ("02_r1", "08_r4", "07_s4"):
        _write_ripple_glm_dataset(tmp_path, epoch=epoch)
    _write_tuning_similarity_table(tmp_path)

    payload = load_glm_behavior_association_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    similarity_table = payload["similarity_table"]
    assert payload["missing_artifacts"] == []
    assert len(similarity_table) == 6
    assert set(similarity_table["epoch_type"]) == {"light", "dark", "sleep"}
    assert set(similarity_table["tuning_epoch"]) == {"08_r4"}
    assert sorted(similarity_table["unit"].unique().tolist()) == [11, 12]
    assert sorted(similarity_table["same_turn_tuning_similarity"].unique().tolist()) == [0.3, 0.6]


def test_load_glm_behavior_association_tables_reports_missing_tuning(
    tmp_path: Path,
) -> None:
    _write_ripple_glm_dataset(tmp_path)

    payload = load_glm_behavior_association_tables(
        tmp_path,
        [("L14", "20240611", "08_r4")],
    )

    assert payload["similarity_table"].empty
    assert payload["missing_artifacts"][0]["artifact"] == "tuning_analysis"


def test_compute_significance_distribution_comparison_uses_session_strata() -> None:
    table = pd.DataFrame(
        {
            "animal_name": ["L14", "L14", "L15", "L15"],
            "date": ["20240611", "20240611", "20241121", "20241121"],
            "epoch": ["08_r4", "08_r4", "10_r5", "10_r5"],
            "same_turn_tuning_similarity": [0.2, 0.8, 0.4, 0.9],
            "ripple_devexp_p_value": [0.2, 0.01, 0.2, 0.01],
        }
    )

    stats = compute_significance_distribution_comparison(
        table,
        metric_column="same_turn_tuning_similarity",
        n_permutations=100,
        random_seed=1,
    )

    assert stats["n_significant"] == 2
    assert stats["n_nonsignificant"] == 2
    assert stats["median_difference"] == pytest.approx(0.55)
    assert 0.0 < float(stats["p_value"]) <= 1.0


def test_load_example_glm_prediction_selects_top_devexp_unit(tmp_path: Path) -> None:
    _write_ripple_glm_dataset(tmp_path)

    example = load_example_glm_prediction(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )

    assert example["unit_id"] == 12
    assert example["ripple_devexp_mean"] == pytest.approx(0.4)
    assert np.allclose(example["observed"], [0.0, 1.0, 3.0])
    assert np.allclose(example["predicted"], [0.2, 1.3, 2.5])


def test_load_first_available_glm_prediction_skips_old_schema(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")
    old_path = get_ripple_glm_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
    )
    old_path.parent.mkdir(parents=True)
    xr.Dataset(
        data_vars={
            "ripple_devexp_mean": (("unit",), np.array([0.1])),
            "ripple_devexp_p_value": (("unit",), np.array([0.5])),
            "ripple_bits_per_spike_mean": (("unit",), np.array([0.01])),
        },
        coords={"unit": np.array([11])},
    ).to_netcdf(old_path)
    _write_ripple_glm_dataset(
        tmp_path,
        animal_name="L19",
        date="20250930",
        epoch="08_r4",
    )

    example = load_first_available_glm_prediction(
        tmp_path,
        preferred_dataset=("L14", "20240611", "08_r4"),
        candidate_datasets=[("L19", "20250930", "08_r4")],
    )

    assert example["unit_id"] == 12
    assert example["ripple_devexp_mean"] == pytest.approx(0.4)


def test_plot_helpers_draw_expected_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peri_table = pd.DataFrame(
        {
            "region": ["v1", "v1", "ca1", "ca1"],
            "unit_id": [11, 11, 101, 101],
            "time_s": [-0.02, 0.0, -0.02, 0.0],
            "mean_rate_hz": [1.0, 2.0, 3.0, 4.0],
        }
    )
    summary_table = pd.DataFrame(
        {
            "region": ["v1", "v1", "ca1", "ca1"],
            "ripple_modulation_index": [0.2, -0.1, 0.3, 0.4],
        }
    )
    glm_table = pd.DataFrame(
        {
            "ripple_devexp_mean": [0.1, 0.4],
            "ripple_devexp_p_value": [0.2, 0.01],
        }
    )
    glm_epoch_table = pd.DataFrame(
        {
            "ripple_devexp_mean": [-2.0, 0.4],
            "ripple_devexp_p_value": [0.2, 0.01],
        }
    )
    glm_epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "epoch": "02_r1",
            "summary_table": glm_epoch_table,
        },
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "epoch": "registered",
            "summary_table": glm_epoch_table,
        },
        {
            "epoch_type": "sleep",
            "label": "Sleep",
            "epoch": "07_s4",
            "summary_table": glm_epoch_table,
        },
    ]
    prediction = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "unit_id": 12,
        "observed": np.array([0.0, 1.0, 3.0]),
        "predicted": np.array([0.2, 1.3, 2.5]),
        "ripple_devexp_mean": 0.4,
        "ripple_devexp_p_value": 0.01,
    }
    trace = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "channel": 101,
        "time_s": np.array([-0.01, 0.0, 0.01]),
        "filtered_lfp": np.array([0.0, 1.0, 0.0]),
        "ripple_duration_s": 0.05,
        "mean_zscore": 5.0,
        "n_ripples": 2,
    }
    xcorr_payload = {
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "02_r1",
        "v1_order_reference_ca1_unit": 11,
        "display_vmax": 5.0,
        "ca1_unit_ids": np.array([11, 10]),
        "v1_unit_ids": np.array([101, 102, 103]),
        "lag_s": np.array([-0.01, 0.0, 0.01]),
        "xcorr": np.ones((2, 3, 3), dtype=float),
    }
    association_payload = {
        "similarity_table": pd.DataFrame(
            {
                "epoch_type": ["light", "light", "dark", "dark", "sleep", "sleep"],
                "same_turn_tuning_similarity": [0.1, 0.5, 0.7, 0.8, 0.4, 0.9],
                "ripple_devexp_mean": [0.05, 0.2, 0.1, 0.3, 0.15, 0.25],
                "ripple_devexp_p_value": [0.01, 0.2, 0.03, 0.4, 0.02, 0.6],
            }
        ),
        "missing_artifacts": [],
    }

    epoch_tables = [
        {
            "epoch_type": "light",
            "label": "Light run",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "02_r1",
            "firing_rate_table": peri_table,
            "summary_table": summary_table,
        },
        {
            "epoch_type": "dark",
            "label": "Dark run",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "08_r4",
            "firing_rate_table": peri_table,
            "summary_table": summary_table,
        },
        {
            "epoch_type": "sleep",
            "label": "Sleep",
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "07_s4",
            "firing_rate_table": peri_table,
            "summary_table": summary_table,
        },
    ]

    fig, axes = plt.subplots(3, 3)
    plot_ripple_lfp_panel(axes[0, 0], trace)
    plot_peri_ripple_heatmap_panel(axes[0, 1], peri_table, regions=("v1", "ca1"))
    plot_modulation_index_panel(axes[0, 2], summary_table, regions=("v1", "ca1"))
    draw_ripple_glm_schematic(axes[1, 0])
    plot_glm_summary_panel(axes[1, 1], glm_table)
    plot_observed_predicted_panel(axes[1, 2], prediction)
    plot_epoch_ripple_heatmap_panel(axes[2, 0], epoch_tables, regions=("v1", "ca1"))
    plot_top_ca1_xcorr_panel(axes[2, 1], xcorr_payload)
    plot_glm_analysis_panel(axes[2, 2], glm_epoch_tables)

    assert len(axes[0, 1].child_axes) == 3
    assert len(axes[2, 0].child_axes) == 10
    assert len(axes[2, 1].images) == 0
    assert len(axes[2, 1].child_axes) == 3
    assert all(len(child_axis.images) == 1 for child_axis in axes[2, 1].child_axes[:2])
    assert len(axes[1, 0].patches) >= 3
    assert len(axes[1, 1].collections) == 1
    assert len(axes[1, 2].collections) == 1
    assert len(axes[2, 2].child_axes) == 7
    assert axes[2, 2].child_axes[1].get_xlim()[0] == pytest.approx(-0.1)
    assert axes[2, 2].child_axes[1].get_xlim()[1] == pytest.approx(0.5)
    assert len(axes[2, 2].child_axes[1].collections) == 2
    assert axes[2, 2].child_axes[2].get_xlim()[0] == pytest.approx(-0.1)
    assert axes[2, 2].child_axes[2].get_xlim()[1] == pytest.approx(0.5)
    assert len(axes[2, 2].child_axes[2].patches) == 2
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_glm_behavior_association_panel(ax, association_payload)
    assert len(ax.child_axes) == 1
    assert len(ax.child_axes[0].patches) > 0
    assert ax.child_axes[0].get_xlabel() == "Dark same-turn\ntuning similarity"
    assert [label.get_text() for label in ax.child_axes[0].get_yticklabels()] == [
        "Light",
        "Dark",
        "Sleep",
    ]
    plt.close(fig)


def test_parse_arguments_defaults_match_figure_2_cli() -> None:
    args = parse_arguments([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == "figure_2"
    assert args.output_format == "pdf"
    assert args.example_dataset == DEFAULT_EXAMPLE_DATASET
    assert args.xcorr_dataset == DEFAULT_XCORR_DATASET
    assert args.xcorr_state == DEFAULT_XCORR_STATE
    assert args.xcorr_top_ca1_units == DEFAULT_XCORR_TOP_CA1_UNITS
    assert args.xcorr_bin_size_s == DEFAULT_XCORR_BIN_SIZE_S
    assert args.xcorr_max_lag_s == DEFAULT_XCORR_MAX_LAG_S
    assert args.xcorr_display_vmax == DEFAULT_XCORR_DISPLAY_VMAX
    assert args.light_epoch is None
    assert args.dark_epoch is None
    assert args.sleep_epoch is None
    assert args.ripple_threshold_zscore == DEFAULT_RIPPLE_THRESHOLD_ZSCORE
    assert args.ripple_window_s == DEFAULT_RIPPLE_WINDOW_S
    assert args.ripple_window_offset_s == DEFAULT_RIPPLE_WINDOW_OFFSET_S
    assert args.ripple_selection == DEFAULT_RIPPLE_SELECTION
    assert args.ridge_strength == DEFAULT_RIDGE_STRENGTH
