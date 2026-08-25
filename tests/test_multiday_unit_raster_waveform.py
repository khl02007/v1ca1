from __future__ import annotations

import json
import pickle

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")

from v1ca1.multiday.visualization import unit_raster_waveform as multiday


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_unit_spike_times_splits_concatenated_sorting(tmp_path):
    sorting_path = tmp_path / "sorting_v1"
    sorting_path.mkdir()
    _write_json(
        sorting_path / "numpysorting_info.json",
        {"unit_ids": [0, 21], "sampling_frequency": 30_000.0},
    )
    spikes = np.array(
        [
            (0, 0, 0),
            (1, 1, 0),
            (4, 1, 0),
            (6, 1, 0),
            (8, 0, 0),
        ],
        dtype=[
            ("sample_index", "<i8"),
            ("unit_index", "<i8"),
            ("segment_index", "<i8"),
        ],
    )
    np.save(sorting_path / "spikes.npy", spikes)
    with (tmp_path / "timestamps_ephys_all.pkl").open("wb") as file:
        pickle.dump(
            {
                "day1": np.array([10.0, 10.1, 10.2, 10.3, 10.4]),
                "day2": np.array([20.0, 20.1, 20.2, 20.3]),
            },
            file,
        )

    result = multiday.load_unit_spike_times_by_date(
        analysis_path=tmp_path,
        region="v1",
        unit_id=21,
        dates=("day2", "day1"),
    )

    assert tuple(result) == ("day2", "day1")
    np.testing.assert_allclose(result["day1"], [10.1, 10.4])
    np.testing.assert_allclose(result["day2"], [20.1])


def test_probe_shank_unit_uses_concatenation_order(tmp_path):
    for shank_index, unit_ids in enumerate(([10, 11], [20, 21, 22])):
        _write_json(
            tmp_path
            / f"curated_sorting_probe0_shank{shank_index}"
            / "numpysorting_info.json",
            {"unit_ids": unit_ids},
        )

    assert multiday._probe_shank_unit(
        tmp_path,
        probe_index=0,
        global_unit_index=3,
    ) == (1, 1, 21)


def test_orient_task_progression_flips_inbound_path():
    position, values = multiday._orient_task_progression(
        np.array([0.0, 50.0, 100.0]),
        np.array([1.0, 2.0, 3.0]),
        "right_to_center",
        total_length_cm=100.0,
    )

    np.testing.assert_allclose(position, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(values, [3.0, 2.0, 1.0])


def test_date_colors_follow_elapsed_days_on_sequential_colormap():
    from matplotlib import colormaps

    colors = multiday._build_date_colors(multiday.DEFAULT_DATES)
    lower, upper = multiday.DATE_COLORMAP_RANGE
    expected_positions = lower + (upper - lower) * np.array(
        [0.0, 1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0, 1.0]
    )
    expected_colors = colormaps.get_cmap(multiday.DATE_COLORMAP)(
        expected_positions
    )

    assert tuple(colors) == multiday.DEFAULT_DATES
    np.testing.assert_allclose(list(colors.values()), expected_colors)
    assert multiday._build_day_labels(multiday.DEFAULT_DATES) == {
        "20240605": "Day 0",
        "20240606": "Day 1",
        "20240607": "Day 2",
        "20240609": "Day 4",
        "20240611": "Day 7",
    }


def test_compute_isi_distributions_uses_fraction_of_all_intervals():
    result = multiday.compute_isi_distributions(
        {"day1": np.array([0.0, 0.001, 0.003, 0.020])},
        bin_size_s=0.001,
        max_isi_s=0.005,
    )

    assert result["n_intervals"] == {"day1": 3}
    np.testing.assert_allclose(
        result["fractions"]["day1"],
        [0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        result["bin_centers_ms"],
        [0.5, 1.5, 2.5, 3.5, 4.5],
    )


def test_smooth_tuning_rate_reduces_single_bin_peak():
    raw = np.array([0.0, 0.0, 10.0, 0.0, 0.0])

    smoothed = multiday._smooth_tuning_rate(raw, sigma_bins=1.0)

    assert smoothed.shape == raw.shape
    assert 0.0 < smoothed[2] < raw[2]
    assert smoothed[1] > 0.0
    np.testing.assert_allclose(
        multiday._smooth_tuning_rate(raw, sigma_bins=0.0),
        raw,
    )


def test_load_figure_data_loads_isi_and_both_tuning_epochs(monkeypatch):
    tuning_epochs = []

    monkeypatch.setattr(
        multiday,
        "load_waveform_data",
        lambda **kwargs: {"loaded_epoch": kwargs["epoch"]},
    )

    def fake_load_tuning_curves(**kwargs):
        tuning_epochs.append(kwargs["epoch"])
        return {date: {} for date in kwargs["dates"]}

    monkeypatch.setattr(multiday, "load_tuning_curves", fake_load_tuning_curves)
    monkeypatch.setattr(
        multiday,
        "load_unit_spike_times_by_date",
        lambda **kwargs: {
            date: np.array([0.0, 0.010, 0.030])
            for date in kwargs["dates"]
        },
    )

    result = multiday.load_figure_data(
        analysis_path=multiday.DEFAULT_ANALYSIS_PATH,
        animal_name="L14",
        region="v1",
        unit_id=21,
        dates=multiday.DEFAULT_DATES,
        waveform_epoch="02_r1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
    )

    assert tuning_epochs == ["08_r4", "02_r1"]
    assert result["waveform_data"] == {"loaded_epoch": "02_r1"}
    assert tuple(result["isi_distributions"]["fractions"]) == (
        multiday.DEFAULT_DATES
    )
    assert result["tuning_smoothing_sigma_bins"] == 1.0


def test_load_tuning_curves_reads_saved_pynapple_artifact(tmp_path):
    output_dir = tmp_path / multiday.DEFAULT_TUNING_CURVE_DIRNAME
    output_dir.mkdir()
    path = multiday.get_tuning_curve_path(
        output_dir,
        region="v1",
        date="20240605",
        epoch="02_r1",
        trajectory="center_to_left",
    )
    curve = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("unit", "tp"),
        coords={"unit": [20, 21], "tp": [0.25, 0.75]},
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240605",
            "region": "v1",
            "epoch": "02_r1",
            "trajectory_type": "center_to_left",
            "n_trials": 3,
        },
    )
    curve.to_netcdf(path)

    result = multiday.load_tuning_curves(
        analysis_path=tmp_path,
        animal_name="L14",
        region="v1",
        unit_id=21,
        dates=("20240605",),
        epoch="02_r1",
        trajectories=("center_to_left",),
    )

    position, rate = result["20240605"]["center_to_left"]
    np.testing.assert_allclose(position, [0.25, 0.75])
    np.testing.assert_allclose(rate, [3.0, 4.0])


def _synthetic_figure_data():
    dates = multiday.DEFAULT_DATES
    trajectories = multiday.DEFAULT_TRAJECTORIES
    position = np.linspace(0.0, 1.0, 20)
    return {
        "animal_name": "L14",
        "region": "v1",
        "unit_id": 21,
        "dates": dates,
        "waveform_epoch": "02_r1",
        "epoch_ids": {"dark": "08_r4", "light": "02_r1"},
        "trajectories": trajectories,
        "waveform_data": {
            "time_ms": np.linspace(-1.0, 0.9, 12),
            "channel_ids": (17, 18, 16),
            "waveforms": {
                date: np.column_stack(
                    [
                        -20.0 * np.exp(-np.linspace(-2, 2, 12) ** 2),
                        -10.0 * np.exp(-np.linspace(-2, 2, 12) ** 2),
                        -5.0 * np.exp(-np.linspace(-2, 2, 12) ** 2),
                    ]
                )
                for date in dates
            },
        },
        "isi_distributions": {
            "bin_centers_ms": np.arange(0.5, 50.0, 1.0),
            "fractions": {
                date: np.exp(
                    -np.arange(0.5, 50.0, 1.0)
                    / (4.0 + date_index)
                )
                / 20.0
                for date_index, date in enumerate(dates)
            },
            "n_intervals": {date: 100 for date in dates},
            "bin_size_s": 0.001,
            "max_isi_s": 0.050,
        },
        "tuning_curves": {
            condition: {
                date: {
                    trajectory: (
                        None
                        if (
                            condition == "dark"
                            and date == dates[0]
                            and trajectory == "right_to_center"
                        )
                        else (
                            position,
                            (
                                condition_index
                                + date_index
                                + trajectory_index
                                + 1
                            )
                            * np.exp(-((position - 0.55) / 0.12) ** 2),
                        )
                    )
                    for trajectory_index, trajectory in enumerate(trajectories)
                }
                for date_index, date in enumerate(dates)
            }
            for condition_index, condition in enumerate(
                multiday.TUNING_EPOCH_ORDER
            )
        },
        "tuning_smoothing_sigma_bins": 1.0,
    }


def test_make_figure_uses_two_eight_left_and_tuning_layout():
    import matplotlib.pyplot as plt

    fig = multiday.make_figure(_synthetic_figure_data())

    assert len(fig.axes) == 16
    assert [axis.get_title() for axis in fig.axes[:3]] == [
        "Ch. 17",
        "Ch. 18",
        "Ch. 16",
    ]
    assert fig.axes[3].get_title() == ""
    assert [axis.get_title() for axis in fig.axes[4:9]] == [
        "Day 0",
        "Day 1",
        "Day 2",
        "Day 4",
        "Day 7",
    ]
    figure_text = {text.get_text() for text in fig.texts}
    assert {
        "A",
        "B",
        "C",
        "Mean waveforms",
        "ISI distribution",
        "Normalized path progression",
    }.issubset(figure_text)
    figure_text_by_label = {text.get_text(): text for text in fig.texts}
    assert (
        figure_text_by_label["Mean waveforms"].get_fontsize()
        == figure_text_by_label["ISI distribution"].get_fontsize()
    )
    assert (
        figure_text_by_label["A"].get_position()[1]
        == figure_text_by_label["Mean waveforms"].get_position()[1]
    )
    assert (
        figure_text_by_label["B"].get_position()[1]
        == figure_text_by_label["ISI distribution"].get_position()[1]
    )
    assert all(not axis.get_xlabel() for axis in fig.axes[:3])
    assert "Time (ms)" in figure_text
    assert fig.axes[3].get_xlabel() == "ISI (ms)"
    assert fig.axes[3].get_ylabel() == "Fraction"
    assert all(not axis.get_ylabel() for axis in fig.axes[:3])
    amplitude_labels = [
        text for text in fig.texts if text.get_text() == "Amplitude (µV)"
    ]
    assert len(amplitude_labels) == 1
    expected_waveform_label_y = 0.5 * (
        fig.axes[0].get_position().y1 + fig.axes[2].get_position().y0
    )
    assert np.isclose(
        amplitude_labels[0].get_position()[1],
        expected_waveform_label_y,
    )
    assert fig.axes[4].get_ylabel() == "FR (Hz)"
    assert fig.axes[9].get_ylabel() == "FR (Hz)"
    assert len({axis.get_ylim()[1] for axis in fig.axes[4:14]}) == 1
    assert not any(
        "No trials" in text.get_text()
        for axis in fig.axes[4:14]
        for text in axis.texts
    )
    assert not fig.axes[14].axison
    assert not fig.axes[15].axison
    assert fig.axes[0].get_position().x1 < fig.axes[1].get_position().x0
    assert fig.axes[1].get_position().x1 < fig.axes[2].get_position().x0
    assert np.isclose(
        fig.axes[0].get_position().y0,
        fig.axes[1].get_position().y0,
    )
    assert np.isclose(
        fig.axes[1].get_position().y0,
        fig.axes[2].get_position().y0,
    )
    assert fig.axes[2].get_position().y0 > fig.axes[3].get_position().y0
    assert fig.axes[4].get_position().x0 > fig.axes[3].get_position().x1
    left_width = fig.axes[3].get_position().x1 - fig.axes[0].get_position().x0
    right_width = fig.axes[8].get_position().x1 - fig.axes[14].get_position().x0
    assert 3.5 < right_width / left_width < 4.5
    assert fig.axes[14].get_position().x0 - fig.axes[2].get_position().x1 < 0.11
    assert 1.0 - fig.axes[8].get_position().x1 < 0.03
    assert np.isclose(fig.get_size_inches()[0], 5.7)
    assert np.isclose(fig.get_size_inches()[1], 2.25)
    assert fig.axes[0].lines[0].get_color() == fig.axes[3].lines[0].get_color()
    assert fig.axes[4].lines[0].get_color() == multiday.EPOCH_TYPE_COLORS["dark"]
    assert fig.axes[4].lines[1].get_color() == multiday.EPOCH_TYPE_COLORS["light"]
    assert fig._suptitle is None
    day_legend_labels = [
        text.get_text() for text in fig.legends[0].get_texts()
    ]
    condition_legend_labels = [
        text.get_text() for text in fig.legends[1].get_texts()
    ]
    assert day_legend_labels == ["Day 0", "Day 1", "Day 2", "Day 4", "Day 7"]
    assert condition_legend_labels == ["Dark", "Light"]
    assert (
        fig.legends[0].get_bbox_to_anchor().x0
        > fig.axes[2].get_position().x1
    )
    condition_anchor = fig.legends[1].get_bbox_to_anchor().transformed(
        fig.transFigure.inverted()
    )
    assert (
        fig.axes[4].get_position().x0
        < condition_anchor.x0
        < fig.axes[4].get_position().x1
    )
    assert (
        fig.axes[4].get_position().y0
        < condition_anchor.y0
        < fig.axes[4].get_position().y1
    )
    assert "02_r1" not in " ".join(text.get_text() for text in fig.texts)
    plt.close(fig)


def test_save_multiday_figure_writes_requested_formats(tmp_path):
    paths = multiday.save_multiday_figure(
        _synthetic_figure_data(),
        output_dir=tmp_path,
        output_name="unit21",
        formats=("png", "pdf"),
        dpi=100,
    )

    assert paths == (tmp_path / "unit21.png", tmp_path / "unit21.pdf")
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_cli_defaults_target_requested_data_and_output():
    args = multiday.parse_arguments([])

    assert args.analysis_path == multiday.DEFAULT_ANALYSIS_PATH
    assert args.output_dir == multiday.DEFAULT_OUTPUT_DIR
    assert args.animal_name == "L14"
    assert args.region == "v1"
    assert args.unit_id == 21
    assert args.waveform_epoch == "02_r1"
    assert args.light_epoch == "02_r1"
    assert args.dark_epoch == "08_r4"
    assert args.n_waveform_channels == 3
    assert args.tuning_smoothing_sigma_bins == 1.0
    assert args.dpi == 600
