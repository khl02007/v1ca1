"""Focused tests for the Figure 2 circular-shift analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_2 as figure


def _expected_exact_profile(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the specified exact signed-lag unit-area overlap profile."""
    dark = np.asarray(dark_curve, dtype=float)
    light = np.asarray(light_curve, dtype=float)
    dark = dark / dark.sum()
    light = light / light.sum()
    n_bins = dark.size
    shifts = np.arange(n_bins, dtype=int)
    lags = shifts.astype(float) / float(n_bins)
    lags = np.where(lags > 0.5, lags - 1.0, lags)
    overlaps = np.asarray(
        [
            np.minimum(dark, np.roll(light, int(shift))).sum()
            for shift in shifts
        ],
        dtype=float,
    )
    order = np.argsort(lags, kind="stable")
    return lags[order], overlaps[order]


def _periodic_linear_interpolation(
    lags: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Interpolate one exact circular profile without smoothing it."""
    order = np.argsort(lags)
    sorted_lags = np.asarray(lags, dtype=float)[order]
    sorted_values = np.asarray(values, dtype=float)[order]
    extended_lags = np.concatenate(
        (
            [sorted_lags[-1] - 1.0],
            sorted_lags,
            [sorted_lags[0] + 1.0],
        )
    )
    extended_values = np.concatenate(
        ([sorted_values[-1]], sorted_values, [sorted_values[0]])
    )
    return np.interp(grid, extended_lags, extended_values)


@pytest.mark.parametrize(
    ("dark", "light"),
    [
        (
            np.asarray([4.0, 0.0, 0.0, 0.0]),
            np.asarray([4.0, 0.0, 0.0, 0.0]),
        ),
        (
            np.asarray([1.0, 4.0, 2.0, 0.0, 3.0]),
            np.asarray([3.0, 0.0, 1.0, 5.0, 2.0]),
        ),
    ],
)
def test_shift_profile_enumerates_exact_signed_normalized_lags(
    dark: np.ndarray,
    light: np.ndarray,
) -> None:
    """Every integer-bin roll appears once with the requested signed lag."""
    result = figure.compute_circular_shift_overlap_profile(dark, light)
    expected_lags, expected_overlaps = _expected_exact_profile(dark, light)

    assert result["profile_status"] == "valid"
    assert result["n_progression_bins"] == dark.size
    np.testing.assert_allclose(
        result["signed_normalized_shifts"],
        expected_lags,
    )
    np.testing.assert_allclose(result["overlap_scores"], expected_overlaps)
    identity_rows = np.isclose(result["signed_normalized_shifts"], 0.0)
    assert np.count_nonzero(
        identity_rows
    ) == 1
    assert result["overlap_scores"][identity_rows][0] == pytest.approx(
        expected_overlaps[np.isclose(expected_lags, 0.0)][0]
    )
    if dark.size % 2 == 0:
        assert 0.5 in result["signed_normalized_shifts"]
        assert -0.5 not in result["signed_normalized_shifts"]


def test_shift_profile_handles_silence_and_rejects_malformed_curves() -> None:
    """One-sided silence is zero; only two-sided silence is excludable."""
    active = np.asarray([0.0, 1.0, 2.0, 0.0])
    silent = np.zeros_like(active)

    dark_only = figure.compute_circular_shift_overlap_profile(active, silent)
    light_only = figure.compute_circular_shift_overlap_profile(silent, active)
    both_silent = figure.compute_circular_shift_overlap_profile(silent, silent)

    for result in (dark_only, light_only):
        assert result["profile_status"] == "one_condition_silent"
        np.testing.assert_allclose(result["overlap_scores"], 0.0)
        assert result["n_progression_bins"] == active.size
    assert both_silent["profile_status"] == "both_conditions_silent"
    assert np.isnan(both_silent["overlap_scores"]).all()
    assert both_silent["n_progression_bins"] == active.size

    with pytest.raises(ValueError, match="aligned 1-D"):
        figure.compute_circular_shift_overlap_profile(active, active[:-1])
    with pytest.raises(ValueError, match="finite"):
        figure.compute_circular_shift_overlap_profile(
            np.asarray([0.0, np.inf, 1.0]),
            np.asarray([0.0, 1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="negative"):
        figure.compute_circular_shift_overlap_profile(
            np.asarray([0.0, -1.0, 2.0]),
            np.asarray([0.0, 1.0, 2.0]),
        )


def _shift_profile_path_table() -> object:
    """Return all four paths with valid and boundary-failing cells."""
    pandas = pytest.importorskip("pandas")
    records = []
    for path in figure.PANEL_B_PATH_ORDER:
        for unit in range(1, 9):
            dark_rate = 0.75
            light_rate = 0.80
            dark_stability = 0.70
            light_stability = 0.65
            passes_dark_path_rate = True
            passes_light_path_rate = True
            passes_dark_stability = True
            passes_light_stability = True
            if unit == 2:
                dark_rate = 0.5
                passes_dark_path_rate = False
            elif unit == 3:
                light_rate = 0.5
                passes_light_path_rate = False
            elif unit == 4:
                dark_stability = 0.5
                passes_dark_stability = False
            elif unit == 5:
                light_stability = 0.5
                passes_light_stability = False
            records.append(
                {
                    "animal_name": "L14",
                    "date": "20240611",
                    "region": "v1",
                    "unit": unit,
                    "path": path,
                    "trajectory_type": path,
                    "dark_epoch": "08_r4",
                    "light_epoch": "10_r2",
                    # Unit 6 shows that this path-specific cohort has no
                    # separate whole-epoch firing-rate gate.
                    "dark_epoch_movement_firing_rate_hz": 0.1,
                    "light_epoch_movement_firing_rate_hz": 0.2,
                    "dark_path_movement_firing_rate_hz": dark_rate,
                    "light_path_movement_firing_rate_hz": light_rate,
                    "dark_stability_correlation": dark_stability,
                    "light_stability_correlation": light_stability,
                    "dark_stability_shape_overlap": 0.80,
                    "light_stability_shape_overlap": 0.60,
                    "dark_shape_overlap_status": "valid",
                    "light_shape_overlap_status": "valid",
                    "passes_dark_path_rate_qc": passes_dark_path_rate,
                    "passes_light_path_rate_qc": passes_light_path_rate,
                    "passes_dark_stability_qc": passes_dark_stability,
                    "passes_light_stability_qc": passes_light_stability,
                    "passes_dark_epoch_rate_qc": unit != 6,
                    "passes_light_epoch_rate_qc": unit != 6,
                    "passes_unit_qc": unit != 6,
                    # Unit 2 deliberately passes this legacy aggregate flag
                    # but fails the explicit dark path-rate requirement.
                    # Unit 8 does the converse to ensure the four explicit
                    # path flags, rather than passes_qc, define the cohort.
                    "passes_qc": unit not in (4, 5, 8),
                    "dark_tuning_curve_path": "/unused/dark.nc",
                    "light_tuning_curve_path": "/unused/light.nc",
                }
            )
    return pandas.DataFrame.from_records(records)


def test_derive_shift_profile_applies_path_filters_and_aligns_variable_bin_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict path filters precede silence and common-grid alignment."""
    path_table = _shift_profile_path_table()
    curves: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
    for path_index, path in enumerate(figure.PANEL_B_PATH_ORDER):
        n_bins = 41 if path_index % 2 == 0 else 44
        valid_dark = np.arange(1, n_bins + 1, dtype=float)
        valid_light = np.roll(valid_dark[::-1], path_index)
        prefix = (
            "L14",
            "20240611",
            "v1",
        )
        suffix = ("08_r4", "10_r2", path)
        for unit in range(1, 9):
            curves[(*prefix, unit, *suffix)] = (valid_dark, valid_light)
        # Unit 7 otherwise passes every path-specific filter but is silent in
        # both conditions. Unit 8 has valid curves and all four explicit path
        # flags despite its deliberately inconsistent passes_qc=False value.
        curves[(*prefix, 7, *suffix)] = (
            np.zeros(n_bins, dtype=float),
            np.zeros(n_bins, dtype=float),
        )
    monkeypatch.setattr(
        figure,
        "_load_panel_h_shift_profile_curve_lookup",
        lambda rows: curves,
    )

    table = figure.derive_panel_h_shift_profile_table(path_table)

    assert tuple(table.columns) == figure.PANEL_H_SHIFT_PROFILE_COLUMNS
    assert set(table["path"].astype(str)) == set(figure.PANEL_B_PATH_ORDER)
    assert set(table["unit"].astype(int)) == {1, 6, 8}
    assert not table["unit"].isin((2, 3, 4, 5, 7)).any()
    assert len(table) == (
        len(figure.PANEL_B_PATH_ORDER)
        * 3
        * figure.PANEL_H_SHIFT_PROFILE_GRID.size
    )
    assert table.groupby(["path", "unit"], observed=True).size().eq(
        figure.PANEL_H_SHIFT_PROFILE_GRID.size
    ).all()
    assert table["cache_version"].eq(
        figure.PANEL_H_SHIFT_PROFILE_CACHE_VERSION
    ).all()

    for path_index, path in enumerate(figure.PANEL_B_PATH_ORDER):
        rows = table[(table["path"].astype(str) == path) & table["unit"].eq(1)]
        rows = rows.sort_values("normalized_shift")
        curve_key = (
            "L14",
            "20240611",
            "v1",
            1,
            "08_r4",
            "10_r2",
            path,
        )
        lags, exact_overlaps = _expected_exact_profile(*curves[curve_key])
        expected = _periodic_linear_interpolation(
            lags,
            exact_overlaps,
            figure.PANEL_H_SHIFT_PROFILE_GRID,
        )
        np.testing.assert_allclose(
            rows["normalized_shift"],
            figure.PANEL_H_SHIFT_PROFILE_GRID,
        )
        np.testing.assert_allclose(rows["overlap"], expected)
        expected_minimum = float(np.min(exact_overlaps))
        expected_split_half = 0.80
        expected_denominator = expected_split_half - expected_minimum
        assert rows["minimum_overlap"].eq(expected_minimum).all()
        assert rows["dark_split_half_overlap"].eq(0.80).all()
        assert rows["light_split_half_overlap"].eq(0.60).all()
        assert rows["split_half_overlap"].eq(expected_split_half).all()
        assert rows["rescaling_denominator"].eq(expected_denominator).all()
        np.testing.assert_allclose(
            rows["rescaled_overlap"],
            (expected - expected_minimum) / expected_denominator,
        )
        assert rows["rescaling_status"].eq("valid").all()
        expected_n_bins = 41 if path_index % 2 == 0 else 44
        assert rows["n_progression_bins"].eq(expected_n_bins).all()
        assert rows["profile_status"].eq("valid").all()
        assert rows.iloc[0]["overlap"] == pytest.approx(
            rows.iloc[-1]["overlap"]
        )

        assert table[
            (table["path"].astype(str) == path) & table["unit"].eq(6)
        ]["profile_status"].eq("valid").all()
        assert table[
            (table["path"].astype(str) == path) & table["unit"].eq(8)
        ]["profile_status"].eq("valid").all()


def test_shift_profile_rescales_each_neuron_path_to_native_minimum_and_dark_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact floor is zero, the split reference is one, without clipping."""
    pandas = pytest.importorskip("pandas")
    path = figure.PANEL_B_PATH_ORDER[0]
    source = pandas.DataFrame.from_records(
        [
            {
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "unit": 1,
                "path": path,
                "dark_epoch": "08_r4",
                "light_epoch": "10_r2",
                "dark_tuning_curve_path": "/unused/dark.nc",
                "light_tuning_curve_path": "/unused/light.nc",
                "passes_dark_path_rate_qc": True,
                "passes_light_path_rate_qc": True,
                "passes_dark_stability_qc": True,
                "passes_light_stability_qc": True,
                "dark_stability_shape_overlap": 0.60,
                "light_stability_shape_overlap": 0.40,
                "dark_shape_overlap_status": "valid",
                "light_shape_overlap_status": "valid",
            }
        ]
    )
    key = ("L14", "20240611", "v1", 1, "08_r4", "10_r2", path)
    curves = {
        key: (
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
    }
    monkeypatch.setattr(
        figure,
        "_load_panel_h_shift_profile_curve_lookup",
        lambda rows: curves,
    )

    table = figure.derive_panel_h_shift_profile_table(source)

    assert table["minimum_overlap"].eq(0.0).all()
    assert table["split_half_overlap"].eq(0.6).all()
    assert table["rescaling_denominator"].eq(0.6).all()
    assert table["rescaling_status"].eq("valid").all()
    np.testing.assert_allclose(
        table["rescaled_overlap"],
        table["overlap"] / 0.6,
    )
    minimum_rows = table["overlap"].eq(table["minimum_overlap"])
    assert table.loc[minimum_rows, "rescaled_overlap"].eq(0.0).all()
    assert (
        (table["split_half_overlap"] - table["minimum_overlap"])
        / table["rescaling_denominator"]
    ).eq(1.0).all()
    identity = table[np.isclose(table["normalized_shift"], 0.0)]
    assert identity.iloc[0]["rescaled_overlap"] == pytest.approx(1.0 / 0.6)
    assert float(table["rescaled_overlap"].max()) > 1.0


def test_shift_profile_retains_invalid_rescaling_rows_for_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid split references and denominators remain cached but unscaled."""
    pandas = pytest.importorskip("pandas")
    path = figure.PANEL_B_PATH_ORDER[0]
    records = []
    for unit in (1, 2, 3, 4):
        records.append(
            {
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "unit": unit,
                "path": path,
                "dark_epoch": "08_r4",
                "light_epoch": "10_r2",
                "dark_tuning_curve_path": "/unused/dark.nc",
                "light_tuning_curve_path": "/unused/light.nc",
                "passes_dark_path_rate_qc": True,
                "passes_light_path_rate_qc": True,
                "passes_dark_stability_qc": True,
                "passes_light_stability_qc": True,
                "dark_stability_shape_overlap": 0.5 if unit != 3 else 1.0,
                "light_stability_shape_overlap": 0.5 if unit != 3 else 1.0,
                "dark_shape_overlap_status": (
                    "nonpositive_curve" if unit == 2 else "valid"
                ),
                "light_shape_overlap_status": (
                    "nonpositive_curve" if unit == 1 else "valid"
                ),
            }
        )
    source = pandas.DataFrame.from_records(records)
    peaked = (
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    flat = (np.ones(4, dtype=float), np.ones(4, dtype=float))
    silent = (np.zeros(4, dtype=float), np.zeros(4, dtype=float))
    curves = {
        ("L14", "20240611", "v1", unit, "08_r4", "10_r2", path): (
            flat if unit == 3 else silent if unit == 4 else peaked
        )
        for unit in (1, 2, 3, 4)
    }
    monkeypatch.setattr(
        figure,
        "_load_panel_h_shift_profile_curve_lookup",
        lambda rows: curves,
    )

    table = figure.derive_panel_h_shift_profile_table(source)

    assert set(table["unit"].astype(int)) == {1, 2, 3}
    status_by_unit = table.groupby("unit")["rescaling_status"].first().to_dict()
    assert status_by_unit == {
        1: "valid",
        2: "invalid_split_half",
        3: "nonpositive_denominator",
    }
    invalid = table[table["rescaling_status"].ne("valid")]
    assert invalid["rescaled_overlap"].isna().all()
    assert table.groupby("unit").size().eq(
        figure.PANEL_H_SHIFT_PROFILE_GRID.size
    ).all()


def test_panel_c_schematic_shows_three_normalized_circular_shifts() -> None:
    """The schematic shows equal-area curves at zero, quarter, and half shifts."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    figure.plot_circular_shift_schematic(axis)
    fig.canvas.draw()

    schematic_text = {text.get_text() for text in axis.texts}
    visible_text = {text.lower() for text in schematic_text}
    assert axis.get_title() == ""
    assert "Norm. dark tuning" in schematic_text
    assert "Norm. light tuning" in schematic_text
    assert "Overlap" in schematic_text
    assert "Dark tuning curve" not in schematic_text
    assert "Light tuning curve" not in schematic_text
    assert "dark fixed" not in visible_text
    assert "light shifted" not in visible_text
    assert "circular wrap" not in visible_text
    assert r"$\delta=0$" in visible_text
    assert r"$\delta=+1/4$" in visible_text
    assert r"$\delta=+1/2$" in visible_text
    assert not any("Each curve normalized" in text for text in schematic_text)
    assert not any("Shading = overlap" in text for text in schematic_text)
    circular_shift_artist = next(
        text for text in axis.texts if text.get_text() == "Circular shift"
    )
    assert circular_shift_artist.get_position() == pytest.approx((0.50, 0.035))
    assert circular_shift_artist.get_ha() == "center"
    assert circular_shift_artist.get_va() == "center"
    assert not any("0 = minimum" in text for text in visible_text)
    assert not any("1 = split-half" in text for text in visible_text)
    assert len(axis.patches) == 1
    overlap_swatch = axis.patches[0]
    assert overlap_swatch.get_alpha() == pytest.approx(0.55)
    assert overlap_swatch.get_facecolor()[:3] == pytest.approx(
        matplotlib.colors.to_rgb("#BDBDBD")
    )

    dark_curves = [
        line
        for line in axis.lines
        if line.get_color().lower() == "#252525"
        and np.asarray(line.get_xdata()).size == 80
    ]
    light_curves = [
        line
        for line in axis.lines
        if line.get_color().lower() == "#e6ab02"
        and np.asarray(line.get_xdata()).size == 80
    ]
    assert len(dark_curves) == 3
    assert len(light_curves) == 3
    row_baselines = (0.617, 0.384, 0.180)
    for baseline, dark_curve, light_curve in zip(
        row_baselines,
        dark_curves,
        light_curves,
        strict=True,
    ):
        dark_response = np.asarray(dark_curve.get_ydata()) - baseline
        light_response = np.asarray(light_curve.get_ydata()) - baseline
        assert np.sum(dark_response) == pytest.approx(np.sum(light_response))
        for response in (dark_response, light_response):
            local_maxima = np.flatnonzero(
                (response[1:-1] > response[:-2])
                & (response[1:-1] > response[2:])
            )
            assert local_maxima.size == 1
    dark_peaks = [int(np.argmax(line.get_ydata())) for line in dark_curves]
    light_peaks = [int(np.argmax(line.get_ydata())) for line in light_curves]
    dark_top_response = np.asarray(dark_curves[0].get_ydata()) - row_baselines[0]
    light_top_response = np.asarray(light_curves[0].get_ydata()) - row_baselines[0]
    assert not np.allclose(dark_top_response, light_top_response)
    assert np.count_nonzero(
        light_top_response >= 0.5 * float(np.max(light_top_response))
    ) > np.count_nonzero(
        dark_top_response >= 0.5 * float(np.max(dark_top_response))
    )
    assert dark_peaks[0] == dark_peaks[1]
    assert dark_peaks[0] == dark_peaks[2]
    assert light_peaks[0] == dark_peaks[0]
    assert (light_peaks[1] - light_peaks[0]) % 80 == 20
    assert (light_peaks[2] - light_peaks[0]) % 80 == 40
    delta_labels = {
        text.get_text(): text
        for text in axis.texts
        if text.get_text()
        in {r"$\Delta=0$", r"$\Delta=+1/4$", r"$\Delta=+1/2$"}
    }
    assert set(delta_labels) == {
        r"$\Delta=0$",
        r"$\Delta=+1/4$",
        r"$\Delta=+1/2$",
    }
    assert delta_labels[r"$\Delta=0$"].get_position() == pytest.approx(
        (0.97, 0.680)
    )
    assert delta_labels[r"$\Delta=0$"].get_ha() == "right"
    assert delta_labels[r"$\Delta=0$"].get_va() == "center"
    assert delta_labels[r"$\Delta=+1/4$"].get_position() == pytest.approx(
        (0.97, 0.490)
    )
    assert delta_labels[r"$\Delta=+1/4$"].get_ha() == "right"
    assert delta_labels[r"$\Delta=+1/4$"].get_va() == "center"
    assert delta_labels[r"$\Delta=+1/2$"].get_position() == pytest.approx(
        (0.97, 0.280)
    )
    assert delta_labels[r"$\Delta=+1/2$"].get_ha() == "right"
    assert delta_labels[r"$\Delta=+1/2$"].get_va() == "center"
    assert all(text.get_bbox_patch() is None for text in delta_labels.values())
    renderer = fig.canvas.get_renderer()
    for label, response_lines in (
        (delta_labels[r"$\Delta=0$"], (dark_curves[0], light_curves[0])),
        (
            delta_labels[r"$\Delta=+1/4$"],
            (dark_curves[1], light_curves[1]),
        ),
        (
            delta_labels[r"$\Delta=+1/2$"],
            (dark_curves[2], light_curves[2]),
        ),
    ):
        label_bounds = label.get_window_extent(renderer)
        label_x_limits = axis.transData.inverted().transform(
            ((label_bounds.x0, label_bounds.y0), (label_bounds.x1, label_bounds.y1))
        )[:, 0]
        local_response_points = []
        for line in response_lines:
            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            within_label_width = (
                (x_data >= label_x_limits[0])
                & (x_data <= label_x_limits[1])
            )
            assert np.any(within_label_width)
            local_response_points.extend(
                axis.transData.transform(
                    np.column_stack(
                        (x_data[within_label_width], y_data[within_label_width])
                    )
                )[:, 1]
            )
        assert label_bounds.y0 > max(local_response_points)
    wrap_arrows = [
        text
        for text in axis.texts
        if getattr(text, "arrow_patch", None) is not None
    ]
    assert len(wrap_arrows) == 2
    arrow_segments = []
    for annotation in wrap_arrows:
        head = np.asarray(annotation.xy, dtype=float)
        tail = np.asarray(annotation.xyann, dtype=float)
        assert head[1] == pytest.approx(tail[1])
        direction = float(np.sign(head[0] - tail[0]))
        assert direction != 0.0
        connection_style = annotation.arrow_patch.get_connectionstyle()
        assert float(getattr(connection_style, "rad", 0.0)) == pytest.approx(0.0)
        arrow_segments.append((tail, head, direction))
    arrow_segments.sort(key=lambda segment: min(segment[0][0], segment[1][0]))
    np.testing.assert_allclose(arrow_segments[0][0], (0.00, 0.035))
    np.testing.assert_allclose(arrow_segments[0][1], (0.23, 0.035))
    np.testing.assert_allclose(arrow_segments[1][0], (0.77, 0.035))
    np.testing.assert_allclose(arrow_segments[1][1], (1.00, 0.035))
    assert arrow_segments[0][2] == arrow_segments[1][2]
    assert arrow_segments[0][0][1] == pytest.approx(arrow_segments[1][0][1])
    assert arrow_segments[0][0][1] == pytest.approx(0.035)
    assert arrow_segments[0][0][1] < row_baselines[-1]
    assert max(arrow_segments[0][0][0], arrow_segments[0][1][0]) < 0.5
    assert min(arrow_segments[1][0][0], arrow_segments[1][1][0]) > 0.5
    assert max(arrow_segments[0][0][0], arrow_segments[0][1][0]) < (
        circular_shift_artist.get_position()[0]
    )
    assert min(arrow_segments[1][0][0], arrow_segments[1][1][0]) > (
        circular_shift_artist.get_position()[0]
    )
    plt.close(fig)


def test_panel_c_uses_one_population_axis_beside_the_schematic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Panel C renders one population profile rather than four path panels."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls = []

    def fake_plot_population(axis: object, table: object) -> None:
        calls.append((axis, table))
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Circular shift")
        axis.set_ylabel("Norm. overlap")

    def fail_pathwise_profiles(*args: object, **kwargs: object) -> None:
        raise AssertionError("Panel B rendered the obsolete four-path axes")

    monkeypatch.setattr(
        figure,
        "plot_panel_h_shift_profiles",
        fail_pathwise_profiles,
    )
    monkeypatch.setattr(
        figure,
        "plot_population_shift_profile",
        fake_plot_population,
    )
    table = object()
    fig, parent_axis = plt.subplots()
    figure.plot_panel_b_circular_shift_analysis(parent_axis, table)
    fig.canvas.draw()

    population_axes = [
        axis
        for axis in fig.axes
        if axis.get_label() == "panel_b_population_shift_profile"
    ]
    schematic_axes = [
        axis
        for axis in fig.axes
        if axis.get_label() == "panel_b_circular_shift_schematic"
    ]
    assert len(calls) == 1
    assert calls[0][1] is table
    assert len(population_axes) == 1
    assert len(schematic_axes) == 1
    assert not any(
        axis.get_label().startswith("panel_b_shift_profile_")
        for axis in fig.axes
    )
    population_axis = population_axes[0]
    schematic_axis = schematic_axes[0]
    assert schematic_axis.get_title() == ""
    assert population_axis.get_title() == ""
    assert population_axis.get_xlabel() == "Circular shift"
    assert population_axis.get_ylabel() == "Norm. overlap"
    parent_bounds = parent_axis.get_position()
    schematic_bounds = schematic_axis.get_position()
    population_bounds = population_axis.get_position()
    schematic_relative_bounds = (
        (schematic_bounds.x0 - parent_bounds.x0) / parent_bounds.width,
        (schematic_bounds.y0 - parent_bounds.y0) / parent_bounds.height,
        schematic_bounds.width / parent_bounds.width,
        schematic_bounds.height / parent_bounds.height,
    )
    population_relative_bounds = (
        (population_bounds.x0 - parent_bounds.x0) / parent_bounds.width,
        (population_bounds.y0 - parent_bounds.y0) / parent_bounds.height,
        population_bounds.width / parent_bounds.width,
        population_bounds.height / parent_bounds.height,
    )
    assert schematic_relative_bounds == pytest.approx(
        (0.1438, 0.0375, 0.2624, 0.825),
        abs=1e-12,
    )
    assert population_relative_bounds == pytest.approx(
        (0.5762, 0.220, 0.280, 0.620),
        abs=1e-12,
    )
    assert schematic_relative_bounds[2] == pytest.approx(
        figure.PANEL_B_SCHEMATIC_RELATIVE_WIDTH
    )
    assert schematic_bounds.x1 < population_bounds.x0
    gutter_fraction = (
        population_bounds.x0 - schematic_bounds.x1
    ) / parent_bounds.width
    assert gutter_fraction == pytest.approx(0.170, abs=1e-12)
    assert schematic_bounds.y0 < population_bounds.y0
    assert schematic_bounds.height > population_bounds.height
    group_center = 0.5 * (schematic_bounds.x0 + population_bounds.x1)
    assert group_center == pytest.approx(
        0.5 * (parent_bounds.x0 + parent_bounds.x1),
        abs=1e-12,
    )
    for child_bounds in (schematic_bounds, population_bounds):
        assert child_bounds.x0 >= parent_bounds.x0
        assert child_bounds.x1 <= parent_bounds.x1
        assert child_bounds.y0 >= parent_bounds.y0
        assert child_bounds.y1 <= parent_bounds.y1
    parent_text = [text.get_text() for text in parent_axis.texts]
    assert "Across neurons: mean; IQR" not in parent_text
    assert "Mean across neurons; shading = IQR" not in parent_text
    assert "Path norm.;\ncell summary" not in parent_text
    connector_arrows = [
        text
        for text in parent_axis.texts
        if getattr(text, "arrow_patch", None) is not None
    ]
    assert connector_arrows == []
    assert len(parent_axis.patches) == 0
    schematic_text = [text.get_text() for text in schematic_axis.texts]
    assert "Overlap" in schematic_text
    assert r"$\Delta=+1/2$" in schematic_text
    assert not any("Each curve normalized" in text for text in schematic_text)
    panel_b_text = [
        text.get_text()
        for axis in (parent_axis, schematic_axis, population_axis)
        for text in axis.texts
    ]
    assert not any("K=" in text for text in panel_b_text)
    plt.close(fig)


def test_panel_b_profile_and_panel_c_scatter_share_x_axis_and_label_level() -> None:
    """The two quantitative panels align their bottom spines and x labels."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 3.0))
    panel_b_axis = fig.add_axes((0.05, 0.10, 0.40, 0.80))
    panel_c_axis = fig.add_axes((0.55, 0.10, 0.40, 0.80))
    schematic_axis = panel_b_axis.inset_axes(
        (0.1088, 0.0375, 0.2624, 0.825)
    )
    schematic_axis.set_label("panel_b_circular_shift_schematic")
    profile_axis = panel_b_axis.inset_axes((0.60, 0.30, 0.35, 0.55))
    profile_axis.set_label("panel_b_population_shift_profile")
    profile_axis.set_xlabel("Circular shift")
    scatter_parent = panel_c_axis.inset_axes((0.45, 0.15, 0.50, 0.75))
    scatter_axis = scatter_parent.inset_axes((0.15, 0.10, 0.60, 0.60))
    scatter_axis.set_xlabel("Dark PII")
    scatter_axis.xaxis.set_label_coords(0.50, -0.18)
    top_axis = scatter_parent.inset_axes((0.15, 0.75, 0.60, 0.15))
    top_axis.set_ylabel("Frac.")
    right_axis = scatter_parent.inset_axes((0.78, 0.10, 0.18, 0.60))
    right_axis.set_xlabel("Frac.")
    fig.canvas.draw()

    assert profile_axis.get_position().y0 != pytest.approx(
        scatter_axis.get_position().y0
    )
    figure._align_panel_b_profile_with_panel_c_scatter(
        fig,
        panel_b_axis,
        panel_c_axis,
    )
    fig.canvas.draw()

    assert profile_axis.get_position().y0 == pytest.approx(
        scatter_axis.get_position().y0,
        abs=1e-12,
    )
    assert schematic_axis.get_position().y0 == pytest.approx(
        panel_b_axis.get_position().y0
        + figure.PANEL_B_SCHEMATIC_RELATIVE_Y
        * panel_b_axis.get_position().height,
        abs=1e-12,
    )
    assert schematic_axis.get_position().height == pytest.approx(
        figure.PANEL_B_SCHEMATIC_RELATIVE_HEIGHT
        * panel_b_axis.get_position().height,
        abs=1e-12,
    )
    quantitative_left = min(
        axis.get_position().x0
        for axis in (scatter_axis, top_axis, right_axis)
    )
    quantitative_right = max(
        axis.get_position().x1
        for axis in (scatter_axis, top_axis, right_axis)
    )
    panel_b_bounds = panel_b_axis.get_position()
    panel_c_bounds = panel_c_axis.get_position()
    expected_width = (
        (quantitative_right - quantitative_left)
        / panel_c_bounds.width
        * panel_b_bounds.width
        * figure.PANEL_B_PROFILE_WIDTH_SCALE_FROM_PANEL_C
    )
    assert profile_axis.get_position().width == pytest.approx(
        expected_width,
        abs=1e-12,
    )
    gap = profile_axis.get_position().x0 - schematic_axis.get_position().x1
    assert gap == pytest.approx(
        figure.PANEL_B_COMPONENT_RELATIVE_GAP * panel_b_bounds.width,
        abs=1e-12,
    )
    group_center = 0.5 * (
        schematic_axis.get_position().x0 + profile_axis.get_position().x1
    )
    assert group_center == pytest.approx(
        0.5 * (panel_b_bounds.x0 + panel_b_bounds.x1),
        abs=1e-12,
    )
    profile_label_y = profile_axis.xaxis.label.get_transform().transform(
        profile_axis.xaxis.label.get_position()
    )[1]
    scatter_label_y = scatter_axis.xaxis.label.get_transform().transform(
        scatter_axis.xaxis.label.get_position()
    )[1]
    assert profile_label_y == pytest.approx(scatter_label_y, abs=1e-9)
    plt.close(fig)


def test_population_profile_averages_paths_within_neuron_before_neuron_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cells with more available paths do not receive greater population weight."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pandas = pytest.importorskip("pandas")
    from matplotlib.axes import Axes

    shifts = np.asarray([-0.5, 0.0, 0.5])
    neuron_path_curves = {
        1: [np.asarray([0.0, 1.0, 0.0])],
        2: [np.zeros(3, dtype=float) for _path in range(4)],
        3: [
            np.asarray([0.1, 0.3, 0.5]),
            np.asarray([0.3, 0.5, 0.7]),
        ],
        4: [np.full(3, value) for value in (0.8, 0.9, 1.0)],
    }
    records = []
    for unit, path_curves in neuron_path_curves.items():
        for path, curve in zip(
            figure.PANEL_B_PATH_ORDER,
            path_curves,
            strict=False,
        ):
            for shift, value in zip(shifts, curve, strict=True):
                records.append(
                    {
                        "animal_name": "L14",
                        "date": "20240611",
                        "region": "v1",
                        "unit": unit,
                        "path": path,
                        "dark_epoch": "08_r4",
                        "light_epoch": "10_r2",
                        "normalized_shift": shift,
                        "overlap": 0.25,
                        "minimum_overlap": 0.10,
                        "dark_split_half_overlap": 0.70,
                        "light_split_half_overlap": 0.50,
                        "split_half_overlap": 0.60,
                        "rescaling_denominator": 0.50,
                        "rescaled_overlap": value,
                        "n_progression_bins": 41,
                        "profile_status": "valid",
                        "rescaling_status": "valid",
                        "cache_version": (
                            figure.PANEL_H_SHIFT_PROFILE_CACHE_VERSION
                        ),
                    }
                )
    for path in figure.PANEL_B_PATH_ORDER:
        for shift in shifts:
            records.append(
                {
                    "animal_name": "L14",
                    "date": "20240611",
                    "region": "v1",
                    "unit": 99,
                    "path": path,
                    "dark_epoch": "08_r4",
                    "light_epoch": "10_r2",
                    "normalized_shift": shift,
                    "overlap": 0.99,
                    "minimum_overlap": 0.99,
                    "dark_split_half_overlap": 0.50,
                    "light_split_half_overlap": 0.50,
                    "split_half_overlap": 0.50,
                    "rescaling_denominator": -0.49,
                    "rescaled_overlap": 99.0,
                    "n_progression_bins": 41,
                    "profile_status": "valid",
                    "rescaling_status": "nonpositive_denominator",
                    "cache_version": figure.PANEL_H_SHIFT_PROFILE_CACHE_VERSION,
                }
            )
    table = pandas.DataFrame.from_records(
        records,
        columns=figure.PANEL_H_SHIFT_PROFILE_COLUMNS,
    )
    fill_calls = []
    original_fill_between = Axes.fill_between

    def recording_fill_between(
        axis: object,
        x: object,
        y1: object,
        y2: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        fill_calls.append(
            {
                "axis": axis,
                "x": np.asarray(x, dtype=float),
                "lower": np.asarray(y1, dtype=float),
                "upper": np.asarray(y2, dtype=float),
                "label": str(kwargs.get("label", "")),
            }
        )
        return original_fill_between(axis, x, y1, y2, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", recording_fill_between)
    fig, axis = plt.subplots()
    figure.plot_population_shift_profile(axis, table)
    fig.canvas.draw()

    neuron_curves = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.4, 0.6],
            [0.9, 0.9, 0.9],
        ]
    )
    expected_median = np.median(neuron_curves, axis=0)
    pooled_path_mean = np.asarray([0.31, 0.45, 0.39])
    assert not np.allclose(expected_median, pooled_path_mean)
    median_lines = [
        line for line in axis.lines if "median" in line.get_label().lower()
    ]
    assert len(median_lines) == 1
    assert median_lines[0].get_label() == "Median across neurons"
    np.testing.assert_allclose(median_lines[0].get_xdata(), shifts)
    np.testing.assert_allclose(median_lines[0].get_ydata(), expected_median)
    assert len(fill_calls) == 1
    np.testing.assert_allclose(fill_calls[0]["x"], shifts)
    np.testing.assert_allclose(
        fill_calls[0]["lower"],
        np.quantile(neuron_curves, 0.25, axis=0),
    )
    np.testing.assert_allclose(
        fill_calls[0]["upper"],
        np.quantile(neuron_curves, 0.75, axis=0),
    )
    assert fill_calls[0]["label"] == "IQR across neurons"
    annotation_artists = [
        text for text in axis.texts if text.get_text().startswith("Δ=0:")
    ]
    assert len(annotation_artists) == 1
    annotation_artist = annotation_artists[0]
    annotation = annotation_artist.get_text()
    assert "N=" not in annotation
    assert "K=" not in annotation
    zero_values = neuron_curves[:, np.flatnonzero(shifts == 0.0)[0]]
    pooled_zero_values = np.asarray(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.8, 0.9, 1.0]
    )
    expected_q25, expected_median, expected_q75 = np.quantile(
        zero_values,
        [0.25, 0.50, 0.75],
    )
    assert expected_median != pytest.approx(np.median(pooled_zero_values))
    expected_annotation = (
        f"Δ=0: med. {expected_median:.2f}\n"
        f"IQR {expected_q25:.2f}–{expected_q75:.2f}"
    )
    assert annotation == expected_annotation
    assert len(annotation.splitlines()) == 2
    assert annotation_artist.get_position() == pytest.approx((0.05, 0.94))
    assert annotation_artist.get_ha() == "left"
    assert annotation_artist.get_va() == "top"
    assert "99" not in annotation
    assert "shift" in axis.get_xlabel().lower()
    assert axis.get_ylabel() == "Norm. overlap"
    plt.close(fig)


def test_shift_profile_cache_records_cohort_grid_and_summary_semantics(
    tmp_path: Path,
) -> None:
    """The cached profile is versioned, addressed, and schema checked."""
    pandas = pytest.importorskip("pandas")
    source_metadata = {
        "cache_version": figure.PANEL_B_TUNING_SIMILARITY_CACHE_VERSION,
        "region": "v1",
        "source_fingerprint": "example",
    }
    metadata = figure.build_panel_h_shift_profile_cache_metadata(
        source_metadata
    )

    assert figure.PANEL_H_SHIFT_PROFILE_CACHE_VERSION == 4
    assert metadata["cache_version"] == 4
    assert metadata["source_path_cache_version"] == (
        figure.PANEL_B_TUNING_SIMILARITY_CACHE_VERSION
    )
    assert metadata["artifact"] == (
        "whole_path_dark_split_half_rescaled_circular_shift_overlap_profile"
    )
    assert metadata["metric"] == (
        "whole_path_dark_split_half_rescaled_unit_area_overlap"
    )
    assert metadata["shift_operation"] == (
        "hold_dark_fixed_and_circularly_roll_light"
    )
    assert metadata["exact_shift_support"] == (
        "every_integer_bin_shift_from_zero_through_n_bins_minus_one"
    )
    assert "signed_circular_lag" in metadata["exact_shift_coordinate"]
    assert "minus_half_to_half" in metadata["exact_shift_coordinate"]
    assert "periodic_linear_interpolation" in metadata["grid_alignment"]
    assert "without_smoothing" in metadata["grid_alignment"]
    assert metadata["minimum_reference"] == (
        "minimum_overlap_across_all_exact_native_integer_bin_shifts"
    )
    assert metadata["minimum_reference_timing"] == (
        "compute_from_exact_native_shift_support_before_common_grid_"
        "interpolation"
    )
    assert metadata["split_half_reference"] == (
        "dark_whole_path_odd_even_unit_area_overlap"
    )
    assert metadata["rescaling"] == (
        "(overlap_minus_minimum_overlap)/(split_half_overlap_minus_minimum_"
        "overlap)"
    )
    assert metadata["denominator_policy"] == (
        "valid_only_when_strictly_greater_than_1e-12"
    )
    assert metadata["clipping"] == "none"
    assert metadata["invalid_rescaling_policy"] == (
        "retain_rows_with_status_and_nan_rescaled_overlap_but_exclude_from_"
        "display"
    )
    np.testing.assert_allclose(
        metadata["common_grid"],
        figure.PANEL_H_SHIFT_PROFILE_GRID,
    )
    assert "strictly_above_threshold" in metadata[
        "unit_path_inclusion"
    ]
    assert "both_dark_and_light" in metadata["unit_path_inclusion"]
    assert "path_movement_firing_rate" in metadata["unit_path_inclusion"]
    assert "odd_even" in metadata["unit_path_inclusion"]
    assert metadata["min_path_movement_firing_rate_hz"] == pytest.approx(0.5)
    assert metadata["min_stability_correlation"] == pytest.approx(0.5)
    assert metadata["whole_epoch_rate_filter"] == "none"
    assert metadata["panel"] == "B"
    assert metadata["population_weighting"] == (
        "equal_mean_across_valid_paths_within_neuron_then_equal_weight_per_"
        "neuron"
    )
    assert metadata["display_summary"] == (
        "median_and_interquartile_range_across_neuron_level_profiles_at_each_"
        "shift"
    )
    assert "silent_in_both_conditions" in metadata[
        "post_filter_exclusion"
    ]
    assert metadata["columns"] == list(figure.PANEL_H_SHIFT_PROFILE_COLUMNS)

    cache_path = figure.build_panel_h_shift_profile_cache_path(
        tmp_path,
        metadata,
    )
    table = pandas.DataFrame(columns=figure.PANEL_H_SHIFT_PROFILE_COLUMNS)
    figure.save_panel_h_shift_profile_cache(cache_path, table, metadata)
    loaded = figure.load_panel_h_shift_profile_cache(cache_path, metadata)
    assert loaded is not None
    assert tuple(loaded.columns) == figure.PANEL_H_SHIFT_PROFILE_COLUMNS

    changed = {**metadata, "common_grid": [-0.5, 0.0, 0.5]}
    assert figure.build_panel_h_shift_profile_cache_path(
        tmp_path,
        changed,
    ) != cache_path
    assert figure.load_panel_h_shift_profile_cache(cache_path, changed) is None

    table.drop(columns=["rescaled_overlap"]).to_parquet(cache_path, index=False)
    assert figure.load_panel_h_shift_profile_cache(cache_path, metadata) is None
