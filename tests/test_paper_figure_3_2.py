from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_3 as figure_3_module
import v1ca1.paper_figures.figure_3_2 as figure_3_2_module
from v1ca1.paper_figures.figure_3_2 import (
    DEFAULT_OUTPUT_NAME,
    PANEL_B_CACHE_VERSION,
    PANEL_B_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_N_SHUFFLES,
    PANEL_B_SHIFT_FRACTION_BOUNDS,
    PANEL_B_SHUFFLE_SEED,
    build_panel_b_cache_metadata,
    build_panel_b_cache_path,
    build_panel_b_percentile_table,
    circular_shift_spikes_within_trial_chunks,
    load_panel_b_cache,
    load_panel_b_tuning_correlation_data,
    make_figure_3_2,
    parse_arguments,
    plot_panel_a_example_grid,
    plot_panel_b_percentile_grid,
    save_panel_b_cache,
)


def _fake_panel_a_example(trajectories: tuple[str, ...]) -> dict[str, object]:
    position = np.asarray([0.0, 0.5, 1.0], dtype=float)
    return {
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "unit_id": 34,
        "trajectories": trajectories,
        "epoch_rates": {
            epoch_key: {
                "epoch": "08_r4" if epoch_key == "dark" else "02_r1",
                "raster_positions": {
                    trajectory: [np.asarray([0.1, 0.4]), np.asarray([0.7])]
                    for trajectory in trajectories
                },
                "firing_rates": {
                    trajectory: (
                        position,
                        np.asarray([0.0, 1.0, 0.5], dtype=float),
                    )
                    for trajectory in trajectories
                },
            }
            for epoch_key in ("dark", "light")
        },
    }


def _fake_correlation_tables():
    pandas = pytest.importorskip("pandas")
    observed_rows = []
    shuffle_rows = []
    for index, trajectory in enumerate(figure_3_module.PANEL_B_TRAJECTORY_TYPES):
        observed_rows.append(
            {
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "dark_epoch": "08_r4",
                "light_epoch": "02_r1",
                "trajectory_type": trajectory,
                "unit": index + 1,
                "correlation": 0.7 + 0.05 * index,
            }
        )
        for shuffle_index in range(2):
            shuffle_rows.append(
                {
                    "animal_name": "L14",
                    "date": "20240611",
                    "region": "v1",
                    "dark_epoch": "08_r4",
                    "light_epoch": "02_r1",
                    "trajectory_type": trajectory,
                    "shuffle": shuffle_index,
                    "unit": index + 1,
                    "shift_fraction_min": 0.3,
                    "shift_fraction_max": 0.6,
                    "correlation": -0.2 + 0.1 * shuffle_index,
                }
            )
    return pandas.DataFrame(observed_rows), pandas.DataFrame(shuffle_rows)


def test_circular_shift_preserves_spike_count_within_trial_movement_chunks() -> None:
    rng = np.random.default_rng(0)
    spike_times = np.asarray([0.1, 0.4, 1.2, 1.8, 2.2], dtype=float)
    trial_chunks = [
        (np.asarray([0.0, 1.0]), np.asarray([0.5, 1.5])),
        (np.asarray([2.0]), np.asarray([2.5])),
    ]

    shifted = circular_shift_spikes_within_trial_chunks(
        spike_times,
        trial_chunks,
        rng=rng,
        shift_fraction_bounds=(0.3, 0.6),
    )

    assert shifted.size == 4
    assert np.all(np.diff(shifted) >= 0.0)
    assert (
        np.count_nonzero((shifted >= 0.0) & (shifted < 0.5))
        + np.count_nonzero((shifted >= 1.0) & (shifted < 1.5))
    ) == 3
    assert np.count_nonzero((shifted >= 2.0) & (shifted < 2.5)) == 1


def test_panel_b_cache_path_and_roundtrip(tmp_path: Path) -> None:
    observed, shuffle = _fake_correlation_tables()
    metadata = build_panel_b_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        n_shuffles=200,
        shuffle_seed=11,
        min_movement_firing_rate_hz=0.5,
        min_tuning_stability_correlation=0.5,
        shift_fraction_bounds=(0.3, 0.6),
    )
    cache_path = build_panel_b_cache_path(tmp_path, metadata)

    assert metadata["cache_version"] == PANEL_B_CACHE_VERSION
    assert metadata["shuffle_unit"] == "light_spike_times"
    assert metadata["shuffle_scope"] == "per_unit_per_trajectory_per_trial"
    assert metadata["shift_fraction_bounds"] == [0.3, 0.6]
    assert cache_path.name.startswith(
        "figure_3_2_panel_b_circular_shuffle_v1_datasets-L14"
    )

    save_panel_b_cache(cache_path, {"observed": observed, "shuffle": shuffle}, metadata)
    loaded = load_panel_b_cache(cache_path, metadata)

    assert loaded is not None
    assert loaded["observed"]["correlation"].tolist() == pytest.approx(
        observed["correlation"].tolist()
    )
    assert loaded["shuffle"]["correlation"].tolist() == pytest.approx(
        shuffle["correlation"].tolist()
    )

    stale_metadata = dict(metadata)
    stale_metadata["n_shuffles"] = 201
    assert load_panel_b_cache(cache_path, stale_metadata) is None


def test_panel_b_loader_uses_matching_circular_shuffle_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed, shuffle = _fake_correlation_tables()
    metadata = build_panel_b_cache_metadata(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        n_shuffles=200,
        shuffle_seed=PANEL_B_SHUFFLE_SEED,
        min_movement_firing_rate_hz=0.5,
        min_tuning_stability_correlation=PANEL_B_MIN_TUNING_STABILITY_CORRELATION,
        shift_fraction_bounds=PANEL_B_SHIFT_FRACTION_BOUNDS,
    )
    save_panel_b_cache(
        build_panel_b_cache_path(tmp_path, metadata),
        {"observed": observed, "shuffle": shuffle},
        metadata,
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "build_panel_b_tuning_correlation_data",
        lambda **_kwargs: pytest.fail("Panel B cache was not used."),
    )

    payload = load_panel_b_tuning_correlation_data(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=50,
        position_offset=10,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        panel_b_cache_dir=tmp_path,
    )

    assert payload["observed"]["trajectory_type"].tolist() == observed[
        "trajectory_type"
    ].tolist()
    assert payload["shuffle"]["shift_fraction_min"].tolist() == pytest.approx(
        shuffle["shift_fraction_min"].tolist()
    )


def test_build_panel_b_percentile_table_uses_each_cells_own_shuffle_null() -> None:
    observed, shuffle = _fake_correlation_tables()

    percentile_table = build_panel_b_percentile_table(observed, shuffle)

    assert len(percentile_table) == len(observed)
    assert set(percentile_table["trajectory_type"]) == set(
        figure_3_module.PANEL_B_TRAJECTORY_TYPES
    )
    assert percentile_table["n_shuffles"].tolist() == [2, 2, 2, 2]
    assert percentile_table["percentile"].tolist() == pytest.approx(
        [100.0, 100.0, 100.0, 100.0]
    )


def test_plot_panel_b_percentile_grid_draws_four_trajectory_panels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed, shuffle = _fake_correlation_tables()
    percentile_table = build_panel_b_percentile_table(observed, shuffle)
    fig, ax = plt.subplots()

    plot_panel_b_percentile_grid(ax, percentile_table)

    assert len(ax.child_axes) == len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    assert ax.child_axes[0].get_legend() is None
    assert any(
        text.get_text().startswith("med. ") for text in ax.child_axes[0].texts
    )
    assert any(
        text.get_text().startswith(">=95%: ") for text in ax.child_axes[0].texts
    )
    assert [line.get_xdata()[0] for line in ax.child_axes[0].lines] == [95.0]
    assert len({child_ax.get_ylim() for child_ax in ax.child_axes}) == 1
    assert all(child_ax.patches for child_ax in ax.child_axes)
    assert [child_ax.get_xlim() for child_ax in ax.child_axes] == [
        pytest.approx((0.0, 100.0))
    ] * len(figure_3_module.PANEL_B_TRAJECTORY_TYPES)
    plt.close(fig)


def test_plot_panel_a_example_grid_draws_all_single_unit_examples() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = [
        _fake_panel_a_example(("center_to_left", "right_to_center"))
        for _index in range(4)
    ]
    fig, ax = plt.subplots()

    plot_panel_a_example_grid(ax, examples)

    assert len(ax.child_axes) == len(examples)
    assert [child.texts[0].get_text() for child in ax.child_axes] == [
        "Cell 1",
        "Cell 2",
        "Cell 3",
        "Cell 4",
    ]
    plt.close(fig)


def test_default_cli_matches_figure_3_2_defaults() -> None:
    args = parse_arguments([])

    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert DEFAULT_OUTPUT_NAME == "figure_3_2"
    assert args.panel_b_n_shuffles == PANEL_B_N_SHUFFLES
    assert args.panel_b_shuffle_seed == PANEL_B_SHUFFLE_SEED
    assert args.panel_b_cache_dir is None
    assert args.refresh_panel_b_cache is False
    assert args.panel_example_cache_dir is None


def test_make_figure_3_2_uses_panel_a_examples_and_panel_b_similarity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_load_panel_quantification_data(**kwargs: object):
        calls["quant_kwargs"] = kwargs
        return {
            "similarity": pandas.DataFrame(),
            "encoding_delta": pandas.DataFrame(),
            "decoding_error": pandas.DataFrame(),
        }

    def fake_load_panel_b_tuning_correlation_data(**kwargs: object):
        calls["panel_b_correlation_kwargs"] = kwargs
        observed, shuffle = _fake_correlation_tables()
        return {"observed": observed, "shuffle": shuffle}

    def fake_load_panel_a_example_data(**kwargs: object):
        calls.setdefault("panel_a_example_kwargs", []).append(kwargs)
        return _fake_panel_a_example(("center_to_left", "right_to_center"))

    def fake_load_panel_a_additional_examples(**kwargs: object):
        calls["panel_a_additional_examples_kwargs"] = kwargs
        return [_fake_panel_a_example(("center_to_left", "right_to_center"))]

    def fake_plot_panel_a_example_grid(ax, examples):
        calls["panel_a_example_count"] = len(examples)
        ax.text(0.5, 0.5, "Panel A examples")

    def fake_save_figure(figure, output_path: Path, dpi: int):
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        calls["figure_text"] = [text.get_text() for text in figure.texts]
        return output_path

    monkeypatch.setattr(
        figure_3_2_module,
        "load_panel_quantification_data",
        fake_load_panel_quantification_data,
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "load_panel_b_tuning_correlation_data",
        fake_load_panel_b_tuning_correlation_data,
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "load_panel_a_example_data",
        fake_load_panel_a_example_data,
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "load_panel_a_additional_examples",
        fake_load_panel_a_additional_examples,
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "plot_panel_a_example_grid",
        fake_plot_panel_a_example_grid,
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "plot_panel_c_vision_tuning_panel",
        lambda ax, *_args: ax.text(0.5, 0.5, "C"),
    )
    monkeypatch.setattr(
        figure_3_2_module,
        "plot_panel_d_route_place_panel",
        lambda ax, *_args: ax.text(0.5, 0.5, "D"),
    )
    monkeypatch.setattr(figure_3_2_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_3_2.svg"
    saved_path = make_figure_3_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=3,
        position_offset=0,
        speed_threshold_cm_s=4.0,
        sigma_bins=0.0,
        dpi=300,
        panel_example_cache_dir=tmp_path / "example-cache",
        panel_b_cache_dir=tmp_path / "panel-b-cache",
        panel_b_n_shuffles=5,
        panel_b_shuffle_seed=11,
    )

    assert saved_path == output_path
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["C", "D"]
    assert {"A", "B"}.issubset(calls["figure_text"])
    assert "Light-dark tuning similarity" in calls["figure_text"]
    assert calls["panel_b_correlation_kwargs"]["n_shuffles"] == 5
    assert calls["panel_b_correlation_kwargs"]["shuffle_seed"] == 11
    assert calls["panel_b_correlation_kwargs"]["panel_b_cache_dir"] == (
        tmp_path / "panel-b-cache"
    )
    assert len(calls["panel_a_example_kwargs"]) == len(figure_3_module.PANEL_A_EXAMPLES)
    assert calls["panel_a_example_count"] == len(figure_3_module.PANEL_A_EXAMPLES) + 1
    assert calls["panel_a_additional_examples_kwargs"]["panel_example_cache_dir"] == (
        tmp_path / "example-cache"
    )
