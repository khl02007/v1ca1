"""Tests for the database-free Figure 2 artifact adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures import figure_2_spyglass as figure

IDENTITY_ROWS = {
    unit_id: {
        "spikesorting_merge_id": "merge",
        "unit_id": str(unit_id),
        "stable_unit_id": f"merge:{unit_id}",
        "group_unit_id": str(unit_id),
    }
    for unit_id in (1, 2, 3)
}


def _movement(epoch: str, rates: dict[int, float]) -> pd.DataFrame:
    """Return a minimal movement-rate table for adapter filtering."""
    return pd.DataFrame.from_records(
        [
            {
                **IDENTITY_ROWS[unit_id],
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "epoch": epoch,
                "movement_firing_rate_hz": rate,
            }
            for unit_id, rate in rates.items()
        ]
    )


def _stability(epoch: str, correlations: dict[int, float]) -> list[pd.DataFrame]:
    """Return four minimal trajectory-specific stability tables."""
    tables = []
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        tables.append(
            pd.DataFrame.from_records(
                [
                    {
                        **IDENTITY_ROWS[unit_id],
                        "animal_name": "L14",
                        "date": "20240611",
                        "region": "v1",
                        "epoch": epoch,
                        "trajectory_type": trajectory_type,
                        "stability_correlation": (
                            correlation if trajectory_index == 0 else 0.0
                        ),
                    }
                    for unit_id, correlation in correlations.items()
                ]
            )
        )
    return tables


def _similarity(epoch: str) -> pd.DataFrame:
    """Return all-unit absolute-overlap rows for the two plotted turns."""
    rows = []
    for unit_id in IDENTITY_ROWS:
        for label, value in (("left_turn", 0.2), ("right_turn", 0.4)):
            rows.append(
                {
                    **IDENTITY_ROWS[unit_id],
                    "animal_name": "L14",
                    "date": "20240611",
                    "region": "v1",
                    "epoch": epoch,
                    "similarity_metric": "absolute_overlap",
                    "comparison_label": label,
                    "similarity": value + 0.01 * unit_id,
                    "similarity_status": "valid",
                }
            )
    return pd.DataFrame.from_records(rows)


def _sessions() -> list[dict[str, Any]]:
    """Return synthetic complete-session manifest rows."""
    return [
        {
            "animal_name": animal_name,
            "date": date,
            "epochs": {"dark": dark, "AB": "02_r1", "BA": "06_r3"},
        }
        for animal_name, date, dark in figure.EXPECTED_DATASETS
    ]


def _render_payload(run_dir: Path) -> dict[str, Any]:
    """Return the minimal already-adapted payload used by source injection."""
    return {
        "run_dir": run_dir,
        "campaign": {
            "run_id": run_dir.name,
            "analysis_parameters": {"pipeline": "figure_2"},
        },
        "datasets": figure.EXPECTED_DATASETS,
        "regions": ("v1",),
        "panel_a_examples": [
            {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "unit_id": unit_id,
                "trajectories": trajectories,
            }
            for animal_name, date, region, unit_id, trajectories in (
                figure.canonical.FIGURE_2_PANEL_A_EXAMPLES
            )
        ],
        "panel_b_shift_profile_table": pd.DataFrame(
            {"normalized_shift": [0.0], "rescaled_overlap": [0.5]}
        ),
        "panel_c_overlap_table": pd.DataFrame({"unit": [1]}),
    }


def _write_campaign_manifest(payload: dict[str, Any]) -> None:
    """Write the exact campaign snapshot needed by a render receipt."""
    run_dir = Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(payload["campaign"]),
        encoding="utf-8",
    )


def test_complete_session_manifest_order_and_epoch_contract() -> None:
    sessions = list(reversed(_sessions()))

    ordered = figure._ordered_sessions(sessions)

    assert [row["animal_name"] for row in ordered] == [
        dataset[0] for dataset in figure.EXPECTED_DATASETS
    ]
    changed = [dict(row) for row in sessions]
    changed[0] = {**changed[0], "epochs": {"dark": "08_r4"}}
    with pytest.raises(ValueError, match="noncanonical epochs"):
        figure._ordered_sessions(changed)


def test_cli_requires_promotion_destination_for_replacement() -> None:
    args = figure.parse_arguments(["--run-id", "run"])

    assert args.promote_to is None
    assert args.replace_promoted_output is False
    with pytest.raises(SystemExit):
        figure.parse_arguments(
            ["--run-id", "run", "--replace-promoted-output"]
        )


def test_path_curve_loader_selects_canonical_analysis_parameter(
    tmp_path: Path,
) -> None:
    """Display-smoothed curves must not shadow the canonical 4-cm curves."""
    records = []
    for parameter_name, n_bins in (
        (figure.TUNING_CURVE_PARAM_NAME, 4),
        ("figure1d_50bin_sigma1p5", 5),
    ):
        path = tmp_path / f"{parameter_name}.nc"
        xr.DataArray(
            np.arange(n_bins, dtype=float).reshape(1, n_bins),
            dims=("unit", "position"),
            coords={
                "unit": [0],
                "stable_unit_id": ("unit", ["merge:1"]),
            },
        ).to_netcdf(path)
        records.append(
            {
                "epoch": "08_r4",
                "region": "v1",
                "trajectory_type": "center_to_left",
                "trial_subset": "all",
                "tuning_curve_param_name": parameter_name,
                "tuning_curve_path": path.name,
            }
        )

    curves, path = figure._load_path_curve(
        records,
        source_root=tmp_path,
        epoch="08_r4",
        trajectory_type="center_to_left",
        trial_subset="all",
    )

    assert path.name == f"{figure.TUNING_CURVE_PARAM_NAME}.nc"
    assert curves["merge:1"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_panel_b_profiles_use_path_qc_and_dark_split_half_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current Panel B uses strict path gates and the dark-only ceiling."""
    session = {
        "animal_name": "L14",
        "date": "20240611",
        "epochs": {"dark": "08_r4", "AB": "02_r1", "BA": "06_r3"},
    }

    def stability_tables(epoch: str) -> list[pd.DataFrame]:
        return [
            pd.DataFrame.from_records(
                [
                    {
                        "stable_unit_id": "merge:100",
                        "unit_id": "100",
                        "trajectory_type": path,
                        "n_odd_spikes": 2,
                        "n_even_spikes": 2,
                        "odd_duration_s": 1.0,
                        "even_duration_s": 1.0,
                        "stability_correlation": (
                            0.5
                            if epoch == "02_r1" and path == "right_to_center"
                            else 0.75
                        ),
                        "stability_shape_overlap": (
                            0.8 if epoch == "08_r4" else 0.95
                        ),
                        "shape_overlap_status": "valid",
                    }
                ]
            )
            for path in figure.canonical.PANEL_B_PATH_ORDER
        ]

    monkeypatch.setattr(
        figure,
        "_load_nwb_sorting_unit_map",
        lambda _session: {"100": 7},
    )
    monkeypatch.setattr(
        figure,
        "_epoch_tuning_curve_records",
        lambda *_args, **_kwargs: ([], tmp_path),
    )
    monkeypatch.setattr(
        figure,
        "_load_stability_tables",
        lambda *_args, epoch, **_kwargs: stability_tables(epoch),
    )
    monkeypatch.setattr(
        figure,
        "_load_path_curve",
        lambda *_args, **_kwargs: (
            {"merge:100": np.asarray([1.0, 0.0, 0.0, 0.0])},
            tmp_path / "curve.nc",
        ),
    )

    table = figure._build_panel_b_shift_profile_table(
        tmp_path,
        [session],
        scratch_root=tmp_path,
    )

    assert list(table.columns) == list(
        figure.canonical.PANEL_H_SHIFT_PROFILE_COLUMNS
    )
    assert set(table["path"]) == set(figure.canonical.PANEL_B_PATH_ORDER) - {
        "right_to_center"
    }
    assert table["unit"].unique().tolist() == [7]
    zero = table[table["normalized_shift"] == 0.0]
    assert zero["dark_split_half_overlap"].tolist() == pytest.approx(
        [0.8] * 3
    )
    assert zero["light_split_half_overlap"].tolist() == pytest.approx(
        [0.95] * 3
    )
    assert zero["split_half_overlap"].tolist() == pytest.approx([0.8] * 3)
    assert zero["rescaled_overlap"].tolist() == pytest.approx([1.25] * 3)


def test_panel_c_intersects_fixed_filters_in_both_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = {
        "animal_name": "L14",
        "date": "20240611",
        "epochs": {"dark": "08_r4", "AB": "02_r1", "BA": "06_r3"},
    }
    similarity = {
        "02_r1": (_similarity("02_r1"), tmp_path / "light.parquet"),
        "08_r4": (_similarity("08_r4"), tmp_path / "dark.parquet"),
    }
    movement = {
        # Unit 2 fails only light MFR; equality at 0.5 must retain unit 1.
        "02_r1": _movement("02_r1", {1: 0.5, 2: 0.49, 3: 1.0}),
        "08_r4": _movement("08_r4", {1: 0.5, 2: 1.0, 3: 1.0}),
    }
    stability = {
        "02_r1": _stability("02_r1", {1: 0.5, 2: 0.8, 3: 0.8}),
        # Unit 3 fails only dark stability.
        "08_r4": _stability("08_r4", {1: 0.5, 2: 0.8, 3: 0.49}),
    }
    monkeypatch.setattr(
        figure,
        "_load_similarity_tables",
        lambda *_args, **_kwargs: similarity,
    )
    monkeypatch.setattr(
        figure,
        "_load_movement_table",
        lambda *_args, epoch, **_kwargs: movement[epoch],
    )
    monkeypatch.setattr(
        figure,
        "_load_stability_tables",
        lambda *_args, epoch, **_kwargs: stability[epoch],
    )

    table = figure._build_panel_c_overlap_table(
        tmp_path,
        [session],
        scratch_root=tmp_path,
    )

    assert table["unit"].tolist() == [1]
    assert table["comparison_label"].tolist() == ["right_turn"]
    assert table["similarity_dark"].tolist() == pytest.approx([0.41])
    assert table["similarity_light"].tolist() == pytest.approx([0.41])


def test_nwb_graph_length_uses_stored_geometry_and_spacing() -> None:
    graph = {
        "node_positions_cm": np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]]),
        "edge_order": np.asarray([[0, 1], [1, 2]]),
        "edge_spacing_cm": np.asarray([2.0]),
    }

    assert figure._graph_length_cm(graph) == 11.0


def test_swap_coordinate_uses_explicit_sorting_id_map() -> None:
    result = {
        "selected_units": pd.DataFrame.from_records(
            [
                {
                    "spikesorting_merge_id": "merge",
                    "unit_id": "100",
                    "stable_unit_id": "merge:100",
                    "group_unit_id": "0",
                },
                {
                    "spikesorting_merge_id": "merge",
                    "unit_id": "101",
                    "stable_unit_id": "merge:101",
                    "group_unit_id": "1",
                },
            ]
        ),
        "dataset": xr.Dataset(coords={"unit": ["0", "1"]}),
    }

    dataset, nwb_unit_ids = figure._numeric_unit_coordinate(
        result,
        sorting_unit_by_nwb_id={"100": 27, "101": 473},
    )

    assert dataset.coords["unit"].values.tolist() == [27, 473]
    assert nwb_unit_ids.tolist() == [100, 101]


def test_source_injection_returns_only_campaign_payload_and_restores() -> None:
    run_dir = Path("/tmp/figure-2-test-run")
    payload = _render_payload(run_dir)
    example_backend = figure.canonical._figure_2
    overlap_backend = figure.canonical._figure_2._figure_2
    original_example_loader = example_backend.load_panel_a_example_data
    original_shift_loader = (
        figure.canonical.load_or_compute_panel_h_shift_profile_table
    )

    with figure._offline_sources(payload):
        example_spec = figure.canonical.FIGURE_2_PANEL_A_EXAMPLES[0]
        example = example_backend.load_panel_a_example_data(
            animal_name=example_spec[0],
            date=example_spec[1],
            region=example_spec[2],
            unit_id=example_spec[3],
            trajectories=example_spec[4],
        )
        assert example is payload["panel_a_examples"][0]
        overlap = overlap_backend.load_panel_b_tuning_overlap_table(
            datasets=figure.EXPECTED_DATASETS,
            region="v1",
        )
        assert overlap is payload["panel_c_overlap_table"]
        assert (
            overlap_backend.filter_panel_b_overlap_by_even_odd_stability(
                overlap,
                datasets=figure.EXPECTED_DATASETS,
                region="v1",
                min_movement_firing_rate_hz=0.5,
                min_stability_correlation=0.5,
            )
            is overlap
        )
        shift = figure.canonical.load_or_compute_panel_h_shift_profile_table(
            datasets=figure.EXPECTED_DATASETS,
            region="v1",
            min_path_movement_firing_rate_hz=0.5,
            min_stability_correlation=0.5,
        )
        assert shift is payload["panel_b_shift_profile_table"]

    assert example_backend.load_panel_a_example_data is original_example_loader
    assert (
        figure.canonical.load_or_compute_panel_h_shift_profile_table
        is original_shift_loader
    )


def test_payload_loader_binds_completed_campaign_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v1ca1.spyglass.offline import figure_2 as campaign

    run_dir = tmp_path / "runs" / "figure2"
    run_dir.mkdir(parents=True)
    sessions = _sessions()
    manifest = {
        "run_id": "figure2",
        "analysis_parameters": {
            "pipeline": campaign.FIGURE_2_PIPELINE,
            "panel_a_examples": figure._current_panel_a_contract(),
        },
        "sessions": sessions,
    }
    marker_a = [object()]
    marker_b = pd.DataFrame({"normalized_shift": [0.0]})
    marker_c = pd.DataFrame({"unit": [1]})
    monkeypatch.setattr(
        campaign,
        "load_figure_2_campaign",
        lambda *_args, **_kwargs: (run_dir, manifest, sessions),
    )
    monkeypatch.setattr(figure, "_load_panel_a_examples", lambda *_args: marker_a)
    monkeypatch.setattr(
        figure,
        "_build_panel_b_shift_profile_table",
        lambda *_args, **_kwargs: marker_b,
    )
    monkeypatch.setattr(
        figure,
        "_build_panel_c_overlap_table",
        lambda *_args, **_kwargs: marker_c,
    )

    payload = figure.load_figure_2_payload(
        run_id="figure2",
        scratch_root=tmp_path,
    )

    assert payload["campaign"] is manifest
    assert payload["panel_a_examples"] is marker_a
    assert payload["panel_b_shift_profile_table"] is marker_b
    assert payload["panel_c_overlap_table"] is marker_c
    assert "panel_d_swap_payload" not in payload
    assert "panel_e_decoding_error_table" not in payload


def test_payload_loader_rejects_historical_four_example_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v1ca1.spyglass.offline import figure_2 as campaign

    run_dir = tmp_path / "runs" / "figure2-old"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "figure2-old",
        "analysis_parameters": {
            "pipeline": campaign.FIGURE_2_PIPELINE,
            "panel_a_examples": [
                {
                    "animal_name": "L14",
                    "date": "20240611",
                    "sorting_unit_id": 34,
                    "trajectory_types": ["center_to_left", "right_to_center"],
                }
            ],
        },
    }
    monkeypatch.setattr(
        campaign,
        "load_figure_2_campaign",
        lambda *_args, **_kwargs: (run_dir, manifest, _sessions()),
    )

    with pytest.raises(ValueError, match="predates the current eight-example"):
        figure.load_figure_2_payload(
            run_id="figure2-old",
            scratch_root=tmp_path,
        )


def test_renderer_uses_canonical_layout_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure2"
    run_dir.mkdir(parents=True)
    payload = _render_payload(run_dir)
    _write_campaign_manifest(payload)
    calls = []

    def _make_figure(**kwargs: Any) -> Path:
        calls.append(kwargs)
        assert (
            figure.canonical.load_or_compute_panel_h_shift_profile_table(
                datasets=figure.EXPECTED_DATASETS,
                region="v1",
                min_path_movement_firing_rate_hz=0.5,
                min_stability_correlation=0.5,
            )
            is payload["panel_b_shift_profile_table"]
        )
        kwargs["output_path"].parent.mkdir(parents=True)
        kwargs["output_path"].write_text("figure", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(figure.canonical, "make_figure_2", _make_figure)
    output = run_dir / "figures" / "figure_2_spyglass.svg"

    returned = figure.render_figure_2(payload, output_path=output)

    assert returned == output
    assert output.read_text(encoding="utf-8") == "figure"
    provenance = figure.get_figure_provenance_path(output)
    assert provenance.is_file()
    provenance_data = json.loads(provenance.read_text(encoding="utf-8"))
    assert provenance_data["artifact_kind"] == figure.FIGURE_ARTIFACT_KIND
    assert provenance_data["figure"]["sha256"] == figure.file_sha256(output)
    assert calls[0]["data_root"] == run_dir.resolve()
    assert calls[0]["output_path"] != output
    assert calls[0]["output_path"].suffix == ".svg"
    assert calls[0]["light_epoch"] == "02_r1"
    assert calls[0]["dark_epoch"] is None
    assert "decoding_n_permutations" not in calls[0]
    assert not list(output.parent.glob(".*.tmp.svg"))
    with pytest.raises(FileExistsError, match="overwrite"):
        figure.render_figure_2(payload, output_path=output)
    with pytest.raises(ValueError, match="inside its campaign run"):
        figure.render_figure_2(
            payload,
            output_path=tmp_path / "outside.svg",
        )

    published_dir = tmp_path / "published"
    published_dir.mkdir()
    published = published_dir / "figure_2.svg"
    assert figure.promote_figure_2(
        payload,
        source_path=output,
        destination_path=published,
    ) == published
    assert published.read_bytes() == output.read_bytes()
    assert figure.get_figure_provenance_path(published).is_file()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        figure.promote_figure_2(
            payload,
            source_path=output,
            destination_path=published,
        )
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches"):
        figure.promote_figure_2(
            payload,
            source_path=output,
            destination_path=published_dir / "tampered.svg",
        )


def test_renderer_removes_partial_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed save must not strand a final or temporary figure file."""
    run_dir = tmp_path / "runs" / "figure2"
    run_dir.mkdir(parents=True)
    payload = _render_payload(run_dir)

    def fail_after_partial_write(**kwargs: Any) -> None:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True)
        output_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(
        figure.canonical,
        "make_figure_2",
        fail_after_partial_write,
    )
    output = run_dir / "figures" / "figure_2_spyglass.svg"

    with pytest.raises(RuntimeError, match="synthetic save failure"):
        figure.render_figure_2(payload, output_path=output)

    assert not output.exists()
    assert not list(output.parent.glob(".*.tmp.svg"))
