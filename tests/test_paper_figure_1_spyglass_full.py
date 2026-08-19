"""Tests for the complete offline Spyglass Figure 1 adapter."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.paper_figures import _figure_1_spyglass_full as full
from v1ca1.paper_figures import figure_1_spyglass as command


def _example(
    animal_name: str,
    date: str,
    epoch: str,
    unit_id: int,
) -> dict[str, Any]:
    """Return a small plotting-compatible example payload."""
    trajectories = full.legacy.PANEL_B_VISUAL_TRAJECTORIES
    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "region": "v1",
        "unit_id": unit_id,
        "raster_positions": {
            trajectory: [np.asarray([0.2, 0.8])] for trajectory in trajectories
        },
        "firing_rates": {
            trajectory: (
                np.linspace(0.01, 0.99, 50),
                np.linspace(0.0, 1.0, 50),
            )
            for trajectory in trajectories
        },
    }


def _render_payload(run_dir: Path) -> dict[str, Any]:
    """Return the minimal payload consumed by the renderer injection layer."""
    campaign = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "analysis_parameters": {"pipeline": full.FULL_FIGURE_PIPELINE},
        "sessions": [],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(campaign),
        encoding="utf-8",
    )
    light_a = _example("L14", "20240611", "02_r1", 229)
    light_b = _example("L14", "20240611", "06_r3", 229)
    dark = _example("L14", "20240611", "08_r4", 229)
    panel_b = {
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "unit_id": 229,
        "epoch_order": ("02_r1", "06_r3", "dark"),
        "epoch_labels": {"02_r1": "02_r1", "06_r3": "06_r3", "dark": "Dark"},
        "epoch_examples": {
            "02_r1": light_a,
            "06_r3": light_b,
            "dark": dark,
        },
        "trajectories": full.legacy.PANEL_B_VISUAL_TRAJECTORIES,
    }
    panels = {
        (order, plotted): np.ones((2, 50), dtype=float)
        for order in full.legacy.PANEL_D_TRAJECTORY_TYPES
        for plotted in full.legacy.PANEL_D_TRAJECTORY_TYPES
    }
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "mode": "full",
        "datasets": full.EXPECTED_DATASETS,
        "regions": ("v1",),
        "panel_b_example": panel_b,
        "panel_c_examples": [
            _example("L14", "20240611", "08_r4", 34),
            _example("L15", "20241121", "10_r5", 473),
        ],
        "panel_d_payload": {"panels_by_region": {"v1": panels}},
        "motor_delta_table": pd.DataFrame(
            columns=full.legacy.MOTOR_DELTA_TABLE_COLUMNS
        ),
        "encoding_delta_table": pd.DataFrame(
            columns=full.legacy.ENCODING_DELTA_TABLE_COLUMNS
        ),
        "decoding_absolute_error_table": pd.DataFrame(
            columns=full.legacy.DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS
        ),
        "decoding_trial_error_table": pd.DataFrame(
            columns=full.legacy.DECODING_TRIAL_ERROR_TABLE_COLUMNS
        ),
    }


def test_cli_defaults_to_panel_d_and_accepts_full_figure_scope() -> None:
    old = command.parse_arguments(
        ["--run-id", "panel-d-run", "--mode", "full"]
    )
    assert old.figure_scope == "panel-d"

    complete = command.parse_arguments(
        [
            "--run-id",
            "full-run",
            "--mode",
            "full",
            "--figure-scope",
            "full-figure",
            "--promote-to",
            "paper_figures/output/figure_1.svg",
            "--replace-promoted-output",
        ]
    )
    assert complete.figure_scope == "full-figure"
    assert complete.asset_dir == command.DEFAULT_ASSET_DIR
    assert complete.promote_to == Path("paper_figures/output/figure_1.svg")
    assert complete.replace_promoted_output

    partial = command.parse_arguments(
        [
            "--run-id",
            "full-run",
            "--mode",
            "l14-validation",
            "--figure-scope",
            "full-figure",
        ]
    )
    assert partial.mode == "l14-validation"

    with pytest.raises(SystemExit):
        command.parse_arguments(
            [
                "--run-id",
                "panel-d-run",
                "--mode",
                "full",
                "--promote-to",
                "figure_1.svg",
            ]
        )
    with pytest.raises(SystemExit):
        command.parse_arguments(
            [
                "--run-id",
                "full-run",
                "--mode",
                "full",
                "--replace-promoted-output",
            ]
        )


def test_full_figure_cli_renders_then_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "full-run"
    source = run_dir / "figures" / "current.svg"
    destination_dir = tmp_path / "published"
    destination_dir.mkdir()
    destination = destination_dir / "figure_1.svg"
    payload = {"run_dir": run_dir}
    calls: list[tuple[Any, ...]] = []

    def _load(**kwargs: Any) -> dict[str, Any]:
        calls.append(("load", kwargs))
        return payload

    def _render(
        observed: Mapping[str, Any],
        **kwargs: Any,
    ) -> Path:
        calls.append(("render", observed, kwargs))
        return source

    def _promote(
        observed: Mapping[str, Any],
        **kwargs: Any,
    ) -> Path:
        calls.append(("promote", observed, kwargs))
        return destination

    monkeypatch.setattr(full, "load_full_figure_1_payload", _load)
    monkeypatch.setattr(full, "render_full_figure_1", _render)
    monkeypatch.setattr(full, "promote_full_figure_1", _promote)
    command.main(
        [
            "--run-id",
            "full-run",
            "--mode",
            "full",
            "--figure-scope",
            "full-figure",
            "--output-path",
            str(source),
            "--promote-to",
            str(destination),
            "--replace-promoted-output",
        ]
    )

    assert [call[0] for call in calls] == ["load", "render", "promote"]
    assert calls[1][2]["output_path"] == source
    assert calls[2][2] == {
        "source_path": source,
        "destination_path": destination,
        "replace": True,
    }


def test_partial_session_selection_accepts_later_completed_sessions() -> None:
    sessions = [
        {
            "animal_name": animal_name,
            "date": date,
            "epochs": [epoch],
        }
        for animal_name, date, epoch in full.EXPECTED_DATASETS
    ]

    selected, datasets = full._ordered_sessions(
        sessions,
        mode="l14-validation",
    )

    assert datasets == (full.L14_DATASET,)
    assert [(row["animal_name"], row["date"]) for row in selected] == [
        full.L14_DATASET[:2]
    ]


def test_model_adapters_use_new_contrast_signs_and_stable_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motor_manifest = tmp_path / "motor" / "manifest.parquet"
    dpp_path = tmp_path / "dpp.parquet"
    motor_manifest.parent.mkdir()
    motor_manifest.touch()
    dpp_path.touch()
    session = {
        "animal_name": "L14",
        "date": "20240611",
        "epochs": ["08_r4"],
        "nwb_file_name": "L1420240611_augmented.nwb",
        "artifacts": {
            "motor_encoding": [
                {
                    "artifact_origin": "computed",
                    "selection": {
                        "nwb_file_name": "L1420240611_augmented.nwb",
                        "epoch": "08_r4",
                    },
                    "artifacts": {
                        "artifact_manifest_path": {
                            "relative_path": str(motor_manifest.relative_to(tmp_path))
                        }
                    },
                }
            ],
            "dpp_encoding": [
                {
                    "artifact_origin": "computed",
                    "selection": {
                        "nwb_file_name": "L1420240611_augmented.nwb",
                        "epoch": "08_r4",
                    },
                    "artifacts": {
                        "dpp_encoding_path": {
                            "relative_path": str(dpp_path.relative_to(tmp_path))
                        }
                    },
                }
            ],
        }
    }
    nested = xr.Dataset(
        {
            "pooled_delta_bits_per_spike": (
                ("delta_metric", "unit"),
                np.asarray([[0.25, np.nan]]),
            )
        },
        coords={
            "delta_metric": [full.legacy.MOTOR_DELTA_METRIC],
            "unit": [0, 1],
            "stable_unit_id": ("unit", ["merge:10", "merge:20"]),
        },
    )
    monkeypatch.setattr(
        full.motor_encoding,
        "load_motor_encoding_artifact",
        lambda path: {
            "artifact_origin": "computed",
            "metadata": {
                "animal_name": "L14",
                "date": "20240611",
                "epoch": "08_r4",
                "region": "v1",
            },
            "nested_cv": nested,
        },
    )
    dpp_table = pd.DataFrame(
        {
            "animal_name": ["L14"],
            "date": ["20240611"],
            "epoch": ["08_r4"],
            "region": ["v1"],
            "stable_unit_id": ["merge:10"],
            "heldout_spike_count": [25],
            "dpp_vs_absolute_place_bits_per_spike": [0.4],
            "dpp_vs_distance_to_reward_bits_per_spike": [-0.2],
        }
    )
    monkeypatch.setattr(
        full.dpp_encoding,
        "load_dpp_encoding_artifact",
        lambda path: dpp_table,
    )

    motor = full._motor_delta_table(tmp_path, [session])
    encoding = full._encoding_delta_table(tmp_path, [session])

    assert motor["unit"].tolist() == ["merge:10"]
    assert motor["delta_log_likelihood_bits_per_spike"].tolist() == [0.25]
    assert dict(
        zip(
            encoding["comparison"],
            encoding["delta_log_likelihood_bits_per_spike"],
            strict=True,
        )
    ) == {
        "dpp_vs_absolute_place": 0.4,
        "dpp_vs_absolute_task_progression": -0.2,
    }


def test_l14_source_injection_uses_l14_for_decoding_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "partial-run"
    run_dir.mkdir(parents=True)
    payload = _render_payload(run_dir)
    payload["datasets"] = (full.L14_DATASET,)
    payload["panel_c_examples"] = payload["panel_c_examples"][:1]
    original_animals = full.legacy.PANEL_H_DECODING_ANIMALS
    original_examples = full.legacy.PANEL_E_EXAMPLES
    calls = []

    def _brackets(table: Any, **kwargs: Any) -> tuple[()]:
        calls.append(kwargs)
        return ()

    monkeypatch.setattr(
        full.legacy,
        "build_decoding_significance_brackets",
        _brackets,
    )
    with full._offline_legacy_sources(payload):
        assert full.legacy.PANEL_H_DECODING_ANIMALS == ("L14",)
        assert full.legacy.PANEL_E_EXAMPLES == (
            ("L14", "20240611", "08_r4", "v1", 34),
        )
        full.legacy.build_decoding_significance_brackets(object())
    assert calls == [{"animal_names": ("L14",)}]
    assert full.legacy.PANEL_H_DECODING_ANIMALS == original_animals
    assert full.legacy.PANEL_E_EXAMPLES == original_examples


def test_renderer_injects_artifacts_restores_loaders_and_guards_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "full-run"
    run_dir.mkdir(parents=True)
    payload = _render_payload(run_dir)
    original_panel_b_loader = full.legacy.load_panel_b_visual_example_data
    calls = []

    def _make_figure(**kwargs: Any) -> Path:
        calls.append(kwargs)
        selected_panel_b = full.legacy.load_panel_b_visual_example_data(
            animal_name="L14",
            date="20240611",
            region="v1",
            unit_id=229,
        )
        assert selected_panel_b is payload["panel_b_example"]
        selected_panel_c = full.legacy.load_or_compute_panel_e_example_data(
            animal_name="L14",
            date="20240611",
            epoch="08_r4",
            region="v1",
            unit_id=34,
        )
        assert selected_panel_c["unit_id"] == 34
        assert full.legacy.load_motor_delta_table(
            datasets=full.EXPECTED_DATASETS,
            region="v1",
        ) is payload["motor_delta_table"]
        kwargs["output_path"].parent.mkdir(parents=True)
        kwargs["output_path"].write_text("figure", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(full.legacy, "make_figure_1", _make_figure)
    output = run_dir / "figures" / "figure_1.svg"
    returned = full.render_full_figure_1(payload, output_path=output)

    assert returned == output
    provenance_path = full.get_full_figure_provenance_path(output)
    provenance = full.load_json(provenance_path)
    assert provenance["schema_version"] == (
        full.FULL_FIGURE_PROVENANCE_SCHEMA_VERSION
    )
    assert provenance["run_id"] == "full-run"
    assert provenance["mode"] == "full"
    assert provenance["figure"]["run_relative_path"] == "figures/figure_1.svg"
    assert provenance["figure"]["sha256"] == full.file_sha256(output)
    assert calls[0]["data_root"] == run_dir.resolve()
    assert calls[0]["decoding_n_permutations"] == (
        full.legacy.DECODING_PERMUTATION_COUNT
    )
    assert calls[0]["decoding_permutation_seed"] == (
        full.legacy.DECODING_PERMUTATION_SEED
    )
    assert full.legacy.load_panel_b_visual_example_data is original_panel_b_loader
    with pytest.raises(FileExistsError, match="overwrite"):
        full.render_full_figure_1(payload, output_path=output)
    with pytest.raises(ValueError, match="inside its run"):
        full.render_full_figure_1(
            payload,
            output_path=tmp_path / "outside.svg",
        )


def test_promoter_validates_receipt_and_requires_explicit_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "full-run"
    run_dir.mkdir(parents=True)
    payload = _render_payload(run_dir)
    monkeypatch.setattr(
        full,
        "code_provenance",
        lambda: {
            "v1ca1_version": "test",
            "v1ca1_git_commit": "a" * 40,
            "v1ca1_git_dirty": False,
        },
    )

    def _make_figure(**kwargs: Any) -> Path:
        kwargs["output_path"].parent.mkdir(parents=True)
        kwargs["output_path"].write_bytes(b"validated figure")
        return kwargs["output_path"]

    monkeypatch.setattr(full.legacy, "make_figure_1", _make_figure)
    source = run_dir / "figures" / "figure_1.svg"
    full.render_full_figure_1(payload, output_path=source)

    publish_dir = tmp_path / "published"
    publish_dir.mkdir()
    destination = publish_dir / "figure_1.svg"
    returned = full.promote_full_figure_1(
        payload,
        source_path=source,
        destination_path=destination,
    )
    assert returned == destination
    assert destination.read_bytes() == source.read_bytes()
    published_provenance = full.load_json(
        full.get_full_figure_provenance_path(destination)
    )
    assert published_provenance["promotion"]["destination_path"] == str(
        destination.resolve()
    )
    assert published_provenance["promotion"]["sha256"] == full.file_sha256(
        source
    )

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        full.promote_full_figure_1(
            payload,
            source_path=source,
            destination_path=destination,
        )
    destination.write_bytes(b"stale")
    full.promote_full_figure_1(
        payload,
        source_path=source,
        destination_path=destination,
        replace=True,
    )
    assert destination.read_bytes() == b"validated figure"

    with pytest.raises(ValueError, match="outside its run"):
        full.promote_full_figure_1(
            payload,
            source_path=source,
            destination_path=run_dir / "figures" / "published.svg",
        )
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="provenance"):
        full.promote_full_figure_1(
            payload,
            source_path=source,
            destination_path=publish_dir / "tampered.svg",
        )
