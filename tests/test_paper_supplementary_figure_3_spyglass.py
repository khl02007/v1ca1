"""Tests for the Spyglass Supplementary Figure 3 adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from v1ca1.paper_figures import supplementary_figure_3_spyglass as figure


def _progression(epoch: str) -> pd.DataFrame:
    """Return one schema-complete motor progression row."""
    return pd.DataFrame(
        {
            "epoch": [epoch],
            "trajectory_type": ["center_to_left"],
            "variable": ["speed_cm_s"],
            "progression_bin_index": [0],
            "progression_bin_start": [0.0],
            "progression_bin_end": [0.1],
            "progression_bin_center": [0.05],
            "sample_count": [10],
            "median": [4.0],
            "q25": [3.0],
            "q75": [5.0],
        }
    )


def _payload(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "campaign": {"run_id": run_dir.name},
        "datasets": figure.EXPECTED_DATASETS,
        "regions": (),
        "motor_progression_table": pd.DataFrame({"value": [1]}),
    }


def test_motor_artifacts_are_pooled_with_dark_light_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = [
        {
            "animal_name": "L12",
            "date": "20240421",
            "epochs": {"dark": "08_r4"},
            "artifacts": {
                "epoch_motor_behavior": [
                    {"epoch": "08_r4", "manifest_path": "dark/manifest.parquet"},
                    {"epoch": figure.LIGHT_EPOCH, "manifest_path": "light/manifest.parquet"},
                ]
            },
        }
    ]
    monkeypatch.setattr(
        figure,
        "_artifact_manifest_path",
        lambda record, **_kwargs: tmp_path / str(record["epoch"]) / "manifest.parquet",
    )

    def load_motor(path: Path) -> dict[str, Any]:
        epoch = path.name
        return {
            "metadata": {
                "animal_name": "L12",
                "date": "20240421",
                "epoch": epoch,
            },
            "artifact_origin": "computed",
            "progression_summary": _progression(epoch),
        }

    monkeypatch.setattr(
        figure.epoch_motor_behavior,
        "load_epoch_motor_behavior_artifact",
        load_motor,
    )

    table = figure._build_motor_progression_table(tmp_path, sessions)

    assert table["epoch_type"].tolist() == ["dark", "light"]
    assert table["dark_epoch"].tolist() == ["08_r4", "08_r4"]
    assert table["light_epoch"].tolist() == [figure.LIGHT_EPOCH] * 2
    assert table["dataset_label"].tolist() == ["L12 20240421"] * 2


def test_context_patches_only_active_motor_loader_and_restores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload(run_dir)
    original = figure.canonical.load_panel_b_motor_progression_table

    with figure._offline_sources(payload):
        observed = figure.canonical.load_panel_b_motor_progression_table(
            data_root=run_dir,
            datasets=payload["datasets"],
            light_epoch=figure.LIGHT_EPOCH,
        )
        assert observed is payload["motor_progression_table"]

    assert figure.canonical.load_panel_b_motor_progression_table is original


def test_render_is_atomic_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps(payload["campaign"]),
        encoding="utf-8",
    )
    output = figure.get_output_path(run_dir=run_dir)

    def render(**kwargs: Any) -> Path:
        assert figure.canonical.load_panel_b_motor_progression_table(
            data_root=kwargs["data_root"],
            datasets=kwargs["datasets"],
        ) is payload["motor_progression_table"]
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("supp3", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(figure.canonical, "make_supplementary_figure_3", render)

    assert figure.render_supplementary_figure_3(
        payload,
        output_path=output,
    ) == output
    assert output.read_text(encoding="utf-8") == "supp3"
    assert figure.get_figure_provenance_path(output).is_file()
    with pytest.raises(FileExistsError):
        figure.render_supplementary_figure_3(payload, output_path=output)
