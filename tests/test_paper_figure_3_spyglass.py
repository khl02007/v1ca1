"""Tests for the database-free current Figure 3 adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.paper_figures import figure_3_spyglass as figure


def _swap_dataset(units: list[int]) -> xr.Dataset:
    """Return one minimal mapped two-model SwapGLM result."""
    trajectories = np.asarray(
        [
            "center_to_left",
            "left_to_center",
            "center_to_right",
            "right_to_center",
        ],
        dtype=str,
    )
    models = np.asarray(
        ["visual", figure.MULTIPLICATIVE_MODEL, figure.ADDITIVE_MODEL],
        dtype=str,
    )
    tp_grid = np.linspace(0.0, 1.0, 5)
    observed_position = np.linspace(0.1, 0.9, 4)
    shape = (len(models), len(trajectories), len(units))
    delta = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0
    predicted = np.broadcast_to(
        np.linspace(1.0, 2.0, len(tp_grid))[None, None, :, None],
        (len(models), len(trajectories), len(tp_grid), len(units)),
    ).copy()
    predicted[1] += 1.0
    predicted[2] += 0.5
    return xr.Dataset(
        data_vars={
            figure.swap_glm.PRIMARY_METRIC: (
                ("model", "trajectory", "unit"),
                delta,
            ),
            "swap_segment_start": (
                ("trajectory",),
                np.full(len(trajectories), 0.25),
            ),
            "swap_segment_end": (
                ("trajectory",),
                np.full(len(trajectories), 0.50),
            ),
            "test_light_observed_rate_hz": (
                ("trajectory", "tp_observed_bin", "unit"),
                np.ones(
                    (len(trajectories), len(observed_position), len(units)),
                    dtype=float,
                ),
            ),
            "test_light_swapped_hz_grid": (
                ("model", "trajectory", "tp_grid", "unit"),
                predicted,
            ),
            "swap_source_trajectory": (
                ("trajectory",),
                trajectories[::-1],
            ),
        },
        coords={
            "model": models,
            "trajectory": trajectories,
            "unit": np.asarray(units, dtype=int),
            "tp_grid": tp_grid,
            "tp_observed_bin": observed_position,
        },
    )


def _loaded_results(tmp_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return synthetic L12 and L19 results covering all four examples."""
    output = {}
    for animal_name, date, units, dark_epoch in (
        ("L12", "20240421", [53, 270], "08_r4"),
        ("L19", "20250930", [66, 31], "08_r4"),
    ):
        output[(animal_name, date)] = {
            "dataset": _swap_dataset(units),
            "selected_units": pd.DataFrame(),
            "eligible_unit_mask": np.ones(len(units), dtype=bool),
            "metadata": {
                "dark_epoch": dark_epoch,
                "light_train_epoch": figure.LIGHT_TRAIN_EPOCH,
                "light_test_epoch": figure.LIGHT_TEST_EPOCH,
            },
            "source_path": tmp_path / f"{animal_name}-swap.parquet",
        }
    return output


def _payload(run_dir: Path) -> dict[str, Any]:
    """Return a minimal already-adapted current Figure 3 payload."""
    return {
        "run_dir": run_dir,
        "campaign": {
            "run_id": run_dir.name,
            "analysis_parameters": {"pipeline": "figure_2"},
        },
        "datasets": figure.EXPECTED_DATASETS,
        "regions": (figure.REGION,),
        "swap_delta": pd.DataFrame({"kind": ["multiplicative"]}),
        "swap_additive_delta": pd.DataFrame({"kind": ["additive"]}),
        "swap_examples": [{"kind": "example"}],
    }


def _write_campaign_manifest(payload: dict[str, Any]) -> None:
    """Write the exact campaign snapshot needed by a render receipt."""
    run_dir = Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(payload["campaign"]),
        encoding="utf-8",
    )


def test_adapter_targets_current_figure_3_and_cli_defaults() -> None:
    assert figure.canonical.DEFAULT_OUTPUT_NAME == "figure_3"
    assert figure.DEFAULT_OUTPUT_NAME == "figure_3_spyglass"
    assert figure.REQUIRED_MODELS == (
        "task_segment_scalar",
        "visual_additive_delta",
    )

    args = figure.parse_arguments(["--run-id", "run"])

    assert args.run_id == "run"
    assert args.output_format == "svg"
    assert args.dpi == 300
    assert args.promote_to is None
    assert args.replace_promoted_output is False
    with pytest.raises(SystemExit):
        figure.parse_arguments(
            ["--run-id", "run", "--replace-promoted-output"]
        )


def test_missing_additive_model_fails_with_recompute_guidance(tmp_path: Path) -> None:
    dataset = _swap_dataset([1]).sel(
        model=["visual", figure.MULTIPLICATIVE_MODEL]
    )

    with pytest.raises(
        ValueError,
        match="missing.*visual_additive_delta.*Recompute",
    ):
        figure._require_swap_models(dataset, source_path=tmp_path / "swap.nc")


def test_delta_tables_and_examples_use_both_models_in_fixed_order(
    tmp_path: Path,
) -> None:
    loaded = _loaded_results(tmp_path)

    multiplicative = figure._delta_table(
        loaded,
        model_name=figure.MULTIPLICATIVE_MODEL,
    )
    additive = figure._delta_table(loaded, model_name=figure.ADDITIVE_MODEL)
    examples = figure._swap_examples(loaded)

    assert len(multiplicative) == len(additive) == 16
    assert set(multiplicative["model_name"]) == {figure.MULTIPLICATIVE_MODEL}
    assert set(additive["model_name"]) == {figure.ADDITIVE_MODEL}
    observed_specs = [
        (
            row["animal_name"],
            row["date"],
            row["region"],
            row["unit_id"],
            row["trajectory"],
        )
        for row in examples
    ]
    assert observed_specs == list(figure.canonical.PANEL_B_SWAP_EXAMPLES)
    assert all(
        set(example["models"])
        == {"visual", figure.MULTIPLICATIVE_MODEL, figure.ADDITIVE_MODEL}
        for example in examples
    )


def test_delta_table_filters_ineligible_and_nonfinite_units(tmp_path: Path) -> None:
    loaded = _loaded_results(tmp_path)
    l12 = loaded[("L12", "20240421")]
    l12["eligible_unit_mask"] = np.asarray([True, False])
    l19 = loaded[("L19", "20250930")]
    l19["dataset"][figure.swap_glm.PRIMARY_METRIC].loc[
        {"model": figure.MULTIPLICATIVE_MODEL, "unit": 66}
    ] = np.nan

    table = figure._delta_table(
        loaded,
        model_name=figure.MULTIPLICATIVE_MODEL,
    )

    assert not (
        (table["animal_name"] == "L12") & (table["unit"] == 270)
    ).any()
    assert not (
        (table["animal_name"] == "L19") & (table["unit"] == 66)
    ).any()
    assert len(table) == 8


def test_offline_context_patches_only_panel_loader_and_restores(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload(run_dir)
    original_loader = figure.canonical.load_figure_3_panel_data
    original_renderer = figure.canonical.make_figure_3

    with figure._offline_panel_data(payload):
        loaded = figure.canonical.load_figure_3_panel_data(
            data_root=run_dir,
            datasets=payload["datasets"],
            regions=payload["regions"],
            dark_epoch=None,
        )
        assert loaded == {
            "swap_delta": payload["swap_delta"],
            "swap_additive_delta": payload["swap_additive_delta"],
            "swap_examples": payload["swap_examples"],
        }
        assert figure.canonical.make_figure_3 is original_renderer

    assert figure.canonical.load_figure_3_panel_data is original_loader


def test_render_is_atomic_run_local_and_does_not_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload(run_dir)
    _write_campaign_manifest(payload)
    output_path = figure.get_output_path(run_dir=run_dir)
    calls = []

    def make_figure_3(**kwargs: Any) -> Path:
        calls.append(kwargs)
        panel_data = figure.canonical.load_figure_3_panel_data(
            data_root=kwargs["data_root"],
            datasets=kwargs["datasets"],
            regions=kwargs["regions"],
            dark_epoch=kwargs["dark_epoch"],
        )
        assert panel_data["swap_examples"] is payload["swap_examples"]
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("current figure 3", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(figure.canonical, "make_figure_3", make_figure_3)

    returned = figure.render_figure_3(payload, output_path=output_path, dpi=144)

    assert returned == output_path
    assert output_path.read_text(encoding="utf-8") == "current figure 3"
    provenance_path = figure.get_figure_provenance_path(output_path)
    assert provenance_path.is_file()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["artifact_kind"] == figure.FIGURE_ARTIFACT_KIND
    assert calls[0]["dpi"] == 144
    assert calls[0]["dark_epoch"] is None
    assert not list(output_path.parent.glob(".*.tmp.svg"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        figure.render_figure_3(payload, output_path=output_path)
    with pytest.raises(ValueError, match="inside its campaign"):
        figure.render_figure_3(
            payload,
            output_path=tmp_path / "outside.svg",
        )

    published_dir = tmp_path / "published"
    published_dir.mkdir()
    published = published_dir / "figure_3.svg"
    assert figure.promote_figure_3(
        payload,
        source_path=output_path,
        destination_path=published,
    ) == published
    assert published.read_bytes() == output_path.read_bytes()
    assert figure.get_figure_provenance_path(published).is_file()


def test_payload_requires_complete_figure_2_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v1ca1.spyglass.offline import figure_2 as offline_figure_2

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sessions = [
        {
            "animal_name": animal_name,
            "date": date,
            "epochs": {
                "dark": dark_epoch,
                "AB": figure.LIGHT_TRAIN_EPOCH,
                "BA": figure.LIGHT_TEST_EPOCH,
            },
        }
        for animal_name, date, dark_epoch in figure.EXPECTED_DATASETS
    ]
    campaign = {
        "analysis_parameters": {"pipeline": offline_figure_2.FIGURE_2_PIPELINE},
        "sessions": [
            {"animal_name": row["animal_name"], "status": "complete"}
            for row in sessions
        ],
    }
    monkeypatch.setattr(
        offline_figure_2,
        "load_figure_2_campaign",
        lambda *_args, **_kwargs: (run_dir, campaign, sessions),
    )
    monkeypatch.setattr(
        figure,
        "_build_panel_data",
        lambda *_args, **_kwargs: {
            "swap_delta": pd.DataFrame(),
            "swap_additive_delta": pd.DataFrame(),
            "swap_examples": [],
        },
    )

    payload = figure.load_figure_3_payload(run_id="run", scratch_root=tmp_path)

    assert payload["run_dir"] == run_dir
    assert payload["sessions"] == sessions
    campaign["sessions"][0]["status"] = "in_progress"
    with pytest.raises(ValueError, match="every Figure 2 session to be complete"):
        figure.load_figure_3_payload(run_id="run", scratch_root=tmp_path)
