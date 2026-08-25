"""Tests for the Spyglass Supplementary Figure 6 adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.paper_figures import supplementary_figure_6_spyglass as figure


def _swap_dataset() -> xr.Dataset:
    """Return visual and scalar held-out scores for two paths and units."""
    models = np.asarray(["visual", figure.SCALAR_MODEL], dtype=str)
    trajectories = np.asarray(["center_to_left", "center_to_right"], dtype=str)
    units = np.asarray([11, 12], dtype=int)
    ll_sum = np.asarray(
        [
            [[10.0, 5.0], [4.0, 4.0]],
            [[8.0, 6.0], [4.0, 3.0]],
        ]
    )
    bits = np.asarray(
        [
            [[1.0, 0.5], [0.4, 0.4]],
            [[0.8, 0.6], [0.4, 0.3]],
        ]
    )
    return xr.Dataset(
        data_vars={
            figure.swap_glm.PRIMARY_METRIC: (
                ("model", "trajectory", "unit"),
                bits - bits[[0]],
            ),
            figure._RAW_LL_SUM: (
                ("model", "trajectory", "unit"),
                ll_sum,
            ),
            figure._RAW_LL_BITS_PER_SPIKE: (
                ("model", "trajectory", "unit"),
                bits,
            ),
            figure._TEST_BIN_COUNT: (("trajectory",), np.asarray([7, 8])),
            figure._SWAP_SEGMENT: (("trajectory",), np.asarray([3, 3])),
        },
        coords={"model": models, "trajectory": trajectories, "unit": units},
    )


def _empirical_summary() -> pd.DataFrame:
    """Return pointwise additive rows plus a deliberately better decoy model."""
    rows = []
    pointwise_ll = {
        ("center_to_left", 11): (9.0, 0.9, 7),
        ("center_to_left", 12): (4.0, 0.4, 7),
        ("center_to_right", 11): (5.0, 0.5, 8),
        ("center_to_right", 12): (4.0, 0.4, 8),
    }
    for model in (
        figure.EMPIRICAL_ADDITIVE_MODEL,
        "empirical_segment_additive_delta",
    ):
        for (trajectory, unit), (ll_sum, bits, n_bins) in pointwise_ll.items():
            if model != figure.EMPIRICAL_ADDITIVE_MODEL:
                ll_sum, bits = 100.0, 10.0
            rows.append(
                {
                    "animal_name": "L12",
                    "date": "20240421",
                    "region": figure.REGION,
                    "dark_train_epoch": "08_r4",
                    "light_train_epoch": figure.LIGHT_TRAIN_EPOCH,
                    "light_test_epoch": figure.LIGHT_TEST_EPOCH,
                    "trajectory": trajectory,
                    "unit": unit,
                    "model": model,
                    "ll_sum": ll_sum,
                    "ll_bits_per_spike": bits,
                    "test_light_bin_count": n_bins,
                    "score_qc_status": "valid",
                    "unit_valid": True,
                }
            )
    return pd.DataFrame.from_records(rows)


def _mixed_inputs(tmp_path: Path) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Return matched synthetic parent and empirical result dictionaries."""
    key = ("L12", "20240421")
    loaded = {
        key: {
            "dataset": _swap_dataset(),
            "metadata": {
                "dark_epoch": "08_r4",
                "light_train_epoch": figure.LIGHT_TRAIN_EPOCH,
                "light_test_epoch": figure.LIGHT_TEST_EPOCH,
            },
            "source_path": tmp_path / "swap-glm" / "manifest.parquet",
        }
    }
    empirical = {
        key: {
            "summary": _empirical_summary(),
            "source_path": tmp_path / "swap-tuning" / "manifest.parquet",
        }
    }
    return loaded, empirical


def _payload(run_dir: Path) -> dict[str, Any]:
    """Return a minimal already-adapted Supplementary Figure 6 payload."""
    return {
        "run_dir": run_dir,
        "campaign": {"run_id": run_dir.name},
        "datasets": figure.EXPECTED_DATASETS,
        "regions": (figure.REGION,),
        "scalar_swap_delta_table": pd.DataFrame({"kind": ["scalar"]}),
        "mixed_full_additive_table": pd.DataFrame({"kind": ["pointwise"]}),
    }


def test_adapter_uses_empirical_pointwise_additive_and_cli_defaults() -> None:
    assert figure.EMPIRICAL_ADDITIVE_MODEL == (
        "empirical_pointwise_additive_delta"
    )
    assert figure.EMPIRICAL_ADDITIVE_MODEL != figure.figure_3_adapter.ADDITIVE_MODEL

    args = figure.parse_arguments(["--run-id", "run"])

    assert args.run_id == "run"
    assert args.output_format == "svg"
    assert args.dpi == 300
    with pytest.raises(SystemExit):
        figure.parse_arguments(["--run-id", "run", "--replace-promoted-output"])


def test_empirical_identity_is_checked_and_mapped_to_sorting_units() -> None:
    selected = pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge", "merge"],
            "unit_id": [101, 102],
            "stable_unit_id": ["merge:101", "merge:102"],
            "group_unit_id": ["group:101", "group:102"],
        }
    )
    summary = pd.concat(
        [selected.iloc[[0]], selected.iloc[[1]], selected.iloc[[0]]],
        ignore_index=True,
    )
    result = {"selected_units": selected, "summary": summary}

    mapped = figure._map_empirical_summary(
        result,
        sorting_unit_by_nwb_id={"101": 11, "102": 12},
    )

    assert mapped["unit"].tolist() == [11, 12, 11]
    changed = summary.copy()
    changed.loc[0, "stable_unit_id"] = "wrong"
    with pytest.raises(ValueError, match="identity disagree"):
        figure._map_empirical_summary(
            {"selected_units": selected, "summary": changed},
            sorting_unit_by_nwb_id={"101": 11, "102": 12},
        )


def test_mixed_table_selects_pointwise_additive_not_decoy(tmp_path: Path) -> None:
    loaded, empirical = _mixed_inputs(tmp_path)
    summary = empirical[("L12", "20240421")]["summary"]
    summary.loc[
        summary["model"].eq(figure.EMPIRICAL_ADDITIVE_MODEL)
        & summary["trajectory"].eq("center_to_left")
        & summary["unit"].eq(11),
        "unit_valid",
    ] = False

    table = figure._build_mixed_full_additive_table(loaded, empirical)

    assert len(table) == 4
    assert table["winner"].tolist() == ["V", "MS", "A", "tie"]
    assert table["A_ll_sum"].tolist() == [9.0, 4.0, 5.0, 4.0]
    assert np.allclose(
        table["delta_V_minus_A_bits_per_spike"],
        [0.1, 0.1, -0.1, 0.0],
    )
    assert np.allclose(
        table["delta_V_minus_task_bits_per_spike"],
        [0.2, -0.1, 0.0, 0.1],
    )


def test_offline_context_patches_only_active_loaders_and_restores(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = _payload(run_dir)
    original_scalar = figure.canonical.load_panel_h_swap_delta_table
    original_mixed = figure.canonical.load_mixed_glm_full_additive_delta_table

    with figure._offline_sources(payload):
        scalar = figure.canonical.load_panel_h_swap_delta_table(
            data_root=run_dir,
            datasets=payload["datasets"],
            region=figure.REGION,
            dark_epoch=None,
            min_movement_firing_rate_hz=(
                figure.figure_2_adapter.MINIMUM_MOVEMENT_FIRING_RATE_HZ
            ),
            min_tuning_stability_correlation=(
                figure.figure_2_adapter.MINIMUM_STABILITY_CORRELATION
            ),
            model_name=figure.SCALAR_MODEL,
        )
        mixed = figure.canonical.load_mixed_glm_full_additive_delta_table(
            data_root=run_dir,
            datasets=payload["datasets"],
            region=figure.REGION,
            dark_epoch=None,
        )
        assert scalar is payload["scalar_swap_delta_table"]
        assert mixed is payload["mixed_full_additive_table"]

    assert figure.canonical.load_panel_h_swap_delta_table is original_scalar
    assert (
        figure.canonical.load_mixed_glm_full_additive_delta_table
        is original_mixed
    )


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
        scalar = figure.canonical.load_panel_h_swap_delta_table(
            data_root=kwargs["data_root"],
            datasets=kwargs["datasets"],
            region=kwargs["region"],
            dark_epoch=kwargs["dark_epoch"],
            min_movement_firing_rate_hz=(
                figure.figure_2_adapter.MINIMUM_MOVEMENT_FIRING_RATE_HZ
            ),
            min_tuning_stability_correlation=(
                figure.figure_2_adapter.MINIMUM_STABILITY_CORRELATION
            ),
            model_name=figure.SCALAR_MODEL,
        )
        mixed = figure.canonical.load_mixed_glm_full_additive_delta_table(
            data_root=kwargs["data_root"],
            datasets=kwargs["datasets"],
            region=kwargs["region"],
            dark_epoch=kwargs["dark_epoch"],
        )
        assert scalar is payload["scalar_swap_delta_table"]
        assert mixed is payload["mixed_full_additive_table"]
        kwargs["output_path"].write_text("supp4", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(figure.canonical, "make_supplementary_figure_6", render)

    assert figure.render_supplementary_figure_6(
        payload,
        output_path=output,
    ) == output
    assert output.read_text(encoding="utf-8") == "supp4"
    assert figure.get_figure_provenance_path(output).is_file()
    with pytest.raises(FileExistsError):
        figure.render_supplementary_figure_6(payload, output_path=output)
