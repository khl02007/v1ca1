"""Tests for the Spyglass Supplementary Figure 2 adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from v1ca1.paper_figures import supplementary_figure_2_spyglass as figure


def _payload(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "campaign": {"run_id": run_dir.name},
        "datasets": figure.EXPECTED_DATASETS,
        "regions": (figure.REGION,),
        "cv_pca_table": pd.DataFrame({"condition": ["dark", "light"]}),
        "decoding_data": {
            "decoding_error": pd.DataFrame({"value": [1.0]}),
            "decoding_significance_labels": {"L12": "*"},
        },
    }


def test_cv_pca_bundle_maps_to_current_paired_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = {
        "animal_name": "L12",
        "date": "20240421",
        "epochs": {"dark": "08_r4"},
        "artifacts": {
            "cv_pca": [
                {
                    "region": figure.REGION,
                    "dark_epoch": "08_r4",
                    "light_epoch": figure.figure_2_adapter.LIGHT_EPOCH,
                }
            ]
        },
    }
    monkeypatch.setattr(
        figure,
        "_artifact_manifest_path",
        lambda *_args, **_kwargs: tmp_path / "cv" / "manifest.parquet",
    )
    monkeypatch.setattr(
        figure.cv_pca,
        "load_cv_pca_artifact",
        lambda _path: {
            "animal_name": "L12",
            "date": "20240421",
            "region": figure.REGION,
            "dark_epoch": "08_r4",
            "light_epoch": figure.figure_2_adapter.LIGHT_EPOCH,
            "artifact_origin": "computed",
            "summary": pd.DataFrame(
                {
                    "condition": ["dark", "light"],
                    "within_cv_participation_ratio": [2.0, 3.0],
                    "n_units": [11, 11],
                }
            ),
        },
    )

    table = figure._build_cv_pca_table(tmp_path, [session])

    assert table["condition"].tolist() == ["dark", "light"]
    assert table["participation_ratio"].tolist() == [2.0, 3.0]
    assert table["n_units"].tolist() == [11, 11]


def test_decoding_builder_uses_fixed_permutation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = pd.DataFrame({"summary": [1]})
    trial = pd.DataFrame({"trial": [2]})
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        figure.figure_2_adapter,
        "_build_panel_e_decoding_tables",
        lambda *_args, **_kwargs: (summary, trial),
    )

    def permutations(_table: Any, *, n_permutations: int, seed: int) -> object:
        calls.append((n_permutations, seed))
        return object()

    monkeypatch.setattr(
        figure.canonical.figure_2,
        "compute_panel_e_decoding_permutation_tests",
        permutations,
    )
    monkeypatch.setattr(
        figure.canonical.figure_2,
        "build_panel_e_decoding_significance_labels",
        lambda _result, *, animal_names: {name: "ns" for name in animal_names},
    )

    output = figure._build_decoding_data(tmp_path, [], scratch_root=tmp_path)

    assert output["decoding_error"] is summary
    assert output["decoding_trial_error"] is trial
    assert calls == [
        (
            figure.canonical.figure_2.DECODING_PERMUTATION_COUNT,
            figure.canonical.figure_2.DECODING_PERMUTATION_SEED,
        )
    ]


def test_context_and_atomic_render_use_only_two_payloads(
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
    original_decoding = figure.canonical.load_panel_a_decoding_data
    original_cv = (
        figure.canonical.supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table
    )

    with figure._offline_sources(payload):
        assert figure.canonical.load_panel_a_decoding_data(
            data_root=run_dir,
            datasets=payload["datasets"],
            region=figure.REGION,
            light_epoch=None,
            dark_epoch=None,
            decoding_n_permutations=(
                figure.canonical.figure_2.DECODING_PERMUTATION_COUNT
            ),
            decoding_permutation_seed=(
                figure.canonical.figure_2.DECODING_PERMUTATION_SEED
            ),
        ) is payload["decoding_data"]
        assert (
            figure.canonical.supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table(
                data_root=run_dir,
                datasets=payload["datasets"],
            )
            is payload["cv_pca_table"]
        )
    assert figure.canonical.load_panel_a_decoding_data is original_decoding
    assert (
        figure.canonical.supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table
        is original_cv
    )

    def render(**kwargs: Any) -> Path:
        assert figure.canonical.load_panel_a_decoding_data(
            data_root=kwargs["data_root"],
            datasets=kwargs["datasets"],
            region=kwargs["region"],
            light_epoch=kwargs["light_epoch"],
            dark_epoch=kwargs["dark_epoch"],
            decoding_n_permutations=kwargs["decoding_n_permutations"],
            decoding_permutation_seed=kwargs["decoding_permutation_seed"],
        ) is payload["decoding_data"]
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("supp2", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(figure.canonical, "make_supplementary_figure_2", render)
    output = figure.get_output_path(run_dir=run_dir)
    assert figure.render_supplementary_figure_2(
        payload,
        output_path=output,
    ) == output
    assert figure.get_figure_provenance_path(output).is_file()
