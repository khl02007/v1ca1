"""Tests for the database-backed manuscript figure command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from v1ca1.paper_figures import generate_spyglass_figures as figures


def test_parse_arguments_accepts_multiple_output_formats() -> None:
    args = figures.parse_arguments(
        ["--output-formats", "svg", "pdf", "png", "--dpi", "600"]
    )

    assert args.output_format == figures.DEFAULT_OUTPUT_FORMAT
    assert args.output_formats == ["svg", "pdf", "png"]
    assert args.dpi == 600


def test_generate_renders_all_formats_from_one_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []

    def build_payload(_database, *, run_dir: Path):
        calls.append(("build", 0))
        return {"run_dir": run_dir}

    def render(_payload, output_path: Path, *, dpi: int) -> Path:
        calls.append((output_path.suffix, dpi))
        output_path.write_text(f"{output_path.suffix}:{dpi}\n")
        return output_path

    monkeypatch.setattr(
        figures,
        "build_supplementary_figure_5_payload",
        build_payload,
    )
    monkeypatch.setattr(
        figures,
        "_render_supplementary_figure_5",
        render,
    )
    database = SimpleNamespace(
        schema_name="test_schema",
        analysis_nwbfile_schema_name="test_nwbfile_schema",
        nwb_root=Path("/nwb"),
        selected_rows=lambda: [{"table": "result", "result_id": "1"}],
    )

    outputs = figures.generate_spyglass_figures(
        database,
        output_dir=tmp_path,
        figure_names=("supplementary_figure_5",),
        output_formats=("svg", "pdf", "png"),
        dpi=600,
    )

    assert [path.suffix for path in outputs] == [".svg", ".pdf", ".png"]
    assert calls == [
        ("build", 0),
        (".svg", 600),
        (".pdf", 600),
        (".png", 600),
    ]
    manifest = json.loads(
        (tmp_path / "spyglass_figure_generation.json").read_text()
    )
    assert manifest["dpi"] == 600
    assert manifest["output_formats"] == ["svg", "pdf", "png"]
    assert [row["format"] for row in manifest["figures"]] == [
        "svg",
        "pdf",
        "png",
    ]
    assert manifest["result_rows"] == [
        {"table": "result", "result_id": "1"}
    ]
