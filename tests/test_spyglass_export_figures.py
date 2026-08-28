"""Tests for logged Spyglass manuscript-figure generation."""

from pathlib import Path

import numpy as np
import pytest

from v1ca1.spyglass import export_figures


class _FakeExportSelection:
    def __init__(self, events, *, export_id=0) -> None:
        self.events = events
        self.export_id = export_id

    def start_export(self, *, paper_id: str, analysis_id: str) -> None:
        self.events.append(("start", paper_id, analysis_id))
        self.export_id = 17

    def stop_export(self) -> None:
        self.events.append(("stop", self.export_id))
        self.export_id = 0


def test_generate_logged_figure_export_wraps_generation(tmp_path):
    events = []
    selection = _FakeExportSelection(events)

    def database_factory(**kwargs):
        events.append(("database", kwargs))
        return "database"

    def generator(database, **kwargs):
        assert selection.export_id == 17
        events.append(("generate", database, kwargs))
        return [tmp_path / "figure_1.svg"]

    outputs = export_figures.generate_logged_figure_export(
        output_dir=tmp_path,
        replace=True,
        export_selection=selection,
        database_factory=database_factory,
        generator=generator,
    )

    assert outputs == [tmp_path / "figure_1.svg"]
    assert [event[0] for event in events] == [
        "start",
        "database",
        "generate",
        "stop",
    ]
    assert events[0] == (
        "start",
        export_figures.DEFAULT_PAPER_ID,
        export_figures.DEFAULT_ANALYSIS_ID,
    )
    assert events[2][2]["output_formats"] == ("svg", "pdf", "png")
    assert events[2][2]["dpi"] == 600
    assert events[2][2]["replace"] is True


def test_generate_logged_figure_export_stops_after_failure(tmp_path):
    events = []
    selection = _FakeExportSelection(events)

    def fail(_database, **_kwargs):
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        export_figures.generate_logged_figure_export(
            output_dir=tmp_path,
            export_selection=selection,
            database_factory=lambda **_kwargs: "database",
            generator=fail,
        )

    assert events == [
        (
            "start",
            export_figures.DEFAULT_PAPER_ID,
            export_figures.DEFAULT_ANALYSIS_ID,
        ),
        ("stop", 17),
    ]
    assert selection.export_id == 0


def test_generate_logged_figure_export_rejects_active_export(tmp_path):
    selection = _FakeExportSelection([], export_id=9)

    with pytest.raises(RuntimeError, match="export 9 is already active"):
        export_figures.generate_logged_figure_export(
            output_dir=tmp_path,
            export_selection=selection,
        )


def test_parse_arguments_uses_export_defaults():
    args = export_figures.parse_arguments([])

    assert args.paper_id == "kyu_v1ca1"
    assert args.analysis_id == "manuscript_figures_v1"
    assert tuple(args.output_formats) == ("svg", "pdf", "png")
    assert args.dpi == 600


def test_multi_file_export_compatibility_normalizes_arrays(monkeypatch):
    from spyglass.utils.mixins.export import ExportMixin

    calls = []

    def parent_copy_to_common(_table, fnames=None):
        calls.append(fnames)

    monkeypatch.setattr(
        ExportMixin,
        "_parent_copy_to_common",
        parent_copy_to_common,
    )
    with export_figures._spyglass_multi_file_export_compatibility():
        ExportMixin._parent_copy_to_common(
            object(),
            np.asarray(["analysis_1.nwb", "analysis_2.nwb"]),
        )

    assert calls == [["analysis_1.nwb", "analysis_2.nwb"]]
    assert ExportMixin._parent_copy_to_common is parent_copy_to_common
