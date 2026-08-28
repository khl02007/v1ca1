"""Log and regenerate manuscript figures for a Spyglass paper export."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from v1ca1.paper_figures._spyglass_database import SpyglassFigureDatabase
from v1ca1.paper_figures.generate_spyglass_figures import (
    DEFAULT_OUTPUT_DIR,
    FIGURE_NAMES,
    generate_spyglass_figures,
)
from v1ca1.spyglass import table_specs
from v1ca1.spyglass.populate_figures import DEFAULT_NWB_ROOT


DEFAULT_PAPER_ID = "kyu_v1ca1"
DEFAULT_ANALYSIS_ID = "manuscript_figures_v1"
DEFAULT_OUTPUT_FORMATS = ("svg", "pdf", "png")
DEFAULT_DPI = 600


def _emit(event: str, **values: Any) -> None:
    """Print one machine-readable export progress record."""
    print(
        json.dumps({"event": event, **values}, default=str, sort_keys=True),
        flush=True,
    )


def _validate_identifier(value: str, *, name: str) -> str:
    """Validate one identifier against the Spyglass export table definition."""
    identifier = str(value)
    if not identifier or identifier.isspace():
        raise ValueError(f"{name} must not be empty.")
    if len(identifier) > 32:
        raise ValueError(f"{name} must contain at most 32 characters.")
    return identifier


@contextmanager
def _spyglass_multi_file_export_compatibility():
    """Normalize multi-file fetch results for the pinned Spyglass exporter."""
    from spyglass.utils.mixins.export import ExportMixin

    original = ExportMixin._parent_copy_to_common

    def parent_copy_to_common(table, fnames=None):
        if fnames is not None and not isinstance(fnames, list):
            if isinstance(fnames, str):
                fnames = [fnames]
            else:
                try:
                    fnames = list(fnames)
                except TypeError:
                    fnames = [fnames]
        return original(table, fnames=fnames)

    ExportMixin._parent_copy_to_common = parent_copy_to_common
    try:
        yield
    finally:
        ExportMixin._parent_copy_to_common = original


def generate_logged_figure_export(
    *,
    paper_id: str = DEFAULT_PAPER_ID,
    analysis_id: str = DEFAULT_ANALYSIS_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_names: Sequence[str] = FIGURE_NAMES,
    output_formats: Sequence[str] = DEFAULT_OUTPUT_FORMATS,
    dpi: int = DEFAULT_DPI,
    replace: bool = False,
    schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
    analysis_nwbfile_schema_name: str = (
        table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME
    ),
    nwb_root: Path = DEFAULT_NWB_ROOT,
    export_selection: Any = None,
    database_factory: Callable[..., SpyglassFigureDatabase] = (
        SpyglassFigureDatabase
    ),
    generator: Callable[..., list[Path]] = generate_spyglass_figures,
) -> list[Path]:
    """Generate figures while Spyglass records fetched rows and NWB files."""
    paper_id = _validate_identifier(paper_id, name="paper_id")
    analysis_id = _validate_identifier(analysis_id, name="analysis_id")
    if export_selection is None:
        from spyglass.common.common_usage import ExportSelection

        export_selection = ExportSelection()
    if export_selection.export_id:
        raise RuntimeError(
            f"Spyglass export {export_selection.export_id} is already active."
        )

    with _spyglass_multi_file_export_compatibility():
        started = False
        export_id = None
        try:
            export_selection.start_export(
                paper_id=paper_id,
                analysis_id=analysis_id,
            )
            started = True
            export_id = int(export_selection.export_id)
            _emit(
                "export_logging_started",
                export_id=export_id,
                paper_id=paper_id,
                analysis_id=analysis_id,
            )
            database = database_factory(
                schema_name=str(schema_name),
                analysis_nwbfile_schema_name=str(
                    analysis_nwbfile_schema_name
                ),
                nwb_root=Path(nwb_root),
            )
            outputs = generator(
                database,
                output_dir=Path(output_dir),
                figure_names=tuple(figure_names),
                output_formats=tuple(output_formats),
                dpi=int(dpi),
                replace=bool(replace),
            )
        except BaseException as error:
            _emit(
                "export_generation_failed",
                export_id=export_id,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            if started:
                export_selection.stop_export()
                _emit("export_logging_stopped", export_id=export_id)

    _emit(
        "export_generation_complete",
        export_id=export_id,
        outputs=len(outputs),
    )
    return outputs


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one logged figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID)
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=FIGURE_NAMES,
        default=FIGURE_NAMES,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-formats",
        nargs="+",
        choices=DEFAULT_OUTPUT_FORMATS,
        default=DEFAULT_OUTPUT_FORMATS,
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--schema-name",
        default=table_specs.DEFAULT_SCHEMA_NAME,
    )
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    parser.add_argument("--nwb-root", type=Path, default=DEFAULT_NWB_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one logged Spyglass manuscript-figure generation."""
    args = parse_arguments(argv)
    generate_logged_figure_export(
        paper_id=args.paper_id,
        analysis_id=args.analysis_id,
        output_dir=args.output_dir,
        figure_names=args.figures,
        output_formats=args.output_formats,
        dpi=args.dpi,
        replace=args.replace,
        schema_name=args.schema_name,
        analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
        nwb_root=args.nwb_root,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_ANALYSIS_ID",
    "DEFAULT_DPI",
    "DEFAULT_OUTPUT_FORMATS",
    "DEFAULT_PAPER_ID",
    "generate_logged_figure_export",
    "main",
]
