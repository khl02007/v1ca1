"""Write supplemental SQL for custom analysis NWB filepath records."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Sequence

from v1ca1.spyglass import table_specs


DEFAULT_PAPER_ID = "kyu_v1ca1"
DEFAULT_FILE_SUFFIX = "_analysis_external"
_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_$]+$")


def _validate_schema_name(value: str) -> str:
    """Return a schema name that is safe to interpolate into SQL."""
    schema_name = str(value)
    if not _SCHEMA_NAME_PATTERN.fullmatch(schema_name):
        raise ValueError(f"Invalid DataJoint schema name: {schema_name!r}.")
    return schema_name


def _fetch_analysis_file_hashes(
    connection: Any,
    relation: Any,
) -> tuple[uuid.UUID, ...]:
    """Fetch raw filepath UUIDs without asking DataJoint to resolve files."""
    query = (
        "SELECT HEX(`analysis_file_abs_path`) FROM "
        f"{relation.full_table_name}{relation.where_clause()} "
        "ORDER BY `analysis_file_abs_path`"
    )
    values = tuple(
        sorted(
            (
                uuid.UUID(hex=str(row[0]))
                for row in connection.query(query).fetchall()
            ),
            key=lambda value: value.hex,
        )
    )
    if not values:
        raise RuntimeError("The paper export contains no custom analysis NWB rows.")
    if len(values) != len(set(values)):
        raise RuntimeError("The paper export contains duplicate filepath UUIDs.")
    return values


def write_analysis_external_mysqldump(
    *,
    paper_id: str = DEFAULT_PAPER_ID,
    analysis_nwbfile_schema_name: str = (
        table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME
    ),
    file_suffix: str = DEFAULT_FILE_SUFFIX,
) -> tuple[Path, int]:
    """Write a restricted dump script for custom external-analysis rows."""
    import datajoint as dj
    from spyglass.common.common_usage import Export, ExportSelection
    from spyglass.settings import export_dir
    from spyglass.utils.sql_helper_fn import SQLDumpHelper

    paper_id = str(paper_id)
    schema_name = _validate_schema_name(analysis_nwbfile_schema_name)
    paper_relation = Export & {"paper_id": paper_id}
    if len(paper_relation) != 1:
        raise ValueError("paper_id must correspond to exactly one paper export.")
    export_key = paper_relation.fetch1("KEY")

    analysis_table_name = f"`{schema_name}`.`analysis_nwbfile`"
    table_relation = Export.Table & export_key & {
        "table_name": analysis_table_name
    }
    if len(table_relation) != 1:
        raise RuntimeError(
            "The paper export must contain exactly one custom "
            "AnalysisNwbfile restriction."
        )
    restriction = table_relation.fetch1("restriction")
    analysis_relation = (
        dj.FreeTable(dj.conn(), analysis_table_name) & restriction
    )
    hashes = _fetch_analysis_file_hashes(dj.conn(), analysis_relation)

    external_table_name = f"`{schema_name}`.`~external_analysis`"
    external_relation = dj.FreeTable(dj.conn(), external_table_name) & [
        {"hash": value} for value in hashes
    ]
    if len(external_relation) != len(hashes):
        raise RuntimeError(
            "Not every exported AnalysisNwbfile row has an external-analysis "
            "record."
        )

    versions = {
        str(value)
        for value in (ExportSelection & export_key).fetch("spyglass_version")
    }
    if len(versions) != 1:
        raise RuntimeError("The paper export must have one Spyglass version.")
    SQLDumpHelper(
        paper_id=paper_id,
        spyglass_version=versions.pop(),
    ).write_mysqldump(
        [external_relation],
        file_suffix=str(file_suffix),
    )
    script_path = (
        Path(export_dir)
        / paper_id
        / f"_ExportSQL_{paper_id}{file_suffix}.sh"
    )
    return script_path, len(hashes)


def _parser() -> argparse.ArgumentParser:
    """Build the supplemental SQL export command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID)
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    parser.add_argument("--file-suffix", default=DEFAULT_FILE_SUFFIX)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Write one supplemental dump script and report its row count."""
    args = _parser().parse_args(argv)
    script_path, row_count = write_analysis_external_mysqldump(
        paper_id=args.paper_id,
        analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
        file_suffix=args.file_suffix,
    )
    print(
        json.dumps(
            {
                "event": "analysis_external_dump_script_written",
                "paper_id": args.paper_id,
                "rows": row_count,
                "script": str(script_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "write_analysis_external_mysqldump"]
