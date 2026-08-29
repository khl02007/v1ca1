"""Validate files logged for a Spyglass paper export against DANDI."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import inspect
from typing import Any


DEFAULT_PAPER_ID = "kyu_v1ca1"
DEFAULT_PROCESSES = 8


@contextmanager
def _spyglass_dandi_validation_compatibility() -> Iterator[bool]:
    """Apply Spyglass's pending warning-table fix when the buggy code is installed."""
    from spyglass.common import common_dandi

    validation_table = common_dandi.DandiValidation
    original_make = validation_table.make
    make_source = inspect.getsource(original_make)
    fixed_markers = (
        'f"{result_map[\'prefix\']}_id"',
        "result_inserts.items()",
    )
    if all(marker in make_source for marker in fixed_markers):
        yield False
        return

    buggy_markers = (
        '"violation_id": i',
        "result_inserts.append(",
        "for part_table, part_keys in result_inserts:",
    )
    if not all(marker in make_source for marker in buggy_markers):
        raise RuntimeError(
            "The installed Spyglass DANDI validator is neither the supported "
            "buggy implementation nor its known fixed implementation."
        )

    def make(self: Any, key: dict[str, Any]) -> None:
        file_path = (common_dandi.Export.File() & key).fetch1("file_path")
        validator_result = list(common_dandi.dandi.validate.validate(file_path))
        results_maps = [
            {
                "table": self.Violations,
                "min_severity": common_dandi.MIN_ERROR_SEVERITY,
                "max_severity": None,
                "prefix": "violation",
            },
            {
                "table": self.Warnings,
                "min_severity": common_dandi.MIN_WARNING_SEVERITY,
                "max_severity": common_dandi.MIN_ERROR_SEVERITY,
                "prefix": "warning",
            },
        ]
        result_inserts = {}
        for result_map in results_maps:
            min_severity_value = result_map["min_severity"]
            max_severity_value = result_map["max_severity"]
            filtered_results = [
                result
                for result in validator_result
                if result.severity is not None
                and result.severity.value >= min_severity_value
                and (
                    max_severity_value is None
                    or result.severity.value < max_severity_value
                )
                and result.id != "DANDI.NO_DANDISET_FOUND"
            ]
            part_keys = [
                {
                    **key,
                    f"{result_map['prefix']}_id": i,
                    "id": result.id[:128],
                    "message": result.message[:255]
                    .replace("'", "")
                    .encode("ascii", "ignore")
                    .decode(),
                    "full_error": str(result).replace("'", "''"),
                    "file_path": file_path,
                }
                for i, result in enumerate(filtered_results)
            ]
            result_inserts[result_map["table"]] = part_keys

        self.insert1(key)
        for part_table, part_keys in result_inserts.items():
            part_table.insert(part_keys)

    validation_table.make = make
    try:
        yield True
    finally:
        validation_table.make = original_make


def validate_dandi_export(
    *,
    paper_id: str = DEFAULT_PAPER_ID,
    processes: int = DEFAULT_PROCESSES,
    force: bool = False,
) -> dict[str, int]:
    """Validate every file associated with one populated paper export."""
    if processes < 1:
        raise ValueError("processes must be at least 1.")

    from spyglass.common.common_dandi import (
        DandiValidation,
        DandiValidationSelection,
    )
    from spyglass.common.common_usage import Export

    paper_key = {"paper_id": str(paper_id)}
    if len(Export & paper_key) != 1:
        raise ValueError("paper_id must correspond to exactly one paper export.")
    export_key = (Export & paper_key).fetch1("KEY")
    file_count = len(Export.File & export_key)

    print(
        f"Validating {file_count} files for paper {paper_id!r} "
        f"with {processes} processes.",
        flush=True,
    )
    with _spyglass_dandi_validation_compatibility() as patched:
        if patched:
            print("Applied Spyglass DANDI validation compatibility fix.", flush=True)
        DandiValidationSelection().check_paper_for_dandi_errors(
            paper_key,
            force=force,
            n_processes=processes,
        )

    counts = {
        "files": file_count,
        "validated": len(DandiValidation & export_key),
        "violations": len(DandiValidation.Violations & export_key),
        "warnings": len(DandiValidation.Warnings & export_key),
    }
    if counts["validated"] != counts["files"]:
        raise RuntimeError(
            f"Validated {counts['validated']} of {counts['files']} exported files."
        )
    print(
        "DANDI validation complete: "
        f"{counts['validated']} files, {counts['violations']} violations, "
        f"{counts['warnings']} warnings.",
        flush=True,
    )
    return counts


def _parser() -> argparse.ArgumentParser:
    """Build the DANDI export-validation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID)
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Validate one Spyglass paper export against DANDI requirements."""
    args = _parser().parse_args(argv)
    validate_dandi_export(
        paper_id=args.paper_id,
        processes=args.processes,
        force=args.force,
    )


if __name__ == "__main__":
    main()


__all__ = ["validate_dandi_export", "main"]
