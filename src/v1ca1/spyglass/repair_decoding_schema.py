"""Repair decoder result keys after quarantining runaway analysis NWBs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from v1ca1.spyglass import table_specs
from v1ca1.spyglass.audit_decoding_schema import (
    DEFAULT_ANALYSIS_ROOT,
    audit_decoding_schema,
    live_schema_audit,
    write_manifest,
)
from v1ca1.spyglass.populate_figures import DEFAULT_NWB_ROOT, _load_runtime


_AFFECTED_TABLES = {
    "path_progression_decoding": (
        "path_progression_decoding_id",
        "analysis_file_name",
    ),
    "path_specific_place_decoding": (
        "path_specific_place_decoding_id",
        "analysis_file_name",
    ),
    "path_progression_decoding.transfer": (
        "path_progression_decoding_id",
        "analysis_file_name",
        "transfer_family",
        "source_trajectory",
        "target_trajectory",
    ),
}
_ALLOWED_FILE_REFERENCE_TABLES = {
    "path_progression_decoding",
    "path_progression_decoding.transfer",
    "path_specific_place_decoding",
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_repair_audit(audit: Mapping[str, Any]) -> None:
    """Require an isolated, exact instance of the known decoder-key defect."""
    if not audit["read_only"] or not audit["code_schema"]["all_safe"]:
        raise RuntimeError("The code schema must be safe before live repair.")
    preconditions = audit["repair_preconditions"]
    if not preconditions["dataset_scope_isolated"]:
        raise RuntimeError("Decoder result rows are not isolated to this dataset.")
    if preconditions["other_artifact_schema_issues"]:
        raise RuntimeError("Other artifact schemas require separate review.")

    problems = {
        row["table"]: tuple(row["live_primary_key"])
        for row in audit["live_schema"]["tables"]
        if row["analysis_file_in_primary_key"]
        or not row["primary_key_matches_selection"]
    }
    if problems != _AFFECTED_TABLES:
        raise RuntimeError(
            f"Live decoder schema does not match the audited defect: {problems!r}."
        )

    registered_count = 0
    for file_row in audit["analysis_files"]:
        references = {
            row["table"] for row in file_row["project_references"]
        }
        if not references <= _ALLOWED_FILE_REFERENCE_TABLES:
            raise RuntimeError(
                f"{file_row['analysis_file_name']} has outside references."
            )
        if not file_row["exists"]:
            raise RuntimeError(
                f"{file_row['analysis_file_name']} is missing on disk."
            )
        if file_row["registered"]:
            registered_count += 1
            if references != {
                "path_progression_decoding",
                "path_progression_decoding.transfer",
            }:
                raise RuntimeError(
                    f"{file_row['analysis_file_name']} has unexpected references."
                )
        elif references:
            raise RuntimeError(
                f"Unregistered {file_row['analysis_file_name']} is referenced."
            )

    expected_registered = audit["decoder_inventory"][
        "path_progression_decoding"
    ]["result_count_global"]
    if registered_count != expected_registered:
        raise RuntimeError(
            f"Found {registered_count} registered files for "
            f"{expected_registered} progression results."
        )


def _dependency_preconditions(tables: Mapping[str, Any]) -> None:
    """Reject live dependents outside the three relations being recreated."""
    progression = tables["path_progression_decoding"]
    transfer = progression.Transfer
    place = tables["path_specific_place_decoding"]
    connection = progression.connection
    connection.dependencies.load()
    expected_progression = {
        progression.full_table_name,
        transfer.full_table_name,
    }
    observed_progression = {
        name
        for name in connection.dependencies.descendants(
            progression.full_table_name
        )
        if not name.isdigit()
    }
    if observed_progression != expected_progression:
        raise RuntimeError(
            "Unexpected PathProgressionDecoding dependents: "
            f"{sorted(observed_progression - expected_progression)!r}."
        )
    observed_place = {
        name
        for name in connection.dependencies.descendants(place.full_table_name)
        if not name.isdigit()
    }
    if observed_place != {place.full_table_name}:
        raise RuntimeError(
            "Unexpected PathSpecificPlaceDecoding dependents: "
            f"{sorted(observed_place - {place.full_table_name})!r}."
        )


def _quarantine_files(
    audit: Mapping[str, Any],
    *,
    quarantine_dir: Path,
) -> list[dict[str, Any]]:
    """Copy and hash every affected file before any database mutation."""
    files_dir = Path(quarantine_dir) / "files"
    files_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for file_row in audit["analysis_files"]:
        source = Path(file_row["path"])
        destination = files_dir / source.name
        if destination.exists():
            raise FileExistsError(destination)
        source_sha256 = _sha256(source)
        shutil.copy2(source, destination)
        destination_sha256 = _sha256(destination)
        if source_sha256 != destination_sha256:
            raise RuntimeError(f"Quarantine copy changed {source.name}.")
        records.append(
            {
                "analysis_file_name": source.name,
                "source": str(source),
                "quarantine": str(destination),
                "size_bytes": source.stat().st_size,
                "sha256": source_sha256,
                "registered": bool(file_row["registered"]),
            }
        )
    return records


def _recreate_decoder_results(
    runtime: Mapping[str, Any],
    *,
    registered_file_names: Sequence[str],
    schema_name: str,
    analysis_nwbfile_schema_name: str,
) -> Mapping[str, Any]:
    """Drop affected result relations, clear their registry rows, and redeclare."""
    tables = runtime["tables"]
    _dependency_preconditions(tables)
    analysis_table = tables["analysis_nwbfile"]
    restriction = [
        {"analysis_file_name": name} for name in registered_file_names
    ]
    registry_rows = analysis_table & restriction
    if len(registry_rows) != len(registered_file_names):
        raise RuntimeError("AnalysisNwbfile registry count changed before repair.")

    tables["path_progression_decoding"].Transfer().drop_quick()
    tables["path_progression_decoding"]().drop_quick()
    tables["path_specific_place_decoding"]().drop_quick()
    deleted = registry_rows.delete_quick(get_count=True)
    if deleted != len(registered_file_names):
        raise RuntimeError(
            f"Deleted {deleted} registry rows; expected "
            f"{len(registered_file_names)}."
        )

    from v1ca1.spyglass.tables import activate

    return activate(
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
        connection=runtime["dj"].conn(),
        create_schema=False,
        create_tables=True,
    )


def _verify_repair(
    tables: Mapping[str, Any],
    *,
    registered_file_names: Sequence[str],
    selection_counts: Mapping[str, int],
) -> None:
    """Require corrected empty results, preserved selections, and no registry rows."""
    live = live_schema_audit(tables)
    if not live["all_safe"]:
        raise RuntimeError("Recreated artifact table keys are still unsafe.")
    if len(tables["path_progression_decoding"]()) != 0:
        raise RuntimeError("PathProgressionDecoding was not recreated empty.")
    if len(tables["path_progression_decoding"].Transfer()) != 0:
        raise RuntimeError("PathProgressionDecoding.Transfer was not recreated empty.")
    if len(tables["path_specific_place_decoding"]()) != 0:
        raise RuntimeError("PathSpecificPlaceDecoding was not recreated empty.")
    for result_name, expected in selection_counts.items():
        observed = len(tables[f"{result_name}_selection"]())
        if observed != expected:
            raise RuntimeError(
                f"{result_name} selection count changed from {expected} "
                f"to {observed}."
            )
    registry = tables["analysis_nwbfile"] & [
        {"analysis_file_name": name} for name in registered_file_names
    ]
    if registry:
        raise RuntimeError("Affected analysis-file registry rows remain.")


def repair_decoding_schema(
    *,
    animal_name: str,
    date: str,
    candidate_analysis_file_names: Sequence[str],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
    analysis_nwbfile_schema_name: str = (
        table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME
    ),
) -> Path:
    """Execute the guarded decoder repair and return its quarantine directory."""
    before = audit_decoding_schema(
        animal_name=animal_name,
        date=date,
        candidate_analysis_file_names=candidate_analysis_file_names,
        nwb_root=nwb_root,
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
    )
    validate_repair_audit(before)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = (
        Path(analysis_root)
        / str(animal_name)
        / str(date)
        / "spyglass_decoder_quarantine"
        / timestamp
    )
    if quarantine_dir.exists():
        raise FileExistsError(quarantine_dir)
    quarantine_dir.mkdir(parents=True)
    write_manifest(before, quarantine_dir / "before.json")
    file_records = _quarantine_files(
        before,
        quarantine_dir=quarantine_dir,
    )
    write_manifest(
        {"phase": "files_verified", "files": file_records},
        quarantine_dir / "repair_state.json",
    )

    runtime = _load_runtime(
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
    )
    registered_names = tuple(
        row["analysis_file_name"]
        for row in before["analysis_files"]
        if row["registered"]
    )
    selection_counts = {
        name: int(entry["selection_count_global"])
        for name, entry in before["decoder_inventory"].items()
    }
    recreated = _recreate_decoder_results(
        runtime,
        registered_file_names=registered_names,
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
    )
    _verify_repair(
        recreated,
        registered_file_names=registered_names,
        selection_counts=selection_counts,
    )
    for record in file_records:
        source = Path(record["source"])
        if _sha256(Path(record["quarantine"])) != record["sha256"]:
            raise RuntimeError(f"Quarantine hash changed for {source.name}.")
        source.unlink()

    all_names = tuple(record["analysis_file_name"] for record in file_records)
    after = audit_decoding_schema(
        animal_name=animal_name,
        date=date,
        candidate_analysis_file_names=all_names,
        nwb_root=nwb_root,
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
    )
    if not after["live_schema"]["all_safe"]:
        raise RuntimeError("Final live schema audit failed.")
    if any(row["registered"] or row["exists"] for row in after["analysis_files"]):
        raise RuntimeError("Affected files remain registered or in the active store.")
    write_manifest(after, quarantine_dir / "after.json")
    write_manifest(
        {
            "phase": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": file_records,
        },
        quarantine_dir / "repair_state.json",
    )
    return quarantine_dir


def _parser() -> argparse.ArgumentParser:
    """Build the explicit destructive repair CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--candidate-analysis-file-name",
        action="append",
        default=[],
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--nwb-root", type=Path, default=DEFAULT_NWB_ROOT)
    parser.add_argument("--schema-name", default=table_specs.DEFAULT_SCHEMA_NAME)
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run only when ``--apply`` explicitly authorizes the live repair."""
    args = _parser().parse_args(argv)
    if not args.apply:
        raise SystemExit("Pass --apply to run the guarded live repair.")
    quarantine_dir = repair_decoding_schema(
        animal_name=args.animal_name,
        date=args.date,
        candidate_analysis_file_names=args.candidate_analysis_file_name,
        analysis_root=args.analysis_root,
        nwb_root=args.nwb_root,
        schema_name=args.schema_name,
        analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
    )
    print(
        json.dumps(
            {
                "event": "decoding_schema_repair_complete",
                "quarantine_dir": str(quarantine_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "repair_decoding_schema", "validate_repair_audit"]
