"""Audit live analysis-NWB keys before repairing the decoder tables."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from v1ca1.spyglass import table_specs
from v1ca1.spyglass.populate_figures import (
    DEFAULT_NWB_ROOT,
    _load_runtime,
    _raw_nwb_path,
    _standard_nwb_file_name,
    select_figure_datasets,
)


DEFAULT_ANALYSIS_ROOT = Path("/stelmo/kyu/analysis")
DECODER_TABLES = {
    "path_progression_decoding": "path_progression_decoding_id",
    "path_specific_place_decoding": "path_specific_place_decoding_id",
}
TRANSFER_PRIMARY_FIELDS = (
    "transfer_family",
    "source_trajectory",
    "target_trajectory",
)


def analysis_nwb_result_table_keys() -> tuple[str, ...]:
    """Return every project result definition linked to AnalysisNwbfile."""
    return tuple(
        name
        for name, definition in table_specs.TABLE_DEFINITIONS.items()
        if "-> AnalysisNwbfile" in definition
    )


def code_schema_audit() -> dict[str, Any]:
    """Check that every code-level analysis-file dependency is secondary."""
    rows = []
    for name in analysis_nwb_result_table_keys():
        definition = table_specs.TABLE_DEFINITIONS[name]
        primary, secondary = definition.split("---", maxsplit=1)
        rows.append(
            {
                "table": name,
                "analysis_file_in_primary_key": (
                    "-> AnalysisNwbfile" in primary
                ),
                "analysis_file_in_secondary_attributes": (
                    "-> AnalysisNwbfile" in secondary
                ),
            }
        )
    transfer_primary, transfer_secondary = (
        table_specs.PATH_PROGRESSION_DECODING_TRANSFER_DEFINITION.split(
            "---", maxsplit=1
        )
    )
    rows.append(
        {
            "table": "path_progression_decoding.transfer",
            "analysis_file_in_primary_key": (
                "-> AnalysisNwbfile" in transfer_primary
            ),
            "analysis_file_in_secondary_attributes": (
                "-> AnalysisNwbfile" in transfer_secondary
            ),
        }
    )
    return {
        "tables": rows,
        "all_safe": all(
            not row["analysis_file_in_primary_key"]
            and row["analysis_file_in_secondary_attributes"]
            for row in rows
        ),
    }


def _row_count(relation: Any) -> int:
    """Count rows after normalizing a table class to a relation."""
    return len(relation())


def live_schema_audit(tables: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every live artifact result key with its selection key."""
    rows = []
    for result_name in analysis_nwb_result_table_keys():
        result_table = tables[result_name]
        selection_table = tables[f"{result_name}_selection"]
        observed = tuple(str(name) for name in result_table.primary_key)
        expected = tuple(str(name) for name in selection_table.primary_key)
        rows.append(
            {
                "table": result_name,
                "expected_primary_key": expected,
                "live_primary_key": observed,
                "analysis_file_in_primary_key": (
                    "analysis_file_name" in observed
                ),
                "primary_key_matches_selection": observed == expected,
                "row_count": _row_count(result_table),
            }
        )

    progression_selection = tables["path_progression_decoding_selection"]
    transfer_table = tables["path_progression_decoding"].Transfer
    expected_transfer = (
        *tuple(str(name) for name in progression_selection.primary_key),
        *TRANSFER_PRIMARY_FIELDS,
    )
    observed_transfer = tuple(
        str(name) for name in transfer_table.primary_key
    )
    rows.append(
        {
            "table": "path_progression_decoding.transfer",
            "expected_primary_key": expected_transfer,
            "live_primary_key": observed_transfer,
            "analysis_file_in_primary_key": (
                "analysis_file_name" in observed_transfer
            ),
            "primary_key_matches_selection": (
                observed_transfer == expected_transfer
            ),
            "row_count": _row_count(transfer_table),
        }
    )
    return {
        "tables": rows,
        "all_safe": all(
            not row["analysis_file_in_primary_key"]
            and row["primary_key_matches_selection"]
            for row in rows
        ),
    }


def _fetch_rows(relation: Any, fields: Sequence[str]) -> list[dict[str, Any]]:
    """Fetch selected fields from one relation as ordinary dictionaries."""
    return [
        dict(row)
        for row in relation.fetch(*tuple(fields), as_dict=True)
    ]


def _decoder_audit(
    tables: Mapping[str, Any],
    *,
    nwb_file_name: str,
) -> dict[str, Any]:
    """Inventory decoder selections, results, and transfer rows."""
    movement_for_session = (
        tables["movement_firing_rate_selection"]
        & {"nwb_file_name": nwb_file_name}
    ).proj()
    audit = {}
    for result_name, id_field in DECODER_TABLES.items():
        selection_table = tables[f"{result_name}_selection"]
        result_table = tables[result_name]
        dataset_selections = selection_table & movement_for_session
        dataset_results = result_table & dataset_selections.proj()
        result_fields = (id_field, "analysis_file_name")
        entry = {
            "selection_count_global": _row_count(selection_table),
            "selection_count_dataset": len(dataset_selections),
            "result_count_global": _row_count(result_table),
            "result_count_dataset": len(dataset_results),
            "selection_ids_dataset": [
                str(value) for value in dataset_selections.fetch(id_field)
            ],
            "result_rows_global": _fetch_rows(result_table, result_fields),
            "result_rows_dataset": _fetch_rows(
                dataset_results, result_fields
            ),
        }
        if result_name == "path_progression_decoding":
            transfer_table = result_table.Transfer
            transfer_fields = (
                id_field,
                "analysis_file_name",
                *TRANSFER_PRIMARY_FIELDS,
            )
            dataset_transfers = transfer_table & dataset_results.proj()
            entry.update(
                {
                    "transfer_count_global": _row_count(transfer_table),
                    "transfer_count_dataset": len(dataset_transfers),
                    "transfer_rows_global": _fetch_rows(
                        transfer_table, transfer_fields
                    ),
                    "transfer_rows_dataset": _fetch_rows(
                        dataset_transfers, transfer_fields
                    ),
                }
            )
        audit[result_name] = entry
    return audit


def _artifact_relations(
    tables: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    """Return all project relations that directly reference analysis files."""
    relations = [
        (name, tables[name]) for name in analysis_nwb_result_table_keys()
    ]
    relations.append(
        (
            "path_progression_decoding.transfer",
            tables["path_progression_decoding"].Transfer,
        )
    )
    return tuple(relations)


def _reference_counts(
    tables: Mapping[str, Any],
    analysis_file_names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Count project-table references to each candidate analysis file."""
    names = tuple(sorted(set(str(name) for name in analysis_file_names)))
    references: dict[str, list[dict[str, Any]]] = {
        name: [] for name in names
    }
    if not names:
        return references
    restriction = [{"analysis_file_name": name} for name in names]
    for table_name, table in _artifact_relations(tables):
        if "analysis_file_name" not in tuple(table.heading.names):
            continue
        counts = Counter(
            str(value)
            for value in (table & restriction).fetch("analysis_file_name")
        )
        for name, count in sorted(counts.items()):
            references[name].append(
                {"table": table_name, "row_count": int(count)}
            )
    return references


def _analysis_file_audit(
    tables: Mapping[str, Any],
    analysis_file_names: Sequence[str],
    *,
    analysis_dir: Path,
) -> list[dict[str, Any]]:
    """Resolve registry state, paths, and references for candidate files."""
    names = tuple(sorted(set(str(name) for name in analysis_file_names)))
    references = _reference_counts(tables, names)
    analysis_table = tables["analysis_nwbfile"]
    rows = []
    for name in names:
        registry = analysis_table & {"analysis_file_name": name}
        flat_path = Path(analysis_dir) / name
        nested_path = (
            Path(analysis_dir) / name[: name.rfind("_")] / name
        )
        path = flat_path if flat_path.exists() else nested_path
        rows.append(
            {
                "analysis_file_name": name,
                "registered": bool(registry),
                "registry_row_count": len(registry),
                "nwb_file_names": [
                    str(value) for value in registry.fetch("nwb_file_name")
                ],
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "project_references": references[name],
            }
        )
    return rows


def audit_decoding_schema(
    *,
    animal_name: str,
    date: str,
    candidate_analysis_file_names: Sequence[str] = (),
    nwb_root: Path = DEFAULT_NWB_ROOT,
    schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
    analysis_nwbfile_schema_name: str = (
        table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME
    ),
) -> dict[str, Any]:
    """Build a read-only live-schema and decoder-row repair manifest."""
    spec = select_figure_datasets(
        animal_name=animal_name,
        date=date,
        all_datasets=False,
    )[0]
    runtime = _load_runtime(
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
    )
    nwb_file_name = _standard_nwb_file_name(
        runtime,
        _raw_nwb_path(spec, nwb_root=Path(nwb_root)),
    )
    tables = runtime["tables"]
    from spyglass.settings import analysis_dir

    decoder = _decoder_audit(tables, nwb_file_name=nwb_file_name)
    result_file_names = {
        str(row["analysis_file_name"])
        for entry in decoder.values()
        for row in entry["result_rows_global"]
    }
    analysis_files = _analysis_file_audit(
        tables,
        (*result_file_names, *candidate_analysis_file_names),
        analysis_dir=Path(analysis_dir),
    )
    non_dataset_rows = {
        name: int(entry["result_count_global"])
        - int(entry["result_count_dataset"])
        for name, entry in decoder.items()
    }
    live_schema = live_schema_audit(tables)
    other_schema_issues = [
        row["table"]
        for row in live_schema["tables"]
        if row["table"]
        not in {
            "path_progression_decoding",
            "path_progression_decoding.transfer",
            "path_specific_place_decoding",
        }
        and (
            row["analysis_file_in_primary_key"]
            or not row["primary_key_matches_selection"]
        )
    ]
    return {
        "audit_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "dataset": {
            "animal_name": str(animal_name),
            "date": str(date),
            "nwb_file_name": nwb_file_name,
        },
        "schemas": {
            "project": schema_name,
            "analysis_nwbfile": analysis_nwbfile_schema_name,
        },
        "code_schema": code_schema_audit(),
        "live_schema": live_schema,
        "decoder_inventory": decoder,
        "analysis_files": analysis_files,
        "repair_preconditions": {
            "non_dataset_decoder_result_rows": non_dataset_rows,
            "other_artifact_schema_issues": other_schema_issues,
            "dataset_scope_isolated": (
                not any(non_dataset_rows.values())
                and not other_schema_issues
            ),
            "would_recreate": (
                "path_progression_decoding.transfer",
                "path_progression_decoding",
                "path_specific_place_decoding",
            ),
            "would_preserve": (
                "decoder parameter rows",
                "decoder selection rows",
                "all non-decoder result rows",
            ),
        },
    }


def default_manifest_path(
    *,
    analysis_root: Path,
    animal_name: str,
    date: str,
) -> Path:
    """Return the conventional per-session decoder audit path."""
    return (
        Path(analysis_root)
        / str(animal_name)
        / str(date)
        / "spyglass_decoding_schema_audit.json"
    )


def write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write one deterministic JSON audit manifest."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    """Build the read-only decoder audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--candidate-analysis-file-name",
        action="append",
        default=[],
    )
    parser.add_argument("--nwb-root", type=Path, default=DEFAULT_NWB_ROOT)
    parser.add_argument(
        "--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument(
        "--schema-name", default=table_specs.DEFAULT_SCHEMA_NAME
    )
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Audit one configured dataset without changing files or database rows."""
    args = _parser().parse_args(argv)
    manifest = audit_decoding_schema(
        animal_name=args.animal_name,
        date=args.date,
        candidate_analysis_file_names=args.candidate_analysis_file_name,
        nwb_root=args.nwb_root,
        schema_name=args.schema_name,
        analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
    )
    manifest_path = args.manifest_path or default_manifest_path(
        analysis_root=args.analysis_root,
        animal_name=args.animal_name,
        date=args.date,
    )
    written = write_manifest(manifest, manifest_path)
    print(
        json.dumps(
            {
                "event": "decoding_schema_audit_complete",
                "manifest_path": str(written),
                "live_schema_safe": manifest["live_schema"]["all_safe"],
                **manifest["repair_preconditions"],
            },
            default=str,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "analysis_nwb_result_table_keys",
    "audit_decoding_schema",
    "code_schema_audit",
    "default_manifest_path",
    "live_schema_audit",
    "main",
    "write_manifest",
]
