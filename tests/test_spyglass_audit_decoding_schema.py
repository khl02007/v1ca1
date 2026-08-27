"""Tests for the read-only decoder schema audit."""

from __future__ import annotations

import json
from pathlib import Path

from v1ca1.spyglass.audit_decoding_schema import (
    analysis_nwb_result_table_keys,
    code_schema_audit,
    default_manifest_path,
    write_manifest,
)


def test_code_schema_audit_covers_every_analysis_nwb_relation() -> None:
    audit = code_schema_audit()

    assert len(analysis_nwb_result_table_keys()) == 17
    assert len(audit["tables"]) == 18
    assert audit["all_safe"] is True
    assert not any(
        row["analysis_file_in_primary_key"] for row in audit["tables"]
    )


def test_manifest_path_and_json_write_are_explicit(tmp_path: Path) -> None:
    path = default_manifest_path(
        analysis_root=tmp_path,
        animal_name="L14",
        date="20240611",
    )

    assert path == (
        tmp_path / "L14" / "20240611" / "spyglass_decoding_schema_audit.json"
    )
    assert write_manifest({"read_only": True}, path) == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "read_only": True
    }
