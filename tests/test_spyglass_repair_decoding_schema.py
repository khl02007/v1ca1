"""Tests for guarded decoder schema repair helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from v1ca1.spyglass.repair_decoding_schema import (
    _quarantine_files,
    validate_repair_audit,
)


def _audit(file_path: Path) -> dict:
    return {
        "read_only": True,
        "code_schema": {"all_safe": True},
        "live_schema": {
            "tables": [
                {
                    "table": "path_progression_decoding",
                    "live_primary_key": [
                        "path_progression_decoding_id",
                        "analysis_file_name",
                    ],
                    "analysis_file_in_primary_key": True,
                    "primary_key_matches_selection": False,
                },
                {
                    "table": "path_specific_place_decoding",
                    "live_primary_key": [
                        "path_specific_place_decoding_id",
                        "analysis_file_name",
                    ],
                    "analysis_file_in_primary_key": True,
                    "primary_key_matches_selection": False,
                },
                {
                    "table": "path_progression_decoding.transfer",
                    "live_primary_key": [
                        "path_progression_decoding_id",
                        "analysis_file_name",
                        "transfer_family",
                        "source_trajectory",
                        "target_trajectory",
                    ],
                    "analysis_file_in_primary_key": True,
                    "primary_key_matches_selection": False,
                },
            ]
        },
        "repair_preconditions": {
            "dataset_scope_isolated": True,
            "other_artifact_schema_issues": [],
        },
        "decoder_inventory": {
            "path_progression_decoding": {"result_count_global": 1}
        },
        "analysis_files": [
            {
                "analysis_file_name": file_path.name,
                "path": str(file_path),
                "exists": True,
                "registered": True,
                "project_references": [
                    {"table": "path_progression_decoding"},
                    {"table": "path_progression_decoding.transfer"},
                ],
            }
        ],
    }


def test_repair_audit_requires_exact_isolated_defect(tmp_path: Path) -> None:
    source = tmp_path / "analysis_TEST.nwb"
    source.write_bytes(b"analysis")
    audit = _audit(source)

    validate_repair_audit(audit)

    audit["repair_preconditions"]["dataset_scope_isolated"] = False
    with pytest.raises(RuntimeError, match="not isolated"):
        validate_repair_audit(audit)


def test_quarantine_copy_is_verified_and_source_is_preserved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "analysis_TEST.nwb"
    source.write_bytes(b"analysis")
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()

    records = _quarantine_files(_audit(source), quarantine_dir=quarantine)

    assert source.read_bytes() == b"analysis"
    assert (quarantine / "files" / source.name).read_bytes() == b"analysis"
    assert records[0]["size_bytes"] == len(b"analysis")
    assert len(records[0]["sha256"]) == 64
