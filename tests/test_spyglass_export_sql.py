from __future__ import annotations

import uuid

import pytest

from v1ca1.spyglass import export_sql


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows
        self.query_text = None

    def query(self, query):
        self.query_text = query
        return _FakeCursor(self._rows)


class _FakeRelation:
    full_table_name = "`schema`.`analysis_nwbfile`"

    @staticmethod
    def where_clause():
        return " WHERE (`analysis_file_name`='example.nwb')"


@pytest.mark.parametrize(
    "value",
    ["schema-name", "schema.name", "schema name", "`schema`", ""],
)
def test_validate_schema_name_rejects_unsafe_values(value):
    with pytest.raises(ValueError, match="Invalid DataJoint schema name"):
        export_sql._validate_schema_name(value)


def test_fetch_analysis_file_hashes_uses_raw_ordered_query():
    first = uuid.UUID("00000000-0000-0000-0000-000000000001")
    second = uuid.UUID("00000000-0000-0000-0000-000000000002")
    connection = _FakeConnection([(second.hex,), (first.hex,)])

    values = export_sql._fetch_analysis_file_hashes(
        connection,
        _FakeRelation(),
    )

    assert values == (first, second)
    assert connection.query_text == (
        "SELECT HEX(`analysis_file_abs_path`) FROM "
        "`schema`.`analysis_nwbfile` WHERE "
        "(`analysis_file_name`='example.nwb') "
        "ORDER BY `analysis_file_abs_path`"
    )


def test_fetch_analysis_file_hashes_rejects_duplicates():
    value = uuid.UUID("00000000-0000-0000-0000-000000000001")
    connection = _FakeConnection([(value.hex,), (value.hex,)])

    with pytest.raises(RuntimeError, match="duplicate filepath UUIDs"):
        export_sql._fetch_analysis_file_hashes(connection, _FakeRelation())
