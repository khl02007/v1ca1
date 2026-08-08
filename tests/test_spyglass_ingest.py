from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_spyglass_package_import_is_passive() -> None:
    sys.modules.pop("v1ca1.spyglass", None)
    sys.modules.pop("v1ca1.spyglass.tables", None)

    import v1ca1.spyglass  # noqa: F401

    assert "v1ca1.spyglass.tables" not in sys.modules


def test_normalize_catalog_requires_every_source_group() -> None:
    from v1ca1.spyglass.ingest import _normalize_catalog

    with pytest.raises(ValueError, match="missing source groups"):
        _normalize_catalog({"epoch_intervals": []})


def test_insert_catalog_rows_uses_explicit_tables() -> None:
    from v1ca1.spyglass.ingest import SOURCE_TABLE_KEYS, _insert_catalog_rows

    class FakeTable:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, object]], bool]] = []

        def insert(self, rows, *, skip_duplicates: bool) -> None:
            self.calls.append((rows, skip_duplicates))

    tables = {key: FakeTable() for key in SOURCE_TABLE_KEYS}
    catalog = {key: [] for key in SOURCE_TABLE_KEYS}
    catalog["epoch_intervals"] = [{"nwb_file_name": "test.nwb", "epoch": "01_s1"}]

    counts = _insert_catalog_rows(catalog, tables, skip_duplicates=True)

    assert counts["epoch_intervals"] == 1
    assert tables["epoch_intervals"].calls == [(catalog["epoch_intervals"], True)]
    assert tables["position"].calls == []


def test_ingest_dry_run_never_activates_tables(monkeypatch) -> None:
    from v1ca1.spyglass import ingest as ingest_module

    catalog = {
        key: [] for key in ingest_module.NWB_CATALOG_KEY_BY_TABLE.values()
    }
    catalog["ripples"] = [{"nwb_file_name": "test.nwb", "epoch": "01_s1"}]
    fake_nwb_module = SimpleNamespace(
        catalog_augmented_nwb=lambda nwbfile, nwb_file_name: catalog
    )
    monkeypatch.setitem(sys.modules, "v1ca1.spyglass.nwb", fake_nwb_module)

    result = ingest_module.ingest_v1ca1_nwb(
        "test.nwb",
        nwbfile=object(),
        dry_run=True,
    )

    assert result["inserted"] is False
    assert result["counts"]["ripple_intervals"] == 1
    assert result["rows"]["ripple_intervals"] == catalog["ripples"]


def test_non_dry_ingestion_rejects_unverifiable_open_nwb_object() -> None:
    from v1ca1.spyglass.ingest import ingest_v1ca1_nwb

    with pytest.raises(ValueError, match="must reopen the path registered"):
        ingest_v1ca1_nwb(
            "test.nwb",
            nwbfile=object(),
            tables={},
        )
