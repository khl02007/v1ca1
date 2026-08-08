"""Explicit ingestion of V1-CA1 objects from an augmented NWB file."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    import pynwb


SOURCE_TABLE_KEYS = (
    "epoch_intervals",
    "trajectory_intervals",
    "ripple_intervals",
    "position",
    "wtrack_graph",
    "spike_sorting_figurl",
)

NWB_CATALOG_KEY_BY_TABLE = {
    "epoch_intervals": "epoch_intervals",
    "trajectory_intervals": "trajectory_intervals",
    "ripple_intervals": "ripples",
    "position": "position",
    "wtrack_graph": "wtrack_graph",
    "spike_sorting_figurl": "spike_sorting_figurl",
}


def _open_nwb_file(nwb_path: Path) -> AbstractContextManager["pynwb.NWBFile"]:
    """Open one NWB file read-only and keep its backing IO alive."""
    from contextlib import contextmanager

    import pynwb

    @contextmanager
    def _reader() -> Iterator["pynwb.NWBFile"]:
        with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
            yield io.read()

    return _reader()


def _resolve_spyglass_dependencies() -> tuple[Any, Any]:
    """Import standard Spyglass tables only when ingestion is requested."""
    from spyglass.common import Nwbfile, Session

    return Nwbfile, Session


def _require_standard_ingestion(nwb_file_name: str, session_table: Any) -> None:
    """Raise when the standard Spyglass Session row has not been ingested."""
    key = {"nwb_file_name": nwb_file_name}
    if len(session_table & key) != 1:
        raise ValueError(
            "Expected exactly one standard Spyglass Session row before V1-CA1 "
            f"ingestion for {nwb_file_name!r}."
        )


def _normalize_catalog(catalog: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Validate the catalog keys and copy rows into insertion-ready dictionaries."""
    missing = [
        catalog_key
        for catalog_key in NWB_CATALOG_KEY_BY_TABLE.values()
        if catalog_key not in catalog
    ]
    if missing:
        raise ValueError(f"Augmented NWB catalog is missing source groups: {missing!r}.")

    normalized: dict[str, list[dict[str, Any]]] = {}
    for table_key, catalog_key in NWB_CATALOG_KEY_BY_TABLE.items():
        rows = catalog[catalog_key]
        normalized[table_key] = [dict(row) for row in rows]
    return normalized


def _insert_catalog_rows(
    catalog: Mapping[str, list[dict[str, Any]]],
    tables: Mapping[str, Any],
    *,
    skip_duplicates: bool,
) -> dict[str, int]:
    """Insert an NWB catalog into an explicitly activated table bundle."""
    missing = [key for key in SOURCE_TABLE_KEYS if key not in tables]
    if missing:
        raise ValueError(f"Activated table bundle is missing source tables: {missing!r}.")

    connections = {
        id(connection): connection
        for table in tables.values()
        if (connection := getattr(table, "connection", None)) is not None
    }
    if len(connections) > 1:
        raise ValueError("All activated source tables must use one DataJoint connection.")
    connection = next(iter(connections.values()), None)
    transaction = (
        connection.transaction
        if connection is not None and hasattr(connection, "transaction")
        else nullcontext()
    )

    counts: dict[str, int] = {}
    with transaction:
        for table_key in SOURCE_TABLE_KEYS:
            rows = catalog[table_key]
            if rows:
                tables[table_key].insert(rows, skip_duplicates=skip_duplicates)
            counts[table_key] = len(rows)
    return counts


def ingest_v1ca1_nwb(
    nwb_file_name: str,
    *,
    nwb_path: Path | str | None = None,
    nwbfile: "pynwb.NWBFile | None" = None,
    tables: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    skip_duplicates: bool = True,
) -> dict[str, Any]:
    """Index project objects in an NWB file after standard Spyglass ingestion.

    The source tables store object pointers and metadata, not copies of the
    underlying interval or position arrays.  Passing ``dry_run=True`` returns
    the rows that would be inserted.  Supplying ``nwbfile`` and/or ``tables``
    supports dependency-free tests; ordinary use resolves the standard
    ``Nwbfile`` and ``Session`` tables lazily.
    """
    from v1ca1.spyglass.nwb import catalog_augmented_nwb

    if not nwb_file_name:
        raise ValueError("nwb_file_name must be a non-empty string.")
    if nwbfile is not None and nwb_path is not None:
        raise ValueError("Pass either nwbfile or nwb_path, not both.")

    nwb_table = None
    if nwbfile is None and nwb_path is None:
        nwb_table, session_table = _resolve_spyglass_dependencies()
        _require_standard_ingestion(nwb_file_name, session_table)
        nwb_path = Path(nwb_table.get_abs_path(nwb_file_name))
    elif not dry_run:
        if nwbfile is not None:
            raise ValueError(
                "Non-dry ingestion must reopen the path registered by standard "
                "Spyglass ingestion; pass nwb_file_name without nwbfile=."
            )
        nwb_table, session_table = _resolve_spyglass_dependencies()
        _require_standard_ingestion(nwb_file_name, session_table)
        registered_path = Path(nwb_table.get_abs_path(nwb_file_name)).resolve()
        supplied_path = Path(nwb_path).resolve()  # type: ignore[arg-type]
        if supplied_path != registered_path:
            raise ValueError(
                f"Supplied NWB path {supplied_path} does not match the path "
                f"registered by Spyglass for {nwb_file_name!r}: {registered_path}."
            )

    nwb_context = (
        nullcontext(nwbfile)
        if nwbfile is not None
        else _open_nwb_file(Path(nwb_path))  # type: ignore[arg-type]
    )
    with nwb_context as opened_nwbfile:
        catalog = _normalize_catalog(
            catalog_augmented_nwb(opened_nwbfile, nwb_file_name=nwb_file_name)
        )

    result: dict[str, Any] = {
        "nwb_file_name": nwb_file_name,
        "counts": {key: len(rows) for key, rows in catalog.items()},
        "rows": catalog,
        "inserted": False,
    }
    if dry_run:
        return result

    if tables is None:
        from v1ca1.spyglass.tables import activate

        tables = activate()

    result["counts"] = _insert_catalog_rows(
        catalog,
        tables,
        skip_duplicates=skip_duplicates,
    )
    result["inserted"] = True
    return result


__all__ = ["SOURCE_TABLE_KEYS", "ingest_v1ca1_nwb"]
