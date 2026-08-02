"""Passive entry points for the project-specific Spyglass pipeline.

Importing this package never activates a DataJoint schema or connects to a
database.  Call :func:`activate` explicitly in a configured Spyglass process.
"""

from __future__ import annotations

from typing import Any


def activate(*args: Any, **kwargs: Any) -> Any:
    """Activate and return the custom table bundle."""
    from v1ca1.spyglass.tables import activate as _activate

    return _activate(*args, **kwargs)


def ingest_v1ca1_nwb(*args: Any, **kwargs: Any) -> Any:
    """Index project data already stored in one ingested augmented NWB file."""
    from v1ca1.spyglass.ingest import ingest_v1ca1_nwb as _ingest_v1ca1_nwb

    return _ingest_v1ca1_nwb(*args, **kwargs)


__all__ = ["activate", "ingest_v1ca1_nwb"]
