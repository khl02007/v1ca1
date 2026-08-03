"""Deterministic identities and provenance snapshots for custom selections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any
import uuid


SELECTION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kyu.v1ca1/spyglass")


def _json_value(value: Any) -> Any:
    """Return one recursively JSON-compatible selection value."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize selection provenance with deterministic ordering."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def provenance_sha256(value: Any) -> str:
    """Return a SHA-256 digest for one canonical provenance payload."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def selection_uuid(analysis_name: str, payload: Mapping[str, Any]) -> uuid.UUID:
    """Return a table-specific UUIDv5 for one complete selection payload."""
    analysis_name = str(analysis_name).strip()
    if not analysis_name:
        raise ValueError("analysis_name must be non-empty.")
    table_namespace = uuid.uuid5(SELECTION_NAMESPACE, analysis_name)
    return uuid.uuid5(table_namespace, canonical_json(payload))


def normalized_strings(values: Sequence[Any], *, name: str) -> list[str]:
    """Return sorted unique non-empty strings for frozen provenance."""
    normalized = sorted(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} must not contain empty values.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique values.")
    return normalized


def unit_identity_sha256(unit_ids: Sequence[Mapping[str, Any]]) -> str:
    """Digest sorted persistent ``(merge_id, unit_id)`` identities."""
    identities = sorted(
        (
            str(unit_id["spikesorting_merge_id"]),
            str(unit_id["unit_id"]),
        )
        for unit_id in unit_ids
    )
    if len(identities) != len(set(identities)):
        raise ValueError("Unit identities must be unique.")
    return provenance_sha256(identities)


__all__ = [
    "SELECTION_NAMESPACE",
    "canonical_json",
    "normalized_strings",
    "provenance_sha256",
    "selection_uuid",
    "unit_identity_sha256",
]
