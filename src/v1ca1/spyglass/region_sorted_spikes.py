"""Immutable registration helpers for region-resolved sorting groups."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
from typing import Any
import uuid

from v1ca1.spyglass.selection import (
    normalized_strings,
    provenance_sha256,
    selection_uuid,
    unit_identity_sha256,
)
from v1ca1.spyglass.spikes import (
    SORTED_SPIKES_GROUP_KEY_FIELDS,
    load_sorted_spikes_group,
)


REGISTRATION_ID_FIELD = "region_sorted_spikes_group_id"
REGISTRATION_VALUE_FIELDS = (
    *SORTED_SPIKES_GROUP_KEY_FIELDS,
    "region_name",
    "sorting_group_members",
    "sorting_group_members_sha256",
    "unit_filter_include_labels",
    "unit_filter_exclude_labels",
    "unit_filter_params_sha256",
    "n_units",
    "selected_units_sha256",
)
REGISTRATION_FIELDS = (REGISTRATION_ID_FIELD, *REGISTRATION_VALUE_FIELDS)


def normalize_region(region: Any) -> str:
    """Return one non-empty, case-insensitive region label."""
    if region is None:
        raise ValueError("region must be a non-empty string.")
    normalized = str(region).strip().casefold()
    if not normalized:
        raise ValueError("region must be a non-empty string.")
    if len(normalized) > 64:
        raise ValueError("region must be at most 64 characters long.")
    return normalized


def _string_field(value: Any, *, name: str) -> str:
    """Return one non-empty string-valued registration field."""
    if value is None:
        raise ValueError(f"{name} must be a non-empty string.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized


def _sequence(value: Any, *, name: str) -> list[Any]:
    """Return a non-string sequence as a list."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a non-string sequence.")
    return list(value)


def _sorting_members_sha256(members: Sequence[str]) -> str:
    """Return the digest used by the standard sorting-group adapter."""
    return hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()


def _snapshot_from_adapter_output(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and reduce one ``load_sorted_spikes_group`` result."""
    if not isinstance(loaded_spikes, Mapping):
        raise TypeError("loaded_spikes must be a mapping.")

    group_key = loaded_spikes.get("group_key")
    if not isinstance(group_key, Mapping):
        raise ValueError("loaded_spikes must contain a mapping-valued group_key.")
    normalized_group_key = {
        field: _string_field(group_key.get(field), name=field)
        for field in SORTED_SPIKES_GROUP_KEY_FIELDS
    }

    members = normalized_strings(
        _sequence(
            loaded_spikes.get("sorting_group_members"),
            name="sorting_group_members",
        ),
        name="sorting_group_members",
    )
    if not members:
        raise ValueError("sorting_group_members must not be empty.")
    expected_members_sha256 = _sorting_members_sha256(members)
    if str(loaded_spikes.get("sorting_group_members_sha256", "")) != (
        expected_members_sha256
    ):
        raise ValueError(
            "sorting_group_members_sha256 does not match sorting_group_members."
        )

    unit_filter = loaded_spikes.get("unit_selection_params")
    if not isinstance(unit_filter, Mapping):
        raise ValueError(
            "loaded_spikes must contain mapping-valued unit_selection_params."
        )
    filter_name = _string_field(
        unit_filter.get("unit_filter_params_name"),
        name="unit_filter_params_name",
    )
    if filter_name != normalized_group_key["unit_filter_params_name"]:
        raise ValueError(
            "unit_selection_params name does not match the sorting group key."
        )
    include_labels = normalized_strings(
        _sequence(unit_filter.get("include_labels"), name="include_labels"),
        name="include_labels",
    )
    exclude_labels = normalized_strings(
        _sequence(unit_filter.get("exclude_labels"), name="exclude_labels"),
        name="exclude_labels",
    )
    filter_snapshot = {
        "unit_filter_params_name": filter_name,
        "include_labels": include_labels,
        "exclude_labels": exclude_labels,
    }
    expected_filter_sha256 = provenance_sha256(filter_snapshot)
    if str(loaded_spikes.get("unit_selection_params_sha256", "")) != (
        expected_filter_sha256
    ):
        raise ValueError(
            "unit_selection_params_sha256 does not match unit_selection_params."
        )

    unit_ids = _sequence(loaded_spikes.get("unit_ids"), name="unit_ids")
    if not all(isinstance(unit_id, Mapping) for unit_id in unit_ids):
        raise TypeError("unit_ids must contain only mappings.")
    raw_n_units = loaded_spikes.get("n_units")
    if isinstance(raw_n_units, bool):
        raise TypeError("n_units must be a non-negative integer.")
    try:
        n_units = int(raw_n_units)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("n_units must be a non-negative integer.") from exc
    if n_units < 0 or n_units != len(unit_ids) or raw_n_units != n_units:
        raise ValueError("n_units must equal the number of unit_ids.")

    return {
        **normalized_group_key,
        "region_name": normalize_region(loaded_spikes.get("region")),
        "sorting_group_members": members,
        "sorting_group_members_sha256": expected_members_sha256,
        "unit_filter_include_labels": include_labels,
        "unit_filter_exclude_labels": exclude_labels,
        "unit_filter_params_sha256": expected_filter_sha256,
        "n_units": n_units,
        "selected_units_sha256": unit_identity_sha256(unit_ids),
    }


def build_region_sorted_spikes_group_row(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic registration row from adapter output.

    The returned row freezes sorting membership, label filtering, and the
    selected unit identity. It intentionally contains neither a per-unit list
    nor spike-time arrays.
    """
    snapshot = _snapshot_from_adapter_output(loaded_spikes)
    return {
        REGISTRATION_ID_FIELD: selection_uuid(
            "RegionSortedSpikesGroup",
            snapshot,
        ),
        **snapshot,
    }


def _registration_value(
    row: Mapping[str, Any],
    *,
    field: str,
) -> Any:
    """Return a normalized value for comparing registration rows."""
    if field in SORTED_SPIKES_GROUP_KEY_FIELDS:
        return _string_field(row.get(field), name=field)
    if field == "region_name":
        return normalize_region(row.get(field))
    if field in {
        "sorting_group_members",
        "unit_filter_include_labels",
        "unit_filter_exclude_labels",
    }:
        return normalized_strings(
            _sequence(row.get(field), name=field),
            name=field,
        )
    if field == "n_units":
        value = row.get(field)
        if isinstance(value, bool):
            raise TypeError("n_units must be a non-negative integer.")
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("n_units must be a non-negative integer.") from exc
        if normalized < 0 or value != normalized:
            raise ValueError("n_units must be a non-negative integer.")
        return normalized
    return _string_field(row.get(field), name=field).casefold()


def validate_region_sorted_spikes_group_row(
    registration_row: Mapping[str, Any],
    loaded_spikes: Mapping[str, Any],
) -> None:
    """Require a stored registration to match freshly loaded source data."""
    if not isinstance(registration_row, Mapping):
        raise TypeError("registration_row must be a mapping.")
    current_row = build_region_sorted_spikes_group_row(loaded_spikes)
    for field in REGISTRATION_VALUE_FIELDS:
        registered = _registration_value(registration_row, field=field)
        current = _registration_value(current_row, field=field)
        if registered != current:
            raise ValueError(
                "RegionSortedSpikesGroup source changed after registration: "
                f"{field}. Create a new registration row."
            )

    try:
        registered_id = uuid.UUID(str(registration_row.get(REGISTRATION_ID_FIELD)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{REGISTRATION_ID_FIELD} must be a valid UUID."
        ) from exc
    if registered_id != current_row[REGISTRATION_ID_FIELD]:
        raise ValueError(
            "RegionSortedSpikesGroup registration UUID does not match its "
            "frozen source snapshot."
        )


def reload_region_sorted_spikes_group(
    registration_row: Mapping[str, Any],
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    region_validator: Callable[..., Any] | None = None,
    time_support: Any | tuple[float, float] | None = None,
    pynapple_module: Any | None = None,
) -> dict[str, Any]:
    """Reload a registered group and verify it against current sources."""
    if not isinstance(registration_row, Mapping):
        raise TypeError("registration_row must be a mapping.")
    group_key = {
        field: registration_row[field]
        for field in SORTED_SPIKES_GROUP_KEY_FIELDS
    }
    loaded_spikes = load_sorted_spikes_group(
        sorted_spikes_group,
        unit_selection_params,
        spike_sorting_output,
        group_key,
        region=normalize_region(registration_row.get("region_name")),
        region_validator=region_validator,
        time_support=time_support,
        allow_empty=True,
        pynapple_module=pynapple_module,
    )
    validate_region_sorted_spikes_group_row(registration_row, loaded_spikes)
    return loaded_spikes


__all__ = [
    "REGISTRATION_FIELDS",
    "REGISTRATION_ID_FIELD",
    "REGISTRATION_VALUE_FIELDS",
    "build_region_sorted_spikes_group_row",
    "normalize_region",
    "reload_region_sorted_spikes_group",
    "validate_region_sorted_spikes_group_row",
]
