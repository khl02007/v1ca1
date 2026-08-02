from __future__ import annotations

"""Database-agnostic adapters for Spyglass spike-sorting outputs.

The functions in this module operate on an injected ``SpikeSortingOutput``-like
object.  Importing the module therefore neither imports Spyglass/DataJoint nor
connects to a database.  NWB ``Units.spike_times`` values are treated as the
canonical seconds representation; SpikeInterface sortings are exposed
separately because their default spike trains are sample indices.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


SUPPORTED_MERGE_PARENTS = (
    "ImportedSpikeSorting",
    "CurationV1",
    "CuratedSpikeSorting",
)
NWB_UNITS_FIELDS = ("object_id", "units")


def _as_python_scalar(value: Any) -> Any:
    """Return a NumPy scalar as its plain-Python value."""
    return value.item() if isinstance(value, np.generic) else value


def _normalize_region(region: Any) -> str:
    """Return a normalized brain-region label for comparisons."""
    normalized = str(_as_python_scalar(region)).strip().casefold()
    if not normalized:
        raise ValueError("region must be a non-empty string.")
    return normalized


def _normalize_parent_name(parent: Any) -> str:
    """Return one supported Spyglass merge-parent name."""
    if not isinstance(parent, str):
        parent = getattr(parent, "table_name", None) or getattr(
            parent, "full_table_name", None
        ) or getattr(type(parent), "__name__", "")
    compact_name = "".join(
        character for character in str(parent) if character.isalnum()
    ).lower()
    for supported_parent in SUPPORTED_MERGE_PARENTS:
        supported_name = "".join(
            character for character in supported_parent if character.isalnum()
        ).lower()
        if supported_name in compact_name:
            return supported_parent
    raise ValueError(
        f"Unsupported spike-sorting merge parent {parent!r}. "
        f"Supported parents are {SUPPORTED_MERGE_PARENTS!r}."
    )


def _restrict_source(source: Any, key: Mapping[str, Any]) -> tuple[Any, bool]:
    """Restrict a relation-like source when it implements ``&``."""
    try:
        return source & dict(key), True
    except (AttributeError, TypeError):
        return source, False


def resolve_merge_parent(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str | None = None,
) -> str:
    """Resolve the merge parent for exactly one spike-sorting output.

    ``merge_parent`` is an explicit, database-free override useful for simple
    injected sources.  Otherwise the adapter first checks relation metadata and
    then uses the public ``source`` attribute stored by Spyglass merge tables.
    """
    if merge_parent is not None:
        return _normalize_parent_name(merge_parent)
    if "source" in key:
        return _normalize_parent_name(key["source"])

    for attribute_name in ("merge_parent", "merge_parent_name", "source_name"):
        parent = getattr(spike_sorting_output, attribute_name, None)
        if parent is not None:
            return _normalize_parent_name(parent)

    restricted, _ = _restrict_source(spike_sorting_output, key)
    fetch1 = getattr(restricted, "fetch1", None)
    if callable(fetch1):
        try:
            return _normalize_parent_name(fetch1("source"))
        except (AttributeError, KeyError, TypeError):
            pass

    get_parent = getattr(spike_sorting_output, "merge_get_parent", None)
    if callable(get_parent):
        return _normalize_parent_name(get_parent(dict(key)))

    raise ValueError(
        "Could not resolve a spike-sorting merge parent. Pass merge_parent="
        " explicitly or inject a relation exposing source/merge_get_parent."
    )


def _normalize_fetch_nwb_result(
    result: Any,
    *,
    fallback_merge_id: Any,
) -> tuple[list[Mapping[str, Any]], list[Any]]:
    """Normalize Spyglass ``fetch_nwb`` output and aligned merge identifiers."""
    returned_merge_ids: Sequence[Any] | None = None
    nwb_payloads = result
    if isinstance(result, tuple) and len(result) == 2:
        nwb_payloads, returned_merge_ids = result

    if isinstance(nwb_payloads, Mapping):
        payload_list = [nwb_payloads]
    else:
        payload_list = list(nwb_payloads)
    if not all(isinstance(payload, Mapping) for payload in payload_list):
        raise TypeError("fetch_nwb must return a mapping or a sequence of mappings.")

    if returned_merge_ids is None:
        if fallback_merge_id is None:
            raise ValueError(
                "fetch_nwb did not return merge ids and key has no merge_id/"
                "spikesorting_merge_id fallback."
            )
        merge_ids = [fallback_merge_id] * len(payload_list)
    else:
        if isinstance(returned_merge_ids, (str, bytes)) or not isinstance(
            returned_merge_ids, Sequence
        ):
            merge_ids = [returned_merge_ids]
        else:
            merge_ids = list(returned_merge_ids)
        if len(merge_ids) != len(payload_list):
            raise ValueError(
                "fetch_nwb returned different numbers of NWB payloads and merge ids: "
                f"{len(payload_list)} vs {len(merge_ids)}."
            )
    return payload_list, merge_ids


def _fetch_nwb_payloads(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Any]]:
    """Fetch NWB payloads through an injected merge relation."""
    restricted, was_restricted = _restrict_source(spike_sorting_output, key)
    fetch_nwb = getattr(restricted, "fetch_nwb", None)
    if not callable(fetch_nwb):
        raise TypeError("spike_sorting_output must expose a callable fetch_nwb method.")

    try:
        result = fetch_nwb(return_merge_ids=True)
    except TypeError as exc:
        if "return_merge_ids" not in str(exc):
            raise
        result = fetch_nwb() if was_restricted else fetch_nwb(dict(key))

    fallback_merge_id = key.get("spikesorting_merge_id", key.get("merge_id"))
    return _normalize_fetch_nwb_result(result, fallback_merge_id=fallback_merge_id)


def _get_units_table(nwb_payload: Mapping[str, Any]) -> Any:
    """Return the units dataframe-like object from one fetched NWB payload."""
    for field_name in NWB_UNITS_FIELDS:
        if field_name in nwb_payload:
            units = nwb_payload[field_name]
            if units is None:
                continue
            return units
    raise ValueError(
        "Fetched NWB payload has no units table. Expected one of "
        f"{NWB_UNITS_FIELDS!r}; found {tuple(nwb_payload)!r}."
    )


def _replace_units_table(nwb_payload: Mapping[str, Any], units: Any) -> dict[str, Any]:
    """Return a shallow NWB payload copy containing a replacement units table."""
    output = dict(nwb_payload)
    for field_name in NWB_UNITS_FIELDS:
        if field_name in output:
            output[field_name] = units
            return output
    raise ValueError(
        "Fetched NWB payload has no replaceable units table. Expected one of "
        f"{NWB_UNITS_FIELDS!r}."
    )


def _filter_imported_units_by_region(units: Any, *, region: str) -> Any:
    """Select one augmented-NWB ``region`` from ImportedSpikeSorting units."""
    columns = getattr(units, "columns", ())
    if "region" not in columns:
        raise ValueError(
            "ImportedSpikeSorting region selection requires the augmented NWB "
            "units table to contain a 'region' column."
        )
    region_values = np.asarray(units["region"], dtype=object).ravel()
    include = np.asarray(
        [_normalize_region(value) == region for value in region_values],
        dtype=bool,
    )
    if include.size != len(units):
        raise ValueError("Imported NWB units region values do not align with unit rows.")
    if not np.any(include):
        available_regions = sorted(
            {_normalize_region(value) for value in region_values}
        )
        raise ValueError(
            f"ImportedSpikeSorting has no units for region {region!r}; "
            f"available regions are {available_regions!r}."
        )
    return units.loc[include].copy()


def _extract_region_names(sort_group_info: Any) -> list[str]:
    """Extract region names from a DataJoint- or dataframe-like object."""
    if isinstance(sort_group_info, Mapping):
        if "region_name" not in sort_group_info:
            raise ValueError("sort-group information has no 'region_name' field.")
        values = sort_group_info["region_name"]
    elif "region_name" in getattr(sort_group_info, "columns", ()):
        values = sort_group_info["region_name"]
    else:
        fetch = getattr(sort_group_info, "fetch", None)
        if not callable(fetch):
            raise ValueError(
                "sort-group information must expose a 'region_name' column or fetch method."
            )
        values = fetch("region_name")

    if isinstance(values, (str, bytes)) or np.asarray(values).ndim == 0:
        values = [values]
    region_names = [_normalize_region(value) for value in list(values)]
    if not region_names:
        raise ValueError("sort-group information contains no region_name rows.")
    return region_names


def _run_region_validator(
    region_validator: Callable[..., Any],
    *,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    merge_parent: str,
    region: str,
) -> None:
    """Run an injected region validator with an explicit named protocol."""
    valid = region_validator(
        spike_sorting_output=spike_sorting_output,
        key=dict(key),
        merge_parent=merge_parent,
        region=region,
    )
    if valid is False:
        raise ValueError(
            f"Injected region validator rejected {merge_parent} for region {region!r}."
        )
    if valid not in (None, True):
        raise TypeError("region_validator must return True, False, or None.")


def _validate_curated_parent_region(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str,
    region: str,
    region_validator: Callable[..., Any] | None,
) -> None:
    """Require a curated sort group's catalog region to match the selection."""
    get_sort_group_info = getattr(spike_sorting_output, "get_sort_group_info", None)
    if callable(get_sort_group_info):
        sort_group_info = get_sort_group_info(dict(key))
        try:
            region_names = _extract_region_names(sort_group_info)
        except ValueError:
            if region_validator is None:
                raise
        else:
            unique_regions = sorted(set(region_names))
            if unique_regions != [region]:
                raise ValueError(
                    f"{merge_parent} sort-group region {unique_regions!r} does not "
                    f"match requested region {region!r}."
                )
            return

    if region_validator is None:
        raise ValueError(
            f"Cannot validate requested region {region!r} for {merge_parent}. "
            "Inject a source with get_sort_group_info or pass region_validator=."
        )
    _run_region_validator(
        region_validator,
        spike_sorting_output=spike_sorting_output,
        key=key,
        merge_parent=merge_parent,
        region=region,
    )


def _spikes_and_unit_ids_from_units(
    units: Any,
    *,
    merge_id: Any,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract aligned seconds arrays and stable unit identifiers."""
    columns = getattr(units, "columns", ())
    if "spike_times" not in columns:
        raise ValueError(
            "Fetched NWB units table is missing its required 'spike_times' column."
        )

    spike_values = units["spike_times"]
    raw_spike_times = (
        spike_values.to_list()
        if hasattr(spike_values, "to_list")
        else list(spike_values)
    )
    raw_unit_ids = list(getattr(units, "index", range(len(raw_spike_times))))
    if len(raw_unit_ids) != len(raw_spike_times):
        raise ValueError(
            "NWB unit ids and spike-time arrays have different lengths: "
            f"{len(raw_unit_ids)} vs {len(raw_spike_times)}."
        )
    if len({_as_python_scalar(unit_id) for unit_id in raw_unit_ids}) != len(raw_unit_ids):
        raise ValueError("NWB units table contains duplicate unit ids.")

    spike_times_s: list[np.ndarray] = []
    unit_ids: list[dict[str, Any]] = []
    unit_metadata: list[dict[str, Any]] = []
    metadata_columns = tuple(
        column
        for column in (
            "sorting_unit_id",
            "region",
            "probe_idx",
            "shank_idx",
            "label",
            "curation_label",
        )
        if column in columns
    )
    for row_index, (raw_unit_id, raw_times) in enumerate(
        zip(raw_unit_ids, raw_spike_times)
    ):
        times = np.asarray(raw_times, dtype=float)
        if times.ndim != 1:
            raise ValueError(
                "Each NWB spike_times entry must be one-dimensional; "
                f"got shape {times.shape}."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("NWB spike_times entries must contain only finite seconds values.")
        if times.size > 1 and np.any(np.diff(times) < 0):
            raise ValueError("NWB spike_times entries must be monotonically nondecreasing.")
        spike_times_s.append(times)
        stable_id = {
            "spikesorting_merge_id": _as_python_scalar(merge_id),
            "unit_id": _as_python_scalar(raw_unit_id),
        }
        unit_ids.append(stable_id)
        metadata = dict(stable_id)
        for column in metadata_columns:
            values = units[column]
            value = values.iloc[row_index] if hasattr(values, "iloc") else values[row_index]
            metadata[column] = _as_python_scalar(value)
        unit_metadata.append(metadata)
    return spike_times_s, unit_ids, unit_metadata


def _fetch_spike_times_seconds_and_metadata(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str | None,
    region: str | None,
    region_validator: Callable[..., Any] | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch canonical seconds plus aligned stable unit metadata."""
    parent = resolve_merge_parent(
        spike_sorting_output,
        key,
        merge_parent=merge_parent,
    )
    normalized_region = None if region is None else _normalize_region(region)
    if normalized_region is not None and parent != "ImportedSpikeSorting":
        _validate_curated_parent_region(
            spike_sorting_output,
            key,
            merge_parent=parent,
            region=normalized_region,
            region_validator=region_validator,
        )
    nwb_payloads, merge_ids = _fetch_nwb_payloads(spike_sorting_output, key)

    spike_times_s: list[np.ndarray] = []
    unit_ids: list[dict[str, Any]] = []
    unit_metadata: list[dict[str, Any]] = []
    for nwb_payload, merge_id in zip(nwb_payloads, merge_ids):
        units = _get_units_table(nwb_payload)
        if normalized_region is not None and parent == "ImportedSpikeSorting":
            units = _filter_imported_units_by_region(
                units,
                region=normalized_region,
            )
        file_spikes, file_unit_ids, file_metadata = _spikes_and_unit_ids_from_units(
            units,
            merge_id=merge_id,
        )
        spike_times_s.extend(file_spikes)
        unit_ids.extend(file_unit_ids)
        unit_metadata.extend(file_metadata)

    stable_ids = [
        (str(unit_id["spikesorting_merge_id"]), str(unit_id["unit_id"]))
        for unit_id in unit_ids
    ]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Fetched spike data contains duplicate stable unit identifiers.")
    return spike_times_s, unit_ids, unit_metadata


def fetch_spike_times_seconds(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str | None = None,
    region: str | None = None,
    region_validator: Callable[..., Any] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Fetch NWB spike times in seconds and their stable unit identifiers.

    The parent is resolved up front so unsupported merge sources fail clearly.
    No per-unit DataJoint rows are required: unit ids come from each NWB units
    dataframe index and are paired with the enclosing merge id.
    """
    spike_times_s, unit_ids, _ = _fetch_spike_times_seconds_and_metadata(
        spike_sorting_output,
        key,
        merge_parent=merge_parent,
        region=region,
        region_validator=region_validator,
    )
    return spike_times_s, unit_ids


def fetch_spike_times_seconds_with_metadata(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str | None = None,
    region: str | None = None,
    region_validator: Callable[..., Any] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch canonical seconds, stable ids, and aligned NWB unit metadata."""
    return _fetch_spike_times_seconds_and_metadata(
        spike_sorting_output,
        key,
        merge_parent=merge_parent,
        region=region,
        region_validator=region_validator,
    )


def build_compatibility_payload(
    spike_times_s: Sequence[np.ndarray],
    unit_ids: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the legacy list representation used by Spyglass spike groups."""
    if len(spike_times_s) != len(unit_ids):
        raise ValueError(
            "spike_times_s and unit_ids must have matching lengths; got "
            f"{len(spike_times_s)} and {len(unit_ids)}."
        )
    return {
        "spike_times": [np.asarray(times, dtype=float) for times in spike_times_s],
        "unit_ids": [dict(unit_id) for unit_id in unit_ids],
        "time_units": "s",
    }


def _build_time_support(
    nap: Any,
    spike_times_s: Sequence[np.ndarray],
    time_support: Any | tuple[float, float] | None,
) -> Any:
    """Return a Pynapple IntervalSet for a TsGroup."""
    if time_support is not None and not isinstance(time_support, tuple):
        return time_support
    if isinstance(time_support, tuple):
        if len(time_support) != 2:
            raise ValueError("time_support tuples must contain exactly (start, end).")
        start, end = (float(value) for value in time_support)
    else:
        nonempty = [np.asarray(times, dtype=float) for times in spike_times_s if len(times)]
        if not nonempty:
            raise ValueError(
                "time_support is required when every unit has an empty spike train."
            )
        start = min(float(times[0]) for times in nonempty)
        end = max(float(times[-1]) for times in nonempty)
        if end <= start:
            end = float(np.nextafter(start, np.inf))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError(
            f"time_support must have finite start < end, got {(start, end)!r}."
        )
    return nap.IntervalSet(start=start, end=end, time_units="s")


def build_spike_tsgroup(
    spike_times_s: Sequence[np.ndarray],
    unit_ids: Sequence[Mapping[str, Any]],
    *,
    time_support: Any | tuple[float, float] | None = None,
    pynapple_module: Any | None = None,
) -> Any:
    """Build a Pynapple ``TsGroup`` with stable ids stored as metadata.

    TsGroup keys are deliberately sequential and ephemeral. Pynapple sorts
    integer keys during construction, so using non-monotonic NWB unit ids as
    keys can otherwise reorder spike trains relative to their metadata.
    """
    if len(spike_times_s) != len(unit_ids):
        raise ValueError(
            "spike_times_s and unit_ids must have matching lengths; got "
            f"{len(spike_times_s)} and {len(unit_ids)}."
        )
    if pynapple_module is None:
        try:
            import pynapple as pynapple_module
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pynapple is required to build a TsGroup from spike times."
            ) from exc

    support = _build_time_support(pynapple_module, spike_times_s, time_support)
    native_unit_ids = [_as_python_scalar(unit_id["unit_id"]) for unit_id in unit_ids]
    merge_ids = [unit_id["spikesorting_merge_id"] for unit_id in unit_ids]
    group_keys = list(range(len(unit_ids)))

    data = {
        group_key: pynapple_module.Ts(
            t=np.asarray(times, dtype=float),
            time_units="s",
            time_support=support,
        )
        for group_key, times in zip(group_keys, spike_times_s)
    }
    metadata = {
        "spikesorting_merge_id": [_as_python_scalar(value) for value in merge_ids],
        "unit_id": native_unit_ids,
    }
    try:
        return pynapple_module.TsGroup(
            data,
            time_support=support,
            time_units="s",
            metadata=metadata,
        )
    except TypeError:
        try:
            return pynapple_module.TsGroup(
                data,
                time_support=support,
                time_units="s",
                **metadata,
            )
        except TypeError:
            group = pynapple_module.TsGroup(
                data,
                time_support=support,
                time_units="s",
            )
        set_info = getattr(group, "set_info", None)
        if callable(set_info):
            set_info(**metadata)
        return group


def load_spikes(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str | None = None,
    region: str | None = None,
    region_validator: Callable[..., Any] | None = None,
    time_support: Any | tuple[float, float] | None = None,
    pynapple_module: Any | None = None,
) -> dict[str, Any]:
    """Load one merge output as seconds arrays, compatibility lists, and TsGroup."""
    parent = resolve_merge_parent(
        spike_sorting_output,
        key,
        merge_parent=merge_parent,
    )
    spike_times_s, unit_ids = fetch_spike_times_seconds(
        spike_sorting_output,
        key,
        merge_parent=parent,
        region=region,
        region_validator=region_validator,
    )
    compatibility = build_compatibility_payload(spike_times_s, unit_ids)
    return {
        "merge_parent": parent,
        "region": None if region is None else _normalize_region(region),
        "spike_times_s": spike_times_s,
        "unit_ids": unit_ids,
        "ts_group": build_spike_tsgroup(
            spike_times_s,
            unit_ids,
            time_support=time_support,
            pynapple_module=pynapple_module,
        ),
        "compatibility": compatibility,
    }


def get_spikeinterface_sorting(
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    *,
    merge_parent: str | None = None,
    region: str | None = None,
    region_validator: Callable[..., Any] | None = None,
    sorting_factory: Callable[..., Any] | None = None,
    nwb_extractor: Callable[..., Any] | None = None,
) -> Any:
    """Return a SpikeInterface sorting with an absolute recording time vector.

    ``CurationV1`` and legacy ``CuratedSpikeSorting`` outputs dispatch to their
    public ``get_sorting`` implementation.  ``ImportedSpikeSorting`` has no
    recording or sample frequency in Spyglass, so callers must inject either a
    ``sorting_factory`` or an ``nwb_extractor``.  These callables receive named
    ``nwb_payloads``, ``key``, and ``merge_parent`` arguments.  SpikeInterface
    returns sample frames by default.  For curated Spyglass parents this
    adapter registers the upstream recording so ``return_times=True`` uses its
    NWB/ephys-reference time vector instead of frame/sampling-rate seconds from
    zero.  An Imported factory must return a sorting with the same explicit
    recording time reference.
    """
    parent = resolve_merge_parent(
        spike_sorting_output,
        key,
        merge_parent=merge_parent,
    )
    normalized_region = None if region is None else _normalize_region(region)
    if parent != "ImportedSpikeSorting":
        if normalized_region is not None:
            _validate_curated_parent_region(
                spike_sorting_output,
                key,
                merge_parent=parent,
                region=normalized_region,
                region_validator=region_validator,
            )
        get_sorting = getattr(spike_sorting_output, "get_sorting", None)
        if not callable(get_sorting):
            raise TypeError(
                f"{parent} source must expose a callable get_sorting method."
            )
        sorting = get_sorting(dict(key))
        get_recording = getattr(spike_sorting_output, "get_recording", None)
        if not callable(get_recording):
            raise TypeError(
                f"{parent} source must expose get_recording so SpikeInterface "
                "seconds retain the NWB time reference."
            )
        recording = get_recording(dict(key))
        _require_recording_time_vector(recording)
        register_recording = getattr(sorting, "register_recording", None)
        if not callable(register_recording):
            raise TypeError("SpikeInterface sorting must expose register_recording().")
        register_recording(recording)
        _require_sorting_recording_time_vector(sorting)
        return sorting

    if sorting_factory is None and nwb_extractor is None:
        raise NotImplementedError(
            "ImportedSpikeSorting cannot infer SpikeInterface sample frames. "
            "Pass sorting_factory= or nwb_extractor= with an explicit sampling model."
        )
    nwb_payloads, _ = _fetch_nwb_payloads(spike_sorting_output, key)
    if normalized_region is not None:
        nwb_payloads = [
            _replace_units_table(
                nwb_payload,
                _filter_imported_units_by_region(
                    _get_units_table(nwb_payload),
                    region=normalized_region,
                ),
            )
            for nwb_payload in nwb_payloads
        ]
    callable_adapter = sorting_factory or nwb_extractor
    sorting = callable_adapter(
        nwb_payloads=nwb_payloads,
        key=dict(key),
        merge_parent=parent,
        region=normalized_region,
    )
    _require_sorting_recording_time_vector(sorting)
    return sorting


def _require_recording_time_vector(recording: Any) -> None:
    """Require an explicit time vector on every SpikeInterface segment."""
    get_num_segments = getattr(recording, "get_num_segments", None)
    has_time_vector = getattr(recording, "has_time_vector", None)
    if not callable(get_num_segments) or not callable(has_time_vector):
        raise TypeError(
            "SpikeInterface recording must expose get_num_segments() and "
            "has_time_vector()."
        )
    missing_segments = [
        segment_index
        for segment_index in range(int(get_num_segments()))
        if not bool(has_time_vector(segment_index=segment_index))
    ]
    if missing_segments:
        raise ValueError(
            "SpikeInterface recording lacks an explicit NWB/ephys time vector "
            f"for segments {missing_segments!r}."
        )


def _require_sorting_recording_time_vector(sorting: Any) -> None:
    """Require a sorting registered to a recording with explicit timestamps."""
    has_recording = getattr(sorting, "has_recording", None)
    if not callable(has_recording) or not bool(has_recording()):
        raise ValueError(
            "SpikeInterface sorting is not registered to its recording; "
            "return_times=True would not retain the NWB/ephys time reference."
        )
    get_num_segments = getattr(sorting, "get_num_segments", None)
    has_time_vector = getattr(sorting, "has_time_vector", None)
    if not callable(get_num_segments) or not callable(has_time_vector):
        raise TypeError(
            "SpikeInterface sorting must expose get_num_segments() and "
            "has_time_vector()."
        )
    missing_segments = [
        segment_index
        for segment_index in range(int(get_num_segments()))
        if not bool(has_time_vector(segment_index=segment_index))
    ]
    if missing_segments:
        raise ValueError(
            "The sorting's registered recording lacks an explicit NWB/ephys "
            f"time vector for segments {missing_segments!r}."
        )


def spikeinterface_times_seconds(
    sorting: Any,
    *,
    spikesorting_merge_id: Any,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Extract NWB/ephys-reference seconds from a validated SI sorting."""
    _require_sorting_recording_time_vector(sorting)
    get_unit_ids = getattr(sorting, "get_unit_ids", None)
    get_spike_train = getattr(sorting, "get_unit_spike_train", None)
    if not callable(get_unit_ids) or not callable(get_spike_train):
        raise TypeError(
            "sorting must expose get_unit_ids and get_unit_spike_train methods."
        )

    spike_times_s: list[np.ndarray] = []
    unit_ids: list[dict[str, Any]] = []
    for unit_id in get_unit_ids():
        times = np.asarray(
            get_spike_train(unit_id=unit_id, return_times=True),
            dtype=float,
        )
        spike_times_s.append(times)
        unit_ids.append(
            {
                "spikesorting_merge_id": _as_python_scalar(spikesorting_merge_id),
                "unit_id": _as_python_scalar(unit_id),
            }
        )
    return spike_times_s, unit_ids
