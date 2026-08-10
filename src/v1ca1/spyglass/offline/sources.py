"""Read Figure 1 inputs directly from an augmented NWB without a database."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass.nwb import (
    catalog_augmented_nwb,
    load_interval_set,
    load_position,
    load_wtrack_graph,
)
from v1ca1.spyglass.selection import unit_identity_sha256
from v1ca1.spyglass.spikes import build_spike_tsgroup


SOURCE_IDENTITY_POLICY = {
    "source": "ImportedSpikeSorting",
    "selection_identity_scope": "offline_surrogate",
    "merge_id": (
        "UUID representation of DataJoint key_hash over nwb_file_name and "
        "source=ImportedSpikeSorting"
    ),
    "unit_id": "NWB Units dataframe index",
    "spike_time_unit": "s",
    "spike_time_reference": "NWB/ephys timestamps",
    "unit_filter": "all units matching the normalized NWB region column",
}


def validate_nwb_session_identity(
    nwbfile: Any,
    *,
    animal_name: str,
    date: str,
) -> None:
    """Require CLI session labels to match the opened NWB source."""
    subject = getattr(nwbfile, "subject", None)
    subject_id = None if subject is None else getattr(subject, "subject_id", None)
    if str(subject_id).strip() != str(animal_name).strip():
        raise ValueError(
            "NWB subject_id does not match --animal-name: "
            f"{subject_id!r} != {animal_name!r}."
        )
    session_start_time = getattr(nwbfile, "session_start_time", None)
    strftime = getattr(session_start_time, "strftime", None)
    if not callable(strftime):
        raise ValueError("NWB session_start_time is unavailable or invalid.")
    nwb_date = strftime("%Y%m%d")
    if nwb_date != str(date):
        raise ValueError(
            "NWB session_start_time date does not match --date: "
            f"{nwb_date!r} != {date!r}."
        )


def imported_spike_sorting_merge_id(nwb_file_name: str) -> str:
    """Return the future Spyglass ImportedSpikeSorting merge UUID."""
    file_name = Path(str(nwb_file_name)).name
    if not file_name or file_name != str(nwb_file_name):
        raise ValueError("nwb_file_name must be one basename.")
    # DataJoint key_hash updates the MD5 with values sorted by key name. The
    # corresponding merge key is {nwb_file_name, source}; no DB import is needed.
    payload = f"{file_name}ImportedSpikeSorting".encode("utf-8")
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return str(uuid.UUID(hex=digest))


def _one_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    description: str,
    **selectors: str,
) -> dict[str, Any]:
    """Return exactly one catalog row matching string-valued selectors."""
    matches = [
        dict(row)
        for row in rows
        if all(str(row.get(name)) == str(value) for name, value in selectors.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} row for {selectors!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def select_figure_1_catalog(
    nwbfile: Any,
    *,
    nwb_file_name: str,
    epoch: str,
    position_role: str = "head",
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
) -> dict[str, Any]:
    """Select and validate one run epoch's Figure 1 NWB catalog rows."""
    selection = select_run_epoch_catalog(
        nwbfile,
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        position_roles=(position_role,),
        trajectory_types=trajectory_types,
        require_dark=True,
    )
    return {
        "epoch_row": selection["epoch_row"],
        "position_row": selection["position_rows"][str(position_role)],
        "trajectory_rows": selection["trajectory_rows"],
        "graph_rows": selection["graph_rows"],
    }


def select_run_epoch_catalog(
    nwbfile: Any,
    *,
    nwb_file_name: str,
    epoch: str,
    position_roles: Sequence[str] = ("head",),
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    graph_configurations: Sequence[str] = (),
    require_dark: bool = False,
) -> dict[str, Any]:
    """Select centimeter positions, intervals, and graphs for one run epoch."""
    catalog = catalog_augmented_nwb(nwbfile, nwb_file_name=nwb_file_name)
    epoch_row = _one_row(
        catalog["epoch_intervals"],
        description="epoch",
        epoch=epoch,
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError(f"Figure 1 tuning requires a run epoch, got {epoch!r}.")
    if require_dark:
        condition = str(epoch_row.get("condition", "")).strip().casefold()
        is_light = epoch_row.get("is_light")
        if condition != "dark" or is_light is None or bool(is_light):
            raise ValueError(
                "Figure 1 tuning requires an explicitly cataloged dark run "
                f"epoch; got condition={epoch_row.get('condition')!r}, "
                f"is_light={is_light!r}."
            )

    requested_roles = tuple(str(value) for value in position_roles)
    if (
        not requested_roles
        or any(not value for value in requested_roles)
        or len(requested_roles) != len(set(requested_roles))
    ):
        raise ValueError("position_roles must be a non-empty unique sequence.")
    position_rows = {
        role: _one_row(
            catalog["position"],
            description="position",
            epoch=epoch,
            position_role=role,
        )
        for role in requested_roles
    }
    if any(row.get("spatial_unit") != "cm" for row in position_rows.values()):
        raise ValueError("Selected positions must use centimeters.")

    requested = tuple(str(value) for value in trajectory_types)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("trajectory_types must be a non-empty unique sequence.")
    trajectory_rows: dict[str, dict[str, Any]] = {}
    graph_rows: dict[str, dict[str, Any]] = {}
    for trajectory_type in requested:
        trajectory_rows[trajectory_type] = _one_row(
            catalog["trajectory_intervals"],
            description="trajectory",
            epoch=epoch,
            trajectory_type=trajectory_type,
        )
    graph_names = tuple(
        dict.fromkeys(
            (*requested, *(str(value) for value in graph_configurations))
        )
    )
    if any(not value for value in graph_names):
        raise ValueError("graph_configurations must contain non-empty names.")
    for configuration_name in graph_names:
        graph_rows[configuration_name] = _one_row(
            catalog["wtrack_graph"],
            description="W-track graph",
            configuration_name=configuration_name,
        )
        if graph_rows[configuration_name].get("coordinate_unit") != "cm":
            raise ValueError("Selected W-track graph must use centimeters.")
    return {
        "epoch_row": epoch_row,
        "position_rows": position_rows,
        "trajectory_rows": trajectory_rows,
        "graph_rows": graph_rows,
    }


def load_figure_1_catalog_objects(
    nwbfile: Any,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Load selected position, lap intervals, and graph inputs into memory."""
    return {
        "position": load_position(
            nwbfile,
            selection["position_row"],
            apply_analysis_offset=True,
        ),
        "trajectory_intervals": {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in selection["trajectory_rows"].items()
        },
        "graph_inputs": {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in selection["graph_rows"].items()
        },
    }


def load_run_epoch_catalog_objects(
    nwbfile: Any,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Load all objects selected by :func:`select_run_epoch_catalog`."""
    return {
        "positions": {
            role: load_position(nwbfile, row, apply_analysis_offset=True)
            for role, row in selection["position_rows"].items()
        },
        "trajectory_intervals": {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in selection["trajectory_rows"].items()
        },
        "graph_inputs": {
            configuration_name: load_wtrack_graph(nwbfile, row)
            for configuration_name, row in selection["graph_rows"].items()
        },
    }


def _native(value: Any) -> Any:
    """Return a NumPy scalar as a plain Python value."""
    return value.item() if isinstance(value, np.generic) else value


def load_nwb_region_spikes(
    nwbfile: Any,
    *,
    nwb_file_name: str,
    region: str,
    time_support: tuple[float, float],
    pynapple_module: Any | None = None,
) -> dict[str, Any]:
    """Load all augmented-NWB units in one region as seconds and a TsGroup."""
    normalized_region = str(region).strip().casefold()
    if not normalized_region:
        raise ValueError("region must be non-empty.")
    units = getattr(nwbfile, "units", None)
    if units is None:
        raise ValueError("Augmented NWB file has no Units table.")
    dataframe = units.to_dataframe()
    missing = sorted({"spike_times", "region"}.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Augmented NWB Units table is missing columns {missing!r}.")
    regions = dataframe["region"].astype(str).str.strip().str.casefold()
    selected = dataframe.loc[regions == normalized_region]
    if selected.index.duplicated().any():
        raise ValueError("Augmented NWB Units table contains duplicate unit ids.")

    merge_id = imported_spike_sorting_merge_id(nwb_file_name)
    spike_times_s: list[np.ndarray] = []
    unit_ids: list[dict[str, Any]] = []
    unit_metadata: list[dict[str, Any]] = []
    metadata_columns = tuple(
        name
        for name in ("sorting_unit_id", "region", "probe_idx", "shank_idx")
        if name in selected.columns
    )
    for unit_id, row in selected.iterrows():
        times = np.asarray(row["spike_times"], dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("Every NWB spike_times entry must be finite and 1-D.")
        if times.size > 1 and np.any(np.diff(times) < 0.0):
            raise ValueError("NWB spike_times must be monotonically nondecreasing.")
        identity = {
            "spikesorting_merge_id": merge_id,
            "unit_id": _native(unit_id),
        }
        spike_times_s.append(times)
        unit_ids.append(identity)
        unit_metadata.append(
            {
                **identity,
                **{name: _native(row[name]) for name in metadata_columns},
            }
        )
    ts_group = build_spike_tsgroup(
        spike_times_s,
        unit_ids,
        time_support=time_support,
        pynapple_module=pynapple_module,
    )
    return {
        "source": "ImportedSpikeSorting",
        "spikesorting_merge_id": merge_id,
        "region": normalized_region,
        "status": "valid" if unit_ids else "no_units",
        "n_units": len(unit_ids),
        "selected_units_sha256": unit_identity_sha256(unit_ids),
        "unit_ids": unit_ids,
        "unit_metadata": unit_metadata,
        "spike_times_s": spike_times_s,
        "ts_group": ts_group,
    }


def resolve_sorting_unit(
    loaded_spikes: Mapping[str, Any],
    *,
    sorting_unit_id: Any,
) -> dict[str, Any]:
    """Resolve one region-local sorting ID to its persistent NWB identity."""
    metadata = loaded_spikes.get("unit_metadata")
    if not isinstance(metadata, Sequence):
        raise ValueError("Loaded spikes do not contain unit_metadata.")
    matches = [
        (group_key, dict(row))
        for group_key, row in enumerate(metadata)
        if str(row.get("sorting_unit_id")) == str(sorting_unit_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one unit for sorting_unit_id "
            f"{sorting_unit_id!r}; found {len(matches)}."
        )
    group_key, row = matches[0]
    return {
        "group_key": int(group_key),
        "sorting_unit_id": _native(row["sorting_unit_id"]),
        "spikesorting_merge_id": str(row["spikesorting_merge_id"]),
        "unit_id": _native(row["unit_id"]),
        "metadata": row,
    }


__all__ = [
    "SOURCE_IDENTITY_POLICY",
    "imported_spike_sorting_merge_id",
    "load_figure_1_catalog_objects",
    "load_nwb_region_spikes",
    "load_run_epoch_catalog_objects",
    "resolve_sorting_unit",
    "select_figure_1_catalog",
    "select_run_epoch_catalog",
    "validate_nwb_session_identity",
]
