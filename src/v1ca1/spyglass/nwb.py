from __future__ import annotations

"""Read small catalog records and selected arrays from augmented NWB objects.

The catalog functions in this module accept an already-open NWB-like object.
They neither open files nor import DataJoint or Spyglass. Large position and
W-track arrays stay in the NWB object until an explicit loader is called.
"""

from collections.abc import Mapping
import json
import re
from typing import Any

import numpy as np


EPHYS_INTERVALS_TABLE_NAME = "ephys_recording_intervals"
TRAJECTORY_INTERVALS_TABLE_NAME = "trajectory_times"
RIPPLES_INTERVALS_TABLE_NAME = "ripples"
POSITION_EPOCHS_TABLE_NAME = "position_epochs"
POSITION_INTERFACE_NAME = "position"
POSITION_SERIES_BY_TYPE = {
    "head": "head_position",
    "body": "body_position",
}
WTRACK_LINEARIZATION_TABLE_NAME = "wtrack_linearization"
SPIKE_SORTING_FIGURLS_TABLE_NAME = "spike_sorting_figurls"
RIPPLE_PROVENANCE_SCRATCH_NAME = "ripple_detection_provenance"

EPOCHS_TABLE_PATH = "/intervals/epochs"
EPHYS_INTERVALS_TABLE_PATH = f"/intervals/{EPHYS_INTERVALS_TABLE_NAME}"
TRAJECTORY_INTERVALS_TABLE_PATH = (
    f"/intervals/{TRAJECTORY_INTERVALS_TABLE_NAME}"
)
RIPPLES_INTERVALS_TABLE_PATH = f"/intervals/{RIPPLES_INTERVALS_TABLE_NAME}"
RIPPLE_PROVENANCE_PATH = f"/scratch/{RIPPLE_PROVENANCE_SCRATCH_NAME}"
POSITION_EPOCHS_TABLE_PATH = f"/intervals/{POSITION_EPOCHS_TABLE_NAME}"
POSITION_INTERFACE_PATH = f"/processing/behavior/{POSITION_INTERFACE_NAME}"
WTRACK_LINEARIZATION_TABLE_PATH = (
    f"/processing/behavior/{WTRACK_LINEARIZATION_TABLE_NAME}"
)
SPIKE_SORTING_FIGURLS_TABLE_PATH = (
    f"/processing/ecephys/{SPIKE_SORTING_FIGURLS_TABLE_NAME}"
)


def _mapping_value(mapping: Any, key: str) -> Any | None:
    """Return one value from a mapping-like NWB container."""
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    get_value = getattr(mapping, "get", None)
    if callable(get_value):
        return get_value(key)
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return None


def _interval_table(nwbfile: Any, table_name: str) -> Any | None:
    """Return one interval table, including the special epochs table."""
    if table_name == "epochs":
        return getattr(nwbfile, "epochs", None)
    return _mapping_value(getattr(nwbfile, "intervals", None), table_name)


def _processing_interface(
    nwbfile: Any,
    module_name: str,
    interface_name: str,
) -> Any | None:
    """Return one processing data interface from an NWB-like object."""
    module = _mapping_value(getattr(nwbfile, "processing", None), module_name)
    if module is None:
        return None
    return _mapping_value(getattr(module, "data_interfaces", None), interface_name)


def _table_column_names(table: Any) -> tuple[str, ...]:
    """Return the ordered column names exposed by one DynamicTable-like object."""
    column_names = getattr(table, "colnames", None)
    if column_names is None:
        raise ValueError(f"NWB table {getattr(table, 'name', '<unknown>')!r} has no columns.")
    return tuple(str(column_name) for column_name in column_names)


def _require_columns(table: Any, path: str, required: set[str]) -> None:
    """Require a set of columns on one DynamicTable-like object."""
    missing = sorted(required.difference(_table_column_names(table)))
    if missing:
        raise ValueError(f"NWB table {path} is missing required columns: {missing!r}.")


def _table_length(table: Any) -> int:
    """Return the number of rows in one DynamicTable-like object."""
    table_id = getattr(table, "id", None)
    if table_id is not None:
        try:
            return int(len(table_id))
        except TypeError:
            pass
    try:
        return int(len(table))
    except TypeError as exc:
        raise ValueError(
            f"Could not determine row count for NWB table {getattr(table, 'name', '<unknown>')!r}."
        ) from exc


def _table_cell(table: Any, column_name: str, row_index: int) -> Any:
    """Read one cell without materializing a whole DynamicTable."""
    try:
        column = table[column_name]
    except (KeyError, TypeError):
        column = getattr(table, column_name, None)
    if column is None:
        raise ValueError(
            f"Could not read column {column_name!r} from NWB table "
            f"{getattr(table, 'name', '<unknown>')!r}."
        )
    try:
        return column[row_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Could not read row {row_index} of column {column_name!r} from "
            f"NWB table {getattr(table, 'name', '<unknown>')!r}."
        ) from exc


def _native_scalar(value: Any) -> Any:
    """Convert one NumPy or byte scalar to a JSON-friendly Python scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _text(value: Any, field_name: str) -> str:
    """Normalize one required scalar text value."""
    value = _native_scalar(value)
    if value is None:
        raise ValueError(f"NWB field {field_name!r} is missing.")
    result = str(value)
    if not result:
        raise ValueError(f"NWB field {field_name!r} is empty.")
    return result


def _optional_text(value: Any) -> str | None:
    """Normalize one optional scalar text value."""
    value = _native_scalar(value)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    result = str(value)
    return result or None


def _centimeter_unit(value: Any, source_path: str) -> str:
    """Normalize supported centimeter spellings and reject mixed coordinates."""
    unit = _optional_text(value)
    if unit is None or unit.strip().casefold() not in {
        "cm",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
    }:
        raise ValueError(
            f"NWB coordinates at {source_path} must use centimeters to match "
            "W-track node_positions_cm."
        )
    return "cm"


def _integer(value: Any, field_name: str) -> int:
    """Normalize one integer-valued NWB cell."""
    value = _native_scalar(value)
    if isinstance(value, bool):
        raise ValueError(f"NWB field {field_name!r} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"NWB field {field_name!r} must be an integer.") from exc
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        numeric_value = float(result)
    if not np.isfinite(numeric_value) or numeric_value != result:
        raise ValueError(f"NWB field {field_name!r} must be an integer.")
    return result


def _float(value: Any, field_name: str) -> float:
    """Normalize one finite floating-point NWB cell."""
    try:
        result = float(_native_scalar(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"NWB field {field_name!r} must be numeric.") from exc
    if not np.isfinite(result):
        raise ValueError(f"NWB field {field_name!r} must be finite.")
    return result


def _boolean(value: Any, field_name: str) -> bool:
    """Normalize one strict Boolean NWB cell."""
    result = _native_scalar(value)
    if not isinstance(result, bool):
        raise ValueError(f"NWB field {field_name!r} must be boolean.")
    return result


def _tags(value: Any) -> list[str]:
    """Normalize one source epochs-table tags cell."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [_text(value, "tags")]
    try:
        return [_text(tag, "tags") for tag in value]
    except TypeError as exc:
        raise ValueError("NWB epochs tags must be a sequence of strings.") from exc


def _object_id(nwb_object: Any) -> str | None:
    """Return an NWB object id when the source exposes one."""
    object_id = getattr(nwb_object, "object_id", None)
    if object_id is None:
        return None
    return str(object_id)


def _source_pointer(path: str, table: Any) -> dict[str, Any]:
    """Build standard table and object pointer fields."""
    object_id = _object_id(table)
    return {
        "source_table_path": path,
        "source_table_object_id": object_id,
        "source_object_path": path,
        "source_object_id": object_id,
    }


def _validated_interval_bounds(
    table: Any,
    *,
    row_index: int,
    path: str,
) -> tuple[float, float]:
    """Return one finite, positive-duration interval row."""
    start_time = _float(_table_cell(table, "start_time", row_index), "start_time")
    stop_time = _float(_table_cell(table, "stop_time", row_index), "stop_time")
    if stop_time <= start_time:
        raise ValueError(
            f"NWB interval table {path} row {row_index} must have start < stop."
        )
    return start_time, stop_time


def _validate_catalog_object_id(
    catalog_row: Mapping[str, Any],
    field_name: str,
    nwb_object: Any,
    *,
    object_label: str,
) -> None:
    """Reject a catalog pointer that belongs to another open NWB object."""
    expected_object_id = catalog_row.get(field_name)
    if expected_object_id is None:
        return
    current_object_id = _object_id(nwb_object)
    if current_object_id is None:
        raise ValueError(
            f"Cannot verify {object_label}: the open NWB object has no object_id."
        )
    if str(expected_object_id) != current_object_id:
        raise ValueError(
            f"Catalog {field_name} does not match the open NWB {object_label}."
        )


def _copy_nwb_file_name(row: dict[str, Any], nwb_file_name: str | None) -> None:
    """Copy an optional Spyglass NWB file key into one catalog row."""
    if nwb_file_name is not None:
        row["nwb_file_name"] = str(nwb_file_name)


def _container_items(container: Any) -> list[tuple[str, Any]]:
    """Return named values from a mapping-like NWB container."""
    if container is None:
        return []
    items = getattr(container, "items", None)
    if callable(items):
        return [(str(name), value) for name, value in items()]
    try:
        return [(str(name), container[name]) for name in container]
    except (KeyError, TypeError) as exc:
        raise ValueError("Could not enumerate an NWB object container.") from exc


def _parse_task_epochs(value: Any, source_path: str) -> list[int]:
    """Parse one 1-based task-epoch scalar, sequence, or comma string."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        residue = re.sub(r"[0-9,;\s\[\]()]+", "", value)
        if residue:
            raise ValueError(
                f"NWB task_epochs at {source_path} contains unsupported text {value!r}."
            )
        raw_values: list[Any] = re.findall(r"\d+", value)
    elif np.isscalar(value):
        raw_values = [value]
    else:
        try:
            raw_values = np.asarray(value, dtype=object).reshape(-1).tolist()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not parse NWB task_epochs at {source_path}.") from exc

    epoch_numbers: list[int] = []
    for raw_value in raw_values:
        epoch_number = _integer(raw_value, "task_epochs")
        if epoch_number <= 0:
            raise ValueError(f"NWB task_epochs at {source_path} must be 1-based.")
        if epoch_number not in epoch_numbers:
            epoch_numbers.append(epoch_number)
    if not epoch_numbers:
        raise ValueError(f"NWB task_epochs at {source_path} is empty.")
    return epoch_numbers


def _task_records_by_epoch(nwbfile: Any) -> dict[int, list[dict[str, Any]]]:
    """Read explicit per-epoch task labels from the tasks processing module."""
    task_module = _mapping_value(getattr(nwbfile, "processing", None), "tasks")
    if task_module is None:
        return {}
    records_by_epoch: dict[int, list[dict[str, Any]]] = {}
    interfaces = getattr(task_module, "data_interfaces", None)
    for interface_name, table in _container_items(interfaces):
        path = f"/processing/tasks/{interface_name}"
        column_names = set(_table_column_names(table))
        if "task_epochs" not in column_names:
            raise ValueError(f"NWB task table {path} is missing 'task_epochs'.")
        for row_index in range(_table_length(table)):
            record = {
                "task_name": (
                    _optional_text(_table_cell(table, "task_name", row_index))
                    if "task_name" in column_names
                    else None
                ),
                "task_description": (
                    _optional_text(_table_cell(table, "task_description", row_index))
                    if "task_description" in column_names
                    else None
                ),
                "task_environment": (
                    _optional_text(_table_cell(table, "task_environment", row_index))
                    if "task_environment" in column_names
                    else None
                ),
                "task_source_path": path,
                "task_object_id": _object_id(table),
            }
            epoch_numbers = _parse_task_epochs(
                _table_cell(table, "task_epochs", row_index),
                path,
            )
            for epoch_number in epoch_numbers:
                records_by_epoch.setdefault(epoch_number, []).append(record)
    return records_by_epoch


def _associated_files_by_epoch(nwbfile: Any) -> dict[int, list[dict[str, Any]]]:
    """Read explicit per-epoch labels from the associated-files module."""
    associated_module = _mapping_value(
        getattr(nwbfile, "processing", None),
        "associated_files",
    )
    if associated_module is None:
        return {}
    records_by_epoch: dict[int, list[dict[str, Any]]] = {}
    interfaces = getattr(associated_module, "data_interfaces", None)
    for interface_name, associated_file in _container_items(interfaces):
        path = f"/processing/associated_files/{interface_name}"
        task_epochs = getattr(associated_file, "task_epochs", None)
        if task_epochs is None:
            raise ValueError(f"NWB associated file {path} is missing task_epochs.")
        record = {
            "name": _optional_text(getattr(associated_file, "name", None)),
            "description": _optional_text(
                getattr(associated_file, "description", None)
            ),
            "source_path": path,
            "object_id": _object_id(associated_file),
        }
        for epoch_number in _parse_task_epochs(task_epochs, path):
            records_by_epoch.setdefault(epoch_number, []).append(record)
    return records_by_epoch


def _explicit_epoch_type(values: list[str], epoch: str) -> tuple[str | None, str | None]:
    """Return an unambiguous run/sleep label and its evidence source."""
    normalized_values = [value.lower() for value in values]
    explicit_types: set[str] = set()
    for value in normalized_values:
        if re.search(r"\bsleep\b", value):
            explicit_types.add("sleep")
        if re.search(r"\brun\b", value):
            explicit_types.add("run")
    if len(explicit_types) == 1:
        return next(iter(explicit_types)), "task"
    if not explicit_types and re.search(r"(?:^|_)s\d*(?:_|$)", epoch.lower()):
        return "sleep", "epoch_tag"
    return None, None


def _condition_token(values: list[str]) -> str | None:
    """Return one unambiguous explicit condition token from raw labels."""
    tokens: set[str] = set()
    for value in values:
        normalized = value.lower()
        for match in re.finditer(r"\bstim(?:ulus)?[\s_-]*([123])\b", normalized):
            tokens.add(f"stim{match.group(1)}")
        for token in ("ab", "gray", "ba", "dark", "bright", "sleep"):
            if re.search(rf"\b{token}\b", normalized):
                tokens.add(token)
    if len(tokens) == 1:
        return next(iter(tokens))
    return None


def _normalize_project_condition(token: str) -> str:
    """Normalize explicit NWB stimulus labels to project condition names."""
    return {
        "stim1": "AB",
        "stim2": "gray",
        "stim3": "BA",
        "ab": "AB",
        "gray": "gray",
        "ba": "BA",
        "dark": "dark",
        "bright": "bright",
        "sleep": "sleep",
    }[token]


def _is_light_condition(condition: str | None) -> bool | None:
    """Return explicit project illumination state after condition normalization."""
    if condition in {"AB", "gray", "BA", "bright"}:
        return True
    if condition == "dark":
        return False
    return None


def _task_epoch_metadata(nwbfile: Any, epoch_tags: list[str]) -> dict[str, dict[str, Any]]:
    """Return audited task and condition metadata keyed by epoch tag."""
    task_records = _task_records_by_epoch(nwbfile)
    associated_records = _associated_files_by_epoch(nwbfile)
    metadata_by_epoch: dict[str, dict[str, Any]] = {}
    for epoch_number, epoch in enumerate(epoch_tags, start=1):
        tasks = task_records.get(epoch_number, [])
        if len(tasks) > 1:
            distinct_tasks = {
                (
                    record["task_name"],
                    record["task_description"],
                    record["task_environment"],
                )
                for record in tasks
            }
            if len(distinct_tasks) > 1:
                raise ValueError(
                    f"NWB task metadata is ambiguous for 1-based epoch {epoch_number}."
                )
        task = tasks[0] if tasks else {}
        associated = associated_records.get(epoch_number, [])
        task_values = [
            str(value)
            for value in (
                task.get("task_name"),
                task.get("task_description"),
                task.get("task_environment"),
            )
            if value is not None
        ]
        associated_values = [
            str(value)
            for record in associated
            for value in (record.get("name"), record.get("description"))
            if value is not None
        ]
        epoch_type, epoch_type_source = _explicit_epoch_type(task_values, epoch)

        condition_token = _condition_token(task_values)
        condition_source: str | None = None
        if condition_token is not None:
            condition_source = f"task:{task.get('task_source_path')}"
        else:
            condition_token = _condition_token(associated_values)
            if condition_token is not None:
                matching_paths = sorted(
                    {
                        str(record["source_path"])
                        for record in associated
                        if _condition_token(
                            [
                                str(value)
                                for value in (
                                    record.get("name"),
                                    record.get("description"),
                                )
                                if value is not None
                            ]
                        )
                        == condition_token
                    }
                )
                condition_source = "associated_file:" + ",".join(matching_paths)
        if condition_token is None and epoch_type == "sleep":
            condition_token = "sleep"
            condition_source = (
                f"task:{task.get('task_source_path')}"
                if task
                else "epoch_tag"
            )

        condition = (
            _normalize_project_condition(condition_token)
            if condition_token is not None
            else None
        )
        if condition_token in {"stim1", "stim2", "stim3"}:
            condition_source = (
                f"{condition_source};project_normalization:"
                f"{condition_token}->{condition}"
            )
        metadata_by_epoch[epoch] = {
            "task_name": task.get("task_name"),
            "task_description": task.get("task_description"),
            "task_environment": task.get("task_environment"),
            "task_source_path": task.get("task_source_path"),
            "task_object_id": task.get("task_object_id"),
            "associated_file_names": [record["name"] for record in associated],
            "associated_file_descriptions": [
                record["description"] for record in associated
            ],
            "associated_file_source_paths": [
                record["source_path"] for record in associated
            ],
            "epoch_type": epoch_type,
            "epoch_type_source": (
                task.get("task_source_path")
                if epoch_type_source == "task"
                else epoch_type_source
            ),
            "condition": condition,
            "condition_source": condition_source,
            "is_light": _is_light_condition(condition),
        }
    return metadata_by_epoch


def _source_epoch_metadata(nwbfile: Any) -> dict[str, dict[str, Any]]:
    """Return source epochs-table metadata keyed by its single epoch tag."""
    epochs = _interval_table(nwbfile, "epochs")
    if epochs is None:
        raise ValueError("NWB file does not contain an epochs table.")
    _require_columns(epochs, EPOCHS_TABLE_PATH, {"start_time", "stop_time", "tags"})

    column_names = set(_table_column_names(epochs))
    raw_metadata_by_epoch: dict[str, dict[str, Any]] = {}
    epoch_tags: list[str] = []
    for row_index in range(_table_length(epochs)):
        tags = _tags(_table_cell(epochs, "tags", row_index))
        if len(tags) != 1:
            raise ValueError(
                "Expected exactly one tag per NWB epochs row, "
                f"found {len(tags)} at row {row_index}: {tags!r}."
            )
        epoch = tags[0]
        if epoch in raw_metadata_by_epoch:
            raise ValueError(f"NWB epochs table contains duplicate tag {epoch!r}.")
        epoch_tags.append(epoch)
        raw_metadata_by_epoch[epoch] = {
            "tags": tags,
            "epochs_table_epoch_type": (
                _optional_text(_table_cell(epochs, "epoch_type", row_index))
                if "epoch_type" in column_names
                else None
            ),
            "epochs_table_condition": (
                _optional_text(_table_cell(epochs, "condition", row_index))
                if "condition" in column_names
                else None
            ),
            "nwb_epoch_start_time": _float(
                _table_cell(epochs, "start_time", row_index),
                "start_time",
            ),
            "nwb_epoch_stop_time": _float(
                _table_cell(epochs, "stop_time", row_index),
                "stop_time",
            ),
        }
    task_metadata_by_epoch = _task_epoch_metadata(nwbfile, epoch_tags)
    metadata_by_epoch: dict[str, dict[str, Any]] = {}
    for epoch in epoch_tags:
        raw_metadata = raw_metadata_by_epoch[epoch]
        task_metadata = task_metadata_by_epoch[epoch]
        if task_metadata["epoch_type"] is None:
            task_metadata["epoch_type"] = raw_metadata["epochs_table_epoch_type"]
            if task_metadata["epoch_type"] is not None:
                task_metadata["epoch_type_source"] = f"{EPOCHS_TABLE_PATH}#epoch_type"
        if task_metadata["condition"] is None:
            raw_condition = raw_metadata["epochs_table_condition"]
            if raw_condition is not None:
                condition_token = _condition_token([raw_condition])
                if condition_token is None:
                    raise ValueError(
                        f"Unsupported condition {raw_condition!r} in "
                        f"{EPOCHS_TABLE_PATH}; expected AB, gray, BA, dark, "
                        "bright, sleep, or stim1/stim2/stim3."
                    )
                task_metadata["condition"] = _normalize_project_condition(
                    condition_token
                )
                task_metadata["condition_source"] = f"{EPOCHS_TABLE_PATH}#condition"
                if condition_token in {"stim1", "stim2", "stim3"}:
                    task_metadata["condition_source"] += (
                        f";project_normalization:{condition_token}->"
                        f"{task_metadata['condition']}"
                    )
        task_metadata["is_light"] = _is_light_condition(
            task_metadata["condition"]
        )
        metadata_by_epoch[epoch] = {
            "tags": raw_metadata["tags"],
            "nwb_epoch_start_time": raw_metadata["nwb_epoch_start_time"],
            "nwb_epoch_stop_time": raw_metadata["nwb_epoch_stop_time"],
            **task_metadata,
        }
    return metadata_by_epoch


def read_epoch_intervals(
    nwbfile: Any,
    *,
    nwb_file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog augmented ephys bounds enriched by explicit epoch metadata."""
    table = _interval_table(nwbfile, EPHYS_INTERVALS_TABLE_NAME)
    if table is None:
        raise ValueError(
            f"NWB file does not contain {EPHYS_INTERVALS_TABLE_PATH}."
        )
    _require_columns(
        table,
        EPHYS_INTERVALS_TABLE_PATH,
        {"start_time", "stop_time", "epoch"},
    )
    source_metadata = _source_epoch_metadata(nwbfile)
    epochs = _interval_table(nwbfile, "epochs")

    rows: list[dict[str, Any]] = []
    seen_epochs: set[str] = set()
    for row_index in range(_table_length(table)):
        epoch = _text(_table_cell(table, "epoch", row_index), "epoch")
        if epoch in seen_epochs:
            raise ValueError(
                f"NWB table {EPHYS_INTERVALS_TABLE_PATH} contains duplicate epoch {epoch!r}."
            )
        if epoch not in source_metadata:
            raise ValueError(
                f"Epoch {epoch!r} in {EPHYS_INTERVALS_TABLE_PATH} is absent from "
                f"{EPOCHS_TABLE_PATH}."
            )
        seen_epochs.add(epoch)
        start_time, stop_time = _validated_interval_bounds(
            table,
            row_index=row_index,
            path=EPHYS_INTERVALS_TABLE_PATH,
        )
        row = {
            "epoch": epoch,
            "start_time": start_time,
            "stop_time": stop_time,
            **source_metadata[epoch],
            **_source_pointer(EPHYS_INTERVALS_TABLE_PATH, table),
            "metadata_table_path": EPOCHS_TABLE_PATH,
            "metadata_table_object_id": _object_id(epochs),
        }
        _copy_nwb_file_name(row, nwb_file_name)
        rows.append(row)
    return rows


def read_trajectory_intervals(
    nwbfile: Any,
    *,
    nwb_file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog one row per epoch and trajectory type without copying laps."""
    table = _interval_table(nwbfile, TRAJECTORY_INTERVALS_TABLE_NAME)
    if table is None:
        return []
    _require_columns(
        table,
        TRAJECTORY_INTERVALS_TABLE_PATH,
        {"start_time", "stop_time", "epoch", "trajectory_type"},
    )

    grouped_counts: dict[tuple[str, str], int] = {}
    for row_index in range(_table_length(table)):
        _validated_interval_bounds(
            table,
            row_index=row_index,
            path=TRAJECTORY_INTERVALS_TABLE_PATH,
        )
        epoch = _text(_table_cell(table, "epoch", row_index), "epoch")
        trajectory_type = _text(
            _table_cell(table, "trajectory_type", row_index),
            "trajectory_type",
        )
        key = (epoch, trajectory_type)
        grouped_counts[key] = grouped_counts.get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    for (epoch, trajectory_type), interval_count in grouped_counts.items():
        row = {
            "epoch": epoch,
            "trajectory_type": trajectory_type,
            "interval_count": int(interval_count),
            **_source_pointer(TRAJECTORY_INTERVALS_TABLE_PATH, table),
        }
        _copy_nwb_file_name(row, nwb_file_name)
        rows.append(row)
    return rows


def _read_ripple_detection_metadata(nwbfile: Any) -> dict[str, Any]:
    """Read queryable detector settings and a pointer to full provenance."""
    provenance_object = _mapping_value(
        getattr(nwbfile, "scratch", None),
        RIPPLE_PROVENANCE_SCRATCH_NAME,
    )
    if provenance_object is None:
        raise ValueError(
            f"NWB file contains {RIPPLES_INTERVALS_TABLE_PATH} but not "
            f"{RIPPLE_PROVENANCE_PATH}."
        )
    raw_data = getattr(provenance_object, "data", None)
    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode("utf-8")
    if not isinstance(raw_data, str):
        try:
            raw_data = raw_data[()]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Could not read JSON text from {RIPPLE_PROVENANCE_PATH}."
            ) from exc
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")
    if not isinstance(raw_data, str):
        raise ValueError(f"NWB scratch object {RIPPLE_PROVENANCE_PATH} is not text.")
    try:
        provenance = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"NWB scratch object {RIPPLE_PROVENANCE_PATH} is not valid JSON."
        ) from exc

    try:
        parameters = provenance["run_log"]["record"]["parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"NWB scratch object {RIPPLE_PROVENANCE_PATH} has no detector parameters."
        ) from exc
    if not isinstance(parameters, Mapping):
        raise ValueError(
            f"Detector parameters in {RIPPLE_PROVENANCE_PATH} must be a mapping."
        )
    threshold = parameters.get("zscore_threshold")
    speed_gated = parameters.get("use_speed_gating")
    threshold_value = (
        None if threshold is None else _float(threshold, "zscore_threshold")
    )
    if threshold_value is not None and threshold_value <= 0:
        raise ValueError("Ripple detector zscore_threshold must be positive.")
    if speed_gated is not None and not isinstance(
        _native_scalar(speed_gated), bool
    ):
        raise ValueError("Ripple detector use_speed_gating must be boolean.")
    return {
        "detector_zscore_threshold": threshold_value,
        "speed_gated": (
            None if speed_gated is None else bool(_native_scalar(speed_gated))
        ),
        "detection_parameters": dict(parameters),
        "provenance_path": RIPPLE_PROVENANCE_PATH,
        "provenance_object_id": _object_id(provenance_object),
    }


def read_ripples(
    nwbfile: Any,
    *,
    nwb_file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog one row per ripple-bearing epoch without copying events."""
    table = _interval_table(nwbfile, RIPPLES_INTERVALS_TABLE_NAME)
    if table is None:
        return []
    _require_columns(
        table,
        RIPPLES_INTERVALS_TABLE_PATH,
        {"start_time", "stop_time", "epoch"},
    )
    detection_metadata = _read_ripple_detection_metadata(nwbfile)

    counts_by_epoch: dict[str, int] = {}
    for row_index in range(_table_length(table)):
        _validated_interval_bounds(
            table,
            row_index=row_index,
            path=RIPPLES_INTERVALS_TABLE_PATH,
        )
        epoch = _text(_table_cell(table, "epoch", row_index), "epoch")
        counts_by_epoch[epoch] = counts_by_epoch.get(epoch, 0) + 1

    rows: list[dict[str, Any]] = []
    for epoch, ripple_count in counts_by_epoch.items():
        row = {
            "epoch": epoch,
            "ripple_count": int(ripple_count),
            **detection_metadata,
            **_source_pointer(RIPPLES_INTERVALS_TABLE_PATH, table),
        }
        _copy_nwb_file_name(row, nwb_file_name)
        rows.append(row)
    return rows


def read_position_index(
    nwbfile: Any,
    *,
    nwb_file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog half-open epoch ranges for head and body position series."""
    table = _interval_table(nwbfile, POSITION_EPOCHS_TABLE_NAME)
    if table is None:
        return []
    _require_columns(
        table,
        POSITION_EPOCHS_TABLE_PATH,
        {
            "start_time",
            "stop_time",
            "epoch",
            "start_index",
            "stop_index_exclusive",
            "sample_count",
            "analysis_start_offset_samples",
            "first_frame",
            "last_frame",
            "video_series_name",
        },
    )
    position_interface = _processing_interface(
        nwbfile,
        "behavior",
        POSITION_INTERFACE_NAME,
    )
    if position_interface is None:
        raise ValueError(
            f"NWB file contains {POSITION_EPOCHS_TABLE_PATH} but not {POSITION_INTERFACE_PATH}."
        )
    spatial_series = getattr(position_interface, "spatial_series", None)
    series_by_type: dict[str, Any] = {}
    for position_type, series_name in POSITION_SERIES_BY_TYPE.items():
        series = _mapping_value(spatial_series, series_name)
        if series is None:
            raise ValueError(
                f"NWB position interface is missing SpatialSeries {series_name!r}."
            )
        _centimeter_unit(
            getattr(series, "unit", None),
            f"{POSITION_INTERFACE_PATH}/{series_name}",
        )
        series_by_type[position_type] = series

    rows: list[dict[str, Any]] = []
    seen_epochs: set[str] = set()
    for row_index in range(_table_length(table)):
        epoch = _text(_table_cell(table, "epoch", row_index), "epoch")
        if epoch in seen_epochs:
            raise ValueError(
                f"NWB table {POSITION_EPOCHS_TABLE_PATH} contains duplicate epoch {epoch!r}."
            )
        seen_epochs.add(epoch)
        start_index = _integer(
            _table_cell(table, "start_index", row_index),
            "start_index",
        )
        stop_index = _integer(
            _table_cell(table, "stop_index_exclusive", row_index),
            "stop_index_exclusive",
        )
        sample_count = _integer(
            _table_cell(table, "sample_count", row_index),
            "sample_count",
        )
        offset = _integer(
            _table_cell(table, "analysis_start_offset_samples", row_index),
            "analysis_start_offset_samples",
        )
        if start_index < 0 or stop_index < start_index:
            raise ValueError(f"Position index bounds are invalid for epoch {epoch!r}.")
        if sample_count != stop_index - start_index:
            raise ValueError(
                f"Position sample_count does not match index bounds for epoch {epoch!r}."
            )
        if offset < 0 or offset > sample_count:
            raise ValueError(
                f"Position analysis offset is outside the stored samples for epoch {epoch!r}."
            )
        start_time, stop_time = _validated_interval_bounds(
            table,
            row_index=row_index,
            path=POSITION_EPOCHS_TABLE_PATH,
        )

        for position_type, series_name in POSITION_SERIES_BY_TYPE.items():
            series = series_by_type[position_type]
            series_path = f"{POSITION_INTERFACE_PATH}/{series_name}"
            row = {
                "epoch": epoch,
                "position_type": position_type,
                "start_index": start_index,
                "stop_index_exclusive": stop_index,
                "sample_count": sample_count,
                "analysis_start_offset_samples": offset,
                "start_time": start_time,
                "stop_time": stop_time,
                "first_frame": _integer(
                    _table_cell(table, "first_frame", row_index),
                    "first_frame",
                ),
                "last_frame": _integer(
                    _table_cell(table, "last_frame", row_index),
                    "last_frame",
                ),
                "video_series_name": _text(
                    _table_cell(table, "video_series_name", row_index),
                    "video_series_name",
                ),
                "spatial_unit": "cm",
                "source_row_index": int(row_index),
                "source_table_path": POSITION_EPOCHS_TABLE_PATH,
                "source_table_object_id": _object_id(table),
                "source_object_path": series_path,
                "source_object_id": _object_id(series),
            }
            _copy_nwb_file_name(row, nwb_file_name)
            rows.append(row)
    return rows


def read_wtrack_graphs(
    nwbfile: Any,
    *,
    nwb_file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog W-track configurations without reading their graph arrays."""
    table = _processing_interface(
        nwbfile,
        "behavior",
        WTRACK_LINEARIZATION_TABLE_NAME,
    )
    if table is None:
        return []
    _require_columns(
        table,
        WTRACK_LINEARIZATION_TABLE_PATH,
        {
            "configuration_name",
            "node_positions_cm",
            "edges",
            "edge_order",
            "edge_spacing_cm",
            "use_hmm",
        },
    )

    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for row_index in range(_table_length(table)):
        configuration_name = _text(
            _table_cell(table, "configuration_name", row_index),
            "configuration_name",
        )
        if configuration_name in seen_names:
            raise ValueError(
                "NWB W-track table contains duplicate configuration "
                f"{configuration_name!r}."
            )
        seen_names.add(configuration_name)
        row = {
            "configuration_name": configuration_name,
            "coordinate_unit": "cm",
            "use_hmm": _boolean(
                _table_cell(table, "use_hmm", row_index),
                "use_hmm",
            ),
            "source_row_index": int(row_index),
            **_source_pointer(WTRACK_LINEARIZATION_TABLE_PATH, table),
        }
        _copy_nwb_file_name(row, nwb_file_name)
        rows.append(row)
    return rows


def read_spike_sorting_figurls(
    nwbfile: Any,
    *,
    nwb_file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog per-shank spike-sorting FigURLs."""
    table = _processing_interface(
        nwbfile,
        "ecephys",
        SPIKE_SORTING_FIGURLS_TABLE_NAME,
    )
    if table is None:
        return []
    required_columns = {
        "probe_idx",
        "shank_idx",
        "sorter",
        "figurl_url",
        "data_uri",
        "curation_uri",
        "source_file",
    }
    _require_columns(table, SPIKE_SORTING_FIGURLS_TABLE_PATH, required_columns)

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for row_index in range(_table_length(table)):
        probe_idx = _integer(_table_cell(table, "probe_idx", row_index), "probe_idx")
        shank_idx = _integer(_table_cell(table, "shank_idx", row_index), "shank_idx")
        pair = (probe_idx, shank_idx)
        if pair in seen_pairs:
            raise ValueError(
                "NWB spike-sorting FigURL table contains duplicate probe/shank "
                f"pair {pair!r}."
            )
        seen_pairs.add(pair)
        row = {
            "probe_idx": probe_idx,
            "shank_idx": shank_idx,
            "sorter": _text(_table_cell(table, "sorter", row_index), "sorter"),
            "figurl_url": _text(
                _table_cell(table, "figurl_url", row_index),
                "figurl_url",
            ),
            "data_uri": _text(_table_cell(table, "data_uri", row_index), "data_uri"),
            "curation_uri": _text(
                _table_cell(table, "curation_uri", row_index),
                "curation_uri",
            ),
            "source_file": _text(
                _table_cell(table, "source_file", row_index),
                "source_file",
            ),
            "source_row_index": int(row_index),
            **_source_pointer(SPIKE_SORTING_FIGURLS_TABLE_PATH, table),
        }
        _copy_nwb_file_name(row, nwb_file_name)
        rows.append(row)
    return rows


def catalog_augmented_nwb(
    nwbfile: Any,
    nwb_file_name: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return all database-free catalog records from one augmented NWB object."""
    return {
        "epoch_intervals": read_epoch_intervals(
            nwbfile,
            nwb_file_name=nwb_file_name,
        ),
        "trajectory_intervals": read_trajectory_intervals(
            nwbfile,
            nwb_file_name=nwb_file_name,
        ),
        "ripples": read_ripples(nwbfile, nwb_file_name=nwb_file_name),
        "position": read_position_index(
            nwbfile,
            nwb_file_name=nwb_file_name,
        ),
        "wtrack_graph": read_wtrack_graphs(
            nwbfile,
            nwb_file_name=nwb_file_name,
        ),
        "spike_sorting_figurl": read_spike_sorting_figurls(
            nwbfile,
            nwb_file_name=nwb_file_name,
        ),
    }


def _resolve_table_path(nwbfile: Any, path: str) -> Any:
    """Resolve a supported source table path from an NWB-like object."""
    interval_prefix = "/intervals/"
    if path.startswith(interval_prefix):
        table_name = path.removeprefix(interval_prefix)
        table = _interval_table(nwbfile, table_name)
    elif path == WTRACK_LINEARIZATION_TABLE_PATH:
        table = _processing_interface(
            nwbfile,
            "behavior",
            WTRACK_LINEARIZATION_TABLE_NAME,
        )
    elif path == SPIKE_SORTING_FIGURLS_TABLE_PATH:
        table = _processing_interface(
            nwbfile,
            "ecephys",
            SPIKE_SORTING_FIGURLS_TABLE_NAME,
        )
    else:
        table = None
    if table is None:
        raise ValueError(f"Could not resolve NWB source table path {path!r}.")
    return table


def _interval_row_indices(table: Any, catalog_row: Mapping[str, Any]) -> list[int]:
    """Select source interval rows described by one grouped catalog row."""
    column_names = set(_table_column_names(table))
    selector_names = [
        name
        for name in ("epoch", "trajectory_type")
        if name in catalog_row and name in column_names
    ]
    if not selector_names:
        raise ValueError("Interval catalog row does not contain a supported selector.")

    selected: list[int] = []
    for row_index in range(_table_length(table)):
        matches = True
        for selector_name in selector_names:
            source_value = _text(
                _table_cell(table, selector_name, row_index),
                selector_name,
            )
            if source_value != str(catalog_row[selector_name]):
                matches = False
                break
        if matches:
            selected.append(row_index)
    return selected


def load_interval_set(nwbfile: Any, catalog_row: Mapping[str, Any]) -> Any:
    """Load all source intervals selected by one catalog row as an IntervalSet."""
    import pynapple as nap

    path = str(catalog_row.get("source_table_path", ""))
    if path not in {
        EPHYS_INTERVALS_TABLE_PATH,
        TRAJECTORY_INTERVALS_TABLE_PATH,
        RIPPLES_INTERVALS_TABLE_PATH,
    }:
        raise ValueError(f"Catalog row does not point to a supported interval table: {path!r}.")
    table = _resolve_table_path(nwbfile, path)
    _validate_catalog_object_id(
        catalog_row,
        "source_table_object_id",
        table,
        object_label=f"interval table {path}",
    )
    _validate_catalog_object_id(
        catalog_row,
        "source_object_id",
        table,
        object_label=f"interval table {path}",
    )
    _require_columns(table, path, {"start_time", "stop_time"})
    selected_indices = _interval_row_indices(table, catalog_row)
    if not selected_indices:
        raise ValueError(f"Catalog selectors no longer match any intervals in {path}.")

    expected_count = catalog_row.get("interval_count", catalog_row.get("ripple_count"))
    if expected_count is not None and len(selected_indices) != int(expected_count):
        raise ValueError(
            f"Catalog count {expected_count} does not match {len(selected_indices)} "
            f"selected rows in {path}."
        )
    starts = np.asarray(
        [
            _float(_table_cell(table, "start_time", index), "start_time")
            for index in selected_indices
        ],
        dtype=float,
    )
    stops = np.asarray(
        [
            _float(_table_cell(table, "stop_time", index), "stop_time")
            for index in selected_indices
        ],
        dtype=float,
    )
    if np.any(stops <= starts):
        raise ValueError(
            f"NWB interval table {path} contains an interval without positive duration."
        )

    intervals = nap.IntervalSet(start=starts, end=stops, time_units="s")
    metadata: dict[str, list[Any]] = {}
    for column_name in _table_column_names(table):
        if column_name in {"start_time", "stop_time"}:
            continue
        values = [
            _native_scalar(_table_cell(table, column_name, index))
            for index in selected_indices
        ]
        if all(np.isscalar(value) or value is None for value in values):
            metadata[column_name] = values
    if metadata:
        set_info = getattr(intervals, "set_info", None)
        if not callable(set_info):
            raise ValueError("Installed pynapple IntervalSet does not support interval metadata.")
        set_info(**metadata)
    return intervals


def _resolve_position_series(nwbfile: Any, position_type: str) -> Any:
    """Resolve one canonical head or body SpatialSeries."""
    if position_type not in POSITION_SERIES_BY_TYPE:
        raise ValueError(
            f"Unsupported position_type {position_type!r}; expected 'head' or 'body'."
        )
    position_interface = _processing_interface(
        nwbfile,
        "behavior",
        POSITION_INTERFACE_NAME,
    )
    if position_interface is None:
        raise ValueError(f"Could not resolve NWB position interface {POSITION_INTERFACE_PATH}.")
    series_name = POSITION_SERIES_BY_TYPE[position_type]
    series = _mapping_value(getattr(position_interface, "spatial_series", None), series_name)
    if series is None:
        raise ValueError(f"Could not resolve SpatialSeries {series_name!r}.")
    return series


def _slice_timestamps(series: Any, start_index: int, stop_index: int) -> np.ndarray:
    """Load a timestamp slice, following a TimeSeries timestamp link if needed."""
    timestamp_source = getattr(series, "timestamps", None)
    seen_ids: set[int] = set()
    while timestamp_source is not None and hasattr(timestamp_source, "timestamps"):
        source_id = id(timestamp_source)
        if source_id in seen_ids:
            raise ValueError("NWB SpatialSeries contains a cyclic timestamps link.")
        seen_ids.add(source_id)
        nested_source = getattr(timestamp_source, "timestamps")
        if nested_source is timestamp_source:
            break
        timestamp_source = nested_source

    if timestamp_source is not None:
        try:
            return np.asarray(timestamp_source[start_index:stop_index], dtype=float).reshape(-1)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Could not read the requested NWB position timestamps.") from exc

    starting_time = getattr(series, "starting_time", None)
    rate = getattr(series, "rate", None)
    if starting_time is None or rate is None:
        raise ValueError("NWB SpatialSeries does not expose timestamps or starting_time/rate.")
    rate_value = _float(rate, "rate")
    if rate_value <= 0:
        raise ValueError("NWB SpatialSeries rate must be positive.")
    return _float(starting_time, "starting_time") + (
        np.arange(start_index, stop_index, dtype=float) / rate_value
    )


def load_position(
    nwbfile: Any,
    catalog_row: Mapping[str, Any],
    *,
    apply_analysis_offset: bool = True,
) -> Any:
    """Load one indexed position component as a second-based ``nap.TsdFrame``."""
    import pynapple as nap

    position_type = str(catalog_row.get("position_type", ""))
    series = _resolve_position_series(nwbfile, position_type)
    _centimeter_unit(
        getattr(series, "unit", None),
        f"{POSITION_INTERFACE_PATH}/{POSITION_SERIES_BY_TYPE[position_type]}",
    )
    if catalog_row.get("spatial_unit", "cm") != "cm":
        raise ValueError("Position catalog spatial_unit must be 'cm'.")
    position_table = _interval_table(nwbfile, POSITION_EPOCHS_TABLE_NAME)
    if position_table is None:
        raise ValueError(f"Could not resolve {POSITION_EPOCHS_TABLE_PATH}.")
    _require_columns(
        position_table,
        POSITION_EPOCHS_TABLE_PATH,
        {
            "epoch",
            "start_index",
            "stop_index_exclusive",
            "sample_count",
            "analysis_start_offset_samples",
        },
    )
    _validate_catalog_object_id(
        catalog_row,
        "source_table_object_id",
        position_table,
        object_label="position epochs table",
    )
    _validate_catalog_object_id(
        catalog_row,
        "source_object_id",
        series,
        object_label=f"{position_type} position series",
    )

    expected_series_path = (
        f"{POSITION_INTERFACE_PATH}/{POSITION_SERIES_BY_TYPE[position_type]}"
    )
    if catalog_row.get("source_table_path") != POSITION_EPOCHS_TABLE_PATH:
        raise ValueError("Position catalog source_table_path is not canonical.")
    if catalog_row.get("source_object_path") != expected_series_path:
        raise ValueError("Position catalog source_object_path is not canonical.")
    row_index = _integer(catalog_row.get("source_row_index"), "source_row_index")
    if row_index < 0 or row_index >= _table_length(position_table):
        raise ValueError("Position catalog source_row_index is outside the source table.")

    epoch = _text(_table_cell(position_table, "epoch", row_index), "epoch")
    if epoch != str(catalog_row.get("epoch", "")):
        raise ValueError("Position catalog epoch does not match its source row.")
    source_start_index = _integer(
        _table_cell(position_table, "start_index", row_index),
        "start_index",
    )
    source_stop_index = _integer(
        _table_cell(position_table, "stop_index_exclusive", row_index),
        "stop_index_exclusive",
    )
    source_sample_count = _integer(
        _table_cell(position_table, "sample_count", row_index),
        "sample_count",
    )
    source_offset = _integer(
        _table_cell(position_table, "analysis_start_offset_samples", row_index),
        "analysis_start_offset_samples",
    )

    start_index = _integer(catalog_row.get("start_index"), "start_index")
    stop_index = _integer(
        catalog_row.get("stop_index_exclusive"),
        "stop_index_exclusive",
    )
    sample_count = _integer(catalog_row.get("sample_count"), "sample_count")
    analysis_offset = _integer(
        catalog_row.get("analysis_start_offset_samples", 0),
        "analysis_start_offset_samples",
    )
    if (
        start_index != source_start_index
        or stop_index != source_stop_index
        or sample_count != source_sample_count
        or analysis_offset != source_offset
    ):
        raise ValueError("Position catalog bounds do not match their source row.")
    if sample_count != stop_index - start_index:
        raise ValueError("Position catalog sample_count does not match its bounds.")
    if analysis_offset < 0 or analysis_offset > sample_count:
        raise ValueError("Position analysis offset is outside its stored samples.")
    if apply_analysis_offset:
        start_index += analysis_offset
    if start_index < 0 or stop_index < start_index:
        raise ValueError("Position catalog row contains invalid half-open sample bounds.")

    try:
        values = np.asarray(series.data[start_index:stop_index], dtype=float)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("Could not read the requested NWB position samples.") from exc
    timestamps = _slice_timestamps(series, start_index, stop_index)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("NWB position SpatialSeries must have shape (n_samples, 2).")
    if values.shape[0] != timestamps.size:
        raise ValueError("NWB position samples and timestamps have different lengths.")
    if not np.all(np.isfinite(timestamps)) or (
        timestamps.size > 1 and np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("NWB position timestamps must be finite and strictly increasing.")
    expected_sample_count = stop_index - start_index
    if values.shape[0] != expected_sample_count:
        raise ValueError(
            "NWB position slice is shorter than its cataloged half-open bounds: "
            f"expected {expected_sample_count}, found {values.shape[0]}."
        )
    return nap.TsdFrame(
        t=timestamps,
        d=values,
        columns=["x", "y"],
        time_units="s",
    )


def load_wtrack_graph(nwbfile: Any, catalog_row: Mapping[str, Any]) -> dict[str, Any]:
    """Load the NumPy graph inputs selected by one W-track catalog row."""
    table = _resolve_table_path(nwbfile, WTRACK_LINEARIZATION_TABLE_PATH)
    _validate_catalog_object_id(
        catalog_row,
        "source_table_object_id",
        table,
        object_label="W-track table",
    )
    _validate_catalog_object_id(
        catalog_row,
        "source_object_id",
        table,
        object_label="W-track table",
    )
    row_index = _integer(catalog_row.get("source_row_index"), "source_row_index")
    if row_index < 0 or row_index >= _table_length(table):
        raise ValueError("W-track catalog source_row_index is outside the source table.")
    configuration_name = _text(
        _table_cell(table, "configuration_name", row_index),
        "configuration_name",
    )
    if configuration_name != str(catalog_row.get("configuration_name", "")):
        raise ValueError("W-track catalog configuration does not match its source row.")
    if catalog_row.get("coordinate_unit", "cm") != "cm":
        raise ValueError("W-track catalog coordinate_unit must be 'cm'.")

    node_positions = np.asarray(
        _table_cell(table, "node_positions_cm", row_index),
        dtype=float,
    )
    edges = np.asarray(_table_cell(table, "edges", row_index), dtype=np.int64)
    edge_order = np.asarray(
        _table_cell(table, "edge_order", row_index),
        dtype=np.int64,
    )
    edge_spacing = np.asarray(
        _table_cell(table, "edge_spacing_cm", row_index),
        dtype=float,
    ).reshape(-1)
    if node_positions.ndim != 2 or node_positions.shape[1] != 2:
        raise ValueError("W-track node_positions_cm must have shape (n_nodes, 2).")
    if not np.all(np.isfinite(node_positions)):
        raise ValueError("W-track node_positions_cm must contain finite values.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("W-track edges must have shape (n_edges, 2).")
    if edge_order.shape != edges.shape:
        raise ValueError("W-track edge_order must have the same shape as edges.")
    if edge_spacing.size != max(0, len(edge_order) - 1):
        raise ValueError("W-track edge_spacing_cm has an unexpected length.")
    if not np.all(np.isfinite(edge_spacing)) or np.any(edge_spacing < 0):
        raise ValueError("W-track edge_spacing_cm must be finite and non-negative.")
    if edges.size and (
        np.any(edges < 0)
        or np.any(edges >= len(node_positions))
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise ValueError("W-track edges contain invalid node indices.")
    if edge_order.size and (
        np.any(edge_order < 0) or np.any(edge_order >= len(node_positions))
    ):
        raise ValueError("W-track edge_order contains invalid node indices.")
    graph_edges = {frozenset((int(start), int(stop))) for start, stop in edges}
    ordered_edges = {
        frozenset((int(start), int(stop))) for start, stop in edge_order
    }
    if graph_edges != ordered_edges or len(graph_edges) != len(edges):
        raise ValueError("W-track edge_order must order every unique graph edge once.")
    edge_order_list = [tuple(int(value) for value in edge) for edge in edge_order]
    edge_spacing_list = [float(value) for value in edge_spacing]
    use_hmm = _boolean(_table_cell(table, "use_hmm", row_index), "use_hmm")
    return {
        "configuration_name": configuration_name,
        "coordinate_unit": "cm",
        "node_positions_cm": node_positions,
        "edges": edges,
        "edge_order": edge_order,
        "edge_spacing_cm": edge_spacing,
        "use_hmm": use_hmm,
        "track_graph_kwargs": {
            "node_positions": node_positions,
            "edges": edges,
        },
        "linearization_kwargs": {
            "edge_order": edge_order_list,
            "edge_spacing": edge_spacing_list,
            "use_HMM": use_hmm,
        },
    }


__all__ = [
    "catalog_augmented_nwb",
    "load_interval_set",
    "load_position",
    "load_wtrack_graph",
    "read_epoch_intervals",
    "read_position_index",
    "read_ripples",
    "read_spike_sorting_figurls",
    "read_trajectory_intervals",
    "read_wtrack_graphs",
]
