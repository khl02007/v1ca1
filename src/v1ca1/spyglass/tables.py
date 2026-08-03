"""Explicit activation for the project-owned Spyglass tables.

Importing this module is passive: DataJoint and Spyglass are imported only by
``activate``. Runtime computation is likewise reached only through explicitly
activated computed tables.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
import hashlib
import math
from numbers import Real
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from v1ca1.spyglass import table_specs


SOURCE_TABLE_KEYS = (
    "epoch_intervals",
    "trajectory_intervals",
    "ripples",
    "position",
    "wtrack_graph",
    "spike_sorting_figurl",
)


def _validate_parameter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one all-scalar RippleModulation parameter row."""
    expected = set(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "RippleModulation parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )

    values = dict(row)
    name = values["ripple_modulation_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "ripple_modulation_param_name must be a non-empty string of at most 64 characters."
        )

    numeric_names = (
        "bin_size_s",
        "time_before_s",
        "time_after_s",
        "response_window_start_s",
        "response_window_end_s",
        "baseline_window_start_s",
        "baseline_window_end_s",
        "expected_detector_zscore_threshold",
    )
    for field_name in numeric_names:
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value

    for field_name in ("bin_size_s", "time_before_s", "time_after_s"):
        if values[field_name] <= 0:
            raise ValueError(f"{field_name} must be positive.")
    if values["response_window_start_s"] >= values["response_window_end_s"]:
        raise ValueError("response window start must be smaller than its end.")
    if values["baseline_window_start_s"] >= values["baseline_window_end_s"]:
        raise ValueError("baseline window start must be smaller than its end.")

    lower_bound = -values["time_before_s"]
    upper_bound = values["time_after_s"]
    for prefix in ("response", "baseline"):
        start = values[f"{prefix}_window_start_s"]
        stop = values[f"{prefix}_window_end_s"]
        if start < lower_bound or stop > upper_bound:
            raise ValueError(
                f"{prefix} window {(start, stop)!r} lies outside "
                f"the peri-ripple window {(lower_bound, upper_bound)!r}."
            )

    if values["heatmap_normalize"] not in {"max", "zscore"}:
        raise ValueError("heatmap_normalize must be 'max' or 'zscore'.")
    if values["expected_detector_zscore_threshold"] <= 0:
        raise ValueError("expected_detector_zscore_threshold must be positive.")
    require_speed_gated = values["require_speed_gated"]
    if isinstance(require_speed_gated, (str, bytes, list, tuple, dict)):
        raise TypeError("require_speed_gated must be a bool scalar.")
    try:
        is_boolean_scalar = require_speed_gated in (True, False)
    except (TypeError, ValueError):
        is_boolean_scalar = False
    if not is_boolean_scalar:
        raise TypeError("require_speed_gated must be a bool scalar.")
    values["require_speed_gated"] = bool(require_speed_gated)
    return values


def _validate_stability_parameter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one TaskProgressionStability parameter row."""
    expected = set(table_specs.DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "TaskProgressionStability parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["task_progression_stability_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "task_progression_stability_param_name must be a non-empty string "
            "of at most 64 characters."
        )
    for field_name in (
        "speed_threshold_cm_s",
        "speed_smoothing_sigma_s",
        "place_bin_size_cm",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be positive and finite.")
        values[field_name] = value
    return values


def _fetch1_dict(table: Any, key: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch one relation row as a plain dictionary."""
    row = (table & dict(key)).fetch1()
    if not isinstance(row, Mapping):
        raise TypeError(f"{table!r}.fetch1() must return a mapping.")
    return dict(row)


def _load_catalog_nwb_object(
    table: Any,
    key: Mapping[str, Any],
    *,
    nwbfile_table: Any,
    loader: Callable[..., Any],
    loader_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Open one row's source NWB read-only and run its database-free loader."""
    import pynwb

    row = _fetch1_dict(table, key)
    nwb_file_name = row.get("nwb_file_name", key.get("nwb_file_name"))
    if nwb_file_name is None:
        raise ValueError("Source-table key does not identify an nwb_file_name.")
    nwb_path = Path(nwbfile_table.get_abs_path(str(nwb_file_name)))
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        return loader(io.read(), row, **dict(loader_kwargs or {}))


def _session_identity(session_table: Any, key: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve artifact identity from standard Session metadata."""
    subject_id, start_time = (session_table & dict(key)).fetch1(
        "subject_id",
        "session_start_time",
    )
    if subject_id is None or not str(subject_id).strip():
        raise ValueError(
            "Session.subject_id is required for RippleModulation artifact paths."
        )
    if isinstance(start_time, (datetime, date)):
        session_date = start_time.strftime("%Y%m%d")
    elif hasattr(start_time, "strftime"):
        session_date = start_time.strftime("%Y%m%d")
    else:
        raise TypeError("Session.session_start_time must provide strftime().")
    return str(subject_id), str(session_date)


def _git_commit(path: Path) -> str | None:
    """Return the repository HEAD containing ``path``, if available."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one existing artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_created_artifacts(paths: list[str]) -> None:
    """Remove only exact artifact files created for a failed table insert."""
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            path.unlink()


def _existing_result_row(
    table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an existing result before any registration-side file writes."""
    try:
        relation = table & dict(key)
    except (AttributeError, TypeError):
        # Minimal injected table fakes used by dependency-free tests do not
        # implement DataJoint relation operators.
        return None
    if not relation:
        return None
    row = relation.fetch1()
    if not isinstance(row, Mapping):
        raise TypeError("Existing computed result must fetch as a mapping.")
    return dict(row)


def _v1ca1_git_commit() -> str | None:
    """Return the local V1-CA1 HEAD without enforcing a particular commit."""
    return _git_commit(Path(__file__).resolve().parents[3])


def _spyglass_git_commit() -> str | None:
    """Return the runtime Spyglass checkout HEAD without enforcing the pin."""
    try:
        import spyglass
    except ModuleNotFoundError:
        return None
    package_path = getattr(spyglass, "__file__", None)
    if package_path is None:
        return None
    return _git_commit(Path(package_path).resolve().parent)


def _intervals_to_frame(intervals: Any, *, epoch: str) -> Any:
    """Convert a Pynapple-like IntervalSet to a detector-table dataframe."""
    import pandas as pd

    as_dataframe = getattr(intervals, "as_dataframe", None)
    if callable(as_dataframe):
        frame = as_dataframe().reset_index(drop=True)
        frame = frame.rename(
            columns={"start": "start_time", "end": "end_time", "stop": "end_time"}
        )
    else:
        starts = getattr(intervals, "start", None)
        stops = getattr(intervals, "end", None)
        if starts is None or stops is None:
            raise TypeError("Loaded ripple intervals do not expose start/end values.")
        frame = pd.DataFrame({"start_time": starts, "end_time": stops})
    missing = [name for name in ("start_time", "end_time") if name not in frame]
    if missing:
        raise ValueError(f"Loaded ripple intervals are missing columns {missing!r}.")
    frame["epoch"] = str(epoch)
    return frame


def _validate_ripple_provenance(
    ripple_row: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> None:
    """Match selected detector metadata to explicit upstream expectations."""
    actual_threshold = ripple_row.get("detector_zscore_threshold")
    if actual_threshold is None or not math.isclose(
        float(actual_threshold),
        float(parameters["expected_detector_zscore_threshold"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Ripples.detector_zscore_threshold does not match "
            "expected_detector_zscore_threshold."
        )
    speed_gated = ripple_row.get("speed_gated")
    if parameters["require_speed_gated"] and (
        speed_gated is None or not bool(speed_gated)
    ):
        raise ValueError("Selected Ripples row must be explicitly speed-gated.")


def _parameter_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate stored scalar columns to the database-free analysis API."""
    return {
        "bin_size_s": parameters["bin_size_s"],
        "time_before_s": parameters["time_before_s"],
        "time_after_s": parameters["time_after_s"],
        "response_window": (
            parameters["response_window_start_s"],
            parameters["response_window_end_s"],
        ),
        "baseline_window": (
            parameters["baseline_window_start_s"],
            parameters["baseline_window_end_s"],
        ),
    }


def _analysis_region(value: Any) -> str:
    """Require the canonical lowercase project analysis region."""
    region = str(value).strip()
    if region not in {"v1", "ca1"}:
        raise ValueError("region must be the canonical lowercase 'v1' or 'ca1'.")
    return region


def _sorting_snapshot_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return selection columns for one resolved sorting-group snapshot."""
    parameters = dict(provenance["unit_selection_params"])
    return {
        "sorting_group_members": list(provenance["sorting_group_members"]),
        "sorting_group_members_sha256": str(
            provenance["sorting_group_members_sha256"]
        ),
        "unit_filter_include_labels": list(parameters["include_labels"]),
        "unit_filter_exclude_labels": list(parameters["exclude_labels"]),
        "unit_filter_params_sha256": str(
            provenance["unit_selection_params_sha256"]
        ),
    }


def _resolve_sorting_snapshot(
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one standard group and its immutable label-filter snapshot."""
    from v1ca1.spyglass.spikes import resolve_sorted_spikes_group_provenance

    return resolve_sorted_spikes_group_provenance(
        sorted_spikes_group,
        unit_selection_params,
        key,
    )


def _validate_frozen_sorting_snapshot(
    selection: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Require current group membership/filter values to match a selection."""
    expected = _sorting_snapshot_fields(provenance)
    for field_name, current_value in expected.items():
        selected_value = selection.get(field_name)
        if field_name.endswith("labels") or field_name == "sorting_group_members":
            selected_value = list(selected_value or ())
        if selected_value != current_value:
            raise ValueError(
                "SortedSpikesGroup membership or UnitSelectionParams changed "
                f"after selection insertion: {field_name}. Create a new selection."
            )


def _parameter_snapshot_field(
    parameters: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, str]:
    """Return one immutable parameter-value digest for a selection row."""
    from v1ca1.spyglass.selection import provenance_sha256

    return {field_name: provenance_sha256(dict(parameters))}


def _validate_frozen_parameters(
    selection: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    """Require current Manual parameters to match their selection snapshot."""
    current = _parameter_snapshot_field(parameters, field_name=field_name)[field_name]
    if str(selection.get(field_name, "")) != current:
        raise ValueError(
            "Analysis parameters changed after selection insertion: "
            f"{field_name}. Create a new selection."
        )


def _ripple_modulation_selection_row(
    *,
    key: Mapping[str, Any],
    ripples_table: Any,
    epoch_intervals_table: Any,
    parameters_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable RippleModulation selection."""
    from v1ca1.spyglass.selection import selection_uuid

    natural_key = {
        field_name: key[field_name]
        for field_name in (
            "nwb_file_name",
            "epoch",
            "ripple_modulation_param_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
        )
    }
    natural_key["region"] = _analysis_region(key["region"])
    _fetch1_dict(ripples_table, natural_key)
    _fetch1_dict(epoch_intervals_table, natural_key)
    parameters = _validate_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    provenance = _resolve_sorting_snapshot(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        key=natural_key,
    )
    snapshot = _sorting_snapshot_fields(provenance)
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    identity_payload = {**natural_key, **snapshot, **parameter_snapshot}
    return {
        "ripple_modulation_id": selection_uuid(
            "RippleModulation",
            identity_payload,
        ),
        **natural_key,
        **snapshot,
        **parameter_snapshot,
    }


def _stability_selection_row(
    *,
    key: Mapping[str, Any],
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    position_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable trajectory-stability selection."""
    from v1ca1.spyglass.selection import selection_uuid

    natural_key = {
        field_name: key[field_name]
        for field_name in (
            "nwb_file_name",
            "epoch",
            "trajectory_type",
            "position_series_name",
            "configuration_name",
            "task_progression_stability_param_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
        )
    }
    natural_key["region"] = _analysis_region(key["region"])
    epoch_row = _fetch1_dict(epoch_intervals_table, natural_key)
    _fetch1_dict(trajectory_intervals_table, natural_key)
    position_row = _fetch1_dict(position_table, natural_key)
    graph_row = _fetch1_dict(wtrack_graph_table, natural_key)
    parameters = _validate_stability_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    if natural_key["trajectory_type"] != natural_key["configuration_name"]:
        raise ValueError(
            "TaskProgressionStability requires configuration_name to equal "
            "trajectory_type."
        )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("TaskProgressionStability requires a run epoch.")
    if position_row.get("spatial_unit") != "cm":
        raise ValueError("TaskProgressionStability position must use centimeters.")
    if graph_row.get("coordinate_unit") != "cm":
        raise ValueError("TaskProgressionStability graph must use centimeters.")
    provenance = _resolve_sorting_snapshot(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        key=natural_key,
    )
    snapshot = _sorting_snapshot_fields(provenance)
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="task_progression_stability_parameters_sha256",
    )
    identity_payload = {**natural_key, **snapshot, **parameter_snapshot}
    return {
        "task_progression_stability_id": selection_uuid(
            "TaskProgressionStability",
            identity_payload,
        ),
        **natural_key,
        **snapshot,
        **parameter_snapshot,
    }


def _sorted_spikes_group_key(key: Mapping[str, Any]) -> dict[str, Any]:
    """Return the session-constrained standard sorting-group key."""
    required = (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
    )
    missing = [name for name in required if name not in key]
    if missing:
        raise ValueError(f"RippleModulation selection is missing group keys {missing!r}.")
    return {name: key[name] for name in required}


def _load_group_unit_data(
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    region: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Load one group through the shared strict sorting adapter."""
    from v1ca1.spyglass.spikes import load_sorted_spikes_group

    return load_sorted_spikes_group(
        sorted_spikes_group,
        unit_selection_params,
        spike_sorting_output,
        key,
        region=region,
        allow_empty=allow_empty,
    )


def _load_group_spikes(
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    region: str,
    time_support: tuple[float, float],
) -> dict[str, Any]:
    """Build one Pynapple TsGroup from all validated sorting-group members."""
    from v1ca1.spyglass.spikes import load_sorted_spikes_group

    return load_sorted_spikes_group(
        sorted_spikes_group,
        unit_selection_params,
        spike_sorting_output,
        key,
        region=region,
        time_support=time_support,
        allow_empty=True,
    )


def _make_ripple_modulation_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one keyed RippleModulation result."""
    import pynwb

    from v1ca1.spyglass.nwb import load_interval_set
    from v1ca1.spyglass.ripple_modulation import (
        compute_epoch_region_ripple_modulation,
        empty_ripple_modulation_result,
        get_ripple_modulation_artifact_paths,
        write_ripple_modulation_artifacts,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    parameters = _validate_parameter_row(_fetch1_dict(parameters_table, key))
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    ripple_row = _fetch1_dict(ripples_table, key)
    _validate_ripple_provenance(ripple_row, parameters)
    epoch_row = _fetch1_dict(epoch_intervals_table, key)
    animal_name, session_date = _session_identity(session_table, key)
    nwb_path = Path(nwbfile_table.get_abs_path(str(key["nwb_file_name"])))

    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if (
        not math.isfinite(epoch_start)
        or not math.isfinite(epoch_stop)
        or epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        ripple_intervals = load_interval_set(io.read(), ripple_row)
    ripple_table = _intervals_to_frame(ripple_intervals, epoch=str(key["epoch"]))
    region = _analysis_region(key["region"])

    loaded_spikes = _load_group_spikes(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=region,
        time_support=(epoch_start, epoch_stop),
    )
    _validate_frozen_sorting_snapshot(key, loaded_spikes)
    if loaded_spikes["status"] == "no_units":
        result = empty_ripple_modulation_result(
            animal_name=animal_name,
            date=session_date,
            epoch=str(key["epoch"]),
            region=region,
            n_ripples=int(ripple_row["ripple_count"]),
            parameters=_parameter_kwargs(parameters),
        )
    else:
        result = compute_epoch_region_ripple_modulation(
            animal_name=animal_name,
            date=session_date,
            epoch=str(key["epoch"]),
            region=region,
            ripple_table=ripple_table,
            epoch_timestamps=[epoch_start, epoch_stop],
            region_spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            **_parameter_kwargs(parameters),
        )
    if not result["summary"].empty:
        summary = result["summary"].copy()
        missing_reason = summary["invalid_reason"].isna()
        nonfinite_response = ~np.isfinite(
            summary["response_zscore"].to_numpy(dtype=float)
        )
        summary.loc[missing_reason & nonfinite_response, "invalid_reason"] = (
            "nonfinite_response_zscore"
        )
        result = {**result, "summary": summary}
    if int(result["n_ripples"]) != int(ripple_row["ripple_count"]):
        raise ValueError(
            "RippleModulation ripple count does not match the selected "
            "Ripples catalog row."
        )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_ripple_modulation_artifact_paths(
        animal_name=animal_name,
        date=session_date,
        epoch=str(key["epoch"]),
        region=region,
        ripple_modulation_id=key["ripple_modulation_id"],
        **path_kwargs,
    )
    created_artifact_paths = [
        str(paths[name])
        for name in ("summary", "peri_ripple_firing_rate")
        if not Path(paths[name]).exists()
    ]
    written = write_ripple_modulation_artifacts(result, paths)
    n_units = int(loaded_spikes["n_units"])
    if n_units == 0:
        analysis_status = "no_units"
        n_valid_units = 0
    elif int(result["n_ripples"]) == 0:
        analysis_status = "no_ripples"
        n_valid_units = 0
    else:
        reasons = result["summary"]["invalid_reason"]
        n_valid_units = int(reasons.isna().sum())
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    return {
        "summary_path": str(written["summary"]),
        "peri_ripple_firing_rate_path": str(written["peri_ripple_firing_rate"]),
        "n_ripples": int(result["n_ripples"]),
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_spikes["unit_ids"]),
        "legacy_artifact_provenance": None,
        "artifact_origin": "computed",
        "_created_artifact_paths": created_artifact_paths,
    }


def _filter_registered_table(
    table: Any,
    *,
    artifact_name: str,
    artifact_key: Mapping[str, str],
    parameters: Mapping[str, Any],
    allow_empty: bool = False,
) -> Any:
    """Select and validate one key from a legacy single- or all-region table."""
    required_key_columns = tuple(artifact_key)
    missing = [column for column in required_key_columns if column not in table.columns]
    if missing:
        raise ValueError(f"{artifact_name} parquet is missing key columns {missing!r}.")

    include = None
    for column, expected in artifact_key.items():
        matches = table[column].astype(str) == str(expected)
        include = matches if include is None else include & matches
    selected = table.loc[include].copy().reset_index(drop=True)
    if selected.empty:
        if allow_empty and table.empty:
            return selected
        raise ValueError(
            f"{artifact_name} parquet has no rows for artifact key {dict(artifact_key)!r}."
        )
    for column, expected in artifact_key.items():
        unique_values = selected[column].astype(str).unique().tolist()
        if unique_values != [str(expected)]:
            raise ValueError(
                f"{artifact_name} parquet has ambiguous {column}: {unique_values!r}."
            )

    for column in ("bin_size_s", "time_before_s", "time_after_s"):
        if column not in selected.columns:
            raise ValueError(f"{artifact_name} parquet is missing parameter column {column!r}.")
        unique_values = selected[column].dropna().astype(float).unique().tolist()
        if len(unique_values) != 1 or not math.isclose(
            unique_values[0],
            float(parameters[column]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{artifact_name} parquet {column} does not match the selected parameters."
            )
    if "n_ripples" not in selected.columns:
        raise ValueError(f"{artifact_name} parquet is missing n_ripples.")
    ripple_counts = selected["n_ripples"].dropna().astype(int).unique().tolist()
    if len(ripple_counts) != 1 or ripple_counts[0] < 0:
        raise ValueError(f"{artifact_name} parquet has ambiguous n_ripples values.")
    return selected


def _attach_registered_unit_identity(
    table: Any,
    *,
    unit_metadata: list[Mapping[str, Any]],
    artifact_name: str,
) -> Any:
    """Key a legacy artifact to stable sorting-merge and NWB unit ids."""
    import pandas as pd

    if "unit_id" not in table.columns:
        raise ValueError(f"{artifact_name} parquet is missing unit_id.")

    catalog_by_stable_id: dict[tuple[str, str], dict[str, Any]] = {}
    catalog_by_sorting_unit_id: dict[str, list[dict[str, Any]]] = {}
    for raw_metadata in unit_metadata:
        metadata = dict(raw_metadata)
        try:
            stable_id = (
                str(metadata["spikesorting_merge_id"]),
                str(metadata["unit_id"]),
            )
        except KeyError as exc:
            raise ValueError("Loaded unit metadata lacks stable unit identity.") from exc
        if stable_id in catalog_by_stable_id:
            raise ValueError(f"Duplicate loaded stable unit identity {stable_id!r}.")
        catalog_by_stable_id[stable_id] = metadata
        if metadata.get("sorting_unit_id") is not None:
            catalog_by_sorting_unit_id.setdefault(
                str(metadata["sorting_unit_id"]), []
            ).append(metadata)

    if not catalog_by_stable_id:
        if not table.empty:
            raise ValueError("Selected SortedSpikesGroup has no unit metadata.")
        output = table.copy()
        output["group_unit_id"] = output["unit_id"].to_numpy(copy=True)
        output["spikesorting_merge_id"] = pd.Series(dtype=str)
        output["unit_id"] = pd.Series(dtype=str)
        output["stable_unit_id"] = pd.Series(dtype=str)
        return output
    row_metadata: list[dict[str, Any]] = []
    has_old_explicit = {
        "spikesorting_merge_id",
        "nwb_unit_id",
    }.issubset(table.columns)
    has_new_explicit = (
        "spikesorting_merge_id" in table.columns
        and "unit_id" in table.columns
        and not has_old_explicit
    )
    if has_old_explicit or has_new_explicit:
        source_unit_column = "nwb_unit_id" if has_old_explicit else "unit_id"
        stable_pairs = zip(
            table["spikesorting_merge_id"].astype(str),
            table[source_unit_column].astype(str),
        )
        for stable_id in stable_pairs:
            if stable_id not in catalog_by_stable_id:
                raise ValueError(
                    f"{artifact_name} contains unit {stable_id!r} outside the "
                    "selected SortedSpikesGroup and region."
                )
            row_metadata.append(catalog_by_stable_id[stable_id])
    else:
        ambiguous_sorting_ids = {
            sorting_unit_id
            for sorting_unit_id, records in catalog_by_sorting_unit_id.items()
            if len(records) != 1
        }
        legacy_ids = table["unit_id"].astype(str).to_list()
        missing_legacy_ids = sorted(
            {
                unit_id
                for unit_id in legacy_ids
                if unit_id not in catalog_by_sorting_unit_id
                or unit_id in ambiguous_sorting_ids
            }
        )
        if missing_legacy_ids:
            raise ValueError(
                f"{artifact_name} legacy unit_id values cannot be mapped uniquely "
                "through the augmented NWB sorting_unit_id column: "
                f"{missing_legacy_ids!r}. Supply artifacts with explicit "
                "spikesorting_merge_id and nwb_unit_id columns."
            )
        row_metadata = [
            catalog_by_sorting_unit_id[unit_id][0] for unit_id in legacy_ids
        ]

    output = table.copy()
    if "unit_id" in output:
        output["group_unit_id"] = output["unit_id"].to_numpy(copy=True)
    output["spikesorting_merge_id"] = [
        str(metadata["spikesorting_merge_id"]) for metadata in row_metadata
    ]
    output["unit_id"] = [str(metadata["unit_id"]) for metadata in row_metadata]
    output["stable_unit_id"] = [
        f"{merge_id}:{source_unit_id}"
        for merge_id, source_unit_id in zip(
            output["spikesorting_merge_id"],
            output["unit_id"],
        )
    ]
    output = output.drop(columns=["nwb_unit_id"], errors="ignore")
    return output


def _register_existing_ripple_modulation_row(
    *,
    key: Mapping[str, Any],
    summary_path: Path,
    peri_ripple_firing_rate_path: Path,
    overwrite: bool,
    parameters_table: Any,
    ripples_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Validate, key, and write existing RippleModulation Parquets."""
    from v1ca1.spyglass.ripple_modulation import (
        plan_register_existing,
        read_planned_artifacts,
        write_planned_artifacts,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent

    parameters = _validate_parameter_row(_fetch1_dict(parameters_table, key))
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    ripple_row = _fetch1_dict(ripples_table, key)
    _validate_ripple_provenance(ripple_row, parameters)
    animal_name, session_date = _session_identity(session_table, key)
    artifact_key = {
        "animal_name": animal_name,
        "date": session_date,
        "epoch": str(key["epoch"]),
        "region": _analysis_region(key["region"]),
    }
    plan_kwargs = {
        **artifact_key,
        "existing_summary_path": Path(summary_path),
        "existing_peri_ripple_firing_rate_path": Path(
            peri_ripple_firing_rate_path
        ),
        "ripple_modulation_id": key["ripple_modulation_id"],
        **_parameter_kwargs(parameters),
        "heatmap_normalize": parameters["heatmap_normalize"],
    }
    if artifact_root is not None:
        plan_kwargs["artifact_root"] = artifact_root
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=artifact_key["region"],
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(key, loaded_units)
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(
            spike_sorting_output,
            {"merge_id": merge_id},
        )
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy RippleModulation registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )
    allow_empty_artifacts = (
        loaded_units["status"] == "no_units"
        or int(ripple_row["ripple_count"]) == 0
    )
    plan = plan_register_existing(**plan_kwargs)
    selected_tables = read_planned_artifacts(
        plan,
        allow_unkeyed_same_path=overwrite,
        allow_empty=allow_empty_artifacts,
    )
    legacy_artifact_provenance = {
        "summary": {
            "source_path": str(Path(summary_path).resolve(strict=True)),
            "sha256": _file_sha256(Path(summary_path)),
        },
        "peri_ripple_firing_rate": {
            "source_path": str(
                Path(peri_ripple_firing_rate_path).resolve(strict=True)
            ),
            "sha256": _file_sha256(Path(peri_ripple_firing_rate_path)),
        },
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "source_spyglass_git_commit": source_spyglass_git_commit,
    }
    selected_summary = _filter_registered_table(
        selected_tables["summary"],
        artifact_name="summary",
        artifact_key=artifact_key,
        parameters=parameters,
        allow_empty=allow_empty_artifacts,
    )
    selected_peri = _filter_registered_table(
        selected_tables["peri_ripple_firing_rate"],
        artifact_name="peri_ripple_firing_rate",
        artifact_key=artifact_key,
        parameters=parameters,
        allow_empty=allow_empty_artifacts,
    )

    selected_summary = _attach_registered_unit_identity(
        selected_summary,
        unit_metadata=loaded_units["unit_metadata"],
        artifact_name="summary",
    )
    selected_peri = _attach_registered_unit_identity(
        selected_peri,
        unit_metadata=loaded_units["unit_metadata"],
        artifact_name="peri_ripple_firing_rate",
    )
    prepared_tables = {
        "summary": selected_summary,
        "peri_ripple_firing_rate": selected_peri,
    }
    for copy in plan["copies"]:
        artifact_name = str(copy["artifact"])
        if (
            not copy.get("copy_required", True)
            and not overwrite
            and not selected_tables[artifact_name].equals(
                prepared_tables[artifact_name]
            )
        ):
            raise ValueError(
                "A same-path registration source requires stable-unit "
                f"normalization and cannot be registered in place: {copy['source']}."
            )

    summary_units = set(selected_summary["stable_unit_id"].astype(str))
    peri_units = set(selected_peri["stable_unit_id"].astype(str))
    catalog_units = {
        f"{unit['spikesorting_merge_id']}:{unit['unit_id']}"
        for unit in loaded_units["unit_ids"]
    }
    no_ripples = int(ripple_row["ripple_count"]) == 0
    if no_ripples and (not selected_summary.empty or not selected_peri.empty):
        raise ValueError(
            "Zero-ripple legacy artifacts must contain canonical empty tables."
        )
    if not no_ripples and (
        len(selected_summary) != len(summary_units)
        or summary_units != peri_units
        or summary_units != catalog_units
    ):
        raise ValueError(
            "Existing summary must contain one row per selected "
            "SortedSpikesGroup unit, and both artifacts must contain exactly "
            "the same units."
        )
    if not selected_summary.empty:
        summary_n_ripples = int(selected_summary["n_ripples"].iloc[0])
        peri_n_ripples = int(selected_peri["n_ripples"].iloc[0])
    else:
        summary_n_ripples = int(ripple_row["ripple_count"])
        peri_n_ripples = summary_n_ripples
    if summary_n_ripples != peri_n_ripples:
        raise ValueError("Existing artifacts disagree on n_ripples.")
    if (
        summary_n_ripples != int(ripple_row["ripple_count"])
    ):
        raise ValueError(
            "Existing artifact n_ripples does not match the selected "
            "Ripples catalog row."
        )
    created_artifact_paths = [
        str(copy["destination"])
        for copy in plan["copies"]
        if (copy.get("copy_required", True) or overwrite)
        and not Path(copy["destination"]).exists()
    ]
    destinations = write_planned_artifacts(
        plan,
        prepared_tables,
        overwrite=overwrite,
    )
    reasons = selected_summary["invalid_reason"]
    n_valid_units = int(reasons.isna().sum())
    n_units = len(catalog_units)
    if n_units == 0:
        analysis_status = "no_units"
    elif summary_n_ripples == 0:
        analysis_status = "no_ripples"
    else:
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    return {
        "summary_path": str(destinations["summary"]),
        "peri_ripple_firing_rate_path": str(destinations["peri_ripple_firing_rate"]),
        "n_ripples": summary_n_ripples,
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_units["unit_ids"]),
        "legacy_artifact_provenance": legacy_artifact_provenance,
        "artifact_origin": "registered_existing",
        "_created_artifact_paths": created_artifact_paths,
    }


def _make_task_progression_stability_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    position_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one trajectory-level stability result."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.stability import (
        compute_selected_stability,
        get_stability_artifact_path,
        write_stability_artifact,
    )

    parameters = _validate_stability_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="task_progression_stability_parameters_sha256",
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, key)
    trajectory_row = _fetch1_dict(trajectory_intervals_table, key)
    position_row = _fetch1_dict(position_table, key)
    graph_row = _fetch1_dict(wtrack_graph_table, key)
    if str(key["trajectory_type"]) != str(key["configuration_name"]):
        raise ValueError(
            "TaskProgressionStability graph configuration must match trajectory_type."
        )
    animal_name, session_date = _session_identity(session_table, key)
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")

    loaded_spikes = _load_group_spikes(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=_analysis_region(key["region"]),
        time_support=(epoch_start, epoch_stop),
    )
    _validate_frozen_sorting_snapshot(key, loaded_spikes)

    nwb_path = Path(nwbfile_table.get_abs_path(str(key["nwb_file_name"])))
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        position = load_position(nwbfile, position_row, apply_analysis_offset=True)
        trajectory_interval = load_interval_set(nwbfile, trajectory_row)
        graph_inputs = load_wtrack_graph(nwbfile, graph_row)

    result = compute_selected_stability(
        animal_name=animal_name,
        date=session_date,
        region=_analysis_region(key["region"]),
        epoch=str(key["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=position,
        trajectory_interval=trajectory_interval,
        graph_inputs=graph_inputs,
        speed_threshold_cm_s=parameters["speed_threshold_cm_s"],
        speed_smoothing_sigma_s=parameters["speed_smoothing_sigma_s"],
        place_bin_size_cm=parameters["place_bin_size_cm"],
    )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    artifact_path = get_stability_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(key["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        region=_analysis_region(key["region"]),
        task_progression_stability_id=key["task_progression_stability_id"],
        **path_kwargs,
    )
    created_artifact_paths = [] if artifact_path.exists() else [str(artifact_path)]
    written_path = write_stability_artifact(result["table"], artifact_path)
    return {
        "stability_path": str(written_path),
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": unit_identity_sha256(loaded_spikes["unit_ids"]),
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


def _validate_legacy_stability_schema(table: Any) -> None:
    """Require every canonical QC field in a legacy stability artifact."""
    from v1ca1.spyglass.stability import empty_stability_table

    required_columns = set(empty_stability_table().columns).difference(
        {
            "spikesorting_merge_id",
            "unit_id",
            "stable_unit_id",
            "group_unit_id",
        }
    )
    required_columns.add("unit")
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(
            f"Existing stability artifact is missing canonical columns {missing!r}."
        )


def _register_existing_task_progression_stability_row(
    *,
    key: Mapping[str, Any],
    stability_path: Path,
    overwrite: bool,
    parameters_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Filter and register one partition of the complete legacy artifact."""
    import pandas as pd

    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent
    from v1ca1.spyglass.stability import (
        empty_stability_table,
        get_stability_artifact_path,
        write_stability_artifact,
    )

    parameters = _validate_stability_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="task_progression_stability_parameters_sha256",
    )
    default_parameters = dict(
        table_specs.DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS
    )
    for field_name in (
        "speed_threshold_cm_s",
        "speed_smoothing_sigma_s",
        "place_bin_size_cm",
    ):
        if not math.isclose(
            parameters[field_name],
            default_parameters[field_name],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Legacy stability registration is only valid for the regenerated "
                f"default parameters; {field_name} differs."
            )
    animal_name, session_date = _session_identity(session_table, key)
    region = _analysis_region(key["region"])
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=region,
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(key, loaded_units)
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(spike_sorting_output, {"merge_id": merge_id})
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy stability registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )

    stability_path = Path(stability_path)
    if not stability_path.is_file():
        raise FileNotFoundError(f"Existing stability artifact not found: {stability_path}")
    table = pd.read_parquet(stability_path)
    _validate_legacy_stability_schema(table)
    expected_partition = {
        "animal_name": animal_name,
        "date": session_date,
        "epoch": str(key["epoch"]),
        "trajectory_type": str(key["trajectory_type"]),
        "region": region,
    }
    include = np.ones(len(table), dtype=bool)
    for column, expected in expected_partition.items():
        include &= table[column].astype(str).to_numpy() == str(expected)
    selected = table.loc[include].copy().reset_index(drop=True)
    if selected.empty and loaded_units["unit_ids"]:
        raise ValueError(
            "Existing stability artifact has no rows for the selected partition."
        )
    legacy_ids = {
        str(metadata["sorting_unit_id"])
        for metadata in loaded_units["unit_metadata"]
        if metadata.get("sorting_unit_id") is not None
    }
    if len(legacy_ids) != len(loaded_units["unit_ids"]):
        raise ValueError(
            "Every imported selected unit needs a unique sorting_unit_id for "
            "legacy stability registration."
        )
    selected = selected.loc[selected["unit"].astype(str).isin(legacy_ids)].copy()
    if loaded_units["unit_ids"]:
        selected = selected.rename(columns={"unit": "unit_id"})
        selected = _attach_registered_unit_identity(
            selected,
            unit_metadata=loaded_units["unit_metadata"],
            artifact_name="stability",
        )
    else:
        selected = empty_stability_table()
    selected_units = set(selected["stable_unit_id"].astype(str))
    expected_units = {
        f"{unit['spikesorting_merge_id']}:{unit['unit_id']}"
        for unit in loaded_units["unit_ids"]
    }
    if len(selected) != len(selected_units) or selected_units != expected_units:
        raise ValueError(
            "Existing stability partition must contain exactly one row per "
            "selected SortedSpikesGroup unit."
        )
    ordered_identity = [
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "group_unit_id",
    ]
    ordered_identity.extend(
        column for column in selected if column not in ordered_identity
    )
    selected = selected.loc[:, ordered_identity]
    n_valid_units = int(selected["stability_status"].astype(str).eq("valid").sum())
    if not expected_units:
        analysis_status = "no_units"
    else:
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    destination = get_stability_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(key["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        region=region,
        task_progression_stability_id=key["task_progression_stability_id"],
        **path_kwargs,
    )
    created_artifact_paths = [] if destination.exists() else [str(destination)]
    written_path = write_stability_artifact(
        selected,
        destination,
        overwrite=overwrite,
    )
    return {
        "stability_path": str(written_path),
        "n_units": len(expected_units),
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_units["unit_ids"]),
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": {
            "source_path": str(stability_path.resolve(strict=True)),
            "sha256": _file_sha256(stability_path),
            "source_v1ca1_git_commit": source_v1ca1_git_commit,
            "source_spyglass_git_commit": source_spyglass_git_commit,
            "assumed_parameters": parameters,
        },
        "_created_artifact_paths": created_artifact_paths,
    }


def _new_schema(schema_factory: Callable[..., Any], context: dict[str, Any]) -> Any:
    """Construct one schema while supporting minimal injectable factories."""
    try:
        return schema_factory(context=context)
    except TypeError:
        schema = schema_factory()
        if hasattr(schema, "context"):
            schema.context = context
    return schema


def _validate_analysis_schema_prefix(
    dj_module: Any,
    analysis_nwbfile_schema_name: str,
) -> None:
    """Fail before DDL when Spyglass's configured custom prefix cannot match."""
    if analysis_nwbfile_schema_name.count("_") != 1:
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    expected_prefix, suffix = analysis_nwbfile_schema_name.split("_", 1)
    if suffix != "nwbfile":
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    custom_config = dj_module.config.get("custom", {})
    configured_prefix = custom_config.get("database.prefix")
    if configured_prefix != expected_prefix:
        raise ValueError(
            "Spyglass custom AnalysisNwbfile activation requires "
            "dj.config['custom']['database.prefix'] to equal "
            f"{expected_prefix!r}; found {configured_prefix!r}."
        )


def _construct_tables(
    *,
    dj_module: Any,
    session_table: Any,
    nwbfile_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    spyglass_mixin: type,
    spyglass_analysis: type,
    schema_factory: Callable[..., Any],
    schema_name: str,
    analysis_nwbfile_schema_name: str,
    connection: Any,
    create_schema: bool,
    create_tables: bool,
    runtime_hooks: Mapping[str, Callable[..., Any]] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Build and decorate tables from injected DataJoint-like dependencies."""
    runtime_hooks = dict(runtime_hooks or {})
    ripple_compute_hook = runtime_hooks.get(
        "ripple_modulation_compute",
        runtime_hooks.get("compute", _make_ripple_modulation_row),
    )
    ripple_register_hook = runtime_hooks.get(
        "ripple_modulation_register_existing",
        runtime_hooks.get(
            "register_existing",
            _register_existing_ripple_modulation_row,
        ),
    )
    stability_compute_hook = runtime_hooks.get(
        "task_progression_stability_compute",
        _make_task_progression_stability_row,
    )
    stability_register_hook = runtime_hooks.get(
        "task_progression_stability_register_existing",
        _register_existing_task_progression_stability_row,
    )
    if not all(
        callable(hook)
        for hook in (
            ripple_compute_hook,
            ripple_register_hook,
            stability_compute_hook,
            stability_register_hook,
        )
    ):
        raise TypeError("Analysis runtime hooks must be callable.")

    main_context: dict[str, Any] = {
        "Session": session_table,
        "SortedSpikesGroup": sorted_spikes_group,
        "UnitSelectionParams": unit_selection_params,
        "SpikeSortingOutput": spike_sorting_output,
    }
    main_schema = _new_schema(schema_factory, main_context)
    main_schema.activate(
        schema_name,
        connection=connection,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=main_context,
    )

    class EpochIntervals(spyglass_mixin, dj_module.Manual):
        definition = table_specs.EPOCH_INTERVALS_DEFINITION

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load one epoch's ephys-reference interval from its NWB file."""
            from v1ca1.spyglass.nwb import load_interval_set

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_interval_set,
            )

    EpochIntervals = main_schema(EpochIntervals)
    main_context["EpochIntervals"] = EpochIntervals

    class TrajectoryIntervals(spyglass_mixin, dj_module.Manual):
        definition = table_specs.TRAJECTORY_INTERVALS_DEFINITION

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load all laps for one epoch and trajectory type."""
            from v1ca1.spyglass.nwb import load_interval_set

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_interval_set,
            )

    TrajectoryIntervals = main_schema(TrajectoryIntervals)
    main_context["TrajectoryIntervals"] = TrajectoryIntervals

    class Ripples(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLES_DEFINITION

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load detector-qualified, speed-gated ripples for one epoch."""
            from v1ca1.spyglass.nwb import load_interval_set

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_interval_set,
            )

    Ripples = main_schema(Ripples)
    main_context["Ripples"] = Ripples

    class Position(spyglass_mixin, dj_module.Manual):
        definition = table_specs.POSITION_DEFINITION

        @classmethod
        def load_position(
            cls,
            key: Mapping[str, Any],
            *,
            apply_analysis_offset: bool = True,
        ) -> Any:
            """Load one explicitly named epoch position series in centimeters."""
            from v1ca1.spyglass.nwb import load_position

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_position,
                loader_kwargs={"apply_analysis_offset": apply_analysis_offset},
            )

    Position = main_schema(Position)
    main_context["Position"] = Position

    class WTrackGraph(spyglass_mixin, dj_module.Manual):
        definition = table_specs.WTRACK_GRAPH_DEFINITION

        @classmethod
        def load_graph(cls, key: Mapping[str, Any]) -> dict[str, Any]:
            """Load track-linearization graph inputs in centimeters."""
            from v1ca1.spyglass.nwb import load_wtrack_graph

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_wtrack_graph,
            )

    WTrackGraph = main_schema(WTrackGraph)
    main_context["WTrackGraph"] = WTrackGraph

    class SpikeSortingFigurl(spyglass_mixin, dj_module.Manual):
        definition = table_specs.SPIKE_SORTING_FIGURL_DEFINITION

        @classmethod
        def get_url(cls, key: Mapping[str, Any]) -> str:
            """Return the indexed spike-sorting FigURL for one probe/shank."""
            return str((cls & dict(key)).fetch1("figurl_url"))

    SpikeSortingFigurl = main_schema(SpikeSortingFigurl)
    main_context["SpikeSortingFigurl"] = SpikeSortingFigurl

    class RippleModulationParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_MODULATION_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one scalar parameter set."""
            validated = _validate_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_default(cls, *, skip_duplicates: bool = True) -> dict[str, Any]:
            """Explicitly insert the canonical no-extra-threshold parameters."""
            return cls.insert_parameters(
                table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    RippleModulationParameters = main_schema(RippleModulationParameters)
    main_context["RippleModulationParameters"] = RippleModulationParameters

    class RippleModulationSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_MODULATION_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one selection."""
            row = _ripple_modulation_selection_row(
                key=key,
                ripples_table=Ripples,
                epoch_intervals_table=EpochIntervals,
                parameters_table=RippleModulationParameters,
                sorted_spikes_group=sorted_spikes_group,
                unit_selection_params=unit_selection_params,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    RippleModulationSelection = main_schema(RippleModulationSelection)
    main_context["RippleModulationSelection"] = RippleModulationSelection

    class RippleModulation(spyglass_mixin, dj_module.Computed):
        definition = table_specs.RIPPLE_MODULATION_DEFINITION
        _compute_hook = staticmethod(ripple_compute_hook)
        _register_existing_hook = staticmethod(ripple_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and register one selected artifact pair."""
            selection = _fetch1_dict(RippleModulationSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleModulationParameters,
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "ripple_modulation_id": selection[
                            "ripple_modulation_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            summary_path: Path | str,
            peri_ripple_firing_rate_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Filter keyed legacy Parquets, write them, then insert one row."""
            if overwrite:
                raise ValueError(
                    "Registered RippleModulation results are immutable; create "
                    "a new selection instead of overwriting an artifact."
                )
            selection = _fetch1_dict(RippleModulationSelection, key)
            result_key = {
                "ripple_modulation_id": selection["ripple_modulation_id"]
            }
            existing = _existing_result_row(cls, result_key)
            if existing is not None:
                if skip_duplicates:
                    return existing
                raise ValueError(
                    "RippleModulation already contains this immutable selection."
                )
            artifact_row = dict(
                cls._register_existing_hook(
                    key=selection,
                    summary_path=Path(summary_path),
                    peri_ripple_firing_rate_path=Path(
                        peri_ripple_firing_rate_path
                    ),
                    overwrite=False,
                    parameters_table=RippleModulationParameters,
                    ripples_table=Ripples,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "ripple_modulation_id": selection["ripple_modulation_id"],
                **artifact_row,
                "artifact_origin": "registered_existing",
                "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                "runtime_spyglass_git_commit": _spyglass_git_commit(),
            }
            try:
                cls.insert1(
                    row,
                    skip_duplicates=False,
                    allow_direct_insert=True,
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise
            return row

    RippleModulation = main_schema(RippleModulation)
    main_context["RippleModulation"] = RippleModulation

    class TaskProgressionStabilityParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.TASK_PROGRESSION_STABILITY_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one numerical stability parameter set."""
            validated = _validate_stability_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_default(cls, *, skip_duplicates: bool = True) -> dict[str, Any]:
            """Explicitly insert the canonical stability parameters."""
            return cls.insert_parameters(
                table_specs.DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    TaskProgressionStabilityParameters = main_schema(
        TaskProgressionStabilityParameters
    )
    main_context["TaskProgressionStabilityParameters"] = (
        TaskProgressionStabilityParameters
    )

    class TaskProgressionStabilitySelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.TASK_PROGRESSION_STABILITY_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one stability selection."""
            row = _stability_selection_row(
                key=key,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                position_table=Position,
                wtrack_graph_table=WTrackGraph,
                parameters_table=TaskProgressionStabilityParameters,
                sorted_spikes_group=sorted_spikes_group,
                unit_selection_params=unit_selection_params,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    TaskProgressionStabilitySelection = main_schema(
        TaskProgressionStabilitySelection
    )
    main_context["TaskProgressionStabilitySelection"] = (
        TaskProgressionStabilitySelection
    )

    class TaskProgressionStability(spyglass_mixin, dj_module.Computed):
        definition = table_specs.TASK_PROGRESSION_STABILITY_DEFINITION
        _compute_hook = staticmethod(stability_compute_hook)
        _register_existing_hook = staticmethod(stability_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one selected stability artifact."""
            selection = _fetch1_dict(TaskProgressionStabilitySelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=TaskProgressionStabilityParameters,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    position_table=Position,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "task_progression_stability_id": selection[
                            "task_progression_stability_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            stability_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Filter the complete legacy Parquet and insert one result row."""
            if overwrite:
                raise ValueError(
                    "Registered TaskProgressionStability results are immutable; "
                    "create a new selection instead of overwriting an artifact."
                )
            selection = _fetch1_dict(TaskProgressionStabilitySelection, key)
            result_key = {
                "task_progression_stability_id": selection[
                    "task_progression_stability_id"
                ]
            }
            existing = _existing_result_row(cls, result_key)
            if existing is not None:
                if skip_duplicates:
                    return existing
                raise ValueError(
                    "TaskProgressionStability already contains this immutable selection."
                )
            artifact_row = dict(
                cls._register_existing_hook(
                    key=selection,
                    stability_path=Path(stability_path),
                    overwrite=False,
                    parameters_table=TaskProgressionStabilityParameters,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "task_progression_stability_id": selection[
                    "task_progression_stability_id"
                ],
                **artifact_row,
                "artifact_origin": "registered_existing",
                "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                "runtime_spyglass_git_commit": _spyglass_git_commit(),
            }
            try:
                cls.insert1(
                    row,
                    skip_duplicates=False,
                    allow_direct_insert=True,
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise
            return row

    TaskProgressionStability = main_schema(TaskProgressionStability)
    main_context["TaskProgressionStability"] = TaskProgressionStability

    analysis_context = {"Nwbfile": nwbfile_table}
    analysis_schema = _new_schema(schema_factory, analysis_context)
    analysis_schema.activate(
        analysis_nwbfile_schema_name,
        connection=(
            connection
            if connection is not None
            else getattr(main_schema, "connection", None)
        ),
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=analysis_context,
    )

    class AnalysisNwbfile(spyglass_analysis, dj_module.Manual):
        definition = table_specs.ANALYSIS_NWBFILE_DEFINITION

        def _register_table(self) -> None:
            """Suppress Spyglass's registry insert during DDL-only activation."""
            return None

        def register_with_spyglass(self) -> None:
            """Explicitly add this table to Spyglass's AnalysisRegistry."""
            spyglass_analysis._register_table(self)

    AnalysisNwbfile = analysis_schema(AnalysisNwbfile)

    return {
        "epoch_intervals": EpochIntervals,
        "trajectory_intervals": TrajectoryIntervals,
        "ripples": Ripples,
        "position": Position,
        "wtrack_graph": WTrackGraph,
        "spike_sorting_figurl": SpikeSortingFigurl,
        "ripple_modulation_parameters": RippleModulationParameters,
        "ripple_modulation_selection": RippleModulationSelection,
        "ripple_modulation": RippleModulation,
        "task_progression_stability_parameters": (
            TaskProgressionStabilityParameters
        ),
        "task_progression_stability_selection": (
            TaskProgressionStabilitySelection
        ),
        "task_progression_stability": TaskProgressionStability,
        "analysis_nwbfile": AnalysisNwbfile,
    }


def activate(
    schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
    *,
    analysis_nwbfile_schema_name: str = table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    connection: Any = None,
    create_schema: bool = True,
    create_tables: bool = True,
    runtime_hooks: Mapping[str, Callable[..., Any]] | None = None,
    artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    """Explicitly import dependencies, activate schemas, and return table classes.

    Activation declares tables only.  It never calls ``insert_default``,
    ``populate``, ``make``, or ``register_existing``.
    """
    import datajoint as dj

    from spyglass.common import Nwbfile, Session
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.utils import SpyglassAnalysis, SpyglassMixin

    _validate_analysis_schema_prefix(dj, analysis_nwbfile_schema_name)

    return _construct_tables(
        dj_module=dj,
        session_table=Session,
        nwbfile_table=Nwbfile,
        sorted_spikes_group=SortedSpikesGroup,
        unit_selection_params=UnitSelectionParams,
        spike_sorting_output=SpikeSortingOutput,
        spyglass_mixin=SpyglassMixin,
        spyglass_analysis=SpyglassAnalysis,
        schema_factory=dj.Schema,
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
        connection=connection,
        create_schema=create_schema,
        create_tables=create_tables,
        runtime_hooks=runtime_hooks,
        artifact_root=None if artifact_root is None else Path(artifact_root),
    )


__all__ = ["SOURCE_TABLE_KEYS", "activate"]
