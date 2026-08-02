"""Explicit activation for the project-owned Spyglass tables.

Importing this module is passive: DataJoint and Spyglass are imported only by
``activate``.  Runtime computation is likewise reached only through an
explicitly activated ``RippleModulationComputed`` table.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
import hashlib
import math
from numbers import Real
from pathlib import Path
import re
import subprocess
from typing import Any

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

    threshold = values["minimum_ripple_mean_zscore"]
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError("minimum_ripple_mean_zscore must be a numeric scalar or None.")
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("minimum_ripple_mean_zscore must be positive when set.")
        values["minimum_ripple_mean_zscore"] = threshold

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
        "minimum_ripple_mean_zscore": parameters["minimum_ripple_mean_zscore"],
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
    if key["unit_filter_params_name"] != "all_units":
        raise ValueError(
            "RippleModulation currently requires unit_filter_params_name='all_units'."
        )
    return {name: key[name] for name in required}


def _sorting_output_sessions(
    spike_sorting_output: Any,
    *,
    merge_id: Any,
) -> set[str]:
    """Resolve the source NWB session for one merge output."""
    merge_key = {"merge_id": merge_id}
    get_parent = getattr(spike_sorting_output, "merge_get_parent", None)
    if not callable(get_parent):
        raise TypeError(
            "SpikeSortingOutput must expose merge_get_parent for session validation."
        )
    parent = get_parent(merge_key)
    heading_names = tuple(getattr(getattr(parent, "heading", None), "names", ()))
    if not heading_names or "nwb_file_name" in heading_names:
        fetch = getattr(parent, "fetch", None)
        if not callable(fetch):
            raise TypeError("SpikeSortingOutput merge parent must expose fetch().")
        try:
            return {str(name) for name in fetch("nwb_file_name")}
        except (KeyError, TypeError, ValueError):
            if "nwb_file_name" in heading_names:
                raise

    get_sort_group_info = getattr(spike_sorting_output, "get_sort_group_info", None)
    if not callable(get_sort_group_info):
        raise ValueError(
            f"Cannot resolve nwb_file_name lineage for SpikeSortingOutput {merge_id!r}."
        )
    sort_group_info = get_sort_group_info(merge_key)
    info_heading_names = tuple(
        getattr(getattr(sort_group_info, "heading", None), "names", ())
    )
    if info_heading_names and "nwb_file_name" not in info_heading_names:
        raise ValueError(
            f"SpikeSortingOutput {merge_id!r} sort-group lineage has no "
            "nwb_file_name."
        )
    fetch = getattr(sort_group_info, "fetch", None)
    if not callable(fetch):
        raise TypeError("SpikeSortingOutput sort-group lineage must expose fetch().")
    try:
        return {str(name) for name in fetch("nwb_file_name")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot resolve nwb_file_name lineage for SpikeSortingOutput {merge_id!r}."
        ) from exc


def _group_membership_provenance(merge_ids: list[Any]) -> tuple[list[str], str]:
    """Return sorted member ids and a deterministic membership digest."""
    member_ids = sorted((str(merge_id) for merge_id in merge_ids))
    if not member_ids or len(set(member_ids)) != len(member_ids):
        raise ValueError("Sorting-group membership must be non-empty and unique.")
    digest = hashlib.sha256("\0".join(member_ids).encode("utf-8")).hexdigest()
    return member_ids, digest


def _safe_group_component(
    key: Mapping[str, Any],
    *,
    merge_ids: list[Any] | None = None,
) -> str:
    """Build a collision-resistant path component for a sorting group."""
    group_key = _sorted_spikes_group_key(key)
    for required_name in ("ripple_modulation_param_name", "nwb_file_name"):
        if required_name not in key or not str(key[required_name]):
            raise ValueError(
                f"RippleModulation selection is missing {required_name!r}."
            )
    identity = (
        f"{key['nwb_file_name']}\0"
        f"{key['ripple_modulation_param_name']}\0"
        f"{group_key['sorted_spikes_group_name']}\0"
        f"{group_key['unit_filter_params_name']}"
    )
    slug = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        str(group_key["sorted_spikes_group_name"]),
    ).strip(".-")
    if not slug:
        slug = "group"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    component = (
        f"group-{slug[:48]}-{group_key['unit_filter_params_name']}-{digest}"
    )
    if merge_ids is not None:
        _, membership_digest = _group_membership_provenance(merge_ids)
        component = f"{component}-members-{membership_digest[:12]}"
    return component


def _group_artifact_paths(
    paths: Mapping[str, Path],
    key: Mapping[str, Any],
    *,
    merge_ids: list[Any] | None = None,
) -> dict[str, Path]:
    """Place an artifact pair beneath its unique sorting-group directory."""
    component = _safe_group_component(key, merge_ids=merge_ids)
    directory = Path(paths["directory"]) / component
    return {
        "directory": directory,
        "summary": directory / Path(paths["summary"]).name,
        "peri_ripple_firing_rate": directory
        / Path(paths["peri_ripple_firing_rate"]).name,
    }


def _group_registration_plan(
    plan: Mapping[str, Any],
    key: Mapping[str, Any],
    *,
    merge_ids: list[Any] | None = None,
) -> dict[str, Any]:
    """Retarget a pure registration plan to the selection's group directory."""
    component = _safe_group_component(key, merge_ids=merge_ids)
    artifact_paths = {
        name: Path(path).parent / component / Path(path).name
        for name, path in dict(plan["artifact_paths"]).items()
    }
    copies = []
    for copy in plan["copies"]:
        name = str(copy["artifact"])
        source = Path(copy["source"])
        destination = artifact_paths[name]
        copies.append(
            {
                **dict(copy),
                "destination": destination,
                "copy_required": source.resolve(strict=False)
                != destination.resolve(strict=False),
            }
        )
    return {**dict(plan), "artifact_paths": artifact_paths, "copies": copies}


def _load_group_unit_data(
    *,
    sorted_spikes_group: Any,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    region: str,
) -> dict[str, Any]:
    """Load canonical seconds and metadata for every member of one group."""
    from v1ca1.spyglass.spikes import fetch_spike_times_seconds_with_metadata

    group_key = _sorted_spikes_group_key(key)
    merge_ids = sorted(
        (sorted_spikes_group.Units & group_key).fetch("spikesorting_merge_id"),
        key=str,
    )
    if not merge_ids:
        raise ValueError(f"SortedSpikesGroup has no members for {group_key!r}.")
    if len({str(merge_id) for merge_id in merge_ids}) != len(merge_ids):
        raise ValueError("SortedSpikesGroup contains duplicate SpikeSortingOutput IDs.")

    spike_times = []
    unit_ids = []
    unit_metadata = []
    for merge_id in merge_ids:
        member_sessions = _sorting_output_sessions(
            spike_sorting_output,
            merge_id=merge_id,
        )
        expected_session = str(group_key["nwb_file_name"])
        if member_sessions != {expected_session}:
            raise ValueError(
                f"SpikeSortingOutput {merge_id!r} belongs to sessions "
                f"{sorted(member_sessions)!r}, not sorting-group session "
                f"{expected_session!r}."
            )
        member_spikes, member_unit_ids, member_metadata = (
            fetch_spike_times_seconds_with_metadata(
                spike_sorting_output,
                {"merge_id": merge_id},
                region=region,
            )
        )
        spike_times.extend(member_spikes)
        unit_ids.extend(member_unit_ids)
        unit_metadata.extend(member_metadata)
    if not unit_ids:
        raise ValueError(
            f"SortedSpikesGroup has no units after validating region {region!r}."
        )
    stable_ids = [
        (str(unit["spikesorting_merge_id"]), str(unit["unit_id"]))
        for unit in unit_ids
    ]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("SortedSpikesGroup produced duplicate stable unit IDs.")
    return {
        "spike_times_s": spike_times,
        "unit_ids": unit_ids,
        "unit_metadata": unit_metadata,
        "merge_ids": merge_ids,
    }


def _load_group_spikes(
    *,
    sorted_spikes_group: Any,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    region: str,
    time_support: tuple[float, float],
) -> dict[str, Any]:
    """Build one Pynapple TsGroup from all validated sorting-group members."""
    from v1ca1.spyglass.spikes import build_spike_tsgroup

    loaded = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=region,
    )
    return {
        **loaded,
        "ts_group": build_spike_tsgroup(
            loaded["spike_times_s"],
            loaded["unit_ids"],
            time_support=time_support,
        ),
    }


def _make_ripple_modulation_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one keyed RippleModulation result."""
    import pynwb

    from v1ca1.spyglass.nwb import load_interval_set
    from v1ca1.spyglass.ripple_modulation import (
        compute_epoch_region_ripple_modulation,
        get_ripple_modulation_artifact_paths,
        write_ripple_modulation_artifacts,
    )
    parameters = _validate_parameter_row(_fetch1_dict(parameters_table, key))
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
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=region,
        time_support=(epoch_start, epoch_stop),
    )
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
    if (
        parameters["minimum_ripple_mean_zscore"] is None
        and int(result["n_ripples"]) != int(ripple_row["ripple_count"])
    ):
        raise ValueError(
            "Unfiltered RippleModulation ripple count does not match the selected "
            "Ripples catalog row."
        )
    if int(result["n_ripples"]) > int(ripple_row["ripple_count"]):
        raise ValueError("RippleModulation selected more ripples than its catalog row.")
    path_kwargs = {
        **_parameter_kwargs(parameters),
        "heatmap_normalize": parameters["heatmap_normalize"],
    }
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = _group_artifact_paths(
        get_ripple_modulation_artifact_paths(
            animal_name=animal_name,
            date=session_date,
            epoch=str(key["epoch"]),
            region=region,
            **path_kwargs,
        ),
        key,
        merge_ids=loaded_spikes["merge_ids"],
    )
    created_artifact_paths = [
        str(paths[name])
        for name in ("summary", "peri_ripple_firing_rate")
        if not Path(paths[name]).exists()
    ]
    written = write_ripple_modulation_artifacts(result, paths)
    sorting_group_members, membership_digest = _group_membership_provenance(
        loaded_spikes["merge_ids"]
    )
    return {
        "summary_path": str(written["summary"]),
        "peri_ripple_firing_rate_path": str(written["peri_ripple_firing_rate"]),
        "n_ripples": int(result["n_ripples"]),
        "n_units": int(len(loaded_spikes["unit_ids"])),
        "sorting_group_members": sorting_group_members,
        "sorting_group_members_sha256": membership_digest,
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
        raise ValueError("Selected SortedSpikesGroup has no unit metadata.")
    stable_columns = {"spikesorting_merge_id", "nwb_unit_id"}
    present_stable_columns = stable_columns.intersection(table.columns)
    if present_stable_columns and present_stable_columns != stable_columns:
        raise ValueError(
            f"{artifact_name} parquet must contain both stable unit columns or neither."
        )

    row_metadata: list[dict[str, Any]] = []
    if present_stable_columns:
        stable_pairs = zip(
            table["spikesorting_merge_id"].astype(str),
            table["nwb_unit_id"].astype(str),
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
    output["spikesorting_merge_id"] = [
        str(metadata["spikesorting_merge_id"]) for metadata in row_metadata
    ]
    output["nwb_unit_id"] = [str(metadata["unit_id"]) for metadata in row_metadata]
    output["unit_id"] = [
        f"{merge_id}:{nwb_unit_id}"
        for merge_id, nwb_unit_id in zip(
            output["spikesorting_merge_id"],
            output["nwb_unit_id"],
        )
    ]
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

    parameters = _validate_parameter_row(_fetch1_dict(parameters_table, key))
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
        **_parameter_kwargs(parameters),
        "heatmap_normalize": parameters["heatmap_normalize"],
    }
    if artifact_root is not None:
        plan_kwargs["artifact_root"] = artifact_root
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=artifact_key["region"],
    )
    plan = _group_registration_plan(
        plan_register_existing(**plan_kwargs),
        key,
        merge_ids=loaded_units["merge_ids"],
    )
    selected_tables = read_planned_artifacts(
        plan,
        allow_unkeyed_same_path=overwrite,
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
    )
    selected_peri = _filter_registered_table(
        selected_tables["peri_ripple_firing_rate"],
        artifact_name="peri_ripple_firing_rate",
        artifact_key=artifact_key,
        parameters=parameters,
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
                f"normalization: {copy['source']}. Pass overwrite=True."
            )

    summary_units = set(selected_summary["unit_id"].astype(str))
    peri_units = set(selected_peri["unit_id"].astype(str))
    catalog_units = {
        f"{unit['spikesorting_merge_id']}:{unit['unit_id']}"
        for unit in loaded_units["unit_ids"]
    }
    if (
        not summary_units
        or len(selected_summary) != len(summary_units)
        or summary_units != peri_units
        or summary_units != catalog_units
    ):
        raise ValueError(
            "Existing summary must contain one row per selected "
            "SortedSpikesGroup unit, and both artifacts must contain exactly "
            "the same units."
        )
    summary_n_ripples = int(selected_summary["n_ripples"].iloc[0])
    peri_n_ripples = int(selected_peri["n_ripples"].iloc[0])
    if summary_n_ripples != peri_n_ripples:
        raise ValueError("Existing artifacts disagree on n_ripples.")
    if (
        parameters["minimum_ripple_mean_zscore"] is None
        and summary_n_ripples != int(ripple_row["ripple_count"])
    ):
        raise ValueError(
            "Unfiltered existing artifact n_ripples does not match the selected "
            "Ripples catalog row."
        )
    if summary_n_ripples > int(ripple_row["ripple_count"]):
        raise ValueError("Existing artifact exceeds its cataloged ripple count.")
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
    sorting_group_members, membership_digest = _group_membership_provenance(
        loaded_units["merge_ids"]
    )
    return {
        "summary_path": str(destinations["summary"]),
        "peri_ripple_firing_rate_path": str(destinations["peri_ripple_firing_rate"]),
        "n_ripples": summary_n_ripples,
        "n_units": len(summary_units),
        "sorting_group_members": sorting_group_members,
        "sorting_group_members_sha256": membership_digest,
        "legacy_artifact_provenance": legacy_artifact_provenance,
        "artifact_origin": "registered_existing",
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
    compute_hook = runtime_hooks.get("compute", _make_ripple_modulation_row)
    register_hook = runtime_hooks.get(
        "register_existing", _register_existing_ripple_modulation_row
    )
    if not callable(compute_hook) or not callable(register_hook):
        raise TypeError("RippleModulation runtime hooks must be callable.")

    main_context: dict[str, Any] = {
        "Session": session_table,
        "SortedSpikesGroup": sorted_spikes_group,
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
            """Load one epoch/type position series in centimeters and seconds."""
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

    RippleModulationSelection = main_schema(RippleModulationSelection)
    main_context["RippleModulationSelection"] = RippleModulationSelection

    class RippleModulationComputed(spyglass_mixin, dj_module.Computed):
        definition = table_specs.RIPPLE_MODULATION_COMPUTED_DEFINITION
        _compute_hook = staticmethod(compute_hook)
        _register_existing_hook = staticmethod(register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and register one selected artifact pair."""
            row = dict(
                self._compute_hook(
                    key=dict(key),
                    parameters_table=RippleModulationParameters,
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
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
                        **dict(key),
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
            artifact_row = dict(
                cls._register_existing_hook(
                    key=dict(key),
                    summary_path=Path(summary_path),
                    peri_ripple_firing_rate_path=Path(
                        peri_ripple_firing_rate_path
                    ),
                    overwrite=overwrite,
                    parameters_table=RippleModulationParameters,
                    ripples_table=Ripples,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
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
                **dict(key),
                **artifact_row,
                "artifact_origin": "registered_existing",
                "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                "runtime_spyglass_git_commit": _spyglass_git_commit(),
            }
            try:
                cls.insert1(row, skip_duplicates=skip_duplicates)
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise
            return row

    RippleModulationComputed = main_schema(RippleModulationComputed)
    main_context["RippleModulationComputed"] = RippleModulationComputed

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
        "ripple_modulation_computed": RippleModulationComputed,
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
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.utils import SpyglassAnalysis, SpyglassMixin

    _validate_analysis_schema_prefix(dj, analysis_nwbfile_schema_name)

    return _construct_tables(
        dj_module=dj,
        session_table=Session,
        nwbfile_table=Nwbfile,
        sorted_spikes_group=SortedSpikesGroup,
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
