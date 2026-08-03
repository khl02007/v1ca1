from __future__ import annotations

"""Database-free RippleModulation computation and artifact planning.

This module adapts the existing ripple-modulation analysis to one explicit
epoch and region.  It contains no DataJoint table declarations or inserts, and
all filesystem mutations require a separate, explicit write/copy call.
"""

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "ripple_modulation"
ARTIFACT_NAMES = ("summary", "peri_ripple_firing_rate")
STABLE_UNIT_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
)


def _ripple_plot_module() -> Any:
    """Import the existing analysis implementation only when it is used."""
    from v1ca1.ripple import plot_ripple_modulation

    return plot_ripple_modulation


def _resolve_parameters(
    *,
    bin_size_s: float | None,
    time_before_s: float | None,
    time_after_s: float | None,
    response_window: tuple[float, float] | None,
    baseline_window: tuple[float, float] | None,
    heatmap_normalize: str | None = None,
) -> dict[str, Any]:
    """Resolve optional values against the canonical analysis defaults."""
    ripple_plot = _ripple_plot_module()
    values = {
        "bin_size_s": float(
            ripple_plot.DEFAULT_BIN_SIZE_S if bin_size_s is None else bin_size_s
        ),
        "time_before_s": float(
            ripple_plot.DEFAULT_TIME_BEFORE_S
            if time_before_s is None
            else time_before_s
        ),
        "time_after_s": float(
            ripple_plot.DEFAULT_TIME_AFTER_S
            if time_after_s is None
            else time_after_s
        ),
        "response_window": (
            (
                float(ripple_plot.DEFAULT_RESPONSE_WINDOW_START_S),
                float(ripple_plot.DEFAULT_RESPONSE_WINDOW_END_S),
            )
            if response_window is None
            else (float(response_window[0]), float(response_window[1]))
        ),
        "baseline_window": (
            (
                float(ripple_plot.DEFAULT_BASELINE_WINDOW_START_S),
                float(ripple_plot.DEFAULT_BASELINE_WINDOW_END_S),
            )
            if baseline_window is None
            else (float(baseline_window[0]), float(baseline_window[1]))
        ),
        "heatmap_normalize": (
            ripple_plot.DEFAULT_HEATMAP_NORMALIZE
            if heatmap_normalize is None
            else str(heatmap_normalize)
        ),
    }
    if values["bin_size_s"] <= 0:
        raise ValueError("bin_size_s must be positive.")
    if values["time_before_s"] <= 0 or values["time_after_s"] <= 0:
        raise ValueError("time_before_s and time_after_s must be positive.")
    if values["response_window"][0] >= values["response_window"][1]:
        raise ValueError("response_window start must be smaller than its end.")
    if values["baseline_window"][0] >= values["baseline_window"][1]:
        raise ValueError("baseline_window start must be smaller than its end.")
    if values["heatmap_normalize"] not in ripple_plot.HEATMAP_NORMALIZE_CHOICES:
        raise ValueError(
            "heatmap_normalize must be one of "
            f"{ripple_plot.HEATMAP_NORMALIZE_CHOICES!r}."
        )
    return values


def _select_epoch_ripples(ripple_table: Any, *, epoch: str) -> Any:
    """Normalize one supplied detector table and select exactly one epoch."""
    as_dataframe = getattr(ripple_table, "as_dataframe", None)
    if callable(as_dataframe):
        table = as_dataframe().copy()
    elif hasattr(ripple_table, "columns") and hasattr(ripple_table, "copy"):
        table = ripple_table.copy()
    else:
        raise TypeError(
            "ripple_table must be a pandas DataFrame-like object or expose "
            "as_dataframe(), as pynapple.IntervalSet does."
        )
    rename_columns: dict[str, str] = {}
    if "start" in table.columns and "start_time" not in table.columns:
        rename_columns["start"] = "start_time"
    if "end" in table.columns and "end_time" not in table.columns:
        rename_columns["end"] = "end_time"
    if "stop_time" in table.columns and "end_time" not in table.columns:
        rename_columns["stop_time"] = "end_time"
    if rename_columns:
        table = table.rename(columns=rename_columns)

    if "epoch" in table.columns:
        epoch_values = table["epoch"].astype(str)
        table = table.loc[epoch_values == str(epoch)].reset_index(drop=True)

    missing_columns = [
        column for column in ("start_time", "end_time") if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            "ripple_table is missing required interval columns: "
            f"{missing_columns!r}."
        )
    table["start_time"] = np.asarray(table["start_time"], dtype=float)
    table["end_time"] = np.asarray(table["end_time"], dtype=float)
    if not np.all(np.isfinite(table[["start_time", "end_time"]].to_numpy(dtype=float))):
        raise ValueError("ripple interval bounds must contain only finite seconds values.")
    if np.any(
        table["end_time"].to_numpy(dtype=float)
        <= table["start_time"].to_numpy(dtype=float)
    ):
        raise ValueError("Every ripple interval must have start_time < end_time.")
    return table


def empty_ripple_modulation_result(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    n_ripples: int,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return typed empty artifacts for a terminal no-unit/event selection."""
    import pandas as pd

    ripple_plot = _ripple_plot_module()
    summary = pd.DataFrame(columns=ripple_plot.SUMMARY_COLUMNS)
    peri = pd.DataFrame(columns=ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS)
    summary = _attach_stable_unit_identity(
        summary,
        region_spikes={},
        stable_unit_ids=[],
    )
    peri = _attach_stable_unit_identity(
        peri,
        region_spikes={},
        stable_unit_ids=[],
    )
    return {
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "region": str(region),
        "n_ripples": int(n_ripples),
        "selected_ripple_table": pd.DataFrame(),
        "summary": summary,
        "peri_ripple_firing_rate": peri,
        "heatmap_payload": None,
        "stable_unit_ids": [],
        "parameters": dict(parameters),
    }


def _attach_stable_unit_identity(
    table: Any,
    *,
    region_spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> Any:
    """Replace temporary TsGroup keys with stable sorting/NWB unit identity."""
    group_keys = list(region_spikes.keys())
    stable_unit_ids = [dict(unit_id) for unit_id in stable_unit_ids]
    if len(group_keys) != len(stable_unit_ids):
        raise ValueError(
            "region_spikes and stable_unit_ids must have matching lengths; got "
            f"{len(group_keys)} and {len(stable_unit_ids)}."
        )

    identity_by_group_key: dict[Any, tuple[str, str, str]] = {}
    composite_ids: set[str] = set()
    for group_key, unit_id in zip(group_keys, stable_unit_ids):
        missing = [
            field
            for field in ("spikesorting_merge_id", "unit_id")
            if field not in unit_id
        ]
        if missing:
            raise ValueError(f"Stable unit identity is missing fields {missing!r}.")
        merge_id = str(unit_id["spikesorting_merge_id"])
        source_unit_id = str(unit_id["unit_id"])
        if not merge_id or not source_unit_id:
            raise ValueError("Stable merge and source unit ids must be non-empty.")
        composite_id = f"{merge_id}:{source_unit_id}"
        if composite_id in composite_ids:
            raise ValueError(f"Duplicate stable unit identity {composite_id!r}.")
        composite_ids.add(composite_id)
        try:
            identity_by_group_key[group_key] = (
                composite_id,
                merge_id,
                source_unit_id,
            )
        except TypeError as exc:
            raise TypeError("TsGroup unit keys must be hashable.") from exc

    output = table.copy()
    if output.empty:
        output["group_unit_id"] = np.asarray([], dtype=int)
        output["spikesorting_merge_id"] = np.asarray([], dtype=str)
        output["stable_unit_id"] = np.asarray([], dtype=str)
        return output
    if "unit_id" not in output.columns:
        raise ValueError("RippleModulation output is missing its unit_id column.")

    identities = []
    for group_key in output["unit_id"].to_list():
        try:
            identities.append(identity_by_group_key[group_key])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"RippleModulation output contains unknown TsGroup key {group_key!r}."
            ) from exc
    output["group_unit_id"] = output["unit_id"].to_numpy(copy=True)
    output["unit_id"] = [identity[2] for identity in identities]
    output["spikesorting_merge_id"] = [identity[1] for identity in identities]
    output["stable_unit_id"] = [identity[0] for identity in identities]
    return output


def compute_epoch_region_ripple_modulation(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    ripple_table: Any,
    epoch_timestamps: np.ndarray,
    region_spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]] | None = None,
    bin_size_s: float | None = None,
    time_before_s: float | None = None,
    time_after_s: float | None = None,
    response_window: tuple[float, float] | None = None,
    baseline_window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute RippleModulation for one supplied epoch and brain region.

    ``ripple_table`` is expected to contain detector-qualified, speed-gated
    events loaded from the NWB/ingestion layer. All its events for ``epoch``
    are used; event detection criteria belong to the selected ``Ripples``
    source row rather than this downstream computation.
    """
    timestamps = np.asarray(epoch_timestamps, dtype=float)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("epoch_timestamps must contain at least two one-dimensional samples.")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("epoch_timestamps must be finite and strictly increasing.")

    ripple_plot = _ripple_plot_module()
    parameters = _resolve_parameters(
        bin_size_s=bin_size_s,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        response_window=response_window,
        baseline_window=baseline_window,
    )
    epoch_ripple_table = _select_epoch_ripples(ripple_table, epoch=epoch)
    selected_ripple_table = epoch_ripple_table
    if not selected_ripple_table.empty:
        starts = selected_ripple_table["start_time"].to_numpy(dtype=float)
        stops = selected_ripple_table["end_time"].to_numpy(dtype=float)
        if np.any(starts < timestamps[0]) or np.any(stops > timestamps[-1]):
            raise ValueError(
                "Selected ripple intervals must lie within epoch_timestamps "
                f"bounds {(float(timestamps[0]), float(timestamps[-1]))!r}."
            )

    if selected_ripple_table.empty:
        import pandas as pd

        summary_table = pd.DataFrame(columns=ripple_plot.SUMMARY_COLUMNS)
        peri_ripple_table = pd.DataFrame(
            columns=ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS
        )
        if stable_unit_ids is not None:
            summary_table = _attach_stable_unit_identity(
                summary_table,
                region_spikes=region_spikes,
                stable_unit_ids=stable_unit_ids,
            )
            peri_ripple_table = _attach_stable_unit_identity(
                peri_ripple_table,
                region_spikes=region_spikes,
                stable_unit_ids=stable_unit_ids,
            )
        return {
            "animal_name": str(animal_name),
            "date": str(date),
            "epoch": str(epoch),
            "region": str(region),
            "n_ripples": 0,
            "selected_ripple_table": selected_ripple_table,
            "summary": summary_table,
            "peri_ripple_firing_rate": peri_ripple_table,
            "heatmap_payload": None,
            "stable_unit_ids": (
                None
                if stable_unit_ids is None
                else [dict(unit_id) for unit_id in stable_unit_ids]
            ),
            "parameters": parameters,
        }

    ripple_start_times, n_ripples = ripple_plot.build_ripple_start_times(
        selected_ripple_table,
        epoch=str(epoch),
        ripple_threshold_zscore=None,
        epoch_timestamps=timestamps,
    )
    heatmap_payload = ripple_plot.build_region_epoch_modulation_result(
        animal_name=str(animal_name),
        date=str(date),
        epoch=str(epoch),
        region=str(region),
        region_spikes=region_spikes,
        ripple_start_times=ripple_start_times,
        n_ripples=n_ripples,
        bin_size_s=parameters["bin_size_s"],
        time_before_s=parameters["time_before_s"],
        time_after_s=parameters["time_after_s"],
        response_window=parameters["response_window"],
        baseline_window=parameters["baseline_window"],
    )

    import pandas as pd

    summary_table = pd.DataFrame(
        heatmap_payload["rows"],
        columns=ripple_plot.SUMMARY_COLUMNS,
    )
    peri_ripple_table = ripple_plot.build_peri_ripple_firing_rate_table(
        {str(region): heatmap_payload}
    )
    if stable_unit_ids is not None:
        summary_table = _attach_stable_unit_identity(
            summary_table,
            region_spikes=region_spikes,
            stable_unit_ids=stable_unit_ids,
        )
        peri_ripple_table = _attach_stable_unit_identity(
            peri_ripple_table,
            region_spikes=region_spikes,
            stable_unit_ids=stable_unit_ids,
        )
        heatmap_payload = {
            **heatmap_payload,
            "unit_ids": summary_table["stable_unit_id"].to_numpy(dtype=object),
        }
    return {
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "region": str(region),
        "n_ripples": int(n_ripples),
        "selected_ripple_table": selected_ripple_table,
        "summary": summary_table,
        "peri_ripple_firing_rate": peri_ripple_table,
        "heatmap_payload": heatmap_payload,
        "stable_unit_ids": (
            None
            if stable_unit_ids is None
            else [dict(unit_id) for unit_id in stable_unit_ids]
        ),
        "parameters": parameters,
    }


def _validate_path_component(value: str, *, name: str) -> str:
    """Return one non-empty path component without traversal or separators."""
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component, got {value!r}.")
    return value


def _validate_uuid_component(value: Any, *, name: str) -> str:
    """Return one canonical UUID path component."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def get_ripple_modulation_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    ripple_modulation_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Plan one UUID-keyed, session-first artifact directory."""
    animal_name = _validate_path_component(animal_name, name="animal_name")
    date = _validate_path_component(date, name="date")
    epoch = _validate_path_component(epoch, name="epoch")
    region = _validate_path_component(region, name="region")
    selection_component = _validate_uuid_component(
        ripple_modulation_id,
        name="ripple_modulation_id",
    )
    output_dir = (
        Path(artifact_root)
        / animal_name
        / date
        / ARTIFACT_DIRNAME
        / epoch
        / region
        / selection_component
    )
    return {
        "directory": output_dir,
        "summary": output_dir / "summary.parquet",
        "peri_ripple_firing_rate": output_dir / "peri_ripple_firing_rate.parquet",
    }


def _legacy_artifact_names(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    parameters: Mapping[str, Any],
) -> dict[str, str]:
    """Return expected standalone-script names with no extra event filter."""
    ripple_plot = _ripple_plot_module()
    stem = ripple_plot.build_epoch_output_stem(
        animal_name=str(animal_name),
        date=str(date),
        epoch=str(epoch),
        region_label=str(region),
        ripple_threshold_zscore=None,
        bin_size_s=float(parameters["bin_size_s"]),
        time_before_s=float(parameters["time_before_s"]),
        time_after_s=float(parameters["time_after_s"]),
        response_window=tuple(parameters["response_window"]),
        baseline_window=tuple(parameters["baseline_window"]),
        heatmap_normalize=str(parameters["heatmap_normalize"]),
    )
    return {
        "summary": f"{stem}_summary.parquet",
        "peri_ripple_firing_rate": f"{stem}_peri_ripple_firing_rate.parquet",
    }


def write_ripple_modulation_artifacts(
    result: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write planned Parquets, refusing to touch existing artifacts by default."""
    paths = {name: Path(artifact_paths[name]) for name in ARTIFACT_NAMES}
    existing_paths = [path for path in paths.values() if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing RippleModulation artifacts: "
            f"{[str(path) for path in existing_paths]!r}."
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    ripple_plot = _ripple_plot_module()
    _write_parquets_atomically(
        {
            "summary": (
                paths["summary"],
                lambda path: ripple_plot.save_summary_table(
                    result["summary"], path
                ),
            ),
            "peri_ripple_firing_rate": (
                paths["peri_ripple_firing_rate"],
                lambda path: ripple_plot.save_peri_ripple_firing_rate_table(
                    result["peri_ripple_firing_rate"], path
                ),
            ),
        }
    )
    return paths


def _write_parquets_atomically(
    writers: Mapping[str, tuple[Path, Callable[[Path], Any]]],
) -> None:
    """Prepare every Parquet beside its destination, then atomically replace it."""
    temporary_paths: dict[str, Path] = {}
    backup_paths: dict[Path, Path] = {}
    replaced_destinations: list[Path] = []
    completed = False
    try:
        for name, (destination, writer) in writers.items():
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp.parquet"
            )
            temporary_paths[name] = temporary_path
            writer(temporary_path)
        for destination, _ in writers.values():
            destination = Path(destination)
            if destination.exists():
                backup_path = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.backup"
                )
                os.replace(destination, backup_path)
                backup_paths[destination] = backup_path
        for name, (destination, _) in writers.items():
            destination = Path(destination)
            os.replace(temporary_paths[name], destination)
            replaced_destinations.append(destination)
        completed = True
    except Exception:
        for destination in replaced_destinations:
            destination.unlink(missing_ok=True)
        for destination, backup_path in backup_paths.items():
            if backup_path.exists():
                os.replace(backup_path, destination)
        raise
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        if completed:
            for backup_path in backup_paths.values():
                backup_path.unlink(missing_ok=True)


def plan_register_existing(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    ripple_modulation_id: Any,
    existing_summary_path: Path,
    existing_peri_ripple_firing_rate_path: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    bin_size_s: float | None = None,
    time_before_s: float | None = None,
    time_after_s: float | None = None,
    response_window: tuple[float, float] | None = None,
    baseline_window: tuple[float, float] | None = None,
    heatmap_normalize: str | None = None,
) -> dict[str, Any]:
    """Return a copy/registration plan containing no database operations."""
    destination_paths = get_ripple_modulation_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        ripple_modulation_id=ripple_modulation_id,
        artifact_root=artifact_root,
    )
    parameters = _resolve_parameters(
        bin_size_s=bin_size_s,
        time_before_s=time_before_s,
        time_after_s=time_after_s,
        response_window=response_window,
        baseline_window=baseline_window,
        heatmap_normalize=heatmap_normalize,
    )
    source_paths = {
        "summary": Path(existing_summary_path),
        "peri_ripple_firing_rate": Path(existing_peri_ripple_firing_rate_path),
    }
    copies = []
    for name in ARTIFACT_NAMES:
        source = source_paths[name]
        destination = destination_paths[name]
        copies.append(
            {
                "artifact": name,
                "source": source,
                "destination": destination,
                "copy_required": source.resolve(strict=False)
                != destination.resolve(strict=False),
            }
        )

    return {
        "operation": "register_existing",
        "key": {
            "animal_name": str(animal_name),
            "date": str(date),
            "epoch": str(epoch),
            "region": str(region),
        },
        "source_paths": source_paths,
        "artifact_paths": {
            name: destination_paths[name] for name in ARTIFACT_NAMES
        },
        "accepted_source_names": {
            name: tuple(
                dict.fromkeys(
                    (
                        _legacy_artifact_names(
                            animal_name=animal_name,
                            date=date,
                            epoch=epoch,
                            region=region,
                            parameters=parameters,
                        )[name],
                        _legacy_artifact_names(
                            animal_name=animal_name,
                            date=date,
                            epoch=epoch,
                            region="all_regions",
                            parameters=parameters,
                        )[name],
                    )
                )
            )
            for name in ARTIFACT_NAMES
        },
        "compute_parameters": parameters,
        "copies": copies,
        "partition_columns": ("animal_name", "date", "epoch", "region"),
        "database_operations": [],
    }


def _select_registration_rows(
    table: Any,
    *,
    key: Mapping[str, Any],
    source: Path,
    allow_empty: bool = False,
) -> Any:
    """Return only rows matching a registration key from one legacy table."""
    required_columns = ("animal_name", "date", "epoch", "region")
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"Existing RippleModulation artifact {source} is missing key "
            f"columns: {missing!r}."
        )

    include = np.ones(len(table), dtype=bool)
    for column in required_columns:
        include &= table[column].astype(str).to_numpy() == str(key[column])
    selected = table.loc[include].reset_index(drop=True)
    if selected.empty:
        if allow_empty and table.empty:
            return selected
        raise ValueError(
            f"Existing RippleModulation artifact {source} has no rows for "
            f"key {dict(key)!r}."
        )
    return selected


def _validate_registration_schema(
    table: Any,
    *,
    artifact_name: str,
    source: Path,
) -> None:
    """Require the complete standalone artifact schema before registration."""
    ripple_plot = _ripple_plot_module()
    required_columns = (
        ripple_plot.SUMMARY_COLUMNS
        if artifact_name == "summary"
        else ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS
    )
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"Existing RippleModulation {artifact_name} artifact {source} is "
            f"missing canonical columns: {missing!r}."
        )


def _validate_peri_ripple_time_grid(
    table: Any,
    *,
    expected: Mapping[str, Any],
    source: Path,
) -> None:
    """Require one complete, common PETH time grid for every legacy unit."""
    if table.empty:
        return
    bin_size_s = float(expected["bin_size_s"])
    expected_grid = np.arange(
        -float(expected["time_before_s"]) + bin_size_s / 2.0,
        float(expected["time_after_s"]),
        bin_size_s,
        dtype=float,
    )
    if not expected_grid.size:
        raise ValueError("RippleModulation parameters produce an empty PETH grid.")
    identity_columns = ["unit_id"]
    if "spikesorting_merge_id" in table.columns:
        identity_columns.insert(0, "spikesorting_merge_id")
        if "nwb_unit_id" in table.columns:
            identity_columns[-1] = "nwb_unit_id"
    for identity, unit_table in table.groupby(identity_columns, sort=False):
        observed = np.sort(unit_table["time_s"].to_numpy(dtype=float))
        if (
            observed.size != expected_grid.size
            or not np.all(np.isfinite(observed))
            or not np.allclose(
                observed,
                expected_grid,
                rtol=1e-7,
                atol=max(1e-12, bin_size_s * 1e-7),
            )
        ):
            raise ValueError(
                "Existing RippleModulation peri-ripple artifact has an "
                f"incomplete or shifted time grid for unit {identity!r}: {source}."
            )
    mean_rate = table["mean_rate_hz"].to_numpy(dtype=float)
    if not np.all(np.isfinite(mean_rate)):
        raise ValueError(
            "Existing RippleModulation peri-ripple mean_rate_hz values must be finite."
        )


def _validate_registration_parameters(
    table: Any,
    *,
    expected: Mapping[str, Any],
    source: Path,
) -> None:
    """Require tabular compute settings to match the destination stem."""
    column_parameters = {
        "bin_size_s": float(expected["bin_size_s"]),
        "time_before_s": float(expected["time_before_s"]),
        "time_after_s": float(expected["time_after_s"]),
    }
    missing = [column for column in column_parameters if column not in table.columns]
    if missing:
        raise ValueError(
            f"Existing RippleModulation artifact {source} is missing compute "
            f"parameter columns: {missing!r}."
        )
    for column, expected_value in column_parameters.items():
        values = table[column].to_numpy(dtype=float)
        if values.size and (
            not np.all(np.isfinite(values))
            or not np.allclose(values, expected_value)
        ):
            unique_values = np.unique(values).tolist()
            raise ValueError(
                f"Existing RippleModulation artifact {source} has {column}="
                f"{unique_values!r}, expected {expected_value!r}."
            )


def read_planned_artifacts(
    plan: Mapping[str, Any],
    *,
    allow_unkeyed_same_path: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Read and validate key-matched Parquets without writing anything."""
    import pandas as pd

    copies = list(plan.get("copies", ()))
    key = dict(plan.get("key", {}))
    required_key_fields = {"animal_name", "date", "epoch", "region"}
    if not required_key_fields.issubset(key):
        raise ValueError(
            "Registration plan key must contain animal_name, date, epoch, and region."
        )
    accepted_source_names = dict(plan.get("accepted_source_names", {}))
    compute_parameters = dict(plan.get("compute_parameters", {}))
    if set(accepted_source_names) != set(ARTIFACT_NAMES) or not compute_parameters:
        raise ValueError("Registration plan is missing source-name or parameter validation.")
    artifact_names = [str(copy.get("artifact")) for copy in copies]
    if sorted(artifact_names) != sorted(ARTIFACT_NAMES):
        raise ValueError(
            "Registration plan must contain exactly one copy for each artifact."
        )
    selected_tables: dict[str, Any] = {}
    for copy in copies:
        artifact_name = str(copy["artifact"])
        source = Path(copy["source"])
        if not source.is_file():
            raise FileNotFoundError(f"Existing artifact was not found: {source}")
        if source.name not in set(accepted_source_names[artifact_name]):
            raise ValueError(
                f"Existing RippleModulation artifact name {source.name!r} does not "
                "match the requested region/all-regions parameter stem. Expected "
                f"one of {accepted_source_names[artifact_name]!r}."
            )
        table = pd.read_parquet(source)
        _validate_registration_schema(
            table,
            artifact_name=artifact_name,
            source=source,
        )
        _validate_registration_parameters(
            table,
            expected=compute_parameters,
            source=source,
        )
        selected = _select_registration_rows(
            table,
            key=key,
            source=source,
            allow_empty=allow_empty,
        )
        if artifact_name == "peri_ripple_firing_rate":
            _validate_peri_ripple_time_grid(
                selected,
                expected=compute_parameters,
                source=source,
            )
        if (
            not copy.get("copy_required", True)
            and len(selected) != len(table)
            and not allow_unkeyed_same_path
        ):
            raise ValueError(
                "A same-path registration source contains rows outside the "
                f"selected artifact key: {source}."
            )
        selected_tables[artifact_name] = selected
    return selected_tables


def write_planned_artifacts(
    plan: Mapping[str, Any],
    tables: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write already validated registration tables to their planned paths."""
    copies = list(plan.get("copies", ()))
    if set(tables) != set(ARTIFACT_NAMES):
        raise ValueError(
            f"tables must contain exactly the artifacts {ARTIFACT_NAMES!r}."
        )
    for copy in copies:
        destination = Path(copy["destination"])
        if copy.get("copy_required", True) and destination.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing RippleModulation artifact: {destination}"
            )

    writers: dict[str, tuple[Path, Callable[[Path], Any]]] = {}
    written_paths: dict[str, Path] = {}
    for copy in copies:
        name = str(copy["artifact"])
        destination = Path(copy["destination"])
        if copy.get("copy_required", True) or overwrite:
            writers[name] = (
                destination,
                lambda path, artifact_name=name: tables[
                    artifact_name
                ].to_parquet(path, index=False),
            )
        written_paths[name] = destination
    if writers:
        _write_parquets_atomically(writers)
    return written_paths


def copy_planned_artifacts(
    plan: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Materialize key-matched Parquets from a registration plan."""
    return write_planned_artifacts(
        plan,
        read_planned_artifacts(
            plan,
            allow_unkeyed_same_path=overwrite,
        ),
        overwrite=overwrite,
    )
