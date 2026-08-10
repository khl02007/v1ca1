"""Build Figure 1 example-cell payloads from an augmented NWB file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.raster.plot_1d_place_field_trajectory import (
    compute_trial_spike_positions,
    make_linear_position_interpolator,
)
from v1ca1.spyglass import movement, path_specific_place
from v1ca1.spyglass.nwb import (
    catalog_augmented_nwb,
    load_interval_set,
    load_position,
    load_wtrack_graph,
)
from v1ca1.spyglass.offline.sources import (
    load_nwb_region_spikes,
    validate_nwb_session_identity,
)
from v1ca1.spyglass.spikes import build_spike_tsgroup
from v1ca1.spyglass.stability import build_task_progression_from_graph


EXAMPLE_PAYLOAD_SCHEMA_VERSION = 1
EXAMPLE_ARTIFACT_DIRNAME = "figure_1_examples"
EXAMPLE_ARTIFACT_FILENAME = "example_payload.npz"
METADATA_ARRAY_NAME = "__metadata__"
DEFAULT_POSITION_ROLE = "head"
DEFAULT_POSITION_OFFSET = 10
DEFAULT_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_SPEED_SMOOTHING_SIGMA_S = 0.1
DEFAULT_POSITION_BIN_COUNT = 50
DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS = 1.5


def _native(value: Any) -> Any:
    """Return a NumPy scalar as its plain-Python value."""
    return value.item() if isinstance(value, np.generic) else value


def _normalized_region(value: Any) -> str:
    """Return one non-empty case-insensitive region label."""
    region = str(_native(value)).strip().casefold()
    if not region:
        raise ValueError("region must be non-empty.")
    return region


def _sorting_unit_token(value: Any) -> str:
    """Return one stable comparison token for a sorting unit identifier."""
    value = _native(value)
    if isinstance(value, bool) or value is None:
        raise ValueError("sorting_unit_id must be a non-boolean scalar.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("sorting_unit_id must be finite and integer-valued.")
        return str(int(numeric))
    token = str(value).strip()
    if not token:
        raise ValueError("sorting_unit_id must be non-empty.")
    return token


def _one_catalog_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    description: str,
    **selectors: str,
) -> dict[str, Any]:
    """Return exactly one catalog row matching the requested selectors."""
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


def _validate_trajectory_types(values: Sequence[str]) -> tuple[str, ...]:
    """Return a non-empty unique tuple of supported trajectory names."""
    trajectories = tuple(str(value) for value in values)
    if not trajectories or len(trajectories) != len(set(trajectories)):
        raise ValueError("trajectory_types must be a non-empty unique sequence.")
    unknown = sorted(set(trajectories).difference(TRAJECTORY_TYPES))
    if unknown:
        raise ValueError(
            f"Unknown trajectory types {unknown!r}; expected {TRAJECTORY_TYPES!r}."
        )
    return trajectories


def select_example_epoch_catalog(
    nwbfile: Any,
    *,
    nwb_file_name: str,
    epoch: str,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    position_role: str = DEFAULT_POSITION_ROLE,
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> dict[str, Any]:
    """Select one light or dark run epoch's example-cell NWB inputs."""
    trajectories = _validate_trajectory_types(trajectory_types)
    if str(position_role) != DEFAULT_POSITION_ROLE:
        raise ValueError("Figure 1 example cells require head position.")
    try:
        requested_offset = float(position_offset)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("position_offset must be numeric.") from exc
    if not np.isfinite(requested_offset) or requested_offset != DEFAULT_POSITION_OFFSET:
        raise ValueError(
            f"Figure 1 example cells require position offset {DEFAULT_POSITION_OFFSET}."
        )

    catalog = catalog_augmented_nwb(nwbfile, nwb_file_name=nwb_file_name)
    epoch_row = _one_catalog_row(
        catalog["epoch_intervals"],
        description="epoch",
        epoch=str(epoch),
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError(f"Figure 1 examples require a run epoch, got {epoch!r}.")

    position_row = _one_catalog_row(
        catalog["position"],
        description="position",
        epoch=str(epoch),
        position_role=DEFAULT_POSITION_ROLE,
    )
    if position_row.get("spatial_unit") != "cm":
        raise ValueError("Figure 1 example position must use centimeters.")
    try:
        stored_offset = float(
            position_row.get("analysis_start_offset_samples", np.nan)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Figure 1 example position has an invalid analysis offset."
        ) from exc
    if not np.isfinite(stored_offset) or stored_offset != DEFAULT_POSITION_OFFSET:
        raise ValueError(
            "Figure 1 example position must declare "
            f"analysis_start_offset_samples={DEFAULT_POSITION_OFFSET}; "
            f"got {position_row.get('analysis_start_offset_samples')!r}."
        )

    trajectory_rows = {}
    graph_rows = {}
    for trajectory_type in trajectories:
        trajectory_rows[trajectory_type] = _one_catalog_row(
            catalog["trajectory_intervals"],
            description="trajectory",
            epoch=str(epoch),
            trajectory_type=trajectory_type,
        )
        graph_rows[trajectory_type] = _one_catalog_row(
            catalog["wtrack_graph"],
            description="W-track graph",
            configuration_name=trajectory_type,
        )
        if graph_rows[trajectory_type].get("coordinate_unit") != "cm":
            raise ValueError("Figure 1 W-track coordinates must use centimeters.")
    return {
        "epoch_row": epoch_row,
        "position_row": position_row,
        "trajectory_rows": trajectory_rows,
        "graph_rows": graph_rows,
    }


def load_example_epoch_sources(
    nwbfile: Any,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Load selected position, trajectory intervals, and W-track inputs."""
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


def resolve_example_unit(
    loaded_spikes: Mapping[str, Any],
    *,
    region: str,
    sorting_unit_id: Any,
) -> dict[str, Any]:
    """Resolve ``(region, sorting_unit_id)`` to one persistent source unit."""
    requested_region = _normalized_region(region)
    if _normalized_region(loaded_spikes.get("region")) != requested_region:
        raise ValueError("Loaded spike region does not match the requested region.")
    if loaded_spikes.get("source") != "ImportedSpikeSorting":
        raise ValueError(
            "Offline Figure 1 examples require augmented-NWB ImportedSpikeSorting."
        )

    metadata = [dict(row) for row in loaded_spikes.get("unit_metadata", ())]
    spike_times = list(loaded_spikes.get("spike_times_s", ()))
    unit_ids = [dict(row) for row in loaded_spikes.get("unit_ids", ())]
    if not (len(metadata) == len(spike_times) == len(unit_ids)):
        raise ValueError(
            "Loaded unit metadata, identities, and spike trains must align."
        )

    requested_token = _sorting_unit_token(sorting_unit_id)
    matches: list[int] = []
    for index, row in enumerate(metadata):
        if "sorting_unit_id" not in row or "region" not in row:
            raise ValueError(
                "Augmented NWB unit metadata requires sorting_unit_id and region."
            )
        if (
            _normalized_region(row["region"]) == requested_region
            and _sorting_unit_token(row["sorting_unit_id"]) == requested_token
        ):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one augmented-NWB unit for "
            f"(region={requested_region!r}, sorting_unit_id={requested_token!r}); "
            f"found {len(matches)}."
        )

    index = matches[0]
    identity = unit_ids[index]
    metadata_identity = {
        name: metadata[index].get(name)
        for name in ("spikesorting_merge_id", "unit_id")
    }
    if identity != metadata_identity:
        raise ValueError("Unit metadata and persistent identity disagree.")
    times = np.asarray(spike_times[index], dtype=float).reshape(-1)
    if not np.all(np.isfinite(times)) or (
        times.size > 1 and np.any(np.diff(times) < 0.0)
    ):
        raise ValueError("Example spike times must be finite, sorted seconds.")

    merge_id = str(identity["spikesorting_merge_id"])
    source_unit_id = _native(identity["unit_id"])
    return {
        "region": requested_region,
        "sorting_unit_id": _native(metadata[index]["sorting_unit_id"]),
        "spikesorting_merge_id": merge_id,
        "unit_id": source_unit_id,
        "stable_unit_id": f"{merge_id}:{source_unit_id}",
        "spike_times_s": times,
        "spike_time_unit": "s",
        "spike_time_reference": "NWB/ephys timestamps",
        "source": "ImportedSpikeSorting",
    }


def _source_pointer(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small source pointer subset needed for provenance."""
    return {
        name: _native(row.get(name))
        for name in (
            "source_table_path",
            "source_table_object_id",
            "source_object_path",
            "source_object_id",
            "source_row_index",
        )
        if row.get(name) is not None
    }


def _extract_example_rate(curve: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized path centers and the sole example unit's rate."""
    values = np.asarray(curve.values, dtype=float)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("Example tuning output must contain exactly one unit.")
    position = np.asarray(curve.coords["path_fraction"].values, dtype=float)
    if position.shape != (values.shape[1],):
        raise ValueError("Example tuning positions do not align with rate bins.")
    return position, values[0]


def compute_example_payload_from_sources(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    example_unit: Mapping[str, Any],
    spikes: Any,
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    movement_analysis_status: str,
    selection: Mapping[str, Any],
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    sigma_bins: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
) -> dict[str, Any]:
    """Compute all-trial rasters and movement-restricted rate curves."""
    trajectories = _validate_trajectory_types(tuple(trajectory_intervals))
    if set(trajectories) != set(graph_inputs):
        raise ValueError("Trajectory intervals and W-track inputs must align.")
    try:
        requested_bin_count = float(position_bin_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("position_bin_count must be numeric.") from exc
    if (
        not np.isfinite(requested_bin_count)
        or requested_bin_count != DEFAULT_POSITION_BIN_COUNT
    ):
        raise ValueError(
            f"Figure 1 examples require {DEFAULT_POSITION_BIN_COUNT} bins."
        )
    if float(sigma_bins) != DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS:
        raise ValueError(
            "Figure 1 examples require Gaussian smoothing sigma "
            f"{DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS} bins."
        )
    if float(speed_threshold_cm_s) != DEFAULT_SPEED_THRESHOLD_CM_S:
        raise ValueError(
            "Figure 1 examples require movement speed threshold "
            f"{DEFAULT_SPEED_THRESHOLD_CM_S} cm/s."
        )
    if float(speed_smoothing_sigma_s) != DEFAULT_SPEED_SMOOTHING_SIGMA_S:
        raise ValueError(
            "Figure 1 examples require speed smoothing sigma "
            f"{DEFAULT_SPEED_SMOOTHING_SIGMA_S} s."
        )

    identity = {
        "spikesorting_merge_id": example_unit["spikesorting_merge_id"],
        "unit_id": example_unit["unit_id"],
    }
    spike_times_s = np.asarray(example_unit["spike_times_s"], dtype=float)
    raster_positions: dict[str, list[np.ndarray]] = {}
    firing_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    analysis_status: dict[str, str] = {}
    for trajectory_type in trajectories:
        progression, _graph_length_cm = build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals[trajectory_type],
            graph_inputs=graph_inputs[trajectory_type],
            trajectory_type=trajectory_type,
        )
        interpolator = make_linear_position_interpolator(progression)
        trials = compute_trial_spike_positions(
            spike_times_s,
            trajectory_intervals[trajectory_type],
            interpolator,
        )
        raster_positions[trajectory_type] = [
            np.asarray(values, dtype=float)[
                (np.asarray(values, dtype=float) >= 0.0)
                & (np.asarray(values, dtype=float) <= 1.0)
            ]
            for values in trials
        ]

        result = path_specific_place.compute_selected_path_specific_place_tuning_curve(
            animal_name=str(animal_name),
            date=str(date),
            region=str(example_unit["region"]),
            epoch=str(epoch),
            trajectory_type=trajectory_type,
            trial_subset="all",
            spikes=spikes,
            stable_unit_ids=(identity,),
            position=position,
            trajectory_intervals=trajectory_intervals[trajectory_type],
            graph_inputs=graph_inputs[trajectory_type],
            movement_intervals=movement_intervals,
            movement_analysis_status=str(movement_analysis_status),
            bin_count=int(position_bin_count),
            sigma_bins=float(sigma_bins),
        )
        firing_rates[trajectory_type] = _extract_example_rate(
            result["tuning_curve"]
        )
        analysis_status[trajectory_type] = str(result["analysis_status"])

    position_row = selection["position_row"]
    metadata = {
        "schema_version": EXAMPLE_PAYLOAD_SCHEMA_VERSION,
        "payload_type": "figure_1_example_raster_and_rate",
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(epoch),
        "nwb_file_name": str(nwb_file_name),
        "region": str(example_unit["region"]),
        "sorting_unit_id": _native(example_unit["sorting_unit_id"]),
        "persistent_unit_identity": {
            "source": str(example_unit["source"]),
            "spikesorting_merge_id": str(example_unit["spikesorting_merge_id"]),
            "unit_id": _native(example_unit["unit_id"]),
            "stable_unit_id": str(example_unit["stable_unit_id"]),
        },
        "spike_source": "RegionSortedSpikesGroup-equivalent augmented NWB Units",
        "spike_time_unit": "s",
        "spike_time_reference": "NWB/ephys timestamps",
        "trajectory_types": list(trajectories),
        "analysis_status": analysis_status,
        "parameters": {
            "position_role": DEFAULT_POSITION_ROLE,
            "spatial_unit": "cm",
            "analysis_start_offset_samples": DEFAULT_POSITION_OFFSET,
            "speed_threshold_cm_s": float(speed_threshold_cm_s),
            "speed_smoothing_sigma_s": float(speed_smoothing_sigma_s),
            "position_bin_count": int(position_bin_count),
            "gaussian_smoothing_sigma_bins": float(sigma_bins),
            "raster_spike_support": "complete_trajectory_intervals",
            "rate_support": "trajectory_intervals_intersect_movement",
        },
        "epoch_provenance": {
            **_source_pointer(selection["epoch_row"]),
            "condition": _native(selection["epoch_row"].get("condition")),
            "is_light": _native(selection["epoch_row"].get("is_light")),
        },
        "position_provenance": {
            **_source_pointer(position_row),
            "position_series_name": str(position_row["position_series_name"]),
            "position_role": str(position_row["position_role"]),
            "spatial_unit": str(position_row["spatial_unit"]),
            "analysis_start_offset_samples": int(
                position_row["analysis_start_offset_samples"]
            ),
        },
        "trajectory_provenance": {
            trajectory_type: _source_pointer(
                selection["trajectory_rows"][trajectory_type]
            )
            for trajectory_type in trajectories
        },
        "wtrack_provenance": {
            trajectory_type: _source_pointer(selection["graph_rows"][trajectory_type])
            for trajectory_type in trajectories
        },
    }
    payload = {
        "metadata": metadata,
        "raster_positions": raster_positions,
        "firing_rates": firing_rates,
    }
    return validate_example_payload(payload)


def compute_nwb_example_payload(
    nwbfile: Any,
    *,
    nwb_file_name: str,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    sorting_unit_id: Any,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    sigma_bins: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
) -> dict[str, Any]:
    """Compute one example payload directly from an already-open NWB file."""
    validate_nwb_session_identity(
        nwbfile,
        animal_name=str(animal_name),
        date=str(date),
    )
    selection = select_example_epoch_catalog(
        nwbfile,
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        trajectory_types=trajectory_types,
        position_offset=position_offset,
    )
    sources = load_example_epoch_sources(nwbfile, selection)
    start = float(selection["epoch_row"]["start_time"])
    stop = float(selection["epoch_row"]["stop_time"])
    loaded_spikes = load_nwb_region_spikes(
        nwbfile,
        nwb_file_name=nwb_file_name,
        region=region,
        time_support=(start, stop),
    )
    example_unit = resolve_example_unit(
        loaded_spikes,
        region=region,
        sorting_unit_id=sorting_unit_id,
    )
    identity = {
        "spikesorting_merge_id": example_unit["spikesorting_merge_id"],
        "unit_id": example_unit["unit_id"],
    }
    spikes = build_spike_tsgroup(
        (example_unit["spike_times_s"],),
        (identity,),
        time_support=(start, stop),
    )
    movement_result = movement.compute_selected_movement_firing_rate(
        animal_name=str(animal_name),
        date=str(date),
        region=_normalized_region(region),
        epoch=str(epoch),
        spikes=spikes,
        stable_unit_ids=(identity,),
        position=sources["position"],
        speed_threshold_cm_s=float(speed_threshold_cm_s),
        speed_smoothing_sigma_s=float(speed_smoothing_sigma_s),
    )
    return compute_example_payload_from_sources(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        nwb_file_name=nwb_file_name,
        example_unit=example_unit,
        spikes=spikes,
        position=sources["position"],
        trajectory_intervals=sources["trajectory_intervals"],
        graph_inputs=sources["graph_inputs"],
        movement_intervals=movement_result["movement_intervals"],
        movement_analysis_status=movement_result["analysis_status"],
        selection=selection,
        position_bin_count=position_bin_count,
        sigma_bins=sigma_bins,
        speed_threshold_cm_s=speed_threshold_cm_s,
        speed_smoothing_sigma_s=speed_smoothing_sigma_s,
    )


def validate_example_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one in-memory example-cell payload."""
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("schema_version") != EXAMPLE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("Unsupported Figure 1 example payload schema version.")
    trajectories = _validate_trajectory_types(metadata.get("trajectory_types", ()))
    raster_positions = dict(payload.get("raster_positions", {}))
    firing_rates = dict(payload.get("firing_rates", {}))
    if set(raster_positions) != set(trajectories) or set(firing_rates) != set(
        trajectories
    ):
        raise ValueError("Example payload trajectory arrays do not match metadata.")

    validated_rasters: dict[str, list[np.ndarray]] = {}
    validated_rates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_bins = int(metadata["parameters"]["position_bin_count"])
    for trajectory_type in trajectories:
        trials = []
        for raw_values in raster_positions[trajectory_type]:
            values = np.asarray(raw_values, dtype=float).reshape(-1)
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError("Raster positions must be finite and within [0, 1].")
            trials.append(values)
        validated_rasters[trajectory_type] = trials

        raw_position, raw_rate = firing_rates[trajectory_type]
        position = np.asarray(raw_position, dtype=float).reshape(-1)
        rate = np.asarray(raw_rate, dtype=float).reshape(-1)
        if position.shape != (expected_bins,) or rate.shape != position.shape:
            raise ValueError("Firing-rate arrays must match the declared bin count.")
        if not np.all(np.isfinite(position)) or np.any(np.diff(position) <= 0.0):
            raise ValueError("Rate positions must be finite and strictly increasing.")
        if np.any(np.isinf(rate)):
            raise ValueError("Firing rates may be finite or NaN, not infinite.")
        validated_rates[trajectory_type] = (position, rate)
    return {
        "metadata": metadata,
        "raster_positions": validated_rasters,
        "firing_rates": validated_rates,
    }


def get_example_payload_path(
    run_dir: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    sorting_unit_id: Any,
) -> Path:
    """Return the run-local path for one example-cell payload."""
    components = (animal_name, date, epoch, _normalized_region(region))
    if any(
        not str(value)
        or str(value) in {".", ".."}
        or Path(str(value)).name != str(value)
        for value in components
    ):
        raise ValueError("Example path identifiers must be single path components.")
    unit_token = _sorting_unit_token(sorting_unit_id)
    if unit_token in {".", ".."} or Path(unit_token).name != unit_token:
        raise ValueError("sorting_unit_id must be one path-safe identifier.")
    return (
        Path(run_dir)
        / str(animal_name)
        / str(date)
        / EXAMPLE_ARTIFACT_DIRNAME
        / str(epoch)
        / _normalized_region(region)
        / f"sorting_unit_{unit_token}"
        / EXAMPLE_ARTIFACT_FILENAME
    )


def _artifact_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Serialize one validated payload to non-pickle NumPy arrays."""
    validated = validate_example_payload(payload)
    metadata = validated["metadata"]
    arrays = {
        METADATA_ARRAY_NAME: np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
    }
    for trajectory_type in metadata["trajectory_types"]:
        trials = validated["raster_positions"][trajectory_type]
        arrays[f"raster_{trajectory_type}_values"] = (
            np.concatenate(trials) if trials else np.asarray([], dtype=float)
        )
        arrays[f"raster_{trajectory_type}_lengths"] = np.asarray(
            [len(values) for values in trials],
            dtype=np.int64,
        )
        position, rate = validated["firing_rates"][trajectory_type]
        arrays[f"rate_{trajectory_type}_position"] = position
        arrays[f"rate_{trajectory_type}_values"] = rate
    return arrays


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_example_payload(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Write one run-local NPZ atomically and refuse every overwrite."""
    output_path = Path(output_path)
    run_dir = Path(run_dir).resolve(strict=False)
    resolved_output = output_path.resolve(strict=False)
    if not resolved_output.is_relative_to(run_dir):
        raise ValueError("Example payload must be written inside run_dir.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite example payload: {output_path}")

    arrays = _artifact_arrays(payload)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(buffer.getbuffer())
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite example payload: {output_path}"
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "payload_path": output_path,
        "artifact_sha256": _file_sha256(output_path),
    }


def load_example_payload(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and verify one run-local example-cell NPZ payload."""
    path = Path(path)
    if expected_sha256 is not None and _file_sha256(path) != str(expected_sha256):
        raise ValueError(f"Example payload SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data[METADATA_ARRAY_NAME].item()))
        raster_positions = {}
        firing_rates = {}
        for trajectory_type in metadata["trajectory_types"]:
            values = np.asarray(data[f"raster_{trajectory_type}_values"], dtype=float)
            lengths = np.asarray(data[f"raster_{trajectory_type}_lengths"], dtype=int)
            if np.any(lengths < 0) or int(np.sum(lengths)) != values.size:
                raise ValueError("Stored raster lengths do not align with values.")
            split_points = np.cumsum(lengths)[:-1]
            raster_positions[trajectory_type] = (
                [
                    np.asarray(part, dtype=float)
                    for part in np.split(values, split_points)
                ]
                if lengths.size
                else []
            )
            firing_rates[trajectory_type] = (
                np.asarray(data[f"rate_{trajectory_type}_position"], dtype=float),
                np.asarray(data[f"rate_{trajectory_type}_values"], dtype=float),
            )
    return validate_example_payload(
        {
            "metadata": metadata,
            "raster_positions": raster_positions,
            "firing_rates": firing_rates,
        }
    )


__all__ = [
    "DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS",
    "DEFAULT_POSITION_BIN_COUNT",
    "DEFAULT_POSITION_OFFSET",
    "DEFAULT_SPEED_THRESHOLD_CM_S",
    "compute_example_payload_from_sources",
    "compute_nwb_example_payload",
    "get_example_payload_path",
    "load_example_epoch_sources",
    "load_example_payload",
    "resolve_example_unit",
    "select_example_epoch_catalog",
    "validate_example_payload",
    "write_example_payload",
]
