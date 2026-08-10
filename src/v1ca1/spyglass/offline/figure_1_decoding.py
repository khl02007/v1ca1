"""Run and adapt Figure 1 cross-path decoding without DataJoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
import v1ca1.spyglass.path_progression_decoding as decoding
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import (
    provenance_sha256,
    selection_uuid,
    unit_identity_sha256,
)
from v1ca1.spyglass.table_specs import (
    MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS,
)


FIGURE_1_DECODING_REGION = "v1"
FIGURE_1_DECODING_PARAMETERS = MappingProxyType(
    dict(MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS)
)
FIGURE_1_DECODING_COMPARISONS = (
    (
        "same_turn_cross_arm",
        "Same turn\ncross arm",
        "same_turn_cross_arm",
        (
            ("center_to_left", "right_to_center"),
            ("right_to_center", "center_to_left"),
            ("center_to_right", "left_to_center"),
            ("left_to_center", "center_to_right"),
        ),
    ),
    (
        "opposite_turn_same_arm",
        "Opposite turn\nsame arm",
        "opposite_turn_same_arm",
        (
            ("center_to_left", "left_to_center"),
            ("left_to_center", "center_to_left"),
            ("center_to_right", "right_to_center"),
            ("right_to_center", "center_to_right"),
        ),
    ),
    (
        "same_inbound_outbound_cross_arm",
        "Same in/out\ncross arm",
        "same_inbound_outbound_cross_arm",
        (
            ("center_to_left", "center_to_right"),
            ("center_to_right", "center_to_left"),
            ("left_to_center", "right_to_center"),
            ("right_to_center", "left_to_center"),
        ),
    ),
)
ABSOLUTE_ERROR_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "comparison",
    "comparison_label",
    "transfer_family",
    "encoding_trajectory",
    "decoding_trajectory",
    "absolute_error",
    "true_path",
    "decoded_path",
)
TRIAL_ERROR_COLUMNS = (
    "animal_name",
    "date",
    "epoch",
    "region",
    "comparison",
    "comparison_label",
    "transfer_family",
    "encoding_trajectory",
    "decoding_trajectory",
    "trial_index",
    "trial_start",
    "trial_end",
    "trial_median_absolute_error",
    "n_samples",
    "true_path",
    "decoded_path",
)
_ARTIFACT_FILE_FIELDS = (
    "artifact_manifest_path",
    "decoding_summary_path",
    "unit_eligibility_path",
    "selected_units_path",
    "binned_error_path",
)


def _source_uuid(value: Any, *, name: str) -> str:
    """Return one canonical UUID identifying an offline upstream source."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def _path_component(value: Any, *, name: str) -> str:
    """Return a non-empty path-safe source label."""
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return value


def _stability_source_ids(
    values: Mapping[str, Any],
) -> dict[str, str]:
    """Return exactly four canonical path-specific stability UUIDs."""
    if set(values) != set(TRAJECTORY_TYPES):
        raise ValueError(
            "stability_source_ids must contain exactly the four trajectories."
        )
    return {
        trajectory_type: _source_uuid(
            values[trajectory_type],
            name=f"{trajectory_type} stability source",
        )
        for trajectory_type in TRAJECTORY_TYPES
    }


def build_figure_1_decoding_selection(
    *,
    nwb_file_name: str,
    epoch: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    stability_source_ids: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic same-epoch Figure 1 decoding selection."""
    nwb_file_name = _path_component(nwb_file_name, name="nwb_file_name")
    epoch = _path_component(epoch, name="epoch")
    region_group_id = _source_uuid(
        region_sorted_spikes_group_id,
        name="region_sorted_spikes_group_id",
    )
    movement_id = _source_uuid(
        movement_firing_rate_id,
        name="movement_firing_rate_id",
    )
    stability_ids = _stability_source_ids(stability_source_ids)
    parameters = dict(FIGURE_1_DECODING_PARAMETERS)
    parameter_sha256 = provenance_sha256(parameters)
    eligibility_rule_sha256 = provenance_sha256(dict(decoding.ELIGIBILITY_RULE))
    decoding_output_rule_sha256 = provenance_sha256(
        dict(decoding.DECODING_OUTPUT_RULE)
    )
    source_fields: dict[str, Any] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        source_fields[f"{trajectory_type}_trajectory_type"] = trajectory_type
        source_fields[f"{trajectory_type}_configuration_name"] = trajectory_type
        stability_id = stability_ids[trajectory_type]
        source_fields[f"{trajectory_type}_stability_id"] = stability_id
        source_fields[f"cohort_{trajectory_type}_stability_id"] = stability_id
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "region_sorted_spikes_group_id": region_group_id,
        "movement_firing_rate_id": movement_id,
        "cohort_movement_firing_rate_id": movement_id,
        **source_fields,
        "path_progression_decoding_param_name": parameters[
            "path_progression_decoding_param_name"
        ],
        "cohort_epoch": epoch,
        "path_progression_decoding_parameters_sha256": parameter_sha256,
        "eligibility_rule_sha256": eligibility_rule_sha256,
        "transfer_spec_sha256": decoding.TRANSFER_SPEC_SHA256,
        "decoding_output_rule_sha256": decoding_output_rule_sha256,
    }
    return {
        "path_progression_decoding_id": str(
            selection_uuid("PathProgressionDecoding", natural_key)
        ),
        **natural_key,
    }


def _output_relative_path(path: Path, *, output_dir: Path) -> str:
    """Return a guarded path relative to the explicit output directory."""
    resolved = Path(path).resolve(strict=True)
    root = Path(output_dir).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path escapes output_dir: {resolved}")
    return resolved.relative_to(root).as_posix()


def _artifact_record(
    *,
    selection: Mapping[str, Any],
    result: Mapping[str, Any],
    written: Mapping[str, Any],
    output_dir: Path,
    input_units_sha256: str,
) -> dict[str, Any]:
    """Return one output-relative, checksummed manifest-ready record."""
    artifact_dir = Path(written["path"]).resolve(strict=True)
    root = Path(output_dir).resolve(strict=False)
    if not artifact_dir.is_relative_to(root):
        raise ValueError("Decoder wrote outside the explicit output directory.")
    record = {
        **dict(selection),
        "region": FIGURE_1_DECODING_REGION,
        "artifact_origin": "computed",
        "selection_identity_scope": "offline_surrogate",
        "selection_sha256": provenance_sha256(dict(selection)),
        "input_units_sha256": str(input_units_sha256),
        "artifact_dir": artifact_dir.relative_to(root).as_posix(),
        **{
            name: _output_relative_path(Path(written[name]), output_dir=root)
            for name in _ARTIFACT_FILE_FIELDS
        },
        **{
            name: int(result[name])
            for name in (
                "n_units_input",
                "n_units_eligible",
                "n_transfer_pairs_expected",
                "n_transfer_pairs_valid",
                "n_decoded_samples",
            )
        },
        "analysis_status": str(result["analysis_status"]),
        "eligible_units_sha256": str(result["eligible_units_sha256"]),
    }
    record["artifact_sha256"] = {
        name: file_sha256(Path(written[name]))
        for name in _ARTIFACT_FILE_FIELDS
    }
    return record


def run_figure_1_decoding(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    stability_source_ids: Mapping[str, Any],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_interval: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute and persist the V1 Figure 1 decoder from loaded NWB inputs."""
    animal_name = _path_component(animal_name, name="animal_name")
    date = _path_component(date, name="date")
    epoch = _path_component(epoch, name="epoch")
    if set(stability_tables_by_trajectory) != set(TRAJECTORY_TYPES):
        raise ValueError(
            "stability_tables_by_trajectory must contain exactly four paths."
        )
    if set(trajectory_intervals) != set(TRAJECTORY_TYPES):
        raise ValueError("trajectory_intervals must contain exactly four paths.")
    if set(graph_inputs) != set(TRAJECTORY_TYPES):
        raise ValueError("graph_inputs must contain exactly four path graphs.")
    stable_unit_ids = tuple(dict(row) for row in stable_unit_ids)
    input_units_sha256 = unit_identity_sha256(stable_unit_ids)
    selection = build_figure_1_decoding_selection(
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        region_sorted_spikes_group_id=region_sorted_spikes_group_id,
        movement_firing_rate_id=movement_firing_rate_id,
        stability_source_ids=stability_source_ids,
    )

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {root}")
    paths = decoding.get_decoding_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        cohort_epoch=epoch,
        region=FIGURE_1_DECODING_REGION,
        path_progression_decoding_id=selection[
            "path_progression_decoding_id"
        ],
        artifact_root=root,
    )
    artifact_dir = Path(paths["artifact_dir"]).resolve(strict=False)
    if not artifact_dir.is_relative_to(root) or artifact_dir == root:
        raise ValueError("Canonical decoder path escapes output_dir.")
    if artifact_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite decoding artifact directory: {artifact_dir}"
        )

    parameters = dict(FIGURE_1_DECODING_PARAMETERS)
    result = decoding.compute_path_progression_decoding(
        animal_name=animal_name,
        date=date,
        region=FIGURE_1_DECODING_REGION,
        epoch=epoch,
        cohort_epoch=epoch,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        target_movement_firing_rate_table=movement_firing_rate_table,
        cohort_movement_firing_rate_table=movement_firing_rate_table,
        target_stability_tables_by_trajectory=stability_tables_by_trajectory,
        cohort_stability_tables_by_trajectory=stability_tables_by_trajectory,
        position=position,
        trajectory_intervals=trajectory_intervals,
        graph_inputs=graph_inputs,
        movement_interval=movement_interval,
        path_progression_decoding_id=selection[
            "path_progression_decoding_id"
        ],
        decoding_bin_size_s=parameters["decoding_bin_size_s"],
        sliding_window_size_bins=parameters["sliding_window_size_bins"],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=None,
        parameter_name=parameters["path_progression_decoding_param_name"],
        parameter_sha256=selection[
            "path_progression_decoding_parameters_sha256"
        ],
        eligibility_rule_sha256=selection["eligibility_rule_sha256"],
        transfer_spec_sha256=selection["transfer_spec_sha256"],
        decoding_output_rule_sha256=selection[
            "decoding_output_rule_sha256"
        ],
    )
    written = decoding.write_decoding_artifact_bundle(
        result,
        paths,
        overwrite=False,
    )
    return {
        "selection": selection,
        "result": result,
        "artifact_record": _artifact_record(
            selection=selection,
            result=result,
            written=written,
            output_dir=root,
            input_units_sha256=input_units_sha256,
        ),
    }


def _interval_arrays(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted start and end arrays for one trajectory interval set."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError("Trajectory interval starts and ends do not match.")
    order = np.argsort(starts)
    return starts[order], ends[order]


def _aligned_absolute_error_with_times(
    true_tsd: Any,
    decoded_tsd: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decoded timestamps and finite aligned absolute errors."""
    if len(np.asarray(decoded_tsd.t)) == 0 or len(np.asarray(true_tsd.t)) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    support = true_tsd.time_support.intersect(decoded_tsd.time_support)
    true_restricted = true_tsd.restrict(support)
    decoded_restricted = decoded_tsd.restrict(support)
    true_at_decoded = true_restricted.interpolate(
        decoded_restricted,
        ep=support,
        left=np.nan,
        right=np.nan,
    )
    timestamps = np.asarray(decoded_restricted.t, dtype=float)
    true_values = np.asarray(true_at_decoded.d, dtype=float)
    decoded_values = np.asarray(decoded_restricted.d, dtype=float)
    valid = (
        np.isfinite(timestamps)
        & np.isfinite(true_values)
        & np.isfinite(decoded_values)
    )
    return timestamps[valid], np.abs(decoded_values[valid] - true_values[valid])


def _output_file_path(
    bundle: Mapping[str, Any],
    *,
    transfer_key: tuple[str, str, str],
    role: str,
) -> Path:
    """Return one checksummed canonical output path from a loaded bundle."""
    family, source, target = transfer_key
    artifact_key = f"cross:{family}:{source}:{target}:{role}"
    manifest = bundle["manifest"]
    rows = manifest.loc[manifest["artifact_key"].astype(str) == artifact_key]
    if len(rows) != 1:
        raise ValueError(f"Decoder manifest is missing {artifact_key!r}.")
    relative = Path(str(rows.iloc[0]["relative_path"]))
    if relative.name != str(relative):
        raise ValueError("Decoder manifest output must be a direct child file.")
    return Path(bundle["path"]) / relative


def load_figure_1_decoding_payload(
    *,
    artifact_manifest_path: Path,
    trajectory_intervals: Mapping[str, Any],
) -> dict[str, Any]:
    """Load canonical decoding outputs for Figure 1 summaries and inference."""
    if set(trajectory_intervals) != set(TRAJECTORY_TYPES):
        raise ValueError("trajectory_intervals must contain exactly four paths.")
    bundle = decoding.load_decoding_artifact_bundle(artifact_manifest_path)
    metadata = bundle["metadata"]
    if str(metadata["region"]).casefold() != FIGURE_1_DECODING_REGION:
        raise ValueError("Figure 1 decoding requires a V1 artifact.")
    if str(metadata["epoch"]) != str(metadata["cohort_epoch"]):
        raise ValueError("Figure 1 decoding requires target epoch as its cohort.")
    expected_parameters = dict(FIGURE_1_DECODING_PARAMETERS)
    if str(metadata["parameter_name"]) != str(
        expected_parameters["path_progression_decoding_param_name"]
    ):
        raise ValueError("Figure 1 decoding uses an unexpected parameter preset.")
    parameters = bundle["parameters"]
    for name in (
        "decoding_bin_size_s",
        "sliding_window_size_bins",
        "spatial_bin_size_cm",
        "minimum_movement_firing_rate_hz",
    ):
        if not np.isclose(
            float(parameters[name]),
            float(expected_parameters[name]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"Figure 1 decoding parameter {name!r} differs.")
    if parameters["minimum_stability_correlation"] is not None:
        raise ValueError("Figure 1 decoding must not apply a stability threshold.")

    sample_tables = []
    trial_rows = []
    for comparison, label, family, pairs in FIGURE_1_DECODING_COMPARISONS:
        for source, target in pairs:
            key = (family, source, target)
            if key not in bundle["cross_path_outputs"]:
                raise ValueError(
                    "Figure 1 requires a valid output for transfer "
                    f"{family}:{source}:{target}."
                )
            output = bundle["cross_path_outputs"][key]
            timestamps, absolute_error = _aligned_absolute_error_with_times(
                output["true"],
                output["decoded"],
            )
            true_path = _output_file_path(
                bundle,
                transfer_key=key,
                role="true",
            )
            decoded_path = _output_file_path(
                bundle,
                transfer_key=key,
                role="decoded",
            )
            common = {
                "animal_name": str(metadata["animal_name"]),
                "date": str(metadata["date"]),
                "epoch": str(metadata["epoch"]),
                "region": FIGURE_1_DECODING_REGION,
                "comparison": comparison,
                "comparison_label": label,
                "transfer_family": family,
                "encoding_trajectory": source,
                "decoding_trajectory": target,
                "true_path": str(true_path),
                "decoded_path": str(decoded_path),
            }
            if absolute_error.size:
                sample_tables.append(
                    pd.DataFrame(
                        {
                            **common,
                            "absolute_error": absolute_error,
                        }
                    ).loc[:, list(ABSOLUTE_ERROR_COLUMNS)]
                )
            starts, ends = _interval_arrays(trajectory_intervals[target])
            for trial_index, (start, end) in enumerate(
                zip(starts, ends, strict=True)
            ):
                in_trial = (timestamps >= start) & (timestamps < end)
                values = absolute_error[in_trial]
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                trial_rows.append(
                    {
                        **common,
                        "trial_index": int(trial_index),
                        "trial_start": float(start),
                        "trial_end": float(end),
                        "trial_median_absolute_error": float(np.median(values)),
                        "n_samples": int(values.size),
                    }
                )
    absolute_error_table = (
        pd.concat(sample_tables, ignore_index=True)
        if sample_tables
        else pd.DataFrame(columns=list(ABSOLUTE_ERROR_COLUMNS))
    )
    trial_error_table = pd.DataFrame.from_records(
        trial_rows,
        columns=list(TRIAL_ERROR_COLUMNS),
    )
    return {
        "bundle": bundle,
        "absolute_error_table": absolute_error_table,
        "trial_error_table": trial_error_table,
    }


__all__ = [
    "ABSOLUTE_ERROR_COLUMNS",
    "FIGURE_1_DECODING_COMPARISONS",
    "FIGURE_1_DECODING_PARAMETERS",
    "FIGURE_1_DECODING_REGION",
    "TRIAL_ERROR_COLUMNS",
    "build_figure_1_decoding_selection",
    "load_figure_1_decoding_payload",
    "run_figure_1_decoding",
]
