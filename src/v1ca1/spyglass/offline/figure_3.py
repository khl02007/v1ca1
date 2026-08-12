"""Run the NWB-only Figure 3 analysis campaign without DataJoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.paper_figures.datasets import PROCESSED_DATASETS
from v1ca1.ripple.detect_ripples import (
    DEFAULT_ENABLE_NOTCH_FILTER,
    DEFAULT_FILTER_ORDER,
    DEFAULT_HIGHCUT_HZ,
    DEFAULT_LOWCUT_HZ,
    DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
    apply_notch_filters_multichannel,
    butter_filter_and_decimate,
    get_ripple_channels_for_session,
)
from v1ca1.spyglass import (
    ripple_cross_region_xcorr,
    ripple_glm,
    ripple_modulation,
)
from v1ca1.spyglass.nwb import catalog_augmented_nwb, load_interval_set
from v1ca1.spyglass.offline.figure_1 import _offline_region_group_id
from v1ca1.spyglass.offline.figure_2 import (
    FIGURE_2_LIGHT_TRAIN_CONDITION,
    FIGURE_2_LIGHT_TRAIN_EPOCH,
    load_figure_2_campaign,
    load_figure_2_session_manifest,
)
from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    DEFAULT_SCRATCH_ROOT,
    MANIFEST_SCHEMA_VERSION,
    SESSION_MANIFEST_FILENAME,
    append_session_manifest,
    code_provenance,
    file_sha256,
    get_run_dir,
    get_session_dir,
    load_json,
    nwb_fingerprint,
    prepare_campaign,
    relative_run_path,
    resolve_run_path,
    utc_now,
    write_json_once,
)
from v1ca1.spyglass.offline.sources import (
    SOURCE_IDENTITY_POLICY,
    load_nwb_region_spikes,
    validate_nwb_session_identity,
)
from v1ca1.spyglass.selection import (
    canonical_json,
    provenance_sha256,
    selection_uuid,
)
from v1ca1.spyglass.table_specs import (
    DEFAULT_RIPPLE_MODULATION_PARAMETERS,
    MANUSCRIPT_MEAN_ACTIVITY_RIPPLE_GLM_PARAMETERS,
    MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS,
    MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS,
)


DEFAULT_PARENT_RUN_ID = "figure2-nwb-gpu-v1"
FIGURE_3_PIPELINE = "figure_3"
FIGURE_3_LIGHT_EPOCH = FIGURE_2_LIGHT_TRAIN_EPOCH
FIGURE_3_REGIONS = ("ca1", "v1")
FIGURE_3_GLM_PARAMETER_PRESETS = (
    MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS,
    MANUSCRIPT_MEAN_ACTIVITY_RIPPLE_GLM_PARAMETERS,
)
FIGURE_3_XCORR_SESSION = ("L15", "20241121")
FIGURE_3_SCHEMATIC_SESSION = FIGURE_3_XCORR_SESSION
SCHEMATIC_FILENAME = "panel_b_schematic.npz"
SCHEMATIC_SCHEMA_VERSION = 1
SCHEMATIC_TIME_BEFORE_S = 0.080
SCHEMATIC_TIME_AFTER_S = 0.220
SCHEMATIC_FILTER_PADDING_S = 2.0
SCHEMATIC_N_UNITS_PER_REGION = 5
SCHEMATIC_TARGET_RIPPLE_DURATION_S = 0.150

_EXPECTED_SESSIONS = {
    (str(animal_name), str(date))
    for animal_name, date, _light, _dark, _sleep in PROCESSED_DATASETS
}
_BUNDLE_PATH_FIELDS = {
    "ripple_glm": (
        "artifact_dir",
        "artifact_manifest_path",
        "selected_units_path",
        "summary_path",
        "result_path",
    ),
    "ripple_cross_region_xcorr": (
        "artifact_dir",
        "artifact_manifest_path",
        "ca1_units_path",
        "v1_units_path",
        "summary_path",
        "result_path",
    ),
}


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: Any,
) -> dict[str, Any]:
    """Return exactly one record matching all requested fields."""
    matches = [
        dict(record)
        for record in records
        if all(str(record.get(name)) == str(value) for name, value in selectors.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {selectors!r}; found {len(matches)}."
        )
    return matches[0]


def _native(value: Any) -> Any:
    """Convert NumPy scalar containers to JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach one immutable digest to an artifact record."""
    output = _native(dict(record))
    if "record_sha256" in output:
        raise ValueError("Cannot reseal an artifact record.")
    output["record_sha256"] = provenance_sha256(output)
    return output


def _checked_record_path(
    record: Mapping[str, Any],
    field: str,
    *,
    run_dir: Path,
) -> Path:
    """Resolve and checksum one file declared by an upstream record."""
    path = resolve_run_path(str(record[field]), run_dir=run_dir)
    digest = record.get("artifact_sha256", {}).get(field)
    if digest is None or not path.is_file() or file_sha256(path) != str(digest):
        raise ValueError(f"Upstream artifact checksum mismatch for {field!r}.")
    return path


def _relative_artifact_record(
    record: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    run_dir: Path,
    path_fields: Sequence[str],
) -> dict[str, Any]:
    """Add guarded run-relative paths and file hashes to one result record."""
    output = dict(record)
    hashes: dict[str, str] = {}
    for field in path_fields:
        path = Path(paths[field])
        output[field] = relative_run_path(path, run_dir=run_dir)
        if path.is_file():
            hashes[field] = file_sha256(path)
    output["artifact_sha256"] = hashes
    return _seal_record(output)


def build_figure_2_parent_snapshot(
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Freeze the validated four-session Figure 2 campaign by checksum."""
    parent_run_dir, campaign, sessions = load_figure_2_campaign(
        parent_run_id,
        scratch_root=scratch_root,
    )
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list) or len(summaries) != len(sessions):
        raise ValueError("Figure 2 parent has an incomplete session index.")
    observed = {
        (str(row.get("animal_name")), str(row.get("date"))) for row in summaries
    }
    if observed != _EXPECTED_SESSIONS:
        raise ValueError(
            "Figure 3 requires exactly the four manuscript sessions; "
            f"observed {sorted(observed)!r}."
        )
    snapshot_sessions = []
    for summary in summaries:
        if str(summary.get("status")) != "complete":
            raise ValueError("Every Figure 2 parent session must be complete.")
        path = resolve_run_path(
            str(summary["session_manifest_path"]), run_dir=parent_run_dir
        )
        snapshot_sessions.append(
            {
                "animal_name": str(summary["animal_name"]),
                "date": str(summary["date"]),
                "session_manifest_path": str(summary["session_manifest_path"]),
                "session_manifest_sha256": file_sha256(path),
            }
        )
    return {
        "run_id": str(parent_run_id),
        "campaign_manifest_sha256": file_sha256(
            parent_run_dir / CAMPAIGN_MANIFEST_FILENAME
        ),
        "sessions": sorted(
            snapshot_sessions,
            key=lambda row: (row["animal_name"], row["date"]),
        ),
    }


def build_figure_3_configuration(
    parent_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable manuscript Figure 3 campaign configuration."""
    return {
        "pipeline": FIGURE_3_PIPELINE,
        "parent_figure_2": dict(parent_snapshot),
        "epoch": FIGURE_3_LIGHT_EPOCH,
        "condition": FIGURE_2_LIGHT_TRAIN_CONDITION,
        "regions": list(FIGURE_3_REGIONS),
        "ripple_event_policy": {
            "detector_zscore_threshold": 2.0,
            "speed_gated": True,
            "additional_mean_zscore_filter": None,
        },
        "ripple_modulation_parameters": dict(DEFAULT_RIPPLE_MODULATION_PARAMETERS),
        "ripple_glm_parameter_presets": [
            dict(parameters) for parameters in FIGURE_3_GLM_PARAMETER_PRESETS
        ],
        "ripple_glm_runtime_policy": {
            "required_jax_platform": "gpu",
            "minimum_visible_gpu_devices": 1,
            "fail_before_session_artifacts": True,
        },
        "ripple_cross_region_xcorr_parameters": dict(
            MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS
        ),
        "ripple_cross_region_xcorr_session": list(FIGURE_3_XCORR_SESSION),
        "panel_b_schematic": {
            "session": list(FIGURE_3_SCHEMATIC_SESSION),
            "time_before_s": SCHEMATIC_TIME_BEFORE_S,
            "time_after_s": SCHEMATIC_TIME_AFTER_S,
            "filter_padding_s": SCHEMATIC_FILTER_PADDING_S,
            "n_units_per_region": SCHEMATIC_N_UNITS_PER_REGION,
            "target_ripple_duration_s": SCHEMATIC_TARGET_RIPPLE_DURATION_S,
            "ripple_channel_policy": "first_session_configured_ca1_channel",
            "lfp_source": "augmented_nwb_acquisition_electrical_series",
            "lowcut_hz": DEFAULT_LOWCUT_HZ,
            "highcut_hz": DEFAULT_HIGHCUT_HZ,
            "filter_order": DEFAULT_FILTER_ORDER,
            "target_sampling_frequency_hz": (
                DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY
            ),
            "notch_filter_enabled": DEFAULT_ENABLE_NOTCH_FILTER,
            "schema_version": SCHEMATIC_SCHEMA_VERSION,
        },
        "artifact_origin": "computed",
        "diagnostic_figures": False,
    }


def prepare_figure_3_campaign(
    *,
    run_id: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create or validate one append-only offline Figure 3 campaign."""
    parent = build_figure_2_parent_snapshot(
        parent_run_id,
        scratch_root=scratch_root,
    )
    configuration = build_figure_3_configuration(parent)
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    if (run_dir / CAMPAIGN_MANIFEST_FILENAME).exists():
        loaded_run_dir, campaign, _sessions = load_figure_3_campaign(
            run_id,
            scratch_root=scratch_root,
        )
        if canonical_json(campaign.get("analysis_parameters")) != canonical_json(
            configuration
        ):
            raise ValueError(
                "Existing campaign uses different analysis parameters; use a new run_id."
            )
        if canonical_json(campaign.get("source_identity_policy")) != canonical_json(
            SOURCE_IDENTITY_POLICY
        ):
            raise ValueError(
                "Existing campaign uses a different source identity policy; use a new run_id."
            )
        return loaded_run_dir, campaign, parent
    run_dir, campaign = prepare_campaign(
        run_id=run_id,
        analysis_parameters=configuration,
        source_identity_policy=SOURCE_IDENTITY_POLICY,
        scratch_root=scratch_root,
    )
    return run_dir, campaign, parent


def _require_jax_gpu(jax_module: Any | None = None) -> dict[str, Any]:
    """Require a real JAX GPU backend and return JSON-safe device provenance."""
    if jax_module is None:
        try:
            import jax as jax_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Offline Figure 3 RippleGLM fitting requires JAX with GPU support."
            ) from exc
    try:
        devices = list(jax_module.devices("gpu"))
        default_backend = str(jax_module.default_backend())
    except Exception as exc:
        raise RuntimeError(
            "Offline Figure 3 RippleGLM fitting requires an available JAX GPU backend."
        ) from exc
    if not devices or default_backend != "gpu":
        raise RuntimeError(
            "Offline Figure 3 refuses to fit RippleGLM on CPU; configure a JAX GPU "
            "backend before starting the session."
        )
    device_rows = []
    for device in devices:
        process_index = getattr(device, "process_index", None)
        if callable(process_index):
            process_index = process_index()
        device_rows.append(
            {
                "id": int(getattr(device, "id", -1)),
                "platform": str(getattr(device, "platform", "")),
                "device_kind": str(getattr(device, "device_kind", "")),
                "process_index": (
                    None if process_index is None else int(process_index)
                ),
            }
        )
    if any(row["platform"] != "gpu" for row in device_rows):
        raise RuntimeError("JAX returned a non-GPU device for the GPU platform.")
    return {
        "jax_version": str(getattr(jax_module, "__version__", "unknown")),
        "default_backend": default_backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "device_count": len(device_rows),
        "devices": device_rows,
    }


def _clear_jax_caches() -> None:
    """Bound compiled-fit cache growth between the two RippleGLM models."""
    import jax

    jax.clear_caches()
    gc.collect()


def _load_parent_session(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load one hash-pinned Figure 2 session and its selected Figure 3 inputs."""
    current = build_figure_2_parent_snapshot(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(dict(parent_snapshot)):
        raise ValueError("Figure 2 parent changed after Figure 3 selection.")
    run_dir = get_run_dir(str(parent_snapshot["run_id"]), scratch_root=scratch_root)
    summary = _one_record(
        parent_snapshot["sessions"],
        label="Figure 2 parent session",
        animal_name=animal_name,
        date=date,
    )
    manifest_path = resolve_run_path(
        str(summary["session_manifest_path"]), run_dir=run_dir
    )
    if file_sha256(manifest_path) != str(summary["session_manifest_sha256"]):
        raise ValueError("Figure 2 parent session checksum changed.")
    session = load_figure_2_session_manifest(
        manifest_path,
        run_dir=run_dir,
        scratch_root=scratch_root,
    )
    if str(session["epochs"]["AB"]) != FIGURE_3_LIGHT_EPOCH:
        raise ValueError("Figure 2 parent light epoch is not the Figure 3 epoch.")
    dark_epoch = str(session["epochs"]["dark"])
    dark_similarity = _one_record(
        session["artifacts"]["path_specific_place_tuning_similarity"],
        label="dark tuning similarity",
        epoch=dark_epoch,
        region="v1",
    )
    parent_artifacts = session["parent_artifacts"]
    dark_movement = dict(parent_artifacts["dark_movement_firing_rate"])
    initial_run_dir = get_run_dir(
        str(parent_artifacts["initial_run_id"]), scratch_root=scratch_root
    )
    for field in ("firing_rate_path", "movement_intervals_path"):
        _checked_record_path(dark_movement, field, run_dir=initial_run_dir)
    nested = dark_similarity.get("artifacts")
    if not isinstance(nested, Mapping) or not nested:
        raise ValueError("Dark tuning similarity lacks its artifact bundle.")
    for metadata in nested.values():
        path = resolve_run_path(str(metadata["relative_path"]), run_dir=run_dir)
        if not path.is_file() or file_sha256(path) != str(metadata["sha256"]):
            raise ValueError("Dark tuning similarity artifact checksum mismatch.")
    selected = {
        "figure_2_run_id": str(parent_snapshot["run_id"]),
        "figure_2_session_manifest_path": str(summary["session_manifest_path"]),
        "figure_2_session_manifest_sha256": str(summary["session_manifest_sha256"]),
        "dark_movement_firing_rate_run_id": str(parent_artifacts["initial_run_id"]),
        "dark_movement_firing_rate": dark_movement,
        "dark_tuning_similarity_run_id": str(parent_snapshot["run_id"]),
        "dark_tuning_similarity": dark_similarity,
    }
    return run_dir, session, selected


def _interval_frame(intervals: Any, *, epoch: str) -> pd.DataFrame:
    """Return one loaded NWB IntervalSet as canonical start/end rows."""
    table = intervals.as_dataframe().copy()
    table = table.rename(
        columns={"start": "start_time", "end": "end_time", "stop": "end_time"}
    )
    missing = sorted({"start_time", "end_time"}.difference(table.columns))
    if missing:
        raise ValueError(f"Loaded NWB intervals are missing columns {missing!r}.")
    table["start_time"] = np.asarray(table["start_time"], dtype=float)
    table["end_time"] = np.asarray(table["end_time"], dtype=float)
    if "epoch" not in table:
        table["epoch"] = str(epoch)
    if not table["epoch"].astype(str).eq(str(epoch)).all():
        raise ValueError("Loaded ripple intervals contain an unexpected epoch.")
    return table.reset_index(drop=True)


def _ripple_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the detector settings and exact NWB object pointers."""
    fields = (
        "nwb_file_name",
        "epoch",
        "ripple_count",
        "detector_zscore_threshold",
        "speed_gated",
        "detection_parameters",
        "provenance_path",
        "provenance_object_id",
        "source_table_path",
        "source_table_object_id",
        "source_object_path",
        "source_object_id",
    )
    provenance = _native({name: row.get(name) for name in fields})
    threshold = float(provenance["detector_zscore_threshold"])
    if not np.isclose(threshold, 2.0, rtol=0.0, atol=1e-12):
        raise ValueError("Figure 3 requires detector threshold 2.0 ripples.")
    if provenance["speed_gated"] is not True:
        raise ValueError("Figure 3 requires speed-gated ripples.")
    return provenance


def _source_identity(
    *,
    nwb_file_name: str,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one compact explicit regional spike-source snapshot."""
    return {
        "offline_region_sorted_spikes_view_id": _offline_region_group_id(
            nwb_file_name=nwb_file_name,
            region=str(loaded["region"]),
            loaded_spikes=loaded,
        ),
        "source": str(loaded["source"]),
        "spikesorting_merge_id": str(loaded["spikesorting_merge_id"]),
        "region": str(loaded["region"]),
        "n_units": int(loaded["n_units"]),
        "selected_units_sha256": str(loaded["selected_units_sha256"]),
    }


def _modulation_parameter_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the table preset to the database-free computation API."""
    return {
        "bin_size_s": float(parameters["bin_size_s"]),
        "time_before_s": float(parameters["time_before_s"]),
        "time_after_s": float(parameters["time_after_s"]),
        "response_window": (
            float(parameters["response_window_start_s"]),
            float(parameters["response_window_end_s"]),
        ),
        "baseline_window": (
            float(parameters["baseline_window_start_s"]),
            float(parameters["baseline_window_end_s"]),
        ),
    }


def _glm_parameter_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one RippleGLM table preset to its computation keywords."""
    names = (
        "ripple_window_s",
        "ripple_window_offset_s",
        "source_window_s",
        "source_window_offset_s",
        "target_window_s",
        "target_window_offset_s",
        "ripple_selection_mode",
        "source_predictor_mode",
        "min_spikes_per_ripple",
        "min_ca1_spikes_per_ripple",
        "n_splits",
        "n_shuffles_ripple",
        "ridge_strength",
        "shuffle_seed",
        "maxiter",
        "tol",
        "expected_detector_zscore_threshold",
        "require_speed_gated",
    )
    return {name: parameters[name] for name in names}


def _xcorr_parameter_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the RippleCrossRegionXCorr preset to computation keywords."""
    names = (
        "bin_size_s",
        "max_lag_s",
        "min_ripple_spikes",
        "extremum_half_width_bins",
        "norm",
        "expected_detector_zscore_threshold",
        "require_speed_gated",
    )
    return {name: parameters[name] for name in names}


def _run_modulation(
    *,
    run_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    ripple_table: pd.DataFrame,
    epoch_bounds: tuple[float, float],
    ripple_row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute, write, and describe one regional RippleModulation result."""
    parameters = dict(DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    parameter_hash = provenance_sha256(parameters)
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "ripple_modulation_param_name": parameters[
            "ripple_modulation_param_name"
        ],
        "ripple_modulation_parameters_sha256": parameter_hash,
        "region_sorted_spikes_group_id": source_identity[
            "offline_region_sorted_spikes_view_id"
        ],
        "ripple_provenance_sha256": provenance_sha256(
            _ripple_provenance(ripple_row)
        ),
    }
    result_id = str(selection_uuid("RippleModulation", natural_key))
    result = ripple_modulation.compute_epoch_region_ripple_modulation(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=str(loaded["region"]),
        ripple_table=ripple_table,
        epoch_timestamps=np.asarray(epoch_bounds, dtype=float),
        region_spikes=loaded["ts_group"],
        stable_unit_ids=loaded["unit_ids"],
        **_modulation_parameter_kwargs(parameters),
    )
    if int(result["n_ripples"]) != int(ripple_row["ripple_count"]):
        raise ValueError("RippleModulation did not preserve the NWB ripple count.")
    if not result["summary"].empty:
        summary = result["summary"].copy()
        missing_reason = summary["invalid_reason"].isna()
        nonfinite = ~np.isfinite(summary["response_zscore"].to_numpy(dtype=float))
        summary.loc[missing_reason & nonfinite, "invalid_reason"] = (
            "nonfinite_response_zscore"
        )
        result = {**result, "summary": summary}
    paths = ripple_modulation.get_ripple_modulation_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=str(loaded["region"]),
        ripple_modulation_id=result_id,
        artifact_root=run_dir,
    )
    written = ripple_modulation.write_ripple_modulation_artifacts(result, paths)
    reasons = result["summary"]["invalid_reason"]
    n_valid = int(reasons.isna().sum()) if len(reasons) else 0
    if int(loaded["n_units"]) == 0:
        status = "no_units"
    elif int(result["n_ripples"]) == 0:
        status = "no_ripples"
    else:
        status = "valid" if n_valid else "no_valid_units"
    record = _relative_artifact_record(
        {
            **natural_key,
            "ripple_modulation_id": result_id,
            "region": str(loaded["region"]),
            "summary_path": written["summary"],
            "peri_ripple_firing_rate_path": written[
                "peri_ripple_firing_rate"
            ],
            "n_ripples": int(result["n_ripples"]),
            "n_units": int(loaded["n_units"]),
            "n_valid_units": n_valid,
            "analysis_status": status,
            "selected_units_sha256": str(loaded["selected_units_sha256"]),
            "artifact_origin": "computed",
        },
        {
            "summary_path": written["summary"],
            "peri_ripple_firing_rate_path": written[
                "peri_ripple_firing_rate"
            ],
        },
        run_dir=run_dir,
        path_fields=("summary_path", "peri_ripple_firing_rate_path"),
    )
    return record, result


def _run_glm(
    *,
    run_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    ripple_table: pd.DataFrame,
    epoch_interval: Any,
    ripple_provenance: Mapping[str, Any],
    ca1: Mapping[str, Any],
    v1: Mapping[str, Any],
    ca1_identity: Mapping[str, Any],
    v1_identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute and persist one fixed-direction RippleGLM bundle."""
    parameters = dict(parameters)
    kwargs = _glm_parameter_kwargs(parameters)
    event_selection = ripple_glm.prepare_ripple_glm_event_selection(
        epoch=epoch,
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        **kwargs,
    )
    parameter_hash = provenance_sha256(parameters)
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "source_region_sorted_spikes_group_id": ca1_identity[
            "offline_region_sorted_spikes_view_id"
        ],
        "target_region_sorted_spikes_group_id": v1_identity[
            "offline_region_sorted_spikes_view_id"
        ],
        "ripple_glm_param_name": parameters["ripple_glm_param_name"],
        "ripple_glm_parameters_sha256": parameter_hash,
        "source_predictor_mode": parameters["source_predictor_mode"],
        "source_ripple_count": int(len(ripple_table)),
        "ripple_provenance_sha256": provenance_sha256(ripple_provenance),
        "n_selected_ripples": int(
            event_selection["n_ripples_after_window_bounds"]
        ),
        "selected_ripple_events_sha256": str(
            event_selection["selected_ripple_events_sha256"]
        ),
        "source_selected_units_sha256": str(ca1["selected_units_sha256"]),
        "target_selected_units_sha256": str(v1["selected_units_sha256"]),
        "ripple_glm_output_rule_sha256": ripple_glm.OUTPUT_RULE_SHA256,
    }
    result_id = str(selection_uuid("RippleGLM", natural_key))
    result = ripple_glm.compute_ripple_glm(
        ripple_glm_id=result_id,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        source_spikes=ca1["ts_group"],
        source_stable_unit_ids=ca1["unit_ids"],
        target_spikes=v1["ts_group"],
        target_stable_unit_ids=v1["unit_ids"],
        upstream_provenance=ripple_provenance,
        parameter_name=str(parameters["ripple_glm_param_name"]),
        parameter_sha256=parameter_hash,
        output_rule_sha256=ripple_glm.OUTPUT_RULE_SHA256,
        expected_selected_ripple_events_sha256=event_selection[
            "selected_ripple_events_sha256"
        ],
        **kwargs,
    )
    paths = ripple_glm.get_ripple_glm_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_glm_id=result_id,
        artifact_root=run_dir,
    )
    written = ripple_glm.write_ripple_glm_artifact(result, paths["artifact_dir"])
    return _relative_artifact_record(
        {
            **natural_key,
            "ripple_glm_id": result_id,
            "parameter_name": str(parameters["ripple_glm_param_name"]),
            "analysis_status": str(result["analysis_status"]),
            "n_ripples": int(result["n_ripples"]),
            "n_source_units": int(result["n_source_units"]),
            "n_target_units": int(result["n_target_units"]),
            "n_source_units_in_fit": int(result["n_source_units_in_fit"]),
            "n_target_units_in_fit": int(result["n_target_units_in_fit"]),
            "n_valid_target_units": int(result["n_valid_target_units"]),
            "artifact_origin": "computed",
        },
        written,
        run_dir=run_dir,
        path_fields=_BUNDLE_PATH_FIELDS["ripple_glm"],
    )


def _run_xcorr(
    *,
    run_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    ripple_table: pd.DataFrame,
    ripple_provenance: Mapping[str, Any],
    ca1: Mapping[str, Any],
    v1: Mapping[str, Any],
    ca1_identity: Mapping[str, Any],
    v1_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute and persist the L15 exact-ripple cross-region xcorr."""
    parameters = dict(MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS)
    event_selection = (
        ripple_cross_region_xcorr.prepare_ripple_cross_region_xcorr_event_selection(
            epoch=epoch,
            ripple_table=ripple_table,
        )
    )
    parameter_hash = provenance_sha256(parameters)
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "source_region_sorted_spikes_group_id": ca1_identity[
            "offline_region_sorted_spikes_view_id"
        ],
        "target_region_sorted_spikes_group_id": v1_identity[
            "offline_region_sorted_spikes_view_id"
        ],
        "ripple_cross_region_xcorr_param_name": parameters[
            "ripple_cross_region_xcorr_param_name"
        ],
        "ripple_cross_region_xcorr_parameters_sha256": parameter_hash,
        "source_ripple_count": int(len(ripple_table)),
        "ripple_provenance_sha256": provenance_sha256(ripple_provenance),
        "selected_ripple_intervals_sha256": event_selection[
            "selected_ripple_intervals_sha256"
        ],
        "source_selected_units_sha256": str(ca1["selected_units_sha256"]),
        "target_selected_units_sha256": str(v1["selected_units_sha256"]),
        "ripple_cross_region_xcorr_output_rule_sha256": (
            ripple_cross_region_xcorr.OUTPUT_RULE_SHA256
        ),
    }
    result_id = str(selection_uuid("RippleCrossRegionXCorr", natural_key))
    result = ripple_cross_region_xcorr.compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=result_id,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_table=ripple_table,
        ca1_spikes=ca1["ts_group"],
        ca1_stable_unit_ids=ca1["unit_ids"],
        v1_spikes=v1["ts_group"],
        v1_stable_unit_ids=v1["unit_ids"],
        upstream_provenance=ripple_provenance,
        parameter_name=str(parameters["ripple_cross_region_xcorr_param_name"]),
        parameter_sha256=parameter_hash,
        output_rule_sha256=ripple_cross_region_xcorr.OUTPUT_RULE_SHA256,
        expected_selected_ripple_intervals_sha256=event_selection[
            "selected_ripple_intervals_sha256"
        ],
        **_xcorr_parameter_kwargs(parameters),
    )
    paths = ripple_cross_region_xcorr.get_ripple_cross_region_xcorr_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_cross_region_xcorr_id=result_id,
        artifact_root=run_dir,
    )
    written = ripple_cross_region_xcorr.write_ripple_cross_region_xcorr_artifact(
        result, paths["artifact_dir"]
    )
    return _relative_artifact_record(
        {
            **natural_key,
            "ripple_cross_region_xcorr_id": result_id,
            "parameter_name": str(
                parameters["ripple_cross_region_xcorr_param_name"]
            ),
            "analysis_status": str(result["analysis_status"]),
            **{
                name: _native(result[name])
                for name in (
                    "n_ripples",
                    "ripple_duration_s",
                    "n_ca1_units",
                    "n_v1_units",
                    "n_ca1_units_in_xcorr",
                    "n_v1_units_in_xcorr",
                    "n_pairs",
                    "n_valid_pairs",
                )
            },
            "artifact_origin": "computed",
        },
        written,
        run_dir=run_dir,
        path_fields=_BUNDLE_PATH_FIELDS["ripple_cross_region_xcorr"],
    )


def _spike_times_by_stable_id(loaded: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Map explicit NWB unit identities to their seconds-valued spike arrays."""
    keys = list(loaded["ts_group"].keys())
    identities = list(loaded["unit_ids"])
    if len(keys) != len(identities):
        raise ValueError("TsGroup keys and stable NWB identities are misaligned.")
    return {
        str(identity["unit_id"]): np.sort(
            np.asarray(loaded["ts_group"][key].t, dtype=float)
        )
        for key, identity in zip(keys, identities, strict=True)
    }


def _unit_metadata_by_stable_id(
    loaded: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return NWB and sorting identities keyed by stable NWB unit id."""
    rows = {
        str(row["unit_id"]): _native(dict(row))
        for row in loaded["unit_metadata"]
    }
    if len(rows) != len(loaded["unit_metadata"]):
        raise ValueError("NWB unit metadata contains duplicate stable identities.")
    return rows


def _window_count(times: np.ndarray, start: float, stop: float) -> int:
    """Count one unit's spikes in a half-open time window."""
    return int(
        np.searchsorted(times, stop, side="left")
        - np.searchsorted(times, start, side="left")
    )


def _select_schematic_event(
    *,
    ripple_table: pd.DataFrame,
    ca1_modulation_summary: pd.DataFrame,
    ca1_spikes: Mapping[str, np.ndarray],
    v1_spikes: Mapping[str, np.ndarray],
    ranked_ca1_unit_ids: Sequence[str] | None = None,
) -> tuple[pd.Series, list[str], list[str], np.ndarray]:
    """Select the fixed real-data schematic event and displayed units."""
    if ranked_ca1_unit_ids is None:
        summary = ca1_modulation_summary.copy()
        summary["unit_id"] = summary["unit_id"].map(str)
        summary = summary.loc[summary["unit_id"].isin(ca1_spikes)].copy()
        if len(summary) < SCHEMATIC_N_UNITS_PER_REGION:
            raise ValueError("Too few CA1 modulation units for the schematic.")
        modulation = pd.to_numeric(
            summary["ripple_modulation_index"], errors="coerce"
        ).to_numpy(dtype=float)
        summary["_finite"] = np.isfinite(modulation)
        summary["_absolute_modulation"] = np.where(
            np.isfinite(modulation), np.abs(modulation), -np.inf
        )
        summary = summary.sort_values(
            ["_finite", "_absolute_modulation", "unit_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        ranked_ca1 = summary["unit_id"].tolist()
    else:
        ranked_ca1 = [str(unit_id) for unit_id in ranked_ca1_unit_ids]
        if len(ranked_ca1) != len(set(ranked_ca1)):
            raise ValueError("Ranked CA1 schematic unit IDs must be unique.")
        if not set(ranked_ca1).issubset(ca1_spikes):
            raise ValueError("Ranked CA1 schematic units are absent from spike data.")
        if len(ranked_ca1) < SCHEMATIC_N_UNITS_PER_REGION:
            raise ValueError("Too few ranked CA1 units for the schematic.")
    best: tuple[tuple[float, ...], pd.Series, list[str], list[str]] | None = None
    for row_index, row in ripple_table.reset_index(drop=True).iterrows():
        start = float(row["start_time"])
        stop = float(row["end_time"])
        window_start = start - SCHEMATIC_TIME_BEFORE_S
        window_stop = start + SCHEMATIC_TIME_AFTER_S
        ca1_counts = {
            unit: _window_count(ca1_spikes[unit], window_start, window_stop)
            for unit in ranked_ca1
        }
        active_ranked = [unit for unit in ranked_ca1 if ca1_counts[unit] > 0]
        selected_ca1 = active_ranked[:SCHEMATIC_N_UNITS_PER_REGION]
        selected_ca1.extend(
            unit
            for unit in ranked_ca1
            if unit not in selected_ca1
        )
        selected_ca1 = selected_ca1[:SCHEMATIC_N_UNITS_PER_REGION]
        v1_counts = {
            unit: _window_count(times, window_start, window_stop)
            for unit, times in v1_spikes.items()
        }
        selected_v1 = sorted(
            v1_counts,
            key=lambda unit: (-v1_counts[unit], str(unit)),
        )[:SCHEMATIC_N_UNITS_PER_REGION]
        if len(selected_v1) < SCHEMATIC_N_UNITS_PER_REGION:
            raise ValueError("Too few V1 units for the schematic.")
        active_ca1 = sum(ca1_counts[unit] > 0 for unit in selected_ca1)
        active_v1 = sum(v1_counts[unit] > 0 for unit in selected_v1)
        mean_zscore = float(row.get("mean_zscore", np.nan))
        score = (
            float(
                active_ca1 >= SCHEMATIC_N_UNITS_PER_REGION
                and active_v1 >= SCHEMATIC_N_UNITS_PER_REGION
            ),
            -abs((stop - start) - SCHEMATIC_TARGET_RIPPLE_DURATION_S),
            float(len(active_ranked)),
            float(active_ca1),
            float(sum(ca1_counts[unit] for unit in selected_ca1)),
            float(active_v1),
            float(sum(v1_counts[unit] for unit in selected_v1)),
            mean_zscore if np.isfinite(mean_zscore) else -np.inf,
            -float(row_index),
        )
        if best is None or score > best[0]:
            best = (score, row, selected_ca1, selected_v1)
    if best is None:
        raise ValueError("No NWB ripple is available for the schematic.")
    return best[1], best[2], best[3], np.asarray(best[0], dtype=float)


def _dataset_searchsorted(dataset: Any, value: float, *, side: str) -> int:
    """Binary-search a monotonic on-disk NWB timestamp dataset."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    low, high = 0, int(len(dataset))
    while low < high:
        middle = (low + high) // 2
        current = float(dataset[middle])
        if current < value or (side == "right" and current == value):
            low = middle + 1
        else:
            high = middle
    return low


def _electrical_series(nwbfile: Any) -> tuple[str, Any]:
    """Resolve the unique timestamped raw ElectricalSeries acquisition."""
    acquisitions = getattr(nwbfile, "acquisition", {})
    preferred = acquisitions.get("e-series")
    candidates = [
        (str(name), value)
        for name, value in acquisitions.items()
        if getattr(value, "timestamps", None) is not None
        and getattr(value, "electrodes", None) is not None
        and getattr(value, "data", None) is not None
    ]
    if preferred is not None and any(value is preferred for _name, value in candidates):
        return "e-series", preferred
    if len(candidates) != 1:
        raise ValueError(
            "Schematic generation requires exactly one timestamped raw "
            "ElectricalSeries acquisition."
        )
    return candidates[0]


def _load_filtered_lfp_snippet(
    nwbfile: Any,
    *,
    channel_id: int,
    ripple_start_s: float,
) -> dict[str, Any]:
    """Filter one padded raw NWB voltage segment into a ripple-band snippet."""
    series_name, series = _electrical_series(nwbfile)
    electrode_indices = np.asarray(series.electrodes.data[:], dtype=int)
    electrode_ids = np.asarray(series.electrodes.table.id[:])[electrode_indices]
    matches = np.flatnonzero(electrode_ids.astype(str) == str(channel_id))
    if matches.size != 1:
        raise ValueError(
            f"Configured ripple channel {channel_id!r} is not unique in the NWB ElectricalSeries."
        )
    channel_index = int(matches[0])
    timestamps = series.timestamps
    requested_start = (
        ripple_start_s - SCHEMATIC_TIME_BEFORE_S - SCHEMATIC_FILTER_PADDING_S
    )
    requested_stop = (
        ripple_start_s + SCHEMATIC_TIME_AFTER_S + SCHEMATIC_FILTER_PADDING_S
    )
    start_index = _dataset_searchsorted(timestamps, requested_start, side="left")
    stop_index = _dataset_searchsorted(timestamps, requested_stop, side="right")
    if stop_index - start_index < 100:
        raise ValueError("NWB has too few raw samples around the schematic ripple.")
    time_s = np.asarray(timestamps[start_index:stop_index], dtype=float)
    raw = np.asarray(series.data[start_index:stop_index, channel_index], dtype=float)
    if raw.ndim != 1 or time_s.shape != raw.shape:
        raise ValueError("NWB voltage data and timestamps are misaligned.")
    duration = float(time_s[-1] - time_s[0])
    sampling_frequency = (len(time_s) - 1) / duration
    unit = str(getattr(series, "unit", "")).strip().casefold()
    conversion = float(getattr(series, "conversion", 1.0))
    offset = float(getattr(series, "offset", 0.0))
    if unit in {"volts", "volt", "v"}:
        raw_uv = (raw * conversion + offset) * 1e6
    elif unit in {"microvolts", "microvolt", "uv", "µv"}:
        raw_uv = raw * conversion + offset
    else:
        raise ValueError(f"Unsupported NWB ElectricalSeries voltage unit {unit!r}.")
    if DEFAULT_ENABLE_NOTCH_FILTER:
        raw_uv = apply_notch_filters_multichannel(
            raw_uv[:, None], sampling_frequency
        )[:, 0]
    decimated_time, filtered, actual_fs = butter_filter_and_decimate(
        time_s,
        raw_uv[:, None],
        sampling_frequency,
        lowcut=DEFAULT_LOWCUT_HZ,
        highcut=DEFAULT_HIGHCUT_HZ,
        order=DEFAULT_FILTER_ORDER,
        target_new_sampling_frequency=DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
    )
    keep = (
        (decimated_time >= ripple_start_s - SCHEMATIC_TIME_BEFORE_S)
        & (decimated_time <= ripple_start_s + SCHEMATIC_TIME_AFTER_S)
    )
    if not np.any(keep):
        raise ValueError("Filtered NWB voltage has no schematic-window samples.")
    return {
        "time_s": np.asarray(decimated_time[keep] - ripple_start_s, dtype=float),
        "filtered_lfp": np.asarray(filtered[keep, 0], dtype=float),
        "sampling_frequency_hz": float(actual_fs),
        "channel": int(channel_id),
        "electrical_series_name": series_name,
        "electrical_series_object_id": str(getattr(series, "object_id", "")),
    }


def _relative_spikes(
    spikes: Mapping[str, np.ndarray],
    unit_ids: Sequence[str],
    *,
    ripple_start_s: float,
) -> list[np.ndarray]:
    """Return selected-unit spike rasters relative to ripple onset."""
    start = ripple_start_s - SCHEMATIC_TIME_BEFORE_S
    stop = ripple_start_s + SCHEMATIC_TIME_AFTER_S
    output = []
    for unit_id in unit_ids:
        times = spikes[str(unit_id)]
        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, stop, side="left")
        output.append(np.asarray(times[left:right] - ripple_start_s, dtype=float))
    return output


def _flatten_rasters(rasters: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ragged spike rasters without object arrays."""
    counts = np.asarray([len(values) for values in rasters], dtype=np.int64)
    flat = (
        np.concatenate([np.asarray(values, dtype=float) for values in rasters])
        if counts.sum()
        else np.asarray([], dtype=float)
    )
    return flat, counts


def _unflatten_rasters(flat: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    """Restore compact ragged spike rasters after strict size validation."""
    flat = np.asarray(flat, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0) or int(counts.sum()) != len(flat):
        raise ValueError("Schematic raster arrays are malformed.")
    edges = np.concatenate(([0], np.cumsum(counts)))
    return [flat[edges[index] : edges[index + 1]] for index in range(len(counts))]


def _write_schematic_payload(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomically create one pickle-free, NWB-derived schematic NPZ."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite schematic payload: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ca1_flat, ca1_counts = _flatten_rasters(payload["ca1_spike_times_s"])
    v1_flat, v1_counts = _flatten_rasters(payload["v1_spike_times_s"])
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=canonical_json(payload["metadata"]),
                time_s=np.asarray(payload["time_s"], dtype=float),
                filtered_lfp=np.asarray(payload["filtered_lfp"], dtype=float),
                ripple_start_s=float(payload["ripple_start_s"]),
                ripple_end_s=float(payload["ripple_end_s"]),
                mean_zscore=float(payload["mean_zscore"]),
                n_ripples=int(payload["n_ripples"]),
                channel=int(payload["channel"]),
                sampling_frequency_hz=float(payload["sampling_frequency_hz"]),
                selection_score=np.asarray(payload["selection_score"], dtype=float),
                ca1_unit_ids=np.asarray(payload["ca1_unit_ids"], dtype=str),
                v1_unit_ids=np.asarray(payload["v1_unit_ids"], dtype=str),
                ca1_unit_identity_json=canonical_json(payload["ca1_unit_identity"]),
                v1_unit_identity_json=canonical_json(payload["v1_unit_identity"]),
                ca1_spike_times_flat=ca1_flat,
                ca1_spike_counts=ca1_counts,
                v1_spike_times_flat=v1_flat,
                v1_spike_counts=v1_counts,
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    load_panel_b_schematic_payload(destination)
    return destination


def load_panel_b_schematic_payload(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate one pickle-free Figure 3 schematic payload."""
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != str(expected_sha256):
        raise ValueError("Panel B schematic checksum mismatch.")
    with np.load(source, allow_pickle=False) as values:
        required = {
            "metadata_json",
            "time_s",
            "filtered_lfp",
            "ripple_start_s",
            "ripple_end_s",
            "mean_zscore",
            "n_ripples",
            "channel",
            "sampling_frequency_hz",
            "selection_score",
            "ca1_unit_ids",
            "v1_unit_ids",
            "ca1_unit_identity_json",
            "v1_unit_identity_json",
            "ca1_spike_times_flat",
            "ca1_spike_counts",
            "v1_spike_times_flat",
            "v1_spike_counts",
        }
        if set(values.files) != required:
            raise ValueError("Panel B schematic payload has an unexpected schema.")
        metadata = json.loads(str(values["metadata_json"]))
        ca1_identity = json.loads(str(values["ca1_unit_identity_json"]))
        v1_identity = json.loads(str(values["v1_unit_identity_json"]))
        payload = {
            "metadata": metadata,
            "animal_name": str(metadata.get("animal_name", "")),
            "date": str(metadata.get("date", "")),
            "epoch": str(metadata.get("epoch", "")),
            "time_s": np.asarray(values["time_s"], dtype=float),
            "filtered_lfp": np.asarray(values["filtered_lfp"], dtype=float),
            "ripple_start_s": float(values["ripple_start_s"]),
            "ripple_end_s": float(values["ripple_end_s"]),
            "mean_zscore": float(values["mean_zscore"]),
            "n_ripples": int(values["n_ripples"]),
            "channel": int(values["channel"]),
            "sampling_frequency_hz": float(values["sampling_frequency_hz"]),
            "ripple_duration_s": (
                float(values["ripple_end_s"]) - float(values["ripple_start_s"])
            ),
            "time_before_s": float(metadata.get("time_before_s", np.nan)),
            "time_after_s": float(metadata.get("time_after_s", np.nan)),
            "n_units_per_region": int(metadata.get("n_units_per_region", -1)),
            "selection_score": np.asarray(values["selection_score"], dtype=float),
            "ca1_unit_ids": np.asarray(values["ca1_unit_ids"], dtype=str),
            "v1_unit_ids": np.asarray(values["v1_unit_ids"], dtype=str),
            "ca1_unit_identity": ca1_identity,
            "v1_unit_identity": v1_identity,
            "ca1_spike_times_s": _unflatten_rasters(
                values["ca1_spike_times_flat"], values["ca1_spike_counts"]
            ),
            "v1_spike_times_s": _unflatten_rasters(
                values["v1_spike_times_flat"], values["v1_spike_counts"]
            ),
        }
    if metadata.get("schema_version") != SCHEMATIC_SCHEMA_VERSION:
        raise ValueError("Unsupported panel B schematic schema version.")
    if payload["time_s"].ndim != 1 or payload["filtered_lfp"].shape != payload[
        "time_s"
    ].shape:
        raise ValueError("Schematic LFP trace and time axis are misaligned.")
    if not np.all(np.isfinite(payload["time_s"])) or not np.all(
        np.isfinite(payload["filtered_lfp"])
    ):
        raise ValueError("Schematic LFP contains non-finite values.")
    for region in FIGURE_3_REGIONS:
        unit_ids = payload[f"{region}_unit_ids"]
        identities = payload[f"{region}_unit_identity"]
        rasters = payload[f"{region}_spike_times_s"]
        if len(unit_ids) != len(identities) or len(unit_ids) != len(rasters):
            raise ValueError(f"Schematic {region} unit identities are misaligned.")
        if [str(row["unit_id"]) for row in identities] != unit_ids.tolist():
            raise ValueError(f"Schematic {region} stable unit mapping changed.")
    return payload


def _build_schematic_payload(
    nwbfile: Any,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    ripple_table: pd.DataFrame,
    ca1: Mapping[str, Any],
    v1: Mapping[str, Any],
    ca1_modulation: Mapping[str, Any],
    selector_kwargs: Mapping[str, Any] | None = None,
    selector_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the L15 panel-B LFP/spike example directly from the NWB."""
    if (selector_kwargs is None) != (selector_policy is None):
        raise ValueError(
            "selector_kwargs and selector_policy must be supplied together."
        )
    ca1_spikes = _spike_times_by_stable_id(ca1)
    v1_spikes = _spike_times_by_stable_id(v1)
    selected, ca1_units, v1_units, score = _select_schematic_event(
        ripple_table=ripple_table,
        ca1_modulation_summary=ca1_modulation["summary"],
        ca1_spikes=ca1_spikes,
        v1_spikes=v1_spikes,
        **({} if selector_kwargs is None else dict(selector_kwargs)),
    )
    ripple_start = float(selected["start_time"])
    ripple_end = float(selected["end_time"])
    channel = get_ripple_channels_for_session(animal_name, date)[0]
    lfp = _load_filtered_lfp_snippet(
        nwbfile,
        channel_id=channel,
        ripple_start_s=ripple_start,
    )
    ca1_metadata = _unit_metadata_by_stable_id(ca1)
    v1_metadata = _unit_metadata_by_stable_id(v1)
    metadata = {
        "schema_version": SCHEMATIC_SCHEMA_VERSION,
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "nwb_file_name": nwb_file_name,
        "artifact_origin": "computed_from_augmented_nwb",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "electrical_series_name": lfp["electrical_series_name"],
        "electrical_series_object_id": lfp["electrical_series_object_id"],
        "time_before_s": SCHEMATIC_TIME_BEFORE_S,
        "time_after_s": SCHEMATIC_TIME_AFTER_S,
        "filter_padding_s": SCHEMATIC_FILTER_PADDING_S,
        "n_units_per_region": SCHEMATIC_N_UNITS_PER_REGION,
        "target_ripple_duration_s": SCHEMATIC_TARGET_RIPPLE_DURATION_S,
        "lowcut_hz": DEFAULT_LOWCUT_HZ,
        "highcut_hz": DEFAULT_HIGHCUT_HZ,
        "filter_order": DEFAULT_FILTER_ORDER,
        "target_sampling_frequency_hz": DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
        "notch_filter_enabled": DEFAULT_ENABLE_NOTCH_FILTER,
    }
    if selector_policy is not None:
        metadata["selector_policy"] = dict(selector_policy)
        metadata["selector_policy_sha256"] = provenance_sha256(selector_policy)
    return {
        "metadata": metadata,
        "time_s": lfp["time_s"],
        "filtered_lfp": lfp["filtered_lfp"],
        "ripple_start_s": ripple_start,
        "ripple_end_s": ripple_end,
        "mean_zscore": float(selected.get("mean_zscore", np.nan)),
        "n_ripples": int(len(ripple_table)),
        "channel": int(channel),
        "sampling_frequency_hz": lfp["sampling_frequency_hz"],
        "selection_score": score,
        "ca1_unit_ids": ca1_units,
        "v1_unit_ids": v1_units,
        "ca1_unit_identity": [ca1_metadata[unit] for unit in ca1_units],
        "v1_unit_identity": [v1_metadata[unit] for unit in v1_units],
        "ca1_spike_times_s": _relative_spikes(
            ca1_spikes, ca1_units, ripple_start_s=ripple_start
        ),
        "v1_spike_times_s": _relative_spikes(
            v1_spikes, v1_units, ripple_start_s=ripple_start
        ),
    }


def _run_figure_3_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Compute and persist one session's NWB-only Figure 3 inputs."""
    animal_name, date = str(animal_name), str(date)
    run_dir, campaign, parent_snapshot = prepare_figure_3_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(run_dir, animal_name=animal_name, date=date)
    if session_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a Figure 3 session: {session_dir}")
    _parent_run_dir, parent_session, parent_artifacts = _load_parent_session(
        parent_snapshot,
        animal_name=animal_name,
        date=date,
        scratch_root=scratch_root,
    )
    gpu_provenance = _require_jax_gpu()
    nwb_path = Path(str(parent_session["nwb_path"])).resolve(strict=True)
    if nwb_path.is_relative_to(run_dir.resolve(strict=False)):
        raise ValueError("Source NWB must remain outside the output run.")

    import pynwb

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=animal_name,
            date=date,
        )
        fingerprint = nwb_fingerprint(nwb_path, nwbfile)
        if canonical_json(fingerprint) != canonical_json(
            parent_session["nwb_fingerprint"]
        ):
            raise ValueError("Source NWB changed relative to the Figure 2 parent.")
        catalog = catalog_augmented_nwb(nwbfile, nwb_file_name=nwb_path.name)
        epoch_row = _one_record(
            catalog["epoch_intervals"],
            label="Figure 3 light epoch",
            epoch=FIGURE_3_LIGHT_EPOCH,
        )
        if (
            str(epoch_row.get("epoch_type")) != "run"
            or str(epoch_row.get("condition")).casefold()
            != FIGURE_2_LIGHT_TRAIN_CONDITION.casefold()
            or epoch_row.get("is_light") is not True
        ):
            raise ValueError("Figure 3 requires the explicitly cataloged AB light run.")
        ripple_row = _one_record(
            catalog["ripples"],
            label="Figure 3 ripple row",
            epoch=FIGURE_3_LIGHT_EPOCH,
        )
        detector_provenance = _ripple_provenance(ripple_row)
        ripple_intervals = load_interval_set(nwbfile, ripple_row)
        epoch_interval = load_interval_set(nwbfile, epoch_row)
        ripple_table = _interval_frame(
            ripple_intervals, epoch=FIGURE_3_LIGHT_EPOCH
        )
        if len(ripple_table) != int(ripple_row["ripple_count"]):
            raise ValueError("NWB ripple catalog count changed while loading events.")
        epoch_bounds = (float(epoch_row["start_time"]), float(epoch_row["stop_time"]))
        loaded = {
            region: load_nwb_region_spikes(
                nwbfile,
                nwb_file_name=nwb_path.name,
                region=region,
                time_support=epoch_bounds,
            )
            for region in FIGURE_3_REGIONS
        }
        identities = {
            region: _source_identity(
                nwb_file_name=nwb_path.name,
                loaded=loaded[region],
            )
            for region in FIGURE_3_REGIONS
        }
        parent_v1 = _one_record(
            parent_session["source_identity"],
            label="Figure 2 V1 spike source",
            region="v1",
        )
        for field in ("spikesorting_merge_id", "selected_units_sha256", "n_units"):
            if str(identities["v1"][field]) != str(parent_v1[field]):
                raise ValueError(f"V1 spike source changed relative to Figure 2: {field}.")

        modulation_records = []
        modulation_results = {}
        for region in FIGURE_3_REGIONS:
            record, result = _run_modulation(
                run_dir=run_dir,
                animal_name=animal_name,
                date=date,
                epoch=FIGURE_3_LIGHT_EPOCH,
                nwb_file_name=nwb_path.name,
                ripple_table=ripple_table,
                epoch_bounds=epoch_bounds,
                ripple_row=ripple_row,
                loaded=loaded[region],
                source_identity=identities[region],
            )
            modulation_records.append(record)
            modulation_results[region] = result

        glm_records = []
        for parameters in FIGURE_3_GLM_PARAMETER_PRESETS:
            try:
                glm_records.append(
                    _run_glm(
                        run_dir=run_dir,
                        animal_name=animal_name,
                        date=date,
                        epoch=FIGURE_3_LIGHT_EPOCH,
                        nwb_file_name=nwb_path.name,
                        ripple_table=ripple_table,
                        epoch_interval=epoch_interval,
                        ripple_provenance=detector_provenance,
                        ca1=loaded["ca1"],
                        v1=loaded["v1"],
                        ca1_identity=identities["ca1"],
                        v1_identity=identities["v1"],
                        parameters=parameters,
                    )
                )
            finally:
                _clear_jax_caches()

        xcorr_records = []
        schematic_records = []
        if (animal_name, date) == FIGURE_3_XCORR_SESSION:
            xcorr_records.append(
                _run_xcorr(
                    run_dir=run_dir,
                    animal_name=animal_name,
                    date=date,
                    epoch=FIGURE_3_LIGHT_EPOCH,
                    nwb_file_name=nwb_path.name,
                    ripple_table=ripple_table,
                    ripple_provenance=detector_provenance,
                    ca1=loaded["ca1"],
                    v1=loaded["v1"],
                    ca1_identity=identities["ca1"],
                    v1_identity=identities["v1"],
                )
            )
        if (animal_name, date) == FIGURE_3_SCHEMATIC_SESSION:
            payload = _build_schematic_payload(
                nwbfile,
                animal_name=animal_name,
                date=date,
                epoch=FIGURE_3_LIGHT_EPOCH,
                nwb_file_name=nwb_path.name,
                ripple_table=ripple_table,
                ca1=loaded["ca1"],
                v1=loaded["v1"],
                ca1_modulation=modulation_results["ca1"],
            )
            payload_path = _write_schematic_payload(
                payload, session_dir / "figure_payloads" / SCHEMATIC_FILENAME
            )
            schematic_records.append(
                _relative_artifact_record(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "epoch": FIGURE_3_LIGHT_EPOCH,
                        "payload_path": payload_path,
                        "schema_version": SCHEMATIC_SCHEMA_VERSION,
                        "n_ripples": int(payload["n_ripples"]),
                        "ripple_start_s": float(payload["ripple_start_s"]),
                        "ripple_end_s": float(payload["ripple_end_s"]),
                        "channel": int(payload["channel"]),
                        "artifact_origin": "computed",
                    },
                    {"payload_path": payload_path},
                    run_dir=run_dir,
                    path_fields=("payload_path",),
                )
            )

    session_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_utc": utc_now(),
        "code_provenance": code_provenance(),
        "runtime_provenance": {"ripple_glm_jax": gpu_provenance},
        "status": "complete",
        "animal_name": animal_name,
        "date": date,
        "nwb_file_name": nwb_path.name,
        "nwb_path": str(nwb_path),
        "nwb_fingerprint": fingerprint,
        "epochs": {"light": FIGURE_3_LIGHT_EPOCH},
        "regions": list(FIGURE_3_REGIONS),
        "parameters": campaign["analysis_parameters"],
        "parent_figure_2": dict(parent_snapshot),
        "parent_artifacts": parent_artifacts,
        "source_identity": [identities[name] for name in FIGURE_3_REGIONS],
        "nwb_sources": {
            "epoch_interval": _native(epoch_row),
            "ripple_intervals": _native(ripple_row),
        },
        "artifacts": {
            "ripple_modulation": modulation_records,
            "ripple_glm": glm_records,
            "ripple_cross_region_xcorr": xcorr_records,
            "panel_b_schematic": schematic_records,
        },
    }
    manifest_path = session_dir / SESSION_MANIFEST_FILENAME
    write_json_once(session_manifest, manifest_path)
    append_session_manifest(campaign, session_manifest, run_dir=run_dir)
    return session_manifest


def run_figure_3_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Claim one session and remove only outputs created by a failed call."""
    run_dir, _campaign, _parent = prepare_figure_3_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(run_dir, animal_name=animal_name, date=date)
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = session_dir.parent / f".{session_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Figure 3 session is already running: {animal_name} {date}"
            ) from exc
        preexisting = session_dir.exists()
        try:
            return _run_figure_3_session(
                run_id=run_id,
                animal_name=animal_name,
                date=date,
                parent_run_id=parent_run_id,
                scratch_root=scratch_root,
            )
        except BaseException:
            if not preexisting and session_dir.exists():
                shutil.rmtree(session_dir)
            raise


def _verify_record(record: Mapping[str, Any], *, run_dir: Path) -> None:
    """Verify one sealed, computed, run-local Figure 3 artifact record."""
    if str(record.get("artifact_origin")) != "computed":
        raise ValueError("Figure 3 artifacts must be computed de novo.")
    unhashed = dict(record)
    observed = str(unhashed.pop("record_sha256", ""))
    if not observed or provenance_sha256(unhashed) != observed:
        raise ValueError("Figure 3 artifact record checksum mismatch.")
    path_hashes = record.get("artifact_sha256")
    if not isinstance(path_hashes, Mapping) or not path_hashes:
        raise ValueError("Figure 3 artifact record lacks file checksums.")
    for field, digest in path_hashes.items():
        path = resolve_run_path(str(record[field]), run_dir=run_dir)
        if not path.is_file() or file_sha256(path) != str(digest):
            raise ValueError(f"Figure 3 artifact checksum mismatch: {path}")


def load_figure_3_session_manifest(
    path: Path,
    *,
    run_dir: Path,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load one completed Figure 3 session and verify all dependencies."""
    manifest_path = Path(path).resolve(strict=True)
    guarded_run_dir = Path(run_dir).resolve(strict=True)
    if not manifest_path.is_relative_to(guarded_run_dir):
        raise ValueError("Figure 3 session manifest escapes its run.")
    session = load_json(manifest_path)
    if (
        session.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or session.get("status") != "complete"
    ):
        raise ValueError("Figure 3 session manifest is not complete.")
    if session.get("epochs") != {"light": FIGURE_3_LIGHT_EPOCH}:
        raise ValueError("Figure 3 session must contain only the fixed light epoch.")
    if session.get("regions") != list(FIGURE_3_REGIONS):
        raise ValueError("Figure 3 session regional order changed.")
    gpu = session.get("runtime_provenance", {}).get("ripple_glm_jax")
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("default_backend") != "gpu"
        or int(gpu.get("device_count", 0)) < 1
        or not isinstance(gpu.get("devices"), list)
        or any(row.get("platform") != "gpu" for row in gpu["devices"])
    ):
        raise ValueError("Figure 3 session lacks valid JAX GPU provenance.")
    artifacts = session.get("artifacts")
    required = {
        "ripple_modulation",
        "ripple_glm",
        "ripple_cross_region_xcorr",
        "panel_b_schematic",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("Figure 3 session has an unexpected artifact schema.")
    special = (
        str(session["animal_name"]), str(session["date"])
    ) == FIGURE_3_XCORR_SESSION
    expected_counts = {
        "ripple_modulation": 2,
        "ripple_glm": 2,
        "ripple_cross_region_xcorr": int(special),
        "panel_b_schematic": int(special),
    }
    for family, expected_count in expected_counts.items():
        records = artifacts[family]
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(
                f"Figure 3 family {family!r} must contain {expected_count} records."
            )
        for record in records:
            _verify_record(record, run_dir=guarded_run_dir)
    if {row["region"] for row in artifacts["ripple_modulation"]} != set(
        FIGURE_3_REGIONS
    ):
        raise ValueError("Figure 3 regional modulation artifacts are incomplete.")
    if {
        row["source_predictor_mode"] for row in artifacts["ripple_glm"]
    } != {"unit_vector", "mean_activity"}:
        raise ValueError("Figure 3 RippleGLM model roles are incomplete.")
    for record in artifacts["ripple_glm"]:
        bundle = ripple_glm.load_ripple_glm_artifact(
            resolve_run_path(record["artifact_dir"], run_dir=guarded_run_dir)
        )
        if (
            str(bundle["ripple_glm_id"]) != str(record["ripple_glm_id"])
            or str(bundle["analysis_status"]) != str(record["analysis_status"])
        ):
            raise ValueError("RippleGLM record disagrees with its bundle.")
    for record in artifacts["ripple_cross_region_xcorr"]:
        bundle = ripple_cross_region_xcorr.load_ripple_cross_region_xcorr_artifact(
            resolve_run_path(record["artifact_dir"], run_dir=guarded_run_dir)
        )
        if str(bundle["ripple_cross_region_xcorr_id"]) != str(
            record["ripple_cross_region_xcorr_id"]
        ):
            raise ValueError("RippleCrossRegionXCorr record disagrees with its bundle.")
    for record in artifacts["panel_b_schematic"]:
        load_panel_b_schematic_payload(
            resolve_run_path(record["payload_path"], run_dir=guarded_run_dir),
            expected_sha256=record["artifact_sha256"]["payload_path"],
        )

    parent = session.get("parent_figure_2")
    if not isinstance(parent, Mapping):
        raise ValueError("Figure 3 session lacks its Figure 2 parent snapshot.")
    current = build_figure_2_parent_snapshot(
        str(parent["run_id"]), scratch_root=scratch_root
    )
    if canonical_json(current) != canonical_json(dict(parent)):
        raise ValueError("Figure 3 parent changed after computation.")
    _parent_run_dir, expected_parent, selected_parent = _load_parent_session(
        parent,
        animal_name=str(session["animal_name"]),
        date=str(session["date"]),
        scratch_root=scratch_root,
    )
    if canonical_json(session.get("parent_artifacts")) != canonical_json(
        selected_parent
    ):
        raise ValueError("Figure 3 parent artifact pointers changed.")
    for field in ("nwb_file_name", "nwb_path", "nwb_fingerprint"):
        if canonical_json(session.get(field)) != canonical_json(
            expected_parent.get(field)
        ):
            raise ValueError(f"Figure 3 {field} differs from its Figure 2 parent.")
    return session


def load_figure_3_campaign(
    run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load a Figure 3 campaign and verify every completed session."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    campaign = load_json(run_dir / CAMPAIGN_MANIFEST_FILENAME)
    if campaign.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported Figure 3 campaign schema version.")
    if str(campaign.get("run_id")) != str(run_id):
        raise ValueError("Figure 3 campaign run_id does not match its directory.")
    if campaign.get("analysis_parameters", {}).get("pipeline") != FIGURE_3_PIPELINE:
        raise ValueError("Selected campaign is not a Figure 3 run.")
    parent = campaign["analysis_parameters"]["parent_figure_2"]
    current = build_figure_2_parent_snapshot(
        str(parent["run_id"]), scratch_root=scratch_root
    )
    if canonical_json(current) != canonical_json(parent):
        raise ValueError("Figure 3 parent changed after campaign selection.")
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list):
        raise ValueError("Figure 3 campaign sessions must be a list.")
    sessions = []
    seen: set[tuple[str, str]] = set()
    for summary in summaries:
        identity = (str(summary.get("animal_name")), str(summary.get("date")))
        if identity in seen:
            raise ValueError(f"Figure 3 contains duplicate session {identity!r}.")
        seen.add(identity)
        session = load_figure_3_session_manifest(
            resolve_run_path(summary["session_manifest_path"], run_dir=run_dir),
            run_dir=run_dir,
            scratch_root=scratch_root,
        )
        if (
            str(session["run_id"]) != str(run_id)
            or (str(session["animal_name"]), str(session["date"])) != identity
        ):
            raise ValueError("Figure 3 campaign and session identities disagree.")
        if canonical_json(session["parameters"]) != canonical_json(
            campaign["analysis_parameters"]
        ):
            raise ValueError("Figure 3 session parameters changed after selection.")
        sessions.append(session)
    return run_dir, campaign, sessions


def _parser() -> argparse.ArgumentParser:
    """Build the explicit one-session Figure 3 CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id", default=DEFAULT_PARENT_RUN_ID)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one Figure 3 session without plotting or database access."""
    args = _parser().parse_args(argv)
    manifest = run_figure_3_session(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        animal_name=args.animal_name,
        date=args.date,
        scratch_root=args.scratch_root,
    )
    print(
        "Completed offline Figure 3 inputs for "
        f"{manifest['animal_name']} {manifest['date']} "
        f"(light={manifest['epochs']['light']})."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PARENT_RUN_ID",
    "FIGURE_3_LIGHT_EPOCH",
    "FIGURE_3_PIPELINE",
    "FIGURE_3_REGIONS",
    "build_figure_2_parent_snapshot",
    "build_figure_3_configuration",
    "load_figure_3_campaign",
    "load_figure_3_session_manifest",
    "load_panel_b_schematic_payload",
    "main",
    "prepare_figure_3_campaign",
    "run_figure_3_session",
]
