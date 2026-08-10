"""Run the remaining Figure 1 analyses without DataJoint or legacy inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import movement
from v1ca1.spyglass.offline.figure_1 import DEFAULT_NWB_ROOT
from v1ca1.spyglass.offline.figure_1_decoding import (
    FIGURE_1_DECODING_PARAMETERS,
    run_figure_1_decoding,
)
from v1ca1.spyglass.offline.figure_1_examples import (
    DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    compute_nwb_example_payload,
    get_example_payload_path,
    write_example_payload,
)
from v1ca1.spyglass.offline.figure_1_models import (
    FIGURE_1_DPP_ENCODING_PARAMETERS,
    FIGURE_1_MOTOR_ENCODING_PARAMETERS,
    run_offline_dpp_encoding,
    run_offline_motor_encoding,
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
    load_campaign_manifest,
    load_json,
    load_session_manifest,
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
    load_run_epoch_catalog_objects,
    select_run_epoch_catalog,
    validate_nwb_session_identity,
)
from v1ca1.spyglass.selection import canonical_json, provenance_sha256


DEFAULT_PARENT_RUN_ID = "figure1d-nwb-v1"
FULL_FIGURE_REGION = "v1"
FULL_W_CONFIGURATION_NAME = "full_w"
FULL_FIGURE_PIPELINE = "figure_1_full"
FULL_FIGURE_EXAMPLES = (
    {
        "panel": "B",
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "02_r1",
        "region": "v1",
        "sorting_unit_id": 229,
    },
    {
        "panel": "B",
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "06_r3",
        "region": "v1",
        "sorting_unit_id": 229,
    },
    {
        "panel": "B",
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "region": "v1",
        "sorting_unit_id": 229,
    },
    {
        "panel": "C",
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "region": "v1",
        "sorting_unit_id": 34,
    },
    {
        "panel": "C",
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "10_r5",
        "region": "v1",
        "sorting_unit_id": 473,
    },
)
_MODEL_ARTIFACT_FIELDS = {
    "dpp_encoding": ("dpp_encoding_path",),
    "motor_encoding": (
        "artifact_manifest_path",
        "selected_units_path",
        "nested_cv_path",
        "full_refit_path",
    ),
}
_DECODING_ARTIFACT_FIELDS = (
    "artifact_manifest_path",
    "decoding_summary_path",
    "unit_eligibility_path",
    "selected_units_path",
    "binned_error_path",
)


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: str,
) -> dict[str, Any]:
    """Return exactly one manifest record matching string selectors."""
    matches = [
        dict(record)
        for record in records
        if all(
            str(record.get(field)) == str(value)
            for field, value in selectors.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {selectors!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def build_parent_snapshot(
    parent_run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Freeze the validated Figure 1D campaign and session manifest hashes."""
    parent_run_dir = get_run_dir(parent_run_id, scratch_root=scratch_root)
    campaign = load_campaign_manifest(
        parent_run_id,
        scratch_root=scratch_root,
        require_artifacts=True,
    )
    if campaign.get("analysis_parameters", {}).get("pipeline") != (
        "figure_1_initial_slice"
    ):
        raise ValueError("Parent campaign is not a Figure 1D offline run.")
    manifest_path = parent_run_dir / CAMPAIGN_MANIFEST_FILENAME
    sessions = []
    for summary in campaign["sessions"]:
        session_path = resolve_run_path(
            summary["session_manifest_path"],
            run_dir=parent_run_dir,
        )
        sessions.append(
            {
                "animal_name": str(summary["animal_name"]),
                "date": str(summary["date"]),
                "session_manifest_path": str(summary["session_manifest_path"]),
                "session_manifest_sha256": file_sha256(session_path),
            }
        )
    return {
        "run_id": str(parent_run_id),
        "manifest_sha256": file_sha256(manifest_path),
        "sessions": sorted(
            sessions,
            key=lambda row: (row["animal_name"], row["date"]),
        ),
    }


def build_full_figure_configuration(
    parent_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable approved analysis configuration."""
    return {
        "pipeline": FULL_FIGURE_PIPELINE,
        "parent_figure_1d": dict(parent_snapshot),
        "region": FULL_FIGURE_REGION,
        "trajectory_types": list(TRAJECTORY_TYPES),
        "full_w_configuration_name": FULL_W_CONFIGURATION_NAME,
        "position_roles": ["head", "body"],
        "example_parameters": {
            "position_role": "head",
            "analysis_start_offset_samples": DEFAULT_POSITION_OFFSET,
            "speed_threshold_cm_s": DEFAULT_SPEED_THRESHOLD_CM_S,
            "position_bin_count": DEFAULT_POSITION_BIN_COUNT,
            "gaussian_smoothing_sigma_bins": (
                DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS
            ),
            "spike_source": "RegionSortedSpikesGroup",
            "artifact_format": "npz",
        },
        "example_specs": [dict(spec) for spec in FULL_FIGURE_EXAMPLES],
        "motor_encoding_parameters": dict(
            FIGURE_1_MOTOR_ENCODING_PARAMETERS
        ),
        "dpp_encoding_parameters": dict(FIGURE_1_DPP_ENCODING_PARAMETERS),
        "path_progression_decoding_parameters": dict(
            FIGURE_1_DECODING_PARAMETERS
        ),
        "diagnostic_figures": False,
        "artifact_origin": "computed",
    }


def prepare_full_figure_campaign(
    *,
    run_id: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create or validate one append-only full-Figure-1 campaign."""
    parent_snapshot = build_parent_snapshot(
        parent_run_id,
        scratch_root=scratch_root,
    )
    configuration = build_full_figure_configuration(parent_snapshot)
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    if (run_dir / CAMPAIGN_MANIFEST_FILENAME).exists():
        loaded_run_dir, campaign, _ = load_full_figure_campaign(
            run_id,
            scratch_root=scratch_root,
        )
        if canonical_json(campaign.get("analysis_parameters")) != canonical_json(
            configuration
        ):
            raise ValueError(
                "Existing campaign uses different analysis parameters; "
                "use a new run_id."
            )
        if canonical_json(campaign.get("source_identity_policy")) != canonical_json(
            SOURCE_IDENTITY_POLICY
        ):
            raise ValueError(
                "Existing campaign uses a different unit identity policy; "
                "use a new run_id."
            )
        return loaded_run_dir, campaign, parent_snapshot
    run_dir, campaign = prepare_campaign(
        run_id=run_id,
        analysis_parameters=configuration,
        source_identity_policy=SOURCE_IDENTITY_POLICY,
        scratch_root=scratch_root,
    )
    return run_dir, campaign, parent_snapshot


def _load_parent_session(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    epoch: str,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Load one hash-pinned parent session manifest."""
    current = build_parent_snapshot(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(dict(parent_snapshot)):
        raise ValueError("Parent Figure 1D campaign changed after selection.")
    parent_run_dir = get_run_dir(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    summary = _one_record(
        parent_snapshot["sessions"],
        label="parent session",
        animal_name=animal_name,
        date=date,
    )
    session_path = resolve_run_path(
        summary["session_manifest_path"],
        run_dir=parent_run_dir,
    )
    if file_sha256(session_path) != str(summary["session_manifest_sha256"]):
        raise ValueError("Parent session manifest checksum changed.")
    session = load_session_manifest(
        session_path,
        run_dir=parent_run_dir,
        require_artifacts=True,
    )
    if str(epoch) not in {str(value) for value in session["epochs"]}:
        raise ValueError(
            f"Parent session does not contain requested dark epoch {epoch!r}."
        )
    return parent_run_dir, session


def _checked_parent_path(
    record: Mapping[str, Any],
    field: str,
    *,
    parent_run_dir: Path,
) -> Path:
    """Resolve and checksum one parent artifact path."""
    path = resolve_run_path(record[field], run_dir=parent_run_dir)
    expected = record.get("artifact_sha256", {}).get(field)
    if expected is None or file_sha256(path) != str(expected):
        raise ValueError(f"Parent artifact checksum mismatch for {field!r}.")
    return path


def load_parent_analysis_inputs(
    parent_run_dir: Path,
    parent_session: Mapping[str, Any],
    *,
    epoch: str,
    region: str = FULL_FIGURE_REGION,
) -> dict[str, Any]:
    """Load the movement and stability results required downstream."""
    artifacts = parent_session["artifacts"]
    movement_record = _one_record(
        artifacts["movement_firing_rate"],
        label="movement artifact",
        epoch=epoch,
        region=region,
    )
    _checked_parent_path(
        movement_record,
        "firing_rate_path",
        parent_run_dir=parent_run_dir,
    )
    _checked_parent_path(
        movement_record,
        "movement_intervals_path",
        parent_run_dir=parent_run_dir,
    )
    movement_dir = resolve_run_path(
        movement_record["artifact_dir"],
        run_dir=parent_run_dir,
    )
    movement_result = movement.load_movement_artifacts(movement_dir)

    stability_records = {}
    stability_tables = {}
    for trajectory_type in TRAJECTORY_TYPES:
        record = _one_record(
            artifacts["path_specific_place_stability"],
            label="stability artifact",
            epoch=epoch,
            region=region,
            trajectory_type=trajectory_type,
        )
        path = _checked_parent_path(
            record,
            "stability_path",
            parent_run_dir=parent_run_dir,
        )
        stability_records[trajectory_type] = record
        stability_tables[trajectory_type] = pd.read_parquet(path)
    source = _one_record(
        parent_session["source_identity"],
        label="regional spike source",
        region=region,
        source="ImportedSpikeSorting",
    )
    return {
        "movement_record": movement_record,
        "movement": movement_result,
        "stability_records": stability_records,
        "stability_tables": stability_tables,
        "source_identity": source,
    }


def _require_parent_fingerprint(
    nwb_path: Path,
    nwbfile: Any,
    parent_session: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the NWB source to be unchanged from the parent run."""
    observed = nwb_fingerprint(nwb_path, nwbfile)
    expected = dict(parent_session["nwb_fingerprint"])
    for field in (
        "resolved_path",
        "size_bytes",
        "mtime_ns",
        "nwb_identifier",
        "units_object_id",
    ):
        if observed.get(field) != expected.get(field):
            raise ValueError(f"Parent NWB fingerprint changed for {field!r}.")
    return observed


def _source_pointer(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact catalog provenance for one selected NWB object."""
    return {
        name: row[name]
        for name in (
            "source_table_path",
            "source_object_path",
        )
        if name in row
    }


def _run_full_figure_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    epoch: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Compute and persist one session's remaining Figure 1 analyses."""
    animal_name, date, epoch = map(str, (animal_name, date, epoch))
    run_dir, campaign, parent_snapshot = prepare_full_figure_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(
        run_dir,
        animal_name=animal_name,
        date=date,
    )
    if session_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite a full-Figure-1 session: {session_dir}"
        )
    parent_run_dir, parent_session = _load_parent_session(
        parent_snapshot,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        scratch_root=scratch_root,
    )
    parent_inputs = load_parent_analysis_inputs(
        parent_run_dir,
        parent_session,
        epoch=epoch,
    )
    nwb_path = Path(str(parent_session["nwb_path"])).resolve(strict=True)
    if nwb_path.is_relative_to(run_dir.resolve(strict=False)):
        raise ValueError("Source NWB must remain outside the output run.")

    import pynwb

    example_records = []
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=animal_name,
            date=date,
        )
        fingerprint = _require_parent_fingerprint(
            nwb_path,
            nwbfile,
            parent_session,
        )
        selection = select_run_epoch_catalog(
            nwbfile,
            nwb_file_name=nwb_path.name,
            epoch=epoch,
            position_roles=("head", "body"),
            trajectory_types=TRAJECTORY_TYPES,
            graph_configurations=(FULL_W_CONFIGURATION_NAME,),
            require_dark=True,
        )
        sources = load_run_epoch_catalog_objects(nwbfile, selection)
        start = float(selection["epoch_row"]["start_time"])
        stop = float(selection["epoch_row"]["stop_time"])
        spikes = load_nwb_region_spikes(
            nwbfile,
            nwb_file_name=nwb_path.name,
            region=FULL_FIGURE_REGION,
            time_support=(start, stop),
        )
        parent_source = parent_inputs["source_identity"]
        if (
            str(spikes["spikesorting_merge_id"])
            != str(parent_source["spikesorting_merge_id"])
            or str(spikes["selected_units_sha256"])
            != str(parent_source["selected_units_sha256"])
        ):
            raise ValueError(
                "NWB regional units changed relative to the parent campaign."
            )
        region_group_id = str(
            parent_source["offline_region_sorted_spikes_view_id"]
        )

        # Resolve the small, manuscript-fixed example set before starting the
        # substantially more expensive population model fits.
        for spec in FULL_FIGURE_EXAMPLES:
            if (
                str(spec["animal_name"]) != animal_name
                or str(spec["date"]) != date
            ):
                continue
            payload = compute_nwb_example_payload(
                nwbfile,
                nwb_file_name=nwb_path.name,
                animal_name=animal_name,
                date=date,
                epoch=str(spec["epoch"]),
                region=str(spec["region"]),
                sorting_unit_id=spec["sorting_unit_id"],
                trajectory_types=TRAJECTORY_TYPES,
            )
            path = get_example_payload_path(
                run_dir,
                animal_name=animal_name,
                date=date,
                epoch=str(spec["epoch"]),
                region=str(spec["region"]),
                sorting_unit_id=spec["sorting_unit_id"],
            )
            written = write_example_payload(
                payload,
                path,
                run_dir=run_dir,
            )
            example_records.append(
                {
                    **dict(spec),
                    "payload_path": relative_run_path(
                        written["payload_path"],
                        run_dir=run_dir,
                    ),
                    "artifact_sha256": written["artifact_sha256"],
                    "persistent_unit_identity": payload["metadata"][
                        "persistent_unit_identity"
                    ],
                    "artifact_origin": "computed",
                }
            )

        movement_record = parent_inputs["movement_record"]
        stability_records = parent_inputs["stability_records"]
        stability_ids = {
            trajectory_type: record[
                "path_specific_place_stability_id"
            ]
            for trajectory_type, record in stability_records.items()
        }
        common_inputs = {
            "animal_name": animal_name,
            "date": date,
            "region": FULL_FIGURE_REGION,
            "epoch": epoch,
            "nwb_file_name": nwb_path.name,
            "region_sorted_spikes_group_id": region_group_id,
            "movement_firing_rate_id": movement_record[
                "movement_firing_rate_id"
            ],
            "stability_ids_by_trajectory": stability_ids,
            "spikes": spikes["ts_group"],
            "stable_unit_ids": spikes["unit_ids"],
            "trajectory_intervals_by_type": sources[
                "trajectory_intervals"
            ],
            "graph_inputs_by_configuration": sources["graph_inputs"],
            "movement_intervals": parent_inputs["movement"][
                "movement_intervals"
            ],
            "movement_firing_rate_table": parent_inputs["movement"]["table"],
            "stability_tables_by_trajectory": parent_inputs[
                "stability_tables"
            ],
        }
        dpp_record = run_offline_dpp_encoding(
            output_dir=run_dir,
            parameters=FIGURE_1_DPP_ENCODING_PARAMETERS,
            position=sources["positions"]["head"],
            **common_inputs,
        )
        motor_record = run_offline_motor_encoding(
            output_dir=run_dir,
            parameters=FIGURE_1_MOTOR_ENCODING_PARAMETERS,
            primary_position=sources["positions"]["head"],
            orientation_reference_position=sources["positions"]["body"],
            primary_position_series_name=selection["position_rows"]["head"][
                "position_series_name"
            ],
            orientation_reference_position_series_name=(
                selection["position_rows"]["body"]["position_series_name"]
            ),
            primary_position_source=selection["position_rows"]["head"][
                "source_object_path"
            ],
            orientation_reference_position_source=selection[
                "position_rows"
            ]["body"]["source_object_path"],
            **common_inputs,
        )
        decoding_result = run_figure_1_decoding(
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            nwb_file_name=nwb_path.name,
            region_sorted_spikes_group_id=region_group_id,
            movement_firing_rate_id=movement_record[
                "movement_firing_rate_id"
            ],
            stability_source_ids=stability_ids,
            spikes=spikes["ts_group"],
            stable_unit_ids=spikes["unit_ids"],
            movement_firing_rate_table=parent_inputs["movement"]["table"],
            stability_tables_by_trajectory=parent_inputs[
                "stability_tables"
            ],
            position=sources["positions"]["head"],
            trajectory_intervals=sources["trajectory_intervals"],
            graph_inputs={
                name: sources["graph_inputs"][name]
                for name in TRAJECTORY_TYPES
            },
            movement_interval=parent_inputs["movement"][
                "movement_intervals"
            ],
            output_dir=run_dir,
        )

    session_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_utc": utc_now(),
        "code_provenance": code_provenance(),
        "status": "complete",
        "animal_name": animal_name,
        "date": date,
        "nwb_file_name": nwb_path.name,
        "nwb_path": str(nwb_path),
        "nwb_fingerprint": fingerprint,
        "epochs": [epoch],
        "regions": [FULL_FIGURE_REGION],
        "trajectories": list(TRAJECTORY_TYPES),
        "parameters": campaign["analysis_parameters"],
        "parent_figure_1d": {
            "run_id": parent_snapshot["run_id"],
            "session_manifest_sha256": _one_record(
                parent_snapshot["sessions"],
                label="parent session",
                animal_name=animal_name,
                date=date,
            )["session_manifest_sha256"],
        },
        "source_identity": parent_inputs["source_identity"],
        "position_selection": {
            role: {
                **_source_pointer(row),
                "position_series_name": row["position_series_name"],
                "position_role": row["position_role"],
                "spatial_unit": row["spatial_unit"],
                "analysis_start_offset_samples": row[
                    "analysis_start_offset_samples"
                ],
            }
            for role, row in selection["position_rows"].items()
        },
        "graph_selection": {
            name: {
                **_source_pointer(row),
                "configuration_name": row["configuration_name"],
                "coordinate_unit": row["coordinate_unit"],
            }
            for name, row in selection["graph_rows"].items()
        },
        "upstream_artifacts": {
            "movement_firing_rate_id": movement_record[
                "movement_firing_rate_id"
            ],
            "stability_ids_by_trajectory": stability_ids,
        },
        "artifacts": {
            "figure_examples": example_records,
            "dpp_encoding": [dpp_record],
            "motor_encoding": [motor_record],
            "path_progression_decoding": [
                decoding_result["artifact_record"]
            ],
        },
    }
    manifest_path = session_dir / SESSION_MANIFEST_FILENAME
    write_json_once(session_manifest, manifest_path)
    append_session_manifest(
        campaign,
        session_manifest,
        run_dir=run_dir,
    )
    return session_manifest


def run_full_figure_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    epoch: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Run one session and remove only its new outputs after a failure."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    session_dir = get_session_dir(
        run_dir,
        animal_name=animal_name,
        date=date,
    )
    session_preexisted = session_dir.exists()
    try:
        return _run_full_figure_session(
            run_id=run_id,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            parent_run_id=parent_run_id,
            scratch_root=scratch_root,
        )
    except BaseException:
        if not session_preexisted and session_dir.exists():
            shutil.rmtree(session_dir)
        raise


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    """Require one declared full-Figure-1 artifact checksum."""
    if not path.is_file() or file_sha256(path) != str(expected_sha256):
        raise ValueError(f"Full-Figure-1 artifact checksum mismatch: {path}")


def load_full_figure_session_manifest(
    path: Path,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Load and checksum one completed full-Figure-1 session manifest."""
    path = Path(path).resolve(strict=True)
    guarded = Path(run_dir).resolve(strict=True)
    if not path.is_relative_to(guarded):
        raise ValueError("Full-Figure-1 session manifest escapes its run.")
    session = load_json(path)
    if (
        session.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or session.get("status") != "complete"
    ):
        raise ValueError("Full-Figure-1 session manifest is not complete.")
    artifacts = session.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("Full-Figure-1 artifacts must be a mapping.")
    required_families = {
        "figure_examples",
        "dpp_encoding",
        "motor_encoding",
        "path_progression_decoding",
    }
    missing_families = sorted(required_families.difference(artifacts))
    if missing_families:
        raise ValueError(
            "Full-Figure-1 manifest is missing artifact families: "
            f"{missing_families!r}."
        )
    for family in required_families:
        if not isinstance(artifacts[family], list):
            raise ValueError(f"Artifact family {family!r} must be a list.")
    for family in (*_MODEL_ARTIFACT_FIELDS, "path_progression_decoding"):
        if len(artifacts[family]) != 1:
            raise ValueError(
                f"Artifact family {family!r} must contain exactly one record."
            )

    for record in artifacts["figure_examples"]:
        if str(record.get("artifact_origin", "")) != "computed":
            raise ValueError("Figure examples must be computed de novo.")
        payload_path = resolve_run_path(record["payload_path"], run_dir=guarded)
        _verify_artifact(payload_path, record["artifact_sha256"])
    for family, expected_fields in _MODEL_ARTIFACT_FIELDS.items():
        record = artifacts[family][0]
        if str(record.get("artifact_origin", "")) != "computed":
            raise ValueError(f"Artifact family {family!r} must be computed de novo.")
        supplied_record_sha256 = str(record.get("record_sha256", ""))
        unhashed = dict(record)
        unhashed.pop("record_sha256", None)
        if supplied_record_sha256 != provenance_sha256(unhashed):
            raise ValueError(f"Artifact family {family!r} record checksum mismatch.")
        model_artifacts = record.get("artifacts")
        if not isinstance(model_artifacts, Mapping) or set(model_artifacts) != set(
            expected_fields
        ):
            raise ValueError(
                f"Artifact family {family!r} must declare exactly "
                f"{list(expected_fields)!r}."
            )
        for artifact in model_artifacts.values():
            artifact_path = resolve_run_path(
                artifact["relative_path"],
                run_dir=guarded,
            )
            _verify_artifact(artifact_path, artifact["sha256"])
    decoding_record = artifacts["path_progression_decoding"][0]
    if str(decoding_record.get("artifact_origin", "")) != "computed":
        raise ValueError("PathProgressionDecoding must be computed de novo.")
    decoding_hashes = decoding_record.get("artifact_sha256")
    if not isinstance(decoding_hashes, Mapping) or set(decoding_hashes) != set(
        _DECODING_ARTIFACT_FIELDS
    ):
        raise ValueError(
            "PathProgressionDecoding must declare its complete artifact bundle."
        )
    for field, digest in decoding_hashes.items():
        artifact_path = resolve_run_path(decoding_record[field], run_dir=guarded)
        _verify_artifact(artifact_path, digest)
    return session


def load_full_figure_campaign(
    run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load a full-Figure-1 campaign and verify every declared artifact."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    campaign = load_json(run_dir / CAMPAIGN_MANIFEST_FILENAME)
    if campaign.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported full-Figure-1 campaign schema version.")
    if str(campaign.get("run_id")) != str(run_id):
        raise ValueError("Full-Figure-1 campaign run_id does not match its directory.")
    if campaign.get("analysis_parameters", {}).get("pipeline") != (
        FULL_FIGURE_PIPELINE
    ):
        raise ValueError("Selected campaign is not a full-Figure-1 run.")
    parent = campaign["analysis_parameters"]["parent_figure_1d"]
    if canonical_json(parent) != canonical_json(
        build_parent_snapshot(parent["run_id"], scratch_root=scratch_root)
    ):
        raise ValueError("Parent Figure 1D campaign changed after selection.")
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list):
        raise ValueError("Full-Figure-1 campaign sessions must be a list.")
    sessions = []
    seen: set[tuple[str, str]] = set()
    for summary in summaries:
        identity = (str(summary.get("animal_name")), str(summary.get("date")))
        if identity in seen:
            raise ValueError(
                f"Full-Figure-1 campaign contains duplicate session {identity!r}."
            )
        seen.add(identity)
        path = resolve_run_path(
            summary["session_manifest_path"],
            run_dir=run_dir,
        )
        session = load_full_figure_session_manifest(path, run_dir=run_dir)
        if (
            str(session.get("run_id")) != str(run_id)
            or str(session["animal_name"]) != str(summary["animal_name"])
            or str(session["date"]) != str(summary["date"])
        ):
            raise ValueError("Campaign and session identities disagree.")
        sessions.append(session)
    return run_dir, campaign, sessions


def _parser() -> argparse.ArgumentParser:
    """Build the explicit single-session full-Figure-1 CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id", default=DEFAULT_PARENT_RUN_ID)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--epoch", required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one explicit session without diagnostic plotting or a database."""
    args = _parser().parse_args(argv)
    manifest = run_full_figure_session(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        animal_name=args.animal_name,
        date=args.date,
        epoch=args.epoch,
        scratch_root=args.scratch_root,
    )
    print(
        "Completed full Figure 1 inputs for "
        f"{manifest['animal_name']} {manifest['date']} "
        f"({manifest['epochs'][0]})."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_NWB_ROOT",
    "DEFAULT_PARENT_RUN_ID",
    "FULL_FIGURE_EXAMPLES",
    "FULL_FIGURE_PIPELINE",
    "FULL_FIGURE_REGION",
    "build_full_figure_configuration",
    "build_parent_snapshot",
    "load_full_figure_campaign",
    "load_full_figure_session_manifest",
    "load_parent_analysis_inputs",
    "main",
    "prepare_full_figure_campaign",
    "run_full_figure_session",
]
