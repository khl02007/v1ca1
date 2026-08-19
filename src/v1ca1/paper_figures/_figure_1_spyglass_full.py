"""Adapt a complete offline campaign to the existing Figure 1 layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures import figure_1 as legacy
from v1ca1.paper_figures.datasets import get_processed_datasets, normalize_dataset_id
from v1ca1.spyglass import dpp_encoding, motor_encoding
from v1ca1.spyglass.nwb import catalog_augmented_nwb, load_interval_set
from v1ca1.spyglass.offline.figure_1_decoding import (
    load_figure_1_decoding_payload,
)
from v1ca1.spyglass.offline.figure_1_examples import load_example_payload
from v1ca1.spyglass.offline.figure_1_full import (
    FULL_FIGURE_EXAMPLES,
    FULL_FIGURE_PIPELINE,
    build_full_figure_configuration,
    load_full_figure_campaign,
)
from v1ca1.spyglass.offline.manifests import (
    DEFAULT_SCRATCH_ROOT,
    code_provenance,
    file_sha256,
    load_json,
    nwb_fingerprint,
    relative_run_path,
    resolve_run_path,
    utc_now,
    write_json_once,
)
from v1ca1.spyglass.offline.sources import validate_nwb_session_identity
from v1ca1.spyglass.selection import canonical_json


DEFAULT_FULL_FIGURE_OUTPUT_NAME = "figure_1_spyglass"
DEFAULT_L14_FIGURE_OUTPUT_NAME = "figure_1_l14_spyglass_validation"
FULL_FIGURE_PROVENANCE_SCHEMA_VERSION = 1
FULL_FIGURE_PROVENANCE_SUFFIX = ".spyglass-provenance.json"
FIGURE_MODES = ("l14-validation", "full")
EXPECTED_DATASETS = tuple(get_processed_datasets())
L14_DATASET = next(
    dataset for dataset in EXPECTED_DATASETS if dataset[0] == "L14"
)
_DPP_COMPARISON_COLUMNS = {
    "dpp_vs_absolute_place": "dpp_vs_absolute_place_bits_per_spike",
    "dpp_vs_absolute_task_progression": (
        "dpp_vs_distance_to_reward_bits_per_spike"
    ),
}


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: Any,
) -> dict[str, Any]:
    """Return exactly one manifest record matching fixed selectors."""
    matches = [
        dict(record)
        for record in records
        if all(
            str(record.get(name)) == str(value)
            for name, value in selectors.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {selectors!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _ordered_sessions(
    sessions: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], tuple[tuple[str, str, str], ...]]:
    """Return the sessions selected by partial or complete figure mode."""
    by_key = {}
    for session in sessions:
        key = (str(session["animal_name"]), str(session["date"]))
        if key in by_key:
            raise ValueError(f"Full Figure 1 campaign duplicates session {key!r}.")
        by_key[key] = dict(session)
    if mode == "l14-validation":
        expected_datasets = (L14_DATASET,)
    elif mode == "full":
        expected_datasets = EXPECTED_DATASETS
    else:
        raise ValueError(f"mode must be one of {FIGURE_MODES!r}.")
    expected_keys = {
        (animal_name, date) for animal_name, date, _epoch in expected_datasets
    }
    if mode == "full" and set(by_key) != expected_keys:
        raise ValueError(
            "Full Figure 1 requires exactly the four manuscript sessions; "
            f"expected {sorted(expected_keys)!r}, got {sorted(by_key)!r}."
        )
    missing = expected_keys.difference(by_key)
    if missing:
        raise ValueError(f"Figure campaign is missing sessions {sorted(missing)!r}.")
    ordered = []
    for animal_name, date, epoch in expected_datasets:
        session = by_key[(animal_name, date)]
        observed_epochs = tuple(str(value) for value in session.get("epochs", ()))
        if observed_epochs != (str(epoch),):
            raise ValueError(
                f"Session {animal_name} {date} must contain only dark epoch "
                f"{epoch!r}; got {observed_epochs!r}."
            )
        ordered.append(session)
    return ordered, expected_datasets


def _artifact_path(
    run_dir: Path,
    record: Mapping[str, Any],
    artifact_name: str,
) -> Path:
    """Resolve one already-checksummed model artifact within the run."""
    artifacts = record.get("artifacts", {})
    if artifact_name not in artifacts:
        raise ValueError(f"Artifact record is missing {artifact_name!r}.")
    return resolve_run_path(
        artifacts[artifact_name]["relative_path"],
        run_dir=run_dir,
    )


def _require_session_identity(
    values: Mapping[str, Any],
    session: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Require one artifact identity to match its enclosing session."""
    expected = {
        "animal_name": str(session["animal_name"]),
        "date": str(session["date"]),
        "epoch": str(session["epochs"][0]),
        "region": "v1",
    }
    mismatches = {
        name: (values.get(name), value)
        for name, value in expected.items()
        if str(values.get(name)) != value
    }
    if mismatches:
        raise ValueError(f"{label} session identity mismatch: {mismatches!r}.")


def _load_examples(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the five fixed example payloads and adapt panels B and C."""
    records = [
        dict(record)
        for session in sessions
        for record in session["artifacts"].get("figure_examples", ())
    ]
    selected_session_keys = {
        (str(animal_name), str(date))
        for animal_name, date, _epoch in datasets
    }
    selected_specs = tuple(
        spec
        for spec in FULL_FIGURE_EXAMPLES
        if (str(spec["animal_name"]), str(spec["date"]))
        in selected_session_keys
    )
    expected_keys = {
        tuple(
            str(spec[name])
            for name in (
                "panel",
                "animal_name",
                "date",
                "epoch",
                "region",
                "sorting_unit_id",
            )
        )
        for spec in selected_specs
    }
    observed_keys = {
        tuple(
            str(record.get(name))
            for name in (
                "panel",
                "animal_name",
                "date",
                "epoch",
                "region",
                "sorting_unit_id",
            )
        )
        for record in records
    }
    if observed_keys != expected_keys or len(records) != len(expected_keys):
        raise ValueError("Full Figure 1 example artifact set is not canonical.")

    loaded_by_key = {}
    for record in records:
        if str(record.get("artifact_origin")) != "computed":
            raise ValueError("Figure 1 examples must be computed de novo.")
        path = resolve_run_path(record["payload_path"], run_dir=run_dir)
        payload = load_example_payload(
            path,
            expected_sha256=str(record["artifact_sha256"]),
        )
        metadata = payload["metadata"]
        key = (
            str(record["animal_name"]),
            str(record["date"]),
            str(record["epoch"]),
            str(record["region"]),
            str(record["sorting_unit_id"]),
        )
        if key != (
            str(metadata["animal_name"]),
            str(metadata["date"]),
            str(metadata["epoch"]),
            str(metadata["region"]),
            str(metadata["sorting_unit_id"]),
        ):
            raise ValueError("Example manifest and NPZ identities disagree.")
        if canonical_json(record.get("persistent_unit_identity", {})) != (
            canonical_json(metadata["persistent_unit_identity"])
        ):
            raise ValueError(
                "Example manifest and NPZ persistent unit identities disagree."
            )
        loaded_by_key[key] = {
            "animal_name": metadata["animal_name"],
            "date": metadata["date"],
            "epoch": metadata["epoch"],
            "region": metadata["region"],
            "unit_id": metadata["sorting_unit_id"],
            "raster_positions": payload["raster_positions"],
            "firing_rates": payload["firing_rates"],
        }

    panel_b_spec = FULL_FIGURE_EXAMPLES[0]
    panel_b_base = (
        str(panel_b_spec["animal_name"]),
        str(panel_b_spec["date"]),
        str(panel_b_spec["region"]),
        str(panel_b_spec["sorting_unit_id"]),
    )
    dark_epoch = next(
        epoch
        for animal_name, date, epoch in datasets
        if (animal_name, date) == panel_b_base[:2]
    )
    epoch_keys = {
        "02_r1": "02_r1",
        "06_r3": "06_r3",
        "dark": dark_epoch,
    }
    panel_b = {
        "animal_name": panel_b_base[0],
        "date": panel_b_base[1],
        "region": panel_b_base[2],
        "unit_id": int(panel_b_base[3]),
        "epoch_order": tuple(epoch_keys),
        "epoch_labels": {
            key: legacy.PANEL_B_VISUAL_EPOCH_LABELS[key]
            for key in epoch_keys
        },
        "epoch_examples": {
            key: loaded_by_key[
                (
                    panel_b_base[0],
                    panel_b_base[1],
                    epoch,
                    panel_b_base[2],
                    panel_b_base[3],
                )
            ]
            for key, epoch in epoch_keys.items()
        },
        "trajectories": legacy.PANEL_B_VISUAL_TRAJECTORIES,
    }
    panel_c = []
    for animal_name, date, epoch, region, sorting_unit_id in legacy.PANEL_E_EXAMPLES:
        if (str(animal_name), str(date)) not in selected_session_keys:
            continue
        panel_c.append(
            loaded_by_key[
                (
                    str(animal_name),
                    str(date),
                    str(epoch),
                    str(region),
                    str(sorting_unit_id),
                )
            ]
        )
    return panel_b, panel_c


def _motor_delta_table(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Adapt de novo MotorEncoding bundles to the existing Panel E table."""
    rows = []
    for session in sessions:
        record = _one_record(
            session["artifacts"].get("motor_encoding", ()),
            label="MotorEncoding artifact",
        )
        if str(record.get("artifact_origin")) != "computed":
            raise ValueError("MotorEncoding must be computed de novo.")
        manifest_path = _artifact_path(
            run_dir,
            record,
            "artifact_manifest_path",
        )
        result = motor_encoding.load_motor_encoding_artifact(
            manifest_path.parent
        )
        if str(result.get("artifact_origin")) != "computed":
            raise ValueError("Loaded MotorEncoding artifact is not de novo.")
        nested = result["nested_cv"]
        values = np.asarray(
            nested["pooled_delta_bits_per_spike"]
            .sel(delta_metric=legacy.MOTOR_DELTA_METRIC)
            .values,
            dtype=float,
        ).reshape(-1)
        unit_coordinate = (
            "stable_unit_id"
            if "stable_unit_id" in nested.coords
            else "unit"
        )
        units = np.asarray(nested.coords[unit_coordinate].values).reshape(-1)
        if units.shape != values.shape:
            raise ValueError("MotorEncoding units and delta values do not align.")
        metadata = result["metadata"]
        _require_session_identity(
            metadata,
            session,
            label="MotorEncoding artifact",
        )
        rows.extend(
            {
                "animal_name": str(metadata["animal_name"]),
                "date": str(metadata["date"]),
                "epoch": str(metadata["epoch"]),
                "region": str(metadata["region"]),
                "unit": unit.item() if isinstance(unit, np.generic) else unit,
                "delta_log_likelihood_bits_per_spike": float(value),
                "source_path": str(manifest_path),
            }
            for unit, value in zip(units, values, strict=True)
            if np.isfinite(value)
        )
    return pd.DataFrame.from_records(
        rows,
        columns=list(legacy.MOTOR_DELTA_TABLE_COLUMNS),
    )


def _encoding_delta_table(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Adapt de novo DPPEncoding Parquets to the existing Panel F table."""
    rows = []
    labels = {
        comparison: label
        for comparison, label, _legacy_column in legacy.ENCODING_DPP_COMPARISONS
    }
    for session in sessions:
        record = _one_record(
            session["artifacts"].get("dpp_encoding", ()),
            label="DPPEncoding artifact",
        )
        if str(record.get("artifact_origin")) != "computed":
            raise ValueError("DPPEncoding must be computed de novo.")
        path = _artifact_path(run_dir, record, "dpp_encoding_path")
        table = dpp_encoding.load_dpp_encoding_artifact(path)
        selection = record.get("selection", {})
        if (
            str(selection.get("nwb_file_name"))
            != str(session["nwb_file_name"])
            or str(selection.get("epoch")) != str(session["epochs"][0])
        ):
            raise ValueError("DPPEncoding selection and session disagree.")
        if not table.empty:
            for column, expected in {
                "animal_name": session["animal_name"],
                "date": session["date"],
                "epoch": session["epochs"][0],
                "region": "v1",
            }.items():
                if set(table[column].astype(str)) != {str(expected)}:
                    raise ValueError(
                        f"DPPEncoding column {column!r} disagrees with session."
                    )
        for comparison, value_column in _DPP_COMPARISON_COLUMNS.items():
            for row in table.to_dict("records"):
                value = float(row[value_column])
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "animal_name": str(row["animal_name"]),
                        "date": str(row["date"]),
                        "epoch": str(row["epoch"]),
                        "region": str(row["region"]),
                        "unit": str(row["stable_unit_id"]),
                        "n_spikes": int(row["heldout_spike_count"]),
                        "comparison": comparison,
                        "comparison_label": labels[comparison],
                        "delta_log_likelihood_bits_per_spike": value,
                        "source_path": str(path),
                    }
                )
    return pd.DataFrame.from_records(
        rows,
        columns=list(legacy.ENCODING_DELTA_TABLE_COLUMNS),
    )


def _trajectory_intervals_from_nwb(
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Load hash-pinned trajectory intervals from the source NWB read-only."""
    import pynwb

    nwb_path = Path(str(session["nwb_path"])).resolve(strict=True)
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=str(session["animal_name"]),
            date=str(session["date"]),
        )
        observed = nwb_fingerprint(nwb_path, nwbfile)
        if canonical_json(observed) != canonical_json(session["nwb_fingerprint"]):
            raise ValueError("Source NWB changed after full Figure 1 computation.")
        catalog = catalog_augmented_nwb(
            nwbfile,
            nwb_file_name=str(session["nwb_file_name"]),
        )
        epoch = str(session["epochs"][0])
        intervals = {}
        for trajectory_type in TRAJECTORY_TYPES:
            row = _one_record(
                catalog["trajectory_intervals"],
                label="NWB trajectory interval",
                epoch=epoch,
                trajectory_type=trajectory_type,
            )
            intervals[trajectory_type] = load_interval_set(nwbfile, row)
    return intervals


def _decoding_tables(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt de novo decoding bundles to sample- and lap-level tables."""
    absolute_tables = []
    trial_tables = []
    for session in sessions:
        record = _one_record(
            session["artifacts"].get("path_progression_decoding", ()),
            label="PathProgressionDecoding artifact",
        )
        if str(record.get("artifact_origin")) != "computed":
            raise ValueError("PathProgressionDecoding must be computed de novo.")
        manifest_path = resolve_run_path(
            record["artifact_manifest_path"],
            run_dir=run_dir,
        )
        loaded = load_figure_1_decoding_payload(
            artifact_manifest_path=manifest_path,
            trajectory_intervals=_trajectory_intervals_from_nwb(session),
        )
        _require_session_identity(
            loaded["bundle"]["metadata"],
            session,
            label="PathProgressionDecoding artifact",
        )
        absolute_tables.append(loaded["absolute_error_table"])
        trial_tables.append(loaded["trial_error_table"])
    return (
        pd.concat(absolute_tables, ignore_index=True).loc[
            :, list(legacy.DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS)
        ],
        pd.concat(trial_tables, ignore_index=True).loc[
            :, list(legacy.DECODING_TRIAL_ERROR_TABLE_COLUMNS)
        ],
    )


def load_full_figure_1_payload(
    *,
    run_id: str,
    mode: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    regions: Sequence[str] = legacy.DEFAULT_REGIONS,
) -> dict[str, Any]:
    """Load all new artifacts needed by the unchanged Figure 1 layout."""
    from v1ca1.paper_figures.figure_1_spyglass import load_figure_1d_payload

    run_dir, campaign, unordered_sessions = load_full_figure_campaign(
        run_id,
        scratch_root=scratch_root,
    )
    if campaign.get("analysis_parameters", {}).get("pipeline") != (
        FULL_FIGURE_PIPELINE
    ):
        raise ValueError("Selected campaign is not a full Figure 1 run.")
    parent_snapshot = campaign["analysis_parameters"]["parent_figure_1d"]
    expected_configuration = build_full_figure_configuration(parent_snapshot)
    if canonical_json(campaign["analysis_parameters"]) != canonical_json(
        expected_configuration
    ):
        raise ValueError("Full Figure 1 campaign parameters are not canonical.")
    sessions, datasets = _ordered_sessions(unordered_sessions, mode=mode)
    regions = tuple(dict.fromkeys(str(region) for region in regions))
    if not regions or any(not region for region in regions):
        raise ValueError("regions must contain at least one non-empty name.")
    parent_payload = load_figure_1d_payload(
        scratch_root=scratch_root,
        run_id=str(parent_snapshot["run_id"]),
        mode=mode,
        regions=regions,
    )
    panel_b, panel_c = _load_examples(
        run_dir,
        sessions,
        datasets=datasets,
    )
    decoding_absolute, decoding_trial = _decoding_tables(run_dir, sessions)
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "mode": mode,
        "sessions": sessions,
        "datasets": datasets,
        "regions": regions,
        "panel_b_example": panel_b,
        "panel_c_examples": panel_c,
        "panel_d_payload": parent_payload,
        "motor_delta_table": _motor_delta_table(run_dir, sessions),
        "encoding_delta_table": _encoding_delta_table(run_dir, sessions),
        "decoding_absolute_error_table": decoding_absolute,
        "decoding_trial_error_table": decoding_trial,
    }


def _requested_datasets(
    kwargs: Mapping[str, Any],
    expected: Sequence[tuple[str, str, str]],
) -> tuple[tuple[str, str, str], ...]:
    """Normalize and validate a legacy loader's requested data sets."""
    observed = tuple(normalize_dataset_id(value) for value in kwargs["datasets"])
    if observed != tuple(expected):
        raise ValueError("Figure layout requested unexpected data sets.")
    return observed


@contextmanager
def _offline_legacy_sources(payload: Mapping[str, Any]):
    """Temporarily inject artifact-backed loaders into the legacy layout."""
    originals = {}
    selected_datasets = tuple(payload["datasets"])
    selected_animals = tuple(dataset[0] for dataset in selected_datasets)

    def replace(name: str, value: Any) -> None:
        originals[name] = getattr(legacy, name)
        setattr(legacy, name, value)

    def panel_b_loader(**kwargs: Any) -> dict[str, Any]:
        expected = payload["panel_b_example"]
        for name in ("animal_name", "date", "region"):
            if str(kwargs[name]) != str(expected[name]):
                raise ValueError(f"Unexpected Panel B {name} selection.")
        if str(kwargs["unit_id"]) != str(expected["unit_id"]):
            raise ValueError("Unexpected Panel B unit selection.")
        return expected

    panel_c_by_key = {
        (
            str(row["animal_name"]),
            str(row["date"]),
            str(row["epoch"]),
            str(row["region"]),
            str(row["unit_id"]),
        ): row
        for row in payload["panel_c_examples"]
    }
    selected_panel_c_specs = tuple(
        (
            row["animal_name"],
            row["date"],
            row["epoch"],
            row["region"],
            row["unit_id"],
        )
        for row in payload["panel_c_examples"]
    )

    def panel_c_loader(**kwargs: Any) -> dict[str, Any]:
        key = tuple(
            str(kwargs[name])
            for name in ("animal_name", "date", "epoch", "region", "unit_id")
        )
        if key not in panel_c_by_key:
            raise ValueError(f"Unexpected Panel C example selection {key!r}.")
        return panel_c_by_key[key]

    def panel_d_loader(**kwargs: Any) -> Any:
        _requested_datasets(kwargs, selected_datasets)
        region = str(kwargs["region"])
        try:
            return payload["panel_d_payload"]["panels_by_region"][region]
        except KeyError as exc:
            raise ValueError(f"Unexpected Panel D region {region!r}.") from exc

    def motor_loader(**kwargs: Any) -> pd.DataFrame:
        _requested_datasets(kwargs, selected_datasets)
        if str(kwargs.get("region", legacy.MOTOR_DELTA_REGION)) != (
            legacy.MOTOR_DELTA_REGION
        ):
            raise ValueError("Unexpected MotorEncoding region.")
        return payload["motor_delta_table"]

    def encoding_loader(**kwargs: Any) -> pd.DataFrame:
        _requested_datasets(kwargs, selected_datasets)
        return payload["encoding_delta_table"]

    def decoding_loader(**kwargs: Any) -> pd.DataFrame:
        _requested_datasets(kwargs, selected_datasets)
        return payload["decoding_absolute_error_table"]

    def decoding_trials_loader(**kwargs: Any) -> pd.DataFrame:
        _requested_datasets(kwargs, selected_datasets)
        return payload["decoding_trial_error_table"]

    original_bracket_builder = legacy.build_decoding_significance_brackets

    def bracket_builder(per_animal_results: Any, **kwargs: Any) -> Any:
        if "animal_names" in kwargs:
            raise ValueError("Artifact-backed renderer owns animal selection.")
        return original_bracket_builder(
            per_animal_results,
            animal_names=selected_animals,
            **kwargs,
        )

    replace("load_panel_b_visual_example_data", panel_b_loader)
    replace("PANEL_E_EXAMPLES", selected_panel_c_specs)
    replace("load_or_compute_panel_e_example_data", panel_c_loader)
    replace("load_or_compute_panel_d_heatmap_panels", panel_d_loader)
    replace("load_motor_delta_table", motor_loader)
    replace("load_encoding_delta_table", encoding_loader)
    replace("load_decoding_absolute_error_table", decoding_loader)
    replace("build_decoding_trial_error_table", decoding_trials_loader)
    replace("PANEL_H_DECODING_ANIMALS", selected_animals)
    replace("build_decoding_significance_brackets", bracket_builder)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(legacy, name, value)


def get_full_figure_output_path(
    *,
    run_dir: Path,
    mode: str,
    output_format: str,
) -> Path:
    """Return the canonical run-local complete Figure 1 output path."""
    if output_format not in legacy.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {legacy.FIGURE_FORMATS!r}."
        )
    if mode == "l14-validation":
        output_name = DEFAULT_L14_FIGURE_OUTPUT_NAME
    elif mode == "full":
        output_name = DEFAULT_FULL_FIGURE_OUTPUT_NAME
    else:
        raise ValueError(f"mode must be one of {FIGURE_MODES!r}.")
    return (
        Path(run_dir)
        / "figures"
        / f"{output_name}.{output_format}"
    )


def get_full_figure_provenance_path(figure_path: Path) -> Path:
    """Return the provenance sidecar path for one complete Figure 1 artifact."""
    figure_path = Path(figure_path)
    return figure_path.with_name(
        f"{figure_path.name}{FULL_FIGURE_PROVENANCE_SUFFIX}"
    )


def _validated_campaign_manifest(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Return the on-disk campaign manifest matching the loaded payload."""
    campaign = payload.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("Complete Figure 1 payload has no campaign snapshot.")
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if canonical_json(manifest) != canonical_json(dict(campaign)):
        raise ValueError(
            "Complete Figure 1 payload and campaign manifest disagree."
        )
    return manifest_path, manifest


def _build_full_figure_provenance(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Build one checksum-bearing receipt for a run-local Figure 1 render."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=True)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Complete Figure 1 output must remain inside its run.")
    manifest_path, campaign = _validated_campaign_manifest(
        payload,
        run_dir=run_dir,
    )
    return {
        "schema_version": FULL_FIGURE_PROVENANCE_SCHEMA_VERSION,
        "artifact_kind": "complete_spyglass_figure_1",
        "created_at_utc": utc_now(),
        "run_id": str(campaign["run_id"]),
        "mode": str(payload["mode"]),
        "datasets": [list(dataset) for dataset in payload["datasets"]],
        "regions": [str(region) for region in payload["regions"]],
        "campaign_manifest": {
            "run_relative_path": relative_run_path(
                manifest_path,
                run_dir=run_dir,
            ),
            "sha256": file_sha256(manifest_path),
        },
        "figure": {
            "run_relative_path": relative_run_path(
                output_path,
                run_dir=run_dir,
            ),
            "format": output_path.suffix.lower().lstrip("."),
            "size_bytes": int(output_path.stat().st_size),
            "sha256": file_sha256(output_path),
        },
        "render_code_provenance": code_provenance(),
    }


def _load_validated_full_figure_provenance(
    payload: Mapping[str, Any],
    *,
    figure_path: Path,
) -> dict[str, Any]:
    """Load and validate the receipt for one run-local Figure 1 artifact."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    figure_path = Path(figure_path).resolve(strict=True)
    if not figure_path.is_relative_to(run_dir):
        raise ValueError("Promoted Figure 1 source must remain inside its run.")
    provenance_path = get_full_figure_provenance_path(figure_path)
    provenance = load_json(provenance_path)
    if provenance.get("schema_version") != (
        FULL_FIGURE_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported complete Figure 1 provenance schema.")
    if provenance.get("artifact_kind") != "complete_spyglass_figure_1":
        raise ValueError("Provenance sidecar is not for complete Figure 1.")

    manifest_path, campaign = _validated_campaign_manifest(
        payload,
        run_dir=run_dir,
    )
    expected_fields = {
        "run_id": str(campaign["run_id"]),
        "mode": str(payload["mode"]),
    }
    for field, expected in expected_fields.items():
        if str(provenance.get(field)) != expected:
            raise ValueError(f"Figure provenance {field!r} does not match.")
    manifest_record = provenance.get("campaign_manifest")
    figure_record = provenance.get("figure")
    if not isinstance(manifest_record, Mapping) or not isinstance(
        figure_record,
        Mapping,
    ):
        raise ValueError("Figure provenance is missing artifact records.")
    if str(manifest_record.get("run_relative_path")) != relative_run_path(
        manifest_path,
        run_dir=run_dir,
    ) or str(manifest_record.get("sha256")) != file_sha256(manifest_path):
        raise ValueError("Campaign manifest changed after Figure 1 rendering.")
    if str(figure_record.get("run_relative_path")) != relative_run_path(
        figure_path,
        run_dir=run_dir,
    ):
        raise ValueError("Figure provenance points to a different artifact.")
    if str(figure_record.get("format")) != figure_path.suffix.lower().lstrip(
        "."
    ):
        raise ValueError("Figure provenance has the wrong output format.")
    if int(figure_record.get("size_bytes", -1)) != figure_path.stat().st_size:
        raise ValueError("Figure 1 size no longer matches its provenance.")
    if str(figure_record.get("sha256")) != file_sha256(figure_path):
        raise ValueError("Figure 1 checksum no longer matches its provenance.")
    return provenance


def promote_full_figure_1(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Atomically publish one validated run-local Figure 1 with a receipt."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    source_path = Path(source_path).resolve(strict=True)
    destination_path = Path(destination_path).resolve(strict=False)
    if destination_path.is_relative_to(run_dir):
        raise ValueError("Figure 1 promotion destination must be outside its run.")
    if source_path.suffix.lower() != destination_path.suffix.lower():
        raise ValueError("Promoted Figure 1 must retain its source format.")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(
            "Figure 1 promotion destination directory does not exist: "
            f"{destination_path.parent}"
        )
    provenance = _load_validated_full_figure_provenance(
        payload,
        figure_path=source_path,
    )
    destination_provenance_path = get_full_figure_provenance_path(
        destination_path
    )
    existing = [
        path
        for path in (destination_path, destination_provenance_path)
        if path.exists()
    ]
    if existing and not replace:
        raise FileExistsError(
            "Refusing to replace promoted Figure 1 artifact(s): "
            + ", ".join(str(path) for path in existing)
        )

    token = uuid.uuid4().hex
    staged_figure = destination_path.with_name(
        f".{destination_path.name}.{token}.tmp"
    )
    staged_provenance = destination_provenance_path.with_name(
        f".{destination_provenance_path.name}.{token}.tmp"
    )
    try:
        shutil.copyfile(source_path, staged_figure)
        source_sha256 = str(provenance["figure"]["sha256"])
        if file_sha256(staged_figure) != source_sha256:
            raise ValueError("Staged Figure 1 checksum differs from its source.")
        published_provenance = {
            **provenance,
            "promotion": {
                "promoted_at_utc": utc_now(),
                "destination_path": str(destination_path),
                "sha256": source_sha256,
                "promotion_code_provenance": code_provenance(),
            },
        }
        staged_provenance.write_text(
            json.dumps(
                json.loads(canonical_json(published_provenance)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if replace:
            os.replace(staged_figure, destination_path)
            os.replace(staged_provenance, destination_provenance_path)
        else:
            os.link(staged_figure, destination_path)
            staged_figure.unlink()
            os.link(staged_provenance, destination_provenance_path)
            staged_provenance.unlink()
    finally:
        staged_figure.unlink(missing_ok=True)
        staged_provenance.unlink(missing_ok=True)
    if file_sha256(destination_path) != str(provenance["figure"]["sha256"]):
        raise ValueError("Promoted Figure 1 checksum verification failed.")
    return destination_path


def render_full_figure_1(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    asset_dir: Path = legacy.DEFAULT_ASSET_DIR,
    dpi: int = 300,
) -> Path:
    """Render the complete legacy layout using only new analysis artifacts."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Complete Figure 1 output must remain inside its run.")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite complete Figure 1 output: {output_path}"
        )
    provenance_path = get_full_figure_provenance_path(output_path)
    if provenance_path.exists():
        raise FileExistsError(
            "Refusing to overwrite complete Figure 1 provenance: "
            f"{provenance_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in legacy.FIGURE_FORMATS:
        raise ValueError(
            f"Figure suffix must be one of {legacy.FIGURE_FORMATS!r}."
        )
    with _offline_legacy_sources(payload):
        rendered = legacy.make_figure_1(
            data_root=run_dir,
            asset_dir=Path(asset_dir),
            output_path=output_path,
            datasets=payload["datasets"],
            regions=payload["regions"],
            position_bin_count=legacy.DEFAULT_POSITION_BIN_COUNT,
            position_offset=legacy.DEFAULT_POSITION_OFFSET,
            speed_threshold_cm_s=legacy.DEFAULT_SPEED_THRESHOLD_CM_S,
            sigma_bins=legacy.DEFAULT_SIGMA_BINS,
            encoding_bin_size_s=legacy.ENCODING_COMPARISON_BIN_SIZE_S,
            encoding_place_bin_size_cm=(
                legacy.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
            ),
            dpi=int(dpi),
            decoding_n_permutations=legacy.DECODING_PERMUTATION_COUNT,
            decoding_permutation_seed=legacy.DECODING_PERMUTATION_SEED,
            panel_d_cache_dir=run_dir / "figures" / "cache",
            panel_e_cache_dir=run_dir / "figures" / "cache",
            panel_dark_light_example_cache_dir=(
                run_dir / "figures" / "cache"
            ),
        )
    if Path(rendered).resolve(strict=True) != output_path:
        raise ValueError("Figure renderer returned an unexpected output path.")
    try:
        write_json_once(
            _build_full_figure_provenance(
                payload,
                output_path=output_path,
            ),
            provenance_path,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


__all__ = [
    "DEFAULT_FULL_FIGURE_OUTPUT_NAME",
    "DEFAULT_L14_FIGURE_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "FULL_FIGURE_PROVENANCE_SCHEMA_VERSION",
    "get_full_figure_output_path",
    "get_full_figure_provenance_path",
    "load_full_figure_1_payload",
    "promote_full_figure_1",
    "render_full_figure_1",
]
