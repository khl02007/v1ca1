"""Render Figure 2 from a completed database-free Spyglass campaign."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures import _dark_light
from v1ca1.paper_figures import figure_2 as legacy
from v1ca1.paper_figures.datasets import get_processed_datasets, normalize_dataset_id
from v1ca1.spyglass import (
    movement,
    path_progression_decoding,
    path_specific_decoding,
    swap_glm,
    tuning_similarity,
)
from v1ca1.spyglass.nwb import (
    catalog_augmented_nwb,
    load_interval_set,
    load_wtrack_graph,
)
from v1ca1.spyglass.offline.figure_1_examples import load_example_payload
from v1ca1.spyglass.offline.manifests import (
    DEFAULT_SCRATCH_ROOT,
    get_run_dir,
    nwb_fingerprint,
    resolve_run_path,
)
from v1ca1.spyglass.offline.sources import validate_nwb_session_identity
from v1ca1.spyglass.selection import canonical_json

DEFAULT_OUTPUT_NAME = "figure_2_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = tuple(get_processed_datasets())
LIGHT_EPOCH = "02_r1"
HELDOUT_LIGHT_EPOCH = "06_r3"
REGION = "v1"
MINIMUM_MOVEMENT_FIRING_RATE_HZ = 0.5
MINIMUM_STABILITY_CORRELATION = 0.5
FORWARD_SWAP_MODEL = "task_segment_scalar"


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: Any,
) -> dict[str, Any]:
    """Return exactly one record matching direct or selection fields."""
    matches = []
    for raw_record in records:
        record = dict(raw_record)
        selection = record.get("selection", {})
        if all(
            str(record.get(name, selection.get(name))) == str(value)
            for name, value in selectors.items()
        ):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {selectors!r}; " f"found {len(matches)}."
        )
    return matches[0]


def _record_artifact_path(
    record: Mapping[str, Any],
    field: str,
    *,
    run_dir: Path,
) -> Path:
    """Resolve a validated nested or original offline artifact pointer."""
    artifacts = record.get("artifacts", {})
    if isinstance(artifacts, Mapping) and field in artifacts:
        value = artifacts[field]
        if not isinstance(value, Mapping) or "relative_path" not in value:
            raise ValueError(f"Artifact field {field!r} is malformed.")
        relative_path = value["relative_path"]
    elif field in record:
        relative_path = record[field]
    else:
        raise ValueError(f"Artifact record is missing {field!r}.")
    return resolve_run_path(str(relative_path), run_dir=run_dir)


def _ordered_sessions(
    sessions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exactly the four manuscript sessions in canonical order."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_session in sessions:
        session = dict(raw_session)
        key = (str(session.get("animal_name")), str(session.get("date")))
        if key in by_key:
            raise ValueError(f"Figure 2 campaign duplicates session {key!r}.")
        by_key[key] = session
    expected_keys = {
        (animal_name, date) for animal_name, date, _dark_epoch in EXPECTED_DATASETS
    }
    if set(by_key) != expected_keys:
        raise ValueError(
            "Figure 2 requires exactly the four manuscript sessions; "
            f"expected {sorted(expected_keys)!r}, got {sorted(by_key)!r}."
        )

    ordered = []
    for animal_name, date, dark_epoch in EXPECTED_DATASETS:
        session = by_key[(animal_name, date)]
        epochs = session.get("epochs")
        expected_epochs = {
            "dark": str(dark_epoch),
            "AB": LIGHT_EPOCH,
            "BA": HELDOUT_LIGHT_EPOCH,
        }
        if (
            not isinstance(epochs, Mapping)
            or {str(name): str(value) for name, value in epochs.items()}
            != expected_epochs
        ):
            raise ValueError(
                f"Session {animal_name} {date} has noncanonical epochs "
                f"{epochs!r}; expected {expected_epochs!r}."
            )
        ordered.append(session)
    return ordered


def _session_dark_epoch(session: Mapping[str, Any]) -> str:
    """Return the declared dark epoch for one Figure 2 session."""
    return str(session["epochs"]["dark"])


def _parent_run_dir(
    session: Mapping[str, Any],
    *,
    parent: str,
    scratch_root: Path,
) -> Path:
    """Return one hash-pinned parent campaign directory."""
    parent_artifacts = session.get("parent_artifacts", {})
    field = f"{parent}_run_id"
    if not isinstance(parent_artifacts, Mapping) or field not in parent_artifacts:
        raise ValueError(f"Session parent_artifacts is missing {field!r}.")
    return get_run_dir(str(parent_artifacts[field]), scratch_root=scratch_root)


def _identity_columns(table: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Return a copy after requiring persistent unit identity columns."""
    required = {
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "group_unit_id",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{label} is missing identity columns {missing!r}.")
    output = table.copy()
    for column in required:
        output[column] = output[column].astype(str)
    return output


def _load_movement_table(
    session: Mapping[str, Any],
    *,
    epoch: str,
    run_dir: Path,
    scratch_root: Path,
) -> pd.DataFrame:
    """Load one epoch-wide movement-rate table from this or its dark parent."""
    if str(epoch) == _session_dark_epoch(session):
        record = dict(session["parent_artifacts"]["dark_movement_firing_rate"])
        source_root = _parent_run_dir(
            session,
            parent="initial",
            scratch_root=scratch_root,
        )
    else:
        record = _one_record(
            session["artifacts"].get("movement_firing_rate", ()),
            label="MovementFiringRate artifact",
            epoch=epoch,
            region=REGION,
        )
        source_root = run_dir
    path = _record_artifact_path(record, "firing_rate_path", run_dir=source_root)
    table = movement.load_movement_firing_rate_artifact(path)
    return _identity_columns(table, label="MovementFiringRate artifact")


def _load_stability_tables(
    session: Mapping[str, Any],
    *,
    epoch: str,
    run_dir: Path,
    scratch_root: Path,
) -> list[pd.DataFrame]:
    """Load the four fixed trajectory stability tables for one epoch."""
    if str(epoch) == _session_dark_epoch(session):
        records = session["parent_artifacts"]["dark_stability"]
        source_root = _parent_run_dir(
            session,
            parent="initial",
            scratch_root=scratch_root,
        )
    else:
        records = session["artifacts"].get(
            "path_specific_place_stability",
            (),
        )
        source_root = run_dir
    tables = []
    for trajectory_type in TRAJECTORY_TYPES:
        record = _one_record(
            records,
            label="PathSpecificPlaceStability artifact",
            epoch=epoch,
            region=REGION,
            trajectory_type=trajectory_type,
        )
        path = _record_artifact_path(record, "stability_path", run_dir=source_root)
        table = pd.read_parquet(path)
        required = {
            "stable_unit_id",
            "unit_id",
            "trajectory_type",
            "stability_correlation",
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(
                f"Stability artifact is missing columns {missing!r}: {path}"
            )
        table = _identity_columns(table, label="Stability artifact")
        if not table.empty and set(table["trajectory_type"].astype(str)) != {
            trajectory_type
        }:
            raise ValueError("Stability artifact trajectory identity is stale.")
        tables.append(table)
    return tables


def _eligible_units(
    movement_table: pd.DataFrame,
    stability_tables: Sequence[pd.DataFrame],
    *,
    minimum_movement_firing_rate_hz: float = MINIMUM_MOVEMENT_FIRING_RATE_HZ,
    minimum_stability_correlation: float = MINIMUM_STABILITY_CORRELATION,
) -> set[str]:
    """Return units passing epoch-wide movement rate and any-path stability."""
    rates = pd.to_numeric(
        movement_table["movement_firing_rate_hz"],
        errors="coerce",
    ).to_numpy(dtype=float)
    active = set(
        movement_table.loc[
            np.isfinite(rates) & (rates >= float(minimum_movement_firing_rate_hz)),
            "stable_unit_id",
        ].astype(str)
    )
    if not stability_tables:
        return set()
    stability = pd.concat(stability_tables, ignore_index=True)
    correlations = pd.to_numeric(
        stability["stability_correlation"],
        errors="coerce",
    ).to_numpy(dtype=float)
    stable = set(
        stability.loc[
            np.isfinite(correlations)
            & (correlations >= float(minimum_stability_correlation)),
            "stable_unit_id",
        ].astype(str)
    )
    return active.intersection(stable)


def _load_similarity_tables(
    session: Mapping[str, Any],
    *,
    run_dir: Path,
) -> dict[str, tuple[pd.DataFrame, Path]]:
    """Load the all-unit absolute-overlap tables keyed by epoch."""
    output: dict[str, tuple[pd.DataFrame, Path]] = {}
    expected_animal_name = str(session["animal_name"])
    expected_date = str(session["date"])
    records = session["artifacts"].get(
        "path_specific_place_tuning_similarity",
        (),
    )
    for record in records:
        if str(record.get("artifact_origin", "")) != "computed":
            raise ValueError("Figure 2 similarity must be computed de novo.")
        path = _record_artifact_path(record, "similarity_path", run_dir=run_dir)
        table = tuning_similarity.load_tuning_similarity_artifact(path)
        if not table.empty:
            animal_names = set(table["animal_name"].astype(str))
            dates = set(table["date"].astype(str))
            epochs = set(table["epoch"].astype(str))
            regions = set(table["region"].astype(str))
            metrics = set(table["similarity_metric"].astype(str))
            if (
                animal_names != {expected_animal_name}
                or dates != {expected_date}
                or len(epochs) != 1
                or regions != {REGION}
                or metrics != {"absolute_overlap"}
            ):
                raise ValueError("Tuning-similarity artifact identity is stale.")
            epoch = next(iter(epochs))
        else:
            epoch = str(
                record.get(
                    "epoch",
                    record.get("selection", {}).get("epoch", ""),
                )
            )
            if not epoch:
                raise ValueError("Cannot identify an empty similarity artifact epoch.")
        if epoch in output:
            raise ValueError(f"Duplicate similarity artifact for epoch {epoch!r}.")
        output[epoch] = (_identity_columns(table, label="Similarity artifact"), path)
    return output


def _build_panel_b_overlap_table(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> pd.DataFrame:
    """Build explicitly filtered, paired light/dark overlap rows."""
    tables = []
    expected_labels = set(_dark_light.PANEL_C_SIMILARITY_COMPARISON_LABELS)
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        dark_epoch = _session_dark_epoch(session)
        similarity_by_epoch = _load_similarity_tables(session, run_dir=run_dir)
        eligible_by_epoch = {}
        for epoch in (LIGHT_EPOCH, dark_epoch):
            movement_table = _load_movement_table(
                session,
                epoch=epoch,
                run_dir=run_dir,
                scratch_root=scratch_root,
            )
            stability_tables = _load_stability_tables(
                session,
                epoch=epoch,
                run_dir=run_dir,
                scratch_root=scratch_root,
            )
            eligible_by_epoch[epoch] = _eligible_units(
                movement_table,
                stability_tables,
            )
        shared_units = eligible_by_epoch[LIGHT_EPOCH].intersection(
            eligible_by_epoch[dark_epoch]
        )

        for epoch_type, epoch in (("light", LIGHT_EPOCH), ("dark", dark_epoch)):
            if epoch not in similarity_by_epoch:
                raise ValueError(
                    f"Missing absolute-overlap artifact for {animal_name} "
                    f"{date} {epoch}."
                )
            table, path = similarity_by_epoch[epoch]
            values = pd.to_numeric(table["similarity"], errors="coerce").to_numpy(
                dtype=float
            )
            keep = (
                table["stable_unit_id"].astype(str).isin(shared_units)
                & table["comparison_label"].astype(str).isin(expected_labels)
                & np.isfinite(values)
            )
            if "similarity_status" in table:
                keep &= table["similarity_status"].astype(str).eq("valid")
            selected = table.loc[keep].copy()
            numeric_units = pd.to_numeric(selected["unit_id"], errors="coerce")
            if numeric_units.isna().any():
                raise ValueError("Figure 2 requires numeric NWB sorting unit IDs.")
            selected = selected.assign(
                epoch_type=epoch_type,
                unit=numeric_units.astype(int),
                source_path=str(path),
            )
            tables.append(
                selected.loc[
                    :,
                    [
                        "animal_name",
                        "date",
                        "epoch_type",
                        "epoch",
                        "unit",
                        "comparison_label",
                        "similarity",
                        "source_path",
                    ],
                ]
            )
    if not tables:
        return pd.DataFrame(
            columns=[
                "animal_name",
                "date",
                "unit",
                "comparison_label",
                "similarity_light",
                "similarity_dark",
            ]
        )
    return _dark_light.build_panel_c_similarity_pairs(
        pd.concat(tables, ignore_index=True)
    )


def _load_panel_a_examples(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Load four cells, each with computed dark and AB payloads."""
    session_by_key = {
        (str(session["animal_name"]), str(session["date"])): session
        for session in sessions
    }
    records = [
        dict(record)
        for session in sessions
        for record in session["artifacts"].get("figure_examples", ())
    ]
    examples = []
    for (
        animal_name,
        date,
        region,
        unit_id,
        trajectories,
    ) in legacy._figure_2.FIGURE_2_PANEL_A_EXAMPLES:
        session = session_by_key[(str(animal_name), str(date))]
        epoch_rates = {}
        for epoch_type, epoch in (
            ("dark", _session_dark_epoch(session)),
            ("light", LIGHT_EPOCH),
        ):
            record = _one_record(
                records,
                label="Figure 2 example payload",
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                region=region,
                sorting_unit_id=unit_id,
            )
            if str(record.get("artifact_origin", "")) != "computed":
                raise ValueError("Figure 2 examples must be computed de novo.")
            path = resolve_run_path(record["payload_path"], run_dir=run_dir)
            loaded = load_example_payload(
                path,
                expected_sha256=str(record["artifact_sha256"]),
            )
            metadata = loaded["metadata"]
            expected_identity = {
                "animal_name": str(animal_name),
                "date": str(date),
                "epoch": str(epoch),
                "region": str(region),
                "sorting_unit_id": str(unit_id),
            }
            observed_identity = {
                name: str(metadata.get(name)) for name in expected_identity
            }
            if observed_identity != expected_identity:
                raise ValueError("Figure 2 example manifest and payload disagree.")
            if tuple(metadata["trajectory_types"]) != tuple(trajectories):
                raise ValueError("Figure 2 example trajectories are noncanonical.")
            epoch_rates[epoch_type] = {
                "animal_name": str(animal_name),
                "date": str(date),
                "epoch": str(epoch),
                "region": str(region),
                "unit_id": int(unit_id),
                "raster_positions": loaded["raster_positions"],
                "firing_rates": loaded["firing_rates"],
            }
        examples.append(
            {
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "unit_id": int(unit_id),
                "trajectories": tuple(trajectories),
                "epoch_rates": epoch_rates,
            }
        )
    return examples


def _load_nwb_sorting_unit_map(
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Map persistent NWB unit IDs to declared sorting IDs read-only."""
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
            raise ValueError("Source NWB changed after Figure 2 computation.")
        units = getattr(nwbfile, "units", None)
        if units is None:
            raise ValueError("Augmented NWB has no Units table.")
        table = units.to_dataframe()
    missing = sorted({"region", "sorting_unit_id"}.difference(table.columns))
    if missing:
        raise ValueError(f"Augmented NWB Units table is missing {missing!r}.")
    selected = table.loc[
        table["region"].astype(str).str.strip().str.casefold().eq(REGION)
    ]
    if selected.index.duplicated().any():
        raise ValueError("Augmented NWB contains duplicate persistent unit IDs.")
    mapping = {
        str(nwb_unit_id): (
            sorting_unit_id.item()
            if isinstance(sorting_unit_id, np.generic)
            else sorting_unit_id
        )
        for nwb_unit_id, sorting_unit_id in selected["sorting_unit_id"].items()
    }
    sorting_ids = [str(value) for value in mapping.values()]
    if any(not value for value in sorting_ids) or len(set(sorting_ids)) != len(
        sorting_ids
    ):
        raise ValueError("V1 sorting_unit_id values must be unique and non-empty.")
    return mapping


def _numeric_unit_coordinate(
    result: Mapping[str, Any],
    *,
    sorting_unit_by_nwb_id: Mapping[str, Any],
) -> tuple[Any, np.ndarray]:
    """Use sorting IDs for examples while retaining NWB IDs for joins."""
    selected = _identity_columns(
        result["selected_units"],
        label="SwapGLM selected_units",
    )
    dataset = result["dataset"]
    group_ids = np.asarray(dataset.coords["unit"].values).astype(str)
    if not np.array_equal(group_ids, selected["group_unit_id"].to_numpy(dtype=str)):
        raise ValueError("SwapGLM dataset and selected-unit audit disagree.")
    nwb_unit_ids = selected["unit_id"].astype(str).to_numpy()
    missing = sorted(set(nwb_unit_ids).difference(sorting_unit_by_nwb_id))
    if missing:
        raise ValueError(
            "SwapGLM selected units are absent from the NWB unit map: " f"{missing!r}."
        )
    sorting_ids = pd.to_numeric(
        [sorting_unit_by_nwb_id[value] for value in nwb_unit_ids],
        errors="coerce",
    )
    if np.any(pd.isna(sorting_ids)) or len(set(sorting_ids)) != len(sorting_ids):
        raise ValueError("Figure 2 requires numeric NWB sorting unit IDs.")
    numeric_nwb_ids = pd.to_numeric(nwb_unit_ids, errors="coerce")
    if np.any(pd.isna(numeric_nwb_ids)):
        raise ValueError("Figure 2 aggregate rows require numeric NWB unit IDs.")
    return (
        dataset.assign_coords(unit=np.asarray(sorting_ids, dtype=int)),
        np.asarray(numeric_nwb_ids, dtype=int),
    )


def _build_panel_d_swap_payload(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Build the forward-only SwapGLM histogram and fixed examples."""
    loaded_by_session: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        record = _one_record(
            session["artifacts"].get("swap_glm", ()),
            label="forward SwapGLM artifact",
        )
        if str(record.get("artifact_origin", "")) != "computed":
            raise ValueError("Figure 2 SwapGLM must be computed de novo.")
        manifest_path = _record_artifact_path(
            record,
            "artifact_manifest_path",
            run_dir=run_dir,
        )
        result = swap_glm.load_swap_glm_artifact(manifest_path.parent)
        metadata = result["metadata"]
        expected = {
            "animal_name": animal_name,
            "date": date,
            "region": REGION,
            "dark_epoch": _session_dark_epoch(session),
            "light_train_epoch": LIGHT_EPOCH,
            "light_test_epoch": HELDOUT_LIGHT_EPOCH,
        }
        if any(str(metadata.get(name)) != value for name, value in expected.items()):
            raise ValueError("SwapGLM metadata does not describe the forward transfer.")
        if str(result.get("artifact_origin", "")) != "computed":
            raise ValueError("Loaded SwapGLM is not a de novo artifact.")
        dataset, unit_ids = _numeric_unit_coordinate(
            result,
            sorting_unit_by_nwb_id=_load_nwb_sorting_unit_map(session),
        )
        selected = result["selected_units"].copy()
        selected["stable_unit_id"] = selected["stable_unit_id"].astype(str)
        dark_movement = _load_movement_table(
            session,
            epoch=expected["dark_epoch"],
            run_dir=run_dir,
            scratch_root=scratch_root,
        )
        dark_stability = _load_stability_tables(
            session,
            epoch=expected["dark_epoch"],
            run_dir=run_dir,
            scratch_root=scratch_root,
        )
        eligible = _eligible_units(dark_movement, dark_stability)
        unit_mask = selected["stable_unit_id"].astype(str).isin(eligible).to_numpy()
        if swap_glm.PRIMARY_METRIC not in dataset:
            raise ValueError("SwapGLM result is missing its primary score.")
        delta = np.asarray(
            dataset[swap_glm.PRIMARY_METRIC].sel(model=FORWARD_SWAP_MODEL).values,
            dtype=float,
        )
        trajectories = np.asarray(dataset.coords["trajectory"].values).astype(str)
        for trajectory_index, trajectory_type in enumerate(trajectories):
            for unit_index, unit_id in enumerate(unit_ids):
                value = float(delta[trajectory_index, unit_index])
                if not unit_mask[unit_index] or not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "region": REGION,
                        "dark_epoch": expected["dark_epoch"],
                        "light_train_epoch": LIGHT_EPOCH,
                        "light_test_epoch": HELDOUT_LIGHT_EPOCH,
                        "model_name": FORWARD_SWAP_MODEL,
                        "trajectory": trajectory_type,
                        "unit": int(unit_id),
                        "delta_ll_bits_per_spike": value,
                        "source_path": str(manifest_path),
                    }
                )
        loaded_by_session[(animal_name, date)] = {
            "dataset": dataset,
            "manifest_path": manifest_path,
            "metadata": metadata,
        }

    examples = []
    for (
        animal_name,
        date,
        region,
        unit_id,
        trajectory,
    ) in legacy._figure_2.PANEL_C_SWAP_EXAMPLES:
        if str(region) != REGION:
            continue
        loaded = loaded_by_session[(str(animal_name), str(date))]
        dataset = loaded["dataset"]
        trajectories = np.asarray(dataset.coords["trajectory"].values).astype(str)
        units = np.asarray(dataset.coords["unit"].values, dtype=int)
        if str(trajectory) not in set(trajectories) or int(unit_id) not in set(units):
            raise ValueError(
                "Configured Figure 2 SwapGLM example is absent: "
                f"{animal_name} {unit_id} {trajectory}."
            )
        example = _dark_light._panel_h_swap_example_from_indices(
            dataset,
            animal_name=str(animal_name),
            date=str(date),
            region=REGION,
            dark_epoch=str(loaded["metadata"]["dark_epoch"]),
            light_train_epoch=LIGHT_EPOCH,
            light_test_epoch=HELDOUT_LIGHT_EPOCH,
            source_path=loaded["manifest_path"],
            trajectory_index=int(np.flatnonzero(trajectories == str(trajectory))[0]),
            unit_index=int(np.flatnonzero(units == int(unit_id))[0]),
            model_name=FORWARD_SWAP_MODEL,
        )
        examples.append(example)
    return {
        "swap_delta": pd.DataFrame.from_records(rows),
        "swap_examples": examples,
    }


def _catalog_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: Any,
) -> dict[str, Any]:
    """Return one NWB catalog row matching exact selectors."""
    return _one_record(rows, label=label, **selectors)


def _graph_length_cm(graph_inputs: Mapping[str, Any]) -> float:
    """Return ordered path length directly from stored centimeter geometry."""
    nodes = np.asarray(graph_inputs["node_positions_cm"], dtype=float)
    edge_order = np.asarray(graph_inputs["edge_order"], dtype=int)
    spacing = np.asarray(graph_inputs["edge_spacing_cm"], dtype=float).reshape(-1)
    lengths = np.linalg.norm(
        nodes[edge_order[:, 1]] - nodes[edge_order[:, 0]],
        axis=1,
    )
    value = float(np.sum(lengths) + np.sum(spacing))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Stored W-track path length must be positive and finite.")
    return value


def _load_nwb_decoding_inputs(
    session: Mapping[str, Any],
    *,
    epochs: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], float]:
    """Load lap intervals and path length from the hash-pinned augmented NWB."""
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
            raise ValueError("Source NWB changed after Figure 2 computation.")
        catalog = catalog_augmented_nwb(
            nwbfile,
            nwb_file_name=str(session["nwb_file_name"]),
        )
        intervals = {}
        for epoch in epochs:
            intervals[str(epoch)] = {
                trajectory_type: load_interval_set(
                    nwbfile,
                    _catalog_row(
                        catalog["trajectory_intervals"],
                        label="trajectory interval",
                        epoch=epoch,
                        trajectory_type=trajectory_type,
                    ),
                )
                for trajectory_type in TRAJECTORY_TYPES
            }
        graph_lengths = []
        for trajectory_type in TRAJECTORY_TYPES:
            graph = load_wtrack_graph(
                nwbfile,
                _catalog_row(
                    catalog["wtrack_graph"],
                    label="W-track graph",
                    configuration_name=trajectory_type,
                ),
            )
            graph_lengths.append(_graph_length_cm(graph))
    path_length = graph_lengths[0]
    if any(
        not np.isclose(length, path_length, rtol=1e-10, atol=1e-12)
        for length in graph_lengths[1:]
    ):
        raise ValueError("The four stored W-track path lengths do not agree.")
    return intervals, path_length


def _cross_artifact_path(
    bundle: Mapping[str, Any],
    key: tuple[str, str, str],
    role: str,
) -> Path:
    """Return one validated cross-decoding NPZ path from its bundle manifest."""
    artifact_key = f"cross:{key[0]}:{key[1]}:{key[2]}:{role}"
    rows = bundle["manifest"].loc[
        bundle["manifest"]["artifact_key"].astype(str).eq(artifact_key)
    ]
    if len(rows) != 1:
        raise ValueError(f"Decoding manifest is missing {artifact_key!r}.")
    return Path(bundle["path"]) / str(rows.iloc[0]["relative_path"])


def _append_decoding_laps(
    records: list[dict[str, Any]],
    *,
    true: Any,
    decoded: Any,
    normalization: float,
    intervals: Any,
    animal_name: str,
    date: str,
    epoch_type: str,
    epoch: str,
    analysis: str,
    comparison: str,
    comparison_label: str,
    transfer_family: str,
    encoding_trajectory: str | None,
    decoding_trajectory: str,
    true_path: Path,
    decoded_path: Path,
) -> np.ndarray:
    """Append lap medians and return all finite normalized errors."""
    timestamps, absolute_error = legacy._align_absolute_error_with_times(
        true,
        decoded,
    )
    absolute_error = np.asarray(absolute_error, dtype=float) / float(normalization)
    legacy._append_panel_e_trial_errors(
        records,
        timestamps=np.asarray(timestamps, dtype=float),
        absolute_error=absolute_error,
        intervals=intervals,
        animal_name=animal_name,
        date=date,
        epoch_type=epoch_type,
        epoch=epoch,
        region=REGION,
        analysis=analysis,
        comparison=comparison,
        comparison_label=comparison_label,
        transfer_family=transfer_family,
        encoding_trajectory=encoding_trajectory,
        decoding_trajectory=decoding_trajectory,
        true_path=true_path,
        decoded_path=decoded_path,
    )
    return absolute_error[np.isfinite(absolute_error)]


def _path_progression_record(
    session: Mapping[str, Any],
    *,
    epoch: str,
    run_dir: Path,
    scratch_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Return one light or parent-dark cross-decoding record and root."""
    if str(epoch) == _session_dark_epoch(session):
        record = dict(session["parent_artifacts"]["dark_path_progression_decoding"])
        source_root = _parent_run_dir(
            session,
            parent="full",
            scratch_root=scratch_root,
        )
    else:
        record = _one_record(
            session["artifacts"].get("path_progression_decoding", ()),
            label="PathProgressionDecoding artifact",
            epoch=epoch,
        )
        source_root = run_dir
    return record, source_root


def _build_panel_e_decoding_tables(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pooled and lap-level decoding tables from new artifacts and NWB."""
    pooled_place: dict[str, list[np.ndarray]] = {"light": [], "dark": []}
    pooled_cross: dict[tuple[str, str, str], list[np.ndarray]] = {}
    trial_records: list[dict[str, Any]] = []
    comparison, comparison_label, transfer_family, pairs = (
        _dark_light.PANEL_E_CROSS_COMPARISONS[0]
    )
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        dark_epoch = _session_dark_epoch(session)
        epoch_intervals, path_length_cm = _load_nwb_decoding_inputs(
            session,
            epochs=(LIGHT_EPOCH, dark_epoch),
        )
        for epoch_type, epoch in (("light", LIGHT_EPOCH), ("dark", dark_epoch)):
            place_record = _one_record(
                session["artifacts"].get("path_specific_place_decoding", ()),
                label="PathSpecificPlaceDecoding artifact",
                epoch=epoch,
            )
            place_manifest = _record_artifact_path(
                place_record,
                "artifact_manifest_path",
                run_dir=run_dir,
            )
            place = path_specific_decoding.load_path_specific_decoding_artifact(
                place_manifest.parent
            )
            place_true_path = _record_artifact_path(
                place_record,
                "true_path",
                run_dir=run_dir,
            )
            place_decoded_path = _record_artifact_path(
                place_record,
                "decoded_path",
                run_dir=run_dir,
            )
            for decoding_trajectory in TRAJECTORY_TYPES:
                _append_decoding_laps(
                    trial_records,
                    true=place["true"],
                    decoded=place["decoded"],
                    normalization=path_length_cm,
                    intervals=epoch_intervals[epoch][decoding_trajectory],
                    animal_name=animal_name,
                    date=date,
                    epoch_type=epoch_type,
                    epoch=epoch,
                    analysis="place",
                    comparison="place",
                    comparison_label="Place",
                    transfer_family="within_epoch",
                    encoding_trajectory=None,
                    decoding_trajectory=decoding_trajectory,
                    true_path=place_true_path,
                    decoded_path=place_decoded_path,
                )
            # The same complete Tsd was used above for each lap family; pool it once.
            timestamps, place_error = legacy._align_absolute_error_with_times(
                place["true"], place["decoded"]
            )
            del timestamps
            finite_place = np.asarray(place_error, dtype=float) / path_length_cm
            pooled_place[epoch_type].append(finite_place[np.isfinite(finite_place)])

            cross_record, cross_root = _path_progression_record(
                session,
                epoch=epoch,
                run_dir=run_dir,
                scratch_root=scratch_root,
            )
            cross_manifest = _record_artifact_path(
                cross_record,
                "artifact_manifest_path",
                run_dir=cross_root,
            )
            cross = path_progression_decoding.load_decoding_artifact_bundle(
                cross_manifest
            )
            cross_values = []
            for encoding_trajectory, decoding_trajectory in pairs:
                key = (
                    str(transfer_family),
                    str(encoding_trajectory),
                    str(decoding_trajectory),
                )
                if key not in cross["cross_path_outputs"]:
                    raise ValueError(f"Cross-decoding bundle is missing {key!r}.")
                output = cross["cross_path_outputs"][key]
                true_path = _cross_artifact_path(cross, key, "true")
                decoded_path = _cross_artifact_path(cross, key, "decoded")
                values = _append_decoding_laps(
                    trial_records,
                    true=output["true"],
                    decoded=output["decoded"],
                    normalization=1.0,
                    intervals=epoch_intervals[epoch][decoding_trajectory],
                    animal_name=animal_name,
                    date=date,
                    epoch_type=epoch_type,
                    epoch=epoch,
                    analysis="cross_trajectory",
                    comparison=str(comparison),
                    comparison_label=str(comparison_label),
                    transfer_family=str(transfer_family),
                    encoding_trajectory=str(encoding_trajectory),
                    decoding_trajectory=str(decoding_trajectory),
                    true_path=true_path,
                    decoded_path=decoded_path,
                )
                if values.size:
                    cross_values.append(values)
            if cross_values:
                pooled_cross.setdefault(
                    (epoch_type, str(comparison), str(comparison_label)),
                    [],
                ).append(np.concatenate(cross_values))

    rows = []
    for epoch_type, values in pooled_place.items():
        finite = [value for value in values if value.size]
        if finite:
            row = _dark_light._summarize_panel_e_errors(
                np.concatenate(finite),
                animal_name=_dark_light.PANEL_E_POOLED_LABEL,
                date=_dark_light.PANEL_E_POOLED_LABEL,
                epoch_type=epoch_type,
                epoch=_dark_light.PANEL_E_POOLED_LABEL,
                analysis="place",
                comparison="place",
                comparison_label="Place",
            )
            if row is not None:
                rows.append(row)
    for (epoch_type, comparison, label), values in pooled_cross.items():
        row = _dark_light._summarize_panel_e_errors(
            np.concatenate(values),
            animal_name=_dark_light.PANEL_E_POOLED_LABEL,
            date=_dark_light.PANEL_E_POOLED_LABEL,
            epoch_type=epoch_type,
            epoch=_dark_light.PANEL_E_POOLED_LABEL,
            analysis="cross_trajectory",
            comparison=comparison,
            comparison_label=label,
        )
        if row is not None:
            rows.append(row)
    return (
        pd.DataFrame.from_records(
            rows,
            columns=_dark_light.PANEL_E_ERROR_SUMMARY_COLUMNS,
        ),
        pd.DataFrame.from_records(
            trial_records,
            columns=legacy.PANEL_E_TRIAL_ERROR_TABLE_COLUMNS,
        ),
    )


def load_figure_2_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load every artifact needed by canonical Figure 2 without legacy paths."""
    from v1ca1.spyglass.offline.figure_2 import (
        FIGURE_2_PIPELINE,
        load_figure_2_campaign,
    )

    run_dir, campaign, unordered_sessions = load_figure_2_campaign(
        run_id,
        scratch_root=scratch_root,
    )
    if str(campaign.get("analysis_parameters", {}).get("pipeline")) != (
        FIGURE_2_PIPELINE
    ):
        raise ValueError("Selected campaign is not a Figure 2 offline run.")
    sessions = _ordered_sessions(unordered_sessions)
    panel_e_summary, panel_e_trials = _build_panel_e_decoding_tables(
        run_dir,
        sessions,
        scratch_root=scratch_root,
    )
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "sessions": sessions,
        "datasets": EXPECTED_DATASETS,
        "regions": (REGION,),
        "panel_a_examples": _load_panel_a_examples(run_dir, sessions),
        "panel_b_overlap_table": _build_panel_b_overlap_table(
            run_dir,
            sessions,
            scratch_root=scratch_root,
        ),
        "panel_d_swap_payload": _build_panel_d_swap_payload(
            run_dir,
            sessions,
            scratch_root=scratch_root,
        ),
        "panel_e_decoding_error_table": panel_e_summary,
        "panel_e_decoding_trial_error_table": panel_e_trials,
    }


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject only retained campaign payloads into the canonical renderer."""
    base = legacy._figure_2
    originals: list[tuple[Any, str, Any]] = []

    def replace(module: Any, name: str, value: Any) -> None:
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    examples_by_key = {
        (
            str(row["animal_name"]),
            str(row["date"]),
            str(row["region"]),
            str(row["unit_id"]),
            tuple(str(value) for value in row["trajectories"]),
        ): row
        for row in payload["panel_a_examples"]
    }

    def load_example(**kwargs: Any) -> dict[str, Any]:
        key = (
            str(kwargs["animal_name"]),
            str(kwargs["date"]),
            str(kwargs["region"]),
            str(kwargs["unit_id"]),
            tuple(str(value) for value in kwargs["trajectories"]),
        )
        if key not in examples_by_key:
            raise ValueError(f"Unexpected Figure 2 example selection {key!r}.")
        return examples_by_key[key]

    expected_datasets = tuple(payload["datasets"])

    def require_request(kwargs: Mapping[str, Any]) -> None:
        observed = tuple(normalize_dataset_id(value) for value in kwargs["datasets"])
        if observed != expected_datasets:
            raise ValueError("Canonical Figure 2 requested unexpected sessions.")
        if str(kwargs.get("region", REGION)) != REGION:
            raise ValueError("Canonical Figure 2 requested an unexpected region.")

    def load_glm(**kwargs: Any) -> dict[str, Any]:
        require_request(kwargs)
        return payload["panel_d_swap_payload"]

    def load_overlap(**kwargs: Any) -> pd.DataFrame:
        require_request(kwargs)
        return payload["panel_b_overlap_table"]

    def keep_filtered(table: Any, **kwargs: Any) -> Any:
        require_request(kwargs)
        if float(kwargs["min_movement_firing_rate_hz"]) != (
            MINIMUM_MOVEMENT_FIRING_RATE_HZ
        ) or float(kwargs["min_stability_correlation"]) != (
            MINIMUM_STABILITY_CORRELATION
        ):
            raise ValueError("Canonical Figure 2 requested unexpected unit filters.")
        if table is not payload["panel_b_overlap_table"]:
            raise ValueError("Canonical Figure 2 attempted to refilter foreign data.")
        return table

    def load_decoding(**kwargs: Any) -> pd.DataFrame:
        require_request(kwargs)
        return payload["panel_e_decoding_error_table"]

    def load_decoding_trials(**kwargs: Any) -> pd.DataFrame:
        require_request(kwargs)
        return payload["panel_e_decoding_trial_error_table"]

    replace(base, "load_panel_glm_data", load_glm)
    replace(base, "load_panel_a_example_data", load_example)
    replace(base, "load_panel_b_tuning_overlap_table", load_overlap)
    replace(base, "filter_panel_b_overlap_by_even_odd_stability", keep_filtered)
    replace(base, "load_panel_e_decoding_error_table", load_decoding)
    replace(legacy, "build_panel_e_decoding_trial_error_table", load_decoding_trials)
    try:
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the canonical run-local Figure 2 output path."""
    output_format = str(output_format).lower()
    if output_format not in legacy._figure_2.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {legacy._figure_2.FIGURE_FORMATS!r}."
        )
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def render_figure_2(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    decoding_n_permutations: int = legacy.DECODING_PERMUTATION_COUNT,
    decoding_permutation_seed: int = legacy.DECODING_PERMUTATION_SEED,
) -> Path:
    """Render the canonical layout using only new campaign-backed inputs."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Figure 2 output must remain inside its campaign run.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Figure 2 output: {output_path}")
    if output_path.suffix.lower().lstrip(".") not in legacy._figure_2.FIGURE_FORMATS:
        raise ValueError("Figure 2 output has an unsupported format.")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    try:
        with _offline_sources(payload):
            rendered = legacy.make_figure_2(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                regions=payload["regions"],
                light_epoch=LIGHT_EPOCH,
                dark_epoch=None,
                position_bin_count=legacy._figure_2.DEFAULT_POSITION_BIN_COUNT,
                position_offset=legacy._figure_2.DEFAULT_POSITION_OFFSET,
                speed_threshold_cm_s=legacy._figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
                sigma_bins=legacy._figure_2.DEFAULT_SIGMA_BINS,
                dpi=int(dpi),
                panel_example_cache_dir=run_dir / "figures" / "cache",
                refresh_panel_example_cache=False,
                decoding_n_permutations=int(decoding_n_permutations),
                decoding_permutation_seed=int(decoding_permutation_seed),
            )
        if Path(rendered).resolve(strict=True) != temporary_path:
            raise ValueError("Figure 2 renderer returned an unexpected output path.")
        os.link(temporary_path, output_path)
        temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the database-free Figure 2 renderer arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    parser.add_argument(
        "--output-format",
        choices=legacy._figure_2.FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--decoding-n-permutations",
        type=int,
        default=legacy.DECODING_PERMUTATION_COUNT,
    )
    parser.add_argument(
        "--decoding-permutation-seed",
        type=int,
        default=legacy.DECODING_PERMUTATION_SEED,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load a complete campaign and render Figure 2 without DataJoint."""
    args = parse_arguments(argv)
    payload = load_figure_2_payload(
        run_id=args.run_id,
        scratch_root=args.scratch_root,
    )
    output_path = (
        get_output_path(
            run_dir=payload["run_dir"],
            output_format=args.output_format,
        )
        if args.output_path is None
        else args.output_path
    )
    render_figure_2(
        payload,
        output_path=output_path,
        dpi=args.dpi,
        decoding_n_permutations=args.decoding_n_permutations,
        decoding_permutation_seed=args.decoding_permutation_seed,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_figure_2_payload",
    "main",
    "render_figure_2",
]
