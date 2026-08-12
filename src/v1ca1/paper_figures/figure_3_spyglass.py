"""Render Figure 3 from a completed database-free Spyglass campaign."""

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

from v1ca1.paper_figures import figure_3 as legacy
from v1ca1.paper_figures.datasets import get_processed_datasets, normalize_dataset_id
from v1ca1.spyglass import (
    movement,
    ripple_cross_region_xcorr,
    ripple_glm,
    tuning_similarity,
)
from v1ca1.spyglass.offline.manifests import (
    DEFAULT_SCRATCH_ROOT,
    get_run_dir,
    nwb_fingerprint,
    resolve_run_path,
)
from v1ca1.spyglass.offline.sources import validate_nwb_session_identity
from v1ca1.spyglass.selection import canonical_json


DEFAULT_OUTPUT_NAME = "figure_3_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = tuple(get_processed_datasets())
LIGHT_EPOCH = "02_r1"
REGIONS = ("ca1", "v1")
DISPLAY_REGIONS = tuple(legacy.DEFAULT_REGIONS)
GLM_SOURCE_MODES = (
    ripple_glm.DEFAULT_SOURCE_PREDICTOR_MODE,
    "mean_activity",
)


class _UnexpectedLegacyRequest(RuntimeError):
    """Signal an attempted source request outside the retained campaign."""


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
            f"Expected exactly one {label} for {selectors!r}; found {len(matches)}."
        )
    return matches[0]


def _record_artifact_path(
    record: Mapping[str, Any],
    field: str,
    *,
    run_dir: Path,
) -> Path:
    """Resolve one validated nested or direct offline artifact pointer."""
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
            raise ValueError(f"Figure 3 campaign duplicates session {key!r}.")
        by_key[key] = session
    expected = {
        (animal_name, date) for animal_name, date, _dark_epoch in EXPECTED_DATASETS
    }
    if set(by_key) != expected:
        raise ValueError(
            "Figure 3 requires exactly the four manuscript sessions; "
            f"expected {sorted(expected)!r}, got {sorted(by_key)!r}."
        )

    ordered = []
    for animal_name, date, _dark_epoch in EXPECTED_DATASETS:
        session = by_key[(animal_name, date)]
        if session.get("epochs") != {"light": LIGHT_EPOCH}:
            raise ValueError(
                f"Session {animal_name} {date} has noncanonical Figure 3 epochs."
            )
        if tuple(session.get("regions", ())) != REGIONS:
            raise ValueError(
                f"Session {animal_name} {date} has noncanonical Figure 3 regions."
            )
        ordered.append(session)
    return ordered


def _require_computed(record: Mapping[str, Any], *, label: str) -> None:
    """Require a newly computed, non-legacy artifact record."""
    if str(record.get("artifact_origin", "")) != "computed":
        raise ValueError(f"{label} must be computed de novo.")


def _load_nwb_sorting_unit_maps(
    session: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Map persistent NWB unit IDs to numeric sorting IDs by region."""
    import pynwb

    nwb_path = Path(str(session["nwb_path"])).resolve(strict=True)
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=str(session["animal_name"]),
            date=str(session["date"]),
        )
        if canonical_json(nwb_fingerprint(nwb_path, nwbfile)) != canonical_json(
            session["nwb_fingerprint"]
        ):
            raise ValueError("Source NWB changed after Figure 3 computation.")
        units = getattr(nwbfile, "units", None)
        if units is None:
            raise ValueError("Augmented NWB has no Units table.")
        table = units.to_dataframe()

    missing = sorted({"region", "sorting_unit_id"}.difference(table.columns))
    if missing:
        raise ValueError(f"Augmented NWB Units table is missing {missing!r}.")
    if table.index.duplicated().any():
        raise ValueError("Augmented NWB contains duplicate persistent unit IDs.")

    output: dict[str, dict[str, int]] = {}
    for region in REGIONS:
        selected = table.loc[
            table["region"].astype(str).str.strip().str.casefold().eq(region)
        ]
        sorting_ids = pd.to_numeric(selected["sorting_unit_id"], errors="coerce")
        if sorting_ids.isna().any():
            raise ValueError(f"{region} sorting_unit_id values must be numeric.")
        mapping = {
            str(nwb_unit_id): int(sorting_unit_id)
            for nwb_unit_id, sorting_unit_id in zip(
                selected.index,
                sorting_ids,
                strict=True,
            )
        }
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"{region} sorting_unit_id values must be unique.")
        output[region] = mapping
    return output


def _map_unit_table(
    table: pd.DataFrame,
    *,
    region: str,
    sorting_unit_by_nwb_id: Mapping[str, int],
    label: str,
) -> pd.DataFrame:
    """Replace source NWB IDs with figure-facing sorting IDs explicitly."""
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
    if "region" in output and not output.empty:
        observed_regions = set(output["region"].astype(str).str.casefold())
        if observed_regions != {str(region).casefold()}:
            raise ValueError(f"{label} has stale region metadata.")
    nwb_unit_ids = output["unit_id"].astype(str)
    expected_stable = (
        output["spikesorting_merge_id"].astype(str) + ":" + nwb_unit_ids
    )
    if not np.array_equal(
        expected_stable.to_numpy(dtype=str),
        output["stable_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError(f"{label} has inconsistent stable unit identities.")
    missing_units = sorted(set(nwb_unit_ids).difference(sorting_unit_by_nwb_id))
    if missing_units:
        raise ValueError(
            f"{label} contains NWB units absent from the Units table: {missing_units!r}."
        )
    output["nwb_unit_id"] = nwb_unit_ids
    output["unit_id"] = np.asarray(
        [sorting_unit_by_nwb_id[value] for value in nwb_unit_ids],
        dtype=int,
    )
    return output


def _load_modulation_epoch_tables(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    unit_maps: Mapping[tuple[str, str], Mapping[str, Mapping[str, int]]],
) -> list[dict[str, Any]]:
    """Load pooled light RippleModulation tables for both regions."""
    summary_tables = []
    firing_rate_tables = []
    light_datasets = []
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        light_datasets.append((animal_name, date, LIGHT_EPOCH))
        for region in REGIONS:
            record = _one_record(
                session["artifacts"].get("ripple_modulation", ()),
                label="RippleModulation artifact",
                epoch=LIGHT_EPOCH,
                region=region,
            )
            _require_computed(record, label="RippleModulation")
            summary_path = _record_artifact_path(
                record,
                "summary_path",
                run_dir=run_dir,
            )
            peri_path = _record_artifact_path(
                record,
                "peri_ripple_firing_rate_path",
                run_dir=run_dir,
            )
            mapping = unit_maps[(animal_name, date)][region]
            summary = _map_unit_table(
                pd.read_parquet(summary_path),
                region=region,
                sorting_unit_by_nwb_id=mapping,
                label="RippleModulation summary",
            )
            peri = _map_unit_table(
                pd.read_parquet(peri_path),
                region=region,
                sorting_unit_by_nwb_id=mapping,
                label="RippleModulation peri-ripple table",
            )
            for table, label, required in (
                (
                    summary,
                    "RippleModulation summary",
                    {"ripple_modulation_index", "response_zscore"},
                ),
                (
                    peri,
                    "RippleModulation peri-ripple table",
                    {"time_s", "mean_rate_hz"},
                ),
            ):
                missing = sorted(required.difference(table.columns))
                if missing:
                    raise ValueError(f"{label} is missing columns {missing!r}.")
                identity = (
                    set(table["animal_name"].astype(str)),
                    set(table["date"].astype(str)),
                    set(table["epoch"].astype(str)),
                )
                if not table.empty and identity != (
                    {animal_name},
                    {date},
                    {LIGHT_EPOCH},
                ):
                    raise ValueError(f"{label} has stale session metadata.")
            summary_tables.append(summary.assign(source_path=str(summary_path)))
            firing_rate_tables.append(peri.assign(source_path=str(peri_path)))

    return [
        {
            "epoch_type": "light",
            "label": legacy.HEATMAP_EPOCH_LABELS["light"],
            "epoch": LIGHT_EPOCH,
            "epochs": tuple(LIGHT_EPOCH for _session in sessions),
            "datasets": tuple(light_datasets),
            "n_datasets": len(light_datasets),
            "firing_rate_table": pd.concat(
                firing_rate_tables,
                ignore_index=True,
                sort=False,
            ),
            "summary_table": pd.concat(
                summary_tables,
                ignore_index=True,
                sort=False,
            ),
        }
    ]


def _map_glm_dataset_units(
    result: Mapping[str, Any],
    *,
    sorting_unit_by_nwb_id: Mapping[str, int],
) -> Any:
    """Assign sorting IDs to one RippleGLM target coordinate after audit."""
    dataset = result["dataset"]
    selected = result["selected_units"].copy()
    target = selected.loc[
        selected["role"].astype(str).eq(ripple_glm.TARGET_ROLE)
        & selected["included_in_fit"].astype(bool)
    ].reset_index(drop=True)
    stable_ids = np.asarray(dataset.coords["unit"].values).astype(str)
    if not np.array_equal(
        stable_ids,
        target["stable_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError("RippleGLM target coordinate and unit audit disagree.")
    nwb_ids = target["unit_id"].astype(str).to_numpy()
    missing = sorted(set(nwb_ids).difference(sorting_unit_by_nwb_id))
    if missing:
        raise ValueError(f"RippleGLM targets are absent from the NWB map: {missing!r}.")
    sorting_ids = np.asarray(
        [sorting_unit_by_nwb_id[value] for value in nwb_ids],
        dtype=int,
    )
    if len(set(sorting_ids.tolist())) != len(sorting_ids):
        raise ValueError("RippleGLM sorting IDs must be unique.")
    return dataset.assign_coords(
        unit=sorting_ids,
        nwb_unit_id=("unit", nwb_ids),
    )


def _validate_glm_result(
    result: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    mode: str,
) -> None:
    """Require the fixed manuscript light RippleGLM specification."""
    expected_identity = {
        "animal_name": animal_name,
        "date": date,
        "epoch": LIGHT_EPOCH,
    }
    if any(str(result.get(name)) != value for name, value in expected_identity.items()):
        raise ValueError("RippleGLM bundle has stale session identity.")
    if str(result.get("artifact_origin")) != "computed":
        raise ValueError("RippleGLM must be computed de novo.")
    parameters = result["parameters"]
    expected_strings = {
        "source_predictor_mode": mode,
        "ripple_selection_mode": legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    }
    if any(str(parameters.get(name)) != value for name, value in expected_strings.items()):
        raise ValueError("RippleGLM bundle has noncanonical model parameters.")
    expected_numbers = {
        "ripple_window_s": legacy.DEFAULT_RIPPLE_WINDOW_S,
        "ripple_window_offset_s": legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        "ridge_strength": legacy.DEFAULT_RIDGE_STRENGTH,
        "expected_detector_zscore_threshold": 2.0,
    }
    for name, expected in expected_numbers.items():
        if not np.isclose(
            float(parameters.get(name, np.nan)),
            float(expected),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"RippleGLM has noncanonical {name}.")
    if int(parameters.get("n_shuffles_ripple", -1)) != (
        ripple_glm.DEFAULT_N_SHUFFLES_RIPPLE
    ) or parameters.get("require_speed_gated") is not True:
        raise ValueError("RippleGLM lacks the fixed shuffle or ripple-event policy.")


def _load_glm_results(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    unit_maps: Mapping[tuple[str, str], Mapping[str, Mapping[str, int]]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load and remap the two light RippleGLM bundles per session."""
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        for mode in GLM_SOURCE_MODES:
            record = _one_record(
                session["artifacts"].get("ripple_glm", ()),
                label="RippleGLM artifact",
                epoch=LIGHT_EPOCH,
                source_predictor_mode=mode,
            )
            _require_computed(record, label="RippleGLM")
            manifest_path = _record_artifact_path(
                record,
                "artifact_manifest_path",
                run_dir=run_dir,
            )
            result = ripple_glm.load_ripple_glm_artifact(manifest_path.parent)
            _validate_glm_result(
                result,
                animal_name=animal_name,
                date=date,
                mode=mode,
            )
            output[(animal_name, date, mode)] = {
                "result": result,
                "dataset": _map_glm_dataset_units(
                    result,
                    sorting_unit_by_nwb_id=unit_maps[(animal_name, date)]["v1"],
                ),
                "manifest_path": manifest_path,
            }
    return output


def _glm_summary_table(
    loaded: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    mode: str,
) -> pd.DataFrame:
    """Return the legacy per-unit summary schema from a validated bundle."""
    dataset = loaded["dataset"]
    required = (
        "ripple_devexp_mean",
        "ripple_devexp_p_value",
        "ripple_bits_per_spike_mean",
    )
    missing = [name for name in required if name not in dataset]
    if missing:
        raise ValueError(f"RippleGLM dataset is missing variables {missing!r}.")
    units = np.asarray(dataset.coords["unit"].values, dtype=int)
    return pd.DataFrame(
        {
            "animal_name": animal_name,
            "date": date,
            "epoch": LIGHT_EPOCH,
            "unit_id": units,
            "ripple_devexp_mean": np.asarray(
                dataset["ripple_devexp_mean"].values,
                dtype=float,
            ),
            "ripple_devexp_p_value": np.asarray(
                dataset["ripple_devexp_p_value"].values,
                dtype=float,
            ),
            "ripple_bits_per_spike_mean": np.asarray(
                dataset["ripple_bits_per_spike_mean"].values,
                dtype=float,
            ),
            "n_ripples": int(loaded["result"]["n_ripples"]),
            "source_predictor_mode": mode,
            "source_path": str(loaded["manifest_path"]),
        }
    )


def _build_glm_epoch_tables(
    sessions: Sequence[Mapping[str, Any]],
    glm_results: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the pooled vector-model light table for Panel C."""
    datasets = []
    tables = []
    mode = ripple_glm.DEFAULT_SOURCE_PREDICTOR_MODE
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        datasets.append((animal_name, date, LIGHT_EPOCH))
        tables.append(
            _glm_summary_table(
                glm_results[(animal_name, date, mode)],
                animal_name=animal_name,
                date=date,
                mode=mode,
            )
        )
    return [
        {
            "epoch_type": "light",
            "label": legacy.HEATMAP_EPOCH_LABELS["light"],
            "epoch": LIGHT_EPOCH,
            "datasets": tuple(datasets),
            "n_datasets": len(datasets),
            "summary_table": pd.concat(tables, ignore_index=True, sort=False),
        }
    ]


def _build_source_comparison_payload(
    sessions: Sequence[Mapping[str, Any]],
    glm_results: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair vector and mean-activity GLM scores by sorting unit ID."""
    rows = []
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        vector = _glm_summary_table(
            glm_results[(animal_name, date, "unit_vector")],
            animal_name=animal_name,
            date=date,
            mode="unit_vector",
        ).rename(
            columns={
                "ripple_devexp_mean": "vector_devexp_mean",
                "ripple_devexp_p_value": "vector_devexp_p_value",
                "ripple_bits_per_spike_mean": "vector_bits_per_spike_mean",
                "source_path": "vector_source_path",
            }
        )
        mean = _glm_summary_table(
            glm_results[(animal_name, date, "mean_activity")],
            animal_name=animal_name,
            date=date,
            mode="mean_activity",
        ).rename(
            columns={
                "ripple_devexp_mean": "mean_activity_devexp_mean",
                "ripple_devexp_p_value": "mean_activity_devexp_p_value",
                "ripple_bits_per_spike_mean": "mean_activity_bits_per_spike_mean",
                "source_path": "mean_activity_source_path",
            }
        )
        joined = vector.merge(
            mean[
                [
                    "animal_name",
                    "date",
                    "epoch",
                    "unit_id",
                    "mean_activity_devexp_mean",
                    "mean_activity_devexp_p_value",
                    "mean_activity_bits_per_spike_mean",
                    "mean_activity_source_path",
                ]
            ],
            on=["animal_name", "date", "epoch", "unit_id"],
            how="inner",
            validate="one_to_one",
        )
        rows.append(
            joined.assign(
                epoch_type="light",
                label=legacy.HEATMAP_EPOCH_LABELS["light"],
                devexp_delta_vector_minus_mean=(
                    joined["vector_devexp_mean"]
                    - joined["mean_activity_devexp_mean"]
                ),
            )
        )
    return {
        "comparison_table": pd.concat(rows, ignore_index=True, sort=False),
        "missing_artifacts": [],
        "ripple_selection": legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
    }


def _build_prediction_examples(
    glm_results: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract the fixed Figure 3 observed/predicted V1 examples."""
    output = []
    for animal_name, date, epoch, unit_id in legacy.DEFAULT_PANEL_B_PREDICTION_EXAMPLES:
        if str(epoch) != LIGHT_EPOCH:
            raise ValueError("Figure 3 prediction examples must use the light epoch.")
        loaded = glm_results[(str(animal_name), str(date), "unit_vector")]
        dataset = loaded["dataset"]
        units = np.asarray(dataset.coords["unit"].values, dtype=int)
        matches = np.flatnonzero(units == int(unit_id))
        if matches.size != 1:
            raise ValueError(
                f"Configured RippleGLM example {animal_name} unit {unit_id} is absent."
            )
        unit_index = int(matches[0])
        required = (
            "ripple_observed_count_oof",
            "ripple_predicted_count_oof",
            "ripple_devexp_mean",
            "ripple_devexp_p_value",
        )
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(f"RippleGLM example lacks variables {missing!r}.")
        output.append(
            {
                "animal_name": str(animal_name),
                "date": str(date),
                "epoch": LIGHT_EPOCH,
                "unit_id": int(unit_id),
                "observed": np.asarray(
                    dataset["ripple_observed_count_oof"].values[:, unit_index],
                    dtype=float,
                ),
                "predicted": np.asarray(
                    dataset["ripple_predicted_count_oof"].values[:, unit_index],
                    dtype=float,
                ),
                "ripple_devexp_mean": float(
                    dataset["ripple_devexp_mean"].values[unit_index]
                ),
                "ripple_devexp_p_value": float(
                    dataset["ripple_devexp_p_value"].values[unit_index]
                ),
                "source_path": str(loaded["manifest_path"]),
            }
        )
    return output


def _parent_run_dir(
    session: Mapping[str, Any],
    *,
    field: str,
    scratch_root: Path,
) -> Path:
    """Return one hash-pinned parent campaign directory."""
    parent = session.get("parent_artifacts", {})
    if not isinstance(parent, Mapping) or field not in parent:
        raise ValueError(f"Session parent_artifacts is missing {field!r}.")
    return get_run_dir(str(parent[field]), scratch_root=scratch_root)


def _load_dark_movement_table(
    session: Mapping[str, Any],
    *,
    scratch_root: Path,
    sorting_unit_by_nwb_id: Mapping[str, int],
) -> pd.DataFrame:
    """Load one parent Figure 1 dark movement-rate table."""
    parent = session["parent_artifacts"]
    record = dict(parent["dark_movement_firing_rate"])
    # The initial Figure 1 manifest predates per-record artifact_origin fields.
    # The Figure 3 campaign loader validates the complete, computed parent chain
    # and the exact parent session/artifact checksums before returning this row.
    source_root = _parent_run_dir(
        session,
        field="dark_movement_firing_rate_run_id",
        scratch_root=scratch_root,
    )
    path = _record_artifact_path(record, "firing_rate_path", run_dir=source_root)
    table = _map_unit_table(
        movement.load_movement_firing_rate_artifact(path),
        region="v1",
        sorting_unit_by_nwb_id=sorting_unit_by_nwb_id,
        label="Dark MovementFiringRate",
    )
    expected_epoch = next(
        dark_epoch
        for animal_name, date, dark_epoch in EXPECTED_DATASETS
        if (animal_name, date)
        == (str(session["animal_name"]), str(session["date"]))
    )
    if not table.empty and set(table["epoch"].astype(str)) != {expected_epoch}:
        raise ValueError("Dark MovementFiringRate has stale epoch metadata.")
    return table.rename(columns={"unit_id": "unit"})[
        ["unit", "movement_firing_rate_hz"]
    ].rename(columns={"movement_firing_rate_hz": "dark_firing_rate_hz"})


def _load_dark_similarity_table(
    session: Mapping[str, Any],
    *,
    scratch_root: Path,
    sorting_unit_by_nwb_id: Mapping[str, int],
) -> pd.DataFrame:
    """Load direct dark overlaps and derive DPPI with the fixed max rule."""
    parent = session["parent_artifacts"]
    record = dict(parent["dark_tuning_similarity"])
    _require_computed(record, label="Dark tuning similarity")
    source_root = _parent_run_dir(
        session,
        field="dark_tuning_similarity_run_id",
        scratch_root=scratch_root,
    )
    path = _record_artifact_path(record, "similarity_path", run_dir=source_root)
    table = _map_unit_table(
        tuning_similarity.load_tuning_similarity_artifact(path),
        region="v1",
        sorting_unit_by_nwb_id=sorting_unit_by_nwb_id,
        label="Dark tuning similarity",
    )
    dark_epoch = str(record["epoch"])
    selected = table.loc[
        table["epoch"].astype(str).eq(dark_epoch)
        & table["region"].astype(str).eq("v1")
        & table["similarity_metric"].astype(str).eq(
            legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
        )
        & table["comparison_family"].astype(str).eq("same_turn")
        & table["comparison_label"].astype(str).isin(("left_turn", "right_turn"))
        & table["similarity_status"].astype(str).eq("valid")
    ].copy()
    selected["similarity"] = pd.to_numeric(selected["similarity"], errors="coerce")
    selected = selected.loc[np.isfinite(selected["similarity"].to_numpy(dtype=float))]
    if selected.empty:
        return pd.DataFrame(
            columns=["unit", "same_turn_tuning_similarity", "tuning_source_path"]
        )
    pooled = (
        selected.groupby("unit_id", sort=True, as_index=False)["similarity"]
        .max()
        .rename(
            columns={
                "unit_id": "unit",
                "similarity": "same_turn_tuning_similarity",
            }
        )
    )
    return pooled.assign(tuning_source_path=str(path))


def _build_behavior_payload(
    sessions: Sequence[Mapping[str, Any]],
    glm_results: Mapping[tuple[str, str, str], Mapping[str, Any]],
    unit_maps: Mapping[tuple[str, str], Mapping[str, Mapping[str, int]]],
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Build Panel E joins from light GLMs and hash-pinned dark parents."""
    devexp_rows = []
    activity_rows = []
    dppi_rows = []
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        mapping = unit_maps[(animal_name, date)]["v1"]
        movement_table = _load_dark_movement_table(
            session,
            scratch_root=scratch_root,
            sorting_unit_by_nwb_id=mapping,
        )
        similarity_table = _load_dark_similarity_table(
            session,
            scratch_root=scratch_root,
            sorting_unit_by_nwb_id=mapping,
        )
        dark_epoch = next(
            value
            for animal, session_date, value in EXPECTED_DATASETS
            if (animal, session_date) == (animal_name, date)
        )
        glm_table = _glm_summary_table(
            glm_results[(animal_name, date, "unit_vector")],
            animal_name=animal_name,
            date=date,
            mode="unit_vector",
        )
        devexp_rows.append(
            legacy.build_glm_dark_activity_devexp_table(
                glm_table,
                movement_table,
                similarity_table,
                animal_name=animal_name,
                date=date,
                glm_epoch=LIGHT_EPOCH,
                epoch_type="light",
                dark_epoch=dark_epoch,
                dark_activity_threshold_hz=legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ,
            )
        )
        activity_rows.append(
            legacy.build_dark_activity_reference_table(
                movement_table,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_epoch,
                dark_activity_threshold_hz=legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ,
            )
        )
        dppi_rows.append(
            legacy.build_dark_active_dppi_reference_table(
                movement_table,
                similarity_table,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_epoch,
                dark_activity_threshold_hz=legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ,
            )
        )
    return {
        "devexp_table": pd.concat(devexp_rows, ignore_index=True, sort=False),
        "dark_activity_reference_table": pd.concat(
            activity_rows,
            ignore_index=True,
            sort=False,
        ),
        "dark_active_dppi_reference_table": pd.concat(
            dppi_rows,
            ignore_index=True,
            sort=False,
        ),
        "missing_artifacts": [],
        "region": "v1",
        "dark_activity_threshold_hz": legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ,
        "tuning_comparison_label": legacy.DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL,
        "tuning_similarity_metric": legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC,
    }


def _build_xcorr_payload(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    unit_maps: Mapping[tuple[str, str], Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    """Load and remap the fixed L15 light RippleCrossRegionXCorr bundle."""
    animal_name, date, epoch = legacy.DEFAULT_XCORR_DATASET
    session = next(
        value
        for value in sessions
        if (str(value["animal_name"]), str(value["date"]))
        == (animal_name, date)
    )
    record = _one_record(
        session["artifacts"].get("ripple_cross_region_xcorr", ()),
        label="RippleCrossRegionXCorr artifact",
        epoch=epoch,
    )
    _require_computed(record, label="RippleCrossRegionXCorr")
    manifest_path = _record_artifact_path(
        record,
        "artifact_manifest_path",
        run_dir=run_dir,
    )
    result = ripple_cross_region_xcorr.load_ripple_cross_region_xcorr_artifact(
        manifest_path.parent
    )
    if (
        str(result["animal_name"]),
        str(result["date"]),
        str(result["epoch"]),
        str(result["artifact_origin"]),
    ) != (animal_name, date, epoch, "computed"):
        raise ValueError("RippleCrossRegionXCorr bundle has stale identity.")
    parameters = result["parameters"]
    expected_parameters = {
        "bin_size_s": legacy.DEFAULT_XCORR_BIN_SIZE_S,
        "max_lag_s": legacy.DEFAULT_XCORR_MAX_LAG_S,
        "expected_detector_zscore_threshold": 2.0,
    }
    if any(
        not np.isclose(
            float(parameters.get(name, np.nan)),
            float(value),
            rtol=0.0,
            atol=1e-12,
        )
        for name, value in expected_parameters.items()
    ) or parameters.get("require_speed_gated") is not True:
        raise ValueError("RippleCrossRegionXCorr has noncanonical parameters.")

    maps = unit_maps[(animal_name, date)]
    summary = result["summary"].copy()
    for region in REGIONS:
        field = f"{region}_unit_id"
        nwb_ids = summary[field].astype(str)
        missing = sorted(set(nwb_ids).difference(maps[region]))
        if missing:
            raise ValueError(f"XCorr {region} units are absent from the NWB map.")
        summary[f"{region}_nwb_unit_id"] = nwb_ids
        summary[field] = np.asarray([maps[region][value] for value in nwb_ids], dtype=int)

    dataset = result["dataset"]
    coordinate_updates: dict[str, Any] = {}
    for region in REGIONS:
        dim = f"{region}_unit"
        id_coordinate = f"{region}_source_unit_id"
        nwb_ids = np.asarray(dataset.coords[id_coordinate].values).astype(str)
        missing = sorted(set(nwb_ids).difference(maps[region]))
        if missing:
            raise ValueError(f"XCorr dataset {region} units are absent from the NWB map.")
        coordinate_updates[dim] = np.asarray(
            [maps[region][value] for value in nwb_ids],
            dtype=int,
        )
        coordinate_updates[f"{region}_nwb_unit_id"] = (dim, nwb_ids)
    dataset = dataset.assign_coords(**coordinate_updates)

    valid_summary = summary.loc[
        summary["status"].astype(str).eq(legacy.PAIR_STATUS_VALID)
    ].copy()
    if valid_summary.empty:
        raise ValueError("RippleCrossRegionXCorr contains no valid pairs.")
    ca1_order = legacy.order_ca1_units_by_best_partner(valid_summary)
    ca1_order = ca1_order[: legacy.DEFAULT_XCORR_TOP_CA1_UNITS]
    if ca1_order.size == 0:
        raise ValueError("RippleCrossRegionXCorr contains no rankable CA1 units.")
    top_ca1 = ca1_order[0]
    top_rows = valid_summary.loc[valid_summary["ca1_unit_id"] == top_ca1].sort_values(
        by=["peak_norm_xcorr", "peak_lag_s"],
        ascending=[False, True],
        kind="stable",
    )
    v1_order = top_rows["v1_unit_id"].to_numpy()
    available_ca1 = np.asarray(dataset.coords["ca1_unit"].values)
    available_v1 = np.asarray(dataset.coords["v1_unit"].values)
    ca1_order = legacy._filter_existing_unit_ids(ca1_order, available_ca1)
    v1_order = legacy._filter_existing_unit_ids(v1_order, available_v1)
    if ca1_order.size == 0 or v1_order.size == 0:
        raise ValueError("XCorr summary units do not overlap its result dataset.")
    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "state": legacy.DEFAULT_XCORR_STATE,
        "summary_path": manifest_path.parent / ripple_cross_region_xcorr.SUMMARY_FILENAME,
        "dataset_path": manifest_path.parent / ripple_cross_region_xcorr.RESULT_FILENAME,
        "summary_table": valid_summary,
        "ca1_unit_ids": ca1_order,
        "v1_unit_ids": v1_order,
        "v1_order_reference_ca1_unit": top_ca1,
        "lag_s": np.asarray(dataset["lag_s"].values, dtype=float),
        "xcorr": np.asarray(
            dataset["xcorr"].sel(ca1_unit=ca1_order, v1_unit=v1_order).values,
            dtype=float,
        ),
        "display_vmax": legacy.DEFAULT_XCORR_DISPLAY_VMAX,
        "attrs": dict(dataset.attrs),
    }


def _legacy_xcorr_display_precision(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy XCorr display arrays at the legacy NetCDF float precision."""
    output = dict(payload)
    output["lag_s"] = np.array(payload["lag_s"], dtype=np.float32, copy=True)
    output["xcorr"] = np.array(payload["xcorr"], dtype=np.float32, copy=True)
    return output


def _map_schematic_unit_ids(
    payload: Mapping[str, Any],
    *,
    unit_maps: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Map schematic identities to sorting IDs with an explicit audit."""
    output = dict(payload)
    for region in REGIONS:
        id_field = f"{region}_unit_ids"
        identity_field = f"{region}_unit_identity"
        identities = [dict(value) for value in output[identity_field]]
        unit_ids = np.asarray(output[id_field]).astype(str)
        if len(identities) != len(unit_ids):
            raise ValueError(f"Schematic {region} identity audit is misaligned.")
        mapped = []
        for unit_id, identity in zip(unit_ids, identities, strict=True):
            if str(identity.get("unit_id")) != unit_id:
                raise ValueError(f"Schematic {region} unit identity is stale.")
            if unit_id not in unit_maps[region]:
                raise ValueError(f"Schematic {region} unit is absent from the NWB map.")
            sorting_id = unit_maps[region][unit_id]
            if int(identity.get("sorting_unit_id")) != sorting_id:
                raise ValueError(f"Schematic {region} sorting identity is stale.")
            mapped.append(sorting_id)
        output[f"{region}_nwb_unit_ids"] = unit_ids
        output[id_field] = np.asarray(mapped, dtype=int)
    return output


def _load_schematic_payload(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    unit_maps: Mapping[tuple[str, str], Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    """Load the mandatory NWB-derived L15 schematic payload."""
    from v1ca1.spyglass.offline.figure_3 import load_panel_b_schematic_payload

    animal_name, date, epoch = legacy.DEFAULT_PANEL_B_SCHEMATIC_DATASET
    session = next(
        value
        for value in sessions
        if (str(value["animal_name"]), str(value["date"]))
        == (animal_name, date)
    )
    record = _one_record(
        session["artifacts"].get("panel_b_schematic", ()),
        label="Panel B schematic artifact",
        epoch=epoch,
    )
    _require_computed(record, label="Panel B schematic")
    path = _record_artifact_path(record, "payload_path", run_dir=run_dir)
    expected_sha256 = record.get("artifact_sha256")
    if isinstance(expected_sha256, Mapping):
        expected_sha256 = expected_sha256.get("payload_path")
    payload = load_panel_b_schematic_payload(
        path,
        expected_sha256=None if expected_sha256 is None else str(expected_sha256),
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Panel B schematic payload lacks source metadata.")
    if (
        str(metadata.get("animal_name")),
        str(metadata.get("date")),
        str(metadata.get("epoch")),
    ) != (animal_name, date, epoch):
        raise ValueError("Panel B schematic payload has stale identity.")
    return _map_schematic_unit_ids(
        payload,
        unit_maps=unit_maps[(animal_name, date)],
    )


def load_figure_3_payload(
    *,
    run_id: str,
    supplement_run_id: str | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load every artifact needed by canonical Figure 3 without legacy paths."""
    from v1ca1.spyglass.offline.figure_3 import (
        FIGURE_3_PIPELINE,
        load_figure_3_campaign,
    )

    run_dir, campaign, unordered_sessions = load_figure_3_campaign(
        run_id,
        scratch_root=scratch_root,
    )
    if str(campaign.get("analysis_parameters", {}).get("pipeline")) != (
        FIGURE_3_PIPELINE
    ):
        raise ValueError("Selected campaign is not a Figure 3 offline run.")
    sessions = _ordered_sessions(unordered_sessions)
    unit_maps = {
        (str(session["animal_name"]), str(session["date"])): (
            _load_nwb_sorting_unit_maps(session)
        )
        for session in sessions
    }
    glm_results = _load_glm_results(run_dir, sessions, unit_maps)
    payload = {
        "run_dir": run_dir,
        "campaign": campaign,
        "sessions": sessions,
        "datasets": EXPECTED_DATASETS,
        "regions": REGIONS,
        "heatmap_epoch_tables": _load_modulation_epoch_tables(
            run_dir,
            sessions,
            unit_maps,
        ),
        "glm_epoch_tables": _build_glm_epoch_tables(sessions, glm_results),
        "schematic_payload": _load_schematic_payload(run_dir, sessions, unit_maps),
        "prediction_examples": _build_prediction_examples(glm_results),
        "behavior_payload": _build_behavior_payload(
            sessions,
            glm_results,
            unit_maps,
            scratch_root=scratch_root,
        ),
        "source_comparison_payload": _build_source_comparison_payload(
            sessions,
            glm_results,
        ),
        "xcorr_payload": _build_xcorr_payload(run_dir, sessions, unit_maps),
    }
    if supplement_run_id is not None:
        from v1ca1.spyglass.offline.figure_3_schematic_supplement import (
            load_figure_3_schematic_supplement,
        )

        supplement_run_dir, supplement, schematic = (
            load_figure_3_schematic_supplement(
                supplement_run_id,
                expected_base_run_id=run_id,
                scratch_root=scratch_root,
            )
        )
        animal_name, date, _epoch = legacy.DEFAULT_PANEL_B_SCHEMATIC_DATASET
        payload["schematic_payload"] = _map_schematic_unit_ids(
            schematic,
            unit_maps=unit_maps[(animal_name, date)],
        )
        payload["base_run_dir"] = run_dir
        payload["run_dir"] = supplement_run_dir
        payload["schematic_supplement"] = supplement
    return payload


def _require_common_request(
    data_root: Path,
    datasets: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
) -> None:
    """Reject any legacy request outside the retained light-only campaign."""
    if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
        strict=True
    ):
        raise _UnexpectedLegacyRequest("Canonical Figure 3 requested a foreign root.")
    observed = tuple(normalize_dataset_id(value) for value in datasets)
    if observed != tuple(payload["datasets"]):
        raise _UnexpectedLegacyRequest("Canonical Figure 3 requested foreign sessions.")
    expected_epochs = {"light_epoch": LIGHT_EPOCH, "dark_epoch": None, "sleep_epoch": None}
    if any(kwargs.get(name) != value for name, value in expected_epochs.items()):
        raise _UnexpectedLegacyRequest("Canonical Figure 3 requested foreign epochs.")


def _require_glm_settings(kwargs: Mapping[str, Any]) -> None:
    """Reject GLM requests that differ from the computed campaign."""
    expected = {
        "ripple_window_s": legacy.DEFAULT_RIPPLE_WINDOW_S,
        "ripple_window_offset_s": legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        "ridge_strength": legacy.DEFAULT_RIDGE_STRENGTH,
    }
    if any(
        not np.isclose(
            float(kwargs.get(name, np.nan)),
            float(value),
            rtol=0.0,
            atol=1e-12,
        )
        for name, value in expected.items()
    ) or str(kwargs.get("ripple_selection")) != (
        legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION
    ):
        raise _UnexpectedLegacyRequest("Canonical Figure 3 requested foreign GLM settings.")


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject only validated campaign payloads into the canonical renderer."""
    originals: list[tuple[Any, str, Any]] = []

    def replace(name: str, value: Any) -> None:
        originals.append((legacy, name, getattr(legacy, name)))
        setattr(legacy, name, value)

    def load_heatmaps(data_root: Path, datasets: Sequence[Any], **kwargs: Any) -> Any:
        _require_common_request(data_root, datasets, kwargs, payload=payload)
        if kwargs.get("ripple_threshold_zscore") is not None:
            raise _UnexpectedLegacyRequest("Figure 3 cannot rethreshold NWB ripples.")
        return payload["heatmap_epoch_tables"]

    def load_glm_epochs(data_root: Path, datasets: Sequence[Any], **kwargs: Any) -> Any:
        _require_common_request(data_root, datasets, kwargs, payload=payload)
        _require_glm_settings(kwargs)
        if tuple(kwargs.get("epoch_types", ())) != legacy.PANEL_C_EPOCH_ORDER:
            raise _UnexpectedLegacyRequest("Figure 3 requested foreign GLM epochs.")
        return payload["glm_epoch_tables"]

    def load_schematic(data_root: Path, **kwargs: Any) -> Any:
        if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
            strict=True
        ):
            raise _UnexpectedLegacyRequest("Figure 3 requested a foreign schematic root.")
        expected = legacy.DEFAULT_PANEL_B_SCHEMATIC_DATASET
        observed = tuple(str(kwargs[name]) for name in ("animal_name", "date", "epoch"))
        if observed != expected or kwargs.get("ripple_threshold_zscore") is not None:
            raise _UnexpectedLegacyRequest("Figure 3 requested a foreign schematic.")
        return payload["schematic_payload"]

    def load_predictions(data_root: Path, **kwargs: Any) -> Any:
        if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
            strict=True
        ):
            raise _UnexpectedLegacyRequest("Figure 3 requested foreign predictions.")
        _require_glm_settings(kwargs)
        return payload["prediction_examples"]

    def load_behavior(data_root: Path, datasets: Sequence[Any], **kwargs: Any) -> Any:
        _require_common_request(data_root, datasets, kwargs, payload=payload)
        _require_glm_settings(kwargs)
        if (
            str(kwargs.get("region", legacy.DEFAULT_PANEL_D_REGION))
            != legacy.DEFAULT_PANEL_D_REGION
            or tuple(kwargs.get("epoch_types", legacy.PANEL_D_EPOCH_ORDER))
            != legacy.PANEL_D_EPOCH_ORDER
            or str(
                kwargs.get(
                    "tuning_similarity_metric",
                    legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC,
                )
            )
            != legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
        ):
            raise _UnexpectedLegacyRequest("Figure 3 requested foreign Panel E inputs.")
        return payload["behavior_payload"]

    def load_source_comparison(
        data_root: Path,
        datasets: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        _require_common_request(data_root, datasets, kwargs, payload=payload)
        _require_glm_settings(kwargs)
        if tuple(kwargs.get("epoch_types", legacy.PANEL_E_GLM_EPOCH_ORDER)) != (
            legacy.PANEL_E_GLM_EPOCH_ORDER
        ):
            raise _UnexpectedLegacyRequest("Figure 3 requested foreign Panel D epochs.")
        return payload["source_comparison_payload"]

    def load_xcorr(data_root: Path, **kwargs: Any) -> Any:
        if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
            strict=True
        ):
            raise _UnexpectedLegacyRequest("Figure 3 requested a foreign XCorr root.")
        expected_identity = legacy.DEFAULT_XCORR_DATASET
        observed_identity = tuple(
            str(kwargs.get(name)) for name in ("animal_name", "date", "epoch")
        )
        expected_numbers = {
            "top_n_ca1_units": legacy.DEFAULT_XCORR_TOP_CA1_UNITS,
            "max_lag_s": legacy.DEFAULT_XCORR_MAX_LAG_S,
            "bin_size_s": legacy.DEFAULT_XCORR_BIN_SIZE_S,
            "display_vmax": legacy.DEFAULT_XCORR_DISPLAY_VMAX,
        }
        if (
            observed_identity != expected_identity
            or str(kwargs.get("state", legacy.DEFAULT_XCORR_STATE))
            != legacy.DEFAULT_XCORR_STATE
            or any(
                not np.isclose(
                    float(kwargs.get(name, np.nan)),
                    float(value),
                    rtol=0.0,
                    atol=1e-12,
                )
                for name, value in expected_numbers.items()
            )
        ):
            raise _UnexpectedLegacyRequest("Figure 3 requested a foreign XCorr result.")
        return _legacy_xcorr_display_precision(payload["xcorr_payload"])

    def forbid_fallback(*_args: Any, **_kwargs: Any) -> Any:
        raise _UnexpectedLegacyRequest("Synthetic or legacy Figure 3 fallback is disabled.")

    replace("load_pooled_ripple_heatmap_epoch_tables", load_heatmaps)
    replace("load_glm_epoch_summary_tables", load_glm_epochs)
    replace("load_or_build_panel_b_schematic_example", load_schematic)
    replace("load_panel_b_prediction_examples", load_predictions)
    replace("load_glm_dark_activity_devexp_tables", load_behavior)
    replace("load_glm_source_predictor_comparison_tables", load_source_comparison)
    replace("load_top_ca1_xcorr_panel_data", load_xcorr)
    replace("load_example_ripple_lfp_trace", forbid_fallback)
    replace("build_panel_b_schematic_example", forbid_fallback)
    replace("load_first_available_glm_prediction", forbid_fallback)
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
    """Return the canonical run-local Figure 3 output path."""
    output_format = str(output_format).lower()
    if output_format not in legacy.FIGURE_FORMATS:
        raise ValueError(f"output_format must be one of {legacy.FIGURE_FORMATS!r}.")
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def render_figure_3(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render the canonical layout using only new campaign-backed inputs."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Figure 3 output must remain inside its campaign run.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Figure 3 output: {output_path}")
    if output_path.suffix.lower().lstrip(".") not in legacy.FIGURE_FORMATS:
        raise ValueError("Figure 3 output has an unsupported format.")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    try:
        with _offline_sources(payload):
            rendered = legacy.make_figure_3(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                example_dataset=legacy.DEFAULT_EXAMPLE_DATASET,
                light_epoch=LIGHT_EPOCH,
                dark_epoch=None,
                sleep_epoch=None,
                regions=DISPLAY_REGIONS,
                ripple_threshold_zscore=None,
                ripple_window_s=legacy.DEFAULT_RIPPLE_WINDOW_S,
                ripple_window_offset_s=legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S,
                ripple_selection=legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
                ridge_strength=legacy.DEFAULT_RIDGE_STRENGTH,
                dark_movement_fr_cache_dir=run_dir / "figures" / "cache",
                refresh_dark_movement_fr_cache=False,
                refresh_panel_b_schematic_cache=False,
                dpi=int(dpi),
                panel_d_tuning_similarity_metric=(
                    legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
                ),
            )
        if Path(rendered).resolve(strict=True) != temporary_path:
            raise ValueError("Figure 3 renderer returned an unexpected output path.")
        os.link(temporary_path, output_path)
        temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the database-free Figure 3 renderer arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--supplement-run-id")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    parser.add_argument(
        "--output-format",
        choices=legacy.FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load a complete campaign and render Figure 3 without DataJoint."""
    args = parse_arguments(argv)
    payload = load_figure_3_payload(
        run_id=args.run_id,
        supplement_run_id=args.supplement_run_id,
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
    render_figure_3(payload, output_path=output_path, dpi=args.dpi)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_figure_3_payload",
    "main",
    "render_figure_3",
]
