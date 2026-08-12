"""Recompute only the Figure 3 schematic against an immutable base campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.paper_figures.datasets import PROCESSED_DATASETS
from v1ca1.spyglass.nwb import catalog_augmented_nwb, load_interval_set
from v1ca1.spyglass.offline import figure_3
from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    DEFAULT_SCRATCH_ROOT,
    MANIFEST_SCHEMA_VERSION,
    code_provenance,
    file_sha256,
    get_run_dir,
    load_json,
    nwb_fingerprint,
    relative_run_path,
    resolve_run_path,
    utc_now,
    write_json_once,
)
from v1ca1.spyglass.offline.sources import (
    load_nwb_region_spikes,
    validate_nwb_session_identity,
)
from v1ca1.spyglass.selection import canonical_json, provenance_sha256


SUPPLEMENT_PIPELINE = "figure_3_schematic_supplement"
SUPPLEMENT_FILENAME = "panel_b_schematic.npz"
SCHEMATIC_SELECTOR_POLICY = {
    "ca1_unit_ranking": [
        {
            "field": "response_zscore",
            "transform": "absolute_value",
            "order": "descending",
            "nonfinite_value": "negative_infinity",
        },
        {
            "field": "ripple_modulation_index",
            "transform": "absolute_value",
            "order": "descending",
            "nonfinite_value": "negative_infinity",
        },
        {
            "field": "sorting_unit_id",
            "transform": "numeric_integer",
            "order": "ascending",
        },
    ],
    "ca1_unit_eligibility": "at_least_one_finite_modulation_metric",
    "v1_unit_ranking": "unchanged_base_window_count_then_nwb_id_string",
    "output_unit_identity": "nwb_unit_id",
}
SCHEMATIC_SELECTOR_POLICY_SHA256 = provenance_sha256(SCHEMATIC_SELECTOR_POLICY)
_EXPECTED_SESSIONS = {
    (str(animal_name), str(date))
    for animal_name, date, _light, _dark, _sleep in PROCESSED_DATASETS
}


def _numeric_sorting_unit_ids(loaded: Mapping[str, Any]) -> dict[str, int]:
    """Map stable NWB IDs to unique integer sorting IDs."""
    metadata = figure_3._unit_metadata_by_stable_id(loaded)
    output: dict[str, int] = {}
    for unit_id, row in metadata.items():
        numeric = pd.to_numeric(row.get("sorting_unit_id"), errors="coerce")
        if not np.isfinite(numeric) or float(numeric) != int(numeric):
            raise ValueError(
                f"Unit {unit_id!r} has a non-integer sorting_unit_id."
            )
        output[unit_id] = int(numeric)
    if len(set(output.values())) != len(output):
        raise ValueError("Sorting unit IDs must be unique within each region.")
    return output


def rank_ca1_schematic_units(
    summary: pd.DataFrame,
    *,
    ca1_spikes: Mapping[str, np.ndarray],
    sorting_unit_ids: Mapping[str, int],
) -> list[str]:
    """Apply the canonical legacy CA1 ranking while retaining NWB IDs."""
    required = {"unit_id", "response_zscore", "ripple_modulation_index"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"CA1 RippleModulation summary is missing {missing!r}.")
    ranked = summary.copy()
    ranked["unit_id"] = ranked["unit_id"].map(str)
    ranked = ranked.loc[ranked["unit_id"].isin(ca1_spikes)].copy()
    if ranked["unit_id"].duplicated().any():
        raise ValueError("CA1 RippleModulation summary contains duplicate NWB units.")
    missing_ids = sorted(set(ranked["unit_id"]).difference(sorting_unit_ids))
    if missing_ids:
        raise ValueError(f"CA1 units lack sorting IDs: {missing_ids!r}.")
    response = pd.to_numeric(ranked["response_zscore"], errors="coerce").to_numpy(
        dtype=float
    )
    modulation = pd.to_numeric(
        ranked["ripple_modulation_index"], errors="coerce"
    ).to_numpy(dtype=float)
    ranked["_response_rank"] = np.where(
        np.isfinite(response), np.abs(response), -np.inf
    )
    ranked["_modulation_rank"] = np.where(
        np.isfinite(modulation), np.abs(modulation), -np.inf
    )
    ranked["_sorting_unit_id"] = np.asarray(
        [sorting_unit_ids[unit_id] for unit_id in ranked["unit_id"]],
        dtype=np.int64,
    )
    ranked = ranked.loc[
        np.isfinite(ranked["_response_rank"].to_numpy(dtype=float))
        | np.isfinite(ranked["_modulation_rank"].to_numpy(dtype=float))
    ].sort_values(
        ["_response_rank", "_modulation_rank", "_sorting_unit_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ranked["unit_id"].tolist()


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


def _load_complete_base_campaign(
    base_run_id: str,
    *,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Strict-load and hash-snapshot exactly four completed Figure 3 sessions."""
    run_dir, campaign, sessions = figure_3.load_figure_3_campaign(
        base_run_id,
        scratch_root=scratch_root,
    )
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list):
        raise ValueError("Base Figure 3 campaign has no session index.")
    identities = {
        (str(row.get("animal_name")), str(row.get("date"))) for row in summaries
    }
    loaded_identities = {
        (str(row.get("animal_name")), str(row.get("date"))) for row in sessions
    }
    if (
        identities != _EXPECTED_SESSIONS
        or loaded_identities != _EXPECTED_SESSIONS
        or len(summaries) != len(_EXPECTED_SESSIONS)
        or len(sessions) != len(_EXPECTED_SESSIONS)
    ):
        raise ValueError(
            "A schematic supplement requires exactly the four manuscript sessions."
        )
    snapshot_sessions = []
    for summary in summaries:
        if str(summary.get("status")) != "complete":
            raise ValueError("Every base Figure 3 session must be complete.")
        manifest_path = resolve_run_path(
            str(summary["session_manifest_path"]),
            run_dir=run_dir,
        )
        snapshot_sessions.append(
            {
                "animal_name": str(summary["animal_name"]),
                "date": str(summary["date"]),
                "session_manifest_path": str(summary["session_manifest_path"]),
                "session_manifest_sha256": file_sha256(manifest_path),
            }
        )
    snapshot = {
        "run_id": str(base_run_id),
        "campaign_manifest_sha256": file_sha256(
            run_dir / CAMPAIGN_MANIFEST_FILENAME
        ),
        "sessions": sorted(
            snapshot_sessions,
            key=lambda row: (row["animal_name"], row["date"]),
        ),
    }
    return run_dir, campaign, sessions, snapshot


def _l15_sources(
    sessions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return L15 plus its exact CA1 modulation and old schematic records."""
    animal_name, date = figure_3.FIGURE_3_SCHEMATIC_SESSION
    session = _one_record(
        sessions,
        label="L15 Figure 3 session",
        animal_name=animal_name,
        date=date,
    )
    artifacts = session.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("The L15 base session lacks its artifact index.")
    modulation = _one_record(
        artifacts.get("ripple_modulation", ()),
        label="L15 CA1 RippleModulation record",
        epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
        region="ca1",
    )
    schematic = _one_record(
        artifacts.get("panel_b_schematic", ()),
        label="superseded L15 schematic record",
        epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
    )
    return session, modulation, schematic


def _source_snapshot(
    session: Mapping[str, Any],
    modulation: Mapping[str, Any],
    schematic: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the exact NWB and base artifact records used by the supplement."""
    return {
        "l15_nwb": {
            "nwb_file_name": str(session["nwb_file_name"]),
            "nwb_path": str(session["nwb_path"]),
            "nwb_fingerprint": dict(session["nwb_fingerprint"]),
        },
        "ca1_modulation_record": dict(modulation),
        "ca1_modulation_record_sha256": str(modulation["record_sha256"]),
        "superseded_schematic_record": dict(schematic),
        "superseded_schematic_record_sha256": str(schematic["record_sha256"]),
    }


def _validate_current_nwb(
    session: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Open the pinned L15 NWB and require its cheap fingerprint to match."""
    import pynwb

    nwb_path = Path(str(session["nwb_path"])).resolve(strict=True)
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=str(session["animal_name"]),
            date=str(session["date"]),
        )
        fingerprint = nwb_fingerprint(nwb_path, nwbfile)
    if canonical_json(fingerprint) != canonical_json(session["nwb_fingerprint"]):
        raise ValueError("The L15 NWB changed relative to the base Figure 3 campaign.")
    return nwb_path, fingerprint


def _build_replacement_payload(
    *,
    run_id: str,
    base_run_id: str,
    base_run_dir: Path,
    base_snapshot: Mapping[str, Any],
    session: Mapping[str, Any],
    modulation: Mapping[str, Any],
    superseded_schematic: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build only the replacement L15 schematic from NWB and modulation output."""
    import pynwb

    nwb_path, fingerprint = _validate_current_nwb(session)
    summary_path = figure_3._checked_record_path(
        modulation,
        "summary_path",
        run_dir=base_run_dir,
    )
    ca1_summary = pd.read_parquet(summary_path)
    animal_name, date = figure_3.FIGURE_3_SCHEMATIC_SESSION
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        catalog = catalog_augmented_nwb(nwbfile, nwb_file_name=nwb_path.name)
        epoch_row = _one_record(
            catalog["epoch_intervals"],
            label="L15 light epoch",
            epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
        )
        ripple_row = _one_record(
            catalog["ripples"],
            label="L15 light ripple intervals",
            epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
        )
        figure_3._ripple_provenance(ripple_row)
        expected_nwb_sources = session.get("nwb_sources")
        if not isinstance(expected_nwb_sources, Mapping) or (
            canonical_json(epoch_row)
            != canonical_json(expected_nwb_sources.get("epoch_interval"))
            or canonical_json(ripple_row)
            != canonical_json(expected_nwb_sources.get("ripple_intervals"))
        ):
            raise ValueError("The L15 NWB interval catalog changed from the base run.")
        ripple_table = figure_3._interval_frame(
            load_interval_set(nwbfile, ripple_row),
            epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
        )
        if len(ripple_table) != int(ripple_row["ripple_count"]):
            raise ValueError("The L15 NWB ripple count changed while loading.")
        epoch_bounds = (float(epoch_row["start_time"]), float(epoch_row["stop_time"]))
        loaded = {
            region: load_nwb_region_spikes(
                nwbfile,
                nwb_file_name=nwb_path.name,
                region=region,
                time_support=epoch_bounds,
            )
            for region in figure_3.FIGURE_3_REGIONS
        }
        identities = {
            region: figure_3._source_identity(
                nwb_file_name=nwb_path.name,
                loaded=loaded[region],
            )
            for region in figure_3.FIGURE_3_REGIONS
        }
        expected_identities = {
            str(row["region"]): dict(row) for row in session["source_identity"]
        }
        if canonical_json(identities) != canonical_json(expected_identities):
            raise ValueError(
                "The L15 regional spike sources changed from the base run."
            )
        ca1_sorting_ids = _numeric_sorting_unit_ids(loaded["ca1"])
        ranked_ca1 = rank_ca1_schematic_units(
            ca1_summary,
            ca1_spikes=figure_3._spike_times_by_stable_id(loaded["ca1"]),
            sorting_unit_ids=ca1_sorting_ids,
        )
        payload = figure_3._build_schematic_payload(
            nwbfile,
            animal_name=animal_name,
            date=date,
            epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
            nwb_file_name=nwb_path.name,
            ripple_table=ripple_table,
            ca1=loaded["ca1"],
            v1=loaded["v1"],
            ca1_modulation={"summary": ca1_summary},
            selector_kwargs={"ranked_ca1_unit_ids": ranked_ca1},
            selector_policy=SCHEMATIC_SELECTOR_POLICY,
        )
    payload["metadata"] = {
        **payload["metadata"],
        "supplement_run_id": str(run_id),
        "base_figure_3_run_id": str(base_run_id),
        "base_campaign_manifest_sha256": str(
            base_snapshot["campaign_manifest_sha256"]
        ),
        "ca1_modulation_record_sha256": str(modulation["record_sha256"]),
        "superseded_schematic_record_sha256": str(
            superseded_schematic["record_sha256"]
        ),
    }
    return payload, fingerprint


def build_figure_3_schematic_supplement(
    *,
    run_id: str,
    base_run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Create one no-overwrite schematic-only Figure 3 supplement run."""
    if str(run_id) == str(base_run_id):
        raise ValueError("Supplement and base Figure 3 run IDs must differ.")
    base_run_dir, _campaign, sessions, base_snapshot = _load_complete_base_campaign(
        base_run_id,
        scratch_root=scratch_root,
    )
    session, modulation, superseded = _l15_sources(sessions)
    figure_3._verify_record(modulation, run_dir=base_run_dir)
    figure_3._verify_record(superseded, run_dir=base_run_dir)
    source = _source_snapshot(session, modulation, superseded)
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    runs_root = (Path(scratch_root).expanduser().resolve(strict=False) / "runs")
    if not run_dir.resolve(strict=False).is_relative_to(runs_root):
        raise ValueError("Schematic supplement run directory escapes its scratch root.")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite a Figure 3 schematic supplement: {run_dir}"
        ) from exc
    try:
        payload, fingerprint = _build_replacement_payload(
            run_id=run_id,
            base_run_id=base_run_id,
            base_run_dir=base_run_dir,
            base_snapshot=base_snapshot,
            session=session,
            modulation=modulation,
            superseded_schematic=superseded,
        )
        payload_path = figure_3._write_schematic_payload(
            payload,
            run_dir
            / str(session["animal_name"])
            / str(session["date"])
            / "figure_payloads"
            / SUPPLEMENT_FILENAME,
        )
        record = figure_3._relative_artifact_record(
            {
                "animal_name": str(session["animal_name"]),
                "date": str(session["date"]),
                "epoch": figure_3.FIGURE_3_LIGHT_EPOCH,
                "payload_path": payload_path,
                "schema_version": figure_3.SCHEMATIC_SCHEMA_VERSION,
                "n_ripples": int(payload["n_ripples"]),
                "ripple_start_s": float(payload["ripple_start_s"]),
                "ripple_end_s": float(payload["ripple_end_s"]),
                "channel": int(payload["channel"]),
                "base_figure_3_run_id": str(base_run_id),
                "ca1_modulation_record_sha256": str(modulation["record_sha256"]),
                "supersedes_record_sha256": str(superseded["record_sha256"]),
                "selector_policy_sha256": SCHEMATIC_SELECTOR_POLICY_SHA256,
                "artifact_origin": "computed",
            },
            {"payload_path": payload_path},
            run_dir=run_dir,
            path_fields=("payload_path",),
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": str(run_id),
            "created_at_utc": utc_now(),
            "code_provenance": code_provenance(),
            "status": "complete",
            "pipeline": SUPPLEMENT_PIPELINE,
            "base_figure_3": base_snapshot,
            "source": source,
            "selector_policy": SCHEMATIC_SELECTOR_POLICY,
            "selector_policy_sha256": SCHEMATIC_SELECTOR_POLICY_SHA256,
            "l15_nwb_fingerprint": fingerprint,
            "artifacts": {"panel_b_schematic": [record]},
        }
        write_json_once(manifest, run_dir / CAMPAIGN_MANIFEST_FILENAME)
    except BaseException:
        shutil.rmtree(run_dir)
        raise
    return manifest


def _verify_base_snapshot(
    snapshot: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Reload the pinned base and reject any campaign or session mutation."""
    base_run_dir, _campaign, sessions, current = _load_complete_base_campaign(
        str(snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(dict(snapshot)):
        raise ValueError("The base Figure 3 campaign changed after supplementation.")
    return base_run_dir, sessions


def load_figure_3_schematic_supplement(
    run_id: str,
    *,
    expected_base_run_id: str | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Strictly load a complete schematic supplement and all pinned inputs."""
    runs_root = (Path(scratch_root).expanduser().resolve(strict=True) / "runs")
    run_dir = get_run_dir(run_id, scratch_root=scratch_root).resolve(strict=True)
    if not run_dir.is_relative_to(runs_root):
        raise ValueError("Schematic supplement run directory escapes its scratch root.")
    manifest = load_json(run_dir / CAMPAIGN_MANIFEST_FILENAME)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or str(manifest.get("run_id")) != str(run_id)
        or manifest.get("status") != "complete"
        or manifest.get("pipeline") != SUPPLEMENT_PIPELINE
    ):
        raise ValueError("Figure 3 schematic supplement manifest is not complete.")
    base_snapshot = manifest.get("base_figure_3")
    if not isinstance(base_snapshot, Mapping):
        raise ValueError("Schematic supplement lacks its base Figure 3 snapshot.")
    if expected_base_run_id is not None and str(base_snapshot.get("run_id")) != str(
        expected_base_run_id
    ):
        raise ValueError("Schematic supplement belongs to a different base campaign.")
    base_run_dir, sessions = _verify_base_snapshot(
        base_snapshot,
        scratch_root=scratch_root,
    )
    policy = manifest.get("selector_policy")
    policy_hash = str(manifest.get("selector_policy_sha256", ""))
    if (
        canonical_json(policy) != canonical_json(SCHEMATIC_SELECTOR_POLICY)
        or provenance_sha256(policy) != policy_hash
        or policy_hash != SCHEMATIC_SELECTOR_POLICY_SHA256
    ):
        raise ValueError("Schematic supplement selector policy changed.")

    session, modulation, superseded = _l15_sources(sessions)
    source = manifest.get("source")
    expected_source = _source_snapshot(session, modulation, superseded)
    if canonical_json(source) != canonical_json(expected_source):
        raise ValueError("Schematic supplement base source records changed.")
    figure_3._verify_record(modulation, run_dir=base_run_dir)
    figure_3._verify_record(superseded, run_dir=base_run_dir)
    _nwb_path, current_fingerprint = _validate_current_nwb(session)
    if canonical_json(manifest.get("l15_nwb_fingerprint")) != canonical_json(
        current_fingerprint
    ):
        raise ValueError("Schematic supplement L15 NWB fingerprint changed.")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "panel_b_schematic"
    }:
        raise ValueError("Schematic supplement has an unexpected artifact schema.")
    record = _one_record(
        artifacts["panel_b_schematic"],
        label="replacement schematic record",
        animal_name=figure_3.FIGURE_3_SCHEMATIC_SESSION[0],
        date=figure_3.FIGURE_3_SCHEMATIC_SESSION[1],
        epoch=figure_3.FIGURE_3_LIGHT_EPOCH,
    )
    figure_3._verify_record(record, run_dir=run_dir)
    expected_record_fields = {
        "base_figure_3_run_id": str(base_snapshot["run_id"]),
        "ca1_modulation_record_sha256": str(modulation["record_sha256"]),
        "supersedes_record_sha256": str(superseded["record_sha256"]),
        "selector_policy_sha256": policy_hash,
    }
    if any(
        str(record.get(name)) != value
        for name, value in expected_record_fields.items()
    ):
        raise ValueError("Replacement schematic record has stale provenance.")
    payload = figure_3.load_panel_b_schematic_payload(
        resolve_run_path(str(record["payload_path"]), run_dir=run_dir),
        expected_sha256=str(record["artifact_sha256"]["payload_path"]),
    )
    duplicated_fields = {
        "schema_version": payload.get("metadata", {}).get("schema_version"),
        "n_ripples": payload.get("n_ripples"),
        "ripple_start_s": payload.get("ripple_start_s"),
        "ripple_end_s": payload.get("ripple_end_s"),
        "channel": payload.get("channel"),
    }
    if any(
        canonical_json(record.get(name)) != canonical_json(value)
        for name, value in duplicated_fields.items()
    ):
        raise ValueError("Replacement schematic record disagrees with its payload.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Replacement schematic payload lacks metadata.")
    expected_metadata = {
        "schema_version": figure_3.SCHEMATIC_SCHEMA_VERSION,
        "animal_name": figure_3.FIGURE_3_SCHEMATIC_SESSION[0],
        "date": figure_3.FIGURE_3_SCHEMATIC_SESSION[1],
        "epoch": figure_3.FIGURE_3_LIGHT_EPOCH,
        "nwb_file_name": str(session["nwb_file_name"]),
        "artifact_origin": "computed_from_augmented_nwb",
        "supplement_run_id": str(run_id),
        "base_figure_3_run_id": str(base_snapshot["run_id"]),
        "base_campaign_manifest_sha256": str(
            base_snapshot["campaign_manifest_sha256"]
        ),
        "ca1_modulation_record_sha256": str(modulation["record_sha256"]),
        "superseded_schematic_record_sha256": str(superseded["record_sha256"]),
        "selector_policy_sha256": policy_hash,
    }
    if any(
        canonical_json(metadata.get(name)) != canonical_json(value)
        for name, value in expected_metadata.items()
    ) or canonical_json(metadata.get("selector_policy")) != canonical_json(policy):
        raise ValueError("Replacement schematic payload has stale provenance.")
    expected_units = int(metadata.get("n_units_per_region", -1))
    if expected_units != figure_3.SCHEMATIC_N_UNITS_PER_REGION:
        raise ValueError("Replacement schematic has a noncanonical unit count.")
    for region in figure_3.FIGURE_3_REGIONS:
        if (
            len(payload[f"{region}_unit_ids"]) != expected_units
            or len(payload[f"{region}_unit_identity"]) != expected_units
            or len(payload[f"{region}_spike_times_s"]) != expected_units
        ):
            raise ValueError(f"Replacement schematic has incomplete {region} rasters.")
    return run_dir, manifest, payload


def _parser() -> argparse.ArgumentParser:
    """Build the schematic-only supplement CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Create one immutable schematic supplement without fitting models."""
    args = _parser().parse_args(argv)
    build_figure_3_schematic_supplement(
        run_id=args.run_id,
        base_run_id=args.base_run_id,
        scratch_root=args.scratch_root,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMATIC_SELECTOR_POLICY",
    "SCHEMATIC_SELECTOR_POLICY_SHA256",
    "SUPPLEMENT_PIPELINE",
    "build_figure_3_schematic_supplement",
    "load_figure_3_schematic_supplement",
    "main",
    "rank_ca1_schematic_units",
]
