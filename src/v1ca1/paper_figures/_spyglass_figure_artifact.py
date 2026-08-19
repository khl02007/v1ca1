"""Validate and publish run-local Spyglass figure artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    code_provenance,
    file_sha256,
    load_json,
    relative_run_path,
    utc_now,
    write_json_once,
)
from v1ca1.spyglass.selection import canonical_json


PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_SUFFIX = ".spyglass-provenance.json"


def get_figure_provenance_path(figure_path: Path) -> Path:
    """Return the receipt path for one run-local or promoted figure."""
    figure_path = Path(figure_path)
    return figure_path.with_name(f"{figure_path.name}{PROVENANCE_SUFFIX}")


def _validated_campaign_manifest(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Return the on-disk campaign manifest matching a loaded payload."""
    campaign = payload.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("Spyglass figure payload has no campaign snapshot.")
    manifest_path = run_dir / CAMPAIGN_MANIFEST_FILENAME
    manifest = load_json(manifest_path)
    if canonical_json(manifest) != canonical_json(dict(campaign)):
        raise ValueError("Figure payload and campaign manifest disagree.")
    return manifest_path, manifest


def write_figure_provenance(
    payload: Mapping[str, Any],
    *,
    figure_path: Path,
    artifact_kind: str,
) -> Path:
    """Write a checksum-bearing receipt for one immutable run-local render."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    figure_path = Path(figure_path).resolve(strict=True)
    if not figure_path.is_relative_to(run_dir):
        raise ValueError("Spyglass figure output must remain inside its run.")
    manifest_path, campaign = _validated_campaign_manifest(
        payload,
        run_dir=run_dir,
    )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_kind": str(artifact_kind),
        "created_at_utc": utc_now(),
        "run_id": str(campaign["run_id"]),
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
                figure_path,
                run_dir=run_dir,
            ),
            "format": figure_path.suffix.lower().lstrip("."),
            "size_bytes": int(figure_path.stat().st_size),
            "sha256": file_sha256(figure_path),
        },
        "render_code_provenance": code_provenance(),
    }
    provenance_path = get_figure_provenance_path(figure_path)
    write_json_once(provenance, provenance_path)
    return provenance_path


def _load_validated_figure_provenance(
    payload: Mapping[str, Any],
    *,
    figure_path: Path,
    artifact_kind: str,
) -> dict[str, Any]:
    """Load and validate one run-local figure receipt."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    figure_path = Path(figure_path).resolve(strict=True)
    if not figure_path.is_relative_to(run_dir):
        raise ValueError("Promoted figure source must remain inside its run.")
    provenance = load_json(get_figure_provenance_path(figure_path))
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported Spyglass figure provenance schema.")
    if str(provenance.get("artifact_kind")) != str(artifact_kind):
        raise ValueError("Figure receipt has the wrong artifact kind.")

    manifest_path, campaign = _validated_campaign_manifest(
        payload,
        run_dir=run_dir,
    )
    if str(provenance.get("run_id")) != str(campaign["run_id"]):
        raise ValueError("Figure receipt has the wrong run ID.")
    manifest_record = provenance.get("campaign_manifest")
    figure_record = provenance.get("figure")
    if not isinstance(manifest_record, Mapping) or not isinstance(
        figure_record,
        Mapping,
    ):
        raise ValueError("Figure receipt is missing artifact records.")
    if str(manifest_record.get("run_relative_path")) != relative_run_path(
        manifest_path,
        run_dir=run_dir,
    ) or str(manifest_record.get("sha256")) != file_sha256(manifest_path):
        raise ValueError("Campaign manifest changed after figure rendering.")
    if str(figure_record.get("run_relative_path")) != relative_run_path(
        figure_path,
        run_dir=run_dir,
    ):
        raise ValueError("Figure receipt points to a different artifact.")
    if str(figure_record.get("format")) != figure_path.suffix.lower().lstrip(
        "."
    ):
        raise ValueError("Figure receipt has the wrong output format.")
    if int(figure_record.get("size_bytes", -1)) != figure_path.stat().st_size:
        raise ValueError("Figure size no longer matches its receipt.")
    if str(figure_record.get("sha256")) != file_sha256(figure_path):
        raise ValueError("Figure checksum no longer matches its receipt.")
    return provenance


def promote_spyglass_figure(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    artifact_kind: str,
    replace: bool = False,
) -> Path:
    """Publish one validated run-local figure and its receipt."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    source_path = Path(source_path).resolve(strict=True)
    destination_path = Path(destination_path).resolve(strict=False)
    if destination_path.is_relative_to(run_dir):
        raise ValueError("Figure promotion destination must be outside its run.")
    if source_path.suffix.lower() != destination_path.suffix.lower():
        raise ValueError("Promoted figure must retain its source format.")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(
            "Figure promotion destination directory does not exist: "
            f"{destination_path.parent}"
        )
    provenance = _load_validated_figure_provenance(
        payload,
        figure_path=source_path,
        artifact_kind=artifact_kind,
    )
    destination_provenance_path = get_figure_provenance_path(destination_path)
    existing = [
        path
        for path in (destination_path, destination_provenance_path)
        if path.exists()
    ]
    if existing and not replace:
        raise FileExistsError(
            "Refusing to replace promoted figure artifact(s): "
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
            raise ValueError("Staged figure checksum differs from its source.")
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
            try:
                os.link(staged_provenance, destination_provenance_path)
            except BaseException:
                destination_path.unlink(missing_ok=True)
                raise
            staged_provenance.unlink()
    finally:
        staged_figure.unlink(missing_ok=True)
        staged_provenance.unlink(missing_ok=True)
    if file_sha256(destination_path) != str(provenance["figure"]["sha256"]):
        raise ValueError("Promoted figure checksum verification failed.")
    return destination_path


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_SUFFIX",
    "get_figure_provenance_path",
    "promote_spyglass_figure",
    "write_figure_provenance",
]
