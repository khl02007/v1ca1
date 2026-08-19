"""Render Supplementary Figure 1 from retained offline Spyglass artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from v1ca1.paper_figures import figure_1_spyglass as figure_1_adapter
from v1ca1.paper_figures import supplementary_figure_1 as canonical
from v1ca1.paper_figures._spyglass_figure_artifact import (
    get_figure_provenance_path,
    promote_spyglass_figure,
    write_figure_provenance,
)
from v1ca1.paper_figures.datasets import normalize_dataset_id
from v1ca1.spyglass.offline.manifests import DEFAULT_SCRATCH_ROOT


DEFAULT_OUTPUT_NAME = "supplementary_figure_1_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = figure_1_adapter.FULL_DATASETS
MOVEMENT_REGION = canonical.DARK_MOVEMENT_FIRING_RATE_REGION
STABILITY_REGIONS = tuple(canonical.STABILITY_REGIONS)
FIGURE_ARTIFACT_KIND = "complete_spyglass_supplementary_figure_1"


def _require_valid_record(record: Mapping[str, Any], *, label: str) -> None:
    """Require a retained artifact record to represent a valid analysis."""
    if str(record.get("analysis_status")) != "valid":
        raise ValueError(f"{label} is not a valid retained analysis artifact.")


def _require_table_identity(
    table: pd.DataFrame,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    label: str,
) -> None:
    """Require one artifact table to match its manifest selection."""
    for column, expected in {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "region": region,
    }.items():
        figure_1_adapter._constant_string_column(
            table,
            column,
            expected,
            label=label,
        )


def _load_session_panel_tables(
    *,
    run_dir: Path,
    session_manifest: Mapping[str, Any],
    epoch: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one session's movement-rate and stability panel rows."""
    animal_name = str(session_manifest["animal_name"])
    date = str(session_manifest["date"])
    artifacts = session_manifest["artifacts"]
    movement_by_region: dict[str, pd.DataFrame] = {}
    unit_digests: dict[str, str] = {}

    for region in STABILITY_REGIONS:
        source_record = figure_1_adapter._matching_record(
            session_manifest["source_identity"],
            expected={"region": region, "source": "ImportedSpikeSorting"},
            label="regional spike source",
        )
        movement_record = figure_1_adapter._matching_record(
            artifacts["movement_firing_rate"],
            expected={"epoch": epoch, "region": region},
            label="movement firing-rate",
        )
        _require_valid_record(
            movement_record,
            label=f"Movement firing-rate record {animal_name} {date} {region}",
        )
        movement_path = figure_1_adapter._record_path(
            run_dir,
            movement_record,
            "firing_rate_path",
            label="Movement firing-rate record",
        )
        movement = figure_1_adapter.load_movement_firing_rate_artifact(
            movement_path
        )
        _require_table_identity(
            movement,
            animal_name=animal_name,
            date=date,
            epoch=str(epoch),
            region=region,
            label="Movement firing-rate artifact",
        )
        required_columns = {
            "unit_id",
            "stable_unit_id",
            "movement_firing_rate_hz",
        }
        missing = sorted(required_columns.difference(movement.columns))
        if missing:
            raise ValueError(
                f"Movement firing-rate artifact is missing columns {missing!r}."
            )
        if movement["stable_unit_id"].astype(str).duplicated().any():
            raise ValueError("Movement firing-rate artifact contains duplicate units.")
        if len(movement) != int(movement_record["n_units"]) or len(
            movement
        ) != int(source_record["n_units"]):
            raise ValueError(
                "Movement artifact and regional spike-source unit counts disagree."
            )
        movement_digest = str(movement_record["selected_units_sha256"])
        source_digest = str(source_record["selected_units_sha256"])
        if movement_digest != source_digest:
            raise ValueError(
                "Movement artifact and regional spike source select different units."
            )
        movement_by_region[region] = movement
        unit_digests[region] = source_digest

    stability_tables = []
    for region in STABILITY_REGIONS:
        expected_units = set(
            movement_by_region[region]["stable_unit_id"].astype(str)
        )
        for trajectory_type in canonical.PANEL_D_TRAJECTORY_TYPES:
            record = figure_1_adapter._matching_record(
                artifacts["path_specific_place_stability"],
                expected={
                    "epoch": epoch,
                    "region": region,
                    "trajectory_type": trajectory_type,
                    "tuning_curve_param_name": (
                        figure_1_adapter.STABILITY_TUNING_PRESET
                    ),
                },
                label="path-specific stability",
            )
            _require_valid_record(
                record,
                label=(
                    "Path-specific stability record "
                    f"{animal_name} {date} {region} {trajectory_type}"
                ),
            )
            if str(record["selected_units_sha256"]) != unit_digests[region]:
                raise ValueError(
                    "Stability artifact and regional spike source select "
                    "different units."
                )
            path = figure_1_adapter._record_path(
                run_dir,
                record,
                "stability_path",
                label="Path-specific stability record",
            )
            table = figure_1_adapter._load_stability_artifact(path)
            _require_table_identity(
                table,
                animal_name=animal_name,
                date=date,
                epoch=str(epoch),
                region=region,
                label="Path-specific stability artifact",
            )
            figure_1_adapter._constant_string_column(
                table,
                "trajectory_type",
                trajectory_type,
                label="Path-specific stability artifact",
            )
            if len(table) != int(record["n_units"]):
                raise ValueError(
                    "Stability artifact and manifest unit counts disagree."
                )
            if set(table["stable_unit_id"].astype(str)) != expected_units:
                raise ValueError(
                    "Stability units do not match the movement firing-rate "
                    "artifact."
                )
            stability_tables.append(table)

    movement_rows = movement_by_region[MOVEMENT_REGION].copy()
    movement_rows["unit"] = movement_rows["unit_id"]
    movement_rows["dark_epoch"] = str(epoch)
    movement_rows["dark_firing_rate_hz"] = movement_rows[
        "movement_firing_rate_hz"
    ]
    stability_rows = pd.concat(stability_tables, ignore_index=True)
    return movement_rows, stability_rows


def load_supplementary_figure_1_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load all active Supplementary Figure 1 data from one Figure 1D run."""
    run_dir, campaign, sessions = (
        figure_1_adapter.load_figure_1_session_manifests(
            scratch_root=Path(scratch_root),
            run_id=run_id,
            mode="full",
        )
    )
    movement_tables = []
    stability_tables = []
    for session, (animal_name, date, epoch) in zip(
        sessions,
        EXPECTED_DATASETS,
        strict=True,
    ):
        if (
            str(session["animal_name"]),
            str(session["date"]),
        ) != (str(animal_name), str(date)):
            raise ValueError("Figure 1D sessions are not in manuscript order.")
        movement, stability = _load_session_panel_tables(
            run_dir=run_dir,
            session_manifest=session,
            epoch=str(epoch),
        )
        movement_tables.append(movement)
        stability_tables.append(stability)
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "datasets": EXPECTED_DATASETS,
        "regions": STABILITY_REGIONS,
        "dark_movement_firing_rate_table": pd.concat(
            movement_tables,
            ignore_index=True,
        ),
        "dark_stability_table": pd.concat(
            stability_tables,
            ignore_index=True,
        ),
    }


def _require_canonical_request(
    *,
    data_root: Path,
    datasets: Sequence[Any],
    payload: Mapping[str, Any],
) -> None:
    """Reject canonical loader requests outside the selected campaign."""
    if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
        strict=True
    ):
        raise ValueError(
            "Canonical Supplementary Figure 1 requested a foreign data root."
        )
    observed = tuple(normalize_dataset_id(value) for value in datasets)
    if observed != tuple(payload["datasets"]):
        raise ValueError(
            "Canonical Supplementary Figure 1 requested foreign sessions."
        )


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject only the two active artifact-backed panel loader seams."""
    original_movement = canonical.load_pooled_dark_movement_firing_rate_table
    original_stability = canonical.load_dark_epoch_stability_table

    def load_movement(
        data_root: Path,
        datasets: Sequence[Any],
        *,
        region: str = MOVEMENT_REGION,
        cache_dir: Path | None = None,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        del cache_dir
        _require_canonical_request(
            data_root=data_root,
            datasets=datasets,
            payload=payload,
        )
        if str(region) != MOVEMENT_REGION:
            raise ValueError(
                "Canonical Supplementary Figure 1 requested a foreign "
                "movement-rate region."
            )
        if refresh_cache:
            raise ValueError(
                "Artifact-backed Supplementary Figure 1 cannot refresh a "
                "legacy cache."
            )
        return payload["dark_movement_firing_rate_table"].copy()

    def load_stability(
        *,
        data_root: Path,
        datasets: Sequence[Any],
        regions: Sequence[str] = STABILITY_REGIONS,
    ) -> pd.DataFrame:
        _require_canonical_request(
            data_root=data_root,
            datasets=datasets,
            payload=payload,
        )
        observed_regions = tuple(str(region) for region in regions)
        if observed_regions != tuple(payload["regions"]):
            raise ValueError(
                "Canonical Supplementary Figure 1 requested foreign "
                "stability regions."
            )
        return payload["dark_stability_table"].copy()

    canonical.load_pooled_dark_movement_firing_rate_table = load_movement
    canonical.load_dark_epoch_stability_table = load_stability
    try:
        yield
    finally:
        canonical.load_pooled_dark_movement_firing_rate_table = original_movement
        canonical.load_dark_epoch_stability_table = original_stability


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the canonical run-local Supplementary Figure 1 output path."""
    output_format = str(output_format).lower()
    if output_format not in canonical.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {canonical.FIGURE_FORMATS!r}."
        )
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def promote_supplementary_figure_1(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Publish a validated Supplementary Figure 1 and its receipt."""
    return promote_spyglass_figure(
        payload,
        source_path=source_path,
        destination_path=destination_path,
        artifact_kind=FIGURE_ARTIFACT_KIND,
        replace=replace,
    )


def render_supplementary_figure_1(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    asset_dir: Path = canonical.DEFAULT_ASSET_DIR,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Atomically render Supplementary Figure 1 inside its campaign run."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError(
            "Supplementary Figure 1 output must remain inside its campaign run."
        )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Supplementary Figure 1 output: {output_path}"
        )
    provenance_path = get_figure_provenance_path(output_path)
    if provenance_path.exists():
        raise FileExistsError(
            "Refusing to overwrite Supplementary Figure 1 provenance: "
            f"{provenance_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in canonical.FIGURE_FORMATS:
        raise ValueError("Supplementary Figure 1 output has an unsupported format.")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    linked_output = False
    try:
        with _offline_sources(payload):
            rendered = canonical.make_supplementary_figure_1(
                data_root=run_dir,
                asset_dir=Path(asset_dir),
                output_path=temporary_path,
                datasets=payload["datasets"],
                dpi=int(dpi),
                dark_movement_fr_cache_dir=run_dir / "figures" / "cache",
                refresh_dark_movement_fr_cache=False,
            )
        if Path(rendered).resolve(strict=True) != temporary_path:
            raise ValueError(
                "Supplementary Figure 1 renderer returned an unexpected "
                "output path."
            )
        os.link(temporary_path, output_path)
        linked_output = True
        temporary_path.unlink()
        write_figure_provenance(
            payload,
            figure_path=output_path,
            artifact_kind=FIGURE_ARTIFACT_KIND,
        )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if linked_output:
            output_path.unlink(missing_ok=True)
            provenance_path.unlink(missing_ok=True)
        raise
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the database-free Supplementary Figure 1 renderer arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    parser.add_argument(
        "--output-format",
        choices=canonical.FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=canonical.DEFAULT_ASSET_DIR,
    )
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument(
        "--replace-promoted-output",
        action="store_true",
        help="Explicitly replace an existing promoted artifact and receipt.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args(argv)
    if args.replace_promoted_output and args.promote_to is None:
        parser.error("--replace-promoted-output requires --promote-to.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Load one Figure 1D campaign and render Supplementary Figure 1."""
    args = parse_arguments(argv)
    payload = load_supplementary_figure_1_payload(
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
    path = render_supplementary_figure_1(
        payload,
        output_path=output_path,
        asset_dir=args.asset_dir,
        dpi=args.dpi,
    )
    print(f"Saved offline Spyglass Supplementary Figure 1 to {path}")
    if args.promote_to is not None:
        promoted = promote_supplementary_figure_1(
            payload,
            source_path=path,
            destination_path=args.promote_to,
            replace=args.replace_promoted_output,
        )
        print(
            "Promoted validated Spyglass Supplementary Figure 1 to "
            f"{promoted}"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_supplementary_figure_1_payload",
    "main",
    "promote_supplementary_figure_1",
    "render_supplementary_figure_1",
]
