"""Render Supplementary Figure 5 from retained Spyglass motor artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from v1ca1.paper_figures import figure_2_spyglass as figure_2_adapter
from v1ca1.paper_figures import supplementary_figure_5 as canonical
from v1ca1.paper_figures._spyglass_figure_artifact import (
    get_figure_provenance_path,
    promote_spyglass_figure,
    write_figure_provenance,
)
from v1ca1.paper_figures.datasets import normalize_dataset_id
from v1ca1.spyglass import epoch_motor_behavior
from v1ca1.spyglass.offline.manifests import DEFAULT_SCRATCH_ROOT


DEFAULT_OUTPUT_NAME = "supplementary_figure_5_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = figure_2_adapter.EXPECTED_DATASETS
LIGHT_EPOCH = canonical.MOTOR_PANEL_LIGHT_EPOCH
FIGURE_ARTIFACT_KIND = "complete_spyglass_supplementary_figure_5"


def _artifact_manifest_path(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
) -> Path:
    """Resolve one checksum-bearing motor artifact manifest."""
    for field in ("artifact_manifest_path", "manifest_path"):
        try:
            return figure_2_adapter._record_artifact_path(
                record,
                field,
                run_dir=run_dir,
            )
        except (KeyError, ValueError):
            continue
    raise ValueError("EpochMotorBehavior record has no artifact manifest path.")


def _build_motor_progression_table(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Pool dark and light progression summaries from validated bundles."""
    tables: list[pd.DataFrame] = []
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        dark_epoch = str(session["epochs"]["dark"])
        for epoch_type, epoch in (("dark", dark_epoch), ("light", LIGHT_EPOCH)):
            record = figure_2_adapter._one_record(
                session["artifacts"].get("epoch_motor_behavior", ()),
                label="EpochMotorBehavior artifact",
                epoch=epoch,
            )
            manifest_path = _artifact_manifest_path(record, run_dir=run_dir)
            loaded = epoch_motor_behavior.load_epoch_motor_behavior_artifact(
                manifest_path.parent
            )
            metadata = loaded["metadata"]
            expected = {
                "animal_name": animal_name,
                "date": date,
                "epoch": epoch,
            }
            if any(
                str(metadata.get(name)) != value
                for name, value in expected.items()
            ):
                raise ValueError(
                    "EpochMotorBehavior metadata disagrees with its session."
                )
            if str(loaded.get("artifact_origin", "")) != "computed":
                raise ValueError(
                    "Supplementary Figure 5 requires computed motor artifacts."
                )
            table = loaded["progression_summary"].copy()
            table["animal_name"] = animal_name
            table["date"] = date
            table["dark_epoch"] = dark_epoch
            table["light_epoch"] = LIGHT_EPOCH
            table["dataset_label"] = f"{animal_name} {date}"
            table["epoch_type"] = epoch_type
            table["source_path"] = str(manifest_path)
            tables.append(table)
    if not tables:
        raise ValueError("Supplementary Figure 5 has no motor artifacts.")
    output = pd.concat(tables, ignore_index=True, sort=False)
    required = set(canonical.MOTOR_PANEL_COLUMNS).union(
        {
            "animal_name",
            "date",
            "dark_epoch",
            "light_epoch",
            "dataset_label",
            "epoch_type",
            "source_path",
        }
    )
    missing = sorted(required.difference(output.columns))
    if missing:
        raise ValueError(
            "EpochMotorBehavior progression summaries are missing columns "
            f"{missing!r}."
        )
    return output


def load_supplementary_figure_5_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load the active Supplementary Figure 5 inputs from one campaign."""
    from v1ca1.spyglass.offline.supplementary_figures import (
        SUPPLEMENTARY_FIGURES_PIPELINE,
        load_supplementary_figures_campaign,
    )

    run_dir, campaign, unordered_sessions = load_supplementary_figures_campaign(
        run_id,
        scratch_root=scratch_root,
    )
    if str(campaign.get("analysis_parameters", {}).get("pipeline")) != (
        SUPPLEMENTARY_FIGURES_PIPELINE
    ):
        raise ValueError("Selected run is not a supplementary-figures campaign.")
    summaries = campaign.get("sessions", ())
    if not summaries or any(
        str(summary.get("status")) != "complete" for summary in summaries
    ):
        raise ValueError("Every supplementary campaign session must be complete.")
    sessions = figure_2_adapter._ordered_sessions(unordered_sessions)
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "sessions": sessions,
        "datasets": EXPECTED_DATASETS,
        "regions": (),
        "motor_progression_table": _build_motor_progression_table(
            run_dir,
            sessions,
        ),
    }


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject only the active motor-table loader into the renderer."""
    original = canonical.load_panel_b_motor_progression_table

    def load_motor(
        *,
        data_root: Path,
        datasets: Sequence[Any],
        light_epoch: str = LIGHT_EPOCH,
    ) -> pd.DataFrame:
        if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
            strict=True
        ):
            raise ValueError("Supplementary Figure 5 requested a foreign root.")
        observed = tuple(normalize_dataset_id(value) for value in datasets)
        if observed != tuple(payload["datasets"]):
            raise ValueError("Supplementary Figure 5 requested foreign sessions.")
        if str(light_epoch) != LIGHT_EPOCH:
            raise ValueError("Supplementary Figure 5 requested a foreign light epoch.")
        return payload["motor_progression_table"]

    canonical.load_panel_b_motor_progression_table = load_motor
    try:
        yield
    finally:
        canonical.load_panel_b_motor_progression_table = original


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the immutable run-local Supplementary Figure 5 path."""
    output_format = str(output_format).lower()
    if output_format not in canonical.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {canonical.FIGURE_FORMATS!r}."
        )
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def promote_supplementary_figure_5(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Publish one validated Supplementary Figure 5 and receipt."""
    return promote_spyglass_figure(
        payload,
        source_path=source_path,
        destination_path=destination_path,
        artifact_kind=FIGURE_ARTIFACT_KIND,
        replace=replace,
    )


def render_supplementary_figure_5(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Atomically render Supplementary Figure 5 inside its campaign."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Output must remain inside its supplementary campaign.")
    provenance_path = get_figure_provenance_path(output_path)
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Supplementary Figure 5: {output_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in canonical.FIGURE_FORMATS:
        raise ValueError("Unsupported Supplementary Figure 5 output format.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    linked = False
    try:
        with _offline_sources(payload):
            rendered = canonical.make_supplementary_figure_5(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                dpi=int(dpi),
            )
        if Path(rendered).resolve(strict=True) != temporary_path:
            raise ValueError("Renderer returned an unexpected output path.")
        os.link(temporary_path, output_path)
        linked = True
        temporary_path.unlink()
        write_figure_provenance(
            payload,
            figure_path=output_path,
            artifact_kind=FIGURE_ARTIFACT_KIND,
        )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if linked:
            output_path.unlink(missing_ok=True)
            provenance_path.unlink(missing_ok=True)
        raise
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Supplementary Figure 5 Spyglass renderer arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument(
        "--output-format",
        choices=canonical.FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--replace-promoted-output", action="store_true")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args(argv)
    if args.replace_promoted_output and args.promote_to is None:
        parser.error("--replace-promoted-output requires --promote-to.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Load one supplementary campaign and render Supplementary Figure 5."""
    args = parse_arguments(argv)
    payload = load_supplementary_figure_5_payload(
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
    path = render_supplementary_figure_5(
        payload,
        output_path=output_path,
        dpi=args.dpi,
    )
    print(f"Saved Spyglass Supplementary Figure 5 to {path}")
    if args.promote_to is not None:
        promoted = promote_supplementary_figure_5(
            payload,
            source_path=path,
            destination_path=args.promote_to,
            replace=args.replace_promoted_output,
        )
        print(f"Promoted Spyglass Supplementary Figure 5 to {promoted}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_supplementary_figure_5_payload",
    "main",
    "promote_supplementary_figure_5",
    "render_supplementary_figure_5",
]
