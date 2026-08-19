"""Render Supplementary Figure 2 from retained Spyglass artifacts."""

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

from v1ca1.paper_figures import figure_2_spyglass as figure_2_adapter
from v1ca1.paper_figures import supplementary_figure_2 as canonical
from v1ca1.paper_figures._spyglass_figure_artifact import (
    get_figure_provenance_path,
    promote_spyglass_figure,
    write_figure_provenance,
)
from v1ca1.paper_figures.datasets import normalize_dataset_id
from v1ca1.spyglass import cv_pca
from v1ca1.spyglass.offline.manifests import DEFAULT_SCRATCH_ROOT


DEFAULT_OUTPUT_NAME = "supplementary_figure_2_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = figure_2_adapter.EXPECTED_DATASETS
REGION = canonical.DEFAULT_REGION
FIGURE_ARTIFACT_KIND = "complete_spyglass_supplementary_figure_2"


def _artifact_manifest_path(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
) -> Path:
    """Resolve one checksum-bearing cvPCA artifact manifest."""
    for field in ("artifact_manifest_path", "manifest_path"):
        try:
            return figure_2_adapter._record_artifact_path(
                record,
                field,
                run_dir=run_dir,
            )
        except (KeyError, ValueError):
            continue
    raise ValueError("cvPCA record has no artifact manifest path.")


def _build_cv_pca_table(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return one paired dark/light participation-ratio table."""
    rows: list[dict[str, Any]] = []
    for session in sessions:
        animal_name = str(session["animal_name"])
        date = str(session["date"])
        dark_epoch = str(session["epochs"]["dark"])
        record = figure_2_adapter._one_record(
            session["artifacts"].get("cv_pca", ()),
            label="cvPCA artifact",
            region=REGION,
            dark_epoch=dark_epoch,
            light_epoch=figure_2_adapter.LIGHT_EPOCH,
        )
        manifest_path = _artifact_manifest_path(record, run_dir=run_dir)
        loaded = cv_pca.load_cv_pca_artifact(manifest_path.parent)
        expected = {
            "animal_name": animal_name,
            "date": date,
            "region": REGION,
            "dark_epoch": dark_epoch,
            "light_epoch": figure_2_adapter.LIGHT_EPOCH,
        }
        if any(str(loaded.get(name)) != value for name, value in expected.items()):
            raise ValueError("cvPCA metadata disagrees with its session.")
        if str(loaded.get("artifact_origin", "")) != "computed":
            raise ValueError("Supplementary Figure 2 requires computed cvPCA.")
        summary = loaded["summary"]
        for condition in ("dark", "light"):
            selected = summary.loc[summary["condition"].astype(str).eq(condition)]
            if len(selected) != 1:
                raise ValueError(
                    "cvPCA must contain one within-condition row for "
                    f"{condition!r}."
                )
            row = selected.iloc[0]
            value = float(row["within_cv_participation_ratio"])
            if not np.isfinite(value):
                raise ValueError("cvPCA participation ratios must be finite.")
            rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "region": REGION,
                    "dark_epoch": dark_epoch,
                    "light_epoch": figure_2_adapter.LIGHT_EPOCH,
                    "condition": condition,
                    "participation_ratio": value,
                    "n_units": int(row["n_units"]),
                    "source_path": manifest_path,
                }
            )
    output = pd.DataFrame.from_records(rows)
    if len(output) != 2 * len(sessions):
        raise ValueError("cvPCA payload does not cover both conditions per session.")
    return output


def _build_decoding_data(
    parent_run_dir: Path,
    parent_sessions: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Reconstruct decoding summaries and fixed permutation labels."""
    summary, trial = figure_2_adapter._build_panel_e_decoding_tables(
        parent_run_dir,
        parent_sessions,
        scratch_root=scratch_root,
    )
    permutation_results = canonical.figure_2.compute_panel_e_decoding_permutation_tests(
        trial,
        n_permutations=canonical.figure_2.DECODING_PERMUTATION_COUNT,
        seed=canonical.figure_2.DECODING_PERMUTATION_SEED,
    )
    animal_names = tuple(
        str(normalize_dataset_id(dataset)[0]) for dataset in EXPECTED_DATASETS
    )
    labels = canonical.figure_2.build_panel_e_decoding_significance_labels(
        permutation_results,
        animal_names=animal_names,
    )
    return {
        "decoding_error": summary,
        "decoding_significance_labels": labels,
        "decoding_trial_error": trial,
    }


def load_supplementary_figure_2_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load every active Supplementary Figure 2 input."""
    from v1ca1.spyglass.offline.supplementary_figures import (
        SUPPLEMENTARY_FIGURES_PIPELINE,
        load_parent_figure_2_sessions,
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
    parent_run_dir, parent_sessions = load_parent_figure_2_sessions(
        sessions,
        scratch_root=scratch_root,
    )
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "sessions": sessions,
        "datasets": EXPECTED_DATASETS,
        "regions": (REGION,),
        "cv_pca_table": _build_cv_pca_table(run_dir, sessions),
        "decoding_data": _build_decoding_data(
            parent_run_dir,
            parent_sessions,
            scratch_root=scratch_root,
        ),
    }


def _require_request(
    *,
    data_root: Path,
    datasets: Sequence[Any],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    payload: Mapping[str, Any],
) -> None:
    """Reject requests outside the fixed supplementary campaign."""
    if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
        strict=True
    ):
        raise ValueError("Supplementary Figure 2 requested a foreign root.")
    observed = tuple(normalize_dataset_id(value) for value in datasets)
    if observed != tuple(payload["datasets"]):
        raise ValueError("Supplementary Figure 2 requested foreign sessions.")
    if str(region) != REGION or light_epoch is not None or dark_epoch is not None:
        raise ValueError("Supplementary Figure 2 requested foreign epoch settings.")


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject only the two active canonical data loaders."""
    original_decoding = canonical.load_panel_a_decoding_data
    original_cv_pca = (
        canonical.supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table
    )

    def load_decoding(
        *,
        data_root: Path,
        datasets: Sequence[Any],
        region: str,
        light_epoch: str | None,
        dark_epoch: str | None,
        decoding_n_permutations: int,
        decoding_permutation_seed: int,
    ) -> dict[str, Any]:
        _require_request(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            payload=payload,
        )
        if int(decoding_n_permutations) != int(
            canonical.figure_2.DECODING_PERMUTATION_COUNT
        ) or int(decoding_permutation_seed) != int(
            canonical.figure_2.DECODING_PERMUTATION_SEED
        ):
            raise ValueError("Supplementary Figure 2 requested foreign permutations.")
        return payload["decoding_data"]

    def load_cv_pca(
        *,
        data_root: Path,
        datasets: Sequence[Any],
    ) -> pd.DataFrame:
        _require_request(
            data_root=data_root,
            datasets=datasets,
            region=REGION,
            light_epoch=None,
            dark_epoch=None,
            payload=payload,
        )
        return payload["cv_pca_table"]

    canonical.load_panel_a_decoding_data = load_decoding
    canonical.supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table = (
        load_cv_pca
    )
    try:
        yield
    finally:
        canonical.load_panel_a_decoding_data = original_decoding
        canonical.supplementary_figure_3.load_panel_a_cv_pca_participation_ratio_table = (
            original_cv_pca
        )


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the immutable run-local Supplementary Figure 2 path."""
    output_format = str(output_format).lower()
    if output_format not in canonical.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {canonical.FIGURE_FORMATS!r}."
        )
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def promote_supplementary_figure_2(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Publish one validated Supplementary Figure 2 and receipt."""
    return promote_spyglass_figure(
        payload,
        source_path=source_path,
        destination_path=destination_path,
        artifact_kind=FIGURE_ARTIFACT_KIND,
        replace=replace,
    )


def render_supplementary_figure_2(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Atomically render Supplementary Figure 2 inside its campaign."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Output must remain inside its supplementary campaign.")
    provenance_path = get_figure_provenance_path(output_path)
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Supplementary Figure 2: {output_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in canonical.FIGURE_FORMATS:
        raise ValueError("Unsupported Supplementary Figure 2 output format.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    linked = False
    try:
        with _offline_sources(payload):
            rendered = canonical.make_supplementary_figure_2(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                region=REGION,
                light_epoch=None,
                dark_epoch=None,
                dpi=int(dpi),
                decoding_n_permutations=(
                    canonical.figure_2.DECODING_PERMUTATION_COUNT
                ),
                decoding_permutation_seed=(
                    canonical.figure_2.DECODING_PERMUTATION_SEED
                ),
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
    """Parse Supplementary Figure 2 Spyglass renderer arguments."""
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
    """Load one supplementary campaign and render Supplementary Figure 2."""
    args = parse_arguments(argv)
    payload = load_supplementary_figure_2_payload(
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
    path = render_supplementary_figure_2(
        payload,
        output_path=output_path,
        dpi=args.dpi,
    )
    print(f"Saved Spyglass Supplementary Figure 2 to {path}")
    if args.promote_to is not None:
        promoted = promote_supplementary_figure_2(
            payload,
            source_path=path,
            destination_path=args.promote_to,
            replace=args.replace_promoted_output,
        )
        print(f"Promoted Spyglass Supplementary Figure 2 to {promoted}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_supplementary_figure_2_payload",
    "main",
    "promote_supplementary_figure_2",
    "render_supplementary_figure_2",
]
