"""Render Supplementary Figure 5 from a retained offline Figure 3 campaign."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any
import uuid

from v1ca1.paper_figures import figure_4_spyglass as figure_4_adapter
from v1ca1.paper_figures import supplementary_figure_5 as canonical
from v1ca1.paper_figures._spyglass_figure_artifact import (
    get_figure_provenance_path,
    promote_spyglass_figure,
    write_figure_provenance,
)
from v1ca1.spyglass.offline.manifests import DEFAULT_SCRATCH_ROOT


DEFAULT_OUTPUT_NAME = "supplementary_figure_5_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = figure_4_adapter.EXPECTED_DATASETS
LIGHT_EPOCH = figure_4_adapter.LIGHT_EPOCH
REGIONS = figure_4_adapter.REGIONS
RIPPLE_WINDOW_S = figure_4_adapter.legacy.DEFAULT_RIPPLE_WINDOW_S
RIPPLE_WINDOW_OFFSET_S = (
    figure_4_adapter.legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S
)
RIPPLE_SELECTION = (
    figure_4_adapter.legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION
)
RIDGE_STRENGTH = figure_4_adapter.legacy.DEFAULT_RIDGE_STRENGTH
FIGURE_ARTIFACT_KIND = "complete_spyglass_supplementary_figure_5"


class _UnexpectedLegacyRequest(RuntimeError):
    """Signal a source request outside the retained campaign contract."""


def load_supplementary_figure_5_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load only the retained artifacts used by Supplementary Figure 5."""
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
        raise ValueError("Selected campaign is not a retained Figure 3 run.")
    sessions = figure_4_adapter._ordered_sessions(unordered_sessions)
    unit_maps = {
        (str(session["animal_name"]), str(session["date"])): (
            figure_4_adapter._load_nwb_sorting_unit_maps(session)
        )
        for session in sessions
    }
    glm_results = figure_4_adapter._load_glm_results(
        run_dir,
        sessions,
        unit_maps,
    )
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "sessions": sessions,
        "datasets": EXPECTED_DATASETS,
        "regions": REGIONS,
        "heatmap_epoch_tables": (
            figure_4_adapter._load_modulation_epoch_tables(
                run_dir,
                sessions,
                unit_maps,
            )
        ),
        "source_comparison_payload": (
            figure_4_adapter._build_source_comparison_payload(
                sessions,
                glm_results,
            )
        ),
    }


def _require_common_request(
    data_root: Path,
    datasets: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
) -> None:
    """Reject renderer requests outside the fixed light-only campaign."""
    try:
        figure_4_adapter._require_common_request(
            data_root,
            datasets,
            kwargs,
            payload=payload,
        )
    except figure_4_adapter._UnexpectedLegacyRequest as exc:
        raise _UnexpectedLegacyRequest(str(exc)) from exc


def _require_glm_settings(kwargs: Mapping[str, Any]) -> None:
    """Reject renderer requests for GLM settings absent from the campaign."""
    try:
        figure_4_adapter._require_glm_settings(kwargs)
    except figure_4_adapter._UnexpectedLegacyRequest as exc:
        raise _UnexpectedLegacyRequest(str(exc)) from exc


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject the two validated payloads bound by the canonical renderer."""
    original_heatmaps = canonical.load_pooled_ripple_heatmap_epoch_tables
    original_source_comparison = (
        canonical.load_glm_source_predictor_comparison_tables
    )

    def load_heatmaps(
        data_root: Path,
        datasets: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        _require_common_request(
            data_root,
            datasets,
            kwargs,
            payload=payload,
        )
        if kwargs.get("ripple_threshold_zscore") is not None:
            raise _UnexpectedLegacyRequest(
                "Supplementary Figure 5 cannot rethreshold retained ripples."
            )
        return payload["heatmap_epoch_tables"]

    def load_source_comparison(
        data_root: Path,
        datasets: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        _require_common_request(
            data_root,
            datasets,
            kwargs,
            payload=payload,
        )
        _require_glm_settings(kwargs)
        if tuple(kwargs.get("epoch_types", ())) != tuple(
            canonical.PANEL_E_GLM_EPOCH_ORDER
        ):
            raise _UnexpectedLegacyRequest(
                "Supplementary Figure 5 requested foreign GLM epochs."
            )
        return payload["source_comparison_payload"]

    canonical.load_pooled_ripple_heatmap_epoch_tables = load_heatmaps
    canonical.load_glm_source_predictor_comparison_tables = (
        load_source_comparison
    )
    try:
        yield
    finally:
        canonical.load_pooled_ripple_heatmap_epoch_tables = original_heatmaps
        canonical.load_glm_source_predictor_comparison_tables = (
            original_source_comparison
        )


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the canonical run-local Supplementary Figure 5 output path."""
    output_format = str(output_format).lower()
    if output_format not in canonical.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {canonical.FIGURE_FORMATS!r}."
        )
    return (
        Path(run_dir)
        / "figures"
        / f"{DEFAULT_OUTPUT_NAME}.{output_format}"
    )


def promote_supplementary_figure_5(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Publish a validated Supplementary Figure 5 and its receipt."""
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
        raise ValueError(
            "Supplementary Figure 5 output must remain inside its campaign run."
        )
    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite Supplementary Figure 5 output: "
            f"{output_path}"
        )
    provenance_path = get_figure_provenance_path(output_path)
    if provenance_path.exists():
        raise FileExistsError(
            "Refusing to overwrite Supplementary Figure 5 provenance: "
            f"{provenance_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in canonical.FIGURE_FORMATS:
        raise ValueError("Supplementary Figure 5 output has an unsupported format.")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    linked_output = False
    try:
        with _offline_sources(payload):
            rendered = canonical.make_supplementary_figure_5(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                regions=payload["regions"],
                light_epoch=LIGHT_EPOCH,
                dark_epoch=None,
                sleep_epoch=None,
                ripple_threshold_zscore=None,
                ripple_window_s=RIPPLE_WINDOW_S,
                ripple_window_offset_s=RIPPLE_WINDOW_OFFSET_S,
                ripple_selection=RIPPLE_SELECTION,
                ridge_strength=RIDGE_STRENGTH,
                dpi=int(dpi),
            )
        if Path(rendered).resolve(strict=True) != temporary_path:
            raise ValueError(
                "Supplementary Figure 5 renderer returned an unexpected path."
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
    """Parse database-free Supplementary Figure 5 renderer arguments."""
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
        "--promote-to",
        type=Path,
        help="Publish the validated artifact and receipt to this path.",
    )
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
    """Load one Figure 3 campaign and render Supplementary Figure 5."""
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
    print(f"Saved offline Spyglass Supplementary Figure 5 to {path}")
    if args.promote_to is not None:
        promoted = promote_supplementary_figure_5(
            payload,
            source_path=path,
            destination_path=args.promote_to,
            replace=args.replace_promoted_output,
        )
        print(f"Promoted validated Supplementary Figure 5 to {promoted}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "FIGURE_ARTIFACT_KIND",
    "get_output_path",
    "load_supplementary_figure_5_payload",
    "main",
    "promote_supplementary_figure_5",
    "render_supplementary_figure_5",
]
