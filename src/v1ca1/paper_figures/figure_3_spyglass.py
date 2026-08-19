"""Render current Figure 3 from a completed offline Figure 2 campaign."""

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

from v1ca1.paper_figures import _dark_light
from v1ca1.paper_figures import figure_2_spyglass as figure_2_adapter
from v1ca1.paper_figures import figure_3 as canonical
from v1ca1.paper_figures._spyglass_figure_artifact import (
    get_figure_provenance_path,
    promote_spyglass_figure,
    write_figure_provenance,
)
from v1ca1.paper_figures.datasets import normalize_dataset_id
from v1ca1.spyglass import swap_glm
from v1ca1.spyglass.offline.manifests import DEFAULT_SCRATCH_ROOT


DEFAULT_OUTPUT_NAME = "figure_3_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = figure_2_adapter.EXPECTED_DATASETS
LIGHT_TRAIN_EPOCH = figure_2_adapter.LIGHT_EPOCH
LIGHT_TEST_EPOCH = figure_2_adapter.HELDOUT_LIGHT_EPOCH
REGION = figure_2_adapter.REGION
MINIMUM_MOVEMENT_FIRING_RATE_HZ = (
    figure_2_adapter.MINIMUM_MOVEMENT_FIRING_RATE_HZ
)
MINIMUM_STABILITY_CORRELATION = figure_2_adapter.MINIMUM_STABILITY_CORRELATION
MULTIPLICATIVE_MODEL = canonical._figure_2.PANEL_C_SWAP_MODEL_NAME
ADDITIVE_MODEL = canonical.PANEL_B_ADDITIVE_MODEL_NAME
REQUIRED_MODELS = (MULTIPLICATIVE_MODEL, ADDITIVE_MODEL)
FIGURE_ARTIFACT_KIND = "complete_spyglass_figure_3"

DELTA_TABLE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "model_name",
    "trajectory",
    "unit",
    "delta_ll_bits_per_spike",
    "source_path",
)


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


def _mapped_swap_dataset(
    result: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
) -> tuple[Any, pd.DataFrame]:
    """Map group-unit coordinates to manuscript sorting-unit identifiers."""
    selected = _identity_columns(
        result["selected_units"],
        label="SwapGLM selected_units",
    )
    dataset = result["dataset"]
    group_unit_ids = np.asarray(dataset.coords["unit"].values).astype(str)
    if not np.array_equal(
        group_unit_ids,
        selected["group_unit_id"].to_numpy(dtype=str),
    ):
        raise ValueError("SwapGLM dataset and selected-unit audit disagree.")

    sorting_unit_by_nwb_id = figure_2_adapter._load_nwb_sorting_unit_map(session)
    nwb_unit_ids = selected["unit_id"].to_numpy(dtype=str)
    missing = sorted(set(nwb_unit_ids).difference(sorting_unit_by_nwb_id))
    if missing:
        raise ValueError(
            "SwapGLM selected units are absent from the NWB unit map: "
            f"{missing!r}."
        )
    sorting_unit_ids = pd.to_numeric(
        [sorting_unit_by_nwb_id[value] for value in nwb_unit_ids],
        errors="coerce",
    )
    if np.any(pd.isna(sorting_unit_ids)) or len(set(sorting_unit_ids)) != len(
        sorting_unit_ids
    ):
        raise ValueError("Figure 3 requires unique numeric sorting unit IDs.")
    return (
        dataset.assign_coords(
            unit=np.asarray(sorting_unit_ids, dtype=int),
            nwb_unit_id=("unit", nwb_unit_ids),
        ),
        selected,
    )


def _require_swap_models(dataset: Any, *, source_path: Path) -> None:
    """Require the two predictions displayed and compared in Figure 3."""
    if "model" not in dataset.coords:
        raise ValueError(
            "Figure 3 SwapGLM dataset has no model coordinate: "
            f"{source_path}"
        )
    available = {
        str(value) for value in np.asarray(dataset.coords["model"].values).reshape(-1)
    }
    missing = [model for model in REQUIRED_MODELS if model not in available]
    if missing:
        raise ValueError(
            "Figure 3 requires SwapGLM models "
            f"{REQUIRED_MODELS!r}; {source_path} is missing {missing!r}. "
            "Recompute the Figure 2 campaign with the additive SwapGLM output."
        )
    if swap_glm.PRIMARY_METRIC not in dataset:
        raise ValueError(
            "Figure 3 SwapGLM dataset is missing its primary score "
            f"{swap_glm.PRIMARY_METRIC!r}: {source_path}"
        )


def _load_session_swap_result(
    run_dir: Path,
    session: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Load, map, and filter one session's de novo SwapGLM artifact."""
    record = figure_2_adapter._one_record(
        session["artifacts"].get("swap_glm", ()),
        label="SwapGLM artifact",
    )
    if str(record.get("artifact_origin", "")) != "computed":
        raise ValueError("Figure 3 requires de novo computed SwapGLM artifacts.")
    manifest_path = figure_2_adapter._record_artifact_path(
        record,
        "artifact_manifest_path",
        run_dir=run_dir,
    )
    result = swap_glm.load_swap_glm_artifact(manifest_path.parent)
    metadata = dict(result["metadata"])
    expected = {
        "animal_name": str(session["animal_name"]),
        "date": str(session["date"]),
        "region": REGION,
        "dark_epoch": str(session["epochs"]["dark"]),
        "light_train_epoch": LIGHT_TRAIN_EPOCH,
        "light_test_epoch": LIGHT_TEST_EPOCH,
    }
    if any(str(metadata.get(name)) != value for name, value in expected.items()):
        raise ValueError("SwapGLM metadata does not describe the Figure 3 transfer.")
    if str(result.get("artifact_origin", "")) != "computed":
        raise ValueError("Loaded Figure 3 SwapGLM is not a de novo artifact.")

    dataset, selected = _mapped_swap_dataset(result, session=session)
    _require_swap_models(dataset, source_path=manifest_path)
    dark_movement = figure_2_adapter._load_movement_table(
        session,
        epoch=expected["dark_epoch"],
        run_dir=run_dir,
        scratch_root=scratch_root,
    )
    dark_stability = figure_2_adapter._load_stability_tables(
        session,
        epoch=expected["dark_epoch"],
        run_dir=run_dir,
        scratch_root=scratch_root,
    )
    eligible = figure_2_adapter._eligible_units(
        dark_movement,
        dark_stability,
        minimum_movement_firing_rate_hz=MINIMUM_MOVEMENT_FIRING_RATE_HZ,
        minimum_stability_correlation=MINIMUM_STABILITY_CORRELATION,
    )
    return {
        "dataset": dataset,
        "selected_units": selected,
        "eligible_unit_mask": selected["stable_unit_id"].isin(eligible).to_numpy(),
        "metadata": metadata,
        "source_path": manifest_path,
    }


def _delta_table(
    loaded_by_session: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    model_name: str,
) -> pd.DataFrame:
    """Return one dark-quality-filtered held-out score table."""
    rows: list[dict[str, Any]] = []
    for (animal_name, date), loaded in loaded_by_session.items():
        dataset = loaded["dataset"]
        delta = np.asarray(
            dataset[swap_glm.PRIMARY_METRIC].sel(model=model_name).values,
            dtype=float,
        )
        trajectories = np.asarray(dataset.coords["trajectory"].values).astype(str)
        units = np.asarray(dataset.coords["unit"].values, dtype=int)
        eligible = np.asarray(loaded["eligible_unit_mask"], dtype=bool)
        if delta.shape != (len(trajectories), len(units)) or eligible.shape != (
            len(units),
        ):
            raise ValueError("SwapGLM score or eligibility dimensions are stale.")
        metadata = loaded["metadata"]
        for trajectory_index, trajectory in enumerate(trajectories):
            for unit_index, unit_id in enumerate(units):
                value = float(delta[trajectory_index, unit_index])
                if not eligible[unit_index] or not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "animal_name": animal_name,
                        "date": date,
                        "region": REGION,
                        "dark_epoch": str(metadata["dark_epoch"]),
                        "light_train_epoch": LIGHT_TRAIN_EPOCH,
                        "light_test_epoch": LIGHT_TEST_EPOCH,
                        "model_name": model_name,
                        "trajectory": trajectory,
                        "unit": int(unit_id),
                        "delta_ll_bits_per_spike": value,
                        "source_path": str(loaded["source_path"]),
                    }
                )
    return pd.DataFrame.from_records(rows, columns=DELTA_TABLE_COLUMNS)


def _swap_examples(
    loaded_by_session: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the four configured examples in the canonical display order."""
    multiplicative_examples = []
    additive_examples = []
    for animal_name, date, region, unit_id, trajectory in (
        canonical.PANEL_B_SWAP_EXAMPLES
    ):
        if str(region) != REGION:
            raise ValueError(
                f"Figure 3 example requests unsupported region {region!r}."
            )
        session_key = (str(animal_name), str(date))
        if session_key not in loaded_by_session:
            raise ValueError(f"Figure 3 example session is absent: {session_key!r}.")
        loaded = loaded_by_session[session_key]
        dataset = loaded["dataset"]
        trajectories = np.asarray(dataset.coords["trajectory"].values).astype(str)
        units = np.asarray(dataset.coords["unit"].values, dtype=int)
        if str(trajectory) not in set(trajectories) or int(unit_id) not in set(units):
            raise ValueError(
                "Configured Figure 3 SwapGLM example is absent: "
                f"{animal_name} {unit_id} {trajectory}."
            )
        trajectory_index = int(np.flatnonzero(trajectories == str(trajectory))[0])
        unit_index = int(np.flatnonzero(units == int(unit_id))[0])
        example_kwargs = {
            "dataset_obj": dataset,
            "animal_name": str(animal_name),
            "date": str(date),
            "region": REGION,
            "dark_epoch": str(loaded["metadata"]["dark_epoch"]),
            "light_train_epoch": LIGHT_TRAIN_EPOCH,
            "light_test_epoch": LIGHT_TEST_EPOCH,
            "source_path": Path(loaded["source_path"]),
            "trajectory_index": trajectory_index,
            "unit_index": unit_index,
        }
        multiplicative = _dark_light._panel_h_swap_example_from_indices(
            **example_kwargs,
            model_name=MULTIPLICATIVE_MODEL,
        )
        additive = _dark_light._panel_h_swap_example_from_indices(
            **example_kwargs,
            model_name=ADDITIVE_MODEL,
        )
        if not np.isfinite(float(multiplicative["delta_ll_bits_per_spike"])) or (
            not np.isfinite(float(additive["delta_ll_bits_per_spike"]))
        ):
            raise ValueError(
                "Configured Figure 3 SwapGLM example has a nonfinite score: "
                f"{animal_name} {unit_id} {trajectory}."
            )
        multiplicative_examples.append(multiplicative)
        additive_examples.append(additive)
    return canonical._merge_additive_predictions(
        multiplicative_examples,
        additive_examples,
    )


def _build_panel_data(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> dict[str, Any]:
    """Build exactly the three canonical Figure 3 loader outputs."""
    loaded_by_session = {
        (str(session["animal_name"]), str(session["date"])): (
            _load_session_swap_result(
                run_dir,
                session,
                scratch_root=scratch_root,
            )
        )
        for session in sessions
    }
    return {
        "swap_delta": _delta_table(
            loaded_by_session,
            model_name=MULTIPLICATIVE_MODEL,
        ),
        "swap_additive_delta": _delta_table(
            loaded_by_session,
            model_name=ADDITIVE_MODEL,
        ),
        "swap_examples": _swap_examples(loaded_by_session),
    }


def load_figure_3_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load current Figure 3 inputs from the matching Figure 2 campaign."""
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
        raise ValueError("Figure 3 must use a completed Figure 2 offline campaign.")
    summaries = campaign.get("sessions", ())
    if not summaries or any(
        str(summary.get("status")) != "complete" for summary in summaries
    ):
        raise ValueError("Figure 3 requires every Figure 2 session to be complete.")
    sessions = figure_2_adapter._ordered_sessions(unordered_sessions)
    panel_data = _build_panel_data(
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
        **panel_data,
    }


def _require_canonical_request(
    *,
    data_root: Path,
    datasets: Sequence[Any],
    regions: Sequence[str],
    dark_epoch: str | None,
    payload: Mapping[str, Any],
) -> None:
    """Reject renderer requests outside the selected campaign contract."""
    if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
        strict=True
    ):
        raise ValueError("Canonical Figure 3 requested a foreign data root.")
    observed_datasets = tuple(normalize_dataset_id(value) for value in datasets)
    if observed_datasets != tuple(payload["datasets"]):
        raise ValueError("Canonical Figure 3 requested foreign sessions.")
    if tuple(str(region) for region in regions) != tuple(payload["regions"]):
        raise ValueError("Canonical Figure 3 requested foreign regions.")
    if dark_epoch is not None:
        raise ValueError("Canonical Figure 3 requested one global dark epoch.")


@contextmanager
def _offline_panel_data(payload: Mapping[str, Any]):
    """Inject only the canonical panel-data loader during rendering."""
    original = canonical.load_figure_3_panel_data

    def load_panel_data(
        *,
        data_root: Path,
        datasets: Sequence[Any],
        regions: Sequence[str],
        dark_epoch: str | None,
    ) -> dict[str, Any]:
        _require_canonical_request(
            data_root=data_root,
            datasets=datasets,
            regions=regions,
            dark_epoch=dark_epoch,
            payload=payload,
        )
        return {
            "swap_delta": payload["swap_delta"],
            "swap_additive_delta": payload["swap_additive_delta"],
            "swap_examples": payload["swap_examples"],
        }

    canonical.load_figure_3_panel_data = load_panel_data
    try:
        yield
    finally:
        canonical.load_figure_3_panel_data = original


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the canonical run-local Figure 3 output path."""
    output_format = str(output_format).lower()
    if output_format not in canonical.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {canonical.FIGURE_FORMATS!r}."
        )
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def promote_figure_3(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Publish a validated run-local Figure 3 and its receipt."""
    return promote_spyglass_figure(
        payload,
        source_path=source_path,
        destination_path=destination_path,
        artifact_kind=FIGURE_ARTIFACT_KIND,
        replace=replace,
    )


def render_figure_3(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Atomically render current Figure 3 inside its Figure 2 campaign."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Figure 3 output must remain inside its campaign run.")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Figure 3 output: {output_path}"
        )
    provenance_path = get_figure_provenance_path(output_path)
    if provenance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Figure 3 provenance: {provenance_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in canonical.FIGURE_FORMATS:
        raise ValueError("Figure 3 output has an unsupported format.")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    linked_output = False
    try:
        with _offline_panel_data(payload):
            rendered = canonical.make_figure_3(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                regions=payload["regions"],
                dark_epoch=None,
                dpi=int(dpi),
            )
        if Path(rendered).resolve(strict=True) != temporary_path:
            raise ValueError("Figure 3 renderer returned an unexpected output path.")
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
    """Parse the database-free current Figure 3 renderer arguments."""
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
        help=(
            "Publish the validated artifact and receipt to this path."
        ),
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
    """Load one Figure 2 campaign and render current Figure 3."""
    args = parse_arguments(argv)
    payload = load_figure_3_payload(
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
    path = render_figure_3(payload, output_path=output_path, dpi=args.dpi)
    print(f"Saved current offline Spyglass Figure 3 to {path}")
    if args.promote_to is not None:
        promoted = promote_figure_3(
            payload,
            source_path=path,
            destination_path=args.promote_to,
            replace=args.replace_promoted_output,
        )
        print(f"Promoted validated Spyglass Figure 3 to {promoted}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_figure_3_payload",
    "main",
    "promote_figure_3",
    "render_figure_3",
]
