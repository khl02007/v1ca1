"""Render Supplementary Figure 6 from retained Spyglass artifacts."""

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
from v1ca1.paper_figures import figure_3_spyglass as figure_3_adapter
from v1ca1.paper_figures import supplementary_figure_6 as canonical
from v1ca1.paper_figures._spyglass_figure_artifact import (
    get_figure_provenance_path,
    promote_spyglass_figure,
    write_figure_provenance,
)
from v1ca1.paper_figures.datasets import normalize_dataset_id
from v1ca1.spyglass import swap_glm, swap_tuning
from v1ca1.spyglass.offline.manifests import DEFAULT_SCRATCH_ROOT


DEFAULT_OUTPUT_NAME = "supplementary_figure_6_spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
EXPECTED_DATASETS = figure_2_adapter.EXPECTED_DATASETS
REGION = figure_2_adapter.REGION
LIGHT_TRAIN_EPOCH = figure_2_adapter.LIGHT_EPOCH
LIGHT_TEST_EPOCH = figure_2_adapter.HELDOUT_LIGHT_EPOCH
SCALAR_MODEL = canonical.SCALAR_MODEL_NAME
EMPIRICAL_ADDITIVE_MODEL = canonical.MIXED_EMPIRICAL_AD_MODEL_NAME
EMPIRICAL_ADDITIVE_LABEL = canonical.MIXED_EMPIRICAL_AD_LABEL
FIGURE_ARTIFACT_KIND = "complete_spyglass_supplementary_figure_6"

_RAW_LL_SUM = "test_light_swapped_segment_swapped_raw_ll_sum"
_RAW_LL_BITS_PER_SPIKE = (
    "test_light_swapped_segment_swapped_raw_ll_bits_per_spike"
)
_TEST_BIN_COUNT = "test_light_swapped_segment_n_bins"
_SWAP_SEGMENT = "swap_segment_index_1based"
_GLM_MODELS = {"V": "visual", "MS": SCALAR_MODEL}
_JOIN_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_train_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "trajectory",
    "unit",
)


def _expected_metadata(session: Mapping[str, Any]) -> dict[str, str]:
    """Return the fixed dark/AB-to-BA identity for one session."""
    return {
        "animal_name": str(session["animal_name"]),
        "date": str(session["date"]),
        "region": REGION,
        "dark_epoch": str(session["epochs"]["dark"]),
        "light_train_epoch": LIGHT_TRAIN_EPOCH,
        "light_test_epoch": LIGHT_TEST_EPOCH,
    }


def _require_metadata(
    observed: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    """Require an artifact to describe the selected manuscript transfer."""
    if any(str(observed.get(name)) != value for name, value in expected.items()):
        raise ValueError(f"{label} metadata disagrees with its session.")


def _require_swap_glm_schema(dataset: Any, *, source_path: Path) -> None:
    """Require the scalar and visual held-out scores used by the figure."""
    required_variables = {
        swap_glm.PRIMARY_METRIC,
        _RAW_LL_SUM,
        _RAW_LL_BITS_PER_SPIKE,
        _TEST_BIN_COUNT,
        _SWAP_SEGMENT,
    }
    missing_variables = sorted(required_variables.difference(dataset.data_vars))
    if missing_variables:
        raise ValueError(
            f"SwapGLM is missing variables {missing_variables!r}: {source_path}"
        )
    if not {"model", "trajectory", "unit"}.issubset(dataset.coords):
        raise ValueError(f"SwapGLM has stale coordinates: {source_path}")
    available_models = {
        str(value)
        for value in np.asarray(dataset.coords["model"].values).reshape(-1)
    }
    missing_models = sorted(set(_GLM_MODELS.values()).difference(available_models))
    if missing_models:
        raise ValueError(
            f"SwapGLM is missing models {missing_models!r}: {source_path}"
        )


def _load_parent_swap_results(
    parent_run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load, map, and dark-quality-filter the parent SwapGLM bundles."""
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for session in sessions:
        expected = _expected_metadata(session)
        record = figure_2_adapter._one_record(
            session["artifacts"].get("swap_glm", ()),
            label="parent SwapGLM artifact",
            region=REGION,
            dark_epoch=expected["dark_epoch"],
            light_train_epoch=LIGHT_TRAIN_EPOCH,
            light_test_epoch=LIGHT_TEST_EPOCH,
        )
        if str(record.get("artifact_origin", "")) != "computed":
            raise ValueError("Supplementary Figure 6 requires computed SwapGLM.")
        manifest_path = figure_2_adapter._record_artifact_path(
            record,
            "artifact_manifest_path",
            run_dir=parent_run_dir,
        )
        result = swap_glm.load_swap_glm_artifact(manifest_path.parent)
        _require_metadata(
            result["metadata"],
            expected,
            label="SwapGLM",
        )
        if str(result.get("artifact_origin", "")) != "computed":
            raise ValueError("Loaded SwapGLM is not a computed artifact.")
        dataset, selected = figure_3_adapter._mapped_swap_dataset(
            result,
            session=session,
        )
        _require_swap_glm_schema(dataset, source_path=manifest_path)

        dark_movement = figure_2_adapter._load_movement_table(
            session,
            epoch=expected["dark_epoch"],
            run_dir=parent_run_dir,
            scratch_root=scratch_root,
        )
        dark_stability = figure_2_adapter._load_stability_tables(
            session,
            epoch=expected["dark_epoch"],
            run_dir=parent_run_dir,
            scratch_root=scratch_root,
        )
        eligible = figure_2_adapter._eligible_units(
            dark_movement,
            dark_stability,
            minimum_movement_firing_rate_hz=(
                figure_2_adapter.MINIMUM_MOVEMENT_FIRING_RATE_HZ
            ),
            minimum_stability_correlation=(
                figure_2_adapter.MINIMUM_STABILITY_CORRELATION
            ),
        )
        key = (expected["animal_name"], expected["date"])
        if key in output:
            raise ValueError(f"Duplicate parent SwapGLM session {key!r}.")
        output[key] = {
            "dataset": dataset,
            "selected_units": selected,
            "eligible_unit_mask": selected["stable_unit_id"].isin(eligible).to_numpy(),
            "metadata": result["metadata"],
            "source_path": manifest_path,
        }
    return output


def _map_empirical_summary(
    result: Mapping[str, Any],
    *,
    sorting_unit_by_nwb_id: Mapping[str, Any],
) -> pd.DataFrame:
    """Map empirical group units to manuscript sorting-unit identifiers."""
    identity_columns = tuple(swap_tuning.IDENTITY_COLUMNS)
    selected = figure_3_adapter._identity_columns(
        result["selected_units"],
        label="SwapTuning selected_units",
    )
    summary = figure_3_adapter._identity_columns(
        result["summary"],
        label="SwapTuning summary",
    )
    if selected["group_unit_id"].duplicated().any():
        raise ValueError("SwapTuning selected units contain duplicate group IDs.")

    selected_identity = selected.loc[:, identity_columns].rename(
        columns={
            name: f"{name}__selected"
            for name in identity_columns
            if name != "group_unit_id"
        }
    )
    checked = summary.merge(
        selected_identity,
        on="group_unit_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    for name in identity_columns:
        if name == "group_unit_id":
            continue
        selected_name = f"{name}__selected"
        if checked[selected_name].isna().any() or not np.array_equal(
            checked[name].astype(str).to_numpy(),
            checked[selected_name].astype(str).to_numpy(),
        ):
            raise ValueError("SwapTuning summary and selected-unit identity disagree.")
    checked = checked.drop(
        columns=[
            f"{name}__selected"
            for name in identity_columns
            if name != "group_unit_id"
        ]
    )

    nwb_unit_ids = checked["unit_id"].astype(str).to_numpy()
    missing = sorted(set(nwb_unit_ids).difference(sorting_unit_by_nwb_id))
    if missing:
        raise ValueError(
            "SwapTuning units are absent from the NWB unit map: "
            f"{missing!r}."
        )
    sorting_units = pd.to_numeric(
        [sorting_unit_by_nwb_id[value] for value in nwb_unit_ids],
        errors="coerce",
    )
    if np.any(pd.isna(sorting_units)):
        raise ValueError("SwapTuning requires numeric manuscript unit IDs.")
    checked["unit"] = np.asarray(sorting_units, dtype=int)
    return checked


def _load_empirical_results(
    run_dir: Path,
    sessions: Sequence[Mapping[str, Any]],
    parent_sessions: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the empirical pointwise-additive score source per session."""
    parent_by_key = {
        (str(session["animal_name"]), str(session["date"])): session
        for session in parent_sessions
    }
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for session in sessions:
        expected = _expected_metadata(session)
        key = (expected["animal_name"], expected["date"])
        if key not in parent_by_key:
            raise ValueError(f"Supplementary session has no parent session: {key!r}.")
        record = figure_2_adapter._one_record(
            session["artifacts"].get("swap_tuning_curve_comparison", ()),
            label="SwapTuningCurveComparison artifact",
            region=REGION,
            dark_epoch=expected["dark_epoch"],
            light_train_epoch=LIGHT_TRAIN_EPOCH,
            light_test_epoch=LIGHT_TEST_EPOCH,
        )
        if str(record.get("artifact_origin", "")) != "computed":
            raise ValueError("Supplementary Figure 6 requires computed SwapTuning.")
        manifest_path = figure_2_adapter._record_artifact_path(
            record,
            "artifact_manifest_path",
            run_dir=run_dir,
        )
        result = swap_tuning.load_swap_tuning_curve_comparison_artifact(
            manifest_path.parent
        )
        _require_metadata(
            result["metadata"],
            expected,
            label="SwapTuning",
        )
        if str(result.get("artifact_origin", "")) != "computed":
            raise ValueError("Loaded SwapTuning is not a computed artifact.")
        summary = _map_empirical_summary(
            result,
            sorting_unit_by_nwb_id=figure_2_adapter._load_nwb_sorting_unit_map(
                parent_by_key[key]
            ),
        )
        if EMPIRICAL_ADDITIVE_MODEL not in set(summary["model"].astype(str)):
            raise ValueError(
                "SwapTuning lacks empirical_pointwise_additive_delta scores."
            )
        output[key] = {"summary": summary, "source_path": manifest_path}
    return output


def _glm_score_table(
    animal_name: str,
    date: str,
    loaded: Mapping[str, Any],
) -> pd.DataFrame:
    """Return visual and scalar raw held-out scores in long unit form."""
    dataset = loaded["dataset"]
    metadata = loaded["metadata"]
    trajectories = np.asarray(dataset.coords["trajectory"].values).astype(str)
    units = np.asarray(dataset.coords["unit"].values, dtype=int)
    trajectory_grid, unit_grid = np.meshgrid(trajectories, units, indexing="ij")
    n_bins = np.asarray(
        dataset[_TEST_BIN_COUNT].transpose("trajectory").values,
        dtype=float,
    )
    swap_segments = np.asarray(
        dataset[_SWAP_SEGMENT].transpose("trajectory").values,
        dtype=int,
    )
    if n_bins.shape != (len(trajectories),) or swap_segments.shape != (
        len(trajectories),
    ):
        raise ValueError("SwapGLM trajectory metadata dimensions are stale.")
    table = pd.DataFrame(
        {
            "animal_name": animal_name,
            "date": date,
            "region": REGION,
            "dark_train_epoch": str(metadata["dark_epoch"]),
            "light_train_epoch": str(metadata["light_train_epoch"]),
            "light_test_epoch": str(metadata["light_test_epoch"]),
            "trajectory": trajectory_grid.ravel(),
            "unit": unit_grid.ravel(),
            "glm_test_light_bin_count": np.repeat(n_bins, len(units)),
            "swap_segment_index_1based": np.repeat(
                swap_segments,
                len(units),
            ),
            "glm_source_path": str(loaded["source_path"]),
        }
    )
    for label, model_name in _GLM_MODELS.items():
        for variable, suffix in (
            (_RAW_LL_SUM, "ll_sum"),
            (_RAW_LL_BITS_PER_SPIKE, "bits_per_spike"),
        ):
            values = np.asarray(
                dataset[variable]
                .sel(model=model_name)
                .transpose("trajectory", "unit")
                .values,
                dtype=float,
            )
            if values.shape != (len(trajectories), len(units)):
                raise ValueError("SwapGLM raw score dimensions are stale.")
            table[f"{label}_{suffix}"] = values.ravel()
    return table


def _build_mixed_full_additive_table(
    loaded_by_session: Mapping[tuple[str, str], Mapping[str, Any]],
    empirical_by_session: Mapping[tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Match parent GLM scores to empirical pointwise-additive scores."""
    tables: list[pd.DataFrame] = []
    for key, loaded in loaded_by_session.items():
        if key not in empirical_by_session:
            raise ValueError(f"No empirical swap result for session {key!r}.")
        animal_name, date = key
        glm_table = _glm_score_table(animal_name, date, loaded)
        empirical_result = empirical_by_session[key]
        summary = empirical_result["summary"]
        required_columns = set(_JOIN_COLUMNS).union(
            {
                "model",
                "ll_sum",
                "ll_bits_per_spike",
                "test_light_bin_count",
                "score_qc_status",
                "unit_valid",
            }
        )
        missing = sorted(required_columns.difference(summary.columns))
        if missing:
            raise ValueError(f"SwapTuning summary is missing columns {missing!r}.")
        # unit_valid summarizes every model and path; the canonical cohort is
        # row-local and is restricted only by the joined finite scores below.
        empirical = summary.loc[
            summary["model"].astype(str).eq(EMPIRICAL_ADDITIVE_MODEL)
        ].copy()
        empirical = empirical.loc[
            : ,
            [
                *_JOIN_COLUMNS,
                "ll_sum",
                "ll_bits_per_spike",
                "test_light_bin_count",
            ],
        ].rename(
            columns={
                "ll_sum": f"{EMPIRICAL_ADDITIVE_LABEL}_ll_sum",
                "ll_bits_per_spike": (
                    f"{EMPIRICAL_ADDITIVE_LABEL}_bits_per_spike"
                ),
                "test_light_bin_count": "empirical_test_light_bin_count",
            }
        )
        empirical["empirical_source_path"] = str(
            empirical_result["source_path"]
        )
        merged = glm_table.merge(
            empirical,
            on=list(_JOIN_COLUMNS),
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        if merged.empty:
            continue
        if not np.allclose(
            merged["glm_test_light_bin_count"].to_numpy(dtype=float),
            merged["empirical_test_light_bin_count"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"GLM and empirical bin counts differ for {animal_name} {date}."
            )

        score_columns = (
            "V_ll_sum",
            "MS_ll_sum",
            f"{EMPIRICAL_ADDITIVE_LABEL}_ll_sum",
            "V_bits_per_spike",
            "MS_bits_per_spike",
            f"{EMPIRICAL_ADDITIVE_LABEL}_bits_per_spike",
        )
        finite = np.ones(len(merged), dtype=bool)
        for column in score_columns:
            finite &= np.isfinite(merged[column].to_numpy(dtype=float))
        merged = merged.loc[finite].copy()
        if merged.empty:
            continue

        labels = np.asarray(["V", "MS", EMPIRICAL_ADDITIVE_LABEL], dtype=object)
        ll_values = merged[
            ["V_ll_sum", "MS_ll_sum", f"{EMPIRICAL_ADDITIVE_LABEL}_ll_sum"]
        ].to_numpy(dtype=float)
        maximum = np.max(ll_values, axis=1)
        tied = (
            np.isclose(ll_values, maximum[:, None], rtol=0.0, atol=1e-12).sum(
                axis=1
            )
            > 1
        )
        merged["winner"] = labels[np.argmax(ll_values, axis=1)]
        merged.loc[tied, "winner"] = "tie"
        merged["delta_V_minus_task_bits_per_spike"] = (
            merged["V_bits_per_spike"] - merged["MS_bits_per_spike"]
        )
        merged[f"delta_V_minus_{EMPIRICAL_ADDITIVE_LABEL}_bits_per_spike"] = (
            merged["V_bits_per_spike"]
            - merged[f"{EMPIRICAL_ADDITIVE_LABEL}_bits_per_spike"]
        )
        tables.append(merged)
    if tables:
        return pd.concat(tables, ignore_index=True, sort=False)
    return pd.DataFrame(
        columns=[
            *_JOIN_COLUMNS,
            "swap_segment_index_1based",
            "winner",
            "V_ll_sum",
            "MS_ll_sum",
            f"{EMPIRICAL_ADDITIVE_LABEL}_ll_sum",
            "V_bits_per_spike",
            "MS_bits_per_spike",
            f"{EMPIRICAL_ADDITIVE_LABEL}_bits_per_spike",
            "delta_V_minus_task_bits_per_spike",
            f"delta_V_minus_{EMPIRICAL_ADDITIVE_LABEL}_bits_per_spike",
            "glm_source_path",
            "empirical_source_path",
        ]
    )


def load_supplementary_figure_6_payload(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load every active Supplementary Figure 6 input."""
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
    parent_run_dir, unordered_parent_sessions = load_parent_figure_2_sessions(
        sessions,
        scratch_root=scratch_root,
    )
    parent_sessions = figure_2_adapter._ordered_sessions(unordered_parent_sessions)
    loaded_by_session = _load_parent_swap_results(
        parent_run_dir,
        parent_sessions,
        scratch_root=scratch_root,
    )
    empirical_by_session = _load_empirical_results(
        run_dir,
        sessions,
        parent_sessions,
    )
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "sessions": sessions,
        "datasets": EXPECTED_DATASETS,
        "regions": (REGION,),
        "scalar_swap_delta_table": figure_3_adapter._delta_table(
            loaded_by_session,
            model_name=SCALAR_MODEL,
        ),
        "mixed_full_additive_table": _build_mixed_full_additive_table(
            loaded_by_session,
            empirical_by_session,
        ),
    }


def _require_request(
    *,
    data_root: Path,
    datasets: Sequence[Any],
    region: str,
    dark_epoch: str | None,
    payload: Mapping[str, Any],
) -> None:
    """Reject canonical requests outside the selected campaign."""
    if Path(data_root).resolve(strict=True) != Path(payload["run_dir"]).resolve(
        strict=True
    ):
        raise ValueError("Supplementary Figure 6 requested a foreign root.")
    observed = tuple(normalize_dataset_id(value) for value in datasets)
    if observed != tuple(payload["datasets"]):
        raise ValueError("Supplementary Figure 6 requested foreign sessions.")
    if str(region) != REGION or dark_epoch is not None:
        raise ValueError("Supplementary Figure 6 requested foreign settings.")


@contextmanager
def _offline_sources(payload: Mapping[str, Any]):
    """Inject only the two active canonical table loaders."""
    original_scalar = canonical.load_panel_h_swap_delta_table
    original_mixed = canonical.load_mixed_glm_full_additive_delta_table

    def load_scalar(
        *,
        data_root: Path,
        datasets: Sequence[Any],
        region: str,
        dark_epoch: str | None,
        light_epoch_pairs: Sequence[tuple[str, str]] = (
            _dark_light.PANEL_H_SWAP_LIGHT_EPOCH_PAIRS
        ),
        min_movement_firing_rate_hz: float | None = None,
        min_tuning_stability_correlation: float | None = None,
        model_name: str = canonical.SCALAR_MODEL_NAME,
    ) -> pd.DataFrame:
        _require_request(
            data_root=data_root,
            datasets=datasets,
            region=region,
            dark_epoch=dark_epoch,
            payload=payload,
        )
        if tuple(tuple(value) for value in light_epoch_pairs) != tuple(
            tuple(value) for value in _dark_light.PANEL_H_SWAP_LIGHT_EPOCH_PAIRS
        ):
            raise ValueError("Supplementary Figure 6 requested foreign transfers.")
        if (
            float(min_movement_firing_rate_hz)
            != float(figure_2_adapter.MINIMUM_MOVEMENT_FIRING_RATE_HZ)
            or float(min_tuning_stability_correlation)
            != float(figure_2_adapter.MINIMUM_STABILITY_CORRELATION)
            or str(model_name) != SCALAR_MODEL
        ):
            raise ValueError("Supplementary Figure 6 requested foreign scalar QC.")
        return payload["scalar_swap_delta_table"]

    def load_mixed(
        *,
        data_root: Path,
        datasets: Sequence[Any],
        region: str,
        dark_epoch: str | None,
        light_train_epoch: str = LIGHT_TRAIN_EPOCH,
        light_test_epoch: str = LIGHT_TEST_EPOCH,
    ) -> pd.DataFrame:
        _require_request(
            data_root=data_root,
            datasets=datasets,
            region=region,
            dark_epoch=dark_epoch,
            payload=payload,
        )
        if (
            str(light_train_epoch) != LIGHT_TRAIN_EPOCH
            or str(light_test_epoch) != LIGHT_TEST_EPOCH
        ):
            raise ValueError("Supplementary Figure 6 requested foreign epochs.")
        return payload["mixed_full_additive_table"]

    canonical.load_panel_h_swap_delta_table = load_scalar
    canonical.load_mixed_glm_full_additive_delta_table = load_mixed
    try:
        yield
    finally:
        canonical.load_panel_h_swap_delta_table = original_scalar
        canonical.load_mixed_glm_full_additive_delta_table = original_mixed


def get_output_path(
    *,
    run_dir: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    """Return the immutable run-local Supplementary Figure 6 path."""
    output_format = str(output_format).lower()
    if output_format not in canonical.FIGURE_FORMATS:
        raise ValueError(
            f"output_format must be one of {canonical.FIGURE_FORMATS!r}."
        )
    return Path(run_dir) / "figures" / f"{DEFAULT_OUTPUT_NAME}.{output_format}"


def promote_supplementary_figure_6(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    destination_path: Path,
    replace: bool = False,
) -> Path:
    """Publish one validated Supplementary Figure 6 and receipt."""
    return promote_spyglass_figure(
        payload,
        source_path=source_path,
        destination_path=destination_path,
        artifact_kind=FIGURE_ARTIFACT_KIND,
        replace=replace,
    )


def render_supplementary_figure_6(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Atomically render Supplementary Figure 6 inside its campaign."""
    run_dir = Path(payload["run_dir"]).resolve(strict=True)
    output_path = Path(output_path).resolve(strict=False)
    if not output_path.is_relative_to(run_dir):
        raise ValueError("Output must remain inside its supplementary campaign.")
    provenance_path = get_figure_provenance_path(output_path)
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Supplementary Figure 6: {output_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in canonical.FIGURE_FORMATS:
        raise ValueError("Unsupported Supplementary Figure 6 output format.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    linked = False
    try:
        with _offline_sources(payload):
            rendered = canonical.make_supplementary_figure_6(
                data_root=run_dir,
                output_path=temporary_path,
                datasets=payload["datasets"],
                region=REGION,
                dark_epoch=None,
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
    """Parse Supplementary Figure 6 Spyglass renderer arguments."""
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
    """Load one campaign and render Supplementary Figure 6."""
    args = parse_arguments(argv)
    payload = load_supplementary_figure_6_payload(
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
    path = render_supplementary_figure_6(
        payload,
        output_path=output_path,
        dpi=args.dpi,
    )
    print(f"Saved Spyglass Supplementary Figure 6 to {path}")
    if args.promote_to is not None:
        promoted = promote_supplementary_figure_6(
            payload,
            source_path=path,
            destination_path=args.promote_to,
            replace=args.replace_promoted_output,
        )
        print(f"Promoted Spyglass Supplementary Figure 6 to {promoted}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_NAME",
    "EXPECTED_DATASETS",
    "get_output_path",
    "load_supplementary_figure_6_payload",
    "main",
    "promote_supplementary_figure_6",
    "render_supplementary_figure_6",
]
