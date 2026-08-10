"""Render Figure 1D validation heatmaps from offline Spyglass artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import REGIONS
from v1ca1.paper_figures.datasets import get_processed_datasets
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_POSITION_OFFSET,
    FIGURE_FORMATS,
    PANEL_D_FIRING_RATE_NORMALIZATION,
    PANEL_D_HEATMAP_CMAP,
    PANEL_D_LINEAR_POSITION_ORIENTATION,
    PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    PANEL_D_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_D_TRAJECTORY_TYPES,
    TASK_PROGRESSION_XLABEL,
    _build_pooled_panel_values_order_and_peaks,
    add_panel_d_heatmap_block_outlines,
    draw_neuron_scale_bar,
    plot_pooled_heatmap_grid,
)
from v1ca1.paper_figures.style import apply_paper_style, figure_size, save_figure
from v1ca1.spyglass.movement import load_movement_firing_rate_artifact
from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    DEFAULT_SCRATCH_ROOT,
    MANIFEST_SCHEMA_VERSION,
    SESSION_MANIFEST_FILENAME,
    file_sha256,
    get_run_dir as get_offline_run_dir,
    load_campaign_manifest as load_offline_campaign_manifest,
    load_session_manifest,
    resolve_run_path,
)
from v1ca1.spyglass.offline.sources import SOURCE_IDENTITY_POLICY
from v1ca1.spyglass.path_specific_place import (
    load_path_specific_place_artifact,
)
from v1ca1.spyglass.table_specs import (
    DEFAULT_MOVEMENT_PARAMETERS,
    FIGURE_1D_TUNING_CURVE_PARAMETERS,
    LEGACY_TUNING_CURVE_PARAMETERS,
)


RUNS_DIRNAME = "runs"
FIGURE_MODES = ("l14-validation", "full")
L14_DATASET = ("L14", "20240611", "08_r4")
FULL_DATASETS = tuple(get_processed_datasets())
FIGURE_1_TUNING_PRESET = str(
    FIGURE_1D_TUNING_CURVE_PARAMETERS["tuning_curve_param_name"]
)
STABILITY_TUNING_PRESET = str(
    LEGACY_TUNING_CURVE_PARAMETERS["tuning_curve_param_name"]
)
FIGURE_1_POSITION_BIN_COUNT = int(
    FIGURE_1D_TUNING_CURVE_PARAMETERS["position_bin_count"]
)
FIGURE_1_SIGMA_BINS = float(
    FIGURE_1D_TUNING_CURVE_PARAMETERS["gaussian_smoothing_sigma_bins"]
)
DEFAULT_REGIONS = REGIONS
DEFAULT_DPI = 300
DEFAULT_OUTPUT_NAMES = {
    "l14-validation": "figure_1d_l14_spyglass_validation",
    "full": "figure_1d_spyglass",
}
ARTIFACT_GROUPS = (
    "movement_firing_rate",
    "path_specific_place_tuning_curve",
    "path_specific_place_stability",
)


def get_run_dir(*, scratch_root: Path, run_id: str) -> Path:
    """Return the immutable campaign directory for one explicit run ID."""
    return get_offline_run_dir(run_id, scratch_root=scratch_root)


def _require_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> None:
    """Require named fields in one manifest object."""
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{label} is missing fields {missing!r}.")


def _resolve_run_relative_path(
    run_dir: Path,
    value: Any,
    *,
    name: str,
    require_file: bool = True,
) -> Path:
    """Resolve one run-relative path and reject absolute paths or escapes."""
    try:
        path = resolve_run_path(str(value), run_dir=run_dir)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the run directory: {value!r}.") from exc
    if require_file and not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def load_campaign_manifest(*, scratch_root: Path, run_id: str) -> dict[str, Any]:
    """Load and minimally validate one offline campaign manifest."""
    manifest = load_offline_campaign_manifest(
        run_id,
        scratch_root=scratch_root,
        require_artifacts=True,
    )
    _require_fields(
        manifest,
        (
            "schema_version",
            "run_id",
            "status",
            "created_at_utc",
            "code_provenance",
            "analysis_parameters",
            "source_identity_policy",
            "sessions",
        ),
        label="Offline campaign manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported offline campaign manifest schema version "
            f"{manifest['schema_version']!r}."
        )
    if str(manifest["run_id"]) != str(run_id):
        raise ValueError("Campaign manifest run_id does not match its directory.")
    if manifest["status"] not in {"in_progress", "complete", "failed"}:
        raise ValueError(f"Unknown campaign status {manifest['status']!r}.")
    if not isinstance(manifest["sessions"], list):
        raise ValueError("Campaign manifest sessions must be a list.")
    if manifest["source_identity_policy"] != SOURCE_IDENTITY_POLICY:
        raise ValueError("Campaign uses an unknown offline source identity policy.")
    return manifest


def _session_key(value: Mapping[str, Any]) -> tuple[str, str]:
    """Return one session key from a manifest row."""
    _require_fields(value, ("animal_name", "date"), label="Session summary")
    return str(value["animal_name"]), str(value["date"])


def _validate_analysis_parameters(parameters: Any) -> None:
    """Require the offline run to match the fixed Figure 1D computation."""
    if not isinstance(parameters, Mapping):
        raise ValueError("Campaign analysis_parameters must be an object.")
    _require_fields(
        parameters,
        (
            "pipeline",
            "movement_parameters",
            "tuning_curve_parameter_presets",
            "stability_tuning_curve_param_name",
            "trial_subsets",
            "position_role",
            "regions",
            "trajectory_types",
            "diagnostic_figures",
        ),
        label="Campaign analysis parameters",
    )
    if str(parameters["pipeline"]) != "figure_1_initial_slice":
        raise ValueError("Campaign does not contain the Figure 1 initial slice.")
    if dict(parameters["movement_parameters"]) != dict(DEFAULT_MOVEMENT_PARAMETERS):
        raise ValueError("Campaign movement parameters do not match Figure 1.")
    presets = parameters["tuning_curve_parameter_presets"]
    if not isinstance(presets, list):
        raise ValueError("Campaign tuning-curve presets must be a list.")
    presets_by_name = {
        str(preset.get("tuning_curve_param_name")): preset
        for preset in presets
        if isinstance(preset, Mapping)
    }
    for expected in (
        LEGACY_TUNING_CURVE_PARAMETERS,
        FIGURE_1D_TUNING_CURVE_PARAMETERS,
    ):
        name = str(expected["tuning_curve_param_name"])
        if name not in presets_by_name or dict(presets_by_name[name]) != dict(expected):
            raise ValueError(f"Campaign tuning preset {name!r} is not canonical.")
    if str(parameters["stability_tuning_curve_param_name"]) != (
        STABILITY_TUNING_PRESET
    ):
        raise ValueError("Campaign stability does not use the legacy tuning preset.")
    if tuple(parameters["trial_subsets"]) != ("all", "odd", "even"):
        raise ValueError("Campaign must compute all, odd, and even trial subsets.")
    if str(parameters["position_role"]) != "head":
        raise ValueError("Figure 1 validation requires the head-position role.")
    if set(parameters["trajectory_types"]) != set(PANEL_D_TRAJECTORY_TYPES):
        raise ValueError("Campaign trajectories do not match Figure 1.")
    if bool(parameters["diagnostic_figures"]):
        raise ValueError("Offline Figure 1 campaigns must disable diagnostic figures.")


def _expected_datasets(mode: str) -> tuple[tuple[str, str, str], ...]:
    """Return the fixed data sets selected by one figure mode."""
    if mode == "l14-validation":
        return (L14_DATASET,)
    if mode == "full":
        return FULL_DATASETS
    raise ValueError(f"mode must be one of {FIGURE_MODES!r}.")


def _validate_nwb_source(session_manifest: Mapping[str, Any]) -> Path:
    """Validate the recorded read-only NWB source without opening it."""
    nwb_path = Path(str(session_manifest["nwb_path"]))
    if not nwb_path.is_absolute():
        raise ValueError("Session nwb_path must be an absolute source path.")
    if not nwb_path.is_file():
        raise FileNotFoundError(f"Session source NWB not found: {nwb_path}")
    if nwb_path.name != str(session_manifest["nwb_file_name"]):
        raise ValueError("Session nwb_path and nwb_file_name disagree.")
    fingerprint = session_manifest["nwb_fingerprint"]
    if not isinstance(fingerprint, Mapping):
        raise ValueError("Session nwb_fingerprint must be an object.")
    _require_fields(
        fingerprint,
        ("resolved_path", "size_bytes", "mtime_ns"),
        label="Session NWB fingerprint",
    )
    resolved = nwb_path.resolve(strict=True)
    stat = resolved.stat()
    if (
        str(fingerprint["resolved_path"]) != str(resolved)
        or int(fingerprint["size_bytes"]) != int(stat.st_size)
        or int(fingerprint["mtime_ns"]) != int(stat.st_mtime_ns)
    ):
        raise ValueError("Session source NWB has changed since offline computation.")
    return nwb_path


def load_figure_1_session_manifests(
    *,
    scratch_root: Path,
    run_id: str,
    mode: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load the completed sessions required for partial or full validation."""
    run_dir = get_run_dir(scratch_root=scratch_root, run_id=run_id)
    campaign = load_campaign_manifest(scratch_root=scratch_root, run_id=run_id)
    if campaign["status"] == "failed":
        raise ValueError("Cannot render a failed offline campaign.")
    _validate_analysis_parameters(campaign["analysis_parameters"])

    summaries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for summary in campaign["sessions"]:
        if not isinstance(summary, Mapping):
            raise ValueError("Each campaign session summary must be an object.")
        key = _session_key(summary)
        if key in summaries:
            raise ValueError(f"Campaign contains duplicate session {key!r}.")
        summaries[key] = summary

    expected = _expected_datasets(mode)
    expected_keys = {(animal_name, date) for animal_name, date, _epoch in expected}
    if mode == "full" and set(summaries) != expected_keys:
        raise ValueError(
            "Full Figure 1 mode requires exactly the four manuscript sessions; "
            f"expected {sorted(expected_keys)!r}, received {sorted(summaries)!r}."
        )
    missing = sorted(expected_keys.difference(summaries))
    if missing:
        raise ValueError(f"Campaign is missing required sessions {missing!r}.")

    session_manifests: list[dict[str, Any]] = []
    for animal_name, date, epoch in expected:
        summary = summaries[(animal_name, date)]
        _require_fields(
            summary,
            ("nwb_file_name", "nwb_path", "session_manifest_path", "status"),
            label=f"Session summary {animal_name} {date}",
        )
        if summary["status"] != "complete":
            raise ValueError(
                f"Session {animal_name} {date} is not complete: "
                f"{summary['status']!r}."
            )
        manifest_path = _resolve_run_relative_path(
            run_dir,
            summary["session_manifest_path"],
            name=f"Session manifest path for {animal_name} {date}",
        )
        session = load_session_manifest(
            manifest_path,
            run_dir=run_dir,
            require_artifacts=True,
        )
        _require_fields(
            session,
            (
                "schema_version",
                "run_id",
                "animal_name",
                "date",
                "nwb_file_name",
                "nwb_path",
                "nwb_fingerprint",
                "code_provenance",
                "status",
                "position_selection",
                "epochs",
                "regions",
                "trajectories",
                "parameters",
                "selection_identity_scope",
                "source_identity",
                "artifacts",
            ),
            label=f"Session manifest {animal_name} {date}",
        )
        if session["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported session manifest schema for {animal_name} {date}."
            )
        if (
            str(session["run_id"]) != str(run_id)
            or _session_key(session) != (animal_name, date)
        ):
            raise ValueError(
                f"Session manifest identity disagrees for {animal_name} {date}."
            )
        if session["status"] != "complete":
            raise ValueError(f"Session manifest {animal_name} {date} is not complete.")
        if session["parameters"] != campaign["analysis_parameters"]:
            raise ValueError(
                f"Campaign and session parameters disagree for {animal_name} {date}."
            )
        if session["selection_identity_scope"] != "offline_surrogate":
            raise ValueError(
                f"Session {animal_name} {date} has an unknown selection identity scope."
            )
        if not isinstance(session["code_provenance"], Mapping) or (
            "v1ca1_git_commit" not in session["code_provenance"]
        ):
            raise ValueError(
                f"Session {animal_name} {date} has invalid code provenance."
            )
        position_selection = session["position_selection"]
        if not isinstance(position_selection, Mapping):
            raise ValueError("Session position_selection must be an object.")
        _require_fields(
            position_selection,
            ("position_role", "spatial_unit", "analysis_start_offset_samples"),
            label=f"Position selection {animal_name} {date}",
        )
        if (
            str(position_selection["position_role"]) != "head"
            or str(position_selection["spatial_unit"]) != "cm"
            or int(position_selection["analysis_start_offset_samples"])
            != DEFAULT_POSITION_OFFSET
        ):
            raise ValueError(
                "Figure 1 validation requires head position in centimeters "
                f"with a {DEFAULT_POSITION_OFFSET}-sample analysis offset."
            )
        if str(session["nwb_file_name"]) != str(summary["nwb_file_name"]) or str(
            session["nwb_path"]
        ) != str(summary["nwb_path"]):
            raise ValueError(
                f"Campaign and session NWB sources disagree for {animal_name} {date}."
            )
        _validate_nwb_source(session)
        if str(epoch) not in {str(value) for value in session["epochs"]}:
            raise ValueError(
                f"Session {animal_name} {date} does not contain expected epoch {epoch}."
            )
        if not isinstance(session["artifacts"], Mapping):
            raise ValueError("Session artifacts must be an object.")
        source_identity = session["source_identity"]
        if not isinstance(source_identity, list):
            raise ValueError("Session source_identity must be a list.")
        source_regions: set[str] = set()
        for record in source_identity:
            if not isinstance(record, Mapping):
                raise ValueError("Every source_identity record must be an object.")
            _require_fields(
                record,
                (
                    "region",
                    "source",
                    "spikesorting_merge_id",
                    "offline_region_sorted_spikes_view_id",
                    "n_units",
                    "selected_units_sha256",
                ),
                label=f"Source identity {animal_name} {date}",
            )
            if str(record["source"]) != "ImportedSpikeSorting":
                raise ValueError("Offline Figure 1 requires imported NWB units.")
            region_name = str(record["region"])
            if region_name in source_regions:
                raise ValueError("Session source_identity has duplicate regions.")
            source_regions.add(region_name)
        if source_regions != {str(value) for value in session["regions"]}:
            raise ValueError(
                "Session source identities do not match its declared regions."
            )
        missing_groups = [
            group for group in ARTIFACT_GROUPS if group not in session["artifacts"]
        ]
        if missing_groups:
            raise ValueError(
                f"Session {animal_name} {date} is missing artifact groups "
                f"{missing_groups!r}."
            )
        session_manifests.append(session)

    if len(session_manifests) > 1:
        parameter_snapshots = {
            json.dumps(session["parameters"], sort_keys=True, separators=(",", ":"))
            for session in session_manifests
        }
        if len(parameter_snapshots) != 1:
            raise ValueError("Figure sessions were computed with different parameters.")
        commits = {
            str(session["code_provenance"]["v1ca1_git_commit"])
            for session in session_manifests
        }
        if len(commits) != 1:
            raise ValueError(
                "Figure sessions were computed from different v1ca1 commits."
            )
    return run_dir, campaign, session_manifests


def _matching_record(
    records: Any,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    """Return exactly one artifact record matching fixed selection fields."""
    if not isinstance(records, list):
        raise ValueError(f"{label} artifact records must be a list.")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and all(str(record.get(name)) == str(value) for name, value in expected.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} record matching {dict(expected)!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _load_stability_artifact(path: Path) -> Any:
    """Load one canonical stability Parquet with required Figure 1 columns."""
    import pandas as pd

    table = pd.read_parquet(path)
    required = {
        "animal_name",
        "date",
        "region",
        "epoch",
        "trajectory_type",
        "stable_unit_id",
        "unit_id",
        "stability_correlation",
        "stability_status",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Stability artifact {path} is missing columns {missing!r}.")
    if table["stable_unit_id"].astype(str).duplicated().any():
        raise ValueError(f"Stability artifact {path} contains duplicate units.")
    return table


def _record_path(
    run_dir: Path,
    record: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> Path:
    """Resolve one named artifact path stored in a manifest record."""
    _require_fields(record, (field,), label=label)
    path = _resolve_run_relative_path(
        run_dir,
        record[field],
        name=f"{label} {field}",
    )
    hashes = record.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or field not in hashes:
        raise ValueError(f"{label} is missing the {field} SHA-256 digest.")
    if file_sha256(path) != str(hashes[field]):
        raise ValueError(f"{label} {field} SHA-256 digest does not match.")
    return path


def _constant_string_column(
    table: Any,
    column: str,
    expected: str,
    *,
    label: str,
) -> None:
    """Require every non-empty table row to carry one expected label."""
    if table.empty:
        return
    values = set(table[column].astype(str))
    if values != {str(expected)}:
        raise ValueError(
            f"{label} column {column!r} must equal {expected!r}; got {values!r}."
        )


def _validate_curve_identity(
    curve: Any,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    trajectory_type: str,
    trial_subset: str,
) -> None:
    """Require one canonical curve to match its manifest selection."""
    expected = {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "region": region,
        "trajectory_type": trajectory_type,
        "trial_subset": trial_subset,
    }
    mismatches = {
        name: (curve.attrs.get(name), value)
        for name, value in expected.items()
        if str(curve.attrs.get(name)) != str(value)
    }
    if mismatches:
        raise ValueError(f"Tuning-curve manifest identity mismatch: {mismatches!r}.")
    if str(curve.attrs.get("binning_mode")) != "bin_count" or int(
        curve.attrs.get("bin_count", -1)
    ) != FIGURE_1_POSITION_BIN_COUNT:
        raise ValueError("Figure 1 tuning curves must use exactly 50 position bins.")
    if not np.isclose(
        float(curve.attrs.get("sigma_bins", np.nan)),
        FIGURE_1_SIGMA_BINS,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Figure 1 tuning curves must use sigma_bins=1.5.")


def _filter_and_label_curve(curve: Any, included_stable_ids: set[str]) -> Any:
    """Filter and expose composite stable identities to Figure 1 helpers."""
    stable_ids = np.asarray(curve.coords["stable_unit_id"].values).astype(str)
    indices = np.flatnonzero(
        np.asarray([stable_id in included_stable_ids for stable_id in stable_ids])
    )
    selected = curve.isel(unit=indices).copy()
    return selected.assign_coords(unit=("unit", stable_ids[indices]))


def load_session_curve_set(
    *,
    run_dir: Path,
    session_manifest: Mapping[str, Any],
    epoch: str,
    region: str,
    min_movement_firing_rate_hz: float = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> dict[str, Any]:
    """Load one session's filtered odd/even curves for Figure 1D."""
    if min_movement_firing_rate_hz < 0.0:
        raise ValueError("min_movement_firing_rate_hz must be non-negative.")
    if not -1.0 <= min_tuning_stability_correlation <= 1.0:
        raise ValueError(
            "min_tuning_stability_correlation must lie between -1 and 1."
        )
    animal_name = str(session_manifest["animal_name"])
    date = str(session_manifest["date"])
    if str(region) not in {str(value) for value in session_manifest["regions"]}:
        raise ValueError(
            f"Session {animal_name} {date} does not contain region {region!r}."
        )
    artifacts = session_manifest["artifacts"]
    source_record = _matching_record(
        session_manifest["source_identity"],
        expected={"region": region, "source": "ImportedSpikeSorting"},
        label="regional spike source",
    )
    _require_fields(
        source_record,
        ("n_units", "selected_units_sha256"),
        label="Regional spike source",
    )

    movement_record = _matching_record(
        artifacts["movement_firing_rate"],
        expected={"epoch": epoch, "region": region},
        label="movement firing-rate",
    )
    _require_fields(
        movement_record,
        ("n_units", "selected_units_sha256"),
        label="Movement firing-rate record",
    )
    if int(movement_record["n_units"]) != int(source_record["n_units"]):
        raise ValueError("Movement and regional spike-source unit counts disagree.")
    movement_path = _record_path(
        run_dir,
        movement_record,
        "firing_rate_path",
        label="Movement firing-rate record",
    )
    movement = load_movement_firing_rate_artifact(movement_path)
    for column, expected in {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "region": region,
    }.items():
        _constant_string_column(
            movement,
            column,
            expected,
            label="Movement firing-rate artifact",
        )
    movement_unit_ids = set(movement["stable_unit_id"].astype(str))
    if movement.empty:
        movement_selected: set[str] = set()
    else:
        rates = np.asarray(movement["movement_firing_rate_hz"], dtype=float)
        movement_selected = set(
            movement.loc[
                np.isfinite(rates) & (rates >= min_movement_firing_rate_hz),
                "stable_unit_id",
            ].astype(str)
        )

    stability_tables = []
    stability_digests: set[str] = set()
    for trajectory_type in PANEL_D_TRAJECTORY_TYPES:
        record = _matching_record(
            artifacts["path_specific_place_stability"],
            expected={
                "epoch": epoch,
                "region": region,
                "trajectory_type": trajectory_type,
                "tuning_curve_param_name": STABILITY_TUNING_PRESET,
            },
            label="path-specific stability",
        )
        _require_fields(
            record,
            ("selected_units_sha256",),
            label="Path-specific stability record",
        )
        stability_digests.add(str(record["selected_units_sha256"]))
        path = _record_path(
            run_dir,
            record,
            "stability_path",
            label="Path-specific stability record",
        )
        table = _load_stability_artifact(path)
        for column, expected in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "region": region,
            "trajectory_type": trajectory_type,
        }.items():
            _constant_string_column(table, column, expected, label="Stability artifact")
        if set(table["stable_unit_id"].astype(str)) != movement_unit_ids:
            raise ValueError(
                "Stability units do not match the movement firing-rate artifact."
            )
        stability_tables.append(table)
    if len(stability_digests) > 1:
        raise ValueError("Stability artifacts disagree on selected-unit identity.")

    stable_selected: set[str] = set()
    for table in stability_tables:
        correlations = np.asarray(table["stability_correlation"], dtype=float)
        statuses = table["stability_status"].astype(str).to_numpy()
        mask = (
            (statuses == "valid")
            & np.isfinite(correlations)
            & (correlations >= min_tuning_stability_correlation)
        )
        stable_selected.update(table.loc[mask, "stable_unit_id"].astype(str))
    included_stable_ids = movement_selected.intersection(stable_selected)

    curves: dict[str, dict[str, Any]] = {"odd": {}, "even": {}}
    tuning_digests: set[str] = set()
    movement_id = str(movement_record.get("movement_firing_rate_id", ""))
    for trial_subset in ("odd", "even"):
        for trajectory_type in PANEL_D_TRAJECTORY_TYPES:
            record = _matching_record(
                artifacts["path_specific_place_tuning_curve"],
                expected={
                    "epoch": epoch,
                    "region": region,
                    "trajectory_type": trajectory_type,
                    "trial_subset": trial_subset,
                    "tuning_curve_param_name": FIGURE_1_TUNING_PRESET,
                },
                label="path-specific tuning curve",
            )
            _require_fields(
                record,
                ("selected_units_sha256",),
                label="Path-specific tuning record",
            )
            if movement_id and (
                str(record.get("movement_firing_rate_id", "")) != movement_id
            ):
                raise ValueError("Tuning curve and movement artifact IDs disagree.")
            tuning_digests.add(str(record["selected_units_sha256"]))
            curve_path = _record_path(
                run_dir,
                record,
                "tuning_curve_path",
                label="Path-specific tuning record",
            )
            curve = load_path_specific_place_artifact(curve_path)
            _validate_curve_identity(
                curve,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                region=region,
                trajectory_type=trajectory_type,
                trial_subset=trial_subset,
            )
            curve_stable_ids = set(
                np.asarray(curve.coords["stable_unit_id"].values).astype(str)
            )
            if curve_stable_ids != movement_unit_ids:
                raise ValueError(
                    "Tuning curve units do not match the movement firing-rate artifact."
                )
            curves[trial_subset][trajectory_type] = _filter_and_label_curve(
                curve,
                included_stable_ids,
            )
    if len(tuning_digests) > 1:
        raise ValueError(
            "Figure 1 tuning artifacts disagree on selected-unit identity."
        )
    movement_digest = str(movement_record.get("selected_units_sha256", ""))
    identity_digests = stability_digests.union(tuning_digests)
    if movement_digest:
        identity_digests.add(movement_digest)
    identity_digests.add(str(source_record["selected_units_sha256"]))
    if len(identity_digests) > 1:
        raise ValueError(
            "Movement, tuning, and stability artifacts select different unit "
            "identities."
        )

    return {
        "animal_name": animal_name,
        "date": date,
        "region": str(region),
        "epoch": str(epoch),
        "odd_curves": curves["odd"],
        "even_curves": curves["even"],
        "included_units": np.asarray(sorted(included_stable_ids), dtype=str),
        "movement_firing_rate_path": str(movement_path),
    }


def load_figure_1d_payload(
    *,
    scratch_root: Path,
    run_id: str,
    mode: str,
    regions: Sequence[str] = DEFAULT_REGIONS,
    min_movement_firing_rate_hz: float = PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ,
    min_tuning_stability_correlation: float = (
        PANEL_D_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> dict[str, Any]:
    """Load and pool the offline artifacts needed for Figure 1D."""
    run_dir, campaign, sessions = load_figure_1_session_manifests(
        scratch_root=scratch_root,
        run_id=run_id,
        mode=mode,
    )
    expected = _expected_datasets(mode)
    regions = tuple(dict.fromkeys(str(region) for region in regions))
    if not regions or any(not region for region in regions):
        raise ValueError("regions must contain at least one non-empty name.")
    curve_sets_by_region: dict[str, list[dict[str, Any]]] = {}
    panels_by_region = {}
    ordered_unit_keys_by_region = {}
    ordered_peak_positions_by_region = {}
    for region in regions:
        curve_sets = [
            load_session_curve_set(
                run_dir=run_dir,
                session_manifest=session,
                epoch=epoch,
                region=region,
                min_movement_firing_rate_hz=min_movement_firing_rate_hz,
                min_tuning_stability_correlation=min_tuning_stability_correlation,
            )
            for session, (_animal_name, _date, epoch) in zip(
                sessions,
                expected,
                strict=True,
            )
        ]
        panels, ordered_unit_keys, ordered_peak_positions = (
            _build_pooled_panel_values_order_and_peaks(
                curve_sets,
                position_bin_count=FIGURE_1_POSITION_BIN_COUNT,
                trajectory_types=PANEL_D_TRAJECTORY_TYPES,
                firing_rate_normalization=PANEL_D_FIRING_RATE_NORMALIZATION,
            )
        )
        curve_sets_by_region[region] = curve_sets
        panels_by_region[region] = panels
        ordered_unit_keys_by_region[region] = ordered_unit_keys
        ordered_peak_positions_by_region[region] = ordered_peak_positions
    return {
        "run_dir": run_dir,
        "campaign": campaign,
        "mode": mode,
        "regions": regions,
        "datasets": expected,
        "curve_sets_by_region": curve_sets_by_region,
        "panels_by_region": panels_by_region,
        "ordered_unit_keys_by_region": ordered_unit_keys_by_region,
        "ordered_peak_positions_by_region": ordered_peak_positions_by_region,
        "min_movement_firing_rate_hz": float(min_movement_firing_rate_hz),
        "min_tuning_stability_correlation": float(
            min_tuning_stability_correlation
        ),
    }


def get_default_output_path(
    *,
    scratch_root: Path,
    run_id: str,
    mode: str,
    output_format: str,
) -> Path:
    """Return a run-local figure path that never points at legacy output."""
    if mode not in FIGURE_MODES:
        raise ValueError(f"mode must be one of {FIGURE_MODES!r}.")
    if output_format not in FIGURE_FORMATS:
        raise ValueError(f"output_format must be one of {FIGURE_FORMATS!r}.")
    return (
        get_run_dir(scratch_root=scratch_root, run_id=run_id)
        / "figures"
        / f"{DEFAULT_OUTPUT_NAMES[mode]}.{output_format}"
    )


def render_figure_1d_validation(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render and save the Figure 1D heatmap panel from one loaded payload."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    run_dir = Path(payload["run_dir"]).resolve()
    resolved_output = output_path.resolve()
    try:
        resolved_output.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(
            "Figure output must remain inside the selected run directory."
        ) from exc
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Figure 1 validation: {output_path}"
        )
    if output_path.suffix.lower().lstrip(".") not in FIGURE_FORMATS:
        raise ValueError(f"Figure suffix must be one of {FIGURE_FORMATS!r}.")

    apply_paper_style()
    regions = tuple(str(region) for region in payload["regions"])
    n_trajectory_rows = len(PANEL_D_TRAJECTORY_TYPES)
    fig, axes = plt.subplots(
        len(regions) * n_trajectory_rows,
        len(PANEL_D_TRAJECTORY_TYPES),
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, 105.0 * len(regions)),
        constrained_layout=True,
        squeeze=False,
    )
    color_image = None
    labels = {
        "center_to_left": "C→L",
        "left_to_center": "L→C",
        "center_to_right": "C→R",
        "right_to_center": "R→C",
    }
    for region_index, region in enumerate(regions):
        row_start = region_index * n_trajectory_rows
        row_stop = row_start + n_trajectory_rows
        region_axes = axes[row_start:row_stop]
        image = plot_pooled_heatmap_grid(
            region_axes,
            payload["panels_by_region"][region],
            trajectory_types=PANEL_D_TRAJECTORY_TYPES,
            axis_orientation=PANEL_D_LINEAR_POSITION_ORIENTATION,
            cmap=PANEL_D_HEATMAP_CMAP,
        )
        if color_image is None and image is not None:
            color_image = image
        for row_index, trajectory_type in enumerate(PANEL_D_TRAJECTORY_TYPES):
            region_axes[row_index, 0].set_ylabel(
                f"{region.upper()}\nOrder: {labels[trajectory_type]}"
                if row_index == 0
                else f"Order: {labels[trajectory_type]}",
                fontsize=6,
            )
        for column_index, trajectory_type in enumerate(PANEL_D_TRAJECTORY_TYPES):
            region_axes[0, column_index].set_title(
                f"Tuning: {labels[trajectory_type]}",
                fontsize=6,
            )
    fig.supxlabel(TASK_PROGRESSION_XLABEL, fontsize=7)
    mode_label = (
        "L14 validation"
        if payload["mode"] == "l14-validation"
        else "four-session reproduction"
    )
    fig.suptitle(
        f"Figure 1D from offline Spyglass artifacts — {mode_label}",
        fontsize=8,
    )
    if color_image is not None:
        colorbar = fig.colorbar(
            color_image,
            ax=axes.ravel().tolist(),
            shrink=0.30,
            pad=0.01,
            ticks=[0.0, 1.0],
        )
        colorbar.set_label("Norm. FR", fontsize=6)
    draw_neuron_scale_bar(axes[-1, -1])
    fig.canvas.draw()
    fig.set_layout_engine(None)
    add_panel_d_heatmap_block_outlines(axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit offline Figure 1D validation command."""
    parser = argparse.ArgumentParser(
        description=(
            "Render Figure 1D validation heatmaps from one retained offline "
            "Spyglass run. This command never reads legacy analysis artifacts."
        )
    )
    parser.add_argument("--run-id", required=True, help="Immutable offline run ID.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=FIGURE_MODES,
        help="Use L14 only for validation or require all four manuscript sessions.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help=f"Offline analysis root. Default: {DEFAULT_SCRATCH_ROOT}",
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help=(
            "Regional sorting view to plot. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default="svg",
        help="Run-local output format. Default: svg",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional output path inside the selected run directory. "
            "Default: <run-dir>/figures/<mode-specific name>."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Rasterization dpi. Default: {DEFAULT_DPI}",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load one offline campaign and render its Figure 1D validation."""
    args = parse_arguments(argv)
    payload = load_figure_1d_payload(
        scratch_root=args.scratch_root,
        run_id=args.run_id,
        mode=args.mode,
        regions=tuple(args.region) if args.region is not None else DEFAULT_REGIONS,
    )
    output_path = (
        get_default_output_path(
            scratch_root=args.scratch_root,
            run_id=args.run_id,
            mode=args.mode,
            output_format=args.output_format,
        )
        if args.output_path is None
        else args.output_path
    )
    path = render_figure_1d_validation(
        payload,
        output_path=output_path,
        dpi=args.dpi,
    )
    print(f"Saved offline Figure 1D validation to {path}")


if __name__ == "__main__":
    main()
