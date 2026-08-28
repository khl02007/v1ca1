"""Populate database-backed Spyglass results required by manuscript figures."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures.datasets import PROCESSED_DATASETS
from v1ca1.spyglass import table_specs


DEFAULT_NWB_ROOT = Path("/stelmo/nwb/raw")
FIGURE_REGIONS = ("v1", "ca1")
TRIAL_SUBSETS = ("all", "odd", "even")
STAGE_ORDER = ("sources", "base", "tuning", "models", "ripple")
STAGE_CHOICES = ("preflight", *STAGE_ORDER, "all", "status")
PRIMARY_POSITION_ROLE = "head"
ORIENTATION_REFERENCE_POSITION_ROLE = "body"
LIGHT_TRAIN_EPOCH = "02_r1"
LIGHT_TEST_EPOCH = "06_r3"
XCORR_DATASET = ("L15", "20241121")
_TUNING_SELECTION_ID = "path_specific_place_tuning_curve_id"
_INDIRECT_SESSION_REFERENCE_FIELDS = {
    "path_specific_place_stability": (
        "odd_path_specific_place_tuning_curve_id"
    ),
    "path_specific_place_tuning_similarity": (
        "center_to_left_tuning_curve_id"
    ),
}


def figure_dataset_specs() -> tuple[dict[str, str], ...]:
    """Return the configured manuscript sessions as explicit mappings."""
    return tuple(
        {
            "animal_name": str(animal_name),
            "date": str(date),
            "light_epoch": str(light_epoch),
            "dark_epoch": str(dark_epoch),
            "sleep_epoch": str(sleep_epoch),
        }
        for animal_name, date, light_epoch, dark_epoch, sleep_epoch in (
            PROCESSED_DATASETS
        )
    )


def select_figure_datasets(
    *,
    animal_name: str | None,
    date: str | None,
    all_datasets: bool,
) -> tuple[dict[str, str], ...]:
    """Resolve either one explicit manuscript session or the full cohort."""
    specs = figure_dataset_specs()
    if all_datasets:
        if animal_name is not None or date is not None:
            raise ValueError(
                "--all-datasets cannot be combined with --animal-name or --date."
            )
        return specs
    if animal_name is None or date is None:
        raise ValueError(
            "Pass both --animal-name and --date, or pass --all-datasets."
        )
    matches = tuple(
        spec
        for spec in specs
        if spec["animal_name"] == str(animal_name)
        and spec["date"] == str(date)
    )
    if len(matches) != 1:
        raise ValueError(
            f"{animal_name} {date} is not one configured manuscript session."
        )
    return matches


def stages_through(stage: str) -> tuple[str, ...]:
    """Return the ordered execution stages requested by one CLI value."""
    if stage == "all":
        return STAGE_ORDER
    if stage in STAGE_ORDER:
        return (stage,)
    if stage in {"preflight", "status"}:
        return ()
    raise ValueError(f"Unknown population stage {stage!r}.")


def planned_result_counts(spec: Mapping[str, str]) -> dict[str, int]:
    """Return exact figure-required result counts for one session."""
    is_xcorr_session = (
        str(spec["animal_name"]),
        str(spec["date"]),
    ) == XCORR_DATASET
    return {
        "epoch_motor_behavior": 2,
        "movement_firing_rate": 4,
        "path_specific_place_tuning_curve": 64,
        "path_specific_place_stability": 12,
        "path_specific_place_tuning_similarity": 2,
        "cv_pca": 1,
        "dpp_encoding": 1,
        "path_progression_decoding": 2,
        "path_specific_place_decoding": 2,
        "motor_encoding": 1,
        "dark_light_glm": 1,
        "swap_glm": 1,
        "swap_tuning_curve_comparison": 1,
        "ripple_modulation": 2,
        "ripple_glm": 2,
        "ripple_cross_region_xcorr": int(is_xcorr_session),
    }


def _raw_nwb_path(spec: Mapping[str, str], *, nwb_root: Path) -> Path:
    """Return the canonical augmented source path for one session."""
    return Path(nwb_root) / (
        f"{spec['animal_name']}{spec['date']}_augmented.nwb"
    )


def _emit(event: str, **values: Any) -> None:
    """Print one immediately visible, machine-readable progress record."""
    print(
        json.dumps({"event": event, **values}, default=str, sort_keys=True),
        flush=True,
    )


def _register_analysis_nwbfile_table(tables: Mapping[str, Any]) -> None:
    """Register the activated custom analysis-NWB table with Spyglass."""
    analysis_nwbfile_table = tables.get("analysis_nwbfile")
    if analysis_nwbfile_table is None:
        raise RuntimeError(
            "Activation did not return the AnalysisNwbfile table."
        )
    analysis_nwbfile_table().register_with_spyglass()


def _load_runtime(
    *,
    schema_name: str,
    analysis_nwbfile_schema_name: str,
) -> dict[str, Any]:
    """Load DataJoint, Spyglass, and the activated project table bundle."""
    import datajoint as dj
    from spyglass.common import Nwbfile, Session, populate_all_common
    from spyglass.data_import.insert_sessions import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.imported import ImportedSpikeSorting
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    from v1ca1.spyglass.tables import activate

    if analysis_nwbfile_schema_name.count("_") != 1:
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    prefix, suffix = analysis_nwbfile_schema_name.split("_", 1)
    if not prefix or suffix != "nwbfile":
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    custom_config = dict(dj.config.get("custom", {}))
    custom_config["database.prefix"] = prefix
    dj.config["custom"] = custom_config
    tables = activate(
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
        connection=dj.conn(),
        create_schema=False,
        create_tables=False,
    )
    _register_analysis_nwbfile_table(tables)
    return {
        "dj": dj,
        "tables": tables,
        "Nwbfile": Nwbfile,
        "Session": Session,
        "populate_all_common": populate_all_common,
        "insert_sessions": insert_sessions,
        "raw_dir": Path(raw_dir),
        "SortedSpikesGroup": SortedSpikesGroup,
        "UnitSelectionParams": UnitSelectionParams,
        "ImportedSpikeSorting": ImportedSpikeSorting,
        "SpikeSortingOutput": SpikeSortingOutput,
        "get_nwb_copy_filename": get_nwb_copy_filename,
    }


def _one_row(table: Any, restriction: Mapping[str, Any], *, label: str) -> dict:
    """Fetch exactly one DataJoint row as a dictionary."""
    relation = table & dict(restriction)
    if len(relation) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {dict(restriction)!r}; "
            f"found {len(relation)}."
        )
    return dict(relation.fetch1())


def _standard_nwb_file_name(
    runtime: Mapping[str, Any], raw_path: Path
) -> str:
    """Return the linked-copy filename standard Spyglass will register."""
    return str(runtime["get_nwb_copy_filename"](raw_path.name))


def _preflight_catalog(
    spec: Mapping[str, str],
    *,
    nwb_root: Path,
    standard_nwb_file_name: str | None = None,
) -> dict[str, Any]:
    """Inspect one source NWB and require every figure-selected catalog row."""
    from v1ca1.spyglass.ingest import ingest_v1ca1_nwb

    raw_path = _raw_nwb_path(spec, nwb_root=nwb_root).resolve(strict=True)
    catalog_name = standard_nwb_file_name or raw_path.name
    preview = ingest_v1ca1_nwb(
        catalog_name,
        nwb_path=raw_path,
        dry_run=True,
    )
    rows = preview["rows"]
    required_epochs = {
        str(spec["light_epoch"]),
        LIGHT_TEST_EPOCH,
        str(spec["dark_epoch"]),
        str(spec["sleep_epoch"]),
    }
    observed_epochs = {str(row["epoch"]) for row in rows["epoch_intervals"]}
    missing_epochs = required_epochs.difference(observed_epochs)
    if missing_epochs:
        raise ValueError(
            f"{raw_path.name} is missing figure epochs {sorted(missing_epochs)!r}."
        )
    required_run_epochs = (
        str(spec["dark_epoch"]),
        str(spec["light_epoch"]),
        LIGHT_TEST_EPOCH,
    )
    positions = {
        (str(row["epoch"]), str(row["position_role"]))
        for row in rows["position"]
    }
    required_positions = {
        (epoch, role)
        for epoch in required_run_epochs
        for role in (
            PRIMARY_POSITION_ROLE,
            ORIENTATION_REFERENCE_POSITION_ROLE,
        )
    }
    missing_positions = required_positions.difference(positions)
    if missing_positions:
        raise ValueError(
            f"{raw_path.name} is missing figure position rows "
            f"{sorted(missing_positions)!r}."
        )
    trajectories = {
        (str(row["epoch"]), str(row["trajectory_type"]))
        for row in rows["trajectory_intervals"]
    }
    required_trajectories = {
        (epoch, trajectory)
        for epoch in required_run_epochs
        for trajectory in TRAJECTORY_TYPES
    }
    missing_trajectories = required_trajectories.difference(trajectories)
    if missing_trajectories:
        raise ValueError(
            f"{raw_path.name} is missing figure trajectories "
            f"{sorted(missing_trajectories)!r}."
        )
    graphs = {str(row["configuration_name"]) for row in rows["wtrack_graph"]}
    required_graphs = {*TRAJECTORY_TYPES, "full_w"}
    missing_graphs = required_graphs.difference(graphs)
    if missing_graphs:
        raise ValueError(
            f"{raw_path.name} is missing graph configurations "
            f"{sorted(missing_graphs)!r}."
        )
    ripple_epochs = {str(row["epoch"]) for row in rows["ripple_intervals"]}
    if str(spec["light_epoch"]) not in ripple_epochs:
        raise ValueError(
            f"{raw_path.name} has no ripple row for {spec['light_epoch']!r}."
        )
    return {
        "raw_path": raw_path,
        "catalog_name": catalog_name,
        "counts": dict(preview["counts"]),
        "planned_results": planned_result_counts(spec),
    }


def _source_status(
    runtime: Mapping[str, Any],
    *,
    standard_nwb_file_name: str,
    group_name: str,
) -> dict[str, Any]:
    """Return standard and project source-row readiness for one session."""
    key = {"nwb_file_name": standard_nwb_file_name}
    tables = runtime["tables"]
    group_key = {
        **key,
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": group_name,
    }
    return {
        "nwbfile": len(runtime["Nwbfile"] & key),
        "session": len(runtime["Session"] & key),
        "imported_spike_sorting": len(runtime["ImportedSpikeSorting"] & key),
        "sorted_spikes_group": len(runtime["SortedSpikesGroup"] & group_key),
        "epoch_intervals": len(tables["epoch_intervals"] & key),
        "region_sorted_spikes_group": len(
            tables["region_sorted_spikes_group"] & group_key
        ),
    }


def _onboard_sources(
    spec: Mapping[str, str],
    *,
    runtime: Mapping[str, Any],
    nwb_root: Path,
) -> dict[str, Any]:
    """Ingest standard sources and register project catalog and region views."""
    from v1ca1.spyglass.ingest import ingest_v1ca1_nwb

    raw_path = _raw_nwb_path(spec, nwb_root=nwb_root).resolve(strict=True)
    configured_raw_dir = Path(runtime["raw_dir"]).resolve(strict=True)
    if raw_path.parent != configured_raw_dir:
        raise ValueError(
            f"NWB root {raw_path.parent} does not match Spyglass raw_dir "
            f"{configured_raw_dir}."
        )
    nwb_file_name = _standard_nwb_file_name(runtime, raw_path)
    group_name = f"{nwb_file_name}_imported_all_units"
    _preflight_catalog(
        spec,
        nwb_root=nwb_root,
        standard_nwb_file_name=nwb_file_name,
    )

    standard_key = {"nwb_file_name": nwb_file_name}
    if not (runtime["Nwbfile"] & standard_key):
        _emit("standard_ingestion_started", nwb_file_name=nwb_file_name)
        runtime["insert_sessions"](
            raw_path.name,
            rollback_on_fail=True,
            raise_err=True,
        )
    elif not (runtime["Session"] & standard_key):
        _emit("common_population_started", nwb_file_name=nwb_file_name)
        runtime["populate_all_common"](
            nwb_file_name,
            rollback_on_fail=True,
            raise_err=True,
        )
    _one_row(runtime["Nwbfile"], standard_key, label="Nwbfile row")
    _one_row(runtime["Session"], standard_key, label="Session row")

    imported = runtime["ImportedSpikeSorting"]
    if not (imported & standard_key):
        _emit("imported_spike_sorting_started", nwb_file_name=nwb_file_name)
        imported().insert_from_nwbfile(nwb_file_name)
    _one_row(imported, standard_key, label="ImportedSpikeSorting row")

    merge_part = runtime["SpikeSortingOutput"].ImportedSpikeSorting
    merge_row = _one_row(
        merge_part,
        standard_key,
        label="ImportedSpikeSorting merge row",
    )
    merge_id = merge_row["merge_id"]
    unit_selection_params = runtime["UnitSelectionParams"]
    all_units_key = {"unit_filter_params_name": "all_units"}
    if not (unit_selection_params & all_units_key):
        unit_selection_params.insert_default()
    _one_row(
        unit_selection_params,
        all_units_key,
        label="all_units UnitSelectionParams row",
    )
    group_key = {
        **standard_key,
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": group_name,
    }
    sorted_group = runtime["SortedSpikesGroup"]
    if not (sorted_group & group_key):
        sorted_group().create_group(
            group_name,
            nwb_file_name,
            unit_filter_params_name="all_units",
            keys=[{"spikesorting_merge_id": merge_id}],
        )
    _one_row(sorted_group, group_key, label="SortedSpikesGroup row")
    member_rows = list((sorted_group.Units & group_key).fetch(as_dict=True))
    if len(member_rows) != 1 or str(
        member_rows[0]["spikesorting_merge_id"]
    ) != str(merge_id):
        raise ValueError(
            f"{group_name!r} does not contain exactly the imported merge row."
        )

    tables = runtime["tables"]
    ingest_v1ca1_nwb(
        nwb_file_name,
        tables=tables,
        skip_duplicates=True,
    )
    tables["region_sorted_spikes_group"].register_regions(
        group_key,
        region_names=FIGURE_REGIONS,
        skip_duplicates=True,
    )
    for region in FIGURE_REGIONS:
        _one_row(
            tables["region_sorted_spikes_group"],
            {**group_key, "region_name": region},
            label=f"{region} RegionSortedSpikesGroup row",
        )
    status = _source_status(
        runtime,
        standard_nwb_file_name=nwb_file_name,
        group_name=group_name,
    )
    _emit("sources_complete", nwb_file_name=nwb_file_name, status=status)
    return {
        "nwb_file_name": nwb_file_name,
        "group_name": group_name,
        "group_key": group_key,
        "status": status,
    }


def _insert_figure_parameters(tables: Mapping[str, Any]) -> None:
    """Insert exactly the parameter definitions used by current figures."""
    tables["movement_parameters"].insert_default(skip_duplicates=True)
    tables["epoch_motor_behavior_parameters"].insert_default(
        skip_duplicates=True
    )
    tables["cv_pca_parameters"].insert_parameters(
        table_specs.MANUSCRIPT_V1_CV_PCA_PARAMETERS,
        skip_duplicates=True,
    )
    tables["ripple_modulation_parameters"].insert_default(
        skip_duplicates=True
    )
    for row in table_specs.TUNING_CURVE_PARAMETER_PRESETS:
        tables["tuning_curve_parameters"].insert_parameters(
            row,
            skip_duplicates=True,
        )
    tables["tuning_similarity_parameters"].insert_parameters(
        table_specs.ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS,
        skip_duplicates=True,
    )
    tables["dpp_encoding_parameters"].insert_default(skip_duplicates=True)
    tables["path_progression_decoding_parameters"].insert_default(
        skip_duplicates=True
    )
    tables["path_specific_place_decoding_parameters"].insert_default(
        skip_duplicates=True
    )
    tables["motor_encoding_parameters"].insert_default(
        region="v1",
        skip_duplicates=True,
    )
    tables["dark_light_glm_parameters"].insert_parameters(
        table_specs.LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS,
        skip_duplicates=True,
    )
    tables["swap_glm_parameters"].insert_parameters(
        table_specs.DEFAULT_SWAP_GLM_PARAMETERS,
        skip_duplicates=True,
    )
    tables["swap_tuning_curve_comparison_parameters"].insert_parameters(
        table_specs.MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS,
        skip_duplicates=True,
    )
    for row in table_specs.RIPPLE_GLM_PARAMETER_PRESETS:
        tables["ripple_glm_parameters"].insert_parameters(
            row,
            skip_duplicates=True,
        )
    tables["ripple_cross_region_xcorr_parameters"].insert_parameters(
        table_specs.MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS,
        skip_duplicates=True,
    )


_RESULT_LOADERS = {
    "epoch_motor_behavior": ("load_epoch_motor_behavior_bundle",),
    "movement_firing_rate": ("load_firing_rates", "load_intervals"),
    "cv_pca": ("load_cv_pca_bundle",),
    "ripple_modulation": ("load_artifacts",),
    "path_specific_place_tuning_curve": ("load_tuning_curve",),
    "path_specific_place_tuning_similarity": ("load_similarity",),
    "path_specific_place_stability": ("load_stability",),
    "dpp_encoding": ("load_dpp_encoding",),
    "path_progression_decoding": ("load_decoding_bundle",),
    "path_specific_place_decoding": ("load_decoding_bundle",),
    "motor_encoding": ("load_motor_encoding_bundle",),
    "dark_light_glm": ("load_dark_light_glm_bundle",),
    "swap_glm": ("load_swap_glm_bundle",),
    "swap_tuning_curve_comparison": (
        "load_swap_tuning_curve_comparison_bundle",
    ),
    "ripple_glm": ("load_ripple_glm_bundle",),
    "ripple_cross_region_xcorr": (
        "load_ripple_cross_region_xcorr_bundle",
    ),
}


def _require_expected_result_primary_key(
    result_table: Any,
    *,
    result_table_key: str,
    id_field: str,
) -> None:
    """Reject a live result table whose key differs from its selection key."""
    observed = tuple(str(name) for name in result_table.primary_key)
    expected = (id_field,)
    if observed != expected:
        raise RuntimeError(
            f"{result_table_key} has live primary key {observed!r}; expected "
            f"{expected!r}. Repair the live schema before population."
        )


def _require_one_pending_populate_job(
    result_table: Any,
    result_key: Mapping[str, Any],
    *,
    result_table_key: str,
) -> None:
    """Require one and only one missing DataJoint job for a result key."""
    pending = (result_table.key_source & dict(result_key)) - result_table
    job_count = len(pending)
    if job_count != 1:
        raise RuntimeError(
            f"{result_table_key} has {job_count} pending populate jobs for "
            f"{dict(result_key)!r}; expected exactly one."
        )


def _insert_populate_verify(
    tables: Mapping[str, Any],
    *,
    selection_table_key: str,
    result_table_key: str,
    id_field: str,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Insert one immutable selection, populate it, and invoke its loaders."""
    selection_table = tables[selection_table_key]
    result_table = tables[result_table_key]
    _require_expected_result_primary_key(
        result_table,
        result_table_key=result_table_key,
        id_field=id_field,
    )
    selection = selection_table.insert_selection(key, skip_duplicates=True)
    result_key = {id_field: selection[id_field]}
    if not (result_table & result_key):
        _require_one_pending_populate_job(
            result_table,
            result_key,
            result_table_key=result_table_key,
        )
        _emit(
            "populate_started",
            table=result_table_key,
            key=result_key,
        )
        result_table.populate(result_key)
    _one_row(result_table, result_key, label=f"{result_table_key} result")
    for loader_name in _RESULT_LOADERS[result_table_key]:
        getattr(result_table, loader_name)(result_key)
    _emit(
        "result_verified",
        table=result_table_key,
        key=result_key,
    )
    return selection


def _region_row(
    tables: Mapping[str, Any],
    *,
    nwb_file_name: str,
    group_name: str,
    region: str,
) -> dict[str, Any]:
    """Return one region view belonging to the canonical imported group."""
    return _one_row(
        tables["region_sorted_spikes_group"],
        {
            "nwb_file_name": nwb_file_name,
            "unit_filter_params_name": "all_units",
            "sorted_spikes_group_name": group_name,
            "region_name": region,
        },
        label=f"{region} RegionSortedSpikesGroup row",
    )


def _position_series_name(
    tables: Mapping[str, Any],
    *,
    nwb_file_name: str,
    epoch: str,
    role: str,
) -> str:
    """Return the unique position series with one semantic role."""
    row = _one_row(
        tables["position"],
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_role": role,
        },
        label=f"{epoch} {role} Position row",
    )
    return str(row["position_series_name"])


def _movement_selection(
    tables: Mapping[str, Any],
    *,
    nwb_file_name: str,
    epoch: str,
    region_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one populated canonical movement selection."""
    row = _one_row(
        tables["movement_firing_rate_selection"],
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "movement_param_name": table_specs.DEFAULT_MOVEMENT_PARAMETERS[
                "movement_param_name"
            ],
            "region_sorted_spikes_group_id": region_row[
                "region_sorted_spikes_group_id"
            ],
        },
        label=f"{epoch} {region_row['region_name']} MovementFiringRateSelection",
    )
    _one_row(
        tables["movement_firing_rate"],
        {"movement_firing_rate_id": row["movement_firing_rate_id"]},
        label="MovementFiringRate result",
    )
    return row


def _populate_base(
    spec: Mapping[str, str],
    *,
    tables: Mapping[str, Any],
    nwb_file_name: str,
    group_name: str,
) -> None:
    """Populate motor summaries and all movement rows required by figures."""
    movement_param_name = table_specs.DEFAULT_MOVEMENT_PARAMETERS[
        "movement_param_name"
    ]
    motor_param_name = table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS[
        "epoch_motor_behavior_param_name"
    ]
    for epoch in (str(spec["dark_epoch"]), str(spec["light_epoch"])):
        _insert_populate_verify(
            tables,
            selection_table_key="epoch_motor_behavior_selection",
            result_table_key="epoch_motor_behavior",
            id_field="epoch_motor_behavior_id",
            key={
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "primary_position_series_name": _position_series_name(
                    tables,
                    nwb_file_name=nwb_file_name,
                    epoch=epoch,
                    role=PRIMARY_POSITION_ROLE,
                ),
                "orientation_reference_position_series_name": (
                    _position_series_name(
                        tables,
                        nwb_file_name=nwb_file_name,
                        epoch=epoch,
                        role=ORIENTATION_REFERENCE_POSITION_ROLE,
                    )
                ),
                "movement_param_name": movement_param_name,
                "epoch_motor_behavior_param_name": motor_param_name,
            },
        )

    region_rows = {
        region: _region_row(
            tables,
            nwb_file_name=nwb_file_name,
            group_name=group_name,
            region=region,
        )
        for region in FIGURE_REGIONS
    }
    movement_roles = (
        (str(spec["dark_epoch"]), "v1"),
        (str(spec["dark_epoch"]), "ca1"),
        (str(spec["light_epoch"]), "v1"),
        (LIGHT_TEST_EPOCH, "v1"),
    )
    for epoch, region in movement_roles:
        _insert_populate_verify(
            tables,
            selection_table_key="movement_firing_rate_selection",
            result_table_key="movement_firing_rate",
            id_field="movement_firing_rate_id",
            key={
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "position_series_name": _position_series_name(
                    tables,
                    nwb_file_name=nwb_file_name,
                    epoch=epoch,
                    role=PRIMARY_POSITION_ROLE,
                ),
                "movement_param_name": movement_param_name,
                "region_sorted_spikes_group_id": region_rows[region][
                    "region_sorted_spikes_group_id"
                ],
            },
        )


def _tuning_selection(
    tables: Mapping[str, Any],
    *,
    movement_firing_rate_id: Any,
    trajectory_type: str,
    parameter_name: str,
    trial_subset: str,
) -> dict[str, Any]:
    """Return one populated path-specific tuning selection."""
    row = _one_row(
        tables["path_specific_place_tuning_curve_selection"],
        {
            "movement_firing_rate_id": movement_firing_rate_id,
            "trajectory_type": trajectory_type,
            "configuration_name": trajectory_type,
            "tuning_curve_param_name": parameter_name,
            "trial_subset": trial_subset,
        },
        label="PathSpecificPlaceTuningCurveSelection",
    )
    _one_row(
        tables["path_specific_place_tuning_curve"],
        {
            "path_specific_place_tuning_curve_id": row[
                "path_specific_place_tuning_curve_id"
            ]
        },
        label="PathSpecificPlaceTuningCurve result",
    )
    return row


def _stability_selection(
    tables: Mapping[str, Any],
    *,
    movement_firing_rate_id: Any,
    trajectory_type: str,
) -> dict[str, Any]:
    """Return one populated stability row for canonical odd/even curves."""
    tuning_name = table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
        "tuning_curve_param_name"
    ]
    odd = _tuning_selection(
        tables,
        movement_firing_rate_id=movement_firing_rate_id,
        trajectory_type=trajectory_type,
        parameter_name=tuning_name,
        trial_subset="odd",
    )
    even = _tuning_selection(
        tables,
        movement_firing_rate_id=movement_firing_rate_id,
        trajectory_type=trajectory_type,
        parameter_name=tuning_name,
        trial_subset="even",
    )
    selection = _one_row(
        tables["path_specific_place_stability_selection"],
        {
            "odd_path_specific_place_tuning_curve_id": odd[
                "path_specific_place_tuning_curve_id"
            ],
            "even_path_specific_place_tuning_curve_id": even[
                "path_specific_place_tuning_curve_id"
            ],
        },
        label="PathSpecificPlaceStabilitySelection",
    )
    _one_row(
        tables["path_specific_place_stability"],
        {
            "path_specific_place_stability_id": selection[
                "path_specific_place_stability_id"
            ]
        },
        label="PathSpecificPlaceStability result",
    )
    return selection


def _populate_tuning(
    spec: Mapping[str, str],
    *,
    tables: Mapping[str, Any],
    nwb_file_name: str,
    group_name: str,
) -> None:
    """Populate figure tuning curves, stability, and similarity rows."""
    region_rows = {
        region: _region_row(
            tables,
            nwb_file_name=nwb_file_name,
            group_name=group_name,
            region=region,
        )
        for region in FIGURE_REGIONS
    }
    movements = {
        (epoch, region): _movement_selection(
            tables,
            nwb_file_name=nwb_file_name,
            epoch=epoch,
            region_row=region_rows[region],
        )
        for epoch, region in (
            (str(spec["dark_epoch"]), "v1"),
            (str(spec["dark_epoch"]), "ca1"),
            (str(spec["light_epoch"]), "v1"),
            (LIGHT_TEST_EPOCH, "v1"),
        )
    }
    legacy_name = table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
        "tuning_curve_param_name"
    ]
    dark_jobs = (
        (
            str(spec["dark_epoch"]),
            region,
            parameter["tuning_curve_param_name"],
            TRIAL_SUBSETS,
        )
        for region in FIGURE_REGIONS
        for parameter in table_specs.TUNING_CURVE_PARAMETER_PRESETS
    )
    light_jobs = (
        (str(spec["light_epoch"]), "v1", legacy_name, TRIAL_SUBSETS),
        (LIGHT_TEST_EPOCH, "v1", legacy_name, ("all",)),
    )
    for epoch, region, parameter_name, subsets in (*dark_jobs, *light_jobs):
        movement_id = movements[(epoch, region)]["movement_firing_rate_id"]
        for trajectory_type in TRAJECTORY_TYPES:
            for trial_subset in subsets:
                _insert_populate_verify(
                    tables,
                    selection_table_key=(
                        "path_specific_place_tuning_curve_selection"
                    ),
                    result_table_key="path_specific_place_tuning_curve",
                    id_field="path_specific_place_tuning_curve_id",
                    key={
                        "movement_firing_rate_id": movement_id,
                        "trajectory_type": trajectory_type,
                        "configuration_name": trajectory_type,
                        "tuning_curve_param_name": parameter_name,
                        "trial_subset": trial_subset,
                    },
                )

    for epoch, region in (
        (str(spec["dark_epoch"]), "v1"),
        (str(spec["dark_epoch"]), "ca1"),
        (str(spec["light_epoch"]), "v1"),
    ):
        movement_id = movements[(epoch, region)]["movement_firing_rate_id"]
        for trajectory_type in TRAJECTORY_TYPES:
            odd = _tuning_selection(
                tables,
                movement_firing_rate_id=movement_id,
                trajectory_type=trajectory_type,
                parameter_name=legacy_name,
                trial_subset="odd",
            )
            even = _tuning_selection(
                tables,
                movement_firing_rate_id=movement_id,
                trajectory_type=trajectory_type,
                parameter_name=legacy_name,
                trial_subset="even",
            )
            _insert_populate_verify(
                tables,
                selection_table_key="path_specific_place_stability_selection",
                result_table_key="path_specific_place_stability",
                id_field="path_specific_place_stability_id",
                key={
                    "odd_path_specific_place_tuning_curve_id": odd[
                        "path_specific_place_tuning_curve_id"
                    ],
                    "even_path_specific_place_tuning_curve_id": even[
                        "path_specific_place_tuning_curve_id"
                    ],
                },
            )

    similarity_name = (
        table_specs.ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS[
            "tuning_similarity_param_name"
        ]
    )
    for epoch in (str(spec["dark_epoch"]), str(spec["light_epoch"])):
        movement_id = movements[(epoch, "v1")]["movement_firing_rate_id"]
        curve_ids = {
            f"{trajectory_type}_tuning_curve_id": _tuning_selection(
                tables,
                movement_firing_rate_id=movement_id,
                trajectory_type=trajectory_type,
                parameter_name=legacy_name,
                trial_subset="all",
            )["path_specific_place_tuning_curve_id"]
            for trajectory_type in TRAJECTORY_TYPES
        }
        _insert_populate_verify(
            tables,
            selection_table_key=(
                "path_specific_place_tuning_similarity_selection"
            ),
            result_table_key="path_specific_place_tuning_similarity",
            id_field="path_specific_place_tuning_similarity_id",
            key={
                **curve_ids,
                "tuning_similarity_param_name": similarity_name,
            },
        )


def _stability_ids(
    tables: Mapping[str, Any],
    *,
    movement_firing_rate_id: Any,
) -> dict[str, Any]:
    """Return the four canonical stability result identifiers."""
    return {
        trajectory_type: _stability_selection(
            tables,
            movement_firing_rate_id=movement_firing_rate_id,
            trajectory_type=trajectory_type,
        )["path_specific_place_stability_id"]
        for trajectory_type in TRAJECTORY_TYPES
    }


def _populate_models(
    spec: Mapping[str, str],
    *,
    tables: Mapping[str, Any],
    nwb_file_name: str,
    group_name: str,
) -> None:
    """Populate the remaining non-ripple figure analysis families."""
    v1_row = _region_row(
        tables,
        nwb_file_name=nwb_file_name,
        group_name=group_name,
        region="v1",
    )
    region_id = v1_row["region_sorted_spikes_group_id"]
    movements = {
        role: _movement_selection(
            tables,
            nwb_file_name=nwb_file_name,
            epoch=epoch,
            region_row=v1_row,
        )
        for role, epoch in (
            ("dark", str(spec["dark_epoch"])),
            ("light_train", str(spec["light_epoch"])),
            ("light_test", LIGHT_TEST_EPOCH),
        )
    }
    stability = {
        role: _stability_ids(
            tables,
            movement_firing_rate_id=movements[role][
                "movement_firing_rate_id"
            ],
        )
        for role in ("dark", "light_train")
    }

    _insert_populate_verify(
        tables,
        selection_table_key="cv_pca_selection",
        result_table_key="cv_pca",
        id_field="cv_pca_id",
        key={
            "nwb_file_name": nwb_file_name,
            "light_epoch": str(spec["light_epoch"]),
            "dark_epoch": str(spec["dark_epoch"]),
            "region_sorted_spikes_group_id": region_id,
            "light_movement_firing_rate_id": movements["light_train"][
                "movement_firing_rate_id"
            ],
            "dark_movement_firing_rate_id": movements["dark"][
                "movement_firing_rate_id"
            ],
            "cv_pca_param_name": table_specs.MANUSCRIPT_V1_CV_PCA_PARAMETERS[
                "cv_pca_param_name"
            ],
        },
    )

    common_stability_fields = {
        f"{trajectory_type}_stability_id": stability["dark"][trajectory_type]
        for trajectory_type in TRAJECTORY_TYPES
    }
    _insert_populate_verify(
        tables,
        selection_table_key="dpp_encoding_selection",
        result_table_key="dpp_encoding",
        id_field="dpp_encoding_id",
        key={
            "region_sorted_spikes_group_id": region_id,
            "movement_firing_rate_id": movements["dark"][
                "movement_firing_rate_id"
            ],
            "full_w_configuration_name": "full_w",
            **common_stability_fields,
            "dpp_encoding_param_name": (
                table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS[
                    "dpp_encoding_param_name"
                ]
            ),
        },
    )

    for role in ("dark", "light_train"):
        movement_id = movements[role]["movement_firing_rate_id"]
        stability_ids = stability[role]
        decoding_stability_fields = {
            f"{trajectory_type}_stability_id": stability_ids[trajectory_type]
            for trajectory_type in TRAJECTORY_TYPES
        }
        cohort_fields = {
            f"cohort_{trajectory_type}_stability_id": stability_ids[
                trajectory_type
            ]
            for trajectory_type in TRAJECTORY_TYPES
        }
        _insert_populate_verify(
            tables,
            selection_table_key="path_progression_decoding_selection",
            result_table_key="path_progression_decoding",
            id_field="path_progression_decoding_id",
            key={
                "region_sorted_spikes_group_id": region_id,
                "movement_firing_rate_id": movement_id,
                "cohort_movement_firing_rate_id": movement_id,
                **decoding_stability_fields,
                **cohort_fields,
                "path_progression_decoding_param_name": (
                    table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS[
                        "path_progression_decoding_param_name"
                    ]
                ),
            },
        )
        _insert_populate_verify(
            tables,
            selection_table_key="path_specific_place_decoding_selection",
            result_table_key="path_specific_place_decoding",
            id_field="path_specific_place_decoding_id",
            key={
                "region_sorted_spikes_group_id": region_id,
                "movement_firing_rate_id": movement_id,
                "path_specific_place_decoding_param_name": (
                    table_specs.MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS[
                        "path_specific_place_decoding_param_name"
                    ]
                ),
            },
        )

    _insert_populate_verify(
        tables,
        selection_table_key="motor_encoding_selection",
        result_table_key="motor_encoding",
        id_field="motor_encoding_id",
        key={
            "region_sorted_spikes_group_id": region_id,
            "movement_firing_rate_id": movements["dark"][
                "movement_firing_rate_id"
            ],
            "primary_position_series_name": _position_series_name(
                tables,
                nwb_file_name=nwb_file_name,
                epoch=str(spec["dark_epoch"]),
                role=PRIMARY_POSITION_ROLE,
            ),
            "orientation_reference_position_series_name": (
                _position_series_name(
                    tables,
                    nwb_file_name=nwb_file_name,
                    epoch=str(spec["dark_epoch"]),
                    role=ORIENTATION_REFERENCE_POSITION_ROLE,
                )
            ),
            "full_w_configuration_name": "full_w",
            **common_stability_fields,
            "motor_encoding_param_name": (
                table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS[
                    "motor_encoding_param_name"
                ]
            ),
        },
    )

    dark_light_selection = _insert_populate_verify(
        tables,
        selection_table_key="dark_light_glm_selection",
        result_table_key="dark_light_glm",
        id_field="dark_light_glm_id",
        key={
            "region_sorted_spikes_group_id": region_id,
            "dark_movement_firing_rate_id": movements["dark"][
                "movement_firing_rate_id"
            ],
            "light_movement_firing_rate_id": movements["light_train"][
                "movement_firing_rate_id"
            ],
            "dark_light_glm_param_name": (
                table_specs.LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS[
                    "dark_light_glm_param_name"
                ]
            ),
        },
    )
    _insert_populate_verify(
        tables,
        selection_table_key="swap_glm_selection",
        result_table_key="swap_glm",
        id_field="swap_glm_id",
        key={
            "dark_light_glm_id": dark_light_selection["dark_light_glm_id"],
            "region_sorted_spikes_group_id": region_id,
            "light_test_movement_firing_rate_id": movements["light_test"][
                "movement_firing_rate_id"
            ],
            "swap_glm_param_name": table_specs.DEFAULT_SWAP_GLM_PARAMETERS[
                "swap_glm_param_name"
            ],
        },
    )

    legacy_name = table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
        "tuning_curve_param_name"
    ]
    swap_curve_fields = {
        f"{role}_{trajectory_type}_tuning_curve_id": _tuning_selection(
            tables,
            movement_firing_rate_id=movements[role][
                "movement_firing_rate_id"
            ],
            trajectory_type=trajectory_type,
            parameter_name=legacy_name,
            trial_subset="all",
        )["path_specific_place_tuning_curve_id"]
        for role in ("dark", "light_train", "light_test")
        for trajectory_type in TRAJECTORY_TYPES
    }
    _insert_populate_verify(
        tables,
        selection_table_key="swap_tuning_curve_comparison_selection",
        result_table_key="swap_tuning_curve_comparison",
        id_field="swap_tuning_curve_comparison_id",
        key={
            "region_sorted_spikes_group_id": region_id,
            **{
                f"{role}_movement_firing_rate_id": movements[role][
                    "movement_firing_rate_id"
                ]
                for role in ("dark", "light_train", "light_test")
            },
            **swap_curve_fields,
            "swap_tuning_curve_comparison_param_name": (
                table_specs.MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS[
                    "swap_tuning_curve_comparison_param_name"
                ]
            ),
        },
    )


def _require_jax_gpu() -> dict[str, Any]:
    """Require the GPU backend recorded by the ripple figure contract."""
    import jax

    devices = list(jax.devices())
    if jax.default_backend() != "gpu" or not devices:
        raise RuntimeError(
            "RippleGLM requires a visible JAX GPU before any ripple rows are "
            "populated."
        )
    return {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
    }


def _populate_ripple(
    spec: Mapping[str, str],
    *,
    tables: Mapping[str, Any],
    nwb_file_name: str,
    group_name: str,
) -> None:
    """Populate the light-epoch ripple result families."""
    gpu = _require_jax_gpu()
    _emit("gpu_preflight_complete", **gpu)
    region_rows = {
        region: _region_row(
            tables,
            nwb_file_name=nwb_file_name,
            group_name=group_name,
            region=region,
        )
        for region in FIGURE_REGIONS
    }
    epoch = str(spec["light_epoch"])
    modulation_name = table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS[
        "ripple_modulation_param_name"
    ]
    for region in FIGURE_REGIONS:
        _insert_populate_verify(
            tables,
            selection_table_key="ripple_modulation_selection",
            result_table_key="ripple_modulation",
            id_field="ripple_modulation_id",
            key={
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "ripple_modulation_param_name": modulation_name,
                "region_sorted_spikes_group_id": region_rows[region][
                    "region_sorted_spikes_group_id"
                ],
            },
        )
    cross_region_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "source_region_sorted_spikes_group_id": region_rows["ca1"][
            "region_sorted_spikes_group_id"
        ],
        "target_region_sorted_spikes_group_id": region_rows["v1"][
            "region_sorted_spikes_group_id"
        ],
    }
    for parameters in table_specs.RIPPLE_GLM_PARAMETER_PRESETS:
        _insert_populate_verify(
            tables,
            selection_table_key="ripple_glm_selection",
            result_table_key="ripple_glm",
            id_field="ripple_glm_id",
            key={
                **cross_region_key,
                "ripple_glm_param_name": parameters[
                    "ripple_glm_param_name"
                ],
            },
        )
    if (str(spec["animal_name"]), str(spec["date"])) == XCORR_DATASET:
        parameters = table_specs.MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS
        _insert_populate_verify(
            tables,
            selection_table_key="ripple_cross_region_xcorr_selection",
            result_table_key="ripple_cross_region_xcorr",
            id_field="ripple_cross_region_xcorr_id",
            key={
                **cross_region_key,
                "ripple_cross_region_xcorr_param_name": parameters[
                    "ripple_cross_region_xcorr_param_name"
                ],
            },
        )


def _population_status(
    spec: Mapping[str, str],
    *,
    runtime: Mapping[str, Any],
    nwb_root: Path,
) -> dict[str, Any]:
    """Return source and result counts for one configured session."""
    raw_path = _raw_nwb_path(spec, nwb_root=nwb_root)
    nwb_file_name = _standard_nwb_file_name(runtime, raw_path)
    group_name = f"{nwb_file_name}_imported_all_units"
    tables = runtime["tables"]
    counts = {}
    for result_key in planned_result_counts(spec):
        selection = tables[f"{result_key}_selection"]
        result = tables[result_key]
        if "nwb_file_name" in selection.heading.names:
            session_selection = selection & {
                "nwb_file_name": nwb_file_name
            }
        else:
            reference_field = _INDIRECT_SESSION_REFERENCE_FIELDS.get(
                result_key
            )
            if reference_field is None:
                raise ValueError(
                    f"Cannot associate {result_key!r} selections with a "
                    "session."
                )
            tuning_selection = tables[
                "path_specific_place_tuning_curve_selection"
            ] & {"nwb_file_name": nwb_file_name}
            session_selection = selection & tuning_selection.proj(
                **{reference_field: _TUNING_SELECTION_ID}
            )
        counts[result_key] = len(result & session_selection.proj())
    return {
        "nwb_file_name": nwb_file_name,
        "sources": _source_status(
            runtime,
            standard_nwb_file_name=nwb_file_name,
            group_name=group_name,
        ),
        "results": counts,
        "planned_results": planned_result_counts(spec),
    }


def populate_figure_dataset(
    spec: Mapping[str, str],
    *,
    stage: str,
    dry_run: bool,
    nwb_root: Path,
    schema_name: str,
    analysis_nwbfile_schema_name: str,
) -> dict[str, Any]:
    """Execute one requested population stage for one manuscript session."""
    raw_path = _raw_nwb_path(spec, nwb_root=nwb_root)
    if dry_run or stage == "preflight":
        preview = _preflight_catalog(spec, nwb_root=nwb_root)
        _emit(
            "preflight_complete",
            animal_name=spec["animal_name"],
            date=spec["date"],
            **preview,
        )
        if dry_run or stage == "preflight":
            return preview

    runtime = _load_runtime(
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
    )
    nwb_file_name = _standard_nwb_file_name(runtime, raw_path)
    group_name = f"{nwb_file_name}_imported_all_units"
    if stage == "status":
        status = _population_status(
            spec,
            runtime=runtime,
            nwb_root=nwb_root,
        )
        _emit(
            "population_status",
            animal_name=spec["animal_name"],
            date=spec["date"],
            **status,
        )
        return status

    for selected_stage in stages_through(stage):
        _emit(
            "stage_started",
            stage=selected_stage,
            animal_name=spec["animal_name"],
            date=spec["date"],
        )
        if selected_stage == "sources":
            source = _onboard_sources(
                spec,
                runtime=runtime,
                nwb_root=nwb_root,
            )
            nwb_file_name = source["nwb_file_name"]
            group_name = source["group_name"]
        else:
            _insert_figure_parameters(runtime["tables"])
            source_status = _source_status(
                runtime,
                standard_nwb_file_name=nwb_file_name,
                group_name=group_name,
            )
            if source_status["region_sorted_spikes_group"] != 2:
                raise ValueError(
                    "Populate the sources stage before computed figure stages."
                )
            stage_functions = {
                "base": _populate_base,
                "tuning": _populate_tuning,
                "models": _populate_models,
                "ripple": _populate_ripple,
            }
            stage_functions[selected_stage](
                spec,
                tables=runtime["tables"],
                nwb_file_name=nwb_file_name,
                group_name=group_name,
            )
        _emit(
            "stage_complete",
            stage=selected_stage,
            animal_name=spec["animal_name"],
            date=spec["date"],
        )
    status = _population_status(
        spec,
        runtime=runtime,
        nwb_root=nwb_root,
    )
    _emit(
        "population_status",
        animal_name=spec["animal_name"],
        date=spec["date"],
        **status,
    )
    return status


def _parser() -> argparse.ArgumentParser:
    """Build the explicit manuscript population CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal-name")
    parser.add_argument("--date")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--stage", choices=STAGE_CHOICES, default="status")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nwb-root", type=Path, default=DEFAULT_NWB_ROOT)
    parser.add_argument(
        "--schema-name",
        default=table_specs.DEFAULT_SCHEMA_NAME,
    )
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Populate one or all configured manuscript sessions."""
    args = _parser().parse_args(argv)
    specs = select_figure_datasets(
        animal_name=args.animal_name,
        date=args.date,
        all_datasets=args.all_datasets,
    )
    for spec in specs:
        populate_figure_dataset(
            spec,
            stage=args.stage,
            dry_run=args.dry_run,
            nwb_root=args.nwb_root,
            schema_name=args.schema_name,
            analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "STAGE_CHOICES",
    "STAGE_ORDER",
    "figure_dataset_specs",
    "main",
    "planned_result_counts",
    "populate_figure_dataset",
    "select_figure_datasets",
    "stages_through",
]
