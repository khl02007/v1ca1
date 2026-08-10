"""Run the initial Figure 1 analysis slice without DataJoint or Spyglass."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
import shutil
from typing import Any

from v1ca1.helper.session import REGIONS, TRAJECTORY_TYPES
from v1ca1.spyglass import movement, path_specific_place, stability
from v1ca1.spyglass.offline.manifests import (
    DEFAULT_SCRATCH_ROOT,
    MANIFEST_SCHEMA_VERSION,
    SESSION_MANIFEST_FILENAME,
    append_session_manifest,
    code_provenance,
    file_sha256,
    get_run_dir,
    get_session_dir,
    nwb_fingerprint,
    prepare_campaign,
    relative_run_path,
    utc_now,
    write_json_once,
)
from v1ca1.spyglass.offline.sources import (
    SOURCE_IDENTITY_POLICY,
    load_figure_1_catalog_objects,
    load_nwb_region_spikes,
    select_figure_1_catalog,
    validate_nwb_session_identity,
)
from v1ca1.spyglass.selection import (
    provenance_sha256,
    selection_uuid,
    unit_identity_sha256,
)
from v1ca1.spyglass.table_specs import (
    DEFAULT_MOVEMENT_PARAMETERS,
    FIGURE_1D_TUNING_CURVE_PARAMETERS,
    LEGACY_TUNING_CURVE_PARAMETERS,
)


DEFAULT_NWB_ROOT = Path("/stelmo/nwb/raw")
DEFAULT_TUNING_PARAMETER_PRESETS = (
    LEGACY_TUNING_CURVE_PARAMETERS,
    FIGURE_1D_TUNING_CURVE_PARAMETERS,
)
DEFAULT_STABILITY_TUNING_PARAM_NAME = str(
    LEGACY_TUNING_CURVE_PARAMETERS["tuning_curve_param_name"]
)
TRIAL_SUBSETS = ("all", "odd", "even")


def _parameter_configuration(
    *,
    movement_parameters: Mapping[str, Any],
    tuning_parameter_presets: Sequence[Mapping[str, Any]],
    stability_tuning_param_name: str,
    position_role: str,
    regions: Sequence[str],
    trajectory_types: Sequence[str],
) -> dict[str, Any]:
    """Validate and freeze the offline Figure 1 parameter configuration."""
    movement_parameters = dict(movement_parameters)
    required_movement = set(DEFAULT_MOVEMENT_PARAMETERS)
    if set(movement_parameters) != required_movement:
        raise ValueError("Movement parameters do not match the declared schema.")
    movement_name = movement_parameters["movement_param_name"]
    if (
        not isinstance(movement_name, str)
        or not movement_name.strip()
        or len(movement_name) > 64
    ):
        raise ValueError(
            "movement_param_name must be a non-empty string of at most 64 "
            "characters."
        )
    for field in ("speed_threshold_cm_s", "speed_smoothing_sigma_s"):
        value = movement_parameters[field]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field} must be one numeric scalar.")
    threshold, sigma = movement._validate_parameters(
        movement_parameters["speed_threshold_cm_s"],
        movement_parameters["speed_smoothing_sigma_s"],
    )
    movement_parameters["speed_threshold_cm_s"] = threshold
    movement_parameters["speed_smoothing_sigma_s"] = sigma

    tuning_parameters: list[dict[str, Any]] = []
    for raw_parameters in tuning_parameter_presets:
        parameters = dict(raw_parameters)
        name = str(parameters.get("tuning_curve_param_name", "")).strip()
        if not name:
            raise ValueError("Every tuning-curve preset requires a non-empty name.")
        validated = path_specific_place.validate_binning_parameters(
            bin_size_cm=parameters.get("place_bin_size_cm"),
            bin_count=parameters.get("position_bin_count"),
            sigma_bins=parameters.get("gaussian_smoothing_sigma_bins", 0.0),
        )
        if str(parameters.get("binning_mode")) != validated["binning_mode"]:
            raise ValueError(
                f"Tuning preset {name!r} has an inconsistent binning_mode."
            )
        tuning_parameters.append(
            {
                "tuning_curve_param_name": name,
                "binning_mode": validated["binning_mode"],
                "place_bin_size_cm": validated["bin_size_cm"],
                "position_bin_count": validated["bin_count"],
                "gaussian_smoothing_sigma_bins": validated["sigma_bins"],
            }
        )
    names = [row["tuning_curve_param_name"] for row in tuning_parameters]
    if len(names) != len(set(names)):
        raise ValueError("Tuning-curve preset names must be unique.")
    if stability_tuning_param_name not in names:
        raise ValueError("The stability tuning preset must be among computed presets.")

    normalized_regions = tuple(str(value).strip().casefold() for value in regions)
    normalized_trajectories = tuple(str(value) for value in trajectory_types)
    if (
        not normalized_regions
        or any(not value for value in normalized_regions)
        or len(normalized_regions) != len(set(normalized_regions))
    ):
        raise ValueError("regions must be a non-empty unique sequence.")
    if (
        not normalized_trajectories
        or any(not value for value in normalized_trajectories)
        or len(normalized_trajectories) != len(set(normalized_trajectories))
    ):
        raise ValueError("trajectory_types must be a non-empty unique sequence.")
    return {
        "pipeline": "figure_1_initial_slice",
        "movement_parameters": movement_parameters,
        "tuning_curve_parameter_presets": tuning_parameters,
        "stability_tuning_curve_param_name": str(stability_tuning_param_name),
        "trial_subsets": list(TRIAL_SUBSETS),
        "position_role": str(position_role),
        "regions": list(normalized_regions),
        "trajectory_types": list(normalized_trajectories),
        "diagnostic_figures": False,
    }


def _offline_region_group_id(
    *,
    nwb_file_name: str,
    region: str,
    loaded_spikes: Mapping[str, Any],
) -> str:
    """Return a deterministic offline regional-view identifier."""
    return str(
        selection_uuid(
            "OfflineRegionSortedSpikesView",
            {
                "nwb_file_name": nwb_file_name,
                "source": "ImportedSpikeSorting",
                "spikesorting_merge_id": loaded_spikes[
                    "spikesorting_merge_id"
                ],
                "region_name": region,
                "unit_filter_params_name": "all_units",
                "n_units": loaded_spikes["n_units"],
                "selected_units_sha256": loaded_spikes[
                    "selected_units_sha256"
                ],
            },
        )
    )


def _movement_selection(
    *,
    nwb_file_name: str,
    epoch: str,
    position_series_name: str,
    region_group_id: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic offline movement selection snapshot."""
    parameters_sha256 = provenance_sha256(dict(parameters))
    payload = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "position_series_name": position_series_name,
        "movement_param_name": parameters["movement_param_name"],
        "region_sorted_spikes_group_id": region_group_id,
        "movement_parameters_sha256": parameters_sha256,
    }
    return {
        "movement_firing_rate_id": str(
            selection_uuid("MovementFiringRate", payload)
        ),
        **payload,
    }


def _tuning_selection(
    *,
    movement_selection: Mapping[str, Any],
    trajectory_type: str,
    trial_subset: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic offline path-specific tuning selection."""
    payload = {
        "nwb_file_name": movement_selection["nwb_file_name"],
        "epoch": movement_selection["epoch"],
        "trajectory_type": trajectory_type,
        "configuration_name": trajectory_type,
        "movement_firing_rate_id": movement_selection[
            "movement_firing_rate_id"
        ],
        "tuning_curve_param_name": parameters["tuning_curve_param_name"],
        "trial_subset": trial_subset,
        "tuning_curve_parameters_sha256": provenance_sha256(dict(parameters)),
    }
    return {
        "path_specific_place_tuning_curve_id": str(
            selection_uuid("PathSpecificPlaceTuningCurve", payload)
        ),
        **payload,
    }


def _curve_attributes(
    selection: Mapping[str, Any],
    *,
    selected_units_sha256: str,
) -> dict[str, str]:
    """Return the immutable selection link stored with one NetCDF artifact."""
    return {
        **{name: str(value) for name, value in selection.items()},
        "selected_units_sha256": str(selected_units_sha256),
    }


def _artifact_record_paths(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
    path_fields: Sequence[str],
) -> dict[str, Any]:
    """Convert artifact paths to guarded run-relative values and add hashes."""
    output = dict(record)
    hashes: dict[str, str] = {}
    for field in path_fields:
        path = Path(output[field])
        output[field] = relative_run_path(path, run_dir=run_dir)
        if path.is_file():
            hashes[field] = file_sha256(path)
    output["artifact_sha256"] = hashes
    return output


def _run_figure_1_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_path: Path | None = None,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    regions: Sequence[str] = REGIONS,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    position_role: str = "head",
    movement_parameters: Mapping[str, Any] = DEFAULT_MOVEMENT_PARAMETERS,
    tuning_parameter_presets: Sequence[Mapping[str, Any]] = (
        DEFAULT_TUNING_PARAMETER_PRESETS
    ),
    stability_tuning_param_name: str = DEFAULT_STABILITY_TUNING_PARAM_NAME,
) -> dict[str, Any]:
    """Compute and persist one session's initial Figure 1 vertical slice."""
    animal_name, date, epoch = map(str, (animal_name, date, epoch))
    if nwb_path is None:
        nwb_path = Path(nwb_root) / f"{animal_name}{date}_augmented.nwb"
    nwb_path = Path(nwb_path).expanduser().resolve(strict=True)
    nwb_file_name = nwb_path.name
    configuration = _parameter_configuration(
        movement_parameters=movement_parameters,
        tuning_parameter_presets=tuning_parameter_presets,
        stability_tuning_param_name=stability_tuning_param_name,
        position_role=position_role,
        regions=regions,
        trajectory_types=trajectory_types,
    )
    run_dir, campaign = prepare_campaign(
        run_id=run_id,
        analysis_parameters=configuration,
        source_identity_policy=SOURCE_IDENTITY_POLICY,
        scratch_root=scratch_root,
    )
    if nwb_path.is_relative_to(run_dir.resolve(strict=False)):
        raise ValueError("Source NWB must not be located inside the output run.")
    session_dir = get_session_dir(
        run_dir,
        animal_name=animal_name,
        date=date,
    )
    if session_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing offline session: {session_dir}"
        )

    import pynwb

    artifact_records: dict[str, list[dict[str, Any]]] = {
        "movement_firing_rate": [],
        "path_specific_place_tuning_curve": [],
        "path_specific_place_stability": [],
    }
    source_identity: list[dict[str, Any]] = []
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=animal_name,
            date=date,
        )
        catalog_selection = select_figure_1_catalog(
            nwbfile,
            nwb_file_name=nwb_file_name,
            epoch=epoch,
            position_role=position_role,
            trajectory_types=configuration["trajectory_types"],
        )
        sources = load_figure_1_catalog_objects(nwbfile, catalog_selection)
        epoch_start = float(catalog_selection["epoch_row"]["start_time"])
        epoch_stop = float(catalog_selection["epoch_row"]["stop_time"])
        fingerprint = nwb_fingerprint(nwb_path, nwbfile)

        for region in configuration["regions"]:
            loaded_spikes = load_nwb_region_spikes(
                nwbfile,
                nwb_file_name=nwb_file_name,
                region=region,
                time_support=(epoch_start, epoch_stop),
            )
            region_group_id = _offline_region_group_id(
                nwb_file_name=nwb_file_name,
                region=region,
                loaded_spikes=loaded_spikes,
            )
            source_identity.append(
                {
                    "region": region,
                    "source": loaded_spikes["source"],
                    "spikesorting_merge_id": loaded_spikes[
                        "spikesorting_merge_id"
                    ],
                    "offline_region_sorted_spikes_view_id": region_group_id,
                    "n_units": loaded_spikes["n_units"],
                    "selected_units_sha256": loaded_spikes[
                        "selected_units_sha256"
                    ],
                }
            )
            movement_selection = _movement_selection(
                nwb_file_name=nwb_file_name,
                epoch=epoch,
                position_series_name=catalog_selection["position_row"][
                    "position_series_name"
                ],
                region_group_id=region_group_id,
                parameters=configuration["movement_parameters"],
            )
            movement_result = movement.compute_selected_movement_firing_rate(
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                spikes=loaded_spikes["ts_group"],
                stable_unit_ids=loaded_spikes["unit_ids"],
                position=sources["position"],
                speed_threshold_cm_s=configuration["movement_parameters"][
                    "speed_threshold_cm_s"
                ],
                speed_smoothing_sigma_s=configuration["movement_parameters"][
                    "speed_smoothing_sigma_s"
                ],
            )
            movement_paths = movement.get_movement_artifact_paths(
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                region=region,
                movement_firing_rate_id=movement_selection[
                    "movement_firing_rate_id"
                ],
                artifact_root=run_dir,
            )
            written_movement = movement.write_movement_artifacts(
                movement_result["table"],
                movement_result["movement_intervals"],
                movement_paths["artifact_dir"],
                overwrite=False,
            )
            movement_loaded = movement.load_movement_artifacts(
                movement_paths["artifact_dir"]
            )
            artifact_records["movement_firing_rate"].append(
                _artifact_record_paths(
                    {
                        **movement_selection,
                        "region": region,
                        "artifact_dir": movement_paths["artifact_dir"],
                        "firing_rate_path": written_movement["firing_rate_path"],
                        "movement_intervals_path": written_movement[
                            "movement_intervals_path"
                        ],
                        "analysis_status": movement_result["analysis_status"],
                        "n_units": movement_result["n_units"],
                        "n_valid_units": movement_result["n_valid_units"],
                        "selected_units_sha256": loaded_spikes[
                            "selected_units_sha256"
                        ],
                    },
                    run_dir=run_dir,
                    path_fields=(
                        "artifact_dir",
                        "firing_rate_path",
                        "movement_intervals_path",
                    ),
                )
            )

            curves: dict[tuple[str, str, str], tuple[Any, dict[str, Any]]] = {}
            for parameters in configuration["tuning_curve_parameter_presets"]:
                for trajectory_type in configuration["trajectory_types"]:
                    for trial_subset in TRIAL_SUBSETS:
                        tuning_selection = _tuning_selection(
                            movement_selection=movement_selection,
                            trajectory_type=trajectory_type,
                            trial_subset=trial_subset,
                            parameters=parameters,
                        )
                        result = (
                            path_specific_place
                            .compute_selected_path_specific_place_tuning_curve(
                                animal_name=animal_name,
                                date=date,
                                region=region,
                                epoch=epoch,
                                trajectory_type=trajectory_type,
                                trial_subset=trial_subset,
                                spikes=loaded_spikes["ts_group"],
                                stable_unit_ids=loaded_spikes["unit_ids"],
                                position=(
                                    sources["position"]
                                    if movement_loaded["analysis_status"]
                                    == "valid"
                                    else None
                                ),
                                trajectory_intervals=sources[
                                    "trajectory_intervals"
                                ][trajectory_type],
                                graph_inputs=sources["graph_inputs"][
                                    trajectory_type
                                ],
                                movement_intervals=movement_loaded[
                                    "movement_intervals"
                                ],
                                movement_analysis_status=movement_loaded[
                                    "analysis_status"
                                ],
                                bin_size_cm=parameters["place_bin_size_cm"],
                                bin_count=parameters["position_bin_count"],
                                sigma_bins=parameters[
                                    "gaussian_smoothing_sigma_bins"
                                ],
                            )
                        )
                        result["tuning_curve"].attrs.update(
                            _curve_attributes(
                                tuning_selection,
                                selected_units_sha256=loaded_spikes[
                                    "selected_units_sha256"
                                ],
                            )
                        )
                        tuning_path = (
                            path_specific_place
                            .get_path_specific_place_artifact_path(
                                animal_name=animal_name,
                                date=date,
                                epoch=epoch,
                                trajectory_type=trajectory_type,
                                trial_subset=trial_subset,
                                region=region,
                                path_specific_place_tuning_curve_id=(
                                    tuning_selection[
                                        "path_specific_place_tuning_curve_id"
                                    ]
                                ),
                                artifact_root=run_dir,
                            )
                        )
                        path_specific_place.write_path_specific_place_artifact(
                            result["tuning_curve"],
                            tuning_path,
                            overwrite=False,
                        )
                        loaded_curve = (
                            path_specific_place.load_path_specific_place_artifact(
                                tuning_path
                            )
                        )
                        key = (
                            parameters["tuning_curve_param_name"],
                            trajectory_type,
                            trial_subset,
                        )
                        curves[key] = (loaded_curve, tuning_selection)
                        artifact_records[
                            "path_specific_place_tuning_curve"
                        ].append(
                            _artifact_record_paths(
                                {
                                    **tuning_selection,
                                    "region": region,
                                    "tuning_curve_path": tuning_path,
                                    "analysis_status": result[
                                        "analysis_status"
                                    ],
                                    "n_units": result["n_units"],
                                    "n_valid_units": result["n_valid_units"],
                                    "n_trials": result["n_trials"],
                                    "n_position_bins": result[
                                        "n_position_bins"
                                    ],
                                    "selected_units_sha256": loaded_spikes[
                                        "selected_units_sha256"
                                    ],
                                },
                                run_dir=run_dir,
                                path_fields=("tuning_curve_path",),
                            )
                        )

            for trajectory_type in configuration["trajectory_types"]:
                odd_curve, odd_selection = curves[
                    (
                        stability_tuning_param_name,
                        trajectory_type,
                        "odd",
                    )
                ]
                even_curve, even_selection = curves[
                    (
                        stability_tuning_param_name,
                        trajectory_type,
                        "even",
                    )
                ]
                stability_payload = {
                    "odd_path_specific_place_tuning_curve_id": odd_selection[
                        "path_specific_place_tuning_curve_id"
                    ],
                    "even_path_specific_place_tuning_curve_id": even_selection[
                        "path_specific_place_tuning_curve_id"
                    ],
                }
                stability_id = str(
                    selection_uuid(
                        "PathSpecificPlaceStability",
                        stability_payload,
                    )
                )
                result = stability.compute_selected_stability_from_tuning_curves(
                    odd_tuning_curve=odd_curve,
                    even_tuning_curve=even_curve,
                    movement_firing_rate_table=movement_loaded["table"],
                )
                stability_path = stability.get_stability_artifact_path(
                    animal_name=animal_name,
                    date=date,
                    epoch=epoch,
                    trajectory_type=trajectory_type,
                    region=region,
                    path_specific_place_stability_id=stability_id,
                    artifact_root=run_dir,
                )
                stability.write_stability_artifact(
                    result["table"],
                    stability_path,
                    overwrite=False,
                )
                artifact_records["path_specific_place_stability"].append(
                    _artifact_record_paths(
                        {
                            "path_specific_place_stability_id": stability_id,
                            **stability_payload,
                            "nwb_file_name": nwb_file_name,
                            "epoch": epoch,
                            "region": region,
                            "trajectory_type": trajectory_type,
                            "tuning_curve_param_name": (
                                stability_tuning_param_name
                            ),
                            "movement_firing_rate_id": movement_selection[
                                "movement_firing_rate_id"
                            ],
                            "stability_path": stability_path,
                            "analysis_status": result["analysis_status"],
                            "n_units": result["n_units"],
                            "n_valid_units": result["n_valid_units"],
                            "selected_units_sha256": unit_identity_sha256(
                                loaded_spikes["unit_ids"]
                            ),
                        },
                        run_dir=run_dir,
                        path_fields=("stability_path",),
                    )
                )

    session_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_utc": utc_now(),
        "code_provenance": code_provenance(),
        "status": "complete",
        "animal_name": animal_name,
        "date": date,
        "nwb_file_name": nwb_file_name,
        "nwb_path": str(nwb_path),
        "nwb_fingerprint": fingerprint,
        "epochs": [epoch],
        "regions": list(configuration["regions"]),
        "trajectories": list(configuration["trajectory_types"]),
        "position_selection": {
            name: catalog_selection["position_row"][name]
            for name in (
                "epoch",
                "position_series_name",
                "position_role",
                "analysis_start_offset_samples",
                "spatial_unit",
                "source_table_path",
                "source_object_path",
            )
        },
        "parameters": configuration,
        "selection_identity_scope": "offline_surrogate",
        "source_identity": source_identity,
        "artifacts": artifact_records,
    }
    manifest_path = session_dir / SESSION_MANIFEST_FILENAME
    write_json_once(session_manifest, manifest_path)
    append_session_manifest(
        campaign,
        session_manifest,
        run_dir=run_dir,
    )
    return session_manifest


def run_figure_1_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_path: Path | None = None,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    regions: Sequence[str] = REGIONS,
    trajectory_types: Sequence[str] = TRAJECTORY_TYPES,
    position_role: str = "head",
    movement_parameters: Mapping[str, Any] = DEFAULT_MOVEMENT_PARAMETERS,
    tuning_parameter_presets: Sequence[Mapping[str, Any]] = (
        DEFAULT_TUNING_PARAMETER_PRESETS
    ),
    stability_tuning_param_name: str = DEFAULT_STABILITY_TUNING_PARAM_NAME,
) -> dict[str, Any]:
    """Run one session and remove only its newly created outputs on failure."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    session_dir = get_session_dir(
        run_dir,
        animal_name=animal_name,
        date=date,
    )
    session_preexisted = session_dir.exists()
    try:
        return _run_figure_1_session(
            run_id=run_id,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            nwb_path=nwb_path,
            nwb_root=nwb_root,
            scratch_root=scratch_root,
            regions=regions,
            trajectory_types=trajectory_types,
            position_role=position_role,
            movement_parameters=movement_parameters,
            tuning_parameter_presets=tuning_parameter_presets,
            stability_tuning_param_name=stability_tuning_param_name,
        )
    except BaseException:
        if not session_preexisted and session_dir.exists():
            shutil.rmtree(session_dir)
        raise


def _parser() -> argparse.ArgumentParser:
    """Build the explicit single-session offline runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--nwb-path", type=Path)
    parser.add_argument("--nwb-root", type=Path, default=DEFAULT_NWB_ROOT)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        dest="regions",
        help="Region to compute; repeat as needed (default: v1 and ca1).",
    )
    parser.add_argument("--position-role", default="head")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one explicit database-free session without diagnostic plotting."""
    args = _parser().parse_args(argv)
    manifest = run_figure_1_session(
        run_id=args.run_id,
        animal_name=args.animal_name,
        date=args.date,
        epoch=args.epoch,
        nwb_path=args.nwb_path,
        nwb_root=args.nwb_root,
        scratch_root=args.scratch_root,
        regions=tuple(args.regions or REGIONS),
        position_role=args.position_role,
    )
    print(
        f"Completed offline Figure 1 inputs for "
        f"{manifest['animal_name']} {manifest['date']} "
        f"({manifest['epochs'][0]})."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_NWB_ROOT",
    "DEFAULT_STABILITY_TUNING_PARAM_NAME",
    "DEFAULT_TUNING_PARAMETER_PRESETS",
    "TRIAL_SUBSETS",
    "main",
    "run_figure_1_session",
]
