"""Read manuscript-figure inputs from populated Spyglass result tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import table_specs
from v1ca1.spyglass.populate_figures import (
    DEFAULT_NWB_ROOT,
    LIGHT_TEST_EPOCH,
    _load_runtime,
    _movement_selection,
    _one_row,
    _region_row,
    _stability_selection,
    _standard_nwb_file_name,
    _tuning_selection,
    figure_dataset_specs,
)


class SpyglassFigureDatabase:
    """Resolve and cache the populated rows used by the manuscript figures."""

    def __init__(
        self,
        *,
        schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
        analysis_nwbfile_schema_name: str = (
            table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME
        ),
        nwb_root: Path = DEFAULT_NWB_ROOT,
    ) -> None:
        self.schema_name = str(schema_name)
        self.analysis_nwbfile_schema_name = str(
            analysis_nwbfile_schema_name
        )
        self.nwb_root = Path(nwb_root).resolve(strict=True)
        self.runtime = _load_runtime(
            schema_name=self.schema_name,
            analysis_nwbfile_schema_name=(
                self.analysis_nwbfile_schema_name
            ),
        )
        self.tables = self.runtime["tables"]
        self.specs = tuple(dict(spec) for spec in figure_dataset_specs())
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._selected_rows: dict[tuple[str, str], dict[str, Any]] = {}

    def spec(self, animal_name: str, date: str) -> dict[str, str]:
        """Return one configured manuscript session."""
        matches = [
            spec
            for spec in self.specs
            if spec["animal_name"] == str(animal_name)
            and spec["date"] == str(date)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one manuscript session for {animal_name} {date}; "
                f"found {len(matches)}."
            )
        return dict(matches[0])

    def raw_nwb_path(self, spec: Mapping[str, str]) -> Path:
        """Return the expected pre-ingestion augmented NWB source path."""
        return (
            self.nwb_root
            / f"{spec['animal_name']}{spec['date']}_augmented.nwb"
        ).resolve(strict=False)

    def nwb_file_name(self, spec: Mapping[str, str]) -> str:
        """Return the standard Spyglass filename registered for one source."""
        cache_key = ("nwb_file_name", spec["animal_name"], spec["date"])
        if cache_key not in self._cache:
            self._cache[cache_key] = _standard_nwb_file_name(
                self.runtime,
                self.raw_nwb_path(spec),
            )
        return str(self._cache[cache_key])

    def registered_nwb_path(self, spec: Mapping[str, str]) -> Path:
        """Fetch and return the augmented NWB copy registered by Spyglass."""
        nwb_file_name = self.nwb_file_name(spec)
        cache_key = ("registered_nwb_path", nwb_file_name)
        if cache_key not in self._cache:
            _one_row(
                self.runtime["Nwbfile"],
                {"nwb_file_name": nwb_file_name},
                label="registered Nwbfile row",
            )
            self._cache[cache_key] = Path(
                self.runtime["Nwbfile"].get_abs_path(nwb_file_name)
            ).resolve(strict=True)
        return Path(self._cache[cache_key])

    def group_name(self, spec: Mapping[str, str]) -> str:
        """Return the canonical imported all-unit sorting group name."""
        return f"{self.nwb_file_name(spec)}_imported_all_units"

    def source_reference(self, table_name: str, result_id: Any) -> Path:
        """Return a stable, display-only reference to one database result."""
        return Path(
            "spyglass"
        ) / self.schema_name / str(table_name) / str(result_id)

    def selected_rows(self) -> list[dict[str, Any]]:
        """Return the exact result identifiers read during this process."""
        return [
            dict(value)
            for _key, value in sorted(self._selected_rows.items())
        ]

    def _region(self, spec: Mapping[str, str], region: str) -> dict[str, Any]:
        cache_key = (
            "region",
            spec["animal_name"],
            spec["date"],
            str(region),
        )
        if cache_key not in self._cache:
            self._cache[cache_key] = _region_row(
                self.tables,
                nwb_file_name=self.nwb_file_name(spec),
                group_name=self.group_name(spec),
                region=str(region),
            )
        return dict(self._cache[cache_key])

    def _computed_bundle(
        self,
        *,
        selection_table: str,
        result_table: str,
        id_field: str,
        restriction: Mapping[str, Any],
        loader: str,
    ) -> tuple[Any, dict[str, Any]]:
        selection = _one_row(
            self.tables[selection_table],
            restriction,
            label=selection_table,
        )
        result_id = selection[id_field]
        cache_key = (result_table, str(result_id), loader)
        result_key = {id_field: result_id}
        result_row = _one_row(
            self.tables[result_table],
            result_key,
            label=result_table,
        )
        origin = result_row.get("artifact_origin")
        if origin is not None and str(origin) != "computed":
            raise ValueError(
                f"{result_table} {result_id} is not a computed result."
            )
        self._selected_rows[(result_table, str(result_id))] = {
            "table": result_table,
            "id_field": id_field,
            "result_id": str(result_id),
            "analysis_file_name": result_row.get("analysis_file_name"),
            "artifact_origin": origin,
        }
        if cache_key not in self._cache:
            self._cache[cache_key] = getattr(
                self.tables[result_table], loader
            )(result_key)
        return self._cache[cache_key], selection

    def movement(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
        region: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one movement firing-rate table and its selection."""
        selection = _movement_selection(
            self.tables,
            nwb_file_name=self.nwb_file_name(spec),
            epoch=str(epoch),
            region_row=self._region(spec, region),
        )
        result_id = selection["movement_firing_rate_id"]
        result_key = {"movement_firing_rate_id": result_id}
        cache_key = ("movement_firing_rate", str(result_id), "rates")
        _one_row(
            self.tables["movement_firing_rate"],
            result_key,
            label="movement_firing_rate",
        )
        self._selected_rows[("movement_firing_rate", str(result_id))] = {
            "table": "movement_firing_rate",
            "id_field": "movement_firing_rate_id",
            "result_id": str(result_id),
            "analysis_file_name": _one_row(
                self.tables["movement_firing_rate"],
                result_key,
                label="movement_firing_rate",
            ).get("analysis_file_name"),
            "artifact_origin": "computed",
        }
        if cache_key not in self._cache:
            self._cache[cache_key] = self.tables[
                "movement_firing_rate"
            ].load_firing_rates(result_key)
        return self._cache[cache_key], selection

    def tuning_curve(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
        region: str,
        trajectory_type: str,
        parameter_name: str,
        trial_subset: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one path-specific tuning curve and its selection."""
        selection = self.tuning_selection(
            spec,
            epoch=epoch,
            region=region,
            trajectory_type=trajectory_type,
            parameter_name=parameter_name,
            trial_subset=trial_subset,
        )
        result_id = selection["path_specific_place_tuning_curve_id"]
        result_key = {"path_specific_place_tuning_curve_id": result_id}
        cache_key = ("path_specific_place_tuning_curve", str(result_id))
        result_row = _one_row(
            self.tables["path_specific_place_tuning_curve"],
            result_key,
            label="path_specific_place_tuning_curve",
        )
        if str(result_row.get("artifact_origin")) != "computed":
            raise ValueError("Tuning-curve result is not computed.")
        self._selected_rows[
            ("path_specific_place_tuning_curve", str(result_id))
        ] = {
            "table": "path_specific_place_tuning_curve",
            "id_field": "path_specific_place_tuning_curve_id",
            "result_id": str(result_id),
            "analysis_file_name": result_row.get("analysis_file_name"),
            "artifact_origin": result_row.get("artifact_origin"),
        }
        if cache_key not in self._cache:
            self._cache[cache_key] = self.tables[
                "path_specific_place_tuning_curve"
            ].load_tuning_curve(result_key)
        return self._cache[cache_key], selection

    def tuning_selection(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
        region: str,
        trajectory_type: str,
        parameter_name: str,
        trial_subset: str,
    ) -> dict[str, Any]:
        """Resolve one path-specific tuning selection without loading output."""
        _movement, movement_selection = self.movement(
            spec,
            epoch=epoch,
            region=region,
        )
        selection = _tuning_selection(
            self.tables,
            movement_firing_rate_id=movement_selection[
                "movement_firing_rate_id"
            ],
            trajectory_type=str(trajectory_type),
            parameter_name=str(parameter_name),
            trial_subset=str(trial_subset),
        )
        return selection

    def stability(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
        region: str,
        trajectory_type: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one canonical odd/even path-stability result."""
        _movement, movement_selection = self.movement(
            spec,
            epoch=epoch,
            region=region,
        )
        selection = _stability_selection(
            self.tables,
            movement_firing_rate_id=movement_selection[
                "movement_firing_rate_id"
            ],
            trajectory_type=str(trajectory_type),
        )
        result_id = selection["path_specific_place_stability_id"]
        result_key = {"path_specific_place_stability_id": result_id}
        cache_key = ("path_specific_place_stability", str(result_id))
        result_row = _one_row(
            self.tables["path_specific_place_stability"],
            result_key,
            label="path_specific_place_stability",
        )
        if str(result_row.get("artifact_origin")) != "computed":
            raise ValueError("Stability result is not computed.")
        self._selected_rows[
            ("path_specific_place_stability", str(result_id))
        ] = {
            "table": "path_specific_place_stability",
            "id_field": "path_specific_place_stability_id",
            "result_id": str(result_id),
            "analysis_file_name": result_row.get("analysis_file_name"),
            "artifact_origin": result_row.get("artifact_origin"),
        }
        if cache_key not in self._cache:
            self._cache[cache_key] = self.tables[
                "path_specific_place_stability"
            ].load_stability(result_key)
        return self._cache[cache_key], selection

    def similarity(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
        region: str = "v1",
    ) -> tuple[Any, dict[str, Any]]:
        """Load the canonical all-path tuning-similarity result."""
        curve_ids = {}
        parameter_name = table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
            "tuning_curve_param_name"
        ]
        for trajectory_type in TRAJECTORY_TYPES:
            selection = self.tuning_selection(
                spec,
                epoch=epoch,
                region=region,
                trajectory_type=trajectory_type,
                parameter_name=parameter_name,
                trial_subset="all",
            )
            curve_ids[f"{trajectory_type}_tuning_curve_id"] = selection[
                "path_specific_place_tuning_curve_id"
            ]
        restriction = {
            **curve_ids,
            "tuning_similarity_param_name": (
                table_specs.ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS[
                    "tuning_similarity_param_name"
                ]
            ),
        }
        return self._computed_bundle(
            selection_table="path_specific_place_tuning_similarity_selection",
            result_table="path_specific_place_tuning_similarity",
            id_field="path_specific_place_tuning_similarity_id",
            restriction=restriction,
            loader="load_similarity",
        )

    def epoch_motor_behavior(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one epoch-level motor-behavior result."""
        return self._computed_bundle(
            selection_table="epoch_motor_behavior_selection",
            result_table="epoch_motor_behavior",
            id_field="epoch_motor_behavior_id",
            restriction={
                "nwb_file_name": self.nwb_file_name(spec),
                "epoch": str(epoch),
            },
            loader="load_epoch_motor_behavior_bundle",
        )

    def cv_pca(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load the manuscript V1 dark/light cvPCA result."""
        return self._computed_bundle(
            selection_table="cv_pca_selection",
            result_table="cv_pca",
            id_field="cv_pca_id",
            restriction={
                "nwb_file_name": self.nwb_file_name(spec),
                "light_epoch": str(spec["light_epoch"]),
                "dark_epoch": str(spec["dark_epoch"]),
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
            },
            loader="load_cv_pca_bundle",
        )

    def _movement_id(
        self, spec: Mapping[str, str], *, epoch: str, region: str = "v1"
    ) -> Any:
        _table, selection = self.movement(spec, epoch=epoch, region=region)
        return selection["movement_firing_rate_id"]

    def dpp_encoding(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load one dark-epoch directional-progression encoding result."""
        return self._computed_bundle(
            selection_table="dpp_encoding_selection",
            result_table="dpp_encoding",
            id_field="dpp_encoding_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "movement_firing_rate_id": self._movement_id(
                    spec, epoch=spec["dark_epoch"]
                ),
            },
            loader="load_dpp_encoding",
        )

    def path_progression_decoding(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one within-epoch cohort cross-path decoding result."""
        movement_id = self._movement_id(spec, epoch=epoch)
        return self._computed_bundle(
            selection_table="path_progression_decoding_selection",
            result_table="path_progression_decoding",
            id_field="path_progression_decoding_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "movement_firing_rate_id": movement_id,
                "cohort_movement_firing_rate_id": movement_id,
            },
            loader="load_decoding_bundle",
        )

    def path_specific_place_decoding(
        self,
        spec: Mapping[str, str],
        *,
        epoch: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one within-epoch path-specific place decoding result."""
        return self._computed_bundle(
            selection_table="path_specific_place_decoding_selection",
            result_table="path_specific_place_decoding",
            id_field="path_specific_place_decoding_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "movement_firing_rate_id": self._movement_id(
                    spec, epoch=epoch
                ),
            },
            loader="load_decoding_bundle",
        )

    def motor_encoding(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load one dark-epoch motor-encoding result."""
        return self._computed_bundle(
            selection_table="motor_encoding_selection",
            result_table="motor_encoding",
            id_field="motor_encoding_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "movement_firing_rate_id": self._movement_id(
                    spec, epoch=spec["dark_epoch"]
                ),
            },
            loader="load_motor_encoding_bundle",
        )

    def dark_light_glm(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load the coupled manuscript dark/light GLM result."""
        return self._computed_bundle(
            selection_table="dark_light_glm_selection",
            result_table="dark_light_glm",
            id_field="dark_light_glm_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "dark_movement_firing_rate_id": self._movement_id(
                    spec, epoch=spec["dark_epoch"]
                ),
                "light_movement_firing_rate_id": self._movement_id(
                    spec, epoch=spec["light_epoch"]
                ),
            },
            loader="load_dark_light_glm_bundle",
        )

    def swap_glm(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load the held-out swapped-light GLM result."""
        return self._computed_bundle(
            selection_table="swap_glm_selection",
            result_table="swap_glm",
            id_field="swap_glm_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "light_test_movement_firing_rate_id": self._movement_id(
                    spec, epoch=LIGHT_TEST_EPOCH
                ),
            },
            loader="load_swap_glm_bundle",
        )

    def swap_tuning(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load the empirical swapped-light tuning comparison result."""
        return self._computed_bundle(
            selection_table="swap_tuning_curve_comparison_selection",
            result_table="swap_tuning_curve_comparison",
            id_field="swap_tuning_curve_comparison_id",
            restriction={
                "region_sorted_spikes_group_id": self._region(spec, "v1")[
                    "region_sorted_spikes_group_id"
                ],
                "dark_movement_firing_rate_id": self._movement_id(
                    spec, epoch=spec["dark_epoch"]
                ),
                "light_train_movement_firing_rate_id": self._movement_id(
                    spec, epoch=spec["light_epoch"]
                ),
                "light_test_movement_firing_rate_id": self._movement_id(
                    spec, epoch=LIGHT_TEST_EPOCH
                ),
            },
            loader="load_swap_tuning_curve_comparison_bundle",
        )

    def ripple_modulation(
        self,
        spec: Mapping[str, str],
        *,
        region: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one light-epoch regional ripple-modulation result."""
        return self._computed_bundle(
            selection_table="ripple_modulation_selection",
            result_table="ripple_modulation",
            id_field="ripple_modulation_id",
            restriction={
                "nwb_file_name": self.nwb_file_name(spec),
                "epoch": str(spec["light_epoch"]),
                "region_sorted_spikes_group_id": self._region(spec, region)[
                    "region_sorted_spikes_group_id"
                ],
            },
            loader="load_artifacts",
        )

    def ripple_glm(
        self,
        spec: Mapping[str, str],
        *,
        source_predictor_mode: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Load one light-epoch CA1-to-V1 ripple GLM result."""
        matches = [
            parameters
            for parameters in table_specs.RIPPLE_GLM_PARAMETER_PRESETS
            if str(parameters["source_predictor_mode"])
            == str(source_predictor_mode)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"No unique ripple GLM preset for {source_predictor_mode!r}."
            )
        return self._computed_bundle(
            selection_table="ripple_glm_selection",
            result_table="ripple_glm",
            id_field="ripple_glm_id",
            restriction={
                "nwb_file_name": self.nwb_file_name(spec),
                "epoch": str(spec["light_epoch"]),
                "ripple_glm_param_name": matches[0][
                    "ripple_glm_param_name"
                ],
            },
            loader="load_ripple_glm_bundle",
        )

    def ripple_cross_region_xcorr(
        self, spec: Mapping[str, str]
    ) -> tuple[Any, dict[str, Any]]:
        """Load the fixed light-epoch cross-region ripple correlogram."""
        return self._computed_bundle(
            selection_table="ripple_cross_region_xcorr_selection",
            result_table="ripple_cross_region_xcorr",
            id_field="ripple_cross_region_xcorr_id",
            restriction={
                "nwb_file_name": self.nwb_file_name(spec),
                "epoch": str(spec["light_epoch"]),
            },
            loader="load_ripple_cross_region_xcorr_bundle",
        )

    def unit_maps(
        self, spec: Mapping[str, str]
    ) -> dict[str, dict[str, int]]:
        """Map persistent NWB unit identifiers to sorting IDs by region."""
        cache_key = ("unit_maps", spec["animal_name"], spec["date"])
        if cache_key in self._cache:
            return self._cache[cache_key]

        import pynwb

        with pynwb.NWBHDF5IO(
            str(self.registered_nwb_path(spec)),
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbfile = io.read()
            units = getattr(nwbfile, "units", None)
            if units is None:
                raise ValueError("Augmented NWB has no Units table.")
            table = units.to_dataframe()
        missing = sorted({"region", "sorting_unit_id"}.difference(table.columns))
        if missing:
            raise ValueError(f"NWB Units table is missing columns {missing!r}.")
        output: dict[str, dict[str, int]] = {}
        for region in ("ca1", "v1"):
            selected = table.loc[
                table["region"].astype(str).str.strip().str.casefold().eq(region)
            ]
            sorting_ids = np.asarray(
                selected["sorting_unit_id"],
                dtype=float,
            )
            if not np.all(np.isfinite(sorting_ids)) or not np.all(
                sorting_ids == sorting_ids.astype(int)
            ):
                raise ValueError(f"{region} sorting_unit_id values are invalid.")
            mapping = {
                str(unit_id): int(sorting_id)
                for unit_id, sorting_id in zip(
                    selected.index,
                    sorting_ids,
                    strict=True,
                )
            }
            if len(set(mapping.values())) != len(mapping):
                raise ValueError(f"{region} sorting_unit_id values are not unique.")
            output[region] = mapping
        self._cache[cache_key] = output
        return output

    def trajectory_inputs(
        self,
        spec: Mapping[str, str],
        *,
        epochs: Sequence[str],
    ) -> tuple[dict[str, dict[str, Any]], float]:
        """Load path intervals and the common W-track path length from NWB."""
        cache_key = (
            "trajectory_inputs",
            spec["animal_name"],
            spec["date"],
            tuple(str(epoch) for epoch in epochs),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        import pynwb

        from v1ca1.spyglass.nwb import (
            catalog_augmented_nwb,
            load_interval_set,
            load_wtrack_graph,
        )

        with pynwb.NWBHDF5IO(
            str(self.registered_nwb_path(spec)),
            mode="r",
            load_namespaces=True,
        ) as io:
            nwbfile = io.read()
            catalog = catalog_augmented_nwb(
                nwbfile,
                nwb_file_name=self.nwb_file_name(spec),
            )
            intervals: dict[str, dict[str, Any]] = {}
            for epoch in epochs:
                intervals[str(epoch)] = {}
                for trajectory_type in TRAJECTORY_TYPES:
                    rows = [
                        row
                        for row in catalog["trajectory_intervals"]
                        if str(row.get("epoch")) == str(epoch)
                        and str(row.get("trajectory_type"))
                        == trajectory_type
                    ]
                    if len(rows) != 1:
                        raise ValueError(
                            "Expected one trajectory interval for "
                            f"{epoch} {trajectory_type}; found {len(rows)}."
                        )
                    intervals[str(epoch)][trajectory_type] = load_interval_set(
                        nwbfile,
                        rows[0],
                    )
            graph_lengths = []
            for trajectory_type in TRAJECTORY_TYPES:
                rows = [
                    row
                    for row in catalog["wtrack_graph"]
                    if str(row.get("configuration_name")) == trajectory_type
                ]
                if len(rows) != 1:
                    raise ValueError(
                        f"Expected one W-track graph for {trajectory_type}."
                    )
                graph = load_wtrack_graph(nwbfile, rows[0])
                nodes = np.asarray(graph["node_positions_cm"], dtype=float)
                edges = np.asarray(graph["edge_order"], dtype=int)
                spacing = np.asarray(
                    graph["edge_spacing_cm"], dtype=float
                ).reshape(-1)
                graph_lengths.append(
                    float(
                        np.linalg.norm(
                            nodes[edges[:, 1]] - nodes[edges[:, 0]],
                            axis=1,
                        ).sum()
                        + spacing.sum()
                    )
                )
        path_length = graph_lengths[0]
        if not np.isfinite(path_length) or path_length <= 0.0 or any(
            not np.isclose(value, path_length, rtol=1e-10, atol=1e-12)
            for value in graph_lengths[1:]
        ):
            raise ValueError("Stored W-track path lengths do not agree.")
        result = (intervals, path_length)
        self._cache[cache_key] = result
        return result

    def example_payloads(
        self,
        specifications: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
        """Compute fixed example-cell panels from each registered source NWB."""
        import pynwb

        from v1ca1.spyglass.offline.figure_1_examples import (
            compute_nwb_example_payload,
        )

        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for specification in specifications:
            key = (
                str(specification["animal_name"]),
                str(specification["date"]),
            )
            grouped.setdefault(key, []).append(specification)

        output: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for (animal_name, date), rows in grouped.items():
            spec = self.spec(animal_name, date)
            with pynwb.NWBHDF5IO(
                str(self.registered_nwb_path(spec)),
                mode="r",
                load_namespaces=True,
            ) as io:
                nwbfile = io.read()
                for row in rows:
                    trajectories = tuple(
                        str(value)
                        for value in row.get(
                            "trajectory_types", TRAJECTORY_TYPES
                        )
                    )
                    cache_key = (
                        "example",
                        animal_name,
                        date,
                        str(row["epoch"]),
                        str(row["region"]),
                        str(row["sorting_unit_id"]),
                        trajectories,
                    )
                    if cache_key not in self._cache:
                        self._cache[cache_key] = compute_nwb_example_payload(
                            nwbfile,
                            nwb_file_name=self.nwb_file_name(spec),
                            animal_name=animal_name,
                            date=date,
                            epoch=str(row["epoch"]),
                            region=str(row["region"]),
                            sorting_unit_id=row["sorting_unit_id"],
                            trajectory_types=trajectories,
                        )
                    key = (
                        animal_name,
                        date,
                        str(row["epoch"]),
                        str(row["region"]),
                        str(row["sorting_unit_id"]),
                    )
                    output[key] = self._cache[cache_key]
        return output


__all__ = ["SpyglassFigureDatabase"]
