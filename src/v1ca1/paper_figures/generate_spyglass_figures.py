"""Generate manuscript figures directly from populated Spyglass tables."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures import _dark_light
from v1ca1.paper_figures import _figure_1_spyglass_full as figure_1_adapter
from v1ca1.paper_figures import figure_1 as figure_1
from v1ca1.paper_figures import figure_1_spyglass as figure_1d_adapter
from v1ca1.paper_figures import figure_2 as figure_2
from v1ca1.paper_figures import figure_2_spyglass as figure_2_adapter
from v1ca1.paper_figures import figure_3 as figure_3
from v1ca1.paper_figures import figure_3_spyglass as figure_3_adapter
from v1ca1.paper_figures import figure_4 as figure_4
from v1ca1.paper_figures import figure_4_spyglass as figure_4_adapter
from v1ca1.paper_figures import supplementary_figure_1 as supplementary_figure_1
from v1ca1.paper_figures import supplementary_figure_1_spyglass as supp_1_adapter
from v1ca1.paper_figures import supplementary_figure_2 as supplementary_figure_2
from v1ca1.paper_figures import supplementary_figure_3 as supplementary_figure_3
from v1ca1.paper_figures import supplementary_figure_4 as supplementary_figure_4
from v1ca1.paper_figures import supplementary_figure_4_spyglass as supp_4_adapter
from v1ca1.paper_figures import supplementary_figure_5 as supplementary_figure_5
from v1ca1.paper_figures import supplementary_figure_5_spyglass as supp_5_adapter
from v1ca1.paper_figures import supplementary_figure_6 as supplementary_figure_6
from v1ca1.paper_figures import supplementary_figure_6_spyglass as supp_6_adapter
from v1ca1.paper_figures import supplementary_figure_7 as supplementary_figure_7
from v1ca1.paper_figures import supplementary_figure_7_spyglass as supp_7_adapter
from v1ca1.paper_figures import supplementary_figure_8 as supplementary_figure_8
from v1ca1.paper_figures._spyglass_database import SpyglassFigureDatabase
from v1ca1.spyglass import ripple_glm, swap_glm, table_specs
from v1ca1.spyglass.offline.figure_1_decoding import (
    FIGURE_1_DECODING_COMPARISONS,
    _aligned_absolute_error_with_times,
    _interval_arrays,
)
from v1ca1.spyglass.offline.figure_1_full import FULL_FIGURE_EXAMPLES


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output" / "spyglass"
DEFAULT_OUTPUT_FORMAT = "svg"
DEFAULT_DPI = 300
FIGURE_NAMES = (
    "figure_1",
    "figure_2",
    "figure_3",
    "figure_4",
    "supplementary_figure_1",
    "supplementary_figure_2",
    "supplementary_figure_3",
    "supplementary_figure_4",
    "supplementary_figure_5",
    "supplementary_figure_6",
    "supplementary_figure_7",
    "supplementary_figure_8",
)


def _result_reference(
    database: SpyglassFigureDatabase,
    table_name: str,
    selection: Mapping[str, Any],
    id_field: str,
) -> Path:
    """Return one display-only source reference for a loaded result."""
    return database.source_reference(table_name, selection[id_field])


def _atomic_render(
    output_path: Path,
    *,
    replace: bool,
    render: Callable[[Path], Path],
) -> Path:
    """Render to a sibling file and publish it atomically."""
    destination = Path(output_path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        rendered = Path(render(temporary)).resolve(strict=True)
        if rendered != temporary:
            raise ValueError("Figure renderer returned an unexpected path.")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _as_example_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one source-NWB example payload to a paper-figure row."""
    metadata = payload["metadata"]
    return {
        "animal_name": str(metadata["animal_name"]),
        "date": str(metadata["date"]),
        "epoch": str(metadata["epoch"]),
        "region": str(metadata["region"]),
        "unit_id": metadata["sorting_unit_id"],
        "raster_positions": payload["raster_positions"],
        "firing_rates": payload["firing_rates"],
    }


def _figure_1_examples(
    database: SpyglassFigureDatabase,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the fixed Figure 1 visual examples from registered source NWBs."""
    specifications = [dict(value) for value in FULL_FIGURE_EXAMPLES]
    loaded = database.example_payloads(specifications)
    rows = {key: _as_example_row(value) for key, value in loaded.items()}
    panel_b_spec = specifications[0]
    panel_b_base = (
        str(panel_b_spec["animal_name"]),
        str(panel_b_spec["date"]),
        str(panel_b_spec["region"]),
        str(panel_b_spec["sorting_unit_id"]),
    )
    dark_epoch = database.spec(*panel_b_base[:2])["dark_epoch"]
    epoch_keys = {"02_r1": "02_r1", "06_r3": "06_r3", "dark": dark_epoch}
    panel_b = {
        "animal_name": panel_b_base[0],
        "date": panel_b_base[1],
        "region": panel_b_base[2],
        "unit_id": int(panel_b_base[3]),
        "epoch_order": tuple(epoch_keys),
        "epoch_labels": {
            key: figure_1.PANEL_B_VISUAL_EPOCH_LABELS[key]
            for key in epoch_keys
        },
        "epoch_examples": {
            key: rows[
                (
                    panel_b_base[0],
                    panel_b_base[1],
                    str(epoch),
                    panel_b_base[2],
                    panel_b_base[3],
                )
            ]
            for key, epoch in epoch_keys.items()
        },
        "trajectories": figure_1.PANEL_B_VISUAL_TRAJECTORIES,
    }
    panel_c = [
        rows[(str(animal), str(date), str(epoch), str(region), str(unit))]
        for animal, date, epoch, region, unit in figure_1.PANEL_E_EXAMPLES
    ]
    return panel_b, panel_c


def _figure_1_curve_set(
    database: SpyglassFigureDatabase,
    spec: Mapping[str, str],
    *,
    region: str,
) -> dict[str, Any]:
    """Load and filter one session's Figure 1 dark odd/even curves."""
    epoch = str(spec["dark_epoch"])
    movement, movement_selection = database.movement(
        spec,
        epoch=epoch,
        region=region,
    )
    movement = movement.copy()
    rates = pd.to_numeric(
        movement["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    movement_selected = set(
        movement.loc[
            np.isfinite(rates)
            & (
                rates
                >= figure_1.PANEL_D_MIN_MOVEMENT_FIRING_RATE_HZ
            ),
            "stable_unit_id",
        ].astype(str)
    )
    stable_selected: set[str] = set()
    for trajectory_type in figure_1.PANEL_D_TRAJECTORY_TYPES:
        stability, _selection = database.stability(
            spec,
            epoch=epoch,
            region=region,
            trajectory_type=trajectory_type,
        )
        correlations = pd.to_numeric(
            stability["stability_correlation"], errors="coerce"
        ).to_numpy(dtype=float)
        statuses = stability["stability_status"].astype(str).to_numpy()
        keep = (
            (statuses == "valid")
            & np.isfinite(correlations)
            & (
                correlations
                >= figure_1.PANEL_D_MIN_TUNING_STABILITY_CORRELATION
            )
        )
        stable_selected.update(
            stability.loc[keep, "stable_unit_id"].astype(str)
        )
    included = movement_selected.intersection(stable_selected)
    curves: dict[str, dict[str, Any]] = {"odd": {}, "even": {}}
    parameter_name = table_specs.FIGURE_1D_TUNING_CURVE_PARAMETERS[
        "tuning_curve_param_name"
    ]
    for trial_subset in curves:
        for trajectory_type in figure_1.PANEL_D_TRAJECTORY_TYPES:
            curve, _selection = database.tuning_curve(
                spec,
                epoch=epoch,
                region=region,
                trajectory_type=trajectory_type,
                parameter_name=parameter_name,
                trial_subset=trial_subset,
            )
            curves[trial_subset][trajectory_type] = (
                figure_1d_adapter._filter_and_label_curve(curve, included)
            )
    return {
        "animal_name": str(spec["animal_name"]),
        "date": str(spec["date"]),
        "region": str(region),
        "epoch": epoch,
        "odd_curves": curves["odd"],
        "even_curves": curves["even"],
        "included_units": np.asarray(sorted(included), dtype=str),
        "movement_firing_rate_path": str(
            _result_reference(
                database,
                "movement_firing_rate",
                movement_selection,
                "movement_firing_rate_id",
            )
        ),
    }


def _figure_1_panel_d_payload(
    database: SpyglassFigureDatabase,
) -> dict[str, Any]:
    """Build pooled Figure 1D heatmap inputs from database tuning rows."""
    panels_by_region = {}
    curve_sets_by_region = {}
    ordered_keys_by_region = {}
    ordered_peaks_by_region = {}
    regions = tuple(figure_1.DEFAULT_REGIONS)
    for region in regions:
        curve_sets = [
            _figure_1_curve_set(database, spec, region=region)
            for spec in database.specs
        ]
        panels, ordered_keys, ordered_peaks = (
            figure_1._build_pooled_panel_values_order_and_peaks(
                curve_sets,
                position_bin_count=figure_1.DEFAULT_POSITION_BIN_COUNT,
                trajectory_types=figure_1.PANEL_D_TRAJECTORY_TYPES,
                firing_rate_normalization=(
                    figure_1.PANEL_D_FIRING_RATE_NORMALIZATION
                ),
            )
        )
        curve_sets_by_region[region] = curve_sets
        panels_by_region[region] = panels
        ordered_keys_by_region[region] = ordered_keys
        ordered_peaks_by_region[region] = ordered_peaks
    return {
        "regions": regions,
        "datasets": figure_1_adapter.EXPECTED_DATASETS,
        "curve_sets_by_region": curve_sets_by_region,
        "panels_by_region": panels_by_region,
        "ordered_unit_keys_by_region": ordered_keys_by_region,
        "ordered_peak_positions_by_region": ordered_peaks_by_region,
    }


def _figure_1_motor_table(
    database: SpyglassFigureDatabase,
) -> pd.DataFrame:
    """Adapt MotorEncoding results to Figure 1's panel table."""
    rows = []
    for spec in database.specs:
        result, selection = database.motor_encoding(spec)
        nested = result["nested_cv"]
        values = np.asarray(
            nested["pooled_delta_bits_per_spike"]
            .sel(delta_metric=figure_1.MOTOR_DELTA_METRIC)
            .values,
            dtype=float,
        ).reshape(-1)
        coordinate = (
            "stable_unit_id" if "stable_unit_id" in nested.coords else "unit"
        )
        units = np.asarray(nested.coords[coordinate].values).reshape(-1)
        metadata = result["metadata"]
        source = _result_reference(
            database,
            "motor_encoding",
            selection,
            "motor_encoding_id",
        )
        rows.extend(
            {
                "animal_name": str(metadata["animal_name"]),
                "date": str(metadata["date"]),
                "epoch": str(metadata["epoch"]),
                "region": str(metadata["region"]),
                "unit": unit.item() if isinstance(unit, np.generic) else unit,
                "delta_log_likelihood_bits_per_spike": float(value),
                "source_path": str(source),
            }
            for unit, value in zip(units, values, strict=True)
            if np.isfinite(value)
        )
    return pd.DataFrame.from_records(
        rows,
        columns=list(figure_1.MOTOR_DELTA_TABLE_COLUMNS),
    )


def _figure_1_encoding_table(
    database: SpyglassFigureDatabase,
) -> pd.DataFrame:
    """Adapt DPPEncoding results to Figure 1's comparison table."""
    rows = []
    labels = {
        comparison: label
        for comparison, label, _column in figure_1.ENCODING_DPP_COMPARISONS
    }
    value_columns = {
        "dpp_vs_absolute_place": "dpp_vs_absolute_place_bits_per_spike",
        "dpp_vs_absolute_task_progression": (
            "dpp_vs_distance_to_reward_bits_per_spike"
        ),
    }
    for spec in database.specs:
        table, selection = database.dpp_encoding(spec)
        source = _result_reference(
            database,
            "dpp_encoding",
            selection,
            "dpp_encoding_id",
        )
        for comparison, value_column in value_columns.items():
            for row in table.to_dict("records"):
                value = float(row[value_column])
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "animal_name": str(row["animal_name"]),
                        "date": str(row["date"]),
                        "epoch": str(row["epoch"]),
                        "region": str(row["region"]),
                        "unit": str(row["stable_unit_id"]),
                        "n_spikes": int(row["heldout_spike_count"]),
                        "comparison": comparison,
                        "comparison_label": labels[comparison],
                        "delta_log_likelihood_bits_per_spike": value,
                        "source_path": str(source),
                    }
                )
    return pd.DataFrame.from_records(
        rows,
        columns=list(figure_1.ENCODING_DELTA_TABLE_COLUMNS),
    )


def _figure_1_decoding_tables(
    database: SpyglassFigureDatabase,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt cross-path decoder bundles to sample- and lap-level tables."""
    sample_tables = []
    trial_rows = []
    for spec in database.specs:
        bundle, selection = database.path_progression_decoding(
            spec,
            epoch=spec["dark_epoch"],
        )
        intervals, _path_length = database.trajectory_inputs(
            spec,
            epochs=(spec["dark_epoch"],),
        )
        metadata = bundle["metadata"]
        source = _result_reference(
            database,
            "path_progression_decoding",
            selection,
            "path_progression_decoding_id",
        )
        for comparison, label, family, pairs in FIGURE_1_DECODING_COMPARISONS:
            for encoding_trajectory, decoding_trajectory in pairs:
                key = (family, encoding_trajectory, decoding_trajectory)
                output = bundle["cross_path_outputs"][key]
                timestamps, errors = _aligned_absolute_error_with_times(
                    output["true"], output["decoded"]
                )
                common = {
                    "animal_name": str(metadata["animal_name"]),
                    "date": str(metadata["date"]),
                    "epoch": str(metadata["epoch"]),
                    "region": "v1",
                    "comparison": comparison,
                    "comparison_label": label,
                    "transfer_family": family,
                    "encoding_trajectory": encoding_trajectory,
                    "decoding_trajectory": decoding_trajectory,
                    "true_path": str(source / f"{family}-{encoding_trajectory}-{decoding_trajectory}-true"),
                    "decoded_path": str(source / f"{family}-{encoding_trajectory}-{decoding_trajectory}-decoded"),
                }
                if errors.size:
                    sample_tables.append(
                        pd.DataFrame({**common, "absolute_error": errors}).loc[
                            :, list(figure_1.DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS)
                        ]
                    )
                starts, ends = _interval_arrays(
                    intervals[str(spec["dark_epoch"])][decoding_trajectory]
                )
                for trial_index, (start, end) in enumerate(
                    zip(starts, ends, strict=True)
                ):
                    in_trial = (timestamps >= start) & (timestamps < end)
                    values = errors[in_trial]
                    values = values[np.isfinite(values)]
                    if values.size:
                        trial_rows.append(
                            {
                                **common,
                                "trial_index": int(trial_index),
                                "trial_start": float(start),
                                "trial_end": float(end),
                                "trial_median_absolute_error": float(
                                    np.median(values)
                                ),
                                "n_samples": int(values.size),
                            }
                        )
    absolute = pd.concat(sample_tables, ignore_index=True).loc[
        :, list(figure_1.DECODING_ABSOLUTE_ERROR_TABLE_COLUMNS)
    ]
    trial = pd.DataFrame.from_records(
        trial_rows,
        columns=list(figure_1.DECODING_TRIAL_ERROR_TABLE_COLUMNS),
    )
    return absolute, trial


def build_figure_1_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build every Figure 1 input directly from populated results."""
    panel_b, panel_c = _figure_1_examples(database)
    decoding_absolute, decoding_trial = _figure_1_decoding_tables(database)
    return {
        "run_dir": Path(run_dir),
        "mode": "full",
        "datasets": figure_1_adapter.EXPECTED_DATASETS,
        "regions": tuple(figure_1.DEFAULT_REGIONS),
        "panel_b_example": panel_b,
        "panel_c_examples": panel_c,
        "panel_d_payload": _figure_1_panel_d_payload(database),
        "motor_delta_table": _figure_1_motor_table(database),
        "encoding_delta_table": _figure_1_encoding_table(database),
        "decoding_absolute_error_table": decoding_absolute,
        "decoding_trial_error_table": decoding_trial,
    }


def _figure_2_sessions(
    database: SpyglassFigureDatabase,
) -> list[dict[str, Any]]:
    """Return minimal session records in manuscript order."""
    return [
        {
            "animal_name": spec["animal_name"],
            "date": spec["date"],
            "epochs": {
                "dark": spec["dark_epoch"],
                "AB": spec["light_epoch"],
                "BA": figure_2_adapter.HELDOUT_LIGHT_EPOCH,
            },
            "_database_spec": spec,
        }
        for spec in database.specs
    ]


@contextmanager
def _database_figure_2_builders(database: SpyglassFigureDatabase):
    """Bind Figure 2 adapter computations to in-memory database results."""
    originals: list[tuple[str, Any]] = []

    def replace(name: str, value: Any) -> None:
        originals.append((name, getattr(figure_2_adapter, name)))
        setattr(figure_2_adapter, name, value)

    def spec_for(session: Mapping[str, Any]) -> Mapping[str, str]:
        return session["_database_spec"]

    def unit_map(session: Mapping[str, Any]) -> dict[str, int]:
        return database.unit_maps(spec_for(session))["v1"]

    def movement_table(
        session: Mapping[str, Any],
        *,
        epoch: str,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        table, _selection = database.movement(
            spec_for(session), epoch=epoch, region="v1"
        )
        return figure_2_adapter._identity_columns(
            table,
            label="MovementFiringRate database result",
        )

    def stability_tables(
        session: Mapping[str, Any],
        *,
        epoch: str,
        **_kwargs: Any,
    ) -> list[pd.DataFrame]:
        return [
            figure_2_adapter._identity_columns(
                database.stability(
                    spec_for(session),
                    epoch=epoch,
                    region="v1",
                    trajectory_type=trajectory_type,
                )[0],
                label="Stability database result",
            )
            for trajectory_type in TRAJECTORY_TYPES
        ]

    def curve_records(
        session: Mapping[str, Any],
        *,
        epoch: str,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        return {"spec": spec_for(session), "epoch": str(epoch)}, Path("spyglass")

    def path_curve(
        records: Mapping[str, Any],
        *,
        epoch: str,
        trajectory_type: str,
        trial_subset: str,
        **_kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], Path]:
        curve, selection = database.tuning_curve(
            records["spec"],
            epoch=epoch,
            region="v1",
            trajectory_type=trajectory_type,
            parameter_name=table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
                "tuning_curve_param_name"
            ],
            trial_subset=trial_subset,
        )
        stable_ids = np.asarray(curve.coords["stable_unit_id"].values).astype(str)
        values = np.asarray(curve.transpose("unit", ...).values, dtype=float)
        source = _result_reference(
            database,
            "path_specific_place_tuning_curve",
            selection,
            "path_specific_place_tuning_curve_id",
        )
        return {
            stable_id: np.asarray(values[index], dtype=float)
            for index, stable_id in enumerate(stable_ids)
        }, source

    def similarity_tables(
        session: Mapping[str, Any],
        **_kwargs: Any,
    ) -> dict[str, tuple[pd.DataFrame, Path]]:
        spec = spec_for(session)
        output = {}
        for epoch in (spec["light_epoch"], spec["dark_epoch"]):
            table, selection = database.similarity(spec, epoch=epoch)
            output[str(epoch)] = (
                figure_2_adapter._identity_columns(
                    table,
                    label="Similarity database result",
                ),
                _result_reference(
                    database,
                    "path_specific_place_tuning_similarity",
                    selection,
                    "path_specific_place_tuning_similarity_id",
                ),
            )
        return output

    replace("_load_nwb_sorting_unit_map", unit_map)
    replace("_load_movement_table", movement_table)
    replace("_load_stability_tables", stability_tables)
    replace("_epoch_tuning_curve_records", curve_records)
    replace("_load_path_curve", path_curve)
    replace("_load_similarity_tables", similarity_tables)
    try:
        yield
    finally:
        for name, value in reversed(originals):
            setattr(figure_2_adapter, name, value)


def _figure_2_examples(
    database: SpyglassFigureDatabase,
) -> list[dict[str, Any]]:
    """Build Figure 2's dark/light example-cell payloads from source NWBs."""
    specifications = []
    for animal_name, date, region, unit_id, trajectories in (
        figure_2.FIGURE_2_PANEL_A_EXAMPLES
    ):
        spec = database.spec(animal_name, date)
        for epoch in (spec["dark_epoch"], spec["light_epoch"]):
            specifications.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "epoch": epoch,
                    "region": region,
                    "sorting_unit_id": unit_id,
                    "trajectory_types": trajectories,
                }
            )
    loaded = database.example_payloads(specifications)
    output = []
    for animal_name, date, region, unit_id, trajectories in (
        figure_2.FIGURE_2_PANEL_A_EXAMPLES
    ):
        spec = database.spec(animal_name, date)
        epoch_rates = {}
        for epoch_type, epoch in (
            ("dark", spec["dark_epoch"]),
            ("light", spec["light_epoch"]),
        ):
            row = _as_example_row(
                loaded[
                    (
                        str(animal_name),
                        str(date),
                        str(epoch),
                        str(region),
                        str(unit_id),
                    )
                ]
            )
            epoch_rates[epoch_type] = row
        output.append(
            {
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "unit_id": int(unit_id),
                "trajectories": tuple(trajectories),
                "epoch_rates": epoch_rates,
            }
        )
    return output


def build_figure_2_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build every current Figure 2 input from populated results."""
    sessions = _figure_2_sessions(database)
    with _database_figure_2_builders(database):
        shift = figure_2_adapter._build_panel_b_shift_profile_table(
            Path(run_dir), sessions, scratch_root=Path(run_dir)
        )
        overlap = figure_2_adapter._build_panel_c_overlap_table(
            Path(run_dir), sessions, scratch_root=Path(run_dir)
        )
    return {
        "run_dir": Path(run_dir),
        "sessions": sessions,
        "datasets": figure_2_adapter.EXPECTED_DATASETS,
        "regions": (figure_2_adapter.REGION,),
        "panel_a_examples": _figure_2_examples(database),
        "panel_b_shift_profile_table": shift,
        "panel_c_overlap_table": overlap,
    }


def _figure_3_loaded_results(
    database: SpyglassFigureDatabase,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map and quality-filter each database SwapGLM result."""
    output = {}
    for spec in database.specs:
        result, selection = database.swap_glm(spec)
        selected = figure_3_adapter._identity_columns(
            result["selected_units"], label="SwapGLM selected_units"
        )
        dataset = result["dataset"]
        group_ids = np.asarray(dataset.coords["unit"].values).astype(str)
        if not np.array_equal(
            group_ids,
            selected["group_unit_id"].to_numpy(dtype=str),
        ):
            raise ValueError("SwapGLM result and unit audit disagree.")
        mapping = database.unit_maps(spec)["v1"]
        nwb_ids = selected["unit_id"].astype(str).to_numpy()
        missing = sorted(set(nwb_ids).difference(mapping))
        if missing:
            raise ValueError(f"SwapGLM units are absent from NWB: {missing!r}.")
        sorting_ids = np.asarray([mapping[value] for value in nwb_ids], dtype=int)
        mapped = dataset.assign_coords(
            unit=sorting_ids,
            nwb_unit_id=("unit", nwb_ids),
        )
        source = _result_reference(
            database,
            "swap_glm",
            selection,
            "swap_glm_id",
        )
        figure_3_adapter._require_swap_models(mapped, source_path=source)
        movement = database.movement(
            spec, epoch=spec["dark_epoch"], region="v1"
        )[0]
        stability = [
            database.stability(
                spec,
                epoch=spec["dark_epoch"],
                region="v1",
                trajectory_type=trajectory_type,
            )[0]
            for trajectory_type in TRAJECTORY_TYPES
        ]
        eligible = figure_2_adapter._eligible_units(movement, stability)
        output[(str(spec["animal_name"]), str(spec["date"]))] = {
            "dataset": mapped,
            "selected_units": selected,
            "eligible_unit_mask": selected["stable_unit_id"].isin(eligible).to_numpy(),
            "metadata": result["metadata"],
            "source_path": source,
        }
    return output


def build_figure_3_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build current Figure 3 from the populated SwapGLM rows."""
    loaded = _figure_3_loaded_results(database)
    return {
        "run_dir": Path(run_dir),
        "datasets": figure_3_adapter.EXPECTED_DATASETS,
        "regions": (figure_3_adapter.REGION,),
        "swap_delta": figure_3_adapter._delta_table(
            loaded, model_name=figure_3_adapter.MULTIPLICATIVE_MODEL
        ),
        "swap_additive_delta": figure_3_adapter._delta_table(
            loaded, model_name=figure_3_adapter.ADDITIVE_MODEL
        ),
        "swap_examples": figure_3_adapter._swap_examples(loaded),
        "_loaded_swap_results": loaded,
    }


def _figure_4_glm_results(
    database: SpyglassFigureDatabase,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load and map both ripple-GLM source models for all sessions."""
    output = {}
    for spec in database.specs:
        animal_name = str(spec["animal_name"])
        date = str(spec["date"])
        for mode in figure_4_adapter.GLM_SOURCE_MODES:
            result, selection = database.ripple_glm(
                spec,
                source_predictor_mode=mode,
            )
            figure_4_adapter._validate_glm_result(
                result,
                animal_name=animal_name,
                date=date,
                mode=mode,
            )
            output[(animal_name, date, mode)] = {
                "result": result,
                "dataset": figure_4_adapter._map_glm_dataset_units(
                    result,
                    sorting_unit_by_nwb_id=database.unit_maps(spec)["v1"],
                ),
                "manifest_path": _result_reference(
                    database,
                    "ripple_glm",
                    selection,
                    "ripple_glm_id",
                ),
            }
    return output


def _figure_4_sessions(
    database: SpyglassFigureDatabase,
) -> list[dict[str, Any]]:
    """Return minimal light-session records for ripple adapter helpers."""
    return [
        {
            "animal_name": str(spec["animal_name"]),
            "date": str(spec["date"]),
            "epochs": {"light": str(spec["light_epoch"])},
            "regions": figure_4_adapter.REGIONS,
            "_database_spec": spec,
        }
        for spec in database.specs
    ]


def _figure_4_modulation_tables(
    database: SpyglassFigureDatabase,
) -> list[dict[str, Any]]:
    """Pool database ripple-modulation tables across the four sessions."""
    summary_tables = []
    firing_rate_tables = []
    datasets = []
    for spec in database.specs:
        animal_name = str(spec["animal_name"])
        date = str(spec["date"])
        epoch = str(spec["light_epoch"])
        datasets.append((animal_name, date, epoch))
        for region in figure_4_adapter.REGIONS:
            result, selection = database.ripple_modulation(spec, region=region)
            mapping = database.unit_maps(spec)[region]
            source = _result_reference(
                database,
                "ripple_modulation",
                selection,
                "ripple_modulation_id",
            )
            summary_tables.append(
                figure_4_adapter._map_unit_table(
                    result["summary"],
                    region=region,
                    sorting_unit_by_nwb_id=mapping,
                    label="RippleModulation summary",
                ).assign(source_path=str(source))
            )
            firing_rate_tables.append(
                figure_4_adapter._map_unit_table(
                    result["peri_ripple_firing_rate"],
                    region=region,
                    sorting_unit_by_nwb_id=mapping,
                    label="RippleModulation peri-ripple table",
                ).assign(source_path=str(source))
            )
    return [
        {
            "epoch_type": "light",
            "label": figure_4_adapter.legacy.HEATMAP_EPOCH_LABELS["light"],
            "epoch": figure_4_adapter.LIGHT_EPOCH,
            "epochs": tuple(
                figure_4_adapter.LIGHT_EPOCH for _spec in database.specs
            ),
            "datasets": tuple(datasets),
            "n_datasets": len(datasets),
            "firing_rate_table": pd.concat(
                firing_rate_tables, ignore_index=True, sort=False
            ),
            "summary_table": pd.concat(
                summary_tables, ignore_index=True, sort=False
            ),
        }
    ]


def _figure_4_dark_movement(
    database: SpyglassFigureDatabase,
    spec: Mapping[str, str],
) -> pd.DataFrame:
    """Return one figure-facing dark movement table."""
    table = database.movement(
        spec,
        epoch=spec["dark_epoch"],
        region="v1",
    )[0]
    mapped = figure_4_adapter._map_unit_table(
        table,
        region="v1",
        sorting_unit_by_nwb_id=database.unit_maps(spec)["v1"],
        label="Dark MovementFiringRate",
    )
    return mapped.rename(columns={"unit_id": "unit"})[
        ["unit", "movement_firing_rate_hz"]
    ].rename(columns={"movement_firing_rate_hz": "dark_firing_rate_hz"})


def _figure_4_dark_similarity(
    database: SpyglassFigureDatabase,
    spec: Mapping[str, str],
) -> pd.DataFrame:
    """Return one figure-facing dark same-turn similarity table."""
    table, selection = database.similarity(spec, epoch=spec["dark_epoch"])
    mapped = figure_4_adapter._map_unit_table(
        table,
        region="v1",
        sorting_unit_by_nwb_id=database.unit_maps(spec)["v1"],
        label="Dark tuning similarity",
    )
    selected = mapped.loc[
        mapped["epoch"].astype(str).eq(str(spec["dark_epoch"]))
        & mapped["region"].astype(str).eq("v1")
        & mapped["similarity_metric"].astype(str).eq(
            figure_4_adapter.legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
        )
        & mapped["comparison_family"].astype(str).eq("same_turn")
        & mapped["comparison_label"].astype(str).isin(
            ("left_turn", "right_turn")
        )
        & mapped["similarity_status"].astype(str).eq("valid")
    ].copy()
    selected["similarity"] = pd.to_numeric(
        selected["similarity"], errors="coerce"
    )
    selected = selected.loc[
        np.isfinite(selected["similarity"].to_numpy(dtype=float))
    ]
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "unit",
                "same_turn_tuning_similarity",
                "tuning_source_path",
            ]
        )
    pooled = (
        selected.groupby("unit_id", sort=True, as_index=False)["similarity"]
        .max()
        .rename(
            columns={
                "unit_id": "unit",
                "similarity": "same_turn_tuning_similarity",
            }
        )
    )
    return pooled.assign(
        tuning_source_path=str(
            _result_reference(
                database,
                "path_specific_place_tuning_similarity",
                selection,
                "path_specific_place_tuning_similarity_id",
            )
        )
    )


def _figure_4_behavior_payload(
    database: SpyglassFigureDatabase,
    glm_results: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Join light ripple GLMs to database dark activity and similarity."""
    devexp_rows = []
    activity_rows = []
    dppi_rows = []
    for spec in database.specs:
        animal_name = str(spec["animal_name"])
        date = str(spec["date"])
        dark_epoch = str(spec["dark_epoch"])
        movement = _figure_4_dark_movement(database, spec)
        similarity = _figure_4_dark_similarity(database, spec)
        glm_table = figure_4_adapter._glm_summary_table(
            glm_results[(animal_name, date, "unit_vector")],
            animal_name=animal_name,
            date=date,
            mode="unit_vector",
        )
        devexp_rows.append(
            figure_4_adapter.legacy.build_glm_dark_activity_devexp_table(
                glm_table,
                movement,
                similarity,
                animal_name=animal_name,
                date=date,
                glm_epoch=figure_4_adapter.LIGHT_EPOCH,
                epoch_type="light",
                dark_epoch=dark_epoch,
                dark_activity_threshold_hz=(
                    figure_4_adapter.legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ
                ),
            )
        )
        activity_rows.append(
            figure_4_adapter.legacy.build_dark_activity_reference_table(
                movement,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_epoch,
                dark_activity_threshold_hz=(
                    figure_4_adapter.legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ
                ),
            )
        )
        dppi_rows.append(
            figure_4_adapter.legacy.build_dark_active_dppi_reference_table(
                movement,
                similarity,
                animal_name=animal_name,
                date=date,
                dark_epoch=dark_epoch,
                dark_activity_threshold_hz=(
                    figure_4_adapter.legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ
                ),
            )
        )
    return {
        "devexp_table": pd.concat(devexp_rows, ignore_index=True, sort=False),
        "dark_activity_reference_table": pd.concat(
            activity_rows, ignore_index=True, sort=False
        ),
        "dark_active_dppi_reference_table": pd.concat(
            dppi_rows, ignore_index=True, sort=False
        ),
        "missing_artifacts": [],
        "region": "v1",
        "dark_activity_threshold_hz": (
            figure_4_adapter.legacy.PANEL_D_DARK_ACTIVITY_THRESHOLD_HZ
        ),
        "tuning_comparison_label": (
            figure_4_adapter.legacy.DEFAULT_PANEL_D_TUNING_COMPARISON_LABEL
        ),
        "tuning_similarity_metric": (
            figure_4_adapter.legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
        ),
    }


def _figure_4_xcorr_payload(
    database: SpyglassFigureDatabase,
) -> dict[str, Any]:
    """Map the populated L15 cross-region ripple correlogram for display."""
    animal_name, date, epoch = figure_4_adapter.legacy.DEFAULT_XCORR_DATASET
    spec = database.spec(animal_name, date)
    result, selection = database.ripple_cross_region_xcorr(spec)
    maps = database.unit_maps(spec)
    summary = result["summary"].copy()
    for region in figure_4_adapter.REGIONS:
        field = f"{region}_unit_id"
        nwb_ids = summary[field].astype(str)
        missing = sorted(set(nwb_ids).difference(maps[region]))
        if missing:
            raise ValueError(f"XCorr {region} units are absent from the NWB.")
        summary[f"{region}_nwb_unit_id"] = nwb_ids
        summary[field] = np.asarray(
            [maps[region][value] for value in nwb_ids], dtype=int
        )
    dataset = result["dataset"]
    coordinate_updates = {}
    for region in figure_4_adapter.REGIONS:
        dimension = f"{region}_unit"
        identity_coordinate = f"{region}_source_unit_id"
        nwb_ids = np.asarray(
            dataset.coords[identity_coordinate].values
        ).astype(str)
        coordinate_updates[dimension] = np.asarray(
            [maps[region][value] for value in nwb_ids], dtype=int
        )
        coordinate_updates[f"{region}_nwb_unit_id"] = (dimension, nwb_ids)
    dataset = dataset.assign_coords(**coordinate_updates)
    valid = summary.loc[
        summary["status"].astype(str).eq(
            figure_4_adapter.legacy.PAIR_STATUS_VALID
        )
    ].copy()
    ca1_order = figure_4_adapter.legacy.order_ca1_units_by_best_partner(valid)[
        : figure_4_adapter.legacy.DEFAULT_XCORR_TOP_CA1_UNITS
    ]
    top_ca1 = ca1_order[0]
    top_rows = valid.loc[valid["ca1_unit_id"] == top_ca1].sort_values(
        by=["peak_norm_xcorr", "peak_lag_s"],
        ascending=[False, True],
        kind="stable",
    )
    v1_order = top_rows["v1_unit_id"].to_numpy()
    ca1_order = figure_4_adapter.legacy._filter_existing_unit_ids(
        ca1_order, np.asarray(dataset.coords["ca1_unit"].values)
    )
    v1_order = figure_4_adapter.legacy._filter_existing_unit_ids(
        v1_order, np.asarray(dataset.coords["v1_unit"].values)
    )
    source = _result_reference(
        database,
        "ripple_cross_region_xcorr",
        selection,
        "ripple_cross_region_xcorr_id",
    )
    return {
        "animal_name": animal_name,
        "date": date,
        "epoch": epoch,
        "state": figure_4_adapter.legacy.DEFAULT_XCORR_STATE,
        "summary_path": source / "summary",
        "dataset_path": source / "dataset",
        "summary_table": valid,
        "ca1_unit_ids": ca1_order,
        "v1_unit_ids": v1_order,
        "v1_order_reference_ca1_unit": top_ca1,
        "lag_s": np.asarray(dataset["lag_s"].values, dtype=float),
        "xcorr": np.asarray(
            dataset["xcorr"]
            .sel(ca1_unit=ca1_order, v1_unit=v1_order)
            .values,
            dtype=float,
        ),
        "display_vmax": figure_4_adapter.legacy.DEFAULT_XCORR_DISPLAY_VMAX,
        "attrs": dict(dataset.attrs),
    }


def _figure_4_schematic_payload(
    database: SpyglassFigureDatabase,
) -> dict[str, Any]:
    """Reconstruct the fixed L15 schematic from NWB plus modulation output."""
    import pynwb

    from v1ca1.spyglass.nwb import catalog_augmented_nwb, load_interval_set
    from v1ca1.spyglass.offline import figure_3 as source_builder
    from v1ca1.spyglass.offline.figure_3_schematic_supplement import (
        SCHEMATIC_SELECTOR_POLICY,
        rank_ca1_schematic_units,
    )
    from v1ca1.spyglass.offline.sources import load_nwb_region_spikes

    animal_name, date, epoch = (
        figure_4_adapter.legacy.DEFAULT_PANEL_B_SCHEMATIC_DATASET
    )
    spec = database.spec(animal_name, date)
    modulation = database.ripple_modulation(spec, region="ca1")[0]
    with pynwb.NWBHDF5IO(
        str(database.registered_nwb_path(spec)),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        catalog = catalog_augmented_nwb(
            nwbfile, nwb_file_name=database.nwb_file_name(spec)
        )
        epoch_rows = [
            row
            for row in catalog["epoch_intervals"]
            if str(row.get("epoch")) == epoch
        ]
        ripple_rows = [
            row
            for row in catalog["ripples"]
            if str(row.get("epoch")) == epoch
        ]
        if len(epoch_rows) != 1 or len(ripple_rows) != 1:
            raise ValueError("L15 schematic NWB catalog is incomplete.")
        ripple_table = source_builder._interval_frame(
            load_interval_set(nwbfile, ripple_rows[0]), epoch=epoch
        )
        bounds = (
            float(epoch_rows[0]["start_time"]),
            float(epoch_rows[0]["stop_time"]),
        )
        loaded = {
            region: load_nwb_region_spikes(
                nwbfile,
                nwb_file_name=database.nwb_file_name(spec),
                region=region,
                time_support=bounds,
            )
            for region in figure_4_adapter.REGIONS
        }
        ranked = rank_ca1_schematic_units(
            modulation["summary"],
            ca1_spikes=source_builder._spike_times_by_stable_id(
                loaded["ca1"]
            ),
            sorting_unit_ids=database.unit_maps(spec)["ca1"],
        )
        payload = source_builder._build_schematic_payload(
            nwbfile,
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            nwb_file_name=database.nwb_file_name(spec),
            ripple_table=ripple_table,
            ca1=loaded["ca1"],
            v1=loaded["v1"],
            ca1_modulation=modulation,
            selector_kwargs={"ranked_ca1_unit_ids": ranked},
            selector_policy=SCHEMATIC_SELECTOR_POLICY,
        )
    return figure_4_adapter._map_schematic_unit_ids(
        payload, unit_maps=database.unit_maps(spec)
    )


def build_figure_4_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build every current Figure 4 input from populated results."""
    sessions = _figure_4_sessions(database)
    glm_results = _figure_4_glm_results(database)
    return {
        "run_dir": Path(run_dir),
        "sessions": sessions,
        "datasets": figure_4_adapter.EXPECTED_DATASETS,
        "regions": figure_4_adapter.REGIONS,
        "heatmap_epoch_tables": _figure_4_modulation_tables(database),
        "glm_epoch_tables": figure_4_adapter._build_glm_epoch_tables(
            sessions, glm_results
        ),
        "schematic_payload": _figure_4_schematic_payload(database),
        "prediction_examples": figure_4_adapter._build_prediction_examples(
            glm_results
        ),
        "behavior_payload": _figure_4_behavior_payload(database, glm_results),
        "source_comparison_payload": (
            figure_4_adapter._build_source_comparison_payload(
                sessions, glm_results
            )
        ),
        "xcorr_payload": _figure_4_xcorr_payload(database),
        "_glm_results": glm_results,
    }


def build_supplementary_figure_1_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build Supplementary Figure 1 movement and stability tables."""
    movement_tables = []
    stability_tables = []
    for spec in database.specs:
        movement = database.movement(
            spec,
            epoch=spec["dark_epoch"],
            region=supp_1_adapter.MOVEMENT_REGION,
        )[0].copy()
        movement["unit"] = movement["unit_id"]
        movement["dark_epoch"] = str(spec["dark_epoch"])
        movement["dark_firing_rate_hz"] = movement[
            "movement_firing_rate_hz"
        ]
        movement_tables.append(movement)
        for region in supp_1_adapter.STABILITY_REGIONS:
            for trajectory_type in supplementary_figure_1.PANEL_D_TRAJECTORY_TYPES:
                stability_tables.append(
                    database.stability(
                        spec,
                        epoch=spec["dark_epoch"],
                        region=region,
                        trajectory_type=trajectory_type,
                    )[0]
                )
    return {
        "run_dir": Path(run_dir),
        "datasets": supp_1_adapter.EXPECTED_DATASETS,
        "regions": supp_1_adapter.STABILITY_REGIONS,
        "dark_movement_firing_rate_table": pd.concat(
            movement_tables, ignore_index=True, sort=False
        ),
        "dark_stability_table": pd.concat(
            stability_tables, ignore_index=True, sort=False
        ),
    }


def _figure_2_decoding_tables(
    database: SpyglassFigureDatabase,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Figure 2 decoding summaries from database decoder bundles."""
    pooled_place: dict[str, list[np.ndarray]] = {"light": [], "dark": []}
    pooled_cross: dict[tuple[str, str, str], list[np.ndarray]] = {}
    individual_place: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    individual_cross: dict[
        tuple[str, str, str, str, str, str], list[np.ndarray]
    ] = {}
    trial_records: list[dict[str, Any]] = []
    comparison, comparison_label, transfer_family, pairs = (
        _dark_light.PANEL_E_CROSS_COMPARISONS[0]
    )
    for spec in database.specs:
        animal_name = str(spec["animal_name"])
        date = str(spec["date"])
        intervals, path_length_cm = database.trajectory_inputs(
            spec,
            epochs=(spec["light_epoch"], spec["dark_epoch"]),
        )
        for epoch_type, epoch in (
            ("light", spec["light_epoch"]),
            ("dark", spec["dark_epoch"]),
        ):
            place, place_selection = database.path_specific_place_decoding(
                spec, epoch=epoch
            )
            place_source = _result_reference(
                database,
                "path_specific_place_decoding",
                place_selection,
                "path_specific_place_decoding_id",
            )
            for decoding_trajectory in TRAJECTORY_TYPES:
                figure_2_adapter._append_decoding_laps(
                    trial_records,
                    true=place["true"],
                    decoded=place["decoded"],
                    normalization=path_length_cm,
                    intervals=intervals[str(epoch)][decoding_trajectory],
                    animal_name=animal_name,
                    date=date,
                    epoch_type=epoch_type,
                    epoch=str(epoch),
                    analysis="place",
                    comparison="place",
                    comparison_label="Place",
                    transfer_family="within_epoch",
                    encoding_trajectory=None,
                    decoding_trajectory=decoding_trajectory,
                    true_path=place_source / "true",
                    decoded_path=place_source / "decoded",
                )
            _timestamps, place_error = (
                figure_2_adapter.panel_helpers._align_absolute_error_with_times(
                    place["true"], place["decoded"]
                )
            )
            finite_place = np.asarray(place_error, dtype=float) / path_length_cm
            finite_place = finite_place[np.isfinite(finite_place)]
            pooled_place[epoch_type].append(finite_place)
            individual_place.setdefault(
                (animal_name, date, epoch_type, str(epoch)), []
            ).append(finite_place)

            cross, cross_selection = database.path_progression_decoding(
                spec, epoch=epoch
            )
            cross_source = _result_reference(
                database,
                "path_progression_decoding",
                cross_selection,
                "path_progression_decoding_id",
            )
            cross_values = []
            for encoding_trajectory, decoding_trajectory in pairs:
                key = (
                    str(transfer_family),
                    str(encoding_trajectory),
                    str(decoding_trajectory),
                )
                output = cross["cross_path_outputs"][key]
                values = figure_2_adapter._append_decoding_laps(
                    trial_records,
                    true=output["true"],
                    decoded=output["decoded"],
                    normalization=1.0,
                    intervals=intervals[str(epoch)][decoding_trajectory],
                    animal_name=animal_name,
                    date=date,
                    epoch_type=epoch_type,
                    epoch=str(epoch),
                    analysis="cross_trajectory",
                    comparison=str(comparison),
                    comparison_label=str(comparison_label),
                    transfer_family=str(transfer_family),
                    encoding_trajectory=str(encoding_trajectory),
                    decoding_trajectory=str(decoding_trajectory),
                    true_path=cross_source / (
                        f"{transfer_family}-{encoding_trajectory}-"
                        f"{decoding_trajectory}-true"
                    ),
                    decoded_path=cross_source / (
                        f"{transfer_family}-{encoding_trajectory}-"
                        f"{decoding_trajectory}-decoded"
                    ),
                )
                if values.size:
                    cross_values.append(values)
            if cross_values:
                concatenated = np.concatenate(cross_values)
                pooled_cross.setdefault(
                    (epoch_type, str(comparison), str(comparison_label)), []
                ).append(concatenated)
                individual_cross.setdefault(
                    (
                        animal_name,
                        date,
                        epoch_type,
                        str(epoch),
                        str(comparison),
                        str(comparison_label),
                    ),
                    [],
                ).append(concatenated)

    pooled_rows = []
    for epoch_type, values in pooled_place.items():
        finite = [value for value in values if value.size]
        if finite:
            row = _dark_light._summarize_panel_e_errors(
                np.concatenate(finite),
                animal_name=_dark_light.PANEL_E_POOLED_LABEL,
                date=_dark_light.PANEL_E_POOLED_LABEL,
                epoch_type=epoch_type,
                epoch=_dark_light.PANEL_E_POOLED_LABEL,
                analysis="place",
                comparison="place",
                comparison_label="Place",
            )
            if row is not None:
                pooled_rows.append(row)
    for (epoch_type, comparison_name, label), values in pooled_cross.items():
        row = _dark_light._summarize_panel_e_errors(
            np.concatenate(values),
            animal_name=_dark_light.PANEL_E_POOLED_LABEL,
            date=_dark_light.PANEL_E_POOLED_LABEL,
            epoch_type=epoch_type,
            epoch=_dark_light.PANEL_E_POOLED_LABEL,
            analysis="cross_trajectory",
            comparison=comparison_name,
            comparison_label=label,
        )
        if row is not None:
            pooled_rows.append(row)

    individual_rows = []
    for (animal_name, date, epoch_type, epoch), values in (
        individual_place.items()
    ):
        finite = [value for value in values if value.size]
        if finite:
            row = _dark_light._summarize_panel_e_errors(
                np.concatenate(finite),
                animal_name=animal_name,
                date=date,
                epoch_type=epoch_type,
                epoch=epoch,
                analysis="place",
                comparison="place",
                comparison_label="Place",
            )
            if row is not None:
                individual_rows.append(row)
    for (
        animal_name,
        date,
        epoch_type,
        epoch,
        comparison_name,
        label,
    ), values in individual_cross.items():
        row = _dark_light._summarize_panel_e_errors(
            np.concatenate(values),
            animal_name=animal_name,
            date=date,
            epoch_type=epoch_type,
            epoch=epoch,
            analysis="cross_trajectory",
            comparison=comparison_name,
            comparison_label=label,
        )
        if row is not None:
            individual_rows.append(row)
    return (
        pd.DataFrame.from_records(
            pooled_rows,
            columns=_dark_light.PANEL_E_ERROR_SUMMARY_COLUMNS,
        ),
        pd.DataFrame.from_records(
            individual_rows,
            columns=_dark_light.PANEL_E_ERROR_SUMMARY_COLUMNS,
        ),
        pd.DataFrame.from_records(
            trial_records,
            columns=(
                figure_2_adapter.panel_helpers.PANEL_E_TRIAL_ERROR_TABLE_COLUMNS
            ),
        ),
    )


def build_supplementary_figure_4_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build decoding and cvPCA inputs for Supplementary Figure 4."""
    cv_rows = []
    for spec in database.specs:
        result, selection = database.cv_pca(spec)
        source = _result_reference(
            database, "cv_pca", selection, "cv_pca_id"
        )
        summary = result["summary"]
        for condition in ("dark", "light"):
            selected = summary.loc[
                summary["condition"].astype(str).eq(condition)
            ]
            if len(selected) != 1:
                raise ValueError(f"cvPCA lacks one {condition} summary row.")
            row = selected.iloc[0]
            cv_rows.append(
                {
                    "animal_name": str(spec["animal_name"]),
                    "date": str(spec["date"]),
                    "region": "v1",
                    "dark_epoch": str(spec["dark_epoch"]),
                    "light_epoch": str(spec["light_epoch"]),
                    "condition": condition,
                    "participation_ratio": float(
                        row["within_cv_participation_ratio"]
                    ),
                    "n_units": int(row["n_units"]),
                    "source_path": source,
                }
            )
    summary, individual, trial = _figure_2_decoding_tables(database)
    permutations = (
        supplementary_figure_4.figure_2.compute_panel_e_decoding_permutation_tests(
            trial,
            n_permutations=(
                supplementary_figure_4.figure_2.DECODING_PERMUTATION_COUNT
            ),
            seed=supplementary_figure_4.figure_2.DECODING_PERMUTATION_SEED,
        )
    )
    animal_names = tuple(spec["animal_name"] for spec in database.specs)
    labels = (
        supplementary_figure_4.figure_2.build_panel_e_decoding_significance_labels(
            permutations,
            animal_names=animal_names,
        )
    )
    result_animals = permutations["animal_name"].astype(str)
    individual_labels = {
        animal_name: (
            supplementary_figure_4.figure_2.build_panel_e_decoding_significance_labels(
                permutations.loc[result_animals == animal_name].copy(),
                animal_names=(animal_name,),
            )
        )
        for animal_name in animal_names
    }
    return {
        "run_dir": Path(run_dir),
        "datasets": supp_4_adapter.EXPECTED_DATASETS,
        "regions": (supp_4_adapter.REGION,),
        "cv_pca_table": pd.DataFrame.from_records(cv_rows),
        "decoding_data": {
            "decoding_error": summary,
            "individual_decoding_error": individual,
            "decoding_significance_labels": labels,
            "individual_decoding_significance_labels": individual_labels,
            "decoding_trial_error": trial,
        },
    }


def build_supplementary_figure_5_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Build motor-progression inputs for Supplementary Figure 5."""
    tables = []
    for spec in database.specs:
        for epoch_type, epoch in (
            ("dark", spec["dark_epoch"]),
            ("light", spec["light_epoch"]),
        ):
            result, selection = database.epoch_motor_behavior(
                spec, epoch=epoch
            )
            table = result["progression_summary"].copy()
            table["animal_name"] = str(spec["animal_name"])
            table["date"] = str(spec["date"])
            table["dark_epoch"] = str(spec["dark_epoch"])
            table["light_epoch"] = str(spec["light_epoch"])
            table["dataset_label"] = (
                f"{spec['animal_name']} {spec['date']}"
            )
            table["epoch_type"] = epoch_type
            table["source_path"] = str(
                _result_reference(
                    database,
                    "epoch_motor_behavior",
                    selection,
                    "epoch_motor_behavior_id",
                )
            )
            tables.append(table)
    output = pd.concat(tables, ignore_index=True, sort=False)
    missing = sorted(
        set(supplementary_figure_5.MOTOR_PANEL_COLUMNS).difference(
            output.columns
        )
    )
    if missing:
        raise ValueError(f"Motor summaries are missing columns {missing!r}.")
    return {
        "run_dir": Path(run_dir),
        "datasets": supp_5_adapter.EXPECTED_DATASETS,
        "regions": (),
        "motor_progression_table": output,
    }


def build_supplementary_figure_6_payload(
    database: SpyglassFigureDatabase,
    *,
    run_dir: Path,
    figure_3_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build scalar-GLM and empirical-additive inputs for Supplement 6."""
    loaded = figure_3_payload["_loaded_swap_results"]
    empirical = {}
    for spec in database.specs:
        result, selection = database.swap_tuning(spec)
        key = (str(spec["animal_name"]), str(spec["date"]))
        summary = supp_6_adapter._map_empirical_summary(
            result,
            sorting_unit_by_nwb_id=database.unit_maps(spec)["v1"],
        )
        empirical[key] = {
            "summary": summary,
            "source_path": _result_reference(
                database,
                "swap_tuning_curve_comparison",
                selection,
                "swap_tuning_curve_comparison_id",
            ),
        }
    return {
        "run_dir": Path(run_dir),
        "datasets": supp_6_adapter.EXPECTED_DATASETS,
        "regions": (supp_6_adapter.REGION,),
        "scalar_swap_delta_table": figure_3_adapter._delta_table(
            loaded, model_name=supp_6_adapter.SCALAR_MODEL
        ),
        "mixed_full_additive_table": (
            supp_6_adapter._build_mixed_full_additive_table(loaded, empirical)
        ),
    }


def build_supplementary_figure_7_payload(
    figure_4_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the two Figure 4 products used by Supplementary Figure 7."""
    return {
        "run_dir": figure_4_payload["run_dir"],
        "datasets": supp_7_adapter.EXPECTED_DATASETS,
        "regions": supp_7_adapter.REGIONS,
        "heatmap_epoch_tables": figure_4_payload["heatmap_epoch_tables"],
        "source_comparison_payload": figure_4_payload[
            "source_comparison_payload"
        ],
    }


def _render_figure_1(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with figure_1_adapter._offline_legacy_sources(payload):
        return figure_1.make_figure_1(
            data_root=Path(payload["run_dir"]),
            asset_dir=figure_1.DEFAULT_ASSET_DIR,
            output_path=output_path,
            datasets=payload["datasets"],
            regions=payload["regions"],
            position_bin_count=figure_1.DEFAULT_POSITION_BIN_COUNT,
            position_offset=figure_1.DEFAULT_POSITION_OFFSET,
            speed_threshold_cm_s=figure_1.DEFAULT_SPEED_THRESHOLD_CM_S,
            sigma_bins=figure_1.DEFAULT_SIGMA_BINS,
            encoding_bin_size_s=figure_1.ENCODING_COMPARISON_BIN_SIZE_S,
            encoding_place_bin_size_cm=(
                figure_1.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
            ),
            dpi=int(dpi),
            decoding_n_permutations=figure_1.DECODING_PERMUTATION_COUNT,
            decoding_permutation_seed=figure_1.DECODING_PERMUTATION_SEED,
            panel_d_cache_dir=Path(payload["run_dir"]) / "cache",
            panel_e_cache_dir=Path(payload["run_dir"]) / "cache",
            panel_dark_light_example_cache_dir=(
                Path(payload["run_dir"]) / "cache"
            ),
        )


def _render_figure_2(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with figure_2_adapter._offline_sources(payload):
        return figure_2.make_figure_2(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            regions=payload["regions"],
            light_epoch=figure_2_adapter.LIGHT_EPOCH,
            dark_epoch=None,
            position_bin_count=figure_2._figure_2.DEFAULT_POSITION_BIN_COUNT,
            position_offset=figure_2._figure_2.DEFAULT_POSITION_OFFSET,
            speed_threshold_cm_s=figure_2._figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
            sigma_bins=figure_2._figure_2.DEFAULT_SIGMA_BINS,
            dpi=int(dpi),
            panel_example_cache_dir=Path(payload["run_dir"]) / "cache",
            refresh_panel_example_cache=False,
            panel_tuning_similarity_cache_dir=(
                Path(payload["run_dir"]) / "cache"
            ),
            refresh_panel_tuning_similarity_cache=False,
        )


def _render_figure_3(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with figure_3_adapter._offline_panel_data(payload):
        return figure_3.make_figure_3(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            regions=payload["regions"],
            dark_epoch=None,
            dpi=int(dpi),
        )


def _render_supplementary_figure_1(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with supp_1_adapter._offline_sources(payload):
        return supplementary_figure_1.make_supplementary_figure_1(
            data_root=Path(payload["run_dir"]),
            asset_dir=supplementary_figure_1.DEFAULT_ASSET_DIR,
            output_path=output_path,
            datasets=payload["datasets"],
            dpi=int(dpi),
            dark_movement_fr_cache_dir=Path(payload["run_dir"]) / "cache",
            refresh_dark_movement_fr_cache=False,
        )


def _render_supplementary_figure_2(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    canonical_bracket_builder = figure_1.build_decoding_significance_brackets
    with figure_1_adapter._offline_legacy_sources(payload):
        adapter_bracket_builder = figure_1.build_decoding_significance_brackets

        def bracket_builder(
            per_animal_results: Any, **kwargs: Any
        ) -> Any:
            return canonical_bracket_builder(per_animal_results, **kwargs)

        figure_1.build_decoding_significance_brackets = bracket_builder
        try:
            return supplementary_figure_2.make_supplementary_figure_2(
                data_root=Path(payload["run_dir"]),
                output_path=output_path,
                datasets=payload["datasets"],
                dpi=int(dpi),
            )
        finally:
            figure_1.build_decoding_significance_brackets = (
                adapter_bracket_builder
            )


def _render_supplementary_figure_3(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with figure_2_adapter._offline_sources(payload):
        return supplementary_figure_3.make_supplementary_figure_3(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            region=figure_2_adapter.REGION,
            light_epoch=figure_2_adapter.LIGHT_EPOCH,
            dark_epoch=None,
            dpi=int(dpi),
            panel_tuning_similarity_cache_dir=(
                Path(payload["run_dir"]) / "cache"
            ),
            refresh_panel_tuning_similarity_cache=False,
        )


def _render_figure_4(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with figure_4_adapter._offline_sources(payload):
        return figure_4.make_figure_4(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            example_dataset=figure_4_adapter.legacy.DEFAULT_EXAMPLE_DATASET,
            light_epoch=figure_4_adapter.LIGHT_EPOCH,
            dark_epoch=None,
            sleep_epoch=None,
            ripple_threshold_zscore=None,
            ripple_window_s=figure_4_adapter.legacy.DEFAULT_RIPPLE_WINDOW_S,
            ripple_window_offset_s=(
                figure_4_adapter.legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S
            ),
            ripple_selection=(
                figure_4_adapter.legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION
            ),
            ridge_strength=figure_4_adapter.legacy.DEFAULT_RIDGE_STRENGTH,
            dark_movement_fr_cache_dir=Path(payload["run_dir"]) / "cache",
            refresh_dark_movement_fr_cache=False,
            refresh_panel_b_schematic_cache=False,
            dpi=int(dpi),
            panel_d_tuning_similarity_metric=(
                figure_4_adapter.legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
            ),
        )


def _render_supplementary_figure_4(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with supp_4_adapter._offline_sources(payload):
        return supplementary_figure_4.make_supplementary_figure_4(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            region=supp_4_adapter.REGION,
            light_epoch=None,
            dark_epoch=None,
            dpi=int(dpi),
            decoding_n_permutations=(
                supplementary_figure_4.figure_2.DECODING_PERMUTATION_COUNT
            ),
            decoding_permutation_seed=(
                supplementary_figure_4.figure_2.DECODING_PERMUTATION_SEED
            ),
        )


def _render_supplementary_figure_5(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with supp_5_adapter._offline_sources(payload):
        return supplementary_figure_5.make_supplementary_figure_5(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            dpi=int(dpi),
        )


def _render_supplementary_figure_6(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with supp_6_adapter._offline_sources(payload):
        return supplementary_figure_6.make_supplementary_figure_6(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            region=supp_6_adapter.REGION,
            dark_epoch=None,
            dpi=int(dpi),
        )


def _render_supplementary_figure_7(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with supp_7_adapter._offline_sources(payload):
        return supplementary_figure_7.make_supplementary_figure_7(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            regions=payload["regions"],
            light_epoch=supp_7_adapter.LIGHT_EPOCH,
            dark_epoch=None,
            sleep_epoch=None,
            ripple_threshold_zscore=None,
            ripple_window_s=supp_7_adapter.RIPPLE_WINDOW_S,
            ripple_window_offset_s=supp_7_adapter.RIPPLE_WINDOW_OFFSET_S,
            ripple_selection=supp_7_adapter.RIPPLE_SELECTION,
            ridge_strength=supp_7_adapter.RIDGE_STRENGTH,
            dpi=int(dpi),
        )


def _render_supplementary_figure_8(
    payload: Mapping[str, Any], output_path: Path, *, dpi: int
) -> Path:
    with figure_4_adapter._offline_sources(payload):
        return supplementary_figure_8.make_supplementary_figure_8(
            data_root=Path(payload["run_dir"]),
            output_path=output_path,
            datasets=payload["datasets"],
            light_epoch=figure_4_adapter.LIGHT_EPOCH,
            dark_epoch=None,
            sleep_epoch=None,
            ripple_window_s=figure_4_adapter.legacy.DEFAULT_RIPPLE_WINDOW_S,
            ripple_window_offset_s=(
                figure_4_adapter.legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S
            ),
            ripple_selection=(
                figure_4_adapter.legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION
            ),
            ridge_strength=figure_4_adapter.legacy.DEFAULT_RIDGE_STRENGTH,
            dark_movement_fr_cache_dir=Path(payload["run_dir"]) / "cache",
            refresh_dark_movement_fr_cache=False,
            dpi=int(dpi),
            tuning_similarity_metric=(
                figure_4_adapter.legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
            ),
        )


def _file_sha256(path: Path) -> str:
    """Return one generated file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(event: str, **values: Any) -> None:
    """Print one machine-readable progress record."""
    print(json.dumps({"event": event, **values}, default=str, sort_keys=True), flush=True)


def generate_spyglass_figures(
    database: SpyglassFigureDatabase,
    *,
    output_dir: Path,
    figure_names: Sequence[str] = FIGURE_NAMES,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    output_formats: Sequence[str] | None = None,
    dpi: int = DEFAULT_DPI,
    replace: bool = False,
) -> list[Path]:
    """Build requested payloads once and render every requested format."""
    selected = tuple(dict.fromkeys(str(name) for name in figure_names))
    unknown = sorted(set(selected).difference(FIGURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown figure names {unknown!r}.")
    requested_formats = (
        (str(output_formats),)
        if isinstance(output_formats, str)
        else tuple(str(value) for value in output_formats or ())
    )
    formats = tuple(
        dict.fromkeys(requested_formats or (str(output_format),))
    )
    unknown_formats = sorted(set(formats).difference(figure_1.FIGURE_FORMATS))
    if unknown_formats:
        raise ValueError(
            f"output formats must be among {figure_1.FIGURE_FORMATS!r}; "
            f"found {unknown_formats!r}."
        )
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive.")
    run_dir = Path(output_dir).resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, Mapping[str, Any]] = {}
    if {"figure_1", "supplementary_figure_2"}.intersection(selected):
        _emit("payload_started", figure="figure_1")
        payloads["figure_1"] = build_figure_1_payload(
            database, run_dir=run_dir
        )
        _emit("payload_complete", figure="figure_1")
    if "supplementary_figure_1" in selected:
        _emit("payload_started", figure="supplementary_figure_1")
        payloads["supplementary_figure_1"] = (
            build_supplementary_figure_1_payload(database, run_dir=run_dir)
        )
        _emit("payload_complete", figure="supplementary_figure_1")
    if {"figure_2", "supplementary_figure_3"}.intersection(selected):
        _emit("payload_started", figure="figure_2")
        payloads["figure_2"] = build_figure_2_payload(
            database, run_dir=run_dir
        )
        _emit("payload_complete", figure="figure_2")
    if {"figure_3", "supplementary_figure_6"}.intersection(selected):
        _emit("payload_started", figure="figure_3")
        payloads["figure_3"] = build_figure_3_payload(
            database, run_dir=run_dir
        )
        _emit("payload_complete", figure="figure_3")
    if {"figure_4", "supplementary_figure_7", "supplementary_figure_8"}.intersection(
        selected
    ):
        _emit("payload_started", figure="figure_4")
        payloads["figure_4"] = build_figure_4_payload(
            database, run_dir=run_dir
        )
        _emit("payload_complete", figure="figure_4")
    if "supplementary_figure_4" in selected:
        _emit("payload_started", figure="supplementary_figure_4")
        payloads["supplementary_figure_4"] = (
            build_supplementary_figure_4_payload(database, run_dir=run_dir)
        )
        _emit("payload_complete", figure="supplementary_figure_4")
    if "supplementary_figure_5" in selected:
        _emit("payload_started", figure="supplementary_figure_5")
        payloads["supplementary_figure_5"] = (
            build_supplementary_figure_5_payload(database, run_dir=run_dir)
        )
        _emit("payload_complete", figure="supplementary_figure_5")
    if "supplementary_figure_6" in selected:
        payloads["supplementary_figure_6"] = (
            build_supplementary_figure_6_payload(
                database,
                run_dir=run_dir,
                figure_3_payload=payloads["figure_3"],
            )
        )
    if "supplementary_figure_7" in selected:
        payloads["supplementary_figure_7"] = (
            build_supplementary_figure_7_payload(payloads["figure_4"])
        )
    if "supplementary_figure_8" in selected:
        payloads["supplementary_figure_8"] = payloads["figure_4"]

    renderers: dict[str, tuple[str, Callable[..., Path]]] = {
        "figure_1": ("figure_1", _render_figure_1),
        "figure_2": ("figure_2", _render_figure_2),
        "figure_3": ("figure_3", _render_figure_3),
        "figure_4": ("figure_4", _render_figure_4),
        "supplementary_figure_1": (
            "supplementary_figure_1",
            _render_supplementary_figure_1,
        ),
        "supplementary_figure_2": (
            "figure_1",
            _render_supplementary_figure_2,
        ),
        "supplementary_figure_3": (
            "figure_2",
            _render_supplementary_figure_3,
        ),
        "supplementary_figure_4": (
            "supplementary_figure_4",
            _render_supplementary_figure_4,
        ),
        "supplementary_figure_5": (
            "supplementary_figure_5",
            _render_supplementary_figure_5,
        ),
        "supplementary_figure_6": (
            "supplementary_figure_6",
            _render_supplementary_figure_6,
        ),
        "supplementary_figure_7": (
            "supplementary_figure_7",
            _render_supplementary_figure_7,
        ),
        "supplementary_figure_8": (
            "supplementary_figure_8",
            _render_supplementary_figure_8,
        ),
    }
    output_records = []
    for current_format in formats:
        for name in selected:
            payload_name, renderer = renderers[name]
            output_path = run_dir / f"{name}.{current_format}"
            _emit(
                "render_started",
                figure=name,
                output_format=current_format,
                output_path=output_path,
            )
            rendered = _atomic_render(
                output_path,
                replace=replace,
                render=lambda temporary, renderer=renderer, payload_name=payload_name: renderer(
                    payloads[payload_name], temporary, dpi=int(dpi)
                ),
            )
            output_records.append(
                {
                    "name": name,
                    "format": current_format,
                    "path": rendered,
                }
            )
            _emit(
                "render_complete",
                figure=name,
                output_format=current_format,
                output_path=output_path,
            )

    manifest = {
        "schema_name": database.schema_name,
        "analysis_nwbfile_schema_name": (
            database.analysis_nwbfile_schema_name
        ),
        "nwb_root": str(database.nwb_root),
        "dpi": int(dpi),
        "output_formats": list(formats),
        "figures": [
            {
                "name": record["name"],
                "format": record["format"],
                "path": str(record["path"]),
                "sha256": _file_sha256(record["path"]),
                "size_bytes": record["path"].stat().st_size,
            }
            for record in output_records
        ],
        "result_rows": database.selected_rows(),
    }
    manifest_path = run_dir / "spyglass_figure_generation.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    _emit(
        "generation_complete",
        figures=len(selected),
        formats=formats,
        outputs=len(output_records),
        manifest=manifest_path,
    )
    return [record["path"] for record in output_records]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the database-backed manuscript figure command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=FIGURE_NAMES,
        default=FIGURE_NAMES,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--output-format",
        choices=figure_1.FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
    )
    format_group.add_argument(
        "--output-formats",
        nargs="+",
        choices=figure_1.FIGURE_FORMATS,
        default=None,
        help="Render all formats after building database payloads once.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--schema-name", default=table_specs.DEFAULT_SCHEMA_NAME
    )
    parser.add_argument(
        "--analysis-nwbfile-schema-name",
        default=table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=Path("/stelmo/nwb/raw"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the populated pipeline and generate requested figures."""
    args = parse_arguments(argv)
    database = SpyglassFigureDatabase(
        schema_name=args.schema_name,
        analysis_nwbfile_schema_name=args.analysis_nwbfile_schema_name,
        nwb_root=args.nwb_root,
    )
    generate_spyglass_figures(
        database,
        output_dir=args.output_dir,
        figure_names=args.figures,
        output_format=args.output_format,
        output_formats=args.output_formats,
        dpi=args.dpi,
        replace=args.replace,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FIGURE_NAMES",
    "build_figure_1_payload",
    "build_figure_2_payload",
    "build_figure_3_payload",
    "build_figure_4_payload",
    "generate_spyglass_figures",
    "main",
]
