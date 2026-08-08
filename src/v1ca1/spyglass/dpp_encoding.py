"""Database-free four-model DPP encoding comparison and Parquet artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES, TURN_TRAJECTORY_PAIRS


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "dpp_encoding"
ARTIFACT_FILENAME = "dpp_encoding.parquet"
FULL_W_CONFIGURATION_NAME = "full_w"
MODEL_NAMES = (
    "path_specific_place",
    "absolute_place",
    "dpp",
    "distance_to_reward",
)
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
STABILITY_COLUMNS = tuple(
    f"{trajectory_type}_stability_correlation"
    for trajectory_type in TRAJECTORY_TYPES
)
MODEL_QC_STATUSES = (
    "valid",
    "zero_training_spikes",
    "zero_heldout_spikes",
    "nonfinite_likelihood",
)
UNIT_QC_STATUSES = (
    "valid",
    "zero_training_spikes",
    "zero_heldout_spikes",
    "partial_model_failure",
    "no_valid_models",
)
CONTRAST_COLUMNS = (
    "dpp_vs_path_specific_place_bits_per_spike",
    "dpp_vs_absolute_place_bits_per_spike",
    "dpp_vs_distance_to_reward_bits_per_spike",
)
TABLE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "dpp_encoding_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "n_folds",
    "evaluation_bin_size_s",
    "spatial_bin_size_cm",
    "gaussian_smoothing_sigma_bins",
    "random_seed",
    "minimum_movement_firing_rate_hz",
    "minimum_stability_correlation",
    "movement_firing_rate_hz",
    *STABILITY_COLUMNS,
    "heldout_spike_count",
    "null_log_likelihood_nats",
    *tuple(
        column
        for model_name in MODEL_NAMES
        for column in (
            f"{model_name}_log_likelihood_nats",
            f"{model_name}_information_bits_per_spike",
            f"{model_name}_qc_status",
        )
    ),
    *CONTRAST_COLUMNS,
    "unit_valid",
    "qc_status",
)
LEGACY_METRIC_COLUMNS = (
    "n_spikes",
    "ll_null",
    "ll_place",
    "ll_generalized_place",
    "ll_tp",
    "ll_gtp",
    "info_bits_place",
    "info_bits_generalized_place",
    "info_bits_tp",
    "info_bits_gtp",
    "delta_bits_place_vs_tp",
    "delta_bits_generalized_place_vs_tp",
    "delta_bits_gtp_vs_tp",
)
LEGACY_MODEL_COLUMNS = {
    "path_specific_place": ("ll_place", "info_bits_place"),
    "absolute_place": (
        "ll_generalized_place",
        "info_bits_generalized_place",
    ),
    "dpp": ("ll_tp", "info_bits_tp"),
    "distance_to_reward": ("ll_gtp", "info_bits_gtp"),
}
_INBOUND_TRAJECTORIES = {"left_to_center", "right_to_center"}
_DPP_OFFSETS = {
    trajectory_type: float(turn_index)
    for turn_index, trajectories in enumerate(TURN_TRAJECTORY_PAIRS.values())
    for trajectory_type in trajectories
}


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe, non-empty path component."""
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return value


def _uuid_component(value: Any, *, name: str) -> str:
    """Return one canonical UUID path component."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def get_dpp_encoding_artifact_path(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    dpp_encoding_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return one UUID-keyed, session-first encoding-comparison path."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "region": region,
        }.items()
    }
    comparison_id = _uuid_component(
        dpp_encoding_id,
        name="dpp_encoding_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["region"]
        / comparison_id
        / ARTIFACT_FILENAME
    )


def empty_dpp_encoding_table() -> pd.DataFrame:
    """Return an empty encoding-comparison table with its canonical schema."""
    columns: dict[str, pd.Series] = {
        name: pd.Series(dtype=str) for name in IDENTITY_COLUMNS
    }
    columns.update(
        {
            "dpp_encoding_id": pd.Series(dtype=str),
            "animal_name": pd.Series(dtype=str),
            "date": pd.Series(dtype=str),
            "region": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "n_folds": pd.Series(dtype=np.int64),
            "evaluation_bin_size_s": pd.Series(dtype=float),
            "spatial_bin_size_cm": pd.Series(dtype=float),
            "gaussian_smoothing_sigma_bins": pd.Series(dtype=float),
            "random_seed": pd.Series(dtype=np.int64),
            "minimum_movement_firing_rate_hz": pd.Series(dtype=float),
            "minimum_stability_correlation": pd.Series(dtype=float),
            "movement_firing_rate_hz": pd.Series(dtype=float),
            **{
                name: pd.Series(dtype=float) for name in STABILITY_COLUMNS
            },
            "heldout_spike_count": pd.Series(dtype=np.int64),
            "null_log_likelihood_nats": pd.Series(dtype=float),
        }
    )
    for model_name in MODEL_NAMES:
        columns[f"{model_name}_log_likelihood_nats"] = pd.Series(dtype=float)
        columns[f"{model_name}_information_bits_per_spike"] = pd.Series(
            dtype=float
        )
        columns[f"{model_name}_qc_status"] = pd.Series(dtype=str)
    columns.update(
        {name: pd.Series(dtype=float) for name in CONTRAST_COLUMNS}
    )
    columns["unit_valid"] = pd.Series(dtype=bool)
    columns["qc_status"] = pd.Series(dtype=str)
    return pd.DataFrame(columns).loc[:, list(TABLE_COLUMNS)]


def _validate_thresholds(
    *,
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
) -> tuple[float, float]:
    """Return finite eligibility thresholds."""
    firing_rate = float(minimum_movement_firing_rate_hz)
    stability = float(minimum_stability_correlation)
    if not np.isfinite(firing_rate) or firing_rate < 0.0:
        raise ValueError(
            "minimum_movement_firing_rate_hz must be non-negative and finite."
        )
    if not np.isfinite(stability) or stability < -1.0 or stability > 1.0:
        raise ValueError(
            "minimum_stability_correlation must be finite and within [-1, 1]."
        )
    return firing_rate, stability


def _selected_identity_table(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return persistent unit identities aligned to current TsGroup keys."""
    from v1ca1.spyglass.path_specific_place import _identity_rows

    rows = _identity_rows(spikes, stable_unit_ids)
    return pd.DataFrame.from_records(
        rows,
        columns=(*IDENTITY_COLUMNS, "_group_key"),
    )


def _require_matching_metadata(
    table: pd.DataFrame,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    table_name: str,
) -> None:
    """Require one non-empty upstream table to match the selected session."""
    for field_name, expected in {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
    }.items():
        if field_name not in table:
            raise ValueError(f"{table_name} is missing {field_name!r}.")
        values = table[field_name].astype(str).unique().tolist()
        if values != [str(expected)]:
            raise ValueError(
                f"{table_name} does not match the selected {field_name}."
            )


def _align_upstream_by_stable_id(
    table: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    table_name: str,
) -> pd.DataFrame:
    """Return one upstream unit table in selected persistent-identity order."""
    required = {"spikesorting_merge_id", "unit_id", "stable_unit_id"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{table_name} is missing identity columns {missing!r}.")
    observed = table.copy()
    for name in required:
        observed[name] = observed[name].astype(str)
    if observed["stable_unit_id"].duplicated().any():
        raise ValueError(f"{table_name} has duplicate stable_unit_id rows.")
    expected_ids = identity["stable_unit_id"].astype(str).tolist()
    observed = observed.set_index("stable_unit_id", drop=False)
    if set(observed.index) != set(expected_ids):
        raise ValueError(
            f"{table_name} identities do not exactly match the selected units."
        )
    observed = observed.loc[expected_ids].reset_index(drop=True)
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        if observed[name].astype(str).tolist() != identity[name].astype(str).tolist():
            raise ValueError(f"{table_name} {name} does not match selected units.")
    return observed


def build_encoding_eligibility_table(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
) -> pd.DataFrame:
    """Return all selected units with fixed firing-rate/stability eligibility."""
    firing_rate_threshold, stability_threshold = _validate_thresholds(
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
    )
    identity = _selected_identity_table(spikes, stable_unit_ids)
    actual_trajectories = set(stability_tables_by_trajectory)
    expected_trajectories = set(TRAJECTORY_TYPES)
    if actual_trajectories != expected_trajectories:
        raise ValueError(
            "stability_tables_by_trajectory must contain exactly the four "
            "trajectory types; "
            f"missing={sorted(expected_trajectories - actual_trajectories)!r}, "
            f"extra={sorted(actual_trajectories - expected_trajectories)!r}."
        )
    if identity.empty:
        if not movement_firing_rate_table.empty or any(
            not table.empty for table in stability_tables_by_trajectory.values()
        ):
            raise ValueError("No selected units require empty upstream unit tables.")
        output = identity.copy()
        output["movement_firing_rate_hz"] = pd.Series(dtype=float)
        for column in STABILITY_COLUMNS:
            output[column] = pd.Series(dtype=float)
        output["eligible"] = pd.Series(dtype=bool)
        return output

    from v1ca1.spyglass.movement import validate_movement_firing_rate_table

    validate_movement_firing_rate_table(movement_firing_rate_table)
    _require_matching_metadata(
        movement_firing_rate_table,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        table_name="MovementFiringRate",
    )
    movement = _align_upstream_by_stable_id(
        movement_firing_rate_table,
        identity,
        table_name="MovementFiringRate",
    )
    rates = pd.to_numeric(
        movement["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(np.isinf(rates)) or np.any(rates[np.isfinite(rates)] < 0.0):
        raise ValueError("Movement firing rates must be non-negative or NaN.")

    output = identity.copy()
    output["movement_firing_rate_hz"] = rates
    stability_arrays: list[np.ndarray] = []
    for trajectory_type in TRAJECTORY_TYPES:
        table = stability_tables_by_trajectory[trajectory_type]
        _require_matching_metadata(
            table,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            table_name=f"Stability[{trajectory_type}]",
        )
        if "trajectory_type" not in table or table[
            "trajectory_type"
        ].astype(str).unique().tolist() != [trajectory_type]:
            raise ValueError(
                f"Stability[{trajectory_type}] has a mismatched trajectory_type."
            )
        observed = _align_upstream_by_stable_id(
            table,
            identity,
            table_name=f"Stability[{trajectory_type}]",
        )
        if "stability_correlation" not in observed:
            raise ValueError(
                f"Stability[{trajectory_type}] is missing stability_correlation."
            )
        correlations = pd.to_numeric(
            observed["stability_correlation"], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(correlations)
        if np.any(np.isinf(correlations)) or np.any(
            (correlations[finite] < -1.0 - 1e-9)
            | (correlations[finite] > 1.0 + 1e-9)
        ):
            raise ValueError("Stability correlations must be within [-1, 1] or NaN.")
        if "stability_status" in observed:
            statuses = observed["stability_status"].astype(str).to_numpy()
            if not np.array_equal(finite, statuses == "valid"):
                raise ValueError(
                    "Finite stability correlations must correspond exactly to "
                    "stability_status='valid'."
                )
        if "firing_rate_hz" in observed:
            stability_rates = pd.to_numeric(
                observed["firing_rate_hz"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.allclose(
                stability_rates,
                rates,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Stability[{trajectory_type}] firing rates disagree with "
                    "MovementFiringRate."
                )
        column = f"{trajectory_type}_stability_correlation"
        output[column] = correlations
        stability_arrays.append(correlations)

    stability_matrix = np.column_stack(stability_arrays)
    output["eligible"] = (
        np.isfinite(rates)
        & (rates >= firing_rate_threshold)
        & np.any(
            np.isfinite(stability_matrix)
            & (stability_matrix >= stability_threshold),
            axis=1,
        )
    )
    return output


def _validate_analysis_parameters(
    *,
    n_folds: int,
    evaluation_bin_size_s: float,
    spatial_bin_size_cm: float,
    gaussian_smoothing_sigma_bins: float,
    random_seed: int,
) -> dict[str, Any]:
    """Return validated fixed cross-validation and tuning parameters."""
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise TypeError("n_folds must be an integer.")
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2.")
    if isinstance(random_seed, bool) or not isinstance(
        random_seed, (int, np.integer)
    ):
        raise TypeError("random_seed must be an integer.")
    if int(random_seed) < 0:
        raise ValueError("random_seed must be non-negative.")
    numeric = {
        "evaluation_bin_size_s": float(evaluation_bin_size_s),
        "spatial_bin_size_cm": float(spatial_bin_size_cm),
        "gaussian_smoothing_sigma_bins": float(
            gaussian_smoothing_sigma_bins
        ),
    }
    if not np.isfinite(numeric["evaluation_bin_size_s"]) or numeric[
        "evaluation_bin_size_s"
    ] <= 0.0:
        raise ValueError("evaluation_bin_size_s must be positive and finite.")
    if not np.isfinite(numeric["spatial_bin_size_cm"]) or numeric[
        "spatial_bin_size_cm"
    ] <= 0.0:
        raise ValueError("spatial_bin_size_cm must be positive and finite.")
    if not np.isfinite(numeric["gaussian_smoothing_sigma_bins"]) or numeric[
        "gaussian_smoothing_sigma_bins"
    ] < 0.0:
        raise ValueError(
            "gaussian_smoothing_sigma_bins must be non-negative and finite."
        )
    return {
        "n_folds": int(n_folds),
        **numeric,
        "random_seed": int(random_seed),
    }


def validate_trajectory_lap_counts(
    trajectory_intervals_by_type: Mapping[str, Any],
    *,
    n_folds: int,
) -> dict[str, int]:
    """Require every trajectory to contain at least one lap per fold."""
    from v1ca1.spyglass.path_specific_place import _extract_interval_bounds

    actual = set(trajectory_intervals_by_type)
    expected = set(TRAJECTORY_TYPES)
    if actual != expected:
        raise ValueError(
            "trajectory_intervals_by_type must contain exactly four trajectories; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )
    counts = {
        trajectory_type: int(
            _extract_interval_bounds(
                trajectory_intervals_by_type[trajectory_type]
            )[0].size
        )
        for trajectory_type in TRAJECTORY_TYPES
    }
    insufficient = {
        name: count for name, count in counts.items() if count < int(n_folds)
    }
    if insufficient:
        raise ValueError(
            "Every trajectory must contain at least n_folds laps; "
            f"n_folds={n_folds}, insufficient={insufficient!r}."
        )
    return counts


def _restrict_feature(feature: Any, support: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned feature timestamps and values on one support."""
    restrict = getattr(feature, "restrict", None)
    if not callable(restrict):
        raise TypeError("Encoding features must expose restrict().")
    restricted = restrict(support)
    times = np.asarray(restricted.t, dtype=float).reshape(-1)
    values = np.asarray(restricted.d, dtype=float).reshape(-1)
    if times.shape != values.shape or not np.all(np.isfinite(times)):
        raise ValueError("Encoding feature values and finite timestamps must align.")
    return times, values


def _values_at_exact_times(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Select feature values at an exact subset of source timestamps."""
    if source_times.size == 0 and target_times.size == 0:
        return np.asarray([], dtype=float)
    indices = np.searchsorted(source_times, target_times)
    if np.any(indices >= source_times.size) or not np.allclose(
        source_times[indices],
        target_times,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Full-W and trajectory-specific features do not share exact timestamps."
        )
    return source_values[indices]


def _build_repeated_block_bin_edges(
    *,
    block_length: float,
    block_count: int,
    bin_width: float,
) -> np.ndarray:
    """Return local-width bin edges aligned to every repeated-block boundary."""
    length = float(block_length)
    width = float(bin_width)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("block_length must be positive and finite.")
    if isinstance(block_count, bool) or not isinstance(
        block_count, (int, np.integer)
    ):
        raise TypeError("block_count must be an integer.")
    count = int(block_count)
    if count < 1:
        raise ValueError("block_count must be positive.")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin_width must be positive and finite.")

    edges = [0.0]
    for block_index in range(count):
        block_start = block_index * length
        local_edges = np.arange(width, length, width, dtype=float)
        edges.extend((block_start + local_edges).tolist())
        edges.append((block_index + 1) * length)
    output = np.asarray(edges, dtype=float)
    if np.any(np.diff(output) <= 0.0):
        raise ValueError("Repeated-block bin edges must be strictly increasing.")
    return output


def build_encoding_model_inputs(
    *,
    position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    spatial_bin_size_cm: float,
) -> dict[str, Any]:
    """Build the four legacy-equivalent model features from selected NWB inputs."""
    from v1ca1.spyglass.dpp import _pool_interval_sets
    from v1ca1.spyglass.path_specific_place import _intersect_intervals
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    expected_graphs = {*TRAJECTORY_TYPES, FULL_W_CONFIGURATION_NAME}
    actual_graphs = set(graph_inputs_by_configuration)
    if actual_graphs != expected_graphs:
        raise ValueError(
            "graph_inputs_by_configuration must contain the four path graphs and "
            f"full_w; missing={sorted(expected_graphs - actual_graphs)!r}, "
            f"extra={sorted(actual_graphs - expected_graphs)!r}."
        )
    place_bin_size = float(spatial_bin_size_cm)
    if not np.isfinite(place_bin_size) or place_bin_size <= 0.0:
        raise ValueError("spatial_bin_size_cm must be positive and finite.")

    support_by_trajectory = {
        trajectory_type: _intersect_intervals(
            trajectory_intervals_by_type[trajectory_type], movement_intervals
        )
        for trajectory_type in TRAJECTORY_TYPES
    }
    pooled_support = _pool_interval_sets(support_by_trajectory)
    progressions: dict[str, Any] = {}
    lengths: dict[str, float] = {}
    time_chunks: list[np.ndarray] = []
    model_chunks = {
        "path_specific_place": [],
        "dpp": [],
        "distance_to_reward": [],
    }
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        progression, graph_length_cm = build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals_by_type[trajectory_type],
            graph_inputs=graph_inputs_by_configuration[trajectory_type],
            trajectory_type=trajectory_type,
        )
        progressions[trajectory_type] = progression
        lengths[trajectory_type] = float(graph_length_cm)
        times, values = _restrict_feature(
            progression, support_by_trajectory[trajectory_type]
        )
        physical_values = values * float(graph_length_cm)
        if trajectory_type in _INBOUND_TRAJECTORIES:
            physical_values = float(graph_length_cm) - physical_values
        time_chunks.append(times)
        model_chunks["path_specific_place"].append(
            physical_values + trajectory_index * float(graph_length_cm)
        )
        model_chunks["dpp"].append(values + _DPP_OFFSETS[trajectory_type])
        model_chunks["distance_to_reward"].append(values)

    common_length = lengths[TRAJECTORY_TYPES[0]]
    if common_length <= 0.0 or any(
        not np.isclose(length, common_length, rtol=1e-10, atol=1e-12)
        for length in lengths.values()
    ):
        raise ValueError(
            "The four directional path graphs must have one common positive length."
        )
    if not time_chunks or not any(chunk.size for chunk in time_chunks):
        raise ValueError("Trajectory intervals contain no movement-supported samples.")
    times = np.concatenate(time_chunks)
    values_by_model = {
        model_name: np.concatenate(chunks)
        for model_name, chunks in model_chunks.items()
    }
    order = np.argsort(times, kind="stable")
    times = times[order]
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(
            "Pooled trajectory feature timestamps must be unique and increasing."
        )
    values_by_model = {
        model_name: values[order]
        for model_name, values in values_by_model.items()
    }

    full_w_progression, full_w_length = build_task_progression_from_graph(
        position=position,
        trajectory_interval=pooled_support,
        graph_inputs=graph_inputs_by_configuration[FULL_W_CONFIGURATION_NAME],
        trajectory_type=FULL_W_CONFIGURATION_NAME,
    )
    full_times, full_values = _restrict_feature(full_w_progression, pooled_support)
    values_by_model["absolute_place"] = _values_at_exact_times(
        full_times,
        full_values * float(full_w_length),
        times,
    )

    import pynapple as nap

    features = {
        model_name: nap.Tsd(
            t=times,
            d=values_by_model[model_name],
            time_support=pooled_support,
            time_units="s",
        )
        for model_name in MODEL_NAMES
    }
    normalized_step = place_bin_size / common_length
    bins = {
        "path_specific_place": _build_repeated_block_bin_edges(
            block_length=common_length,
            block_count=len(TRAJECTORY_TYPES),
            bin_width=place_bin_size,
        ),
        "absolute_place": np.arange(
            0.0,
            float(full_w_length) + place_bin_size,
            place_bin_size,
            dtype=float,
        ),
        "dpp": _build_repeated_block_bin_edges(
            block_length=1.0,
            block_count=2,
            bin_width=normalized_step,
        ),
        "distance_to_reward": np.arange(
            0.0,
            1.0 + normalized_step,
            normalized_step,
            dtype=float,
        ),
    }
    return {
        "features": features,
        "bins": bins,
        "smoothing_boundaries_by_model": {
            "path_specific_place": tuple(
                common_length * index
                for index in range(1, len(TRAJECTORY_TYPES))
            ),
            "absolute_place": (),
            "dpp": (1.0,),
            "distance_to_reward": (),
        },
        "pooled_support": pooled_support,
        "support_by_trajectory": support_by_trajectory,
        "common_path_length_cm": float(common_length),
        "full_w_length_cm": float(full_w_length),
    }


def build_strict_cross_validation_folds(
    *,
    trajectory_intervals_by_type: Mapping[str, Any],
    movement_intervals: Any,
    n_folds: int,
    random_seed: int,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Build independently shuffled folds with movement in every split."""
    from v1ca1.spyglass.dpp import _pool_interval_sets
    from v1ca1.task_progression.encoding_comparison import (
        build_single_trajectory_train_test_folds,
    )

    validate_trajectory_lap_counts(
        trajectory_intervals_by_type,
        n_folds=n_folds,
    )
    seed_sequence = np.random.SeedSequence(int(random_seed))
    child_sequences = seed_sequence.spawn(len(TRAJECTORY_TYPES))
    trajectory_folds: dict[str, tuple[dict[int, Any], dict[int, Any]]] = {}
    for trajectory_type, child_sequence in zip(
        TRAJECTORY_TYPES,
        child_sequences,
        strict=True,
    ):
        trajectory_seed = int(
            child_sequence.generate_state(1, dtype=np.uint32)[0]
        )
        trajectory_folds[trajectory_type] = (
            build_single_trajectory_train_test_folds(
                trajectory_intervals_by_type[trajectory_type],
                n_folds=n_folds,
                random_state=trajectory_seed,
            )
        )

    strict_train: dict[int, Any] = {}
    strict_test: dict[int, Any] = {}
    for fold in range(int(n_folds)):
        train = _pool_interval_sets(
            {
                trajectory_type: trajectory_folds[trajectory_type][0][fold]
                for trajectory_type in TRAJECTORY_TYPES
            }
        ).intersect(movement_intervals)
        test = _pool_interval_sets(
            {
                trajectory_type: trajectory_folds[trajectory_type][1][fold]
                for trajectory_type in TRAJECTORY_TYPES
            }
        ).intersect(movement_intervals)
        train_duration = float(train.tot_length())
        test_duration = float(test.tot_length())
        if (
            not np.isfinite(train_duration)
            or not np.isfinite(test_duration)
            or train_duration <= 0.0
            or test_duration <= 0.0
        ):
            raise ValueError(
                "Every cross-validation fold must have nonzero movement-supported "
                f"train and test duration; fold={fold}, train={train_duration}, "
                f"test={test_duration}."
            )
        strict_train[fold] = train
        strict_test[fold] = test
    return strict_train, strict_test


def _poisson_log_likelihood_nats(
    spike_counts: np.ndarray,
    firing_rates_hz: np.ndarray,
    *,
    bin_size_s: float,
) -> float:
    """Return total Poisson log likelihood including the factorial term."""
    from scipy.special import gammaln

    counts = np.asarray(spike_counts, dtype=np.int64).reshape(-1)
    rates = np.asarray(firing_rates_hz, dtype=float).reshape(-1)
    if counts.shape != rates.shape:
        raise ValueError("Spike counts and predicted rates must align.")
    if np.any(counts < 0) or not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        return np.nan
    means = rates * float(bin_size_s)
    value = np.sum(counts * np.log(means) - means - gammaln(counts + 1.0))
    return float(value) if np.isfinite(value) else np.nan


def _fold_feature_values(
    *,
    reference_counts: Any,
    features: Mapping[str, Any],
    test_fold: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Interpolate all models onto one held-out grid and return a common mask."""
    values: dict[str, np.ndarray] = {}
    expected_times = np.asarray(reference_counts.t, dtype=float).reshape(-1)
    for model_name in MODEL_NAMES:
        interpolated = features[model_name].interpolate(
            reference_counts,
            ep=test_fold,
        )
        times = np.asarray(interpolated.t, dtype=float).reshape(-1)
        model_values = np.asarray(interpolated.d, dtype=float).reshape(-1)
        if not np.array_equal(times, expected_times) or (
            model_values.shape != times.shape
        ):
            raise ValueError(
                "All encoding models must use identical held-out time bins."
            )
        values[model_name] = model_values
    finite_masks = {
        model_name: np.isfinite(values[model_name])
        for model_name in MODEL_NAMES
    }
    common_mask = finite_masks[MODEL_NAMES[0]]
    if any(
        not np.array_equal(common_mask, finite_masks[model_name])
        for model_name in MODEL_NAMES[1:]
    ):
        raise ValueError(
            "Encoding-model finite masks diverge on held-out time bins."
        )
    if not np.any(common_mask):
        raise ValueError("A held-out fold has no common finite model bins.")
    return values, common_mask


def _smooth_tuning_curve_in_blocks(
    tuning_curve: Any,
    *,
    pos_dim: str,
    sigma_bins: float,
    block_boundaries: Sequence[float],
) -> Any:
    """Smooth one tuning curve without crossing coordinate-block boundaries."""
    from v1ca1.task_progression.encoding_comparison import (
        smooth_pf_along_position_nan_aware,
    )

    boundaries = np.asarray(block_boundaries, dtype=float).reshape(-1)
    if boundaries.size == 0:
        return smooth_pf_along_position_nan_aware(
            tuning_curve,
            pos_dim=pos_dim,
            sigma_bins=sigma_bins,
        )
    if not np.all(np.isfinite(boundaries)) or np.any(
        np.diff(boundaries) <= 0.0
    ):
        raise ValueError(
            "Tuning-curve block boundaries must be finite and increasing."
        )

    coordinates = np.asarray(
        tuning_curve.coords[pos_dim], dtype=float
    ).reshape(-1)
    if not np.all(np.isfinite(coordinates)) or np.any(
        np.diff(coordinates) <= 0.0
    ):
        raise ValueError("Tuning-curve coordinates must be finite and increasing.")
    block_ids = np.searchsorted(boundaries, coordinates, side="right")
    smoothed_values = np.asarray(tuning_curve.values, dtype=float).copy()
    axis = tuning_curve.get_axis_num(pos_dim)
    for block_id in range(boundaries.size + 1):
        indices = np.flatnonzero(block_ids == block_id)
        if indices.size == 0:
            raise ValueError("Every tuning-curve coordinate block must contain bins.")
        if indices[-1] - indices[0] + 1 != indices.size:
            raise ValueError("Tuning-curve coordinate blocks must be contiguous.")
        block_slice = slice(int(indices[0]), int(indices[-1]) + 1)
        block_curve = tuning_curve.isel({pos_dim: block_slice})
        smoothed_block = smooth_pf_along_position_nan_aware(
            block_curve,
            pos_dim=pos_dim,
            sigma_bins=sigma_bins,
        )
        output_slice = [slice(None)] * smoothed_values.ndim
        output_slice[axis] = block_slice
        smoothed_values[tuple(output_slice)] = np.asarray(
            smoothed_block.values,
            dtype=float,
        )
    return tuning_curve.copy(data=smoothed_values)


def _raw_training_curves(
    *,
    spikes: Any,
    features: Mapping[str, Any],
    bins: Mapping[str, np.ndarray],
    smoothing_boundaries_by_model: Mapping[str, Sequence[float]],
    train_fold: Any,
    sigma_bins: float,
) -> dict[str, Any]:
    """Fit and smooth all model tuning curves within one training fold."""
    import pynapple as nap

    curves: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        raw = nap.compute_tuning_curves(
            data=spikes,
            features=features[model_name],
            bins=[np.asarray(bins[model_name], dtype=float)],
            epochs=train_fold,
            feature_names=[model_name],
        )
        curves[model_name] = (
            raw
            if float(sigma_bins) == 0.0
            else _smooth_tuning_curve_in_blocks(
                raw,
                pos_dim=model_name,
                sigma_bins=sigma_bins,
                block_boundaries=smoothing_boundaries_by_model[model_name],
            )
        )
    return curves


def _count_spikes_in_support(unit_spikes: Any, support: Any) -> int:
    """Count spikes in intervals without restricting an empty Pynapple array."""
    from v1ca1.spyglass.path_specific_place import _extract_interval_bounds

    timestamps = np.asarray(unit_spikes.t, dtype=float).reshape(-1)
    starts, ends = _extract_interval_bounds(support)
    if timestamps.size == 0 or starts.size == 0:
        return 0
    selected = np.zeros(timestamps.shape, dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        selected |= (timestamps >= float(start)) & (timestamps <= float(end))
    return int(np.sum(selected))


def _model_rates_for_bins(
    *,
    tuning_curve: Any,
    positions: np.ndarray,
    bin_edges: np.ndarray,
    fill_rate_hz: float,
) -> np.ndarray:
    """Return finite positive model rates for held-out feature values."""
    curve = np.asarray(getattr(tuning_curve, "values", tuning_curve), dtype=float)
    curve = curve.reshape(-1)
    edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    if curve.size != edges.size - 1:
        return np.full(np.asarray(positions).shape, np.nan, dtype=float)
    indices = np.digitize(np.asarray(positions, dtype=float), edges) - 1
    indices = np.clip(indices, 0, curve.size - 1)
    rates = curve[indices]
    rates = np.where(np.isfinite(rates), rates, float(fill_rate_hz))
    rates = np.maximum(rates, 1e-10)
    return rates


def _evaluate_encoding_models(
    *,
    spikes: Any,
    features: Mapping[str, Any],
    bins: Mapping[str, np.ndarray],
    smoothing_boundaries_by_model: Mapping[str, Sequence[float]],
    train_folds: Mapping[int, Any],
    test_folds: Mapping[int, Any],
    n_folds: int,
    evaluation_bin_size_s: float,
    gaussian_smoothing_sigma_bins: float,
) -> dict[Any, dict[str, Any]]:
    """Return strict common-mask fold totals for each eligible group unit."""
    unit_ids = list(spikes.keys())
    stores = {
        unit_id: {
            "heldout_spike_count": 0,
            "null_log_likelihood_nats": 0.0,
            "zero_training_spikes": False,
            "model_log_likelihood_nats": {
                model_name: 0.0 for model_name in MODEL_NAMES
            },
            "model_failed": {model_name: False for model_name in MODEL_NAMES},
        }
        for unit_id in unit_ids
    }
    if not unit_ids:
        return stores
    nonempty_unit_ids = [
        unit_id
        for unit_id in unit_ids
        if np.asarray(spikes[unit_id].t).size > 0
    ]
    if not nonempty_unit_ids:
        for store in stores.values():
            store["zero_training_spikes"] = True
            store["model_failed"] = {
                model_name: True for model_name in MODEL_NAMES
            }
        return stores
    reference_unit = nonempty_unit_ids[0]
    for fold in range(int(n_folds)):
        train_fold = train_folds[fold]
        test_fold = test_folds[fold]
        train_duration_s = float(train_fold.tot_length())
        reference_counts = spikes[reference_unit].count(
            evaluation_bin_size_s,
            test_fold,
        )
        values_by_model, common_mask = _fold_feature_values(
            reference_counts=reference_counts,
            features=features,
            test_fold=test_fold,
        )
        train_spike_counts = {
            unit_id: _count_spikes_in_support(spikes[unit_id], train_fold)
            for unit_id in unit_ids
        }
        active_unit_ids = [
            unit_id
            for unit_id in unit_ids
            if train_spike_counts[unit_id] > 0
        ]
        curves = (
            _raw_training_curves(
                spikes=spikes[active_unit_ids],
                features=features,
                bins=bins,
                smoothing_boundaries_by_model=smoothing_boundaries_by_model,
                train_fold=train_fold,
                sigma_bins=gaussian_smoothing_sigma_bins,
            )
            if active_unit_ids
            else {}
        )
        reference_times = np.asarray(reference_counts.t, dtype=float).reshape(-1)
        for unit_id in unit_ids:
            unit_spikes = spikes[unit_id]
            n_train_spikes = train_spike_counts[unit_id]
            if n_train_spikes <= 0:
                stores[unit_id]["zero_training_spikes"] = True
            train_rate_hz = max(n_train_spikes / train_duration_s, 1e-10)
            if np.asarray(unit_spikes.t).size == 0:
                counts = np.zeros(reference_times.shape, dtype=np.int64)
            else:
                binned = unit_spikes.count(evaluation_bin_size_s, test_fold)
                binned_times = np.asarray(binned.t, dtype=float).reshape(-1)
                counts = np.asarray(binned.d, dtype=np.int64).reshape(-1)
                if not np.array_equal(binned_times, reference_times):
                    raise ValueError(
                        "All eligible units must use identical held-out time bins."
                    )
            counts = counts[common_mask]
            stores[unit_id]["heldout_spike_count"] += int(counts.sum())
            if n_train_spikes <= 0:
                stores[unit_id]["model_failed"] = {
                    model_name: True for model_name in MODEL_NAMES
                }
                continue
            null_rates = np.full(counts.shape, train_rate_hz, dtype=float)
            null_ll = _poisson_log_likelihood_nats(
                counts,
                null_rates,
                bin_size_s=evaluation_bin_size_s,
            )
            if not np.isfinite(null_ll):
                stores[unit_id]["model_failed"] = {
                    model_name: True for model_name in MODEL_NAMES
                }
            else:
                stores[unit_id]["null_log_likelihood_nats"] += null_ll

            for model_name in MODEL_NAMES:
                try:
                    tuning_curve = curves[model_name].sel(unit=unit_id)
                except (KeyError, ValueError):
                    stores[unit_id]["model_failed"][model_name] = True
                    continue
                rates = _model_rates_for_bins(
                    tuning_curve=tuning_curve,
                    positions=values_by_model[model_name][common_mask],
                    bin_edges=bins[model_name],
                    fill_rate_hz=train_rate_hz,
                )
                model_ll = _poisson_log_likelihood_nats(
                    counts,
                    rates,
                    bin_size_s=evaluation_bin_size_s,
                )
                if not np.isfinite(model_ll):
                    stores[unit_id]["model_failed"][model_name] = True
                else:
                    stores[unit_id]["model_log_likelihood_nats"][
                        model_name
                    ] += model_ll
    return stores


def _unit_metric_row(
    *,
    store: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one strict fold store into canonical metrics and QC."""
    n_spikes = int(store["heldout_spike_count"])
    zero_train = bool(store["zero_training_spikes"])
    null_ll = float(store["null_log_likelihood_nats"])
    row: dict[str, Any] = {"heldout_spike_count": n_spikes}
    if zero_train:
        row["null_log_likelihood_nats"] = np.nan
        unit_status = "zero_training_spikes"
    elif n_spikes <= 0:
        row["null_log_likelihood_nats"] = np.nan
        unit_status = "zero_heldout_spikes"
    elif not np.isfinite(null_ll):
        row["null_log_likelihood_nats"] = np.nan
        unit_status = "no_valid_models"
    else:
        row["null_log_likelihood_nats"] = null_ll
        unit_status = "valid"

    valid_models: dict[str, bool] = {}
    for model_name in MODEL_NAMES:
        ll_column = f"{model_name}_log_likelihood_nats"
        info_column = f"{model_name}_information_bits_per_spike"
        status_column = f"{model_name}_qc_status"
        model_ll = float(store["model_log_likelihood_nats"][model_name])
        if zero_train:
            status = "zero_training_spikes"
        elif n_spikes <= 0:
            status = "zero_heldout_spikes"
        elif bool(store["model_failed"][model_name]) or not (
            np.isfinite(model_ll) and np.isfinite(null_ll)
        ):
            status = "nonfinite_likelihood"
        else:
            status = "valid"
        valid_models[model_name] = status == "valid"
        row[status_column] = status
        if status == "valid":
            row[ll_column] = model_ll
            row[info_column] = (model_ll - null_ll) / (np.log(2.0) * n_spikes)
        else:
            row[ll_column] = np.nan
            row[info_column] = np.nan

    contrast_targets = {
        "dpp_vs_path_specific_place_bits_per_spike": "path_specific_place",
        "dpp_vs_absolute_place_bits_per_spike": "absolute_place",
        "dpp_vs_distance_to_reward_bits_per_spike": "distance_to_reward",
    }
    for column, alternative in contrast_targets.items():
        if valid_models["dpp"] and valid_models[alternative]:
            row[column] = (
                row["dpp_log_likelihood_nats"]
                - row[f"{alternative}_log_likelihood_nats"]
            ) / (np.log(2.0) * n_spikes)
        else:
            row[column] = np.nan

    if unit_status == "valid" and not all(valid_models.values()):
        unit_status = (
            "partial_model_failure" if any(valid_models.values()) else "no_valid_models"
        )
    row["unit_valid"] = unit_status == "valid"
    row["qc_status"] = unit_status
    return row


def compute_selected_dpp_encoding(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
    dpp_encoding_id: Any,
    n_folds: int = 5,
    evaluation_bin_size_s: float = 0.05,
    spatial_bin_size_cm: float = 4.0,
    gaussian_smoothing_sigma_bins: float = 1.0,
    random_seed: int = 47,
) -> dict[str, Any]:
    """Fit and score four strict CV encoding models for eligible units."""
    parameters = _validate_analysis_parameters(
        n_folds=n_folds,
        evaluation_bin_size_s=evaluation_bin_size_s,
        spatial_bin_size_cm=spatial_bin_size_cm,
        gaussian_smoothing_sigma_bins=gaussian_smoothing_sigma_bins,
        random_seed=random_seed,
    )
    comparison_id = _uuid_component(
        dpp_encoding_id,
        name="dpp_encoding_id",
    )
    firing_rate_threshold, stability_threshold = _validate_thresholds(
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
    )
    eligibility = build_encoding_eligibility_table(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        movement_firing_rate_table=movement_firing_rate_table,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
    )
    eligible_mask = eligibility["eligible"].to_numpy(dtype=bool)
    if not np.any(eligible_mask):
        table = empty_dpp_encoding_table()
        return {"table": table, **summarize_dpp_encoding_table(table)}

    from v1ca1.task_progression.encoding_comparison import (
        select_spikes_by_unit_mask,
    )

    eligible_spikes = select_spikes_by_unit_mask(spikes, eligible_mask)
    eligible = eligibility.loc[eligible_mask].reset_index(drop=True)
    if list(eligible_spikes.keys()) != eligible["_group_key"].tolist():
        raise ValueError("Eligible TsGroup keys do not preserve selected-unit order.")
    model_inputs = build_encoding_model_inputs(
        position=position,
        trajectory_intervals_by_type=trajectory_intervals_by_type,
        graph_inputs_by_configuration=graph_inputs_by_configuration,
        movement_intervals=movement_intervals,
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
    )
    train_folds, test_folds = build_strict_cross_validation_folds(
        trajectory_intervals_by_type=trajectory_intervals_by_type,
        movement_intervals=movement_intervals,
        n_folds=parameters["n_folds"],
        random_seed=parameters["random_seed"],
    )
    stores = _evaluate_encoding_models(
        spikes=eligible_spikes,
        features=model_inputs["features"],
        bins=model_inputs["bins"],
        smoothing_boundaries_by_model=model_inputs[
            "smoothing_boundaries_by_model"
        ],
        train_folds=train_folds,
        test_folds=test_folds,
        n_folds=parameters["n_folds"],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
    )
    rows: list[dict[str, Any]] = []
    for _, unit in eligible.iterrows():
        group_unit_id = unit["_group_key"]
        rows.append(
            {
                **{
                    name: str(unit[name])
                    for name in IDENTITY_COLUMNS
                },
                "dpp_encoding_id": comparison_id,
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "epoch": str(epoch),
                "n_folds": int(parameters["n_folds"]),
                "evaluation_bin_size_s": float(
                    parameters["evaluation_bin_size_s"]
                ),
                "spatial_bin_size_cm": float(
                    parameters["spatial_bin_size_cm"]
                ),
                "gaussian_smoothing_sigma_bins": float(
                    parameters["gaussian_smoothing_sigma_bins"]
                ),
                "random_seed": int(parameters["random_seed"]),
                "minimum_movement_firing_rate_hz": firing_rate_threshold,
                "minimum_stability_correlation": stability_threshold,
                "movement_firing_rate_hz": float(
                    unit["movement_firing_rate_hz"]
                ),
                **{
                    column: float(unit[column])
                    for column in STABILITY_COLUMNS
                },
                **_unit_metric_row(store=stores[group_unit_id]),
            }
        )
    table = pd.DataFrame.from_records(rows).loc[:, list(TABLE_COLUMNS)]
    validate_dpp_encoding_table(table)
    return {"table": table, **summarize_dpp_encoding_table(table)}


def _validate_nonnegative_integer_column(
    table: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Return one validated non-negative integer column."""
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(f"{column} must contain non-negative integers.")
    return np.rint(values).astype(np.int64)


def validate_dpp_encoding_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and return one canonical eligible-unit comparison table."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Encoding-comparison artifact must be a pandas DataFrame.")
    missing = sorted(set(TABLE_COLUMNS).difference(table.columns))
    extra = sorted(set(table.columns).difference(TABLE_COLUMNS))
    if missing or extra:
        raise ValueError(
            "Encoding-comparison table must have the exact canonical schema; "
            f"missing={missing!r}, extra={extra!r}."
        )
    if table.empty:
        return table

    identity = table.loc[:, list(IDENTITY_COLUMNS)].astype(str)
    expected_stable = identity["spikesorting_merge_id"] + ":" + identity["unit_id"]
    if not np.array_equal(
        expected_stable.to_numpy(dtype=str),
        identity["stable_unit_id"].to_numpy(dtype=str),
    ):
        raise ValueError("Stable unit identities are inconsistent.")
    if identity["stable_unit_id"].duplicated().any() or identity[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError("Encoding-comparison identities must be one-to-one.")
    for field_name in (
        "dpp_encoding_id",
        "animal_name",
        "date",
        "region",
        "epoch",
    ):
        values = table[field_name].astype(str).unique().tolist()
        if len(values) != 1 or not values[0]:
            raise ValueError(f"{field_name} must contain one non-empty value.")
    _uuid_component(
        table["dpp_encoding_id"].iloc[0],
        name="dpp_encoding_id",
    )
    n_folds_values = _validate_nonnegative_integer_column(table, "n_folds")
    random_seed_values = _validate_nonnegative_integer_column(table, "random_seed")
    if np.any(n_folds_values < 2):
        raise ValueError("n_folds must be at least 2.")
    for field_name in (
        "n_folds",
        "evaluation_bin_size_s",
        "spatial_bin_size_cm",
        "gaussian_smoothing_sigma_bins",
        "random_seed",
        "minimum_movement_firing_rate_hz",
        "minimum_stability_correlation",
    ):
        if table[field_name].nunique(dropna=False) != 1:
            raise ValueError(f"{field_name} must be constant within one artifact.")
    _validate_analysis_parameters(
        n_folds=int(n_folds_values[0]),
        evaluation_bin_size_s=float(table["evaluation_bin_size_s"].iloc[0]),
        spatial_bin_size_cm=float(table["spatial_bin_size_cm"].iloc[0]),
        gaussian_smoothing_sigma_bins=float(
            table["gaussian_smoothing_sigma_bins"].iloc[0]
        ),
        random_seed=int(random_seed_values[0]),
    )
    _validate_thresholds(
        minimum_movement_firing_rate_hz=float(
            table["minimum_movement_firing_rate_hz"].iloc[0]
        ),
        minimum_stability_correlation=float(
            table["minimum_stability_correlation"].iloc[0]
        ),
    )

    rates = pd.to_numeric(
        table["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("Eligible-unit movement firing rates must be finite.")
    firing_rate_threshold = float(
        table["minimum_movement_firing_rate_hz"].iloc[0]
    )
    if np.any(rates < firing_rate_threshold):
        raise ValueError(
            "Every artifact row must meet the movement firing-rate threshold."
        )
    stability_arrays: list[np.ndarray] = []
    for column in STABILITY_COLUMNS:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if np.any(np.isinf(values)) or np.any(
            (values[finite] < -1.0 - 1e-9) | (values[finite] > 1.0 + 1e-9)
        ):
            raise ValueError(f"{column} must be within [-1, 1] or NaN.")
        stability_arrays.append(values)
    stability_threshold = float(
        table["minimum_stability_correlation"].iloc[0]
    )
    stability_matrix = np.column_stack(stability_arrays)
    if np.any(
        ~np.any(
            np.isfinite(stability_matrix)
            & (stability_matrix >= stability_threshold),
            axis=1,
        )
    ):
        raise ValueError(
            "Every artifact row must meet the any-trajectory stability threshold."
        )

    spike_counts = _validate_nonnegative_integer_column(
        table, "heldout_spike_count"
    )
    null_values = pd.to_numeric(
        table["null_log_likelihood_nats"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(np.isinf(null_values)):
        raise ValueError("Null log likelihood may be finite or NaN, not infinite.")

    model_valid: dict[str, np.ndarray] = {}
    model_ll: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        status = table[f"{model_name}_qc_status"].astype(str).to_numpy()
        if not set(status).issubset(MODEL_QC_STATUSES):
            raise ValueError(f"Unsupported {model_name} QC status.")
        ll_values = pd.to_numeric(
            table[f"{model_name}_log_likelihood_nats"], errors="coerce"
        ).to_numpy(dtype=float)
        info_values = pd.to_numeric(
            table[f"{model_name}_information_bits_per_spike"], errors="coerce"
        ).to_numpy(dtype=float)
        if np.any(np.isinf(ll_values)) or np.any(np.isinf(info_values)):
            raise ValueError(f"{model_name} metrics may not be infinite.")
        valid = status == "valid"
        if not np.array_equal(
            valid,
            np.isfinite(ll_values) & np.isfinite(info_values),
        ):
            raise ValueError(
                f"Finite {model_name} metrics must correspond exactly to valid QC."
            )
        if np.any(valid & ((spike_counts <= 0) | ~np.isfinite(null_values))):
            raise ValueError(
                f"Valid {model_name} metrics require spikes and finite null LL."
            )
        expected_info = np.full(len(table), np.nan, dtype=float)
        expected_info[valid] = (
            ll_values[valid] - null_values[valid]
        ) / (np.log(2.0) * spike_counts[valid])
        if not np.allclose(
            info_values,
            expected_info,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"{model_name} information values are inconsistent.")
        model_valid[model_name] = valid
        model_ll[model_name] = ll_values

    contrast_targets = {
        "dpp_vs_path_specific_place_bits_per_spike": "path_specific_place",
        "dpp_vs_absolute_place_bits_per_spike": "absolute_place",
        "dpp_vs_distance_to_reward_bits_per_spike": "distance_to_reward",
    }
    for column, alternative in contrast_targets.items():
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        expected_valid = model_valid["dpp"] & model_valid[alternative]
        if np.any(np.isinf(values)) or not np.array_equal(
            np.isfinite(values), expected_valid
        ):
            raise ValueError(f"{column} finite-value QC is inconsistent.")
        expected = np.full(len(table), np.nan, dtype=float)
        expected[expected_valid] = (
            model_ll["dpp"][expected_valid]
            - model_ll[alternative][expected_valid]
        ) / (np.log(2.0) * spike_counts[expected_valid])
        if not np.allclose(
            values, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        ):
            raise ValueError(f"{column} values are inconsistent.")

    all_models_valid = np.logical_and.reduce(list(model_valid.values()))
    unit_valid = table["unit_valid"].to_numpy()
    if unit_valid.dtype != bool and not all(
        isinstance(value, (bool, np.bool_)) for value in unit_valid
    ):
        raise ValueError("unit_valid must contain booleans.")
    unit_valid = unit_valid.astype(bool)
    if not np.array_equal(unit_valid, all_models_valid):
        raise ValueError("unit_valid must mean that all four models are valid.")
    statuses = table["qc_status"].astype(str).to_numpy()
    if not set(statuses).issubset(UNIT_QC_STATUSES):
        raise ValueError("Unsupported unit qc_status.")
    if not np.array_equal(statuses == "valid", unit_valid):
        raise ValueError("Valid unit QC must correspond exactly to unit_valid.")
    valid_model_count = np.sum(np.column_stack(list(model_valid.values())), axis=1)
    model_status_matrix = np.column_stack(
        [
            table[f"{model_name}_qc_status"].astype(str).to_numpy()
            for model_name in MODEL_NAMES
        ]
    )
    for row_index, status in enumerate(statuses):
        if status == "zero_training_spikes" and not np.all(
            model_status_matrix[row_index] == "zero_training_spikes"
        ):
            raise ValueError(
                "zero_training_spikes unit QC requires the same model QC."
            )
        if np.any(
            model_status_matrix[row_index] == "zero_training_spikes"
        ) and status != "zero_training_spikes":
            raise ValueError(
                "zero_training_spikes model QC requires matching unit QC."
            )
        if status == "zero_training_spikes" and np.isfinite(
            null_values[row_index]
        ):
            raise ValueError("zero_training_spikes requires a NaN null LL.")
        if status == "zero_heldout_spikes" and not (
            spike_counts[row_index] == 0
            and np.all(
                model_status_matrix[row_index] == "zero_heldout_spikes"
            )
        ):
            raise ValueError(
                "zero_heldout_spikes unit QC requires zero spikes and "
                "matching model QC."
            )
        if np.any(
            model_status_matrix[row_index] == "zero_heldout_spikes"
        ) and status != "zero_heldout_spikes":
            raise ValueError(
                "zero_heldout_spikes model QC requires matching unit QC."
            )
        if status == "zero_heldout_spikes" and np.isfinite(
            null_values[row_index]
        ):
            raise ValueError("zero_heldout_spikes requires a NaN null LL.")
        if status == "partial_model_failure" and not (
            0 < valid_model_count[row_index] < len(MODEL_NAMES)
        ):
            raise ValueError(
                "partial_model_failure requires some but not all valid models."
            )
        if status == "no_valid_models" and valid_model_count[row_index] != 0:
            raise ValueError("no_valid_models requires every model to be invalid.")
        if status == "no_valid_models" and not np.all(
            model_status_matrix[row_index] == "nonfinite_likelihood"
        ):
            raise ValueError(
                "no_valid_models requires nonfinite_likelihood model QC."
            )
    return table


def summarize_dpp_encoding_table(table: pd.DataFrame) -> dict[str, Any]:
    """Return result-level counts and status for one canonical table."""
    validate_dpp_encoding_table(table)
    from v1ca1.spyglass.selection import unit_identity_sha256

    n_units_eligible = int(len(table))
    n_units_valid = int(table["unit_valid"].sum()) if n_units_eligible else 0
    if n_units_eligible == 0:
        status = "no_eligible_units"
    elif n_units_valid:
        status = "valid"
    else:
        status = "no_valid_units"
    identities = table.loc[
        :, ["spikesorting_merge_id", "unit_id"]
    ].to_dict("records")
    return {
        "analysis_status": status,
        "n_units_eligible": n_units_eligible,
        "n_units_valid": n_units_valid,
        "eligible_units_sha256": unit_identity_sha256(identities),
    }


def load_dpp_encoding_artifact(path: Path) -> pd.DataFrame:
    """Load and validate one canonical encoding-comparison Parquet."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Encoding-comparison artifact not found: {path}")
    return validate_dpp_encoding_table(pd.read_parquet(path))


def write_dpp_encoding_artifact(
    table: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated Parquet without implicit overwrite."""
    validate_dpp_encoding_table(table)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite encoding-comparison artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.backup")
    had_existing = path.exists()
    try:
        table.to_parquet(temporary, index=False)
        load_dpp_encoding_artifact(temporary)
        if had_existing:
            os.replace(path, backup)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        if backup.exists():
            path.unlink(missing_ok=True)
            os.replace(backup, path)
        raise
    else:
        backup.unlink(missing_ok=True)
    return path


def _legacy_unit_column(table: pd.DataFrame) -> pd.Series:
    """Return legacy ephemeral unit identifiers from a column or named index."""
    if "unit" in table.columns:
        units = table["unit"]
    elif table.index.name == "unit":
        units = pd.Series(table.index, index=table.index)
    else:
        raise ValueError("Legacy encoding table must have a 'unit' column or index.")
    if units.duplicated().any():
        raise ValueError("Legacy encoding unit identifiers must be unique.")
    return units


def _resolve_legacy_identity(
    legacy_unit: Any,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]],
) -> str:
    """Return one persistent stable ID for a legacy ephemeral unit."""
    candidates = [legacy_unit]
    string_value = str(legacy_unit)
    if string_value != legacy_unit:
        candidates.append(string_value)
    resolved = [
        unit_identity_resolver[candidate]
        for candidate in candidates
        if candidate in unit_identity_resolver
    ]
    if not resolved:
        raise ValueError(f"No persistent identity for legacy unit {legacy_unit!r}.")
    stable_ids = {
        f"{str(value['spikesorting_merge_id'])}:{str(value['unit_id'])}"
        for value in resolved
        if "spikesorting_merge_id" in value and "unit_id" in value
    }
    if len(stable_ids) != 1:
        raise ValueError(f"Ambiguous persistent identity for {legacy_unit!r}.")
    return next(iter(stable_ids))


def normalize_legacy_dpp_encoding_table(
    legacy_table: pd.DataFrame,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]],
    dpp_encoding_id: Any,
    n_folds: int = 5,
    evaluation_bin_size_s: float = 0.05,
    spatial_bin_size_cm: float = 4.0,
    gaussian_smoothing_sigma_bins: float = 1.0,
    random_seed: int = 47,
) -> pd.DataFrame:
    """Normalize one exact-eligible-set legacy wide summary into canonical form."""
    if not isinstance(legacy_table, pd.DataFrame):
        raise TypeError("legacy_table must be a pandas DataFrame.")
    missing = sorted(set(LEGACY_METRIC_COLUMNS).difference(legacy_table.columns))
    if missing:
        raise ValueError(f"Legacy encoding table is missing columns {missing!r}.")
    parameters = _validate_analysis_parameters(
        n_folds=n_folds,
        evaluation_bin_size_s=evaluation_bin_size_s,
        spatial_bin_size_cm=spatial_bin_size_cm,
        gaussian_smoothing_sigma_bins=gaussian_smoothing_sigma_bins,
        random_seed=random_seed,
    )
    comparison_id = _uuid_component(
        dpp_encoding_id,
        name="dpp_encoding_id",
    )
    firing_rate_threshold, stability_threshold = _validate_thresholds(
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
    )
    eligibility = build_encoding_eligibility_table(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        movement_firing_rate_table=movement_firing_rate_table,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
    )
    eligible = eligibility.loc[eligibility["eligible"]].reset_index(drop=True)
    legacy_units = _legacy_unit_column(legacy_table)
    stable_ids = [
        _resolve_legacy_identity(unit, unit_identity_resolver)
        for unit in legacy_units.tolist()
    ]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("Legacy units do not resolve one-to-one.")
    expected_ids = eligible["stable_unit_id"].astype(str).tolist()
    if set(stable_ids) != set(expected_ids) or len(stable_ids) != len(expected_ids):
        raise ValueError(
            "Legacy encoding units do not exactly match the eligible upstream set."
        )
    source = legacy_table.copy()
    source["_stable_unit_id"] = stable_ids
    source = source.set_index("_stable_unit_id", drop=False).loc[expected_ids]

    rows: list[dict[str, Any]] = []
    for _, unit in eligible.iterrows():
        stable_id = str(unit["stable_unit_id"])
        legacy = source.loc[stable_id]
        n_spikes_value = float(legacy["n_spikes"])
        if (
            not np.isfinite(n_spikes_value)
            or n_spikes_value < 0.0
            or not np.isclose(n_spikes_value, round(n_spikes_value))
        ):
            raise ValueError("Legacy n_spikes must contain non-negative integers.")
        n_spikes = int(round(n_spikes_value))
        per_spike_null = float(legacy["ll_null"])
        store = {
            "heldout_spike_count": n_spikes,
            "null_log_likelihood_nats": (
                per_spike_null * n_spikes
                if n_spikes > 0 and np.isfinite(per_spike_null)
                else np.nan
            ),
            "zero_training_spikes": False,
            "model_log_likelihood_nats": {},
            "model_failed": {},
        }
        for model_name, (ll_column, info_column) in LEGACY_MODEL_COLUMNS.items():
            per_spike_ll = float(legacy[ll_column])
            valid = n_spikes > 0 and np.isfinite(
                store["null_log_likelihood_nats"]
            ) and np.isfinite(per_spike_ll)
            store["model_log_likelihood_nats"][model_name] = (
                per_spike_ll * n_spikes if valid else np.nan
            )
            store["model_failed"][model_name] = not valid
            legacy_info = float(legacy[info_column])
            expected_info = (
                (per_spike_ll - per_spike_null) / np.log(2.0)
                if valid
                else np.nan
            )
            if not np.isclose(
                legacy_info,
                expected_info,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"Legacy {info_column} is internally inconsistent.")

        legacy_delta_checks = {
            "delta_bits_place_vs_tp": ("ll_place", "ll_tp"),
            "delta_bits_generalized_place_vs_tp": (
                "ll_generalized_place",
                "ll_tp",
            ),
            "delta_bits_gtp_vs_tp": ("ll_gtp", "ll_tp"),
        }
        for delta_column, (first_ll, second_ll) in legacy_delta_checks.items():
            first = float(legacy[first_ll])
            second = float(legacy[second_ll])
            expected_delta = (
                (first - second) / np.log(2.0)
                if np.isfinite(first) and np.isfinite(second)
                else np.nan
            )
            if not np.isclose(
                float(legacy[delta_column]),
                expected_delta,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"Legacy {delta_column} is internally inconsistent.")

        rows.append(
            {
                **{name: str(unit[name]) for name in IDENTITY_COLUMNS},
                "dpp_encoding_id": comparison_id,
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "epoch": str(epoch),
                "n_folds": int(parameters["n_folds"]),
                "evaluation_bin_size_s": float(
                    parameters["evaluation_bin_size_s"]
                ),
                "spatial_bin_size_cm": float(
                    parameters["spatial_bin_size_cm"]
                ),
                "gaussian_smoothing_sigma_bins": float(
                    parameters["gaussian_smoothing_sigma_bins"]
                ),
                "random_seed": int(parameters["random_seed"]),
                "minimum_movement_firing_rate_hz": firing_rate_threshold,
                "minimum_stability_correlation": stability_threshold,
                "movement_firing_rate_hz": float(
                    unit["movement_firing_rate_hz"]
                ),
                **{
                    column: float(unit[column])
                    for column in STABILITY_COLUMNS
                },
                **_unit_metric_row(store=store),
            }
        )
    if not rows:
        return empty_dpp_encoding_table()
    table = pd.DataFrame.from_records(rows).loc[:, list(TABLE_COLUMNS)]
    return validate_dpp_encoding_table(table)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_existing_dpp_encoding_artifact(
    source_path: Path,
    destination_path: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]],
    dpp_encoding_id: Any,
    n_folds: int = 5,
    evaluation_bin_size_s: float = 0.05,
    spatial_bin_size_cm: float = 4.0,
    gaussian_smoothing_sigma_bins: float = 1.0,
    random_seed: int = 47,
    source_v1ca1_git_commit: str | None = None,
) -> dict[str, Any]:
    """Normalize and register one exact-eligible legacy summary Parquet."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy encoding artifact not found: {source}")
    legacy = pd.read_parquet(source)
    table = normalize_legacy_dpp_encoding_table(
        legacy,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        movement_firing_rate_table=movement_firing_rate_table,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
        unit_identity_resolver=unit_identity_resolver,
        dpp_encoding_id=dpp_encoding_id,
        n_folds=n_folds,
        evaluation_bin_size_s=evaluation_bin_size_s,
        spatial_bin_size_cm=spatial_bin_size_cm,
        gaussian_smoothing_sigma_bins=gaussian_smoothing_sigma_bins,
        random_seed=random_seed,
    )
    written = write_dpp_encoding_artifact(table, destination_path)
    return {
        "table": table,
        "path": written,
        **summarize_dpp_encoding_table(table),
        "legacy_artifact_provenance": {
            "source_path": str(source.resolve()),
            "source_sha256": _file_sha256(source),
            "source_v1ca1_git_commit": source_v1ca1_git_commit,
            "legacy_log_likelihood_units": "nats_per_spike",
            "canonical_log_likelihood_units": "total_nats",
            "eligible_unit_set_validated": True,
        },
        "_created_artifact_paths": [str(written)],
    }


__all__ = [
    "ARTIFACT_DIRNAME",
    "ARTIFACT_FILENAME",
    "CONTRAST_COLUMNS",
    "DEFAULT_ARTIFACT_ROOT",
    "FULL_W_CONFIGURATION_NAME",
    "IDENTITY_COLUMNS",
    "LEGACY_METRIC_COLUMNS",
    "MODEL_NAMES",
    "MODEL_QC_STATUSES",
    "STABILITY_COLUMNS",
    "TABLE_COLUMNS",
    "UNIT_QC_STATUSES",
    "build_encoding_eligibility_table",
    "build_encoding_model_inputs",
    "build_strict_cross_validation_folds",
    "compute_selected_dpp_encoding",
    "empty_dpp_encoding_table",
    "get_dpp_encoding_artifact_path",
    "load_dpp_encoding_artifact",
    "normalize_legacy_dpp_encoding_table",
    "register_existing_dpp_encoding_artifact",
    "summarize_dpp_encoding_table",
    "validate_dpp_encoding_table",
    "validate_trajectory_lap_counts",
    "write_dpp_encoding_artifact",
]
