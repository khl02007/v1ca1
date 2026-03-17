from __future__ import annotations

"""Shared helpers for ripple-time decoding workflows."""

from typing import Any

import numpy as np
import pandas as pd

from v1ca1.task_progression._session import (
    build_combined_task_progression_bins,
    build_linear_position_bins,
)


REPRESENTATIONS = ("place", "task_progression")
FEATURE_NAME_BY_REPRESENTATION = {
    "place": "linpos",
    "task_progression": "tp",
}


def select_decode_epochs(
    run_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return the run epochs to decode."""
    if not requested_epochs:
        return list(run_epochs)

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            "Decode epochs must be run epochs because tuning curves are built from saved "
            f"run epochs. Missing run epochs: {missing_epochs!r}"
        )
    return list(requested_epochs)


def validate_train_epoch(
    run_epochs: list[str],
    train_epoch: str | None,
    *,
    flag_name: str = "--train-epoch",
) -> str | None:
    """Validate an optional shared training epoch."""
    if train_epoch is None:
        return None
    if train_epoch not in run_epochs:
        raise ValueError(
            f"{flag_name} must be one of the saved run epochs {run_epochs!r}. "
            f"Got {train_epoch!r}."
        )
    return str(train_epoch)


def get_representation_inputs(
    session: dict[str, Any],
    *,
    animal_name: str,
    representation: str,
) -> tuple[dict[str, Any], np.ndarray, str]:
    """Return the requested feature Tsds, bins, and feature name."""
    if representation == "place":
        return (
            session["linear_position_by_run"],
            build_linear_position_bins(animal_name),
            FEATURE_NAME_BY_REPRESENTATION[representation],
        )
    if representation == "task_progression":
        return (
            session["task_progression_by_run"],
            build_combined_task_progression_bins(animal_name),
            FEATURE_NAME_BY_REPRESENTATION[representation],
        )
    raise ValueError(f"Unsupported representation {representation!r}.")


def get_tsgroup_unit_ids(tsgroup: Any) -> np.ndarray:
    """Return unit ids from a pynapple TsGroup-like object."""
    if hasattr(tsgroup, "get_unit_ids"):
        return np.asarray(tsgroup.get_unit_ids())
    if hasattr(tsgroup, "keys"):
        return np.asarray(list(tsgroup.keys()))
    raise ValueError("Could not extract unit ids from the provided TsGroup-like object.")


def build_region_unit_mask_table(
    *,
    unit_ids: np.ndarray,
    movement_firing_rates_hz: np.ndarray,
    min_movement_fr_hz: float,
    region: str,
) -> pd.DataFrame:
    """Return the movement-rate mask table for one region."""
    unit_ids = np.asarray(unit_ids)
    movement_firing_rates_hz = np.asarray(movement_firing_rates_hz, dtype=float)
    if movement_firing_rates_hz.shape[0] != unit_ids.size:
        raise ValueError(
            f"{region} movement firing rates do not match the saved unit count."
        )

    keep_unit = movement_firing_rates_hz >= float(min_movement_fr_hz)
    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "movement_firing_rate_hz": movement_firing_rates_hz,
            "passes_movement_firing_rate": keep_unit,
            "keep_unit": keep_unit,
        }
    )


def empty_ripple_table() -> pd.DataFrame:
    """Return an empty ripple table."""
    return pd.DataFrame(
        {
            "start_time": pd.Series(dtype=float),
            "end_time": pd.Series(dtype=float),
        }
    )


def compute_tuning_curves_for_epoch(
    *,
    spikes: Any,
    feature: Any,
    movement_interval: Any,
    bin_edges: np.ndarray,
    feature_name: str,
) -> Any:
    """Compute one epoch's tuning curves with explicit feature naming."""
    import pynapple as nap

    return nap.compute_tuning_curves(
        data=spikes,
        features=feature,
        bins=[np.asarray(bin_edges, dtype=float)],
        epochs=movement_interval,
        feature_names=[feature_name],
    )


def make_intervalset_from_bounds(
    starts: np.ndarray,
    ends: np.ndarray,
) -> Any:
    """Create one IntervalSet from aligned start and end arrays."""
    import pynapple as nap

    return nap.IntervalSet(
        start=np.asarray(starts, dtype=float),
        end=np.asarray(ends, dtype=float),
        time_units="s",
    )


def make_empty_tsd(time_support: Any | None = None) -> Any:
    """Return an empty second-based Tsd."""
    import pynapple as nap

    kwargs: dict[str, Any] = {"time_units": "s"}
    if time_support is not None:
        kwargs["time_support"] = time_support
    return nap.Tsd(
        t=np.array([], dtype=float),
        d=np.array([], dtype=float),
        **kwargs,
    )


def extract_time_values(tsd_like: Any) -> np.ndarray:
    """Return time values from a pynapple time series or frame."""
    if hasattr(tsd_like, "t"):
        return np.asarray(tsd_like.t, dtype=float)
    if hasattr(tsd_like, "index"):
        index = getattr(tsd_like, "index")
        if hasattr(index, "values"):
            return np.asarray(index.values, dtype=float)
    raise ValueError("Could not extract time values from the provided pynapple object.")


def concatenate_tsds(tsds: list[Any], time_support: Any) -> Any:
    """Concatenate decoded Tsds into one sorted Tsd."""
    import pynapple as nap

    if not tsds:
        return make_empty_tsd(time_support=time_support)

    times = [np.asarray(tsd.t, dtype=float) for tsd in tsds if len(np.asarray(tsd.t)) > 0]
    values = [np.asarray(tsd.d, dtype=float) for tsd in tsds if len(np.asarray(tsd.t)) > 0]
    if not times:
        return make_empty_tsd(time_support=time_support)

    all_times = np.concatenate(times)
    all_values = np.concatenate(values)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=time_support,
        time_units="s",
    )


def interpolate_nans_1d(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in one 1D array."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()

    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, np.nan, dtype=float)
    if np.all(finite):
        return values.copy()

    indices = np.arange(values.size, dtype=float)
    output = values.copy()
    output[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return output


def deranged_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deranged permutation when possible."""
    if size <= 1:
        return np.arange(size, dtype=int)
    for _ in range(128):
        permutation = rng.permutation(size)
        if not np.any(permutation == np.arange(size, dtype=int)):
            return permutation
    return np.roll(np.arange(size, dtype=int), 1)


def assemble_decoded_ripple_epoch_data(
    *,
    spikes: Any,
    tuning_curves: Any,
    ripple_table: pd.DataFrame,
    epoch_interval: Any,
    bin_size_s: float,
) -> dict[str, Any]:
    """Decode one region on ripple bins and return ripple-aligned decoded states."""
    import pynapple as nap

    decoded_chunks: list[Any] = []
    ripple_starts: list[float] = []
    ripple_ends: list[float] = []
    ripple_source_indices: list[int] = []
    decoded_state_chunks: list[np.ndarray] = []
    bin_time_chunks: list[np.ndarray] = []
    ripple_id_chunks: list[np.ndarray] = []
    skipped_ripples: list[dict[str, Any]] = []
    kept_ripple_count = 0

    for ripple_row_index, ripple_row in ripple_table.reset_index(drop=True).iterrows():
        ripple_ep = make_intervalset_from_bounds(
            np.array([float(ripple_row["start_time"])], dtype=float),
            np.array([float(ripple_row["end_time"])], dtype=float),
        ).intersect(epoch_interval)
        if float(ripple_ep.tot_length()) <= 0.0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "No overlap with decode epoch interval.",
                }
            )
            continue

        decoded, _ = nap.decode_bayes(
            tuning_curves=tuning_curves,
            data=spikes,
            epochs=ripple_ep,
            sliding_window_size=None,
            bin_size=float(bin_size_s),
        )
        decoded_times = extract_time_values(decoded).reshape(-1)
        if decoded_times.size == 0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "Decoding returned no time bins.",
                }
            )
            continue

        decoded_state = interpolate_nans_1d(np.asarray(decoded.d, dtype=float).reshape(-1))
        if decoded_state.shape[0] != decoded_times.size:
            raise ValueError("Decoded state values do not match decoded timestamps.")
        if not np.any(np.isfinite(decoded_state)):
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "Decoded state was non-finite for all ripple bins.",
                }
            )
            continue

        decoded_chunks.append(decoded)
        decoded_state_chunks.append(decoded_state)
        bin_time_chunks.append(decoded_times)
        ripple_id_chunks.append(np.full(decoded_state.shape[0], kept_ripple_count, dtype=int))
        ripple_starts.append(float(np.asarray(ripple_ep.start, dtype=float)[0]))
        ripple_ends.append(float(np.asarray(ripple_ep.end, dtype=float)[0]))
        ripple_source_indices.append(int(ripple_row_index))
        kept_ripple_count += 1

    ripple_support = make_intervalset_from_bounds(
        np.asarray(ripple_starts, dtype=float),
        np.asarray(ripple_ends, dtype=float),
    )
    decoded_tsd = concatenate_tsds(decoded_chunks, ripple_support)
    if not decoded_state_chunks:
        return {
            "decoded_tsd": decoded_tsd,
            "decoded_state": np.array([], dtype=float),
            "bin_times_s": np.array([], dtype=float),
            "ripple_ids": np.array([], dtype=int),
            "n_ripples_kept": 0,
            "n_bins": 0,
            "ripple_start_times_s": np.array([], dtype=float),
            "ripple_end_times_s": np.array([], dtype=float),
            "ripple_source_indices": np.array([], dtype=int),
            "skipped_ripples": skipped_ripples,
        }

    return {
        "decoded_tsd": decoded_tsd,
        "decoded_state": np.concatenate(decoded_state_chunks).astype(float, copy=False),
        "bin_times_s": np.concatenate(bin_time_chunks).astype(float, copy=False),
        "ripple_ids": np.concatenate(ripple_id_chunks).astype(int, copy=False),
        "n_ripples_kept": int(kept_ripple_count),
        "n_bins": int(sum(chunk.shape[0] for chunk in decoded_state_chunks)),
        "ripple_start_times_s": np.asarray(ripple_starts, dtype=float),
        "ripple_end_times_s": np.asarray(ripple_ends, dtype=float),
        "ripple_source_indices": np.asarray(ripple_source_indices, dtype=int),
        "skipped_ripples": skipped_ripples,
    }
