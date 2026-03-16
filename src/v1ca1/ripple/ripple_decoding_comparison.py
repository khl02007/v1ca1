from __future__ import annotations

"""Compare CA1 and V1 ripple decoding content for one session."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS, load_ephys_timestamps_by_epoch
from v1ca1.ripple._decoding import (
    REPRESENTATIONS,
    assemble_decoded_ripple_epoch_data,
    compute_tuning_curves_for_epoch,
    deranged_permutation,
    empty_ripple_table,
    get_representation_inputs,
    select_decode_epochs,
    validate_train_epoch,
)
from v1ca1.ripple.ripple_glm import build_epoch_intervals, load_ripple_tables
from v1ca1.task_progression._session import (
    compute_movement_firing_rates,
    get_analysis_path,
    prepare_task_progression_session,
)


DEFAULT_BIN_SIZE_S = 0.002
DEFAULT_N_SHUFFLES = 100
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_CA1_MIN_MOVEMENT_FR_HZ = 0.5
DEFAULT_V1_MIN_MOVEMENT_FR_HZ = 0.5

METRIC_LABELS = {
    "pearson_r": "Pearson r",
    "mean_abs_difference": "Mean Absolute Difference",
    "mean_abs_difference_normalized": "Normalized Mean Absolute Difference",
    "mean_signed_difference": "Mean Signed Difference",
    "start_difference": "Start Difference",
    "end_difference": "End Difference",
}
METRIC_DIRECTION = {
    "pearson_r": "higher",
    "mean_abs_difference": "lower",
    "mean_abs_difference_normalized": "lower",
    "mean_signed_difference": "zero",
    "start_difference": "zero",
    "end_difference": "zero",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for ripple decoding comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode CA1 and V1 separately during ripples and compare their decoded content."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--representation",
        required=True,
        choices=REPRESENTATIONS,
        help="Decoded state representation: place or task progression",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument(
        "--decode-epoch",
        help=(
            "Optional single run epoch to decode. If --train-epoch is omitted, "
            "this epoch is also used for tuning."
        ),
    )
    decode_group.add_argument(
        "--decode-epochs",
        nargs="+",
        help=(
            "Optional run epochs to decode. When this plural form is used with "
            "--train-epoch, all listed decode epochs share that training epoch."
        ),
    )
    parser.add_argument(
        "--train-epoch",
        help=(
            "Optional run epoch used to build both CA1 and V1 tuning curves. "
            "If --decode-epoch is omitted, this epoch is also used as the decode epoch. "
            "If neither epoch flag is provided, the script iterates over all run epochs "
            "with train_epoch == decode_epoch."
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Time bin size in seconds for ripple decoding. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--ca1-min-movement-fr-hz",
        type=float,
        default=DEFAULT_CA1_MIN_MOVEMENT_FR_HZ,
        help=(
            "Minimum CA1 firing rate during movement in the training epoch. "
            f"Default: {DEFAULT_CA1_MIN_MOVEMENT_FR_HZ}"
        ),
    )
    parser.add_argument(
        "--v1-min-movement-fr-hz",
        type=float,
        default=DEFAULT_V1_MIN_MOVEMENT_FR_HZ,
        help=(
            "Minimum V1 firing rate during movement in the training epoch. "
            f"Default: {DEFAULT_V1_MIN_MOVEMENT_FR_HZ}"
        ),
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help=f"Number of shuffle refits per epoch summary. Default: {DEFAULT_N_SHUFFLES}",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for ripple-trajectory shuffles. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI argument ranges."""
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.ca1_min_movement_fr_hz < 0:
        raise ValueError("--ca1-min-movement-fr-hz must be non-negative.")
    if args.v1_min_movement_fr_hz < 0:
        raise ValueError("--v1-min-movement-fr-hz must be non-negative.")
    if args.n_shuffles < 0:
        raise ValueError("--n-shuffles must be non-negative.")


def require_xarray():
    """Return `xarray` and fail clearly when NetCDF outputs are requested."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `xarray` to save ripple comparison outputs as NetCDF."
        ) from exc
    return xr


def get_pyplot():
    """Return pyplot configured for headless script execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def resolve_train_epoch(decode_epoch: str, requested_train_epoch: str | None) -> str:
    """Return the training epoch for one decoded epoch."""
    return str(requested_train_epoch) if requested_train_epoch is not None else str(decode_epoch)


def resolve_epoch_pairs(
    run_epochs: list[str],
    *,
    requested_decode_epoch: str | None,
    requested_decode_epochs: list[str] | None,
    requested_train_epoch: str | None,
) -> list[tuple[str, str]]:
    """Return ordered (decode_epoch, train_epoch) pairs for this run."""
    validated_train_epoch = validate_train_epoch(run_epochs, requested_train_epoch)
    if requested_decode_epoch is not None:
        decode_epoch = select_decode_epochs(run_epochs, [requested_decode_epoch])[0]
        train_epoch = resolve_train_epoch(decode_epoch, validated_train_epoch)
        return [(decode_epoch, train_epoch)]

    if requested_decode_epochs:
        decode_epochs = select_decode_epochs(run_epochs, requested_decode_epochs)
        return [
            (decode_epoch, resolve_train_epoch(decode_epoch, validated_train_epoch))
            for decode_epoch in decode_epochs
        ]

    if validated_train_epoch is not None:
        return [(validated_train_epoch, validated_train_epoch)]

    return [(epoch, epoch) for epoch in run_epochs]


def _subset_spikes(spikes: Any, unit_ids: list[Any]) -> Any:
    """Return a TsGroup restricted to the requested unit ids."""
    import pynapple as nap

    return nap.TsGroup({unit_id: spikes[unit_id] for unit_id in unit_ids}, time_units="s")


def build_region_unit_mask_table(
    *,
    unit_ids: np.ndarray,
    movement_firing_rates_hz: np.ndarray,
    min_movement_fr_hz: float,
    region: str,
) -> pd.DataFrame:
    """Return one auditable movement-rate unit mask table."""
    if movement_firing_rates_hz.shape[0] != unit_ids.size:
        raise ValueError(
            f"{region} movement firing rates do not match the saved unit count."
        )

    keep_unit = np.asarray(movement_firing_rates_hz, dtype=float) >= float(min_movement_fr_hz)
    return pd.DataFrame(
        {
            "unit_id": np.asarray(unit_ids),
            "movement_firing_rate_hz": np.asarray(movement_firing_rates_hz, dtype=float),
            "passes_movement_firing_rate": keep_unit,
            "keep_unit": keep_unit,
        }
    )


def _build_ripple_lookup(decoded_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return one lookup from ripple source index to decoded blocks."""
    lookup: dict[int, dict[str, Any]] = {}
    ripple_ids = np.asarray(decoded_data["ripple_ids"], dtype=int)
    for local_ripple_id, ripple_source_index in enumerate(
        np.asarray(decoded_data["ripple_source_indices"], dtype=int)
    ):
        mask = ripple_ids == int(local_ripple_id)
        lookup[int(ripple_source_index)] = {
            "ripple_id": int(local_ripple_id),
            "state": np.asarray(decoded_data["decoded_state"], dtype=float)[mask],
            "times": np.asarray(decoded_data["bin_times_s"], dtype=float)[mask],
            "start_time_s": float(decoded_data["ripple_start_times_s"][local_ripple_id]),
            "end_time_s": float(decoded_data["ripple_end_times_s"][local_ripple_id]),
        }
    return lookup


def align_decoded_ripple_data(
    ca1_decoded: dict[str, Any],
    v1_decoded: dict[str, Any],
) -> dict[str, Any]:
    """Align CA1 and V1 decoded traces on common ripple bins."""
    ca1_lookup = _build_ripple_lookup(ca1_decoded)
    v1_lookup = _build_ripple_lookup(v1_decoded)
    common_ripple_source_indices = sorted(set(ca1_lookup) & set(v1_lookup))

    ca1_state_chunks: list[np.ndarray] = []
    v1_state_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    ripple_id_chunks: list[np.ndarray] = []
    ripple_source_indices: list[int] = []
    ripple_start_times_s: list[float] = []
    ripple_end_times_s: list[float] = []
    skipped_ripples: list[dict[str, Any]] = []
    kept_ripple_count = 0

    for ripple_source_index in common_ripple_source_indices:
        ca1_block = ca1_lookup[ripple_source_index]
        v1_block = v1_lookup[ripple_source_index]
        ca1_times = np.asarray(ca1_block["times"], dtype=float)
        v1_times = np.asarray(v1_block["times"], dtype=float)
        ca1_times_rounded = np.round(ca1_times, decimals=9)
        v1_times_rounded = np.round(v1_times, decimals=9)
        common_times, ca1_indices, v1_indices = np.intersect1d(
            ca1_times_rounded,
            v1_times_rounded,
            assume_unique=False,
            return_indices=True,
        )
        if common_times.size == 0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": "CA1 and V1 decoded bins did not overlap.",
                }
            )
            continue

        ca1_state = np.asarray(ca1_block["state"], dtype=float)[ca1_indices]
        v1_state = np.asarray(v1_block["state"], dtype=float)[v1_indices]
        valid = np.isfinite(ca1_state) & np.isfinite(v1_state)
        if not np.any(valid):
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": "Aligned CA1 and V1 bins were non-finite.",
                }
            )
            continue

        ca1_state = ca1_state[valid]
        v1_state = v1_state[valid]
        aligned_times = np.asarray(ca1_times[ca1_indices], dtype=float)[valid]
        ca1_state_chunks.append(ca1_state)
        v1_state_chunks.append(v1_state)
        time_chunks.append(aligned_times)
        ripple_id_chunks.append(np.full(ca1_state.shape[0], kept_ripple_count, dtype=int))
        ripple_source_indices.append(int(ripple_source_index))
        ripple_start_times_s.append(float(max(ca1_block["start_time_s"], v1_block["start_time_s"])))
        ripple_end_times_s.append(float(min(ca1_block["end_time_s"], v1_block["end_time_s"])))
        kept_ripple_count += 1

    if not ca1_state_chunks:
        return {
            "ca1_decoded_state": np.array([], dtype=float),
            "v1_decoded_state": np.array([], dtype=float),
            "bin_times_s": np.array([], dtype=float),
            "ripple_ids": np.array([], dtype=int),
            "n_ripples": 0,
            "n_bins": 0,
            "ripple_source_indices": np.array([], dtype=int),
            "ripple_start_times_s": np.array([], dtype=float),
            "ripple_end_times_s": np.array([], dtype=float),
            "skipped_ripples": skipped_ripples,
        }

    return {
        "ca1_decoded_state": np.concatenate(ca1_state_chunks).astype(float, copy=False),
        "v1_decoded_state": np.concatenate(v1_state_chunks).astype(float, copy=False),
        "bin_times_s": np.concatenate(time_chunks).astype(float, copy=False),
        "ripple_ids": np.concatenate(ripple_id_chunks).astype(int, copy=False),
        "n_ripples": int(kept_ripple_count),
        "n_bins": int(sum(chunk.shape[0] for chunk in ca1_state_chunks)),
        "ripple_source_indices": np.asarray(ripple_source_indices, dtype=int),
        "ripple_start_times_s": np.asarray(ripple_start_times_s, dtype=float),
        "ripple_end_times_s": np.asarray(ripple_end_times_s, dtype=float),
        "skipped_ripples": skipped_ripples,
    }


def compute_per_ripple_metrics(
    *,
    ca1_state: np.ndarray,
    v1_state: np.ndarray,
    state_span: float,
) -> dict[str, float | int]:
    """Return coherence metrics for one ripple's aligned decoded states."""
    ca1_state = np.asarray(ca1_state, dtype=float).reshape(-1)
    v1_state = np.asarray(v1_state, dtype=float).reshape(-1)
    if ca1_state.shape != v1_state.shape:
        raise ValueError("CA1 and V1 ripple states must share the same shape.")

    diff = v1_state - ca1_state
    mean_abs_difference = float(np.mean(np.abs(diff))) if diff.size else np.nan
    if diff.size >= 2 and np.nanstd(ca1_state) > 0.0 and np.nanstd(v1_state) > 0.0:
        pearson_r = float(np.corrcoef(ca1_state, v1_state)[0, 1])
    else:
        pearson_r = np.nan

    if state_span > 0:
        mean_abs_difference_normalized = mean_abs_difference / float(state_span)
    else:
        mean_abs_difference_normalized = np.nan

    return {
        "pearson_r": pearson_r,
        "mean_abs_difference": mean_abs_difference,
        "mean_abs_difference_normalized": float(mean_abs_difference_normalized),
        "mean_signed_difference": float(np.mean(diff)) if diff.size else np.nan,
        "start_difference": float(diff[0]) if diff.size else np.nan,
        "end_difference": float(diff[-1]) if diff.size else np.nan,
        "n_bins": int(diff.size),
    }


def build_per_ripple_metric_table(
    *,
    aligned_data: dict[str, Any],
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    state_span: float,
    v1_state_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return one ripple-level coherence table for an aligned epoch."""
    ca1_state_all = np.asarray(aligned_data["ca1_decoded_state"], dtype=float)
    if v1_state_override is None:
        v1_state_all = np.asarray(aligned_data["v1_decoded_state"], dtype=float)
    else:
        v1_state_all = np.asarray(v1_state_override, dtype=float)
    ripple_ids = np.asarray(aligned_data["ripple_ids"], dtype=int)

    rows: list[dict[str, Any]] = []
    for ripple_id, ripple_source_index in enumerate(
        np.asarray(aligned_data["ripple_source_indices"], dtype=int)
    ):
        mask = ripple_ids == int(ripple_id)
        metrics = compute_per_ripple_metrics(
            ca1_state=ca1_state_all[mask],
            v1_state=v1_state_all[mask],
            state_span=state_span,
        )
        rows.append(
            {
                "representation": str(representation),
                "train_epoch": str(train_epoch),
                "decode_epoch": str(decode_epoch),
                "ripple_id": int(ripple_id),
                "ripple_source_index": int(ripple_source_index),
                "ripple_start_time_s": float(aligned_data["ripple_start_times_s"][ripple_id]),
                "ripple_end_time_s": float(aligned_data["ripple_end_times_s"][ripple_id]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def shuffle_ripple_state_blocks_by_length(
    state: np.ndarray,
    ripple_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Shuffle whole ripple-state blocks within matching lengths only."""
    state = np.asarray(state, dtype=float).reshape(-1)
    ripple_ids = np.asarray(ripple_ids, dtype=int).reshape(-1)
    if state.shape[0] != ripple_ids.size:
        raise ValueError("State rows must match ripple_ids length.")
    if state.size == 0:
        return state.copy(), False

    unique_ripple_ids = np.unique(ripple_ids)
    blocks = [state[ripple_ids == ripple_id] for ripple_id in unique_ripple_ids]
    block_indices_by_length: dict[int, list[int]] = {}
    for block_index, block in enumerate(blocks):
        block_indices_by_length.setdefault(int(block.size), []).append(block_index)

    shuffled_blocks = list(blocks)
    any_changed = False
    for block_indices in block_indices_by_length.values():
        if len(block_indices) < 2:
            continue
        permutation = deranged_permutation(len(block_indices), rng)
        if not np.array_equal(permutation, np.arange(len(block_indices), dtype=int)):
            any_changed = True
        for target_index, source_offset in zip(block_indices, permutation, strict=False):
            shuffled_blocks[target_index] = blocks[block_indices[source_offset]]

    return np.concatenate(shuffled_blocks).astype(float, copy=False), any_changed


def summarize_metric_against_shuffle(
    observed: float,
    null_samples: np.ndarray,
    *,
    direction: str,
) -> dict[str, float]:
    """Summarize one epoch metric against its shuffle null."""
    null_samples = np.asarray(null_samples, dtype=float).reshape(-1)
    finite_null = null_samples[np.isfinite(null_samples)]
    if not np.isfinite(observed):
        return {
            "shuffle_mean": float(np.mean(finite_null)) if finite_null.size else np.nan,
            "shuffle_sd": float(np.std(finite_null, ddof=0)) if finite_null.size else np.nan,
            "p_value": np.nan,
        }
    if finite_null.size == 0:
        return {
            "shuffle_mean": np.nan,
            "shuffle_sd": np.nan,
            "p_value": np.nan,
        }

    if direction == "higher":
        p_value = (1.0 + float(np.sum(finite_null >= observed))) / float(finite_null.size + 1)
    elif direction == "lower":
        p_value = (1.0 + float(np.sum(finite_null <= observed))) / float(finite_null.size + 1)
    elif direction == "zero":
        p_value = (1.0 + float(np.sum(np.abs(finite_null) <= abs(observed)))) / float(
            finite_null.size + 1
        )
    else:
        raise ValueError(f"Unsupported metric direction {direction!r}.")

    return {
        "shuffle_mean": float(np.mean(finite_null)),
        "shuffle_sd": float(np.std(finite_null, ddof=0)),
        "p_value": float(p_value),
    }


def build_epoch_summary_table(
    *,
    aligned_data: dict[str, Any],
    ripple_table: pd.DataFrame,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    state_span: float,
    n_shuffles: int,
    shuffle_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    """Return one epoch summary table plus per-shuffle null samples."""
    observed_table = build_per_ripple_metric_table(
        aligned_data=aligned_data,
        representation=representation,
        train_epoch=train_epoch,
        decode_epoch=decode_epoch,
        state_span=state_span,
    )
    observed_metrics = {
        metric_name: float(np.nanmean(observed_table[metric_name].to_numpy(dtype=float)))
        for metric_name in METRIC_LABELS
    }
    null_samples = {
        metric_name: np.full(int(n_shuffles), np.nan, dtype=float)
        for metric_name in METRIC_LABELS
    }

    effective_shuffles = 0
    if aligned_data["n_ripples"] >= 2 and n_shuffles > 0:
        rng = np.random.default_rng(shuffle_seed)
        for shuffle_index in range(n_shuffles):
            shuffled_v1_state, changed = shuffle_ripple_state_blocks_by_length(
                aligned_data["v1_decoded_state"],
                aligned_data["ripple_ids"],
                rng,
            )
            if not changed:
                continue
            shuffled_table = build_per_ripple_metric_table(
                aligned_data=aligned_data,
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
                state_span=state_span,
                v1_state_override=shuffled_v1_state,
            )
            for metric_name in METRIC_LABELS:
                null_samples[metric_name][shuffle_index] = float(
                    np.nanmean(shuffled_table[metric_name].to_numpy(dtype=float))
                )
            effective_shuffles += 1

    summary_row: dict[str, Any] = {
        "representation": str(representation),
        "train_epoch": str(train_epoch),
        "decode_epoch": str(decode_epoch),
        "n_ripples": int(aligned_data["n_ripples"]),
        "n_ripple_bins": int(aligned_data["n_bins"]),
        "n_ripple_events_input": int(len(ripple_table)),
        "n_effective_shuffles": int(effective_shuffles),
    }
    for metric_name, observed in observed_metrics.items():
        summary_row[metric_name] = observed
        shuffle_summary = summarize_metric_against_shuffle(
            observed,
            null_samples[metric_name],
            direction=METRIC_DIRECTION[metric_name],
        )
        summary_row[f"{metric_name}_shuffle_mean"] = shuffle_summary["shuffle_mean"]
        summary_row[f"{metric_name}_shuffle_sd"] = shuffle_summary["shuffle_sd"]
        summary_row[f"{metric_name}_p_value"] = shuffle_summary["p_value"]

    return pd.DataFrame([summary_row]), null_samples, effective_shuffles


def build_output_stem(
    *,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
) -> str:
    """Return the shared output stem for one decoded epoch."""
    return f"{representation}_train-{train_epoch}_decode-{decode_epoch}"


def build_decoded_output_name(
    *,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    region: str,
) -> str:
    """Return the decoded-state filename stem for one region and epoch."""
    return f"{build_output_stem(representation=representation, train_epoch=train_epoch, decode_epoch=decode_epoch)}_{region}_decoded"


def build_epoch_dataset_name(
    *,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
) -> str:
    """Return the combined epoch dataset filename."""
    return (
        f"{build_output_stem(representation=representation, train_epoch=train_epoch, decode_epoch=decode_epoch)}"
        "_comparison_dataset.nc"
    )


def build_epoch_dataset(
    *,
    aligned_data: dict[str, Any],
    ca1_mask_table: pd.DataFrame,
    v1_mask_table: pd.DataFrame,
    per_ripple_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    shuffle_samples: dict[str, np.ndarray],
    animal_name: str,
    date: str,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    bin_size_s: float,
    sources: dict[str, Any],
    skipped_ripples: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> Any:
    """Build one epoch-level xarray dataset with decoded states and coherence summaries."""
    xr = require_xarray()

    summary_row = summary_table.iloc[0]
    attrs = {
        "schema_version": "1",
        "animal_name": animal_name,
        "date": date,
        "representation": representation,
        "train_epoch": train_epoch,
        "decode_epoch": decode_epoch,
        "comparison_direction": "ca1_vs_v1_decoded_state",
        "bin_size_s": float(bin_size_s),
        "n_ripples": int(aligned_data["n_ripples"]),
        "n_ripple_bins": int(aligned_data["n_bins"]),
        "n_effective_shuffles": int(summary_row["n_effective_shuffles"]),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        "skipped_ripples_json": json.dumps(skipped_ripples, sort_keys=True),
    }

    data_vars: dict[str, Any] = {
        "bin_time_s": (("bin",), np.asarray(aligned_data["bin_times_s"], dtype=float)),
        "ripple_id": (("bin",), np.asarray(aligned_data["ripple_ids"], dtype=int)),
        "ca1_decoded_state": (
            ("bin",),
            np.asarray(aligned_data["ca1_decoded_state"], dtype=float),
        ),
        "v1_decoded_state": (
            ("bin",),
            np.asarray(aligned_data["v1_decoded_state"], dtype=float),
        ),
        "ripple_source_index": (
            ("ripple",),
            np.asarray(aligned_data["ripple_source_indices"], dtype=int),
        ),
        "ripple_start_time_s": (
            ("ripple",),
            np.asarray(aligned_data["ripple_start_times_s"], dtype=float),
        ),
        "ripple_end_time_s": (
            ("ripple",),
            np.asarray(aligned_data["ripple_end_times_s"], dtype=float),
        ),
        "ca1_movement_firing_rate_hz": (
            ("ca1_unit",),
            ca1_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "ca1_keep_unit": (("ca1_unit",), ca1_mask_table["keep_unit"].to_numpy(dtype=bool)),
        "v1_movement_firing_rate_hz": (
            ("v1_unit",),
            v1_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "v1_keep_unit": (("v1_unit",), v1_mask_table["keep_unit"].to_numpy(dtype=bool)),
    }
    for metric_name in METRIC_LABELS:
        data_vars[metric_name] = (
            ("ripple",),
            per_ripple_table[metric_name].to_numpy(dtype=float),
        )
        data_vars[f"{metric_name}_observed"] = float(summary_row[metric_name])
        data_vars[f"{metric_name}_shuffle"] = (
            ("shuffle",),
            np.asarray(shuffle_samples[metric_name], dtype=float),
        )
        data_vars[f"{metric_name}_shuffle_mean"] = float(summary_row[f"{metric_name}_shuffle_mean"])
        data_vars[f"{metric_name}_shuffle_sd"] = float(summary_row[f"{metric_name}_shuffle_sd"])
        data_vars[f"{metric_name}_p_value"] = float(summary_row[f"{metric_name}_p_value"])

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "bin": np.arange(int(aligned_data["n_bins"]), dtype=int),
            "ripple": np.arange(int(aligned_data["n_ripples"]), dtype=int),
            "shuffle": np.arange(int(len(next(iter(shuffle_samples.values()), []))), dtype=int),
            "ca1_unit": ca1_mask_table["unit_id"].to_numpy(),
            "v1_unit": v1_mask_table["unit_id"].to_numpy(),
        },
        attrs=attrs,
    )


def plot_epoch_summary(
    *,
    aligned_data: dict[str, Any],
    summary_table: pd.DataFrame,
    shuffle_samples: dict[str, np.ndarray],
    animal_name: str,
    date: str,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    out_path: Path,
) -> Path:
    """Plot binwise agreement plus two shuffle-null summaries for one epoch."""
    plt = get_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ca1_state = np.asarray(aligned_data["ca1_decoded_state"], dtype=float)
    v1_state = np.asarray(aligned_data["v1_decoded_state"], dtype=float)
    valid = np.isfinite(ca1_state) & np.isfinite(v1_state)
    if np.any(valid):
        axes[0].scatter(ca1_state[valid], v1_state[valid], s=12, alpha=0.5, color="tab:blue")
        finite_values = np.concatenate([ca1_state[valid], v1_state[valid]])
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        axes[0].plot([value_min, value_max], [value_min, value_max], color="black", alpha=0.6)
    else:
        axes[0].text(0.5, 0.5, "No finite aligned bins", ha="center", va="center")
    axes[0].set_xlabel("CA1 decoded state")
    axes[0].set_ylabel("V1 decoded state")
    axes[0].set_title("Binwise decoded-state agreement")
    axes[0].grid(True, alpha=0.2)

    summary_row = summary_table.iloc[0]
    for axis, metric_name in zip(
        axes[1:],
        ("pearson_r", "mean_abs_difference"),
        strict=False,
    ):
        shuffle_values = np.asarray(shuffle_samples[metric_name], dtype=float)
        finite_shuffle = shuffle_values[np.isfinite(shuffle_values)]
        if finite_shuffle.size:
            axis.hist(finite_shuffle, bins=min(20, max(finite_shuffle.size, 5)), color="0.75")
            axis.axvline(float(summary_row[metric_name]), color="tab:red", linewidth=2)
        else:
            axis.text(0.5, 0.5, "No effective shuffles", ha="center", va="center")
        axis.set_xlabel(METRIC_LABELS[metric_name])
        axis.set_ylabel("Shuffle count")
        axis.set_title(
            f"Observed vs shuffle\np={summary_row[f'{metric_name}_p_value']:.3g}"
            if np.isfinite(summary_row[f"{metric_name}_p_value"])
            else "Observed vs shuffle\np=NaN"
        )
        axis.grid(True, alpha=0.2)

    figure.suptitle(
        (
            f"{animal_name} {date} {representation} "
            f"train {train_epoch} decode {decode_epoch}"
        ),
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def main() -> None:
    """Run the ripple decoding comparison CLI."""
    args = parse_arguments()
    validate_arguments(args)

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=REGIONS,
    )
    epoch_pairs = resolve_epoch_pairs(
        session["run_epochs"],
        requested_decode_epoch=args.decode_epoch,
        requested_decode_epochs=args.decode_epochs,
        requested_train_epoch=args.train_epoch,
    )
    feature_by_epoch, bin_edges, feature_name = get_representation_inputs(
        session,
        animal_name=args.animal_name,
        representation=args.representation,
    )
    state_span = float(bin_edges[-1] - bin_edges[0]) if len(bin_edges) >= 2 else np.nan
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = analysis_path / "ripple_decoding_comparison"
    fig_dir = analysis_path / "figs" / "ripple_decoding_comparison"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _epoch_tags, timestamps_ephys_by_epoch, ephys_source = load_ephys_timestamps_by_epoch(
        analysis_path
    )
    epoch_intervals = build_epoch_intervals(timestamps_ephys_by_epoch)
    ripple_tables, ripple_source = load_ripple_tables(analysis_path)

    sources = dict(session["sources"])
    sources["timestamps_ephys"] = ephys_source
    sources["ripple_events"] = ripple_source
    fit_parameters = {
        "bin_size_s": args.bin_size_s,
        "ca1_min_movement_fr_hz": args.ca1_min_movement_fr_hz,
        "v1_min_movement_fr_hz": args.v1_min_movement_fr_hz,
        "n_shuffles": args.n_shuffles,
        "shuffle_seed": args.shuffle_seed,
        "comparison_direction": "ca1_vs_v1_decoded_state",
    }

    ca1_spikes_all = session["spikes_by_region"]["ca1"]
    v1_spikes_all = session["spikes_by_region"]["v1"]
    ca1_unit_ids_all = np.asarray(list(ca1_spikes_all.keys()))
    v1_unit_ids_all = np.asarray(list(v1_spikes_all.keys()))

    saved_decoded_paths: list[Path] = []
    saved_ripple_table_paths: list[Path] = []
    saved_summary_table_paths: list[Path] = []
    saved_dataset_paths: list[Path] = []
    saved_figure_paths: list[Path] = []
    skipped_epochs: list[dict[str, Any]] = []

    for decode_epoch, train_epoch in epoch_pairs:
        ripple_table = ripple_tables.get(decode_epoch, empty_ripple_table())
        if ripple_table.empty:
            skipped_epochs.append(
                {"epoch": decode_epoch, "reason": "No ripple events were found for this epoch."}
            )
            print(f"Skipping {args.animal_name} {args.date} {decode_epoch}: no ripple events")
            continue

        ca1_mask_table = build_region_unit_mask_table(
            unit_ids=ca1_unit_ids_all,
            movement_firing_rates_hz=np.asarray(
                movement_firing_rates["ca1"][train_epoch],
                dtype=float,
            ),
            min_movement_fr_hz=args.ca1_min_movement_fr_hz,
            region="ca1",
        )
        v1_mask_table = build_region_unit_mask_table(
            unit_ids=v1_unit_ids_all,
            movement_firing_rates_hz=np.asarray(
                movement_firing_rates["v1"][train_epoch],
                dtype=float,
            ),
            min_movement_fr_hz=args.v1_min_movement_fr_hz,
            region="v1",
        )
        ca1_keep_unit_ids = ca1_mask_table.loc[
            ca1_mask_table["keep_unit"], "unit_id"
        ].tolist()
        v1_keep_unit_ids = v1_mask_table.loc[
            v1_mask_table["keep_unit"], "unit_id"
        ].tolist()
        if not ca1_keep_unit_ids or not v1_keep_unit_ids:
            skipped_epochs.append(
                {
                    "epoch": decode_epoch,
                    "reason": "No CA1 or V1 units passed the movement firing-rate threshold.",
                    "train_epoch": train_epoch,
                }
            )
            print(
                f"Skipping {args.animal_name} {args.date} {decode_epoch}: "
                "no CA1 or V1 units passed movement-rate filtering"
            )
            continue

        ca1_spikes = _subset_spikes(ca1_spikes_all, ca1_keep_unit_ids)
        v1_spikes = _subset_spikes(v1_spikes_all, v1_keep_unit_ids)
        try:
            ca1_tuning_curves = compute_tuning_curves_for_epoch(
                spikes=ca1_spikes,
                feature=feature_by_epoch[train_epoch],
                movement_interval=session["movement_by_run"][train_epoch],
                bin_edges=bin_edges,
                feature_name=feature_name,
            )
            v1_tuning_curves = compute_tuning_curves_for_epoch(
                spikes=v1_spikes,
                feature=feature_by_epoch[train_epoch],
                movement_interval=session["movement_by_run"][train_epoch],
                bin_edges=bin_edges,
                feature_name=feature_name,
            )
            ca1_decoded = assemble_decoded_ripple_epoch_data(
                spikes=ca1_spikes,
                tuning_curves=ca1_tuning_curves,
                ripple_table=ripple_table,
                epoch_interval=epoch_intervals[decode_epoch],
                bin_size_s=args.bin_size_s,
            )
            v1_decoded = assemble_decoded_ripple_epoch_data(
                spikes=v1_spikes,
                tuning_curves=v1_tuning_curves,
                ripple_table=ripple_table,
                epoch_interval=epoch_intervals[decode_epoch],
                bin_size_s=args.bin_size_s,
            )
        except Exception as exc:
            skipped_epochs.append(
                {
                    "epoch": decode_epoch,
                    "reason": "Failed to assemble ripple decoding inputs.",
                    "error": str(exc),
                    "train_epoch": train_epoch,
                }
            )
            print(
                f"Skipping {args.animal_name} {args.date} {decode_epoch}: "
                f"failed to decode ripple inputs: {exc}"
            )
            continue

        aligned_data = align_decoded_ripple_data(ca1_decoded, v1_decoded)
        if aligned_data["n_ripples"] == 0 or aligned_data["n_bins"] == 0:
            skipped_epochs.append(
                {
                    "epoch": decode_epoch,
                    "reason": "No common aligned CA1/V1 ripple bins were available.",
                    "train_epoch": train_epoch,
                    "ca1_skipped_ripples": ca1_decoded["skipped_ripples"],
                    "v1_skipped_ripples": v1_decoded["skipped_ripples"],
                    "alignment_skipped_ripples": aligned_data["skipped_ripples"],
                }
            )
            print(
                f"Skipping {args.animal_name} {args.date} {decode_epoch}: "
                "no common aligned CA1/V1 ripple bins"
            )
            continue

        per_ripple_table = build_per_ripple_metric_table(
            aligned_data=aligned_data,
            representation=args.representation,
            train_epoch=train_epoch,
            decode_epoch=decode_epoch,
            state_span=state_span,
        )
        summary_table, shuffle_samples, _effective_shuffles = build_epoch_summary_table(
            aligned_data=aligned_data,
            ripple_table=ripple_table,
            representation=args.representation,
            train_epoch=train_epoch,
            decode_epoch=decode_epoch,
            state_span=state_span,
            n_shuffles=args.n_shuffles,
            shuffle_seed=args.shuffle_seed,
        )
        skipped_ripples = {
            "ca1": ca1_decoded["skipped_ripples"],
            "v1": v1_decoded["skipped_ripples"],
            "alignment": aligned_data["skipped_ripples"],
        }
        epoch_dataset = build_epoch_dataset(
            aligned_data=aligned_data,
            ca1_mask_table=ca1_mask_table,
            v1_mask_table=v1_mask_table,
            per_ripple_table=per_ripple_table,
            summary_table=summary_table,
            shuffle_samples=shuffle_samples,
            animal_name=args.animal_name,
            date=args.date,
            representation=args.representation,
            train_epoch=train_epoch,
            decode_epoch=decode_epoch,
            bin_size_s=args.bin_size_s,
            sources=sources,
            skipped_ripples=skipped_ripples,
            fit_parameters=fit_parameters,
        )

        ca1_decoded_path = (
            data_dir
            / f"{build_decoded_output_name(representation=args.representation, train_epoch=train_epoch, decode_epoch=decode_epoch, region='ca1')}.npz"
        )
        v1_decoded_path = (
            data_dir
            / f"{build_decoded_output_name(representation=args.representation, train_epoch=train_epoch, decode_epoch=decode_epoch, region='v1')}.npz"
        )
        ca1_decoded["decoded_tsd"].save(ca1_decoded_path)
        v1_decoded["decoded_tsd"].save(v1_decoded_path)
        saved_decoded_paths.extend([ca1_decoded_path, v1_decoded_path])

        output_stem = build_output_stem(
            representation=args.representation,
            train_epoch=train_epoch,
            decode_epoch=decode_epoch,
        )
        ripple_metrics_path = data_dir / f"{output_stem}_ripple_metrics.parquet"
        summary_table_path = data_dir / f"{output_stem}_epoch_summary.parquet"
        dataset_path = data_dir / build_epoch_dataset_name(
            representation=args.representation,
            train_epoch=train_epoch,
            decode_epoch=decode_epoch,
        )
        figure_path = fig_dir / f"{output_stem}_coherence_summary.png"

        per_ripple_table.to_parquet(ripple_metrics_path, index=False)
        summary_table.to_parquet(summary_table_path, index=False)
        epoch_dataset.to_netcdf(dataset_path)
        plot_epoch_summary(
            aligned_data=aligned_data,
            summary_table=summary_table,
            shuffle_samples=shuffle_samples,
            animal_name=args.animal_name,
            date=args.date,
            representation=args.representation,
            train_epoch=train_epoch,
            decode_epoch=decode_epoch,
            out_path=figure_path,
        )
        saved_ripple_table_paths.append(ripple_metrics_path)
        saved_summary_table_paths.append(summary_table_path)
        saved_dataset_paths.append(dataset_path)
        saved_figure_paths.append(figure_path)

    if not saved_summary_table_paths:
        raise RuntimeError(
            "All requested decode epochs were skipped. "
            f"Epoch reasons: {skipped_epochs!r}"
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.ripple.ripple_decoding_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "representation": args.representation,
            "data_root": args.data_root,
            "epoch_pairs": [
                {"decode_epoch": decode_epoch, "train_epoch": train_epoch}
                for decode_epoch, train_epoch in epoch_pairs
            ],
            "decode_epoch": args.decode_epoch,
            "decode_epochs": args.decode_epochs,
            "train_epoch": args.train_epoch,
            "bin_size_s": args.bin_size_s,
            "ca1_min_movement_fr_hz": args.ca1_min_movement_fr_hz,
            "v1_min_movement_fr_hz": args.v1_min_movement_fr_hz,
            "n_shuffles": args.n_shuffles,
            "shuffle_seed": args.shuffle_seed,
            "comparison_direction": "ca1_vs_v1_decoded_state",
        },
        outputs={
            "sources": sources,
            "saved_decoded_paths": saved_decoded_paths,
            "saved_ripple_table_paths": saved_ripple_table_paths,
            "saved_summary_table_paths": saved_summary_table_paths,
            "saved_dataset_paths": saved_dataset_paths,
            "saved_figure_paths": saved_figure_paths,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
