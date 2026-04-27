from __future__ import annotations

"""Classify ripple trajectory identity in CA1 and V1 for one session."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    REGIONS,
    TRAJECTORY_TYPES,
    get_analysis_path,
)
from v1ca1.ripple._decoding import (
    build_region_unit_mask_table,
    concatenate_tsds,
    extract_time_values,
    get_tsgroup_unit_ids,
    make_empty_tsd,
    make_intervalset_from_bounds,
    select_decode_epochs,
)
from v1ca1.ripple.ripple_glm import (
    build_epoch_intervals,
    load_ripple_tables,
)
from v1ca1.task_progression._session import prepare_task_progression_session

DEFAULT_BIN_SIZE_S = 0.01
DEFAULT_N_SPLITS = 5
DEFAULT_N_SHUFFLES = 100
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_CA1_MIN_MOVEMENT_FR_HZ = 0.0
DEFAULT_V1_MIN_MOVEMENT_FR_HZ = 0.5
DEFAULT_DECODER = "cosine_nearest_centroid"

DECODER_NAMES = (
    "cosine_nearest_centroid",
    "poisson_naive_bayes",
)

LABEL_SCHEMES = {
    "trajectory": {
        "class_names": tuple(TRAJECTORY_TYPES),
        "label_by_trajectory": {
            trajectory_type: trajectory_type for trajectory_type in TRAJECTORY_TYPES
        },
    },
    "turn_group": {
        "class_names": ("left_turn", "right_turn"),
        "label_by_trajectory": {
            "center_to_left": "left_turn",
            "right_to_center": "left_turn",
            "center_to_right": "right_turn",
            "left_to_center": "right_turn",
        },
    },
    "arm": {
        "class_names": ("left", "right"),
        "label_by_trajectory": {
            "center_to_left": "left",
            "left_to_center": "left",
            "center_to_right": "right",
            "right_to_center": "right",
        },
    },
}


def get_class_index_by_name(class_names: tuple[str, ...]) -> dict[str, int]:
    return {
        class_name: class_index for class_index, class_name in enumerate(class_names)
    }


def get_probability_columns(class_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"posterior_{class_name}" for class_name in class_names)


def get_score_columns(
    *, decoder_name: str, class_names: tuple[str, ...]
) -> tuple[str, ...]:
    prefix = "log_likelihood" if decoder_name == "poisson_naive_bayes" else "similarity"
    return tuple(f"{prefix}_{class_name}" for class_name in class_names)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.n_shuffles < 0:
        raise ValueError("--n-shuffles must be non-negative.")
    if args.ca1_min_movement_fr_hz < 0:
        raise ValueError("--ca1-min-movement-fr-hz must be non-negative.")
    if args.v1_min_movement_fr_hz < 0:
        raise ValueError("--v1-min-movement-fr-hz must be non-negative.")


def resolve_label_schemes(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(LABEL_SCHEMES)
    resolved: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if value not in seen:
            seen.add(value)
            resolved.append(value)
    return resolved


def _subset_spikes(spikes: Any, unit_ids: list[Any]) -> Any:
    import pynapple as nap

    return nap.TsGroup({unit_id: spikes[unit_id] for unit_id in unit_ids}, time_units="s")


def _intervalset_bounds(intervalset: Any) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(getattr(intervalset, "start", []), dtype=float).reshape(-1)
    ends = np.asarray(getattr(intervalset, "end", []), dtype=float).reshape(-1)
    if starts.shape != ends.shape:
        raise ValueError("IntervalSet start and end bounds do not share the same shape.")
    return starts, ends


def _first_last_interval_times(intervalset: Any) -> tuple[float, float]:
    starts, ends = _intervalset_bounds(intervalset)
    if starts.size == 0:
        return np.nan, np.nan
    return float(starts[0]), float(ends[-1])


def validate_count_columns(counts: Any, unit_ids: np.ndarray, label: str) -> None:
    count_columns = np.asarray(counts.columns)
    if count_columns.shape != unit_ids.shape or not np.array_equal(count_columns, unit_ids):
        raise ValueError(
            f"{label} spike count columns do not match the expected unit order. "
            f"Expected {unit_ids!r}, got {count_columns!r}."
        )


def count_spikes_in_interval(
    spikes: Any,
    unit_ids: np.ndarray,
    intervalset: Any,
    *,
    bin_size_s: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    counts = spikes.count(float(bin_size_s), ep=intervalset)
    validate_count_columns(counts, unit_ids, label=label)
    count_matrix = np.asarray(counts.d, dtype=float)
    if count_matrix.ndim == 1:
        count_matrix = count_matrix[:, np.newaxis]
    count_times = extract_time_values(counts).reshape(-1)
    if count_times.size != count_matrix.shape[0]:
        raise ValueError(
            f"{label} count timestamps do not match the number of spike-count rows."
        )
    return count_matrix, count_times


def compute_movement_firing_rates(
    spikes_by_region: dict[str, Any],
    movement_by_run: dict[str, Any],
    run_epochs: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    movement_rates: dict[str, dict[str, np.ndarray]] = {}
    for region, spikes in spikes_by_region.items():
        movement_rates[region] = {}
        for epoch in run_epochs:
            movement_interval = movement_by_run[epoch]
            duration = float(movement_interval.tot_length())
            counts = np.asarray(spikes.count(ep=movement_interval).to_numpy(), dtype=float)
            if counts.ndim == 1:
                counts = counts[np.newaxis, :]
            total_counts = np.sum(counts, axis=0).ravel()
            if duration <= 0:
                movement_rates[region][epoch] = np.zeros(total_counts.shape, dtype=float)
            else:
                movement_rates[region][epoch] = total_counts / duration
    return movement_rates


def build_behavioral_events(
    *,
    spikes_by_region: dict[str, Any],
    unit_ids_by_region: dict[str, np.ndarray],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    bin_size_s: float,
    label_scheme_name: str,
    class_names: tuple[str, ...],
    label_by_trajectory: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    skipped_events: list[dict[str, Any]] = []
    class_index_by_name = get_class_index_by_name(class_names)

    for trajectory_type in TRAJECTORY_TYPES:
        source_intervalset = trajectory_intervals[trajectory_type]
        source_starts, source_ends = _intervalset_bounds(source_intervalset)
        for source_index, (start_time_s, end_time_s) in enumerate(zip(source_starts, source_ends)):
            source_ep = make_intervalset_from_bounds(
                np.array([float(start_time_s)], dtype=float),
                np.array([float(end_time_s)], dtype=float),
            )
            event_support = source_ep.intersect(movement_interval)
            if float(event_support.tot_length()) <= 0.0:
                skipped_events.append(
                    {
                        "trajectory_type": trajectory_type,
                        "source_index": int(source_index),
                        "reason": "No overlap with movement interval.",
                    }
                )
                continue

            region_counts: dict[str, np.ndarray] = {}
            region_times: dict[str, np.ndarray] = {}
            invalid_reason: str | None = None
            for region in REGIONS:
                count_matrix, count_times = count_spikes_in_interval(
                    spikes_by_region[region],
                    unit_ids_by_region[region],
                    event_support,
                    bin_size_s=bin_size_s,
                    label=region.upper(),
                )
                if count_matrix.shape[0] == 0:
                    invalid_reason = f"{region.upper()} counting produced zero bins."
                    break
                region_counts[region] = count_matrix
                region_times[region] = count_times

            if invalid_reason is not None:
                skipped_events.append(
                    {
                        "trajectory_type": trajectory_type,
                        "source_index": int(source_index),
                        "reason": invalid_reason,
                    }
                )
                continue

            ca1_times = np.asarray(region_times["ca1"], dtype=float)
            v1_times = np.asarray(region_times["v1"], dtype=float)
            if ca1_times.shape != v1_times.shape or not np.allclose(
                ca1_times,
                v1_times,
                rtol=1e-9,
                atol=1e-9,
            ):
                skipped_events.append(
                    {
                        "trajectory_type": trajectory_type,
                        "source_index": int(source_index),
                        "reason": "CA1 and V1 traversal bins did not align.",
                    }
                )
                continue

            event_start_time_s, event_end_time_s = _first_last_interval_times(event_support)
            events.append(
                {
                    "event_id": int(len(events)),
                    "label_scheme": str(label_scheme_name),
                    "trajectory_type": trajectory_type,
                    "class_label": str(label_by_trajectory[trajectory_type]),
                    "class_index": int(
                        class_index_by_name[label_by_trajectory[trajectory_type]]
                    ),
                    "source_index": int(source_index),
                    "source_start_time_s": float(start_time_s),
                    "source_end_time_s": float(end_time_s),
                    "event_start_time_s": float(event_start_time_s),
                    "event_end_time_s": float(event_end_time_s),
                    "duration_s": float(event_support.tot_length()),
                    "n_bins": int(ca1_times.size),
                    "interval_support": event_support,
                    "bin_times_s": ca1_times,
                    "region_counts": region_counts,
                }
            )

    return events, skipped_events


def fit_poisson_naive_bayes(
    *,
    count_matrices: list[np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    bin_size_s: float,
) -> dict[str, np.ndarray]:
    if not count_matrices:
        raise ValueError("No count matrices were provided for model fitting.")

    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.shape[0] != len(count_matrices):
        raise ValueError("The number of labels must match the number of count matrices.")

    n_units = int(np.asarray(count_matrices[0], dtype=float).shape[1])
    rates_hz = np.zeros((n_classes, n_units), dtype=float)
    class_event_counts = np.zeros(n_classes, dtype=int)
    class_bin_counts = np.zeros(n_classes, dtype=int)

    for class_index in range(n_classes):
        class_matrices = [
            np.asarray(count_matrix, dtype=float)
            for count_matrix, label in zip(count_matrices, labels)
            if int(label) == class_index
        ]
        if not class_matrices:
            raise ValueError(f"Class index {class_index} had no training events.")

        total_bins = int(sum(matrix.shape[0] for matrix in class_matrices))
        total_counts = np.sum(np.concatenate(class_matrices, axis=0), axis=0)
        total_duration_s = float(total_bins) * float(bin_size_s)
        rates_hz[class_index] = (
            total_counts / total_duration_s if total_duration_s > 0.0 else np.zeros(n_units)
        )
        class_event_counts[class_index] = int(len(class_matrices))
        class_bin_counts[class_index] = int(total_bins)

    return {
        "rates_hz": np.asarray(rates_hz, dtype=float),
        "class_event_counts": np.asarray(class_event_counts, dtype=int),
        "class_bin_counts": np.asarray(class_bin_counts, dtype=int),
    }


def build_event_rate_vector(
    *,
    count_matrix: np.ndarray,
    bin_size_s: float,
) -> np.ndarray:
    counts = np.asarray(count_matrix, dtype=float)
    if counts.ndim != 2:
        raise ValueError(f"Expected a 2D count matrix, got shape {counts.shape}.")
    if counts.shape[0] == 0:
        raise ValueError("Count matrices must include at least one time bin.")

    total_counts = np.sum(counts, axis=0)
    duration_s = float(counts.shape[0]) * float(bin_size_s)
    if duration_s <= 0.0:
        return np.zeros(total_counts.shape, dtype=float)
    return (total_counts / duration_s).astype(float, copy=False)


def fit_cosine_nearest_centroid(
    *,
    count_matrices: list[np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    bin_size_s: float,
) -> dict[str, np.ndarray]:
    if not count_matrices:
        raise ValueError("No count matrices were provided for model fitting.")

    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.shape[0] != len(count_matrices):
        raise ValueError("The number of labels must match the number of count matrices.")

    event_rate_vectors = np.stack(
        [
            build_event_rate_vector(count_matrix=count_matrix, bin_size_s=bin_size_s)
            for count_matrix in count_matrices
        ],
        axis=0,
    )
    n_units = int(event_rate_vectors.shape[1])
    centroids = np.zeros((n_classes, n_units), dtype=float)
    class_event_counts = np.zeros(n_classes, dtype=int)

    for class_index in range(n_classes):
        class_vectors = event_rate_vectors[labels == class_index]
        if class_vectors.shape[0] == 0:
            raise ValueError(f"Class index {class_index} had no training events.")
        centroids[class_index] = np.mean(class_vectors, axis=0)
        class_event_counts[class_index] = int(class_vectors.shape[0])

    return {
        "centroids": centroids,
        "class_event_counts": class_event_counts,
    }


def score_count_matrix(
    *,
    count_matrix: np.ndarray,
    model: dict[str, np.ndarray],
    bin_size_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(count_matrix, dtype=float)
    if counts.ndim != 2:
        raise ValueError(f"Expected a 2D count matrix, got shape {counts.shape}.")
    if counts.shape[0] == 0:
        raise ValueError("Count matrices must include at least one time bin.")

    total_counts = np.sum(counts, axis=0)
    n_bins = int(counts.shape[0])
    rates_hz = np.asarray(model["rates_hz"], dtype=float)
    lam = np.clip(rates_hz * float(bin_size_s), 1e-12, None)
    log_prior = -np.log(float(rates_hz.shape[0]))
    scores = total_counts[np.newaxis, :] @ np.log(lam).T
    scores = scores.reshape(-1) - float(n_bins) * np.sum(lam, axis=1) + log_prior
    posterior = np.exp(scores - logsumexp(scores))
    return posterior.astype(float, copy=False), scores.astype(float, copy=False)


def score_count_matrix_cosine(
    *,
    count_matrix: np.ndarray,
    model: dict[str, np.ndarray],
    bin_size_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_rate_vector = build_event_rate_vector(
        count_matrix=count_matrix,
        bin_size_s=bin_size_s,
    )
    centroids = np.asarray(model["centroids"], dtype=float)
    if centroids.ndim != 2:
        raise ValueError(f"Expected a 2D centroid array, got shape {centroids.shape}.")

    feature_norm = float(np.linalg.norm(event_rate_vector))
    centroid_norms = np.linalg.norm(centroids, axis=1)
    denominator = np.maximum(feature_norm * centroid_norms, 1e-12)
    scores = (centroids @ event_rate_vector) / denominator
    posterior = np.exp(scores - logsumexp(scores))
    return posterior.astype(float, copy=False), scores.astype(float, copy=False)


def fit_decoder(
    *,
    decoder_name: str,
    count_matrices: list[np.ndarray],
    labels: np.ndarray,
    n_classes: int,
    bin_size_s: float,
) -> dict[str, np.ndarray]:
    if decoder_name == "poisson_naive_bayes":
        return fit_poisson_naive_bayes(
            count_matrices=count_matrices,
            labels=labels,
            n_classes=n_classes,
            bin_size_s=bin_size_s,
        )
    if decoder_name == "cosine_nearest_centroid":
        return fit_cosine_nearest_centroid(
            count_matrices=count_matrices,
            labels=labels,
            n_classes=n_classes,
            bin_size_s=bin_size_s,
        )
    raise ValueError(f"Unsupported decoder: {decoder_name}")


def score_decoder(
    *,
    decoder_name: str,
    count_matrix: np.ndarray,
    model: dict[str, np.ndarray],
    bin_size_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if decoder_name == "poisson_naive_bayes":
        return score_count_matrix(
            count_matrix=count_matrix,
            model=model,
            bin_size_s=bin_size_s,
        )
    if decoder_name == "cosine_nearest_centroid":
        return score_count_matrix_cosine(
            count_matrix=count_matrix,
            model=model,
            bin_size_s=bin_size_s,
        )
    raise ValueError(f"Unsupported decoder: {decoder_name}")


def summarize_posterior(
    posterior: np.ndarray,
    scores: np.ndarray,
    *,
    decoder_name: str,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    posterior = np.asarray(posterior, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if posterior.shape[0] != len(class_names):
        raise ValueError("Posterior length did not match the number of trajectory classes.")

    probability_columns = get_probability_columns(class_names)
    score_columns = get_score_columns(
        decoder_name=decoder_name,
        class_names=class_names,
    )
    predicted_class_index = int(np.argmax(posterior))
    sorted_posterior = np.sort(posterior)
    margin_to_second = (
        float(sorted_posterior[-1] - sorted_posterior[-2])
        if sorted_posterior.size >= 2
        else np.nan
    )
    safe_posterior = np.clip(posterior, 1e-12, None)
    entropy = float(-np.sum(safe_posterior * np.log(safe_posterior)))
    summary = {
        "predicted_class_index": predicted_class_index,
        "predicted_class": class_names[predicted_class_index],
        "max_posterior": float(posterior[predicted_class_index]),
        "margin_to_second": margin_to_second,
        "entropy": entropy,
    }
    for column_name, value in zip(probability_columns, posterior):
        summary[column_name] = float(value)
    for column_name, value in zip(score_columns, scores):
        summary[column_name] = float(value)
    return summary


def compute_region_metrics(
    *,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)
    predicted_labels = np.asarray(predicted_labels, dtype=int).reshape(-1)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("True and predicted labels must share the same shape.")

    confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names), dtype=int),
    )
    per_class_recall = recall_score(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names), dtype=int),
        average=None,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion.astype(int, copy=False),
    }
    for class_name, recall_value in zip(class_names, per_class_recall):
        metrics[f"recall_{class_name}"] = float(recall_value)
    return metrics


def build_behavior_cv_outputs(
    *,
    events: list[dict[str, Any]],
    decoder_name: str,
    n_splits: int,
    shuffle_seed: int,
    bin_size_s: float,
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    class_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels = np.asarray([event["class_index"] for event in events], dtype=int)
    probability_columns = get_probability_columns(class_names)
    score_columns = get_score_columns(
        decoder_name=decoder_name,
        class_names=class_names,
    )
    class_counts = np.bincount(labels, minlength=len(class_names))
    if np.any(class_counts == 0):
        missing = [class_names[index] for index, count in enumerate(class_counts) if count == 0]
        raise ValueError(
            "Traversal CV requires all classes in the selected label scheme. "
            f"Missing classes: {missing!r}"
        )

    effective_n_splits = int(min(n_splits, int(np.min(class_counts))))
    if effective_n_splits < 2:
        raise ValueError(
            "Not enough traversal events for CV: "
            f"min_class_count={int(np.min(class_counts))}, requested_n_splits={n_splits}"
        )

    splitter = StratifiedKFold(
        n_splits=effective_n_splits,
        shuffle=True,
        random_state=int(shuffle_seed),
    )
    n_events = len(events)
    fold_indices = np.full(n_events, -1, dtype=int)
    predictions_by_region: dict[str, np.ndarray] = {}
    posteriors_by_region: dict[str, np.ndarray] = {}
    scores_by_region: dict[str, np.ndarray] = {}
    for region in REGIONS:
        predictions_by_region[region] = np.full(n_events, -1, dtype=int)
        posteriors_by_region[region] = np.full(
            (n_events, len(class_names)),
            np.nan,
            dtype=float,
        )
        scores_by_region[region] = np.full(
            (n_events, len(class_names)),
            np.nan,
            dtype=float,
        )

    for fold_index, (train_index, test_index) in enumerate(splitter.split(np.arange(n_events), labels)):
        fold_indices[test_index] = int(fold_index)
        train_labels = labels[train_index]
        for region in REGIONS:
            model = fit_decoder(
                decoder_name=decoder_name,
                count_matrices=[
                    events[event_index]["region_counts"][region]
                    for event_index in train_index.tolist()
                ],
                labels=train_labels,
                n_classes=len(class_names),
                bin_size_s=bin_size_s,
            )
            for event_index in test_index.tolist():
                posterior, scores = score_decoder(
                    decoder_name=decoder_name,
                    count_matrix=events[event_index]["region_counts"][region],
                    model=model,
                    bin_size_s=bin_size_s,
                )
                predictions_by_region[region][event_index] = int(np.argmax(posterior))
                posteriors_by_region[region][event_index] = posterior
                scores_by_region[region][event_index] = scores

    if np.any(fold_indices < 0):
        raise RuntimeError("Not all traversal events received a held-out CV prediction.")

    prediction_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        row: dict[str, Any] = {
            "animal_name": str(animal_name),
            "date": str(date),
            "decode_epoch": str(decode_epoch),
            "label_scheme": str(label_scheme_name),
            "decoder": str(decoder_name),
            "event_id": int(event["event_id"]),
            "fold_index": int(fold_indices[event_index]),
            "trajectory_type": str(event["trajectory_type"]),
            "true_class_index": int(event["class_index"]),
            "true_class": str(class_names[int(event["class_index"])]),
            "source_index": int(event["source_index"]),
            "source_start_time_s": float(event["source_start_time_s"]),
            "source_end_time_s": float(event["source_end_time_s"]),
            "event_start_time_s": float(event["event_start_time_s"]),
            "event_end_time_s": float(event["event_end_time_s"]),
            "duration_s": float(event["duration_s"]),
            "n_bins": int(event["n_bins"]),
        }
        for region in REGIONS:
            posterior = posteriors_by_region[region][event_index]
            scores = scores_by_region[region][event_index]
            region_summary = summarize_posterior(
                posterior,
                scores,
                decoder_name=decoder_name,
                class_names=class_names,
            )
            row[f"{region}_predicted_class_index"] = int(region_summary["predicted_class_index"])
            row[f"{region}_predicted_class"] = str(region_summary["predicted_class"])
            row[f"{region}_correct"] = bool(
                region_summary["predicted_class_index"] == int(event["class_index"])
            )
            row[f"{region}_max_posterior"] = float(region_summary["max_posterior"])
            row[f"{region}_margin_to_second"] = float(region_summary["margin_to_second"])
            row[f"{region}_entropy"] = float(region_summary["entropy"])
            for column_name in probability_columns:
                row[f"{region}_{column_name}"] = float(region_summary[column_name])
            for column_name in score_columns:
                row[f"{region}_{column_name}"] = float(region_summary[column_name])
        prediction_rows.append(row)
    prediction_table = pd.DataFrame(prediction_rows)

    summary_rows: list[dict[str, Any]] = []
    metrics_by_region: dict[str, Any] = {}
    for region in REGIONS:
        metrics = compute_region_metrics(
            true_labels=labels,
            predicted_labels=predictions_by_region[region],
            class_names=class_names,
        )
        metrics_by_region[region] = metrics
        summary_row: dict[str, Any] = {
            "animal_name": str(animal_name),
            "date": str(date),
            "decode_epoch": str(decode_epoch),
            "label_scheme": str(label_scheme_name),
            "decoder": str(decoder_name),
            "region": str(region),
            "n_events": int(n_events),
            "effective_n_splits": int(effective_n_splits),
            "min_class_count": int(np.min(class_counts)),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }
        confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
        for true_index, true_name in enumerate(class_names):
            summary_row[f"recall_{true_name}"] = float(metrics[f"recall_{true_name}"])
            for pred_index, pred_name in enumerate(class_names):
                summary_row[f"confusion_{true_name}_to_{pred_name}"] = int(
                    confusion[true_index, pred_index]
                )
        summary_rows.append(summary_row)
    summary_table = pd.DataFrame(summary_rows)
    return prediction_table, summary_table, metrics_by_region


def classify_ripples(
    *,
    spikes_by_region: dict[str, Any],
    unit_ids_by_region: dict[str, np.ndarray],
    models_by_region: dict[str, dict[str, np.ndarray]],
    ripple_table: pd.DataFrame,
    epoch_interval: Any,
    bin_size_s: float,
    decoder_name: str,
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    class_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ripple_rows: list[dict[str, Any]] = []
    skipped_ripples: list[dict[str, Any]] = []
    ripple_chunks_by_region: dict[str, list[Any]] = {region: [] for region in REGIONS}
    ripple_starts: list[float] = []
    ripple_ends: list[float] = []
    ripple_source_indices: list[int] = []
    posterior_cube: list[np.ndarray] = []
    probability_columns = get_probability_columns(class_names)
    score_columns = get_score_columns(
        decoder_name=decoder_name,
        class_names=class_names,
    )

    for ripple_source_index, ripple_row in ripple_table.reset_index(drop=True).iterrows():
        ripple_ep = make_intervalset_from_bounds(
            np.array([float(ripple_row["start_time"])], dtype=float),
            np.array([float(ripple_row["end_time"])], dtype=float),
        ).intersect(epoch_interval)
        if float(ripple_ep.tot_length()) <= 0.0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": "No overlap with decode epoch interval.",
                }
            )
            continue

        region_counts: dict[str, np.ndarray] = {}
        region_times: dict[str, np.ndarray] = {}
        invalid_reason: str | None = None
        for region in REGIONS:
            count_matrix, count_times = count_spikes_in_interval(
                spikes_by_region[region],
                unit_ids_by_region[region],
                ripple_ep,
                bin_size_s=bin_size_s,
                label=region.upper(),
            )
            if count_matrix.shape[0] == 0:
                invalid_reason = f"{region.upper()} counting produced zero bins."
                break
            region_counts[region] = count_matrix
            region_times[region] = count_times

        if invalid_reason is not None:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": invalid_reason,
                }
            )
            continue

        ca1_times = np.asarray(region_times["ca1"], dtype=float)
        v1_times = np.asarray(region_times["v1"], dtype=float)
        if ca1_times.shape != v1_times.shape or not np.allclose(
            ca1_times,
            v1_times,
            rtol=1e-9,
            atol=1e-9,
        ):
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": "CA1 and V1 ripple bins did not align.",
                }
            )
            continue

        ripple_start_time_s, ripple_end_time_s = _first_last_interval_times(ripple_ep)
        row: dict[str, Any] = {
            "animal_name": str(animal_name),
            "date": str(date),
            "decode_epoch": str(decode_epoch),
            "label_scheme": str(label_scheme_name),
            "decoder": str(decoder_name),
            "ripple_id": int(len(ripple_rows)),
            "ripple_source_index": int(ripple_source_index),
            "ripple_start_time_s": float(ripple_start_time_s),
            "ripple_end_time_s": float(ripple_end_time_s),
            "duration_s": float(ripple_ep.tot_length()),
            "n_bins": int(ca1_times.size),
        }
        per_region_posteriors: list[np.ndarray] = []
        for region in REGIONS:
            posterior, scores = score_decoder(
                decoder_name=decoder_name,
                count_matrix=region_counts[region],
                model=models_by_region[region],
                bin_size_s=bin_size_s,
            )
            summary = summarize_posterior(
                posterior,
                scores,
                decoder_name=decoder_name,
                class_names=class_names,
            )
            row[f"{region}_predicted_class_index"] = int(summary["predicted_class_index"])
            row[f"{region}_predicted_class"] = str(summary["predicted_class"])
            row[f"{region}_max_posterior"] = float(summary["max_posterior"])
            row[f"{region}_margin_to_second"] = float(summary["margin_to_second"])
            row[f"{region}_entropy"] = float(summary["entropy"])
            for column_name in probability_columns:
                row[f"{region}_{column_name}"] = float(summary[column_name])
            for column_name in score_columns:
                row[f"{region}_{column_name}"] = float(summary[column_name])
            per_region_posteriors.append(np.asarray(posterior, dtype=float))

        ca1_posterior = per_region_posteriors[REGIONS.index("ca1")]
        v1_posterior = per_region_posteriors[REGIONS.index("v1")]
        ca1_class_index = int(row["ca1_predicted_class_index"])
        v1_class_index = int(row["v1_predicted_class_index"])
        row["same_label"] = bool(ca1_class_index == v1_class_index)
        row["js_distance"] = float(jensenshannon(ca1_posterior, v1_posterior))
        row["v1_posterior_for_ca1_class"] = float(v1_posterior[ca1_class_index])
        row["ca1_posterior_for_v1_class"] = float(ca1_posterior[v1_class_index])
        row["max_posterior_difference"] = float(
            row["v1_max_posterior"] - row["ca1_max_posterior"]
        )
        row["entropy_difference"] = float(row["v1_entropy"] - row["ca1_entropy"])
        row["abs_max_posterior_difference"] = float(abs(row["max_posterior_difference"]))
        row["abs_entropy_difference"] = float(abs(row["entropy_difference"]))
        ripple_rows.append(row)
        posterior_cube.append(np.stack(per_region_posteriors, axis=0))
        ripple_chunks_by_region["ca1"].append(
            make_empty_tsd(time_support=ripple_ep)
            if ca1_times.size == 0
            else _make_label_tsd(ca1_times, np.full(ca1_times.size, ca1_class_index), ripple_ep)
        )
        ripple_chunks_by_region["v1"].append(
            make_empty_tsd(time_support=ripple_ep)
            if v1_times.size == 0
            else _make_label_tsd(v1_times, np.full(v1_times.size, v1_class_index), ripple_ep)
        )
        ripple_starts.append(float(ripple_start_time_s))
        ripple_ends.append(float(ripple_end_time_s))
        ripple_source_indices.append(int(ripple_source_index))

    ripple_support = make_intervalset_from_bounds(
        np.asarray(ripple_starts, dtype=float),
        np.asarray(ripple_ends, dtype=float),
    )
    decoded_tsds = {
        region: concatenate_tsds(ripple_chunks_by_region[region], ripple_support)
        for region in REGIONS
    }
    metadata = {
        "decoded_label_tsds": decoded_tsds,
        "ripple_source_indices": np.asarray(ripple_source_indices, dtype=int),
        "posterior_cube": (
            np.stack(posterior_cube, axis=1).astype(float, copy=False)
            if posterior_cube
            else np.empty((len(REGIONS), 0, len(class_names)), dtype=float)
        ),
        "skipped_ripples": skipped_ripples,
    }
    return pd.DataFrame(ripple_rows), metadata


def _make_label_tsd(times: np.ndarray, labels: np.ndarray, time_support: Any) -> Any:
    import pynapple as nap

    return nap.Tsd(
        t=np.asarray(times, dtype=float),
        d=np.asarray(labels, dtype=float),
        time_support=time_support,
        time_units="s",
    )


def deranged_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=int)
    indices = np.arange(size, dtype=int)
    for _ in range(128):
        permutation = rng.permutation(size)
        if not np.any(permutation == indices):
            return permutation
    return np.roll(indices, 1)


def empirical_p_value(
    observed: float,
    null_samples: np.ndarray,
    *,
    higher_is_better: bool,
) -> float:
    values = np.asarray(null_samples, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(observed):
        return np.nan
    if higher_is_better:
        extreme = finite >= observed
    else:
        extreme = finite <= observed
    return float((np.sum(extreme) + 1) / (finite.size + 1))


def summarize_ripple_table_against_shuffle(
    *,
    ripple_table: pd.DataFrame,
    n_shuffles: int,
    shuffle_seed: int,
    class_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray], int]:
    if ripple_table.empty:
        summary = {
            "same_label_fraction": np.nan,
            "mean_js_distance": np.nan,
            "mean_ca1_max_posterior": np.nan,
            "mean_v1_max_posterior": np.nan,
            "mean_v1_posterior_for_ca1_class": np.nan,
            "same_label_fraction_shuffle_mean": np.nan,
            "same_label_fraction_shuffle_sd": np.nan,
            "same_label_fraction_p_value": np.nan,
            "mean_js_distance_shuffle_mean": np.nan,
            "mean_js_distance_shuffle_sd": np.nan,
            "mean_js_distance_p_value": np.nan,
            "mean_v1_posterior_for_ca1_class_shuffle_mean": np.nan,
            "mean_v1_posterior_for_ca1_class_shuffle_sd": np.nan,
            "mean_v1_posterior_for_ca1_class_p_value": np.nan,
        }
        return summary, {"same_label_fraction": np.array([])}, 0

    ca1_labels = ripple_table["ca1_predicted_class_index"].to_numpy(dtype=int)
    v1_labels = ripple_table["v1_predicted_class_index"].to_numpy(dtype=int)
    probability_columns = get_probability_columns(class_names)
    ca1_posterior = ripple_table[
        [f"ca1_{column_name}" for column_name in probability_columns]
    ].to_numpy(dtype=float)
    v1_posterior = ripple_table[
        [f"v1_{column_name}" for column_name in probability_columns]
    ].to_numpy(dtype=float)

    observed_same_label_fraction = float(np.mean(ca1_labels == v1_labels))
    observed_mean_js_distance = float(
        np.mean(
            [
                jensenshannon(ca1_row, v1_row)
                for ca1_row, v1_row in zip(ca1_posterior, v1_posterior)
            ]
        )
    )
    observed_mean_ca1_support = float(
        np.mean(v1_posterior[np.arange(len(ripple_table)), ca1_labels])
    )

    null_samples = {
        "same_label_fraction": np.full(int(n_shuffles), np.nan, dtype=float),
        "mean_js_distance": np.full(int(n_shuffles), np.nan, dtype=float),
        "mean_v1_posterior_for_ca1_class": np.full(int(n_shuffles), np.nan, dtype=float),
    }
    effective_shuffles = 0
    if len(ripple_table) >= 2 and n_shuffles > 0:
        rng = np.random.default_rng(int(shuffle_seed))
        for shuffle_index in range(int(n_shuffles)):
            permutation = deranged_permutation(len(ripple_table), rng)
            shuffled_v1_posterior = v1_posterior[permutation]
            shuffled_v1_labels = np.argmax(shuffled_v1_posterior, axis=1)
            null_samples["same_label_fraction"][shuffle_index] = float(
                np.mean(ca1_labels == shuffled_v1_labels)
            )
            null_samples["mean_js_distance"][shuffle_index] = float(
                np.mean(
                    [
                        jensenshannon(ca1_row, v1_row)
                        for ca1_row, v1_row in zip(ca1_posterior, shuffled_v1_posterior)
                    ]
                )
            )
            null_samples["mean_v1_posterior_for_ca1_class"][shuffle_index] = float(
                np.mean(shuffled_v1_posterior[np.arange(len(ripple_table)), ca1_labels])
            )
            effective_shuffles += 1

    summary = {
        "same_label_fraction": observed_same_label_fraction,
        "mean_js_distance": observed_mean_js_distance,
        "mean_ca1_max_posterior": float(
            ripple_table["ca1_max_posterior"].mean()
        ),
        "mean_v1_max_posterior": float(
            ripple_table["v1_max_posterior"].mean()
        ),
        "mean_v1_posterior_for_ca1_class": observed_mean_ca1_support,
        "same_label_fraction_shuffle_mean": float(
            np.nanmean(null_samples["same_label_fraction"])
        )
        if effective_shuffles > 0
        else np.nan,
        "same_label_fraction_shuffle_sd": float(
            np.nanstd(null_samples["same_label_fraction"], ddof=0)
        )
        if effective_shuffles > 0
        else np.nan,
        "same_label_fraction_p_value": empirical_p_value(
            observed_same_label_fraction,
            null_samples["same_label_fraction"],
            higher_is_better=True,
        ),
        "mean_js_distance_shuffle_mean": float(
            np.nanmean(null_samples["mean_js_distance"])
        )
        if effective_shuffles > 0
        else np.nan,
        "mean_js_distance_shuffle_sd": float(
            np.nanstd(null_samples["mean_js_distance"], ddof=0)
        )
        if effective_shuffles > 0
        else np.nan,
        "mean_js_distance_p_value": empirical_p_value(
            observed_mean_js_distance,
            null_samples["mean_js_distance"],
            higher_is_better=False,
        ),
        "mean_v1_posterior_for_ca1_class_shuffle_mean": float(
            np.nanmean(null_samples["mean_v1_posterior_for_ca1_class"])
        )
        if effective_shuffles > 0
        else np.nan,
        "mean_v1_posterior_for_ca1_class_shuffle_sd": float(
            np.nanstd(null_samples["mean_v1_posterior_for_ca1_class"], ddof=0)
        )
        if effective_shuffles > 0
        else np.nan,
        "mean_v1_posterior_for_ca1_class_p_value": empirical_p_value(
            observed_mean_ca1_support,
            null_samples["mean_v1_posterior_for_ca1_class"],
            higher_is_better=True,
        ),
    }
    return summary, null_samples, int(effective_shuffles)


def build_epoch_summary_table(
    *,
    behavior_summary: pd.DataFrame,
    ripple_summary: dict[str, float],
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    decoder_name: str,
    class_names: tuple[str, ...],
    bin_size_s: float,
    n_events: int,
    n_ripples: int,
    n_effective_shuffles: int,
) -> pd.DataFrame:
    summary_row: dict[str, Any] = {
        "animal_name": str(animal_name),
        "date": str(date),
        "decode_epoch": str(decode_epoch),
        "label_scheme": str(label_scheme_name),
        "decoder": str(decoder_name),
        "bin_size_s": float(bin_size_s),
        "n_behavior_events": int(n_events),
        "n_ripples": int(n_ripples),
        "n_effective_shuffles": int(n_effective_shuffles),
    }
    for region in REGIONS:
        region_row = behavior_summary.loc[behavior_summary["region"] == region]
        if region_row.empty:
            continue
        row = region_row.iloc[0]
        summary_row[f"{region}_accuracy"] = float(row["accuracy"])
        summary_row[f"{region}_balanced_accuracy"] = float(row["balanced_accuracy"])
        summary_row[f"{region}_macro_f1"] = float(row["macro_f1"])
        summary_row[f"{region}_effective_n_splits"] = int(row["effective_n_splits"])
        for class_name in class_names:
            summary_row[f"{region}_recall_{class_name}"] = float(row[f"recall_{class_name}"])
    summary_row.update({key: make_json_safe(value) for key, value in ripple_summary.items()})
    return pd.DataFrame([summary_row])


def require_xarray():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `xarray` to save raw ripple trajectory outputs as NetCDF."
        ) from exc
    return xr


def build_epoch_dataset(
    *,
    ripple_table: pd.DataFrame,
    ripple_metadata: dict[str, Any],
    ca1_mask_table: pd.DataFrame,
    v1_mask_table: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    epoch_summary: pd.DataFrame,
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    decoder_name: str,
    class_names: tuple[str, ...],
    bin_size_s: float,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> Any:
    xr = require_xarray()
    n_ripples = int(len(ripple_table))
    posterior_cube = np.asarray(ripple_metadata["posterior_cube"], dtype=float)
    if posterior_cube.shape != (len(REGIONS), n_ripples, len(class_names)):
        raise ValueError(
            f"Posterior cube had unexpected shape {posterior_cube.shape}; "
            f"expected {(len(REGIONS), n_ripples, len(class_names))}."
        )

    data_vars: dict[str, Any] = {
        "ripple_posterior": (("region", "ripple", "class"), posterior_cube),
        "ripple_start_time_s": (
            ("ripple",),
            ripple_table["ripple_start_time_s"].to_numpy(dtype=float)
            if n_ripples
            else np.array([], dtype=float),
        ),
        "ripple_end_time_s": (
            ("ripple",),
            ripple_table["ripple_end_time_s"].to_numpy(dtype=float)
            if n_ripples
            else np.array([], dtype=float),
        ),
        "ripple_n_bins": (
            ("ripple",),
            ripple_table["n_bins"].to_numpy(dtype=int) if n_ripples else np.array([], dtype=int),
        ),
        "ripple_source_index": (
            ("ripple",),
            ripple_table["ripple_source_index"].to_numpy(dtype=int)
            if n_ripples
            else np.array([], dtype=int),
        ),
        "ripple_same_label": (
            ("ripple",),
            ripple_table["same_label"].to_numpy(dtype=bool)
            if n_ripples
            else np.array([], dtype=bool),
        ),
        "ripple_js_distance": (
            ("ripple",),
            ripple_table["js_distance"].to_numpy(dtype=float)
            if n_ripples
            else np.array([], dtype=float),
        ),
        "ca1_unit_id": (("ca1_unit",), ca1_mask_table["unit_id"].to_numpy()),
        "ca1_keep_unit": (
            ("ca1_unit",),
            ca1_mask_table["keep_unit"].to_numpy(dtype=bool),
        ),
        "ca1_movement_firing_rate_hz": (
            ("ca1_unit",),
            ca1_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "v1_unit_id": (("v1_unit",), v1_mask_table["unit_id"].to_numpy()),
        "v1_keep_unit": (
            ("v1_unit",),
            v1_mask_table["keep_unit"].to_numpy(dtype=bool),
        ),
        "v1_movement_firing_rate_hz": (
            ("v1_unit",),
            v1_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
    }
    if n_ripples:
        for region in REGIONS:
            data_vars[f"{region}_predicted_class_index"] = (
                ("ripple",),
                ripple_table[f"{region}_predicted_class_index"].to_numpy(dtype=int),
            )
            data_vars[f"{region}_max_posterior"] = (
                ("ripple",),
                ripple_table[f"{region}_max_posterior"].to_numpy(dtype=float),
            )
            data_vars[f"{region}_margin_to_second"] = (
                ("ripple",),
                ripple_table[f"{region}_margin_to_second"].to_numpy(dtype=float),
            )
            data_vars[f"{region}_entropy"] = (
                ("ripple",),
                ripple_table[f"{region}_entropy"].to_numpy(dtype=float),
            )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "region": np.asarray(REGIONS, dtype=object),
            "ripple": np.arange(n_ripples, dtype=int),
            "class": np.asarray(class_names, dtype=object),
            "ca1_unit": np.arange(len(ca1_mask_table), dtype=int),
            "v1_unit": np.arange(len(v1_mask_table), dtype=int),
        },
        attrs={
            "animal_name": str(animal_name),
            "date": str(date),
            "decode_epoch": str(decode_epoch),
            "label_scheme": str(label_scheme_name),
            "decoder": str(decoder_name),
            "bin_size_s": float(bin_size_s),
            "sources_json": json.dumps(make_json_safe(sources), sort_keys=True),
            "fit_parameters_json": json.dumps(make_json_safe(fit_parameters), sort_keys=True),
            "behavior_summary_json": behavior_summary.to_json(orient="records"),
            "epoch_summary_json": epoch_summary.to_json(orient="records"),
        },
    )
    return dataset


def format_bin_size_token(bin_size_s: float) -> str:
    token = np.format_float_positional(float(bin_size_s), trim="-")
    return f"binsize-{token.replace('.', 'p')}s"


def build_output_stem(
    *,
    decoder_name: str,
    label_scheme_name: str,
    decode_epoch: str,
    bin_size_s: float,
) -> str:
    return (
        f"trajectory_identity_{decoder_name}_{label_scheme_name}_decode-{decode_epoch}_"
        f"{format_bin_size_token(bin_size_s)}"
    )


def build_dataset_name(
    *,
    decoder_name: str,
    label_scheme_name: str,
    decode_epoch: str,
    bin_size_s: float,
) -> str:
    return (
        f"{build_output_stem(decoder_name=decoder_name, label_scheme_name=label_scheme_name, decode_epoch=decode_epoch, bin_size_s=bin_size_s)}"
        "_dataset.nc"
    )


def _load_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_behavior_confusions(
    *,
    metrics_by_region: dict[str, Any],
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    decoder_name: str,
    class_names: tuple[str, ...],
    out_path: Path,
) -> Path:
    plt = _load_plotting()
    figure, axes = plt.subplots(1, len(REGIONS), figsize=(12, 5), constrained_layout=True)
    if len(REGIONS) == 1:
        axes = [axes]
    for axis, region in zip(axes, REGIONS):
        confusion = np.asarray(metrics_by_region[region]["confusion_matrix"], dtype=float)
        image = axis.imshow(confusion, cmap="Blues", aspect="equal")
        axis.set_xticks(np.arange(len(class_names)))
        axis.set_yticks(np.arange(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right")
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{region.upper()} CV confusion\n"
            f"acc={metrics_by_region[region]['accuracy']:.3f}, "
            f"bal={metrics_by_region[region]['balanced_accuracy']:.3f}"
        )
        for row_index in range(confusion.shape[0]):
            for col_index in range(confusion.shape[1]):
                axis.text(
                    col_index,
                    row_index,
                    int(confusion[row_index, col_index]),
                    ha="center",
                    va="center",
                    color="black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{animal_name} {date} {label_scheme_name} {decoder_name} CV decode {decode_epoch}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def plot_ripple_label_contingency(
    *,
    ripple_table: pd.DataFrame,
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    decoder_name: str,
    class_names: tuple[str, ...],
    out_path: Path,
) -> Path:
    plt = _load_plotting()
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    contingency = confusion_matrix(
        ripple_table["ca1_predicted_class_index"].to_numpy(dtype=int),
        ripple_table["v1_predicted_class_index"].to_numpy(dtype=int),
        labels=np.arange(len(class_names), dtype=int),
    )
    image = axis.imshow(contingency, cmap="Greens", aspect="equal")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("V1 ripple label")
    axis.set_ylabel("CA1 ripple label")
    axis.set_title("Ripple label contingency")
    for row_index in range(contingency.shape[0]):
        for col_index in range(contingency.shape[1]):
            axis.text(
                col_index,
                row_index,
                int(contingency[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{animal_name} {date} {label_scheme_name} {decoder_name} ripple labels {decode_epoch}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def plot_confidence_scatter(
    *,
    ripple_table: pd.DataFrame,
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    decoder_name: str,
    out_path: Path,
) -> Path:
    plt = _load_plotting()
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    same_label = ripple_table["same_label"].to_numpy(dtype=bool)
    colors = np.where(same_label, "tab:green", "tab:gray")
    axis.scatter(
        ripple_table["ca1_max_posterior"].to_numpy(dtype=float),
        ripple_table["v1_max_posterior"].to_numpy(dtype=float),
        c=colors,
        alpha=0.8,
        edgecolor="none",
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.5", linewidth=1)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("CA1 max posterior")
    axis.set_ylabel("V1 max posterior")
    axis.set_title("Ripple confidence comparison")
    figure.suptitle(
        f"{animal_name} {date} {label_scheme_name} {decoder_name} ripple confidence {decode_epoch}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def plot_same_label_shuffle(
    *,
    observed_value: float,
    shuffle_values: np.ndarray,
    p_value: float,
    animal_name: str,
    date: str,
    decode_epoch: str,
    label_scheme_name: str,
    decoder_name: str,
    out_path: Path,
) -> Path:
    plt = _load_plotting()
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    finite_shuffle = np.asarray(shuffle_values, dtype=float)
    finite_shuffle = finite_shuffle[np.isfinite(finite_shuffle)]
    if finite_shuffle.size:
        axis.hist(finite_shuffle, bins=min(20, max(5, finite_shuffle.size)), color="0.75")
        axis.axvline(float(observed_value), color="tab:red", linewidth=2)
    else:
        axis.text(0.5, 0.5, "No effective shuffles", ha="center", va="center")
    axis.set_xlabel("Same-label fraction")
    axis.set_ylabel("Count")
    axis.set_title(
        f"Observed vs shuffle\np={p_value:.3g}" if np.isfinite(p_value) else "Observed vs shuffle"
    )
    figure.suptitle(
        f"{animal_name} {date} {label_scheme_name} {decoder_name} same-label shuffle {decode_epoch}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def get_log_dir(analysis_path: Path) -> Path:
    return analysis_path / "ripple_trajectory_identity" / "logs"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify ripple trajectory identity separately in CA1 and V1."
    )
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Session date in YYYYMMDD format",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--decode-epochs",
        nargs="+",
        help="Optional run epochs to decode. Default: all run epochs.",
    )
    parser.add_argument(
        "--label-schemes",
        nargs="+",
        choices=tuple(LABEL_SCHEMES),
        default=list(LABEL_SCHEMES),
        help=(
            "Trajectory label schemes to decode. "
            "Default: trajectory turn_group arm"
        ),
    )
    parser.add_argument(
        "--decoder",
        choices=DECODER_NAMES,
        default=DEFAULT_DECODER,
        help=(
            "Decoder used for both behavioral CV and ripple classification. "
            f"Default: {DEFAULT_DECODER}"
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Time bin size in seconds for traversal and ripple counts. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Requested number of traversal CV folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help=f"Number of ripple-level shuffle permutations. Default: {DEFAULT_N_SHUFFLES}",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for traversal CV and ripple shuffles. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    parser.add_argument(
        "--ca1-min-movement-fr-hz",
        type=float,
        default=DEFAULT_CA1_MIN_MOVEMENT_FR_HZ,
        help=(
            "Minimum CA1 firing rate during movement used for unit masking. "
            f"Default: {DEFAULT_CA1_MIN_MOVEMENT_FR_HZ}"
        ),
    )
    parser.add_argument(
        "--v1-min-movement-fr-hz",
        type=float,
        default=DEFAULT_V1_MIN_MOVEMENT_FR_HZ,
        help=(
            "Minimum V1 firing rate during movement used for unit masking. "
            f"Default: {DEFAULT_V1_MIN_MOVEMENT_FR_HZ}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    label_schemes = resolve_label_schemes(args.label_schemes)

    analysis_path = get_analysis_path(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    decode_epochs = select_decode_epochs(session["run_epochs"], args.decode_epochs)
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    epoch_intervals = build_epoch_intervals(session["timestamps_ephys_by_epoch"])
    ripple_tables, ripple_source = load_ripple_tables(analysis_path)

    data_dir = analysis_path / "ripple_trajectory_identity"
    fig_dir = analysis_path / "figs" / "ripple_trajectory_identity"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sources = dict(session["sources"])
    sources["ripple_events"] = ripple_source
    fit_parameters = {
        "animal_name": args.animal_name,
        "date": args.date,
        "decode_epochs": decode_epochs,
        "label_schemes": label_schemes,
        "decoder": args.decoder,
        "bin_size_s": args.bin_size_s,
        "n_splits": args.n_splits,
        "n_shuffles": args.n_shuffles,
        "shuffle_seed": args.shuffle_seed,
        "ca1_min_movement_fr_hz": args.ca1_min_movement_fr_hz,
        "v1_min_movement_fr_hz": args.v1_min_movement_fr_hz,
        "label_scheme_definitions": make_json_safe(LABEL_SCHEMES),
        "comparison_direction": "ca1_vs_v1_trajectory_identity",
    }

    saved_behavior_prediction_paths: list[Path] = []
    saved_behavior_summary_paths: list[Path] = []
    saved_ripple_table_paths: list[Path] = []
    saved_epoch_summary_paths: list[Path] = []
    saved_dataset_paths: list[Path] = []
    saved_decoded_paths: list[Path] = []
    saved_figure_paths: list[Path] = []
    successful_jobs: list[dict[str, Any]] = []
    skipped_epochs: list[dict[str, Any]] = []

    for decode_epoch in decode_epochs:
        ripple_table = ripple_tables.get(decode_epoch, pd.DataFrame())
        if ripple_table.empty:
            skipped_epochs.append(
                {
                    "decode_epoch": decode_epoch,
                    "reason": "No ripple events were found for this epoch.",
                }
            )
            print(f"Skipping {decode_epoch}: no ripple events")
            continue

        ca1_mask_table = build_region_unit_mask_table(
            unit_ids=get_tsgroup_unit_ids(session["spikes_by_region"]["ca1"]),
            movement_firing_rates_hz=np.asarray(
                movement_firing_rates["ca1"][decode_epoch],
                dtype=float,
            ),
            min_movement_fr_hz=args.ca1_min_movement_fr_hz,
            region="ca1",
        )
        v1_mask_table = build_region_unit_mask_table(
            unit_ids=get_tsgroup_unit_ids(session["spikes_by_region"]["v1"]),
            movement_firing_rates_hz=np.asarray(
                movement_firing_rates["v1"][decode_epoch],
                dtype=float,
            ),
            min_movement_fr_hz=args.v1_min_movement_fr_hz,
            region="v1",
        )
        ca1_keep_unit_ids = ca1_mask_table.loc[ca1_mask_table["keep_unit"], "unit_id"].tolist()
        v1_keep_unit_ids = v1_mask_table.loc[v1_mask_table["keep_unit"], "unit_id"].tolist()
        if not ca1_keep_unit_ids or not v1_keep_unit_ids:
            skipped_epochs.append(
                {
                    "decode_epoch": decode_epoch,
                    "reason": "No CA1 or V1 units passed the movement firing-rate threshold.",
                }
            )
            print(f"Skipping {decode_epoch}: no CA1 or V1 units passed movement-rate filtering")
            continue

        spikes_by_region = {
            "ca1": _subset_spikes(session["spikes_by_region"]["ca1"], ca1_keep_unit_ids),
            "v1": _subset_spikes(session["spikes_by_region"]["v1"], v1_keep_unit_ids),
        }
        unit_ids_by_region = {
            "ca1": np.asarray(ca1_keep_unit_ids),
            "v1": np.asarray(v1_keep_unit_ids),
        }

        for label_scheme_name in label_schemes:
            label_scheme = LABEL_SCHEMES[label_scheme_name]
            class_names = tuple(label_scheme["class_names"])
            label_by_trajectory = dict(label_scheme["label_by_trajectory"])

            behavioral_events, skipped_behavioral_events = build_behavioral_events(
                spikes_by_region=spikes_by_region,
                unit_ids_by_region=unit_ids_by_region,
                trajectory_intervals=session["trajectory_intervals"][decode_epoch],
                movement_interval=session["movement_by_run"][decode_epoch],
                bin_size_s=args.bin_size_s,
                label_scheme_name=label_scheme_name,
                class_names=class_names,
                label_by_trajectory=label_by_trajectory,
            )
            if not behavioral_events:
                skipped_epochs.append(
                    {
                        "decode_epoch": decode_epoch,
                        "label_scheme": label_scheme_name,
                        "reason": "No behavioral traversal events were available after movement filtering.",
                        "skipped_behavioral_events": skipped_behavioral_events,
                    }
                )
                print(
                    f"Skipping {decode_epoch} {label_scheme_name}: "
                    "no usable behavioral traversal events"
                )
                continue

            try:
                behavior_predictions, behavior_summary, behavior_metrics = build_behavior_cv_outputs(
                    events=behavioral_events,
                    decoder_name=args.decoder,
                    n_splits=args.n_splits,
                    shuffle_seed=args.shuffle_seed,
                    bin_size_s=args.bin_size_s,
                    animal_name=args.animal_name,
                    date=args.date,
                    decode_epoch=decode_epoch,
                    label_scheme_name=label_scheme_name,
                    class_names=class_names,
                )
            except Exception as exc:
                skipped_epochs.append(
                    {
                        "decode_epoch": decode_epoch,
                        "label_scheme": label_scheme_name,
                        "reason": "Behavioral trajectory CV failed.",
                        "error": str(exc),
                        "skipped_behavioral_events": skipped_behavioral_events,
                    }
                )
                print(
                    f"Skipping {decode_epoch} {label_scheme_name}: "
                    f"behavioral trajectory CV failed: {exc}"
                )
                continue

            final_models = {
                region: fit_decoder(
                    decoder_name=args.decoder,
                    count_matrices=[event["region_counts"][region] for event in behavioral_events],
                    labels=np.asarray(
                        [event["class_index"] for event in behavioral_events],
                        dtype=int,
                    ),
                    n_classes=len(class_names),
                    bin_size_s=args.bin_size_s,
                )
                for region in REGIONS
            }
            ripple_predictions, ripple_metadata = classify_ripples(
                spikes_by_region=spikes_by_region,
                unit_ids_by_region=unit_ids_by_region,
                models_by_region=final_models,
                ripple_table=ripple_table,
                epoch_interval=epoch_intervals[decode_epoch],
                bin_size_s=args.bin_size_s,
                decoder_name=args.decoder,
                animal_name=args.animal_name,
                date=args.date,
                decode_epoch=decode_epoch,
                label_scheme_name=label_scheme_name,
                class_names=class_names,
            )
            if ripple_predictions.empty:
                skipped_epochs.append(
                    {
                        "decode_epoch": decode_epoch,
                        "label_scheme": label_scheme_name,
                        "reason": "No usable ripple events were available after binning.",
                        "skipped_behavioral_events": skipped_behavioral_events,
                        "skipped_ripples": ripple_metadata["skipped_ripples"],
                    }
                )
                print(
                    f"Skipping {decode_epoch} {label_scheme_name}: "
                    "no usable ripple events after binning"
                )
                continue

            ripple_summary, shuffle_samples, effective_shuffles = (
                summarize_ripple_table_against_shuffle(
                    ripple_table=ripple_predictions,
                    n_shuffles=args.n_shuffles,
                    shuffle_seed=args.shuffle_seed,
                    class_names=class_names,
                )
            )
            epoch_summary = build_epoch_summary_table(
                behavior_summary=behavior_summary,
                ripple_summary=ripple_summary,
                animal_name=args.animal_name,
                date=args.date,
                decode_epoch=decode_epoch,
                label_scheme_name=label_scheme_name,
                decoder_name=args.decoder,
                class_names=class_names,
                bin_size_s=args.bin_size_s,
                n_events=len(behavioral_events),
                n_ripples=len(ripple_predictions),
                n_effective_shuffles=effective_shuffles,
            )

            output_stem = build_output_stem(
                decoder_name=args.decoder,
                label_scheme_name=label_scheme_name,
                decode_epoch=decode_epoch,
                bin_size_s=args.bin_size_s,
            )
            behavior_prediction_path = data_dir / f"{output_stem}_behavior_cv_predictions.parquet"
            behavior_summary_path = data_dir / f"{output_stem}_behavior_summary.parquet"
            ripple_table_path = data_dir / f"{output_stem}_ripple_classification.parquet"
            epoch_summary_path = data_dir / f"{output_stem}_epoch_summary.parquet"
            dataset_path = data_dir / build_dataset_name(
                decoder_name=args.decoder,
                label_scheme_name=label_scheme_name,
                decode_epoch=decode_epoch,
                bin_size_s=args.bin_size_s,
            )

            behavior_predictions.to_parquet(behavior_prediction_path, index=False)
            behavior_summary.to_parquet(behavior_summary_path, index=False)
            ripple_predictions.to_parquet(ripple_table_path, index=False)
            epoch_summary.to_parquet(epoch_summary_path, index=False)

            saved_behavior_prediction_paths.append(behavior_prediction_path)
            saved_behavior_summary_paths.append(behavior_summary_path)
            saved_ripple_table_paths.append(ripple_table_path)
            saved_epoch_summary_paths.append(epoch_summary_path)

            dataset_saved = False
            try:
                epoch_dataset = build_epoch_dataset(
                    ripple_table=ripple_predictions,
                    ripple_metadata=ripple_metadata,
                    ca1_mask_table=ca1_mask_table,
                    v1_mask_table=v1_mask_table,
                    behavior_summary=behavior_summary,
                    epoch_summary=epoch_summary,
                    animal_name=args.animal_name,
                    date=args.date,
                    decode_epoch=decode_epoch,
                    label_scheme_name=label_scheme_name,
                    decoder_name=args.decoder,
                    class_names=class_names,
                    bin_size_s=args.bin_size_s,
                    sources=sources,
                    fit_parameters=fit_parameters,
                )
                epoch_dataset.to_netcdf(dataset_path)
                saved_dataset_paths.append(dataset_path)
                dataset_saved = True
            except ModuleNotFoundError:
                dataset_path = None

            for region in REGIONS:
                decoded_path = data_dir / f"{output_stem}_{region}_ripple_labels.npz"
                ripple_metadata["decoded_label_tsds"][region].save(decoded_path)
                saved_decoded_paths.append(decoded_path)

            behavior_confusion_figure = plot_behavior_confusions(
                metrics_by_region=behavior_metrics,
                animal_name=args.animal_name,
                date=args.date,
                decode_epoch=decode_epoch,
                label_scheme_name=label_scheme_name,
                decoder_name=args.decoder,
                class_names=class_names,
                out_path=fig_dir / f"{output_stem}_behavior_confusions.png",
            )
            contingency_figure = plot_ripple_label_contingency(
                ripple_table=ripple_predictions,
                animal_name=args.animal_name,
                date=args.date,
                decode_epoch=decode_epoch,
                label_scheme_name=label_scheme_name,
                decoder_name=args.decoder,
                class_names=class_names,
                out_path=fig_dir / f"{output_stem}_ripple_contingency.png",
            )
            confidence_figure = plot_confidence_scatter(
                ripple_table=ripple_predictions,
                animal_name=args.animal_name,
                date=args.date,
                decode_epoch=decode_epoch,
                label_scheme_name=label_scheme_name,
                decoder_name=args.decoder,
                out_path=fig_dir / f"{output_stem}_confidence_scatter.png",
            )
            shuffle_figure = plot_same_label_shuffle(
                observed_value=float(ripple_summary["same_label_fraction"]),
                shuffle_values=np.asarray(shuffle_samples["same_label_fraction"], dtype=float),
                p_value=float(ripple_summary["same_label_fraction_p_value"]),
                animal_name=args.animal_name,
                date=args.date,
                decode_epoch=decode_epoch,
                label_scheme_name=label_scheme_name,
                decoder_name=args.decoder,
                out_path=fig_dir / f"{output_stem}_same_label_shuffle.png",
            )
            saved_figure_paths.extend(
                [
                    behavior_confusion_figure,
                    contingency_figure,
                    confidence_figure,
                    shuffle_figure,
                ]
            )

            successful_jobs.append(
                {
                    "decode_epoch": decode_epoch,
                    "label_scheme": label_scheme_name,
                    "decoder": args.decoder,
                    "class_names": list(class_names),
                    "n_behavior_events": int(len(behavioral_events)),
                    "n_ripples": int(len(ripple_predictions)),
                    "n_ca1_units": int(len(ca1_keep_unit_ids)),
                    "n_v1_units": int(len(v1_keep_unit_ids)),
                    "behavior_prediction_path": behavior_prediction_path,
                    "behavior_summary_path": behavior_summary_path,
                    "ripple_table_path": ripple_table_path,
                    "epoch_summary_path": epoch_summary_path,
                    "dataset_path": dataset_path if dataset_saved else None,
                    "behavior_confusion_figure": behavior_confusion_figure,
                    "contingency_figure": contingency_figure,
                    "confidence_figure": confidence_figure,
                    "shuffle_figure": shuffle_figure,
                    "skipped_behavioral_events": skipped_behavioral_events,
                    "skipped_ripples": ripple_metadata["skipped_ripples"],
                }
            )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name=Path(__file__).stem,
        parameters=fit_parameters,
        outputs={
            "sources": sources,
            "saved_behavior_prediction_paths": saved_behavior_prediction_paths,
            "saved_behavior_summary_paths": saved_behavior_summary_paths,
            "saved_ripple_table_paths": saved_ripple_table_paths,
            "saved_epoch_summary_paths": saved_epoch_summary_paths,
            "saved_dataset_paths": saved_dataset_paths,
            "saved_decoded_paths": saved_decoded_paths,
            "saved_figure_paths": saved_figure_paths,
            "successful_jobs": successful_jobs,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")
    if not successful_jobs:
        print("No decode epochs completed successfully.")


if __name__ == "__main__":
    main()
