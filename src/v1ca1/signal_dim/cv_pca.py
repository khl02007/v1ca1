from __future__ import annotations

"""Cross-validated PCA for light/dark task-progression response geometry.

This script builds paired light/dark V1 or CA1 tuning tensors from disjoint
groups of W-track laps, estimates within-condition cvPCA spectra, and scores
how well source-condition subspaces capture target-condition activity.
"""

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from v1ca1.helper.run_logging import write_run_log
from v1ca1.signal_dim._shared import (
    DEFAULT_BIN_SIZE_CM,
    DEFAULT_DATA_ROOT,
    DEFAULT_MIN_OCCUPANCY_S,
    DEFAULT_N_GROUPS,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REGIONS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    TRAJECTORY_TYPES,
    get_light_and_dark_epochs,
    get_signal_dim_figure_dir,
    get_signal_dim_output_dir,
    load_signal_dim_session,
    occupancy_mask_1d,
    sample_lap_indices,
    split_lap_indices_into_groups,
)


ArrayF = npt.NDArray[np.floating]
ArrayB = npt.NDArray[np.bool_]

DEFAULT_N_RANDOM_REPEATS = 1
DEFAULT_UNIT_FILTER_MODE = "shared-active"
DEFAULT_NORMALIZATION = "zscore"
DEFAULT_MIN_CONDITION_SD_HZ = 1e-6
DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_OUTPUT_DIRNAME = "cv_pca"
DEFAULT_RESIDUAL_POWER_COMPONENT = "pr"
DEFAULT_RESIDUALIZED_COMPONENT = "pr"
CONDITIONS = ("dark", "light")
RESIDUALIZED_COMPONENT_MODES = ("pr", "80", "90")
UNIT_CLASSES = (
    "shared_active",
    "dark_only",
    "light_only",
    "inactive_or_low_mod",
)


UnitFilterMode = Literal["shared-active", "dark-active", "union-active"]
NormalizationMode = Literal["zscore", "center"]
ResidualPowerComponentMode = Literal["pr", "80", "90"]


@dataclass(frozen=True)
class UnitSelection:
    """Neuron inclusion metadata for one light/dark pair."""

    keep_mask: ArrayB
    unit_classes: np.ndarray
    dark_active: ArrayB
    light_active: ArrayB
    dark_modulated: ArrayB
    light_modulated: ArrayB


@dataclass(frozen=True)
class PairTuningTensors:
    """Paired light/dark grouped tuning tensors with aligned conditions."""

    dark: ArrayF
    light: ArrayF
    unit_ids: np.ndarray
    unit_classes: np.ndarray
    dark_firing_rate_hz: ArrayF
    light_firing_rate_hz: ArrayF
    dark_condition_sd_hz: ArrayF
    light_condition_sd_hz: ArrayF
    condition_trajectory: np.ndarray
    condition_bin_center: ArrayF
    condition_bin_index: np.ndarray
    n_valid_bins_by_trajectory: dict[str, int]
    unit_selection: UnitSelection


def _stable_seed_component(value: str | int) -> int:
    """Return a deterministic uint32 seed component for one value."""
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFFFFFFFF
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


def _rng_for_epoch(base_seed: int, epoch: str) -> np.random.Generator:
    """Return a reproducible RNG for one epoch's lap grouping."""
    seed_sequence = np.random.SeedSequence(
        [int(base_seed) & 0xFFFFFFFF, _stable_seed_component(epoch)]
    )
    return np.random.default_rng(seed_sequence)


def _validate_group_count(n_groups: int) -> None:
    """Reject group counts too small for leave-one-group-out cvPCA."""
    if int(n_groups) < 3:
        raise ValueError("cvPCA requires at least 3 lap groups.")


def _normalize_train_and_target(
    x_train: ArrayF,
    x_target: ArrayF,
    *,
    normalization: NormalizationMode,
    min_scale: float,
) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
    """Condition-center train/target data and optionally scale by train SD."""
    train = np.asarray(x_train, dtype=float)
    target = np.asarray(x_target, dtype=float)
    if train.ndim != 2 or target.ndim != 2:
        raise ValueError("x_train and x_target must both be 2D arrays.")
    if train.shape[1] != target.shape[1]:
        raise ValueError(
            "x_train and x_target must have the same number of units. "
            f"Got {train.shape[1]} and {target.shape[1]}."
        )

    train_center = np.mean(train, axis=0)
    target_center = np.mean(target, axis=0)
    train_centered = train - train_center
    target_centered = target - target_center
    if normalization == "center":
        scale = np.ones(train.shape[1], dtype=float)
        return train_centered, target_centered, train_center, scale
    if normalization != "zscore":
        raise ValueError(f"Unsupported normalization mode: {normalization!r}")

    scale = np.std(train_centered, axis=0)
    safe_scale = np.where(scale >= float(min_scale), scale, 1.0)
    return (
        train_centered / safe_scale,
        target_centered / safe_scale,
        train_center,
        safe_scale,
    )


def _fit_pcs(x_train: ArrayF) -> ArrayF:
    """Return right singular vectors as unit-space principal axes."""
    train = np.asarray(x_train, dtype=float)
    if train.ndim != 2:
        raise ValueError(f"Expected a 2D train matrix, got shape {train.shape}.")
    if min(train.shape) == 0:
        raise ValueError("Cannot fit PCA to an empty train matrix.")
    _, _, vt = np.linalg.svd(train, full_matrices=False)
    return vt.T


def _participation_ratio(values: ArrayF) -> float:
    """Return participation ratio from nonnegative spectral values."""
    spectrum = np.asarray(values, dtype=float).reshape(-1)
    spectrum = spectrum[np.isfinite(spectrum)]
    spectrum = np.clip(spectrum, 0.0, np.inf)
    denom = float(np.sum(spectrum**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(spectrum) ** 2 / denom)


def _components_for_fraction(cumulative: ArrayF, fraction: float) -> float:
    """Return the first 1-based component index reaching a cumulative fraction."""
    values = np.asarray(cumulative, dtype=float).reshape(-1)
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    reached = np.flatnonzero(valid & (values >= float(fraction)))
    if reached.size == 0:
        return float("nan")
    return float(reached[0] + 1)


def _class_counts(unit_classes: np.ndarray) -> dict[str, int]:
    """Return counts for each configured unit class."""
    classes = np.asarray(unit_classes, dtype=str)
    return {unit_class: int(np.sum(classes == unit_class)) for unit_class in UNIT_CLASSES}


def classify_and_select_units(
    *,
    unit_ids: np.ndarray,
    dark_firing_rate_hz: ArrayF,
    light_firing_rate_hz: ArrayF,
    dark_condition_sd_hz: ArrayF,
    light_condition_sd_hz: ArrayF,
    min_firing_rate_hz: float,
    min_condition_sd_hz: float,
    unit_filter_mode: UnitFilterMode,
) -> UnitSelection:
    """Return cell classes and the requested inclusion mask."""
    unit_ids = np.asarray(unit_ids)
    dark_fr = np.asarray(dark_firing_rate_hz, dtype=float).reshape(-1)
    light_fr = np.asarray(light_firing_rate_hz, dtype=float).reshape(-1)
    dark_sd = np.asarray(dark_condition_sd_hz, dtype=float).reshape(-1)
    light_sd = np.asarray(light_condition_sd_hz, dtype=float).reshape(-1)
    n_units = unit_ids.size
    for name, values in {
        "dark_firing_rate_hz": dark_fr,
        "light_firing_rate_hz": light_fr,
        "dark_condition_sd_hz": dark_sd,
        "light_condition_sd_hz": light_sd,
    }.items():
        if values.size != n_units:
            raise ValueError(f"{name} must have one value per unit.")

    dark_active = np.isfinite(dark_fr) & (dark_fr >= float(min_firing_rate_hz))
    light_active = np.isfinite(light_fr) & (light_fr >= float(min_firing_rate_hz))
    dark_modulated = np.isfinite(dark_sd) & (dark_sd >= float(min_condition_sd_hz))
    light_modulated = np.isfinite(light_sd) & (light_sd >= float(min_condition_sd_hz))
    dark_usable = dark_active & dark_modulated
    light_usable = light_active & light_modulated

    if unit_filter_mode == "shared-active":
        keep_mask = dark_usable & light_usable
    elif unit_filter_mode == "dark-active":
        keep_mask = dark_usable
    elif unit_filter_mode == "union-active":
        keep_mask = dark_usable | light_usable
    else:
        raise ValueError(f"Unsupported unit_filter_mode: {unit_filter_mode!r}")

    unit_classes = np.full(n_units, "inactive_or_low_mod", dtype=object)
    unit_classes[dark_usable & ~light_usable] = "dark_only"
    unit_classes[light_usable & ~dark_usable] = "light_only"
    unit_classes[dark_usable & light_usable] = "shared_active"

    return UnitSelection(
        keep_mask=np.asarray(keep_mask, dtype=bool),
        unit_classes=np.asarray(unit_classes, dtype=str),
        dark_active=np.asarray(dark_active, dtype=bool),
        light_active=np.asarray(light_active, dtype=bool),
        dark_modulated=np.asarray(dark_modulated, dtype=bool),
        light_modulated=np.asarray(light_modulated, dtype=bool),
    )


def _condition_sd(f_tensor: ArrayF) -> ArrayF:
    """Return per-unit condition SD from the repeat-averaged tuning tensor."""
    tensor = np.asarray(f_tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError(f"Expected F with shape (repeat, condition, unit), got {tensor.shape}.")
    return np.std(np.mean(tensor, axis=0), axis=0)


def _mean_valid(values: ArrayF, axis: int | tuple[int, ...]) -> ArrayF:
    """Return nanmean without emitting warnings for all-NaN slices."""
    array = np.asarray(values, dtype=float)
    valid_count = np.sum(np.isfinite(array), axis=axis)
    summed = np.nansum(array, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = summed / valid_count
    return np.where(valid_count > 0, mean, np.nan)


def _within_cv_pca_from_tensor(
    tensor: ArrayF,
    *,
    normalization: NormalizationMode,
    min_scale: float,
) -> dict[str, Any]:
    """Return within-condition cvPCA spectrum and dimensionality for one tensor."""
    values = np.asarray(tensor, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected tensor with shape (group, condition, unit), got {values.shape}.")

    n_groups, n_conditions, n_units = values.shape
    _validate_group_count(n_groups)
    if n_units == 0:
        raise ValueError("Cannot run cvPCA with zero units.")

    n_components = min(n_conditions, n_units)
    diagonal = np.full((n_groups, n_components), np.nan, dtype=float)
    for fold in range(n_groups):
        train_group_mask = np.ones(n_groups, dtype=bool)
        train_group_mask[fold] = False
        x_train_raw = np.mean(values[train_group_mask], axis=0)
        x_target_raw = values[fold]
        x_train, x_target, _, _ = _normalize_train_and_target(
            x_train_raw,
            x_target_raw,
            normalization=normalization,
            min_scale=min_scale,
        )
        pcs = _fit_pcs(x_train)[:, :n_components]
        train_scores = x_train @ pcs
        target_scores = x_target @ pcs
        diagonal[fold] = np.mean(train_scores * target_scores, axis=0)

    signed_spectrum = _mean_valid(diagonal, axis=0)
    positive_spectrum = np.clip(signed_spectrum, 0.0, np.inf)
    cumulative = np.full(n_components, np.nan, dtype=float)
    n80 = float("nan")
    n90 = float("nan")
    total_positive = float(np.nansum(positive_spectrum))
    if total_positive > 0.0:
        cumulative = np.cumsum(positive_spectrum) / total_positive
        n80 = _components_for_fraction(cumulative, 0.8)
        n90 = _components_for_fraction(cumulative, 0.9)

    return {
        "signed_spectrum": signed_spectrum,
        "positive_spectrum": positive_spectrum,
        "cumulative": cumulative,
        "participation_ratio": _participation_ratio(positive_spectrum),
        "n_components_80": n80,
        "n_components_90": n90,
    }


def _residualized_light_tensor_for_dark_fold(
    *,
    dark_tensor: ArrayF,
    light_tensor: ArrayF,
    dark_source_fold: int,
    component_count: int,
    min_scale: float,
) -> ArrayF:
    """Return light groups after removing dark PCs fit without one dark group."""
    dark = np.asarray(dark_tensor, dtype=float)
    light = np.asarray(light_tensor, dtype=float)
    n_groups = int(dark.shape[0])
    if int(component_count) < 1:
        raise ValueError("component_count must be positive.")

    train_group_mask = np.ones(n_groups, dtype=bool)
    train_group_mask[int(dark_source_fold)] = False
    dark_train_raw = np.mean(dark[train_group_mask], axis=0)
    dark_centered = dark_train_raw - np.mean(dark_train_raw, axis=0)
    scale = np.std(dark_centered, axis=0)
    safe_scale = np.where(scale >= float(min_scale), scale, 1.0)
    dark_train = dark_centered / safe_scale

    n_components = min(dark_train.shape)
    component_count = min(int(component_count), n_components)
    pcs = _fit_pcs(dark_train)[:, :component_count]

    residualized = np.full_like(light, np.nan, dtype=float)
    for group in range(n_groups):
        light_raw = light[group]
        light_centered = light_raw - np.mean(light_raw, axis=0)
        light_normalized = light_centered / safe_scale
        reconstruction = (light_normalized @ pcs) @ pcs.T
        residualized[group] = light_normalized - reconstruction
    return residualized


def _compute_residualized_light_metrics(
    *,
    dark_tensor: ArrayF,
    light_tensor: ArrayF,
    cutoff_components: dict[str, int],
    min_scale: float,
) -> dict[str, Any]:
    """Compute cvPCA dimensionality of light after removing dark subspaces."""
    dark = np.asarray(dark_tensor, dtype=float)
    light = np.asarray(light_tensor, dtype=float)
    n_groups, n_conditions, n_units = dark.shape
    n_components = min(n_conditions, n_units)
    n_modes = len(RESIDUALIZED_COMPONENT_MODES)

    signed_by_fold = np.full((n_modes, n_groups, n_components), np.nan, dtype=np.float32)
    cutoff_values = np.full((n_modes,), np.nan, dtype=np.float32)
    signed_mean = np.full((n_modes, n_components), np.nan, dtype=np.float32)
    positive_mean = np.full_like(signed_mean, np.nan)
    cumulative = np.full_like(signed_mean, np.nan)
    pr = np.full((n_modes,), np.nan, dtype=np.float32)
    n80 = np.full((n_modes,), np.nan, dtype=np.float32)
    n90 = np.full((n_modes,), np.nan, dtype=np.float32)

    for mode_index, mode in enumerate(RESIDUALIZED_COMPONENT_MODES):
        component_count = int(cutoff_components.get(mode, -1))
        if component_count < 1:
            continue
        component_count = min(component_count, n_components)
        cutoff_values[mode_index] = np.float32(component_count)

        for dark_source_fold in range(n_groups):
            residualized = _residualized_light_tensor_for_dark_fold(
                dark_tensor=dark,
                light_tensor=light,
                dark_source_fold=dark_source_fold,
                component_count=component_count,
                min_scale=min_scale,
            )
            fold_metrics = _within_cv_pca_from_tensor(
                residualized,
                normalization="center",
                min_scale=min_scale,
            )
            signed_by_fold[mode_index, dark_source_fold] = fold_metrics[
                "signed_spectrum"
            ].astype(np.float32)

        mode_signed = _mean_valid(signed_by_fold[mode_index], axis=0)
        mode_positive = np.clip(mode_signed, 0.0, np.inf)
        signed_mean[mode_index] = mode_signed.astype(np.float32)
        positive_mean[mode_index] = mode_positive.astype(np.float32)
        total_positive = float(np.nansum(mode_positive))
        if total_positive > 0.0:
            mode_cumulative = np.cumsum(mode_positive) / total_positive
            cumulative[mode_index] = mode_cumulative.astype(np.float32)
            n80[mode_index] = np.float32(_components_for_fraction(mode_cumulative, 0.8))
            n90[mode_index] = np.float32(_components_for_fraction(mode_cumulative, 0.9))
        pr[mode_index] = np.float32(_participation_ratio(mode_positive))

    return {
        "cutoff_components": cutoff_values,
        "spectrum_signed_by_fold": signed_by_fold,
        "spectrum_signed": signed_mean,
        "spectrum_positive": positive_mean,
        "cumulative_shared_variance": cumulative,
        "participation_ratio": pr,
        "n_components_80": n80,
        "n_components_90": n90,
    }


def compute_cv_pca_metrics(
    tensors_by_condition: dict[str, ArrayF],
    *,
    unit_classes: np.ndarray,
    normalization: NormalizationMode = DEFAULT_NORMALIZATION,
    min_scale: float = DEFAULT_MIN_CONDITION_SD_HZ,
    save_residual_matrices: bool = False,
) -> dict[str, Any]:
    """Compute within-condition cvPCA and cross-condition projection metrics."""
    tensors = {
        condition: np.asarray(tensors_by_condition[condition], dtype=float)
        for condition in CONDITIONS
    }
    shapes = {condition: tensors[condition].shape for condition in CONDITIONS}
    first_shape = shapes[CONDITIONS[0]]
    for condition, shape in shapes.items():
        if len(shape) != 3:
            raise ValueError(f"{condition} tensor must have shape (group, condition, unit).")
        if shape != first_shape:
            raise ValueError(f"Dark and light tensors must have the same shape. Got {shapes!r}.")

    n_groups, n_conditions, n_units = first_shape
    _validate_group_count(n_groups)
    if n_units == 0:
        raise ValueError("No units remain after filtering.")

    unit_classes = np.asarray(unit_classes, dtype=str)
    if unit_classes.size != n_units:
        raise ValueError("unit_classes must have one value per retained unit.")

    n_components = min(n_conditions, n_units)
    dims_5 = (len(CONDITIONS), len(CONDITIONS), n_groups, n_groups, n_components)
    dims_4 = (len(CONDITIONS), len(CONDITIONS), n_groups, n_groups)
    score_covariance = np.full(dims_5, np.nan, dtype=np.float32)
    test_variance = np.full(dims_5, np.nan, dtype=np.float32)
    cumulative_captured = np.full(dims_5, np.nan, dtype=np.float32)
    residual_fraction = np.full(dims_5, np.nan, dtype=np.float32)
    total_test_variance = np.full(dims_4, np.nan, dtype=np.float32)
    valid_projection = np.zeros(dims_4, dtype=bool)
    residual_by_class = np.full((*dims_5, len(UNIT_CLASSES)), np.nan, dtype=np.float32)
    residual_matrices = (
        np.full((*dims_5, n_conditions, n_units), np.nan, dtype=np.float32)
        if save_residual_matrices
        else None
    )

    condition_index = {condition: index for index, condition in enumerate(CONDITIONS)}
    unit_class_masks = {
        unit_class: np.asarray(unit_classes == unit_class, dtype=bool)
        for unit_class in UNIT_CLASSES
    }

    for source_condition in CONDITIONS:
        source_tensor = tensors[source_condition]
        source_i = condition_index[source_condition]
        for source_fold in range(n_groups):
            train_group_mask = np.ones(n_groups, dtype=bool)
            train_group_mask[source_fold] = False
            x_train_raw = np.mean(source_tensor[train_group_mask], axis=0)

            for target_condition in CONDITIONS:
                target_tensor = tensors[target_condition]
                target_i = condition_index[target_condition]
                target_groups = (
                    [source_fold]
                    if target_condition == source_condition
                    else range(n_groups)
                )
                for target_group in target_groups:
                    x_target_raw = target_tensor[target_group]
                    x_train, x_target, _, _ = _normalize_train_and_target(
                        x_train_raw,
                        x_target_raw,
                        normalization=normalization,
                        min_scale=min_scale,
                    )
                    pcs = _fit_pcs(x_train)
                    pcs = pcs[:, :n_components]
                    train_scores = x_train @ pcs
                    target_scores = x_target @ pcs
                    total_variance = float(np.mean(np.sum(x_target**2, axis=1)))
                    if not np.isfinite(total_variance) or total_variance <= 0.0:
                        continue

                    covariance = np.mean(train_scores * target_scores, axis=0)
                    component_variance = np.mean(target_scores**2, axis=0)
                    cumulative = np.cumsum(component_variance) / total_variance
                    residual = 1.0 - cumulative

                    score_covariance[
                        source_i, target_i, source_fold, target_group, :
                    ] = covariance.astype(np.float32)
                    test_variance[
                        source_i, target_i, source_fold, target_group, :
                    ] = component_variance.astype(np.float32)
                    cumulative_captured[
                        source_i, target_i, source_fold, target_group, :
                    ] = cumulative.astype(np.float32)
                    residual_fraction[
                        source_i, target_i, source_fold, target_group, :
                    ] = residual.astype(np.float32)
                    total_test_variance[
                        source_i, target_i, source_fold, target_group
                    ] = np.float32(total_variance)
                    valid_projection[source_i, target_i, source_fold, target_group] = True

                    reconstruction = np.zeros_like(x_target)
                    for component_index in range(n_components):
                        scores_k = target_scores[:, [component_index]]
                        pc_k = pcs[:, [component_index]]
                        reconstruction = reconstruction + scores_k @ pc_k.T
                        residual_matrix = x_target - reconstruction
                        for class_index, unit_class in enumerate(UNIT_CLASSES):
                            mask = unit_class_masks[unit_class]
                            class_energy = (
                                float(np.sum(residual_matrix[:, mask] ** 2))
                                if np.any(mask)
                                else 0.0
                            )
                            residual_by_class[
                                source_i,
                                target_i,
                                source_fold,
                                target_group,
                                component_index,
                                class_index,
                            ] = np.float32(class_energy / total_variance)
                        if residual_matrices is not None:
                            residual_matrices[
                                source_i,
                                target_i,
                                source_fold,
                                target_group,
                                component_index,
                            ] = residual_matrix.astype(np.float32)

    within_signed = np.full((len(CONDITIONS), n_components), np.nan, dtype=np.float32)
    within_positive = np.full_like(within_signed, np.nan)
    within_cumulative = np.full_like(within_signed, np.nan)
    within_pr = np.full((len(CONDITIONS),), np.nan, dtype=np.float32)
    within_n80 = np.full((len(CONDITIONS),), np.nan, dtype=np.float32)
    within_n90 = np.full((len(CONDITIONS),), np.nan, dtype=np.float32)

    for condition in CONDITIONS:
        index = condition_index[condition]
        diagonal = np.full((n_groups, n_components), np.nan, dtype=float)
        for fold in range(n_groups):
            diagonal[fold] = score_covariance[index, index, fold, fold, :]
        signed_spectrum = _mean_valid(diagonal, axis=0)
        positive_spectrum = np.clip(signed_spectrum, 0.0, np.inf)
        total_positive = float(np.nansum(positive_spectrum))
        if total_positive > 0.0:
            cumulative = np.cumsum(positive_spectrum) / total_positive
            within_cumulative[index] = cumulative.astype(np.float32)
            within_n80[index] = np.float32(_components_for_fraction(cumulative, 0.8))
            within_n90[index] = np.float32(_components_for_fraction(cumulative, 0.9))
        within_signed[index] = signed_spectrum.astype(np.float32)
        within_positive[index] = positive_spectrum.astype(np.float32)
        within_pr[index] = np.float32(_participation_ratio(positive_spectrum))

    dark_index = condition_index["dark"]
    residualized_cutoffs = {
        "pr": _finite_component_count(within_pr[dark_index]),
        "80": _finite_component_count(within_n80[dark_index]),
        "90": _finite_component_count(within_n90[dark_index]),
    }
    residualized_light_metrics = _compute_residualized_light_metrics(
        dark_tensor=tensors["dark"],
        light_tensor=tensors["light"],
        cutoff_components=residualized_cutoffs,
        min_scale=min_scale,
    )

    result = {
        "score_covariance_by_component": score_covariance,
        "test_variance_by_component": test_variance,
        "cumulative_test_variance_captured": cumulative_captured,
        "residual_fraction": residual_fraction,
        "total_test_variance": total_test_variance,
        "valid_projection": valid_projection,
        "residual_fraction_by_unit_class": residual_by_class,
        "within_cv_spectrum_signed": within_signed,
        "within_cv_spectrum_positive": within_positive,
        "within_cv_cumulative_shared_variance": within_cumulative,
        "within_cv_participation_ratio": within_pr,
        "within_cv_n_components_80": within_n80,
        "within_cv_n_components_90": within_n90,
        "residualized_light_cutoff_components": residualized_light_metrics[
            "cutoff_components"
        ],
        "residualized_light_cv_spectrum_signed_by_fold": residualized_light_metrics[
            "spectrum_signed_by_fold"
        ],
        "residualized_light_cv_spectrum_signed": residualized_light_metrics[
            "spectrum_signed"
        ],
        "residualized_light_cv_spectrum_positive": residualized_light_metrics[
            "spectrum_positive"
        ],
        "residualized_light_cv_cumulative_shared_variance": residualized_light_metrics[
            "cumulative_shared_variance"
        ],
        "residualized_light_cv_participation_ratio": residualized_light_metrics[
            "participation_ratio"
        ],
        "residualized_light_cv_n_components_80": residualized_light_metrics[
            "n_components_80"
        ],
        "residualized_light_cv_n_components_90": residualized_light_metrics[
            "n_components_90"
        ],
    }
    if residual_matrices is not None:
        result["residual_matrix"] = residual_matrices
    return result


def _extract_tsgroup_unit_ids(tsgroup: Any) -> np.ndarray:
    """Return unit ids from a pynapple TsGroup-like object."""
    if hasattr(tsgroup, "get_unit_ids"):
        return np.asarray(tsgroup.get_unit_ids())
    if hasattr(tsgroup, "keys"):
        return np.asarray(list(tsgroup.keys()))
    raise ValueError("Could not extract unit ids from the TsGroup-like object.")


def _build_epoch_grouped_tuning(
    *,
    session: dict[str, Any],
    epoch: str,
    region: str,
    bin_edges: ArrayF,
    n_groups: int,
    min_occupancy_s: float,
    group_seed: int,
) -> tuple[dict[str, list[ArrayF]], dict[str, list[ArrayB]]]:
    """Build grouped trajectory tuning curves and occupancy masks for one epoch."""
    import pynapple as nap
    import track_linearization as tl

    from v1ca1.helper.wtrack import get_wtrack_branch_side

    rng = _rng_for_epoch(group_seed, epoch)
    position_offset = int(session["position_offset"])
    trajectory_times = session["trajectory_times"]
    movement_epoch = session["movement_by_epoch"][epoch]
    position_dict = session["position_dict"]
    timestamps_position_dict = session["timestamps_position_dict"]
    spikes = session["spikes_by_region"][region]
    track_graphs_by_side = session["track_graphs_by_side"]
    edge_orders_by_side = session["edge_orders_by_side"]
    linear_edge_spacing = session["linear_edge_spacing"]
    n_bins = int(np.asarray(bin_edges).size - 1)

    traj_starts: dict[str, np.ndarray] = {}
    traj_ends: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    for trajectory in TRAJECTORY_TYPES:
        starts = np.asarray(trajectory_times[epoch][trajectory][:, 0], dtype=float)
        ends = np.asarray(trajectory_times[epoch][trajectory][:, -1], dtype=float)
        traj_starts[trajectory] = starts
        traj_ends[trajectory] = ends
        n_trials[trajectory] = int(starts.size)
        if starts.size < n_groups:
            raise ValueError(
                f"n_groups={n_groups} exceeds available laps for {epoch} "
                f"{trajectory}: n_available={starts.size}."
            )

    linear_position: dict[str, Any] = {}
    for trajectory in TRAJECTORY_TYPES:
        branch_side = get_wtrack_branch_side(trajectory)
        position_df = tl.get_linearized_position(
            position=position_dict[epoch][position_offset:],
            track_graph=track_graphs_by_side[branch_side],
            edge_order=edge_orders_by_side[branch_side],
            edge_spacing=linear_edge_spacing,
        )
        trajectory_epoch = nap.IntervalSet(
            start=traj_starts[trajectory],
            end=traj_ends[trajectory],
        )
        linear_position[trajectory] = nap.Tsd(
            t=np.asarray(timestamps_position_dict[epoch][position_offset:], dtype=float),
            d=np.asarray(position_df["linear_position"], dtype=float),
            time_support=trajectory_epoch,
            time_units="s",
        )

    tuning_by_trajectory: dict[str, list[ArrayF]] = {
        trajectory: [] for trajectory in TRAJECTORY_TYPES
    }
    occupancy_by_trajectory: dict[str, list[ArrayB]] = {
        trajectory: [] for trajectory in TRAJECTORY_TYPES
    }
    for trajectory in TRAJECTORY_TYPES:
        selected_laps = sample_lap_indices(
            n_trials[trajectory],
            lap_fraction=1.0,
            n_groups=n_groups,
            rng=rng,
        )
        lap_groups = split_lap_indices_into_groups(
            selected_laps,
            n_groups,
            rng=rng,
        )
        for group_indices in lap_groups:
            group_indices = np.sort(group_indices)
            group_epoch = nap.IntervalSet(
                start=traj_starts[trajectory][group_indices],
                end=traj_ends[trajectory][group_indices],
            )
            use_epoch = group_epoch.intersect(movement_epoch)
            tuning_curve = nap.compute_tuning_curves(
                data=spikes,
                features=linear_position[trajectory],
                bins=[np.asarray(bin_edges, dtype=float)],
                epochs=use_epoch,
                feature_names=["linpos"],
            )
            tuning_values = np.asarray(tuning_curve.to_numpy(), dtype=float)
            if tuning_values.shape[1] != n_bins:
                raise RuntimeError(
                    "Tuning curve bin count does not match requested bin edges."
                )
            occupancy_mask = occupancy_mask_1d(
                linear_position[trajectory],
                use_epoch,
                np.asarray(bin_edges, dtype=float),
                min_occupancy_s=min_occupancy_s,
            )
            if occupancy_mask.shape[0] != n_bins:
                raise RuntimeError("Occupancy mask length mismatch with bins.")
            tuning_by_trajectory[trajectory].append(tuning_values)
            occupancy_by_trajectory[trajectory].append(occupancy_mask)

    return tuning_by_trajectory, occupancy_by_trajectory


def _stack_shared_condition_tensor(
    tuning_by_trajectory: dict[str, list[ArrayF]],
    keep_masks: dict[str, ArrayB],
) -> ArrayF:
    """Stack trajectory tuning curves into a grouped condition-by-unit tensor."""
    n_groups = len(next(iter(tuning_by_trajectory.values())))
    grouped_matrices: list[ArrayF] = []
    for group_index in range(n_groups):
        pieces: list[ArrayF] = []
        for trajectory in TRAJECTORY_TYPES:
            tuning = np.nan_to_num(
                np.asarray(tuning_by_trajectory[trajectory][group_index], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            pieces.append(tuning[:, keep_masks[trajectory]].T)
        grouped_matrices.append(np.concatenate(pieces, axis=0))
    return np.stack(grouped_matrices, axis=0)


def build_pairwise_tuning_tensors(
    session: dict[str, Any],
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    bin_size_cm: float,
    n_groups: int,
    min_occupancy_s: float,
    group_seed: int,
    min_firing_rate_hz: float,
    min_condition_sd_hz: float,
    unit_filter_mode: UnitFilterMode,
) -> PairTuningTensors:
    """Build aligned light/dark grouped tuning tensors for one region and pair."""
    _validate_group_count(n_groups)

    track_total_length = float(session["track_total_length"])
    bin_edges = np.arange(
        0.0,
        track_total_length + float(bin_size_cm),
        float(bin_size_cm),
        dtype=float,
    )
    if bin_edges.size < 3:
        raise ValueError("Binning produced fewer than two bins; decrease bin_size_cm.")

    dark_tuning, dark_occupancy = _build_epoch_grouped_tuning(
        session=session,
        epoch=dark_epoch,
        region=region,
        bin_edges=bin_edges,
        n_groups=n_groups,
        min_occupancy_s=min_occupancy_s,
        group_seed=group_seed,
    )
    light_tuning, light_occupancy = _build_epoch_grouped_tuning(
        session=session,
        epoch=light_epoch,
        region=region,
        bin_edges=bin_edges,
        n_groups=n_groups,
        min_occupancy_s=min_occupancy_s,
        group_seed=group_seed,
    )

    keep_masks: dict[str, ArrayB] = {}
    n_valid_bins_by_trajectory: dict[str, int] = {}
    condition_trajectory: list[str] = []
    condition_bin_center: list[float] = []
    condition_bin_index: list[int] = []
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for trajectory in TRAJECTORY_TYPES:
        keep = np.logical_and.reduce(
            [*dark_occupancy[trajectory], *light_occupancy[trajectory]]
        )
        if int(np.sum(keep)) < 2:
            raise ValueError(
                f"Too few shared valid bins for {trajectory}: {int(np.sum(keep))}."
            )
        keep_masks[trajectory] = np.asarray(keep, dtype=bool)
        n_valid_bins_by_trajectory[trajectory] = int(np.sum(keep))
        kept_indices = np.flatnonzero(keep)
        condition_trajectory.extend([trajectory] * kept_indices.size)
        condition_bin_center.extend(bin_centers[kept_indices].tolist())
        condition_bin_index.extend(kept_indices.astype(int).tolist())

    dark_tensor_all = _stack_shared_condition_tensor(dark_tuning, keep_masks)
    light_tensor_all = _stack_shared_condition_tensor(light_tuning, keep_masks)
    unit_ids = _extract_tsgroup_unit_ids(session["spikes_by_region"][region])
    if unit_ids.size != dark_tensor_all.shape[2]:
        unit_ids = np.arange(dark_tensor_all.shape[2], dtype=int)

    dark_fr = np.asarray(
        session["movement_firing_rates_by_region"][region][dark_epoch],
        dtype=float,
    )
    light_fr = np.asarray(
        session["movement_firing_rates_by_region"][region][light_epoch],
        dtype=float,
    )
    dark_sd = _condition_sd(dark_tensor_all)
    light_sd = _condition_sd(light_tensor_all)
    unit_selection = classify_and_select_units(
        unit_ids=unit_ids,
        dark_firing_rate_hz=dark_fr,
        light_firing_rate_hz=light_fr,
        dark_condition_sd_hz=dark_sd,
        light_condition_sd_hz=light_sd,
        min_firing_rate_hz=min_firing_rate_hz,
        min_condition_sd_hz=min_condition_sd_hz,
        unit_filter_mode=unit_filter_mode,
    )
    keep_units = unit_selection.keep_mask
    if not np.any(keep_units):
        raise ValueError(
            "No units remain after applying firing-rate and condition-SD filters."
        )

    return PairTuningTensors(
        dark=dark_tensor_all[:, :, keep_units],
        light=light_tensor_all[:, :, keep_units],
        unit_ids=np.asarray(unit_ids)[keep_units],
        unit_classes=unit_selection.unit_classes[keep_units],
        dark_firing_rate_hz=dark_fr[keep_units],
        light_firing_rate_hz=light_fr[keep_units],
        dark_condition_sd_hz=dark_sd[keep_units],
        light_condition_sd_hz=light_sd[keep_units],
        condition_trajectory=np.asarray(condition_trajectory, dtype=str),
        condition_bin_center=np.asarray(condition_bin_center, dtype=float),
        condition_bin_index=np.asarray(condition_bin_index, dtype=int),
        n_valid_bins_by_trajectory=n_valid_bins_by_trajectory,
        unit_selection=unit_selection,
    )


def build_result_dataset(
    *,
    pair_tensors: PairTuningTensors,
    metrics: dict[str, Any],
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    settings: dict[str, Any],
) -> Any:
    """Build an xarray Dataset containing cvPCA curves and metadata."""
    import xarray as xr

    n_groups, n_conditions, n_units = pair_tensors.dark.shape
    n_components = int(metrics["within_cv_spectrum_signed"].shape[1])
    data_vars: dict[str, Any] = {
        "score_covariance_by_component": (
            ("source_condition", "target_condition", "source_fold", "target_group", "component"),
            metrics["score_covariance_by_component"],
        ),
        "test_variance_by_component": (
            ("source_condition", "target_condition", "source_fold", "target_group", "component"),
            metrics["test_variance_by_component"],
        ),
        "cumulative_test_variance_captured": (
            ("source_condition", "target_condition", "source_fold", "target_group", "component"),
            metrics["cumulative_test_variance_captured"],
        ),
        "residual_fraction": (
            ("source_condition", "target_condition", "source_fold", "target_group", "component"),
            metrics["residual_fraction"],
        ),
        "total_test_variance": (
            ("source_condition", "target_condition", "source_fold", "target_group"),
            metrics["total_test_variance"],
        ),
        "valid_projection": (
            ("source_condition", "target_condition", "source_fold", "target_group"),
            metrics["valid_projection"],
        ),
        "residual_fraction_by_unit_class": (
            (
                "source_condition",
                "target_condition",
                "source_fold",
                "target_group",
                "component",
                "unit_class",
            ),
            metrics["residual_fraction_by_unit_class"],
        ),
        "within_cv_spectrum_signed": (
            ("within_condition", "component"),
            metrics["within_cv_spectrum_signed"],
        ),
        "within_cv_spectrum_positive": (
            ("within_condition", "component"),
            metrics["within_cv_spectrum_positive"],
        ),
        "within_cv_cumulative_shared_variance": (
            ("within_condition", "component"),
            metrics["within_cv_cumulative_shared_variance"],
        ),
        "within_cv_participation_ratio": (
            ("within_condition",),
            metrics["within_cv_participation_ratio"],
        ),
        "within_cv_n_components_80": (
            ("within_condition",),
            metrics["within_cv_n_components_80"],
        ),
        "within_cv_n_components_90": (
            ("within_condition",),
            metrics["within_cv_n_components_90"],
        ),
        "residualized_light_cutoff_components": (
            ("residualized_component",),
            metrics["residualized_light_cutoff_components"],
        ),
        "residualized_light_cv_spectrum_signed_by_fold": (
            ("residualized_component", "source_fold", "component"),
            metrics["residualized_light_cv_spectrum_signed_by_fold"],
        ),
        "residualized_light_cv_spectrum_signed": (
            ("residualized_component", "component"),
            metrics["residualized_light_cv_spectrum_signed"],
        ),
        "residualized_light_cv_spectrum_positive": (
            ("residualized_component", "component"),
            metrics["residualized_light_cv_spectrum_positive"],
        ),
        "residualized_light_cv_cumulative_shared_variance": (
            ("residualized_component", "component"),
            metrics["residualized_light_cv_cumulative_shared_variance"],
        ),
        "residualized_light_cv_participation_ratio": (
            ("residualized_component",),
            metrics["residualized_light_cv_participation_ratio"],
        ),
        "residualized_light_cv_n_components_80": (
            ("residualized_component",),
            metrics["residualized_light_cv_n_components_80"],
        ),
        "residualized_light_cv_n_components_90": (
            ("residualized_component",),
            metrics["residualized_light_cv_n_components_90"],
        ),
        "dark_firing_rate_hz": ("unit", pair_tensors.dark_firing_rate_hz.astype(np.float32)),
        "light_firing_rate_hz": ("unit", pair_tensors.light_firing_rate_hz.astype(np.float32)),
        "dark_condition_sd_hz": ("unit", pair_tensors.dark_condition_sd_hz.astype(np.float32)),
        "light_condition_sd_hz": ("unit", pair_tensors.light_condition_sd_hz.astype(np.float32)),
        "unit_class_per_unit": ("unit", pair_tensors.unit_classes.astype(str)),
        "condition_trajectory": ("condition", pair_tensors.condition_trajectory.astype(str)),
        "condition_bin_center": (
            "condition",
            pair_tensors.condition_bin_center.astype(np.float32),
        ),
        "condition_bin_index": ("condition", pair_tensors.condition_bin_index.astype(int)),
    }
    if "residual_matrix" in metrics:
        data_vars["residual_matrix"] = (
            (
                "source_condition",
                "target_condition",
                "source_fold",
                "target_group",
                "component",
                "condition",
                "unit",
            ),
            metrics["residual_matrix"],
        )

    attrs = {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "script": "v1ca1.signal_dim.cv_pca",
        "n_groups": int(n_groups),
        "n_conditions": int(n_conditions),
        "n_units": int(n_units),
        "n_components": int(n_components),
        "normalization": str(settings["normalization"]),
        "unit_filter_mode": str(settings["unit_filter_mode"]),
        "min_firing_rate_hz": float(settings["min_firing_rate_hz"]),
        "min_condition_sd_hz": float(settings["min_condition_sd_hz"]),
        "bin_size_cm": float(settings["bin_size_cm"]),
        "min_occupancy_s": float(settings["min_occupancy_s"]),
        "random_seed": int(settings.get("random_seed", DEFAULT_RANDOM_SEED)),
        "n_random_repeats": int(settings.get("n_random_repeats", 1)),
        "n_valid_bins_by_trajectory_json": json.dumps(
            {
                key: int(value)
                for key, value in pair_tensors.n_valid_bins_by_trajectory.items()
            },
            sort_keys=True,
        ),
        "unit_class_counts_kept_json": json.dumps(
            _class_counts(pair_tensors.unit_classes),
            sort_keys=True,
        ),
        "unit_class_counts_all_json": json.dumps(
            _class_counts(pair_tensors.unit_selection.unit_classes),
            sort_keys=True,
        ),
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "source_condition": np.asarray(CONDITIONS, dtype=str),
            "target_condition": np.asarray(CONDITIONS, dtype=str),
            "within_condition": np.asarray(CONDITIONS, dtype=str),
            "residualized_component": np.asarray(RESIDUALIZED_COMPONENT_MODES, dtype=str),
            "source_fold": np.arange(n_groups, dtype=int),
            "target_group": np.arange(n_groups, dtype=int),
            "component": np.arange(1, n_components + 1, dtype=int),
            "condition": np.arange(n_conditions, dtype=int),
            "unit": pair_tensors.unit_ids,
            "unit_class": np.asarray(UNIT_CLASSES, dtype=str),
        },
        attrs=attrs,
    )


def _mean_projection_curve(dataset: Any, variable: str, source: str, target: str) -> np.ndarray:
    """Return one mean curve across valid projection combinations."""
    values = dataset[variable].sel(source_condition=source, target_condition=target)
    valid = dataset["valid_projection"].sel(source_condition=source, target_condition=target)
    masked = values.where(valid)
    return np.asarray(masked.mean(dim=("source_fold", "target_group"), skipna=True).values)


def _mean_projection_scalar(
    dataset: Any,
    variable: str,
    source: str,
    target: str,
    component_count: int,
) -> float:
    """Return a mean projection scalar at one 1-based component count."""
    if component_count < 1:
        return float("nan")
    components = np.asarray(dataset.coords["component"].values, dtype=int)
    if component_count > int(components[-1]):
        return float("nan")
    curve = _mean_projection_curve(dataset, variable, source, target)
    return float(curve[component_count - 1])


def _finite_component_count(value: Any) -> int:
    """Return a positive component count, or -1 when unavailable."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1.0:
        return -1
    return int(round(numeric))


def _component_count_for_condition(
    dataset: Any,
    *,
    condition: str,
    component_mode: ResidualPowerComponentMode,
) -> int:
    """Return the component count selected by one within-condition metric."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition!r}")
    if component_mode == "pr":
        value = dataset["within_cv_participation_ratio"].sel(
            within_condition=condition
        )
    elif component_mode == "80":
        value = dataset["within_cv_n_components_80"].sel(within_condition=condition)
    elif component_mode == "90":
        value = dataset["within_cv_n_components_90"].sel(within_condition=condition)
    else:
        raise ValueError(f"Unsupported residual-power component mode: {component_mode!r}")

    component_count = _finite_component_count(value.values)
    max_component = int(np.asarray(dataset.coords["component"].values, dtype=int)[-1])
    if component_count < 1:
        return -1
    return min(component_count, max_component)


def _condition_epoch(dataset: Any, condition: str) -> str:
    """Return the epoch label for one condition slot."""
    if condition == "dark":
        return str(dataset.attrs["dark_epoch"])
    if condition == "light":
        return str(dataset.attrs["light_epoch"])
    raise ValueError(f"Unsupported condition: {condition!r}")


def _condition_label(dataset: Any, condition: str) -> str:
    """Return a condition label that includes the epoch identity."""
    return f"{condition} ({_condition_epoch(dataset, condition)})"


def _condition_tick_label(dataset: Any, condition: str) -> str:
    """Return a compact multi-line condition tick label."""
    return f"{condition}\n{_condition_epoch(dataset, condition)}"


def _direction_label(dataset: Any, source: str, target: str) -> str:
    """Return a source-to-target label with condition epochs."""
    return f"{_condition_label(dataset, source)}->{_condition_label(dataset, target)}"


def _capture_fraction_at_component(
    dataset: Any,
    *,
    source: str,
    target: str,
    component_count: int,
) -> float:
    """Return mean captured fraction for one projection at one component count."""
    residual = _mean_projection_scalar(
        dataset,
        "residual_fraction",
        source,
        target,
        component_count,
    )
    return float("nan") if not np.isfinite(residual) else 1.0 - residual


def _mean_residual_power_by_condition_unit(
    dataset: Any,
    *,
    source: str,
    target: str,
    component_count: int,
) -> np.ndarray:
    """Return fold-averaged residual power with shape (condition, unit)."""
    if "residual_matrix" not in dataset:
        raise ValueError("Dataset does not contain residual_matrix.")
    if int(component_count) < 1:
        raise ValueError("component_count must be positive.")

    residual = dataset["residual_matrix"].sel(
        source_condition=source,
        target_condition=target,
        component=int(component_count),
    )
    valid = dataset["valid_projection"].sel(
        source_condition=source,
        target_condition=target,
    )
    residual_power = residual.where(valid) ** 2
    averaged = residual_power.mean(dim=("source_fold", "target_group"), skipna=True)
    return np.asarray(averaged.values, dtype=float)


def _row_quantiles(values: ArrayF) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 25th, 50th, and 75th percentiles for each row."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")
    q25 = np.full(array.shape[0], np.nan, dtype=float)
    q50 = np.full(array.shape[0], np.nan, dtype=float)
    q75 = np.full(array.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(array):
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        q25[row_index], q50[row_index], q75[row_index] = np.quantile(
            finite,
            [0.25, 0.50, 0.75],
        )
    return q25, q50, q75


def build_summary_table(
    dataset: Any,
    *,
    settings: dict[str, Any],
) -> Any:
    """Build one compact parquet-ready summary table for a result dataset."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    class_counts_kept = json.loads(dataset.attrs["unit_class_counts_kept_json"])
    class_counts_all = json.loads(dataset.attrs["unit_class_counts_all_json"])
    for source in CONDITIONS:
        source_index = CONDITIONS.index(source)
        pr_value = float(dataset["within_cv_participation_ratio"].values[source_index])
        k_pr = int(round(pr_value)) if np.isfinite(pr_value) else -1
        k_80 = _finite_component_count(
            dataset["within_cv_n_components_80"].values[source_index]
        )
        k_90 = _finite_component_count(
            dataset["within_cv_n_components_90"].values[source_index]
        )
        for target in CONDITIONS:
            row = {
                "animal_name": dataset.attrs["animal_name"],
                "date": dataset.attrs["date"],
                "region": dataset.attrs["region"],
                "dark_epoch": dataset.attrs["dark_epoch"],
                "light_epoch": dataset.attrs["light_epoch"],
                "unit_filter_mode": dataset.attrs["unit_filter_mode"],
                "normalization": dataset.attrs["normalization"],
                "source_condition": source,
                "target_condition": target,
                "projection_direction": f"{source}_to_{target}",
                "is_cross_condition": bool(source != target),
                "n_units": int(dataset.attrs["n_units"]),
                "source_cv_participation_ratio": pr_value,
                "source_n_components_pr_rounded": np.nan if k_pr < 1 else k_pr,
                "source_n_components_80": np.nan if k_80 < 1 else k_80,
                "source_n_components_90": np.nan if k_90 < 1 else k_90,
                "min_firing_rate_hz": float(settings["min_firing_rate_hz"]),
                "min_condition_sd_hz": float(settings["min_condition_sd_hz"]),
                "bin_size_cm": float(settings["bin_size_cm"]),
                "n_groups": int(settings["n_groups"]),
                "min_occupancy_s": float(settings["min_occupancy_s"]),
            }
            for unit_class in UNIT_CLASSES:
                row[f"n_kept_{unit_class}"] = int(class_counts_kept[unit_class])
                row[f"n_all_{unit_class}"] = int(class_counts_all[unit_class])
            for label, component_count in {
                "pr": k_pr,
                "80": k_80,
                "90": k_90,
            }.items():
                residual = _mean_projection_scalar(
                    dataset,
                    "residual_fraction",
                    source,
                    target,
                    component_count,
                )
                row[f"residual_fraction_at_{label}"] = residual
                row[f"captured_fraction_at_{label}"] = (
                    float("nan") if not np.isfinite(residual) else 1.0 - residual
                )
                for unit_class in UNIT_CLASSES:
                    if component_count < 1:
                        row[f"residual_fraction_{unit_class}_at_{label}"] = np.nan
                        continue
                    component = int(component_count)
                    class_values = dataset["residual_fraction_by_unit_class"].sel(
                        source_condition=source,
                        target_condition=target,
                        unit_class=unit_class,
                    )
                    if component > int(dataset.coords["component"].values[-1]):
                        row[f"residual_fraction_{unit_class}_at_{label}"] = np.nan
                    else:
                        selected = class_values.sel(component=component)
                        valid = dataset["valid_projection"].sel(
                            source_condition=source,
                            target_condition=target,
                        )
                        row[f"residual_fraction_{unit_class}_at_{label}"] = float(
                            selected.where(valid).mean(
                                dim=("source_fold", "target_group"),
                                skipna=True,
                            ).values
                        )
            rows.append(row)
    return pd.DataFrame(rows)


def build_residualized_light_summary_table(
    dataset: Any,
    *,
    settings: dict[str, Any],
) -> Any:
    """Build a compact table for residualized-light cvPCA metrics."""
    import pandas as pd

    dark_pr = float(
        dataset["within_cv_participation_ratio"].sel(within_condition="dark").values
    )
    light_pr = float(
        dataset["within_cv_participation_ratio"].sel(within_condition="light").values
    )
    rows: list[dict[str, Any]] = []
    for mode in RESIDUALIZED_COMPONENT_MODES:
        residualized_pr = float(
            dataset["residualized_light_cv_participation_ratio"]
            .sel(residualized_component=mode)
            .values
        )
        row = {
            "animal_name": dataset.attrs["animal_name"],
            "date": dataset.attrs["date"],
            "region": dataset.attrs["region"],
            "dark_epoch": dataset.attrs["dark_epoch"],
            "light_epoch": dataset.attrs["light_epoch"],
            "unit_filter_mode": dataset.attrs["unit_filter_mode"],
            "normalization": dataset.attrs["normalization"],
            "residualized_component": mode,
            "n_units": int(dataset.attrs["n_units"]),
            "dark_cv_participation_ratio": dark_pr,
            "light_cv_participation_ratio": light_pr,
            "residualized_light_cutoff_components": float(
                dataset["residualized_light_cutoff_components"]
                .sel(residualized_component=mode)
                .values
            ),
            "residualized_light_cv_participation_ratio": residualized_pr,
            "residualized_light_cv_n_components_80": float(
                dataset["residualized_light_cv_n_components_80"]
                .sel(residualized_component=mode)
                .values
            ),
            "residualized_light_cv_n_components_90": float(
                dataset["residualized_light_cv_n_components_90"]
                .sel(residualized_component=mode)
                .values
            ),
            "residualized_light_pr_over_light_pr": (
                float("nan")
                if not np.isfinite(residualized_pr) or not np.isfinite(light_pr) or light_pr <= 0
                else residualized_pr / light_pr
            ),
            "bin_size_cm": float(settings["bin_size_cm"]),
            "n_groups": int(settings["n_groups"]),
            "min_occupancy_s": float(settings["min_occupancy_s"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_normalized_capture_summary_table(dataset: Any, *, settings: dict[str, Any]) -> Any:
    """Build same-k normalized cross-condition capture metrics."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for source, target, within_source, direction in (
        ("dark", "light", "light", "dark_to_light"),
        ("light", "dark", "dark", "light_to_dark"),
    ):
        for mode in RESIDUALIZED_COMPONENT_MODES:
            component_count = _component_count_for_condition(
                dataset,
                condition=source,
                component_mode=mode,
            )
            cross_capture = _capture_fraction_at_component(
                dataset,
                source=source,
                target=target,
                component_count=component_count,
            )
            within_capture = _capture_fraction_at_component(
                dataset,
                source=within_source,
                target=target,
                component_count=component_count,
            )
            cross_residual = _mean_projection_scalar(
                dataset,
                "residual_fraction",
                source,
                target,
                component_count,
            )
            within_residual = _mean_projection_scalar(
                dataset,
                "residual_fraction",
                within_source,
                target,
                component_count,
            )
            rows.append(
                {
                    "animal_name": dataset.attrs["animal_name"],
                    "date": dataset.attrs["date"],
                    "region": dataset.attrs["region"],
                    "dark_epoch": dataset.attrs["dark_epoch"],
                    "light_epoch": dataset.attrs["light_epoch"],
                    "unit_filter_mode": dataset.attrs["unit_filter_mode"],
                    "normalization": dataset.attrs["normalization"],
                    "projection_direction": direction,
                    "source_condition": source,
                    "target_condition": target,
                    "target_within_source_condition": within_source,
                    "cutoff_mode": mode,
                    "cutoff_source_condition": source,
                    "cutoff_components": component_count,
                    "cross_captured_fraction": cross_capture,
                    "target_within_captured_fraction": within_capture,
                    "normalized_capture_fraction": (
                        float("nan")
                        if not np.isfinite(cross_capture)
                        or not np.isfinite(within_capture)
                        or within_capture <= 0.0
                        else cross_capture / within_capture
                    ),
                    "cross_residual_fraction": cross_residual,
                    "target_within_residual_fraction": within_residual,
                    "excess_residual_fraction": (
                        float("nan")
                        if not np.isfinite(cross_residual)
                        or not np.isfinite(within_residual)
                        else cross_residual - within_residual
                    ),
                    "bin_size_cm": float(settings["bin_size_cm"]),
                    "n_groups": int(settings["n_groups"]),
                    "min_occupancy_s": float(settings["min_occupancy_s"]),
                    "n_units": int(dataset.attrs["n_units"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_residual_power_by_trajectory(
    dataset: Any,
    *,
    fig_dir: Path,
    stem: str,
    component_mode: ResidualPowerComponentMode,
) -> Path | None:
    """Save per-trajectory residual power medians and IQRs across units."""
    if "residual_matrix" not in dataset:
        return None

    import matplotlib.pyplot as plt

    component_count = _component_count_for_condition(
        dataset,
        condition="dark",
        component_mode=component_mode,
    )
    if component_count < 1:
        return None

    condition_trajectory = np.asarray(dataset["condition_trajectory"].values, dtype=str)
    condition_bin_center = np.asarray(dataset["condition_bin_center"].values, dtype=float)
    condition_bin_index = np.asarray(dataset["condition_bin_index"].values, dtype=int)
    series_specs = (
        ("dark", "light", _direction_label(dataset, "dark", "light"), "tab:orange"),
        ("light", "light", _direction_label(dataset, "light", "light"), "tab:red"),
    )
    power_by_series = {
        label: _mean_residual_power_by_condition_unit(
            dataset,
            source=source,
            target=target,
            component_count=component_count,
        )
        for source, target, label, _ in series_specs
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharey=True)
    for axis, trajectory in zip(axes.flat, TRAJECTORY_TYPES, strict=True):
        trajectory_mask = condition_trajectory == trajectory
        axis.set_title(trajectory.replace("_", " "))
        axis.grid(True, alpha=0.25)
        if not np.any(trajectory_mask):
            axis.text(
                0.5,
                0.5,
                "No retained bins",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue

        trajectory_rows = np.flatnonzero(trajectory_mask)
        row_order = np.lexsort(
            (
                condition_bin_center[trajectory_rows],
                condition_bin_index[trajectory_rows],
            )
        )
        trajectory_rows = trajectory_rows[row_order]
        x_values = condition_bin_center[trajectory_rows]

        for _, _, label, color in series_specs:
            q25, median, q75 = _row_quantiles(power_by_series[label])
            axis.plot(
                x_values,
                median[trajectory_rows],
                color=color,
                linewidth=1.5,
                label=label,
            )
            axis.fill_between(
                x_values,
                q25[trajectory_rows],
                q75[trajectory_rows],
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )

    for axis in axes[1, :]:
        axis.set_xlabel("Task progression (cm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Residual power")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    cutoff_label = component_mode.upper() if component_mode == "pr" else component_mode
    fig.suptitle(
        "Residual power by trajectory "
        f"({cutoff_label} cutoff, {component_count} source PCs; line=median, band=IQR)"
    )
    residual_power_path = fig_dir / f"{stem}_residual_power_by_trajectory.png"
    fig.tight_layout()
    fig.savefig(residual_power_path, dpi=300)
    plt.close(fig)
    return residual_power_path


def _plot_residualized_light_dimensionality(
    dataset: Any,
    *,
    fig_dir: Path,
    stem: str,
    component_mode: ResidualPowerComponentMode,
) -> Path | None:
    """Save dimensionality of light after removing a dark subspace."""
    import matplotlib.pyplot as plt

    if component_mode not in RESIDUALIZED_COMPONENT_MODES:
        raise ValueError(f"Unsupported residualized component mode: {component_mode!r}")

    residualized_pr = float(
        dataset["residualized_light_cv_participation_ratio"]
        .sel(residualized_component=component_mode)
        .values
    )
    if not np.isfinite(residualized_pr):
        return None

    cutoff_components = float(
        dataset["residualized_light_cutoff_components"]
        .sel(residualized_component=component_mode)
        .values
    )
    labels = (
        _condition_tick_label(dataset, "dark"),
        _condition_tick_label(dataset, "light"),
        "light resid",
    )
    positions = np.arange(len(labels), dtype=float)
    residualized_values = {
        "within_cv_participation_ratio": residualized_pr,
        "within_cv_n_components_80": float(
            dataset["residualized_light_cv_n_components_80"]
            .sel(residualized_component=component_mode)
            .values
        ),
        "within_cv_n_components_90": float(
            dataset["residualized_light_cv_n_components_90"]
            .sel(residualized_component=component_mode)
            .values
        ),
    }

    fig, axes = plt.subplots(figsize=(10, 4), ncols=3)
    metric_specs = (
        (
            "within_cv_participation_ratio",
            "Participation ratio",
            "cvPCA PR",
        ),
        (
            "within_cv_n_components_80",
            "PCs for 80%",
            "n PCs",
        ),
        (
            "within_cv_n_components_90",
            "PCs for 90%",
            "n PCs",
        ),
    )
    for axis, (variable, title, ylabel) in zip(axes, metric_specs, strict=True):
        within_values = np.asarray(dataset[variable].values, dtype=float)
        values = np.asarray(
            [within_values[0], within_values[1], residualized_values[variable]],
            dtype=float,
        )
        axis.bar(positions, values, color=["0.35", "0.70", "tab:orange"], width=0.65)
        axis.plot(positions, values, color="black", marker="o", linewidth=1.0)
        axis.set_xticks(positions, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)

    cutoff_label = component_mode.upper() if component_mode == "pr" else component_mode
    fig.suptitle(
        "Residualized-light cvPCA "
        f"({cutoff_label} dark cutoff, {cutoff_components:.0f} PCs removed)"
    )
    residualized_path = fig_dir / f"{stem}_residualized_light_dimensionality.png"
    fig.tight_layout()
    fig.savefig(residualized_path, dpi=300)
    plt.close(fig)
    return residualized_path


def _plot_normalized_capture_summary(
    dataset: Any,
    *,
    fig_dir: Path,
    stem: str,
) -> Path | None:
    """Save same-k normalized capture for cross-condition projections."""
    import matplotlib.pyplot as plt

    table = build_normalized_capture_summary_table(dataset, settings={
        "bin_size_cm": dataset.attrs["bin_size_cm"],
        "n_groups": dataset.attrs["n_groups"],
        "min_occupancy_s": dataset.attrs["min_occupancy_s"],
    })
    if table.empty:
        return None

    direction_labels = {
        "dark_to_light": _direction_label(dataset, "dark", "light"),
        "light_to_dark": _direction_label(dataset, "light", "dark"),
    }
    cutoff_labels = {"pr": "PR", "80": "80%", "90": "90%"}
    fig, axes = plt.subplots(figsize=(10, 4), ncols=2, sharey=True)
    for axis, direction in zip(axes, ("dark_to_light", "light_to_dark"), strict=True):
        direction_table = table.loc[table["projection_direction"] == direction]
        values = []
        labels = []
        for mode in RESIDUALIZED_COMPONENT_MODES:
            selected = direction_table.loc[direction_table["cutoff_mode"] == mode]
            if selected.empty:
                values.append(np.nan)
            else:
                values.append(float(selected["normalized_capture_fraction"].iloc[0]))
            labels.append(cutoff_labels[mode])
        x_values = np.arange(len(labels), dtype=float)
        axis.bar(x_values, values, color=["0.35", "0.55", "0.75"], width=0.65)
        axis.axhline(1.0, color="black", linewidth=1.0, linestyle=":", alpha=0.6)
        axis.set_xticks(x_values, labels)
        axis.set_title(direction_labels[direction])
        axis.set_ylabel("Cross capture / target-within capture")
        axis.set_ylim(0.0, max(1.05, float(np.nanmax(values)) * 1.1 if np.isfinite(values).any() else 1.05))
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Same-k normalized capture")
    path = fig_dir / f"{stem}_normalized_capture_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_summary_figures(
    dataset: Any,
    *,
    fig_dir: Path,
    stem: str,
    residual_power_component: ResidualPowerComponentMode = DEFAULT_RESIDUAL_POWER_COMPONENT,
    residualized_component: ResidualPowerComponentMode = DEFAULT_RESIDUALIZED_COMPONENT,
) -> list[Path]:
    """Save basic cvPCA spectrum and cross-projection residual figures."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, axes = plt.subplots(figsize=(10, 4), ncols=2)
    for condition in CONDITIONS:
        components = np.asarray(dataset.coords["component"].values, dtype=int)
        spectrum = np.asarray(
            dataset["within_cv_spectrum_positive"].sel(within_condition=condition).values,
            dtype=float,
        )
        cumulative = np.asarray(
            dataset["within_cv_cumulative_shared_variance"]
            .sel(within_condition=condition)
            .values,
            dtype=float,
        )
        axes[0].plot(components, spectrum, marker="o", label=_condition_label(dataset, condition))
        axes[1].plot(components, cumulative, marker="o", label=_condition_label(dataset, condition))
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Shared variance")
    axes[0].set_title("Within-condition cvPCA")
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("Cumulative shared variance")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Cumulative spectrum")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    spectrum_path = fig_dir / f"{stem}_within_cv_spectrum.png"
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=300)
    plt.close(fig)
    saved.append(spectrum_path)

    fig, axes = plt.subplots(figsize=(10, 4), ncols=3)
    condition_positions = np.arange(len(CONDITIONS), dtype=float)
    metric_specs = (
        (
            "within_cv_participation_ratio",
            "Participation ratio",
            "cvPCA PR",
        ),
        (
            "within_cv_n_components_80",
            "PCs for 80%",
            "n PCs",
        ),
        (
            "within_cv_n_components_90",
            "PCs for 90%",
            "n PCs",
        ),
    )
    for axis, (variable, title, ylabel) in zip(axes, metric_specs, strict=True):
        values = np.asarray(dataset[variable].values, dtype=float)
        axis.bar(condition_positions, values, color=["0.35", "0.70"], width=0.65)
        axis.plot(condition_positions, values, color="black", marker="o", linewidth=1.0)
        axis.set_xticks(
            condition_positions,
            [_condition_tick_label(dataset, condition) for condition in CONDITIONS],
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Within-condition cvPCA dimensionality")
    pr_path = fig_dir / f"{stem}_within_cv_pr_summary.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)
    saved.append(pr_path)

    fig, axis = plt.subplots(figsize=(7, 5))
    components = np.asarray(dataset.coords["component"].values, dtype=int)
    for source in CONDITIONS:
        for target in CONDITIONS:
            curve = _mean_projection_curve(dataset, "residual_fraction", source, target)
            axis.plot(
                components,
                curve,
                marker="o",
                label=_direction_label(dataset, source, target),
            )
    marker_specs = (
        ("dark", "pr", ":", "0.25"),
        ("dark", "80", "--", "0.25"),
        ("dark", "90", "-.", "0.25"),
        ("light", "pr", ":", "0.55"),
        ("light", "80", "--", "0.55"),
        ("light", "90", "-.", "0.55"),
    )
    for condition, mode, linestyle, color in marker_specs:
        component_count = _component_count_for_condition(
            dataset,
            condition=condition,
            component_mode=mode,
        )
        if component_count > 0:
            label = f"{_condition_label(dataset, condition)} {mode.upper() if mode == 'pr' else mode + '%'}"
            axis.axvline(
                component_count,
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.65,
                label=label,
            )
    axis.set_xlabel("Source PCs")
    axis.set_ylabel("Target residual fraction")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Subspace projection residuals")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    residual_path = fig_dir / f"{stem}_projection_residuals.png"
    fig.tight_layout()
    fig.savefig(residual_path, dpi=300)
    plt.close(fig)
    saved.append(residual_path)

    residual_power_path = _plot_residual_power_by_trajectory(
        dataset,
        fig_dir=fig_dir,
        stem=stem,
        component_mode=residual_power_component,
    )
    if residual_power_path is not None:
        saved.append(residual_power_path)

    residualized_path = _plot_residualized_light_dimensionality(
        dataset,
        fig_dir=fig_dir,
        stem=stem,
        component_mode=residualized_component,
    )
    if residualized_path is not None:
        saved.append(residualized_path)

    normalized_capture_path = _plot_normalized_capture_summary(
        dataset,
        fig_dir=fig_dir,
        stem=stem,
    )
    if normalized_capture_path is not None:
        saved.append(normalized_capture_path)

    return saved


def plot_repeat_dimensionality_summary(
    repeat_table: Any,
    *,
    fig_dir: Path,
    stem: str,
    residualized_repeat_table: Any | None = None,
    residualized_component: ResidualPowerComponentMode = DEFAULT_RESIDUALIZED_COMPONENT,
) -> Path | None:
    """Save dimensionality means and seed-to-seed variability for repeat runs."""
    if repeat_table is None or repeat_table.empty:
        return None
    if int(repeat_table["repeat_index"].nunique()) < 2:
        return None

    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    within = repeat_table[
        repeat_table["source_condition"] == repeat_table["target_condition"]
    ].copy()
    if within.empty:
        return None
    residualized = None
    if residualized_repeat_table is not None and not residualized_repeat_table.empty:
        residualized = residualized_repeat_table.loc[
            residualized_repeat_table["residualized_component"] == residualized_component
        ].copy()
        if residualized.empty:
            residualized = None

    fig, axes = plt.subplots(figsize=(10, 4), ncols=3)
    dark_epoch = str(within["dark_epoch"].iloc[0]) if "dark_epoch" in within else "dark"
    light_epoch = str(within["light_epoch"].iloc[0]) if "light_epoch" in within else "light"
    labels = [f"dark\n{dark_epoch}", f"light\n{light_epoch}"]
    if residualized is not None:
        labels.append("light resid")
    condition_positions = np.arange(len(labels), dtype=float)
    metric_specs = (
        (
            "source_cv_participation_ratio",
            "residualized_light_cv_participation_ratio",
            "Participation ratio",
            "cvPCA PR",
        ),
        (
            "source_n_components_80",
            "residualized_light_cv_n_components_80",
            "PCs for 80%",
            "n PCs",
        ),
        (
            "source_n_components_90",
            "residualized_light_cv_n_components_90",
            "PCs for 90%",
            "n PCs",
        ),
    )
    rng = np.random.default_rng(0)
    for axis, (column, residualized_column, title, ylabel) in zip(
        axes,
        metric_specs,
        strict=True,
    ):
        means: list[float] = []
        sds: list[float] = []
        value_groups: list[np.ndarray] = []
        for condition in CONDITIONS:
            values = np.asarray(
                within.loc[within["source_condition"] == condition, column],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            sds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
            value_groups.append(values)
        if residualized is not None:
            values = np.asarray(residualized[residualized_column], dtype=float)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            sds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
            value_groups.append(values)

        colors = ["0.35", "0.70", "tab:orange"][: len(labels)]
        axis.bar(condition_positions, means, color=colors, width=0.65)
        axis.errorbar(
            condition_positions,
            means,
            yerr=sds,
            color="black",
            linestyle="none",
            capsize=4,
            linewidth=1.0,
        )
        for position, values in zip(condition_positions, value_groups, strict=True):
            jitter = rng.uniform(-0.08, 0.08, size=values.size)
            axis.scatter(
                np.full(values.size, position) + jitter,
                values,
                color="black",
                s=12,
                alpha=0.55,
                zorder=3,
            )
        axis.set_xticks(condition_positions, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle("cvPCA dimensionality across random lap groupings")
    repeat_path = fig_dir / f"{stem}_within_cv_pr_repeats.png"
    fig.tight_layout()
    fig.savefig(repeat_path, dpi=300)
    plt.close(fig)
    return repeat_path


def _region_threshold(region: str, args: argparse.Namespace) -> float:
    """Return the configured movement firing-rate threshold for one region."""
    if region == "v1":
        return float(args.v1_min_fr_hz)
    if region == "ca1":
        return float(args.ca1_min_fr_hz)
    raise ValueError(f"Unsupported region: {region!r}")


def _output_stem(region: str, light_epoch: str, dark_epoch: str) -> str:
    """Return a stable file stem for one region and light/dark pair."""
    return f"{region}_{light_epoch}_vs_{dark_epoch}_cv_pca"


def _repeat_output_stem(stem: str, *, n_repeats: int, random_seed: int) -> str:
    """Return a seed-specific stem only when running repeat analyses."""
    if int(n_repeats) <= 1:
        return stem
    return f"{stem}_seed{int(random_seed)}"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for light/dark cvPCA."""
    parser = argparse.ArgumentParser(
        description="Fit light/dark cross-validated PCA on task-progression tuning tensors"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument("--dark-epoch", required=True, help="Epoch label to treat as dark")
    parser.add_argument(
        "--light-epoch",
        help=(
            "Optional run epoch label to treat as light. If omitted, every usable "
            "non-dark run epoch inferred from cleaned position data is analyzed "
            "as a separate light/dark pair."
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=DEFAULT_REGIONS,
        default=list(DEFAULT_REGIONS),
        help="Region(s) to analyze. Default: v1 ca1",
    )
    parser.add_argument(
        "--bin-size-cm",
        type=float,
        default=DEFAULT_BIN_SIZE_CM,
        help=(
            "Spatial bin size in cm for trajectory tuning curves. "
            f"Default: {DEFAULT_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=DEFAULT_N_GROUPS,
        help=f"Number of random disjoint lap groups. Default: {DEFAULT_N_GROUPS}",
    )
    parser.add_argument(
        "--min-occupancy-s",
        type=float,
        default=DEFAULT_MIN_OCCUPANCY_S,
        help=f"Minimum occupancy per retained bin and group. Default: {DEFAULT_MIN_OCCUPANCY_S}",
    )
    parser.add_argument(
        "--unit-filter-mode",
        choices=("shared-active", "dark-active", "union-active"),
        default=DEFAULT_UNIT_FILTER_MODE,
        help=f"Neuron inclusion rule. Default: {DEFAULT_UNIT_FILTER_MODE}",
    )
    parser.add_argument(
        "--normalization",
        choices=("zscore", "center"),
        default=DEFAULT_NORMALIZATION,
        help="Per-fold normalization after condition-centering. Default: zscore",
    )
    parser.add_argument(
        "--v1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help="Minimum movement firing rate for V1 units. Default: 0.5",
    )
    parser.add_argument(
        "--ca1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help="Minimum movement firing rate for CA1 units. Default: 0.0",
    )
    parser.add_argument(
        "--min-condition-sd-hz",
        type=float,
        default=DEFAULT_MIN_CONDITION_SD_HZ,
        help=(
            "Minimum repeat-averaged condition SD for numerical inclusion. "
            f"Default: {DEFAULT_MIN_CONDITION_SD_HZ}"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Base random seed for lap grouping. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--n-random-repeats",
        type=int,
        default=DEFAULT_N_RANDOM_REPEATS,
        help=(
            "Number of random lap groupings to run. Values greater than 1 save "
            "seed-specific outputs plus a repeat summary and PR error-bar figure. "
            f"Default: {DEFAULT_N_RANDOM_REPEATS}"
        ),
    )
    parser.add_argument(
        "--save-residual-matrices",
        action="store_true",
        help="Also save component-wise residual matrices in the NetCDF output.",
    )
    parser.add_argument(
        "--residual-power-component",
        choices=("pr", "80", "90"),
        default=DEFAULT_RESIDUAL_POWER_COMPONENT,
        help=(
            "Component cutoff for the residual-power trajectory figure when "
            "--save-residual-matrices is used. Default: pr"
        ),
    )
    parser.add_argument(
        "--residualized-component",
        choices=("pr", "80", "90"),
        default=DEFAULT_RESIDUALIZED_COMPONENT,
        help=(
            "Dark component cutoff shown in the residualized-light dimensionality "
            "figure. All cutoffs are saved in the NetCDF and summary table. Default: pr"
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip summary figure generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for saved NetCDF and parquet outputs. Default: "
            "analysis_path / 'signal_dim' / 'cv_pca'"
        ),
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        help=(
            "Directory for saved figures. Default: "
            "analysis_path / 'figs' / 'signal_dim' / 'cv_pca'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the light/dark cvPCA workflow for one session."""
    args = parse_arguments()
    _validate_group_count(args.n_groups)
    if args.n_random_repeats < 1:
        raise ValueError("--n-random-repeats must be positive.")
    if args.bin_size_cm <= 0.0:
        raise ValueError("--bin-size-cm must be positive.")
    if args.min_occupancy_s <= 0.0:
        raise ValueError("--min-occupancy-s must be positive.")

    session = load_signal_dim_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=list(args.regions),
        position_offset=DEFAULT_POSITION_OFFSET,
        speed_threshold_cm_s=DEFAULT_SPEED_THRESHOLD_CM_S,
    )
    run_epochs = list(session["run_epochs"])
    light_epochs, dark_epoch = get_light_and_dark_epochs(
        run_epochs,
        args.light_epoch,
        args.dark_epoch,
    )

    analysis_path = session["analysis_path"]
    output_dir = args.output_dir or get_signal_dim_output_dir(
        analysis_path,
        DEFAULT_OUTPUT_DIRNAME,
    )
    fig_dir = args.fig_dir or get_signal_dim_figure_dir(
        analysis_path,
        DEFAULT_OUTPUT_DIRNAME,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    saved_datasets: list[Path] = []
    saved_tables: list[Path] = []
    saved_figures: list[Path] = []
    skipped_pairs: list[dict[str, Any]] = []

    for region in args.regions:
        min_fr = _region_threshold(region, args)
        for light_epoch in light_epochs:
            stem = _output_stem(region, light_epoch, dark_epoch)
            repeat_tables: list[Any] = []
            residualized_repeat_tables: list[Any] = []
            normalized_capture_repeat_tables: list[Any] = []
            for repeat_index in range(args.n_random_repeats):
                random_seed = int(args.random_seed) + repeat_index
                output_stem = _repeat_output_stem(
                    stem,
                    n_repeats=args.n_random_repeats,
                    random_seed=random_seed,
                )
                settings = {
                    "animal_name": args.animal_name,
                    "date": args.date,
                    "region": region,
                    "light_epoch": light_epoch,
                    "dark_epoch": dark_epoch,
                    "data_root": args.data_root,
                    "run_epochs": run_epochs,
                    "bin_size_cm": args.bin_size_cm,
                    "n_groups": args.n_groups,
                    "min_occupancy_s": args.min_occupancy_s,
                    "unit_filter_mode": args.unit_filter_mode,
                    "normalization": args.normalization,
                    "min_firing_rate_hz": min_fr,
                    "min_condition_sd_hz": args.min_condition_sd_hz,
                    "random_seed": random_seed,
                    "n_random_repeats": args.n_random_repeats,
                    "save_residual_matrices": args.save_residual_matrices,
                }
                print(
                    f"Running cvPCA for {args.animal_name} {args.date} "
                    f"{region} {light_epoch} vs {dark_epoch} "
                    f"seed {random_seed}."
                )
                try:
                    pair_tensors = build_pairwise_tuning_tensors(
                        session,
                        region=region,
                        light_epoch=light_epoch,
                        dark_epoch=dark_epoch,
                        bin_size_cm=args.bin_size_cm,
                        n_groups=args.n_groups,
                        min_occupancy_s=args.min_occupancy_s,
                        group_seed=random_seed,
                        min_firing_rate_hz=min_fr,
                        min_condition_sd_hz=args.min_condition_sd_hz,
                        unit_filter_mode=args.unit_filter_mode,
                    )
                    metrics = compute_cv_pca_metrics(
                        {"dark": pair_tensors.dark, "light": pair_tensors.light},
                        unit_classes=pair_tensors.unit_classes,
                        normalization=args.normalization,
                        min_scale=args.min_condition_sd_hz,
                        save_residual_matrices=args.save_residual_matrices,
                    )
                except Exception as exc:
                    skipped_pairs.append(
                        {
                            "region": region,
                            "light_epoch": light_epoch,
                            "dark_epoch": dark_epoch,
                            "repeat_index": repeat_index,
                            "random_seed": random_seed,
                            "reason": str(exc),
                        }
                    )
                    print(
                        f"Skipping {region} {light_epoch} vs {dark_epoch} "
                        f"seed {random_seed}: {exc}"
                    )
                    continue

                dataset = build_result_dataset(
                    pair_tensors=pair_tensors,
                    metrics=metrics,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    settings=settings,
                )
                dataset_path = output_dir / f"{output_stem}.nc"
                dataset.to_netcdf(dataset_path)
                saved_datasets.append(dataset_path)

                summary_table = build_summary_table(dataset, settings=settings)
                if args.n_random_repeats > 1:
                    summary_table.insert(0, "repeat_index", repeat_index)
                    summary_table.insert(1, "random_seed", random_seed)
                    repeat_tables.append(summary_table)
                table_path = output_dir / f"{output_stem}_summary.parquet"
                summary_table.to_parquet(table_path, index=False)
                saved_tables.append(table_path)

                residualized_table = build_residualized_light_summary_table(
                    dataset,
                    settings=settings,
                )
                normalized_capture_table = build_normalized_capture_summary_table(
                    dataset,
                    settings=settings,
                )
                if args.n_random_repeats > 1:
                    residualized_table.insert(0, "repeat_index", repeat_index)
                    residualized_table.insert(1, "random_seed", random_seed)
                    normalized_capture_table.insert(0, "repeat_index", repeat_index)
                    normalized_capture_table.insert(1, "random_seed", random_seed)
                    residualized_repeat_tables.append(residualized_table)
                    normalized_capture_repeat_tables.append(normalized_capture_table)
                residualized_table_path = (
                    output_dir / f"{output_stem}_residualized_light_summary.parquet"
                )
                residualized_table.to_parquet(residualized_table_path, index=False)
                saved_tables.append(residualized_table_path)
                normalized_capture_table_path = (
                    output_dir / f"{output_stem}_normalized_capture_summary.parquet"
                )
                normalized_capture_table.to_parquet(
                    normalized_capture_table_path,
                    index=False,
                )
                saved_tables.append(normalized_capture_table_path)

                if not args.no_figures and args.n_random_repeats == 1:
                    saved_figures.extend(
                        plot_summary_figures(
                            dataset,
                            fig_dir=fig_dir,
                            stem=output_stem,
                            residual_power_component=args.residual_power_component,
                            residualized_component=args.residualized_component,
                        )
                    )
                print(
                    f"Saved cvPCA outputs for {region} {light_epoch} "
                    f"vs {dark_epoch} seed {random_seed}."
                )

            if args.n_random_repeats > 1 and repeat_tables:
                import pandas as pd

                repeat_table = pd.concat(repeat_tables, ignore_index=True)
                repeat_table_path = output_dir / f"{stem}_random_repeats_summary.parquet"
                repeat_table.to_parquet(repeat_table_path, index=False)
                saved_tables.append(repeat_table_path)
                residualized_repeat_table = (
                    pd.concat(residualized_repeat_tables, ignore_index=True)
                    if residualized_repeat_tables
                    else None
                )
                if residualized_repeat_table is not None:
                    residualized_repeat_table_path = (
                        output_dir
                        / f"{stem}_residualized_light_random_repeats_summary.parquet"
                    )
                    residualized_repeat_table.to_parquet(
                        residualized_repeat_table_path,
                        index=False,
                    )
                    saved_tables.append(residualized_repeat_table_path)
                if normalized_capture_repeat_tables:
                    normalized_capture_repeat_table = pd.concat(
                        normalized_capture_repeat_tables,
                        ignore_index=True,
                    )
                    normalized_capture_repeat_table_path = (
                        output_dir
                        / f"{stem}_normalized_capture_random_repeats_summary.parquet"
                    )
                    normalized_capture_repeat_table.to_parquet(
                        normalized_capture_repeat_table_path,
                        index=False,
                    )
                    saved_tables.append(normalized_capture_repeat_table_path)
                if not args.no_figures:
                    repeat_figure = plot_repeat_dimensionality_summary(
                        repeat_table,
                        fig_dir=fig_dir,
                        stem=stem,
                        residualized_repeat_table=residualized_repeat_table,
                        residualized_component=args.residualized_component,
                    )
                    if repeat_figure is not None:
                        saved_figures.append(repeat_figure)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.signal_dim.cv_pca",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": args.regions,
            "run_epochs": run_epochs,
            "light_epochs": light_epochs,
            "dark_epoch": dark_epoch,
            "bin_size_cm": args.bin_size_cm,
            "n_groups": args.n_groups,
            "min_occupancy_s": args.min_occupancy_s,
            "unit_filter_mode": args.unit_filter_mode,
            "normalization": args.normalization,
            "v1_min_fr_hz": args.v1_min_fr_hz,
            "ca1_min_fr_hz": args.ca1_min_fr_hz,
            "min_condition_sd_hz": args.min_condition_sd_hz,
            "random_seed": args.random_seed,
            "n_random_repeats": args.n_random_repeats,
            "save_residual_matrices": args.save_residual_matrices,
            "residual_power_component": args.residual_power_component,
            "residualized_component": args.residualized_component,
            "output_dir": output_dir,
            "fig_dir": None if args.no_figures else fig_dir,
        },
        outputs={
            "saved_datasets": saved_datasets,
            "saved_summary_tables": saved_tables,
            "saved_figures": saved_figures,
            "skipped_pairs": skipped_pairs,
            "sources": session["position_source"],
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
