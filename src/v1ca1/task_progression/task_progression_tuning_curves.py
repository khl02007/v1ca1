from __future__ import annotations

"""Compute and save place and task-progression tuning curves for one session.

This script loads one dataset through the shared task-progression session
helpers, rebuilds the movement-restricted linear-position and combined
task-progression coordinates for each run epoch, computes tuning curves for
each region and run epoch, smooths those curves in the same way used by the
encoding workflow, and saves the result as NetCDF-backed xarray outputs.

Each saved file preserves one tuning curve as an xarray `DataArray`, written
under `analysis_path / "task_progression_tuning_curves"`. Run metadata is
recorded under `analysis_path / "v1ca1_log"`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    build_combined_task_progression_bins,
    build_linear_position_bins,
    get_analysis_path,
    prepare_task_progression_session,
)


DEFAULT_SIGMA_BINS = 1.0


def smooth_pf_along_position_nan_aware(
    pf: Any,
    pos_dim: str,
    sigma_bins: float,
    *,
    eps: float = 1e-12,
    mode: str = "nearest",
) -> Any:
    """Smooth a tuning curve without turning unsupported bins into zeros."""
    axis = pf.get_axis_num(pos_dim)
    values = np.asarray(pf.values, dtype=np.float64)

    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)

    numerator = gaussian_filter1d(filled, sigma=sigma_bins, axis=axis, mode=mode)
    denominator = gaussian_filter1d(
        mask.astype(np.float64),
        sigma=sigma_bins,
        axis=axis,
        mode=mode,
    )
    smoothed = numerator / np.maximum(denominator, eps)
    smoothed = np.where(denominator > eps, smoothed, np.nan)
    return pf.copy(data=smoothed)


def compute_tuning_curves_for_epoch(
    spikes: Any,
    linear_position: Any,
    task_progression: Any,
    movement_interval: Any,
    position_bins: np.ndarray,
    task_progression_bins: np.ndarray,
    sigma_bins: float,
) -> tuple[Any, Any]:
    """Compute smoothed place and task-progression tuning curves for one epoch."""
    import pynapple as nap

    place_tuning = nap.compute_tuning_curves(
        data=spikes,
        features=linear_position,
        bins=[position_bins],
        epochs=movement_interval,
        feature_names=["linpos"],
    )
    place_tuning = smooth_pf_along_position_nan_aware(
        place_tuning,
        pos_dim="linpos",
        sigma_bins=sigma_bins,
    )

    task_progression_tuning = nap.compute_tuning_curves(
        data=spikes,
        features=task_progression,
        bins=[task_progression_bins],
        epochs=movement_interval,
        feature_names=["tp"],
    )
    task_progression_tuning = smooth_pf_along_position_nan_aware(
        task_progression_tuning,
        pos_dim="tp",
        sigma_bins=sigma_bins,
    )
    return place_tuning, task_progression_tuning


def _make_netcdf_safe_attr_value(value: Any) -> Any:
    """Convert one attribute value into a NetCDF-safe representation."""
    def _to_json_safe(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [_to_json_safe(item) for item in obj.tolist()]
        if isinstance(obj, dict):
            return {str(key): _to_json_safe(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_safe(item) for item in obj]
        return obj

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return json.dumps(_to_json_safe(value))
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_to_json_safe(value))
    return str(value)


def prepare_tuning_curve_for_save(
    tuning_curve: Any,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    model_name: str,
) -> Any:
    """Return one tuning curve with NetCDF-safe attrs for xarray export."""
    output = tuning_curve.rename("firing_rate_hz").copy()
    output.attrs = {
        key: _make_netcdf_safe_attr_value(value)
        for key, value in getattr(tuning_curve, "attrs", {}).items()
    }
    output.attrs.update(
        {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "model_name": model_name,
        }
    )
    return output


def save_tuning_curves(
    tuning_curves: dict[str, dict[str, dict[str, Any]]],
    data_dir: Path,
) -> list[Path]:
    """Write one NetCDF tuning curve per region, epoch, and model."""
    saved_paths: list[Path] = []
    for region, region_curves in tuning_curves.items():
        for epoch, model_curves in region_curves.items():
            for model_name, tuning_curve in model_curves.items():
                path = data_dir / f"{region}_{epoch}_{model_name}_tuning_curves.nc"
                tuning_curve.to_netcdf(path)
                saved_paths.append(path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the tuning-curve export workflow."""
    parser = argparse.ArgumentParser(
        description="Compute and save place and task-progression tuning curves"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore per epoch. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in tuning-curve bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    return parser.parse_args()


def main() -> None:
    """Compute and save place and task-progression tuning curves for one session."""
    args = parse_arguments()
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = analysis_path / "task_progression_tuning_curves"
    data_dir.mkdir(parents=True, exist_ok=True)

    position_bins = build_linear_position_bins(args.animal_name)
    task_progression_bins = build_combined_task_progression_bins(args.animal_name)

    tuning_curves_by_region: dict[str, dict[str, dict[str, Any]]] = {region: {} for region in REGIONS}
    for region in REGIONS:
        for epoch in session["run_epochs"]:
            place_tuning, task_progression_tuning = compute_tuning_curves_for_epoch(
                spikes=session["spikes_by_region"][region],
                linear_position=session["linear_position_by_run"][epoch],
                task_progression=session["task_progression_by_run"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                position_bins=position_bins,
                task_progression_bins=task_progression_bins,
                sigma_bins=args.sigma_bins,
            )
            tuning_curves_by_region[region][epoch] = {
                "place": prepare_tuning_curve_for_save(
                    place_tuning,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    model_name="place",
                ),
                "task_progression": prepare_tuning_curve_for_save(
                    task_progression_tuning,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    model_name="task_progression",
                ),
            }

    saved_netcdf_paths = save_tuning_curves(tuning_curves_by_region, data_dir=data_dir)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.task_progression_tuning_curves",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "sigma_bins": args.sigma_bins,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "n_position_bins": int(len(position_bins) - 1),
            "n_task_progression_bins": int(len(task_progression_bins) - 1),
            "saved_netcdf_paths": saved_netcdf_paths,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
