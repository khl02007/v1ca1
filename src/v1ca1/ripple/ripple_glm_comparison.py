from __future__ import annotations

"""Compare legacy and current ripple GLM inputs and ripple outputs for one epoch."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT
from v1ca1.ripple.ripple_glm import (
    DEFAULT_MAXITER,
    DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE,
    DEFAULT_MIN_SPIKES_PER_RIPPLE,
    DEFAULT_N_SHUFFLES_RIPPLE,
    DEFAULT_N_SPLITS,
    DEFAULT_RIDGE_STRENGTH,
    DEFAULT_RIPPLE_WINDOW_S,
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    DEFAULT_SHUFFLE_SEED,
    DEFAULT_TOL,
    RIPPLE_SELECTION_MODE_ALL,
    fit_ripple_glm_train_on_ripple,
    format_ridge_strength_suffix,
    format_ripple_selection_suffix,
    format_ripple_window_suffix,
    prepare_ripple_glm_session,
)

METRIC_NAMES = ("pseudo_r2", "mae", "devexp", "bits_per_spike")


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate argument ranges for the comparison workflow."""
    if args.ripple_window_s <= 0:
        raise ValueError("--ripple-window-s must be positive.")
    if not np.isfinite(args.ripple_window_offset_s):
        raise ValueError("--ripple-window-offset-s must be finite.")
    if args.min_spikes_per_ripple < 0:
        raise ValueError("--min-spikes-per-ripple must be non-negative.")
    if args.min_ca1_spikes_per_ripple < 0:
        raise ValueError("--min-ca1-spikes-per-ripple must be non-negative.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.n_shuffles_ripple < 0:
        raise ValueError("--n-shuffles-ripple must be non-negative.")
    if args.ridge_strength < 0:
        raise ValueError("--ridge-strength must be non-negative.")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive.")
    if args.tol <= 0:
        raise ValueError("--tol must be positive.")


def ensure_epoch(session: dict[str, Any], epoch: str, *, label: str) -> None:
    """Validate that one epoch exists in the requested session payload."""
    if epoch not in session["epoch_tags"]:
        raise ValueError(
            f"Epoch {epoch!r} was not found in {label} epoch tags: {session['epoch_tags']!r}"
        )
    if epoch not in session["ripple_tables"]:
        raise ValueError(f"Epoch {epoch!r} is missing ripple events in {label}.")
    if epoch not in session["epoch_intervals"]:
        raise ValueError(f"Epoch {epoch!r} is missing interval bounds in {label}.")


def build_fit_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Return the fit-parameter bundle shared across the comparison runs."""
    return {
        "ripple_window_s": float(args.ripple_window_s),
        "ripple_window_offset_s": float(args.ripple_window_offset_s),
        "min_spikes_per_ripple": float(args.min_spikes_per_ripple),
        "min_ca1_spikes_per_ripple": float(args.min_ca1_spikes_per_ripple),
        "n_splits": int(args.n_splits),
        "n_shuffles_ripple": int(args.n_shuffles_ripple),
        "ridge_strength": float(args.ridge_strength),
        "shuffle_seed": int(args.shuffle_seed),
        "maxiter": int(args.maxiter),
        "tol": float(args.tol),
    }


def default_legacy_output_path(
    analysis_path: Path,
    *,
    epoch: str,
    ripple_window_s: float,
    min_spikes_per_ripple: float,
    ridge_strength: float,
) -> Path:
    """Return the default legacy .npz output path for one epoch."""
    return (
        analysis_path
        / "ripple"
        / "glm_results_train_on_ripple"
        / (
            f"{epoch}_train_ripple_predict_ripple_and_pre_exclude_ripples_"
            f"False_ripple_window_{ripple_window_s}"
            f"_min_spikes_per_ripple_{min_spikes_per_ripple}"
            f"_ridge_strength_{ridge_strength}.npz"
        )
    )


def default_current_output_path(
    analysis_path: Path,
    *,
    epoch: str,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ridge_strength: float,
) -> Path:
    """Return the default current .nc output path for one epoch."""
    ripple_selection_suffix = format_ripple_selection_suffix(RIPPLE_SELECTION_MODE_ALL)
    return (
        analysis_path
        / "ripple_glm"
        / (
            f"{epoch}_{format_ripple_window_suffix(ripple_window_s, ripple_window_offset_s=ripple_window_offset_s)}_"
            f"{ripple_selection_suffix}_"
            f"{format_ridge_strength_suffix(ridge_strength)}_samplewise_ripple_glm.nc"
        )
    )


def default_report_path(
    analysis_path: Path,
    *,
    epoch: str,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    ridge_strength: float,
    legacy_date: str,
) -> Path:
    """Return the default JSON report path for one comparison run."""
    return (
        analysis_path
        / "ripple_glm"
        / "comparison"
        / (
            f"{epoch}_{format_ripple_window_suffix(ripple_window_s, ripple_window_offset_s=ripple_window_offset_s)}_"
            f"{format_ridge_strength_suffix(ridge_strength)}_"
            f"legacy_{legacy_date}_comparison.json"
        )
    )


def build_ripple_ep(
    ripple_table: Any,
    *,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> Any:
    """Build the ripple IntervalSet using the same rules as the current fitter."""
    import pynapple as nap

    if "start_time" in ripple_table.columns:
        ripple_starts = np.asarray(ripple_table["start_time"], dtype=float)
        ripple_ends = np.asarray(ripple_table["end_time"], dtype=float)
    else:
        ripple_starts = np.asarray(ripple_table["start"], dtype=float)
        ripple_ends = np.asarray(ripple_table["end"], dtype=float)

    ripple_starts = ripple_starts + float(ripple_window_offset_s)
    if ripple_window_s is None:
        return nap.IntervalSet(
            start=ripple_starts,
            end=ripple_ends + float(ripple_window_offset_s),
            time_units="s",
        )
    return nap.IntervalSet(
        start=ripple_starts,
        end=ripple_starts + float(ripple_window_s),
        time_units="s",
    )


def summarize_session_inputs(
    session: dict[str, Any],
    *,
    epoch: str,
    ripple_window_s: float | None,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
) -> dict[str, Any]:
    """Summarize one session's ripple inputs for one epoch."""
    ripple_table = session["ripple_tables"][epoch]
    ripple_ep = build_ripple_ep(
        ripple_table,
        ripple_window_s=ripple_window_s,
        ripple_window_offset_s=ripple_window_offset_s,
    )
    return {
        "analysis_path": str(session["analysis_path"]),
        "sources": session["sources"],
        "raw_ripple_rows": int(len(ripple_table)),
        "fixed_window_ripple_count": int(len(ripple_ep)),
        "ripple_table_columns": [str(column_name) for column_name in ripple_table.columns],
    }


def run_current_fit(
    session: dict[str, Any],
    *,
    epoch: str,
    fit_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Run the current ripple GLM fitter for one epoch."""
    return fit_ripple_glm_train_on_ripple(
        epoch=epoch,
        spikes=session["spikes_by_region"],
        epoch_interval=session["epoch_intervals"][epoch],
        ripple_table=session["ripple_tables"][epoch],
        min_spikes_per_ripple=fit_parameters["min_spikes_per_ripple"],
        min_ca1_spikes_per_ripple=fit_parameters["min_ca1_spikes_per_ripple"],
        n_shuffles_ripple=fit_parameters["n_shuffles_ripple"],
        shuffle_seed=fit_parameters["shuffle_seed"],
        ripple_window_s=fit_parameters["ripple_window_s"],
        ripple_window_offset_s=fit_parameters["ripple_window_offset_s"],
        n_splits=fit_parameters["n_splits"],
        ridge_strength=fit_parameters["ridge_strength"],
        maxiter=fit_parameters["maxiter"],
        tol=fit_parameters["tol"],
    )


def metric_mean(values: np.ndarray) -> float | None:
    """Return the NaN-aware mean for one metric array."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return None
    return float(np.nanmean(array))


def build_metric_means(metric_arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Return compact metric means for the available metric partitions."""
    return {
        partition_name: {
            metric_name: metric_mean(metric_values)
            for metric_name, metric_values in partition_metrics.items()
        }
        for partition_name, partition_metrics in metric_arrays.items()
    }


def normalize_result_from_fit(
    *,
    label: str,
    result_source: str,
    fit_parameters: dict[str, Any],
    input_summary: dict[str, Any],
    results: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Normalize one fresh fit output into a comparison-ready record."""
    metric_arrays = {
        "ripple": {
            "pseudo_r2": np.asarray(results["pseudo_r2_ripple_folds"], dtype=float),
            "mae": np.asarray(results["mae_ripple_folds"], dtype=float),
            "devexp": np.asarray(results["devexp_ripple_folds"], dtype=float),
            "bits_per_spike": np.asarray(results["bits_per_spike_ripple_folds"], dtype=float),
        },
    }
    v1_unit_ids = np.asarray(results["v1_unit_ids"])
    ca1_unit_ids = np.asarray(results["ca1_unit_ids"])
    return {
        "label": label,
        "result_source": result_source,
        "output_path": None if output_path is None else str(output_path),
        "fit_parameters": dict(fit_parameters),
        "input_summary": dict(input_summary),
        "n_ripples": int(results["n_ripples"]),
        "n_v1_units": int(len(v1_unit_ids)),
        "n_ca1_units": int(len(ca1_unit_ids)),
        "v1_unit_ids": v1_unit_ids,
        "ca1_unit_ids": ca1_unit_ids,
        "metric_arrays": metric_arrays,
        "metric_means": build_metric_means(metric_arrays),
    }


def load_legacy_saved_output(
    output_path: Path,
    *,
    label: str,
    fit_parameters: dict[str, Any],
    input_summary: dict[str, Any],
) -> dict[str, Any]:
    """Load one saved legacy .npz output into a comparison-ready record."""
    loaded = np.load(output_path, allow_pickle=True)
    metric_arrays = {
        "ripple": {
            "pseudo_r2": np.asarray(loaded["pseudo_r2_ripple_folds"], dtype=float),
            "mae": np.asarray(loaded["mae_ripple_folds"], dtype=float),
            "devexp": np.asarray(loaded["devexp_ripple_folds"], dtype=float),
            "bits_per_spike": np.asarray(loaded["bits_per_spike_ripple_folds"], dtype=float),
        },
    }
    saved_parameters = {
        "ripple_window_s": (
            None
            if np.asarray(loaded["ripple_window"]).dtype == object
            and loaded["ripple_window"].item() is None
            else float(np.asarray(loaded["ripple_window"]).reshape(-1)[0])
        ),
        "min_spikes_per_ripple": float(
            np.asarray(loaded["min_spikes_per_ripple"]).reshape(-1)[0]
        ),
        "min_ca1_spikes_per_ripple": fit_parameters["min_ca1_spikes_per_ripple"],
        "n_splits": int(np.asarray(loaded["n_splits"]).reshape(-1)[0]),
        "n_shuffles_ripple": int(np.asarray(loaded["n_shuffles_ripple"]).reshape(-1)[0]),
        "ridge_strength": fit_parameters["ridge_strength"],
        "shuffle_seed": int(np.asarray(loaded["random_seed"]).reshape(-1)[0]),
        "maxiter": fit_parameters["maxiter"],
        "tol": fit_parameters["tol"],
    }
    v1_unit_ids = np.asarray(loaded["v1_unit_ids"])
    ca1_unit_ids = np.asarray(loaded["ca1_unit_ids"])
    return {
        "label": label,
        "result_source": "legacy_saved_npz",
        "output_path": str(output_path),
        "fit_parameters": saved_parameters,
        "input_summary": dict(input_summary),
        "n_ripples": int(np.asarray(loaded["n_ripples"]).reshape(-1)[0]),
        "n_v1_units": int(len(v1_unit_ids)),
        "n_ca1_units": int(len(ca1_unit_ids)),
        "v1_unit_ids": v1_unit_ids,
        "ca1_unit_ids": ca1_unit_ids,
        "metric_arrays": metric_arrays,
        "metric_means": build_metric_means(metric_arrays),
    }


def load_current_saved_output(
    output_path: Path,
    *,
    label: str,
    input_summary: dict[str, Any],
) -> dict[str, Any]:
    """Load one saved current .nc output into a comparison-ready record."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to load saved ripple GLM NetCDF outputs."
        ) from exc

    dataset = xr.load_dataset(output_path)
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    metric_arrays = {
        "ripple": {
            "pseudo_r2": np.asarray(dataset["pseudo_r2_ripple_folds"].values, dtype=float),
            "mae": np.asarray(dataset["mae_ripple_folds"].values, dtype=float),
            "devexp": np.asarray(dataset["devexp_ripple_folds"].values, dtype=float),
            "bits_per_spike": np.asarray(
                dataset["bits_per_spike_ripple_folds"].values,
                dtype=float,
            ),
        },
    }
    v1_unit_ids = np.asarray(dataset["unit"].values)
    ca1_unit_ids = np.asarray(dataset["ca1_unit_id"].values)
    return {
        "label": label,
        "result_source": "current_saved_netcdf",
        "output_path": str(output_path),
        "fit_parameters": fit_parameters,
        "input_summary": dict(input_summary),
        "n_ripples": int(dataset.attrs["n_ripples"]),
        "n_v1_units": int(len(v1_unit_ids)),
        "n_ca1_units": int(len(ca1_unit_ids)),
        "v1_unit_ids": v1_unit_ids,
        "ca1_unit_ids": ca1_unit_ids,
        "metric_arrays": metric_arrays,
        "metric_means": build_metric_means(metric_arrays),
    }


def summarize_array_diff(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return compact agreement statistics for one pair of aligned arrays."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
    if not np.any(finite_mask):
        return {
            "n_finite_compared": 0,
            "allclose": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    abs_diff = np.abs(left_values[finite_mask] - right_values[finite_mask])
    return {
        "n_finite_compared": int(abs_diff.size),
        "allclose": bool(
            np.allclose(left_values, right_values, equal_nan=True, rtol=1e-6, atol=1e-6)
        ),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
    }


def build_unit_index(unit_ids: np.ndarray) -> dict[Any, int]:
    """Return one index lookup for a unit-id array."""
    return {
        unit_id.item() if hasattr(unit_id, "item") else unit_id: idx
        for idx, unit_id in enumerate(unit_ids)
    }


def compare_runs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two run summaries when their inputs and retained units match."""
    left_v1 = np.asarray(left["v1_unit_ids"])
    right_v1 = np.asarray(right["v1_unit_ids"])
    left_ca1 = np.asarray(left["ca1_unit_ids"])
    right_ca1 = np.asarray(right["ca1_unit_ids"])

    issues: list[str] = []
    if left["n_ripples"] != right["n_ripples"]:
        issues.append(
            f"n_ripples differs: {left['n_ripples']} vs {right['n_ripples']}"
        )

    left_v1_set = set(left_v1.tolist())
    right_v1_set = set(right_v1.tolist())
    left_ca1_set = set(left_ca1.tolist())
    right_ca1_set = set(right_ca1.tolist())
    if left_v1_set != right_v1_set:
        issues.append(
            "V1 unit ids differ: "
            f"left-only={sorted(left_v1_set - right_v1_set)!r}, "
            f"right-only={sorted(right_v1_set - left_v1_set)!r}"
        )
    if left_ca1_set != right_ca1_set:
        issues.append(
            "CA1 unit ids differ: "
            f"left-only={sorted(left_ca1_set - right_ca1_set)!r}, "
            f"right-only={sorted(right_ca1_set - left_ca1_set)!r}"
        )

    comparison = {
        "left": left["label"],
        "right": right["label"],
        "comparable": not issues,
        "issues": issues,
    }
    if issues:
        return comparison

    right_v1_index = build_unit_index(right_v1)
    aligned_right_indices = [
        right_v1_index[unit_id.item() if hasattr(unit_id, "item") else unit_id]
        for unit_id in left_v1
    ]
    metric_diffs: dict[str, Any] = {"ripple": {}}
    for metric_name in METRIC_NAMES:
        left_ripple = np.asarray(left["metric_arrays"]["ripple"][metric_name], dtype=float)
        right_ripple = np.asarray(
            right["metric_arrays"]["ripple"][metric_name],
            dtype=float,
        )[:, aligned_right_indices]
        metric_diffs["ripple"][metric_name] = summarize_array_diff(left_ripple, right_ripple)

    comparison["metric_diffs"] = metric_diffs
    return comparison


def build_report_entry(run: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe summary for one run."""
    input_summary = dict(run["input_summary"])
    return {
        "label": run["label"],
        "result_source": run["result_source"],
        "output_path": run["output_path"],
        "analysis_path": input_summary["analysis_path"],
        "sources": input_summary["sources"],
        "raw_ripple_rows": input_summary["raw_ripple_rows"],
        "fixed_window_ripple_count": input_summary["fixed_window_ripple_count"],
        "n_ripples": run["n_ripples"],
        "n_v1_units": run["n_v1_units"],
        "n_ca1_units": run["n_ca1_units"],
        "v1_unit_ids": np.asarray(run["v1_unit_ids"]).tolist(),
        "ca1_unit_ids": np.asarray(run["ca1_unit_ids"]).tolist(),
        "fit_parameters": run["fit_parameters"],
        "metric_means": run["metric_means"],
    }


def write_report(output_path: Path, report: dict[str, Any]) -> Path:
    """Write one JSON comparison report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for one ripple GLM comparison run."""
    parser = argparse.ArgumentParser(
        description="Compare legacy and current ripple GLM inputs and outputs."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Current session date in YYYYMMDD format")
    parser.add_argument(
        "--epoch",
        required=True,
        help="Single epoch label to compare, for example 02_r1.",
    )
    parser.add_argument(
        "--legacy-date",
        help=(
            "Optional legacy session directory name under the same animal. "
            "Defaults to <date>_old."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--legacy-output-path",
        type=Path,
        help="Optional path to a saved legacy .npz output for the requested epoch.",
    )
    parser.add_argument(
        "--current-output-path",
        type=Path,
        help="Optional path to a saved current .nc output for the requested epoch.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional path for the JSON comparison report.",
    )
    parser.add_argument(
        "--refit-current",
        action="store_true",
        help="Run the current fitter on the current artifacts instead of loading a saved .nc.",
    )
    parser.add_argument(
        "--ripple-window-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_S,
        help=f"Fixed ripple window length in seconds. Default: {DEFAULT_RIPPLE_WINDOW_S}",
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        default=DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        help=(
            "Offset in seconds applied to the ripple window relative to ripple start time. "
            f"Default: {DEFAULT_RIPPLE_WINDOW_OFFSET_S}"
        ),
    )
    parser.add_argument(
        "--min-spikes-per-ripple",
        type=float,
        default=DEFAULT_MIN_SPIKES_PER_RIPPLE,
        help=(
            "Minimum average V1 spikes per ripple required to keep one unit. "
            f"Default: {DEFAULT_MIN_SPIKES_PER_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--min-ca1-spikes-per-ripple",
        type=float,
        default=DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE,
        help=(
            "Minimum average CA1 spikes per ripple required to keep one source unit. "
            f"Default: {DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of ripple CV folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--n-shuffles-ripple",
        type=int,
        default=DEFAULT_N_SHUFFLES_RIPPLE,
        help=(
            "Number of shuffle refits per fold for held-out ripple evaluation. "
            f"Default: {DEFAULT_N_SHUFFLES_RIPPLE}"
        ),
    )
    parser.add_argument(
        "--ridge-strength",
        type=float,
        default=DEFAULT_RIDGE_STRENGTH,
        help=f"Single ridge regularization strength to compare. Default: {DEFAULT_RIDGE_STRENGTH}",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for response shuffles. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help=f"Maximum number of LBFGS iterations. Default: {DEFAULT_MAXITER}",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"LBFGS optimizer tolerance. Default: {DEFAULT_TOL}",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Optional value to assign to CUDA_VISIBLE_DEVICES before fitting, "
            "for example '0' or '0,1'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the ripple GLM legacy-vs-current comparison for one epoch."""
    args = parse_arguments()
    validate_arguments(args)

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(
            "Setting CUDA_VISIBLE_DEVICES="
            f"{args.cuda_visible_devices!r} for ripple GLM comparison."
        )

    legacy_date = args.legacy_date if args.legacy_date is not None else f"{args.date}_old"
    fit_parameters = build_fit_parameters(args)

    legacy_session = prepare_ripple_glm_session(
        animal_name=args.animal_name,
        date=legacy_date,
        data_root=args.data_root,
    )
    current_session = prepare_ripple_glm_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    ensure_epoch(legacy_session, args.epoch, label=f"legacy session {legacy_date!r}")
    ensure_epoch(current_session, args.epoch, label=f"current session {args.date!r}")

    legacy_input_summary = summarize_session_inputs(
        legacy_session,
        epoch=args.epoch,
        ripple_window_s=fit_parameters["ripple_window_s"],
        ripple_window_offset_s=fit_parameters["ripple_window_offset_s"],
    )
    current_input_summary = summarize_session_inputs(
        current_session,
        epoch=args.epoch,
        ripple_window_s=fit_parameters["ripple_window_s"],
        ripple_window_offset_s=fit_parameters["ripple_window_offset_s"],
    )

    legacy_output_path = (
        args.legacy_output_path
        if args.legacy_output_path is not None
        else default_legacy_output_path(
            Path(legacy_input_summary["analysis_path"]),
            epoch=args.epoch,
            ripple_window_s=fit_parameters["ripple_window_s"],
            min_spikes_per_ripple=fit_parameters["min_spikes_per_ripple"],
            ridge_strength=fit_parameters["ridge_strength"],
        )
    )
    current_output_path = (
        args.current_output_path
        if args.current_output_path is not None
        else default_current_output_path(
            Path(current_input_summary["analysis_path"]),
            epoch=args.epoch,
            ripple_window_s=fit_parameters["ripple_window_s"],
            ripple_window_offset_s=fit_parameters["ripple_window_offset_s"],
            ridge_strength=fit_parameters["ridge_strength"],
        )
    )
    report_path = (
        args.output_path
        if args.output_path is not None
        else default_report_path(
            Path(current_input_summary["analysis_path"]),
            epoch=args.epoch,
            ripple_window_s=fit_parameters["ripple_window_s"],
            ripple_window_offset_s=fit_parameters["ripple_window_offset_s"],
            ridge_strength=fit_parameters["ridge_strength"],
            legacy_date=legacy_date,
        )
    )

    runs: list[dict[str, Any]] = []
    if legacy_output_path.exists():
        print(f"Loading legacy saved output: {legacy_output_path}")
        runs.append(
            load_legacy_saved_output(
                legacy_output_path,
                label="legacy_artifacts_legacy_code_saved_output",
                fit_parameters=fit_parameters,
                input_summary=legacy_input_summary,
            )
        )
    else:
        print(f"Legacy saved output not found, skipping step 1: {legacy_output_path}")

    print(
        "Running current fitter on legacy artifacts for "
        f"{args.animal_name} {legacy_date} {args.epoch}."
    )
    legacy_fit_results = run_current_fit(
        legacy_session,
        epoch=args.epoch,
        fit_parameters=fit_parameters,
    )
    runs.append(
        normalize_result_from_fit(
            label="legacy_artifacts_current_code_refit",
            result_source="current_code_refit",
            fit_parameters=fit_parameters,
            input_summary=legacy_input_summary,
            results=legacy_fit_results,
        )
    )

    if current_output_path.exists() and not args.refit_current:
        print(f"Loading current saved output: {current_output_path}")
        runs.append(
            load_current_saved_output(
                current_output_path,
                label="current_artifacts_current_code_saved_output",
                input_summary=current_input_summary,
            )
        )
    else:
        print(
            "Running current fitter on current artifacts for "
            f"{args.animal_name} {args.date} {args.epoch}."
        )
        current_fit_results = run_current_fit(
            current_session,
            epoch=args.epoch,
            fit_parameters=fit_parameters,
        )
        runs.append(
            normalize_result_from_fit(
                label="current_artifacts_current_code_refit",
                result_source="current_code_refit",
                fit_parameters=fit_parameters,
                input_summary=current_input_summary,
                results=current_fit_results,
            )
        )

    comparisons = [
        compare_runs(left_run, right_run)
        for run_index, left_run in enumerate(runs)
        for right_run in runs[run_index + 1 :]
    ]
    report = {
        "animal_name": args.animal_name,
        "date": args.date,
        "legacy_date": legacy_date,
        "epoch": args.epoch,
        "runs": [build_report_entry(run) for run in runs],
        "comparisons": comparisons,
    }
    write_report(report_path, report)
    log_path = write_run_log(
        analysis_path=Path(current_input_summary["analysis_path"]),
        script_name="v1ca1.ripple.ripple_glm_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "legacy_date": legacy_date,
            "epoch": args.epoch,
            "data_root": args.data_root,
            "legacy_output_path": legacy_output_path,
            "current_output_path": current_output_path,
            "report_path": report_path,
            "refit_current": args.refit_current,
            "fit_parameters": fit_parameters,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        outputs={
            "runs": [build_report_entry(run) for run in runs],
            "comparisons": comparisons,
            "report_path": report_path,
        },
    )
    print(f"Saved ripple GLM comparison report to {report_path}")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
