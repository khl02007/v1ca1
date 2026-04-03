from __future__ import annotations

"""Select one ridge strength per saved ripple GLM setup.

This CLI scans one session's saved `ripple_glm/*.nc` outputs and groups them by
fixed ripple window plus ripple-selection mode. Within each `(epoch, ridge)`
dataset, it averages held-out ripple deviance explained across folds per V1
unit, then takes the median across units. Session-level ridge selection uses
the mean of those per-epoch scores across the intersection of epochs that are
available and valid for every candidate ridge in the setup. When multiple
ridges are within the configured absolute tolerance of the best session-level
score, the larger ridge is selected.

The script writes two parquet tables under `analysis_path / "ripple_glm"`:

1. one setup-level summary table with one row per `(setup, ridge)`
2. one long per-epoch score table with one row per `(setup, ridge, epoch)`

A JSON run log is also written under `v1ca1_log/`.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, get_analysis_path
from v1ca1.ripple.ripple_glm import (
    DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    format_ripple_window_offset_suffix,
    format_ripple_window_suffix,
)


DEFAULT_TIE_TOL = 0.01
DEFAULT_MIN_COMMON_EPOCHS = 2
DEFAULT_MIN_FINITE_UNITS = 5

FILENAME_PATTERN = re.compile(
    r"^(?P<epoch>.+)_(?P<ripple_window_suffix>rw_[^_]+(?:_off_[^_]+)?)_"
    r"(?P<ripple_selection_mode>[^_]+)_(?P<ridge_strength_suffix>ridge_[^_]+)"
    r"_samplewise_ripple_glm\.nc$"
)

SUMMARY_COLUMNS = [
    "setup_label",
    "ripple_window_s",
    "ripple_window_offset_s",
    "ripple_window_suffix",
    "ripple_selection_mode",
    "ridge_strength",
    "ridge_strength_suffix",
    "selected",
    "selection_status",
    "is_within_tie_tol_of_best",
    "session_mean_epoch_median_devexp",
    "best_session_mean_epoch_median_devexp",
    "score_delta_from_best",
    "epoch_win_count",
    "epoch_win_fraction",
    "n_candidate_ridges",
    "candidate_ridge_strengths_json",
    "tied_ridge_strengths_json",
    "n_available_epochs",
    "available_epochs_json",
    "n_common_file_epochs",
    "common_file_epochs_json",
    "n_valid_epochs_for_ridge",
    "valid_epochs_for_ridge_json",
    "n_common_valid_epochs",
    "common_valid_epochs_json",
    "tie_tol",
    "min_common_epochs",
    "min_finite_units",
]

EPOCH_SCORE_COLUMNS = [
    "setup_label",
    "ripple_window_s",
    "ripple_window_offset_s",
    "ripple_window_suffix",
    "ripple_selection_mode",
    "epoch",
    "ridge_strength",
    "ridge_strength_suffix",
    "epoch_median_devexp",
    "n_finite_units",
    "n_units_total",
    "n_folds",
    "n_ripples",
    "epoch_score_valid",
    "in_common_file_epoch_set",
    "in_common_valid_epoch_set",
    "epoch_is_winner",
    "output_path",
]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for ripple GLM ridge selection."""
    parser = argparse.ArgumentParser(
        description="Select one ridge strength per saved ripple GLM setup"
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
        "--ripple-window-s",
        type=float,
        help="Optional fixed ripple window to analyze. Default: scan all saved windows.",
    )
    parser.add_argument(
        "--ripple-window-offset-s",
        type=float,
        help="Optional ripple-window offset to analyze. Default: scan all saved offsets.",
    )
    parser.add_argument(
        "--ripple-selection-mode",
        type=str,
        help="Optional ripple-selection mode to analyze. Default: scan all saved modes.",
    )
    parser.add_argument(
        "--ridge-strengths",
        nargs="+",
        type=float,
        help="Optional subset of ridge strengths to compare. Default: infer from saved files.",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=DEFAULT_TIE_TOL,
        help=(
            "Absolute tolerance for near-ties in session-level deviance explained. "
            f"Default: {DEFAULT_TIE_TOL}"
        ),
    )
    parser.add_argument(
        "--min-common-epochs",
        type=int,
        default=DEFAULT_MIN_COMMON_EPOCHS,
        help=(
            "Minimum number of epochs required in the shared comparison set for one setup. "
            f"Default: {DEFAULT_MIN_COMMON_EPOCHS}"
        ),
    )
    parser.add_argument(
        "--min-finite-units",
        type=int,
        default=DEFAULT_MIN_FINITE_UNITS,
        help=(
            "Minimum number of units with finite epoch scores required for one "
            f"(epoch, ridge) score. Default: {DEFAULT_MIN_FINITE_UNITS}"
        ),
    )
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate argument ranges for ridge selection."""
    if args.ripple_window_s is not None and args.ripple_window_s <= 0:
        raise ValueError("--ripple-window-s must be positive.")
    if args.ripple_window_offset_s is not None and not np.isfinite(args.ripple_window_offset_s):
        raise ValueError("--ripple-window-offset-s must be finite.")
    if args.tie_tol < 0:
        raise ValueError("--tie-tol must be non-negative.")
    if args.min_common_epochs <= 0:
        raise ValueError("--min-common-epochs must be positive.")
    if args.min_finite_units <= 0:
        raise ValueError("--min-finite-units must be positive.")
    if args.ridge_strengths is not None and any(value < 0 for value in args.ridge_strengths):
        raise ValueError("--ridge-strengths must be non-negative.")


def _parse_suffix_float(encoded_value: str) -> float:
    """Parse one filesystem-friendly encoded float value."""
    sign = -1.0 if encoded_value.startswith("m") else 1.0
    encoded_abs = encoded_value[1:] if encoded_value.startswith("m") else encoded_value
    return sign * float(encoded_abs.replace("p", "."))


def _parse_ripple_window_suffix(ripple_window_suffix: str) -> tuple[float, float]:
    """Parse one saved ripple-window suffix into window and offset seconds."""
    match = re.fullmatch(r"rw_(?P<window>[^_]+s)(?:_off_(?P<offset>[^_]+s))?", ripple_window_suffix)
    if match is None:
        raise ValueError(f"Invalid ripple-window suffix: {ripple_window_suffix!r}")
    ripple_window_s = _parse_suffix_float(match.group("window")[:-1])
    offset_suffix = match.group("offset")
    if offset_suffix is None:
        return ripple_window_s, DEFAULT_RIPPLE_WINDOW_OFFSET_S
    return ripple_window_s, _parse_suffix_float(offset_suffix[:-1])


def _parse_ridge_strength_suffix(ridge_strength_suffix: str) -> float:
    """Parse one saved ridge suffix into a float."""
    if not ridge_strength_suffix.startswith("ridge_"):
        raise ValueError(f"Invalid ridge suffix: {ridge_strength_suffix!r}")
    return float(ridge_strength_suffix.removeprefix("ridge_"))


def parse_saved_output_path(path: Path) -> dict[str, Any]:
    """Parse one saved ripple GLM NetCDF path into comparable metadata."""
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse saved ripple GLM filename: {path.name}")

    ripple_window_suffix = match.group("ripple_window_suffix")
    ridge_strength_suffix = match.group("ridge_strength_suffix")
    ripple_selection_mode = match.group("ripple_selection_mode")
    epoch = match.group("epoch")
    ripple_window_s, ripple_window_offset_s = _parse_ripple_window_suffix(ripple_window_suffix)
    return {
        "epoch": epoch,
        "ripple_window_s": ripple_window_s,
        "ripple_window_offset_s": ripple_window_offset_s,
        "ripple_window_suffix": ripple_window_suffix,
        "ripple_selection_mode": ripple_selection_mode,
        "ridge_strength": _parse_ridge_strength_suffix(ridge_strength_suffix),
        "ridge_strength_suffix": ridge_strength_suffix,
        "output_path": path,
        "setup_label": f"{ripple_window_suffix}_{ripple_selection_mode}",
    }


def _float_matches(left: float, right: float) -> bool:
    """Return whether two floats match under a tight tolerance."""
    return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))


def collect_saved_output_records(
    data_dir: Path,
    *,
    ripple_window_s: float | None = None,
    ripple_window_offset_s: float | None = None,
    ripple_selection_mode: str | None = None,
    ridge_strengths: list[float] | None = None,
) -> pd.DataFrame:
    """Scan saved ripple GLM outputs and return one row per parsed dataset."""
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*_samplewise_ripple_glm.nc")):
        try:
            row = parse_saved_output_path(path)
        except ValueError:
            continue
        if ripple_window_s is not None and not _float_matches(row["ripple_window_s"], ripple_window_s):
            continue
        if (
            ripple_window_offset_s is not None
            and not _float_matches(row["ripple_window_offset_s"], ripple_window_offset_s)
        ):
            continue
        if (
            ripple_selection_mode is not None
            and row["ripple_selection_mode"] != ripple_selection_mode
        ):
            continue
        if ridge_strengths is not None and not any(
            _float_matches(row["ridge_strength"], ridge_strength) for ridge_strength in ridge_strengths
        ):
            continue
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No saved ripple GLM NetCDF outputs matched under {data_dir}.")
    return pd.DataFrame(rows)


def compute_epoch_median_devexp(
    devexp_ripple_folds: np.ndarray,
    *,
    min_finite_units: int,
) -> dict[str, Any]:
    """Collapse held-out deviance explained to one epoch-level score."""
    values = np.asarray(devexp_ripple_folds, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            "Expected devexp_ripple_folds to have shape (fold, unit), "
            f"got {values.shape!r}."
        )

    finite_by_unit = np.isfinite(values)
    unit_has_finite = np.any(finite_by_unit, axis=0)
    unit_scores = np.full(values.shape[1], np.nan, dtype=float)
    if np.any(unit_has_finite):
        unit_scores[unit_has_finite] = np.nanmean(values[:, unit_has_finite], axis=0)

    n_finite_units = int(np.isfinite(unit_scores).sum())
    epoch_score_valid = n_finite_units >= int(min_finite_units)
    epoch_median_devexp = (
        float(np.nanmedian(unit_scores))
        if epoch_score_valid
        else np.nan
    )
    return {
        "epoch_median_devexp": epoch_median_devexp,
        "n_finite_units": n_finite_units,
        "n_units_total": int(values.shape[1]),
        "n_folds": int(values.shape[0]),
        "epoch_score_valid": bool(epoch_score_valid),
    }


def load_epoch_score_row(
    output_path: Path,
    *,
    min_finite_units: int,
) -> dict[str, Any]:
    """Load one saved NetCDF and return the epoch score summary row."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to load saved ripple GLM NetCDF outputs."
        ) from exc

    dataset = xr.load_dataset(output_path)
    try:
        score_row = compute_epoch_median_devexp(
            np.asarray(dataset["devexp_ripple_folds"].values, dtype=float),
            min_finite_units=min_finite_units,
        )
        score_row["n_ripples"] = int(dataset.attrs["n_ripples"])
        return score_row
    finally:
        close_method = getattr(dataset, "close", None)
        if callable(close_method):
            close_method()


def _choose_best_ridge(
    score_by_ridge: dict[float, float],
    *,
    tie_tol: float,
) -> tuple[float | None, float | None, list[float]]:
    """Return the selected ridge, best score, and tied ridges."""
    finite_scores = {
        float(ridge_strength): float(score)
        for ridge_strength, score in score_by_ridge.items()
        if np.isfinite(score)
    }
    if not finite_scores:
        return None, None, []

    best_score = max(finite_scores.values())
    tied_ridges = sorted(
        ridge_strength
        for ridge_strength, score in finite_scores.items()
        if best_score - score <= float(tie_tol)
    )
    return float(max(tied_ridges)), float(best_score), tied_ridges


def _json_list(values: list[Any]) -> str:
    """Return a JSON-safe compact list representation."""
    return json.dumps(values)


def summarize_setup_group(
    group: pd.DataFrame,
    *,
    tie_tol: float,
    min_common_epochs: int,
    min_finite_units: int,
    score_loader: Callable[..., dict[str, Any]] = load_epoch_score_row,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return setup-level summary rows and per-epoch score rows for one setup."""
    if group.empty:
        return [], []

    setup_label = str(group["setup_label"].iloc[0])
    ripple_window_s = float(group["ripple_window_s"].iloc[0])
    ripple_window_offset_s = (
        float(group["ripple_window_offset_s"].iloc[0])
        if "ripple_window_offset_s" in group.columns
        else DEFAULT_RIPPLE_WINDOW_OFFSET_S
    )
    ripple_window_suffix = str(group["ripple_window_suffix"].iloc[0])
    ripple_selection_mode = str(group["ripple_selection_mode"].iloc[0])
    candidate_ridges = sorted(float(value) for value in group["ridge_strength"].unique())

    epochs_by_ridge: dict[float, list[str]] = {}
    records_by_key: dict[tuple[float, str], dict[str, Any]] = {}
    for record in group.to_dict("records"):
        ridge_strength = float(record["ridge_strength"])
        epoch = str(record["epoch"])
        key = (ridge_strength, epoch)
        if key in records_by_key:
            raise ValueError(
                f"Found duplicate saved outputs for setup {setup_label!r}, "
                f"ridge {ridge_strength:.1e}, epoch {epoch!r}."
            )
        records_by_key[key] = record
        epochs_by_ridge.setdefault(ridge_strength, []).append(epoch)

    common_file_epoch_set = set(epochs_by_ridge[candidate_ridges[0]])
    for ridge_strength in candidate_ridges[1:]:
        common_file_epoch_set &= set(epochs_by_ridge[ridge_strength])
    common_file_epochs = sorted(common_file_epoch_set)

    epoch_rows: list[dict[str, Any]] = []
    for ridge_strength in candidate_ridges:
        for epoch in common_file_epochs:
            record = records_by_key[(ridge_strength, epoch)]
            score_row = score_loader(
                Path(record["output_path"]),
                min_finite_units=min_finite_units,
            )
            epoch_rows.append(
                {
                    "setup_label": setup_label,
                    "ripple_window_s": ripple_window_s,
                    "ripple_window_offset_s": ripple_window_offset_s,
                    "ripple_window_suffix": ripple_window_suffix,
                    "ripple_selection_mode": ripple_selection_mode,
                    "epoch": epoch,
                    "ridge_strength": ridge_strength,
                    "ridge_strength_suffix": str(record["ridge_strength_suffix"]),
                    "epoch_median_devexp": float(score_row["epoch_median_devexp"])
                    if np.isfinite(score_row["epoch_median_devexp"])
                    else np.nan,
                    "n_finite_units": int(score_row["n_finite_units"]),
                    "n_units_total": int(score_row["n_units_total"]),
                    "n_folds": int(score_row["n_folds"]),
                    "n_ripples": int(score_row["n_ripples"]),
                    "epoch_score_valid": bool(score_row["epoch_score_valid"]),
                    "in_common_file_epoch_set": True,
                    "in_common_valid_epoch_set": False,
                    "epoch_is_winner": False,
                    "output_path": str(record["output_path"]),
                }
            )

    epoch_rows_by_ridge: dict[float, list[dict[str, Any]]] = {ridge: [] for ridge in candidate_ridges}
    for row in epoch_rows:
        epoch_rows_by_ridge[float(row["ridge_strength"])].append(row)

    common_valid_epoch_set = set(common_file_epochs)
    for ridge_strength in candidate_ridges:
        valid_epochs_for_ridge = {
            str(row["epoch"])
            for row in epoch_rows_by_ridge[ridge_strength]
            if row["epoch_score_valid"]
        }
        common_valid_epoch_set &= valid_epochs_for_ridge
    common_valid_epochs = sorted(common_valid_epoch_set)

    for row in epoch_rows:
        row["in_common_valid_epoch_set"] = (
            row["epoch_score_valid"] and row["epoch"] in common_valid_epoch_set
        )

    if len(common_file_epochs) < int(min_common_epochs):
        selection_status = "insufficient_common_file_epochs"
        session_scores_by_ridge: dict[float, float] = {}
        selected_ridge = None
        best_score = None
        tied_ridges: list[float] = []
        epoch_winner_by_epoch: dict[str, float] = {}
    elif len(common_valid_epochs) < int(min_common_epochs):
        selection_status = "insufficient_common_valid_epochs"
        session_scores_by_ridge = {}
        selected_ridge = None
        best_score = None
        tied_ridges = []
        epoch_winner_by_epoch = {}
    else:
        selection_status = "scored"
        session_scores_by_ridge = {}
        epoch_winner_by_epoch = {}
        for ridge_strength in candidate_ridges:
            ridge_epoch_scores = [
                float(row["epoch_median_devexp"])
                for row in epoch_rows_by_ridge[ridge_strength]
                if row["in_common_valid_epoch_set"]
            ]
            session_scores_by_ridge[ridge_strength] = float(np.mean(ridge_epoch_scores))

        selected_ridge, best_score, tied_ridges = _choose_best_ridge(
            session_scores_by_ridge,
            tie_tol=tie_tol,
        )

        for epoch in common_valid_epochs:
            epoch_score_by_ridge = {
                float(row["ridge_strength"]): float(row["epoch_median_devexp"])
                for row in epoch_rows
                if row["epoch"] == epoch and row["in_common_valid_epoch_set"]
            }
            epoch_winner, _, _ = _choose_best_ridge(epoch_score_by_ridge, tie_tol=tie_tol)
            if epoch_winner is not None:
                epoch_winner_by_epoch[epoch] = float(epoch_winner)

        for row in epoch_rows:
            row["epoch_is_winner"] = (
                row["in_common_valid_epoch_set"]
                and epoch_winner_by_epoch.get(str(row["epoch"])) == float(row["ridge_strength"])
            )

    summary_rows: list[dict[str, Any]] = []
    for ridge_strength in candidate_ridges:
        available_epochs = sorted(str(epoch) for epoch in epochs_by_ridge[ridge_strength])
        valid_epochs_for_ridge = sorted(
            str(row["epoch"])
            for row in epoch_rows_by_ridge[ridge_strength]
            if row["epoch_score_valid"]
        )
        session_score = session_scores_by_ridge.get(ridge_strength, np.nan)
        is_selected = (
            selection_status == "scored"
            and selected_ridge is not None
            and _float_matches(ridge_strength, selected_ridge)
        )
        tied_with_best = (
            selection_status == "scored"
            and any(_float_matches(ridge_strength, tied_ridge) for tied_ridge in tied_ridges)
        )
        score_delta = (
            np.nan
            if best_score is None or not np.isfinite(session_score)
            else float(best_score - session_score)
        )
        epoch_win_count = int(
            sum(
                bool(row["epoch_is_winner"])
                for row in epoch_rows_by_ridge[ridge_strength]
            )
        )
        summary_rows.append(
            {
                "setup_label": setup_label,
                "ripple_window_s": ripple_window_s,
                "ripple_window_offset_s": ripple_window_offset_s,
                "ripple_window_suffix": ripple_window_suffix,
                "ripple_selection_mode": ripple_selection_mode,
                "ridge_strength": ridge_strength,
                "ridge_strength_suffix": str(group.loc[
                    _setup_mask_for_ridge(group, ridge_strength),
                    "ridge_strength_suffix",
                ].iloc[0]),
                "selected": bool(is_selected),
                "selection_status": selection_status,
                "is_within_tie_tol_of_best": bool(tied_with_best),
                "session_mean_epoch_median_devexp": float(session_score)
                if np.isfinite(session_score)
                else np.nan,
                "best_session_mean_epoch_median_devexp": float(best_score)
                if best_score is not None
                else np.nan,
                "score_delta_from_best": float(score_delta)
                if np.isfinite(score_delta)
                else np.nan,
                "epoch_win_count": epoch_win_count,
                "epoch_win_fraction": (
                    float(epoch_win_count / len(common_valid_epochs))
                    if common_valid_epochs
                    else np.nan
                ),
                "n_candidate_ridges": int(len(candidate_ridges)),
                "candidate_ridge_strengths_json": _json_list(candidate_ridges),
                "tied_ridge_strengths_json": _json_list(tied_ridges),
                "n_available_epochs": int(len(available_epochs)),
                "available_epochs_json": _json_list(available_epochs),
                "n_common_file_epochs": int(len(common_file_epochs)),
                "common_file_epochs_json": _json_list(common_file_epochs),
                "n_valid_epochs_for_ridge": int(len(valid_epochs_for_ridge)),
                "valid_epochs_for_ridge_json": _json_list(valid_epochs_for_ridge),
                "n_common_valid_epochs": int(len(common_valid_epochs)),
                "common_valid_epochs_json": _json_list(common_valid_epochs),
                "tie_tol": float(tie_tol),
                "min_common_epochs": int(min_common_epochs),
                "min_finite_units": int(min_finite_units),
            }
        )

    return summary_rows, epoch_rows


def _setup_mask_for_ridge(group: pd.DataFrame, ridge_strength: float) -> pd.Series:
    """Return the boolean mask for one ridge value inside one setup group."""
    return np.isclose(
        group["ridge_strength"].to_numpy(dtype=float),
        float(ridge_strength),
        rtol=1e-12,
        atol=1e-12,
    )


def build_output_stem(
    *,
    ripple_window_s: float | None,
    ripple_window_offset_s: float | None,
    ripple_selection_mode: str | None,
) -> str:
    """Return the output stem for one ridge-selection run."""
    stem_parts = ["ripple_glm_ridge_selection"]
    if ripple_window_s is not None:
        stem_parts.append(
            format_ripple_window_suffix(
                ripple_window_s,
                ripple_window_offset_s=(
                    DEFAULT_RIPPLE_WINDOW_OFFSET_S
                    if ripple_window_offset_s is None
                    else ripple_window_offset_s
                ),
            )
        )
    elif ripple_window_offset_s is not None:
        stem_parts.append(format_ripple_window_offset_suffix(ripple_window_offset_s))
    if ripple_selection_mode is not None:
        stem_parts.append(str(ripple_selection_mode))
    return "_".join(stem_parts)


def save_parquet_table(table: pd.DataFrame, output_path: Path) -> Path:
    """Write one DataFrame to parquet with a descriptive dependency error."""
    try:
        table.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Saving parquet outputs requires `pyarrow` or `fastparquet` to be installed."
        ) from exc
    return output_path


def print_setup_selection_summary(summary_table: pd.DataFrame) -> None:
    """Print one compact selection line per setup."""
    for setup_label, group in summary_table.groupby("setup_label", sort=True):
        selected_rows = group.loc[group["selected"]]
        if not selected_rows.empty:
            row = selected_rows.iloc[0]
            tied_ridges = json.loads(str(row["tied_ridge_strengths_json"]))
            print(
                f"{setup_label}: selected ridge={float(row['ridge_strength']):.1e}; "
                f"score={float(row['session_mean_epoch_median_devexp']):.4f}; "
                f"comparison_epochs={int(row['n_common_valid_epochs'])}; "
                f"ties={len(tied_ridges)}"
            )
            continue

        row = group.iloc[0]
        print(
            f"{setup_label}: no ridge selected "
            f"({row['selection_status']}; common_file_epochs={int(row['n_common_file_epochs'])}; "
            f"common_valid_epochs={int(row['n_common_valid_epochs'])})"
        )


def main(argv: list[str] | None = None) -> None:
    """Run ridge selection over one session's saved ripple GLM outputs."""
    args = parse_arguments(argv)
    validate_arguments(args)

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = analysis_path / "ripple_glm"
    output_stem = build_output_stem(
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection_mode=args.ripple_selection_mode,
    )
    summary_path = data_dir / f"{output_stem}_summary.parquet"
    epoch_scores_path = data_dir / f"{output_stem}_epoch_scores.parquet"

    print(f"Scanning saved ripple GLM outputs under {data_dir}...")
    records = collect_saved_output_records(
        data_dir,
        ripple_window_s=args.ripple_window_s,
        ripple_window_offset_s=args.ripple_window_offset_s,
        ripple_selection_mode=args.ripple_selection_mode,
        ridge_strengths=None if args.ridge_strengths is None else list(args.ridge_strengths),
    )

    summary_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for _, group in records.groupby(
        ["ripple_window_s", "ripple_window_offset_s", "ripple_selection_mode"],
        sort=True,
    ):
        group_summary_rows, group_epoch_rows = summarize_setup_group(
            group,
            tie_tol=args.tie_tol,
            min_common_epochs=args.min_common_epochs,
            min_finite_units=args.min_finite_units,
        )
        summary_rows.extend(group_summary_rows)
        epoch_rows.extend(group_epoch_rows)

    if not summary_rows:
        raise RuntimeError("No comparable ripple GLM setup summaries were produced.")

    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table.loc[
        :,
        [column for column in SUMMARY_COLUMNS if column in summary_table.columns],
    ].sort_values(
        ["ripple_window_s", "ripple_window_offset_s", "ripple_selection_mode", "ridge_strength"],
        kind="stable",
    ).reset_index(drop=True)

    epoch_score_table = pd.DataFrame(epoch_rows, columns=EPOCH_SCORE_COLUMNS)
    if not epoch_score_table.empty:
        epoch_score_table = epoch_score_table.sort_values(
            [
                "ripple_window_s",
                "ripple_window_offset_s",
                "ripple_selection_mode",
                "epoch",
                "ridge_strength",
            ],
            kind="stable",
        ).reset_index(drop=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    save_parquet_table(summary_table, summary_path)
    save_parquet_table(epoch_score_table, epoch_scores_path)

    print_setup_selection_summary(summary_table)
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.ripple.ripple_glm_select_ridge",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "ripple_window_s": args.ripple_window_s,
            "ripple_window_offset_s": args.ripple_window_offset_s,
            "ripple_selection_mode": args.ripple_selection_mode,
            "ridge_strengths": args.ridge_strengths,
            "tie_tol": args.tie_tol,
            "min_common_epochs": args.min_common_epochs,
            "min_finite_units": args.min_finite_units,
        },
        outputs={
            "n_saved_outputs_scanned": int(len(records)),
            "summary_path": summary_path,
            "epoch_scores_path": epoch_scores_path,
            "n_setup_rows": int(len(summary_table)),
            "n_epoch_rows": int(len(epoch_score_table)),
        },
    )
    print(f"Saved ridge-selection summary to {summary_path}")
    print(f"Saved ridge-selection epoch scores to {epoch_scores_path}")
    print(f"Wrote run log to {log_path}")


if __name__ == "__main__":
    main()
