"""Database-free ripple-restricted CA1-to-V1 cross-correlation artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "ripple_cross_region_xcorr"
MANIFEST_FILENAME = "manifest.parquet"
CA1_UNITS_FILENAME = "ca1_units.parquet"
V1_UNITS_FILENAME = "v1_units.parquet"
SUMMARY_FILENAME = "summary.parquet"
RESULT_FILENAME = "ripple_cross_region_xcorr.nc"
BUNDLE_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"

SOURCE_REGION = "ca1"
TARGET_REGION = "v1"
DEFAULT_BIN_SIZE_S = 0.005
DEFAULT_MAX_LAG_S = 0.5
DEFAULT_MIN_RIPPLE_SPIKES = 30
DEFAULT_EXTREMUM_HALF_WIDTH_BINS = 1
DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD = 2.0
DEFAULT_REQUIRE_SPEED_GATED = True

OUTPUT_RULE = {
    "version": 1,
    "direction": "ca1_reference_to_v1_target",
    "interval_scope": "exact_selected_detected_ripple_intervals_for_one_epoch",
    "epoch_pooling": False,
    "fixed_ripple_windows": False,
    "state_intervals": False,
    "normalization": "pynapple_target_rate_norm_true",
    "bin_size_s": DEFAULT_BIN_SIZE_S,
    "max_lag_s": DEFAULT_MAX_LAG_S,
    "minimum_ripple_spikes_per_unit": DEFAULT_MIN_RIPPLE_SPIKES,
    "extremum_half_width_bins": DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
    "detector_event_policy": "zscore_threshold_2_and_speed_gated",
    "unit_audit_policy": "retain_all_ca1_and_v1_input_units",
    "unit_identity_policy": "stable_sorting_identity_with_runtime_group_key_audit",
    "terminal_artifact_policy": "explicit_empty_and_partial_statuses",
    "legacy_registration_policy": (
        "imported_spike_sorting_identity_resolution_and_exact_nwb_recomputation"
    ),
    "legacy_comparison_policy": "all_four_scientific_artifacts_tight_equal",
    "time_unit": "s",
    "time_reference": "augmented_nwb_ephys_timestamps",
}

IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
UNIT_AUDIT_COLUMNS = (
    "region",
    *IDENTITY_COLUMNS,
    "input_unit_index",
    "ripple_spike_count",
    "minimum_ripple_spikes",
    "passes_ripple_spike_threshold",
    "included_in_xcorr",
    "n_valid_pairs",
    "unit_qc_status",
)
UNIT_QC_STATUSES = (
    "excluded_spike_threshold",
    "not_computed",
    "valid",
    "no_valid_pairs",
)
PAIR_STATUS_VALID = "valid"
PAIR_STATUS_NO_FINITE_BINS = "no_finite_bins"
PAIR_STATUSES = (PAIR_STATUS_VALID, PAIR_STATUS_NO_FINITE_BINS)
PAIR_SUMMARY_COLUMNS = (
    "ripple_cross_region_xcorr_id",
    "animal_name",
    "date",
    "epoch",
    "ca1_spikesorting_merge_id",
    "ca1_unit_id",
    "ca1_stable_unit_id",
    "ca1_group_unit_id",
    "v1_spikesorting_merge_id",
    "v1_unit_id",
    "v1_stable_unit_id",
    "v1_group_unit_id",
    "n_ca1_ripple_spikes",
    "n_v1_ripple_spikes",
    "peak_lag_s",
    "peak_norm_xcorr",
    "status",
)
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_valid_pairs",
    "no_ripples",
    "no_ca1_units",
    "no_v1_units",
    "no_eligible_ca1_units",
    "no_eligible_v1_units",
)
NONTERMINAL_STATUSES = ("valid", "partial_valid", "no_valid_pairs")
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "ripple_cross_region_xcorr_id",
    "animal_name",
    "date",
    "epoch",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "upstream_provenance_json",
    "selected_ripple_intervals_sha256",
    "n_ripples",
    "ripple_duration_s",
    "n_ca1_units",
    "n_v1_units",
    "n_ca1_units_in_xcorr",
    "n_v1_units_in_xcorr",
    "n_pairs",
    "n_valid_pairs",
    "ca1_units_sha256",
    "v1_units_sha256",
    "summary_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "bundle_schema_version",
)


def _screen_module() -> Any:
    """Import the existing scientific implementation only when required."""
    from v1ca1.xcorr import screen_xcorr

    return screen_xcorr


def _provenance_sha256(value: Any) -> str:
    """Return the shared deterministic provenance digest."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(value)


OUTPUT_RULE_SHA256 = _provenance_sha256(OUTPUT_RULE)


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe, non-empty path component."""
    component = str(value)
    if not component or Path(component).name != component or component in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return component


def _uuid_string(value: Any, *, name: str) -> str:
    """Return one canonical UUID string."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_sha256(table: pd.DataFrame) -> str:
    """Return a deterministic digest for one canonical table."""
    hashed = pd.util.hash_pandas_object(table, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _database_bool(value: Any, *, name: str) -> bool:
    """Normalize one bool or database integer 0/1 without accepting truthy junk."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(int(value))
    raise TypeError(f"{name} must be a bool or database integer 0/1.")


def _boolean_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    """Return one strictly validated boolean array."""
    return np.asarray(
        [_database_bool(value, name=name) for value in values], dtype=bool
    )


def _canonical_json_mapping(
    value: Mapping[str, Any], *, name: str
) -> tuple[dict[str, Any], str]:
    """Return a JSON-roundtripped provenance mapping and canonical JSON."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping.")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable.") from exc
    return normalized, encoded


def get_ripple_cross_region_xcorr_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_cross_region_xcorr_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first cross-region xcorr bundle."""
    animal_name = _path_component(animal_name, name="animal_name")
    date = _path_component(date, name="date")
    epoch = _path_component(epoch, name="epoch")
    result_id = _uuid_string(ripple_cross_region_xcorr_id, name="ripple_cross_region_xcorr_id")
    artifact_dir = (
        Path(artifact_root)
        / animal_name
        / date
        / ARTIFACT_DIRNAME
        / epoch
        / result_id
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "ca1_units_path": artifact_dir / CA1_UNITS_FILENAME,
        "v1_units_path": artifact_dir / V1_UNITS_FILENAME,
        "summary_path": artifact_dir / SUMMARY_FILENAME,
        "result_path": artifact_dir / RESULT_FILENAME,
    }


def get_legacy_ripple_cross_region_xcorr_paths(
    analysis_path: Path, *, epoch: str
) -> dict[str, Path]:
    """Return the four canonical legacy exact-ripple xcorr artifact paths."""
    epoch = _path_component(epoch, name="epoch")
    artifact_dir = (
        Path(analysis_path) / "xcorr" / "screen_pairs" / "ripple" / epoch
    )
    return {
        "artifact_dir": artifact_dir,
        "ca1_unit_filter_path": artifact_dir / "ca1_unit_filter.parquet",
        "v1_unit_filter_path": artifact_dir / "v1_unit_filter.parquet",
        "summary_path": artifact_dir / "xcorr_summary.parquet",
        "result_path": artifact_dir / "xcorr.nc",
    }


def validate_ripple_cross_region_xcorr_parameters(
    *,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_ripple_spikes: int = DEFAULT_MIN_RIPPLE_SPIKES,
    extremum_half_width_bins: int = DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
    norm: bool = True,
    expected_detector_zscore_threshold: float = (
        DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD
    ),
    require_speed_gated: bool = DEFAULT_REQUIRE_SPEED_GATED,
) -> dict[str, Any]:
    """Return the fixed, validated ripple xcorr scientific parameters."""
    bin_size = float(bin_size_s)
    max_lag = float(max_lag_s)
    detector_threshold = float(expected_detector_zscore_threshold)
    fixed_floats = {
        "bin_size_s": (bin_size, DEFAULT_BIN_SIZE_S),
        "max_lag_s": (max_lag, DEFAULT_MAX_LAG_S),
        "expected_detector_zscore_threshold": (
            detector_threshold,
            DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD,
        ),
    }
    for name, (observed, expected) in fixed_floats.items():
        if not np.isfinite(observed) or not np.isclose(
            observed, expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{name} must equal the fixed value {expected!r}.")
    for name, raw, expected in (
        ("min_ripple_spikes", min_ripple_spikes, DEFAULT_MIN_RIPPLE_SPIKES),
        (
            "extremum_half_width_bins",
            extremum_half_width_bins,
            DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
        ),
    ):
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
        if int(raw) != expected:
            raise ValueError(f"{name} must equal the fixed value {expected!r}.")
    if not _database_bool(norm, name="norm"):
        raise ValueError("RippleCrossRegionXCorr requires pynapple norm=True.")
    if not _database_bool(require_speed_gated, name="require_speed_gated"):
        raise ValueError("RippleCrossRegionXCorr requires speed-gated ripple events.")
    return {
        "bin_size_s": bin_size,
        "max_lag_s": max_lag,
        "min_ripple_spikes": int(min_ripple_spikes),
        "extremum_half_width_bins": int(extremum_half_width_bins),
        "norm": True,
        "expected_detector_zscore_threshold": detector_threshold,
        "require_speed_gated": True,
    }


def _effective_parameters(
    *,
    parameter_name: str,
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
    **values: Any,
) -> dict[str, Any]:
    """Validate parameters and immutable parameter/output-rule hashes."""
    parameters = validate_ripple_cross_region_xcorr_parameters(**values)
    name = _path_component(parameter_name, name="parameter_name")
    expected_parameter_hash = _provenance_sha256(
        {"ripple_cross_region_xcorr_param_name": name, **parameters}
    )
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_hash
    if str(parameter_sha256) != expected_parameter_hash:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    if output_rule_sha256 is None:
        output_rule_sha256 = OUTPUT_RULE_SHA256
    if str(output_rule_sha256) != OUTPUT_RULE_SHA256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": name,
        "parameter_sha256": str(parameter_sha256),
        "output_rule_sha256": str(output_rule_sha256),
        **parameters,
    }


def _metadata(
    *, ripple_cross_region_xcorr_id: Any, animal_name: str, date: str, epoch: str
) -> dict[str, str]:
    """Return validated immutable selection metadata."""
    return {
        "ripple_cross_region_xcorr_id": _uuid_string(
            ripple_cross_region_xcorr_id, name="ripple_cross_region_xcorr_id"
        ),
        "animal_name": _path_component(animal_name, name="animal_name"),
        "date": _path_component(date, name="date"),
        "epoch": _path_component(epoch, name="epoch"),
    }


def _validate_upstream_provenance(
    upstream_provenance: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Require the selected RippleIntervals row to carry fixed detector provenance."""
    if not isinstance(upstream_provenance, Mapping):
        raise TypeError("upstream_provenance must be a mapping.")
    raw = dict(upstream_provenance)
    missing = [
        name
        for name in ("detector_zscore_threshold", "speed_gated")
        if name not in raw
    ]
    if missing:
        raise ValueError(
            "upstream_provenance must contain detector_zscore_threshold and "
            "speed_gated from the selected RippleIntervals row."
        )
    detector_value = raw["detector_zscore_threshold"]
    if isinstance(detector_value, bool) or not isinstance(detector_value, Real):
        raise TypeError(
            "upstream_provenance detector_zscore_threshold must be numeric."
        )
    detector_threshold = float(detector_value)
    speed_gated = _database_bool(
        raw["speed_gated"], name="upstream_provenance speed_gated"
    )
    normalized, encoded = _canonical_json_mapping(
        {
            **raw,
            "detector_zscore_threshold": detector_threshold,
            "speed_gated": speed_gated,
        },
        name="upstream_provenance",
    )
    if not np.isfinite(detector_threshold) or not np.isclose(
        detector_threshold,
        float(parameters["expected_detector_zscore_threshold"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Selected RippleIntervals detector threshold does not equal 2.0."
        )
    if not speed_gated:
        raise ValueError(
            "Selected RippleIntervals provenance must have speed_gated=True."
        )
    return normalized, encoded


def _normalize_ripple_table(ripple_table: Any, *, epoch: str) -> pd.DataFrame:
    """Return exact finite detected-ripple bounds for one epoch without pooling."""
    as_dataframe = getattr(ripple_table, "as_dataframe", None)
    if callable(as_dataframe):
        table = as_dataframe().copy()
    elif isinstance(ripple_table, pd.DataFrame):
        table = ripple_table.copy()
    else:
        table = pd.DataFrame(ripple_table)
    rename = {}
    if "start" in table and "start_time" not in table:
        rename["start"] = "start_time"
    if "stop_time" in table and "end_time" not in table:
        rename["stop_time"] = "end_time"
    if "end" in table and "end_time" not in table:
        rename["end"] = "end_time"
    table = table.rename(columns=rename)
    if "epoch" in table:
        table = table.loc[table["epoch"].astype(str) == str(epoch)]
    missing = [name for name in ("start_time", "end_time") if name not in table]
    if missing:
        raise ValueError(f"ripple_table is missing required columns {missing!r}.")
    table = table.loc[:, ["start_time", "end_time"]].copy().reset_index(drop=True)
    for name in ("start_time", "end_time"):
        table[name] = pd.to_numeric(table[name], errors="raise")
    bounds = table.to_numpy(dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Ripple interval bounds must be finite seconds values.")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Every ripple interval must have start_time < end_time.")
    table = table.sort_values(["start_time", "end_time"], kind="stable").reset_index(
        drop=True
    )
    starts = table["start_time"].to_numpy(dtype=float)
    ends = table["end_time"].to_numpy(dtype=float)
    if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError(
            "Selected detected ripple intervals must not overlap; overlapping "
            "events cannot be preserved exactly by pynapple IntervalSet."
        )
    return table


def _ripple_intervals_sha256(table: pd.DataFrame) -> str:
    """Return the exact ordered detected-ripple interval digest."""
    return _provenance_sha256(
        {
            "start_time_s": table["start_time"].to_numpy(dtype=float).tolist(),
            "end_time_s": table["end_time"].to_numpy(dtype=float).tolist(),
        }
    )


def prepare_ripple_cross_region_xcorr_event_selection(
    *, epoch: str, ripple_table: Any
) -> dict[str, Any]:
    """Return canonical exact ripple intervals and their immutable selection hash."""
    selected = _normalize_ripple_table(
        ripple_table, epoch=_path_component(epoch, name="epoch")
    )
    starts = selected["start_time"].to_numpy(dtype=float)
    ends = selected["end_time"].to_numpy(dtype=float)
    return {
        "selected_ripple_table": selected,
        "ripple_start_time_s": starts,
        "ripple_end_time_s": ends,
        "n_ripples": int(len(selected)),
        "ripple_duration_s": float(np.sum(ends - starts)),
        "selected_ripple_intervals_sha256": _ripple_intervals_sha256(selected),
    }


def _require_expected_ripple_hash(
    observed: str, expected: str | None
) -> None:
    """Reject event drift after a wrapper freezes one exact-ripple selection."""
    if expected is not None and str(expected) != observed:
        raise ValueError(
            "Selected ripple intervals changed after selection; the expected "
            "SHA-256 digest does not match."
        )


def _build_exact_ripple_intervalset(table: pd.DataFrame, *, epoch: str) -> Any:
    """Build one IntervalSet and reject any implicit boundary transformation."""
    screen = _screen_module()
    interval_rows = table.rename(
        columns={"start_time": "start", "end_time": "end"}
    )
    intervals = screen.build_state_intervalset({str(epoch): interval_rows}, [str(epoch)])
    starts, ends = screen.get_interval_bounds(intervals)
    expected_starts = table["start_time"].to_numpy(dtype=float)
    expected_ends = table["end_time"].to_numpy(dtype=float)
    if not np.array_equal(starts, expected_starts) or not np.array_equal(
        ends, expected_ends
    ):
        raise ValueError(
            "pynapple changed the selected ripple boundaries; exact detected "
            "ripple intervals are required."
        )
    return intervals


def _identity_table(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    *,
    region: str,
) -> pd.DataFrame:
    """Return persistent identities aligned to one runtime TsGroup."""
    from v1ca1.spyglass.movement import _stable_identity_rows

    rows = _stable_identity_rows(spikes, stable_unit_ids)
    table = pd.DataFrame.from_records(rows, columns=IDENTITY_COLUMNS)
    if table.empty:
        table = pd.DataFrame(
            {column: pd.Series(dtype=object) for column in IDENTITY_COLUMNS}
        )
    for name in IDENTITY_COLUMNS:
        table[name] = table[name].map(str)
    table.insert(0, "region", region)
    table["input_unit_index"] = np.arange(len(table), dtype=int)
    return table


def _build_unit_audit(
    *,
    identity: pd.DataFrame,
    spikes: Any,
    intervals: Any,
    region: str,
    min_ripple_spikes: int,
) -> pd.DataFrame:
    """Return one all-input unit audit using the existing spike-count helper."""
    screen = _screen_module()
    counts = screen.build_unit_spike_count_table(
        spikes,
        intervals,
        region=region,
        min_state_spikes=min_ripple_spikes,
    )
    if len(counts) != len(identity):
        raise ValueError("Spike-count rows do not align to stable unit identities.")
    if counts["unit_id"].map(str).tolist() != identity["group_unit_id"].tolist():
        raise ValueError("Spike-count runtime unit order differs from stable identities.")
    output = identity.copy()
    output["ripple_spike_count"] = counts["state_spike_count"].to_numpy(dtype=int)
    output["minimum_ripple_spikes"] = int(min_ripple_spikes)
    output["passes_ripple_spike_threshold"] = counts[
        "passes_state_spike_count"
    ].to_numpy(dtype=bool)
    output["included_in_xcorr"] = False
    output["n_valid_pairs"] = 0
    output["unit_qc_status"] = np.where(
        output["passes_ripple_spike_threshold"],
        "not_computed",
        "excluded_spike_threshold",
    )
    return output.loc[:, list(UNIT_AUDIT_COLUMNS)]


def _expected_lag_times(parameters: Mapping[str, Any]) -> np.ndarray:
    """Return the canonical pynapple lag centers for the fixed settings."""
    ratio = float(parameters["max_lag_s"]) / float(parameters["bin_size_s"])
    rounded = int(round(ratio))
    if not np.isclose(ratio, rounded, rtol=0.0, atol=1e-10):
        raise ValueError("max_lag_s must be an integer multiple of bin_size_s.")
    return np.arange(-rounded + 1, rounded, dtype=float) * float(
        parameters["bin_size_s"]
    )


def _terminal_status(
    ca1_units: pd.DataFrame,
    v1_units: pd.DataFrame,
    *,
    n_ripples: int,
) -> str | None:
    """Return the fixed terminal status implied by events and unit audits."""
    if n_ripples == 0:
        return "no_ripples"
    if ca1_units.empty:
        return "no_ca1_units"
    if v1_units.empty:
        return "no_v1_units"
    if not ca1_units["passes_ripple_spike_threshold"].any():
        return "no_eligible_ca1_units"
    if not v1_units["passes_ripple_spike_threshold"].any():
        return "no_eligible_v1_units"
    return None


def _identity_lookup(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return unique runtime-key-to-persistent-identity rows."""
    output: dict[str, dict[str, Any]] = {}
    for row in table.to_dict("records"):
        key = str(row["group_unit_id"])
        if key in output:
            raise ValueError("String-normalized runtime unit ids must be unique.")
        output[key] = row
    return output


def _canonical_pair_summary(
    legacy_summary: pd.DataFrame,
    *,
    metadata: Mapping[str, str],
    ca1_units: pd.DataFrame,
    v1_units: pd.DataFrame,
) -> pd.DataFrame:
    """Replace runtime pair identifiers with persistent identities."""
    ca1_lookup = _identity_lookup(ca1_units)
    v1_lookup = _identity_lookup(v1_units)
    rows: list[dict[str, Any]] = []
    for source in legacy_summary.to_dict("records"):
        try:
            ca1 = ca1_lookup[str(source["ca1_unit_id"])]
            v1 = v1_lookup[str(source["v1_unit_id"])]
        except KeyError as exc:
            raise ValueError("Pair summary contains an unknown runtime unit id.") from exc
        rows.append(
            {
                **metadata,
                "ca1_spikesorting_merge_id": ca1["spikesorting_merge_id"],
                "ca1_unit_id": ca1["unit_id"],
                "ca1_stable_unit_id": ca1["stable_unit_id"],
                "ca1_group_unit_id": ca1["group_unit_id"],
                "v1_spikesorting_merge_id": v1["spikesorting_merge_id"],
                "v1_unit_id": v1["unit_id"],
                "v1_stable_unit_id": v1["stable_unit_id"],
                "v1_group_unit_id": v1["group_unit_id"],
                "n_ca1_ripple_spikes": int(source["n_ca1_state_spikes"]),
                "n_v1_ripple_spikes": int(source["n_v1_state_spikes"]),
                "peak_lag_s": float(source["peak_lag_s"]),
                "peak_norm_xcorr": float(source["peak_norm_xcorr"]),
                "status": str(source["status"]),
            }
        )
    return pd.DataFrame.from_records(rows, columns=PAIR_SUMMARY_COLUMNS)


def _annotate_unit_qc(
    audit: pd.DataFrame,
    *,
    pair_summary: pd.DataFrame,
    region: str,
    computed: bool,
) -> pd.DataFrame:
    """Annotate included units with pair-level validity without dropping units."""
    output = audit.copy()
    if not computed:
        return output
    eligible = output["passes_ripple_spike_threshold"].to_numpy(dtype=bool)
    output.loc[eligible, "included_in_xcorr"] = True
    stable_column = f"{region}_stable_unit_id"
    valid_rows = pair_summary.loc[pair_summary["status"] == PAIR_STATUS_VALID]
    counts = valid_rows.groupby(stable_column, sort=False).size().to_dict()
    valid_counts = output["stable_unit_id"].map(counts).fillna(0).astype(int)
    output.loc[eligible, "n_valid_pairs"] = valid_counts.loc[eligible]
    output.loc[eligible, "unit_qc_status"] = np.where(
        valid_counts.loc[eligible].to_numpy(dtype=int) > 0,
        "valid",
        "no_valid_pairs",
    )
    return output.loc[:, list(UNIT_AUDIT_COLUMNS)]


def _dataset_attrs(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    selected_ripple_intervals_sha256: str,
    analysis_status: str,
    artifact_origin: str,
    legacy_artifact_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return canonical scalar attributes for the xcorr NetCDF result."""
    effective = {
        name: value
        for name, value in parameters.items()
        if name not in {"parameter_name", "parameter_sha256", "output_rule_sha256"}
    }
    return {
        "ripple_cross_region_xcorr_result_schema_version": RESULT_SCHEMA_VERSION,
        **metadata,
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
        "interval_scope": "exact_selected_detected_ripple_intervals",
        "normalization": "pynapple_target_rate_norm_true",
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "effective_parameters_json": json.dumps(
            effective, sort_keys=True, separators=(",", ":")
        ),
        "upstream_provenance_json": upstream_provenance_json,
        "selected_ripple_intervals_sha256": selected_ripple_intervals_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance_json": json.dumps(
            dict(legacy_artifact_provenance or {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }


def _make_dataset(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    ripple_table: pd.DataFrame,
    ca1_units: pd.DataFrame,
    v1_units: pd.DataFrame,
    lag_times: np.ndarray,
    xcorr: np.ndarray,
    analysis_status: str,
    artifact_origin: str = "computed",
    legacy_artifact_provenance: Mapping[str, Any] | None = None,
) -> Any:
    """Build one identity-stable NetCDF-backed xcorr tensor."""
    import xarray as xr

    selected_hash = _ripple_intervals_sha256(ripple_table)
    ca1_selected = ca1_units.loc[ca1_units["included_in_xcorr"]].reset_index(drop=True)
    v1_selected = v1_units.loc[v1_units["included_in_xcorr"]].reset_index(drop=True)
    expected_shape = (len(ca1_selected), len(v1_selected), len(lag_times))
    values = np.asarray(xcorr, dtype=float)
    if values.shape != expected_shape:
        raise ValueError(
            f"xcorr has shape {values.shape!r}; expected {expected_shape!r}."
        )
    return xr.Dataset(
        data_vars={
            "xcorr": (("ca1_unit", "v1_unit", "lag_s"), values),
            "ripple_start_time_s": (
                ("ripple",),
                ripple_table["start_time"].to_numpy(dtype=float),
            ),
            "ripple_end_time_s": (
                ("ripple",),
                ripple_table["end_time"].to_numpy(dtype=float),
            ),
        },
        coords={
            "ca1_unit": ca1_selected["stable_unit_id"].to_numpy(dtype=str),
            "v1_unit": v1_selected["stable_unit_id"].to_numpy(dtype=str),
            "lag_s": np.asarray(lag_times, dtype=float),
            "ca1_spikesorting_merge_id": (
                ("ca1_unit",),
                ca1_selected["spikesorting_merge_id"].to_numpy(dtype=str),
            ),
            "ca1_source_unit_id": (
                ("ca1_unit",),
                ca1_selected["unit_id"].to_numpy(dtype=str),
            ),
            "ca1_group_unit_id": (
                ("ca1_unit",),
                ca1_selected["group_unit_id"].to_numpy(dtype=str),
            ),
            "v1_spikesorting_merge_id": (
                ("v1_unit",),
                v1_selected["spikesorting_merge_id"].to_numpy(dtype=str),
            ),
            "v1_source_unit_id": (
                ("v1_unit",),
                v1_selected["unit_id"].to_numpy(dtype=str),
            ),
            "v1_group_unit_id": (
                ("v1_unit",),
                v1_selected["group_unit_id"].to_numpy(dtype=str),
            ),
        },
        attrs=_dataset_attrs(
            metadata=metadata,
            parameters=parameters,
            upstream_provenance_json=upstream_provenance_json,
            selected_ripple_intervals_sha256=selected_hash,
            analysis_status=analysis_status,
            artifact_origin=artifact_origin,
            legacy_artifact_provenance=legacy_artifact_provenance,
        ),
    )


def compute_ripple_cross_region_xcorr(
    *,
    ripple_cross_region_xcorr_id: Any,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_table: Any,
    ca1_spikes: Any,
    ca1_stable_unit_ids: Sequence[Mapping[str, Any]],
    v1_spikes: Any,
    v1_stable_unit_ids: Sequence[Mapping[str, Any]],
    upstream_provenance: Mapping[str, Any],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    expected_selected_ripple_intervals_sha256: str | None = None,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_ripple_spikes: int = DEFAULT_MIN_RIPPLE_SPIKES,
    extremum_half_width_bins: int = DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
    norm: bool = True,
    expected_detector_zscore_threshold: float = (
        DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD
    ),
    require_speed_gated: bool = DEFAULT_REQUIRE_SPEED_GATED,
) -> dict[str, Any]:
    """Compute CA1-reference/V1-target xcorr inside exact detected ripples."""
    metadata = _metadata(
        ripple_cross_region_xcorr_id=ripple_cross_region_xcorr_id,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    parameters = _effective_parameters(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        bin_size_s=bin_size_s,
        max_lag_s=max_lag_s,
        min_ripple_spikes=min_ripple_spikes,
        extremum_half_width_bins=extremum_half_width_bins,
        norm=norm,
        expected_detector_zscore_threshold=expected_detector_zscore_threshold,
        require_speed_gated=require_speed_gated,
    )
    provenance, provenance_json = _validate_upstream_provenance(
        upstream_provenance, parameters=parameters
    )
    event_selection = prepare_ripple_cross_region_xcorr_event_selection(
        epoch=metadata["epoch"], ripple_table=ripple_table
    )
    ripples = event_selection["selected_ripple_table"]
    selected_hash = event_selection["selected_ripple_intervals_sha256"]
    _require_expected_ripple_hash(
        selected_hash, expected_selected_ripple_intervals_sha256
    )
    intervals = _build_exact_ripple_intervalset(ripples, epoch=metadata["epoch"])
    ca1_identity = _identity_table(
        ca1_spikes, ca1_stable_unit_ids, region=SOURCE_REGION
    )
    v1_identity = _identity_table(v1_spikes, v1_stable_unit_ids, region=TARGET_REGION)
    ca1_units = _build_unit_audit(
        identity=ca1_identity,
        spikes=ca1_spikes,
        intervals=intervals,
        region=SOURCE_REGION,
        min_ripple_spikes=parameters["min_ripple_spikes"],
    )
    v1_units = _build_unit_audit(
        identity=v1_identity,
        spikes=v1_spikes,
        intervals=intervals,
        region=TARGET_REGION,
        min_ripple_spikes=parameters["min_ripple_spikes"],
    )
    terminal = _terminal_status(ca1_units, v1_units, n_ripples=len(ripples))
    lag_times = _expected_lag_times(parameters)
    pair_summary = pd.DataFrame(columns=PAIR_SUMMARY_COLUMNS)
    if terminal is None:
        screen = _screen_module()
        ca1_runtime_ids = np.asarray(
            list(ca1_spikes.keys()), dtype=object
        )[ca1_units["passes_ripple_spike_threshold"].to_numpy(dtype=bool)]
        v1_runtime_ids = np.asarray(
            list(v1_spikes.keys()), dtype=object
        )[v1_units["passes_ripple_spike_threshold"].to_numpy(dtype=bool)]
        filtered_ca1 = screen.subset_spikes_by_unit_ids(ca1_spikes, ca1_runtime_ids)
        filtered_v1 = screen.subset_spikes_by_unit_ids(v1_spikes, v1_runtime_ids)
        frame = screen.compute_xcorr(
            ca1_spikes=filtered_ca1,
            v1_spikes=filtered_v1,
            intervals=intervals,
            bin_size_s=parameters["bin_size_s"],
            max_lag_s=parameters["max_lag_s"],
        )
        observed_lags, xcorr = screen.xcorr_frame_to_tensor(
            frame, ca1_runtime_ids, v1_runtime_ids
        )
        if not np.allclose(
            observed_lags, lag_times, rtol=0.0, atol=1e-12
        ):
            raise ValueError("pynapple returned an unexpected xcorr lag grid.")
        legacy_summary = screen.build_pair_summary_table(
            xcorr=xcorr,
            ca1_unit_ids=ca1_runtime_ids,
            v1_unit_ids=v1_runtime_ids,
            ca1_spike_counts=ca1_units.loc[
                ca1_units["passes_ripple_spike_threshold"], "ripple_spike_count"
            ].to_numpy(dtype=int),
            v1_spike_counts=v1_units.loc[
                v1_units["passes_ripple_spike_threshold"], "ripple_spike_count"
            ].to_numpy(dtype=int),
            lag_times=observed_lags,
            extremum_half_width_bins=parameters["extremum_half_width_bins"],
        )
        pair_summary = _canonical_pair_summary(
            legacy_summary,
            metadata=metadata,
            ca1_units=ca1_units,
            v1_units=v1_units,
        )
        valid_pairs = pair_summary["status"].eq(PAIR_STATUS_VALID)
        if valid_pairs.all():
            analysis_status = "valid"
        elif valid_pairs.any():
            analysis_status = "partial_valid"
        else:
            analysis_status = "no_valid_pairs"
        ca1_units = _annotate_unit_qc(
            ca1_units,
            pair_summary=pair_summary,
            region=SOURCE_REGION,
            computed=True,
        )
        v1_units = _annotate_unit_qc(
            v1_units,
            pair_summary=pair_summary,
            region=TARGET_REGION,
            computed=True,
        )
    else:
        analysis_status = terminal
        xcorr = np.full(
            (
                int(ca1_units["included_in_xcorr"].sum()),
                int(v1_units["included_in_xcorr"].sum()),
                len(lag_times),
            ),
            np.nan,
            dtype=float,
        )
    dataset = _make_dataset(
        metadata=metadata,
        parameters=parameters,
        upstream_provenance_json=provenance_json,
        ripple_table=ripples,
        ca1_units=ca1_units,
        v1_units=v1_units,
        lag_times=lag_times,
        xcorr=xcorr,
        analysis_status=analysis_status,
    )
    return validate_ripple_cross_region_xcorr_result(
        {
            **metadata,
            "parameters": parameters,
            "upstream_provenance": provenance,
            "selected_ripple_intervals_sha256": selected_hash,
            "ca1_units": ca1_units,
            "v1_units": v1_units,
            "summary": pair_summary,
            "dataset": dataset,
            "analysis_status": analysis_status,
            "artifact_origin": "computed",
            "legacy_artifact_provenance": {},
        }
    )


def _validate_unit_audit(
    table: Any,
    *,
    region: str,
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate one all-input regional unit audit exactly."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != UNIT_AUDIT_COLUMNS:
        raise ValueError(f"{region} unit audit does not match its canonical schema.")
    output = table.copy().reset_index(drop=True)
    if not output["region"].astype(str).eq(region).all():
        raise ValueError(f"{region} unit audit contains another region.")
    for name in IDENTITY_COLUMNS:
        output[name] = output[name].map(str)
        if output[name].eq("").any():
            raise ValueError(f"{region} unit audit has empty {name} values.")
    if output["stable_unit_id"].duplicated().any() or output[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError(f"{region} unit identities must be unique.")
    expected_stable = (
        output["spikesorting_merge_id"] + ":" + output["unit_id"]
    )
    if not output["stable_unit_id"].equals(expected_stable):
        raise ValueError(f"{region} stable unit ids are not canonical.")
    expected_indices = np.arange(len(output), dtype=int)
    if not np.array_equal(
        pd.to_numeric(output["input_unit_index"], errors="raise").to_numpy(),
        expected_indices,
    ):
        raise ValueError(f"{region} input unit indices are not contiguous.")
    counts = pd.to_numeric(output["ripple_spike_count"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0) or not np.allclose(
        counts, np.rint(counts), rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{region} ripple spike counts must be non-negative integers.")
    output["ripple_spike_count"] = counts.astype(int)
    thresholds = pd.to_numeric(
        output["minimum_ripple_spikes"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.all(thresholds == parameters["min_ripple_spikes"]):
        raise ValueError(f"{region} unit thresholds differ from parameters.")
    passes = _boolean_array(
        output["passes_ripple_spike_threshold"].tolist(),
        name=f"{region} passes_ripple_spike_threshold",
    )
    output["passes_ripple_spike_threshold"] = passes
    expected_passes = counts >= parameters["min_ripple_spikes"]
    if not np.array_equal(passes, expected_passes):
        raise ValueError(f"{region} spike-threshold flags differ from counts.")
    included = _boolean_array(
        output["included_in_xcorr"].tolist(),
        name=f"{region} included_in_xcorr",
    )
    output["included_in_xcorr"] = included
    if np.any(included & ~passes):
        raise ValueError(f"{region} excluded units cannot be included in xcorr.")
    valid_counts = pd.to_numeric(output["n_valid_pairs"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(valid_counts)) or np.any(valid_counts < 0.0) or not np.allclose(
        valid_counts, np.rint(valid_counts), rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{region} valid-pair counts must be non-negative integers.")
    output["n_valid_pairs"] = valid_counts.astype(int)
    statuses = output["unit_qc_status"].astype(str)
    if not statuses.isin(UNIT_QC_STATUSES).all():
        raise ValueError(f"{region} unit audit has unsupported QC statuses.")
    expected_status = np.full(len(output), "excluded_spike_threshold", dtype=object)
    expected_status[passes & ~included] = "not_computed"
    expected_status[passes & included & (valid_counts > 0)] = "valid"
    expected_status[passes & included & (valid_counts == 0)] = "no_valid_pairs"
    if not np.array_equal(statuses.to_numpy(dtype=object), expected_status):
        raise ValueError(f"{region} unit QC statuses are inconsistent.")
    if np.any((~included) & (valid_counts != 0)):
        raise ValueError(f"{region} non-included units cannot claim valid pairs.")
    return output.loc[:, list(UNIT_AUDIT_COLUMNS)]


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    name: str,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Raise a concise validation error for mismatched scientific tables."""
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as exc:
        raise ValueError(f"{name} differs from the canonical result.") from exc


def _validate_dataset(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    selected_ripple_intervals_sha256: str,
    ca1_units: pd.DataFrame,
    v1_units: pd.DataFrame,
    analysis_status: str,
    artifact_origin: str,
    legacy_artifact_provenance: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate NetCDF identity, interval, tensor, and pair-summary arithmetic."""
    if dataset is None or not hasattr(dataset, "attrs") or not hasattr(dataset, "sizes"):
        raise TypeError("RippleCrossRegionXCorr dataset must be xarray Dataset-like.")
    expected_attrs = {
        "ripple_cross_region_xcorr_result_schema_version": RESULT_SCHEMA_VERSION,
        **metadata,
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
        "interval_scope": "exact_selected_detected_ripple_intervals",
        "normalization": "pynapple_target_rate_norm_true",
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        "upstream_provenance_json": upstream_provenance_json,
        "selected_ripple_intervals_sha256": selected_ripple_intervals_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
    for name, expected in expected_attrs.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"RippleCrossRegionXCorr dataset has mismatched {name}.")
    effective = {
        name: value
        for name, value in parameters.items()
        if name not in {"parameter_name", "parameter_sha256", "output_rule_sha256"}
    }
    try:
        dataset_parameters = json.loads(
            str(dataset.attrs.get("effective_parameters_json", "{}"))
        )
        dataset_legacy = json.loads(
            str(dataset.attrs.get("legacy_artifact_provenance_json", "{}"))
        )
    except json.JSONDecodeError as exc:
        raise ValueError("RippleCrossRegionXCorr dataset has malformed JSON attrs.") from exc
    if dataset_parameters != effective:
        raise ValueError("RippleCrossRegionXCorr dataset parameters differ from the row.")
    if dataset_legacy != dict(legacy_artifact_provenance):
        raise ValueError("RippleCrossRegionXCorr dataset legacy provenance differs from the row.")
    required_dimensions = ("ripple", "ca1_unit", "v1_unit", "lag_s")
    if any(name not in dataset.dims for name in required_dimensions):
        raise ValueError("RippleCrossRegionXCorr dataset lacks canonical dimensions.")
    for name in ("ripple_start_time_s", "ripple_end_time_s"):
        if name not in dataset or dataset[name].dims != ("ripple",):
            raise ValueError(f"RippleCrossRegionXCorr dataset lacks canonical {name}.")
    starts = np.asarray(dataset["ripple_start_time_s"].values, dtype=float)
    ends = np.asarray(dataset["ripple_end_time_s"].values, dtype=float)
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Persisted ripple intervals must be finite.")
    if np.any(ends <= starts) or (len(starts) > 1 and np.any(starts[1:] < ends[:-1])):
        raise ValueError("Persisted ripple intervals must be positive and non-overlapping.")
    ripple_table = pd.DataFrame({"start_time": starts, "end_time": ends})
    if _ripple_intervals_sha256(ripple_table) != selected_ripple_intervals_sha256:
        raise ValueError("Persisted ripple intervals do not match their digest.")
    lag_times = np.asarray(dataset.coords["lag_s"].values, dtype=float)
    if not np.allclose(
        lag_times,
        _expected_lag_times(parameters),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("RippleCrossRegionXCorr lag coordinates differ from parameters.")
    selected_by_region = {
        SOURCE_REGION: ca1_units.loc[ca1_units["included_in_xcorr"]].reset_index(
            drop=True
        ),
        TARGET_REGION: v1_units.loc[v1_units["included_in_xcorr"]].reset_index(
            drop=True
        ),
    }
    coordinate_columns = {
        SOURCE_REGION: {
            "ca1_unit": "stable_unit_id",
            "ca1_spikesorting_merge_id": "spikesorting_merge_id",
            "ca1_source_unit_id": "unit_id",
            "ca1_group_unit_id": "group_unit_id",
        },
        TARGET_REGION: {
            "v1_unit": "stable_unit_id",
            "v1_spikesorting_merge_id": "spikesorting_merge_id",
            "v1_source_unit_id": "unit_id",
            "v1_group_unit_id": "group_unit_id",
        },
    }
    for region, coordinate_map in coordinate_columns.items():
        selected = selected_by_region[region]
        for coordinate, column in coordinate_map.items():
            if coordinate not in dataset.coords or not np.array_equal(
                np.asarray(dataset.coords[coordinate].values).astype(str),
                selected[column].to_numpy(dtype=str),
            ):
                raise ValueError(
                    f"RippleCrossRegionXCorr {region} coordinate {coordinate} is misaligned."
                )
    if "xcorr" not in dataset or dataset["xcorr"].dims != (
        "ca1_unit",
        "v1_unit",
        "lag_s",
    ):
        raise ValueError("RippleCrossRegionXCorr dataset lacks its canonical tensor.")
    values = np.asarray(dataset["xcorr"].values, dtype=float)
    expected_shape = (
        len(selected_by_region[SOURCE_REGION]),
        len(selected_by_region[TARGET_REGION]),
        len(lag_times),
    )
    if values.shape != expected_shape:
        raise ValueError("RippleCrossRegionXCorr tensor shape differs from unit coordinates.")
    if analysis_status not in NONTERMINAL_STATUSES:
        if values.shape[:2] != (0, 0):
            raise ValueError("Terminal RippleCrossRegionXCorr tensors must have empty unit axes.")
        return pd.DataFrame(columns=PAIR_SUMMARY_COLUMNS)
    screen = _screen_module()
    ca1_selected = selected_by_region[SOURCE_REGION]
    v1_selected = selected_by_region[TARGET_REGION]
    legacy_summary = screen.build_pair_summary_table(
        xcorr=values,
        ca1_unit_ids=ca1_selected["group_unit_id"].to_numpy(dtype=object),
        v1_unit_ids=v1_selected["group_unit_id"].to_numpy(dtype=object),
        ca1_spike_counts=ca1_selected["ripple_spike_count"].to_numpy(dtype=int),
        v1_spike_counts=v1_selected["ripple_spike_count"].to_numpy(dtype=int),
        lag_times=lag_times,
        extremum_half_width_bins=parameters["extremum_half_width_bins"],
    )
    return _canonical_pair_summary(
        legacy_summary,
        metadata=metadata,
        ca1_units=ca1_units,
        v1_units=v1_units,
    )


def validate_ripple_cross_region_xcorr_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one canonical ripple-only cross-region xcorr result."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    metadata = _metadata(
        ripple_cross_region_xcorr_id=result.get("ripple_cross_region_xcorr_id"),
        animal_name=result.get("animal_name"),
        date=result.get("date"),
        epoch=result.get("epoch"),
    )
    raw_parameters = result.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("result parameters must be a mapping.")
    parameter_keys = (
        "bin_size_s",
        "max_lag_s",
        "min_ripple_spikes",
        "extremum_half_width_bins",
        "norm",
        "expected_detector_zscore_threshold",
        "require_speed_gated",
    )
    missing = [name for name in parameter_keys if name not in raw_parameters]
    if missing:
        raise ValueError(f"RippleCrossRegionXCorr parameters are missing {missing!r}.")
    parameters = _effective_parameters(
        parameter_name=raw_parameters.get("parameter_name"),
        parameter_sha256=raw_parameters.get("parameter_sha256"),
        output_rule_sha256=raw_parameters.get("output_rule_sha256"),
        **{name: raw_parameters[name] for name in parameter_keys},
    )
    provenance, provenance_json = _validate_upstream_provenance(
        result.get("upstream_provenance"), parameters=parameters
    )
    selected_hash = str(result.get("selected_ripple_intervals_sha256", ""))
    if len(selected_hash) != 64:
        raise ValueError("selected_ripple_intervals_sha256 must be a SHA-256 digest.")
    analysis_status = str(result.get("analysis_status", ""))
    if analysis_status not in ANALYSIS_STATUSES:
        raise ValueError("RippleCrossRegionXCorr has an unsupported analysis_status.")
    artifact_origin = str(result.get("artifact_origin", ""))
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("RippleCrossRegionXCorr has an unsupported artifact_origin.")
    legacy = result.get("legacy_artifact_provenance", {})
    if not isinstance(legacy, Mapping):
        raise TypeError("legacy_artifact_provenance must be a mapping.")
    legacy = dict(legacy)
    if artifact_origin == "computed" and legacy:
        raise ValueError("Computed RippleCrossRegionXCorr results cannot claim legacy provenance.")
    if artifact_origin == "registered_existing" and not legacy:
        raise ValueError("Registered RippleCrossRegionXCorr results require legacy provenance.")
    ca1_units = _validate_unit_audit(
        result.get("ca1_units"), region=SOURCE_REGION, parameters=parameters
    )
    v1_units = _validate_unit_audit(
        result.get("v1_units"), region=TARGET_REGION, parameters=parameters
    )
    dataset = result.get("dataset")
    expected_summary = _validate_dataset(
        dataset,
        metadata=metadata,
        parameters=parameters,
        upstream_provenance_json=provenance_json,
        selected_ripple_intervals_sha256=selected_hash,
        ca1_units=ca1_units,
        v1_units=v1_units,
        analysis_status=analysis_status,
        artifact_origin=artifact_origin,
        legacy_artifact_provenance=legacy,
    )
    summary = result.get("summary")
    if not isinstance(summary, pd.DataFrame) or tuple(summary.columns) != PAIR_SUMMARY_COLUMNS:
        raise ValueError("RippleCrossRegionXCorr summary does not match its canonical schema.")
    _assert_frame_equal(summary, expected_summary, name="RippleCrossRegionXCorr summary")
    n_ripples = int(dataset.sizes["ripple"])
    terminal = _terminal_status(ca1_units, v1_units, n_ripples=n_ripples)
    if analysis_status in NONTERMINAL_STATUSES:
        if terminal is not None:
            raise ValueError(
                "Nonterminal RippleCrossRegionXCorr result has terminal inputs: " + terminal
            )
        valid_pairs = summary["status"].eq(PAIR_STATUS_VALID)
        expected_status = (
            "valid"
            if valid_pairs.all()
            else "partial_valid" if valid_pairs.any() else "no_valid_pairs"
        )
        if analysis_status != expected_status:
            raise ValueError("RippleCrossRegionXCorr status differs from pair QC.")
        expected_ca1 = _annotate_unit_qc(
            ca1_units.assign(
                included_in_xcorr=ca1_units["passes_ripple_spike_threshold"],
                n_valid_pairs=0,
                unit_qc_status=np.where(
                    ca1_units["passes_ripple_spike_threshold"],
                    "no_valid_pairs",
                    "excluded_spike_threshold",
                ),
            ),
            pair_summary=summary,
            region=SOURCE_REGION,
            computed=True,
        )
        expected_v1 = _annotate_unit_qc(
            v1_units.assign(
                included_in_xcorr=v1_units["passes_ripple_spike_threshold"],
                n_valid_pairs=0,
                unit_qc_status=np.where(
                    v1_units["passes_ripple_spike_threshold"],
                    "no_valid_pairs",
                    "excluded_spike_threshold",
                ),
            ),
            pair_summary=summary,
            region=TARGET_REGION,
            computed=True,
        )
        _assert_frame_equal(ca1_units, expected_ca1, name="CA1 unit QC")
        _assert_frame_equal(v1_units, expected_v1, name="V1 unit QC")
    elif terminal != analysis_status:
        raise ValueError("RippleCrossRegionXCorr terminal status differs from its inputs.")
    n_valid_pairs = int(summary["status"].eq(PAIR_STATUS_VALID).sum())
    return {
        **metadata,
        "parameters": parameters,
        "upstream_provenance": provenance,
        "selected_ripple_intervals_sha256": selected_hash,
        "ca1_units": ca1_units,
        "v1_units": v1_units,
        "summary": expected_summary,
        "dataset": dataset,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance": legacy,
        "n_ripples": n_ripples,
        "ripple_duration_s": float(
            np.sum(
                np.asarray(dataset["ripple_end_time_s"].values, dtype=float)
                - np.asarray(dataset["ripple_start_time_s"].values, dtype=float)
            )
        ),
        "n_ca1_units": len(ca1_units),
        "n_v1_units": len(v1_units),
        "n_ca1_units_in_xcorr": int(ca1_units["included_in_xcorr"].sum()),
        "n_v1_units_in_xcorr": int(v1_units["included_in_xcorr"].sum()),
        "n_pairs": len(expected_summary),
        "n_valid_pairs": n_valid_pairs,
        "ca1_units_sha256": _table_sha256(ca1_units),
        "v1_units_sha256": _table_sha256(v1_units),
        "summary_sha256": _table_sha256(expected_summary),
    }


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable manifest values repeated for every bundle artifact."""
    return {
        "ripple_cross_region_xcorr_id": result["ripple_cross_region_xcorr_id"],
        "animal_name": result["animal_name"],
        "date": result["date"],
        "epoch": result["epoch"],
        "parameter_name": result["parameters"]["parameter_name"],
        "parameter_sha256": result["parameters"]["parameter_sha256"],
        "output_rule_sha256": result["parameters"]["output_rule_sha256"],
        "upstream_provenance_json": json.dumps(
            result["upstream_provenance"], sort_keys=True, separators=(",", ":")
        ),
        "selected_ripple_intervals_sha256": result[
            "selected_ripple_intervals_sha256"
        ],
        "n_ripples": result["n_ripples"],
        "ripple_duration_s": result["ripple_duration_s"],
        "n_ca1_units": result["n_ca1_units"],
        "n_v1_units": result["n_v1_units"],
        "n_ca1_units_in_xcorr": result["n_ca1_units_in_xcorr"],
        "n_v1_units_in_xcorr": result["n_v1_units_in_xcorr"],
        "n_pairs": result["n_pairs"],
        "n_valid_pairs": result["n_valid_pairs"],
        "ca1_units_sha256": result["ca1_units_sha256"],
        "v1_units_sha256": result["v1_units_sha256"],
        "summary_sha256": result["summary_sha256"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": json.dumps(
            result["legacy_artifact_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def _load_dataset(path: Path) -> Any:
    """Load one NetCDF dataset eagerly and close its backing file."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def write_ripple_cross_region_xcorr_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write, checksum, and reload one complete xcorr bundle."""
    validated = validate_ripple_cross_region_xcorr_result(result)
    destination = Path(path)
    if destination.name != validated["ripple_cross_region_xcorr_id"]:
        raise ValueError("Artifact directory name must equal ripple_cross_region_xcorr_id.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite RippleCrossRegionXCorr artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        validated["ca1_units"].to_parquet(
            temporary / CA1_UNITS_FILENAME, index=False
        )
        validated["v1_units"].to_parquet(temporary / V1_UNITS_FILENAME, index=False)
        validated["summary"].to_parquet(temporary / SUMMARY_FILENAME, index=False)
        validated["dataset"].to_netcdf(temporary / RESULT_FILENAME)
        common = _manifest_common(validated)
        rows = []
        for artifact_key, filename, artifact_kind in (
            ("ca1_units", CA1_UNITS_FILENAME, "parquet"),
            ("v1_units", V1_UNITS_FILENAME, "parquet"),
            ("summary", SUMMARY_FILENAME, "parquet"),
            ("ripple_cross_region_xcorr", RESULT_FILENAME, "netcdf"),
        ):
            artifact_path = temporary / filename
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "relative_path": filename,
                    "artifact_kind": artifact_kind,
                    "file_size_bytes": artifact_path.stat().st_size,
                    "sha256": _file_sha256(artifact_path),
                    **common,
                }
            )
        pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_parquet(
            temporary / MANIFEST_FILENAME, index=False
        )
        load_ripple_cross_region_xcorr_artifact(temporary, _allow_temporary_name=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "artifact_dir": destination,
        "artifact_manifest_path": destination / MANIFEST_FILENAME,
        "ca1_units_path": destination / CA1_UNITS_FILENAME,
        "v1_units_path": destination / V1_UNITS_FILENAME,
        "summary_path": destination / SUMMARY_FILENAME,
        "result_path": destination / RESULT_FILENAME,
    }


def load_ripple_cross_region_xcorr_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one complete RippleCrossRegionXCorr bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RippleCrossRegionXCorr manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or len(manifest) != 4:
        raise ValueError("RippleCrossRegionXCorr manifest does not have the canonical schema.")
    expected = {
        "ca1_units": (CA1_UNITS_FILENAME, "parquet"),
        "v1_units": (V1_UNITS_FILENAME, "parquet"),
        "summary": (SUMMARY_FILENAME, "parquet"),
        "ripple_cross_region_xcorr": (RESULT_FILENAME, "netcdf"),
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("RippleCrossRegionXCorr manifest lacks canonical artifacts.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("RippleCrossRegionXCorr manifest names or kinds are stale.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"RippleCrossRegionXCorr artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or _file_sha256(
            artifact_path
        ) != str(row["sha256"]):
            raise ValueError(f"RippleCrossRegionXCorr checksum mismatch: {artifact_path}")
    first = manifest.iloc[0]
    for name in MANIFEST_COLUMNS[5:]:
        if not np.all(manifest[name].astype(str) == str(first[name])):
            raise ValueError(f"RippleCrossRegionXCorr manifest has inconsistent {name!r}.")
    result_id = str(first["ripple_cross_region_xcorr_id"])
    if not _allow_temporary_name and directory.name != result_id:
        raise ValueError("Artifact directory name does not match ripple_cross_region_xcorr_id.")
    dataset = _load_dataset(directory / RESULT_FILENAME)
    try:
        effective = json.loads(str(dataset.attrs["effective_parameters_json"]))
        upstream = json.loads(str(first["upstream_provenance_json"]))
        legacy = json.loads(str(first["legacy_artifact_provenance_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("RippleCrossRegionXCorr artifact contains malformed provenance.") from exc
    validated = validate_ripple_cross_region_xcorr_result(
        {
            "ripple_cross_region_xcorr_id": result_id,
            "animal_name": str(first["animal_name"]),
            "date": str(first["date"]),
            "epoch": str(first["epoch"]),
            "parameters": {
                "parameter_name": str(first["parameter_name"]),
                "parameter_sha256": str(first["parameter_sha256"]),
                "output_rule_sha256": str(first["output_rule_sha256"]),
                **effective,
            },
            "upstream_provenance": upstream,
            "selected_ripple_intervals_sha256": str(
                first["selected_ripple_intervals_sha256"]
            ),
            "ca1_units": pd.read_parquet(directory / CA1_UNITS_FILENAME),
            "v1_units": pd.read_parquet(directory / V1_UNITS_FILENAME),
            "summary": pd.read_parquet(directory / SUMMARY_FILENAME),
            "dataset": dataset,
            "analysis_status": str(first["analysis_status"]),
            "artifact_origin": str(first["artifact_origin"]),
            "legacy_artifact_provenance": legacy,
        }
    )
    integer_fields = (
        "n_ripples",
        "n_ca1_units",
        "n_v1_units",
        "n_ca1_units_in_xcorr",
        "n_v1_units_in_xcorr",
        "n_pairs",
        "n_valid_pairs",
    )
    if any(validated[name] != int(first[name]) for name in integer_fields):
        raise ValueError("RippleCrossRegionXCorr manifest counts differ from its artifacts.")
    if not np.isclose(
        validated["ripple_duration_s"],
        float(first["ripple_duration_s"]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("RippleCrossRegionXCorr manifest duration differs from its dataset.")
    for name in ("ca1_units_sha256", "v1_units_sha256", "summary_sha256"):
        if validated[name] != str(first[name]):
            raise ValueError(f"RippleCrossRegionXCorr manifest {name} is mismatched.")
    return {**validated, "manifest": manifest}


def _load_legacy_unit_filter(path: Path, *, region: str) -> pd.DataFrame:
    """Load one legacy five-column regional spike-count filter."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy {region} unit filter not found: {source_path}")
    table = pd.read_parquet(source_path)
    expected_columns = (
        "region",
        "unit_id",
        "state_spike_count",
        "passes_state_spike_count",
        "keep_unit",
    )
    if tuple(table.columns) != expected_columns:
        raise ValueError(f"Legacy {region} unit filter has an unsupported schema.")
    output = table.copy().reset_index(drop=True)
    if not output["region"].astype(str).eq(region).all():
        raise ValueError(f"Legacy {region} unit filter contains another region.")
    legacy_ids = output["unit_id"].map(str)
    if legacy_ids.eq("").any() or legacy_ids.duplicated().any():
        raise ValueError(f"Legacy {region} unit identifiers must be unique and non-empty.")
    counts = pd.to_numeric(output["state_spike_count"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0) or not np.allclose(
        counts, np.rint(counts), rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"Legacy {region} spike counts must be non-negative integers.")
    passes = _boolean_array(
        output["passes_state_spike_count"].tolist(),
        name=f"Legacy {region} passes_state_spike_count",
    )
    keep = _boolean_array(
        output["keep_unit"].tolist(), name=f"Legacy {region} keep_unit"
    )
    output["passes_state_spike_count"] = passes
    output["keep_unit"] = keep
    if not np.array_equal(passes, counts >= DEFAULT_MIN_RIPPLE_SPIKES):
        raise ValueError(f"Legacy {region} spike-threshold flags are stale.")
    if not np.array_equal(keep, passes):
        raise ValueError(f"Legacy {region} keep_unit differs from threshold passing.")
    output["legacy_unit_id"] = legacy_ids
    output["state_spike_count"] = counts.astype(int)
    return output


def _resolve_legacy_identities(
    table: pd.DataFrame,
    *,
    region: str,
    resolver: Callable[[Sequence[Any]], Sequence[Mapping[str, Any]]],
    expected_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Resolve legacy sorting ids independently of current ephemeral TsGroup keys."""
    if not callable(resolver):
        raise TypeError(f"{region}_legacy_identity_resolver must be callable.")
    raw_ids = table["unit_id"].tolist()
    resolved = [dict(value) for value in resolver(raw_ids)]
    if len(resolved) != len(raw_ids):
        raise ValueError(
            f"{region} legacy identity resolver returned the wrong number of rows."
        )
    rows = []
    legacy_to_stable: dict[str, str] = {}
    for legacy_id, identity in zip(raw_ids, resolved, strict=True):
        missing = [
            name
            for name in ("spikesorting_merge_id", "unit_id")
            if name not in identity
        ]
        if missing:
            raise ValueError(
                f"{region} legacy identity resolver is missing fields {missing!r}."
            )
        merge_id = str(identity["spikesorting_merge_id"])
        unit_id = str(identity["unit_id"])
        stable_id = str(identity.get("stable_unit_id", f"{merge_id}:{unit_id}"))
        if stable_id != f"{merge_id}:{unit_id}":
            raise ValueError(f"{region} resolver returned a noncanonical stable unit id.")
        legacy_key = str(legacy_id)
        if stable_id in legacy_to_stable.values():
            raise ValueError(f"{region} resolver returned duplicate persistent units.")
        legacy_to_stable[legacy_key] = stable_id
        rows.append(
            {
                "legacy_unit_id": legacy_key,
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": stable_id,
            }
        )
    resolved_table = pd.DataFrame.from_records(rows)
    expected_stable = set(expected_audit["stable_unit_id"].astype(str))
    if set(resolved_table.get("stable_unit_id", pd.Series(dtype=str))) != expected_stable:
        raise ValueError(
            f"Legacy {region} resolved units differ from selected NWB spike units."
        )
    return resolved_table, legacy_to_stable


def _compare_legacy_unit_filter(
    source: pd.DataFrame,
    resolved: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    region: str,
) -> None:
    """Compare a legacy filter to recomputed counts through stable identity."""
    observed = source.merge(
        resolved.loc[:, ["legacy_unit_id", "stable_unit_id"]],
        on="legacy_unit_id",
        how="left",
        validate="one_to_one",
    ).loc[
        :,
        [
            "stable_unit_id",
            "state_spike_count",
            "passes_state_spike_count",
            "keep_unit",
        ],
    ]
    canonical = expected.loc[
        :,
        [
            "stable_unit_id",
            "ripple_spike_count",
            "passes_ripple_spike_threshold",
            "passes_ripple_spike_threshold",
        ],
    ].copy()
    canonical.columns = observed.columns
    observed = observed.sort_values("stable_unit_id", kind="stable").reset_index(drop=True)
    canonical = canonical.sort_values("stable_unit_id", kind="stable").reset_index(drop=True)
    _assert_frame_equal(
        observed,
        canonical,
        name=f"Legacy {region} unit filter",
        rtol=0.0,
        atol=0.0,
    )


def _compare_legacy_pair_summary(
    path: Path,
    *,
    expected: pd.DataFrame,
    ca1_legacy_to_stable: Mapping[str, str],
    v1_legacy_to_stable: Mapping[str, str],
) -> None:
    """Require all legacy pair-summary values to match the NWB recomputation."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy xcorr summary not found: {source_path}")
    source = pd.read_parquet(source_path)
    source_columns = (
        "ca1_unit_id",
        "v1_unit_id",
        "n_ca1_state_spikes",
        "n_v1_state_spikes",
        "peak_lag_s",
        "peak_norm_xcorr",
        "status",
    )
    if tuple(source.columns) != source_columns:
        raise ValueError("Legacy xcorr summary has an unsupported schema.")
    observed = source.copy()
    try:
        observed["ca1_stable_unit_id"] = observed["ca1_unit_id"].map(
            lambda value: ca1_legacy_to_stable[str(value)]
        )
        observed["v1_stable_unit_id"] = observed["v1_unit_id"].map(
            lambda value: v1_legacy_to_stable[str(value)]
        )
    except KeyError as exc:
        raise ValueError("Legacy pair summary contains an unresolved unit id.") from exc
    observed = observed.loc[
        :,
        [
            "ca1_stable_unit_id",
            "v1_stable_unit_id",
            "n_ca1_state_spikes",
            "n_v1_state_spikes",
            "peak_lag_s",
            "peak_norm_xcorr",
            "status",
        ],
    ]
    canonical = expected.loc[
        :,
        [
            "ca1_stable_unit_id",
            "v1_stable_unit_id",
            "n_ca1_ripple_spikes",
            "n_v1_ripple_spikes",
            "peak_lag_s",
            "peak_norm_xcorr",
            "status",
        ],
    ].copy()
    canonical.columns = observed.columns
    sort_columns = ["ca1_stable_unit_id", "v1_stable_unit_id"]
    observed = observed.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    canonical = canonical.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    _assert_frame_equal(
        observed,
        canonical,
        name="Legacy xcorr pair summary",
        rtol=2e-7,
        atol=2e-8,
    )


def _compare_legacy_dataset(
    path: Path,
    *,
    expected: Any,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    ca1_legacy_to_stable: Mapping[str, str],
    v1_legacy_to_stable: Mapping[str, str],
) -> None:
    """Require the legacy xcorr tensor and lag grid to match recomputation."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy xcorr NetCDF not found: {source_path}")
    source = _load_dataset(source_path)
    expected_attrs = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "state": "ripple",
        "epoch_group_label": metadata["epoch"],
        "bin_size_s": parameters["bin_size_s"],
        "max_lag_s": parameters["max_lag_s"],
        "min_state_spikes": parameters["min_ripple_spikes"],
        "extremum_half_width_bins": parameters["extremum_half_width_bins"],
    }
    for name, expected_value in expected_attrs.items():
        if name not in source.attrs:
            raise ValueError(f"Legacy xcorr dataset is missing attribute {name!r}.")
        observed = source.attrs[name]
        if isinstance(expected_value, float):
            matches = np.isclose(
                float(observed), expected_value, rtol=0.0, atol=1e-12
            )
        else:
            matches = str(observed) == str(expected_value)
        if not matches:
            raise ValueError(f"Legacy xcorr dataset has mismatched {name!r}.")
    try:
        selected_epochs = json.loads(str(source.attrs["selected_epochs_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Legacy xcorr dataset lacks selected epoch provenance.") from exc
    if selected_epochs != [metadata["epoch"]]:
        raise ValueError("Legacy xcorr dataset is not one exact, unpooled epoch.")
    interval_source = str(source.attrs.get("state_interval_source", ""))
    if not interval_source or interval_source.startswith("fixed_ripple_windows["):
        raise ValueError(
            "Legacy xcorr dataset must use exact detected ripple intervals, "
            "not fixed ripple windows."
        )
    if "xcorr" not in source or source["xcorr"].dims != (
        "ca1_unit",
        "v1_unit",
        "lag_s",
    ):
        raise ValueError("Legacy xcorr dataset lacks its canonical tensor.")
    try:
        source_ca1_stable = [
            ca1_legacy_to_stable[str(value)]
            for value in np.asarray(source.coords["ca1_unit"].values).tolist()
        ]
        source_v1_stable = [
            v1_legacy_to_stable[str(value)]
            for value in np.asarray(source.coords["v1_unit"].values).tolist()
        ]
    except KeyError as exc:
        raise ValueError("Legacy xcorr dataset contains an unresolved unit id.") from exc
    expected_ca1 = np.asarray(expected.coords["ca1_unit"].values).astype(str).tolist()
    expected_v1 = np.asarray(expected.coords["v1_unit"].values).astype(str).tolist()
    if set(source_ca1_stable) != set(expected_ca1) or set(source_v1_stable) != set(
        expected_v1
    ):
        raise ValueError("Legacy xcorr tensor units differ from NWB recomputation.")
    if len(set(source_ca1_stable)) != len(source_ca1_stable) or len(
        set(source_v1_stable)
    ) != len(source_v1_stable):
        raise ValueError("Legacy xcorr tensor contains duplicate resolved units.")
    ca1_index = {stable: index for index, stable in enumerate(source_ca1_stable)}
    v1_index = {stable: index for index, stable in enumerate(source_v1_stable)}
    reordered = np.asarray(source["xcorr"].values, dtype=float)[
        np.ix_(
            [ca1_index[value] for value in expected_ca1],
            [v1_index[value] for value in expected_v1],
            np.arange(source.sizes["lag_s"], dtype=int),
        )
    ]
    source_lags = np.asarray(source.coords["lag_s"].values, dtype=float)
    expected_lags = np.asarray(expected.coords["lag_s"].values, dtype=float)
    if not np.allclose(source_lags, expected_lags, rtol=0.0, atol=2e-8):
        raise ValueError("Legacy xcorr lag grid differs from NWB recomputation.")
    if not np.allclose(
        reordered,
        np.asarray(expected["xcorr"].values, dtype=float),
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    ):
        raise ValueError("Legacy xcorr tensor differs from exact NWB recomputation.")


def register_existing_ripple_cross_region_xcorr_artifact(
    *,
    source_ca1_unit_filter_path: Path,
    source_v1_unit_filter_path: Path,
    source_summary_path: Path,
    source_result_path: Path,
    destination_path: Path,
    ripple_cross_region_xcorr_id: Any,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_table: Any,
    ca1_spikes: Any,
    ca1_stable_unit_ids: Sequence[Mapping[str, Any]],
    v1_spikes: Any,
    v1_stable_unit_ids: Sequence[Mapping[str, Any]],
    upstream_provenance: Mapping[str, Any],
    ca1_legacy_identity_resolver: Callable[
        [Sequence[Any]], Sequence[Mapping[str, Any]]
    ],
    v1_legacy_identity_resolver: Callable[
        [Sequence[Any]], Sequence[Mapping[str, Any]]
    ],
    ca1_sorting_type: str,
    v1_sorting_type: str,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    expected_selected_ripple_intervals_sha256: str | None = None,
    bin_size_s: float = DEFAULT_BIN_SIZE_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_ripple_spikes: int = DEFAULT_MIN_RIPPLE_SPIKES,
    extremum_half_width_bins: int = DEFAULT_EXTREMUM_HALF_WIDTH_BINS,
    norm: bool = True,
    expected_detector_zscore_threshold: float = (
        DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD
    ),
    require_speed_gated: bool = DEFAULT_REQUIRE_SPEED_GATED,
    source_v1ca1_git_commit: str | None = None,
    source_spyglass_git_commit: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recompute exact NWB inputs and register only four matching legacy artifacts."""
    if str(ca1_sorting_type) != "ImportedSpikeSorting" or str(
        v1_sorting_type
    ) != "ImportedSpikeSorting":
        raise ValueError(
            "Legacy RippleCrossRegionXCorr registration requires "
            "ImportedSpikeSorting for both CA1 and V1 groups."
        )
    source_paths = {
        "ca1_unit_filter": Path(source_ca1_unit_filter_path),
        "v1_unit_filter": Path(source_v1_unit_filter_path),
        "summary": Path(source_summary_path),
        "result": Path(source_result_path),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Legacy {name} artifact not found: {path}")
    recomputed = compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=ripple_cross_region_xcorr_id,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        ripple_table=ripple_table,
        ca1_spikes=ca1_spikes,
        ca1_stable_unit_ids=ca1_stable_unit_ids,
        v1_spikes=v1_spikes,
        v1_stable_unit_ids=v1_stable_unit_ids,
        upstream_provenance=upstream_provenance,
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        expected_selected_ripple_intervals_sha256=(
            expected_selected_ripple_intervals_sha256
        ),
        bin_size_s=bin_size_s,
        max_lag_s=max_lag_s,
        min_ripple_spikes=min_ripple_spikes,
        extremum_half_width_bins=extremum_half_width_bins,
        norm=norm,
        expected_detector_zscore_threshold=expected_detector_zscore_threshold,
        require_speed_gated=require_speed_gated,
    )
    if recomputed["analysis_status"] not in NONTERMINAL_STATUSES:
        raise ValueError(
            "Legacy registration requires a recomputed nonterminal xcorr result."
        )
    ca1_source = _load_legacy_unit_filter(
        source_paths["ca1_unit_filter"], region=SOURCE_REGION
    )
    v1_source = _load_legacy_unit_filter(
        source_paths["v1_unit_filter"], region=TARGET_REGION
    )
    ca1_resolved, ca1_legacy_to_stable = _resolve_legacy_identities(
        ca1_source,
        region=SOURCE_REGION,
        resolver=ca1_legacy_identity_resolver,
        expected_audit=recomputed["ca1_units"],
    )
    v1_resolved, v1_legacy_to_stable = _resolve_legacy_identities(
        v1_source,
        region=TARGET_REGION,
        resolver=v1_legacy_identity_resolver,
        expected_audit=recomputed["v1_units"],
    )
    _compare_legacy_unit_filter(
        ca1_source,
        ca1_resolved,
        recomputed["ca1_units"],
        region=SOURCE_REGION,
    )
    _compare_legacy_unit_filter(
        v1_source,
        v1_resolved,
        recomputed["v1_units"],
        region=TARGET_REGION,
    )
    _compare_legacy_pair_summary(
        source_paths["summary"],
        expected=recomputed["summary"],
        ca1_legacy_to_stable=ca1_legacy_to_stable,
        v1_legacy_to_stable=v1_legacy_to_stable,
    )
    metadata = {
        name: recomputed[name]
        for name in ("ripple_cross_region_xcorr_id", "animal_name", "date", "epoch")
    }
    _compare_legacy_dataset(
        source_paths["result"],
        expected=recomputed["dataset"],
        metadata=metadata,
        parameters=recomputed["parameters"],
        ca1_legacy_to_stable=ca1_legacy_to_stable,
        v1_legacy_to_stable=v1_legacy_to_stable,
    )
    legacy_provenance = {
        "registration_policy": (
            "exact_nwb_recomputation_and_all_four_legacy_artifact_comparison"
        ),
        "unit_identity_policy": (
            "separate_region_imported_sorting_id_resolvers_to_persistent_identity"
        ),
        "ca1_sorting_type": str(ca1_sorting_type),
        "v1_sorting_type": str(v1_sorting_type),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": {
            name: _file_sha256(path) for name, path in source_paths.items()
        },
        "source_v1ca1_git_commit": (
            "unknown" if source_v1ca1_git_commit is None else str(source_v1ca1_git_commit)
        ),
        "source_spyglass_git_commit": (
            "unknown" if source_spyglass_git_commit is None else str(source_spyglass_git_commit)
        ),
        "compared_artifacts": [
            "ca1_unit_filter",
            "v1_unit_filter",
            "summary",
            "result",
        ],
        "detector_zscore_threshold": recomputed["parameters"][
            "expected_detector_zscore_threshold"
        ],
        "speed_gated": recomputed["parameters"]["require_speed_gated"],
    }
    registered_dataset = recomputed["dataset"].copy(deep=True)
    registered_dataset.attrs = _dataset_attrs(
        metadata=metadata,
        parameters=recomputed["parameters"],
        upstream_provenance_json=json.dumps(
            recomputed["upstream_provenance"], sort_keys=True, separators=(",", ":")
        ),
        selected_ripple_intervals_sha256=recomputed[
            "selected_ripple_intervals_sha256"
        ],
        analysis_status=recomputed["analysis_status"],
        artifact_origin="registered_existing",
        legacy_artifact_provenance=legacy_provenance,
    )
    registered = validate_ripple_cross_region_xcorr_result(
        {
            **metadata,
            "parameters": recomputed["parameters"],
            "upstream_provenance": recomputed["upstream_provenance"],
            "selected_ripple_intervals_sha256": recomputed[
                "selected_ripple_intervals_sha256"
            ],
            "ca1_units": recomputed["ca1_units"],
            "v1_units": recomputed["v1_units"],
            "summary": recomputed["summary"],
            "dataset": registered_dataset,
            "analysis_status": recomputed["analysis_status"],
            "artifact_origin": "registered_existing",
            "legacy_artifact_provenance": legacy_provenance,
        }
    )
    paths = write_ripple_cross_region_xcorr_artifact(
        registered, destination_path, overwrite=overwrite
    )
    return {**registered, **paths}


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD",
    "DEFAULT_REQUIRE_SPEED_GATED",
    "MANIFEST_COLUMNS",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "PAIR_SUMMARY_COLUMNS",
    "RESULT_SCHEMA_VERSION",
    "UNIT_AUDIT_COLUMNS",
    "compute_ripple_cross_region_xcorr",
    "get_ripple_cross_region_xcorr_artifact_paths",
    "get_legacy_ripple_cross_region_xcorr_paths",
    "load_ripple_cross_region_xcorr_artifact",
    "prepare_ripple_cross_region_xcorr_event_selection",
    "register_existing_ripple_cross_region_xcorr_artifact",
    "validate_ripple_cross_region_xcorr_parameters",
    "validate_ripple_cross_region_xcorr_result",
    "write_ripple_cross_region_xcorr_artifact",
]
