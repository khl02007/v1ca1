from __future__ import annotations

import inspect
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from v1ca1.ripple import plot_ripple_modulation as ripple_plot
from v1ca1.spyglass import ripple_modulation


RIPPLE_MODULATION_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")


class _FakeTs:
    def __init__(self, t: np.ndarray, **_kwargs: Any) -> None:
        self.t = np.asarray(t, dtype=float)


class _FakeIntervalSet:
    def __init__(self, start: float, end: float, **_kwargs: Any) -> None:
        self.start = np.asarray([start], dtype=float)
        self.end = np.asarray([end], dtype=float)


class _FakePerievent:
    def __init__(self, counts: pd.DataFrame) -> None:
        self._counts = counts

    def count(self, _bin_size_s: float) -> pd.DataFrame:
        return self._counts


def _install_fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> None:
    def _compute_perievent(
        *,
        timestamps: types.SimpleNamespace,
        tref: _FakeTs,
        minmax: tuple[float, float],
        time_unit: str,
    ) -> _FakePerievent:
        assert tref.t.ndim == 1
        assert minmax == (-0.2, 0.2)
        assert time_unit == "s"
        if timestamps.unit_id == 11:
            values = [[0.0, 1.0], [2.0, 1.0], [4.0, 5.0], [3.0, 2.0]]
        else:
            values = [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
        return _FakePerievent(
            pd.DataFrame(
                values,
                index=np.array([-0.15, -0.05, 0.05, 0.15], dtype=float),
            )
        )

    fake_pynapple = types.SimpleNamespace(
        Ts=_FakeTs,
        IntervalSet=_FakeIntervalSet,
        compute_perievent=_compute_perievent,
    )
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)


def _ripple_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": ["02_r1", "02_r1", "03_r2"],
            "start_time": [0.25, 0.75, 1.25],
            "end_time": [0.30, 0.80, 1.30],
            "mean_zscore": [1.5, 5.0, 6.0],
        }
    )


def _compute(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    _install_fake_pynapple(monkeypatch)
    ripple_table = kwargs.pop("ripple_table", _ripple_table())
    return ripple_modulation.compute_epoch_region_ripple_modulation(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_table=ripple_table,
        epoch_timestamps=np.linspace(0.0, 1.0, 1001),
        region_spikes={
            11: types.SimpleNamespace(unit_id=11),
            12: types.SimpleNamespace(unit_id=12),
        },
        bin_size_s=0.1,
        time_before_s=0.2,
        time_after_s=0.2,
        response_window=(0.0, 0.2),
        baseline_window=(-0.2, 0.0),
        **kwargs,
    )


def _legacy_artifact_paths(
    directory: Path,
    *,
    region_label: str = "all_regions",
    ripple_threshold_zscore: float | None = None,
) -> dict[str, Path]:
    """Return standalone-script artifact paths for registration tests."""
    stem = ripple_plot.build_epoch_output_stem(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region_label=region_label,
        ripple_threshold_zscore=ripple_threshold_zscore,
        bin_size_s=ripple_plot.DEFAULT_BIN_SIZE_S,
        time_before_s=ripple_plot.DEFAULT_TIME_BEFORE_S,
        time_after_s=ripple_plot.DEFAULT_TIME_AFTER_S,
        response_window=(
            ripple_plot.DEFAULT_RESPONSE_WINDOW_START_S,
            ripple_plot.DEFAULT_RESPONSE_WINDOW_END_S,
        ),
        baseline_window=(
            ripple_plot.DEFAULT_BASELINE_WINDOW_START_S,
            ripple_plot.DEFAULT_BASELINE_WINDOW_END_S,
        ),
        heatmap_normalize=ripple_plot.DEFAULT_HEATMAP_NORMALIZE,
    )
    return {
        "summary": directory / f"{stem}_summary.parquet",
        "peri_ripple_firing_rate": (
            directory / f"{stem}_peri_ripple_firing_rate.parquet"
        ),
    }


def _legacy_summary_table(
    *,
    regions: tuple[str, ...] = ("v1", "ca1"),
) -> pd.DataFrame:
    """Return a complete standalone summary fixture."""
    rows = []
    for unit_id, region in enumerate(regions, start=11):
        rows.append(
            {
                "animal_name": "RatA",
                "date": "20240101",
                "epoch": "02_r1",
                "region": region,
                "unit_id": unit_id,
                "n_ripples": 2,
                "bin_size_s": ripple_plot.DEFAULT_BIN_SIZE_S,
                "time_before_s": ripple_plot.DEFAULT_TIME_BEFORE_S,
                "time_after_s": ripple_plot.DEFAULT_TIME_AFTER_S,
                "baseline_mean_hz": 1.0,
                "baseline_std_hz": 0.5,
                "response_mean_hz": 1.5,
                "ripple_modulation_index": 0.2,
                "response_zscore": 1.0,
                "invalid_reason": None,
            }
        )
    return pd.DataFrame(rows, columns=ripple_plot.SUMMARY_COLUMNS)


def _legacy_peri_table(
    *,
    regions: tuple[str, ...] = ("v1", "ca1"),
) -> pd.DataFrame:
    """Return complete standalone peri-ripple traces on the default grid."""
    bin_size_s = ripple_plot.DEFAULT_BIN_SIZE_S
    time_values = (
        -ripple_plot.DEFAULT_TIME_BEFORE_S
        + (bin_size_s / 2.0)
        + np.arange(
            round(
                (
                    ripple_plot.DEFAULT_TIME_BEFORE_S
                    + ripple_plot.DEFAULT_TIME_AFTER_S
                )
                / bin_size_s
            )
        )
        * bin_size_s
    )
    rows = []
    for unit_id, region in enumerate(regions, start=11):
        for time_s in time_values:
            rows.append(
                {
                    "animal_name": "RatA",
                    "date": "20240101",
                    "epoch": "02_r1",
                    "region": region,
                    "unit_id": unit_id,
                    "n_ripples": 2,
                    "bin_size_s": bin_size_s,
                    "time_before_s": ripple_plot.DEFAULT_TIME_BEFORE_S,
                    "time_after_s": ripple_plot.DEFAULT_TIME_AFTER_S,
                    "time_s": float(time_s),
                    "mean_rate_hz": 1.0,
                }
            )
    return pd.DataFrame(
        rows,
        columns=ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS,
    )


def _write_legacy_artifacts(
    paths: dict[str, Path],
    *,
    summary: pd.DataFrame | None = None,
    peri: pd.DataFrame | None = None,
) -> None:
    """Write one pair of legacy registration fixtures."""
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    (summary if summary is not None else _legacy_summary_table()).to_parquet(
        paths["summary"],
        index=False,
    )
    (peri if peri is not None else _legacy_peri_table()).to_parquet(
        paths["peri_ripple_firing_rate"],
        index=False,
    )


def _nwb_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one complete stable-identity table pair for NWB tests."""
    summary = _legacy_summary_table(regions=("v1",))
    peri = _legacy_peri_table(regions=("v1",))
    for table in (summary, peri):
        table["unit_id"] = table["unit_id"].astype(str)
        table["group_unit_id"] = 11
        table["spikesorting_merge_id"] = "merge-a"
        table["stable_unit_id"] = "merge-a:11"
    return summary, peri


def _registration_plan(
    paths: dict[str, Path],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Plan registration of one fixture pair into a UUID destination."""
    return ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_modulation_id=RIPPLE_MODULATION_ID,
        existing_summary_path=paths["summary"],
        existing_peri_ripple_firing_rate_path=paths[
            "peri_ripple_firing_rate"
        ],
        artifact_root=artifact_root,
    )


def test_compute_one_epoch_region_uses_all_detector_events_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)

    assert result["epoch"] == "02_r1"
    assert result["region"] == "v1"
    assert result["n_ripples"] == 2
    assert "minimum_ripple_mean_zscore" not in result
    assert result["selected_ripple_table"]["epoch"].unique().tolist() == ["02_r1"]
    assert result["selected_ripple_table"]["mean_zscore"].tolist() == [1.5, 5.0]
    assert list(result["summary"].columns) == ripple_plot.SUMMARY_COLUMNS
    assert list(result["peri_ripple_firing_rate"].columns) == (
        ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS
    )
    assert result["summary"].shape[0] == 2
    assert result["peri_ripple_firing_rate"].shape[0] == 8
    assert set(result["summary"]["region"]) == {"v1"}


def test_compute_accepts_pynapple_intervalset_dataframe_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IntervalSetInput:
        def as_dataframe(self) -> pd.DataFrame:
            return _ripple_table().rename(
                columns={"start_time": "start", "end_time": "end"}
            )

    _install_fake_pynapple(monkeypatch)
    result = ripple_modulation.compute_epoch_region_ripple_modulation(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_table=_IntervalSetInput(),
        epoch_timestamps=np.linspace(0.0, 1.0, 1001),
        region_spikes={11: types.SimpleNamespace(unit_id=11)},
        bin_size_s=0.1,
        time_before_s=0.2,
        time_after_s=0.2,
        response_window=(0.0, 0.2),
        baseline_window=(-0.2, 0.0),
    )

    assert result["n_ripples"] == 2
    assert list(result["selected_ripple_table"]["start_time"]) == [0.25, 0.75]


def test_compute_rejects_ripples_outside_selected_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ripple_table = pd.DataFrame(
        {
            "epoch": ["02_r1"],
            "start_time": [0.95],
            "end_time": [1.05],
        }
    )

    with pytest.raises(ValueError, match="within epoch_timestamps"):
        _compute(monkeypatch, ripple_table=ripple_table)


def test_compute_and_registration_apis_have_no_secondary_mean_zscore_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compute_parameters = inspect.signature(
        ripple_modulation.compute_epoch_region_ripple_modulation
    ).parameters
    registration_parameters = inspect.signature(
        ripple_modulation.plan_register_existing
    ).parameters

    assert "minimum_ripple_mean_zscore" not in compute_parameters
    assert "minimum_ripple_mean_zscore" not in registration_parameters
    with pytest.raises(TypeError, match="minimum_ripple_mean_zscore"):
        _compute(monkeypatch, minimum_ripple_mean_zscore=4.0)

    legacy_paths = _legacy_artifact_paths(tmp_path)
    with pytest.raises(TypeError, match="minimum_ripple_mean_zscore"):
        ripple_modulation.plan_register_existing(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            region="v1",
            ripple_modulation_id=RIPPLE_MODULATION_ID,
            existing_summary_path=legacy_paths["summary"],
            existing_peri_ripple_firing_rate_path=legacy_paths[
                "peri_ripple_firing_rate"
            ],
            artifact_root=tmp_path / "native",
            minimum_ripple_mean_zscore=4.0,
        )


def test_compute_writes_persistent_and_convenience_unit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(
        monkeypatch,
        stable_unit_ids=[
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
            {"spikesorting_merge_id": "merge-b", "unit_id": 12},
        ],
    )

    expected_stable_ids = ["merge-a:11", "merge-b:12"]
    assert ripple_modulation.STABLE_UNIT_COLUMNS == (
        "spikesorting_merge_id",
        "unit_id",
    )
    assert result["summary"]["unit_id"].tolist() == ["11", "12"]
    assert result["summary"]["spikesorting_merge_id"].tolist() == [
        "merge-a",
        "merge-b",
    ]
    assert result["summary"]["stable_unit_id"].tolist() == expected_stable_ids
    assert result["summary"]["group_unit_id"].tolist() == [11, 12]
    assert set(result["peri_ripple_firing_rate"]["unit_id"]) == {"11", "12"}
    assert set(result["peri_ripple_firing_rate"]["stable_unit_id"]) == set(
        expected_stable_ids
    )
    assert result["heatmap_payload"]["unit_ids"].tolist() == expected_stable_ids


def test_stable_unit_identity_must_align_with_tsgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        _compute(
            monkeypatch,
            stable_unit_ids=[
                {"spikesorting_merge_id": "merge-a", "unit_id": 11}
            ],
        )


def test_compute_preserves_empty_parquet_schemas_when_epoch_has_no_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(
        monkeypatch,
        ripple_table=_ripple_table().loc[lambda table: table["epoch"] == "03_r2"],
        stable_unit_ids=[
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
            {"spikesorting_merge_id": "merge-b", "unit_id": 12},
        ],
    )

    assert result["n_ripples"] == 0
    assert result["summary"].empty
    assert result["peri_ripple_firing_rate"].empty
    assert list(result["summary"].columns) == [
        *ripple_plot.SUMMARY_COLUMNS,
        "group_unit_id",
        "spikesorting_merge_id",
        "stable_unit_id",
    ]
    assert list(result["peri_ripple_firing_rate"].columns) == (
        [
            *ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS,
            "group_unit_id",
            "spikesorting_merge_id",
            "stable_unit_id",
        ]
    )


@pytest.mark.parametrize("empty", [False, True])
def test_ripple_modulation_nwb_tables_roundtrip_real_hdf5(
    tmp_path: Path,
    empty: bool,
) -> None:
    from pynwb import NWBHDF5IO, NWBFile

    summary, peri = _nwb_tables()
    if empty:
        summary = summary.iloc[0:0]
        peri = peri.iloc[0:0]
    summary_object = (
        ripple_modulation.ripple_modulation_summary_to_dynamic_table(
            summary
        )
    )
    peri_object = (
        ripple_modulation.peri_ripple_firing_rate_to_dynamic_table(peri)
    )
    summary_id = str(summary_object.object_id)
    peri_id = str(peri_object.object_id)
    nwbfile = NWBFile(
        session_description="RippleModulation storage test",
        identifier=f"ripple-modulation-{empty}",
        session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    nwbfile.add_scratch(summary_object)
    nwbfile.add_scratch(peri_object)
    path = tmp_path / f"ripple-modulation-{empty}.nwb"
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)

    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        stored = io.read()
        loaded_summary = (
            ripple_modulation.ripple_modulation_summary_from_dynamic_table(
                stored.objects[summary_id]
            )
        )
        loaded_peri = (
            ripple_modulation.peri_ripple_firing_rate_from_dynamic_table(
                stored.objects[peri_id]
            )
        )
        validated_summary, validated_peri = (
            ripple_modulation.validate_ripple_modulation_tables(
                loaded_summary,
                loaded_peri,
            )
        )

    assert not __import__("pynwb").validate(path=path)
    assert list(validated_summary.columns) == list(
        ripple_modulation.SUMMARY_TABLE_COLUMNS
    )
    assert list(validated_peri.columns) == list(
        ripple_modulation.PERI_RIPPLE_FIRING_RATE_TABLE_COLUMNS
    )
    assert validated_summary.empty is empty
    assert validated_peri.empty is empty
    if not empty:
        assert validated_summary["invalid_reason"].isna().all()
        assert validated_summary["stable_unit_id"].tolist() == ["merge-a:11"]
        assert validated_peri["stable_unit_id"].unique().tolist() == [
            "merge-a:11"
        ]
    assert ripple_modulation.ripple_modulation_summary_sha256(
        validated_summary
    ) == ripple_modulation.ripple_modulation_summary_sha256(summary)
    assert ripple_modulation.peri_ripple_firing_rate_sha256(
        validated_peri
    ) == ripple_modulation.peri_ripple_firing_rate_sha256(peri)


def test_ripple_modulation_nwb_tables_cross_validate_identity_and_grid() -> None:
    summary, peri = _nwb_tables()
    ripple_modulation.validate_ripple_modulation_tables(summary, peri)

    wrong_identity = peri.copy()
    wrong_identity["stable_unit_id"] = "merge-a:12"
    with pytest.raises(ValueError, match="stable_unit_id must equal"):
        ripple_modulation.validate_ripple_modulation_tables(
            summary,
            wrong_identity,
        )

    incomplete = peri.iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete or shifted time grid"):
        ripple_modulation.validate_ripple_modulation_tables(summary, incomplete)


def test_artifact_paths_are_uuid_keyed_and_do_not_create_directories(
    tmp_path: Path,
) -> None:
    paths = ripple_modulation.get_ripple_modulation_artifact_paths(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_modulation_id=RIPPLE_MODULATION_ID,
        artifact_root=tmp_path,
    )

    expected_directory = (
        tmp_path
        / "RatA"
        / "20240101"
        / "ripple_modulation"
        / "02_r1"
        / "v1"
        / str(RIPPLE_MODULATION_ID)
    )
    assert paths["directory"] == expected_directory
    assert paths["summary"].parent == expected_directory
    assert paths["peri_ripple_firing_rate"].parent == expected_directory
    assert paths["summary"].name == "summary.parquet"
    assert (
        paths["peri_ripple_firing_rate"].name
        == "peri_ripple_firing_rate.parquet"
    )
    assert not expected_directory.exists()

    with pytest.raises(ValueError, match="UUID"):
        ripple_modulation.get_ripple_modulation_artifact_paths(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            region="v1",
            ripple_modulation_id="not-a-uuid",
            artifact_root=tmp_path,
        )


def test_register_existing_plan_has_no_database_writes_and_copy_is_explicit(
    tmp_path: Path,
) -> None:
    existing_dir = tmp_path / "legacy"
    existing_dir.mkdir()
    source_paths = _legacy_artifact_paths(existing_dir)
    summary_source = source_paths["summary"]
    peri_source = source_paths["peri_ripple_firing_rate"]
    _write_legacy_artifacts(source_paths)

    plan = ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_modulation_id=RIPPLE_MODULATION_ID,
        existing_summary_path=summary_source,
        existing_peri_ripple_firing_rate_path=peri_source,
        artifact_root=tmp_path / "native",
    )

    assert plan["operation"] == "register_existing"
    assert plan["database_operations"] == []
    assert all(copy["copy_required"] for copy in plan["copies"])
    assert not plan["artifact_paths"]["summary"].exists()
    assert plan["artifact_paths"]["summary"] == (
        tmp_path
        / "native"
        / "RatA"
        / "20240101"
        / "ripple_modulation"
        / "02_r1"
        / "v1"
        / str(RIPPLE_MODULATION_ID)
        / "summary.parquet"
    )
    assert set(plan["accepted_source_names"]["summary"]) == {
        _legacy_artifact_paths(existing_dir, region_label="v1")["summary"].name,
        source_paths["summary"].name,
    }

    copied = ripple_modulation.copy_planned_artifacts(plan)

    copied_summary = pd.read_parquet(copied["summary"])
    copied_peri = pd.read_parquet(copied["peri_ripple_firing_rate"])
    assert copied_summary["region"].tolist() == ["v1"]
    assert set(copied_peri["region"]) == {"v1"}
    assert len(copied_peri) == 50
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ripple_modulation.copy_planned_artifacts(plan)


@pytest.mark.parametrize(
    ("artifact_name", "missing_column"),
    [
        ("summary", "response_zscore"),
        ("peri_ripple_firing_rate", "mean_rate_hz"),
    ],
)
def test_register_existing_requires_full_canonical_artifact_schemas(
    tmp_path: Path,
    artifact_name: str,
    missing_column: str,
) -> None:
    paths = _legacy_artifact_paths(tmp_path / "legacy")
    summary = _legacy_summary_table()
    peri = _legacy_peri_table()
    if artifact_name == "summary":
        summary = summary.drop(columns=missing_column)
    else:
        peri = peri.drop(columns=missing_column)
    _write_legacy_artifacts(paths, summary=summary, peri=peri)
    plan = _registration_plan(paths, artifact_root=tmp_path / "native")

    with pytest.raises(ValueError):
        ripple_modulation.read_planned_artifacts(plan)


@pytest.mark.parametrize("grid_error", ["incomplete", "missing", "shifted"])
def test_register_existing_rejects_invalid_per_unit_time_grid(
    tmp_path: Path,
    grid_error: str,
) -> None:
    paths = _legacy_artifact_paths(tmp_path / "legacy")
    peri = _legacy_peri_table()
    v1_indices = peri.index[peri["region"] == "v1"]
    if grid_error == "incomplete":
        peri = peri.drop(index=v1_indices[-1])
    elif grid_error == "missing":
        peri.loc[v1_indices[len(v1_indices) // 2], "time_s"] = np.nan
    else:
        peri.loc[v1_indices, "time_s"] += (
            ripple_plot.DEFAULT_BIN_SIZE_S / 2.0
        )
    _write_legacy_artifacts(paths, peri=peri)
    plan = _registration_plan(paths, artifact_root=tmp_path / "native")

    with pytest.raises(ValueError):
        ripple_modulation.read_planned_artifacts(plan)


def test_empty_legacy_tables_require_explicit_allow_empty(tmp_path: Path) -> None:
    paths = _legacy_artifact_paths(tmp_path / "legacy")
    empty_summary = pd.DataFrame(columns=ripple_plot.SUMMARY_COLUMNS)
    empty_peri = pd.DataFrame(
        columns=ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS
    )
    _write_legacy_artifacts(
        paths,
        summary=empty_summary,
        peri=empty_peri,
    )
    plan = _registration_plan(paths, artifact_root=tmp_path / "native")

    with pytest.raises(ValueError, match="no rows"):
        ripple_modulation.read_planned_artifacts(plan)

    loaded = ripple_modulation.read_planned_artifacts(
        plan,
        allow_empty=True,
    )

    assert loaded["summary"].empty
    assert loaded["peri_ripple_firing_rate"].empty
    assert list(loaded["summary"].columns) == ripple_plot.SUMMARY_COLUMNS
    assert (
        list(loaded["peri_ripple_firing_rate"].columns)
        == ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS
    )


def test_register_existing_rejects_artifact_without_matching_key(tmp_path: Path) -> None:
    paths = _legacy_artifact_paths(tmp_path)
    _write_legacy_artifacts(
        paths,
        summary=_legacy_summary_table(regions=("ca1",)),
        peri=_legacy_peri_table(regions=("ca1",)),
    )
    plan = ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_modulation_id=RIPPLE_MODULATION_ID,
        existing_summary_path=paths["summary"],
        existing_peri_ripple_firing_rate_path=paths[
            "peri_ripple_firing_rate"
        ],
        artifact_root=tmp_path / "native",
    )

    with pytest.raises(ValueError, match="has no rows for key"):
        ripple_modulation.copy_planned_artifacts(plan)


def test_register_existing_rejects_thresholded_legacy_name(tmp_path: Path) -> None:
    thresholded_paths = _legacy_artifact_paths(
        tmp_path,
        ripple_threshold_zscore=2.0,
    )
    _write_legacy_artifacts(thresholded_paths)
    plan = ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_modulation_id=RIPPLE_MODULATION_ID,
        existing_summary_path=thresholded_paths["summary"],
        existing_peri_ripple_firing_rate_path=thresholded_paths[
            "peri_ripple_firing_rate"
        ],
        artifact_root=tmp_path / "native",
    )

    with pytest.raises(ValueError, match="does not match"):
        ripple_modulation.copy_planned_artifacts(plan)


def test_atomic_pair_write_leaves_no_partial_destination(tmp_path: Path) -> None:
    destinations = {
        "summary": tmp_path / "summary.parquet",
        "peri": tmp_path / "peri.parquet",
    }

    def write_summary(path: Path) -> None:
        path.write_bytes(b"summary")

    def fail_peri(_path: Path) -> None:
        raise RuntimeError("failed second writer")

    with pytest.raises(RuntimeError, match="failed second writer"):
        ripple_modulation._write_parquets_atomically(
            {
                "summary": (destinations["summary"], write_summary),
                "peri": (destinations["peri"], fail_peri),
            }
        )

    assert not destinations["summary"].exists()
    assert not destinations["peri"].exists()
    assert list(tmp_path.iterdir()) == []
