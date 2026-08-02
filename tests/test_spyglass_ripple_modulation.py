from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from v1ca1.ripple import plot_ripple_modulation as ripple_plot
from v1ca1.spyglass import ripple_modulation


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
    return ripple_modulation.compute_epoch_region_ripple_modulation(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        ripple_table=_ripple_table(),
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


def test_compute_one_epoch_region_uses_all_detector_events_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)

    assert result["epoch"] == "02_r1"
    assert result["region"] == "v1"
    assert result["n_ripples"] == 2
    assert result["minimum_ripple_mean_zscore"] is None
    assert result["selected_ripple_table"]["epoch"].unique().tolist() == ["02_r1"]
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


def test_optional_minimum_event_mean_zscore_is_the_only_extra_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch, minimum_ripple_mean_zscore=4.0)

    assert result["n_ripples"] == 1
    assert np.allclose(result["selected_ripple_table"]["start_time"], [0.75])
    assert set(result["summary"]["n_ripples"]) == {1}


def test_compute_writes_stable_merge_and_nwb_unit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(
        monkeypatch,
        stable_unit_ids=[
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
            {"spikesorting_merge_id": "merge-b", "unit_id": 12},
        ],
    )

    expected_ids = ["merge-a:11", "merge-b:12"]
    assert result["summary"]["unit_id"].tolist() == expected_ids
    assert result["summary"]["spikesorting_merge_id"].tolist() == [
        "merge-a",
        "merge-b",
    ]
    assert result["summary"]["nwb_unit_id"].tolist() == ["11", "12"]
    assert set(result["peri_ripple_firing_rate"]["unit_id"]) == set(expected_ids)
    assert result["heatmap_payload"]["unit_ids"].tolist() == expected_ids


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


def test_compute_preserves_empty_parquet_schemas_after_filtering_all_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch, minimum_ripple_mean_zscore=10.0)

    assert result["n_ripples"] == 0
    assert result["summary"].empty
    assert result["peri_ripple_firing_rate"].empty
    assert list(result["summary"].columns) == ripple_plot.SUMMARY_COLUMNS
    assert list(result["peri_ripple_firing_rate"].columns) == (
        ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS
    )


def test_artifact_paths_are_session_first_and_do_not_create_directories(
    tmp_path: Path,
) -> None:
    paths = ripple_modulation.get_ripple_modulation_artifact_paths(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        artifact_root=tmp_path,
    )

    expected_directory = (
        tmp_path / "RatA" / "20240101" / "ripple_modulation" / "02_r1" / "v1"
    )
    assert paths["directory"] == expected_directory
    assert paths["summary"].parent == expected_directory
    assert paths["peri_ripple_firing_rate"].parent == expected_directory
    assert paths["summary"].name.endswith("_summary.parquet")
    assert "mean_zscore_all_detected" in paths["summary"].name
    assert not expected_directory.exists()


def test_register_existing_plan_has_no_database_writes_and_copy_is_explicit(
    tmp_path: Path,
) -> None:
    existing_dir = tmp_path / "legacy"
    existing_dir.mkdir()
    source_names = ripple_modulation.get_ripple_modulation_artifact_paths(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="all_regions",
        artifact_root=existing_dir,
    )
    summary_source = existing_dir / source_names["summary"].name
    peri_source = existing_dir / source_names["peri_ripple_firing_rate"].name
    legacy_rows = {
        "animal_name": ["RatA", "RatA"],
        "date": ["20240101", "20240101"],
        "epoch": ["02_r1", "02_r1"],
        "region": ["v1", "ca1"],
        "bin_size_s": [0.02, 0.02],
        "time_before_s": [0.5, 0.5],
        "time_after_s": [0.5, 0.5],
    }
    pd.DataFrame({**legacy_rows, "response_zscore": [1.0, 2.0]}).to_parquet(
        summary_source,
        index=False,
    )
    pd.DataFrame({**legacy_rows, "time_s": [0.0, 0.0]}).to_parquet(
        peri_source,
        index=False,
    )

    plan = ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        existing_summary_path=summary_source,
        existing_peri_ripple_firing_rate_path=peri_source,
        artifact_root=tmp_path / "native",
    )

    assert plan["operation"] == "register_existing"
    assert plan["database_operations"] == []
    assert all(copy["copy_required"] for copy in plan["copies"])
    assert not plan["artifact_paths"]["summary"].exists()

    copied = ripple_modulation.copy_planned_artifacts(plan)

    copied_summary = pd.read_parquet(copied["summary"])
    copied_peri = pd.read_parquet(copied["peri_ripple_firing_rate"])
    assert copied_summary["region"].tolist() == ["v1"]
    assert copied_peri["region"].tolist() == ["v1"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ripple_modulation.copy_planned_artifacts(plan)


def test_register_existing_rejects_artifact_without_matching_key(tmp_path: Path) -> None:
    source_name = ripple_modulation.get_ripple_modulation_artifact_paths(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="all_regions",
        artifact_root=tmp_path,
    )["summary"].name
    source = tmp_path / source_name
    pd.DataFrame(
        {
            "animal_name": ["RatA"],
            "date": ["20240101"],
            "epoch": ["02_r1"],
            "region": ["ca1"],
            "bin_size_s": [0.02],
            "time_before_s": [0.5],
            "time_after_s": [0.5],
        }
    ).to_parquet(source, index=False)
    plan = ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
        existing_summary_path=source,
        existing_peri_ripple_firing_rate_path=source,
        artifact_root=tmp_path / "native",
    )

    with pytest.raises(ValueError, match="has no rows for key"):
        ripple_modulation.copy_planned_artifacts(plan)


def test_register_existing_rejects_mismatched_threshold_stem(tmp_path: Path) -> None:
    thresholded_paths = ripple_modulation.get_ripple_modulation_artifact_paths(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="all_regions",
        artifact_root=tmp_path,
        minimum_ripple_mean_zscore=2.0,
    )
    table = pd.DataFrame(
        {
            "animal_name": ["RatA"],
            "date": ["20240101"],
            "epoch": ["02_r1"],
            "region": ["v1"],
            "bin_size_s": [0.02],
            "time_before_s": [0.5],
            "time_after_s": [0.5],
        }
    )
    for artifact in ripple_modulation.ARTIFACT_NAMES:
        path = thresholded_paths[artifact]
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    plan = ripple_modulation.plan_register_existing(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        region="v1",
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
