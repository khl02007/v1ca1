"""Tests for database-free ripple-only RippleCrossRegionXCorr artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass import ripple_cross_region_xcorr as module


RESULT_ID = uuid.UUID("aeaa376a-b5c2-5d61-a449-2e046cf9abf0")


class _Spike:
    """Minimal spike-time object exposing the pynapple-compatible ``t`` field."""

    def __init__(self, times: np.ndarray):
        self.t = np.asarray(times, dtype=float)


def _spikes() -> tuple[dict[str, _Spike], dict[str, _Spike]]:
    """Return two regions with one passing and one excluded runtime unit each."""
    ca1 = {
        "ca1-runtime-a": _Spike(np.linspace(0.101, 0.199, 31)),
        "ca1-runtime-b": _Spike(np.linspace(0.11, 0.19, 5)),
    }
    v1 = {
        "v1-runtime-a": _Spike(np.linspace(0.102, 0.198, 32)),
        "v1-runtime-b": _Spike(np.linspace(0.12, 0.18, 4)),
    }
    return ca1, v1


def _identities() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return persistent identities aligned to the two runtime mappings."""
    return (
        [
            {"spikesorting_merge_id": "merge-ca1", "unit_id": "11"},
            {"spikesorting_merge_id": "merge-ca1", "unit_id": "12"},
        ],
        [
            {"spikesorting_merge_id": "merge-v1", "unit_id": "21"},
            {"spikesorting_merge_id": "merge-v1", "unit_id": "22"},
        ],
    )


def _ripples() -> pd.DataFrame:
    """Return two exact detector-qualified ripple intervals in one epoch."""
    return pd.DataFrame(
        {
            "epoch": ["02_r1", "02_r1", "04_r2"],
            "start_time": [0.1, 0.4, 10.0],
            "end_time": [0.2, 0.5, 10.1],
        }
    )


def _provenance() -> dict[str, object]:
    """Return selected-RippleIntervals provenance with the required detector policy."""
    return {
        "ripple_interval_list_name": "02_r1_ripples",
        "detector_zscore_threshold": 2.0,
        "speed_gated": True,
        "source_nwb_file_name": "L14_20240611_.nwb",
    }


def _patch_science(
    monkeypatch: pytest.MonkeyPatch,
    *,
    curve: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Patch only pynapple construction/computation while retaining pure helpers."""
    from v1ca1.xcorr import screen_xcorr

    calls: list[dict[str, object]] = []

    def build_intervalset(intervals_by_epoch, selected_epochs):
        assert len(selected_epochs) == 1
        rows = intervals_by_epoch[selected_epochs[0]]
        return SimpleNamespace(
            start=rows["start"].to_numpy(dtype=float),
            end=rows["end"].to_numpy(dtype=float),
        )

    def subset(spikes, unit_ids):
        return {unit_id: spikes[unit_id] for unit_id in unit_ids.tolist()}

    def compute_xcorr(**kwargs):
        calls.append(kwargs)
        ca1_ids = list(kwargs["ca1_spikes"])
        v1_ids = list(kwargs["v1_spikes"])
        lags = module._expected_lag_times(
            module.validate_ripple_cross_region_xcorr_parameters()
        )
        values = (
            np.linspace(0.5, 2.5, len(lags))
            if curve is None
            else np.asarray(curve, dtype=float)
        )
        columns = pd.MultiIndex.from_tuples(
            [(ca1_id, v1_id) for ca1_id in ca1_ids for v1_id in v1_ids]
        )
        return pd.DataFrame(
            np.tile(values[:, None], (1, len(columns))),
            index=lags,
            columns=columns,
        )

    monkeypatch.setattr(screen_xcorr, "build_state_intervalset", build_intervalset)
    monkeypatch.setattr(screen_xcorr, "subset_spikes_by_unit_ids", subset)
    monkeypatch.setattr(screen_xcorr, "compute_xcorr", compute_xcorr)
    return calls


def _compute(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return one complete result through the public compute entry point."""
    _patch_science(monkeypatch)
    ca1_spikes, v1_spikes = _spikes()
    ca1_ids, v1_ids = _identities()
    return module.compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_table=_ripples(),
        ca1_spikes=ca1_spikes,
        ca1_stable_unit_ids=ca1_ids,
        v1_spikes=v1_spikes,
        v1_stable_unit_ids=v1_ids,
        upstream_provenance=_provenance(),
    )


def test_parameters_paths_and_legacy_paths_are_fixed(tmp_path: Path) -> None:
    parameters = module.validate_ripple_cross_region_xcorr_parameters()
    assert parameters == {
        "bin_size_s": 0.005,
        "max_lag_s": 0.5,
        "min_ripple_spikes": 30,
        "extremum_half_width_bins": 1,
        "norm": True,
        "expected_detector_zscore_threshold": 2.0,
        "require_speed_gated": True,
    }
    with pytest.raises(ValueError, match="fixed value"):
        module.validate_ripple_cross_region_xcorr_parameters(bin_size_s=0.01)
    with pytest.raises(ValueError, match="norm=True"):
        module.validate_ripple_cross_region_xcorr_parameters(norm=False)
    with pytest.raises(ValueError, match="speed-gated"):
        module.validate_ripple_cross_region_xcorr_parameters(require_speed_gated=False)
    assert module.validate_ripple_cross_region_xcorr_parameters(
        norm=np.int64(1), require_speed_gated=np.bool_(True)
    )["require_speed_gated"] is True
    with pytest.raises(TypeError, match="database integer 0/1"):
        module.validate_ripple_cross_region_xcorr_parameters(norm="yes")
    paths = module.get_ripple_cross_region_xcorr_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_cross_region_xcorr_id=RESULT_ID,
        artifact_root=tmp_path,
    )
    assert paths["artifact_dir"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple_cross_region_xcorr"
        / "02_r1"
        / str(RESULT_ID)
    )
    legacy = module.get_legacy_ripple_cross_region_xcorr_paths(
        tmp_path / "L14" / "20240611", epoch="02_r1"
    )
    assert legacy["result_path"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "xcorr"
        / "screen_pairs"
        / "ripple"
        / "02_r1"
        / "xcorr.nc"
    )


def test_event_selection_helper_freezes_one_exact_epoch() -> None:
    selection = module.prepare_ripple_cross_region_xcorr_event_selection(
        epoch="02_r1", ripple_table=_ripples().iloc[::-1]
    )
    assert tuple(selection) == (
        "selected_ripple_table",
        "ripple_start_time_s",
        "ripple_end_time_s",
        "n_ripples",
        "ripple_duration_s",
        "selected_ripple_intervals_sha256",
    )
    assert selection["selected_ripple_table"].to_dict("list") == {
        "start_time": [0.1, 0.4],
        "end_time": [0.2, 0.5],
    }
    assert selection["n_ripples"] == 2
    assert selection["ripple_duration_s"] == pytest.approx(0.2)
    assert len(selection["selected_ripple_intervals_sha256"]) == 64


def test_compute_uses_exact_ripples_stable_units_and_ca1_to_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_science(monkeypatch)
    ca1_spikes, v1_spikes = _spikes()
    ca1_ids, v1_ids = _identities()
    selection = module.prepare_ripple_cross_region_xcorr_event_selection(
        epoch="02_r1", ripple_table=_ripples()
    )
    result = module.compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_table=_ripples(),
        ca1_spikes=ca1_spikes,
        ca1_stable_unit_ids=ca1_ids,
        v1_spikes=v1_spikes,
        v1_stable_unit_ids=v1_ids,
        upstream_provenance=_provenance(),
        expected_selected_ripple_intervals_sha256=selection[
            "selected_ripple_intervals_sha256"
        ],
    )
    assert result["analysis_status"] == "valid"
    assert result["artifact_origin"] == "computed"
    assert result["n_ripples"] == 2
    assert result["ripple_duration_s"] == pytest.approx(0.2)
    assert result["n_ca1_units"] == 2
    assert result["n_v1_units"] == 2
    assert result["n_ca1_units_in_xcorr"] == 1
    assert result["n_v1_units_in_xcorr"] == 1
    assert result["n_pairs"] == result["n_valid_pairs"] == 1
    assert len(calls) == 1
    assert list(calls[0]["ca1_spikes"]) == ["ca1-runtime-a"]
    assert list(calls[0]["v1_spikes"]) == ["v1-runtime-a"]
    assert calls[0]["bin_size_s"] == 0.005
    assert calls[0]["max_lag_s"] == 0.5
    np.testing.assert_array_equal(calls[0]["intervals"].start, [0.1, 0.4])
    np.testing.assert_array_equal(calls[0]["intervals"].end, [0.2, 0.5])
    assert result["ca1_units"]["ripple_spike_count"].tolist() == [31, 5]
    assert result["v1_units"]["ripple_spike_count"].tolist() == [32, 4]
    assert result["ca1_units"]["unit_qc_status"].tolist() == [
        "valid",
        "excluded_spike_threshold",
    ]
    assert result["summary"].loc[0, "ca1_stable_unit_id"] == "merge-ca1:11"
    assert result["summary"].loc[0, "v1_stable_unit_id"] == "merge-v1:21"
    assert result["summary"].loc[0, "ca1_group_unit_id"] == "ca1-runtime-a"
    assert result["dataset"].coords["ca1_unit"].item() == "merge-ca1:11"
    assert result["dataset"].coords["v1_unit"].item() == "merge-v1:21"
    np.testing.assert_array_equal(
        result["dataset"]["ripple_start_time_s"].values, [0.1, 0.4]
    )


def test_compute_rejects_wrong_ripple_provenance_and_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_science(monkeypatch)
    ca1_spikes, v1_spikes = _spikes()
    ca1_ids, v1_ids = _identities()
    kwargs = {
        "ripple_cross_region_xcorr_id": RESULT_ID,
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "02_r1",
        "ripple_table": _ripples(),
        "ca1_spikes": ca1_spikes,
        "ca1_stable_unit_ids": ca1_ids,
        "v1_spikes": v1_spikes,
        "v1_stable_unit_ids": v1_ids,
        "upstream_provenance": _provenance(),
    }
    with pytest.raises(ValueError, match="threshold"):
        module.compute_ripple_cross_region_xcorr(
            **{
                **kwargs,
                "upstream_provenance": {
                    **_provenance(),
                    "detector_zscore_threshold": 1.5,
                },
            }
        )
    with pytest.raises(ValueError, match="speed_gated=True"):
        module.compute_ripple_cross_region_xcorr(
            **{
                **kwargs,
                "upstream_provenance": {**_provenance(), "speed_gated": False},
            }
        )
    with pytest.raises(TypeError, match="database integer 0/1"):
        module.compute_ripple_cross_region_xcorr(
            **{
                **kwargs,
                "upstream_provenance": {**_provenance(), "speed_gated": "yes"},
            }
        )
    normalized = module.compute_ripple_cross_region_xcorr(
        **{
            **kwargs,
            "upstream_provenance": {
                **_provenance(),
                "detector_zscore_threshold": np.float64(2.0),
                "speed_gated": np.int64(1),
            },
        }
    )
    assert normalized["upstream_provenance"]["speed_gated"] is True
    with pytest.raises(ValueError, match="expected SHA-256 digest"):
        module.compute_ripple_cross_region_xcorr(
            **kwargs,
            expected_selected_ripple_intervals_sha256="0" * 64,
        )
    overlaps = pd.DataFrame(
        {
            "epoch": ["02_r1", "02_r1"],
            "start": [0.1, 0.15],
            "end": [0.2, 0.25],
        }
    )
    with pytest.raises(ValueError, match="must not overlap"):
        module.compute_ripple_cross_region_xcorr(**{**kwargs, "ripple_table": overlaps})


def test_terminal_no_ripples_writes_explicit_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_science(monkeypatch)
    ca1_spikes, v1_spikes = _spikes()
    ca1_ids, v1_ids = _identities()
    result = module.compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        epoch="missing_epoch",
        ripple_table=_ripples(),
        ca1_spikes=ca1_spikes,
        ca1_stable_unit_ids=ca1_ids,
        v1_spikes=v1_spikes,
        v1_stable_unit_ids=v1_ids,
        upstream_provenance=_provenance(),
    )
    assert not calls
    assert result["analysis_status"] == "no_ripples"
    assert result["summary"].empty
    assert result["dataset"].sizes["ripple"] == 0
    assert result["dataset"].sizes["ca1_unit"] == 0
    assert result["dataset"].sizes["v1_unit"] == 0
    assert result["ca1_units"]["unit_qc_status"].eq(
        "excluded_spike_threshold"
    ).all()
    tampered_units = result["ca1_units"].copy()
    tampered_units.loc[0, "included_in_xcorr"] = True
    with pytest.raises(ValueError, match="excluded units cannot be included"):
        module.validate_ripple_cross_region_xcorr_result(
            {**result, "ca1_units": tampered_units}
        )


def test_terminal_missing_region_preserves_other_region_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_science(monkeypatch)
    _, v1_spikes = _spikes()
    _, v1_ids = _identities()
    result = module.compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_table=_ripples(),
        ca1_spikes={},
        ca1_stable_unit_ids=[],
        v1_spikes=v1_spikes,
        v1_stable_unit_ids=v1_ids,
        upstream_provenance=_provenance(),
    )
    assert not calls
    assert result["analysis_status"] == "no_ca1_units"
    assert result["ca1_units"].empty
    assert result["v1_units"]["unit_qc_status"].tolist() == [
        "not_computed",
        "excluded_spike_threshold",
    ]
    assert not result["v1_units"]["included_in_xcorr"].any()


def test_validation_rejects_tensor_summary_and_qc_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)
    broken_dataset = result["dataset"].copy(deep=True)
    broken_dataset["xcorr"].values[0, 0, 0] = 1e6
    with pytest.raises(ValueError, match="summary differs"):
        module.validate_ripple_cross_region_xcorr_result(
            {**result, "dataset": broken_dataset}
        )
    broken_units = result["ca1_units"].copy()
    broken_units.loc[0, "ripple_spike_count"] = 1
    with pytest.raises(ValueError, match="threshold flags"):
        module.validate_ripple_cross_region_xcorr_result(
            {**result, "ca1_units": broken_units}
        )
    broken_boolean = result["ca1_units"].copy()
    broken_boolean["included_in_xcorr"] = broken_boolean[
        "included_in_xcorr"
    ].astype(object)
    broken_boolean.loc[0, "included_in_xcorr"] = "yes"
    with pytest.raises(TypeError, match="database integer 0/1"):
        module.validate_ripple_cross_region_xcorr_result(
            {**result, "ca1_units": broken_boolean}
        )
    broken_attrs = result["dataset"].copy(deep=True)
    broken_attrs.attrs["interval_scope"] = "pooled_state"
    with pytest.raises(ValueError, match="interval_scope"):
        module.validate_ripple_cross_region_xcorr_result(
            {**result, "dataset": broken_attrs}
        )


def test_atomic_roundtrip_checksums_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)
    paths = module.get_ripple_cross_region_xcorr_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_cross_region_xcorr_id=RESULT_ID,
        artifact_root=tmp_path,
    )
    written = module.write_ripple_cross_region_xcorr_artifact(
        result, paths["artifact_dir"]
    )
    assert all(path.is_file() for name, path in written.items() if name != "artifact_dir")
    loaded = module.load_ripple_cross_region_xcorr_artifact(paths["artifact_dir"])
    assert loaded["analysis_status"] == "valid"
    assert loaded["n_valid_pairs"] == 1
    with pytest.raises(FileExistsError):
        module.write_ripple_cross_region_xcorr_artifact(result, paths["artifact_dir"])
    with written["summary_path"].open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_ripple_cross_region_xcorr_artifact(paths["artifact_dir"])


def _write_legacy_artifacts(
    result: dict[str, object],
    path: Path,
) -> dict[str, Path]:
    """Write four legacy artifacts whose sorting ids differ from runtime keys."""
    import xarray as xr

    path.mkdir(parents=True)
    legacy_ids = {
        "merge-ca1:11": 9001,
        "merge-ca1:12": 9002,
        "merge-v1:21": 8001,
        "merge-v1:22": 8002,
    }
    for region, filename in (
        ("ca1", "ca1_unit_filter.parquet"),
        ("v1", "v1_unit_filter.parquet"),
    ):
        audit = result[f"{region}_units"]
        table = pd.DataFrame(
            {
                "region": region,
                "unit_id": audit["stable_unit_id"].map(legacy_ids).to_numpy(),
                "state_spike_count": audit["ripple_spike_count"].to_numpy(),
                "passes_state_spike_count": audit[
                    "passes_ripple_spike_threshold"
                ].to_numpy(),
                "keep_unit": audit["passes_ripple_spike_threshold"].to_numpy(),
            }
        )
        table.to_parquet(path / filename, index=False)
    summary = result["summary"]
    legacy_summary = pd.DataFrame(
        {
            "ca1_unit_id": summary["ca1_stable_unit_id"].map(legacy_ids),
            "v1_unit_id": summary["v1_stable_unit_id"].map(legacy_ids),
            "n_ca1_state_spikes": summary["n_ca1_ripple_spikes"],
            "n_v1_state_spikes": summary["n_v1_ripple_spikes"],
            "peak_lag_s": summary["peak_lag_s"],
            "peak_norm_xcorr": summary["peak_norm_xcorr"],
            "status": summary["status"],
        }
    )
    legacy_summary.to_parquet(path / "xcorr_summary.parquet", index=False)
    source = result["dataset"]
    legacy_dataset = xr.Dataset(
        data_vars={
            "xcorr": (
                ("ca1_unit", "v1_unit", "lag_s"),
                np.asarray(source["xcorr"].values, dtype=np.float32),
            )
        },
        coords={
            "ca1_unit": [legacy_ids[value] for value in source.ca1_unit.values],
            "v1_unit": [legacy_ids[value] for value in source.v1_unit.values],
            "lag_s": np.asarray(source.lag_s.values, dtype=np.float32),
        },
        attrs={
            "animal_name": "L14",
            "date": "20240611",
            "state": "ripple",
            "epoch_group_label": "02_r1",
            "bin_size_s": 0.005,
            "max_lag_s": 0.5,
            "min_state_spikes": 30,
            "extremum_half_width_bins": 1,
            "selected_epochs_json": '["02_r1"]',
            "state_interval_source": "parquet",
        },
    )
    legacy_dataset.to_netcdf(path / "xcorr.nc")
    return {
        "source_ca1_unit_filter_path": path / "ca1_unit_filter.parquet",
        "source_v1_unit_filter_path": path / "v1_unit_filter.parquet",
        "source_summary_path": path / "xcorr_summary.parquet",
        "source_result_path": path / "xcorr.nc",
    }


def test_legacy_registration_resolves_sorting_ids_and_compares_all_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)
    legacy_paths = _write_legacy_artifacts(result, tmp_path / "legacy")
    ca1_spikes, v1_spikes = _spikes()
    ca1_ids, v1_ids = _identities()
    ca1_map = {9001: ca1_ids[0], 9002: ca1_ids[1]}
    v1_map = {8001: v1_ids[0], 8002: v1_ids[1]}
    destination = (
        tmp_path / "registered" / str(RESULT_ID)
    )
    registered = module.register_existing_ripple_cross_region_xcorr_artifact(
        **legacy_paths,
        destination_path=destination,
        ripple_cross_region_xcorr_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        ripple_table=_ripples(),
        ca1_spikes=ca1_spikes,
        ca1_stable_unit_ids=ca1_ids,
        v1_spikes=v1_spikes,
        v1_stable_unit_ids=v1_ids,
        upstream_provenance=_provenance(),
        ca1_legacy_identity_resolver=lambda ids: [ca1_map[value] for value in ids],
        v1_legacy_identity_resolver=lambda ids: [v1_map[value] for value in ids],
        ca1_sorting_type="ImportedSpikeSorting",
        v1_sorting_type="ImportedSpikeSorting",
        expected_selected_ripple_intervals_sha256=(
            result["selected_ripple_intervals_sha256"]
        ),
        source_v1ca1_git_commit="v1-source",
        source_spyglass_git_commit="sg-source",
    )
    assert registered["artifact_origin"] == "registered_existing"
    provenance = registered["legacy_artifact_provenance"]
    assert set(provenance["compared_artifacts"]) == {
        "ca1_unit_filter",
        "v1_unit_filter",
        "summary",
        "result",
    }
    assert provenance["source_v1ca1_git_commit"] == "v1-source"
    assert provenance["source_spyglass_git_commit"] == "sg-source"
    assert provenance["ca1_sorting_type"] == "ImportedSpikeSorting"
    assert provenance["v1_sorting_type"] == "ImportedSpikeSorting"
    assert destination.is_dir()
    loaded = module.load_ripple_cross_region_xcorr_artifact(destination)
    assert loaded["legacy_artifact_provenance"] == provenance


def test_legacy_registration_rejects_any_scientific_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)
    legacy_paths = _write_legacy_artifacts(result, tmp_path / "legacy")
    ca1_spikes, v1_spikes = _spikes()
    ca1_ids, v1_ids = _identities()
    ca1_map = {9001: ca1_ids[0], 9002: ca1_ids[1]}
    v1_map = {8001: v1_ids[0], 8002: v1_ids[1]}
    common = {
        **legacy_paths,
        "destination_path": tmp_path / "registered" / str(RESULT_ID),
        "ripple_cross_region_xcorr_id": RESULT_ID,
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "02_r1",
        "ripple_table": _ripples(),
        "ca1_spikes": ca1_spikes,
        "ca1_stable_unit_ids": ca1_ids,
        "v1_spikes": v1_spikes,
        "v1_stable_unit_ids": v1_ids,
        "upstream_provenance": _provenance(),
        "ca1_legacy_identity_resolver": lambda ids: [ca1_map[value] for value in ids],
        "v1_legacy_identity_resolver": lambda ids: [v1_map[value] for value in ids],
        "ca1_sorting_type": "ImportedSpikeSorting",
        "v1_sorting_type": "ImportedSpikeSorting",
    }
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        module.register_existing_ripple_cross_region_xcorr_artifact(
            **{**common, "ca1_sorting_type": "CurationV1"}
        )
    with pytest.raises(ValueError, match="expected SHA-256 digest"):
        module.register_existing_ripple_cross_region_xcorr_artifact(
            **common,
            expected_selected_ripple_intervals_sha256="0" * 64,
        )

    import xarray as xr

    with xr.open_dataset(legacy_paths["source_result_path"]) as source_dataset:
        fixed_window_dataset = source_dataset.load()
    fixed_window_dataset.attrs["state_interval_source"] = (
        "fixed_ripple_windows[parquet]"
    )
    fixed_window_dataset.to_netcdf(legacy_paths["source_result_path"])
    with pytest.raises(ValueError, match="exact detected ripple intervals"):
        module.register_existing_ripple_cross_region_xcorr_artifact(**common)
    fixed_window_dataset.attrs["state_interval_source"] = "parquet"
    fixed_window_dataset.to_netcdf(legacy_paths["source_result_path"])

    summary = pd.read_parquet(legacy_paths["source_summary_path"])
    summary.loc[0, "peak_norm_xcorr"] += 0.1
    summary.to_parquet(legacy_paths["source_summary_path"], index=False)
    with pytest.raises(ValueError, match="pair summary"):
        module.register_existing_ripple_cross_region_xcorr_artifact(
            **common,
        )
