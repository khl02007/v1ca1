from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.nwb.replace_epoch_bounds import DEFAULT_OUTPUT_SUFFIX, replace_epoch_bounds


def _write_test_nwb(nwb_path: Path, epoch_tags: list[str], epoch_bounds: list[tuple[float, float]]) -> None:
    pynwb = pytest.importorskip("pynwb")

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    for epoch_tag, (start_time, stop_time) in zip(epoch_tags, epoch_bounds, strict=True):
        nwbfile.add_epoch(
            start_time=float(start_time),
            stop_time=float(stop_time),
            tags=[str(epoch_tag)],
        )

    with pynwb.NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)


def _write_timestamps_ephys_npz(
    analysis_path: Path,
    epoch_tags: list[str],
    epoch_bounds: list[tuple[float, float]],
) -> None:
    nap = pytest.importorskip("pynapple")

    starts = np.asarray([bounds[0] for bounds in epoch_bounds], dtype=float)
    stops = np.asarray([bounds[1] for bounds in epoch_bounds], dtype=float)
    intervals = nap.IntervalSet(start=starts, end=stops, time_units="s")
    intervals.set_info(epoch=list(epoch_tags))
    intervals.save(analysis_path / "timestamps_ephys.npz")


def _read_epoch_bounds(nwb_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    pynwb = pytest.importorskip("pynwb")

    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        epoch_tags, start_times, stop_times = extract_epoch_metadata(nwbfile)
    return epoch_tags, start_times, stop_times


def test_replace_epoch_bounds_writes_new_nwb_file(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    source_epoch_tags = ["01_s1", "02_r1"]
    source_epoch_bounds = [(0.0, 1.0), (2.0, 3.0)]
    replacement_epoch_bounds = [(0.25, 0.95), (2.1, 2.85)]
    _write_test_nwb(nwb_path, source_epoch_tags, source_epoch_bounds)
    _write_timestamps_ephys_npz(analysis_path, source_epoch_tags, replacement_epoch_bounds)

    output_path = replace_epoch_bounds(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
    )

    assert output_path == nwb_root / f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}"
    assert output_path.exists()

    source_tags, source_starts, source_stops = _read_epoch_bounds(nwb_path)
    output_tags, output_starts, output_stops = _read_epoch_bounds(output_path)

    assert source_tags == source_epoch_tags
    assert np.allclose(source_starts, [0.0, 2.0])
    assert np.allclose(source_stops, [1.0, 3.0])
    assert output_tags == source_epoch_tags
    assert np.allclose(output_starts, [0.25, 2.1])
    assert np.allclose(output_stops, [0.95, 2.85])


def test_replace_epoch_bounds_rejects_mismatched_epoch_labels(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1", "02_r1"], [(0.0, 1.0), (2.0, 3.0)])
    _write_timestamps_ephys_npz(
        analysis_path,
        ["02_r1", "01_s1"],
        [(2.1, 2.9), (0.1, 0.9)],
    )

    with pytest.raises(ValueError, match="do not match the NWB epochs table"):
        replace_epoch_bounds(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_replace_epoch_bounds_requires_timestamps_ephys_npz(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])

    with pytest.raises(FileNotFoundError, match="timestamps_ephys.npz not found"):
        replace_epoch_bounds(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_replace_epoch_bounds_rejects_unreadable_timestamps_npz(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    (analysis_path / "timestamps_ephys.npz").write_text("not a valid npz", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to load"):
        replace_epoch_bounds(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_replace_epoch_bounds_requires_overwrite_for_existing_output(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    output_path = nwb_root / f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    output_path.write_bytes(b"existing output")

    with pytest.raises(FileExistsError, match="Output path already exists"):
        replace_epoch_bounds(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )
