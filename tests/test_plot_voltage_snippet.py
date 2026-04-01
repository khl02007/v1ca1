from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v1ca1.nwb.plot_voltage_snippet import (
    align_channel_metadata_to_recording,
    FILTERED_FIGURE_STEM,
    RAW_FIGURE_STEM,
    build_channel_metadata,
    compute_snippet_frame_bounds,
    compute_snippet_num_frames,
    get_output_paths,
    main,
    order_shank_channels,
    parse_arguments,
    plot_voltage_snippet,
    resolve_snippet_selection,
    validate_probe_shank_layout,
)


def _make_shank_metadata(
    recording_channel_ids: list[int] | None,
    channel_ids: list[int],
    y_values: list[float] | None,
    rel_y_values: list[float] | None = None,
    probe_electrodes: list[int] | None = None,
    probe_shank_values: list[int] | None = None,
    probe_group_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return one dataframe shaped like the NWB electrodes table."""
    num_channels = len(channel_ids)
    metadata = pd.DataFrame(
        {
            "channel_id": channel_ids,
            "location": ["CA1"] * num_channels,
        }
    )
    if y_values is not None:
        metadata["y"] = y_values
    if rel_y_values is not None:
        metadata["rel_y"] = rel_y_values
    if probe_electrodes is not None:
        metadata["probe_electrode"] = probe_electrodes
    if probe_shank_values is not None:
        metadata["probe_shank"] = probe_shank_values
    if probe_group_labels is not None:
        metadata["group_name"] = probe_group_labels
    if recording_channel_ids is None:
        recording_channel_ids = channel_ids
    metadata.index = pd.Index(recording_channel_ids, name="id")
    return metadata


def _write_test_nwb(
    nwb_path: Path,
    sampling_frequency: float = 20000.0,
    num_frames: int = 4000,
    num_probes: int = 1,
    include_all_shanks: bool = True,
    repeated_channel_ids_per_probe: bool = False,
) -> None:
    """Write one small NWB file with an ElectricalSeries and electrode metadata."""
    pynwb = pytest.importorskip("pynwb")

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device("test-device")
    nwbfile.add_electrode_column("channel_id", "Lab channel id")
    nwbfile.add_electrode_column("probe_shank", "Shank index")
    nwbfile.add_electrode_column("probe_electrode", "Electrode index within shank")

    num_shanks = 4 if include_all_shanks else 3
    num_channels = 0
    for probe_idx in range(num_probes):
        electrode_group = nwbfile.create_electrode_group(
            name=f"test-electrode-group-{probe_idx}",
            description="test electrode group",
            location="test location",
            device=device,
        )
        for shank_idx in range(num_shanks):
            for channel_in_shank in range(32):
                if repeated_channel_ids_per_probe:
                    channel_id = 32 * shank_idx + channel_in_shank
                else:
                    channel_id = 128 * probe_idx + 32 * shank_idx + channel_in_shank
                depth = 1000.0 - 20.0 * channel_in_shank
                nwbfile.add_electrode(
                    id=int(num_channels),
                    x=float(shank_idx * 40.0),
                    y=depth,
                    z=0.0,
                    imp=np.nan,
                    location="CA1",
                    filtering="none",
                    group=electrode_group,
                    channel_id=int(channel_id),
                    probe_shank=int(shank_idx),
                    probe_electrode=int(channel_in_shank),
                )
                num_channels += 1

    electrode_region = nwbfile.create_electrode_table_region(
        list(range(num_channels)),
        "all electrodes",
    )
    time = np.arange(num_frames, dtype=float) / sampling_frequency
    traces = np.zeros((num_frames, num_channels), dtype=np.int16)
    for channel_index in range(num_channels):
        traces[:, channel_index] = np.round(
            40.0 * np.sin(2.0 * np.pi * (channel_index + 1) * time * 5.0)
            + channel_index
        ).astype(np.int16)

    acquisition_series = pynwb.ecephys.ElectricalSeries(
        name="ElectricalSeries",
        data=traces,
        electrodes=electrode_region,
        rate=sampling_frequency,
        filtering="none",
    )
    nwbfile.add_acquisition(acquisition_series)

    with pynwb.NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)


def _write_timestamps_ephys_pickle(
    analysis_path: Path,
    timestamps_by_epoch: dict[str, np.ndarray],
) -> None:
    """Write one legacy timestamps_ephys pickle for helper-backed tests."""
    analysis_path.mkdir(parents=True, exist_ok=True)
    with open(analysis_path / "timestamps_ephys.pkl", "wb") as file:
        pickle.dump(
            {
                epoch: np.asarray(timestamps, dtype=float)
                for epoch, timestamps in timestamps_by_epoch.items()
            },
            file,
        )


def test_build_channel_metadata_groups_channels_by_probe_and_shank() -> None:
    electrode_df = _make_shank_metadata(
        recording_channel_ids=None,
        channel_ids=[0, 31, 32, 127, 128, 159],
        y_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        probe_shank_values=[0, 0, 1, 3, 0, 0],
    )

    metadata = build_channel_metadata(electrode_df)

    assert metadata["probe_idx"].tolist() == [0, 0, 0, 0, 1, 1]
    assert metadata["shank_idx"].tolist() == [0, 0, 1, 3, 0, 0]


def test_order_shank_channels_prefers_y_then_probe_electrode() -> None:
    metadata = _make_shank_metadata(
        recording_channel_ids=None,
        channel_ids=[2, 1, 3],
        y_values=[50.0, 90.0, 50.0],
        rel_y_values=[5.0, 4.0, 3.0],
        probe_electrodes=[2, 1, 7],
    )

    ordered = order_shank_channels(build_channel_metadata(metadata))

    assert ordered["channel_id"].tolist() == [1, 3, 2]


def test_order_shank_channels_falls_back_to_rel_y() -> None:
    metadata = _make_shank_metadata(
        recording_channel_ids=None,
        channel_ids=[0, 1, 2],
        y_values=[0.0, 0.0, 0.0],
        rel_y_values=[20.0, 80.0, 50.0],
        probe_electrodes=[0, 1, 2],
    )

    ordered = order_shank_channels(build_channel_metadata(metadata))

    assert ordered["channel_id"].tolist() == [1, 2, 0]


def test_order_shank_channels_requires_depth_coordinate() -> None:
    metadata = _make_shank_metadata(
        recording_channel_ids=None,
        channel_ids=[0, 1, 2],
        y_values=[0.0, 0.0, 0.0],
        rel_y_values=[1.0, 1.0, 1.0],
        probe_electrodes=[0, 1, 2],
    )

    with pytest.raises(ValueError, match="Could not determine a within-shank depth ordering"):
        order_shank_channels(build_channel_metadata(metadata))


def test_validate_probe_shank_layout_requires_four_shanks() -> None:
    channel_ids = list(range(96))
    metadata = _make_shank_metadata(
        recording_channel_ids=None,
        channel_ids=channel_ids,
        y_values=list(np.linspace(100.0, 0.0, num=len(channel_ids))),
        probe_shank_values=[channel_id // 32 for channel_id in channel_ids],
    )

    with pytest.raises(ValueError, match="exactly four shanks"):
        validate_probe_shank_layout(build_channel_metadata(metadata))


def test_build_channel_metadata_accepts_repeated_channel_ids_across_probes() -> None:
    repeated_probe_channel_ids = list(range(128)) + list(range(128))
    metadata = _make_shank_metadata(
        recording_channel_ids=list(range(256)),
        channel_ids=repeated_probe_channel_ids,
        y_values=list(np.linspace(0.0, -127.0, num=128)) * 2,
        probe_shank_values=[channel_id // 32 for channel_id in repeated_probe_channel_ids],
        probe_electrodes=[channel_id % 32 for channel_id in repeated_probe_channel_ids],
        probe_group_labels=(["0"] * 128) + (["1"] * 128),
    )

    channel_metadata = build_channel_metadata(metadata)

    assert channel_metadata["recording_channel_id"].tolist() == list(range(256))
    assert channel_metadata["probe_idx"].tolist()[:4] == [0, 0, 0, 0]
    assert channel_metadata["probe_idx"].tolist()[128:132] == [1, 1, 1, 1]
    assert channel_metadata["channel_id"].tolist()[:4] == [0, 1, 2, 3]
    assert channel_metadata["channel_id"].tolist()[128:132] == [0, 1, 2, 3]


def test_build_channel_metadata_requires_probe_grouping_when_channel_ids_repeat() -> None:
    metadata = _make_shank_metadata(
        recording_channel_ids=list(range(256)),
        channel_ids=list(range(128)) + list(range(128)),
        y_values=list(np.linspace(0.0, -127.0, num=128)) * 2,
        probe_shank_values=([0] * 256),
    )

    with pytest.raises(ValueError, match="duplicate channel ids but does not expose"):
        build_channel_metadata(metadata)


def test_align_channel_metadata_to_recording_uses_recording_channel_id() -> None:
    class _FakeRecording:
        def get_channel_ids(self):
            return np.asarray([101, 100], dtype=int)

    metadata = _make_shank_metadata(
        recording_channel_ids=[100, 101],
        channel_ids=[0, 0],
        y_values=[0.0, -40.0],
        probe_shank_values=[0, 0],
        probe_electrodes=[0, 1],
        probe_group_labels=["0", "1"],
    )

    aligned = align_channel_metadata_to_recording(
        _FakeRecording(),
        build_channel_metadata(metadata),
    )

    assert aligned["recording_channel_id"].tolist() == [101, 100]
    assert aligned["channel_id"].tolist() == [0, 0]
    assert aligned["probe_idx"].tolist() == [1, 0]
    assert aligned["recording_channel_index"].tolist() == [0, 1]


def test_compute_snippet_frame_bounds_validates_bounds() -> None:
    assert compute_snippet_frame_bounds(
        start_time_s=1.25,
        duration_s=0.150,
        sampling_frequency=1000.0,
        total_frames=2000,
    ) == (1250, 1400)

    with pytest.raises(ValueError, match="extends beyond the recording bounds"):
        compute_snippet_frame_bounds(
            start_time_s=1.95,
            duration_s=0.100,
            sampling_frequency=1000.0,
            total_frames=2000,
        )


def test_resolve_snippet_selection_explicit_does_not_require_helper_outputs(tmp_path: Path) -> None:
    selection = resolve_snippet_selection(
        requested_start_time_s=0.05,
        requested_epoch="02_r1",
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=2000,
        analysis_path=tmp_path / "analysis",
        random_seed=0,
    )

    assert selection["resolved_start_time_s"] == pytest.approx(0.05)
    assert selection["start_time_source"] == "explicit"
    assert selection["selected_run_epoch"] is None
    assert selection["epoch_timestamp_source"] is None
    assert selection["epoch_relative_start_index"] is None
    assert selection["start_frame"] == 50
    assert selection["end_frame"] == 100


def test_resolve_snippet_selection_uses_first_run_epoch_deterministically(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis" / "animal" / "20240101"
    timestamps_by_epoch = {
        "01_s1": np.arange(0, 100, dtype=float) / 1000.0,
        "02_r1": np.arange(100, 300, dtype=float) / 1000.0,
        "03_r2": np.arange(300, 500, dtype=float) / 1000.0,
    }
    _write_timestamps_ephys_pickle(analysis_path, timestamps_by_epoch)

    selection = resolve_snippet_selection(
        requested_start_time_s=None,
        requested_epoch=None,
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=500,
        analysis_path=analysis_path,
        random_seed=0,
    )

    duration_frames = compute_snippet_num_frames(
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=500,
    )
    expected_epoch_start_index = int(
        np.random.default_rng(0).integers(
            0,
            len(timestamps_by_epoch["02_r1"]) - duration_frames + 1,
        )
    )
    assert selection["start_time_source"] == "first_run_epoch_random"
    assert selection["selected_run_epoch"] == "02_r1"
    assert selection["epoch_timestamp_source"] == "pickle"
    assert selection["epoch_relative_start_index"] == expected_epoch_start_index
    assert selection["start_frame"] == 100 + expected_epoch_start_index
    assert selection["end_frame"] == selection["start_frame"] + duration_frames
    assert selection["resolved_start_time_s"] == pytest.approx(
        timestamps_by_epoch["02_r1"][expected_epoch_start_index]
    )


def test_resolve_snippet_selection_uses_requested_epoch_deterministically(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis" / "animal" / "20240101"
    timestamps_by_epoch = {
        "01_s1": np.arange(0, 100, dtype=float) / 1000.0,
        "02_r1": np.arange(100, 300, dtype=float) / 1000.0,
        "03_s2": np.arange(300, 500, dtype=float) / 1000.0,
    }
    _write_timestamps_ephys_pickle(analysis_path, timestamps_by_epoch)

    selection = resolve_snippet_selection(
        requested_start_time_s=None,
        requested_epoch="03_s2",
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=500,
        analysis_path=analysis_path,
        random_seed=0,
    )

    duration_frames = compute_snippet_num_frames(
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=500,
    )
    expected_epoch_start_index = int(
        np.random.default_rng(0).integers(
            0,
            len(timestamps_by_epoch["03_s2"]) - duration_frames + 1,
        )
    )
    assert selection["start_time_source"] == "requested_epoch_random"
    assert selection["selected_run_epoch"] == "03_s2"
    assert selection["epoch_timestamp_source"] == "pickle"
    assert selection["epoch_relative_start_index"] == expected_epoch_start_index
    assert selection["start_frame"] == 300 + expected_epoch_start_index
    assert selection["end_frame"] == selection["start_frame"] + duration_frames
    assert selection["resolved_start_time_s"] == pytest.approx(
        timestamps_by_epoch["03_s2"][expected_epoch_start_index]
    )


def test_resolve_snippet_selection_changes_with_seed(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis" / "animal" / "20240101"
    timestamps_by_epoch = {
        "01_s1": np.arange(0, 100, dtype=float) / 1000.0,
        "02_r1": np.arange(100, 1300, dtype=float) / 1000.0,
    }
    _write_timestamps_ephys_pickle(analysis_path, timestamps_by_epoch)

    selection_seed_0 = resolve_snippet_selection(
        requested_start_time_s=None,
        requested_epoch=None,
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=1300,
        analysis_path=analysis_path,
        random_seed=0,
    )
    selection_seed_1 = resolve_snippet_selection(
        requested_start_time_s=None,
        requested_epoch=None,
        duration_s=0.050,
        sampling_frequency=1000.0,
        total_frames=1300,
        analysis_path=analysis_path,
        random_seed=1,
    )

    assert selection_seed_0["epoch_relative_start_index"] != selection_seed_1["epoch_relative_start_index"]


def test_resolve_snippet_selection_requires_helper_outputs_for_auto_mode(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run `python -m v1ca1.helper.get_timestamps`"):
        resolve_snippet_selection(
            requested_start_time_s=None,
            requested_epoch=None,
            duration_s=0.050,
            sampling_frequency=1000.0,
            total_frames=2000,
            analysis_path=tmp_path / "analysis",
            random_seed=0,
        )


def test_resolve_snippet_selection_requires_requested_epoch_to_exist(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis" / "animal" / "20240101"
    _write_timestamps_ephys_pickle(
        analysis_path,
        {
            "01_s1": np.arange(0, 100, dtype=float) / 1000.0,
            "02_r1": np.arange(100, 300, dtype=float) / 1000.0,
        },
    )

    with pytest.raises(ValueError, match="Requested epoch '03_s2' was not found"):
        resolve_snippet_selection(
            requested_start_time_s=None,
            requested_epoch="03_s2",
            duration_s=0.050,
            sampling_frequency=1000.0,
            total_frames=300,
            analysis_path=analysis_path,
            random_seed=0,
        )


def test_resolve_snippet_selection_requires_run_epoch(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis" / "animal" / "20240101"
    _write_timestamps_ephys_pickle(
        analysis_path,
        {
            "01_s1": np.arange(0, 100, dtype=float) / 1000.0,
            "02_s2": np.arange(100, 300, dtype=float) / 1000.0,
        },
    )

    with pytest.raises(ValueError, match="Could not infer any run epochs"):
        resolve_snippet_selection(
            requested_start_time_s=None,
            requested_epoch=None,
            duration_s=0.050,
            sampling_frequency=1000.0,
            total_frames=300,
            analysis_path=analysis_path,
            random_seed=0,
        )


def test_resolve_snippet_selection_requires_long_enough_first_run_epoch(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis" / "animal" / "20240101"
    _write_timestamps_ephys_pickle(
        analysis_path,
        {
            "01_s1": np.arange(0, 100, dtype=float) / 1000.0,
            "02_r1": np.arange(100, 120, dtype=float) / 1000.0,
        },
    )

    with pytest.raises(ValueError, match="shorter than the requested snippet duration"):
        resolve_snippet_selection(
            requested_start_time_s=None,
            requested_epoch=None,
            duration_s=0.050,
            sampling_frequency=1000.0,
            total_frames=120,
            analysis_path=analysis_path,
            random_seed=0,
        )


def test_plot_voltage_snippet_writes_two_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pynwb")
    pytest.importorskip("spikeinterface.full")

    animal_name = "animal"
    date = "20240101"
    analysis_root = tmp_path / "analysis"
    nwb_root = tmp_path / "raw"
    nwb_root.mkdir(parents=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    _write_test_nwb(nwb_path)

    outputs = plot_voltage_snippet(
        animal_name=animal_name,
        date=date,
        start_time_s=0.05,
        broadband_duration_s=0.100,
        spike_band_duration_s=0.050,
        analysis_root=analysis_root,
        nwb_root=nwb_root,
    )
    raw_path = outputs["raw_figure_path"]
    filtered_path = outputs["filtered_figure_path"]

    expected_raw_path, expected_filtered_path = get_output_paths(
        animal_name=animal_name,
        date=date,
        start_time_s=0.05,
        broadband_duration_s=0.100,
        spike_band_duration_s=0.050,
        analysis_root=analysis_root,
    )
    assert raw_path == expected_raw_path
    assert filtered_path == expected_filtered_path
    assert raw_path != filtered_path
    assert raw_path.exists()
    assert filtered_path.exists()
    assert raw_path.name.startswith(RAW_FIGURE_STEM)
    assert filtered_path.name.startswith(FILTERED_FIGURE_STEM)
    assert raw_path.suffix == ".png"
    assert filtered_path.suffix == ".png"
    assert outputs["broadband_duration_s"] == pytest.approx(0.100)
    assert outputs["spike_band_duration_s"] == pytest.approx(0.050)
    assert outputs["selection_duration_s"] == pytest.approx(0.100)

    log_dir = analysis_root / animal_name / date / "v1ca1_log"
    assert not log_dir.exists()


def test_plot_voltage_snippet_auto_start_uses_resolved_time_in_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pynwb")
    pytest.importorskip("spikeinterface.full")

    animal_name = "animal"
    date = "20240101"
    analysis_root = tmp_path / "analysis"
    analysis_path = analysis_root / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_root.mkdir(parents=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    _write_test_nwb(nwb_path)

    timestamps_by_epoch = {
        "01_s1": np.arange(0, 1000, dtype=float) / 20000.0,
        "02_r1": np.arange(1000, 4000, dtype=float) / 20000.0,
    }
    _write_timestamps_ephys_pickle(analysis_path, timestamps_by_epoch)
    expected_selection = resolve_snippet_selection(
        requested_start_time_s=None,
        requested_epoch=None,
        duration_s=0.050,
        sampling_frequency=20000.0,
        total_frames=4000,
        analysis_path=analysis_path,
        random_seed=0,
    )

    outputs = plot_voltage_snippet(
        animal_name=animal_name,
        date=date,
        start_time_s=None,
        broadband_duration_s=0.050,
        spike_band_duration_s=0.050,
        analysis_root=analysis_root,
        nwb_root=nwb_root,
        random_seed=0,
    )
    raw_path = outputs["raw_figure_path"]
    filtered_path = outputs["filtered_figure_path"]

    expected_raw_path, expected_filtered_path = get_output_paths(
        animal_name=animal_name,
        date=date,
        start_time_s=float(expected_selection["resolved_start_time_s"]),
        broadband_duration_s=0.050,
        spike_band_duration_s=0.050,
        analysis_root=analysis_root,
    )
    assert raw_path == expected_raw_path
    assert filtered_path == expected_filtered_path
    assert raw_path.exists()
    assert filtered_path.exists()

    assert outputs["resolved_start_time_s"] == pytest.approx(
        float(expected_selection["resolved_start_time_s"])
    )
    assert outputs["start_time_source"] == "first_run_epoch_random"
    assert outputs["selected_run_epoch"] == "02_r1"
    assert outputs["epoch_timestamp_source"] == "pickle"
    assert outputs["epoch_relative_start_index"] == expected_selection["epoch_relative_start_index"]


def test_plot_voltage_snippet_auto_start_uses_helper_frame_index_for_absolute_timestamps(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pynwb")
    pytest.importorskip("spikeinterface.full")

    animal_name = "animal"
    date = "20240105"
    analysis_root = tmp_path / "analysis"
    analysis_path = analysis_root / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_root.mkdir(parents=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    _write_test_nwb(nwb_path)

    absolute_offset_s = 2_567_123.0
    timestamps_by_epoch = {
        "01_s1": absolute_offset_s + np.arange(0, 1000, dtype=float) / 20000.0,
        "02_r1": absolute_offset_s + np.arange(1000, 4000, dtype=float) / 20000.0,
    }
    _write_timestamps_ephys_pickle(analysis_path, timestamps_by_epoch)
    expected_selection = resolve_snippet_selection(
        requested_start_time_s=None,
        requested_epoch=None,
        duration_s=0.050,
        sampling_frequency=20000.0,
        total_frames=4000,
        analysis_path=analysis_path,
        random_seed=0,
    )

    outputs = plot_voltage_snippet(
        animal_name=animal_name,
        date=date,
        start_time_s=None,
        broadband_duration_s=0.050,
        spike_band_duration_s=0.050,
        analysis_root=analysis_root,
        nwb_root=nwb_root,
        random_seed=0,
    )

    assert outputs["start_frame"] == expected_selection["start_frame"]
    assert outputs["end_frame"] == expected_selection["end_frame"]
    assert outputs["resolved_start_time_s"] == pytest.approx(
        float(expected_selection["resolved_start_time_s"])
    )
    assert outputs["raw_figure_path"].exists()
    assert outputs["filtered_figure_path"].exists()


def test_plot_voltage_snippet_accepts_repeated_channel_ids_across_probes(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pynwb")
    pytest.importorskip("spikeinterface.full")

    animal_name = "animal"
    date = "20240102"
    analysis_root = tmp_path / "analysis"
    nwb_root = tmp_path / "raw"
    nwb_root.mkdir(parents=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    _write_test_nwb(
        nwb_path,
        num_probes=2,
        repeated_channel_ids_per_probe=True,
    )

    outputs = plot_voltage_snippet(
        animal_name=animal_name,
        date=date,
        start_time_s=0.05,
        broadband_duration_s=0.050,
        spike_band_duration_s=0.050,
        analysis_root=analysis_root,
        nwb_root=nwb_root,
    )
    raw_path = outputs["raw_figure_path"]
    filtered_path = outputs["filtered_figure_path"]

    assert raw_path.exists()
    assert filtered_path.exists()
    assert raw_path.name.startswith(RAW_FIGURE_STEM)
    assert filtered_path.name.startswith(FILTERED_FIGURE_STEM)


def test_main_writes_run_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pynwb")
    pytest.importorskip("spikeinterface.full")

    animal_name = "animal"
    date = "20240103"
    analysis_root = tmp_path / "analysis"
    analysis_path = analysis_root / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_root.mkdir(parents=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    _write_test_nwb(nwb_path)
    _write_timestamps_ephys_pickle(
        analysis_path,
        {
            "01_s1": np.arange(0, 1000, dtype=float) / 20000.0,
            "02_r1": np.arange(1000, 4000, dtype=float) / 20000.0,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_voltage_snippet.py",
            "--animal-name",
            animal_name,
            "--date",
            date,
            "--broadband-duration-s",
            "0.05",
            "--spike-band-duration-s",
            "0.05",
            "--data-root",
            str(analysis_root),
            "--nwb-root",
            str(nwb_root),
        ],
    )
    main()

    log_dir = analysis_root / animal_name / date / "v1ca1_log"
    log_paths = sorted(log_dir.glob("v1ca1_nwb_plot_voltage_snippet_*.json"))
    assert len(log_paths) == 1
    log_record = json.loads(log_paths[0].read_text(encoding="utf-8"))
    assert log_record["parameters"]["requested_start_time_s"] is None
    assert log_record["parameters"]["requested_epoch"] is None
    assert str(log_record["parameters"]["data_root"]) == str(analysis_root)
    assert log_record["parameters"]["broadband_duration_s"] == 0.05
    assert log_record["parameters"]["spike_band_duration_s"] == 0.05
    assert log_record["parameters"]["random_seed"] == 0
    assert log_record["outputs"]["start_time_source"] == "first_run_epoch_random"
    assert log_record["outputs"]["raw_figure_path"].endswith(".png")


def test_main_accepts_epoch_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("pynwb")
    pytest.importorskip("spikeinterface.full")

    animal_name = "animal"
    date = "20240104"
    analysis_root = tmp_path / "analysis"
    analysis_path = analysis_root / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_root.mkdir(parents=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    _write_test_nwb(nwb_path)
    _write_timestamps_ephys_pickle(
        analysis_path,
        {
            "01_s1": np.arange(0, 1000, dtype=float) / 20000.0,
            "02_r1": np.arange(1000, 2500, dtype=float) / 20000.0,
            "03_s2": np.arange(2500, 4000, dtype=float) / 20000.0,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_voltage_snippet.py",
            "--animal-name",
            animal_name,
            "--date",
            date,
            "--epoch",
            "03_s2",
            "--broadband-duration-s",
            "0.05",
            "--spike-band-duration-s",
            "0.05",
            "--data-root",
            str(analysis_root),
            "--nwb-root",
            str(nwb_root),
        ],
    )
    main()

    log_dir = analysis_root / animal_name / date / "v1ca1_log"
    log_paths = sorted(log_dir.glob("v1ca1_nwb_plot_voltage_snippet_*.json"))
    assert len(log_paths) == 1
    log_record = json.loads(log_paths[0].read_text(encoding="utf-8"))
    assert str(log_record["parameters"]["data_root"]) == str(analysis_root)
    assert log_record["parameters"]["broadband_duration_s"] == 0.05
    assert log_record["parameters"]["spike_band_duration_s"] == 0.05
    assert log_record["parameters"]["requested_epoch"] == "03_s2"
    assert log_record["outputs"]["selected_run_epoch"] == "03_s2"
    assert log_record["outputs"]["start_time_source"] == "requested_epoch_random"


def test_parse_arguments_uses_separate_duration_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_voltage_snippet.py",
            "--animal-name",
            "animal",
            "--date",
            "20240105",
        ],
    )

    args = parse_arguments()

    assert args.broadband_duration_s == pytest.approx(1.0)
    assert args.spike_band_duration_s == pytest.approx(0.2)
