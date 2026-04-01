from __future__ import annotations

"""Plot short NWB voltage snippets grouped by probe and shank."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_NWB_ROOT, get_run_epochs, load_ephys_timestamps_by_epoch

if TYPE_CHECKING:
    from pandas import DataFrame


DEFAULT_ANALYSIS_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_BROADBAND_DURATION_S = 1.0
DEFAULT_SPIKE_BAND_DURATION_S = 0.2
DEFAULT_RANDOM_SEED = 0
EXPECTED_SHANK_COUNT = 4
CHANNELS_PER_PROBE = 128
CHANNELS_PER_SHANK = 32
FILTER_FREQ_MIN_HZ = 600.0
FILTER_FREQ_MAX_HZ = 5000.0
RAW_FIGURE_STEM = "voltage_snippet_raw"
FILTERED_FIGURE_STEM = "voltage_snippet_bp600_5000"
_SPIKEINTERFACE = None


def get_spikeinterface():
    """Import SpikeInterface lazily and configure shared job settings once."""
    global _SPIKEINTERFACE
    if _SPIKEINTERFACE is None:
        import spikeinterface.full as si

        si.set_global_job_kwargs(chunk_size=30000, n_jobs=8, progress_bar=True)
        _SPIKEINTERFACE = si
    return _SPIKEINTERFACE


def get_analysis_path(
    animal_name: str,
    date: str,
    analysis_root: Path,
) -> Path:
    """Return the analysis directory for one animal/date session."""
    return analysis_root / animal_name / date


def get_nwb_path(animal_name: str, date: str, nwb_root: Path) -> Path:
    """Return the NWB file path for one animal/date session."""
    return nwb_root / f"{animal_name}{date}.nwb"


def format_seconds_for_filename(time_s: float) -> str:
    """Return one filesystem-safe seconds token such as `12p345s`."""
    return f"{time_s:.3f}".replace(".", "p") + "s"


def format_duration_for_filename(duration_s: float) -> str:
    """Return one filesystem-safe duration token such as `dur0p15s`."""
    return "dur" + f"{duration_s:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "s"


def get_output_paths(
    animal_name: str,
    date: str,
    start_time_s: float,
    broadband_duration_s: float,
    spike_band_duration_s: float,
    analysis_root: Path,
) -> tuple[Path, Path]:
    """Return the raw and filtered output figure paths for one snippet."""
    analysis_path = get_analysis_path(animal_name, date, analysis_root)
    fig_dir = analysis_path / "figs" / "nwb"
    raw_token = (
        f"{format_seconds_for_filename(start_time_s)}_"
        f"{format_duration_for_filename(broadband_duration_s)}"
    )
    filtered_token = (
        f"{format_seconds_for_filename(start_time_s)}_"
        f"{format_duration_for_filename(spike_band_duration_s)}"
    )
    return (
        fig_dir / f"{RAW_FIGURE_STEM}_{raw_token}.png",
        fig_dir / f"{FILTERED_FIGURE_STEM}_{filtered_token}.png",
    )


def load_electrode_table(nwb_path: Path) -> "DataFrame":
    """Return the NWB electrodes table as a dataframe."""
    import pynwb

    with pynwb.NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        if nwbfile.electrodes is None:
            raise ValueError("NWB file does not contain an electrodes table.")
        return nwbfile.electrodes.to_dataframe().copy()


def build_channel_metadata(electrode_df: "DataFrame") -> "DataFrame":
    """Add derived probe and shank metadata keyed by recording channel id."""
    channel_metadata = electrode_df.copy()
    recording_channel_ids = np.asarray(channel_metadata.index.tolist(), dtype=int)
    if recording_channel_ids.size == 0:
        raise ValueError("NWB electrodes table is empty.")
    if len(set(recording_channel_ids.tolist())) != recording_channel_ids.size:
        raise ValueError("NWB electrodes table contains duplicate recording channel ids.")
    channel_metadata["recording_channel_id"] = recording_channel_ids

    if "channel_id" in channel_metadata.columns:
        channel_ids = np.asarray(channel_metadata["channel_id"], dtype=int)
    else:
        channel_ids = recording_channel_ids
        channel_metadata["channel_id"] = channel_ids

    channel_metadata["channel_id"] = channel_ids

    probe_group_column = None
    for candidate in ("group_name", "group"):
        if candidate in channel_metadata.columns:
            candidate_values = channel_metadata[candidate]
            if candidate_values.notna().any():
                probe_group_column = candidate
                break

    if probe_group_column is not None:
        probe_group_labels = channel_metadata[probe_group_column].astype(str)
        ordered_probe_labels = sorted(probe_group_labels.unique().tolist())
        probe_idx_by_label = {
            probe_label: probe_idx for probe_idx, probe_label in enumerate(ordered_probe_labels)
        }
        channel_metadata["probe_group_label"] = probe_group_labels
        channel_metadata["probe_idx"] = probe_group_labels.map(probe_idx_by_label).astype(int)
    else:
        if len(set(channel_ids.tolist())) != channel_ids.size:
            raise ValueError(
                "NWB electrodes table contains duplicate channel ids but does not expose "
                "`group_name` or `group` metadata needed to distinguish probes."
            )
        channel_metadata["probe_group_label"] = (channel_ids // CHANNELS_PER_PROBE).astype(str)
        channel_metadata["probe_idx"] = channel_ids // CHANNELS_PER_PROBE

    if "probe_shank" in channel_metadata.columns and channel_metadata["probe_shank"].notna().any():
        probe_shank = pd.to_numeric(channel_metadata["probe_shank"], errors="coerce")
        if probe_shank.isna().any():
            raise ValueError("NWB electrodes table contains non-numeric `probe_shank` values.")
        channel_metadata["shank_idx"] = probe_shank.astype(int)
    else:
        channel_metadata["shank_idx"] = (channel_ids % CHANNELS_PER_PROBE) // CHANNELS_PER_SHANK

    if "probe_shank" in channel_metadata.columns:
        probe_shank = pd.to_numeric(channel_metadata["probe_shank"], errors="coerce")
        valid_probe_shank = probe_shank.notna().to_numpy(dtype=bool)
        mismatched = np.zeros(channel_metadata.shape[0], dtype=bool)
        if np.any(valid_probe_shank):
            mismatched[valid_probe_shank] = (
                probe_shank.loc[valid_probe_shank].astype(int).to_numpy()
                != channel_metadata.loc[valid_probe_shank, "shank_idx"].to_numpy(dtype=int)
            )
        if mismatched.any():
            mismatched_rows = channel_metadata.loc[mismatched, ["channel_id", "probe_shank", "shank_idx"]]
            raise ValueError(
                "NWB electrodes table has `probe_shank` values that disagree with the "
                f"derived shank index from channel ids:\n{mismatched_rows.to_string(index=False)}"
            )

    if "probe_electrode" not in channel_metadata.columns:
        channel_metadata["probe_electrode"] = np.zeros(channel_ids.shape, dtype=int)
    channel_metadata["probe_electrode"] = (
        pd.to_numeric(channel_metadata["probe_electrode"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return channel_metadata


def validate_probe_shank_layout(channel_metadata: "DataFrame") -> list[int]:
    """Require four shanks for every detected probe."""
    probe_indices = sorted(channel_metadata["probe_idx"].astype(int).unique().tolist())
    for probe_idx in probe_indices:
        probe_metadata = channel_metadata.loc[channel_metadata["probe_idx"] == probe_idx]
        shanks = sorted(probe_metadata["shank_idx"].astype(int).unique().tolist())
        expected_shanks = list(range(EXPECTED_SHANK_COUNT))
        if shanks != expected_shanks:
            raise ValueError(
                "Expected each detected probe to expose exactly four shanks indexed "
                f"{expected_shanks}, but probe {probe_idx} has shanks {shanks}."
            )
    return probe_indices


def get_depth_column(shank_metadata: "DataFrame") -> str:
    """Return the electrode coordinate column used to order channels by depth."""
    for candidate in ("y", "rel_y"):
        if candidate not in shank_metadata.columns:
            continue
        candidate_values = pd.to_numeric(shank_metadata[candidate], errors="coerce")
        finite_values = candidate_values[np.isfinite(candidate_values.to_numpy(dtype=float))]
        if finite_values.nunique() > 1:
            return candidate
    channel_ids = shank_metadata["channel_id"].astype(int).tolist()
    raise ValueError(
        "Could not determine a within-shank depth ordering because neither `y` nor "
        "`rel_y` varies across the channels in this shank. "
        f"Channel ids: {channel_ids!r}"
    )


def order_shank_channels(shank_metadata: "DataFrame") -> "DataFrame":
    """Return one shank's channels sorted from highest to lowest depth."""
    ordered_metadata = shank_metadata.copy()
    depth_column = get_depth_column(ordered_metadata)
    ordered_metadata["_depth_order"] = pd.to_numeric(
        ordered_metadata[depth_column],
        errors="coerce",
    )
    ordered_metadata = ordered_metadata.sort_values(
        by=["_depth_order", "probe_electrode", "channel_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return ordered_metadata.drop(columns="_depth_order")


def align_channel_metadata_to_recording(
    recording: Any,
    channel_metadata: "DataFrame",
) -> "DataFrame":
    """Align NWB channel metadata to the SpikeInterface channel order."""
    recording_channel_ids = np.asarray(recording.get_channel_ids(), dtype=int)
    indexed_metadata = channel_metadata.set_index("recording_channel_id", drop=False)
    missing_channel_ids = [
        int(channel_id)
        for channel_id in recording_channel_ids
        if int(channel_id) not in indexed_metadata.index
    ]
    if missing_channel_ids:
        raise ValueError(
            "NWB electrodes table is missing metadata for recording channel ids "
            f"{missing_channel_ids!r}."
        )
    if not indexed_metadata.index.is_unique:
        raise ValueError("NWB electrodes table contains duplicate recording channel ids.")

    aligned_metadata = indexed_metadata.loc[recording_channel_ids].copy()
    aligned_metadata["recording_channel_index"] = np.arange(recording_channel_ids.size, dtype=int)
    return aligned_metadata.reset_index(drop=True)


def compute_snippet_frame_bounds(
    start_time_s: float,
    duration_s: float,
    sampling_frequency: float,
    total_frames: int,
) -> tuple[int, int]:
    """Return inclusive-exclusive frame bounds for one snippet."""
    if start_time_s < 0:
        raise ValueError("--start-time-s must be non-negative.")
    start_frame = int(round(start_time_s * sampling_frequency))
    duration_frames = compute_snippet_num_frames(
        duration_s=duration_s,
        sampling_frequency=sampling_frequency,
        total_frames=total_frames,
    )
    end_frame = start_frame + duration_frames
    if start_frame >= total_frames:
        raise ValueError(
            "Requested snippet start is outside the recording bounds: "
            f"start frame {start_frame}, total frames {total_frames}."
        )
    if end_frame > total_frames:
        raise ValueError(
            "Requested snippet extends beyond the recording bounds: "
            f"end frame {end_frame}, total frames {total_frames}."
        )
    return start_frame, end_frame


def compute_snippet_num_frames(
    duration_s: float,
    sampling_frequency: float,
    total_frames: int,
) -> int:
    """Return the number of frames needed for one snippet duration."""
    if duration_s <= 0:
        raise ValueError("Snippet duration must be positive.")
    if sampling_frequency <= 0:
        raise ValueError("Recording sampling frequency must be positive.")
    if total_frames <= 0:
        raise ValueError("Recording does not contain any frames.")

    duration_frames = int(round(duration_s * sampling_frequency))
    if duration_frames <= 0:
        raise ValueError(
            "Snippet duration is too short for the recording sampling frequency and "
            "would select zero frames."
        )
    return duration_frames


def resolve_snippet_selection(
    requested_start_time_s: float | None,
    requested_epoch: str | None,
    duration_s: float,
    sampling_frequency: float,
    total_frames: int,
    analysis_path: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Resolve snippet start time and frame bounds from explicit or helper-driven input."""
    if requested_start_time_s is not None:
        start_frame, end_frame = compute_snippet_frame_bounds(
            start_time_s=requested_start_time_s,
            duration_s=duration_s,
            sampling_frequency=sampling_frequency,
            total_frames=total_frames,
        )
        return {
            "requested_start_time_s": float(requested_start_time_s),
            "resolved_start_time_s": float(requested_start_time_s),
            "start_time_source": "explicit",
            "random_seed": int(random_seed),
            "selected_run_epoch": None,
            "epoch_timestamp_source": None,
            "epoch_relative_start_index": None,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    duration_frames = compute_snippet_num_frames(
        duration_s=duration_s,
        sampling_frequency=sampling_frequency,
        total_frames=total_frames,
    )
    try:
        epoch_tags, timestamps_by_epoch, epoch_timestamp_source = load_ephys_timestamps_by_epoch(
            analysis_path
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not load helper ephys timestamp outputs under "
            f"{analysis_path}. Run `python -m v1ca1.helper.get_timestamps` for this "
            "session or pass `--start-time-s` explicitly."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            "Could not read helper ephys timestamp outputs under "
            f"{analysis_path}. Run `python -m v1ca1.helper.get_timestamps` for this "
            "session again or pass `--start-time-s` explicitly."
        ) from exc

    if requested_epoch is not None:
        if requested_epoch not in epoch_tags:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} was not found under {analysis_path}. "
                f"Available epoch tags: {epoch_tags!r}"
            )
        selected_epoch = requested_epoch
        start_time_source = "requested_epoch_random"
    else:
        try:
            run_epochs = get_run_epochs(epoch_tags)
        except ValueError as exc:
            raise ValueError(
                "Could not infer any run epochs from helper timestamp outputs under "
                f"{analysis_path}. Available epoch tags: {epoch_tags!r}. "
                "Pass `--start-time-s` explicitly if you want to bypass run-epoch selection."
            ) from exc

        selected_epoch = run_epochs[0]
        start_time_source = "first_run_epoch_random"

    selected_epoch_position = epoch_tags.index(selected_epoch)
    epoch_timestamps = np.asarray(timestamps_by_epoch[selected_epoch], dtype=float)
    if epoch_timestamps.size < duration_frames:
        raise ValueError(
            "The selected epoch is shorter than the requested snippet duration: "
            f"epoch {selected_epoch!r} has {epoch_timestamps.size} samples, but "
            f"{duration_frames} are required."
        )

    max_epoch_start_index = epoch_timestamps.size - duration_frames
    rng = np.random.default_rng(random_seed)
    epoch_relative_start_index = int(rng.integers(0, max_epoch_start_index + 1))
    epoch_start_offset = sum(
        len(np.asarray(timestamps_by_epoch[epoch], dtype=float))
        for epoch in epoch_tags[:selected_epoch_position]
    )
    start_frame = int(epoch_start_offset + epoch_relative_start_index)
    end_frame = int(start_frame + duration_frames)
    if end_frame > total_frames:
        raise ValueError(
            "Helper ephys timestamp outputs are inconsistent with the recording length: "
            f"requested snippet ends at frame {end_frame}, but the recording has only "
            f"{total_frames} frames."
        )

    return {
        "requested_start_time_s": None,
        "resolved_start_time_s": float(epoch_timestamps[epoch_relative_start_index]),
        "start_time_source": start_time_source,
        "random_seed": int(random_seed),
        "selected_run_epoch": str(selected_epoch),
        "epoch_timestamp_source": str(epoch_timestamp_source),
        "epoch_relative_start_index": epoch_relative_start_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def compute_panel_spacing(traces: np.ndarray, filtered: bool) -> float:
    """Return one robust vertical spacing value for a panel."""
    abs_traces = np.abs(np.asarray(traces, dtype=float))
    if abs_traces.size == 0:
        return 1.0

    percentile_scale = float(np.percentile(abs_traces, 99.5))
    median_abs = float(np.median(abs_traces))
    robust_scale = max(percentile_scale, median_abs * 6.0, 1e-6)
    multiplier = 2.8 if not filtered else 1.6
    return robust_scale * multiplier


def plot_voltage_figure(
    traces: np.ndarray,
    channel_metadata: "DataFrame",
    probe_indices: list[int],
    sampling_frequency: float,
    start_time_s: float,
    duration_s: float,
    output_path: Path,
    filtered: bool,
) -> Path:
    """Save one figure of vertically offset traces grouped by probe and shank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_probes = len(probe_indices)
    figure_height = max(3.0, 2.6 * n_probes)
    figure, axes = plt.subplots(
        n_probes,
        EXPECTED_SHANK_COUNT,
        figsize=(18, figure_height),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    time_ms = np.arange(traces.shape[0], dtype=float) / sampling_frequency * 1000.0
    x_padding_ms = max(2.0, duration_s * 1000.0 * 0.06)
    trace_color = "black" if not filtered else "tab:blue"
    label = (
        f"Bandpass {FILTER_FREQ_MIN_HZ:.0f}-{FILTER_FREQ_MAX_HZ:.0f} Hz"
        if filtered
        else "Raw broadband"
    )
    figure.suptitle(
        f"{label} voltage snippet at {start_time_s:.3f} s ({duration_s:.3f} s)",
        fontsize=14,
    )

    for row_index, probe_idx in enumerate(probe_indices):
        for shank_idx in range(EXPECTED_SHANK_COUNT):
            axis = axes[row_index, shank_idx]
            shank_metadata = channel_metadata.loc[
                (channel_metadata["probe_idx"] == probe_idx)
                & (channel_metadata["shank_idx"] == shank_idx)
            ]
            ordered_metadata = order_shank_channels(shank_metadata)
            panel_traces = traces[:, ordered_metadata["recording_channel_index"].to_numpy(dtype=int)]
            spacing = compute_panel_spacing(panel_traces, filtered=filtered)
            offsets = spacing * np.arange(panel_traces.shape[1] - 1, -1, -1, dtype=float)

            for trace_index, (_, channel_row) in enumerate(ordered_metadata.iterrows()):
                offset = offsets[trace_index]
                axis.plot(
                    time_ms,
                    panel_traces[:, trace_index] + offset,
                    color=trace_color,
                    linewidth=0.7,
                )
                axis.text(
                    -0.4 * x_padding_ms,
                    offset,
                    str(int(channel_row["channel_id"])),
                    ha="right",
                    va="center",
                    fontsize=6,
                    clip_on=False,
                )

            probe_group_label = str(ordered_metadata["probe_group_label"].iloc[0])
            axis.set_title(f"Probe {probe_idx} (group {probe_group_label}) shank {shank_idx}")
            axis.grid(axis="x", alpha=0.2)
            axis.set_xlim(-x_padding_ms, time_ms[-1] + x_padding_ms)
            axis.set_yticks([])
            if offsets.size:
                axis.set_ylim(-spacing, offsets[0] + spacing)
            if shank_idx == 0:
                axis.set_ylabel(f"Probe {probe_idx}\n(group {probe_group_label})")
            if row_index == n_probes - 1:
                axis.set_xlabel("Time from snippet start (ms)")

    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_voltage_snippet(
    animal_name: str,
    date: str,
    start_time_s: float | None = None,
    epoch: str | None = None,
    broadband_duration_s: float = DEFAULT_BROADBAND_DURATION_S,
    spike_band_duration_s: float = DEFAULT_SPIKE_BAND_DURATION_S,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Load one NWB session, save snippet figures, and return output metadata."""
    si = get_spikeinterface()
    analysis_path = get_analysis_path(animal_name, date, analysis_root)
    nwb_path = get_nwb_path(animal_name, date, nwb_root)

    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    print(f"Processing {animal_name} {date}.")
    recording = si.read_nwb_recording(nwb_path)
    if recording.get_num_segments() != 1:
        raise ValueError(
            "plot_voltage_snippet.py currently supports only single-segment recordings."
        )

    channel_metadata = build_channel_metadata(load_electrode_table(nwb_path))
    channel_metadata = align_channel_metadata_to_recording(recording, channel_metadata)
    probe_indices = validate_probe_shank_layout(channel_metadata)

    sampling_frequency = float(recording.get_sampling_frequency())
    if FILTER_FREQ_MAX_HZ >= sampling_frequency / 2.0:
        raise ValueError(
            "Bandpass high cutoff must be below the Nyquist frequency: "
            f"{FILTER_FREQ_MAX_HZ} Hz vs Nyquist {sampling_frequency / 2.0:.3f} Hz."
        )

    total_frames = int(recording.get_num_frames(segment_index=0))
    selection_duration_s = max(broadband_duration_s, spike_band_duration_s)
    selection = resolve_snippet_selection(
        requested_start_time_s=start_time_s,
        requested_epoch=epoch,
        duration_s=selection_duration_s,
        sampling_frequency=sampling_frequency,
        total_frames=total_frames,
        analysis_path=analysis_path,
        random_seed=random_seed,
    )
    resolved_start_time_s = float(selection["resolved_start_time_s"])
    start_frame = int(selection["start_frame"])
    broadband_num_frames = compute_snippet_num_frames(
        duration_s=broadband_duration_s,
        sampling_frequency=sampling_frequency,
        total_frames=total_frames,
    )
    spike_band_num_frames = compute_snippet_num_frames(
        duration_s=spike_band_duration_s,
        sampling_frequency=sampling_frequency,
        total_frames=total_frames,
    )
    broadband_start_frame = start_frame
    broadband_end_frame = int(broadband_start_frame + broadband_num_frames)
    spike_band_start_frame = start_frame
    spike_band_end_frame = int(spike_band_start_frame + spike_band_num_frames)
    if broadband_end_frame > total_frames:
        raise ValueError(
            "Requested broadband snippet extends beyond the recording bounds: "
            f"end frame {broadband_end_frame}, total frames {total_frames}."
        )
    if spike_band_end_frame > total_frames:
        raise ValueError(
            "Requested spike-band snippet extends beyond the recording bounds: "
            f"end frame {spike_band_end_frame}, total frames {total_frames}."
        )
    raw_output_path, filtered_output_path = get_output_paths(
        animal_name=animal_name,
        date=date,
        start_time_s=resolved_start_time_s,
        broadband_duration_s=broadband_duration_s,
        spike_band_duration_s=spike_band_duration_s,
        analysis_root=analysis_root,
    )
    raw_traces = np.asarray(
        recording.get_traces(
            start_frame=broadband_start_frame,
            end_frame=broadband_end_frame,
            segment_index=0,
        ),
        dtype=float,
    )

    recording_filtered = si.bandpass_filter(
        recording,
        freq_min=FILTER_FREQ_MIN_HZ,
        freq_max=FILTER_FREQ_MAX_HZ,
        dtype=np.float64,
    )
    filtered_traces = np.asarray(
        recording_filtered.get_traces(
            start_frame=spike_band_start_frame,
            end_frame=spike_band_end_frame,
            segment_index=0,
        ),
        dtype=float,
    )

    plot_voltage_figure(
        traces=raw_traces,
        channel_metadata=channel_metadata,
        probe_indices=probe_indices,
        sampling_frequency=sampling_frequency,
        start_time_s=resolved_start_time_s,
        duration_s=broadband_duration_s,
        output_path=raw_output_path,
        filtered=False,
    )
    plot_voltage_figure(
        traces=filtered_traces,
        channel_metadata=channel_metadata,
        probe_indices=probe_indices,
        sampling_frequency=sampling_frequency,
        start_time_s=resolved_start_time_s,
        duration_s=spike_band_duration_s,
        output_path=filtered_output_path,
        filtered=True,
    )

    print(f"Saved raw snippet figure to {raw_output_path}")
    print(f"Saved filtered snippet figure to {filtered_output_path}")
    return {
        "analysis_path": analysis_path,
        "nwb_path": nwb_path,
        "raw_figure_path": raw_output_path,
        "filtered_figure_path": filtered_output_path,
        "resolved_start_time_s": resolved_start_time_s,
        "start_time_source": selection["start_time_source"],
        "selected_run_epoch": selection["selected_run_epoch"],
        "epoch_timestamp_source": selection["epoch_timestamp_source"],
        "epoch_relative_start_index": selection["epoch_relative_start_index"],
        "start_frame": start_frame,
        "end_frame": int(selection["end_frame"]),
        "selection_duration_s": selection_duration_s,
        "broadband_duration_s": broadband_duration_s,
        "spike_band_duration_s": spike_band_duration_s,
        "broadband_end_frame": broadband_end_frame,
        "spike_band_end_frame": spike_band_end_frame,
        "sampling_frequency_hz": sampling_frequency,
        "probe_indices": probe_indices,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for voltage snippet plotting."""
    parser = argparse.ArgumentParser(
        description="Plot raw and bandpass-filtered NWB voltage snippets by probe and shank"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
    parser.add_argument(
        "--epoch",
        help=(
            "Epoch tag to sample from when `--start-time-s` is omitted. "
            "If omitted, the script chooses a random chunk from the first run epoch."
        ),
    )
    parser.add_argument(
        "--start-time-s",
        type=float,
        help=(
            "Snippet start time in seconds from the start of the recording. "
            "If provided, this overrides `--epoch`."
        ),
    )
    parser.add_argument(
        "--broadband-duration-s",
        type=float,
        default=DEFAULT_BROADBAND_DURATION_S,
        help=(
            "Broadband snippet duration in seconds. "
            f"Default: {DEFAULT_BROADBAND_DURATION_S}"
        ),
    )
    parser.add_argument(
        "--spike-band-duration-s",
        type=float,
        default=DEFAULT_SPIKE_BAND_DURATION_S,
        help=(
            "Spike-band snippet duration in seconds. "
            f"Default: {DEFAULT_SPIKE_BAND_DURATION_S}"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed used when `--start-time-s` is omitted. "
            f"Default: {DEFAULT_RANDOM_SEED}"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help=f"Base directory for session analysis data. Default: {DEFAULT_ANALYSIS_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint."""
    args = parse_arguments()
    outputs = plot_voltage_snippet(
        animal_name=args.animal_name,
        date=args.date,
        start_time_s=args.start_time_s,
        epoch=args.epoch,
        broadband_duration_s=args.broadband_duration_s,
        spike_band_duration_s=args.spike_band_duration_s,
        analysis_root=args.data_root,
        nwb_root=args.nwb_root,
        random_seed=args.random_seed,
    )
    log_path = write_run_log(
        analysis_path=outputs["analysis_path"],
        script_name="v1ca1.nwb.plot_voltage_snippet",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "requested_start_time_s": args.start_time_s,
            "requested_epoch": args.epoch,
            "broadband_duration_s": args.broadband_duration_s,
            "spike_band_duration_s": args.spike_band_duration_s,
            "data_root": args.data_root,
            "nwb_root": args.nwb_root,
            "random_seed": int(args.random_seed),
            "filter_freq_min_hz": FILTER_FREQ_MIN_HZ,
            "filter_freq_max_hz": FILTER_FREQ_MAX_HZ,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
