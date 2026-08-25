"""Plot multiday waveforms, ISIs, and light/dark path tuning for one unit."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.wtrack import get_wtrack_segment_edges
from v1ca1.multiday.compute_tuning_curves import (
    DEFAULT_OUTPUT_DIRNAME as DEFAULT_TUNING_CURVE_DIRNAME,
    get_tuning_curve_path,
)
from v1ca1.paper_figures.style import (
    EPOCH_TYPE_COLORS,
    RASTER_TICK_KWARGS,
    apply_paper_style,
    save_figure,
)


DEFAULT_ANALYSIS_PATH = Path(
    "/stelmo/kyu/analysis/L14/"
    "20240605_20240606_20240607_20240609_20240611"
)
DEFAULT_OUTPUT_DIR = Path("/home/kyu/repos/v1ca1/paper_figures")
DEFAULT_OUTPUT_NAME = "multiday_L14_v1_unit21"
DEFAULT_ANIMAL_NAME = "L14"
DEFAULT_REGION = "v1"
DEFAULT_UNIT_ID = 21
DEFAULT_LIGHT_EPOCH = "02_r1"
DEFAULT_DARK_EPOCH = "08_r4"
DEFAULT_WAVEFORM_EPOCH = DEFAULT_LIGHT_EPOCH
DEFAULT_DATES = (
    "20240605",
    "20240606",
    "20240607",
    "20240609",
    "20240611",
)
DEFAULT_DAY_NUMBER_BY_DATE = {
    "20240605": 0,
    "20240606": 1,
    "20240607": 2,
    "20240609": 4,
    "20240611": 7,
}
DEFAULT_TRAJECTORIES = ("center_to_left", "right_to_center")
DEFAULT_POSITION_OFFSET = 10
DEFAULT_N_WAVEFORM_CHANNELS = 3
DEFAULT_ISI_BIN_SIZE_S = 0.001
DEFAULT_ISI_MAX_S = 0.050
DEFAULT_TUNING_SMOOTHING_SIGMA_BINS = 1.0
DEFAULT_DPI = 600
DEFAULT_FORMATS = ("png",)
DEFAULT_FIGURE_WIDTH_IN = 5.7
DEFAULT_FIGURE_HEIGHT_IN = 2.25
SEGMENT_BOUNDARIES = tuple(
    float(value)
    for value in get_wtrack_segment_edges(DEFAULT_ANIMAL_NAME)[1:-1]
)
TUNING_EPOCH_ORDER = ("dark", "light")
DATE_COLORMAP = "viridis"
DATE_COLORMAP_RANGE = (0.08, 0.82)
SPIKE_READ_CHUNK_SIZE = 5_000_000


def _get_day_numbers(dates: Sequence[str]) -> np.ndarray:
    """Return configured study days or elapsed calendar days as a fallback."""
    dates = tuple(str(date) for date in dates)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("dates must contain unique identifiers.")
    if all(date in DEFAULT_DAY_NUMBER_BY_DATE for date in dates):
        return np.asarray(
            [DEFAULT_DAY_NUMBER_BY_DATE[date] for date in dates],
            dtype=float,
        )
    try:
        parsed_dates = tuple(
            datetime.strptime(date, "%Y%m%d") for date in dates
        )
    except ValueError:
        return np.arange(len(dates), dtype=float)
    first_date = min(parsed_dates)
    return np.asarray(
        [(date - first_date).days for date in parsed_dates],
        dtype=float,
    )


def _build_date_colors(dates: Sequence[str]) -> dict[str, Any]:
    """Map dates onto a sequential colormap using study-day positions."""
    from matplotlib import colormaps

    dates = tuple(str(date) for date in dates)
    day_numbers = _get_day_numbers(dates)
    day_range = float(np.max(day_numbers) - np.min(day_numbers))
    normalized_offsets = (
        np.zeros_like(day_numbers)
        if day_range == 0.0
        else (day_numbers - np.min(day_numbers)) / day_range
    )
    lower, upper = DATE_COLORMAP_RANGE
    color_positions = lower + (upper - lower) * normalized_offsets
    colormap = colormaps.get_cmap(DATE_COLORMAP)
    return {
        date: colormap(float(position))
        for date, position in zip(dates, color_positions, strict=True)
    }


def _build_day_labels(dates: Sequence[str]) -> dict[str, str]:
    """Return concise study-day legend labels for each date."""
    dates = tuple(str(date) for date in dates)
    day_numbers = _get_day_numbers(dates)
    return {
        date: f"Day {int(day_number)}"
        for date, day_number in zip(dates, day_numbers, strict=True)
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and report a missing artifact clearly."""
    if not path.exists():
        raise FileNotFoundError(f"Missing multiday artifact: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_pickle(path: Path) -> Any:
    """Load one legacy multiday pickle and report a missing artifact clearly."""
    if not path.exists():
        raise FileNotFoundError(f"Missing multiday artifact: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def _sorting_unit_index(
    analysis_path: Path,
    region: str,
    unit_id: int,
) -> int:
    """Return the zero-based sorting index for one displayed unit ID."""
    info = _load_json(
        analysis_path / f"sorting_{region}" / "numpysorting_info.json"
    )
    unit_ids = [int(value) for value in info["unit_ids"]]
    try:
        return unit_ids.index(int(unit_id))
    except ValueError as error:
        raise ValueError(
            f"Unit {unit_id} is absent from sorting_{region}; "
            f"available IDs span {unit_ids[:3]}...{unit_ids[-3:]}."
        ) from error


def _probe_shank_unit(
    analysis_path: Path,
    *,
    probe_index: int,
    global_unit_index: int,
) -> tuple[int, int, int]:
    """Map a concatenated V1 unit index to shank, local index, and sorter ID."""
    shank_paths = sorted(
        analysis_path.glob(f"curated_sorting_probe{probe_index}_shank*"),
        key=lambda path: int(path.name.rsplit("shank", maxsplit=1)[1]),
    )
    if not shank_paths:
        raise FileNotFoundError(
            f"No curated probe {probe_index} shank sortings under {analysis_path}."
        )

    remaining_index = int(global_unit_index)
    for shank_path in shank_paths:
        shank_index = int(shank_path.name.rsplit("shank", maxsplit=1)[1])
        info = _load_json(shank_path / "numpysorting_info.json")
        unit_ids = [int(value) for value in info["unit_ids"]]
        if remaining_index < len(unit_ids):
            return shank_index, remaining_index, unit_ids[remaining_index]
        remaining_index -= len(unit_ids)

    raise ValueError(
        f"Global unit index {global_unit_index} exceeds the units on probe "
        f"{probe_index}."
    )


def _extract_unit_spike_frames(
    spikes_path: Path,
    unit_index: int,
    *,
    chunk_size: int = SPIKE_READ_CHUNK_SIZE,
) -> np.ndarray:
    """Read only one unit's sample indices from a NumPySorting artifact."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not spikes_path.exists():
        raise FileNotFoundError(f"Missing multiday artifact: {spikes_path}")

    spikes = np.load(spikes_path, mmap_mode="r")
    required_fields = {"sample_index", "unit_index"}
    observed_fields = set(spikes.dtype.names or ())
    if not required_fields.issubset(observed_fields):
        raise ValueError(
            f"{spikes_path} lacks NumPySorting fields {sorted(required_fields)}."
        )

    frame_blocks: list[np.ndarray] = []
    for start in range(0, len(spikes), chunk_size):
        block = spikes[start : start + chunk_size]
        selected = block["unit_index"] == int(unit_index)
        if np.any(selected):
            frame_blocks.append(
                np.asarray(block["sample_index"][selected], dtype=np.int64)
            )
    if not frame_blocks:
        return np.array([], dtype=np.int64)
    return np.concatenate(frame_blocks)


def load_unit_spike_times_by_date(
    *,
    analysis_path: Path,
    region: str,
    unit_id: int,
    dates: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return absolute spike times for one unit, split across requested days."""
    analysis_path = Path(analysis_path)
    requested_dates = tuple(str(date) for date in dates)
    if not requested_dates or len(set(requested_dates)) != len(requested_dates):
        raise ValueError("dates must contain unique date identifiers.")

    unit_index = _sorting_unit_index(analysis_path, region, unit_id)
    spike_frames = _extract_unit_spike_frames(
        analysis_path / f"sorting_{region}" / "spikes.npy",
        unit_index,
    )
    timestamps_by_date = _load_pickle(analysis_path / "timestamps_ephys_all.pkl")
    if not isinstance(timestamps_by_date, Mapping):
        raise ValueError("timestamps_ephys_all.pkl must contain a date mapping.")

    output: dict[str, np.ndarray] = {}
    sample_offset = 0
    for date, timestamps in timestamps_by_date.items():
        date = str(date)
        timestamps = np.asarray(timestamps, dtype=float)
        next_offset = sample_offset + timestamps.size
        if date in requested_dates:
            start = int(np.searchsorted(spike_frames, sample_offset, side="left"))
            stop = int(np.searchsorted(spike_frames, next_offset, side="left"))
            local_frames = spike_frames[start:stop] - sample_offset
            output[date] = np.asarray(timestamps[local_frames], dtype=float)
        sample_offset = next_offset

    del timestamps_by_date
    gc.collect()
    missing_dates = [date for date in requested_dates if date not in output]
    if missing_dates:
        raise ValueError(
            "Requested dates are absent from timestamps_ephys_all.pkl: "
            f"{missing_dates!r}."
        )
    if spike_frames.size and int(spike_frames[-1]) >= sample_offset:
        raise ValueError(
            "The concatenated sorting extends beyond timestamps_ephys_all.pkl."
        )
    return {date: output[date] for date in requested_dates}


def compute_isi_distributions(
    spike_times_by_date: Mapping[str, np.ndarray],
    *,
    bin_size_s: float = DEFAULT_ISI_BIN_SIZE_S,
    max_isi_s: float = DEFAULT_ISI_MAX_S,
) -> dict[str, Any]:
    """Return per-day fractions of ISIs in fixed-width bins."""
    if not np.isfinite(bin_size_s) or bin_size_s <= 0.0:
        raise ValueError("bin_size_s must be positive and finite.")
    if not np.isfinite(max_isi_s) or max_isi_s <= bin_size_s:
        raise ValueError("max_isi_s must be finite and greater than bin_size_s.")
    bin_edges_s = np.arange(
        0.0,
        max_isi_s + 0.5 * bin_size_s,
        bin_size_s,
    )
    fractions: dict[str, np.ndarray] = {}
    n_intervals: dict[str, int] = {}
    for date, spike_times in spike_times_by_date.items():
        spike_times = np.asarray(spike_times, dtype=float)
        spike_times = spike_times[np.isfinite(spike_times)]
        spike_times.sort()
        intervals = np.diff(spike_times)
        intervals = intervals[intervals >= 0.0]
        counts, _ = np.histogram(intervals, bins=bin_edges_s)
        fractions[str(date)] = (
            counts.astype(float) / intervals.size
            if intervals.size
            else np.zeros(counts.shape, dtype=float)
        )
        n_intervals[str(date)] = int(intervals.size)
    return {
        "bin_centers_ms": 1_000.0
        * 0.5
        * (bin_edges_s[:-1] + bin_edges_s[1:]),
        "fractions": fractions,
        "n_intervals": n_intervals,
        "bin_size_s": float(bin_size_s),
        "max_isi_s": float(max_isi_s),
    }


def load_waveform_data(
    *,
    analysis_path: Path,
    region: str,
    unit_id: int,
    dates: Sequence[str],
    epoch: str,
    reference_date: str,
    probe_index: int = 0,
    n_channels: int = DEFAULT_N_WAVEFORM_CHANNELS,
) -> dict[str, Any]:
    """Load one unit's top-channel mean templates for every requested day."""
    if region != "v1":
        raise ValueError(
            "This multiday waveform layout currently maps V1 probe 0 only."
        )
    if n_channels <= 0:
        raise ValueError("n_channels must be positive.")

    analysis_path = Path(analysis_path)
    dates = tuple(str(date) for date in dates)
    if reference_date not in dates:
        raise ValueError("reference_date must be one of dates.")
    global_unit_index = _sorting_unit_index(analysis_path, region, unit_id)
    shank_index, local_unit_index, sorter_unit_id = _probe_shank_unit(
        analysis_path,
        probe_index=probe_index,
        global_unit_index=global_unit_index,
    )

    def analyzer_path(date: str) -> Path:
        return (
            analysis_path
            / "waveforms"
            / f"waveforms_probe{probe_index}_shank{shank_index}"
            / f"{date}_{epoch}"
        )

    reference_path = analyzer_path(reference_date)
    reference_info = _load_json(
        reference_path / "sorting" / "numpysorting_info.json"
    )
    reference_unit_ids = [int(value) for value in reference_info["unit_ids"]]
    if sorter_unit_id not in reference_unit_ids:
        raise ValueError(
            f"Sorter unit {sorter_unit_id} is absent from {reference_path}."
        )
    template_unit_index = reference_unit_ids.index(sorter_unit_id)
    reference_templates = np.load(
        reference_path / "extensions" / "templates" / "average.npy"
    )
    amplitudes = np.nanmax(
        np.abs(reference_templates[template_unit_index]),
        axis=0,
    )
    channel_indices = np.argsort(-amplitudes)[:n_channels]

    recording_info = _load_json(
        reference_path / "recording_info" / "recording_attributes.json"
    )
    channel_ids = np.asarray(recording_info["channel_ids"])[channel_indices]
    sampling_frequency = float(recording_info["sampling_frequency"])
    template_params = _load_json(
        reference_path / "extensions" / "templates" / "params.json"
    )
    ms_before = float(template_params["ms_before"])
    n_samples = int(reference_templates.shape[1])
    time_ms = np.arange(n_samples, dtype=float) / sampling_frequency * 1_000.0
    time_ms -= ms_before

    waveforms: dict[str, np.ndarray] = {}
    for date in dates:
        current_path = analyzer_path(date)
        current_info = _load_json(
            current_path / "sorting" / "numpysorting_info.json"
        )
        current_unit_ids = [int(value) for value in current_info["unit_ids"]]
        if sorter_unit_id not in current_unit_ids:
            raise ValueError(
                f"Sorter unit {sorter_unit_id} is absent from {current_path}."
            )
        current_unit_index = current_unit_ids.index(sorter_unit_id)
        templates = np.load(
            current_path / "extensions" / "templates" / "average.npy"
        )
        if templates.shape[1:] != reference_templates.shape[1:]:
            raise ValueError(
                "Waveform template shapes differ across dates: "
                f"{templates.shape[1:]} vs {reference_templates.shape[1:]}."
            )
        waveforms[date] = np.asarray(
            templates[current_unit_index, :, channel_indices],
            dtype=float,
        ).T

    return {
        "probe_index": probe_index,
        "shank_index": shank_index,
        "local_unit_index": local_unit_index,
        "sorter_unit_id": sorter_unit_id,
        "time_ms": time_ms,
        "channel_ids": tuple(int(value) for value in channel_ids),
        "waveforms": waveforms,
    }


def _classifier_path(
    analysis_path: Path,
    *,
    region: str,
    date: str,
    epoch: str,
    trajectory: str,
) -> Path:
    """Return the legacy trajectory-classifier path used by the source plots."""
    return (
        analysis_path
        / "classifier_1d_trajectory"
        / (
            f"classifier_{region}_{date}_{epoch}_1d_{trajectory}"
            "_movement_fit_position_std_4.0_discrete_var_switching"
            "_place_bin_size_1.0_movement_var_4.0.pkl"
        )
    )


def _orient_task_progression(
    position: np.ndarray,
    values: np.ndarray,
    trajectory: str,
    *,
    total_length_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and orient a path so zero is always its starting well."""
    progression = np.asarray(position, dtype=float) / float(total_length_cm)
    values = np.asarray(values, dtype=float)
    if trajectory.endswith("_to_center"):
        progression = 1.0 - progression
    order = np.argsort(progression)
    return progression[order], values[order]


def load_tuning_curves(
    *,
    analysis_path: Path,
    animal_name: str,
    region: str,
    unit_id: int,
    dates: Sequence[str],
    epoch: str,
    trajectories: Sequence[str],
    tuning_curve_dir: Path | None = None,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]]:
    """Load Pynapple task-progression tuning curves for the requested unit."""
    import xarray as xr

    analysis_path = Path(analysis_path)
    tuning_curve_dir = (
        analysis_path / DEFAULT_TUNING_CURVE_DIRNAME
        if tuning_curve_dir is None
        else Path(tuning_curve_dir)
    )
    output: dict[
        str,
        dict[str, tuple[np.ndarray, np.ndarray] | None],
    ] = {}
    for date in dates:
        output[str(date)] = {}
        for trajectory in trajectories:
            path = get_tuning_curve_path(
                tuning_curve_dir,
                region=region,
                date=str(date),
                epoch=epoch,
                trajectory=str(trajectory),
            )
            if not path.exists():
                raise FileNotFoundError(
                    "Missing Pynapple multiday tuning curve: "
                    f"{path}. Run `python -m "
                    "v1ca1.multiday.compute_tuning_curves` first."
                )
            with xr.open_dataarray(path) as tuning_curve:
                for key, expected in (
                    ("animal_name", animal_name),
                    ("date", str(date)),
                    ("region", region),
                    ("epoch", epoch),
                    ("trajectory_type", str(trajectory)),
                ):
                    observed = str(tuning_curve.attrs.get(key, ""))
                    if observed != str(expected):
                        raise ValueError(
                            f"{path} has {key}={observed!r}; expected "
                            f"{str(expected)!r}."
                        )
                unit_values = np.asarray(
                    tuning_curve.coords["unit"],
                    dtype=int,
                )
                if int(unit_id) not in unit_values:
                    raise ValueError(
                        f"Unit {unit_id} is absent from {path}; available "
                        f"IDs span {unit_values[:3]}...{unit_values[-3:]}."
                    )
                position = np.asarray(
                    tuning_curve.coords["tp"],
                    dtype=float,
                )
                rate_hz = np.asarray(
                    tuning_curve.sel(unit=int(unit_id)),
                    dtype=float,
                )
                n_trials = int(tuning_curve.attrs.get("n_trials", 0))
            if n_trials == 0 or not np.isfinite(rate_hz).any():
                output[str(date)][str(trajectory)] = None
                continue
            if position.shape != rate_hz.shape:
                raise ValueError(
                    f"Tuning coordinate and values differ in {path}: "
                    f"{position.shape} vs {rate_hz.shape}."
                )
            output[str(date)][str(trajectory)] = (position, rate_hz)
    return output


def load_raster_positions(
    *,
    analysis_path: Path,
    animal_name: str,
    dates: Sequence[str],
    epoch: str,
    trajectories: Sequence[str],
    spike_times_by_date: Mapping[str, np.ndarray],
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Linearize this unit's trial spikes and orient them as task progression."""
    from scipy.interpolate import interp1d
    import track_linearization as tl

    from v1ca1.helper.wtrack import get_wtrack_total_length
    from v1ca1.raster.plot_1d_place_field_trajectory import (
        get_branch_aligned_track_graph,
    )

    if position_offset < 0:
        raise ValueError("position_offset must be non-negative.")
    analysis_path = Path(analysis_path)
    timestamps_position = _load_pickle(analysis_path / "timestamps_position.pkl")
    trajectory_times = _load_pickle(analysis_path / "trajectory_times.pkl")
    total_length_cm = float(get_wtrack_total_length(animal_name))

    output: dict[str, dict[str, list[np.ndarray]]] = {}
    for date in dates:
        date = str(date)
        position = np.asarray(
            _load_pickle(
                analysis_path / "position" / f"{date}_{animal_name}_{epoch}.pkl"
            ),
            dtype=float,
        )
        timestamps = np.asarray(timestamps_position[date][epoch], dtype=float)
        if min(position.shape[0], timestamps.size) <= position_offset:
            raise ValueError(
                f"Position offset {position_offset} removes all {date} samples."
            )
        position = position[position_offset:]
        timestamps = timestamps[position_offset:]
        spike_times = np.asarray(spike_times_by_date[date], dtype=float)
        output[date] = {}

        for trajectory in trajectories:
            trajectory = str(trajectory)
            track_graph, edge_order = get_branch_aligned_track_graph(
                animal_name,
                trajectory,
            )
            linear_position = tl.get_linearized_position(
                position=position,
                track_graph=track_graph,
                edge_order=edge_order,
                edge_spacing=0,
            )["linear_position"].to_numpy(dtype=float)
            interpolator = interp1d(
                timestamps,
                linear_position,
                kind="linear",
                bounds_error=False,
                assume_sorted=True,
            )
            trials: list[np.ndarray] = []
            for start, end in trajectory_times[date][epoch][trajectory]:
                lower = int(np.searchsorted(spike_times, float(start), side="right"))
                upper = int(np.searchsorted(spike_times, float(end), side="right"))
                positions = np.asarray(
                    interpolator(spike_times[lower:upper]),
                    dtype=float,
                )
                valid = np.isfinite(positions)
                positions = positions[valid]
                positions = positions[
                    (positions >= 0.0) & (positions <= total_length_cm)
                ]
                progression = positions / total_length_cm
                if trajectory.endswith("_to_center"):
                    progression = 1.0 - progression
                trials.append(progression)
            output[date][trajectory] = trials
    return output


def load_figure_data(
    *,
    analysis_path: Path,
    animal_name: str,
    region: str,
    unit_id: int,
    dates: Sequence[str],
    waveform_epoch: str,
    light_epoch: str,
    dark_epoch: str,
    trajectories: Sequence[str] = DEFAULT_TRAJECTORIES,
    reference_date: str | None = None,
    n_waveform_channels: int = DEFAULT_N_WAVEFORM_CHANNELS,
    tuning_curve_dir: Path | None = None,
    tuning_smoothing_sigma_bins: float = (
        DEFAULT_TUNING_SMOOTHING_SIGMA_BINS
    ),
) -> dict[str, Any]:
    """Load every data component needed for the multiday unit figure."""
    dates = tuple(str(date) for date in dates)
    trajectories = tuple(str(value) for value in trajectories)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("dates must contain unique date identifiers.")
    if trajectories != DEFAULT_TRAJECTORIES:
        raise ValueError(
            "This figure requires center_to_left and right_to_center, in that order."
        )
    if (
        not np.isfinite(tuning_smoothing_sigma_bins)
        or tuning_smoothing_sigma_bins < 0.0
    ):
        raise ValueError(
            "tuning_smoothing_sigma_bins must be non-negative and finite."
        )
    reference_date = dates[0] if reference_date is None else str(reference_date)
    waveform_data = load_waveform_data(
        analysis_path=analysis_path,
        region=region,
        unit_id=unit_id,
        dates=dates,
        epoch=waveform_epoch,
        reference_date=reference_date,
        n_channels=n_waveform_channels,
    )
    epoch_ids = {"dark": str(dark_epoch), "light": str(light_epoch)}
    tuning_curves = {
        condition: load_tuning_curves(
            analysis_path=analysis_path,
            animal_name=animal_name,
            region=region,
            unit_id=unit_id,
            dates=dates,
            epoch=epoch_ids[condition],
            trajectories=trajectories,
            tuning_curve_dir=tuning_curve_dir,
        )
        for condition in TUNING_EPOCH_ORDER
    }
    spike_times_by_date = load_unit_spike_times_by_date(
        analysis_path=analysis_path,
        region=region,
        unit_id=unit_id,
        dates=dates,
    )
    return {
        "analysis_path": Path(analysis_path),
        "animal_name": animal_name,
        "region": region,
        "unit_id": int(unit_id),
        "dates": dates,
        "waveform_epoch": str(waveform_epoch),
        "epoch_ids": epoch_ids,
        "trajectories": trajectories,
        "waveform_data": waveform_data,
        "isi_distributions": compute_isi_distributions(
            spike_times_by_date
        ),
        "tuning_curves": tuning_curves,
        "tuning_smoothing_sigma_bins": float(
            tuning_smoothing_sigma_bins
        ),
    }


def _add_segment_boundaries(ax: Any) -> None:
    """Add the task-segment guides used by Condensed2."""
    for boundary in SEGMENT_BOUNDARIES:
        ax.axvline(
            boundary,
            color="#A6A6A6",
            linewidth=0.45,
            zorder=1,
        )


def _style_data_axis(ax: Any) -> None:
    """Apply the compact open-axis style used by Condensed2."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=1.8, pad=1.0)


def _plot_raster_axis(
    ax: Any,
    trial_positions: Sequence[np.ndarray],
    *,
    color: str,
    show_ylabel: bool,
) -> None:
    """Plot one day's trial rasters on normalized task progression."""
    for trial_index, positions in enumerate(trial_positions, start=1):
        positions = np.asarray(positions, dtype=float)
        if positions.size:
            ax.plot(
                positions,
                np.full(positions.shape, trial_index, dtype=float),
                "|",
                color=color,
                **RASTER_TICK_KWARGS,
            )
    _add_segment_boundaries(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.5, max(1.5, len(trial_positions) + 0.5))
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([])
    ax.set_yticks([])
    if show_ylabel:
        ax.set_ylabel("Trials", labelpad=2.0)
    _style_data_axis(ax)


def _plot_rate_axis(
    ax: Any,
    position: np.ndarray,
    rate_hz: np.ndarray,
    *,
    color: str,
    y_max: float,
    show_ylabel: bool,
) -> None:
    """Plot one day's tuning curve in the Condensed2 style."""
    ax.plot(position, rate_hz, color=color, linewidth=1.0, zorder=3)
    _add_segment_boundaries(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, y_max])
    ax.set_yticklabels(["0", f"{y_max:g}"])
    if show_ylabel:
        ax.set_ylabel("FR (Hz)", labelpad=2.0)
    else:
        ax.set_yticklabels([])
    _style_data_axis(ax)


def _smooth_tuning_rate(
    rate_hz: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    """Gaussian-smooth one tuning curve for display without changing artifacts."""
    from scipy.ndimage import gaussian_filter1d

    values = np.asarray(rate_hz, dtype=float)
    if values.ndim != 1:
        raise ValueError("Tuning rates must be one-dimensional.")
    if not np.isfinite(sigma_bins) or sigma_bins < 0.0:
        raise ValueError("sigma_bins must be non-negative and finite.")
    if sigma_bins == 0.0 or values.size < 2:
        return values.copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()
    filled = values.copy()
    if not np.all(finite):
        indices = np.arange(values.size, dtype=float)
        filled[~finite] = np.interp(
            indices[~finite],
            indices[finite],
            values[finite],
        )
    return gaussian_filter1d(filled, sigma=float(sigma_bins), mode="nearest")


def make_figure(data: Mapping[str, Any]) -> Any:
    """Build the multiday unit figure without writing it to disk."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic

    apply_paper_style()
    dates = tuple(data["dates"])
    if not dates:
        raise ValueError("Figure data must contain at least one date.")
    trajectories = tuple(data["trajectories"])
    if trajectories != DEFAULT_TRAJECTORIES:
        raise ValueError(
            "Figure data must contain center_to_left and right_to_center."
        )

    smoothing_sigma_bins = float(
        data.get(
            "tuning_smoothing_sigma_bins",
            DEFAULT_TUNING_SMOOTHING_SIGMA_BINS,
        )
    )
    tuning_rates = []
    for trajectory in trajectories:
        for condition in TUNING_EPOCH_ORDER:
            for date in dates:
                curve = data["tuning_curves"][condition][date][trajectory]
                if curve is not None:
                    tuning_rates.append(
                        _smooth_tuning_rate(curve[1], smoothing_sigma_bins)
                    )
    finite_rate_maxima = [
        float(np.nanmax(values))
        for values in tuning_rates
        if np.isfinite(values).any()
    ]
    rate_y_max = (
        1.0
        if not finite_rate_maxima
        else max(1.0, 5.0 * np.ceil(max(finite_rate_maxima) / 5.0))
    )
    waveform_values = np.concatenate(
        [
            np.asarray(data["waveform_data"]["waveforms"][date], dtype=float).ravel()
            for date in dates
        ]
    )
    waveform_min = float(np.floor(np.nanmin(waveform_values) / 25.0) * 25.0)
    waveform_max = float(np.ceil(np.nanmax(waveform_values) / 25.0) * 25.0)
    date_colors = _build_date_colors(dates)
    day_labels = _build_day_labels(dates)
    fig = plt.figure(
        figsize=(DEFAULT_FIGURE_WIDTH_IN, DEFAULT_FIGURE_HEIGHT_IN),
    )
    outer_grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=(2.0, 8.0),
        left=0.070,
        right=0.980,
        bottom=0.175,
        top=0.900,
        wspace=0.22,
    )
    left_grid = outer_grid[0].subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=(2.0, 1.0),
        hspace=0.75,
    )
    waveform_grid = left_grid[0].subgridspec(
        nrows=1,
        ncols=len(data["waveform_data"]["channel_ids"]),
        wspace=0.45,
    )
    tuning_grid = outer_grid[1].subgridspec(
        nrows=2,
        ncols=len(dates) + 2,
        width_ratios=(0.40, 0.28, *(1.0 for _ in dates)),
        hspace=0.42,
        wspace=0.32,
    )

    time_ms = np.asarray(data["waveform_data"]["time_ms"], dtype=float)
    channel_ids = tuple(data["waveform_data"]["channel_ids"])
    waveform_axes = [
        fig.add_subplot(waveform_grid[0, channel_index])
        for channel_index in range(len(channel_ids))
    ]
    for channel_index, (waveform_ax, channel_id) in enumerate(
        zip(waveform_axes, channel_ids, strict=True)
    ):
        for date in dates:
            waveforms = np.asarray(
                data["waveform_data"]["waveforms"][date],
                dtype=float,
            )
            waveform_ax.plot(
                time_ms,
                waveforms[:, channel_index],
                color=date_colors[date],
                linewidth=1.0,
            )
        waveform_ax.axhline(0.0, color="0.85", linewidth=0.4, zorder=0)
        waveform_ax.set_xlim(-1.0, 1.0)
        waveform_ax.set_ylim(waveform_min, waveform_max)
        waveform_ax.set_title(
            (
                f"Ch. {channel_id}"
                if len(channel_ids) > 2
                else f"Channel {channel_id}"
            ),
            fontsize=5.5 if len(channel_ids) > 2 else 6.5,
            pad=2.0,
        )
        waveform_ax.set_xticks([-1.0, 0.0, 1.0])
        if channel_index > 0:
            waveform_ax.set_yticklabels([])
        _style_data_axis(waveform_ax)

    waveform_left = waveform_axes[0].get_position().x0
    waveform_right = waveform_axes[-1].get_position().x1
    waveform_bottom = min(axis.get_position().y0 for axis in waveform_axes)
    fig.text(
        0.5 * (waveform_left + waveform_right),
        waveform_bottom - 0.080,
        "Time (ms)",
        ha="center",
        va="top",
    )
    waveform_label_y = 0.5 * (
        waveform_axes[0].get_position().y1
        + waveform_axes[-1].get_position().y0
    )
    fig.text(
        waveform_axes[0].get_position().x0 - 0.047,
        waveform_label_y,
        "Amplitude (µV)",
        ha="center",
        va="center",
        rotation=90,
    )

    isi_axis = fig.add_subplot(left_grid[1, 0])
    isi_data = data["isi_distributions"]
    isi_bin_centers_ms = np.asarray(
        isi_data["bin_centers_ms"],
        dtype=float,
    )
    for date in dates:
        isi_axis.plot(
            isi_bin_centers_ms,
            np.asarray(isi_data["fractions"][date], dtype=float),
            color=date_colors[date],
            linewidth=1.0,
        )
    isi_axis.set_xlim(0.0, 1_000.0 * float(isi_data["max_isi_s"]))
    isi_axis.set_ylim(bottom=0.0)
    isi_axis.set_xlabel("ISI (ms)", labelpad=2.0)
    isi_axis.set_ylabel("Fraction", labelpad=2.0)
    _style_data_axis(isi_axis)

    tuning_axes = np.empty((len(trajectories), len(dates)), dtype=object)
    for row, trajectory in enumerate(trajectories):
        for column, date in enumerate(dates):
            ax = fig.add_subplot(tuning_grid[row, column + 2])
            tuning_axes[row, column] = ax
            for condition in TUNING_EPOCH_ORDER:
                curve = data["tuning_curves"][condition][date][trajectory]
                if curve is None:
                    continue
                position, rate_hz = curve
                ax.plot(
                    np.asarray(position, dtype=float),
                    _smooth_tuning_rate(
                        rate_hz,
                        smoothing_sigma_bins,
                    ),
                    color=EPOCH_TYPE_COLORS[condition],
                    linewidth=1.0,
                    zorder=3,
                )
            _add_segment_boundaries(ax)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, rate_y_max)
            ax.set_xticks([0.0, 1.0])
            ax.set_yticks([0.0, rate_y_max])
            if column == 0:
                ax.set_ylabel("FR (Hz)", labelpad=2.0)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(day_labels[date], fontsize=6.5, pad=2.0)
                ax.set_xticklabels([])
            _style_data_axis(ax)

    trajectory_axes = []
    for row, trajectory in enumerate(trajectories):
        trajectory_axis = fig.add_subplot(tuning_grid[row, 0])
        draw_w_track_schematic(
            trajectory_axis,
            trajectory_name=trajectory,
            track_linewidth=0.35,
            trajectory_linewidth=1.4,
            arrow_mutation_scale=7.5,
            fill_track=False,
        )
        trajectory_axes.append(trajectory_axis)

    tuning_bottom = min(ax.get_position().y0 for ax in tuning_axes[-1])
    tuning_left = tuning_axes[0, 0].get_position().x0
    tuning_right = tuning_axes[0, -1].get_position().x1
    fig.text(
        0.5 * (tuning_left + tuning_right),
        tuning_bottom - 0.080,
        "Normalized path progression",
        ha="center",
        va="top",
        fontsize=7.0,
    )

    panel_a_left = waveform_axes[0].get_position().x0
    panel_a_top = waveform_axes[0].get_position().y1
    fig.text(
        panel_a_left - 0.035,
        panel_a_top + 0.045,
        "A",
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
    )
    fig.text(
        0.5 * (waveform_left + waveform_right),
        panel_a_top + 0.045,
        "Mean waveforms",
        fontsize=7.0,
        ha="center",
        va="bottom",
    )
    isi_bounds = isi_axis.get_position()
    panel_b_title_y = isi_bounds.y1 + 0.018
    fig.text(
        panel_a_left - 0.035,
        panel_b_title_y,
        "B",
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
    )
    fig.text(
        0.5 * (isi_bounds.x0 + isi_bounds.x1),
        panel_b_title_y,
        "ISI distribution",
        fontsize=7.0,
        ha="center",
        va="bottom",
    )
    tuning_top = max(ax.get_position().y1 for ax in tuning_axes[0])
    panel_c_left = trajectory_axes[0].get_position().x0
    fig.text(
        panel_c_left - 0.015,
        tuning_top + 0.045,
        "C",
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=date_colors[date],
            linewidth=1.1,
            label=day_labels[date],
        )
        for date in dates
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(
            waveform_right + 0.008,
            0.5
            * (
                waveform_axes[0].get_position().y1
                + waveform_axes[-1].get_position().y0
            ),
        ),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    condition_handles = [
        Line2D(
            [0],
            [0],
            color=EPOCH_TYPE_COLORS[condition],
            linewidth=1.1,
            label=condition.title(),
        )
        for condition in TUNING_EPOCH_ORDER
    ]
    fig.legend(
        handles=condition_handles,
        loc="upper left",
        bbox_to_anchor=(
            tuning_axes[0, 0].get_position().x0 + 0.008,
            tuning_axes[0, 0].get_position().y1 - 0.012,
        ),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.3,
    )
    return fig


def save_multiday_figure(
    data: Mapping[str, Any],
    *,
    output_dir: Path,
    output_name: str,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    """Build and save the requested multiday figure formats."""
    import matplotlib.pyplot as plt

    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    normalized_formats = tuple(str(value).lower().lstrip(".") for value in formats)
    if (
        not normalized_formats
        or len(set(normalized_formats)) != len(normalized_formats)
    ):
        raise ValueError("formats must contain unique output formats.")
    unsupported = set(normalized_formats).difference({"png", "pdf", "svg"})
    if unsupported:
        raise ValueError(f"Unsupported output formats: {sorted(unsupported)!r}.")

    fig = make_figure(data)
    output_paths = tuple(
        save_figure(
            fig,
            Path(output_dir) / f"{output_name}.{output_format}",
            dpi=dpi,
            bbox_inches=None,
        )
        for output_format in normalized_formats
    )
    plt.close(fig)
    return output_paths


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the multiday unit figure."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot five-day waveforms, ISI distributions, and light/dark "
            "trajectory tuning curves for one multiday-sorted unit."
        ),
    )
    parser.add_argument(
        "--analysis-path",
        type=Path,
        default=DEFAULT_ANALYSIS_PATH,
        help=f"Multiday analysis directory. Default: {DEFAULT_ANALYSIS_PATH}",
    )
    parser.add_argument("--animal-name", default=DEFAULT_ANIMAL_NAME)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--unit-id", type=int, default=DEFAULT_UNIT_ID)
    parser.add_argument(
        "--waveform-epoch",
        default=DEFAULT_WAVEFORM_EPOCH,
    )
    parser.add_argument("--light-epoch", default=DEFAULT_LIGHT_EPOCH)
    parser.add_argument("--dark-epoch", default=DEFAULT_DARK_EPOCH)
    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        help="Date to include, repeatable. Default: the five L14 dates.",
    )
    parser.add_argument(
        "--reference-date",
        help="Date used to select the top waveform channels. Default: first date.",
    )
    parser.add_argument(
        "--n-waveform-channels",
        type=int,
        default=DEFAULT_N_WAVEFORM_CHANNELS,
    )
    parser.add_argument(
        "--tuning-curve-dir",
        type=Path,
        help=(
            "Directory containing Pynapple tuning curves. Default: "
            "<analysis-path>/multiday_tuning_curves."
        ),
    )
    parser.add_argument(
        "--tuning-smoothing-sigma-bins",
        type=float,
        default=DEFAULT_TUNING_SMOOTHING_SIGMA_BINS,
        help=(
            "Gaussian display smoothing in 4-cm tuning bins. "
            f"Default: {DEFAULT_TUNING_SMOOTHING_SIGMA_BINS:g}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=("png", "pdf", "svg"),
        help="Output format, repeatable. Default: png.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    """Generate the configured multiday unit figure."""
    args = parse_arguments(argv)
    dates = DEFAULT_DATES if args.dates is None else tuple(args.dates)
    formats = DEFAULT_FORMATS if args.formats is None else tuple(args.formats)
    data = load_figure_data(
        analysis_path=args.analysis_path,
        animal_name=args.animal_name,
        region=args.region,
        unit_id=args.unit_id,
        dates=dates,
        waveform_epoch=args.waveform_epoch,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        reference_date=args.reference_date,
        n_waveform_channels=args.n_waveform_channels,
        tuning_curve_dir=args.tuning_curve_dir,
        tuning_smoothing_sigma_bins=args.tuning_smoothing_sigma_bins,
    )
    output_paths = save_multiday_figure(
        data,
        output_dir=args.output_dir,
        output_name=args.output_name,
        formats=formats,
        dpi=args.dpi,
    )
    for output_path in output_paths:
        print(output_path)
    return output_paths


if __name__ == "__main__":
    main()
