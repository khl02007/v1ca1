from __future__ import annotations

"""Write a copy of one NWB file with epoch bounds replaced from saved ephys intervals."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, DEFAULT_NWB_ROOT, get_analysis_path

if TYPE_CHECKING:
    import pandas as pd
    import pynapple as nap
    import pynwb


DEFAULT_OUTPUT_SUFFIX = "_epoch_bounds_replaced.nwb"


def _extract_interval_dataframe(intervals: "nap.IntervalSet") -> "pd.DataFrame":
    """Return a dataframe-like view of one pynapple IntervalSet."""
    if hasattr(intervals, "as_dataframe"):
        return intervals.as_dataframe()
    if hasattr(intervals, "_metadata"):
        return intervals._metadata.copy()  # type: ignore[attr-defined]
    raise ValueError("Could not read metadata from timestamps_ephys.npz.")


def _extract_epoch_tags(intervals: "nap.IntervalSet") -> list[str]:
    """Extract saved epoch labels from timestamps_ephys.npz."""
    try:
        epoch_info = intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is not None:
        epoch_array = np.asarray(epoch_info)
        if epoch_array.size:
            return [str(epoch) for epoch in epoch_array.tolist()]

    interval_df = _extract_interval_dataframe(intervals)
    if "epoch" in interval_df.columns:
        return [str(epoch) for epoch in interval_df["epoch"].tolist()]

    raise ValueError("timestamps_ephys.npz does not contain saved epoch labels.")


def _extract_interval_bounds(intervals: "nap.IntervalSet") -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned start and end arrays from timestamps_ephys.npz."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "timestamps_ephys.npz has mismatched start/end arrays: "
            f"{starts.shape} vs {ends.shape}."
        )
    return starts, ends


def load_epoch_bounds_npz(analysis_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load saved epoch labels and bounds from timestamps_ephys.npz."""
    import pynapple as nap

    npz_path = analysis_path / "timestamps_ephys.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"timestamps_ephys.npz not found: {npz_path}")

    try:
        intervals = nap.load_file(npz_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {npz_path}.") from exc

    epoch_tags = _extract_epoch_tags(intervals)
    start_times, stop_times = _extract_interval_bounds(intervals)
    if len(epoch_tags) != start_times.size:
        raise ValueError(
            "Mismatch between saved epoch labels and interval bounds in "
            f"{npz_path}."
        )
    if start_times.size == 0:
        raise ValueError(f"timestamps_ephys.npz does not contain any epochs: {npz_path}")
    if np.any(stop_times < start_times):
        raise ValueError(f"timestamps_ephys.npz contains an interval with stop < start: {npz_path}")

    return epoch_tags, start_times, stop_times


def resolve_output_path(
    nwb_path: Path,
    animal_name: str,
    date: str,
    output_path: Path | None,
) -> Path:
    """Return the requested output path, defaulting to a sibling NWB copy."""
    if output_path is not None:
        return output_path
    return nwb_path.with_name(f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}")


def validate_output_path(source_path: Path, output_path: Path, overwrite: bool) -> None:
    """Validate the destination path for the corrected NWB file."""
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Output path must differ from the source NWB path.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. Pass --overwrite to replace it."
        )


def validate_epoch_tags(
    nwb_epoch_tags: list[str],
    replacement_epoch_tags: list[str],
) -> None:
    """Require exact epoch count and order agreement between NWB and npz."""
    if nwb_epoch_tags != replacement_epoch_tags:
        raise ValueError(
            "Epoch labels from timestamps_ephys.npz do not match the NWB epochs table. "
            f"NWB: {nwb_epoch_tags!r}; timestamps_ephys.npz: {replacement_epoch_tags!r}"
        )


def _set_column_values(column: Any, values: np.ndarray, column_name: str) -> None:
    """Replace one DynamicTable column with new values."""
    value_array = np.asarray(values, dtype=float)
    targets: list[Any] = []
    if hasattr(column, "data"):
        targets.append(column.data)
    targets.append(column)

    errors: list[str] = []
    for target in targets:
        try:
            target[:] = value_array
            return
        except Exception as exc:
            errors.append(f"{type(target).__name__} slice assignment failed: {exc}")

        try:
            for index, value in enumerate(value_array.tolist()):
                target[index] = float(value)
            return
        except Exception as exc:
            errors.append(f"{type(target).__name__} item assignment failed: {exc}")

    if hasattr(column, "data"):
        try:
            column.data = value_array.tolist()
            return
        except Exception as exc:
            errors.append(f"{type(column).__name__} data replacement failed: {exc}")

    raise TypeError(
        f"Could not update epochs column {column_name!r}. "
        + " ".join(errors)
    )


def replace_epoch_bounds_in_memory(
    nwbfile: "pynwb.NWBFile",
    replacement_start_times: np.ndarray,
    replacement_stop_times: np.ndarray,
) -> list[dict[str, Any]]:
    """Replace epoch start/stop times on one in-memory NWB object."""
    if nwbfile.epochs is None:
        raise ValueError("NWB file does not contain an epochs table.")

    epoch_tags, old_start_times, old_stop_times = extract_epoch_metadata(nwbfile)
    if not (
        len(epoch_tags)
        == replacement_start_times.size
        == replacement_stop_times.size
    ):
        raise ValueError("Replacement epoch bounds do not match the NWB epoch count.")

    epoch_table = nwbfile.epochs
    _set_column_values(epoch_table["start_time"], replacement_start_times, "start_time")
    _set_column_values(epoch_table["stop_time"], replacement_stop_times, "stop_time")

    return [
        {
            "epoch": epoch,
            "old_start_time_s": float(old_start),
            "old_stop_time_s": float(old_stop),
            "new_start_time_s": float(new_start),
            "new_stop_time_s": float(new_stop),
        }
        for epoch, old_start, old_stop, new_start, new_stop in zip(
            epoch_tags,
            old_start_times,
            old_stop_times,
            replacement_start_times,
            replacement_stop_times,
            strict=True,
        )
    ]


def write_nwb_copy(
    read_io: Any,
    nwbfile: "pynwb.NWBFile",
    output_path: Path,
) -> None:
    """Write the corrected NWB object to a new file."""
    import pynwb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pynwb.NWBHDF5IO(output_path, "w") as write_io:
        export = getattr(write_io, "export", None)
        if callable(export):
            export(src_io=read_io, nwbfile=nwbfile)
        else:
            write_io.write(nwbfile)


def replace_epoch_bounds(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a new NWB file with epoch bounds replaced from timestamps_ephys.npz."""
    import pynwb

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    resolved_output_path = resolve_output_path(
        nwb_path=nwb_path,
        animal_name=animal_name,
        date=date,
        output_path=output_path,
    )

    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    replacement_epoch_tags, replacement_start_times, replacement_stop_times = (
        load_epoch_bounds_npz(analysis_path)
    )
    validate_output_path(source_path=nwb_path, output_path=resolved_output_path, overwrite=overwrite)

    print(f"Processing {animal_name} {date}.")
    with pynwb.NWBHDF5IO(nwb_path, "r") as read_io:
        nwbfile = read_io.read()
        nwb_epoch_tags, _nwb_start_times, _nwb_stop_times = extract_epoch_metadata(nwbfile)
        validate_epoch_tags(
            nwb_epoch_tags=nwb_epoch_tags,
            replacement_epoch_tags=replacement_epoch_tags,
        )
        epoch_replacements = replace_epoch_bounds_in_memory(
            nwbfile=nwbfile,
            replacement_start_times=replacement_start_times,
            replacement_stop_times=replacement_stop_times,
        )
        write_nwb_copy(
            read_io=read_io,
            nwbfile=nwbfile,
            output_path=resolved_output_path,
        )

    outputs = {
        "source_nwb_path": nwb_path,
        "output_nwb_path": resolved_output_path,
        "epoch_column_names": list(getattr(nwbfile.epochs, "colnames", [])),
        "epoch_replacements": epoch_replacements,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.nwb.replace_epoch_bounds",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "output_path": resolved_output_path,
            "overwrite": overwrite,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")
    return resolved_output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the epoch-bound replacement CLI."""
    parser = argparse.ArgumentParser(
        description="Write a new NWB file with epoch bounds replaced from timestamps_ephys.npz"
    )
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path for the corrected NWB copy. Default: sibling *_epoch_bounds_replaced.nwb",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the epoch-bound replacement CLI."""
    args = parse_arguments()
    replace_epoch_bounds(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
