"""Convert legacy position pickles into the canonical combined parquet."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    coerce_position_array,
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    get_analysis_path,
    load_position_timestamps,
)

DEFAULT_POSITION_NAME = "position.pkl"
DEFAULT_BODY_POSITION_NAME = "body_position.pkl"
OUTPUT_COLUMNS = (
    "epoch",
    "frame",
    "frame_time_s",
    "head_x_cm",
    "head_y_cm",
    "body_x_cm",
    "body_y_cm",
)


def _validate_name(value: str, flag: str) -> str:
    """Return one validated CLI-provided file or directory name."""
    if not value:
        raise ValueError(f"{flag} must be a non-empty string.")
    return value


def _load_position_pickle_from_path(
    input_path: Path,
) -> dict[str, np.ndarray]:
    """Load one per-epoch XY pickle from an explicit filesystem path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Position file not found: {input_path}")

    with open(input_path, "rb") as file:
        position_dict = pickle.load(file)

    return {
        str(epoch): coerce_position_array(value)
        for epoch, value in position_dict.items()
    }


def _validate_epoch_subset(
    position_by_epoch: dict[str, np.ndarray],
    epoch_tags: list[str],
    *,
    input_name: str,
) -> None:
    """Ensure one per-epoch position mapping only contains known epochs."""
    extra_epochs = sorted(set(position_by_epoch) - set(epoch_tags))
    if extra_epochs:
        raise ValueError(
            f"{input_name} contains epochs not found in saved position timestamps: {extra_epochs!r}"
        )


def _validate_epoch_lengths(
    position_by_epoch: dict[str, np.ndarray],
    timestamps_position: dict[str, np.ndarray],
    *,
    input_name: str,
) -> None:
    """Ensure one per-epoch position mapping matches saved timestamp counts."""
    for epoch, position_xy in position_by_epoch.items():
        position_array = np.asarray(position_xy, dtype=float)
        expected_times = np.asarray(timestamps_position[epoch], dtype=float)
        if position_array.shape[0] != expected_times.size:
            raise ValueError(
                f"{input_name} row count does not match saved position timestamps "
                f"for epoch {epoch!r}: {position_array.shape[0]} vs {expected_times.size}."
            )


def convert_legacy_position_pickles(
    animal_name: str,
    date: str,
    head_position_path: Path,
    body_position_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dirname: str = DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    output_name: str = DEFAULT_CLEAN_DLC_POSITION_NAME,
    overwrite: bool = False,
) -> Path:
    """Convert one session's legacy head/body pickles into one combined parquet."""
    import pandas as pd

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    output_dirname = _validate_name(output_dirname, "--output-dirname")
    output_name = _validate_name(output_name, "--output-name")
    head_position_path = Path(head_position_path)
    body_position_path = Path(body_position_path)

    epoch_tags, timestamps_position, timestamp_source = load_position_timestamps(analysis_path)
    head_position = _load_position_pickle_from_path(
        head_position_path,
    )
    body_position = _load_position_pickle_from_path(
        body_position_path,
    )

    _validate_epoch_subset(
        head_position,
        epoch_tags,
        input_name=str(head_position_path),
    )
    _validate_epoch_subset(
        body_position,
        epoch_tags,
        input_name=str(body_position_path),
    )
    _validate_epoch_lengths(
        head_position,
        timestamps_position,
        input_name=str(head_position_path),
    )
    _validate_epoch_lengths(
        body_position,
        timestamps_position,
        input_name=str(body_position_path),
    )

    written_epochs = [epoch for epoch in epoch_tags if epoch in head_position]
    if not written_epochs:
        raise ValueError(
            f"{head_position_path} does not contain any epochs matching saved position timestamps."
        )

    output_dir = analysis_path / output_dirname
    output_path = output_dir / output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output parquet already exists: {output_path}. Pass --overwrite to replace it."
        )

    tables: list[pd.DataFrame] = []
    missing_body_epochs: list[str] = []
    for epoch in written_epochs:
        frame_times = np.asarray(timestamps_position[epoch], dtype=float)
        head_xy = np.asarray(head_position[epoch], dtype=float)
        if epoch in body_position:
            body_xy = np.asarray(body_position[epoch], dtype=float)
        else:
            body_xy = np.full(head_xy.shape, np.nan, dtype=float)
            missing_body_epochs.append(epoch)

        tables.append(
            pd.DataFrame(
                {
                    "epoch": epoch,
                    "frame": np.arange(frame_times.size, dtype=int),
                    "frame_time_s": frame_times,
                    "head_x_cm": head_xy[:, 0],
                    "head_y_cm": head_xy[:, 1],
                    "body_x_cm": body_xy[:, 0],
                    "body_y_cm": body_xy[:, 1],
                }
            )
        )

    combined = pd.concat(tables, ignore_index=True)
    combined = combined.loc[:, list(OUTPUT_COLUMNS)]

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.position.convert_legacy_position_pickles",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "head_position_path": head_position_path,
            "body_position_path": body_position_path,
            "data_root": data_root,
            "output_dirname": output_dirname,
            "output_name": output_name,
            "overwrite": overwrite,
        },
        outputs={
            "position_timestamp_source": timestamp_source,
            "head_position_source": head_position_path,
            "body_position_source": body_position_path,
            "written_epochs": written_epochs,
            "written_epoch_count": int(len(written_epochs)),
            "missing_body_epochs": missing_body_epochs,
            "missing_body_epoch_count": int(len(missing_body_epochs)),
            "combined_frame_count": int(combined.shape[0]),
            "combined_output_path": output_path,
        },
    )
    print(f"Saved combined legacy position parquet to {output_path}")
    print(f"Saved run metadata to {log_path}")
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the legacy position conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert legacy head/body position pickles into one combined parquet."
    )
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--head-position-path",
        type=Path,
        required=True,
        help="Path to the legacy head-position pickle.",
    )
    parser.add_argument(
        "--body-position-path",
        type=Path,
        required=True,
        help="Path to the legacy body-position pickle.",
    )
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        help=(
            "Directory under the session analysis path used for the combined parquet. "
            f"Default: {DEFAULT_CLEAN_DLC_POSITION_DIRNAME}"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_CLEAN_DLC_POSITION_NAME,
        help=f"Combined parquet filename written under the output directory. Default: {DEFAULT_CLEAN_DLC_POSITION_NAME}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output parquet when it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the legacy position conversion CLI."""
    args = parse_arguments()
    convert_legacy_position_pickles(
        animal_name=args.animal_name,
        date=args.date,
        head_position_path=args.head_position_path,
        body_position_path=args.body_position_path,
        data_root=args.data_root,
        output_dirname=args.output_dirname,
        output_name=args.output_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
