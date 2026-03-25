"""Combine per-epoch cleaned DLC position outputs into one session parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, get_analysis_path, load_position_timestamps
from v1ca1.position.clean_dlc_position import DEFAULT_OUTPUT_DIRNAME as DEFAULT_INPUT_DIRNAME

DEFAULT_OUTPUT_NAME = "position.parquet"
REQUIRED_INPUT_COLUMNS = (
    "epoch",
    "frame",
    "frame_time_s",
    "head_x_cleaned",
    "head_y_cleaned",
    "body_x_cleaned",
    "body_y_cleaned",
)
OUTPUT_COLUMNS = (
    "epoch",
    "frame",
    "frame_time_s",
    "head_x",
    "head_y",
    "body_x",
    "body_y",
)


def _validate_epoch_table(
    table: pd.DataFrame,
    path: Path,
    timestamps_position: dict[str, np.ndarray],
) -> tuple[str, pd.DataFrame]:
    """Validate one per-epoch cleaned parquet and return the curated output table."""
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Cleaned DLC parquet is missing required columns: {missing_columns!r} in {path}")
    if table.empty:
        raise ValueError(f"Cleaned DLC parquet is empty: {path}")

    epoch_values = table["epoch"].astype(str).unique().tolist()
    if len(epoch_values) != 1:
        raise ValueError(f"Cleaned DLC parquet must contain exactly one epoch: {path}")
    epoch = str(epoch_values[0])
    if epoch not in timestamps_position:
        raise ValueError(
            f"Cleaned DLC parquet epoch {epoch!r} was not found in saved position timestamps: {path}"
        )

    frame_numbers = table["frame"].to_numpy(dtype=int)
    if np.unique(frame_numbers).size != frame_numbers.size:
        raise ValueError(f"Cleaned DLC parquet contains duplicate frames for epoch {epoch!r}: {path}")
    if frame_numbers.size > 1 and np.any(np.diff(frame_numbers) < 0):
        raise ValueError(f"Cleaned DLC parquet frames are not monotonic for epoch {epoch!r}: {path}")

    frame_times = table["frame_time_s"].to_numpy(dtype=float)
    expected_times = np.asarray(timestamps_position[epoch], dtype=float)
    if frame_times.size != expected_times.size:
        raise ValueError(
            "Cleaned DLC parquet row count does not match saved position timestamps "
            f"for epoch {epoch!r}: {frame_times.size} vs {expected_times.size} in {path}"
        )
    if not np.allclose(frame_times, expected_times, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Cleaned DLC parquet timestamps do not match saved position timestamps "
            f"for epoch {epoch!r}: {path}"
        )

    curated = table.loc[
        :,
        [
            "epoch",
            "frame",
            "frame_time_s",
            "head_x_cleaned",
            "head_y_cleaned",
            "body_x_cleaned",
            "body_y_cleaned",
        ],
    ].rename(
        columns={
            "head_x_cleaned": "head_x",
            "head_y_cleaned": "head_y",
            "body_x_cleaned": "body_x",
            "body_y_cleaned": "body_y",
        }
    )
    return epoch, curated


def combine_clean_dlc_position(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    input_dirname: str = DEFAULT_INPUT_DIRNAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Combine all cleaned per-epoch DLC position parquets for one session."""
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not input_dirname:
        raise ValueError("--input-dirname must be a non-empty string.")
    if not output_name:
        raise ValueError("--output-name must be a non-empty string.")

    input_dir = analysis_path / input_dirname
    if not input_dir.exists():
        raise FileNotFoundError(f"Cleaned DLC input directory not found: {input_dir}")

    epoch_tags, timestamps_position, timestamp_source = load_position_timestamps(analysis_path)
    candidate_paths = sorted(path for path in input_dir.glob("*.parquet") if path.name != output_name)
    if not candidate_paths:
        raise FileNotFoundError(f"No cleaned DLC parquet files were found under {input_dir}")

    tables_by_epoch: dict[str, pd.DataFrame] = {}
    source_paths_by_epoch: dict[str, Path] = {}
    for path in candidate_paths:
        table = pd.read_parquet(path)
        epoch, curated = _validate_epoch_table(
            table=table,
            path=path,
            timestamps_position=timestamps_position,
        )
        if epoch in tables_by_epoch:
            raise ValueError(
                "Found duplicate cleaned DLC outputs for one epoch: "
                f"{epoch!r} in {source_paths_by_epoch[epoch]} and {path}"
            )
        tables_by_epoch[epoch] = curated
        source_paths_by_epoch[epoch] = path

    processed_epochs = [epoch for epoch in epoch_tags if epoch in tables_by_epoch]
    if not processed_epochs:
        raise ValueError(
            f"No cleaned DLC parquet files matched saved position epochs under {input_dir}"
        )

    combined = pd.concat([tables_by_epoch[epoch] for epoch in processed_epochs], ignore_index=True)
    combined = combined.loc[:, list(OUTPUT_COLUMNS)]

    output_path = input_dir / output_name
    combined.to_parquet(output_path, index=False)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.position.combine_clean_dlc_position",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "input_dirname": input_dirname,
            "output_name": output_name,
        },
        outputs={
            "position_timestamp_source": timestamp_source,
            "processed_epochs": processed_epochs,
            "processed_epoch_count": int(len(processed_epochs)),
            "combined_frame_count": int(combined.shape[0]),
            "source_paths_by_epoch": source_paths_by_epoch,
            "combined_output_path": output_path,
        },
    )
    print(f"Saved combined cleaned DLC position parquet to {output_path}")
    print(f"Saved run metadata to {log_path}")
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the cleaned DLC combine CLI."""
    parser = argparse.ArgumentParser(
        description="Combine cleaned per-epoch DLC position parquets for one session."
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
        "--input-dirname",
        default=DEFAULT_INPUT_DIRNAME,
        help=(
            "Directory under the session analysis path containing per-epoch cleaned DLC parquets. "
            f"Default: {DEFAULT_INPUT_DIRNAME}"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Combined parquet filename written under the input directory. Default: {DEFAULT_OUTPUT_NAME}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the cleaned DLC combine CLI."""
    args = parse_arguments()
    combine_clean_dlc_position(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        input_dirname=args.input_dirname,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
