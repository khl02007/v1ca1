from __future__ import annotations

"""List processed data sets used by manuscript figure-combine scripts."""

import argparse
import json
from collections.abc import Sequence


DatasetId = tuple[str, str]
DATASET_OUTPUT_FORMATS = ("plain", "shell", "csv", "json")
PROCESSED_DATASETS: tuple[DatasetId, ...] = (
    ("L12", "20240421"),
    ("L14", "20240611"),
    ("L15", "20241121"),
    ("L16", "20250302"),
    ("L19", "20250930"),
)


def get_processed_datasets() -> list[DatasetId]:
    """Return animal/date data sets to include in paper-figure combines."""
    return [(str(animal_name), str(date)) for animal_name, date in PROCESSED_DATASETS]


def format_processed_datasets(
    datasets: Sequence[DatasetId],
    output_format: str = "plain",
) -> str:
    """Format processed data sets for command-line or script consumption."""
    if output_format == "plain":
        return "\n".join(f"{animal_name} {date}" for animal_name, date in datasets)
    if output_format == "shell":
        return "\n".join(
            f"--animal-name {animal_name} --date {date}"
            for animal_name, date in datasets
        )
    if output_format == "csv":
        rows = ["animal_name,date"]
        rows.extend(f"{animal_name},{date}" for animal_name, date in datasets)
        return "\n".join(rows)
    if output_format == "json":
        records = [
            {"animal_name": animal_name, "date": date}
            for animal_name, date in datasets
        ]
        return json.dumps(records, indent=2)
    raise ValueError(
        f"Unknown output_format {output_format!r}. "
        f"Expected one of {DATASET_OUTPUT_FORMATS!r}."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the processed-data-set listing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "List processed V1-CA1 data sets configured for paper-figure combines."
        )
    )
    parser.add_argument(
        "--format",
        choices=DATASET_OUTPUT_FORMATS,
        default="plain",
        help=(
            "Output format. 'plain' prints one 'animal date' pair per line; "
            "'shell' prints reusable --animal-name/--date argument pairs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the processed-data-set listing CLI."""
    args = parse_args(argv)
    output = format_processed_datasets(
        get_processed_datasets(),
        output_format=args.format,
    )
    if output:
        print(output)


if __name__ == "__main__":
    main()
