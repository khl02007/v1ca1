from __future__ import annotations

"""List processed data sets used by manuscript figure-combine scripts."""

import argparse
import json
from collections.abc import Sequence


DatasetId = tuple[str, str, str]
DEFAULT_DARK_EPOCH = "08_r4"
DEFAULT_DARK_EPOCH_BY_ANIMAL = {
    "L15": "10_r5",
}
DATASET_OUTPUT_FORMATS = ("plain", "shell", "csv", "json")
PROCESSED_DATASETS: tuple[DatasetId, ...] = (
    ("L12", "20240421", "08_r4"),
    ("L14", "20240611", "08_r4"),
    ("L15", "20241121", "10_r5"),
    # ("L16", "20250302", "08_r4"),
    ("L19", "20250930", "08_r4"),
)


def get_dataset_dark_epoch(animal_name: str) -> str:
    """Return the default dark epoch label for one animal."""
    return DEFAULT_DARK_EPOCH_BY_ANIMAL.get(str(animal_name), DEFAULT_DARK_EPOCH)


def make_dataset_id(
    animal_name: str,
    date: str,
    dark_epoch: str | None = None,
) -> DatasetId:
    """Return one normalized paper-figure data-set identifier."""
    dark_epoch = (
        get_dataset_dark_epoch(animal_name) if dark_epoch is None else dark_epoch
    )
    return str(animal_name), str(date), str(dark_epoch)


def get_processed_datasets() -> list[DatasetId]:
    """Return animal/date/dark-epoch data sets to include in figure combines."""
    return [
        make_dataset_id(animal_name, date, dark_epoch)
        for animal_name, date, dark_epoch in PROCESSED_DATASETS
    ]


def normalize_dataset_id(dataset: DatasetId | tuple[str, str]) -> DatasetId:
    """Return a three-field data-set identifier from old or current tuples."""
    if len(dataset) == 2:
        animal_name, date = dataset
        return make_dataset_id(animal_name, date)
    if len(dataset) == 3:
        animal_name, date, dark_epoch = dataset
        return make_dataset_id(animal_name, date, dark_epoch)
    raise ValueError(
        "Data-set identifiers must be (animal_name, date) or "
        "(animal_name, date, dark_epoch)."
    )


def format_processed_datasets(
    datasets: Sequence[DatasetId | tuple[str, str]],
    output_format: str = "plain",
) -> str:
    """Format processed data sets for command-line or script consumption."""
    datasets = [normalize_dataset_id(dataset) for dataset in datasets]
    if output_format == "plain":
        return "\n".join(
            f"{animal_name} {date} {dark_epoch}"
            for animal_name, date, dark_epoch in datasets
        )
    if output_format == "shell":
        return "\n".join(
            f"--animal-name {animal_name} --date {date} --epoch {dark_epoch}"
            for animal_name, date, dark_epoch in datasets
        )
    if output_format == "csv":
        rows = ["animal_name,date,dark_epoch"]
        rows.extend(
            f"{animal_name},{date},{dark_epoch}"
            for animal_name, date, dark_epoch in datasets
        )
        return "\n".join(rows)
    if output_format == "json":
        records = [
            {
                "animal_name": animal_name,
                "date": date,
                "dark_epoch": dark_epoch,
            }
            for animal_name, date, dark_epoch in datasets
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
            "Output format. 'plain' prints one 'animal date dark_epoch' "
            "triple per line; 'shell' prints reusable "
            "--animal-name/--date/--epoch argument triples."
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
