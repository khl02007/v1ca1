from __future__ import annotations

"""List processed data sets used by manuscript figure-combine scripts."""

import argparse
import json
from collections.abc import Sequence


DatasetId = tuple[str, str, str]
FigureEpochDatasetId = tuple[str, str, str, str, str]
DEFAULT_DARK_EPOCH = "08_r4"
DEFAULT_LIGHT_EPOCH = "02_r1"
DEFAULT_SLEEP_EPOCH = "07_s4"
DEFAULT_DARK_EPOCH_BY_ANIMAL = {
    "L15": "10_r5",
}
DEFAULT_LIGHT_EPOCH_BY_ANIMAL: dict[str, str] = {}
DEFAULT_SLEEP_EPOCH_BY_ANIMAL: dict[str, str] = {}
DATASET_OUTPUT_FORMATS = ("plain", "shell", "csv", "json")
PROCESSED_DATASETS: tuple[FigureEpochDatasetId, ...] = (
    ("L12", "20240421", "02_r1", "08_r4", "07_s4"),
    ("L14", "20240611", "02_r1", "08_r4", "07_s4"),
    ("L15", "20241121", "02_r1", "10_r5", "07_s4"),
    # ("L16", "20250302", "02_r1", "08_r4", "07_s4"),
    ("L19", "20250930", "02_r1", "08_r4", "07_s4"),
)


def get_dataset_dark_epoch(animal_name: str) -> str:
    """Return the default dark epoch label for one animal."""
    return DEFAULT_DARK_EPOCH_BY_ANIMAL.get(str(animal_name), DEFAULT_DARK_EPOCH)


def get_dataset_light_epoch(animal_name: str) -> str:
    """Return the default light run epoch label for one animal."""
    return DEFAULT_LIGHT_EPOCH_BY_ANIMAL.get(str(animal_name), DEFAULT_LIGHT_EPOCH)


def get_dataset_sleep_epoch(animal_name: str) -> str:
    """Return the default sleep epoch label for one animal."""
    return DEFAULT_SLEEP_EPOCH_BY_ANIMAL.get(str(animal_name), DEFAULT_SLEEP_EPOCH)


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


def make_figure_epoch_dataset_id(
    animal_name: str,
    date: str,
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
) -> FigureEpochDatasetId:
    """Return one normalized light/dark/sleep paper-figure data-set ID."""
    return (
        str(animal_name),
        str(date),
        (
            get_dataset_light_epoch(animal_name)
            if light_epoch is None
            else str(light_epoch)
        ),
        get_dataset_dark_epoch(animal_name) if dark_epoch is None else str(dark_epoch),
        (
            get_dataset_sleep_epoch(animal_name)
            if sleep_epoch is None
            else str(sleep_epoch)
        ),
    )


def make_figure_2_epoch_ids(
    animal_name: str,
    date: str,
    *,
    light_epoch: str | None = None,
    dark_epoch: str | None = None,
    sleep_epoch: str | None = None,
) -> dict[str, DatasetId]:
    """Return registered light, dark, and sleep epoch IDs for one session."""
    return {
        "light": make_dataset_id(
            animal_name,
            date,
            get_dataset_light_epoch(animal_name) if light_epoch is None else light_epoch,
        ),
        "dark": make_dataset_id(
            animal_name,
            date,
            get_dataset_dark_epoch(animal_name) if dark_epoch is None else dark_epoch,
        ),
        "sleep": make_dataset_id(
            animal_name,
            date,
            get_dataset_sleep_epoch(animal_name) if sleep_epoch is None else sleep_epoch,
        ),
    }


def get_processed_datasets() -> list[DatasetId]:
    """Return animal/date/dark-epoch data sets to include in figure combines."""
    return [
        make_dataset_id(animal_name, date, dark_epoch)
        for (
            animal_name,
            date,
            _light_epoch,
            dark_epoch,
            _sleep_epoch,
        ) in PROCESSED_DATASETS
    ]


def get_processed_figure_epoch_datasets() -> list[FigureEpochDatasetId]:
    """Return registered animal/date/light/dark/sleep data sets."""
    return [
        make_figure_epoch_dataset_id(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            sleep_epoch=sleep_epoch,
        )
        for animal_name, date, light_epoch, dark_epoch, sleep_epoch in PROCESSED_DATASETS
    ]


def normalize_dataset_id(
    dataset: DatasetId | FigureEpochDatasetId | tuple[str, str],
) -> DatasetId:
    """Return a three-field data-set identifier from old or current tuples."""
    if len(dataset) == 2:
        animal_name, date = dataset
        return make_dataset_id(animal_name, date)
    if len(dataset) == 3:
        animal_name, date, dark_epoch = dataset
        return make_dataset_id(animal_name, date, dark_epoch)
    if len(dataset) == 5:
        animal_name, date, _light_epoch, dark_epoch, _sleep_epoch = dataset
        return make_dataset_id(animal_name, date, dark_epoch)
    raise ValueError(
        "Data-set identifiers must be (animal_name, date) or "
        "(animal_name, date, dark_epoch) or "
        "(animal_name, date, light_epoch, dark_epoch, sleep_epoch)."
    )


def normalize_figure_epoch_dataset_id(
    dataset: DatasetId | FigureEpochDatasetId | tuple[str, str],
) -> FigureEpochDatasetId:
    """Return a five-field light/dark/sleep figure data-set identifier."""
    if len(dataset) == 2:
        animal_name, date = dataset
        return make_figure_epoch_dataset_id(animal_name, date)
    if len(dataset) == 3:
        animal_name, date, dark_epoch = dataset
        return make_figure_epoch_dataset_id(
            animal_name,
            date,
            dark_epoch=dark_epoch,
        )
    if len(dataset) == 5:
        animal_name, date, light_epoch, dark_epoch, sleep_epoch = dataset
        return make_figure_epoch_dataset_id(
            animal_name,
            date,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            sleep_epoch=sleep_epoch,
        )
    raise ValueError(
        "Data-set identifiers must be (animal_name, date), "
        "(animal_name, date, dark_epoch), or "
        "(animal_name, date, light_epoch, dark_epoch, sleep_epoch)."
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


def format_processed_figure_epoch_datasets(
    datasets: Sequence[DatasetId | FigureEpochDatasetId | tuple[str, str]],
    output_format: str = "plain",
) -> str:
    """Format processed light/dark/sleep data sets for scripts."""
    datasets = [normalize_figure_epoch_dataset_id(dataset) for dataset in datasets]
    if output_format == "plain":
        return "\n".join(
            f"{animal_name} {date} {light_epoch} {dark_epoch} {sleep_epoch}"
            for animal_name, date, light_epoch, dark_epoch, sleep_epoch in datasets
        )
    if output_format == "shell":
        return "\n".join(
            f"--animal-name {animal_name} --date {date} "
            f"--light-epoch {light_epoch} --dark-epoch {dark_epoch} "
            f"--sleep-epoch {sleep_epoch}"
            for animal_name, date, light_epoch, dark_epoch, sleep_epoch in datasets
        )
    if output_format == "csv":
        rows = ["animal_name,date,light_epoch,dark_epoch,sleep_epoch"]
        rows.extend(
            f"{animal_name},{date},{light_epoch},{dark_epoch},{sleep_epoch}"
            for animal_name, date, light_epoch, dark_epoch, sleep_epoch in datasets
        )
        return "\n".join(rows)
    if output_format == "json":
        records = [
            {
                "animal_name": animal_name,
                "date": date,
                "light_epoch": light_epoch,
                "dark_epoch": dark_epoch,
                "sleep_epoch": sleep_epoch,
            }
            for animal_name, date, light_epoch, dark_epoch, sleep_epoch in datasets
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
    parser.add_argument(
        "--include-light-sleep",
        action="store_true",
        help=(
            "Print animal/date/light_epoch/dark_epoch/sleep_epoch entries instead "
            "of the default animal/date/dark_epoch entries."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the processed-data-set listing CLI."""
    args = parse_args(argv)
    if args.include_light_sleep:
        output = format_processed_figure_epoch_datasets(
            get_processed_figure_epoch_datasets(),
            output_format=args.format,
        )
    else:
        output = format_processed_datasets(
            get_processed_datasets(),
            output_format=args.format,
        )
    if output:
        print(output)


if __name__ == "__main__":
    main()
