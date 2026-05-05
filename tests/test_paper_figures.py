from __future__ import annotations

import pytest

import v1ca1.paper_figures.datasets as datasets_module
from v1ca1.paper_figures import (
    DEFAULT_DPI,
    PAPER_RC_PARAMS,
    REGION_COLORS,
    figure_size,
    get_processed_datasets,
    mm_to_inches,
)


def test_mm_to_inches_uses_print_units() -> None:
    assert mm_to_inches(25.4) == pytest.approx(1.0)


def test_figure_size_returns_matplotlib_inches() -> None:
    assert figure_size(50.8, 25.4) == pytest.approx((2.0, 1.0))


def test_paper_style_embeds_editable_text_for_vector_exports() -> None:
    assert DEFAULT_DPI == 300
    assert PAPER_RC_PARAMS["pdf.fonttype"] == 42
    assert PAPER_RC_PARAMS["svg.fonttype"] == "none"
    assert REGION_COLORS == {"v1": "#4C72B0", "ca1": "#DD8452"}


def test_get_processed_datasets_uses_paper_figure_registry() -> None:
    assert get_processed_datasets() == [
        ("L12", "20240421", "08_r4"),
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L19", "20250930", "08_r4"),
    ]


def test_figure_2_epoch_registry_uses_light_dark_sleep_epochs() -> None:
    assert datasets_module.get_dataset_light_epoch("L14") == "02_r1"
    assert datasets_module.get_dataset_dark_epoch("L15") == "10_r5"
    assert datasets_module.get_dataset_sleep_epoch("L14") == "07_s4"
    assert datasets_module.get_processed_figure_epoch_datasets() == [
        ("L12", "20240421", "02_r1", "08_r4", "07_s4"),
        ("L14", "20240611", "02_r1", "08_r4", "07_s4"),
        ("L15", "20241121", "02_r1", "10_r5", "07_s4"),
        ("L19", "20250930", "02_r1", "08_r4", "07_s4"),
    ]
    assert datasets_module.make_figure_2_epoch_ids("L15", "20241121") == {
        "light": ("L15", "20241121", "02_r1"),
        "dark": ("L15", "20241121", "10_r5"),
        "sleep": ("L15", "20241121", "07_s4"),
    }


def test_format_processed_datasets_supports_shell_arguments_with_dark_epoch() -> None:
    formatted = datasets_module.format_processed_datasets(
        [("L14", "20240611", "08_r4"), ("L16", "20250302", "08_r4")],
        output_format="shell",
    )

    assert formatted == (
        "--animal-name L14 --date 20240611 --epoch 08_r4\n"
        "--animal-name L16 --date 20250302 --epoch 08_r4"
    )


def test_format_processed_datasets_infers_dark_epoch_for_old_two_field_tuples() -> None:
    formatted = datasets_module.format_processed_datasets(
        [("L15", "20241121")],
        output_format="csv",
    )

    assert formatted == "animal_name,date,dark_epoch\nL15,20241121,10_r5"


def test_format_processed_figure_epoch_datasets_includes_light_and_sleep() -> None:
    formatted = datasets_module.format_processed_figure_epoch_datasets(
        [("L15", "20241121", "10_r5")],
        output_format="plain",
    )

    assert formatted == "L15 20241121 02_r1 10_r5 07_s4"
