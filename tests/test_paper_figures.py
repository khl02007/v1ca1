from __future__ import annotations

import pytest

import v1ca1.paper_figures.datasets as datasets_module
from v1ca1.paper_figures import (
    DEFAULT_DPI,
    PAPER_RC_PARAMS,
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


def test_get_processed_datasets_uses_paper_figure_registry() -> None:
    assert get_processed_datasets() == [
        ("L12", "20240421"),
        ("L14", "20240611"),
        ("L15", "20241121"),
        ("L16", "20250302"),
        ("L19", "20250930"),
    ]


def test_format_processed_datasets_supports_shell_arguments() -> None:
    formatted = datasets_module.format_processed_datasets(
        [("L14", "20240611"), ("L16", "20250302")],
        output_format="shell",
    )

    assert formatted == (
        "--animal-name L14 --date 20240611\n"
        "--animal-name L16 --date 20250302"
    )
