from __future__ import annotations

"""Publication figure helpers for the V1-CA1 project."""

from v1ca1.paper_figures.style import (
    DEFAULT_DPI,
    PANEL_LABEL_KWARGS,
    PAPER_RC_PARAMS,
    apply_paper_style,
    figure_size,
    label_axis,
    mm_to_inches,
    save_figure,
)
from v1ca1.paper_figures.datasets import get_processed_datasets

__all__ = [
    "DEFAULT_DPI",
    "PANEL_LABEL_KWARGS",
    "PAPER_RC_PARAMS",
    "apply_paper_style",
    "figure_size",
    "get_processed_datasets",
    "label_axis",
    "mm_to_inches",
    "save_figure",
]
