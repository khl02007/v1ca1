from __future__ import annotations

"""Shared matplotlib style helpers for manuscript figures."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


DEFAULT_DPI = 300
PAPER_RC_PARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
PANEL_LABEL_KWARGS: dict[str, Any] = {
    "fontsize": 8,
    "fontweight": "bold",
    "ha": "left",
    "va": "bottom",
}


def mm_to_inches(length_mm: float) -> float:
    """Convert a length in millimeters to inches."""
    return float(length_mm) / 25.4


def figure_size(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Return a matplotlib figure size tuple from millimeter dimensions."""
    return mm_to_inches(width_mm), mm_to_inches(height_mm)


def apply_paper_style(overrides: Mapping[str, Any] | None = None) -> None:
    """Apply shared manuscript plotting defaults to matplotlib rcParams."""
    import matplotlib.pyplot as plt

    rc_params = PAPER_RC_PARAMS.copy()
    if overrides is not None:
        rc_params.update(overrides)
    plt.rcParams.update(rc_params)


def label_axis(
    ax: "Axes",
    label: str,
    x: float = -0.08,
    y: float = 1.04,
    **kwargs: Any,
) -> None:
    """Add a panel label in axes-relative coordinates."""
    text_kwargs = PANEL_LABEL_KWARGS.copy()
    text_kwargs.update(kwargs)
    ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)


def save_figure(
    figure: "Figure",
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    **kwargs: Any,
) -> Path:
    """Save a figure, creating the output directory if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight"}
    save_kwargs.update(kwargs)
    figure.savefig(output_path, **save_kwargs)
    return output_path
