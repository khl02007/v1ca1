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
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
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
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.cal": "Liberation Sans",
    "mathtext.sf": "Liberation Sans",
    "mathtext.tt": "Liberation Mono",
}
PANEL_LABEL_KWARGS: dict[str, Any] = {
    "fontsize": 8,
    "fontweight": "bold",
    "ha": "left",
    "va": "bottom",
}
REGION_COLORS: dict[str, str] = {
    "v1": "#2166AC",
    "ca1": "#B35806",
}
EPOCH_TYPE_COLORS: dict[str, str] = {
    "light": "#E6AB02",
    "dark": "#252525",
    "sleep": "#7B3294",
}
VISUAL_CONDITION_COLORS: dict[str, str] = {
    "02_r1": "#1B9E77",
    "06_r3": "#D01C8B",
    "dark": EPOCH_TYPE_COLORS["dark"],
}
TRAJECTORY_COLORS: dict[str, str] = {
    "center_to_left": "#5DA5DA",
    "center_to_right": "#F15854",
    "right_to_center": "#60BD68",
    "left_to_center": "#B2912F",
}
MODEL_CLASS_COLORS: dict[str, str] = {
    "visual": "#4D4D4D",
    "task_segment_bump": "#E7298A",
    "task_segment_scalar": "#7570B3",
}
ENCODING_COMPARISON_COLORS: dict[str, str] = {
    "dpp_vs_absolute_place": "#7570B3",
    "dpp_vs_absolute_task_progression": "#A6761D",
}
ANIMAL_COLORS: dict[str, str] = {
    "L14": "#66C2A5",
    "L15": "#FC8D62",
    "L16": "#8DA0CB",
    "L19": "#E78AC3",
}
NEUTRAL_COLORS: dict[str, str] = {
    "empirical": "black",
    "axis": "#404040",
    "segment_boundary": "#A6A6A6",
    "nonsignificant": "#B3B3B3",
    "dark_epoch_background": "#F2F2F2",
}
SCHEMATIC_COLORS: dict[str, str] = {
    "visual_stimulus": "#E6AB02",
    "light_basis": EPOCH_TYPE_COLORS["light"],
    "dark_basis": "#737373",
    "trajectory_arrow": "#F15854",
    "ripple_trace": "#6A51A3",
    "ripple_span": "#E6AB02",
    "ripple_onset": "#B35806",
    "ripple_window_fill": "#F1F1F1",
    "ca1_count_fill": "#F8EFE7",
    "v1_count_fill": "#E8EEF8",
    "glm_fill": "#E8F3E8",
}
RASTER_TICK_KWARGS: dict[str, Any] = {
    "markersize": 0.55,
    "markeredgewidth": 0.21,
}
HISTOGRAM_KWARGS: dict[str, Any] = {
    "alpha": 0.52,
    "edgecolor": "none",
}
COMPACT_HISTOGRAM_KWARGS: dict[str, Any] = {
    "alpha": 0.48,
    "edgecolor": "none",
}
EMPHASIS_HISTOGRAM_KWARGS: dict[str, Any] = {
    "alpha": 0.65,
    "edgecolor": "none",
}
OUTLINED_HISTOGRAM_KWARGS: dict[str, Any] = {
    "alpha": 0.60,
    "edgecolor": "white",
    "linewidth": 0.25,
}
EPOCH_HISTOGRAM_ALPHA: dict[str, float] = {
    "light": 0.48,
    "dark": 0.34,
    "sleep": 0.40,
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
