from __future__ import annotations

"""Generate parametrically drawn W-track schematic panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from v1ca1.helper.plot_wtrack_schematic import (
    STIMULUS_LAYOUTS,
    W_TRACK_TRAJECTORY_NAMES,
    draw_large_ovals,
    draw_segmented_basis_circles,
    get_w_track_geometry,
    plot_w_track_trajectory,
    trajectory_points,
)
from v1ca1.paper_figures.style import SCHEMATIC_COLORS, TRAJECTORY_COLORS, save_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


DEFAULT_OUTPUT_DIR = Path("paper_figures") / "output"
DEFAULT_OUTPUT_NAME = "w_track_schematic"
DEFAULT_OUTPUT_FORMAT = "pdf"
FIGURE_FORMATS = ("pdf", "svg", "png", "tiff")
DEFAULT_BASIS_SEGMENT_STYLES: list[dict[str, Any]] = [
    {"edge_color": "black", "fill_color": "none", "radius": 0.34, "spacing": 0.34},
    {"edge_color": "black", "fill_color": "none", "radius": 0.34, "spacing": 0.34},
    {
        "edge_color": SCHEMATIC_COLORS["light_basis"],
        "fill_color": SCHEMATIC_COLORS["light_basis"],
        "fill_alpha": 0.35,
        "radius": 0.34,
        "spacing": 0.34,
    },
]
DEFAULT_OVAL_REGIONS = ["center_arm", "left_center_connector", "left_arm"]
DEFAULT_OVAL_STYLES: list[dict[str, Any]] = [
    {"edge_color": "black", "fill_color": "none"},
    {"edge_color": "black", "fill_color": "none"},
    {
        "edge_color": SCHEMATIC_COLORS["light_basis"],
        "fill_color": SCHEMATIC_COLORS["light_basis"],
        "fill_alpha": 0.38,
    },
]


def draw_w_track_schematic(
    ax: "Axes",
    *,
    trajectory_name: str,
    arrow_color: str | None = None,
    track_edge_color: str = "black",
    track_linewidth: float = 0.8,
    trajectory_linewidth: float = 1.2,
    arrow_mutation_scale: float = 10.0,
    fill_track: bool = False,
) -> "Axes":
    """Draw a compact W-track outline and trajectory arrow on an existing axis."""
    from matplotlib.patches import Polygon

    if arrow_color is None:
        arrow_color = TRAJECTORY_COLORS.get(
            trajectory_name,
            SCHEMATIC_COLORS["trajectory_arrow"],
        )

    outline, points, dims = get_w_track_geometry()
    path = trajectory_points(trajectory_name, points)
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="black" if fill_track else "none",
            edgecolor=track_edge_color,
            linewidth=track_linewidth,
            joinstyle="miter",
            zorder=1,
        )
    )

    xs, ys = zip(*path, strict=True)
    ax.plot(
        xs[:-1],
        ys[:-1],
        color=arrow_color,
        linewidth=trajectory_linewidth,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )
    ax.annotate(
        "",
        xy=path[-1],
        xytext=path[-2],
        arrowprops={
            "arrowstyle": "-|>",
            "color": arrow_color,
            "lw": trajectory_linewidth,
            "mutation_scale": arrow_mutation_scale,
            "shrinkA": 0,
            "shrinkB": 0,
            "connectionstyle": "arc3,rad=0",
        },
        zorder=6,
    )

    x5 = dims["x5"]
    y2 = dims["y2"]
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, x5 + 0.35)
    ax.set_ylim(-0.25, y2 + 0.25)
    ax.axis("off")
    return ax


def _add_compact_stimulus_labels(
    ax: "Axes",
    dims: dict[str, float],
    *,
    stimulus_layout: str,
    label_color: str,
    label_fontsize: float,
) -> None:
    """Add compact outside stimulus labels to an existing W-track axis."""
    label_maps = {
        "stim1": {"left": "A", "right": "B", "center": "C"},
        "stim2": {"left": "B", "right": "A", "center": "C"},
    }
    if stimulus_layout not in label_maps:
        raise ValueError(f"stimulus_layout must be one of {STIMULUS_LAYOUTS!r}.")

    labels = label_maps[stimulus_layout]
    x0, x5 = dims["x0"], dims["x5"]
    y0, y2 = dims["y0"], dims["y2"]
    ax.text(
        x0 - 0.82,
        y2 / 2,
        labels["left"],
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color=label_color,
    )
    ax.text(
        x5 + 0.82,
        y2 / 2,
        labels["right"],
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color=label_color,
    )
    ax.text(
        x5 / 2,
        y0 - 0.78,
        labels["center"],
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color=label_color,
    )


def draw_w_track_basis_schematic(
    ax: "Axes",
    *,
    trajectory_name: str,
    stimulus_layout: str = "stim1",
    show_labels: bool = False,
    label_color: str = "black",
    label_fontsize: float = 8.0,
    arrow_color: str | None = None,
    track_edge_color: str = "black",
    track_linewidth: float = 0.8,
    trajectory_linewidth: float = 1.0,
    arrow_mutation_scale: float = 8.0,
    fill_track_black: bool = False,
    show_arrow: bool = True,
    show_basis: bool = False,
    basis_segment_styles: list[dict[str, Any]] | None = None,
    basis_edge_color: str = "black",
    basis_fill_color: str = "none",
    basis_fill_alpha: float = 1.0,
    basis_radius: float = 0.34,
    basis_spacing: float = 0.34,
    basis_linewidth: float = 1.0,
    show_large_ovals: bool = False,
    oval_regions: str | list[str] | None = None,
    oval_styles: list[dict[str, Any]] | None = None,
    oval_edge_color: str = SCHEMATIC_COLORS["light_basis"],
    oval_fill_color: str = SCHEMATIC_COLORS["light_basis"],
    oval_fill_alpha: float = 0.35,
    oval_linewidth: float = 1.0,
) -> "Axes":
    """Draw a W-track with optional basis/segment overlays on an existing axis."""
    from matplotlib.patches import Polygon

    if arrow_color is None:
        arrow_color = TRAJECTORY_COLORS.get(
            trajectory_name,
            SCHEMATIC_COLORS["trajectory_arrow"],
        )

    outline, points, dims = get_w_track_geometry()
    path = trajectory_points(trajectory_name, points)
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="black" if fill_track_black else "white",
            edgecolor=track_edge_color,
            linewidth=track_linewidth,
            joinstyle="miter",
            zorder=1,
        )
    )

    if show_basis:
        draw_segmented_basis_circles(
            ax,
            trajectory_name,
            points,
            segment_styles=basis_segment_styles,
            default_edge_color=basis_edge_color,
            default_fill_color=basis_fill_color,
            default_fill_alpha=basis_fill_alpha,
            default_radius=basis_radius,
            default_spacing=basis_spacing,
            default_linewidth=basis_linewidth,
        )

    if show_large_ovals:
        draw_large_ovals(
            ax,
            dims,
            oval_regions=oval_regions,
            oval_styles=oval_styles,
            default_edge_color=oval_edge_color,
            default_fill_color=oval_fill_color,
            default_fill_alpha=oval_fill_alpha,
            default_linewidth=oval_linewidth,
        )

    xs, ys = zip(*path, strict=True)
    if show_arrow:
        ax.plot(
            xs[:-1],
            ys[:-1],
            color=arrow_color,
            linewidth=trajectory_linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )
        ax.annotate(
            "",
            xy=path[-1],
            xytext=path[-2],
            arrowprops={
                "arrowstyle": "-|>",
                "color": arrow_color,
                "lw": trajectory_linewidth,
                "mutation_scale": arrow_mutation_scale,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": "arc3,rad=0",
            },
            zorder=6,
        )
    else:
        ax.plot(
            xs,
            ys,
            color=arrow_color,
            linewidth=trajectory_linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )

    if show_labels:
        _add_compact_stimulus_labels(
            ax,
            dims,
            stimulus_layout=stimulus_layout,
            label_color=label_color,
            label_fontsize=label_fontsize,
        )

    x5 = dims["x5"]
    y2 = dims["y2"]
    ax.set_aspect("equal")
    ax.set_xlim(-1.05 if show_labels else -0.35, x5 + (1.05 if show_labels else 0.35))
    ax.set_ylim(-0.95 if show_labels else -0.25, y2 + 0.25)
    ax.axis("off")
    return ax


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the schematic output path for one requested format."""
    if output_format not in FIGURE_FORMATS:
        raise ValueError(
            f"Unknown output format {output_format!r}. Expected one of {FIGURE_FORMATS!r}."
        )
    return Path(output_dir) / f"{output_name}.{output_format}"


def make_w_track_schematic(
    *,
    output_path: Path,
    trajectory_name: str,
    stimulus_layout: str,
    label_color: str,
    show_labels: bool,
    show_arrow: bool,
    arrow_color: str | None,
    show_basis: bool,
    fill_track_black: bool,
    show_large_ovals: bool,
    dpi: int,
) -> Path:
    """Build and save one W-track schematic using the paper-figure defaults."""
    if arrow_color is None:
        arrow_color = TRAJECTORY_COLORS.get(
            trajectory_name,
            SCHEMATIC_COLORS["trajectory_arrow"],
        )

    fig: Figure
    ax: Axes
    fig, ax = plot_w_track_trajectory(
        trajectory_name=trajectory_name,
        show_labels=show_labels,
        stimulus_layout=stimulus_layout,
        label_color=label_color,
        show_arrow=show_arrow,
        arrow_color=arrow_color,
        show_basis=show_basis,
        basis_segment_styles=DEFAULT_BASIS_SEGMENT_STYLES,
        show_large_ovals=show_large_ovals,
        oval_regions=DEFAULT_OVAL_REGIONS,
        oval_styles=DEFAULT_OVAL_STYLES,
        fill_track_black=fill_track_black,
    )
    save_figure(fig, output_path, dpi=dpi)

    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Saved W-track schematic to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for W-track schematic generation."""
    parser = argparse.ArgumentParser(
        description="Generate a parametrically drawn W-track schematic."
    )
    parser.add_argument(
        "--trajectory-name",
        choices=W_TRACK_TRAJECTORY_NAMES,
        default="center_to_left",
        help="Trajectory to draw. Default: center_to_left",
    )
    parser.add_argument(
        "--stimulus-layout",
        choices=STIMULUS_LAYOUTS,
        default="stim1",
        help="Outside stimulus label layout. Default: stim1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for schematic output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--label-color",
        default="black",
        help="Stimulus label color. Default: black",
    )
    parser.add_argument(
        "--arrow-color",
        default=None,
        help=(
            "Trajectory arrow color. Default: the shared trajectory palette."
        ),
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide outside stimulus labels.",
    )
    parser.add_argument(
        "--no-arrow",
        action="store_true",
        help="Draw the trajectory without an arrowhead.",
    )
    parser.add_argument(
        "--no-basis",
        action="store_true",
        help="Hide circular basis functions.",
    )
    parser.add_argument(
        "--fill-track-black",
        action="store_true",
        help="Fill the track body in black.",
    )
    parser.add_argument(
        "--show-large-ovals",
        action="store_true",
        help="Show large oval overlays from the default schematic example.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run W-track schematic generation."""
    args = parse_arguments(argv)
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    make_w_track_schematic(
        output_path=output_path,
        trajectory_name=args.trajectory_name,
        stimulus_layout=args.stimulus_layout,
        label_color=args.label_color,
        show_labels=not args.no_labels,
        show_arrow=not args.no_arrow,
        arrow_color=args.arrow_color,
        show_basis=not args.no_basis,
        fill_track_black=args.fill_track_black,
        show_large_ovals=args.show_large_ovals,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
