from __future__ import annotations

"""Generate parametrically drawn W-track schematic panels."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from v1ca1.helper.plot_wtrack_schematic import (
    STIMULUS_LAYOUTS,
    W_TRACK_TRAJECTORY_NAMES,
    get_w_track_geometry,
    plot_w_track_trajectory,
    trajectory_points,
)
from v1ca1.paper_figures.style import save_figure

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
        "edge_color": "red",
        "fill_color": "pink",
        "fill_alpha": 0.35,
        "radius": 0.34,
        "spacing": 0.34,
    },
]
DEFAULT_OVAL_REGIONS = ["center_arm", "left_center_connector", "left_arm"]
DEFAULT_OVAL_STYLES: list[dict[str, Any]] = [
    {"edge_color": "black", "fill_color": "none"},
    {"edge_color": "black", "fill_color": "none"},
    {"edge_color": "red", "fill_color": "#f4c7d7", "fill_alpha": 0.7},
]


def draw_w_track_schematic(
    ax: "Axes",
    *,
    trajectory_name: str,
    arrow_color: str = "dodgerblue",
    track_edge_color: str = "black",
    track_linewidth: float = 0.8,
    trajectory_linewidth: float = 1.2,
    arrow_mutation_scale: float = 10.0,
    fill_track: bool = False,
) -> "Axes":
    """Draw a compact W-track outline and trajectory arrow on an existing axis."""
    from matplotlib.patches import Polygon

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
    arrow_color: str,
    show_basis: bool,
    fill_track_black: bool,
    show_large_ovals: bool,
    dpi: int,
) -> Path:
    """Build and save one W-track schematic using the paper-figure defaults."""
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
        default="dodgerblue",
        help="Trajectory arrow color. Default: dodgerblue",
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
