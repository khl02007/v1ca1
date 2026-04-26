from __future__ import annotations

"""Draw publication-style schematic W-track trajectory figures."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


W_TRACK_TRAJECTORY_NAMES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
W_TRACK_OVAL_REGIONS = (
    "left_arm",
    "center_arm",
    "right_arm",
    "left_center_connector",
    "center_right_connector",
)
STIMULUS_LAYOUTS = ("stim1", "stim2")

Point = tuple[float, float]
StyleDict = dict[str, Any]


def get_w_track_geometry() -> tuple[list[Point], dict[str, Point], dict[str, float]]:
    """Return schematic W-track coordinates with uniform corridor width."""
    corridor_w = 0.7
    gap = 1.0
    height = 4.0
    margin = 0.28

    x0 = 0.0
    x1 = x0 + corridor_w
    x2 = x1 + gap
    x3 = x2 + corridor_w
    x4 = x3 + gap
    x5 = x4 + corridor_w

    y0 = 0.0
    y1 = corridor_w
    y2 = height

    outline = [
        (x0, y0),
        (x5, y0),
        (x5, y2),
        (x4, y2),
        (x4, y1),
        (x3, y1),
        (x3, y2),
        (x2, y2),
        (x2, y1),
        (x1, y1),
        (x1, y2),
        (x0, y2),
    ]

    points = {
        "left": ((x0 + x1) / 2, y2 - margin),
        "center": ((x2 + x3) / 2, y2 - margin),
        "right": ((x4 + x5) / 2, y2 - margin),
        "bottom_left": ((x0 + x1) / 2, y1 / 2),
        "bottom_center": ((x2 + x3) / 2, y1 / 2),
        "bottom_right": ((x4 + x5) / 2, y1 / 2),
    }

    dims = {
        "x0": x0,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "y0": y0,
        "y1": y1,
        "y2": y2,
        "corridor_w": corridor_w,
    }
    return outline, points, dims


def trajectory_points(trajectory_name: str, points: dict[str, Point]) -> list[Point]:
    """Return path points for a named schematic W-track trajectory."""
    paths = {
        "center_to_left": [
            points["center"],
            points["bottom_center"],
            points["bottom_left"],
            points["left"],
        ],
        "center_to_right": [
            points["center"],
            points["bottom_center"],
            points["bottom_right"],
            points["right"],
        ],
        "left_to_center": [
            points["left"],
            points["bottom_left"],
            points["bottom_center"],
            points["center"],
        ],
        "right_to_center": [
            points["right"],
            points["bottom_right"],
            points["bottom_center"],
            points["center"],
        ],
    }
    if trajectory_name not in paths:
        raise ValueError(
            f"Unknown trajectory_name {trajectory_name!r}. "
            f"Expected one of {W_TRACK_TRAJECTORY_NAMES!r}."
        )
    return paths[trajectory_name]


def trajectory_segments(trajectory_name: str, points: dict[str, Point]) -> list[list[Point]]:
    """Return the three directed trajectory segments."""
    path = trajectory_points(trajectory_name, points)
    return [[path[index], path[index + 1]] for index in range(len(path) - 1)]


def sample_path(
    path: list[Point] | np.ndarray,
    spacing: float = 0.38,
    include_endpoints: bool = True,
) -> np.ndarray:
    """Sample evenly spaced centers along a polyline."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must have shape (n_points, 2).")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_length = float(segment_lengths.sum())

    if total_length == 0:
        return path.copy()

    distances = list(np.arange(0, total_length, spacing))

    if include_endpoints:
        if not distances or distances[0] != 0:
            distances = [0.0] + distances
        if distances[-1] != total_length:
            distances.append(total_length)

    centers = []
    for distance in distances:
        remaining = float(distance)
        for start, end, segment_length in zip(path[:-1], path[1:], segment_lengths):
            if remaining <= segment_length or np.isclose(remaining, segment_length):
                frac = 0.0 if segment_length == 0 else remaining / segment_length
                centers.append(start + frac * (end - start))
                break
            remaining -= segment_length

    centers = np.asarray(centers, dtype=float)

    if len(centers) > 1:
        keep = [0]
        for index in range(1, len(centers)):
            if not np.allclose(centers[index], centers[keep[-1]]):
                keep.append(index)
        centers = centers[keep]

    return centers


def draw_basis_circles(
    ax: "Axes",
    path: list[Point] | np.ndarray,
    radius: float = 0.35,
    spacing: float = 0.38,
    edge_color: str = "black",
    fill_color: str = "none",
    linewidth: float = 4,
    fill_alpha: float = 1.0,
) -> None:
    """Draw circular basis functions with independent outline and fill."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle

    facecolor = "none" if fill_color == "none" else to_rgba(fill_color, fill_alpha)
    for x, y in sample_path(path, spacing=spacing, include_endpoints=True):
        ax.add_patch(
            Circle(
                (x, y),
                radius=radius,
                facecolor=facecolor,
                edgecolor=edge_color,
                linewidth=linewidth,
                zorder=3,
            )
        )


def draw_segmented_basis_circles(
    ax: "Axes",
    trajectory_name: str,
    points: dict[str, Point],
    segment_styles: list[StyleDict] | None = None,
    default_edge_color: str = "black",
    default_fill_color: str = "none",
    default_fill_alpha: float = 1.0,
    default_radius: float = 0.35,
    default_spacing: float = 0.38,
    default_linewidth: float = 4,
) -> None:
    """Draw basis circles with separate styling for each trajectory segment."""
    segments = trajectory_segments(trajectory_name, points)

    if segment_styles is None:
        segment_styles = [{} for _segment in segments]

    if len(segment_styles) != len(segments):
        raise ValueError("basis_segment_styles must contain exactly three dictionaries.")

    for segment, style in zip(segments, segment_styles, strict=True):
        draw_basis_circles(
            ax,
            segment,
            radius=style.get("radius", default_radius),
            spacing=style.get("spacing", default_spacing),
            edge_color=style.get("edge_color", default_edge_color),
            fill_color=style.get("fill_color", default_fill_color),
            fill_alpha=style.get("fill_alpha", default_fill_alpha),
            linewidth=style.get("linewidth", default_linewidth),
        )


def get_oval_specs(dims: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Return oval geometry for each arm and connector segment."""
    x0, x1 = dims["x0"], dims["x1"]
    x2, x3 = dims["x2"], dims["x3"]
    x4, x5 = dims["x4"], dims["x5"]
    y1, y2 = dims["y1"], dims["y2"]
    corridor_w = dims["corridor_w"]

    return {
        "left_arm": {
            "xy": ((x0 + x1) / 2, (y1 + y2) / 2),
            "width": corridor_w * 0.95,
            "height": y2 - y1 + 0.25,
            "angle": 0,
        },
        "center_arm": {
            "xy": ((x2 + x3) / 2, (y1 + y2) / 2),
            "width": corridor_w * 0.95,
            "height": y2 - y1 + 0.25,
            "angle": 0,
        },
        "right_arm": {
            "xy": ((x4 + x5) / 2, (y1 + y2) / 2),
            "width": corridor_w * 0.95,
            "height": y2 - y1 + 0.25,
            "angle": 0,
        },
        "left_center_connector": {
            "xy": ((x0 + x3) / 2, y1 / 2),
            "width": x3 - x0 + 0.15,
            "height": corridor_w * 0.95,
            "angle": 0,
        },
        "center_right_connector": {
            "xy": ((x2 + x5) / 2, y1 / 2),
            "width": x5 - x2 + 0.15,
            "height": corridor_w * 0.95,
            "angle": 0,
        },
    }


def draw_large_ovals(
    ax: "Axes",
    dims: dict[str, float],
    oval_regions: str | list[str] | None,
    oval_styles: list[StyleDict] | None = None,
    default_edge_color: str = "red",
    default_fill_color: str = "none",
    default_fill_alpha: float = 1.0,
    default_linewidth: float = 4,
) -> None:
    """Draw one or more large ovals over selected regions."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Ellipse

    specs = get_oval_specs(dims)

    if oval_regions is None:
        return

    if isinstance(oval_regions, str):
        oval_regions = [oval_regions]

    if oval_styles is None:
        oval_styles = [{} for _region in oval_regions]

    if len(oval_styles) != len(oval_regions):
        raise ValueError("oval_styles must match the number of oval_regions.")

    for region, style in zip(oval_regions, oval_styles, strict=True):
        if region not in specs:
            raise ValueError(
                f"Unknown oval region {region!r}. Expected one of {W_TRACK_OVAL_REGIONS!r}."
            )
        fill_color = style.get("fill_color", default_fill_color)
        facecolor = (
            "none"
            if fill_color == "none"
            else to_rgba(fill_color, style.get("fill_alpha", default_fill_alpha))
        )

        ax.add_patch(
            Ellipse(
                **specs[region],
                facecolor=facecolor,
                edgecolor=style.get("edge_color", default_edge_color),
                linewidth=style.get("linewidth", default_linewidth),
                zorder=4,
            )
        )


def draw_path(
    ax: "Axes",
    path: list[Point],
    show_arrow: bool = True,
    arrow_color: str = "orangered",
) -> None:
    """Draw a trajectory, optionally with an arrowhead."""
    xs, ys = zip(*path, strict=True)

    if show_arrow:
        ax.plot(
            xs[:-1],
            ys[:-1],
            color=arrow_color,
            linewidth=3,
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
                "lw": 3,
                "mutation_scale": 24,
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
            linewidth=3,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )


def add_labels(
    ax: "Axes",
    dims: dict[str, float],
    stimulus_layout: str = "stim1",
    label_color: str = "black",
) -> None:
    """Add outside stimulus labels for the selected stimulus layout."""
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
        x0 - 1.0,
        y2 / 2,
        labels["left"],
        ha="center",
        va="center",
        fontsize=42,
        color=label_color,
    )
    ax.text(
        x5 + 1.0,
        y2 / 2,
        labels["right"],
        ha="center",
        va="center",
        fontsize=42,
        color=label_color,
    )
    ax.text(
        x5 / 2,
        y0 - 0.9,
        labels["center"],
        ha="center",
        va="center",
        fontsize=42,
        color=label_color,
    )


def plot_w_track_trajectory(
    trajectory_name: str = "center_to_left",
    show_labels: bool = True,
    stimulus_layout: str = "stim1",
    label_color: str = "black",
    show_arrow: bool = True,
    arrow_color: str = "orangered",
    show_basis: bool = False,
    basis_edge_color: str = "black",
    basis_fill_color: str = "none",
    basis_fill_alpha: float = 1.0,
    basis_radius: float = 0.35,
    basis_spacing: float = 0.38,
    basis_linewidth: float = 4,
    basis_segment_styles: list[StyleDict] | None = None,
    fill_track_black: bool = False,
    show_large_ovals: bool = False,
    oval_regions: str | list[str] | None = None,
    oval_edge_color: str = "red",
    oval_fill_color: str = "none",
    oval_fill_alpha: float = 1.0,
    oval_linewidth: float = 4,
    oval_styles: list[StyleDict] | None = None,
    figsize: tuple[float, float] = (5, 4),
) -> tuple["Figure", "Axes"]:
    """Draw a schematic W-track with optional trajectory, basis circles, and ovals."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    outline, points, dims = get_w_track_geometry()
    path = trajectory_points(trajectory_name, points)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="black" if fill_track_black else "white",
            edgecolor="black",
            linewidth=3,
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

    draw_path(ax, path, show_arrow=show_arrow, arrow_color=arrow_color)

    if show_labels:
        add_labels(ax, dims, stimulus_layout=stimulus_layout, label_color=label_color)

    x5 = dims["x5"]
    y2 = dims["y2"]

    ax.set_aspect("equal")
    ax.set_xlim(-1.6 if show_labels else -0.4, x5 + (1.6 if show_labels else 0.4))
    ax.set_ylim(-1.3 if show_labels else -0.4, y2 + 0.4)
    ax.axis("off")

    return fig, ax


def _highlight_basis_segment_styles(
    highlight_segment: int | None,
    radius: float,
    spacing: float,
) -> list[StyleDict] | None:
    """Return CLI basis styles with one optional highlighted segment."""
    if highlight_segment is None:
        return None

    styles: list[StyleDict] = [
        {
            "edge_color": "black",
            "fill_color": "none",
            "radius": radius,
            "spacing": spacing,
        }
        for _index in range(3)
    ]
    styles[highlight_segment - 1] = {
        "edge_color": "red",
        "fill_color": "pink",
        "fill_alpha": 0.35,
        "radius": radius,
        "spacing": spacing,
    }
    return styles


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Draw a schematic W-track trajectory.")
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
        help="Outside label layout. Default: stim1",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI. Default: 300",
    )
    parser.add_argument(
        "--arrow-color",
        default="orangered",
        help="Trajectory arrow color. Default: orangered",
    )
    parser.add_argument(
        "--label-color",
        default="black",
        help="Stimulus label color. Default: black",
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
        "--show-basis",
        action="store_true",
        help="Draw circular basis functions along the trajectory.",
    )
    parser.add_argument(
        "--highlight-basis-segment",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Highlight one basis segment, numbered from trajectory start.",
    )
    parser.add_argument(
        "--basis-radius",
        type=float,
        default=0.35,
        help="Basis circle radius. Default: 0.35",
    )
    parser.add_argument(
        "--basis-spacing",
        type=float,
        default=0.38,
        help="Basis circle center spacing. Default: 0.38",
    )
    parser.add_argument(
        "--fill-track-black",
        action="store_true",
        help="Fill the track body in black.",
    )
    parser.add_argument(
        "--oval-region",
        action="append",
        choices=W_TRACK_OVAL_REGIONS,
        default=None,
        help="Region to outline with a large oval. May be passed more than once.",
    )
    parser.add_argument(
        "--oval-edge-color",
        default="red",
        help="Large oval outline color. Default: red",
    )
    parser.add_argument(
        "--oval-fill-color",
        default="none",
        help="Large oval fill color. Default: none",
    )
    parser.add_argument(
        "--oval-fill-alpha",
        type=float,
        default=1.0,
        help="Large oval fill alpha. Default: 1.0",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure after saving. Ignored when --output-path is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the schematic W-track plotting CLI."""
    args = parse_arguments()
    if args.output_path is not None and not args.show:
        import matplotlib

        matplotlib.use("Agg")

    basis_segment_styles = _highlight_basis_segment_styles(
        highlight_segment=args.highlight_basis_segment,
        radius=args.basis_radius,
        spacing=args.basis_spacing,
    )
    fig, _ax = plot_w_track_trajectory(
        trajectory_name=args.trajectory_name,
        show_labels=not args.no_labels,
        stimulus_layout=args.stimulus_layout,
        label_color=args.label_color,
        show_arrow=not args.no_arrow,
        arrow_color=args.arrow_color,
        show_basis=args.show_basis,
        basis_radius=args.basis_radius,
        basis_spacing=args.basis_spacing,
        basis_segment_styles=basis_segment_styles,
        fill_track_black=args.fill_track_black,
        show_large_ovals=args.oval_region is not None,
        oval_regions=args.oval_region,
        oval_edge_color=args.oval_edge_color,
        oval_fill_color=args.oval_fill_color,
        oval_fill_alpha=args.oval_fill_alpha,
    )

    import matplotlib.pyplot as plt

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved W-track schematic to {args.output_path}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
