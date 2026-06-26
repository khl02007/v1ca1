from __future__ import annotations

from pathlib import Path

import pytest

from v1ca1.paper_figures.w_track_schematic import (
    DEFAULT_BASIS_SEGMENT_STYLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OVAL_REGIONS,
    DEFAULT_OVAL_STYLES,
    build_output_path,
    draw_w_track_schematic,
    parse_arguments,
)
from v1ca1.paper_figures.style import SCHEMATIC_COLORS


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures/output"), "schematic", "svg") == Path(
        "paper_figures/output/schematic.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures/output"), "schematic", "jpg")


def test_parse_arguments_defaults_match_paper_schematic_example() -> None:
    args = parse_arguments([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == "w_track_schematic"
    assert args.output_format == "pdf"
    assert args.trajectory_name == "center_to_left"
    assert args.stimulus_layout == "stim1"
    assert args.arrow_color is None
    assert not args.no_basis


def test_default_styles_match_three_segment_example() -> None:
    assert len(DEFAULT_BASIS_SEGMENT_STYLES) == 3
    assert (
        DEFAULT_BASIS_SEGMENT_STYLES[-1]["edge_color"]
        == SCHEMATIC_COLORS["light_basis"]
    )
    assert (
        DEFAULT_BASIS_SEGMENT_STYLES[-1]["fill_color"]
        == SCHEMATIC_COLORS["light_basis"]
    )
    assert DEFAULT_OVAL_REGIONS == ["center_arm", "left_center_connector", "left_arm"]
    assert len(DEFAULT_OVAL_STYLES) == len(DEFAULT_OVAL_REGIONS)


def test_draw_w_track_schematic_draws_outline_and_arrow_without_basis() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = plt.subplots()
    draw_w_track_schematic(
        ax,
        trajectory_name="center_to_left",
        arrow_color="dodgerblue",
        fill_track=False,
    )

    assert any(isinstance(patch, Polygon) for patch in ax.patches)
    assert not any(isinstance(patch, Rectangle) for patch in ax.patches)
    assert not any(isinstance(patch, Circle) for patch in ax.patches)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_draw_w_track_schematic_can_fill_named_regions() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon, Rectangle
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_w_track_schematic(
        ax,
        trajectory_name="center_to_left",
        region_fill_colors={"left_arm": "pink"},
    )

    rectangles = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    polygons = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    assert len(rectangles) == 1
    assert len(polygons) == 2
    assert rectangles[0].get_facecolor() == pytest.approx(to_rgba("pink"))
    assert rectangles[0].get_x() == pytest.approx(0.0)
    assert rectangles[0].get_y() == pytest.approx(0.7)
    assert rectangles[0].get_width() == pytest.approx(0.7)
    assert rectangles[0].get_height() == pytest.approx(3.3)
    assert polygons[-1].get_facecolor()[3] == pytest.approx(0.0)
    assert polygons[-1].get_edgecolor() == pytest.approx(to_rgba("black"))
    plt.close(fig)


def test_draw_w_track_schematic_can_set_region_fill_alpha() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_w_track_schematic(
        ax,
        trajectory_name="center_to_left",
        region_fill_colors={"left_arm": "pink"},
        region_fill_alpha=0.35,
    )

    rectangles = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert len(rectangles) == 1
    assert rectangles[0].get_facecolor() == pytest.approx(to_rgba("pink", 0.35))
    plt.close(fig)


def test_draw_w_track_schematic_accepts_connector_aliases() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_w_track_schematic(
        ax,
        trajectory_name="center_to_right",
        region_fill_colors={"left_connector": "#cccccc"},
    )

    rectangles = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert len(rectangles) == 1
    assert rectangles[0].get_x() == pytest.approx(0.0)
    assert rectangles[0].get_y() == pytest.approx(0.0)
    assert rectangles[0].get_width() == pytest.approx(2.4)
    assert rectangles[0].get_height() == pytest.approx(0.7)
    plt.close(fig)


def test_draw_w_track_schematic_rejects_unknown_region_fill_names() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unknown W-track region"):
        draw_w_track_schematic(
            ax,
            trajectory_name="center_to_left",
            region_fill_colors={"unknown_arm": "pink"},
        )
    plt.close(fig)
