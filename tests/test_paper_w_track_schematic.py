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
    assert args.arrow_color == "dodgerblue"
    assert not args.no_basis


def test_default_styles_match_three_segment_example() -> None:
    assert len(DEFAULT_BASIS_SEGMENT_STYLES) == 3
    assert DEFAULT_BASIS_SEGMENT_STYLES[-1]["edge_color"] == "red"
    assert DEFAULT_BASIS_SEGMENT_STYLES[-1]["fill_color"] == "pink"
    assert DEFAULT_OVAL_REGIONS == ["center_arm", "left_center_connector", "left_arm"]
    assert len(DEFAULT_OVAL_STYLES) == len(DEFAULT_OVAL_REGIONS)


def test_draw_w_track_schematic_draws_outline_and_arrow_without_basis() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    fig, ax = plt.subplots()
    draw_w_track_schematic(
        ax,
        trajectory_name="center_to_left",
        arrow_color="dodgerblue",
        fill_track=False,
    )

    assert any(isinstance(patch, Polygon) for patch in ax.patches)
    assert not any(isinstance(patch, Circle) for patch in ax.patches)
    assert len(ax.lines) == 1
    plt.close(fig)
